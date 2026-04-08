"""
MSHDR v2.0: Supervised Contrastive Manifold-Aware Autoencoder (SCMAA)
=======================================================================

Novel Contributions over v1
----------------------------
1. Adaptive Ensemble Feature Selection
   MI + ANOVA F-statistics + Random Forest importance → weighted ensemble score
   → elbow-based cutoff (no fixed top-k guess)

2. Fisher-Weighted Manifold Fusion
   PCA, LDA, Isomap, UMAP run in parallel.
   Each embedding is scored by the multiclass Fisher criterion (J = tr(S_W⁻¹ S_B)).
   Softmax-normalised scores act as attention weights → scaled weighted concatenation.
   LDA gets high weight (it's supervised); random manifolds get low weight.

3. Supervised Contrastive VAE (SCVAE)  ← core novelty
   Joint multi-task loss forces the 4-D bottleneck to be both compact AND discriminative:
       L = α·MSE(x̂,x)  +  β(t)·KL(q‖p)  +  γ·SupCon(z,y)  +  δ·CE(ŷ,y)
   β(t) is linearly annealed 0→β_max (curriculum: learn to reconstruct first,
   then regularise the latent space).
   During inference, z = μ (deterministic mean, zero noise).

4. Hierarchical Stacking Ensemble
   Level-1: RF + XGB + MLP trained on 4-D latent features.
   Out-of-fold (OOF) predictions stack into meta-features.
   Level-2: Logistic Regression meta-learner — no fixed weights, adapts to data.

Root-cause of v1 failure (61 % accuracy)
------------------------------------------
• Pure reconstruction loss → bottleneck captures variance, NOT class structure.
• Equal-weight concatenation dilutes LDA (the only supervised signal) with PCA/Isomap noise.
• Fixed [0.4, 0.6] voting weights → not adaptive to actual model strengths.
"""

from __future__ import annotations

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier
)
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import Isomap
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

from src.utils.config import RANDOM_SEED

warnings.filterwarnings('ignore')

# ── MPS fallback: allow unsupported ops to silently run on CPU ──────────────
# Must be set BEFORE torch is imported.
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

# ── optional heavy dependencies ──────────────────────────────────────────────

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader as TorchLoader, TensorDataset
    _TORCH = True
except ImportError:
    _TORCH = False

try:
    from xgboost import XGBClassifier
    _XGB = True
except ImportError:
    XGBClassifier = GradientBoostingClassifier
    _XGB = False

try:
    import lightgbm as lgb
    _LGB = True
except ImportError:
    _LGB = False

try:
    import umap as umap_lib
    _UMAP = True
except ImportError:
    _UMAP = False


# ════════════════════════════════════════════════════════════════════════════
# 1. PyTorch modules (SCVAE + SupCon Loss)
# ════════════════════════════════════════════════════════════════════════════

if _TORCH:
    class SupervisedContrastiveLoss(nn.Module):
        """
        Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).
        Pulls same-class embeddings together; pushes different-class apart.
        Temperature τ controls sharpness of the distribution.
        """
        def __init__(self, temperature: float = 0.07):
            super().__init__()
            self.tau = temperature

        def forward(self, z: 'torch.Tensor', y: 'torch.Tensor') -> 'torch.Tensor':
            N = z.size(0)
            device = z.device

            z_norm = F.normalize(z, dim=1)                        # [N, D]
            sim = torch.matmul(z_norm, z_norm.T) / self.tau       # [N, N]

            # Positive mask: same class, different sample
            labels = y.view(-1, 1)
            pos_mask = torch.eq(labels, labels.T).float().to(device)
            eye = torch.eye(N, device=device)
            pos_mask = pos_mask - eye

            # Numerical stability: subtract row max
            sim_max, _ = torch.max(sim, dim=1, keepdim=True)
            sim = sim - sim_max.detach()

            exp_sim = torch.exp(sim) * (1.0 - eye)               # exclude diagonal
            log_prob = sim - torch.log(exp_sim.sum(1, keepdim=True) + 1e-8)

            n_pos = pos_mask.sum(1)
            mean_log_pos = (pos_mask * log_prob).sum(1) / (n_pos + 1e-8)

            has_pos = (n_pos > 0).float()
            loss = -(mean_log_pos * has_pos).sum() / (has_pos.sum() + 1e-8)
            return loss

    class SCVAE(nn.Module):
        """
        Supervised Contrastive Variational Autoencoder.

        Architecture:
            Encoder: input_dim → h1 → h2 → [μ, log_σ²]  (latent_dim each)
            Reparameterisation: z = μ + σ·ε
            Decoder: latent_dim → h2 → h1 → input_dim  (Sigmoid output, range [0,1])
            Classifier head: latent_dim → h_clf → n_classes

        At inference:  z = μ  (deterministic, no noise)
        """
        def __init__(self, input_dim: int, latent_dim: int = 4, n_classes: int = 2):
            super().__init__()
            h1 = max(input_dim * 4, 32)
            h2 = max(input_dim * 2, 16)
            h_c = max(latent_dim * 2, 8)

            self.enc = nn.Sequential(
                nn.Linear(input_dim, h1),
                nn.BatchNorm1d(h1),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(h1, h2),
                nn.BatchNorm1d(h2),
                nn.ReLU(),
            )
            self.fc_mu     = nn.Linear(h2, latent_dim)
            self.fc_logvar = nn.Linear(h2, latent_dim)

            self.dec = nn.Sequential(
                nn.Linear(latent_dim, h2),
                nn.BatchNorm1d(h2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(h2, h1),
                nn.BatchNorm1d(h1),
                nn.ReLU(),
                nn.Linear(h1, input_dim),
                nn.Sigmoid(),
            )

            self.clf = nn.Sequential(
                nn.Linear(latent_dim, h_c),
                nn.ReLU(),
                nn.Linear(h_c, n_classes),
            )

        def encode(self, x):
            h = self.enc(x)
            return self.fc_mu(h), self.fc_logvar(h)

        def reparameterise(self, mu, logvar):
            if self.training:
                std = torch.exp(0.5 * logvar).clamp(max=10.0)
                return mu + std * torch.randn_like(std)
            return mu                                           # deterministic inference

        def forward(self, x):
            mu, logvar = self.encode(x)
            z     = self.reparameterise(mu, logvar)
            x_hat = self.dec(z)
            logits = self.clf(z)
            return x_hat, mu, logvar, z, logits

        @torch.no_grad()
        def encode_deterministic(self, x: 'torch.Tensor') -> np.ndarray:
            self.eval()
            mu, _ = self.encode(x)
            return mu.cpu().numpy()


# ════════════════════════════════════════════════════════════════════════════
# 2. Helper utilities
# ════════════════════════════════════════════════════════════════════════════

def _fisher_criterion(X: np.ndarray, y: np.ndarray) -> float:
    """
    Multiclass Fisher Linear Discriminant criterion  J = tr(S_W⁻¹ S_B).
    Higher value → better class separability in this embedding space.
    Returns 0.0 on numerical failure.
    """
    classes = np.unique(y)
    mu_all  = X.mean(axis=0)
    d = X.shape[1]
    S_B = np.zeros((d, d))
    S_W = np.zeros((d, d))
    for c in classes:
        Xc = X[y == c]
        nc = len(Xc)
        dc = (Xc.mean(axis=0) - mu_all).reshape(-1, 1)
        S_B += nc * (dc @ dc.T)
        S_W += (Xc - Xc.mean(axis=0)).T @ (Xc - Xc.mean(axis=0))
    S_W += np.eye(d) * 1e-6   # regularise for stability
    try:
        J = float(np.trace(np.linalg.solve(S_W, S_B)))
    except np.linalg.LinAlgError:
        J = float(np.trace(np.linalg.pinv(S_W) @ S_B))
    return max(J, 0.0)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def _train_scvae(
    model: 'SCVAE',
    X: np.ndarray,
    y: np.ndarray,
    epochs: int       = 100,
    batch_size: int   = 512,
    lr: float         = 1e-3,
    alpha: float      = 1.0,    # reconstruction weight
    beta_max: float   = 0.5,    # max KL weight (β-VAE style)
    gamma: float      = 0.5,    # supervised contrastive weight
    delta: float      = 1.0,    # classification weight
    warmup_epochs: int = 40,    # β annealing period
    patience: int     = 15,
    device: str       = 'cpu',
) -> 'SCVAE':
    """
    Train SCVAE with annealed multi-task loss:
        L = α·Recon + β(t)·KL + γ·SupCon + δ·CE

    β(t) = β_max × min(1, t / warmup_epochs)   (linear warmup)

    Curriculum effect: early epochs focus on reconstruction (β≈0),
    later epochs regularise the latent space and push for separability.
    """
    sup_con = SupervisedContrastiveLoss(temperature=0.07)
    ce_loss = nn.CrossEntropyLoss()
    opt     = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched   = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

    Xt = torch.FloatTensor(X).to(device)
    yt = torch.LongTensor(y).to(device)
    ds = TensorDataset(Xt, yt)
    loader = TorchLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    model.to(device).train()
    best_loss, best_state = float('inf'), None
    patience_cnt = 0

    print(f"  Training SCVAE: up to {epochs} epochs (patience={patience}) ...", flush=True)
    for epoch in range(epochs):
        beta = beta_max * min(1.0, epoch / max(warmup_epochs, 1))
        epoch_loss = 0.0

        for xb, yb in loader:
            opt.zero_grad()
            x_hat, mu, logvar, z, logits = model(xb)

            L_recon = F.mse_loss(x_hat, xb)
            L_kl    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            L_con   = sup_con(z, yb)
            L_cls   = ce_loss(logits, yb)

            loss = alpha * L_recon + beta * L_kl + gamma * L_con + delta * L_cls
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            epoch_loss += loss.item()

        sched.step()
        avg = epoch_loss / len(loader)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    epoch {epoch+1:>3}/{epochs}  loss={avg:.4f}  patience={patience_cnt}/{patience}", flush=True)

        if avg < best_loss - 1e-5:
            best_loss = avg
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"    Early stop at epoch {epoch+1}", flush=True)
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


# ════════════════════════════════════════════════════════════════════════════
# 3. Stage classes
# ════════════════════════════════════════════════════════════════════════════

class AdaptiveFeatureSelector:
    """
    Ensemble feature scorer combining three complementary criteria:
      • Mutual Information  (non-linear dependence with label)
      • ANOVA F-statistic   (linear class separation)
      • Random Forest importance (tree-based, handles interactions)

    Final score = 0.4 × MI_norm + 0.3 × ANOVA_norm + 0.3 × RF_norm.
    Elbow detection on sorted scores determines the cutoff adaptively.
    """

    def __init__(self, random_state: int = 42):
        self.rs = random_state
        self.scores_: pd.Series | None = None
        self.selected_features_: list[str] = []

    def fit_transform(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: np.ndarray,
        min_k: int = 5,
        max_k: int = 12,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        cols = X_train.columns.tolist()

        mi   = mutual_info_classif(X_train, y_train, random_state=self.rs)
        anova, _ = f_classif(X_train, y_train)
        rf   = RandomForestClassifier(
            n_estimators=100, random_state=self.rs, n_jobs=-1
        ).fit(X_train, y_train).feature_importances_

        # Normalise each scorer to [0,1]
        def _norm(v):
            rng = v.max() - v.min()
            return (v - v.min()) / (rng + 1e-12)

        ensemble = 0.4 * _norm(mi) + 0.3 * _norm(anova) + 0.3 * _norm(rf)

        self.scores_ = pd.Series(ensemble, index=cols).sort_values(ascending=False)

        # Elbow detection: largest drop in sorted scores
        scores_sorted = self.scores_.values
        drops = scores_sorted[:-1] - scores_sorted[1:]
        elbow_k = int(np.argmax(drops) + 1)           # +1: keep up to and including elbow
        k = int(np.clip(elbow_k, min_k, max_k))

        self.selected_features_ = self.scores_.head(k).index.tolist()
        print(f"  Elbow k={k} | Selected: {self.selected_features_}")

        return X_train[self.selected_features_], X_test[self.selected_features_]


class FisherManifoldFusion:
    """
    Runs PCA, LDA, Isomap (+ UMAP if available) in parallel.
    Scores each embedding by Fisher's criterion J = tr(S_W⁻¹ S_B).
    Attention weights = softmax(J_scores).
    Each embedding is StandardScaled, then multiplied by its weight
    before horizontal concatenation.
    """

    METHODS = ['pca', 'lda', 'isomap', 'umap']

    def __init__(self, random_state: int = 42):
        self.rs = random_state
        self.transformers_: dict = {}
        self.scalers_:      dict = {}
        self.weights_:      dict = {}
        self.dims_:         dict = {'pca': 3, 'lda': 1, 'isomap': 2, 'umap': 2}

    # Max samples used to *fit* non-linear manifold methods (Isomap, UMAP).
    # Fitting on the full 46K dataset causes O(n²) blowup — subsampling to
    # MANIFOLD_FIT_SAMPLES preserves the manifold structure while staying fast.
    # The fitted transformer then maps all training/test points via transform().
    MANIFOLD_FIT_SAMPLES = 2000

    def _subsample_for_fit(self, X, y):
        """Return a class-stratified subsample for fitting slow manifold methods."""
        n = len(X)
        if n <= self.MANIFOLD_FIT_SAMPLES:
            return X, y
        rng = np.random.default_rng(self.rs)
        classes, counts = np.unique(y, return_counts=True)
        idx = []
        for c, cnt in zip(classes, counts):
            c_idx = np.where(y == c)[0]
            k = max(1, int(self.MANIFOLD_FIT_SAMPLES * cnt / n))
            idx.append(rng.choice(c_idx, min(k, len(c_idx)), replace=False))
        return X[np.concatenate(idx)], y[np.concatenate(idx)]

    def _fit_transform_method(self, name, X_train, y_train):
        if name == 'pca':
            print(f"    fitting {name}...", flush=True)
            t = PCA(n_components=self.dims_['pca'], random_state=self.rs)
            return t, t.fit_transform(X_train)
        if name == 'lda':
            print(f"    fitting {name}...", flush=True)
            t = LDA(n_components=self.dims_['lda'])
            Xt = t.fit_transform(X_train, y_train)
            return t, Xt.reshape(-1, 1) if Xt.ndim == 1 else Xt
        if name == 'isomap':
            print(f"    fitting {name} on subsample ({self.MANIFOLD_FIT_SAMPLES} pts)...", flush=True)
            X_sub, _ = self._subsample_for_fit(X_train, y_train)
            t = Isomap(n_components=self.dims_['isomap'], n_neighbors=10, n_jobs=-1)
            t.fit(X_sub)
            return t, t.transform(X_train)
        if name == 'umap' and _UMAP:
            print(f"    fitting {name} on subsample ({self.MANIFOLD_FIT_SAMPLES} pts)...", flush=True)
            X_sub, _ = self._subsample_for_fit(X_train, y_train)
            t = umap_lib.UMAP(n_components=self.dims_['umap'],
                              random_state=self.rs, n_neighbors=15, min_dist=0.1)
            t.fit(X_sub)
            return t, t.transform(X_train)
        return None, None

    def fit_transform(
        self, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:

        train_parts, test_parts = [], []
        fisher_scores = {}

        for name in self.METHODS:
            transformer, Xt_train = self._fit_transform_method(name, X_train, y_train)
            if transformer is None:
                continue

            # Transform test set
            if name == 'umap':
                Xt_test = transformer.transform(X_test)
            elif name == 'lda':
                Xt_test = transformer.transform(X_test)
                if Xt_test.ndim == 1:
                    Xt_test = Xt_test.reshape(-1, 1)
                    Xt_train = Xt_train.reshape(-1, 1)
            else:
                Xt_test = transformer.transform(X_test)

            # Standardise each embedding independently
            sc = StandardScaler().fit(Xt_train)
            Xt_train_s = sc.transform(Xt_train)
            Xt_test_s  = sc.transform(Xt_test)

            J = _fisher_criterion(Xt_train_s, y_train)
            fisher_scores[name] = J

            self.transformers_[name] = transformer
            self.scalers_[name]      = sc
            train_parts.append((name, Xt_train_s))
            test_parts.append((name, Xt_test_s))

        # Attention weights via softmax on Fisher scores
        names  = [n for n, _ in train_parts]
        scores = np.array([fisher_scores[n] for n in names])
        weights = _softmax(scores)
        self.weights_ = dict(zip(names, weights))

        print("  Fisher scores:", {n: f"{fisher_scores[n]:.2f}" for n in names})
        print("  Attention weights:", {n: f"{w:.3f}" for n, w in self.weights_.items()})

        # Weighted concatenation
        X_train_fused = np.hstack([w * X for (n, X), w in zip(train_parts, weights)])
        X_test_fused  = np.hstack([w * X for (n, X), w in zip(test_parts,  weights)])

        return X_train_fused, X_test_fused


class HierarchicalStackingEnsemble:
    """
    Two-level stacking ensemble.

    Level-1 base learners (trained on latent features):
        • Random Forest
        • XGBoost  (or GradientBoosting fallback)
        • LightGBM (or MLP fallback)

    Level-2 meta-learner:
        • Logistic Regression trained on 5-fold OOF probability predictions.
        Weights adapt to actual model performance — no manual tuning.
    """

    def __init__(self, random_state: int = 42):
        self.rs = random_state
        self.meta = LogisticRegression(C=1.0, max_iter=1000, random_state=random_state)
        self._build_base_learners()

    def _build_base_learners(self):
        self.base_learners = [
            ('rf', RandomForestClassifier(
                n_estimators=100, n_jobs=-1, random_state=self.rs)),
        ]
        if _XGB:
            self.base_learners.append(('xgb', XGBClassifier(
                n_estimators=100, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.8,
                random_state=self.rs, eval_metric='logloss', verbosity=0)))
        else:
            self.base_learners.append(('gb', GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.05, max_depth=4,
                random_state=self.rs)))

        if _LGB:
            self.base_learners.append(('lgb', lgb.LGBMClassifier(
                n_estimators=100, learning_rate=0.05,
                random_state=self.rs, verbose=-1)))
        else:
            self.base_learners.append(('mlp', MLPClassifier(
                hidden_layer_sizes=(64, 32), activation='relu',
                alpha=1e-4, random_state=self.rs,
                max_iter=300, early_stopping=True)))

    def fit_predict(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (y_pred, y_proba) on X_test.
        Uses 3-fold OOF for meta-feature construction.
        """
        n_base  = len(self.base_learners)
        skf     = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.rs)
        oof     = np.zeros((len(X_train), n_base))
        test_ps = np.zeros((len(X_test),  n_base))

        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            Xtr, Xval = X_train[tr_idx], X_train[val_idx]
            ytr, _    = y_train[tr_idx], y_train[val_idx]

            for b, (name, clf) in enumerate(self.base_learners):
                clf_f = clone(clf)
                clf_f.fit(Xtr, ytr)
                oof[val_idx, b]  = clf_f.predict_proba(Xval)[:, 1]
                test_ps[:, b]   += clf_f.predict_proba(X_test)[:, 1] / skf.n_splits

        self.meta.fit(oof, y_train)
        y_pred  = self.meta.predict(test_ps)
        y_proba = self.meta.predict_proba(test_ps)[:, 1]
        return y_pred, y_proba


# ════════════════════════════════════════════════════════════════════════════
# 4. Main Pipeline
# ════════════════════════════════════════════════════════════════════════════

class MSHDRv2:
    """
    MSHDR v2.0 — Supervised Contrastive Manifold-Aware Autoencoder (SCMAA)

    Four-stage pipeline:
        Stage 1 → Adaptive Ensemble Feature Selection
        Stage 2 → Fisher-Weighted Manifold Fusion
        Stage 3 → Supervised Contrastive VAE  (PyTorch) / FactorAnalysis+LDA fallback
        Stage 4 → Hierarchical Stacking Ensemble
    """

    LATENT_DIM = 4      # VAE bottleneck dimensionality
    LABEL      = 'MSHDR_v2'

    def __init__(
        self,
        X_train: pd.DataFrame,
        X_test:  pd.DataFrame,
        y_train: np.ndarray,
        y_test:  np.ndarray,
    ):
        self._X_train_orig = X_train.copy()
        self._X_test_orig  = X_test.copy()
        self.y_train = np.asarray(y_train)
        self.y_test  = np.asarray(y_test)
        self.results: dict = {}

        # Working copies (will be updated per stage)
        self.X_train: np.ndarray | pd.DataFrame = X_train
        self.X_test:  np.ndarray | pd.DataFrame = X_test

    # ── Stage 1 ──────────────────────────────────────────────────────────────

    def stage1_adaptive_feature_selection(self) -> 'MSHDRv2':
        print("\n--- Stage 1: Adaptive Ensemble Feature Selection ---")
        sel = AdaptiveFeatureSelector(random_state=RANDOM_SEED)
        self.X_train, self.X_test = sel.fit_transform(
            self.X_train, self.X_test, self.y_train
        )
        self.feature_selector = sel
        return self

    # ── Stage 2 ──────────────────────────────────────────────────────────────

    def stage2_fisher_manifold_fusion(self) -> 'MSHDRv2':
        print("\n--- Stage 2: Fisher-Weighted Manifold Fusion ---")
        X_tr = (self.X_train.values
                if isinstance(self.X_train, pd.DataFrame)
                else self.X_train)
        X_te = (self.X_test.values
                if isinstance(self.X_test, pd.DataFrame)
                else self.X_test)

        fusion = FisherManifoldFusion(random_state=RANDOM_SEED)
        self.X_train, self.X_test = fusion.fit_transform(X_tr, X_te, self.y_train)
        self.manifold_fusion = fusion
        print(f"  Fused representation: {self.X_train.shape[1]}D")
        return self

    # ── Stage 3 ──────────────────────────────────────────────────────────────

    def stage3_supervised_contrastive_vae(self) -> 'MSHDRv2':
        print("\n--- Stage 3: Supervised Contrastive VAE ---")

        # MinMaxScaler maps training data to [0,1] — matches Sigmoid decoder output.
        # Test-set values are clipped to [0,1] to handle any out-of-range extrapolation.
        self._vae_scaler = MinMaxScaler()
        X_tr_s = self._vae_scaler.fit_transform(self.X_train)
        X_te_s = np.clip(self._vae_scaler.transform(self.X_test), 0.0, 1.0)

        input_dim = X_tr_s.shape[1]

        if _TORCH:
            if torch.cuda.is_available():
                device = 'cuda'                  # NVIDIA GPU — best performance
            elif (torch.backends.mps.is_available()
                  and torch.backends.mps.is_built()
                  and os.environ.get('MSHDR_USE_MPS', '0') == '1'):
                device = 'mps'                   # Apple Silicon — opt-in only
            else:
                device = 'cpu'                   # safe default on Mac
            print(f"  PyTorch device: {device}")
            model = SCVAE(input_dim=input_dim, latent_dim=self.LATENT_DIM)
            model = _train_scvae(
                model, X_tr_s, self.y_train,
                epochs=100, batch_size=512,
                alpha=1.0, beta_max=0.5, gamma=0.5, delta=1.0,
                warmup_epochs=40, patience=15, device=device,
            )
            Xt = torch.FloatTensor(X_tr_s).to(device)
            Xe = torch.FloatTensor(X_te_s).to(device)
            self.X_train = model.encode_deterministic(Xt)
            self.X_test  = model.encode_deterministic(Xe)
            self._vae_model = model
        else:
            # Fallback: FactorAnalysis (probabilistic PCA) + LDA
            print("  PyTorch unavailable — using FactorAnalysis + LDA fallback")
            fa = FactorAnalysis(n_components=self.LATENT_DIM, random_state=RANDOM_SEED)
            fa_train = fa.fit_transform(X_tr_s)
            fa_test  = fa.transform(X_te_s)
            # LDA further squeezes to 1D then concat with FA for discriminability
            lda2 = LDA(n_components=1)
            lda_tr = lda2.fit_transform(fa_train, self.y_train).reshape(-1, 1)
            lda_te = lda2.transform(fa_test).reshape(-1, 1)
            self.X_train = np.hstack([fa_train, lda_tr])
            self.X_test  = np.hstack([fa_test,  lda_te])
            self._vae_model = None

        print(f"  Latent representation: {self.X_train.shape[1]}D")
        return self

    # ── Stage 4 ──────────────────────────────────────────────────────────────

    def stage4_hierarchical_stacking(self) -> dict:
        print("\n--- Stage 4: Hierarchical Stacking Ensemble ---")
        ensemble = HierarchicalStackingEnsemble(random_state=RANDOM_SEED)
        X_tr = self.X_train.values if isinstance(self.X_train, pd.DataFrame) else np.asarray(self.X_train)
        X_te = self.X_test.values  if isinstance(self.X_test,  pd.DataFrame) else np.asarray(self.X_test)
        y_pred, y_proba = ensemble.fit_predict(X_tr, self.y_train, X_te)

        self._last_y_pred = y_pred   # expose for confusion matrix in main.py
        self.results[self.LABEL] = {
            'Accuracy':  accuracy_score(self.y_test, y_pred),
            'Precision': precision_score(self.y_test, y_pred, zero_division=0),
            'Recall':    recall_score(self.y_test, y_pred, zero_division=0),
            'F1-Score':  f1_score(self.y_test, y_pred, zero_division=0),
            'AUC-ROC':   roc_auc_score(self.y_test, y_proba),
        }
        m = self.results[self.LABEL]
        print(f"  {self.LABEL} | Acc: {m['Accuracy']:.4f} | "
              f"F1: {m['F1-Score']:.4f} | AUC: {m['AUC-ROC']:.4f}")
        return self.results

    # ── Entry point ──────────────────────────────────────────────────────────

    def run_all(self) -> dict:
        print("\n" + "=" * 80)
        print("PHASE 3: MSHDR v2.0  (SCMAA Pipeline)")
        print("=" * 80)
        print(f"PyTorch available : {_TORCH}")
        print(f"XGBoost available : {_XGB}")
        print(f"LightGBM available: {_LGB}")
        print(f"UMAP available    : {_UMAP}")

        self.stage1_adaptive_feature_selection()
        self.stage2_fisher_manifold_fusion()
        self.stage3_supervised_contrastive_vae()
        self.stage4_hierarchical_stacking()

        return self.results
