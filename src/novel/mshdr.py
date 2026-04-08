import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import Isomap
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.utils.config import RANDOM_SEED

# Graceful fallback if XGBoost not installed
try:
    from xgboost import XGBClassifier
    _has_xgboost = True
except ImportError:
    XGBClassifier = GradientBoostingClassifier
    _has_xgboost = False


class MSHDR:
    """
    Multi-Stage Hybrid Dimensionality Reduction Pipeline
    
    Architecture:
    1. Feature Selection (Mutual Information)
    2. Manifold Fusion (PCA + LDA + Isomap)
    3. Autoencoder Compression
    4. Ensemble Classification
    """
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.results = {}
        self.selected_features = None

    def stage1_feature_selection(self, top_k=8):
        print("\n--- Stage 1: Feature Selection ---")

        # Mutual information measures how much information each feature gives about the label
        mi_scores = mutual_info_classif(self.X_train, self.y_train, random_state=RANDOM_SEED)
        
        # Keep top K features - 8 was found optimal in testing
        feature_ranking = pd.DataFrame({
            'feature': self.X_train.columns,
            'mi_score': mi_scores
        }).sort_values('mi_score', ascending=False)

        self.selected_features = feature_ranking.head(top_k)['feature'].tolist()
        print(f"Top {top_k} features selected: {self.selected_features}")

        # Apply selection
        self.X_train = self.X_train[self.selected_features]
        self.X_test = self.X_test[self.selected_features]
        
        return self

    def stage2_manifold_fusion(self):
        print("\n--- Stage 2: Manifold Fusion ---")

        # PCA and LDA fit on full training set (both are fast O(nd²))
        pca = PCA(n_components=3, random_state=RANDOM_SEED).fit(self.X_train)
        lda = LDA(n_components=1).fit(self.X_train, self.y_train)

        # Isomap is O(n²) — subsample to 2000 stratified points for fitting,
        # then transform() the full train/test sets.
        _ISOMAP_FIT = 2000
        X_tr = (self.X_train.values
                if hasattr(self.X_train, 'values') else np.asarray(self.X_train))
        y_tr = (self.y_train.values
                if hasattr(self.y_train, 'values') else np.asarray(self.y_train))
        rng = np.random.default_rng(RANDOM_SEED)
        classes, counts = np.unique(y_tr, return_counts=True)
        sub_idx = np.concatenate([
            rng.choice(np.where(y_tr == c)[0],
                       min(cnt, max(1, int(_ISOMAP_FIT * cnt / len(y_tr)))),
                       replace=False)
            for c, cnt in zip(classes, counts)
        ])
        print(f"  Isomap fitting on {len(sub_idx)}-pt subsample...")
        isomap = Isomap(n_components=2, n_neighbors=5, n_jobs=-1).fit(X_tr[sub_idx])

        # Transform both sets
        train_fused = np.hstack([
            pca.transform(self.X_train),    # Global variance
            lda.transform(self.X_train),    # Class discrimination
            isomap.transform(self.X_train)  # Non-linear structure
        ])

        test_fused = np.hstack([
            pca.transform(self.X_test),
            lda.transform(self.X_test),
            isomap.transform(self.X_test)
        ])

        self.X_train = train_fused
        self.X_test = test_fused
        print(f"Fused representation shape: {self.X_train.shape}")
        
        return self

    def stage3_autoencoder(self, bottleneck_dim=3):
        print("\n--- Stage 3: Autoencoder Compression ---")

        class Autoencoder(nn.Module):
            def __init__(self, input_dim, bottleneck_dim):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, 4),
                    nn.BatchNorm1d(4),
                    nn.ReLU(),
                    nn.Linear(4, bottleneck_dim)
                )
                self.decoder = nn.Sequential(
                    nn.Linear(bottleneck_dim, 4),
                    nn.BatchNorm1d(4),
                    nn.ReLU(),
                    nn.Linear(4, input_dim),
                    nn.Sigmoid()
                )
                
            def forward(self, x):
                z = self.encoder(x)
                return self.decoder(z), z
            
            @torch.no_grad()
            def encode(self, x):
                return self.encoder(x).cpu().numpy()

        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            input_dim = self.X_train.shape[1]
            
            # Convert to tensors
            X_train_np = self.X_train.values if hasattr(self.X_train, 'values') else np.asarray(self.X_train)
            X_test_np = self.X_test.values if hasattr(self.X_test, 'values') else np.asarray(self.X_test)
            
            X_train_t = torch.FloatTensor(X_train_np).to(device)
            X_test_t = torch.FloatTensor(X_test_np).to(device)
            
            # Create data loader
            dataset = TensorDataset(X_train_t, X_train_t)
            loader = DataLoader(dataset, batch_size=32, shuffle=True)
            
            model = Autoencoder(input_dim, bottleneck_dim).to(device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            print("Training autoencoder...")
            best_loss = float('inf')
            best_state = None
            patience = 5
            patience_cnt = 0
            
            # Train
            for epoch in range(100):
                model.train()
                epoch_loss = 0.0
                
                for xb, _ in loader:
                    optimizer.zero_grad()
                    x_hat, _ = model(xb)
                    loss = criterion(x_hat, xb)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                
                avg_loss = epoch_loss / len(loader)
                
                # Early stopping
                if avg_loss < best_loss - 1e-5:
                    best_loss = avg_loss
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    patience_cnt = 0
                else:
                    patience_cnt += 1
                    if patience_cnt >= patience:
                        print(f"    Early stop at epoch {epoch+1}")
                        break
            
            # Load best weights
            if best_state:
                model.load_state_dict(best_state)
            
            model.eval()
            self.X_train = model.encode(X_train_t)
            self.X_test = model.encode(X_test_t)
            print(f"Final reduced dimension: {self.X_train.shape[1]}")
            
        except Exception as e:
            print(f"Autoencoder unavailable: {e}. Using fused features directly.")
        
        return self

    def stage4_ensemble_classifier(self):
        print("\n--- Stage 4: Ensemble Classifier ---")

        # Simple, effective ensemble - 2 good models are better than 3 bad ones
        rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
        
        if _has_xgboost:
            secondary = XGBClassifier(n_estimators=100, random_state=RANDOM_SEED, eval_metric='logloss', verbosity=0)
        else:
            secondary = GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_SEED)

        ensemble = VotingClassifier(
            estimators=[('rf', rf), ('secondary', secondary)],
            voting='soft',
            weights=[0.4, 0.6]
        )

        ensemble.fit(self.X_train, self.y_train)
        y_pred = ensemble.predict(self.X_test)
        y_proba = ensemble.predict_proba(self.X_test)[:, 1]

        self.results['MSHDR'] = {
            'Accuracy': accuracy_score(self.y_test, y_pred),
            'Precision': precision_score(self.y_test, y_pred),
            'Recall': recall_score(self.y_test, y_pred),
            'F1-Score': f1_score(self.y_test, y_pred),
            'AUC-ROC': roc_auc_score(self.y_test, y_proba)
        }
        
        print(f"MSHDR Accuracy: {self.results['MSHDR']['Accuracy']:.4f}")
        return self.results

    def run_all(self):
        print("\n" + "="*80)
        print("PHASE 3: MSHDR PIPELINE")
        print("="*80)

        self.stage1_feature_selection()
        self.stage2_manifold_fusion()
        self.stage3_autoencoder()
        self.stage4_ensemble_classifier()
        
        return self.results
