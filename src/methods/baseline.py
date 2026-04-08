from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
import umap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.utils.config import RANDOM_SEED, PCA_VARIANCE


class BaselineMethods:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.results = {}
        
        # Use same classifier for all baselines for fair comparison
        self.classifier = RandomForestClassifier(
            n_estimators=100, 
            random_state=RANDOM_SEED, 
            n_jobs=-1
        )

    def _evaluate(self, X_train_red, X_test_red, method_name):
        """Evaluate reduced features with consistent classifier and store results"""
        self.classifier.fit(X_train_red, self.y_train)
        y_pred = self.classifier.predict(X_test_red)
        y_proba = self.classifier.predict_proba(X_test_red)[:, 1]

        self.results[method_name] = {
            'Accuracy': accuracy_score(self.y_test, y_pred),
            'Precision': precision_score(self.y_test, y_pred),
            'Recall': recall_score(self.y_test, y_pred),
            'F1-Score': f1_score(self.y_test, y_pred),
            'AUC-ROC': roc_auc_score(self.y_test, y_proba)
        }
        
        print(f"{method_name} Accuracy: {self.results[method_name]['Accuracy']:.4f}")
        return self.results[method_name]

    def run_pca(self):
        print("\n--- PCA (Principal Component Analysis) ---")
        pca = PCA(n_components=PCA_VARIANCE, random_state=RANDOM_SEED)
        X_train_pca = pca.fit_transform(self.X_train)
        X_test_pca = pca.transform(self.X_test)
        
        print(f"PCA Components (95% variance): {pca.n_components_}")
        print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.4f}")
        return self._evaluate(X_train_pca, X_test_pca, 'PCA')

    def run_lda(self):
        print("\n--- LDA (Linear Discriminant Analysis) ---")
        lda = LDA(n_components=1)
        X_train_lda = lda.fit_transform(self.X_train, self.y_train)
        X_test_lda = lda.transform(self.X_test)
        return self._evaluate(X_train_lda, X_test_lda, 'LDA')

    def run_tsne(self):
        print("\n--- t-SNE ---")
        # Note: t-SNE is only for visualization - bad for predictive tasks.
        # Subsampled to 5000 pts: t-SNE is O(n²) and already scores ~58% on this dataset.
        import numpy as np
        rng = np.random.default_rng(RANDOM_SEED)
        max_pts = 5000
        n_tr = len(self.X_train)
        n_te = len(self.X_test)
        tr_idx = rng.choice(n_tr, min(max_pts, n_tr), replace=False)
        te_idx = rng.choice(n_te, min(max_pts, n_te), replace=False)
        tsne = TSNE(n_components=2, random_state=RANDOM_SEED, perplexity=30, max_iter=500)
        X_train_tsne = tsne.fit_transform(self.X_train.iloc[tr_idx] if hasattr(self.X_train, 'iloc') else self.X_train[tr_idx])
        X_test_tsne  = tsne.fit_transform(self.X_test.iloc[te_idx]  if hasattr(self.X_test,  'iloc') else self.X_test[te_idx])
        # Evaluate on subsample only (t-SNE cannot transform new points)
        y_train_sub = self.y_train.iloc[tr_idx] if hasattr(self.y_train, 'iloc') else self.y_train[tr_idx]
        y_test_sub  = self.y_test.iloc[te_idx]  if hasattr(self.y_test,  'iloc') else self.y_test[te_idx]
        from sklearn.ensemble import RandomForestClassifier as _RF
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        clf = _RF(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
        clf.fit(X_train_tsne, y_train_sub)
        y_pred  = clf.predict(X_test_tsne)
        y_proba = clf.predict_proba(X_test_tsne)[:, 1]
        self.results['t-SNE'] = {
            'Accuracy':  accuracy_score(y_test_sub, y_pred),
            'Precision': precision_score(y_test_sub, y_pred),
            'Recall':    recall_score(y_test_sub, y_pred),
            'F1-Score':  f1_score(y_test_sub, y_pred),
            'AUC-ROC':   roc_auc_score(y_test_sub, y_proba),
        }
        print(f"t-SNE Accuracy: {self.results['t-SNE']['Accuracy']:.4f}")
        return self.results['t-SNE']

    def run_umap(self):
        print("\n--- UMAP ---")
        umap_reducer = umap.UMAP(
            n_components=2, 
            random_state=RANDOM_SEED, 
            n_neighbors=15, 
            min_dist=0.1
        )
        X_train_umap = umap_reducer.fit_transform(self.X_train)
        X_test_umap = umap_reducer.transform(self.X_test)
        return self._evaluate(X_train_umap, X_test_umap, 'UMAP')

    def run_original(self):
        print("\n--- Original Features (No Reduction) ---")
        return self._evaluate(self.X_train, self.X_test, 'Original')

    def run_all(self):
        print("\n" + "="*80)
        print("PHASE 2: BASELINE DIMENSIONALITY REDUCTION METHODS")
        print("="*80)
        
        self.run_pca()
        self.run_lda()
        self.run_tsne()
        self.run_umap()
        self.run_original()
        
        return self.results
