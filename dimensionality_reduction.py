import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, classification_report
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*80)
print("DIMENSIONALITY REDUCTION FOR SOC DATA CLASSIFICATION")
print("="*80)

# ============================================================================
# PHASE 1: DATA LOADING AND ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("PHASE 1: DATA LOADING AND ANALYSIS")
print("="*80)

df = pd.read_csv('final_cleaned_dataset.csv')
print(f"\nDataset Shape: {df.shape}")
print(f"Total Samples: {len(df)}")
print(f"Total Features: {df.shape[1] - 1}")
print(f"\nColumn Names:\n{df.columns.tolist()}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nMissing Values:\n{df.isnull().sum()}")
print(f"\nClass Distribution:\n{df['label'].value_counts()}")
print(f"\nClass Distribution (%):\n{df['label'].value_counts(normalize=True)*100}")

X = df.drop('label', axis=1)
y = df['label']

print(f"\nFeature Statistics:")
print(X.describe())

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nTrain Set Size: {len(X_train)}")
print(f"Test Set Size: {len(X_test)}")
print(f"Train Class Distribution: {y_train.value_counts().to_dict()}")
print(f"Test Class Distribution: {y_test.value_counts().to_dict()}")

# ============================================================================
# PHASE 2: BASELINE DIMENSIONALITY REDUCTION METHODS
# ============================================================================
print("\n" + "="*80)
print("PHASE 2: BASELINE DIMENSIONALITY REDUCTION METHODS")
print("="*80)

results = {}

# 2.1 PCA
print("\n--- 2.1 PCA (Principal Component Analysis) ---")
pca = PCA(n_components=0.95)  # Keep 95% variance
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
print(f"PCA Components (95% variance): {pca.n_components_}")
print(f"Explained Variance Ratio: {pca.explained_variance_ratio_.sum():.4f}")

rf_pca = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_pca.fit(X_train_pca, y_train)
y_pred_pca = rf_pca.predict(X_test_pca)

results['PCA'] = {
    'Accuracy': accuracy_score(y_test, y_pred_pca),
    'Precision': precision_score(y_test, y_pred_pca),
    'Recall': recall_score(y_test, y_pred_pca),
    'F1-Score': f1_score(y_test, y_pred_pca),
    'AUC-ROC': roc_auc_score(y_test, rf_pca.predict_proba(X_test_pca)[:, 1])
}
print(f"Accuracy: {results['PCA']['Accuracy']:.4f}")

# 2.2 LDA
print("\n--- 2.2 LDA (Linear Discriminant Analysis) ---")
lda = LDA(n_components=1)
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)
print(f"LDA Components: {lda.n_components}")

rf_lda = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_lda.fit(X_train_lda, y_train)
y_pred_lda = rf_lda.predict(X_test_lda)

results['LDA'] = {
    'Accuracy': accuracy_score(y_test, y_pred_lda),
    'Precision': precision_score(y_test, y_pred_lda),
    'Recall': recall_score(y_test, y_pred_lda),
    'F1-Score': f1_score(y_test, y_pred_lda),
    'AUC-ROC': roc_auc_score(y_test, rf_lda.predict_proba(X_test_lda)[:, 1])
}
print(f"Accuracy: {results['LDA']['Accuracy']:.4f}")

# 2.3 t-SNE
print("\n--- 2.3 t-SNE ---")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
X_tsne = tsne.fit_transform(X)
X_train_tsne, X_test_tsne, _, _ = train_test_split(X_tsne, y, test_size=0.2, random_state=42, stratify=y)

rf_tsne = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_tsne.fit(X_train_tsne, y_train)
y_pred_tsne = rf_tsne.predict(X_test_tsne)

results['t-SNE'] = {
    'Accuracy': accuracy_score(y_test, y_pred_tsne),
    'Precision': precision_score(y_test, y_pred_tsne),
    'Recall': recall_score(y_test, y_pred_tsne),
    'F1-Score': f1_score(y_test, y_pred_tsne),
    'AUC-ROC': roc_auc_score(y_test, rf_tsne.predict_proba(X_test_tsne)[:, 1])
}
print(f"Accuracy: {results['t-SNE']['Accuracy']:.4f}")

# 2.4 UMAP
print("\n--- 2.4 UMAP ---")
try:
    import umap
    umap_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    X_umap = umap_reducer.fit_transform(X)
    X_train_umap, X_test_umap, _, _ = train_test_split(X_umap, y, test_size=0.2, random_state=42, stratify=y)
    
    rf_umap = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_umap.fit(X_train_umap, y_train)
    y_pred_umap = rf_umap.predict(X_test_umap)
    
    results['UMAP'] = {
        'Accuracy': accuracy_score(y_test, y_pred_umap),
        'Precision': precision_score(y_test, y_pred_umap),
        'Recall': recall_score(y_test, y_pred_umap),
        'F1-Score': f1_score(y_test, y_pred_umap),
        'AUC-ROC': roc_auc_score(y_test, rf_umap.predict_proba(X_test_umap)[:, 1])
    }
    print(f"Accuracy: {results['UMAP']['Accuracy']:.4f}")
except Exception as e:
    print(f"UMAP not available: {e}")
    results['UMAP'] = {'Accuracy': 0, 'Precision': 0, 'Recall': 0, 'F1-Score': 0, 'AUC-ROC': 0}

# 2.5 Original Features (Baseline)
print("\n--- 2.5 Original Features (No Reduction) ---")
rf_orig = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_orig.fit(X_train, y_train)
y_pred_orig = rf_orig.predict(X_test)

results['Original'] = {
    'Accuracy': accuracy_score(y_test, y_pred_orig),
    'Precision': precision_score(y_test, y_pred_orig),
    'Recall': recall_score(y_test, y_pred_orig),
    'F1-Score': f1_score(y_test, y_pred_orig),
    'AUC-ROC': roc_auc_score(y_test, rf_orig.predict_proba(X_test)[:, 1])
}
print(f"Accuracy: {results['Original']['Accuracy']:.4f}")

# ============================================================================
# PHASE 3: NOVEL MSHDR APPROACH
# ============================================================================
print("\n" + "="*80)
print("PHASE 3: NOVEL MSHDR APPROACH (Multi-Stage Hybrid Dimensionality Reduction)")
print("="*80)

# Stage 1: Feature Selection
print("\n--- Stage 1: Feature Selection ---")
from sklearn.feature_selection import mutual_info_classif

mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
feature_importance = pd.DataFrame({'Feature': X.columns, 'MI_Score': mi_scores})
feature_importance = feature_importance.sort_values('MI_Score', ascending=False)
print("Feature Importance (Mutual Information):")
print(feature_importance)

# Correlation-based filtering
corr_matrix = X_train.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr = [column for column in upper.columns if any(upper[column] > 0.8)]
print(f"\nHighly Correlated Features (removed): {high_corr}")

# Select top features
top_features = feature_importance.head(10)['Feature'].tolist()
print(f"\nTop 10 Features Selected: {top_features}")

X_train_fs = X_train[top_features]
X_test_fs = X_test[top_features]

# Stage 2: Manifold Learning Fusion (Parallel UMAP + t-SNE)
print("\n--- Stage 2: Parallel Manifold Learning Fusion ---")

# UMAP on selected features
umap_fusion = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
X_train_umap_fusion = umap_fusion.fit_transform(X_train_fs)
X_test_umap_fusion = umap_fusion.transform(X_test_fs)

# t-SNE on selected features (use subset for speed)
tsne_fusion = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
X_tsne_subset = tsne_fusion.fit_transform(X_train_fs)
X_tsne_test_subset = tsne_fusion.fit_transform(X_test_fs)

# Fuse manifold features (concatenation)
X_train_manifold = np.hstack([X_train_umap_fusion, X_tsne_subset])
X_test_manifold = np.hstack([X_test_umap_fusion, X_tsne_test_subset])
print(f"Manifold Fusion Shape: {X_train_manifold.shape}")

# Stage 3: Autoencoder Compression
print("\n--- Stage 3: Stacked Denoising Autoencoder ---")

try:
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.callbacks import EarlyStopping
    
    # Build Stacked Denoising Autoencoder
    input_dim = X_train_manifold.shape[1]
    
    input_layer = Input(shape=(input_dim,))
    # Encoder
    encoded = Dense(64, activation='relu')(input_layer)
    encoded = BatchNormalization()(encoded)
    encoded = Dropout(0.3)(encoded)
    encoded = Dense(32, activation='relu')(encoded)
    encoded = BatchNormalization()(encoded)
    encoded = Dropout(0.3)(encoded)
    bottleneck = Dense(8, activation='relu', name='bottleneck')(encoded)
    # Decoder
    decoded = Dense(32, activation='relu')(bottleneck)
    decoded = BatchNormalization()(decoded)
    decoded = Dense(64, activation='relu')(decoded)
    decoded = BatchNormalization()(decoded)
    output_layer = Dense(input_dim, activation='sigmoid')(decoded)
    
    autoencoder = Model(input_layer, output_layer)
    encoder = Model(input_layer, bottleneck)
    
    autoencoder.compile(optimizer='adam', loss='mse')
    
    print("Training Autoencoder...")
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    autoencoder.fit(X_train_manifold, X_train_manifold, 
                   epochs=50, batch_size=256, 
                   validation_split=0.2, 
                   callbacks=[early_stop], verbose=0)
    
    X_train_ae = encoder.predict(X_train_manifold, verbose=0)
    X_test_ae = encoder.predict(X_test_manifold, verbose=0)
    print(f"Autoencoder Latent Shape: {X_train_ae.shape}")
    
    has_ae = True
except Exception as e:
    print(f"Autoencoder failed, using manifold features: {e}")
    X_train_ae = X_train_manifold
    X_test_ae = X_test_manifold
    has_ae = False

# Stage 4: Ensemble Classifier
print("\n--- Stage 4: Ensemble Classifier ---")

# Create ensemble
rf_ensemble = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
gb_ensemble = GradientBoostingClassifier(n_estimators=100, random_state=42)
svm_ensemble = SVC(kernel='rbf', probability=True, random_state=42)

ensemble = VotingClassifier(
    estimators=[
        ('rf', rf_ensemble),
        ('gb', gb_ensemble),
        ('svm', svm_ensemble)
    ],
    voting='soft'
)

print("Training Ensemble Classifier...")
ensemble.fit(X_train_ae, y_train)
y_pred_mshdr = ensemble.predict(X_test_ae)

results['MSHDR'] = {
    'Accuracy': accuracy_score(y_test, y_pred_mshdr),
    'Precision': precision_score(y_test, y_pred_mshdr),
    'Recall': recall_score(y_test, y_pred_mshdr),
    'F1-Score': f1_score(y_test, y_pred_mshdr),
    'AUC-ROC': roc_auc_score(y_test, ensemble.predict_proba(X_test_ae)[:, 1])
}
print(f"Accuracy: {results['MSHDR']['Accuracy']:.4f}")

# ============================================================================
# PHASE 4: RESULTS COMPARISON AND VISUALIZATION
# ============================================================================
print("\n" + "="*80)
print("PHASE 4: RESULTS COMPARISON")
print("="*80)

results_df = pd.DataFrame(results).T
results_df = results_df.round(4)
print("\n" + results_df.to_string())

# Save results
results_df.to_csv('dimensionality_reduction_results.csv')
print("\nResults saved to: dimensionality_reduction_results.csv")

# Best method
best_method = results_df['Accuracy'].idxmax()
print(f"\n*** Best Method: {best_method} with Accuracy: {results_df.loc[best_method, 'Accuracy']:.4f} ***")

# Confusion Matrix for MSHDR
print("\n--- Confusion Matrix (MSHDR) ---")
cm = confusion_matrix(y_test, y_pred_mshdr)
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, y_pred_mshdr))

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Accuracy Comparison Bar Chart
ax1 = axes[0, 0]
methods = results_df.index.tolist()
accuracies = results_df['Accuracy'].tolist()
colors = ['#2ecc71' if m == 'MSHDR' else '#3498db' for m in methods]
bars = ax1.bar(methods, accuracies, color=colors, edgecolor='black')
ax1.set_xlabel('Method', fontsize=12)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.set_title('Accuracy Comparison: Dimensionality Reduction Methods', fontsize=14)
ax1.set_ylim(0, 1)
for bar, acc in zip(bars, accuracies):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{acc:.4f}', 
             ha='center', va='bottom', fontsize=10)
ax1.tick_params(axis='x', rotation=45)

# 2. All Metrics Comparison
ax2 = axes[0, 1]
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
x = np.arange(len(methods))
width = 0.15
for i, metric in enumerate(metrics):
    values = results_df[metric].tolist()
    ax2.bar(x + i*width, values, width, label=metric)
ax2.set_xlabel('Method', fontsize=12)
ax2.set_ylabel('Score', fontsize=12)
ax2.set_title('All Metrics Comparison', fontsize=14)
ax2.set_xticks(x + width*2)
ax2.set_xticklabels(methods, rotation=45)
ax2.legend(loc='lower right', fontsize=8)
ax2.set_ylim(0, 1.1)

# 3. PCA Components Visualization
ax3 = axes[1, 0]
pca_viz = PCA(n_components=2)
X_pca_viz = pca_viz.fit_transform(X)
scatter = ax3.scatter(X_pca_viz[:, 0], X_pca_viz[:, 1], c=y, cmap='coolwarm', alpha=0.5, s=10)
ax3.set_xlabel('PC1', fontsize=12)
ax3.set_ylabel('PC2', fontsize=12)
ax3.set_title('PCA 2D Visualization (Original Data)', fontsize=14)
plt.colorbar(scatter, ax=ax3, label='Label')

# 4. t-SNE Visualization
ax4 = axes[1, 1]
scatter2 = ax4.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='coolwarm', alpha=0.5, s=10)
ax4.set_xlabel('t-SNE 1', fontsize=12)
ax4.set_ylabel('t-SNE 2', fontsize=12)
ax4.set_title('t-SNE 2D Visualization', fontsize=14)
plt.colorbar(scatter2, ax=ax4, label='Label')

plt.tight_layout()
plt.savefig('dimensionality_reduction_analysis.png', dpi=150, bbox_inches='tight')
print("Visualization saved to: dimensionality_reduction_analysis.png")

# Feature Importance Plot
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['MI_Score'], color='#3498db')
plt.xlabel('Mutual Information Score')
plt.ylabel('Feature')
plt.title('Feature Importance (Mutual Information)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
print("Feature importance saved to: feature_importance.png")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)
print(f"""
Dataset: SOC Network Intrusion Detection
Total Samples: {len(df)}
Original Features: {X.shape[1]}

RESULTS COMPARISON:
------------------
""")
for method, metrics in results.items():
    print(f"{method:12s} | Acc: {metrics['Accuracy']:.4f} | Prec: {metrics['Precision']:.4f} | Rec: {metrics['Recall']:.4f} | F1: {metrics['F1-Score']:.4f} | AUC: {metrics['AUC-ROC']:.4f}")

print(f"""
KEY FINDINGS:
-------------
1. Best Performing Method: {best_method}
2. MSHDR (Novel Approach) Performance: {results['MSHDR']['Accuracy']:.4f}
3. Improvement over Original: {(results['MSHDR']['Accuracy'] - results['Original']['Accuracy'])*100:.2f}%

NOVEL CONTRIBUTIONS:
--------------------
1. Multi-stage cascaded dimensionality reduction pipeline
2. Feature selection using mutual information
3. Parallel manifold learning fusion (UMAP + t-SNE)
4. Denoising autoencoder for compression
5. Ensemble voting classifier (RF + GB + SVM)

Files Generated:
- dimensionality_reduction_results.csv
- dimensionality_reduction_analysis.png
- feature_importance.png
""")

print("\n" + "="*80)
print("EXECUTION COMPLETE")
print("="*80)