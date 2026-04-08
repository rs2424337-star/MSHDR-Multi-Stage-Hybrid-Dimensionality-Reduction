import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from src.utils.config import VIS_DIR

# Set consistent style for all plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class Visualizer:
    def __init__(self, results_df, y_test, X_all, y_all):
        self.results_df = results_df
        self.y_test = y_test
        self.X_all = X_all
        self.y_all = y_all

    def plot_accuracy_comparison(self, ax):
        methods = self.results_df.index.tolist()
        accuracies = self.results_df['Accuracy']
        
        # Highlight MSHDR in green for comparison
        colors = ['#27ae60' if m == 'MSHDR' else '#3498db' for m in methods]
        bars = ax.bar(methods, accuracies, color=colors, edgecolor='black', alpha=0.8)
        
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy Comparison')
        ax.set_ylim(0.5, 1.05)  # Zoom in on relevant range
        
        # Add value labels on top
        for bar, acc in zip(bars, accuracies):
            ax.text(
                bar.get_x() + bar.get_width()/2, 
                bar.get_height() + 0.005,
                f'{acc:.4f}', 
                ha='center', 
                va='bottom',
                fontsize=9
            )
            
        ax.tick_params(axis='x', rotation=45)

    def plot_metrics_comparison(self, ax):
        methods = self.results_df.index.tolist()
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        x = np.arange(len(methods))
        width = 0.2

        for i, metric in enumerate(metrics):
            ax.bar(x + i*width, self.results_df[metric], width, label=metric, alpha=0.8)

        ax.set_ylabel('Score')
        ax.set_title('Performance Metrics')
        ax.set_xticks(x + width*1.5)
        ax.set_xticklabels(methods, rotation=45)
        ax.legend(loc='lower right', fontsize=8)
        ax.set_ylim(0.5, 1.05)

    def plot_pca_2d(self, ax):
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(self.X_all)
        
        scatter = ax.scatter(
            X_pca[:, 0], X_pca[:, 1], 
            c=self.y_all, 
            cmap='coolwarm', 
            alpha=0.4, 
            s=8,
            edgecolors='none'
        )
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        ax.set_title('PCA 2D Projection')

    def plot_tsne_2d(self, ax):
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_jobs=-1)
        X_tsne = tsne.fit_transform(self.X_all)
        
        scatter = ax.scatter(
            X_tsne[:, 0], X_tsne[:, 1], 
            c=self.y_all, 
            cmap='coolwarm', 
            alpha=0.4, 
            s=8,
            edgecolors='none'
        )
        
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.set_title('t-SNE 2D Projection')

    def plot_feature_importance(self, feature_importance_df):
        plt.figure(figsize=(10, 6))
        
        # Horizontal bar plot - highest at top
        bars = plt.barh(
            feature_importance_df['feature'], 
            feature_importance_df['mi_score'], 
            color='#3498db',
            alpha=0.8
        )
        
        plt.xlabel('Mutual Information Score')
        plt.ylabel('Feature')
        plt.title('Feature Importance Ranking')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        plt.savefig(f'{VIS_DIR}/feature_importance.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ Feature importance plot saved")

    def plot_confusion_matrix(self, y_pred, method_name='MSHDR'):
        cm = confusion_matrix(self.y_test, y_pred)
        
        plt.figure(figsize=(7, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['Normal', 'Attack'],
            yticklabels=['Normal', 'Attack'],
            cbar=False
        )
        
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {method_name}')
        plt.tight_layout()
        
        plt.savefig(f'{VIS_DIR}/confusion_matrix_{method_name.lower()}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Confusion matrix saved for {method_name}")

    def generate_all(self, y_pred_mshdr=None, feature_importance_df=None):
        print("\n" + "="*80)
        print("PHASE 4: GENERATING VISUALIZATIONS")
        print("="*80)

        # Main comparison figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('Dimensionality Reduction Performance Analysis', fontsize=16, y=0.95)
        
        self.plot_accuracy_comparison(axes[0, 0])
        self.plot_metrics_comparison(axes[0, 1])
        self.plot_pca_2d(axes[1, 0])
        self.plot_tsne_2d(axes[1, 1])
        
        plt.tight_layout()
        plt.savefig(f'{VIS_DIR}/performance_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ Performance comparison plot saved")

        if feature_importance_df is not None:
            self.plot_feature_importance(feature_importance_df)

        if y_pred_mshdr is not None:
            self.plot_confusion_matrix(y_pred_mshdr)
