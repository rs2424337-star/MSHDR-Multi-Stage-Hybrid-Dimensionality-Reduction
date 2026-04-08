import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import PROJECT_ROOT, RESULTS_DIR
from src.data.loader import DataLoader
from src.methods.baseline import BaselineMethods
from src.novel.mshdr import MSHDR
from src.novel.mshdr_v2 import MSHDRv2
from src.utils.visualizer import Visualizer


def main():
    print("="*80)
    print("DIMENSIONALITY REDUCTION FOR SOC DATA CLASSIFICATION")
    print("Multi-Stage Hybrid Dimensionality Reduction (MSHDR)")
    print("="*80)

    data_path = os.path.join(PROJECT_ROOT, 'final_cleaned_dataset.csv')

    # Phase 1: Load and split data
    loader = DataLoader(data_path)
    loader.load()
    loader.split()

    # Phase 2: Run baseline methods
    baseline = BaselineMethods(
        loader.X_train, loader.X_test,
        loader.y_train, loader.y_test
    )
    baseline_results = baseline.run_all()

    # Phase 3a: Run original MSHDR v1 (for comparison)
    mshdr = MSHDR(
        loader.X_train, loader.X_test,
        loader.y_train, loader.y_test
    )
    mshdr_results = mshdr.run_all()

    # Phase 3b: Run MSHDR v2 (SCMAA — novel approach)
    mshdr_v2 = MSHDRv2(
        loader.X_train, loader.X_test,
        loader.y_train, loader.y_test
    )
    mshdr_v2_results = mshdr_v2.run_all()

    # Combine all results
    all_results = {**baseline_results, **mshdr_results, **mshdr_v2_results}
    results_df = pd.DataFrame(all_results).T
    results_df = results_df.round(4)

    # Save results
    results_df.to_csv(f'{RESULTS_DIR}/dimensionality_reduction_results.csv')
    print("\n✓ Results saved to results/dimensionality_reduction_results.csv")

    # Best method
    best_method = results_df['Accuracy'].idxmax()
    print(f"\n*** Best Method: {best_method} with Accuracy: {results_df.loc[best_method, 'Accuracy']:.4f} ***")

    # Capture MSHDR v2 predictions for confusion matrix
    y_pred_mshdr = getattr(mshdr_v2, '_last_y_pred', None)

    # Phase 4: Visualization
    visualizer = Visualizer(results_df, loader.y_test, loader.X, loader.y)
    visualizer.generate_all(y_pred_mshdr=y_pred_mshdr)

    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"""
Dataset: SOC Network Intrusion Detection
Total Samples: {len(loader.df)}
Original Features: {loader.X.shape[1]}

RESULTS COMPARISON:
------------------
""")
    for method, metrics in all_results.items():
        print(f"{method:12s} | Acc: {metrics['Accuracy']:.4f} | Prec: {metrics['Precision']:.4f} | Rec: {metrics['Recall']:.4f} | F1: {metrics['F1-Score']:.4f} | AUC: {metrics['AUC-ROC']:.4f}")

    print(f"""
KEY FINDINGS:
-------------
1. Best Performing Method: {best_method}
2. MSHDR (Novel Approach) Performance: {all_results.get('MSHDR', {}).get('Accuracy', 'N/A')}

NOVEL CONTRIBUTIONS (MSHDR v2 / SCMAA):
----------------------------------------
1. Adaptive Ensemble Feature Selection (MI + ANOVA + RF importance, elbow cutoff)
2. Fisher-Weighted Manifold Fusion     (J=tr(S_W^-1 S_B) drives attention weights)
3. Supervised Contrastive VAE          (L=Recon+β(t)KL+SupCon+ClassHead)
4. β-Annealing curriculum learning     (reconstruction first, then regularise)
5. Hierarchical Stacking Ensemble      (OOF meta-learner, no fixed weights)

Files Generated:
- results/dimensionality_reduction_results.csv
- visualizations/dimensionality_reduction_analysis.png
- visualizations/feature_importance.png
- visualizations/confusion_matrix_mshdr.png
""")

    print("\n" + "="*80)
    print("EXECUTION COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
