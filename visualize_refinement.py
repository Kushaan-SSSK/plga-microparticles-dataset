
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import warnings

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

def main():
    print("=== Generating Refined Visualizations ===")
    
    # 1. Uncertainty Analysis (Peppas n)
    try:
        preds = pd.read_csv('all_predictions_and_uncertainty.csv')
        df_n = preds[preds['Target'] == 'Peppas_n'].copy()
        
        if not df_n.empty:
            plt.figure(figsize=(8,6))
            df_n['AbsError'] = (df_n['Actual'] - df_n['Predicted']).abs()
            
            # Scatter plot with regression line
            sns.regplot(data=df_n, x='Uncertainty', y='AbsError', 
                        scatter_kws={'alpha':0.3, 'color':'blue'}, line_kws={'color':'red'})
            
            # Calculate correlation
            corr = df_n['Uncertainty'].corr(df_n['AbsError'])
            
            plt.title(f'Uncertainty Calibration (Peppas n)\nCorrelation: {corr:.2f}')
            plt.xlabel('Ensemble Uncertainty (Std Dev)')
            plt.ylabel('Absolute Prediction Error')
            plt.tight_layout()
            plt.savefig('Figure4_UncertaintyCalibration.png')
            print("Generated Figure4_UncertaintyCalibration.png")
            
        # 2. Burst Confusion Matrix
        df_b = preds[preds['Target'] == 'Burst_Class'].copy()
        if not df_b.empty:
            plt.figure(figsize=(6,5))
            cm = confusion_matrix(df_b['Actual'], df_b['Predicted'])
            # Normalize by row (True Class)
            cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            sns.heatmap(cmn, annot=True, fmt='.2f', cmap='Blues', 
                        xticklabels=['Low (<10%)', 'Med', 'High (>40%)'],
                        yticklabels=['Low', 'Med', 'High'])
            plt.title('Burst Release Classification Accuracy')
            plt.xlabel('Predicted Class')
            plt.ylabel('Actual Class')
            plt.tight_layout()
            plt.savefig('Figure5_BurstClassification.png')
            print("Generated Figure5_BurstClassification.png")
            
    except FileNotFoundError:
        print("all_predictions_and_uncertainty.csv not found.")

    # 3. Benchmarking
    try:
        bench = pd.read_csv('benchmark_results.csv')
        # Filter to relevant targets
        bench = bench[bench['Target'].isin(['Peppas_n', 'Peppas_K'])]
        
        plt.figure(figsize=(10,6))
        sns.barplot(data=bench, x='Target', y='R2', hue='Model', palette='viridis')
        plt.axhline(0, color='k', linestyle='--', linewidth=1)
        plt.title('Rigorous Benchmarking: Stacked Ensemble vs Baselines')
        plt.ylabel('R2 Score (10-Fold CV)')
        plt.tight_layout()
        plt.savefig('Figure6_Benchmarking.png')
        print("Generated Figure6_Benchmarking.png")
    except FileNotFoundError:
        print("benchmark_results.csv not found.")

if __name__ == "__main__":
    main()
