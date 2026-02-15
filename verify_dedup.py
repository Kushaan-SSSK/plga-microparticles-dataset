"""
Verification script: compute formulation-level (deduplicated) statistics
for all targets to determine correct numbers for the manuscript.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, f1_score, confusion_matrix
from collections import Counter

preds = pd.read_csv('all_predictions_and_uncertainty.csv')
perf = pd.read_csv('performance_metrics.csv')
bench = pd.read_csv('benchmark_results.csv')

print("=" * 70)
print("RAW (TIMEPOINT-LEVEL) COUNTS")
print("=" * 70)
for target in ['Peppas_n', 'Peppas_K', 'Burst_24h']:
    d = preds[preds['Target'] == target]
    print(f"  {target}: {len(d)} rows")

print("\n" + "=" * 70)
print("DEDUPLICATION: formulation-level")
print("=" * 70)

# Deduplicate: each formulation has identical (Actual, Predicted, Target)
# The Leverage column varies per timepoint within a formulation, so we
# need to aggregate it (mean leverage per formulation).
preds_form = (
    preds
    .groupby(['Actual', 'Predicted', 'Target'], as_index=False)
    .agg({
        'Residuals': 'first',
        'Uncertainty': 'first',
        'Leverage': 'mean',       # mean leverage across timepoints
        'Std_Residual': 'first',
    })
)

for target in ['Peppas_n', 'Peppas_K', 'Burst_24h']:
    d = preds_form[preds_form['Target'] == target]
    print(f"  {target}: {len(d)} unique formulations")

print("\n" + "=" * 70)
print("ISSUE 1: Target summary stats (formulation-level)")
print("=" * 70)
for target in ['Peppas_n', 'Peppas_K', 'Burst_24h']:
    d = preds_form[preds_form['Target'] == target]
    vals = d['Actual']
    print(f"  {target}: N={len(vals)}, mean={vals.mean():.3f} ± {vals.std():.3f}, "
          f"min={vals.min():.3f}, max={vals.max():.3f}")

print("\n" + "=" * 70)
print("ISSUE 2: AD leverage threshold (formulation-level)")
print("=" * 70)
p = 15
for target in ['Peppas_n', 'Peppas_K', 'Burst_24h']:
    d = preds_form[preds_form['Target'] == target].copy()
    n_form = len(d)
    h_star = 3 * p / n_form
    safe = d[d['Leverage'] < h_star]
    unsafe = d[d['Leverage'] >= h_star]
    
    r2_safe = r2_score(safe['Actual'], safe['Predicted']) if len(safe) > 10 else np.nan
    r2_unsafe = r2_score(unsafe['Actual'], unsafe['Predicted']) if len(unsafe) > 10 else np.nan
    mae_safe = mean_absolute_error(safe['Actual'], safe['Predicted']) if len(safe) > 10 else np.nan
    mae_unsafe = mean_absolute_error(unsafe['Actual'], unsafe['Predicted']) if len(unsafe) > 10 else np.nan
    
    print(f"\n  {target}: N={n_form}, h*={h_star:.4f}")
    print(f"    In-domain:     {len(safe):4d} formulations, R²={r2_safe:.3f}, MAE={mae_safe:.3f}")
    print(f"    High-leverage: {len(unsafe):4d} formulations, R²={r2_unsafe:.3f}, MAE={mae_unsafe:.3f}")

print("\n" + "=" * 70)
print("ISSUE 3: Burst class distribution (formulation-level)")
print("=" * 70)
db = preds_form[preds_form['Target'] == 'Burst_24h']
burst_vals = db['Actual'].values
y_true = np.zeros(len(burst_vals), dtype=int)
y_true[(burst_vals >= 0.10) & (burst_vals < 0.40)] = 1
y_true[burst_vals >= 0.40] = 2

dist = Counter(y_true)
n_total = len(burst_vals)
print(f"  Total formulations: {n_total}")
print(f"  Low  (<10%):  {dist[0]:4d}  ({100*dist[0]/n_total:.1f}%)")
print(f"  Med  (10-40%): {dist[1]:4d}  ({100*dist[1]/n_total:.1f}%)")
print(f"  High (>=40%):  {dist[2]:4d}  ({100*dist[2]/n_total:.1f}%)")

# Confusion matrix (assuming all predicted as majority class)
y_pred = np.full_like(y_true, 2)
cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
print(f"\n  Confusion matrix:\n{cm}")
print(f"  Macro-F1: {macro_f1:.3f}")

print("\n" + "=" * 70)
print("ISSUE 4: Table 2 vs Table 3 comparison")
print("=" * 70)
print("\n  performance_metrics.csv (Table 2):")
for _, row in perf.iterrows():
    if pd.notna(row['R2']):
        print(f"    {row['Target']}: R²={row['R2']:.3f}, MAE={row['MAE']:.3f}, RMSE={row['RMSE']:.3f}")

print("\n  benchmark_results.csv (Table 3) — StackedEnsemble only:")
for _, row in bench[bench['Model'] == 'StackedEnsemble'].iterrows():
    print(f"    {row['Target']}: R²={row['R2']:.3f}, MAE={row['MAE']:.3f}")

print("\n  DISCREPANCY for Peppas_n:")
t2_r2 = perf[perf['Target'] == 'Peppas_n']['R2'].values[0]
t3_r2 = bench[(bench['Target'] == 'Peppas_n') & (bench['Model'] == 'StackedEnsemble')]['R2'].values[0]
print(f"    Table 2: R²={t2_r2:.3f}")
print(f"    Table 3: R²={t3_r2:.3f}")
print(f"    Delta:   {abs(t2_r2 - t3_r2):.3f}")

print("\n" + "=" * 70)
print("DONE — Use formulation-level numbers above to update manuscript")
print("=" * 70)
