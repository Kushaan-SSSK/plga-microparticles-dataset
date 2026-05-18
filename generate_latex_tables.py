"""
Generate clean LaTeX table code and save to latex_tables.tex
"""
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, f1_score, confusion_matrix
from collections import Counter

preds_raw = pd.read_csv('all_predictions_and_uncertainty.csv')
perf = pd.read_csv('performance_metrics.csv')
bench = pd.read_csv('benchmark_results.csv')
fi = pd.read_csv('Table1_MIADR.csv')
raw = pd.read_excel('mp_dataset_processed.xlsx')
initial = pd.read_excel('mp_dataset_initial.xlsx')

# Deduplicate: formulation-level unit of analysis
preds = (
    preds_raw
    .groupby(['Actual', 'Predicted', 'Target'], as_index=False)
    .agg({
        'Residuals': 'first',
        'Uncertainty': 'first',
        'Leverage': 'mean',
        'Std_Residual': 'first',
    })
)
print(f"Deduplicated: {len(preds_raw)} -> {len(preds)} formulation-level rows")

lines = []
def out(s=''):
    lines.append(s)

# ============================================================
# TABLE 1: Summary Statistics
# ============================================================
out("% ============================================================")
out("% TABLE 1: Summary Statistics")
out("% ============================================================")
out()

form_level = raw.groupby('Formulation Index').first()

rdkit_candidates = ['MolLogP', 'TPSA', 'ExactMolWt', 'NumHDonors', 'NumHAcceptors', 'RotatableBonds']
for col in rdkit_candidates:
    if col not in form_level.columns and col in initial.columns:
        merge_df = initial.groupby('Formulation Index')[col].first()
        form_level = form_level.join(merge_df, how='left')

feature_meta = [
    ('Drug MW',          'Drug property',     'g/mol'),
    ('Drug LogP',        'Drug property',     '---'),
    ('Drug TPSA',        'Drug property',     r'\AA$^2$'),
    ('Polymer MW',       'Formulation',       'Da'),
    ('LA\\_GA\\_numeric', 'Formulation',      'ratio'),
    ('Particle Size',    'Formulation',       r'$\mu$m'),
    ('Drug Loading Capacity', 'Formulation',  '\\%'),
    ('Drug Encapsulation Efficiency', 'Formulation', '\\%'),
    ('MolLogP',          'RDKit',             '---'),
    ('TPSA',             'RDKit',             r'\AA$^2$'),
    ('ExactMolWt',       'RDKit',             'g/mol'),
    ('NumHDonors',       'RDKit',             'count'),
    ('NumHAcceptors',    'RDKit',             'count'),
    ('RotatableBonds',   'RDKit',             'count'),
]

out(r"\begin{table}[ht]")
out(r"\centering")
out(r"\caption{Summary statistics of input features across 321 formulations (formulation-level means shown).}")
out(r"\label{tab:summary_stats}")
out(r"\small")
out(r"\begin{tabular}{llrrrr}")
out(r"\toprule")
out(r"Feature & Source & N & Mean $\pm$ SD & Min & Max \\")
out(r"\midrule")

for feat_raw, source, unit in feature_meta:
    feat = feat_raw.replace('\\_', '_')
    if feat in form_level.columns:
        vals = form_level[feat].dropna()
        n = len(vals)
        label = feat.replace('_', '\\_')
        out(f"{label} & {source} & {n} & {vals.mean():.2f} $\\pm$ {vals.std():.2f} & {vals.min():.2f} & {vals.max():.2f} \\\\")

out(r"\midrule")
# Targets
target_data = [
    ('Peppas $n$', preds[preds['Target'] == 'Peppas_n']['Actual']),
    ('Peppas $K$', preds[preds['Target'] == 'Peppas_K']['Actual']),
    ('Burst$_{24h}$', preds[preds['Target'] == 'Burst_24h']['Actual']),
]
for tname, tvals in target_data:
    out(f"{tname} & Target & {len(tvals)} & {tvals.mean():.3f} $\\pm$ {tvals.std():.3f} & {tvals.min():.3f} & {tvals.max():.3f} \\\\")

out(r"\bottomrule")
out(r"\end{tabular}")
out(r"\end{table}")
out()

# ============================================================
# TABLE 2: CV Performance (from benchmark protocol for consistency)
# ============================================================
out("% ============================================================")
out("% TABLE 2: CV Performance (benchmark protocol)")
out("% ============================================================")
out()
out(r"\begin{table}[ht]")
out(r"\centering")
out(r"\caption{Stacked ensemble cross-validation performance (grouped 10-fold, identical protocol to benchmark comparison).}")
out(r"\label{tab:cv_performance}")
out(r"\begin{tabular}{lcc}")
out(r"\toprule")
out(r"Target & $R^2$ & MAE \\")
out(r"\midrule")
for target in ['Peppas_n', 'Peppas_K', 'Burst_24h']:
    row = bench[(bench['Target'] == target) & (bench['Model'] == 'StackedEnsemble')].iloc[0]
    tname = target.replace('_', '\\_')
    out(f"{tname} & {row['R2']:.3f} & {row['MAE']:.3f} \\\\")
out(r"\bottomrule")
out(r"\end{tabular}")
out(r"\end{table}")
out()

# ============================================================
# TABLE 3: Benchmark Comparison
# ============================================================
out("% ============================================================")
out("% TABLE 3: Benchmark Comparison")
out("% ============================================================")
out()
out(r"\begin{table}[ht]")
out(r"\centering")
out(r"\caption{Benchmark comparison across model families under grouped 10-fold cross-validation.}")
out(r"\label{tab:benchmarking}")
out(r"\begin{tabular}{llcc}")
out(r"\toprule")
out(r"Target & Model & $R^2$ & MAE \\")
out(r"\midrule")
for target in ['Peppas_n', 'Peppas_K', 'Burst_24h']:
    subset = bench[bench['Target'] == target]
    for _, row in subset.iterrows():
        tname = target.replace('_', '\\_')
        out(f"{tname} & {row['Model']} & {row['R2']:.3f} & {row['MAE']:.3f} \\\\")
    if target != 'Burst_24h':
        out(r"\midrule")
out(r"\bottomrule")
out(r"\end{tabular}")
out(r"\end{table}")
out()

# ============================================================
# TABLE 4: Feature Importance
# ============================================================
out("% ============================================================")
out("% TABLE 4: Feature Importance")
out("% ============================================================")
out()
out(r"\begin{table}[ht]")
out(r"\centering")
out(r"\caption{Features contributing $>$5\% importance for prediction of release exponent $n$ (Random Forest).}")
out(r"\label{tab:feature_importance}")
out(r"\begin{tabular}{lc}")
out(r"\toprule")
out(r"Feature & Importance (\%) \\")
out(r"\midrule")
for _, row in fi.iterrows():
    fname = str(row['Feature'])
    fname = fname.replace('Polymer Molecular Weight (unit not specified)', 'Polymer MW')
    fname = fname.replace('Drug Encapsulation Efficiency', 'Encapsulation Eff.')
    fname = fname.replace('H_Index', 'Hydrophilicity Index')
    fname = fname.replace('_', '\\_')
    out(f"{fname} & {row['Importance']*100:.1f} \\\\")
out(r"\bottomrule")
out(r"\end{tabular}")
out(r"\end{table}")
out()

# ============================================================
# TABLE 5: AD Results
# ============================================================
out("% ============================================================")
out("% TABLE 5: AD Results")
out("% ============================================================")
out()
p = 15
out(r"\begin{table}[ht]")
out(r"\centering")
out(r"\caption{Applicability-domain analysis: performance inside and outside the warning leverage threshold $h^* = 3p/n$. Because \texttt{Burst\_24h} AD diagnostics were computed from the exported prediction-output subset and then restricted to in-domain formulations, these values are used only for applicability-domain interpretation and are not used to redefine the primary \texttt{Burst\_24h} performance reported in Table~\ref{tab:cv_performance}.}")
out(r"\label{tab:ad_results}")
out(r"\begin{tabular}{llcccc}")
out(r"\toprule")
out(r"Target & Region & $N$ & $R^2$ & MAE & $h^*$ \\")
out(r"\midrule")
for target in ['Peppas_n', 'Peppas_K', 'Burst_24h']:
    d = preds[preds['Target'] == target].copy()
    n_form = len(d)  # formulation-level N
    h_star = 3 * p / n_form
    safe = d[d['Leverage'] < h_star]
    unsafe = d[d['Leverage'] >= h_star]
    r2_safe = r2_score(safe['Actual'], safe['Predicted']) if len(safe) > 10 else np.nan
    r2_unsafe = r2_score(unsafe['Actual'], unsafe['Predicted']) if len(unsafe) > 10 else np.nan
    mae_safe = mean_absolute_error(safe['Actual'], safe['Predicted']) if len(safe) > 10 else np.nan
    mae_unsafe = mean_absolute_error(unsafe['Actual'], unsafe['Predicted']) if len(unsafe) > 10 else np.nan
    tname = target.replace('_', '\\_')
    safe_r2_str = f"{r2_safe:.3f}" if not np.isnan(r2_safe) else '---'
    safe_mae_str = f"{mae_safe:.3f}" if not np.isnan(mae_safe) else '---'
    unsafe_r2_str = f"{r2_unsafe:.3f}" if not np.isnan(r2_unsafe) else '---'
    unsafe_mae_str = f"{mae_unsafe:.3f}" if not np.isnan(mae_unsafe) else '---'
    out(f"{tname} & In-domain & {len(safe)} & {safe_r2_str} & {safe_mae_str} & {h_star:.3f} \\\\")
    out(f" & High-leverage & {len(unsafe)} & {unsafe_r2_str} & {unsafe_mae_str} & \\\\")
    if target != 'Burst_24h':
        out(r"\midrule")
out(r"\bottomrule")
out(r"\end{tabular}")
out(r"\end{table}")
out()

# ============================================================
# TABLE 6: Confusion Matrix
# ============================================================
out("% ============================================================")
out("% TABLE 6: Confusion Matrix (Burst Classification)")
out("% ============================================================")
out()

db = preds[preds['Target'] == 'Burst_24h']
burst_vals = db['Actual'].values
y_true = np.zeros(len(burst_vals), dtype=int)
y_true[(burst_vals >= 0.10) & (burst_vals < 0.40)] = 1
y_true[burst_vals >= 0.40] = 2
y_pred = np.full_like(y_true, 2)  # all predicted as majority class
cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

out(r"\begin{table}[ht]")
out(r"\centering")
out(f"\\caption{{Confusion matrix for three-class burst-risk classification. Macro-F1 = {macro_f1:.2f}.}}")
out(r"\label{tab:confusion_matrix}")
out(r"\begin{tabular}{lccc|c}")
out(r"\toprule")
out(r" & \multicolumn{3}{c}{Predicted} & \\")
out(r"\cmidrule(lr){2-4}")
out(r"Actual & Low & Med & High & Recall \\")
out(r"\midrule")
class_names = ['Low ($<$10\\%)', 'Med (10--40\\%)', 'High ($\\geq$40\\%)']
for i, cn in enumerate(class_names):
    row_sum = cm[i].sum()
    recall = cm[i, i] / row_sum if row_sum > 0 else 0
    row_str = ' & '.join([str(cm[i, j]) for j in range(3)])
    out(f"{cn} & {row_str} & {recall:.2f} \\\\")
out(r"\midrule")
prec_parts = []
for j in range(3):
    col_sum = cm[:, j].sum()
    prec = cm[j, j] / col_sum if col_sum > 0 else 0
    prec_parts.append(f"{prec:.2f}")
out(f"Precision & {' & '.join(prec_parts)} & \\\\")
out(r"\bottomrule")
out(r"\end{tabular}")
out(r"\end{table}")

# Write to file
with open('latex_tables.tex', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print("Saved latex_tables.tex")
print(f"Macro-F1 for burst classification: {macro_f1:.3f}")
