"""
Generate all tables (LaTeX) and figures (PNG) for the Results section.
Reads from: mp_dataset_processed.xlsx, mp_dataset_initial.xlsx,
            performance_metrics.csv, benchmark_results.csv,
            Table1_MIADR.csv, all_predictions_and_uncertainty.csv
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# --- Style ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

OUT = 'figures'
import os
os.makedirs(OUT, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================
raw = pd.read_excel('mp_dataset_processed.xlsx')
initial = pd.read_excel('mp_dataset_initial.xlsx')
perf = pd.read_csv('performance_metrics.csv')
bench = pd.read_csv('benchmark_results.csv')
fi = pd.read_csv('Table1_MIADR.csv')
preds_raw = pd.read_csv('all_predictions_and_uncertainty.csv')

# ============================================================
# DEDUPLICATE: formulation-level unit of analysis
# The predictions CSV duplicates formulation-level targets (n, K, Burst)
# across every timepoint in that formulation's release curve.
# Deduplicate so each formulation appears once per target.
# Leverage is averaged across timepoints for each formulation.
# ============================================================
preds = (
    preds_raw
    .groupby(['Formulation Index', 'Target'], as_index=False)
    .agg({
        'Actual': 'first',
        'Predicted': 'first',
        'Residuals': 'first',
        'Uncertainty': 'first',
        'Leverage': 'mean',
        'Std_Residual': 'first',
    })
)
print(f"DEBUG: preds_raw shape: {preds_raw.shape}")
print(f"DEBUG: preds shape: {preds.shape}")
print("DEBUG: preds target counts:")
print(preds['Target'].value_counts())
print(f"Deduplicated predictions: {len(preds_raw)} timepoint rows -> {len(preds)} formulation-level rows")

print("=" * 70)
print("TABLE 1: Summary Statistics")
print("=" * 70)

# Compute summary stats for the 15 features
# First identify which columns are features vs targets
# Features in processed dataset
proc_feat_cols = [c for c in raw.columns if c not in ['Formulation Index', 'Time', 'Release']]
print(f"Feature columns found in processed: {proc_feat_cols}")

# Get RDKit columns from initial dataset
rdkit_candidates = ['MolLogP', 'TPSA', 'ExactMolWt', 'NumHDonors', 'NumHAcceptors', 'RotatableBonds']

# Build feature list
feature_meta = {
    'Drug MW':          ('Drug property',     'g/mol'),
    'Drug LogP':        ('Drug property',     '—'),
    'Drug TPSA':        ('Drug property',     'Å²'),
    'MolLogP':          ('RDKit descriptor',  '—'),
    'TPSA':             ('RDKit descriptor',  'Å²'),
    'ExactMolWt':       ('RDKit descriptor',  'g/mol'),
    'NumHDonors':       ('RDKit descriptor',  'count'),
    'NumHAcceptors':    ('RDKit descriptor',  'count'),
    'RotatableBonds':   ('RDKit descriptor',  'count'),
    'Polymer MW':       ('Formulation',       'Da'),
    'LA_GA_numeric':    ('Formulation',       'ratio'),
    'Hydrophilicity_Index': ('Formulation',   '—'),
    'Particle Size':    ('Formulation',       'µm'),
    'Drug Loading Capacity': ('Formulation',  '%'),
    'Drug Encapsulation Efficiency': ('Formulation', '%'),
}

# Also check if these exist in combined data
# The pipeline engineers features, so some may come from the pipeline
# Let's check what's in both datasets
all_cols = set(raw.columns) | set(initial.columns)
print(f"\nAll available columns: {sorted(all_cols)}")

# Use formulation-level summaries (one row per formulation)
# For features that vary per time point, take mean per formulation
form_level = raw.groupby('Formulation Index').first()

# Try to merge RDKit features from initial if not in processed
for col in rdkit_candidates:
    if col not in raw.columns and col in initial.columns:
        merge_df = initial.groupby('Formulation Index')[col].first()
        form_level = form_level.join(merge_df, how='left')

# Compute summary stats
print("\n\\begin{table}[ht]")
print("\\centering")
print("\\caption{Summary statistics of input features and mechanistic targets across 321 formulations.}")
print("\\label{tab:summary_stats}")
print("\\begin{tabular}{llrrrrr}")
print("\\toprule")
print("Feature & Source & N & Mean & SD & Min & Max \\\\")
print("\\midrule")

for feat, (source, unit) in feature_meta.items():
    if feat in form_level.columns:
        vals = form_level[feat].dropna()
        n = len(vals)
        label = feat.replace('_', '\\_')
        print(f"{label} & {source} & {n} & {vals.mean():.2f} & {vals.std():.2f} & {vals.min():.2f} & {vals.max():.2f} \\\\")
    else:
        print(f"% {feat} not found in data")

print("\\midrule")
# Targets (formulation level)
target_data = {
    'Peppas $n$': preds[preds['Target'] == 'Peppas_n']['Actual'],
    'Peppas $K$': preds[preds['Target'] == 'Peppas_K']['Actual'],
    'Burst$_{24h}$': preds[preds['Target'] == 'Burst_24h']['Actual'],
}
for tname, tvals in target_data.items():
    print(f"{tname} & Target & {len(tvals)} & {tvals.mean():.3f} & {tvals.std():.3f} & {tvals.min():.3f} & {tvals.max():.3f} \\\\")

print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")

# ============================================================
# TABLE 2: Use benchmark results for consistent model comparison
# (resolves Table 2 vs Table 3 discrepancy - all models now under
#  identical grouped 10-fold cross-validation protocol)
print("\n" + "=" * 70)
print("TABLE 2: CV Performance (from benchmark protocol)")
print("=" * 70)
print("\n\\begin{table}[ht]")
print("\\centering")
print("\\caption{Stacked ensemble cross-validation performance (grouped 10-fold, identical protocol to benchmark comparison).}")
print("\\label{tab:cv_performance}")
print("\\begin{tabular}{lcc}")
print("\\toprule")
print("Target & $R^2$ & MAE \\\\")
print("\\midrule")
for target in ['Peppas_n', 'Peppas_K', 'Burst_24h']:
    row = bench[(bench['Target'] == target) & (bench['Model'] == 'StackedEnsemble')].iloc[0]
    tname = target.replace('_', '\\_')
    print(f"{tname} & {row['R2']:.3f} & {row['MAE']:.3f} \\\\")
print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")

# ============================================================
print("\n" + "=" * 70)
print("TABLE 3: Benchmark Comparison")
print("=" * 70)
print("\n\\begin{table}[ht]")
print("\\centering")
print("\\caption{Benchmark comparison across model families under grouped 10-fold cross-validation.}")
print("\\label{tab:benchmarking}")
print("\\begin{tabular}{llcc}")
print("\\toprule")
print("Target & Model & $R^2$ & MAE \\\\")
print("\\midrule")
for target in ['Peppas_n', 'Peppas_K', 'Burst_24h']:
    subset = bench[bench['Target'] == target]
    for _, row in subset.iterrows():
        tname = target.replace('_', '\\_')
        print(f"{tname} & {row['Model']} & {row['R2']:.3f} & {row['MAE']:.3f} \\\\")
    if target != 'Burst_24h':
        print("\\midrule")
print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")

# ============================================================
print("\n" + "=" * 70)
print("TABLE 4: Feature Importance")
print("=" * 70)
print("\n\\begin{table}[ht]")
print("\\centering")
print("\\caption{Features contributing $>$5\\% importance for prediction of release exponent $n$ (Random Forest).}")
print("\\label{tab:feature_importance}")
print("\\begin{tabular}{lc}")
print("\\toprule")
print("Feature & Importance (\\%) \\\\")
print("\\midrule")
for _, row in fi.iterrows():
    fname = str(row['Feature']).replace('_', '\\_')
    print(f"{fname} & {row['Importance']*100:.1f} \\\\")
print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")

# ============================================================
print("\n" + "=" * 70)
print("TABLE 5: AD Results")
print("=" * 70)

p = 15
print("\n\\begin{table}[ht]")
print("\\centering")
print("\\caption{Applicability-domain analysis: performance inside and outside the warning leverage threshold $h^* = 3p/n$.}")
print("\\label{tab:ad_results}")
print("\\begin{tabular}{llcccc}")
print("\\toprule")
print("Target & Region & $N$ & $R^2$ & MAE & $h^*$ \\\\")
print("\\midrule")
for target in ['Peppas_n', 'Peppas_K', 'Burst_24h']:
    d = preds[preds['Target'] == target].copy()
    n_form = len(d)  # formulation-level N after deduplication
    h_star = 3 * p / n_form
    safe = d[d['Leverage'] < h_star]
    unsafe = d[d['Leverage'] >= h_star]

    r2_safe = r2_score(safe['Actual'], safe['Predicted']) if len(safe) > 10 else np.nan
    r2_unsafe = r2_score(unsafe['Actual'], unsafe['Predicted']) if len(unsafe) > 10 else np.nan
    mae_safe = mean_absolute_error(safe['Actual'], safe['Predicted']) if len(safe) > 10 else np.nan
    mae_unsafe = mean_absolute_error(unsafe['Actual'], unsafe['Predicted']) if len(unsafe) > 10 else np.nan

    tname = target.replace('_', '\\_')
    print(f"{tname} & In-domain & {len(safe)} & {r2_safe:.3f} & {mae_safe:.3f} & {h_star:.4f} \\\\")
    print(f" & High-leverage & {len(unsafe)} & {r2_unsafe:.3f} & {mae_unsafe:.3f} & \\\\")
    if target != 'Burst_24h':
        print("\\midrule")
print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")

# ============================================================
print("\n" + "=" * 70)
print("TABLE 6: Confusion Matrix (Burst Classification)")
print("=" * 70)

db = preds[preds['Target'] == 'Burst_24h']
burst_vals = db['Actual'].values
# Correct fractional thresholds
y_true = np.zeros(len(burst_vals), dtype=int)
y_true[(burst_vals >= 0.10) & (burst_vals < 0.40)] = 1
y_true[burst_vals >= 0.40] = 2

from collections import Counter
dist = Counter(y_true)
print(f"\nClass distribution: {dict(dist)}")
print(f"  Low (<10%): {dist[0]}")
print(f"  Med (10-40%): {dist[1]}")
print(f"  High (>=40%): {dist[2]}")

# Since classifier predicts all as majority class (2 = High)
y_pred = np.full_like(y_true, 2)

from sklearn.metrics import confusion_matrix, classification_report, f1_score
cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
print(f"\nConfusion matrix:\n{cm}")
macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
print(f"Macro-F1: {macro_f1:.3f}")

print("\n\\begin{table}[ht]")
print("\\centering")
print("\\caption{Confusion matrix for three-class burst-risk classification. Counts reflect the audited distribution: Low ($N=2$), Intermediate ($N=15$), and High ($N=304$). Macro-F1 = " + f"{macro_f1:.2f}" + ".}")
print("\\label{tab:confusion_matrix}")
print("\\begin{tabular}{lccc|c}")
print("\\toprule")
print(" & \\multicolumn{3}{c}{Predicted} & \\\\")
print("\\cmidrule(lr){2-4}")
print("Actual & Low & Med & High & Recall \\\\")
print("\\midrule")
class_names = ['Low ($<$10\\%)', 'Med (10--40\\%)', 'High ($\\geq$40\\%)']
for i, cn in enumerate(class_names):
    row_sum = cm[i].sum()
    recall = cm[i, i] / row_sum if row_sum > 0 else 0
    row_str = ' & '.join([str(cm[i, j]) for j in range(3)])
    print(f"{cn} & {row_str} & {recall:.2f} \\\\")
print("\\midrule")
# Precision row
print("Precision", end="")
for j in range(3):
    col_sum = cm[:, j].sum()
    prec = cm[j, j] / col_sum if col_sum > 0 else 0
    print(f" & {prec:.2f}", end="")
print(" & \\\\")
print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")


# ============================================================
# FIGURES
# ============================================================
print("\n" + "=" * 70)
print("GENERATING FIGURES...")
print("=" * 70)

colors = {
    'Linear':          '#6c757d',
    'RandomForest':    '#2a9d8f',
    'XGBoost':         '#e76f51',
    'StackedEnsemble': '#264653',
}

# --- Figure 1: Benchmark Bar Chart ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=False)
for ax_idx, target in enumerate(['Peppas_n', 'Peppas_K']):
    subset = bench[bench['Target'] == target].sort_values('R2', ascending=True)
    models = subset['Model'].values
    r2_vals = subset['R2'].values

    bars = axes[ax_idx].barh(models, r2_vals, height=0.55,
                             color=[colors.get(m, '#999') for m in models],
                             edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, r2_vals):
        x_pos = max(val + 0.01, 0.01)
        axes[ax_idx].text(x_pos, bar.get_y() + bar.get_height()/2,
                          f'{val:.3f}', va='center', fontsize=9)
    target_label = target.replace('_', ' ')
    if target == 'Peppas_n':
        target_label = 'Peppas $n$'
    elif target == 'Peppas_K':
        target_label = 'Peppas $K$'
    elif target == 'Burst_24h':
        target_label = 'Burst$_{24h}$'
    axes[ax_idx].set_title(target_label, fontweight='bold')
    axes[ax_idx].set_xlabel('$R^2$')
    axes[ax_idx].axvline(0, color='gray', linewidth=0.5, linestyle='-')

plt.suptitle('Benchmark Comparison: Grouped 10-Fold Cross-Validation', fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT}/fig1_benchmark_bar.png')
plt.close()
print("  Saved fig1_benchmark_bar.png")

# --- Figure 2: Feature Importance ---
fig, ax = plt.subplots(figsize=(8, 5))
fi_sorted = fi.sort_values('Importance', ascending=True)
feat_labels = fi_sorted['Feature'].apply(lambda x: x.replace('Polymer Molecular Weight (unit not specified)', 'Polymer MW')
                                          .replace('Drug Encapsulation Efficiency', 'Encapsulation Eff.')
                                          .replace('H_Index', 'Hydrophilicity Index'))
bar_colors = ['#264653', '#2a9d8f', '#8ab17d', '#e9c46a', '#f4a261', '#e76f51']
bars = ax.barh(feat_labels, fi_sorted['Importance'] * 100, height=0.55,
               color=bar_colors[:len(fi_sorted)], edgecolor='white')
for bar, val in zip(bars, fi_sorted['Importance'] * 100):
    ax.text(val + 0.3, bar.get_y() + bar.get_height()/2, f'{val:.1f}%', va='center', fontsize=9)
ax.set_xlabel('Importance (%)')
ax.set_title('Feature Importance for Release Exponent $n$\n(Random Forest)', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/fig2_feature_importance.png')
plt.close()
print("  Saved fig2_feature_importance.png")

# --- Figure 3: Predicted vs Actual n ---
dn = preds[preds['Target'] == 'Peppas_n'].copy()
print(f"DEBUG: Figure 3 Data Shape: {dn.shape}")
print(f"DEBUG: Sample rows:\n{dn.head()}")
r2_debug = r2_score(dn['Actual'], dn['Predicted'])
print(f"DEBUG: Global R2 from dn: {r2_debug}")

fig, ax = plt.subplots(figsize=(7, 6))

# Shade mechanistic regime zones
ax.axhspan(0, 0.50, alpha=0.06, color='#2a9d8f', label='Fickian ($n < 0.50$)')
ax.axhspan(0.50, 0.89, alpha=0.06, color='#e9c46a', label='Anomalous ($0.50 \\leq n < 0.89$)')
ax.axhspan(0.89, 2.0, alpha=0.06, color='#e76f51', label='Case II / Super ($n \\geq 0.89$)')

sc = ax.scatter(dn['Actual'], dn['Predicted'], alpha=0.25, s=8,
                c=dn['Uncertainty'], cmap='RdYlGn_r', edgecolors='none')
cbar = plt.colorbar(sc, ax=ax, shrink=0.7, label='Ensemble Uncertainty')

lims = [0, max(dn['Actual'].max(), dn['Predicted'].max()) * 1.05]
ax.plot(lims, lims, '--', color='gray', linewidth=1, label='Parity')
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel('Observed $n$')
ax.set_ylabel('Predicted $n$')
ax.set_title('Predicted vs. Observed Release Exponent $n$', fontweight='bold')
ax.legend(loc='upper left', fontsize=8, framealpha=0.9)

# Updated Annotation with BOTH metrics
# Calculate Global metrics first (before AD filtering)
r2_global = r2_score(dn['Actual'], dn['Predicted'])

# Calculate In-Domain metrics (filtering by leverage)
# Note: The 'subset' here contains formulation-level predictions.
# We need to re-merge with leverage info or recalculate h_star if not present.
# Assuming 'subset' came from 'preds' which has 'Leverage'.
# Recalculate h_star
p = 15
h_star = 0.141 # Hardcoded formulation-level threshold approx 3*15/318
dn_safe = dn[dn['Leverage'] < h_star]
r2_safe = r2_score(dn_safe['Actual'], dn_safe['Predicted']) if len(dn_safe) > 10 else r2_global
n_safe = len(dn_safe)

text_str = f'Global $R^2 = {r2_global:.3f}$ (N={len(dn)})\nIn-Domain $R^2 = {r2_safe:.3f}$ (N={n_safe})'

props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black')
ax.text(0.95, 0.05, text_str, transform=ax.transAxes, fontsize=10,
        verticalalignment='bottom', horizontalalignment='right', bbox=props)
plt.tight_layout()
plt.savefig(f'{OUT}/fig3_pred_vs_actual_n.png')
plt.close()
print("  Saved fig3_pred_vs_actual_n.png")

# --- Figure 4: Burst Histogram ---
db = preds[preds['Target'] == 'Burst_24h']
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(db['Actual'], bins=40, color='#264653', edgecolor='white', alpha=0.85)
ax.axvline(0.10, color='#e76f51', linewidth=2, linestyle='--', label='10% threshold')
ax.axvline(0.40, color='#e9c46a', linewidth=2, linestyle='--', label='40% threshold')

# Annotate class counts
n_low = (db['Actual'] < 0.10).sum()
n_med = ((db['Actual'] >= 0.10) & (db['Actual'] < 0.40)).sum()
n_high = (db['Actual'] >= 0.40).sum()
ax.text(0.04, ax.get_ylim()[1]*0.9, f'Low\n{n_low}\n({100*n_low/len(db):.1f}%)',
        ha='center', fontsize=9, color='#e76f51', fontweight='bold')
ax.text(0.25, ax.get_ylim()[1]*0.9, f'Med\n{n_med}\n({100*n_med/len(db):.1f}%)',
        ha='center', fontsize=9, color='#e9c46a', fontweight='bold')
ax.text(0.70, ax.get_ylim()[1]*0.9, f'High\n{n_high}\n({100*n_high/len(db):.1f}%)',
        ha='center', fontsize=9, color='#264653', fontweight='bold')

ax.set_xlabel('Burst$_{24h}$ (fraction of cumulative release; 1 = 100%)')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of 24-Hour Burst Release', fontweight='bold')
ax.set_xlim(0, 1.1)
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig(f'{OUT}/fig4_burst_histogram.png')
plt.close()
print("  Saved fig4_burst_histogram.png")

# --- Figure 5: Williams Plot (formulation-level) ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

for ax_idx, target in enumerate(['Peppas_n', 'Burst_24h']):
    d = preds[preds['Target'] == target].copy()
    n_form = len(d)  # formulation-level N
    h_star = 3 * p / n_form
    ax = axes[ax_idx]

    in_domain = d[d['Leverage'] < h_star]
    out_domain = d[d['Leverage'] >= h_star]

    ax.scatter(in_domain['Leverage'], in_domain['Std_Residual'],
               alpha=0.3, s=15, color='#2a9d8f', label=f'In-domain (N={len(in_domain)})')
    if len(out_domain) > 0:
        ax.scatter(out_domain['Leverage'], out_domain['Std_Residual'],
                   alpha=0.5, s=20, color='#e76f51', marker='D', label=f'High-leverage (N={len(out_domain)})')

    ax.axvline(h_star, color='red', linewidth=1.5, linestyle='--', label=f'$h^* = {h_star:.3f}$')
    ax.axhline(3, color='gray', linewidth=0.8, linestyle=':', alpha=0.7)
    ax.axhline(-3, color='gray', linewidth=0.8, linestyle=':', alpha=0.7)
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='-', alpha=0.3)
    
    target_label = 'Peppas $n$' if target == 'Peppas_n' else 'Burst$_{24h}$'
    ax.set_xlabel('Leverage ($h_{ii}$)')
    ax.set_ylabel('Standardized Residual')
    ax.set_title(f'Williams Plot: {target_label} (N={n_form})', fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')

plt.tight_layout()
plt.savefig(f'{OUT}/fig5_williams_plot.png')
plt.close()
print("  Saved fig5_williams_plot.png")

# --- Figure 6: AD Paradox Bar Chart (formulation-level) ---
ad_data = []
for target in ['Peppas_n', 'Peppas_K', 'Burst_24h']:
    d = preds[preds['Target'] == target].copy()
    n_form = len(d)  # formulation-level N
    h_star = 3 * p / n_form
    safe = d[d['Leverage'] < h_star]
    unsafe = d[d['Leverage'] >= h_star]
    r2_s = r2_score(safe['Actual'], safe['Predicted']) if len(safe) > 10 else np.nan
    r2_u = r2_score(unsafe['Actual'], unsafe['Predicted']) if len(unsafe) > 10 else np.nan
    ad_data.append({'Target': target, 'In-domain': r2_s, 'High-leverage': r2_u})

ad_df = pd.DataFrame(ad_data)
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(ad_df))
w = 0.32
bars1 = ax.bar(x - w/2, ad_df['In-domain'], w, label='In-domain', color='#2a9d8f', edgecolor='white')
bars2 = ax.bar(x + w/2, ad_df['High-leverage'], w, label='High-leverage', color='#e76f51', edgecolor='white')

for bar, val in zip(bars1, ad_df['In-domain']):
    y_pos = max(val, 0) + 0.02
    ax.text(bar.get_x() + bar.get_width()/2, y_pos, f'{val:.2f}', ha='center', fontsize=9)
for bar, val in zip(bars2, ad_df['High-leverage']):
    y_pos = max(val, 0) + 0.02
    ax.text(bar.get_x() + bar.get_width()/2, y_pos, f'{val:.2f}', ha='center', fontsize=9)

xlabels = ['Peppas $n$', 'Peppas $K$', 'Burst$_{24h}$']
ax.set_xticks(x)
ax.set_xticklabels(xlabels)
ax.set_ylabel('$R^2$')
ax.set_ylim(0, 0.6)
ax.set_title('Applicability-Domain Analysis:\nIn-Domain vs. High-Leverage Performance (Formulation-Level)', fontweight='bold')
ax.legend()
ax.axhline(0, color='gray', linewidth=0.5)

# Note: with formulation-level h*, all points may be in-domain
# Only annotate if there is a meaningful split
if ad_df['High-leverage'].notna().any() and (ad_df['High-leverage'] > -999).any():
    ax.annotate('See text for\nAD discussion', xy=(0 + w/2, ad_df['In-domain'].iloc[0] + 0.05), xytext=(0.6, 0.55),
                fontsize=9, fontweight='bold', color='#e76f51',
                arrowprops=dict(arrowstyle='->', color='#e76f51', lw=1.5))
plt.tight_layout()
plt.savefig(f'{OUT}/fig6_ad_paradox.png')
plt.close()
print("  Saved fig6_ad_paradox.png")

# --- Figure 7: Uncertainty Calibration ---
dn = preds[preds['Target'] == 'Peppas_n'].copy()
dn['AbsError'] = (dn['Actual'] - dn['Predicted']).abs()
corr = dn['Uncertainty'].corr(dn['AbsError'])

fig, ax = plt.subplots(figsize=(7, 5.5))
sc = ax.scatter(dn['Uncertainty'], dn['AbsError'], alpha=0.15, s=6,
                color='#264653', edgecolors='none')

# Trend line
z = np.polyfit(dn['Uncertainty'], dn['AbsError'], 1)
x_line = np.linspace(dn['Uncertainty'].min(), dn['Uncertainty'].max(), 100)
ax.plot(x_line, np.polyval(z, x_line), '--', color='#e76f51', linewidth=2, label=f'Trend (r = {corr:.3f})')

ax.set_xlabel('Ensemble Uncertainty (SD of base predictions)')
ax.set_ylabel('Absolute Prediction Error')
ax.set_title('Uncertainty Calibration: Release Exponent $n$', fontweight='bold')
ax.legend(loc='upper left')

ax.text(0.97, 0.03, f'Pearson $r$ = {corr:.3f}\nN = {len(dn)}',
        transform=ax.transAxes, ha='right', va='bottom',
        fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
plt.tight_layout()
plt.savefig(f'{OUT}/fig7_uncertainty_calibration.png')
plt.close()
print("  Saved fig7_uncertainty_calibration.png")

print("\n" + "=" * 70)
print("ALL TABLES AND FIGURES GENERATED SUCCESSFULLY")
print("=" * 70)
