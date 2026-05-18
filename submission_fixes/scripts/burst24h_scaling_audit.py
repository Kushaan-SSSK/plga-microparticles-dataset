"""
scripts/regenerate_fig5_burst_audit.py

Audits Burst_24h scaling in all_predictions_and_uncertainty.csv,
corrects percentage-scale outliers for plotting only (values > 1.5 / 100),
saves audit CSVs, and regenerates figures/Fig_5.png.

Matplotlib only. No seaborn.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED_CSV   = os.path.join(BASE, 'all_predictions_and_uncertainty.csv')
OUT_DIR    = os.path.join(BASE, 'outputs')
FIG_DIR    = os.path.join(BASE, 'figures')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

AUDIT_CSV   = os.path.join(OUT_DIR, 'burst24h_scaling_audit.csv')
SUMMARY_CSV = os.path.join(OUT_DIR, 'burst24h_scaling_summary.csv')
COUNTS_CSV  = os.path.join(OUT_DIR, 'burst24h_class_counts_after_audit.csv')
FIG5_PNG    = os.path.join(FIG_DIR, 'Fig_5.png')

# ---------------------------------------------------------------------------
# 1. Load predictions CSV
# ---------------------------------------------------------------------------
print("Loading predictions CSV...")
df_all = pd.read_csv(PRED_CSV)
print(f"  Total rows: {len(df_all)}")
print(f"  Columns: {list(df_all.columns)}")

# ---------------------------------------------------------------------------
# 2. Filter to Burst_24h
# ---------------------------------------------------------------------------
if 'Target' not in df_all.columns:
    raise ValueError("Column 'Target' not found in CSV. Cannot filter to Burst_24h.")

df = df_all[df_all['Target'] == 'Burst_24h'].copy()
print(f"  Burst_24h rows (before dedup): {len(df)}")

# ---------------------------------------------------------------------------
# 3. Identify actual burst column
# ---------------------------------------------------------------------------
actual_col = None
for candidate in ['y_true', 'Actual', 'Observed', 'true', 'actual']:
    if candidate in df.columns:
        actual_col = candidate
        break

if actual_col is None:
    print(f"Available columns: {list(df.columns)}")
    raise ValueError("Could not find an actual/observed burst column. See available columns above.")

print(f"  Using actual column: '{actual_col}'")

# ---------------------------------------------------------------------------
# 4. Identify formulation identifier
# ---------------------------------------------------------------------------
id_col = None
for candidate in ['Formulation Index', 'Formulation_Index', 'formulation_id', 'ID']:
    if candidate in df.columns:
        id_col = candidate
        break

if id_col is None:
    print("  No formulation ID column found; using row index.")

# ---------------------------------------------------------------------------
# 5. Deduplicate to one row per formulation
# ---------------------------------------------------------------------------
if id_col is not None:
    df_form = df.groupby(id_col, as_index=False)[actual_col].first()
else:
    df_form = df[[actual_col]].drop_duplicates().reset_index(drop=True)

print(f"  Unique formulations (Burst_24h): {len(df_form)}")

burst_vals = df_form[actual_col].values.copy()

# ---------------------------------------------------------------------------
# 6. Build row-level audit for suspicious values (> 1.5)
# ---------------------------------------------------------------------------
SUSPICIOUS_THRESHOLD = 1.5

audit_rows = []
for i, val in enumerate(burst_vals):
    if val > SUSPICIOUS_THRESHOLD:
        form_id = df_form[id_col].iloc[i] if id_col is not None else f"row_{i}"
        corrected = val / 100.0
        audit_rows.append({
            'Row_Index': i,
            'Formulation_ID': form_id,
            'Original_Value': val,
            'Proposed_Corrected_Value': corrected,
            'Reason': f'Value > {SUSPICIOUS_THRESHOLD}; likely percentage-scale encoding; divided by 100'
        })

audit_df = pd.DataFrame(audit_rows)
audit_df.to_csv(AUDIT_CSV, index=False)
print(f"\nAudit table ({len(audit_rows)} suspicious rows) saved to: {AUDIT_CSV}")
if len(audit_rows) > 0:
    print(audit_df.to_string(index=False))
else:
    print("  No suspicious values found (all Burst_24h <= 1.5).")

# ---------------------------------------------------------------------------
# 7. Apply correction for plotting only
# ---------------------------------------------------------------------------
burst_corrected = burst_vals.copy().astype(float)
n_corrected = 0
for row in audit_rows:
    idx = row['Row_Index']
    burst_corrected[idx] = row['Proposed_Corrected_Value']
    n_corrected += 1

# ---------------------------------------------------------------------------
# 8. Summary statistics after correction
# ---------------------------------------------------------------------------
n_total      = len(burst_corrected)
b_min        = float(np.min(burst_corrected))
b_max        = float(np.max(burst_corrected))
b_mean       = float(np.mean(burst_corrected))
b_sd         = float(np.std(burst_corrected, ddof=1))
n_above_one  = int(np.sum(burst_corrected > 1.0))

summary = {
    'N':                      [n_total],
    'Min':                    [round(b_min, 4)],
    'Max':                    [round(b_max, 4)],
    'Mean':                   [round(b_mean, 4)],
    'SD':                     [round(b_sd, 4)],
    'N_above_1.0':            [n_above_one],
    'N_corrected_from_pct':   [n_corrected],
}
summary_df = pd.DataFrame(summary)
summary_df.to_csv(SUMMARY_CSV, index=False)

print("\nSummary after correction:")
print(f"  N = {n_total}")
print(f"  Min = {b_min:.4f}")
print(f"  Max = {b_max:.4f}")
print(f"  Mean = {b_mean:.4f}")
print(f"  SD = {b_sd:.4f}")
print(f"  N values > 1.0 = {n_above_one}")
print(f"  N corrected from percentage scale = {n_corrected}")
print(f"  Saved to: {SUMMARY_CSV}")

# ---------------------------------------------------------------------------
# 9. Class counts after correction
# ---------------------------------------------------------------------------
n_low  = int(np.sum(burst_corrected < 0.10))
n_med  = int(np.sum((burst_corrected >= 0.10) & (burst_corrected < 0.40)))
n_high = int(np.sum(burst_corrected >= 0.40))
n_check = n_low + n_med + n_high

counts_df = pd.DataFrame([
    {'Class': 'Low (Burst_24h < 0.10)',              'N': n_low,  'Pct': round(100*n_low/n_total, 1)},
    {'Class': 'Intermediate (0.10 <= Burst_24h < 0.40)', 'N': n_med,  'Pct': round(100*n_med/n_total, 1)},
    {'Class': 'High (Burst_24h >= 0.40)',             'N': n_high, 'Pct': round(100*n_high/n_total, 1)},
    {'Class': 'Total',                                'N': n_check,'Pct': 100.0},
])
counts_df.to_csv(COUNTS_CSV, index=False)

print("\nClass counts after audit correction:")
print(f"  Low  (< 0.10):       {n_low}  ({100*n_low/n_total:.1f}%)")
print(f"  Med  (0.10-0.40):   {n_med}  ({100*n_med/n_total:.1f}%)")
print(f"  High (>= 0.40):     {n_high}  ({100*n_high/n_total:.1f}%)")
print(f"  Total: {n_check}")
print(f"  Saved to: {COUNTS_CSV}")

# ---------------------------------------------------------------------------
# 10. Regenerate Figure 5
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

fig, ax = plt.subplots(figsize=(8, 5))

# Clip display range to [0, 1.15] for histogram bins
plot_vals = burst_corrected.copy()

ax.hist(plot_vals, bins=40, range=(0, 1.15),
        color='#264653', edgecolor='white', linewidth=0.4, alpha=0.88)

ax.axvline(0.10, color='#e76f51', linewidth=1.8, linestyle='--', label='0.10 threshold (Low/Intermediate)')
ax.axvline(0.40, color='#e9c46a', linewidth=1.8, linestyle='--', label='0.40 threshold (Intermediate/High)')

ax.set_xlim(0, 1.1)
ax.set_xlabel('Burst$_{24h}$ (fraction of cumulative release; 1 = 100%)', fontsize=12)
ax.set_ylabel('Number of formulations', fontsize=12)
ax.set_title('Distribution of 24-Hour Burst Release (N = 321)', fontweight='bold')

# Annotation box with class counts
annot_text = (
    f'Low: {n_low} ({100*n_low/n_total:.1f}%)\n'
    f'Intermediate: {n_med} ({100*n_med/n_total:.1f}%)\n'
    f'High: {n_high} ({100*n_high/n_total:.1f}%)'
)
ax.text(0.97, 0.97, annot_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#cccccc', alpha=0.9))

ax.legend(loc='upper left', fontsize=9, framealpha=0.9)

plt.tight_layout()
plt.savefig(FIG5_PNG, dpi=300)
plt.close()

print(f"\nFigure 5 saved to: {FIG5_PNG}")
print("Done.")
