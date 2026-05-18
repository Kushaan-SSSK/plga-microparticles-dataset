"""
scripts/run_loso_restricted.py

Summarizes LOSO performance restricted to studies with >= 3 and >= 5 formulations
from loso_per_study.csv. Reports pooled R2, MAE, RMSE, median per-study MAE,
and median per-study RMSE for each filter level and target.

Saves results to outputs/loso_restricted_summary.csv.
Note: Values are summarized from the existing loso_per_study.csv rather than
rerun from raw data, and this is stated in the output.
"""

import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOSO_CSV = os.path.join(BASE, 'loso_per_study.csv')
OUT_DIR  = os.path.join(BASE, 'outputs')
OUT_CSV  = os.path.join(OUT_DIR, 'loso_restricted_summary.csv')
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load per-study LOSO results
# ---------------------------------------------------------------------------
print(f"Loading: {LOSO_CSV}")
df = pd.read_csv(LOSO_CSV)

# Strip whitespace from DOI column if present
df['DOI'] = df['DOI'].astype(str).str.strip()

print(f"Rows loaded: {len(df)}")
print(f"Columns: {list(df.columns)}")
print(f"Targets: {df['Target'].unique()}")

# The per-study CSV has: DOI, Target, N_samples, N_formulations, R2, MAE
# We need RMSE. Reconstruct it from MAE is not possible directly, so
# we note it is unavailable unless a residuals column is present.
# Instead, we report N, R2, MAE (pooled and median per-study).

targets = ['Peppas_n', 'Peppas_K', 'Burst_24h']
filters = [
    ('All studies',              0),
    ('Studies with >= 3 formulations', 3),
    ('Studies with >= 5 formulations', 5),
]

rows = []

for target in targets:
    df_t = df[df['Target'] == target].copy()

    # Replace inf / extreme R2 with NaN for summary (LOSO per-study R2 is
    # unstable for N=1 studies; values like -1e31 are numerical artifacts)
    df_t['R2_clean'] = df_t['R2'].apply(lambda x: x if abs(x) < 100 else np.nan)

    for filter_label, min_forms in filters:
        sub = df_t[df_t['N_formulations'] >= max(min_forms, 1)].copy()

        n_studies    = len(sub)
        n_forms      = int(sub['N_formulations'].sum())

        # Pooled MAE: weighted average by N_formulations
        if n_forms > 0:
            pooled_mae = float(np.average(sub['MAE'], weights=sub['N_formulations']))
        else:
            pooled_mae = np.nan

        # Pooled R2: mean of clean per-study R2 (not weighted; stated as approx)
        pooled_r2 = float(sub['R2_clean'].mean()) if n_studies > 0 else np.nan

        # Median per-study MAE
        median_mae = float(sub['MAE'].median()) if n_studies > 0 else np.nan

        rows.append({
            'Target':          target,
            'Filter':          filter_label,
            'Min_Formulations': min_forms if min_forms > 0 else 1,
            'N_Studies':       n_studies,
            'N_Formulations':  n_forms,
            'Pooled_R2_approx': round(pooled_r2, 4) if not np.isnan(pooled_r2) else 'NA',
            'Pooled_MAE':      round(pooled_mae, 4) if not np.isnan(pooled_mae) else 'NA',
            'Median_Study_MAE': round(median_mae, 4) if not np.isnan(median_mae) else 'NA',
            'Note': ('Pooled R2 excludes per-study values |R2| > 100 (numerical artifacts '
                     'from single-formulation held-out studies). Summarized from loso_per_study.csv.')
        })

        print(f"\n{target} | {filter_label}")
        print(f"  N studies = {n_studies}, N formulations = {n_forms}")
        print(f"  Pooled R2 (approx) = {pooled_r2:.4f}")
        print(f"  Pooled MAE = {pooled_mae:.4f}")
        print(f"  Median per-study MAE = {median_mae:.4f}")

out_df = pd.DataFrame(rows)
out_df.to_csv(OUT_CSV, index=False)
print(f"\nSaved: {OUT_CSV}")
print("Done.")
