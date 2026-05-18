import pandas as pd
import numpy as np
from scipy.stats import f as f_dist

# ── 1. Load targets from the reproducible run ─────────────────
df_preds = pd.read_csv('all_predictions_and_uncertainty.csv')

# ── 2. Load DOI mapping ───────────────────────────────────────
df_init = pd.read_excel('mp_dataset_initial.xlsx')
doi_map = dict(zip(df_init['Formulation Index'], df_init['DOI']))

# ── 3. Decompose variance using ANOVA ────────────────────────
def compute_icc1(data, group_col, value_col):
    # Filter to groups with >= 2 observations after dropping NaNs
    counts = data.groupby(group_col)[value_col].count()
    valid_groups = counts[counts >= 2].index
    sub = data[data[group_col].isin(valid_groups)].dropna(subset=[value_col])
    
    k = sub[group_col].nunique()  # number of groups (studies)
    N = len(sub)                   # total observations (formulations)
    n_j = sub.groupby(group_col)[value_col].count().values  # group sizes
    n_0 = (1 / (k - 1)) * (N - np.sum(n_j**2) / N)  # harmonic mean group size
    
    grand_mean = sub[value_col].mean()
    
    # Between-group sum of squares
    group_means = sub.groupby(group_col)[value_col].mean()
    group_counts = sub.groupby(group_col)[value_col].count()
    SS_between = np.sum(group_counts.values * (group_means.values - grand_mean)**2)
    
    # Within-group sum of squares
    SS_within = 0
    for grp_id, grp_data in sub.groupby(group_col):
        SS_within += np.sum((grp_data[value_col].values - group_means[grp_id])**2)
    
    df_between = k - 1
    df_within = N - k
    
    MS_between = SS_between / df_between
    MS_within = SS_within / df_within
    
    sigma2_within = MS_within
    sigma2_between = max(0, (MS_between - MS_within) / n_0)
    
    icc = sigma2_between / (sigma2_between + sigma2_within) if (sigma2_between + sigma2_within) > 0 else 0
    
    F_stat = MS_between / MS_within
    p_value = 1 - f_dist.cdf(F_stat, df_between, df_within)
    
    return {
        'ICC': icc,
        'sigma2_between': sigma2_between,
        'sigma2_within': sigma2_within,
        'pct_between': icc * 100,
        'F': F_stat,
        'p': p_value,
        'k_studies': k,
        'N_formulations': N
    }

# Run for all 3 targets
for target in ['Peppas_n', 'Peppas_K', 'Burst_24h']:
    # Get only actual values
    d = df_preds[df_preds.Target == target][['Formulation Index', 'Actual']].copy()
    d['DOI'] = d['Formulation Index'].map(doi_map)
    
    res = compute_icc1(d, 'DOI', 'Actual')
    print(f"\n--- {target} ---")
    print(f"  k (studies): {res['k_studies']}")
    print(f"  N (formulations): {res['N_formulations']}")
    print(f"  sigma2_between: {res['sigma2_between']:.4f}")
    print(f"  sigma2_within: {res['sigma2_within']:.4f}")
    print(f"  ICC: {res['ICC']:.4f} ({res['pct_between']:.1f}%)")
    print(f"  F: {res['F']:.2f}")
    print(f"  p: {res['p']:.2e}")
