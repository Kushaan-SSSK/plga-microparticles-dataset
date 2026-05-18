import sys
sys.path.append('.')
from src.plga_pipeline_v2 import PLGAPrecisionPipeline
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

p = PLGAPrecisionPipeline('mp_dataset_processed.xlsx', 'mp_dataset_initial.xlsx')
p.engineer_features()
p.engineer_targets()
p.build_ensemble()
p.train_and_validate()

feature_cols = ['Drug MW', 'Drug LogP', 'Drug TPSA', 'MolLogP', 'TPSA', 'ExactMolWt', 
                'NumHDonors', 'NumHAcceptors', 'RotatableBonds',
                'Polymer MW', 'LA_GA_numeric', 'Hydrophilicity_Index',
                'Particle Size', 'Drug Loading Capacity', 'Drug Encapsulation Efficiency']

X = p.df[feature_cols].fillna(0).values
X_s = StandardScaler().fit_transform(X)
H = X_s @ np.linalg.pinv(X_s.T @ X_s) @ X_s.T
leverage = np.diag(H)

for target in ['Peppas_n', 'Peppas_K', 'Burst_24h']:
    # Get oof predictions
    d = p.results[target]
    # Merge leverage using Formulation Index
    # Wait, p.results[target] has Formulation Index
    # We can match leverage by index because p.df and X are in the same order, but p.results[target] is out-of-fold predictions.
    # Actually, p.results[target] has all_groups which is Formulation Index.
    # Let's map Formulation Index to its leverage.
    lev_map = dict(zip(p.df['Formulation Index'], leverage))
    d['Leverage'] = d['Formulation Index'].map(lev_map)
    
    N = len(d)
    h_star = 3 * 15 / N
    safe = d[d.Leverage <= h_star]
    high = d[d.Leverage > h_star]
    
    print(f"\n--- {target} (h* = {h_star:.4f}) ---")
    for name, df_sub in [('Safe', safe), ('High', high)]:
        if len(df_sub) == 0:
            print(f"  {name}: N=0")
            continue
        r2 = r2_score(df_sub['Actual'], df_sub['Predicted'])
        mae = mean_absolute_error(df_sub['Actual'], df_sub['Predicted'])
        rmse = np.sqrt(mean_squared_error(df_sub['Actual'], df_sub['Predicted']))
        print(f"  {name}: N={len(df_sub)}, R2={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")
