import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

preds_raw = pd.read_csv(r'c:\Users\kusha\Downloads\A Dataset on Formulation Parameters and Characteristics of Drug-Loaded PLGA Microparticles\A Dataset on Formulation Parameters and Characteristics of Drug-Loaded PLGA Microparticles\all_predictions_and_uncertainty.csv')
preds = (
    preds_raw
    .groupby(['Formulation Index', 'Target'], as_index=False)
    .agg({
        'Actual': 'first',
        'Predicted': 'first',
        'Leverage': 'mean',
    })
)

for target in ['Peppas_n', 'Peppas_K', 'Burst_24h']:
    d = preds[preds['Target'] == target]
    n_form = len(d)
    p = 15
    h_star = 3 * p / n_form
    safe = d[d['Leverage'] < h_star]
    
    r2_full = r2_score(d['Actual'], d['Predicted'])
    r2_safe = r2_score(safe['Actual'], safe['Predicted']) if len(safe) > 0 else np.nan
    
    print(f"{target}:")
    print(f"  Total N: {len(d)}, R2: {r2_full:.4f}")
    print(f"  In-domain N: {len(safe)}, R2: {r2_safe:.4f}")
    print(f"  h_star: {h_star:.4f}")
