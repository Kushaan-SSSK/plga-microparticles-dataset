
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingRegressor
from sklearn.svm import SVR
from sklearn.impute import SimpleImputer
from src.plga_pipeline_v2 import PLGAPrecisionPipeline
import warnings

warnings.filterwarnings('ignore')

def run_benchmarks():
    print("=== Running Rigorous Benchmarks ===")
    
    # Reuse pipeline for data consistency
    print("Loading data via PLGAPrecisionPipeline...")
    pipeline = PLGAPrecisionPipeline('mp_dataset_processed.xlsx', 'mp_dataset_initial.xlsx')
    pipeline.engineer_features()
    pipeline.engineer_targets()
    # Apply same feature cleaning as pipeline
    pipeline.df = pipeline.df.fillna(pipeline.df.mean(numeric_only=True))
    
    df = pipeline.df
    targets = ['Peppas_n', 'Peppas_K', 'Burst_24h']
    
    feature_cols = ['Drug MW', 'Drug LogP', 'Drug TPSA', 'MolLogP', 'TPSA', 'ExactMolWt', 
                    'NumHDonors', 'NumHAcceptors', 'RotatableBonds',
                    'Polymer MW', 'LA_GA_numeric', 'Hydrophilicity_Index',
                    'Particle Size', 'Drug Loading Capacity', 'Drug Encapsulation Efficiency']
    
    X = df[feature_cols]
    groups = df['Formulation Index']
    
    # Define models
    models = {
        'Linear': LinearRegression(),
        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.05, n_jobs=-1, objective='reg:squarederror'),
    }
    
    # Re-define ensemble to match production
    rf_ens = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    xgb_ens = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, n_jobs=-1, objective='reg:squarederror')
    svr_ens = SVR(kernel='rbf', C=10, gamma='scale')
    stack = StackingRegressor(
        estimators=[('rf', rf_ens), ('xgb', xgb_ens), ('svr', svr_ens)],
        final_estimator=Ridge(alpha=1.0),
        cv=5,
        n_jobs=-1
    )
    models['StackedEnsemble'] = stack
    
    results = []
    
    gkf = GroupKFold(n_splits=10)
    
    for target in targets:
        print(f"\nTarget: {target}")
        y = df[target]
        valid_mask = y.notna()
        X_curr = X[valid_mask]
        y_curr = y[valid_mask]
        groups_curr = groups[valid_mask]
        
        for name, model in models.items():
            print(f"  Benchmarking {name}...")
            
            pipe = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler()),
                ('model', model)
            ])
            
            try:
                preds = cross_val_predict(pipe, X_curr, y_curr, cv=gkf, groups=groups_curr, n_jobs=-1)
                
                r2 = r2_score(y_curr, preds)
                mae = mean_absolute_error(y_curr, preds)
                
                results.append({
                    'Target': target,
                    'Model': name,
                    'R2': r2,
                    'MAE': mae
                })
                print(f"    -> R2: {r2:.3f}")
            except Exception as e:
                print(f"    -> Failed: {e}")
                
    res_df = pd.DataFrame(results)
    res_df.to_csv('benchmark_results.csv', index=False)
    print("\nBenchmarks saved to benchmark_results.csv")
    print(res_df.pivot(index='Target', columns='Model', values='R2'))

if __name__ == "__main__":
    run_benchmarks()
