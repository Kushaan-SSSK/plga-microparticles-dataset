
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.plga_pipeline_v2 import PLGAPrecisionPipeline
import warnings

warnings.filterwarnings('ignore')

def check_and_run():
    print("=== Checking Burst Statistics ===")
    
    # Load data using pipeline logic to ensure consistent feature engineering
    pipeline = PLGAPrecisionPipeline('mp_dataset_processed.xlsx', 'mp_dataset_initial.xlsx')
    pipeline.engineer_features()
    pipeline.engineer_targets()
    
    df = pipeline.df
    burst = df['Burst_24h']
    
    print(f"\nFull Dataset N={len(burst)}")
    print(f"Min: {burst.min():.4f}")
    print(f"Max: {burst.max():.4f}")
    print(f"Mean: {burst.mean():.4f}")
    print(f"SD: {burst.std():.4f}")
    
    # Check outliers > 1.2
    outliers = df[df['Burst_24h'] > 1.2]
    print(f"\nOutliers > 1.2: N={len(outliers)}")
    if len(outliers) > 0:
        print(outliers[['Formulation Index', 'Burst_24h']])
        
    print("\n=== Rerunning Benchmarks with Filter (Burst <= 1.2) ===")
    # Filter
    df_clean = df[df['Burst_24h'] <= 1.2].copy()
    print(f"Clean Dataset N={len(df_clean)}")
    print(f"New Max: {df_clean['Burst_24h'].max():.4f}")
    print(f"New SD: {df_clean['Burst_24h'].std():.4f}")
    
    # Run Benchmark
    run_benchmark_on_df(df_clean)

def run_benchmark_on_df(df):
    target = 'Burst_24h'
    
    feature_cols = ['Drug MW', 'Drug LogP', 'Drug TPSA', 'MolLogP', 'TPSA', 'ExactMolWt', 
                    'NumHDonors', 'NumHAcceptors', 'RotatableBonds',
                    'Polymer MW', 'LA_GA_numeric', 'Hydrophilicity_Index',
                    'Particle Size', 'Drug Loading Capacity', 'Drug Encapsulation Efficiency']
    
    X = df[feature_cols]
    y = df[target]
    groups = df['Formulation Index']
    
    # Simple imputation for benchmark (same as benchmark_baselines.py)
    X = X.fillna(X.mean(numeric_only=True))
    
    models = {
        'Linear': LinearRegression(),
        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.05, n_jobs=-1, objective='reg:squarederror'),
    }
    
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
    
    gkf = GroupKFold(n_splits=10)
    
    print(f"\nBenchmarking {target} (Filtered)...")
    
    for name, model in models.items():
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        
        try:
            preds = cross_val_predict(pipe, X, y, cv=gkf, groups=groups, n_jobs=-1)
            r2 = r2_score(y, preds)
            mae = mean_absolute_error(y, preds)
            print(f"{name}: R2={r2:.4f}, MAE={mae:.4f}")
            if name == 'StackedEnsemble':
                with open('burst_mae.txt', 'w') as f:
                    f.write(f"{mae:.4f}")
        except Exception as e:
            print(f"{name}: Failed ({e})")

if __name__ == "__main__":
    check_and_run()
