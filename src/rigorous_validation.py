import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.plga_pipeline_v2 import PLGAPrecisionPipeline

def rigorous_validation():
    print("=== RIGOROUS VALIDATION: LEAKAGE CHECK & AD ANALYSIS ===")
    
    # 1. Load Data (Reuse pipeline for raw loading & feature engineering ONLY)
    print("Loading and engineering features (no imputation yet)...")
    pipeline = PLGAPrecisionPipeline('mp_dataset_processed.xlsx', 'mp_dataset_initial.xlsx')
    pipeline.engineer_features()
    pipeline.engineer_targets()
    
    df = pipeline.df
    print(f"Total Data Shape: {df.shape}")
    
    # 2. Strict Train/Test Split (80/20 Grouped)
    print("\nSTEP 1: Strict 80/20 Grouped Split...")
    groups = df['Formulation Index']
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(df, groups=groups))
    
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()
    
    print(f"Train Set: {len(train_df)} samples")
    print(f"Test Set:  {len(test_df)} samples")
    
    # Define features
    feature_cols = ['Drug MW', 'Drug LogP', 'Drug TPSA', 'MolLogP', 'TPSA', 'ExactMolWt', 
                    'NumHDonors', 'NumHAcceptors', 'RotatableBonds',
                    'Polymer MW', 'LA_GA_numeric', 'Hydrophilicity_Index',
                    'Particle Size', 'Drug Loading Capacity', 'Drug Encapsulation Efficiency']
    
    X_train_raw = train_df[feature_cols]
    X_test_raw = test_df[feature_cols]
    
    # 3. Leakage-Free Preprocessing
    # Fit Imputer and Scaler on TRAIN only
    print("\nSTEP 2: Preprocessing (Fit on Train ONLY)...")
    preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    
    # 4. Burst Release Classification (The "Leakage Test")
    print("\nSTEP 3: Burst Release Classification Check...")
    target = 'Burst_24h'
    
    # Create Classes
    y_train_val = train_df[target]
    y_test_val = test_df[target]
    
    # Filter NaNs
    train_mask = y_train_val.notna()
    test_mask = y_test_val.notna()
    
    X_train_b = X_train[train_mask]
    y_train_b = y_train_val[train_mask]
    X_test_b = X_test[test_mask]
    y_test_b = y_test_val[test_mask]
    
    def get_classes(y):
        y_class = np.zeros_like(y, dtype=int)
        y_class[(y >= 10) & (y < 40)] = 1
        y_class[y >= 40] = 2
        return y_class
        
    y_train_cls = get_classes(y_train_b)
    y_test_cls = get_classes(y_test_b)
    
    print(f"Training Burst Classifier on {len(y_train_cls)} samples...")
    clf = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, n_jobs=-1, 
                            use_label_encoder=False, objective='multi:softprob', num_class=3, eval_metric='mlogloss', random_state=42)
    clf.fit(X_train_b, y_train_cls)
    
    train_preds = clf.predict(X_train_b)
    test_preds = clf.predict(X_test_b)
    
    train_acc = accuracy_score(y_train_cls, train_preds)
    test_acc = accuracy_score(y_test_cls, test_preds)
    
    print(f"  -> Train Accuracy: {train_acc:.4f}")
    print(f"  -> Test Accuracy:  {test_acc:.4f} (Previous reported: 1.0)")
    print("  -> Test Confusion Matrix:")
    print(confusion_matrix(y_test_cls, test_preds))
    
    if test_acc < 0.99:
        print("  [CONCLUSION] Leakage Confirmed. 100% was an artifact of improper validation.")
    else:
        print("  [CONCLUSION] 100% Accuracy held! Signal is extremely strong.")

    # 5. Applicability Domain Analysis (The "Hero")
    print("\nSTEP 4: Applicability Domain Analysis (Peppas_n)...")
    target_n = 'Peppas_n'
    
    y_train_n = train_df[target_n]
    y_test_n = test_df[target_n]
    
    mask_train = y_train_n.notna()
    mask_test = y_test_n.notna()
    
    X_train_n = X_train[mask_train]
    y_train_n = y_train_n[mask_train]
    X_test_n = X_test[mask_test]
    y_test_n = y_test_n[mask_test]
    
    # Train Regressor (XGBoost for speed/consistency)
    reg = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.05, n_jobs=-1, objective='reg:squarederror', random_state=42)
    reg.fit(X_train_n, y_train_n)
    
    y_pred_test = reg.predict(X_test_n)
    
    # Calculate AD Statistics
    # H = X_test_scaled * (X_train_scaled.T * X_train_scaled)^-1 * X_train_scaled.T ... tricky for test set
    # Standard approach for Test Set AD: Distance to training centroid or similar.
    # But sticking to Williams Plot logic: Leverage of test point regarding the HAT matrix of Training Data.
    # h_i = x_i^T (X_train^T X_train)^-1 x_i
    
    # Compute (X^T X)^-1 from Training Data
    try:
        XtX_inv = np.linalg.pinv(np.dot(X_train_n.T, X_train_n))
        
        # Compute Leverage for Test Points
        levs = []
        for i in range(len(X_test_n)):
            x_vec = X_test_n[i]
            # h = x^T * (XtX)^-1 * x
            h = np.dot(x_vec.T, np.dot(XtX_inv, x_vec))
            levs.append(h)
        levs = np.array(levs)
        
        # Calculate Warning Leverage h*
        p = X_train_n.shape[1]
        n_train = X_train_n.shape[0]
        h_star = 3 * p / n_train
        
        print(f"  Warning Leverage h*: {h_star:.4f}")
        
        # Identify "Safe" vs "Unsafe"
        # Standardized Residuals not available without "true" sigma, but we can use prediction error.
        residuals = y_test_n - y_pred_test
        # Standardize by Training RMSE or similar? 
        # Williams plot usually uses standardized residuals of the *model fit*.
        # Here we check if AE is lower in low leverage.
        
        safe_mask = levs < h_star
        unsafe_mask = ~safe_mask
        
        print(f"  Test Points in Domain: {sum(safe_mask)} / {len(levs)}")
        
        if sum(safe_mask) > 0:
            r2_safe = r2_score(y_test_n[safe_mask], y_pred_test[safe_mask])
            mae_safe = mean_absolute_error(y_test_n[safe_mask], y_pred_test[safe_mask])
            print(f"  [SAFE ZONE] R2: {r2_safe:.4f}, MAE: {mae_safe:.4f}")
        else:
             print("  [SAFE ZONE] No points.")
             
        if sum(unsafe_mask) > 0:
            r2_unsafe = r2_score(y_test_n[unsafe_mask], y_pred_test[unsafe_mask])
            mae_unsafe = mean_absolute_error(y_test_n[unsafe_mask], y_pred_test[unsafe_mask])
            print(f"  [UNSAFE ZONE] R2: {r2_unsafe:.4f}, MAE: {mae_unsafe:.4f}")
        else:
            print("  [UNSAFE ZONE] No points.")
            
        # Diff
        if sum(safe_mask) > 0 and sum(unsafe_mask) > 0:
            print(f"  -> Improvement in Safe Zone: +{r2_safe - r2_unsafe:.4f} R2")
            
    except Exception as e:
        print(f"AD Analysis Failed: {e}")

if __name__ == "__main__":
    rigorous_validation()
