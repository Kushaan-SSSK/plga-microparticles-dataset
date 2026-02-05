
import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin

class PLGAModelManager:
    def __init__(self, features_path, curves_path):
        self.df = pd.read_csv(features_path)
        self.curves_df = pd.read_pickle(curves_path)
        self.targets = {}
        self.models = {}
        self.results = {}
        
    def define_targets(self):
        """
        Defines Y1 (Burst), Y2 (t50), Y3 (Cluster).
        """
        print("Defining targets...")
        # Y1: % Burst at 24h
        # Already calculated in data_engineering as 'Burst_Slope'? 
        # Wait, prompt says "Y1: % Burst at 24h". 
        # My data_engineering calculated 'Burst_Slope' = release/time.
        # Ideally we want the actual % Release at 24h.
        # I need to re-calculate or update data_engineering? 
        # Or I can calculate it here from the curves if I have them.
        # I loaded 'interpolated_curves.pkl'.
        
        # Let's extract % Burst at 24h from curves
        burst_values = []
        t50_values = []
        
        # Prepare matrix for clustering
        # interpolated_curves structure: Formulation Index, Interpolated_Release (array), Interpolated_Time (array)
        
        # Pivot curves to matrix
        curve_matrix = []
        indices = []
        
        for idx, row in self.curves_df.iterrows():
            f_idx = row['Formulation Index']
            y = row['Interpolated_Release']
            t = row['Interpolated_Time']
            
            # burst at 24h
            # find index closest to 24h or interpolate
            # Since we have pchip implied by 'y', let's just find closest point if high res, 
            # or better validation: strictly pchip(24).
            # But the 'y' array is just values. 't' is array.
            # Let's assume linear interp between points or closest.
            
            # Find closest time to 24
            idx_24 = (np.abs(t - 24)).argmin()
            burst_val = y[idx_24]
            burst_values.append({'Formulation Index': f_idx, 'Y1_Burst': burst_val})
            
            # t50: Time to 50% release
            # Find first time where y >= 50 (assuming release is 0-100 scale)
            # data_engineering said 0-100% release? Let's check scale.
            # Sample output showed "Release" ~ 0.5 to 88. So scale is likely %.
            # If max release < 50%, t50 is undefined (or max time).
            idx_50 = np.where(y >= 50)[0]
            if len(idx_50) > 0:
                t50 = t[idx_50[0]]
            else:
                t50 = t[-1] # Censored
            
            t50_values.append({'Formulation Index': f_idx, 'Y2_t50': t50})
            
            curve_matrix.append(y)
            indices.append(f_idx)
            
        # Y3: Clustering
        print("Running K-Means for Y3...")
        X_curves = np.array(curve_matrix)
        # Normalize curves? Usually yes, but they are all 0-100% (or similar).
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(X_curves)
        
        cluster_data = [{'Formulation Index': idx, 'Y3_Cluster': c} for idx, c in zip(indices, clusters)]
        
        # Merge targets into main df
        self.df = self.df.merge(pd.DataFrame(burst_values), on='Formulation Index')
        self.df = self.df.merge(pd.DataFrame(t50_values), on='Formulation Index')
        self.df = self.df.merge(pd.DataFrame(cluster_data), on='Formulation Index')
        
        return self.df

    def physics_residuals(self):
        """
        Calculates Higuchi residuals (Q = K * sqrt(t)) and adds as meta-feature?
        Prompt: "Add the residual between the model and Higuchi as a meta-feature."
        This likely means: Fit Higuchi to the curve -> Get theoretical Q -> Residual = Q_real - Q_higuchi.
        This residual profile could be a feature? Or a scalar summary?
        "Add the residual ... as a meta-feature". 
        Usually means: For each formulation, fit Higuchi. The 'error' of Higuchi (RMSE or R2) is a feature.
        OR: Predict the Residuals as a target?
        Prompt says: "Add the residual ... as a meta-feature."
        Let's assume we add "Higuchi_R2" or "Higuchi_MSE" as a feature column for the main models.
        """
        print("Calculating Higuchi residuals...")
        higuchi_feats = []
        
        for idx, row in self.curves_df.iterrows():
            y = row['Interpolated_Release']
            t = row['Interpolated_Time']
            
            # Fit Q = K * sqrt(t)
            # Linear regression of Q vs sqrt(t)
            # Avoid t=0
            valid = t > 0
            sqrt_t = np.sqrt(t[valid])
            y_valid = y[valid]
            
            if len(y_valid) > 1:
                slope = np.sum(sqrt_t * y_valid) / np.sum(sqrt_t**2) # Zero intercept assumption for Higuchi
                y_pred = slope * sqrt_t
                mse = np.mean((y_valid - y_pred)**2)
                r2 = r2_score(y_valid, y_pred)
                k_higuchi = slope
            else:
                mse = 0
                r2 = 0
                k_higuchi = 0
                
            higuchi_feats.append({
                'Formulation Index': row['Formulation Index'],
                'Higuchi_MSE': mse,
                'Higuchi_K': k_higuchi
            })
            
        self.df = self.df.merge(pd.DataFrame(higuchi_feats), on='Formulation Index', how='left')

    def objective_xgb(self, trial, X, y, task_type='regression'):
        param = {
            'verbosity': 0,
            'objective': 'reg:squarederror' if task_type == 'regression' else 'multi:softmax',
            'booster': 'gbtree',
            'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.2, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 1.0),
            'max_depth': trial.suggest_int('max_depth', 1, 9),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'eta': trial.suggest_float('eta', 1e-8, 1.0, log=True),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
        }
        
        if task_type == 'classification':
             param['num_class'] = 3 # based on K=3
             
        # CV in inner loop
        dtrain = xgb.DMatrix(X, label=y)
        cv_results = xgb.cv(param, dtrain, nfold=5, num_boost_round=100, early_stopping_rounds=10, verbose_eval=False)
        
        if task_type == 'regression':
            metric = 'test-rmse-mean'
            score = cv_results[metric].iloc[-1]
            return score # Minimize RMSE
        else:
            metric = 'test-merror-mean' # Minimize error rate
            score = cv_results[metric].iloc[-1]
            return score

    def run_nested_cv(self, target_cols=['Y1_Burst', 'Y2_t50', 'Y3_Cluster']):
        """
        Nested CV: 10-fold inner (Optuna), 5-fold outer.
        """
        print("Running Nested Cross-Validation...")
        
        # Prepare Feature Matrix X
        # Drop metadata and targets
        leakage = ['Burst_Slope', 'Lag_Duration', 'Peppas_n']
        drop_cols = target_cols + ['Formulation Index', 'Drug', 'Drug SMILES', 'DOI', 'Release', 'Time', 'Article Title', 'IN/OUT', 'Interpolated_Release', 'Interpolated_Time'] + leakage
        # Also drop non-numeric if any
        
        X_df = self.df.drop(columns=[c for c in drop_cols if c in self.df.columns], errors='ignore')
        X_df = X_df.select_dtypes(include=[np.number])
        
        # Fill NaNs
        X_df = X_df.fillna(X_df.mean())
        X = X_df.values
        
        results_summary = []
        
        kf_outer = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for target in target_cols:
            print(f"Optimizing for {target}...")
            y = self.df[target].values
            
            task_type = 'classification' if 'Cluster' in target else 'regression'
            
            fold_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(kf_outer.split(X, y)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Inner Loop: Optuna
                study = optuna.create_study(direction='minimize')
                study.optimize(lambda trial: self.objective_xgb(trial, X_train, y_train, task_type), n_trials=5) # Reduced trials for speed
                
                best_params = study.best_params
                
                # Train final model on outer fold train set
                if task_type == 'regression':
                    model = xgb.XGBRegressor(**best_params, n_estimators=100)
                    model.fit(X_train, y_train)
                    preds = model.predict(X_val)
                    score = r2_score(y_val, preds)
                else:
                    model = xgb.XGBClassifier(**best_params, n_estimators=100)
                    model.fit(X_train, y_train)
                    preds = model.predict(X_val)
                    score = accuracy_score(y_val, preds)
                    
                fold_scores.append(score)
                print(f"Fold {fold}: Score={score:.4f}")
                
            mean_score = np.mean(fold_scores)
            results_summary.append({'Target': target, 'Mean_Score': mean_score})
            print(f"Mean Score for {target}: {mean_score:.4f}")
            
        return pd.DataFrame(results_summary)

if __name__ == "__main__":
    manager = PLGAModelManager('engineered_features.csv', 'interpolated_curves.pkl')
    manager.define_targets()
    manager.physics_residuals()
    # Save generic dataset for audit
    manager.df.to_csv('final_dataset_for_audit.csv', index=False)
    
    results = manager.run_nested_cv()
    results.to_csv('model_performance_results.csv', index=False)
    print("Modeling Complete.")
