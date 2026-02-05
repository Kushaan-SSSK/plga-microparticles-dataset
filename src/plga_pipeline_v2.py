
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy.optimize import curve_fit

# Set Plotting Style
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

class PLGAPrecisionPipeline:
    def __init__(self, raw_path, initial_path):
        self.raw_df = pd.read_excel(raw_path)
        self.initial_df = pd.read_excel(initial_path)
        self.df = None
        self.targets = ['Peppas_n', 'Peppas_K', 'Burst_24h']
        self.models = {}
        self.results = {}
        self.ad_metrics = {}

    def _calculate_rdkit_descriptors(self, smiles):
        if pd.isna(smiles): return [np.nan]*6
        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol: return [np.nan]*6
            return [
                Descriptors.MolLogP(mol),
                Descriptors.TPSA(mol),
                Descriptors.ExactMolWt(mol),
                Lipinski.NumHDonors(mol),
                Lipinski.NumHAcceptors(mol),
                Lipinski.NumRotatableBonds(mol)
            ]
        except:
            return [np.nan]*6

    def engineer_features(self):
        print("STEP 1: Feature Engineering...")
        
        # 1. Consolidate Data
        # Extract unique formulation parameters from raw_df (which has Time/Release repeats)
        # We want one row per "Formulation Index" with all parameters
        # Columns to keep from raw_df:
        param_cols = ['Formulation Index', 'Drug MW', 'Drug LogP', 'Drug TPSA', 'Polymer MW', 'LA/GA', 
                      'Particle Size', 'Drug Loading Capacity', 'Drug Encapsulation Efficiency']
        
        # Check which columns actually exist in raw_df
        existing_cols = [c for c in param_cols if c in self.raw_df.columns]
        formulation_params = self.raw_df[existing_cols].drop_duplicates(subset='Formulation Index')
        
        # Merge with initial_df to get Drug SMILES if needed (or other metadata)
        # initial_df has SMILES
        if 'Drug SMILES' in self.initial_df.columns:
            formulation_params = formulation_params.merge(
                self.initial_df[['Formulation Index', 'Drug SMILES']], 
                on='Formulation Index', 
                how='left'
            )
        
        # 2. Chemical Featurization (RDKit)
        print("  - Calculating RDKit descriptors...")
        # Recalculate Descriptors from SMILES to ensure we have "Computed" values vs "Reported" values
        # And specifically exact features like NumHDonors which might not be in raw_df
        if 'Drug SMILES' in formulation_params.columns:
            formulation_params['RDKit_Descriptors'] = formulation_params['Drug SMILES'].apply(self._calculate_rdkit_descriptors)
            
            rdkit_cols = ['MolLogP', 'TPSA', 'ExactMolWt', 'NumHDonors', 'NumHAcceptors', 'RotatableBonds']
            rdkit_df = pd.DataFrame(formulation_params['RDKit_Descriptors'].tolist(), columns=rdkit_cols)
            # Concat or assign? assign is safer to align indices
            # reset_index to be sure
            formulation_params = formulation_params.reset_index(drop=True)
            rdkit_df = rdkit_df.reset_index(drop=True)
            
            formulation_params = pd.concat([formulation_params, rdkit_df], axis=1)
        
        # 3. Polymer Engineering
        print("  - Engineering Polymer features...")
        # LA/GA Ratio
        # Check if 'LA/GA' is in params
        if 'LA/GA' in formulation_params.columns:
             formulation_params['LA_GA_numeric'] = formulation_params['LA/GA']
        else:
             formulation_params['LA_GA_numeric'] = 1.0 # default
             
        # Polymer MW
        # raw_df has 'Polymer MW', initial_df has 'Polymer Mw'. 
        # We used 'Polymer MW' from raw_df in param_cols.
        if 'Polymer MW' not in formulation_params.columns:
             if 'Polymer Mw' in self.initial_df.columns:
                  # Merge it in
                  formulation_params = formulation_params.merge(self.initial_df[['Formulation Index', 'Polymer Mw']], on='Formulation Index')
                  formulation_params['Polymer MW'] = formulation_params['Polymer Mw']
        
        # Clean MW
        formulation_params['Polymer_Mw_Clean'] = formulation_params['Polymer MW'].fillna(1).replace(0, 1)
        
        # Hydrophilicity Index
        formulation_params['Hydrophilicity_Index'] = (1.0 / (formulation_params['LA_GA_numeric'] + 1e-6)) * (1.0 / formulation_params['Polymer_Mw_Clean'])
        
        print("DEBUG: formulation_params columns:", formulation_params.columns.tolist())
        print("DEBUG: formulation_params shape:", formulation_params.shape)
        self.df = formulation_params

    def engineer_targets(self):
        print("STEP 2: Target Engineering (Mechanistic)...")
        results = []
        
        grouped = self.raw_df.groupby('Formulation Index')
        
        for idx, group in grouped:
            if len(group) < 5: continue # Filter < 5 points
            
            group = group.sort_values('Time')
            t = group['Time'].values
            y = group['Release'].values
            
            # Target C: 24h Burst
            # Interpolate or find closest
            try:
                # Simple linear interp for 24h
                burst_24 = np.interp(24, t, y)
            except:
                burst_24 = np.nan
                
            # Korsmeyer-Peppas: Mt/Minf = K * t^n
            # Fit to first 60% release
            mask = y < 60
            if mask.sum() < 3: # Need points for fit
                # Fallback: fit all if max < 60, or take first N
                if len(y) >= 3:
                    t_fit = t[:5]
                    y_fit = y[:5]
                else:
                    continue
            else:
                t_fit = t[mask]
                y_fit = y[mask]
                
            # Log-Log Fit: log(Q) = log(K) + n*log(t)
            # Avoid t=0, y=0
            valid = (t_fit > 0) & (y_fit > 0)
            t_log = np.log(t_fit[valid])
            y_log = np.log(y_fit[valid])
            
            if len(t_log) < 3: continue
            
            try:
                slope, intercept = np.polyfit(t_log, y_log, 1)
                n = slope
                K = np.exp(intercept)
                
                # Bounds check (Physical limits)
                # n usually 0 to 1. K > 0.
                if n < 0 or n > 2: n = np.nan # Outlier
                    
                results.append({
                    'Formulation Index': idx,
                    'Peppas_n': n,
                    'Peppas_K': K,
                    'Burst_24h': burst_24
                })
            except:
                continue
                
        target_df = pd.DataFrame(results)
        print("DEBUG: target_df shape:", target_df.shape)
        # Check duplicate keys in target_df?
        if target_df['Formulation Index'].duplicated().any():
             print("WARNING: Duplicates in target_df Formulation Index!")
             
        self.df = self.df.merge(target_df, on='Formulation Index', how='inner')
        print(f"  - Features + Targets merged. Final shape: {self.df.shape}")

    def build_ensemble(self):
        print("STEP 3: Stacked Ensemble Architecture...")
        
        # Defined Level 0 Models
        rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
        xgb_mod = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, n_jobs=-1, objective='reg:squarederror')
        svr = SVR(kernel='rbf', C=10, gamma='scale')
        
        # Level 1 Meta-Learner
        ridge = Ridge(alpha=1.0)
        
        # Stacking Ensemble
        self.ensemble = StackingRegressor(
            estimators=[('rf', rf), ('xgb', xgb_mod), ('svr', svr)],
            final_estimator=ridge,
            cv=5, # Internal CV for stacking
            n_jobs=-1
        )
        
    def train_and_validate(self):
        print("STEP 3b: Training & Validation (10-Fold Grouped)...")
        print("DEBUG: Columns in self.df:", self.df.columns.tolist())
        
        # Features
        feature_cols = ['Drug MW', 'Drug LogP', 'Drug TPSA', 'MolLogP', 'TPSA', 'ExactMolWt', 
                        'NumHDonors', 'NumHAcceptors', 'RotatableBonds',
                        'Polymer MW', 'LA_GA_numeric', 'Hydrophilicity_Index',
                        'Particle Size', 'Drug Loading Capacity', 'Drug Encapsulation Efficiency']
        
        # Clean Features (fill NaNs)
        X = self.df[feature_cols].copy()
        X = X.fillna(X.mean())
        
        groups = self.df['Formulation Index']
        gkf = GroupKFold(n_splits=10)
        
        metrics_list = []
        
        for target in self.targets:
            print(f"  - Training for {target}...")
            y = self.df[target].copy()
            
            # Remove NaNs
            valid_mask = y.notna()
            X_curr = X[valid_mask].values
            y_curr = y[valid_mask].values
            groups_curr = groups[valid_mask].values
            
            # Storage
            all_actual = []
            all_pred = []
            all_std = []
            all_indices = []
            
            # Manual CV Loop for Uncertainty
            for train_idx, test_idx in gkf.split(X_curr, y_curr, groups=groups_curr):
                X_train, X_test = X_curr[train_idx], X_curr[test_idx]
                y_train, y_test = y_curr[train_idx], y_curr[test_idx]
                
                # Preprocessing
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Fit Ensemble
                # Re-instantiate to avoid leakage
                rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
                xgb_mod = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=6, n_jobs=-1, objective='reg:squarederror')
                svr = SVR(kernel='rbf', C=10, gamma='scale')
                
                stack = StackingRegressor(
                    estimators=[('rf', rf), ('xgb', xgb_mod), ('svr', svr)],
                    final_estimator=Ridge(alpha=1.0),
                    cv=3,
                    n_jobs=-1
                )
                
                stack.fit(X_train_scaled, y_train)
                preds = stack.predict(X_test_scaled)
                
                # Uncertainty (Std Dev of Base Learners)
                # Access fitted base learners
                base_preds = []
                for name, est in stack.named_estimators_.items():
                    base_preds.append(est.predict(X_test_scaled))
                
                # Add overall variance
                ensemble_std = np.std(base_preds, axis=0)
                
                all_actual.extend(y_test)
                all_pred.extend(preds)
                all_std.extend(ensemble_std)
                all_indices.extend(test_idx) # Keep track? Not strictly needed if sequential, but good for debug
                
            # Metrics
            all_actual = np.array(all_actual)
            all_pred = np.array(all_pred)
            all_std = np.array(all_std)
            
            r2 = r2_score(all_actual, all_pred)
            mae = mean_absolute_error(all_actual, all_pred)
            rmse = np.sqrt(mean_squared_error(all_actual, all_pred))
            
            print(f"    {target}: R2={r2:.3f}, MAE={mae:.3f}")
            
            metrics_list.append({'Target': target, 'R2': r2, 'MAE': mae, 'RMSE': rmse})
            
            self.results[target] = pd.DataFrame({
                'Actual': all_actual,
                'Predicted': all_pred,
                'Residuals': all_actual - all_pred,
                'Uncertainty': all_std
            })
            
            # Refit on full data for export
            final_pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('model', self.ensemble)
            ])
            final_pipe.fit(X_curr, y_curr)
            self.models[target] = final_pipe

        # BURST CLASSIFICATION
        print("  - Running Burst Classification...")
        # Create Classes: Low (<10), Med (10-40), High (>40)
        burst_y = self.df['Burst_24h'].copy()
        # Drop NaNs
        valid_b = burst_y.notna()
        X_b = X[valid_b].values
        y_b_val = burst_y[valid_b].values
        groups_b = groups[valid_b].values
        
        y_class = np.zeros_like(y_b_val, dtype=int)
        y_class[(y_b_val >= 10) & (y_b_val < 40)] = 1
        y_class[y_b_val >= 40] = 2
        
        clf = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, n_jobs=-1, 
                                use_label_encoder=False, objective='multi:softprob', num_class=3, eval_metric='mlogloss')
        
        # Cross val
        class_preds = cross_val_predict(clf, X_b, y_class, cv=gkf, groups=groups_b)
        
        from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
        acc = accuracy_score(y_class, class_preds)
        print(f"    Burst Classification Accuracy: {acc:.3f}")
        
        self.results['Burst_Class'] = {
            'Actual': y_class,
            'Predicted': class_preds,
            'ConfusionMatrix': confusion_matrix(y_class, class_preds)
        }
        metrics_list.append({'Target': 'Burst_Class', 'R2': np.nan, 'MAE': np.nan, 'RMSE': np.nan, 'Accuracy': acc})

        pd.DataFrame(metrics_list).to_csv('performance_metrics.csv', index=False)
        joblib.dump(self.models, 'Final_Model.joblib')

    def analyze_applicability_domain(self):
        print("STEP 4: Applicability Domain (Williams Plot)...")
        # Leverage (Hat Matrix diagonal) vs Standardized Residuals
        # H = X(X^TX)^-1X^T
        
        for target in self.targets:
            res_df = self.results[target]
            
            # Reconstruct X for valid rows used in training
            feature_cols = ['Drug MW', 'Drug LogP', 'Drug TPSA', 'MolLogP', 'TPSA', 'ExactMolWt', 
                        'NumHDonors', 'NumHAcceptors', 'RotatableBonds',
                        'Polymer MW', 'LA_GA_numeric', 'Hydrophilicity_Index',
                        'Particle Size', 'Drug Loading Capacity', 'Drug Encapsulation Efficiency']
            
            valid_idx = self.df[self.df[target].notna()].index
            X = self.df.loc[valid_idx, feature_cols].fillna(0).values
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Calculate Hat Matrix Diagonal (Leverage)
            # H = diag(X (X.T X)^-1 X.T)
            # Use pseudo-inverse for stability
            try:
                H = np.dot(X_scaled, np.linalg.pinv(np.dot(X_scaled.T, X_scaled)))
                H = np.dot(H, X_scaled.T)
                leverage = np.diagonal(H)
            except:
                leverage = np.zeros(len(X))
                
            # Standardized Residuals
            residuals = res_df['Residuals'].values
            std_resid = residuals / (np.std(residuals) + 1e-6)
            
            res_df['Leverage'] = leverage
            res_df['Std_Residual'] = std_resid
            
            # Define AD limits
            # Warning Leverage h* = 3p/n
            p = X.shape[1]
            n = X.shape[0]
            h_star = 3 * p / n
            
            # High Certainty Subset
            high_cert = res_df[(res_df['Leverage'] < h_star) & (np.abs(res_df['Std_Residual']) < 3)]
            
            acc_full = r2_score(res_df['Actual'], res_df['Predicted'])
            acc_ad = r2_score(high_cert['Actual'], high_cert['Predicted'])
            
            print(f"  {target} AD Analysis:")
            print(f"    Full R2: {acc_full:.4f}")
            print(f"    High-Certainty R2: {acc_ad:.4f} (Coverage: {len(high_cert)/len(res_df):.1%})")
            
            self.ad_metrics[target] = {'h_star': h_star, 'data': res_df}

    def generate_visualizations(self):
        print("STEP 5: Visualization...")
        
        # Figure 1: Mechanism Map (n values)
        plt.figure(figsize=(10, 6))
        
        # Determine mechanism zones
        # 0.43 = Fickian, 0.85 = Case II
        
        n_data = self.results['Peppas_n']
        curr_min = min(n_data['Actual'].min(), n_data['Predicted'].min())
        curr_max = max(n_data['Actual'].max(), n_data['Predicted'].max())
        
        sns.scatterplot(data=n_data, x='Actual', y='Predicted', alpha=0.6, edgecolor='w')
        plt.plot([curr_min, curr_max], [curr_min, curr_max], 'k--', lw=2)
        
        # Add zones
        plt.axvspan(0.3, 0.5, color='green', alpha=0.1, label='Fickian Diffusion')
        plt.axvspan(0.8, 0.9, color='orange', alpha=0.1, label='Case II Relaxation')
        
        plt.title('Mechanism Map: Predicted vs Actual Release Exponent (n)')
        plt.xlabel('Actual n')
        plt.ylabel('Predicted n')
        plt.legend()
        plt.tight_layout()
        plt.savefig('Figure1_MechanismMap.png')
        
        # Figure 2: AD Plot for Burst
        plt.figure(figsize=(10, 6))
        ad_data = self.ad_metrics['Burst_24h']['data']
        h_star = self.ad_metrics['Burst_24h']['h_star']
        
        plt.scatter(ad_data['Leverage'], ad_data['Std_Residual'], alpha=0.6, c='blue')
        plt.axhline(y=3, color='r', linestyle='--')
        plt.axhline(y=-3, color='r', linestyle='--')
        plt.axvline(x=h_star, color='r', linestyle='--', label='Warning Leverage $h^*$')
        
        plt.fill_between([0, h_star], -3, 3, color='green', alpha=0.1, label='Applicability Domain')
        
        plt.title('Applicability Domain: Williams Plot (Burst Release)')
        plt.xlabel('Leverage (Model Space Distance)')
        plt.ylabel('Standardized Residuals')
        plt.legend()
        plt.tight_layout()
        plt.savefig('Figure2_ApplicabilityDomain.png')
        
        # Figure 3: Feature Importance (From RandomForest in Ensemble)
        # Extract RF from Peppas_n model pipeline
        # pipe['model'].estimators_[0] is ('rf', RF)
        
        rf_model = self.models['Peppas_n'].named_steps['model'].estimators_[0]
        importances = rf_model.feature_importances_
        feature_cols = ['Drug MW', 'Drug LogP', 'Drug TPSA', 'MolLogP', 'TPSA', 'ExactMolWt', 
                        'NumHDonors', 'NumHAcceptors', 'RotatableBonds',
                        'Polymer MW', 'LA_GA_numeric', 'Hydrophilicity_Index',
                        'Particle Size', 'Drug Loading Capacity', 'Drug Encapsulation Efficiency']
        
        imp_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances}).sort_values('Importance', ascending=False).head(10)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=imp_df, x='Importance', y='Feature', palette='viridis')
        plt.title('Top 10 Drivers of Release Mechanism (n)')
        plt.tight_layout()
        plt.savefig('Figure3_FeatureImportance.png')
        
        print("Visualizations saved.")

if __name__ == "__main__":
    pipeline = PLGAPrecisionPipeline('mp_dataset_processed.xlsx', 'mp_dataset_initial.xlsx')
    pipeline.engineer_features()
    pipeline.engineer_targets()
    pipeline.build_ensemble()
    pipeline.train_and_validate()
    pipeline.analyze_applicability_domain()
    pipeline.generate_visualizations()
    print("\n=== Precision Pipeline Complete ===")
    print("Artifacts Generated: Figures 1-3, Final_Model.joblib, performance_metrics.csv")
    
    # Export detailed predictions
    all_res = []
    for target, res in pipeline.results.items():
        if isinstance(res, pd.DataFrame):
            df = res.copy()
            df['Target'] = target
            all_res.append(df)
        elif isinstance(res, dict) and 'Actual' in res:
             # Convert Burst dict to DF
             df = pd.DataFrame({
                 'Actual': res['Actual'],
                 'Predicted': res['Predicted']
             })
             # Confusion Matrix is lost here but reproduced in visualization script if needed? 
             # Actually visualize_refinement computes it from Actual/Predicted so it's fine.
             df['Target'] = target
             all_res.append(df)

    if all_res:
        pd.concat(all_res).to_csv('all_predictions_and_uncertainty.csv', index=False)
        print("Detailed predictions exported to all_predictions_and_uncertainty.csv")
