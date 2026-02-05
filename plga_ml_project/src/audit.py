
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder

class SensitivityAuditor:
    def __init__(self, dataset_path):
        self.df = pd.read_csv(dataset_path)
        # Preprocessing: Encoding categoricals
        self.le = LabelEncoder()
        # Identify categorical columns
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                self.df[col] = self.df[col].astype(str)
                self.df[col] = self.le.fit_transform(self.df[col])
                
        self.results = []
        
    def get_feature_sets(self):
        # Define feature sets
        all_cols = self.df.columns.tolist()
        targets = ['Y1_Burst', 'Y2_t50', 'Y3_Cluster']
        exclude = targets + ['Formulation Index', 'Drug', 'DOI', 'Article Title', 'Higuchi_MSE', 'Higuchi_K'] 
        # Higuchi features are "Physics Informed", maybe part of Advanced? Or separate? 
        # Prompt: "Run Model C: 'Advanced' (Process-heavy) parameters."
        # Feature Ablation: 
        # Model A: Full Dataset.
        # Model B: "Basic" (Drug MW, Polymer MW, Size).
        # Model C: "Advanced" (Process-heavy).
        
        # Identify columns
        # Basic
        basic_feats = ['Drug MW', 'Polymer MW', 'Particle Size', 'Drug LogP', 'Drug TPSA'] # Assuming these are basic
        
        # Advanced (Process)
        # Look for process params: 'Initial Drug-to-Polymer Ratio', 'Solubility Enhancer Concentration', 'LA/GA', 'Method' (if exists)
        # Let's find columns that match
        available = [c for c in all_cols if c not in exclude]
        
        feat_basic = [c for c in basic_feats if c in available]
        
        # Advanced: Everything that is NOT basic and NOT chemical properties?
        # Chemical properties: 'MolLogP', 'TPSA', 'H_Donors', 'H_Acceptors', 'Rot_Bonds', 'H_Index'.
        # Process: 'Initial Drug-to-Polymer Ratio', 'Solubility Enhancer Concentration', 'LA/GA', 'Method' (if specific col exists)
        # Let's define sets explicitly
        
        feat_full = available
        
        feat_advanced = [c for c in available if c not in feat_basic and 'LogP' not in c and 'TPSA' not in c and 'H_' not in c and 'Rot_Bonds' not in c]
        # This leaves: 'LA/GA', 'Initial Drug-to-Polymer Ratio', 'Drug Loading Capacity', 'Drug Encapsulation Efficiency', 'Solubility Enhancer Concentration', 'Burst_Slope', 'Lag_Duration', 'Peppas_n'
        # Wait, 'Burst_Slope', 'Lag_Duration', 'Peppas_n' are FEATURES derived from release curve.
        # They should NOT be input features if we are predicting Release characteristics?
        # The prompt says: "Target Definition: ... Y1: % Burst ... Y2: t50 ... Y3: Kinetic Profile Cluster".
        # So Burst Slope (tsfresh) is actually a proxy for Y1?
        # If we use Burst Slope as input to predict Burst (Y1), that's leakage.
        # "Step 1 ... Feature Extraction ... to extract 5 key features...: Initial Burst Slope...".
        # "Step 2 ... Target Definition: Y1: % Burst at 24h".
        # Yes, using 'Burst_Slope' to predict 'Y1_Burst' is 100% leakage.
        # So tsfresh features should be TARGETS or intermediate analysis, NOT inputs for prediction.
        # INPUTS should be Formulation Parameters (Drug, Polymer, Process).
        # OUTPUTS are Release Characteristics (Burst, t50, Curve Shape).
        # So I must EXCLUDE release-derived features from Inputs!
        
        leakage_cols = ['Burst_Slope', 'Lag_Duration', 'Peppas_n', 'Release', 'Time', 'interpolated_curves']
        feat_full = [c for c in feat_full if c not in leakage_cols]
        feat_basic = [c for c in feat_basic if c in feat_full]
        feat_advanced = [c for c in feat_advanced if c in feat_full]
        
        return {
            'Model A (Full)': feat_full,
            'Model B (Basic)': feat_basic,
            'Model C (Advanced)': feat_advanced
        }

    def run_ablation(self):
        print("Running Ablation Study...")
        feature_sets = self.get_feature_sets()
        targets = ['Y1_Burst', 'Y2_t50'] # Y3 is classification, let's stick to regression for R2 gap
        
        for target in targets:
            y = self.df[target].values
            
            for model_name, feats in feature_sets.items():
                if not feats:
                    continue
                
                X = self.df[feats].values
                # Simple XGB
                model = xgb.XGBRegressor(n_estimators=100, objective='reg:squarederror')
                scores = cross_val_score(model, X, y, cv=5, scoring='r2')
                mean_r2 = np.mean(scores)
                
                self.results.append({
                    'Target': target,
                    'Model': model_name,
                    'R2': mean_r2,
                    'Features': len(feats)
                })
                print(f"{target} - {model_name}: R2={mean_r2:.4f}")
                
    def calculate_informational_gap(self):
        print("Calculating Informational Gap...")
        # Gamma = 1 - R2_best
        # We need "best" model for each target (Model A)
        
        df_res = pd.DataFrame(self.results)
        gap_data = []
        
        for target in df_res['Target'].unique():
            best_r2 = df_res[df_res['Target'] == target]['R2'].max()
            gamma = 1 - best_r2
            gap_data.append({'Target': target, 'Gamma': gamma})
            print(f"Gap for {target}: {gamma:.4f}")
            
    def export_outliers(self):
        # Identify formulations where model failed most (High Residuals)
        # Train Full Model on Y2 (t50) as representative
        print("identifying outliers...")
        target = 'Y2_t50'
        feats = self.get_feature_sets()['Model A (Full)']
        X = self.df[feats].values
        y = self.df[target].values
        
        model = xgb.XGBRegressor(n_estimators=100)
        kf = KFold(n_splits=5)
        preds = np.zeros_like(y)
        
        for train_idx, val_idx in kf.split(X):
             model.fit(X[train_idx], y[train_idx])
             preds[val_idx] = model.predict(X[val_idx])
             
        residuals = np.abs(y - preds)
        self.df['Residual_Y2'] = residuals
        
        outliers = self.df.sort_values('Residual_Y2', ascending=False).head(20)
        outliers[['Formulation Index', target, 'Residual_Y2']].to_csv('outlier_formulations.csv', index=False)
        print("Outliers exported.")

if __name__ == "__main__":
    auditor = SensitivityAuditor('final_dataset_for_audit.csv')
    auditor.run_ablation()
    auditor.calculate_informational_gap()
    auditor.export_outliers()
    pd.DataFrame(auditor.results).to_csv('audit_results.csv', index=False)
    print("Audit Complete.")
