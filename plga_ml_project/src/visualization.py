
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from math import pi
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

# Set style
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

class PLGAVisualizer:
    def __init__(self, dataset_path, original_data_path):
        self.df = pd.read_csv(dataset_path)
        self.raw_df = pd.read_excel(original_data_path)
        self.model = None
        self.feature_importance = None
        
    def classify_drugs(self):
        # Heuristic for drug class
        def get_class(mw):
            if pd.isna(mw): return 'Small Molecule' # Default
            if mw < 1000: return 'Small Molecule'
            elif mw < 10000: return 'Peptide'
            else: return 'Protein'
            
        if 'Drug MW' in self.df.columns:
             self.df['Drug_Class'] = self.df['Drug MW'].apply(get_class)
        else:
             print("Warning: Drug MW not found. Using random classes for demo.")
             self.df['Drug_Class'] = np.random.choice(['Small Molecule', 'Peptide'], size=len(self.df))
             
        print("Drugs classified.")
        
    def train_reference_model(self):
        # Train a model on Y2 (t50) to get residuals and importance
        print("Training reference model for visualization...")
        target = 'Y2_t50'
        # Drop non-features
        exclude = ['Formulation Index', 'Drug', 'DOI', 'Article Title', 'Higuchi_MSE', 'Higuchi_K', 
                   'Y1_Burst', 'Y2_t50', 'Y3_Cluster', 'Drug_Class', 'Residual_Y2', 'Release', 'Time',
                   'Burst_Slope', 'Lag_Duration', 'Peppas_n', 'Predicted', 'Residuals', 'Uncertainty', 'Std_Residual', 'Leverage', 'Z_Score']
        
        # Handle categoricals
        X_df = self.df.drop(columns=[c for c in exclude if c in self.df.columns], errors='ignore')
        
        # Encoder
        for col in X_df.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X_df[col] = le.fit_transform(X_df[col].astype(str))
            
        self.feature_names = X_df.columns.tolist()
        X = X_df.fillna(X_df.mean()).values
        y = self.df[target].values
        
        self.model = xgb.XGBRegressor(n_estimators=100)
        self.model.fit(X, y)
        
        # Residuals
        preds = self.model.predict(X)
        residuals = np.abs(y - preds)
        self.df['Residual_Y2'] = residuals
        
        # Importances
        self.feature_importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
    def plot_radar_chart(self):
        print("Generating Radar Chart...")
        # Metrics by Drug Class: R2 (proxy 1-MeanRelativeError?), MAE, Coverage (1-Missingness?)
        # Let's use Residuals as metric (inverted), and maybe data count.
        # "Radar Chart showing model performance (R2, MAE, Coverage)"
        # I'll calculate R2 per class.
        
        metrics = []
        classes = self.df['Drug_Class'].unique()
        
        for cls in classes:
            sub_df = self.df[self.df['Drug_Class'] == cls]
            if len(sub_df) < 5: continue
            
            # Recalculate R2 for this subset (using global preds would be better)
            # But I only have residuals in self.df current state.
            # R2 = 1 - u/v. u = sum(res^2), v = sum((y-mean)^2)
            y = sub_df['Y2_t50']
            res = sub_df['Residual_Y2']
            u = np.sum(res**2)
            v = np.sum((y - y.mean())**2)
            r2 = 1 - u/v if v > 0 else 0
            
            mae = np.mean(res)
            
            # Coverage: % of samples in this class
            coverage = len(sub_df) / len(self.df)
            
            metrics.append({
                'Class': cls,
                'R2': max(0, r2), # clip neg
                'MAE_Inv': 1/(mae+1), # Invert for radar (higher is better)
                'Count_Log': np.log1p(len(sub_df))
            })
            
        # Normalize for Plot
        radar_df = pd.DataFrame(metrics)
        scaler = MinMaxScaler()
        plot_data = radar_df.copy()
        plot_data[['R2', 'MAE_Inv', 'Count_Log']] = scaler.fit_transform(plot_data[['R2', 'MAE_Inv', 'Count_Log']])
        
        # Plot
        categories = ['R2', 'MAE_Inv', 'Count_Log']
        N = len(categories)
        
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        
        for i, row in plot_data.iterrows():
            values = row[categories].values.flatten().tolist()
            values += values[:1]
            ax.plot(angles, values, linewidth=1, linestyle='solid', label=row['Class'])
            ax.fill(angles, values, alpha=0.1)
            
        plt.xticks(angles[:-1], categories)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.title('Model Performance by Drug Class', y=1.1)
        plt.tight_layout()
        plt.savefig('Figure1_Radar.png')
        print("Figure 1 saved.")

    def plot_deficit_heatmap(self):
        print("Generating Deficit Heatmap...")
        # Correlate Missingness dict with Residuals
        # Need to map back to original data to find missing cells
        # Merge raw_df with df residuals on Formulation Index
        
        merged = self.raw_df.merge(self.df[['Formulation Index', 'Residual_Y2']], on='Formulation Index')
        
        # Identify "Informational" columns (parameters)
        info_cols = ['Drug MW', 'Polymer Mw', 'Polymer Mn', 'PDI', 'LA/GA', 'Particle Size', 
                     'Drug Loading Capacity', 'Drug Encapsulation Efficiency', 'Formulation Method',
                     'Polymer Molecular Weight (unit not specified)'] 
        # Check which exists
        cols_to_check = [c for c in info_cols if c in merged.columns]
        
        # Create binary missingness matrix
        missingness = merged[cols_to_check].isnull().astype(int)
        
        # Filter out constant columns (variance == 0) to avoid NaN correlations
        missingness = missingness.loc[:, missingness.var() > 0]
        
        missingness['Residual'] = merged['Residual_Y2']
        
        # Correlation
        corr = missingness.corr()['Residual'].drop('Residual').sort_values()
        
        # Plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr.to_frame(), annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation: Missing Parameter vs Prediction Error')
        plt.ylabel('Parameter Missing in Literature')
        plt.tight_layout()
        plt.savefig('Figure2_Deficit_Heatmap.png')
        print("Figure 2 saved.")

    def generate_miadr_table(self):
        print("Generating MIADR Table...")
        # Feature Importance > 5% (0.05)
        miadr = self.feature_importance[self.feature_importance['Importance'] > 0.05]
        miadr.to_csv('Table1_MIADR.csv', index=False)
        print("Table 1 saved.")

if __name__ == "__main__":
    viz = PLGAVisualizer('final_dataset_for_audit.csv', 'mp_dataset_initial.xlsx')
    viz.classify_drugs()
    viz.train_reference_model()
    viz.plot_radar_chart()
    viz.plot_deficit_heatmap()
    viz.generate_miadr_table()
    print("Visualization Complete.")
