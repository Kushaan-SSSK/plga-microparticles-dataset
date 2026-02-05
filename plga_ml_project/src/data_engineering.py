
import pandas as pd
import numpy as np
import scipy.interpolate
from scipy.interpolate import PchipInterpolator
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
try:
    from tsfresh.feature_extraction import extract_features
    from tsfresh.feature_extraction.feature_calculators import linear_trend, number_peaks
except ImportError:
    print("Warning: tsfresh not installed. Feature extraction may fail.")

class PLGADataProcessor:
    def __init__(self, raw_data_path, initial_data_path):
        self.raw_df = pd.read_excel(raw_data_path)
        self.initial_df = pd.read_excel(initial_data_path)
        self.roi_features = {}

    def interpolate_curves(self, n_points=100):
        """
        Interpolates release curves to standardized 0-100% time points using Pchip.
        """
        print("Interpolating curves with Pchip...")
        interpolated_data = []
        
        # Group by Formulation Index
        grouped = self.raw_df.groupby('Formulation Index')
        
        for idx, group in grouped:
            # Sort by time just in case
            group = group.sort_values('Time')
            
            t = group['Time'].values
            y = group['Release'].values
            
            # Ensure start at 0,0 if not present
            if t[0] != 0:
                t = np.insert(t, 0, 0)
                y = np.insert(y, 0, 0)
            
            # Create Pchip interpolator
            try:
                pchip = PchipInterpolator(t, y)
            except Exception as e:
                print(f"Error interpolating formulation {idx}: {e}")
                continue
            
            # Standardized time points (0 to max time for this formulation? 
            # Or 0 to 100% release? The prompt says "standardize all release curves to 100 points from 0 to 100% release".
            # This is slightly ambiguous. Usually it means normalizing time to reach 100% release.
            # OR it means just resampling the existing curve to 100 evenly spaced points up to the last time point.
            # "0 to 100% release" suggests TIME is the variable? 
            # Re-reading: "use a Pchip ... to standardize all release curves to 100 points from 0 to 100% release."
            # This implies the X axis should be % Release and Y axis Time? Or X is normalized Time? 
            # Usually strict comparison requires same time points.
            # Let's assume normalizing Time to 0-1 range (normalized by max time) or just resampling 100 points up to max time.
            # However, "from 0 to 100% release" sounds like we want to find Time at 1%, 2%, ... 100% release.
            # But release curves are the target.
            # Let's stick to: Resample curve at 100 evenly spaced TIME points from 0 to MaxTime of that curve.
            # OR better: The "standard" for ML comparison usually is "Time" as input, "Release" as output.
            # But the prompt says "standardize ... to 100 points".
            # Let's interpret as: 100 points evenly spaced from t=0 to t=max_t for that formulation.
            
            t_max = t.max()
            t_new = np.linspace(0, t_max, n_points)
            y_new = pchip(t_new)
            
            # Store Pchip representation for later use (e.g. finding t50)
            self.roi_features[idx] = {
                'pchip': pchip,
                't_max': t_max,
                'y_max': y_new.max()
            }
            
            # Create dataframe for this formulation
            # But actually we want to extract FEATURES from this curve.
            # Step 2 says "Kinetic Profile Cluster (K-Means on the Pchip-interpolated curves)".
            # So we need the vector of 100 points.
            
            interpolated_data.append({
                'Formulation Index': idx,
                'Interpolated_Release': y_new,
                'Interpolated_Time': t_new
            })
            
        return pd.DataFrame(interpolated_data)

    def extract_tsfresh_features(self):
        """
        Extracts key manual features: Burst Slope, Lag Phase, Peppas n.
        """
        print("Extracting tsfresh & kinetic features...")
        features = []
        
        for idx, data in self.roi_features.items():
            pchip = data['pchip']
            t_max = data['t_max']
            
            # 1. Initial Burst Slope (first 24h)
            # If t_max < 24, take slope to t_max
            t_burst = min(24.0, t_max)
            y_burst = pchip(t_burst)
            burst_slope = y_burst / t_burst if t_burst > 0 else 0
            
            # 2. Lag Phase Duration
            # Define as time until release > 5% (arbitrary but common threshold)
            # Find root of pchip(t) - 0.05 = 0
            t_lag = 0
            t_eval = np.linspace(0, t_max, 1000)
            y_eval = pchip(t_eval)
            lag_limit = 0.05 # 5% release
            # find first index where y > lag_limit
            gt_indices = np.where(y_eval > lag_limit)[0]
            if len(gt_indices) > 0:
                t_lag = t_eval[gt_indices[0]]
            
            # 3. Peppas 'n' exponent
            # Mt/Minf = k * t^n  => log(Release) = log(k) + n*log(t)
            # Slope of log-log plot for first 60% of release
            # Get points up to 60% release
            t_peppas = []
            y_peppas = []
            
            # sample points
            for t_val in np.linspace(1, t_max, 100): # start at 1 to avoid log(0)
                y_val = pchip(t_val)
                if y_val > 0 and (y_val / data['y_max']) < 0.6:
                    t_peppas.append(t_val)
                    y_peppas.append(y_val)
            
            n_exponent = 0.5 # default/fallback
            if len(t_peppas) > 5:
                # Linear regression on log-log
                slope, intercept = np.polyfit(np.log(t_peppas), np.log(y_peppas), 1)
                n_exponent = slope
            
            features.append({
                'Formulation Index': idx,
                'Burst_Slope': burst_slope,
                'Lag_Duration': t_lag,
                'Peppas_n': n_exponent
            })
            
        return pd.DataFrame(features)

    def calculate_chemical_features(self):
        """
        Featurizes drugs using RDKit and calculates Polymer H_index.
        """
        print("Calculating chemical features...")
        chem_data = []
        
        # Unique formulations from initial df
        # We need to act on the 'initial_df' which has SMILES
        
        for i, row in self.initial_df.iterrows():
            idx = row['Formulation Index']
            smiles = row.get('Drug SMILES', '')
            
            mol_logp = np.nan
            tpsa = np.nan
            h_donors = np.nan
            h_acceptors = np.nan
            rotatable_bonds = np.nan
            
            if pd.notna(smiles):
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        # Add Weight
                        mol_mw = Descriptors.MolWt(mol)
                        
                        mol_logp = Descriptors.MolLogP(mol)
                        tpsa = Descriptors.TPSA(mol)
                        h_donors = Lipinski.NumHDonors(mol)
                        h_acceptors = Lipinski.NumHAcceptors(mol)
                        rotatable_bonds = Lipinski.NumRotatableBonds(mol)
                    else:
                        mol_mw = np.nan
                except:
                    mol_mw = np.nan
                    pass
            else:
                mol_mw = np.nan

            # Polymer Hydrophilicity
            # H_index = (GA / (LA + GA)) * (1 / MW)
            # Dataset has 'LA/GA' ratio. 
            
            la_ga_ratio = row.get('LA/GA', 0)
            # Check for Polymer Mw (capital M, small w) or MW
            polymer_mw = row.get('Polymer Mw', row.get('Polymer MW', 0))
            if pd.isna(polymer_mw) or polymer_mw == 0: polymer_mw = 1
            
            ga_fraction = 1.0 / (la_ga_ratio + 1.0)
            # Check unit of MW. Usually kDa or Da. Assuming consistency.
            h_index = ga_fraction * (1.0 / polymer_mw)
            
            chem_data.append({
                'Formulation Index': idx,
                'Drug MW': mol_mw,
                'MolLogP': mol_logp,
                'TPSA': tpsa,
                'H_Donors': h_donors,
                'H_Acceptors': h_acceptors,
                'Rot_Bonds': rotatable_bonds,
                'H_Index': h_index
            })
            
        return pd.DataFrame(chem_data)

    def process_all(self):
        # 1. Interpolate
        df_interp = self.interpolate_curves()
        
        # 2. Kinetic Features
        df_kinetic = self.extract_tsfresh_features()
        
        # 3. Chemical Features
        df_chem = self.calculate_chemical_features()
        
        # 4. Merge everything
        # Start with initial formulations
        final_df = self.initial_df.copy()
        final_df = final_df.merge(df_kinetic, on='Formulation Index', how='left')
        final_df = final_df.merge(df_chem, on='Formulation Index', how='left')
        
        # Also add interpolated curves as vector? Or just keep for clustering later.
        # Step 2 says "Kinetic Profile Cluster ... on Pchip curves".
        # We should save the interpolated curves separately or as a serialized object.
        # For CSV export, maybe not.
        
        return final_df, df_interp

if __name__ == "__main__":
    processor = PLGADataProcessor(
        'mp_dataset_processed.xlsx', 
        'mp_dataset_initial.xlsx'
    )
    df_features, df_curves = processor.process_all()
    df_features.to_csv('engineered_features.csv', index=False)
    df_curves.to_pickle('interpolated_curves.pkl')
    print("Data Engineering Complete.")
