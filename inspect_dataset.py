import pandas as pd
import numpy as np

def inspect_data():
    print("Loading datasets...")
    # Load initial (raw-ish) and processed data
    try:
        df_initial = pd.read_excel('mp_dataset_initial.xlsx')
        print("Loaded mp_dataset_initial.xlsx")
        print(f"Columns: {df_initial.columns.tolist()}")
    except Exception as e:
        print(f"Error loading initial: {e}")
        df_initial = pd.DataFrame()

    try:
        df_raw = pd.read_excel('mp_dataset_processed.xlsx') 
        # Note: pipeline uses mp_dataset_processed.xlsx as 'raw_path' in init, 
        # but let's check what's actually in there. 
        # The pipeline calls clean data 'mp_dataset_processed.xlsx' usually? 
        # Let's check the pipeline code: 
        # pipeline = PLGAPrecisionPipeline('mp_dataset_processed.xlsx', 'mp_dataset_initial.xlsx')
        # So 'mp_dataset_processed.xlsx' contains the time-series likely?
        print("Loaded mp_dataset_processed.xlsx")
        print(f"Columns: {df_raw.columns.tolist()}")
    except Exception as e:
        print(f"Error loading processed: {e}")
        df_raw = pd.DataFrame()

    print("\n--- 1. Drug(s) Studied ---")
    if 'Drug Name' in df_initial.columns:
        drugs = df_initial['Drug Name'].unique()
        print(f"Unique Drugs ({len(drugs)}): {drugs}")
    elif 'Drug' in df_initial.columns:
        drugs = df_initial['Drug'].unique()
        print(f"Unique Drugs ({len(drugs)}): {drugs}")
    else:
        print("Drug Name column not found in initial.")

    print("\n--- 2. PLGA Details ---")
    # LA/GA Ratio
    if 'LA/GA' in df_initial.columns:
        laga = df_initial['LA/GA'].unique()
        print(f"LA/GA Ratios: {laga}")
    
    # MW Range
    if 'Polymer Mw' in df_initial.columns:
        mw = df_initial['Polymer Mw']
        print(f"Polymer Mw (Initial) - Min: {mw.min()}, Max: {mw.max()}, Mean: {mw.mean():.2f}")
    elif 'Polymer MW' in df_raw.columns:
        mw = df_raw['Polymer MW']
        print(f"Polymer MW (Raw) - Min: {mw.min()}, Max: {mw.max()}, Mean: {mw.mean():.2f}")

    # End Group
    end_group_cols = [c for c in df_initial.columns if 'End' in c or 'Group' in c or 'Acid' in c or 'Ester' in c]
    print(f"Potential End Group Columns: {end_group_cols}")
    if end_group_cols:
        for c in end_group_cols:
            print(f"  {c} unique: {df_initial[c].unique()[:10]}")

    print("\n--- 3. Particle Type ---")
    # Size
    if 'Particle Size' in df_initial.columns:
        ps = df_initial['Particle Size'] # Assuming microns?
        print(f"Particle Size (Initial) - Min: {ps.min()}, Max: {ps.max()}, Mean: {ps.mean():.2f}")
    elif 'Particle Size' in df_raw.columns:
        ps = df_raw['Particle Size']
        print(f"Particle Size (Raw) - Min: {ps.min()}, Max: {ps.max()}, Mean: {ps.mean():.2f}")
        
    print("\n--- 4. Release Context ---")
    context_cols = [c for c in df_initial.columns if 'Media' in c or 'pH' in c or 'Temp' in c or 'vitro' in c or 'vivo' in c]
    print(f"Potential Release Context Columns: {context_cols}")
    for c in context_cols:
         print(f"  {c} unique: {df_initial[c].unique()[:10]}")

if __name__ == "__main__":
    inspect_data()
