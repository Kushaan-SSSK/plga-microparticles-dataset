import pandas as pd
import numpy as np

def extract_all_metadata():
    """
    Comprehensive extraction of dataset metadata for the IJP paper.
    """
    
    # Load datasets
    df_initial = pd.read_excel('mp_dataset_initial.xlsx')
    df_processed = pd.read_excel('mp_dataset_processed.xlsx')
    
    print("=" * 80)
    print("1. DATASET PROVENANCE")
    print("=" * 80)
    
    # Total formulations
    n_formulations = df_initial['Formulation Index'].nunique()
    n_rows = len(df_processed)
    print(f"Total Formulation Indices (unique formulations): {n_formulations}")
    print(f"Total Rows (time-series data points): {n_rows}")
    
    # Source Papers (DOIs)
    if 'DOI' in df_initial.columns:
        dois = df_initial['DOI'].dropna().unique()
        print(f"\nNumber of Source Papers (unique DOIs): {len(dois)}")
        print("\n--- LIST OF ALL DOIs ---")
        for i, doi in enumerate(sorted(dois), 1):
            print(f"{i}. {doi.strip()}")
    
    print("\n" + "=" * 80)
    print("2. FULL LIST OF DRUGS")
    print("=" * 80)
    
    if 'Drug' in df_initial.columns:
        drugs = df_initial['Drug'].dropna().unique()
        print(f"Total Unique Drugs: {len(drugs)}")
        for i, drug in enumerate(sorted(drugs), 1):
            print(f"{i}. {drug.strip() if isinstance(drug, str) else drug}")
    
    print("\n" + "=" * 80)
    print("3. BURST RELEASE STATISTICS (from processed data if available)")
    print("=" * 80)
    
    # Try to load targets
    try:
        targets = pd.read_csv('performance_metrics.csv')
        print(targets.to_string())
    except:
        print("Could not load performance_metrics.csv")
    
    # Count burst classes if possible
    # Need to calculate from raw data
    # Burst is defined as release at 24h
    # Looking at the code: Burst_24h is interpolated from release data
    
    print("\n" + "=" * 80)
    print("4. FEATURE SET USED (from code)")
    print("=" * 80)
    features = [
        'Drug MW', 'Drug LogP', 'Drug TPSA',  # From raw data
        'MolLogP', 'TPSA', 'ExactMolWt', 'NumHDonors', 'NumHAcceptors', 'RotatableBonds',  # RDKit
        'Polymer MW', 'LA_GA_numeric', 'Hydrophilicity_Index',  # Polymer
        'Particle Size', 'Drug Loading Capacity', 'Drug Encapsulation Efficiency'  # Formulation
    ]
    print("Feature columns used in the model:")
    for f in features:
        print(f"  - {f}")
    
    print("\nConfirmed MISSING (not in dataset):")
    missing = ['Emulsion Type', 'PVA %', 'Solvent', 'Stirring Rate', 'Evaporation Temp']
    for m in missing:
        print(f"  - {m}")

if __name__ == "__main__":
    extract_all_metadata()
