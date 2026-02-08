import pandas as pd

def extract_dois_and_drugs():
    df = pd.read_excel('mp_dataset_initial.xlsx')
    
    print("=" * 80)
    print("COMPLETE LIST OF DOIs (Paper Sources)")
    print("=" * 80)
    
    if 'DOI' in df.columns:
        dois = df['DOI'].dropna().unique()
        for i, doi in enumerate(sorted(dois), 1):
            clean_doi = str(doi).strip()
            print(f"{i}. https://doi.org/{clean_doi}")
    
    print("\n" + "=" * 80)
    print("COMPLETE LIST OF DRUGS")
    print("=" * 80)
    
    if 'Drug' in df.columns:
        drugs = df['Drug'].dropna().unique()
        for i, drug in enumerate(sorted(drugs), 1):
            print(f"{i}. {str(drug).strip()}")
    
    print("\n" + "=" * 80)
    print("BURST RELEASE CLASS COUNTS")
    print("=" * 80)
    
    # Load processed data to count burst classes
    df_proc = pd.read_excel('mp_dataset_processed.xlsx')
    
    # Get unique formulations with 24h release
    grouped = df_proc.groupby('Formulation Index')
    
    high_burst = 0
    low_burst = 0
    middle = 0
    
    import numpy as np
    
    for idx, group in grouped:
        t = group['Time'].values
        y = group['Release'].values
        
        try:
            burst_24 = np.interp(24, t, y)
            
            if burst_24 > 0.4:  # High burst
                high_burst += 1
            elif burst_24 < 0.1:  # Low burst
                low_burst += 1
            else:
                middle += 1
        except:
            pass
    
    print(f"High Burst (>40% at 24h):   {high_burst}")
    print(f"Low Burst (<10% at 24h):    {low_burst}")
    print(f"Middle (Excluded):          {middle}")
    print(f"Total:                      {high_burst + low_burst + middle}")

if __name__ == "__main__":
    extract_dois_and_drugs()
