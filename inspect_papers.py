import pandas as pd

def inspect_papers():
    print("Loading mp_dataset_all_papers.xlsx...")
    try:
        df = pd.read_excel('mp_dataset_all_papers.xlsx')
        print(f"Columns: {df.columns.tolist()}")
        
        # Look for relevant columns
        keywords = ['medium', 'buffer', 'ph', 'temp', 'vitro', 'vivo', 'method', 'end', 'group', 'acid', 'ester']
        for col in df.columns:
            if any(k in col.lower() for k in keywords):
                print(f"\nPotential Match: {col}")
                print(f"Unique values: {df[col].unique()[:20]}")
                
        # Also check for 'Paper' or 'Source' context
        if 'Paper ID' in df.columns:
            print(f"\nNumber of Papers: {df['Paper ID'].nunique()}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_papers()
