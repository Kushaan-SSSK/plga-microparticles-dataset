import pandas as pd
import numpy as np

def fuzzy_search():
    print("Loading mp_dataset_initial.xlsx...")
    df = pd.read_excel('mp_dataset_initial.xlsx')
    
    keywords = ['pbs', 'buffer', 'ph 7.4', '37', 'acid', 'ester', 'capped', 'uncapped']
    
    print(f"Searching for keywords: {keywords}")
    
    # Convert entire DF to string
    df_str = df.astype(str).apply(lambda x: x.str.lower())
    
    found = False
    for col in df_str.columns:
        for key in keywords:
            matches = df_str[col].str.contains(key, na=False)
            if matches.any():
                print(f"Match in column '{col}':")
                print(df.loc[matches, col].unique()[:5])
                found = True
                
    if not found:
        print("No keywords found in the dataset values.")

    # Also check the notebook content just in case
    print("\nChecking notebook header...")
    try:
        with open('mp_dataset_data_analysis.ipynb', 'r', encoding='utf-8') as f:
            content = f.read()
            # simple dumb search
            if 'PBS' in content: print("Found 'PBS' in notebook.")
            if 'acid' in content.lower(): print("Found 'acid' in notebook.")
            if 'ester' in content.lower(): print("Found 'ester' in notebook.")
            if '37' in content: print("Found '37' in notebook.")
    except Exception as e:
        print(f"Error reading notebook: {e}")

if __name__ == "__main__":
    fuzzy_search()
