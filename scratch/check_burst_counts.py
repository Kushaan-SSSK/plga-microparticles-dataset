import pandas as pd
import numpy as np

def main():
    raw_df = pd.read_excel('mp_dataset_processed.xlsx')
    results = []
    grouped = raw_df.groupby('Formulation Index')
    
    for idx, group in grouped:
        if len(group) < 5: continue
        group = group.sort_values('Time')
        t = group['Time'].values
        y = group['Release'].values
        burst_24 = np.interp(24, t, y)
        results.append(burst_24)
    
    burst_vals = np.array(results)
    print(f"Total: {len(burst_vals)}")
    print(f"Low (<0.10): {sum(burst_vals < 0.10)}")
    print(f"Med (0.10 to <0.40): {sum((burst_vals >= 0.10) & (burst_vals < 0.40))}")
    print(f"High (>=0.40): {sum(burst_vals >= 0.40)}")
    
    # Check for values exactly at boundaries
    print(f"Values == 0.10: {sum(burst_vals == 0.10)}")
    print(f"Values == 0.40: {sum(burst_vals == 0.40)}")
    
    # Sort and show boundary values
    sorted_burst = np.sort(burst_vals)
    print(f"Values near 0.40: {sorted_burst[(sorted_burst > 0.38) & (sorted_burst < 0.42)]}")

if __name__ == '__main__':
    main()
