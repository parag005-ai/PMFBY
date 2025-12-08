"""Process all official state data files."""
import pandas as pd
import numpy as np
import os

print("=" * 70)
print("PROCESSING ALL OFFICIAL DES DATA FILES")
print("=" * 70)

data_dir = 'data'
files = [
    'haryana_data_real.xls',
    'madhyapradesh_data.xls',
    'maharashtra_data.xls'
]

all_data = {}

for filename in files:
    filepath = os.path.join(data_dir, filename)
    if not os.path.exists(filepath):
        print(f"\n[SKIP] {filename} - not found")
        continue
    
    state = filename.replace('_data_real.xls', '').replace('_data.xls', '').upper()
    print(f"\n{'='*70}")
    print(f"STATE: {state}")
    print(f"File: {filename}")
    print('='*70)
    
    try:
        tables = pd.read_html(filepath)
        df = tables[0]
        
        # Save as CSV
        csv_path = filepath.replace('.xls', '_clean.csv')
        df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")
        print(f"Shape: {df.shape}")
        print(f"\nData Preview:")
        print(df.to_string())
        
        all_data[state] = df
        
    except Exception as e:
        print(f"Error: {e}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
for state, df in all_data.items():
    print(f"{state}: {df.shape[0]} rows x {df.shape[1]} cols")
