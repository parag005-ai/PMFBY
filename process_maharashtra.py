"""Process Maharashtra data files."""
import pandas as pd

# Read both files as HTML tables
print("=" * 70)
print("MAHARASHTRA DATA - OFFICIAL GOVERNMENT DATA")
print("=" * 70)

# File 1
print("\n[1] maharashtra_data.xls")
try:
    tables1 = pd.read_html('data/maharashtra_data.xls')
    df1 = tables1[0]
    print(f"    Shape: {df1.shape}")
    print(f"    Columns: {list(df1.columns)}")
    df1.to_csv('data/maharashtra_data_clean.csv', index=False)
    print("    Saved as: maharashtra_data_clean.csv")
    print()
    print(df1.to_string())
except Exception as e:
    print(f"    Error: {e}")

print("\n" + "=" * 70)

# File 2
print("\n[2] maharshtra_data2.xls")
try:
    tables2 = pd.read_html('data/maharshtra_data2.xls')
    df2 = tables2[0]
    print(f"    Shape: {df2.shape}")
    print(f"    Columns: {list(df2.columns)}")
    df2.to_csv('data/maharashtra_data2_clean.csv', index=False)
    print("    Saved as: maharashtra_data2_clean.csv")
    print()
    print(df2.to_string())
except Exception as e:
    print(f"    Error: {e}")

print("\n" + "=" * 70)
