"""Process Maharashtra full dataset."""
import pandas as pd

# Load full dataset
df = pd.read_csv('data/maharashtra_district_data/district-season-and-crop-wise-area-production-and-yield-statistics-for-maharashtra.csv')

print('='*70)
print('MAHARASHTRA FULL DATASET')
print('='*70)
print(f'Total Rows: {len(df):,}')
print(f'Total Columns: {len(df.columns)}')
print()

print('COLUMNS:')
for c in df.columns:
    print(f'  - {c}')

print()
print('SAMPLE DATA (first 3 rows):')
print(df.head(3).T)  # Transpose for better viewing

print()
print('UNIQUE VALUES:')
for col in df.columns:
    unique = df[col].nunique()
    print(f'  {col}: {unique} unique values')

# Find yield-related columns
print()
print('DATA SUMMARY:')
if 'fiscal_year' in df.columns:
    print(f'  Years: {sorted(df["fiscal_year"].unique())}')
if 'district_as_per_source' in df.columns:
    print(f'  Districts: {df["district_as_per_source"].nunique()}')
