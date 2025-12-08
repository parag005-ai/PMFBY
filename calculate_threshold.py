"""
Process Official Crop Statistics and Calculate PMFBY Thresholds
Data Source: Agricultural Statistics at a Glance, Government of India
"""

import pandas as pd
import numpy as np

# Load official statistics
df = pd.read_csv('data/official_crop_statistics.csv')

# Calculate yield (kg/ha)
df['yield_kg_ha'] = ((df['production_'] / df['area_']) * 1000).round(2)

print('='*70)
print('OFFICIAL CROP STATISTICS - LOADED SUCCESSFULLY')
print('='*70)
print(f"Districts: {df['district_name'].nunique()}")
print(f"Crops: {df['crop'].unique().tolist()}")
print(f"Years: {df['crop_year'].min()} - {df['crop_year'].max()}")
print(f"Total Records: {len(df)}")
print()

# Show Karnal Rice data
print('='*70)
print('KARNAL RICE DATA (Haryana)')
print('='*70)
karnal_rice = df[(df['district_name']=='KARNAL') & (df['crop']=='Rice')]
karnal_rice = karnal_rice.sort_values('crop_year', ascending=False)
print(karnal_rice[['crop_year','area_','production_','yield_kg_ha']].to_string(index=False))

# Calculate PMFBY threshold
yields = karnal_rice['yield_kg_ha'].values
sorted_yields = np.sort(yields)

# Exclude 2 worst years (calamity years)
used_yields = sorted_yields[2:]  # Remove 2 lowest
excluded = sorted_yields[:2]

avg_yield = np.mean(used_yields)
threshold_80 = avg_yield * 0.80

print()
print('='*70)
print('PMFBY THRESHOLD CALCULATION (Official Formula)')
print('='*70)
print(f"All 7 years yields (kg/ha): {list(yields.round(0).astype(int))}")
print(f"Excluded calamity years:    {list(excluded.round(0).astype(int))}")
print(f"Used for average (5 years): {list(used_yields.round(0).astype(int))}")
print()
print(f"Average Yield:    {avg_yield:.0f} kg/ha")
print(f"Indemnity Level:  80%")
print(f"THRESHOLD YIELD:  {threshold_80:.0f} kg/ha")
print()
print(f"Formula: {avg_yield:.0f} x 80% = {threshold_80:.0f} kg/ha")
print()
print("Source: Based on Agricultural Statistics at a Glance, Govt of India")
print('='*70)
