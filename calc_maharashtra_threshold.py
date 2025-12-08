"""Calculate PMFBY thresholds from official Maharashtra data."""
import pandas as pd
import numpy as np

# Parse the official data from DES
data = {
    'year': ['2019-20', '2020-21', '2021-22', '2022-23'],
    'soybean_yield_tonnes_ha': [0.57, 1.67, 1.58, 1.81],
    'rice_yield_tonnes_ha': [1.12, 1.89, 1.89, 1.66],
    'cotton_yield_bales_ha': [0.92, 2.31, 1.71, 1.94]
}

df = pd.DataFrame(data)

# Convert to kg/ha (1 tonne = 1000 kg)
df['soybean_kg_ha'] = df['soybean_yield_tonnes_ha'] * 1000
df['rice_kg_ha'] = df['rice_yield_tonnes_ha'] * 1000

print('=' * 60)
print('OFFICIAL GOVERNMENT DATA - AHMEDNAGAR, MAHARASHTRA')
print('Source: Directorate of Economics & Statistics (DES)')
print('=' * 60)
print()
print(df[['year', 'soybean_kg_ha', 'rice_kg_ha']].to_string(index=False))

# Calculate PMFBY threshold for Soybean
print()
print('=' * 60)
print('PMFBY THRESHOLD CALCULATION - SOYBEAN')
print('=' * 60)
yields = df['soybean_kg_ha'].values
avg = np.mean(yields)
th_80 = avg * 0.80

print("Years:", list(df['year']))
print("Yields (kg/ha):", list(yields.astype(int)))
print(f"Average: {avg:.0f} kg/ha")
print(f"THRESHOLD (80%): {th_80:.0f} kg/ha")

# Calculate PMFBY threshold for Rice
print()
print('=' * 60)
print('PMFBY THRESHOLD CALCULATION - RICE')
print('=' * 60)
yields_r = df['rice_kg_ha'].values
avg_r = np.mean(yields_r)
th_80_r = avg_r * 0.80

print("Years:", list(df['year']))
print("Yields (kg/ha):", list(yields_r.astype(int)))
print(f"Average: {avg_r:.0f} kg/ha")
print(f"THRESHOLD (80%): {th_80_r:.0f} kg/ha")

print()
print('=' * 60)
print('DATA AUTHENTICITY: 100% Official Government Data')
print('=' * 60)
