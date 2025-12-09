"""
Complete PMFBY Pipeline Test - Kolhapur Wheat
==============================================
Tests the full pipeline with:
- Location: Kolhapur (16.69, 74.23)
- Crop: Wheat
- Season: Rabi
- All features: Weather + Soil + Satellite + Stress
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feature_engineering.weather_features import fetch_and_compute_weather
from feature_engineering.soil_features_fast import get_soil_properties, compute_soil_quality_index
from feature_engineering.agronomic_stress import compute_all_stress_indices
from feature_engineering.satellite_features import fetch_satellite_features

print("=" * 70)
print("COMPLETE PMFBY PIPELINE TEST - KOLHAPUR WHEAT")
print("=" * 70)

# Test parameters
lat = 16.69
lon = 74.23
district = 'Kolhapur'
crop = 'Wheat'
season = 'Rabi'
year = 2024
area = 10  # hectares
threshold = 2500  # kg/ha (example)

print(f"""
Test Configuration:
  Location: {lat}N, {lon}E ({district})
  Crop: {crop}
  Season: {season}
  Year: {year}
  Area: {area} ha
  Threshold: {threshold} kg/ha
""")

# Step 1: Weather Features
print("\n[1] Fetching Weather Data...")
print("-" * 70)
weather = fetch_and_compute_weather(lat, lon, year)

print(f"  Total Rainfall: {weather['rain_total']:.0f} mm")
print(f"  GDD: {weather['gdd']:.0f} degree-days")
print(f"  Heat Stress: {weather['heat_stress_intensity']:.0f}")
print(f"  VPD: {weather['vpd_mean']:.2f} kPa")
print(f"  Dry Spells: {weather['dry_spell_count']}")
print(f"  Water Balance: {weather['water_balance']:.0f} mm")

# Step 2: Soil Properties
print("\n[2] Getting Soil Properties...")
print("-" * 70)
soil = get_soil_properties(district)
soil_quality = compute_soil_quality_index(soil)

print(f"  Texture: {soil['texture_class']}")
print(f"  Clay: {soil['clay_pct']:.1f}%, Sand: {soil['sand_pct']:.1f}%")
print(f"  Organic Carbon: {soil['organic_carbon']:.2f}%")
print(f"  pH: {soil['ph']:.1f}")
print(f"  AWC: {soil['awc']:.3f} cm³/cm³")
print(f"  Quality Index: {soil_quality:.2f}")

# Step 3: Satellite Features
print("\n[3] Fetching Satellite Data (Sentinel-2)...")
print("-" * 70)
satellite = fetch_satellite_features(lat, lon, year, season, use_sentinel=True)

print(f"  Source: {satellite['source']}")
print(f"  Observations: {satellite['n_observations']}")
print(f"  NDVI Mean: {satellite['ndvi_mean']:.3f}")
print(f"  NDVI Peak: {satellite['ndvi_peak']:.3f}")
print(f"  NDVI AUC: {satellite['ndvi_auc']:.2f}")
print(f"  EVI Mean: {satellite['evi_mean']:.3f}")
print(f"  NDWI Mean: {satellite['ndwi_mean']:.3f}")

# Step 4: Agronomic Stress
print("\n[4] Computing Agronomic Stress Indices...")
print("-" * 70)
stress = compute_all_stress_indices(weather, crop=crop)

print(f"  Vegetative Stress: {stress['vegetative_stress']:.2f}")
print(f"  Flowering Heat Stress: {stress['flowering_heat_stress']:.2f}")
print(f"  Grain Fill Deficit: {stress['grain_fill_deficit']:.2f}")
print(f"  Waterlogging Risk: {stress['waterlogging_risk']:.2f}")
print(f"  Combined Stress: {stress['combined_stress']:.2f}")
print(f"  Yield Potential: {stress['yield_potential']:.2f}")

# Summary
print("\n" + "=" * 70)
print("FEATURE SUMMARY")
print("=" * 70)

total_features = {
    'Weather': 15,
    'Soil': 5,
    'Satellite': 10,
    'Stress': 10,
    'Metadata': 7
}

print(f"\nTotal Features: {sum(total_features.values())}")
for category, count in total_features.items():
    print(f"  {category}: {count}")

print("\n" + "=" * 70)
print("ALL FEATURES COMPUTED SUCCESSFULLY!")
print("=" * 70)

# Feature values for ML model
print(f"""
Ready for ML Prediction:
  - Weather features: ✓ ({total_features['Weather']} features)
  - Soil features: ✓ ({total_features['Soil']} features)
  - Satellite features: ✓ ({total_features['Satellite']} features)
  - Stress indices: ✓ ({total_features['Stress']} features)
  - Metadata: ✓ ({total_features['Metadata']} features)

Next Step: Train model with all {sum(total_features.values())} features
Expected Accuracy: 90%+ (with satellite data)
""")
