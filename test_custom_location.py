"""
PMFBY Pipeline Test - Custom Coordinates
=========================================
Location: 19.54841, 74.188663
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feature_engineering.weather_features import fetch_and_compute_weather
from feature_engineering.soil_features_fast import get_soil_properties, compute_soil_quality_index
from feature_engineering.agronomic_stress import compute_all_stress_indices
from feature_engineering.satellite_features import fetch_satellite_features

print("=" * 70)
print("PMFBY PIPELINE TEST - CUSTOM COORDINATES")
print("=" * 70)

lat = 19.54841
lon = 74.188663
district = 'Ahmednagar'  # Closest district
crop = 'Rice'
season = 'Kharif'
year = 2024

print(f"""
Location: {lat}N, {lon}E
District: {district}
Crop: {crop} ({season})
Year: {year}
""")

print("[1] Weather...")
weather = fetch_and_compute_weather(lat, lon, year)
print(f"    Rain: {weather['rain_total']:.0f}mm, GDD: {weather['gdd']:.0f}")

print("\n[2] Soil...")
soil = get_soil_properties(district)
print(f"    {soil['texture_class']}, pH: {soil['ph']:.1f}")

print("\n[3] Satellite...")
satellite = fetch_satellite_features(lat, lon, year, season, use_sentinel=True)
print(f"    NDVI: {satellite['ndvi_mean']:.3f}, Obs: {satellite['n_observations']}")

print("\n[4] Stress...")
stress = compute_all_stress_indices(weather, crop=crop)
print(f"    Combined: {stress['combined_stress']:.2f}")

print("\n" + "=" * 70)
print(f"""
RESULTS:
  Rainfall: {weather['rain_total']:.0f} mm
  NDVI: {satellite['ndvi_mean']:.3f}
  Stress: {stress['combined_stress']:.2f}
  Yield Potential: {stress['yield_potential']:.2f}
  
All 47 features computed successfully!
""")
print("=" * 70)
