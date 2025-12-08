"""
Run PMFBY prediction for specific coordinates
Location: 19.071591, 74.774179 (Ahmednagar, Maharashtra)
"""

import pickle
import numpy as np
import requests
from datetime import datetime

# Your specific coordinates
LAT = 19.071591
LON = 74.774179

print("=" * 70)
print("PMFBY YIELD PREDICTION - SPECIFIC LOCATION")
print("=" * 70)
print(f"\n[*] Farm Location: {LAT}N, {LON}E (Ahmednagar)")
print(f"[*] Year: {datetime.now().year}")

# Load model
print("\n[1] LOADING MODEL...")
with open('models/trained/yield_model_with_weather.pkl', 'rb') as f:
    model_data = pickle.load(f)
print(f"    [OK] Model loaded (R2 = {model_data['metrics']['r2']:.4f})")

# Fetch weather
print("\n[2] FETCHING WEATHER FROM NASA POWER...")
year = datetime.now().year - 1  # Use last year for complete data

url = "https://power.larc.nasa.gov/api/temporal/daily/point"
params = {
    'start': f"{year}0601", 'end': f"{year}1130",
    'latitude': LAT, 'longitude': LON,
    'community': 'AG',
    'parameters': 'PRECTOTCORR,T2M,T2M_MAX,T2M_MIN,RH2M',
    'format': 'JSON'
}

response = requests.get(url, params=params, timeout=60)
data = response.json()
params_data = data.get('properties', {}).get('parameter', {})

rain = [r for r in params_data.get('PRECTOTCORR', {}).values() if r > -900]
t2m = [t for t in params_data.get('T2M', {}).values() if t > -900]
t2m_max = [t for t in params_data.get('T2M_MAX', {}).values() if t > -900]
rh = [r for r in params_data.get('RH2M', {}).values() if r > -900]

weather = {
    'rain_total': sum(rain),
    'rain_days': sum(1 for r in rain if r > 1),
    'temp_mean': np.mean(t2m),
    'temp_max': max(t2m_max),
    'heat_days': sum(1 for t in t2m_max if t > 35),
    'humidity_mean': np.mean(rh)
}

print(f"    [OK] Weather Data for {year} (June-November):")
print(f"      - Total Rainfall: {weather['rain_total']:.1f} mm")
print(f"      - Rainy Days: {weather['rain_days']} days")
print(f"      - Mean Temperature: {weather['temp_mean']:.1f} C")
print(f"      - Max Temperature: {weather['temp_max']:.1f} C")
print(f"      - Heat Stress Days: {weather['heat_days']} days (>35C)")
print(f"      - Mean Humidity: {weather['humidity_mean']:.1f}%")

# Prepare features
print("\n[3] PREPARING FEATURES...")

# Use encoders from model
encoders = model_data['encoders']
district_encoded = encoders['district'].transform(['Ahmednagar'])[0]
crop_encoded = encoders['crop'].transform(['Rice'])[0]
season_encoded = encoders['season'].transform(['Kharif'])[0]

area_ha = 10  # 10 hectares

X = np.array([[
    district_encoded,      # X1: District
    crop_encoded,          # X2: Crop (Rice)
    season_encoded,        # X3: Season (Kharif)
    year,                  # X4: Year
    LAT,                   # X5: Latitude
    LON,                   # X6: Longitude
    np.log1p(area_ha),     # X7: log(Area+1)
    1,                     # X8: Is major crop (Rice = yes)
    1,                     # X9: Is Kharif
    weather['rain_total'], # X10
    weather['rain_days'],  # X11
    weather['temp_mean'],  # X12
    weather['temp_max'],   # X13
    weather['heat_days'],  # X14
    weather['humidity_mean'] # X15
]])

print("    [OK] Features prepared (15 features)")

# Predict
print("\n[4] PREDICTING YIELD...")
model = model_data['model']
predicted_yield = model.predict(X)[0]
mae = model_data['metrics']['mae']

print(f"    [OK] Predicted Yield: {predicted_yield:.0f} kg/ha")
print(f"    [OK] Confidence Interval: [{predicted_yield - 1.5*mae:.0f}, {predicted_yield + 1.5*mae:.0f}] kg/ha")

# PMFBY Calculation
print("\n[5] PMFBY LOSS CALCULATION...")
threshold = 1640  # Average Rice yield in Ahmednagar from DES data

shortfall = max(0, threshold - predicted_yield)
loss_pct = (shortfall / threshold) * 100 if shortfall > 0 else 0
claim_triggered = loss_pct >= 33

print(f"""
    +-------------------------------------------------------------------+
    |  PMFBY ASSESSMENT                                                 |
    +-------------------------------------------------------------------+
    |  Threshold Yield (DES Average): {threshold:>8} kg/ha                |
    |  Predicted Yield:               {predicted_yield:>8.0f} kg/ha                |
    |  Shortfall:                     {shortfall:>8.0f} kg/ha                |
    |  Loss Percentage:               {loss_pct:>8.2f}%                      |
    |  Claim Trigger (>=33%):         {'YES' if claim_triggered else 'NO':>8}                        |
    +-------------------------------------------------------------------+
""")

print("=" * 70)
print(f"Location: {LAT}, {LON}")
print(f"Crop: Rice (Kharif)")
print(f"Status: {'CLAIM TRIGGERED' if claim_triggered else 'NO CLAIM - Yield Above Threshold'}")
print("=" * 70)
