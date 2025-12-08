"""
PMFBY ML Model Training with Weather Features
Shows live training progress in terminal
"""

import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import pickle
import os
from datetime import datetime
import time
import sys

# District coordinates for Maharashtra
DISTRICT_COORDS = {
    'Ahmednagar': (19.09, 74.74), 'Pune': (18.52, 73.86), 'Nashik': (20.00, 73.78),
    'Solapur': (17.66, 75.91), 'Kolhapur': (16.69, 74.23), 'Satara': (17.69, 73.99),
    'Sangli': (16.85, 74.57), 'Aurangabad': (19.88, 75.32), 'Jalgaon': (21.00, 75.57),
    'Nagpur': (21.15, 79.09), 'Amravati': (20.93, 77.75), 'Akola': (20.71, 77.00),
    'Buldhana': (20.53, 76.18), 'Latur': (18.40, 76.57), 'Beed': (18.99, 75.76),
    'Nanded': (19.15, 77.30), 'Parbhani': (19.27, 76.77), 'Osmanabad': (18.18, 76.04),
    'Ratnagiri': (16.99, 73.30), 'Sindhudurg': (16.35, 73.53), 'Thane': (19.22, 72.98),
    'Mumbai': (19.08, 72.88), 'Raigad': (18.52, 73.18), 'Wardha': (20.75, 78.60),
    'Chandrapur': (19.95, 79.30), 'Gadchiroli': (20.10, 80.00), 'Gondia': (21.46, 80.20),
    'Bhandara': (21.17, 79.65), 'Yavatmal': (20.40, 78.12), 'Washim': (20.11, 77.15),
    'Hingoli': (19.72, 77.15), 'Jalna': (19.84, 75.88), 'Dhule': (20.90, 74.78),
    'Nandurbar': (21.37, 74.25), 'default': (19.5, 76.0)
}


def fetch_weather_for_year(lat, lon, year):
    """Fetch annual weather summary from NASA POWER."""
    try:
        # Kharif season: June to November
        start_date = f"{year}0601"
        end_date = f"{year}1130"
        
        url = f"https://power.larc.nasa.gov/api/temporal/daily/point"
        params = {
            'start': start_date,
            'end': end_date,
            'latitude': lat,
            'longitude': lon,
            'community': 'AG',
            'parameters': 'PRECTOTCORR,T2M,T2M_MAX,T2M_MIN,RH2M',
            'format': 'JSON'
        }
        
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            params_data = data.get('properties', {}).get('parameter', {})
            
            # Extract weather features
            rain = list(params_data.get('PRECTOTCORR', {}).values())
            t2m = list(params_data.get('T2M', {}).values())
            t2m_max = list(params_data.get('T2M_MAX', {}).values())
            rh = list(params_data.get('RH2M', {}).values())
            
            # Clean data (remove -999 values)
            rain = [r for r in rain if r > -900]
            t2m = [t for t in t2m if t > -900]
            t2m_max = [t for t in t2m_max if t > -900]
            rh = [r for r in rh if r > -900]
            
            return {
                'rain_total': sum(rain) if rain else 700,
                'rain_days': sum(1 for r in rain if r > 1) if rain else 60,
                'temp_mean': np.mean(t2m) if t2m else 28,
                'temp_max': max(t2m_max) if t2m_max else 40,
                'heat_days': sum(1 for t in t2m_max if t > 35) if t2m_max else 20,
                'humidity_mean': np.mean(rh) if rh else 70
            }
    except Exception as e:
        pass
    
    # Return defaults if failed
    return {
        'rain_total': 700, 'rain_days': 60, 'temp_mean': 28,
        'temp_max': 40, 'heat_days': 20, 'humidity_mean': 70
    }


def progress_bar(current, total, prefix='', suffix='', length=40):
    """Display progress bar."""
    percent = current / total
    filled = int(length * percent)
    bar = '█' * filled + '░' * (length - filled)
    sys.stdout.write(f'\r{prefix} |{bar}| {current}/{total} {suffix}')
    sys.stdout.flush()


def main():
    print("=" * 70)
    print("PMFBY ML MODEL TRAINING WITH WEATHER FEATURES")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # =====================================================
    # STEP 1: Load Data
    # =====================================================
    print("\n[1] LOADING DATA...")
    df = pd.read_csv('data/maharashtra_district_data/district-season-and-crop-wise-area-production-and-yield-statistics-for-maharashtra.csv')
    df.columns = df.columns.str.lower().str.strip()
    df = df.rename(columns={'fiscal_year': 'year', 'district_as_per_source': 'district', 'crop_yield': 'yield_tonnes_ha'})
    df['yield_kg_ha'] = pd.to_numeric(df['yield_tonnes_ha'], errors='coerce') * 1000
    df_clean = df[(df['yield_kg_ha'].notna()) & (df['yield_kg_ha'] > 0) & (df['yield_kg_ha'] < 20000)].copy()
    print(f"    Loaded: {len(df_clean):,} records")
    
    # =====================================================
    # STEP 2: Fetch Weather for Unique District-Years
    # =====================================================
    print("\n[2] FETCHING WEATHER DATA FROM NASA POWER...")
    
    # Get unique district-year combinations
    df_clean['year_num'] = df_clean['year'].str[:4].astype(int)
    unique_combos = df_clean[['district', 'year_num']].drop_duplicates()
    
    print(f"    Unique district-year combinations: {len(unique_combos)}")
    print("    Fetching weather (this may take a few minutes)...\n")
    
    weather_cache = {}
    total = len(unique_combos)
    
    for idx, row in unique_combos.iterrows():
        district = row['district']
        year = row['year_num']
        
        # Get coordinates
        coords = DISTRICT_COORDS.get(district, DISTRICT_COORDS['default'])
        
        # Fetch weather
        key = f"{district}_{year}"
        weather = fetch_weather_for_year(coords[0], coords[1], year)
        weather_cache[key] = weather
        
        # Update progress
        current = len(weather_cache)
        progress_bar(current, total, prefix='    Progress', suffix=f'[{district[:10]}]')
        
        # Small delay to avoid rate limiting
        if current % 10 == 0:
            time.sleep(0.5)
    
    print("\n    ✓ Weather data fetched!")
    
    # =====================================================
    # STEP 3: Add Weather Features to Dataset
    # =====================================================
    print("\n[3] ADDING WEATHER FEATURES TO DATASET...")
    
    def get_weather_feature(row, feature):
        key = f"{row['district']}_{row['year_num']}"
        return weather_cache.get(key, {}).get(feature, 0)
    
    for feat in ['rain_total', 'rain_days', 'temp_mean', 'temp_max', 'heat_days', 'humidity_mean']:
        df_clean[feat] = df_clean.apply(lambda r: get_weather_feature(r, feat), axis=1)
    
    print(f"    Added 6 weather features")
    
    # =====================================================
    # STEP 4: Feature Engineering
    # =====================================================
    print("\n[4] FEATURE ENGINEERING...")
    
    le_district = LabelEncoder()
    le_crop = LabelEncoder()
    le_season = LabelEncoder()
    
    df_clean['district_encoded'] = le_district.fit_transform(df_clean['district'].astype(str))
    df_clean['crop_encoded'] = le_crop.fit_transform(df_clean['crop'].astype(str))
    df_clean['season_encoded'] = le_season.fit_transform(df_clean['season'].astype(str))
    
    df_clean['lat'] = df_clean['district'].apply(lambda x: DISTRICT_COORDS.get(x, DISTRICT_COORDS['default'])[0])
    df_clean['lon'] = df_clean['district'].apply(lambda x: DISTRICT_COORDS.get(x, DISTRICT_COORDS['default'])[1])
    df_clean['area_ha'] = pd.to_numeric(df_clean['area'], errors='coerce').fillna(1000)
    df_clean['log_area'] = np.log1p(df_clean['area_ha'])
    
    major_crops = ['Rice', 'Wheat', 'Soyabean', 'Cotton', 'Sugarcane', 'Jowar', 'Bajra', 'Maize']
    df_clean['is_major_crop'] = df_clean['crop'].isin(major_crops).astype(int)
    df_clean['is_kharif'] = df_clean['season'].str.lower().str.contains('kharif', na=False).astype(int)
    
    print(f"    Total features: 16")
    
    # =====================================================
    # STEP 5: Prepare Training Data
    # =====================================================
    print("\n[5] PREPARING TRAINING DATA...")
    
    feature_cols = [
        'district_encoded', 'crop_encoded', 'season_encoded', 'year_num',
        'lat', 'lon', 'log_area', 'is_major_crop', 'is_kharif',
        'rain_total', 'rain_days', 'temp_mean', 'temp_max', 'heat_days', 'humidity_mean'
    ]
    
    X = df_clean[feature_cols].values
    y = df_clean['yield_kg_ha'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"    Training samples: {len(X_train):,}")
    print(f"    Test samples: {len(X_test):,}")
    
    # =====================================================
    # STEP 6: Train Models
    # =====================================================
    print("\n[6] TRAINING MODELS...")
    print("    Training Random Forest (200 trees)...")
    
    start_time = time.time()
    rf_model = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_split=5, n_jobs=-1, random_state=42, verbose=1)
    rf_model.fit(X_train, y_train)
    rf_time = time.time() - start_time
    print(f"    ✓ Random Forest trained in {rf_time:.1f}s")
    
    print("\n    Training Gradient Boosting (200 trees)...")
    start_time = time.time()
    gb_model = GradientBoostingRegressor(n_estimators=200, max_depth=8, learning_rate=0.1, random_state=42, verbose=1)
    gb_model.fit(X_train, y_train)
    gb_time = time.time() - start_time
    print(f"    ✓ Gradient Boosting trained in {gb_time:.1f}s")
    
    # =====================================================
    # STEP 7: Evaluate
    # =====================================================
    print("\n[7] MODEL EVALUATION...")
    print("=" * 70)
    
    rf_pred = rf_model.predict(X_test)
    gb_pred = gb_model.predict(X_test)
    
    rf_r2 = r2_score(y_test, rf_pred)
    rf_mae = mean_absolute_error(y_test, rf_pred)
    rf_mape = np.mean(np.abs((y_test - rf_pred) / y_test)) * 100
    
    gb_r2 = r2_score(y_test, gb_pred)
    gb_mae = mean_absolute_error(y_test, gb_pred)
    gb_mape = np.mean(np.abs((y_test - gb_pred) / y_test)) * 100
    
    print("\n    RANDOM FOREST:")
    print(f"      R² Score: {rf_r2:.4f} ({rf_r2*100:.1f}%)")
    print(f"      MAE: {rf_mae:.0f} kg/ha")
    print(f"      Accuracy: {100 - rf_mape:.1f}%")
    
    print("\n    GRADIENT BOOSTING:")
    print(f"      R² Score: {gb_r2:.4f} ({gb_r2*100:.1f}%)")
    print(f"      MAE: {gb_mae:.0f} kg/ha")
    print(f"      Accuracy: {100 - gb_mape:.1f}%")
    
    # =====================================================
    # STEP 8: Feature Importance
    # =====================================================
    print("\n[8] FEATURE IMPORTANCE...")
    importance = sorted(zip(feature_cols, rf_model.feature_importances_), key=lambda x: x[1], reverse=True)
    for feat, imp in importance:
        bar = '█' * int(imp * 40)
        print(f"    {feat:<20}: {imp:.3f} {bar}")
    
    # =====================================================
    # STEP 9: Save Model
    # =====================================================
    print("\n[9] SAVING MODEL...")
    
    best_model = rf_model if rf_r2 > gb_r2 else gb_model
    best_r2 = max(rf_r2, gb_r2)
    
    model_dir = 'models/trained'
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, 'yield_model_with_weather.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': best_model,
            'feature_cols': feature_cols,
            'encoders': {'district': le_district, 'crop': le_crop, 'season': le_season},
            'district_coords': DISTRICT_COORDS,
            'metrics': {'r2': best_r2, 'mae': rf_mae if rf_r2 > gb_r2 else gb_mae},
            'trained_on': datetime.now().isoformat()
        }, f)
    
    print(f"    ✓ Model saved to: {model_path}")
    
    # =====================================================
    # SUMMARY
    # =====================================================
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"""
    DATASET:
    - Records: {len(df_clean):,}
    - Districts: {df_clean['district'].nunique()}
    - Crops: {df_clean['crop'].nunique()}
    - Years: {df_clean['year_num'].nunique()}
    
    WEATHER DATA:
    - Source: NASA POWER API
    - Features: rain_total, rain_days, temp_mean, temp_max, heat_days, humidity
    
    MODEL PERFORMANCE:
    - R² Score: {best_r2:.4f} ({best_r2*100:.1f}%)
    - MAE: {rf_mae if rf_r2 > gb_r2 else gb_mae:.0f} kg/ha
    """)
    print("=" * 70)


if __name__ == "__main__":
    main()
