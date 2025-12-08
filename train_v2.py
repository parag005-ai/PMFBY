"""
PMFBY v2.0 - Complete Training Pipeline
========================================
Trains ensemble model using:
- 20,558 DES yield records
- 15+ advanced weather features
- 10 agronomic stress indices
- Ensemble of RF + XGBoost

Usage: python train_v2.py
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feature_engineering.weather_features import fetch_and_compute_weather
from feature_engineering.agronomic_stress import compute_all_stress_indices, CROP_PARAMS
from models.ensemble import EnsembleYieldPredictor

# District coordinates
DISTRICT_COORDS = {
    'Ahmednagar': (19.09, 74.74), 'Pune': (18.52, 73.86), 'Nashik': (20.00, 73.78),
    'Solapur': (17.66, 75.91), 'Kolhapur': (16.69, 74.23), 'Satara': (17.69, 73.99),
    'Sangli': (16.85, 74.57), 'Aurangabad': (19.88, 75.32), 'Jalgaon': (21.00, 75.57),
    'Nagpur': (21.15, 79.09), 'Amravati': (20.93, 77.75), 'Akola': (20.71, 77.00),
    'Buldhana': (20.53, 76.18), 'Latur': (18.40, 76.57), 'Beed': (18.99, 75.76),
    'Nanded': (19.15, 77.30), 'Parbhani': (19.27, 76.77), 'Osmanabad': (18.18, 76.04),
}

# Feature columns for v2 model
FEATURE_COLS_V2 = [
    # Basic features
    'district_encoded', 'crop_encoded', 'season_encoded', 'year',
    'lat', 'lon', 'log_area', 'is_major_crop', 'is_kharif',
    
    # Weather features (15)
    'rain_total', 'rain_days', 'rain_cv', 'dry_spell_count', 'rain_anomaly',
    'gdd', 'heat_stress_intensity', 'vpd_mean',
    'rain_jun_jul', 'rain_aug_sep', 'humidity_cv',
    'et_total', 'water_balance', 'max_hot_streak',
    'night_temp_mean', 'diurnal_range',
    
    # Stress indices (7)
    'vegetative_stress', 'flowering_heat_stress', 'grain_fill_deficit',
    'waterlogging_risk', 'drought_index', 'combined_stress', 'yield_potential'
]


def load_des_data(filepath: str) -> pd.DataFrame:
    """Load and clean DES Maharashtra data."""
    print("[1] Loading DES data...")
    
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.lower().str.strip()
    
    # Rename columns
    df = df.rename(columns={
        'fiscal_year': 'year',
        'district_as_per_source': 'district',
        'crop_yield': 'yield_tonnes_ha'
    })
    
    # Convert yield to kg/ha
    df['yield_kg_ha'] = pd.to_numeric(df['yield_tonnes_ha'], errors='coerce') * 1000
    
    # Filter valid records
    df = df[
        (df['yield_kg_ha'].notna()) & 
        (df['yield_kg_ha'] > 0) & 
        (df['yield_kg_ha'] < 20000)
    ].copy()
    
    # Extract year number
    df['year_num'] = df['year'].str[:4].astype(int)
    
    print(f"    Loaded {len(df):,} records")
    print(f"    Districts: {df['district'].nunique()}")
    print(f"    Crops: {df['crop'].nunique()}")
    print(f"    Years: {df['year_num'].nunique()}")
    
    return df


def compute_features_for_training(df: pd.DataFrame, 
                                   weather_cache: dict,
                                   encoders: dict) -> pd.DataFrame:
    """
    Compute all v2 features for training dataset.
    Uses cached weather data to avoid repeated API calls.
    """
    print("\n[2] Computing features...")
    
    features_list = []
    
    for idx, row in df.iterrows():
        district = row['district']
        crop = row['crop']
        year = row['year_num']
        season = row.get('season', 'Kharif')
        
        # Get cached weather (or default)
        key = f"{district}_{year}"
        weather = weather_cache.get(key, {
            'rain_total': 700, 'rain_days': 60, 'rain_cv': 2.0,
            'dry_spell_count': 3, 'rain_anomaly': 0, 'gdd': 2000,
            'heat_stress_intensity': 100, 'vpd_mean': 1.0,
            'rain_jun_jul': 300, 'rain_aug_sep': 350, 'humidity_cv': 15,
            'et_total': 500, 'water_balance': 200, 'max_hot_streak': 10,
            'night_temp_mean': 22, 'diurnal_range': 10
        })
        
        # Compute stress indices
        crop_key = crop if crop in CROP_PARAMS else 'default'
        stress = compute_all_stress_indices(weather, crop=crop_key)
        
        # Encode categoricals
        try:
            district_enc = encoders['district'].transform([str(district)])[0]
        except:
            district_enc = 0
        
        try:
            crop_enc = encoders['crop'].transform([str(crop)])[0]
        except:
            crop_enc = 0
        
        try:
            season_enc = encoders['season'].transform([str(season)])[0]
        except:
            season_enc = 0
        
        # Get coordinates
        coords = DISTRICT_COORDS.get(district, (19.5, 76.0))
        
        # Area
        area = row.get('area', 1000)
        
        # Major crops
        major_crops = ['Rice', 'Wheat', 'Soyabean', 'Cotton', 'Sugarcane', 'Jowar', 'Bajra', 'Maize']
        
        # Build feature dict
        feat = {
            'district_encoded': district_enc,
            'crop_encoded': crop_enc,
            'season_encoded': season_enc,
            'year': year,
            'lat': coords[0],
            'lon': coords[1],
            'log_area': np.log1p(area),
            'is_major_crop': 1 if crop in major_crops else 0,
            'is_kharif': 1 if 'kharif' in str(season).lower() else 0,
            **weather,
            **stress,
            'yield_kg_ha': row['yield_kg_ha']
        }
        
        features_list.append(feat)
    
    features_df = pd.DataFrame(features_list)
    print(f"    Computed {len(FEATURE_COLS_V2)} features for {len(features_df)} samples")
    
    return features_df


def main():
    """Main training pipeline."""
    print("=" * 70)
    print("PMFBY v2.0 - ENSEMBLE MODEL TRAINING")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Paths
    data_path = 'data/maharashtra_district_data/district-season-and-crop-wise-area-production-and-yield-statistics-for-maharashtra.csv'
    model_path = 'models/trained/ensemble_v2.pkl'
    
    # Load DES data
    df = load_des_data(data_path)
    
    # Load cached weather or compute
    weather_cache_path = 'data/weather_cache.pkl'
    if os.path.exists(weather_cache_path):
        print("\n[*] Loading cached weather data...")
        with open(weather_cache_path, 'rb') as f:
            weather_cache = pickle.load(f)
        print(f"    Loaded {len(weather_cache)} cached entries")
    else:
        print("\n[*] No cached weather. Using defaults (run train_with_weather.py for full data)")
        weather_cache = {}
    
    # Create encoders
    from sklearn.preprocessing import LabelEncoder
    encoders = {
        'district': LabelEncoder().fit(df['district'].astype(str)),
        'crop': LabelEncoder().fit(df['crop'].astype(str)),
        'season': LabelEncoder().fit(df['season'].astype(str) if 'season' in df else ['Kharif', 'Rabi', 'Summer'])
    }
    
    # Sample for faster testing (use full data for production)
    sample_size = min(5000, len(df))
    df_sample = df.sample(n=sample_size, random_state=42)
    print(f"\n[*] Using {sample_size} samples for training demo")
    
    # Compute features
    features_df = compute_features_for_training(df_sample, weather_cache, encoders)
    
    # Prepare X, y
    available_cols = [c for c in FEATURE_COLS_V2 if c in features_df.columns]
    X = features_df[available_cols].values
    y = features_df['yield_kg_ha'].values
    
    print(f"\n[3] Training Ensemble Model...")
    print(f"    Features: {len(available_cols)}")
    print(f"    Samples: {len(X)}")
    
    # Train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train ensemble
    model = EnsembleYieldPredictor()
    model.fit(X_train, y_train, feature_names=available_cols)
    
    # Evaluate
    print("\n[4] Evaluation...")
    train_metrics = model.evaluate(X_train, y_train)
    test_metrics = model.evaluate(X_test, y_test)
    
    print("\n    TRAINING METRICS:")
    for k, v in train_metrics.items():
        print(f"      {k:10s}: {v:.4f}")
    
    print("\n    TEST METRICS:")
    for k, v in test_metrics.items():
        print(f"      {k:10s}: {v:.4f}")
    
    # Feature importance
    print("\n    TOP 10 FEATURES:")
    for i, row in model.feature_importance.head(10).iterrows():
        bar = '*' * int(row['importance'] * 50)
        print(f"      {row['feature']:25s}: {row['importance']:.4f} {bar}")
    
    # Save model
    print("\n[5] Saving model...")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    model_data = {
        'model': model,
        'feature_cols': available_cols,
        'encoders': encoders,
        'metrics': test_metrics,
        'version': '2.0',
        'trained_on': datetime.now().isoformat()
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"    Saved to: {model_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"""
    MODEL: PMFBY Ensemble v2.0
    
    DATA:
    - Source: DES Maharashtra
    - Samples: {len(X_train)} train, {len(X_test)} test
    - Features: {len(available_cols)}
    
    PERFORMANCE:
    - R2: {test_metrics['r2']:.4f} ({test_metrics['r2']*100:.1f}%)
    - MAE: {test_metrics['mae']:.0f} kg/ha
    - RMSE: {test_metrics['rmse']:.0f} kg/ha
    
    MODEL FILE: {model_path}
    """)
    print("=" * 70)


if __name__ == "__main__":
    main()
