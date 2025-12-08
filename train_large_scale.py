"""
PMFBY ML Model Training - Large Scale
Using 21,729 real records from Maharashtra DES data
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("PMFBY ML MODEL TRAINING - LARGE SCALE (21,729 RECORDS)")
print("=" * 70)

# =====================================================
# STEP 1: Load Maharashtra Full Dataset
# =====================================================
print("\n[1] LOADING MAHARASHTRA FULL DATASET...")

df = pd.read_csv('data/maharashtra_district_data/district-season-and-crop-wise-area-production-and-yield-statistics-for-maharashtra.csv')

print(f"    Loaded: {len(df):,} rows")
print(f"    Columns: {list(df.columns)}")

# =====================================================
# STEP 2: Clean and Prepare Data
# =====================================================
print("\n[2] CLEANING DATA...")

# Standardize column names
df.columns = df.columns.str.lower().str.strip()

# Rename for clarity
df = df.rename(columns={
    'fiscal_year': 'year',
    'district_as_per_source': 'district',
    'crop_yield': 'yield_tonnes_ha'
})

# Convert yield to kg/ha
df['yield_kg_ha'] = pd.to_numeric(df['yield_tonnes_ha'], errors='coerce') * 1000

# Filter valid records
df_clean = df[
    (df['yield_kg_ha'].notna()) & 
    (df['yield_kg_ha'] > 0) & 
    (df['yield_kg_ha'] < 20000)  # Remove outliers
].copy()

print(f"    Clean records: {len(df_clean):,}")
print(f"    Districts: {df_clean['district'].nunique()}")
print(f"    Crops: {df_clean['crop'].nunique()}")
print(f"    Years: {df_clean['year'].nunique()}")
print(f"    Yield range: {df_clean['yield_kg_ha'].min():.0f} - {df_clean['yield_kg_ha'].max():.0f} kg/ha")

# Show top crops
print("\n    Top crops by frequency:")
print(df_clean['crop'].value_counts().head(10).to_string())

# =====================================================
# STEP 3: Feature Engineering
# =====================================================
print("\n[3] FEATURE ENGINEERING...")

# Encode categorical variables
le_district = LabelEncoder()
le_crop = LabelEncoder()
le_season = LabelEncoder()

df_clean['district_encoded'] = le_district.fit_transform(df_clean['district'].astype(str))
df_clean['crop_encoded'] = le_crop.fit_transform(df_clean['crop'].astype(str))
df_clean['season_encoded'] = le_season.fit_transform(df_clean['season'].astype(str))

# Extract year as numeric
df_clean['year_num'] = df_clean['year'].str[:4].astype(int)

# Create region features (approximate lat/lon based on district)
# This is a simplified approach - in production use actual coordinates
district_coords = {
    'Ahmednagar': (19.09, 74.74), 'Pune': (18.52, 73.86), 'Nashik': (20.00, 73.78),
    'Solapur': (17.66, 75.91), 'Kolhapur': (16.69, 74.23), 'Satara': (17.69, 73.99),
    'Sangli': (16.85, 74.57), 'Aurangabad': (19.88, 75.32), 'Jalgaon': (21.00, 75.57),
    'Nagpur': (21.15, 79.09), 'Amravati': (20.93, 77.75), 'Akola': (20.71, 77.00),
    'Buldhana': (20.53, 76.18), 'Latur': (18.40, 76.57), 'Beed': (18.99, 75.76),
    'Nanded': (19.15, 77.30), 'Parbhani': (19.27, 76.77), 'Osmanabad': (18.18, 76.04)
}

# Default coordinates for unknown districts
default_lat, default_lon = 19.5, 76.0

df_clean['lat'] = df_clean['district'].apply(
    lambda x: district_coords.get(x, (default_lat, default_lon))[0]
)
df_clean['lon'] = df_clean['district'].apply(
    lambda x: district_coords.get(x, (default_lat, default_lon))[1]
)

# Area as a feature (larger farms tend to have different yields)
df_clean['area_ha'] = pd.to_numeric(df_clean['area'], errors='coerce').fillna(1000)
df_clean['log_area'] = np.log1p(df_clean['area_ha'])

# Identify major crop categories
major_crops = ['Rice', 'Wheat', 'Soyabean', 'Cotton', 'Sugarcane', 'Jowar', 'Bajra', 'Maize', 'Groundnut', 'Tur']
df_clean['is_major_crop'] = df_clean['crop'].isin(major_crops).astype(int)

# Season binary features
df_clean['is_kharif'] = (df_clean['season'].str.lower().str.contains('kharif', na=False)).astype(int)
df_clean['is_rabi'] = (df_clean['season'].str.lower().str.contains('rabi', na=False)).astype(int)

print(f"    Features created: {len([c for c in df_clean.columns if c not in ['state', 'crop', 'district', 'note', 'unit']])}")

# =====================================================
# STEP 4: Prepare Training Data
# =====================================================
print("\n[4] PREPARING TRAINING DATA...")

feature_cols = [
    'district_encoded', 'crop_encoded', 'season_encoded', 'year_num',
    'lat', 'lon', 'log_area', 'is_major_crop', 'is_kharif', 'is_rabi'
]

X = df_clean[feature_cols].values
y = df_clean['yield_kg_ha'].values

print(f"    Feature matrix: {X.shape}")
print(f"    Target vector: {y.shape}")

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"    Training samples: {len(X_train):,}")
print(f"    Test samples: {len(X_test):,}")

# =====================================================
# STEP 5: Train Models
# =====================================================
print("\n[5] TRAINING MODELS...")

# Random Forest
print("    Training Random Forest...")
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# Gradient Boosting
print("    Training Gradient Boosting...")
gb_model = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    random_state=42
)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)

# =====================================================
# STEP 6: Evaluate Models
# =====================================================
print("\n[6] MODEL EVALUATION...")
print("-" * 70)

# Random Forest metrics
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)
rf_mape = np.mean(np.abs((y_test - rf_pred) / y_test)) * 100

# Gradient Boosting metrics
gb_mae = mean_absolute_error(y_test, gb_pred)
gb_rmse = np.sqrt(mean_squared_error(y_test, gb_pred))
gb_r2 = r2_score(y_test, gb_pred)
gb_mape = np.mean(np.abs((y_test - gb_pred) / y_test)) * 100

print("\n    RANDOM FOREST:")
print(f"      MAE: {rf_mae:.0f} kg/ha")
print(f"      RMSE: {rf_rmse:.0f} kg/ha")
print(f"      R²: {rf_r2:.4f}")
print(f"      MAPE: {rf_mape:.1f}%")
print(f"      ACCURACY: {100 - rf_mape:.1f}%")

print("\n    GRADIENT BOOSTING:")
print(f"      MAE: {gb_mae:.0f} kg/ha")
print(f"      RMSE: {gb_rmse:.0f} kg/ha")
print(f"      R²: {gb_r2:.4f}")
print(f"      MAPE: {gb_mape:.1f}%")
print(f"      ACCURACY: {100 - gb_mape:.1f}%")

# Cross-validation
print("\n    CROSS-VALIDATION (5-fold)...")
cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='r2')
print(f"      CV R² scores: {cv_scores.round(3)}")
print(f"      Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# =====================================================
# STEP 7: Feature Importance
# =====================================================
print("\n[7] FEATURE IMPORTANCE...")
print("-" * 70)

importance = list(zip(feature_cols, rf_model.feature_importances_))
importance.sort(key=lambda x: x[1], reverse=True)

for feat, imp in importance:
    bar = '*' * int(imp * 50)
    print(f"    {feat:<20}: {imp:.4f} {bar}")

# =====================================================
# STEP 8: Save Best Model
# =====================================================
print("\n[8] SAVING MODEL...")

best_model = rf_model if rf_r2 > gb_r2 else gb_model
best_name = "Random Forest" if rf_r2 > gb_r2 else "Gradient Boosting"
best_r2 = max(rf_r2, gb_r2)
best_mape = rf_mape if rf_r2 > gb_r2 else gb_mape

model_dir = 'models/trained'
os.makedirs(model_dir, exist_ok=True)

model_data = {
    'model': best_model,
    'feature_cols': feature_cols,
    'model_name': best_name,
    'encoders': {
        'district': le_district,
        'crop': le_crop,
        'season': le_season
    },
    'metrics': {
        'r2': best_r2,
        'mape': best_mape,
        'mae': rf_mae if rf_r2 > gb_r2 else gb_mae,
        'rmse': rf_rmse if rf_r2 > gb_r2 else gb_rmse
    },
    'training_samples': len(X_train),
    'test_samples': len(X_test),
    'trained_on': datetime.now().isoformat(),
    'data_source': 'DES Maharashtra (21,729 records)'
}

model_path = os.path.join(model_dir, 'yield_model_v2.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(model_data, f)

print(f"    Model saved to: {model_path}")

# =====================================================
# SUMMARY
# =====================================================
print("\n" + "=" * 70)
print("TRAINING COMPLETE - SUMMARY")
print("=" * 70)
print(f"""
    DATA:
    - Source: DES Maharashtra Official Data
    - Total Records: {len(df_clean):,}
    - Districts: {df_clean['district'].nunique()}
    - Crops: {df_clean['crop'].nunique()}
    - Years: {df_clean['year'].nunique()}
    
    MODEL:
    - Algorithm: {best_name}
    - Features: {len(feature_cols)}
    - Training samples: {len(X_train):,}
    - Test samples: {len(X_test):,}
    
    PERFORMANCE:
    - R² Score: {best_r2:.4f} ({best_r2*100:.1f}%)
    - MAPE: {best_mape:.1f}%
    - Accuracy: {100 - best_mape:.1f}%
    - MAE: {rf_mae if rf_r2 > gb_r2 else gb_mae:.0f} kg/ha
    
    STATUS: ✓ Model trained successfully!
""")
print("=" * 70)
