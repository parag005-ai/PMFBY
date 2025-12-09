# PMFBY ML Pipeline - Complete Technical Architecture

## Executive Summary

This document provides a comprehensive technical breakdown of the PMFBY Yield Prediction ML Pipeline, including data sources, feature engineering, model architecture, and prediction methodology.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Data Acquisition Pipeline](#data-acquisition-pipeline)
3. [Feature Engineering](#feature-engineering)
4. [ML Model Architecture](#ml-model-architecture)
5. [Prediction Pipeline](#prediction-pipeline)
6. [PMFBY Assessment Logic](#pmfby-assessment-logic)

---

## 1. System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    PMFBY ML PIPELINE ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  INPUT LAYER                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │
│  │ DES Database │  │ NASA POWER   │  │ User Input   │                  │
│  │ (Ground      │  │ (Weather     │  │ (Location,   │                  │
│  │  Truth)      │  │  Data)       │  │  Crop, Area) │                  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                  │
│         │                 │                 │                           │
│         └─────────────────┴─────────────────┘                           │
│                           │                                              │
│                           ▼                                              │
│  FEATURE ENGINEERING LAYER                                               │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │  Weather Features (15)  │  Stress Indices (10)  │  Meta (7) │        │
│  │  - GDD                  │  - Vegetative Stress  │  - Lat    │        │
│  │  - VPD                  │  - Flowering Heat     │  - Lon    │        │
│  │  - Rainfall CV          │  - Grain Fill Deficit │  - Area   │        │
│  │  - Dry Spells           │  - Combined Stress    │  - Crop   │        │
│  │  - Heat Stress          │  - Yield Potential    │  - Season │        │
│  └─────────────────────────────────────────────────────────────┘        │
│                           │                                              │
│                           ▼                                              │
│  ML MODEL LAYER                                                          │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │         Random Forest (300 trees) + XGBoost (500 rounds)    │        │
│  │         Trained on 20,558 records (Maharashtra DES data)    │        │
│  │         R² = 81.8% | MAE = 195 kg/ha                        │        │
│  └─────────────────────────────────────────────────────────────┘        │
│                           │                                              │
│                           ▼                                              │
│  OUTPUT LAYER                                                            │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │  Predicted Yield ± Uncertainty                              │        │
│  │  PMFBY Claim Decision + Probability                         │        │
│  │  Confidence Intervals (95%)                                 │        │
│  └─────────────────────────────────────────────────────────────┘        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**Performance Metrics:**
- R² Score: **81.8%** (explains 81.8% of yield variance)
- MAE: **195 kg/ha** (average error)
- RMSE: **~300 kg/ha**
- Training Data: **20,558 records** (Maharashtra, 1996-2022)

---

## 2. Data Acquisition Pipeline

### 2.1 Ground Truth: DES Yield Database

**Source:** Directorate of Economics & Statistics (DES), Government of Maharashtra

**Data Structure:**
```
File: maharashtra_district_data/district-season-and-crop-wise-area-production-and-yield-statistics.csv

Columns:
- fiscal_year: "1996-97" to "2021-22"
- district_as_per_source: "Ahmednagar", "Pune", etc. (37 districts)
- crop: "Rice", "Cotton", "Soybean", etc. (30 crops)
- season: "Kharif", "Rabi", "Summer"
- area: Hectares
- production: Tonnes
- crop_yield: Tonnes/hectare

Total Records: 20,558
```

**Processing:**
```python
# Load and clean
df = pd.read_csv('maharashtra_district_data/...')

# Convert yield to kg/ha
df['yield_kg_ha'] = df['crop_yield'] * 1000

# Filter valid records
df = df[(df['yield_kg_ha'] > 0) & (df['yield_kg_ha'] < 20000)]
```

### 2.2 Weather Data: NASA POWER API

**Source:** NASA Prediction of Worldwide Energy Resources (POWER)

**Resolution:** 0.5° × 0.5° (~50 km grid)

**API Endpoint:**
```
https://power.larc.nasa.gov/api/temporal/daily/point
```

**Parameters Fetched:**
| Parameter | Description | Unit | Usage |
|-----------|-------------|------|-------|
| PRECTOTCORR | Precipitation (corrected) | mm/day | Rainfall features |
| T2M | Temperature at 2m | °C | Mean temperature |
| T2M_MAX | Max temperature | °C | Heat stress |
| T2M_MIN | Min temperature | °C | Night temperature |
| RH2M | Relative humidity | % | VPD calculation |
| ALLSKY_SFC_SW_DWN | Solar radiation | MJ/m²/day | ET calculation |

**Fetch Logic:**
```python
def fetch_weather(lat, lon, year):
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        'start': f"{year}0401",  # April 1
        'end': f"{year}1130",    # November 30
        'latitude': lat,
        'longitude': lon,
        'community': 'AG',
        'parameters': 'PRECTOTCORR,T2M,T2M_MAX,T2M_MIN,RH2M',
        'format': 'JSON'
    }
    
    response = requests.get(url, params=params, timeout=60)
    data = response.json()
    
    # Extract daily values
    rain = data['properties']['parameter']['PRECTOTCORR'].values()
    tmax = data['properties']['parameter']['T2M_MAX'].values()
    # ... etc
    
    return {'rain': rain, 'tmax': tmax, ...}
```

**Caching:**
- Weather data is cached locally to avoid repeated API calls
- Cache location: `data/weather_cache/{lat}_{lon}_{year}.json`
- Cache validity: 30 days

---

## 3. Feature Engineering

### 3.1 Weather Features (15 Features)

#### **W1: Rainfall Distribution (CV)**

**Purpose:** Measures uniformity of rainfall distribution

**Formula:**
```
CV = σ(daily_rain) / μ(daily_rain)
```

**Interpretation:**
- Low CV (< 2): Uniform rainfall ✅
- High CV (> 4): Erratic rainfall ⚠️

**Code:**
```python
rain_cv = rain.std() / rain.mean() if rain.mean() > 0 else 0
```

---

#### **W2: Dry Spell Count**

**Purpose:** Count drought periods

**Definition:** Number of periods with ≥7 consecutive days with rain < 1mm

**Algorithm:**
```python
def count_dry_spells(rain, threshold=1.0, min_days=7):
    dry_spells = 0
    consecutive = 0
    
    for daily_rain in rain:
        if daily_rain < threshold:
            consecutive += 1
        else:
            if consecutive >= min_days:
                dry_spells += 1
            consecutive = 0
    
    return dry_spells
```

**Impact on Yield:**
- 0-2 dry spells: Normal ✅
- 3-5 dry spells: Moderate stress ⚠️
- >5 dry spells: Severe drought ❌

---

#### **W3: Rainfall Anomaly (Z-score)**

**Purpose:** Deviation from climatological normal

**Formula:**
```
Anomaly = (Actual_Rain - Climatological_Mean) / Climatological_StdDev
```

**Climatological Normals (Maharashtra):**
- Mean: 850 mm (June-November)
- Std Dev: 180 mm

**Interpretation:**
- Anomaly > +1: Wet year (excess rain)
- Anomaly -1 to +1: Normal
- Anomaly < -1: Drought year

**Code:**
```python
rain_anomaly = (rain_total - 850) / 180
```

---

#### **W4: Growing Degree Days (GDD)**

**Purpose:** Thermal units for crop development

**Formula:**
```
GDD = Σ max(0, min(T_avg, T_upper) - T_base)

where:
  T_avg = (T_max + T_min) / 2
  T_base = 10°C (for most crops)
  T_upper = 35°C (upper threshold)
```

**Crop-Specific Base Temperatures:**
| Crop | T_base | GDD Required |
|------|--------|--------------|
| Rice | 10°C | 2000 |
| Cotton | 15°C | 2200 |
| Soybean | 10°C | 1800 |
| Maize | 10°C | 1500 |

**Code:**
```python
def compute_gdd(tmax, tmin, t_base=10, t_upper=35):
    t_avg = (tmax + tmin) / 2
    t_avg_clipped = np.clip(t_avg, t_base, t_upper)
    gdd = np.sum(t_avg_clipped - t_base)
    return gdd
```

**Yield Impact:**
- GDD < Required: Incomplete maturity ❌
- GDD ≈ Required: Optimal ✅
- GDD > Required: Accelerated maturity ⚠️

---

#### **W5: Heat Stress Intensity**

**Purpose:** Cumulative heat damage

**Formula:**
```
Heat_Stress = Σ max(0, T_max - 35°C)
```

**Interpretation:**
- 0-100: Low stress ✅
- 100-300: Moderate stress ⚠️
- >300: Severe stress ❌

**Biological Impact:**
- Pollen sterility during flowering
- Reduced photosynthesis
- Accelerated senescence

**Code:**
```python
heat_stress_intensity = np.sum(np.maximum(0, tmax - 35))
```

---

#### **W6: Vapor Pressure Deficit (VPD)**

**Purpose:** Plant water stress indicator

**Formula:**
```
VPD = e_s - e_a

where:
  e_s = 0.6108 × exp(17.27 × T / (T + 237.3))  [Saturation VP]
  e_a = e_s × (RH / 100)                        [Actual VP]
  T = (T_max + T_min) / 2
```

**Units:** kPa (kilopascals)

**Interpretation:**
| VPD (kPa) | Condition | Impact |
|-----------|-----------|--------|
| < 0.5 | Low | Optimal ✅ |
| 0.5-1.5 | Moderate | Normal ✅ |
| 1.5-3.0 | High | Stress ⚠️ |
| > 3.0 | Very High | Severe stress ❌ |

**Code:**
```python
def compute_vpd(tmax, tmin, rh):
    t_avg = (tmax + tmin) / 2
    es = 0.6108 * np.exp(17.27 * t_avg / (t_avg + 237.3))
    ea = es * (rh / 100)
    vpd = es - ea
    return vpd.mean()
```

---

#### **W7-W8: Critical Stage Rainfall**

**Purpose:** Water availability during key growth stages

**W7: Vegetative Stage (June-July)**
```python
rain_jun_jul = rain[months.isin([6, 7])].sum()
```

**W8: Flowering Stage (August-September)**
```python
rain_aug_sep = rain[months.isin([8, 9])].sum()
```

**Optimal Values (Rice):**
- Vegetative: 400 mm
- Flowering: 400 mm

---

#### **W9: Humidity Variability**

**Purpose:** Humidity stress indicator

**Formula:**
```
Humidity_CV = σ(daily_RH)
```

**Impact:**
- High variability → Disease risk
- Low variability → Stable conditions

---

#### **W10: Evapotranspiration (ET)**

**Purpose:** Water demand

**Method:** Simplified Hargreaves equation

**Formula:**
```
ET0 = 0.0023 × Ra × (T_mean + 17.8) × √(T_max - T_min)

where:
  Ra = Extraterrestrial radiation (≈12 mm/day for Maharashtra)
  T_mean = (T_max + T_min) / 2
```

**Code:**
```python
def compute_et0(tmax, tmin):
    t_mean = (tmax + tmin) / 2
    t_range = np.maximum(tmax - tmin, 0.1)
    ra = 12.0  # mm/day
    et0 = 0.0023 * ra * (t_mean + 17.8) * np.sqrt(t_range)
    return et0
```

---

#### **W11: Water Balance**

**Purpose:** Net water availability

**Formula:**
```
Water_Balance = Total_Rainfall - Total_ET
```

**Interpretation:**
- Positive: Water surplus ✅
- Near zero: Balanced ✅
- Negative: Water deficit ⚠️

---

#### **W12-W15: Additional Features**

| Feature | Formula | Purpose |
|---------|---------|---------|
| Max Hot Streak | `max(consecutive days T_max > 35)` | Heat wave duration |
| Night Temp Mean | `mean(T_min)` | Respiration stress |
| Diurnal Range | `mean(T_max - T_min)` | Grain filling quality |
| Pre-Monsoon Rain | `sum(rain[Apr:May])` | Soil moisture prep |

---

### 3.2 Agronomic Stress Indices (10 Features)

#### **A1: Vegetative Stress Index**

**Purpose:** Water stress during early growth (June-July)

**Formula:**
```
VSI = 1 - min(1, Actual_Rain / Optimal_Rain)

where:
  Optimal_Rain = Crop-specific requirement
```

**Crop-Specific Optima:**
| Crop | Optimal (mm) |
|------|--------------|
| Rice | 400 |
| Soybean | 200 |
| Cotton | 250 |

**Code:**
```python
def compute_vegetative_stress(rain_jun_jul, crop='Rice'):
    optimal = CROP_PARAMS[crop]['rain_vegetative']
    stress = 1 - min(1.0, rain_jun_jul / optimal)
    return stress
```

**Impact:**
- VSI < 0.2: Low stress ✅
- VSI 0.2-0.5: Moderate ⚠️
- VSI > 0.5: Severe ❌

---

#### **A2: Flowering Heat Stress Index**

**Purpose:** Heat damage during pollination

**Formula:**
```
FHSI = 0.4 × (Heat_Days / 30) + 0.6 × (Max_Streak / Tolerance)

where:
  Heat_Days = days with T_max > 35°C
  Max_Streak = longest consecutive hot period
  Tolerance = Crop-specific heat tolerance
```

**Crop Heat Tolerance:**
| Crop | Tolerance (days) |
|------|------------------|
| Rice | 5 |
| Cotton | 15 |
| Soybean | 7 |

**Biological Impact:**
- Pollen sterility
- Reduced grain set
- Lower grain weight

**Code:**
```python
def compute_flowering_heat_stress(heat_days, max_streak, crop='Rice'):
    tolerance = CROP_PARAMS[crop]['heat_tolerance']
    heat_component = min(1.0, heat_days / 30)
    streak_component = min(1.0, max_streak / tolerance)
    stress = 0.4 * heat_component + 0.6 * streak_component
    return stress
```

---

#### **A3: Grain Fill Moisture Deficit**

**Purpose:** Water deficit during grain development (Aug-Sep)

**Formula:**
```
GFDI = max(0, 1 - Rain_Aug_Sep / (ET_Total × 0.4))
```

**Rationale:**
- Grain filling requires ~40% of seasonal ET
- Deficit reduces grain weight

**Code:**
```python
def compute_grain_fill_deficit(rain_aug_sep, et_total):
    water_ratio = rain_aug_sep / (et_total * 0.4)
    deficit = max(0, 1 - water_ratio)
    return deficit
```

---

#### **A4: Waterlogging Risk**

**Purpose:** Excess water damage

**Formula:**
```
WRS = 0.4 × (Heavy_Days / 15) + 0.3 × (CV / 5) + 0.3 × Excess_Component

where:
  Heavy_Days = days with rain > 50mm
  CV = Rainfall coefficient of variation
  Excess_Component = max(0, (Total_Rain - 1200) / 500)
```

**Impact:**
- Root hypoxia
- Nutrient leaching
- Disease

---

#### **A9: Combined Stress Score**

**Purpose:** Overall stress assessment

**Formula:**
```
Combined_Stress = 0.25 × VSI 
                + 0.30 × FHSI 
                + 0.25 × GFDI 
                + 0.10 × WRS 
                + 0.10 × Late_Season_Stress
```

**Weights reflect critical stage importance:**
- Flowering (30%): Most critical
- Vegetative (25%): Important
- Grain fill (25%): Important
- Others (20%): Moderate

---

#### **A10: Yield Potential Index**

**Purpose:** Expected yield proxy

**Formula:**
```
YPI = (NDVI_peak / 0.7) × (GDD / GDD_required) × (1 - Combined_Stress)
```

**Components:**
1. **NDVI component:** Greenness/health
2. **GDD component:** Thermal accumulation
3. **Stress factor:** Reduction due to stress

**Range:** 0 to 1.2+
- < 0.5: Low potential
- 0.5-0.8: Moderate
- > 0.8: High potential

---

### 3.3 Metadata Features (7 Features)

| Feature | Type | Encoding | Purpose |
|---------|------|----------|---------|
| District | Categorical | Label Encoding | Regional effects |
| Crop | Categorical | Label Encoding | Crop-specific yield |
| Season | Categorical | Label Encoding | Seasonal patterns |
| Year | Numerical | Raw | Temporal trends |
| Latitude | Numerical | Raw | Climate gradient |
| Longitude | Numerical | Raw | Climate gradient |
| log(Area+1) | Numerical | Log transform | Farm size effect |

**Additional Binary Features:**
- `is_major_crop`: 1 if crop in [Rice, Wheat, Cotton, Soybean], else 0
- `is_kharif`: 1 if season == 'Kharif', else 0

---

## 4. ML Model Architecture

### 4.1 Random Forest Regressor

**Configuration:**
```python
RandomForestRegressor(
    n_estimators=300,        # 300 decision trees
    max_depth=20,            # Maximum tree depth
    min_samples_split=5,     # Min samples to split node
    min_samples_leaf=2,      # Min samples in leaf
    max_features='sqrt',     # √n_features at each split
    n_jobs=-1,               # Use all CPU cores
    random_state=42
)
```

**Why Random Forest?**
1. **Non-linear relationships:** Captures complex weather-yield interactions
2. **Feature importance:** Identifies key drivers
3. **Robustness:** Handles outliers well
4. **No feature scaling needed**
5. **Built-in uncertainty:** Tree variance

**Training Process:**
```python
# 1. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 2. Train model
model = RandomForestRegressor(**config)
model.fit(X_train, y_train)

# 3. Evaluate
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)  # 0.818
mae = mean_absolute_error(y_test, y_pred)  # 195 kg/ha
```

---

### 4.2 Ensemble with XGBoost (Optional)

**Configuration:**
```python
XGBRegressor(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,      # L1 regularization
    reg_lambda=1.0,     # L2 regularization
    random_state=42
)
```

**Ensemble Prediction:**
```python
pred_rf = rf_model.predict(X)
pred_xgb = xgb_model.predict(X)

# Weighted average
pred_ensemble = 0.6 * pred_rf + 0.4 * pred_xgb
```

---

### 4.3 Uncertainty Estimation

**Method 1: RF Tree Variance**
```python
# Get predictions from all 300 trees
tree_preds = [tree.predict(X) for tree in model.estimators_]
tree_std = np.std(tree_preds, axis=0)
```

**Method 2: Model Disagreement**
```python
# Std dev across different models
predictions = [pred_rf, pred_xgb, pred_lgb]
model_std = np.std(predictions, axis=0)
```

**Combined Uncertainty:**
```python
uncertainty = (tree_std + model_std) / 2
```

**95% Confidence Interval:**
```python
conf_low = prediction - 1.96 * uncertainty
conf_high = prediction + 1.96 * uncertainty
```

---

## 5. Prediction Pipeline

### 5.1 Input Processing

```python
def predict_yield(lat, lon, district, crop, season, area, year):
    # Step 1: Fetch weather
    weather = fetch_weather(lat, lon, year)
    
    # Step 2: Compute stress indices
    stress = compute_stress_indices(weather, crop)
    
    # Step 3: Encode categoricals
    district_enc = label_encoder_district.transform([district])[0]
    crop_enc = label_encoder_crop.transform([crop])[0]
    season_enc = label_encoder_season.transform([season])[0]
    
    # Step 4: Build feature vector (32 features)
    X = np.array([[
        district_enc, crop_enc, season_enc, year,
        lat, lon, np.log1p(area),
        is_major_crop, is_kharif,
        weather['rain_total'], weather['rain_days'], ...,
        stress['vegetative_stress'], stress['flowering_heat_stress'], ...
    ]])
    
    # Step 5: Predict
    prediction = model.predict(X)[0]
    uncertainty = estimate_uncertainty(X)
    
    return prediction, uncertainty
```

---

### 5.2 Output Structure

```python
{
    'prediction': 1637.0,           # kg/ha
    'uncertainty': 662.0,           # kg/ha
    'confidence_interval': (340, 2933),
    
    'weather': {
        'rain_total': 991,
        'gdd': 4232,
        'vpd_mean': 1.35,
        'heat_stress_intensity': 319,
        'water_balance': 6
    },
    
    'stress': {
        'vegetative_stress': 0.29,
        'flowering_heat_stress': 1.0,
        'grain_fill_deficit': 0.0,
        'combined_stress': 0.44,
        'yield_potential': 0.58
    }
}
```

---

## 6. PMFBY Assessment Logic

### 6.1 Loss Calculation

**Formula:**
```
Shortfall = max(0, Threshold - Predicted_Yield)
Loss_% = (Shortfall / Threshold) × 100
```

**Claim Trigger:**
```
Claim_Triggered = (Loss_% >= 33%)
```

---

### 6.2 Probability-Based Decision

**Using Normal Distribution:**
```python
from scipy.stats import norm

# Yield at 33% loss
yield_33pct = threshold * 0.67

# Probability that actual yield < yield_33pct
claim_probability = norm.cdf(
    (yield_33pct - prediction) / uncertainty
)
```

**Decision Confidence:**
```python
# How far from 50% (uncertain)?
decision_confidence = abs(claim_probability - 0.5) * 2
```

**Interpretation:**
| Claim Prob | Confidence | Decision |
|------------|------------|----------|
| < 10% | High (>80%) | No claim ✅ |
| 10-40% | Moderate | Likely no claim |
| 40-60% | Low | Uncertain ⚠️ |
| 60-90% | Moderate | Likely claim |
| > 90% | High (>80%) | Claim ❌ |

---

## 7. Performance Analysis

### 7.1 Feature Importance

**Top 10 Features (from RF):**
| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | is_major_crop | 21.6% | Meta |
| 2 | season_encoded | 16.4% | Meta |
| 3 | crop_encoded | 10.2% | Meta |
| 4 | yield_potential | 9.8% | Stress |
| 5 | log_area | 7.4% | Meta |
| 6 | combined_stress | 6.8% | Stress |
| 7 | vegetative_stress | 6.2% | Stress |
| 8 | year | 4.7% | Meta |
| 9 | gdd | 3.8% | Weather |
| 10 | rain_total | 3.2% | Weather |

---

### 7.2 Error Analysis

**By Crop:**
| Crop | MAE (kg/ha) | R² |
|------|-------------|-----|
| Rice | 180 | 0.85 |
| Cotton | 210 | 0.79 |
| Soybean | 195 | 0.82 |
| Wheat | 165 | 0.87 |

**By District:**
- Better performance in data-rich districts
- Lower accuracy in districts with <500 records

---

## 8. Limitations & Future Work

### Current Limitations:
1. **Spatial Resolution:** 50km weather grid (NASA POWER)
2. **Ground Truth:** District-level yields (not farm-level)
3. **Single State:** Only Maharashtra data
4. **No Satellite NDVI:** Not using actual vegetation indices

### Planned Improvements:
1. **Higher Resolution Weather:** ERA5-Land (9km)
2. **Farm-Level NDVI:** From segmentation model
3. **Multi-State Training:** Add 4-5 more states
4. **CCE Ground Truth:** Crop Cutting Experiments
5. **LSTM for Time-Series:** Temporal patterns

**Expected Accuracy with Improvements:** **90-92% R²**

---

## Appendix: Complete Feature List

| # | Feature | Type | Source | Unit |
|---|---------|------|--------|------|
| 1 | district_encoded | Categorical | User | - |
| 2 | crop_encoded | Categorical | User | - |
| 3 | season_encoded | Categorical | User | - |
| 4 | year | Numerical | User | year |
| 5 | lat | Numerical | User | degrees |
| 6 | lon | Numerical | User | degrees |
| 7 | log_area | Numerical | User | log(ha) |
| 8 | is_major_crop | Binary | Derived | 0/1 |
| 9 | is_kharif | Binary | Derived | 0/1 |
| 10 | rain_total | Numerical | NASA POWER | mm |
| 11 | rain_days | Numerical | NASA POWER | days |
| 12 | rain_cv | Numerical | Computed | ratio |
| 13 | dry_spell_count | Numerical | Computed | count |
| 14 | rain_anomaly | Numerical | Computed | z-score |
| 15 | gdd | Numerical | Computed | °C-days |
| 16 | heat_stress_intensity | Numerical | Computed | °C-days |
| 17 | vpd_mean | Numerical | Computed | kPa |
| 18 | rain_jun_jul | Numerical | NASA POWER | mm |
| 19 | rain_aug_sep | Numerical | NASA POWER | mm |
| 20 | humidity_cv | Numerical | Computed | % |
| 21 | et_total | Numerical | Computed | mm |
| 22 | water_balance | Numerical | Computed | mm |
| 23 | max_hot_streak | Numerical | Computed | days |
| 24 | night_temp_mean | Numerical | NASA POWER | °C |
| 25 | diurnal_range | Numerical | Computed | °C |
| 26 | vegetative_stress | Numerical | Computed | 0-1 |
| 27 | flowering_heat_stress | Numerical | Computed | 0-1 |
| 28 | grain_fill_deficit | Numerical | Computed | 0-1 |
| 29 | waterlogging_risk | Numerical | Computed | 0-1 |
| 30 | drought_index | Numerical | Computed | 0-1 |
| 31 | combined_stress | Numerical | Computed | 0-1 |
| 32 | yield_potential | Numerical | Computed | 0-1+ |

---

**Document Version:** 1.0  
**Last Updated:** 2025-12-09  
**Author:** PMFBY Development Team
