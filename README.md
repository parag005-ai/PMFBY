# PMFBY Yield Prediction System

AI/ML-based crop yield prediction system for Pradhan Mantri Fasal Bima Yojana (PMFBY) insurance claims.

## Features

- **81.8% Accuracy** (R² score) on test data
- Real-time weather data from NASA POWER API
- 32+ advanced features including:
  - Weather features (GDD, VPD, rainfall distribution)
  - Agronomic stress indices
  - Crop-specific parameters
- Uncertainty quantification with confidence intervals
- PMFBY claim decision with probability scoring

## Model Performance

| Metric | Value |
|--------|-------|
| R² Score | 81.8% |
| MAE | 195 kg/ha |
| Training Data | 20,558 records (Maharashtra) |
| Features | 32 |

## Quick Start

### Installation

```bash
pip install numpy pandas scikit-learn requests scipy xgboost
```

### Run Prediction

```bash
python predict_unified.py
```

## File Structure

```
pmfby_engine/
├── feature_engineering/
│   ├── weather_features.py       # Weather feature extraction
│   ├── agronomic_stress.py       # Crop stress indices
│   └── weather_multi_source.py   # Multi-source weather fetcher
├── models/
│   ├── ensemble.py               # Ensemble ML model
│   └── trained/
│       └── yield_model_with_weather.pkl  # Trained model (81.8% R²)
├── predict_unified.py            # Main prediction script
├── train_v2.py                   # Training pipeline
└── pmfby_predict.py              # Original prediction script
```

## Usage Example

```python
from predict_unified import run_unified_prediction

result = run_unified_prediction(
    lat=19.071591,
    lon=74.774179,
    district='Ahmednagar',
    crop='Rice',
    season='Kharif',
    area=10,
    threshold=1640
)

print(f"Predicted Yield: {result['prediction']:.0f} kg/ha")
print(f"Uncertainty: ±{result['uncertainty']:.0f} kg/ha")
print(f"Claim Triggered: {result['pmfby']['claim_triggered']}")
```

## Data Sources

- **Yield Data**: DES (Directorate of Economics & Statistics), Maharashtra
- **Weather**: NASA POWER API (50km resolution)
- **Soil**: ICAR soil database

## Key Components

### 1. Weather Features (15+)
- Growing Degree Days (GDD)
- Vapor Pressure Deficit (VPD)
- Dry spell count
- Heat stress intensity
- Critical stage rainfall

### 2. Agronomic Stress Indices (10)
- Vegetative stress
- Flowering heat stress
- Grain fill moisture deficit
- Combined stress score
- Yield potential index

### 3. ML Model
- Random Forest Regressor (300 trees)
- XGBoost (optional ensemble)
- Uncertainty estimation via tree variance

## Output

```
PMFBY UNIFIED PREDICTION
========================

MODEL PERFORMANCE:
    R2 Score: 81.8%
    MAE: 195 kg/ha

YIELD PREDICTION:
    Predicted Yield: 1637 kg/ha
    Uncertainty: +/- 662 kg/ha
    95% Confidence: [340, 2933] kg/ha

PMFBY ASSESSMENT:
    Threshold Yield: 1640 kg/ha
    Shortfall: 3 kg/ha
    Loss Percentage: 0.2%

CLAIM DECISION:
    Triggered (>=33%): NO
    Claim Probability: 20.8%
    Decision Confidence: 58.4%
```

## Future Enhancements

- [ ] Add Sentinel-2 NDVI integration
- [ ] Multi-state training data
- [ ] Higher resolution weather (ERA5-Land)
- [ ] Farm-level CCE ground truth
- [ ] Ensemble with LSTM for time-series

## License

MIT

## Authors

Developed for PMFBY insurance claim assessment.
