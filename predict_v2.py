"""
PMFBY v2.0 - Yield Prediction with Uncertainty
================================================
Interactive prediction using ensemble model with confidence intervals.

Features:
- Real-time weather from NASA POWER
- 15+ weather features + 10 agronomic stress indices
- Ensemble RF + XGBoost predictions
- Uncertainty estimation and confidence intervals
- PMFBY claim decision with confidence score

Usage: python predict_v2.py
"""

import os
import sys
import pickle
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feature_engineering.weather_features import fetch_and_compute_weather
from feature_engineering.agronomic_stress import compute_all_stress_indices, CROP_PARAMS

# Configuration
MODEL_PATH = 'models/trained/ensemble_v2.pkl'

DISTRICT_COORDS = {
    'Ahmednagar': (19.09, 74.74), 'Pune': (18.52, 73.86), 'Nashik': (20.00, 73.78),
    'Solapur': (17.66, 75.91), 'Kolhapur': (16.69, 74.23), 'Satara': (17.69, 73.99),
    'Sangli': (16.85, 74.57), 'Aurangabad': (19.88, 75.32), 'Jalgaon': (21.00, 75.57),
    'Nagpur': (21.15, 79.09), 'Amravati': (20.93, 77.75), 'Akola': (20.71, 77.00),
}

MAJOR_CROPS = ['Rice', 'Wheat', 'Soyabean', 'Cotton', 'Sugarcane', 'Jowar', 'Bajra', 'Maize']
SEASONS = ['Kharif', 'Rabi', 'Summer']


def load_model():
    """Load trained ensemble model."""
    print("\n[1] Loading Ensemble Model v2.0...")
    
    if not os.path.exists(MODEL_PATH):
        print(f"    ERROR: Model not found at {MODEL_PATH}")
        print("    Run train_v2.py first.")
        sys.exit(1)
    
    with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
    
    print(f"    Model: Ensemble (RF + XGBoost)")
    print(f"    Features: {len(model_data['feature_cols'])}")
    print(f"    R2: {model_data['metrics']['r2']:.4f}")
    
    return model_data


def calculate_pmfby_with_confidence(pred, uncertainty, threshold):
    """Calculate PMFBY loss with confidence-aware decision."""
    from scipy.stats import norm
    
    shortfall = max(0, threshold - pred)
    loss_pct = (shortfall / threshold) * 100 if shortfall > 0 else 0
    
    # Probability of loss >= 33%
    if uncertainty > 0:
        # What yield corresponds to 33% loss?
        yield_33pct = threshold * 0.67  # 67% of threshold
        
        # Probability that actual yield < yield_33pct
        prob_claim = norm.cdf((yield_33pct - pred) / uncertainty)
    else:
        prob_claim = 1.0 if loss_pct >= 33 else 0.0
    
    return {
        'threshold': threshold,
        'predicted_yield': pred,
        'uncertainty': uncertainty,
        'shortfall': shortfall,
        'loss_percentage': loss_pct,
        'claim_triggered': loss_pct >= 33,
        'claim_probability': prob_claim,
        'decision_confidence': abs(prob_claim - 0.5) * 2  # 0=uncertain, 1=confident
    }


def run_prediction(lat, lon, district, crop, season, area, threshold, year=None):
    """Run complete v2 prediction pipeline."""
    
    if year is None:
        year = datetime.now().year - 1  # Use last year for complete weather
    
    # Load model
    model_data = load_model()
    model = model_data['model']
    feature_cols = model_data['feature_cols']
    encoders = model_data['encoders']
    
    # Fetch weather
    print(f"\n[2] Fetching Weather for ({lat:.4f}, {lon:.4f}) Year {year}...")
    weather = fetch_and_compute_weather(lat, lon, year)
    
    print("    Weather Features:")
    print(f"      - Rainfall: {weather['rain_total']:.0f} mm")
    print(f"      - GDD: {weather['gdd']:.0f} degree-days")
    print(f"      - Heat Stress: {weather['heat_stress_intensity']:.0f}")
    print(f"      - VPD: {weather['vpd_mean']:.2f} kPa")
    print(f"      - Water Balance: {weather['water_balance']:.0f} mm")
    
    # Compute stress indices
    print("\n[3] Computing Agronomic Stress Indices...")
    crop_key = crop if crop in CROP_PARAMS else 'default'
    stress = compute_all_stress_indices(weather, crop=crop_key)
    
    print("    Stress Indices:")
    print(f"      - Vegetative Stress: {stress['vegetative_stress']:.2f}")
    print(f"      - Flowering Heat Stress: {stress['flowering_heat_stress']:.2f}")
    print(f"      - Combined Stress: {stress['combined_stress']:.2f}")
    print(f"      - Yield Potential: {stress['yield_potential']:.2f}")
    
    # Encode categoricals
    print("\n[4] Preparing Features...")
    try:
        district_enc = encoders['district'].transform([district])[0]
    except:
        district_enc = 0
    
    try:
        crop_enc = encoders['crop'].transform([crop])[0]
    except:
        crop_enc = 0
    
    try:
        season_enc = encoders['season'].transform([season])[0]
    except:
        season_enc = 0
    
    # Build feature dict
    features = {
        'district_encoded': district_enc,
        'crop_encoded': crop_enc,
        'season_encoded': season_enc,
        'year': year,
        'lat': lat,
        'lon': lon,
        'log_area': np.log1p(area),
        'is_major_crop': 1 if crop in MAJOR_CROPS else 0,
        'is_kharif': 1 if season.lower() == 'kharif' else 0,
        **weather,
        **stress
    }
    
    # Create feature vector in correct order
    X = np.array([[features.get(col, 0) for col in feature_cols]])
    print(f"    Features: {X.shape[1]}")
    
    # Predict with uncertainty
    print("\n[5] Making Prediction with Uncertainty...")
    result = model.predict_with_uncertainty(X)
    
    pred = result['prediction'][0]
    uncertainty = result['uncertainty'][0]
    conf_low = result['confidence_low'][0]
    conf_high = result['confidence_high'][0]
    
    print(f"    Predicted Yield: {pred:.0f} kg/ha")
    print(f"    Uncertainty: +/- {uncertainty:.0f} kg/ha")
    print(f"    95% CI: [{conf_low:.0f}, {conf_high:.0f}] kg/ha")
    
    # PMFBY Calculation
    print("\n[6] PMFBY Assessment...")
    pmfby = calculate_pmfby_with_confidence(pred, uncertainty, threshold)
    
    # Final Output
    print("\n" + "=" * 70)
    print(" PMFBY v2.0 PREDICTION RESULTS")
    print("=" * 70)
    print(f"""
    LOCATION:
        Coordinates: {lat:.4f}N, {lon:.4f}E
        District: {district}
        
    CROP:
        Crop: {crop}
        Season: {season}
        Area: {area} ha
        
    WEATHER SUMMARY:
        Total Rainfall: {weather['rain_total']:.0f} mm
        GDD: {weather['gdd']:.0f} degree-days
        Heat Stress Days: {weather['heat_days']} days
        Combined Stress: {stress['combined_stress']:.2f}
        
    YIELD PREDICTION:
        Predicted Yield: {pred:.0f} kg/ha
        Uncertainty: +/- {uncertainty:.0f} kg/ha
        95% Confidence: [{conf_low:.0f}, {conf_high:.0f}] kg/ha
        
    PMFBY ASSESSMENT:
        Threshold Yield: {threshold:.0f} kg/ha
        Shortfall: {pmfby['shortfall']:.0f} kg/ha
        Loss Percentage: {pmfby['loss_percentage']:.1f}%
        
    CLAIM DECISION:
        Triggered (>=33%): {'YES' if pmfby['claim_triggered'] else 'NO'}
        Claim Probability: {pmfby['claim_probability']*100:.1f}%
        Decision Confidence: {pmfby['decision_confidence']*100:.1f}%
    """)
    print("=" * 70)
    
    return {
        'prediction': pred,
        'uncertainty': uncertainty,
        'confidence_interval': (conf_low, conf_high),
        'weather': weather,
        'stress': stress,
        'pmfby': pmfby
    }


def main():
    """Interactive prediction mode."""
    print("=" * 70)
    print("   PMFBY v2.0 - YIELD PREDICTION WITH UNCERTAINTY")
    print("=" * 70)
    print("   Model: Ensemble (Random Forest + XGBoost)")
    print("   Features: 32 (Weather + Stress Indices)")
    print("=" * 70)
    
    # Default values for demo
    lat = 19.071591
    lon = 74.774179
    district = 'Ahmednagar'
    crop = 'Rice'
    season = 'Kharif'
    area = 10
    threshold = 1640
    
    print(f"\nRunning prediction for:")
    print(f"  Location: {lat}, {lon} ({district})")
    print(f"  Crop: {crop} ({season})")
    print(f"  Threshold: {threshold} kg/ha")
    
    result = run_prediction(lat, lon, district, crop, season, area, threshold)
    
    return result


if __name__ == "__main__":
    main()
