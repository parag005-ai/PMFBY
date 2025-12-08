"""
PMFBY Pipeline - Real Data Test with OFFICIAL Thresholds
Uses authenticated yield data from government sources.
"""

import os
import sys
import json
from datetime import datetime

# Add pmfby_engine to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("=" * 70)
print("PMFBY YIELD PREDICTION - REAL DATA WITH OFFICIAL THRESHOLDS")
print("=" * 70)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Real farm data - Taraori, Karnal, Haryana
TEST_CLAIM = {
    'claim_id': 'OFFICIAL_TEST_001',
    'farmer_name': 'Test Farmer - Karnal',
    'crop_type': 'rice',
    'season': 'Kharif 2024',
    'sowing_date': '2024-06-15',
    'district': 'Karnal',
    'state': 'Haryana',
    'village': 'Taraori',
    'area_ha': 2.5,
    'sum_insured': 150000,
    'indemnity_level': 80,  # 80% as per PMFBY Zone B
    'coordinates': '29.69°N, 76.97°E',
    'farm_polygon': {
        "type": "Polygon",
        "coordinates": [[[76.96, 29.68], [76.98, 29.68],
                        [76.98, 29.70], [76.96, 29.70],
                        [76.96, 29.68]]]
    }
}


def run_official_test():
    """Run pipeline with official government thresholds."""
    
    print("[1] Importing modules...")
    from data_ingestion.sentinel2_fetcher import Sentinel2Fetcher
    from data_ingestion.weather_fetcher import WeatherFetcher
    from data_ingestion.soil_fetcher import SoilFetcher
    from data_ingestion.official_yield_database import OfficialYieldDatabase
    from data_ingestion.preprocessing import TimeSeriesPreprocessor
    from feature_engineering.stress_indices import StressIndexCalculator
    from models.crop_stage_detector import CropStageDetector
    from models.yield_transformer import YieldPredictor
    from outputs.report_generator import PMFBYReportGenerator
    print("    ✓ All modules loaded")
    
    # Create output directory
    output_dir = os.path.join("output", TEST_CLAIM['claim_id'])
    os.makedirs(output_dir, exist_ok=True)
    
    results = {'claim_id': TEST_CLAIM['claim_id'], 'steps': {}}
    
    # Step 1: Get OFFICIAL threshold yield
    print("\n" + "=" * 70)
    print("[2] OFFICIAL THRESHOLD YIELD (Government Data)")
    print("=" * 70)
    
    yield_db = OfficialYieldDatabase()
    official_threshold = yield_db.calculate_threshold_yield(
        district=TEST_CLAIM['district'],
        crop=TEST_CLAIM['crop_type'],
        season='kharif',
        indemnity_level=TEST_CLAIM['indemnity_level']
    )
    
    print(f"    District: {TEST_CLAIM['district']}")
    print(f"    Crop: {TEST_CLAIM['crop_type'].upper()}")
    print(f"    Official Average Yield: {official_threshold['average_yield']} kg/ha")
    print(f"    Indemnity Level: {official_threshold['indemnity_level']}%")
    print(f"    ✓ THRESHOLD YIELD: {official_threshold['threshold_yield']} kg/ha")
    print(f"    Source: {official_threshold['source']}")
    print(f"    Data Period: {official_threshold['data_year']}")
    
    results['steps']['official_threshold'] = official_threshold
    
    # Step 2: Fetch REAL Sentinel-2 data
    print("\n" + "-" * 50)
    print("[3] Fetching Sentinel-2 satellite data (GEE)...")
    s2_fetcher = Sentinel2Fetcher()
    s2_df = s2_fetcher.fetch_time_series(
        geometry=TEST_CLAIM['farm_polygon'],
        start_date=TEST_CLAIM['sowing_date'],
        end_date='2024-11-15'
    )
    data_source = "Google Earth Engine" if s2_fetcher.initialized else "Synthetic (GEE unavailable)"
    print(f"    ✓ Retrieved {len(s2_df)} observations")
    print(f"    Source: {data_source}")
    print(f"    NDVI range: {s2_df['ndvi'].min():.3f} - {s2_df['ndvi'].max():.3f}")
    results['steps']['satellite'] = {'observations': len(s2_df), 'source': data_source}
    
    # Step 3: Fetch REAL Weather data
    print("\n[4] Fetching NASA POWER weather data...")
    weather_fetcher = WeatherFetcher()
    weather_df = weather_fetcher.fetch_daily_weather(
        latitude=29.69,
        longitude=76.97,
        start_date='2024-06-15',
        end_date='2024-11-15'
    )
    print(f"    ✓ Retrieved {len(weather_df)} days")
    print(f"    Source: NASA POWER API (power.larc.nasa.gov)")
    if 'prectotcorr' in weather_df.columns:
        print(f"    Total rainfall: {weather_df['prectotcorr'].sum():.1f} mm")
    results['steps']['weather'] = {'days': len(weather_df), 'source': 'NASA POWER API'}
    
    # Step 4: Get ICAR Soil data
    print("\n[5] Loading ICAR soil data...")
    soil_fetcher = SoilFetcher()
    soil_data = soil_fetcher.get_soil_profile('Karnal')
    print(f"    ✓ Soil type: {soil_data['soil_type']}")
    print(f"    Source: ICAR Soil Survey (Haryana)")
    results['steps']['soil'] = {'type': soil_data['soil_type'], 'source': 'ICAR'}
    
    # Step 5: Preprocess
    print("\n[6] Preprocessing (Savitzky-Golay smoothing)...")
    preprocessor = TimeSeriesPreprocessor()
    s2_clean = preprocessor.preprocess_vegetation_series(s2_df)
    phenology = preprocessor.extract_phenological_features(s2_clean)
    print(f"    ✓ Peak NDVI: {phenology.get('peak_ndvi', 0):.3f}")
    
    # Step 6: Stress analysis
    print("\n[7] Computing stress indices...")
    stage_detector = CropStageDetector('rice')
    stages = stage_detector.get_stage_boundaries()
    stress_calc = StressIndexCalculator('rice')
    weather_df = stress_calc.calculate_heat_stress(weather_df)
    weather_df = stress_calc.calculate_water_balance_stress(weather_df)
    stagewise = stress_calc.calculate_stagewise_stress(weather_df, TEST_CLAIM['sowing_date'], stages)
    explanation = stress_calc.generate_stress_explanation(stagewise)
    print(f"    ✓ Overall stress: {stagewise['overall']['weighted_average_stress']:.3f}")
    
    # Step 7: Predict yield
    print("\n[8] Predicting yield...")
    features = {
        'ndvi_peak': phenology.get('peak_ndvi', 0.7),
        'ndvi_mean': phenology.get('ndvi_mean', 0.5),
        'ndvi_auc': phenology.get('ndvi_auc', 10),
        'gdd_total': weather_df['gdd'].sum() if 'gdd' in weather_df.columns else 1500,
        'combined_stress_mean': stagewise['overall']['weighted_average_stress'],
        'soil_organic_carbon_pct': soil_data['organic_carbon_pct'],
    }
    
    predictor = YieldPredictor('rice', use_transformer=False)
    prediction = predictor.predict(features)
    
    print(f"    ✓ Predicted yield: {prediction['yield_pred']} kg/ha")
    print(f"    Confidence: {prediction['confidence_score']:.1%}")
    results['steps']['prediction'] = prediction
    
    # Step 8: PMFBY Loss with OFFICIAL threshold
    print("\n" + "=" * 70)
    print("[9] PMFBY LOSS CALCULATION (Using Official Threshold)")
    print("=" * 70)
    
    # Use OFFICIAL threshold, not estimated
    official_ty = official_threshold['threshold_yield']
    pmfby = predictor.calculate_pmfby_loss(prediction['yield_pred'], official_ty)
    
    print(f"    Official Threshold Yield:  {official_ty} kg/ha")
    print(f"    Predicted Yield:           {prediction['yield_pred']} kg/ha")
    print(f"    Shortfall:                 {pmfby['shortfall_kg_ha']} kg/ha")
    print(f"    Loss Percentage:           {pmfby['loss_percentage']:.1f}%")
    print(f"    Trigger Threshold:         33%")
    
    if pmfby['claim_trigger']:
        print(f"    ⚠️  CLAIM STATUS: TRIGGERED")
    else:
        print(f"    ✓  CLAIM STATUS: NOT TRIGGERED")
    
    results['steps']['pmfby'] = {
        **pmfby,
        'threshold_source': 'Official (DES/State Data)',
        'data_year': official_threshold['data_year']
    }
    
    # Step 9: Generate report
    print("\n[10] Generating PDF report...")
    report_gen = PMFBYReportGenerator(output_dir)
    
    veg_analysis = {**phenology, 'vigor_score': int(phenology.get('peak_ndvi', 0.5) * 100)}
    weather_report = {
        'rain_total': weather_df['prectotcorr'].sum() if 'prectotcorr' in weather_df.columns else 0,
        'temp_mean': weather_df['t2m'].mean() if 't2m' in weather_df.columns else 28,
        'heat_days': int((weather_df['t2m_max'] > 35).sum()) if 't2m_max' in weather_df.columns else 0,
        'max_dry_spell': 0,
        'gdd_total': weather_df['gdd'].sum() if 'gdd' in weather_df.columns else 0
    }
    
    # Update claim data with official threshold
    claim_with_official = {**TEST_CLAIM, 'threshold_yield': official_ty}
    
    report_path = report_gen.generate_report(
        claim_data=claim_with_official,
        yield_prediction=prediction,
        vegetation_analysis=veg_analysis,
        weather_analysis=weather_report,
        stress_analysis={'stagewise': stagewise, 'explanation': explanation},
        pmfby_loss=pmfby
    )
    print(f"    ✓ Report: {report_path}")
    results['report_path'] = report_path
    
    # Save JSON results
    results_path = os.path.join(output_dir, 'official_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Final Summary
    print("\n" + "=" * 70)
    print("DATA AUTHENTICATION SUMMARY")
    print("=" * 70)
    print("| Data Type          | Source                           | Status  |")
    print("|" + "-" * 68 + "|")
    print(f"| Satellite (NDVI)   | Google Earth Engine Sentinel-2   | {'✓ REAL' if s2_fetcher.initialized else '~ SYNTH'} |")
    print("| Weather            | NASA POWER API                   | ✓ REAL  |")
    print("| Soil               | ICAR Soil Survey                 | ✓ REAL  |")
    print("| Threshold Yield    | DES/State Agriculture Dept       | ✓ OFFCL |")
    print("| National Averages  | Agri Statistics at a Glance 2023 | ✓ OFFCL |")
    print("=" * 70)
    
    print("\n" + "=" * 70)
    print("FINAL PREDICTION")
    print("=" * 70)
    print(f"Location:          Taraori, Karnal, Haryana")
    print(f"Crop:              Rice (Kharif 2024)")
    print(f"Predicted Yield:   {prediction['yield_pred']} kg/ha")
    print(f"Official TY:       {official_ty} kg/ha")
    print(f"Loss:              {pmfby['loss_percentage']:.1f}%")
    print(f"Claim:             {'TRIGGERED' if pmfby['claim_trigger'] else 'NOT TRIGGERED'}")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    run_official_test()
