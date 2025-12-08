"""
PMFBY Pipeline - Real Data Test
Runs the complete pipeline with real satellite and weather data.
"""

import os
import sys
import json
from datetime import datetime

# Add pmfby_engine to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("PMFBY YIELD PREDICTION - REAL DATA TEST")
print("=" * 60)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Real farm data for testing
# Location: Taraori village, Karnal district, Haryana
# This is a prime rice-wheat belt area

TEST_CLAIM = {
    'claim_id': 'REAL_TEST_001',
    'farmer_name': 'Test Farmer - Karnal',
    'crop_type': 'rice',
    'season': 'Kharif 2024',
    'sowing_date': '2024-06-15',
    'district': 'Karnal',
    'village': 'Taraori',
    'area_ha': 2.5,
    'sum_insured': 150000,
    'threshold_yield': 3200,  # Karnal average is ~3200 kg/ha
    'coordinates': '29.69°N, 76.97°E',
    'farm_polygon': {
        "type": "Polygon",
        "coordinates": [[[76.96, 29.68], [76.98, 29.68],
                        [76.98, 29.70], [76.96, 29.70],
                        [76.96, 29.68]]]
    }
}

def run_real_test():
    """Run the pipeline with real data."""
    
    # Import modules
    print("[1] Importing PMFBY modules...")
    
    try:
        from data_ingestion.sentinel2_fetcher import Sentinel2Fetcher
        from data_ingestion.weather_fetcher import WeatherFetcher
        from data_ingestion.soil_fetcher import SoilFetcher
        from data_ingestion.preprocessing import TimeSeriesPreprocessor
        from feature_engineering.stress_indices import StressIndexCalculator
        from feature_engineering.feature_extraction import FeatureExtractor
        from models.crop_stage_detector import CropStageDetector
        from models.yield_transformer import YieldPredictor
        from outputs.report_generator import PMFBYReportGenerator
        print("    ✓ All modules imported successfully")
    except ImportError as e:
        print(f"    ✗ Import error: {e}")
        print("    Installing missing dependencies...")
        os.system("pip install pandas numpy scipy requests")
        return
    
    # Create output directory
    output_dir = os.path.join("output", TEST_CLAIM['claim_id'])
    os.makedirs(output_dir, exist_ok=True)
    
    results = {'claim_id': TEST_CLAIM['claim_id'], 'steps': {}}
    
    # Step 1: Fetch Sentinel-2 data
    print("\n[2] Fetching Sentinel-2 satellite data...")
    s2_fetcher = Sentinel2Fetcher()
    s2_df = s2_fetcher.fetch_time_series(
        geometry=TEST_CLAIM['farm_polygon'],
        start_date=TEST_CLAIM['sowing_date'],
        end_date='2024-11-15'
    )
    print(f"    ✓ Retrieved {len(s2_df)} observations")
    print(f"    NDVI range: {s2_df['ndvi'].min():.3f} - {s2_df['ndvi'].max():.3f}")
    results['steps']['satellite'] = {'observations': len(s2_df), 'source': 'GEE' if s2_fetcher.initialized else 'synthetic'}
    
    # Step 2: Fetch weather data
    print("\n[3] Fetching NASA POWER weather data...")
    weather_fetcher = WeatherFetcher()
    weather_df = weather_fetcher.fetch_daily_weather(
        latitude=29.69,
        longitude=76.97,
        start_date='2024-06-15',
        end_date='2024-11-15'
    )
    print(f"    ✓ Retrieved {len(weather_df)} days of weather data")
    if 'prectotcorr' in weather_df.columns:
        print(f"    Total rainfall: {weather_df['prectotcorr'].sum():.1f} mm")
    if 't2m_max' in weather_df.columns:
        print(f"    Max temperature: {weather_df['t2m_max'].max():.1f}°C")
    results['steps']['weather'] = {'days': len(weather_df)}
    
    # Step 3: Get soil data
    print("\n[4] Loading ICAR soil data for Karnal...")
    soil_fetcher = SoilFetcher()
    soil_data = soil_fetcher.get_soil_profile('Karnal')
    suitability = soil_fetcher.assess_suitability('Karnal', 'rice')
    print(f"    ✓ Soil type: {soil_data['soil_type']}")
    print(f"    pH: {soil_data['ph']}, Organic Carbon: {soil_data['organic_carbon_pct']}%")
    print(f"    Rice suitability: {suitability['suitability_class']} ({suitability['suitability_score']}/100)")
    results['steps']['soil'] = suitability
    
    # Step 4: Preprocess data
    print("\n[5] Preprocessing time series (Savitzky-Golay smoothing)...")
    preprocessor = TimeSeriesPreprocessor()
    s2_clean = preprocessor.preprocess_vegetation_series(s2_df)
    phenology = preprocessor.extract_phenological_features(s2_clean)
    print(f"    ✓ Peak NDVI: {phenology.get('peak_ndvi', 0):.3f}")
    print(f"    Season length: {phenology.get('season_length_days', 0)} days")
    results['steps']['phenology'] = phenology
    
    # Step 5: Detect crop stages
    print("\n[6] Detecting crop growth stages...")
    stage_detector = CropStageDetector('rice')
    stages = stage_detector.get_stage_boundaries()
    s2_clean = stage_detector.add_stage_labels(s2_clean, TEST_CLAIM['sowing_date'])
    print("    Stage boundaries:")
    for stage, (start, end) in stages.items():
        print(f"      {stage}: day {start} - {end}")
    
    # Step 6: Calculate stress indices
    print("\n[7] Computing stress indices...")
    stress_calc = StressIndexCalculator('rice')
    
    # Merge weather for stress
    weather_df['date'] = weather_df['date'].astype(str).str[:10]
    s2_clean['date_str'] = s2_clean['date'].astype(str).str[:10]
    
    # Calculate stresses on weather data
    weather_df = stress_calc.calculate_heat_stress(weather_df)
    weather_df = stress_calc.calculate_water_balance_stress(weather_df)
    
    stagewise = stress_calc.calculate_stagewise_stress(weather_df, TEST_CLAIM['sowing_date'], stages)
    explanation = stress_calc.generate_stress_explanation(stagewise)
    
    print(f"    ✓ Overall stress: {stagewise['overall']['weighted_average_stress']:.3f}")
    print(f"    Most stressed stage: {stagewise['overall']['most_stressed_stage']}")
    print(f"    Explanation: {explanation[:80]}...")
    results['steps']['stress'] = {'overall': stagewise['overall'], 'explanation': explanation}
    
    # Step 7: Extract features
    print("\n[8] Extracting ML features...")
    extractor = FeatureExtractor()
    
    # Add stress to features dict
    features = {
        'ndvi_peak': phenology.get('peak_ndvi', 0.7),
        'ndvi_mean': phenology.get('ndvi_mean', 0.5),
        'ndvi_auc': phenology.get('ndvi_auc', 10),
        'gdd_total': weather_df['gdd'].sum() if 'gdd' in weather_df.columns else 1500,
        'combined_stress_mean': stagewise['overall']['weighted_average_stress'],
        'soil_organic_carbon_pct': soil_data['organic_carbon_pct'],
        'rain_total': weather_df['prectotcorr'].sum() if 'prectotcorr' in weather_df.columns else 800
    }
    print(f"    ✓ Extracted {len(features)} key features")
    
    # Step 8: Predict yield
    print("\n[9] Predicting yield...")
    predictor = YieldPredictor('rice', use_transformer=False)
    prediction = predictor.predict(features)
    
    print(f"    ✓ Predicted yield: {prediction['yield_pred']} kg/ha")
    print(f"    Confidence interval: {prediction['yield_low_10']} - {prediction['yield_high_90']} kg/ha")
    print(f"    Confidence score: {prediction['confidence_score']:.1%}")
    results['steps']['prediction'] = prediction
    
    # Step 9: Calculate PMFBY loss
    print("\n[10] Calculating PMFBY loss...")
    pmfby = predictor.calculate_pmfby_loss(
        prediction['yield_pred'],
        TEST_CLAIM['threshold_yield']
    )
    
    print(f"    Threshold yield: {pmfby['threshold_yield']} kg/ha")
    print(f"    Shortfall: {pmfby['shortfall_kg_ha']} kg/ha")
    print(f"    Loss percentage: {pmfby['loss_percentage']:.1f}%")
    
    if pmfby['claim_trigger']:
        print(f"    ⚠️  CLAIM TRIGGERED (Loss > 33%)")
    else:
        print(f"    ✓ No claim (Loss < 33% threshold)")
    results['steps']['pmfby'] = pmfby
    
    # Step 10: Generate report
    print("\n[11] Generating PDF report...")
    report_gen = PMFBYReportGenerator(output_dir)
    
    veg_analysis = {**phenology, 'vigor_score': int(phenology.get('peak_ndvi', 0.5) * 100)}
    weather_report = {
        'rain_total': weather_df['prectotcorr'].sum() if 'prectotcorr' in weather_df.columns else 0,
        'temp_mean': weather_df['t2m'].mean() if 't2m' in weather_df.columns else 28,
        'heat_days': int((weather_df['t2m_max'] > 35).sum()) if 't2m_max' in weather_df.columns else 0,
        'max_dry_spell': 0,
        'gdd_total': weather_df['gdd'].sum() if 'gdd' in weather_df.columns else 0
    }
    
    report_path = report_gen.generate_report(
        claim_data=TEST_CLAIM,
        yield_prediction=prediction,
        vegetation_analysis=veg_analysis,
        weather_analysis=weather_report,
        stress_analysis={'stagewise': stagewise, 'explanation': explanation},
        pmfby_loss=pmfby
    )
    print(f"    ✓ Report saved: {report_path}")
    results['report_path'] = report_path
    
    # Save results JSON
    results_path = os.path.join(output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"    ✓ Results saved: {results_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("PREDICTION SUMMARY")
    print("=" * 60)
    print(f"Farm Location:     Taraori, Karnal, Haryana")
    print(f"Crop:              Rice (Kharif 2024)")
    print(f"Predicted Yield:   {prediction['yield_pred']} kg/ha")
    print(f"Threshold Yield:   {TEST_CLAIM['threshold_yield']} kg/ha")
    print(f"Loss:              {pmfby['loss_percentage']:.1f}%")
    print(f"Claim Status:      {'TRIGGERED' if pmfby['claim_trigger'] else 'NOT TRIGGERED'}")
    print(f"Confidence:        {prediction['confidence_score']:.1%}")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    run_real_test()
