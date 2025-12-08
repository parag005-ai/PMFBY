"""
PMFBY Yield Prediction Engine
Master Pipeline

Orchestrates the complete end-to-end yield prediction workflow:
Data Ingestion → Feature Engineering → Prediction → Aggregation → Reporting
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# Import pipeline modules
from data_ingestion.sentinel2_fetcher import Sentinel2Fetcher
from data_ingestion.sentinel1_fetcher import Sentinel1Fetcher
from data_ingestion.weather_fetcher import WeatherFetcher
from data_ingestion.soil_fetcher import SoilFetcher
from data_ingestion.preprocessing import TimeSeriesPreprocessor, DataMerger

from feature_engineering.stress_indices import StressIndexCalculator
from feature_engineering.feature_extraction import FeatureExtractor

from models.crop_stage_detector import CropStageDetector
from models.yield_transformer import YieldPredictor

from aggregation.aggregator import (
    PixelToFarmAggregator,
    FarmToVillageAggregator,
    VillageToDistrictAggregator,
    BiasCorrector
)

from outputs.report_generator import PMFBYReportGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PMFBYPipeline:
    """
    Master orchestrator for PMFBY yield prediction pipeline.
    
    Complete workflow:
    1. Fetch satellite data (Sentinel-2, Sentinel-1)
    2. Fetch weather data (NASA POWER)
    3. Get soil data (ICAR)
    4. Preprocess and merge time series
    5. Calculate stress indices
    6. Extract features
    7. Detect crop stages
    8. Predict yield
    9. Calculate PMFBY loss
    10. Generate report
    """
    
    def __init__(
        self,
        output_dir: str = "output",
        gee_project_id: Optional[str] = None
    ):
        """
        Initialize pipeline.
        
        Args:
            output_dir: Base output directory
            gee_project_id: Optional GEE project ID
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize components
        logger.info("Initializing PMFBY Pipeline components...")
        
        self.s2_fetcher = Sentinel2Fetcher(gee_project_id)
        self.s1_fetcher = Sentinel1Fetcher(gee_project_id)
        self.weather_fetcher = WeatherFetcher()
        self.soil_fetcher = SoilFetcher()
        
        self.preprocessor = TimeSeriesPreprocessor()
        self.data_merger = DataMerger()
        
        self.feature_extractor = FeatureExtractor()
        
        self.aggregator = PixelToFarmAggregator()
        self.bias_corrector = BiasCorrector()
        
        self.report_generator = PMFBYReportGenerator(
            os.path.join(output_dir, "reports")
        )
        
        logger.info("Pipeline initialized successfully")
    
    def run_farm_prediction(
        self,
        claim_data: Dict
    ) -> Dict:
        """
        Run complete prediction pipeline for a single farm.
        
        Args:
            claim_data: Dictionary with required fields:
                - claim_id: Unique identifier
                - farm_polygon: GeoJSON geometry
                - sowing_date: YYYY-MM-DD
                - crop_type: rice, wheat, etc.
                - district: District name
                - village: (optional) Village name
                - farmer_name: (optional)
                - area_ha: (optional) Farm area
                - threshold_yield: (optional) Custom threshold
                - sum_insured: (optional)
                
        Returns:
            Dictionary with complete prediction results
        """
        claim_id = claim_data.get('claim_id', f"CLAIM_{datetime.now().strftime('%Y%m%d%H%M%S')}")
        logger.info(f"Starting prediction for claim: {claim_id}")
        
        # Create output directory for this claim
        claim_dir = os.path.join(self.output_dir, claim_id)
        os.makedirs(claim_dir, exist_ok=True)
        
        results = {
            'claim_id': claim_id,
            'claim_data': claim_data,
            'timestamp': datetime.now().isoformat(),
            'status': 'processing'
        }
        
        try:
            # Extract parameters
            farm_polygon = claim_data.get('farm_polygon')
            sowing_date = claim_data.get('sowing_date')
            crop_type = claim_data.get('crop_type', 'rice')
            district = claim_data.get('district', 'unknown')
            
            # Calculate date range (sowing to maturity)
            stage_detector = CropStageDetector(crop_type)
            stages = stage_detector.get_stage_boundaries()
            max_days = max(info[1] for info in stages.values())
            
            sowing_dt = pd.to_datetime(sowing_date)
            end_date = (sowing_dt + pd.Timedelta(days=max_days + 10)).strftime('%Y-%m-%d')
            
            # Get centroid for weather data
            if farm_polygon:
                coords = farm_polygon.get('coordinates', [[[0, 0]]])[0]
                centroid_lon = np.mean([c[0] for c in coords])
                centroid_lat = np.mean([c[1] for c in coords])
            else:
                # Default coordinates (Karnal, Haryana)
                centroid_lat, centroid_lon = 29.69, 76.97
                farm_polygon = {
                    "type": "Polygon",
                    "coordinates": [[[centroid_lon-0.01, centroid_lat-0.01],
                                    [centroid_lon+0.01, centroid_lat-0.01],
                                    [centroid_lon+0.01, centroid_lat+0.01],
                                    [centroid_lon-0.01, centroid_lat+0.01],
                                    [centroid_lon-0.01, centroid_lat-0.01]]]
                }
            
            # Step 1: Fetch satellite data
            logger.info("Step 1: Fetching Sentinel-2 data...")
            s2_df = self.s2_fetcher.fetch_time_series(
                geometry=farm_polygon,
                start_date=sowing_date,
                end_date=end_date
            )
            results['s2_observations'] = len(s2_df)
            
            logger.info("Step 1b: Fetching Sentinel-1 SAR data...")
            s1_df = self.s1_fetcher.fetch_time_series(
                geometry=farm_polygon,
                start_date=sowing_date,
                end_date=end_date
            )
            results['s1_observations'] = len(s1_df)
            
            # Step 2: Fetch weather data
            logger.info("Step 2: Fetching weather data...")
            weather_df = self.weather_fetcher.fetch_daily_weather(
                latitude=centroid_lat,
                longitude=centroid_lon,
                start_date=sowing_date,
                end_date=end_date
            )
            results['weather_days'] = len(weather_df)
            
            # Step 3: Get soil data
            logger.info("Step 3: Getting soil data...")
            soil_data = self.soil_fetcher.get_soil_profile(district)
            soil_suitability = self.soil_fetcher.assess_suitability(district, crop_type)
            results['soil_suitability'] = soil_suitability['suitability_class']
            
            # Step 4: Preprocess satellite data
            logger.info("Step 4: Preprocessing time series...")
            s2_clean = self.preprocessor.preprocess_vegetation_series(s2_df)
            
            # Step 5: Merge data sources
            logger.info("Step 5: Merging data sources...")
            merged_df = self.data_merger.merge_timeseries(s2_clean, s1_df, weather_df)
            merged_df = self.data_merger.add_soil_features(merged_df, soil_data)
            
            # Step 6: Add crop stages
            logger.info("Step 6: Detecting crop stages...")
            merged_df = stage_detector.add_stage_labels(merged_df, sowing_date)
            
            # Step 7: Calculate stress indices
            logger.info("Step 7: Calculating stress indices...")
            stress_calculator = StressIndexCalculator(crop_type)
            merged_df = stress_calculator.calculate_heat_stress(merged_df)
            merged_df = stress_calculator.calculate_moisture_stress(merged_df)
            merged_df = stress_calculator.calculate_water_balance_stress(merged_df)
            merged_df = stress_calculator.calculate_combined_stress(merged_df)
            
            stagewise_stress = stress_calculator.calculate_stagewise_stress(
                merged_df, sowing_date, stages
            )
            stress_explanation = stress_calculator.generate_stress_explanation(stagewise_stress)
            
            results['stress_analysis'] = {
                'stagewise': stagewise_stress,
                'explanation': stress_explanation
            }
            
            # Step 8: Extract features
            logger.info("Step 8: Extracting features...")
            features = self.feature_extractor.extract_all_features(
                merged_df, soil_data, sowing_date
            )
            
            # Extract phenological features
            phenology_features = self.preprocessor.extract_phenological_features(s2_clean)
            features.update(phenology_features)
            
            results['features'] = {k: v for k, v in features.items() if k != 'sequence'}
            
            # Step 9: Predict yield
            logger.info("Step 9: Predicting yield...")
            predictor = YieldPredictor(crop_type, use_transformer=False)  # Use empirical for now
            prediction = predictor.predict(features)
            results['yield_prediction'] = prediction
            
            # Step 10: Calculate PMFBY loss
            logger.info("Step 10: Calculating PMFBY loss...")
            threshold_yield = claim_data.get(
                'threshold_yield',
                predictor.params['base_yield'] * 1.1  # Default threshold
            )
            
            pmfby_loss = predictor.calculate_pmfby_loss(
                prediction['yield_pred'],
                threshold_yield
            )
            results['pmfby_analysis'] = pmfby_loss
            
            # Step 11: Weather stress analysis
            logger.info("Step 11: Analyzing weather stress...")
            weather_stress = self.weather_fetcher.analyze_stress_conditions(
                weather_df, stages, sowing_date
            )
            results['weather_analysis'] = weather_stress
            
            # Step 12: Generate report
            logger.info("Step 12: Generating report...")
            
            # Prepare data for report
            veg_analysis = {
                'ndvi_peak': features.get('ndvi_peak', 0),
                'ndvi_mean': features.get('ndvi_mean', 0),
                'ndvi_auc': features.get('ndvi_auc', 0),
                'season_length': phenology_features.get('season_length_days', 0),
                'peak_date': phenology_features.get('peak_date', 'N/A'),
                'vigor_score': int(features.get('ndvi_peak', 0.5) * 100)
            }
            
            weather_report = {
                'rain_total': features.get('rain_total', 0),
                'temp_mean': features.get('temp_mean', 0),
                'heat_days': features.get('heat_days', 0),
                'max_dry_spell': features.get('max_dry_spell', 0),
                'gdd_total': features.get('gdd_total', 0)
            }
            
            report_path = self.report_generator.generate_report(
                claim_data=claim_data,
                yield_prediction=prediction,
                vegetation_analysis=veg_analysis,
                weather_analysis=weather_report,
                stress_analysis=results['stress_analysis'],
                pmfby_loss=pmfby_loss
            )
            results['report_path'] = report_path
            
            # Save merged data
            merged_path = os.path.join(claim_dir, 'merged_timeseries.csv')
            merged_df.to_csv(merged_path, index=False)
            results['data_path'] = merged_path
            
            # Save complete results as JSON
            results_path = os.path.join(claim_dir, 'prediction_results.json')
            with open(results_path, 'w') as f:
                # Convert non-serializable items
                json_results = {k: v for k, v in results.items() 
                              if not isinstance(v, (pd.DataFrame, np.ndarray))}
                json.dump(json_results, f, indent=2, default=str)
            
            results['status'] = 'completed'
            results['results_path'] = results_path
            
            logger.info(f"Pipeline completed successfully for {claim_id}")
            logger.info(f"  Predicted Yield: {prediction['yield_pred']} kg/ha")
            logger.info(f"  Loss: {pmfby_loss['loss_percentage']:.1f}%")
            logger.info(f"  Claim Trigger: {pmfby_loss['claim_trigger']}")
            
        except Exception as e:
            logger.error(f"Pipeline failed for {claim_id}: {str(e)}")
            results['status'] = 'failed'
            results['error'] = str(e)
            
            # Save error state
            error_path = os.path.join(claim_dir, 'error_log.json')
            with open(error_path, 'w') as f:
                json.dump({
                    'claim_id': claim_id,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)
        
        return results
    
    def run_batch_predictions(
        self,
        claims: List[Dict]
    ) -> List[Dict]:
        """
        Run predictions for multiple farms.
        
        Args:
            claims: List of claim data dictionaries
            
        Returns:
            List of prediction results
        """
        results = []
        total = len(claims)
        
        logger.info(f"Starting batch prediction for {total} claims")
        
        for i, claim in enumerate(claims):
            logger.info(f"Processing claim {i+1}/{total}: {claim.get('claim_id', 'unknown')}")
            result = self.run_farm_prediction(claim)
            results.append(result)
        
        # Summary
        completed = sum(1 for r in results if r['status'] == 'completed')
        failed = sum(1 for r in results if r['status'] == 'failed')
        triggered = sum(1 for r in results 
                       if r.get('pmfby_analysis', {}).get('claim_trigger', False))
        
        logger.info(f"Batch complete: {completed} succeeded, {failed} failed, {triggered} claims triggered")
        
        return results


def main():
    """Test the master pipeline."""
    # Initialize pipeline
    pipeline = PMFBYPipeline(output_dir="output/pmfby_test")
    
    # Test claim data
    test_claim = {
        'claim_id': 'TEST_CLAIM_001',
        'farmer_name': 'Test Farmer',
        'crop_type': 'rice',
        'sowing_date': '2024-06-15',
        'district': 'Karnal',
        'village': 'Taraori',
        'area_ha': 2.5,
        'sum_insured': 125000,
        'threshold_yield': 3000,
        'farm_polygon': {
            "type": "Polygon",
            "coordinates": [[[76.95, 29.68], [76.98, 29.68],
                            [76.98, 29.71], [76.95, 29.71],
                            [76.95, 29.68]]]
        }
    }
    
    # Run prediction
    result = pipeline.run_farm_prediction(test_claim)
    
    print("\n" + "=" * 60)
    print("PMFBY PREDICTION RESULT")
    print("=" * 60)
    print(f"Claim ID: {result['claim_id']}")
    print(f"Status: {result['status']}")
    
    if result['status'] == 'completed':
        pred = result['yield_prediction']
        pmfby = result['pmfby_analysis']
        
        print(f"\nYield Prediction:")
        print(f"  Predicted: {pred['yield_pred']} kg/ha")
        print(f"  Range: {pred['yield_low_10']} - {pred['yield_high_90']} kg/ha")
        print(f"  Confidence: {pred['confidence_score']:.1%}")
        
        print(f"\nPMFBY Analysis:")
        print(f"  Threshold: {pmfby['threshold_yield']} kg/ha")
        print(f"  Loss: {pmfby['loss_percentage']:.1f}%")
        print(f"  Claim Trigger: {'YES' if pmfby['claim_trigger'] else 'NO'}")
        
        print(f"\nReport: {result.get('report_path', 'N/A')}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
    
    return result


if __name__ == "__main__":
    main()
