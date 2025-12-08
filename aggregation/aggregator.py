"""
PMFBY Yield Prediction Engine
Aggregation Module

Implements pixel → farm → village → district aggregation
with area-weighted statistics for PMFBY reporting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PixelToFarmAggregator:
    """
    Aggregates pixel-level predictions to farm-level statistics.
    
    Uses robust statistics (median, IQR) to handle outliers
    and missing pixels.
    """
    
    def __init__(self, min_valid_pixels: int = 10):
        """
        Initialize aggregator.
        
        Args:
            min_valid_pixels: Minimum valid pixels required for reliable estimate
        """
        self.min_valid_pixels = min_valid_pixels
    
    def aggregate_pixels(
        self,
        pixel_values: np.ndarray,
        weights: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Aggregate pixel values to farm-level statistics.
        
        Args:
            pixel_values: Array of pixel-level values
            weights: Optional weights for each pixel
            
        Returns:
            Dictionary with robust statistics
        """
        # Remove NaN values
        valid_mask = ~np.isnan(pixel_values)
        valid_values = pixel_values[valid_mask]
        
        if len(valid_values) < self.min_valid_pixels:
            logger.warning(f"Only {len(valid_values)} valid pixels. Using available data.")
        
        if len(valid_values) == 0:
            return {
                'mean': np.nan,
                'median': np.nan,
                'std': np.nan,
                'p25': np.nan,
                'p75': np.nan,
                'iqr': np.nan,
                'valid_pixels': 0,
                'total_pixels': len(pixel_values),
                'reliability': 'LOW'
            }
        
        # Calculate weighted statistics if weights provided
        if weights is not None:
            valid_weights = weights[valid_mask]
            weighted_mean = np.average(valid_values, weights=valid_weights)
        else:
            weighted_mean = np.mean(valid_values)
        
        # Robust statistics
        median = np.median(valid_values)
        std = np.std(valid_values)
        p25 = np.percentile(valid_values, 25)
        p75 = np.percentile(valid_values, 75)
        iqr = p75 - p25
        
        # Reliability assessment
        coverage = len(valid_values) / len(pixel_values)
        cv = std / median if median > 0 else 999
        
        if coverage >= 0.8 and cv < 0.3:
            reliability = 'HIGH'
        elif coverage >= 0.5 and cv < 0.5:
            reliability = 'MEDIUM'
        else:
            reliability = 'LOW'
        
        return {
            'mean': round(float(weighted_mean), 4),
            'median': round(float(median), 4),
            'std': round(float(std), 4),
            'p25': round(float(p25), 4),
            'p75': round(float(p75), 4),
            'iqr': round(float(iqr), 4),
            'min': round(float(np.min(valid_values)), 4),
            'max': round(float(np.max(valid_values)), 4),
            'valid_pixels': int(len(valid_values)),
            'total_pixels': int(len(pixel_values)),
            'coverage': round(coverage, 3),
            'coefficient_of_variation': round(cv, 3),
            'reliability': reliability
        }


class FarmToVillageAggregator:
    """
    Aggregates farm-level predictions to village/GP level.
    
    Uses area-weighted averaging for representative estimates.
    """
    
    def __init__(self):
        """Initialize aggregator."""
        pass
    
    def aggregate_farms(
        self,
        farms: List[Dict]
    ) -> Dict:
        """
        Aggregate multiple farms to village-level statistics.
        
        Args:
            farms: List of dictionaries with farm data
                  Each must have: farm_id, yield_pred, area_ha
                  Optional: yield_low, yield_high, confidence
                  
        Returns:
            Dictionary with village-level aggregated statistics
        """
        if not farms:
            return {'error': 'No farms provided'}
        
        # Extract data
        farm_ids = [f.get('farm_id', i) for i, f in enumerate(farms)]
        yields = np.array([f['yield_pred'] for f in farms])
        areas = np.array([f.get('area_ha', 1.0) for f in farms])
        
        # Handle missing values
        valid_mask = ~np.isnan(yields) & ~np.isnan(areas)
        yields = yields[valid_mask]
        areas = areas[valid_mask]
        
        if len(yields) == 0:
            return {'error': 'No valid farm data'}
        
        # Area-weighted mean
        total_area = np.sum(areas)
        weighted_yield = np.sum(yields * areas) / total_area
        
        # Simple statistics
        simple_mean = np.mean(yields)
        simple_std = np.std(yields)
        
        # Uncertainty aggregation (if available)
        if 'yield_low_10' in farms[0]:
            lows = np.array([f.get('yield_low_10', f['yield_pred'] * 0.85) for f in farms])
            highs = np.array([f.get('yield_high_90', f['yield_pred'] * 1.15) for f in farms])
            lows = lows[valid_mask]
            highs = highs[valid_mask]
            
            weighted_low = np.sum(lows * areas) / total_area
            weighted_high = np.sum(highs * areas) / total_area
        else:
            weighted_low = weighted_yield * 0.85
            weighted_high = weighted_yield * 1.15
        
        # Confidence aggregation
        if 'confidence_score' in farms[0]:
            confidences = np.array([f.get('confidence_score', 0.7) for f in farms])
            confidences = confidences[valid_mask]
            avg_confidence = np.average(confidences, weights=areas)
        else:
            avg_confidence = 0.7
        
        return {
            'weighted_yield': round(float(weighted_yield), 2),
            'yield_low_10': round(float(weighted_low), 2),
            'yield_high_90': round(float(weighted_high), 2),
            'simple_mean_yield': round(float(simple_mean), 2),
            'yield_std': round(float(simple_std), 2),
            'total_area_ha': round(float(total_area), 4),
            'farm_count': int(len(yields)),
            'confidence_score': round(float(avg_confidence), 3),
            'yield_range': {
                'min': round(float(np.min(yields)), 2),
                'max': round(float(np.max(yields)), 2),
                'p25': round(float(np.percentile(yields, 25)), 2),
                'p75': round(float(np.percentile(yields, 75)), 2)
            }
        }


class VillageToDistrictAggregator:
    """
    Aggregates village-level data to block/district level
    for PMFBY reporting and CCE comparison.
    """
    
    def __init__(self):
        """Initialize aggregator."""
        pass
    
    def aggregate_villages(
        self,
        villages: List[Dict]
    ) -> Dict:
        """
        Aggregate villages to block/district level.
        
        Args:
            villages: List of village data dictionaries
                     Each must have: village_code, weighted_yield, total_area_ha
                     
        Returns:
            District-level summary
        """
        if not villages:
            return {'error': 'No villages provided'}
        
        # Extract data
        village_codes = [v.get('village_code', i) for i, v in enumerate(villages)]
        yields = np.array([v['weighted_yield'] for v in villages])
        areas = np.array([v.get('total_area_ha', 1.0) for v in villages])
        farm_counts = np.array([v.get('farm_count', 1) for v in villages])
        
        # Filter valid
        valid_mask = ~np.isnan(yields) & ~np.isnan(areas)
        yields = yields[valid_mask]
        areas = areas[valid_mask]
        farm_counts = farm_counts[valid_mask]
        
        if len(yields) == 0:
            return {'error': 'No valid village data'}
        
        # Area-weighted district yield
        total_area = np.sum(areas)
        district_yield = np.sum(yields * areas) / total_area
        
        # Confidence intervals
        if 'yield_low_10' in villages[0]:
            lows = np.array([v.get('yield_low_10', v['weighted_yield'] * 0.85) for v in villages])
            highs = np.array([v.get('yield_high_90', v['weighted_yield'] * 1.15) for v in villages])
            lows = lows[valid_mask]
            highs = highs[valid_mask]
            
            district_low = np.sum(lows * areas) / total_area
            district_high = np.sum(highs * areas) / total_area
        else:
            district_low = district_yield * 0.85
            district_high = district_yield * 1.15
        
        # Performance classification
        yield_mean = np.mean(yields)
        yield_std = np.std(yields)
        
        low_villages = np.sum(yields < yield_mean - yield_std)
        avg_villages = np.sum((yields >= yield_mean - yield_std) & (yields <= yield_mean + yield_std))
        high_villages = np.sum(yields > yield_mean + yield_std)
        
        return {
            'district_yield': round(float(district_yield), 2),
            'yield_low_10': round(float(district_low), 2),
            'yield_high_90': round(float(district_high), 2),
            'total_area_ha': round(float(total_area), 2),
            'total_villages': int(len(yields)),
            'total_farms': int(np.sum(farm_counts)),
            'yield_distribution': {
                'mean': round(float(yield_mean), 2),
                'std': round(float(yield_std), 2),
                'min': round(float(np.min(yields)), 2),
                'max': round(float(np.max(yields)), 2),
                'cv': round(float(yield_std / yield_mean) if yield_mean > 0 else 0, 3)
            },
            'village_classification': {
                'below_average': int(low_villages),
                'average': int(avg_villages),
                'above_average': int(high_villages)
            }
        }


class BiasCorrector:
    """
    Applies bias correction to predictions using historical CCE data.
    
    Methods:
    - Linear correction (slope, intercept)
    - Quantile mapping
    """
    
    def __init__(self):
        """Initialize bias corrector."""
        self.correction_models = {}
    
    def fit_linear(
        self,
        predicted: np.ndarray,
        actual: np.ndarray,
        crop: str,
        district: str,
        season: str
    ) -> Dict:
        """
        Fit linear bias correction model.
        
        Args:
            predicted: Array of predicted yields
            actual: Array of actual CCE yields
            crop: Crop type
            district: District code
            season: Season (kharif/rabi)
            
        Returns:
            Fitted parameters
        """
        if len(predicted) < 3:
            logger.warning("Insufficient data for linear correction. Using 1:1 mapping.")
            params = {'slope': 1.0, 'intercept': 0.0, 'r2': 0.0}
        else:
            # Linear regression
            from numpy.polynomial import polynomial as P
            coef, stats = P.polyfit(predicted, actual, 1, full=True)
            intercept, slope = coef
            
            # R-squared
            ss_res = stats[0][0] if len(stats[0]) > 0 else 0
            ss_tot = np.sum((actual - np.mean(actual)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            params = {
                'slope': round(float(slope), 4),
                'intercept': round(float(intercept), 4),
                'r2': round(float(r2), 4)
            }
        
        key = f"{crop}_{district}_{season}"
        self.correction_models[key] = {
            'type': 'linear',
            'params': params,
            'n_samples': len(predicted)
        }
        
        return params
    
    def apply_correction(
        self,
        predicted: float,
        crop: str,
        district: str,
        season: str
    ) -> Dict:
        """
        Apply bias correction to prediction.
        
        Args:
            predicted: Raw predicted yield
            crop: Crop type
            district: District code
            season: Season
            
        Returns:
            Corrected prediction with metadata
        """
        key = f"{crop}_{district}_{season}"
        
        if key not in self.correction_models:
            logger.info(f"No correction model for {key}. Using raw prediction.")
            return {
                'raw_prediction': predicted,
                'corrected_prediction': predicted,
                'correction_applied': False
            }
        
        model = self.correction_models[key]
        params = model['params']
        
        if model['type'] == 'linear':
            corrected = predicted * params['slope'] + params['intercept']
        else:
            corrected = predicted
        
        # Ensure positive
        corrected = max(0, corrected)
        
        return {
            'raw_prediction': round(predicted, 2),
            'corrected_prediction': round(corrected, 2),
            'correction_applied': True,
            'correction_type': model['type'],
            'model_r2': params.get('r2', None)
        }


def main():
    """Test aggregation modules."""
    print("\n=== Pixel to Farm Aggregation ===")
    pixel_agg = PixelToFarmAggregator()
    
    # Simulate pixel values
    pixels = np.random.normal(0.65, 0.1, 100)
    pixels[5:10] = np.nan  # Add some missing
    
    farm_stats = pixel_agg.aggregate_pixels(pixels)
    for k, v in farm_stats.items():
        print(f"  {k}: {v}")
    
    print("\n=== Farm to Village Aggregation ===")
    village_agg = FarmToVillageAggregator()
    
    farms = [
        {'farm_id': 'F001', 'yield_pred': 2800, 'area_ha': 1.5, 'yield_low_10': 2400, 'yield_high_90': 3200, 'confidence_score': 0.85},
        {'farm_id': 'F002', 'yield_pred': 3100, 'area_ha': 2.0, 'yield_low_10': 2700, 'yield_high_90': 3500, 'confidence_score': 0.78},
        {'farm_id': 'F003', 'yield_pred': 2500, 'area_ha': 0.8, 'yield_low_10': 2100, 'yield_high_90': 2900, 'confidence_score': 0.82},
        {'farm_id': 'F004', 'yield_pred': 2950, 'area_ha': 1.2, 'yield_low_10': 2550, 'yield_high_90': 3350, 'confidence_score': 0.80},
    ]
    
    village_stats = village_agg.aggregate_farms(farms)
    for k, v in village_stats.items():
        print(f"  {k}: {v}")
    
    print("\n=== Village to District Aggregation ===")
    district_agg = VillageToDistrictAggregator()
    
    villages = [
        {'village_code': 'V001', 'weighted_yield': 2850, 'total_area_ha': 50, 'farm_count': 25},
        {'village_code': 'V002', 'weighted_yield': 3100, 'total_area_ha': 75, 'farm_count': 40},
        {'village_code': 'V003', 'weighted_yield': 2600, 'total_area_ha': 30, 'farm_count': 15},
        {'village_code': 'V004', 'weighted_yield': 2900, 'total_area_ha': 60, 'farm_count': 32},
    ]
    
    district_stats = district_agg.aggregate_villages(villages)
    for k, v in district_stats.items():
        print(f"  {k}: {v}")
    
    print("\n=== Bias Correction ===")
    corrector = BiasCorrector()
    
    # Fit on historical data
    pred = np.array([2500, 2800, 3000, 2700, 3200])
    actual = np.array([2600, 2900, 2950, 2800, 3100])
    
    params = corrector.fit_linear(pred, actual, 'rice', 'karnal', 'kharif')
    print(f"  Fitted params: {params}")
    
    # Apply correction
    corrected = corrector.apply_correction(2800, 'rice', 'karnal', 'kharif')
    print(f"  Correction result: {corrected}")
    
    return district_stats


if __name__ == "__main__":
    main()
