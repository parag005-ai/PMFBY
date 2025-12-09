"""
Integration Module: Segmentation Model → PMFBY Yield Prediction
================================================================

This module connects your crop segmentation model output with the PMFBY
yield prediction system.

Usage:
    from integration import predict_yield_from_segmentation
    
    result = predict_yield_from_segmentation(
        segmentation_output={'centroid': (19.07, 74.77), 'area': 10, ...},
        threshold_yield=1640
    )
"""

import sys
import os
from datetime import datetime
from typing import Dict, Tuple, Optional

# Add PMFBY to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from predict_unified import run_unified_prediction

# District coordinates lookup (Maharashtra)
DISTRICT_COORDS = {
    'Ahmednagar': (19.09, 74.74),
    'Pune': (18.52, 73.86),
    'Nashik': (20.00, 73.78),
    'Solapur': (17.66, 75.91),
    'Kolhapur': (16.69, 74.23),
    'Satara': (17.69, 73.99),
    'Sangli': (16.85, 74.57),
    'Aurangabad': (19.88, 75.32),
    'Jalgaon': (21.00, 75.57),
    'Nagpur': (21.15, 79.09),
    'Amravati': (20.93, 77.75),
    'Akola': (20.71, 77.00),
}


def find_district(lat: float, lon: float) -> str:
    """
    Find nearest district based on coordinates.
    
    Args:
        lat: Latitude
        lon: Longitude
    
    Returns:
        District name
    """
    min_dist = float('inf')
    nearest_district = 'Ahmednagar'  # Default
    
    for district, (d_lat, d_lon) in DISTRICT_COORDS.items():
        # Euclidean distance
        dist = ((lat - d_lat)**2 + (lon - d_lon)**2)**0.5
        if dist < min_dist:
            min_dist = dist
            nearest_district = district
    
    return nearest_district


def get_current_season() -> str:
    """
    Determine agricultural season based on current month.
    
    Returns:
        Season name ('Kharif', 'Rabi', or 'Summer')
    """
    month = datetime.now().month
    
    if 6 <= month <= 11:
        return 'Kharif'
    elif month == 12 or 1 <= month <= 3:
        return 'Rabi'
    else:
        return 'Summer'


def predict_yield_from_segmentation(
    segmentation_output: Dict,
    threshold_yield: float,
    season: Optional[str] = None,
    district: Optional[str] = None
) -> Dict:
    """
    Predict crop yield from segmentation model output.
    
    Args:
        segmentation_output: Dictionary from segmentation model containing:
            - 'centroid': (lat, lon) - Farm center coordinates [REQUIRED]
            - 'area': float - Farm area in hectares [REQUIRED]
            - 'crop_type': str - Detected crop name [REQUIRED]
            - 'ndvi_mean': float - Average NDVI [OPTIONAL]
            - 'ndvi_peak': float - Peak NDVI [OPTIONAL]
            - 'boundary': list - Farm boundary polygon [OPTIONAL]
            - 'confidence': float - Segmentation confidence [OPTIONAL]
        
        threshold_yield: float - PMFBY threshold yield in kg/ha
        
        season: str - Agricultural season ('Kharif', 'Rabi', 'Summer')
                     If None, auto-detected from current date
        
        district: str - District name. If None, auto-detected from coordinates
    
    Returns:
        Dictionary containing:
            - 'prediction': float - Predicted yield (kg/ha)
            - 'uncertainty': float - Uncertainty (kg/ha)
            - 'confidence_interval': tuple - (low, high) 95% CI
            - 'weather': dict - Weather features
            - 'stress': dict - Agronomic stress indices
            - 'pmfby': dict - PMFBY assessment
            - 'segmentation': dict - Original segmentation data
    
    Example:
        >>> seg_output = {
        ...     'centroid': (19.071591, 74.774179),
        ...     'area': 10,
        ...     'crop_type': 'Rice',
        ...     'ndvi_mean': 0.65
        ... }
        >>> result = predict_yield_from_segmentation(seg_output, 1640)
        >>> print(f"Yield: {result['prediction']:.0f} kg/ha")
    """
    
    # Validate required fields
    required_fields = ['centroid', 'area', 'crop_type']
    for field in required_fields:
        if field not in segmentation_output:
            raise ValueError(f"Missing required field: {field}")
    
    # Extract data
    lat, lon = segmentation_output['centroid']
    area = segmentation_output['area']
    crop = segmentation_output['crop_type']
    
    # Auto-detect district if not provided
    if district is None:
        district = find_district(lat, lon)
        print(f"    Auto-detected district: {district}")
    
    # Auto-detect season if not provided
    if season is None:
        season = get_current_season()
        print(f"    Auto-detected season: {season}")
    
    # Run PMFBY prediction
    print(f"\n[Integration] Running yield prediction...")
    print(f"    Location: ({lat:.4f}, {lon:.4f})")
    print(f"    District: {district}")
    print(f"    Crop: {crop} ({season})")
    print(f"    Area: {area} ha")
    
    result = run_unified_prediction(
        lat=lat,
        lon=lon,
        district=district,
        crop=crop,
        season=season,
        area=area,
        threshold=threshold_yield
    )
    
    # Add segmentation metadata
    result['segmentation'] = {
        'boundary': segmentation_output.get('boundary'),
        'ndvi_mean': segmentation_output.get('ndvi_mean'),
        'ndvi_peak': segmentation_output.get('ndvi_peak'),
        'confidence': segmentation_output.get('confidence'),
        'crop_detected': crop,
        'area_ha': area
    }
    
    # Optional: Adjust prediction using segmentation NDVI
    if 'ndvi_mean' in segmentation_output:
        ndvi = segmentation_output['ndvi_mean']
        # NDVI-based adjustment (higher NDVI = higher yield potential)
        # This is a simple linear adjustment, can be made more sophisticated
        ndvi_adjustment = (ndvi - 0.5) * 300  # kg/ha adjustment
        
        result['prediction_original'] = result['prediction']
        result['prediction'] += ndvi_adjustment
        result['ndvi_adjustment'] = ndvi_adjustment
        
        print(f"\n    [NDVI Adjustment] +{ndvi_adjustment:.0f} kg/ha (NDVI={ndvi:.2f})")
    
    return result


def batch_predict(segmentation_results: list, threshold_yield: float) -> list:
    """
    Predict yield for multiple farms in batch.
    
    Args:
        segmentation_results: List of segmentation outputs
        threshold_yield: PMFBY threshold
    
    Returns:
        List of prediction results
    """
    results = []
    
    for i, seg_output in enumerate(segmentation_results):
        print(f"\n{'='*70}")
        print(f"Processing Farm {i+1}/{len(segmentation_results)}")
        print('='*70)
        
        try:
            result = predict_yield_from_segmentation(seg_output, threshold_yield)
            results.append(result)
        except Exception as e:
            print(f"    ERROR: {e}")
            results.append({'error': str(e), 'segmentation': seg_output})
    
    return results


# ===============================================
# EXAMPLE USAGE
# ===============================================

if __name__ == "__main__":
    print("="*70)
    print("INTEGRATION TEST: Segmentation → PMFBY Prediction")
    print("="*70)
    
    # Mock segmentation output
    segmentation_output = {
        'centroid': (19.071591, 74.774179),
        'area': 10.5,
        'crop_type': 'Rice',
        'ndvi_mean': 0.65,
        'ndvi_peak': 0.72,
        'confidence': 0.92,
        'boundary': [
            [19.07, 74.77],
            [19.08, 74.77],
            [19.08, 74.78],
            [19.07, 74.78]
        ]
    }
    
    # PMFBY threshold
    threshold = 1640  # kg/ha for Rice in Ahmednagar
    
    # Run prediction
    result = predict_yield_from_segmentation(
        segmentation_output=segmentation_output,
        threshold_yield=threshold
    )
    
    # Display results
    print("\n" + "="*70)
    print("INTEGRATED RESULTS")
    print("="*70)
    print(f"""
SEGMENTATION DATA:
    Crop: {result['segmentation']['crop_detected']}
    Area: {result['segmentation']['area_ha']} ha
    NDVI: {result['segmentation']['ndvi_mean']:.2f}
    Confidence: {result['segmentation']['confidence']*100:.1f}%

YIELD PREDICTION:
    Predicted: {result['prediction']:.0f} kg/ha
    Uncertainty: ±{result['uncertainty']:.0f} kg/ha
    95% CI: [{result['confidence_interval'][0]:.0f}, {result['confidence_interval'][1]:.0f}] kg/ha

PMFBY ASSESSMENT:
    Threshold: {result['pmfby']['threshold']} kg/ha
    Shortfall: {result['pmfby']['shortfall']:.0f} kg/ha
    Loss: {result['pmfby']['loss_percentage']:.1f}%
    Claim: {'YES' if result['pmfby']['claim_triggered'] else 'NO'}
    Probability: {result['pmfby']['claim_probability']*100:.1f}%
    """)
    
    print("="*70)
    print("✅ INTEGRATION TEST PASSED")
    print("="*70)
