# Integration Guide: Segmentation Model + PMFBY Yield Prediction

This guide explains how to integrate the PMFBY Yield Prediction System with your existing crop segmentation model.

---

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    INTEGRATED PIPELINE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. SEGMENTATION MODEL (Your Existing System)                   │
│     Input: Satellite Image                                      │
│     Output: {                                                   │
│         'boundary': polygon coordinates,                        │
│         'centroid': (lat, lon),                                 │
│         'area': hectares,                                       │
│         'crop_type': 'Rice',                                    │
│         'ndvi_mean': 0.65,                                      │
│         'confidence': 0.92                                      │
│     }                                                            │
│                                                                  │
│                          ▼                                       │
│                                                                  │
│  2. PMFBY YIELD PREDICTION (This System)                        │
│     Input: Segmentation output + threshold                      │
│     Process:                                                    │
│       - Fetch weather for centroid location                     │
│       - Compute agronomic stress indices                        │
│       - Predict yield with uncertainty                          │
│     Output: {                                                   │
│         'predicted_yield': 1637 kg/ha,                          │
│         'uncertainty': ±662 kg/ha,                              │
│         'claim_triggered': False,                               │
│         'claim_probability': 20.8%                              │
│     }                                                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Integration Steps

### Step 1: Install PMFBY System

On your device with the segmentation model:

```bash
# Clone the PMFBY repository
git clone https://github.com/parag005-ai/PMFBY.git

# Install dependencies
cd PMFBY
pip install -r requirements.txt
```

### Step 2: Create Integration Module

Create a file `integration.py` in your project:

```python
"""
Integration module connecting Segmentation Model → PMFBY Prediction
"""

import sys
sys.path.append('./PMFBY')  # Add PMFBY to path

from predict_unified import run_unified_prediction

def predict_yield_from_segmentation(segmentation_output, threshold_yield):
    """
    Takes segmentation model output and predicts crop yield.
    
    Args:
        segmentation_output: Dict from your segmentation model with:
            - 'centroid': (lat, lon) - Farm center coordinates
            - 'area': float - Farm area in hectares
            - 'crop_type': str - Detected crop (e.g., 'Rice', 'Cotton')
            - 'ndvi_mean': float (optional) - Average NDVI
            - 'boundary': polygon (optional) - Farm boundary
        
        threshold_yield: float - PMFBY threshold in kg/ha
    
    Returns:
        Dict with yield prediction and PMFBY assessment
    """
    
    # Extract data from segmentation
    lat, lon = segmentation_output['centroid']
    area = segmentation_output['area']
    crop = segmentation_output['crop_type']
    
    # Determine district (you can add a lat/lon → district lookup)
    district = find_district(lat, lon)  # Implement this
    
    # Determine season based on current date
    import datetime
    month = datetime.datetime.now().month
    if 6 <= month <= 11:
        season = 'Kharif'
    elif 11 <= month or month <= 3:
        season = 'Rabi'
    else:
        season = 'Summer'
    
    # Run PMFBY prediction
    result = run_unified_prediction(
        lat=lat,
        lon=lon,
        district=district,
        crop=crop,
        season=season,
        area=area,
        threshold=threshold_yield
    )
    
    # Add segmentation data to result
    result['segmentation'] = {
        'boundary': segmentation_output.get('boundary'),
        'ndvi': segmentation_output.get('ndvi_mean'),
        'confidence': segmentation_output.get('confidence')
    }
    
    return result


def find_district(lat, lon):
    """
    Map coordinates to district name.
    You can use reverse geocoding or a lookup table.
    """
    # Simple lookup for Maharashtra districts
    districts = {
        (19.0, 74.7): 'Ahmednagar',
        (18.5, 73.8): 'Pune',
        (20.0, 73.7): 'Nashik',
        # Add more...
    }
    
    # Find nearest district (simple approach)
    min_dist = float('inf')
    nearest = 'Ahmednagar'  # Default
    
    for coords, district in districts.items():
        dist = ((lat - coords[0])**2 + (lon - coords[1])**2)**0.5
        if dist < min_dist:
            min_dist = dist
            nearest = district
    
    return nearest
```

### Step 3: Use in Your Pipeline

```python
# In your main segmentation pipeline

# Step 1: Run segmentation (your existing code)
segmentation_result = your_segmentation_model.predict(satellite_image)
# Output: {'centroid': (19.07, 74.77), 'area': 10, 'crop_type': 'Rice', ...}

# Step 2: Get PMFBY threshold (from database or user input)
threshold = 1640  # kg/ha for Rice in Ahmednagar

# Step 3: Predict yield
from integration import predict_yield_from_segmentation

yield_result = predict_yield_from_segmentation(
    segmentation_output=segmentation_result,
    threshold_yield=threshold
)

# Step 4: Display results
print(f"Farm Location: {yield_result['segmentation']['boundary']}")
print(f"Predicted Yield: {yield_result['prediction']:.0f} kg/ha")
print(f"Uncertainty: ±{yield_result['uncertainty']:.0f} kg/ha")
print(f"Claim Triggered: {yield_result['pmfby']['claim_triggered']}")
print(f"Claim Probability: {yield_result['pmfby']['claim_probability']*100:.1f}%")
```

---

## Expected Input Format

Your segmentation model should output:

```python
{
    'centroid': (19.071591, 74.774179),      # (lat, lon) - Required
    'area': 10.5,                             # hectares - Required
    'crop_type': 'Rice',                      # Crop name - Required
    'boundary': [[lat1,lon1], [lat2,lon2]], # Optional
    'ndvi_mean': 0.65,                        # Optional (can improve accuracy)
    'ndvi_peak': 0.72,                        # Optional
    'confidence': 0.92                        # Optional
}
```

---

## Expected Output Format

The integrated system returns:

```python
{
    'model_r2': 0.818,
    'prediction': 1637.0,                    # kg/ha
    'uncertainty': 662.0,                    # kg/ha
    'confidence_interval': (340, 2933),      # kg/ha
    
    'weather': {
        'rain_total': 991,
        'gdd': 4232,
        'heat_stress_intensity': 319,
        ...
    },
    
    'stress': {
        'vegetative_stress': 0.29,
        'flowering_heat_stress': 1.0,
        'combined_stress': 0.44,
        'yield_potential': 0.58
    },
    
    'pmfby': {
        'threshold': 1640,
        'predicted_yield': 1637,
        'shortfall': 3,
        'loss_percentage': 0.2,
        'claim_triggered': False,
        'claim_probability': 0.208,
        'decision_confidence': 0.584
    },
    
    'segmentation': {
        'boundary': [...],
        'ndvi': 0.65,
        'confidence': 0.92
    }
}
```

---

## Folder Structure

```
your_project/
├── segmentation_model/          # Your existing model
│   ├── model.py
│   ├── weights/
│   └── ...
│
├── PMFBY/                        # Cloned from GitHub
│   ├── predict_unified.py
│   ├── feature_engineering/
│   ├── models/trained/
│   └── ...
│
├── integration.py                # Integration code (create this)
├── main.py                       # Your main pipeline
└── requirements.txt              # Combined dependencies
```

---

## Complete Example

```python
# main.py - Complete integrated pipeline

from segmentation_model import SegmentationModel
from integration import predict_yield_from_segmentation

# Initialize your segmentation model
seg_model = SegmentationModel()

# Load satellite image
image = load_satellite_image('farm_image.tif')

# Step 1: Segment the farm
print("Running segmentation...")
seg_result = seg_model.predict(image)

print(f"Detected Crop: {seg_result['crop_type']}")
print(f"Farm Area: {seg_result['area']} ha")
print(f"Location: {seg_result['centroid']}")

# Step 2: Predict yield
print("\nPredicting yield...")
threshold = 1640  # Get from database based on crop/district

yield_result = predict_yield_from_segmentation(
    segmentation_output=seg_result,
    threshold_yield=threshold
)

# Step 3: Display results
print("\n" + "="*70)
print("INTEGRATED RESULTS")
print("="*70)
print(f"""
SEGMENTATION:
  Crop: {seg_result['crop_type']}
  Area: {seg_result['area']} ha
  NDVI: {seg_result.get('ndvi_mean', 'N/A')}
  Confidence: {seg_result.get('confidence', 'N/A')*100:.1f}%

YIELD PREDICTION:
  Predicted: {yield_result['prediction']:.0f} kg/ha
  Uncertainty: ±{yield_result['uncertainty']:.0f} kg/ha
  95% CI: [{yield_result['confidence_interval'][0]:.0f}, 
           {yield_result['confidence_interval'][1]:.0f}] kg/ha

PMFBY ASSESSMENT:
  Threshold: {yield_result['pmfby']['threshold']} kg/ha
  Loss: {yield_result['pmfby']['loss_percentage']:.1f}%
  Claim: {'YES' if yield_result['pmfby']['claim_triggered'] else 'NO'}
  Probability: {yield_result['pmfby']['claim_probability']*100:.1f}%
""")
```

---

## Advanced: Using Segmentation NDVI

To improve accuracy, you can pass NDVI from your segmentation:

```python
# Modify predict_unified.py to accept NDVI

def run_unified_prediction(lat, lon, district, crop, season, area, threshold, 
                           ndvi_from_segmentation=None):
    # ... existing code ...
    
    # If NDVI provided from segmentation, use it to adjust prediction
    if ndvi_from_segmentation is not None:
        # NDVI adjustment factor
        ndvi_factor = (ndvi_from_segmentation - 0.5) * 500  # kg/ha
        pred += ndvi_factor
    
    # ... rest of code ...
```

---

## Testing the Integration

```python
# test_integration.py

# Mock segmentation output for testing
test_segmentation = {
    'centroid': (19.071591, 74.774179),
    'area': 10,
    'crop_type': 'Rice',
    'ndvi_mean': 0.65,
    'confidence': 0.92
}

# Test prediction
result = predict_yield_from_segmentation(
    segmentation_output=test_segmentation,
    threshold_yield=1640
)

assert 'prediction' in result
assert 'pmfby' in result
assert result['prediction'] > 0

print("✅ Integration test passed!")
```

---

## Troubleshooting

### Issue: Module not found
```python
# Add PMFBY to Python path
import sys
sys.path.append('/path/to/PMFBY')
```

### Issue: Slow weather API
- First run takes 30-60 seconds (fetching weather)
- Subsequent runs are instant (cached)
- Cache location: `PMFBY/data/weather_cache/`

### Issue: District not found
- Add your districts to the `find_district()` function
- Or use reverse geocoding API (Google Maps, Nominatim)

---

## Next Steps

1. ✅ Clone PMFBY repo
2. ✅ Create `integration.py`
3. ✅ Test with mock data
4. ✅ Integrate into your pipeline
5. ⬜ Add district lookup
6. ⬜ Optimize for batch processing
7. ⬜ Add NDVI integration for higher accuracy

---

## Support

For issues or questions:
- GitHub: https://github.com/parag005-ai/PMFBY
- Check `README.md` for API documentation
