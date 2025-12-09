# Google Earth Engine Setup Guide

## Prerequisites

1. **Google Account** - You need a Google account
2. **Earth Engine Access** - Sign up at https://earthengine.google.com/signup/

## Installation Steps

### Step 1: Install Earth Engine Python API

```bash
pip install earthengine-api
```

### Step 2: Authenticate

```bash
earthengine authenticate
```

This will:
1. Open a browser window
2. Ask you to sign in with your Google account
3. Generate an authentication token
4. Save credentials to your system

### Step 3: Verify Installation

```python
import ee
ee.Initialize()
print("Earth Engine initialized successfully!")
```

## Quick Test

```bash
cd d:\tr\pmfby_engine
python feature_engineering/satellite_features.py
```

## Troubleshooting

### Error: "Earth Engine not initialized"
**Solution:** Run `earthengine authenticate` first

### Error: "Project not found"
**Solution:** 
```python
ee.Initialize(project='your-project-id')
```

### Error: "Quota exceeded"
**Solution:** Earth Engine has usage limits. Wait or upgrade to paid tier.

## Alternative: Use MODIS (No Authentication)

If you can't authenticate, the code will fall back to default values.
Or use MODIS which has more lenient quotas:

```python
features = fetch_satellite_features(
    lat, lon, year, season,
    use_sentinel=False  # Use MODIS instead
)
```

## Usage in Pipeline

```python
from feature_engineering.satellite_features import fetch_satellite_features

# Fetch NDVI for a farm
satellite = fetch_satellite_features(
    lat=19.071591,
    lon=74.774179,
    year=2024,
    season='Kharif'
)

print(f"NDVI Mean: {satellite['ndvi_mean']:.2f}")
print(f"NDVI Peak: {satellite['ndvi_peak']:.2f}")
```
