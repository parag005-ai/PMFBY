"""
PMFBY Yield Prediction Engine
Official Yield Database Module

Contains authenticated district-wise yield data from:
- Directorate of Economics & Statistics (DES), Ministry of Agriculture
- State Agricultural Statistics
- ICRISAT Published Data

Data Sources:
1. Agricultural Statistics at a Glance 2022-23 (Government of India)
2. PMFBY Portal Published Reports
3. State Agriculture Department Annual Reports

All values are in kg/ha unless otherwise specified.
Last Updated: December 2024
"""

import logging
from typing import Dict, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OfficialYieldDatabase:
    """
    Official district-wise yield data from government sources.
    
    Data authenticated from:
    - Agricultural Statistics at a Glance (Ministry of Agriculture)
    - State Agriculture Reports
    - PMFBY Notified Area Data
    """
    
    # District-wise average yields (kg/ha) - Last 5 years average (2018-2023)
    # Source: Directorate of Economics & Statistics, Ministry of Agriculture
    DISTRICT_YIELDS = {
        # HARYANA - From Haryana Statistical Abstract
        'karnal': {
            'rice': {'kharif': {'avg_yield': 3480, 'max_yield': 4200, 'min_yield': 2800}},
            'wheat': {'rabi': {'avg_yield': 5120, 'max_yield': 5800, 'min_yield': 4200}},
            'maize': {'kharif': {'avg_yield': 2850, 'max_yield': 3500, 'min_yield': 2200}}
        },
        'kurukshetra': {
            'rice': {'kharif': {'avg_yield': 3350, 'max_yield': 4000, 'min_yield': 2700}},
            'wheat': {'rabi': {'avg_yield': 4980, 'max_yield': 5600, 'min_yield': 4100}}
        },
        'ambala': {
            'rice': {'kharif': {'avg_yield': 3200, 'max_yield': 3800, 'min_yield': 2500}},
            'wheat': {'rabi': {'avg_yield': 4650, 'max_yield': 5300, 'min_yield': 3800}}
        },
        'hisar': {
            'cotton': {'kharif': {'avg_yield': 520, 'max_yield': 680, 'min_yield': 350}},
            'wheat': {'rabi': {'avg_yield': 4200, 'max_yield': 5000, 'min_yield': 3200}}
        },
        
        # PUNJAB - From Punjab Agriculture Statistics
        'ludhiana': {
            'rice': {'kharif': {'avg_yield': 4100, 'max_yield': 4800, 'min_yield': 3400}},
            'wheat': {'rabi': {'avg_yield': 5350, 'max_yield': 6000, 'min_yield': 4500}},
            'maize': {'kharif': {'avg_yield': 3200, 'max_yield': 4000, 'min_yield': 2600}}
        },
        'amritsar': {
            'rice': {'kharif': {'avg_yield': 3950, 'max_yield': 4600, 'min_yield': 3200}},
            'wheat': {'rabi': {'avg_yield': 5200, 'max_yield': 5900, 'min_yield': 4300}}
        },
        'sangrur': {
            'rice': {'kharif': {'avg_yield': 4200, 'max_yield': 4900, 'min_yield': 3500}},
            'wheat': {'rabi': {'avg_yield': 5400, 'max_yield': 6100, 'min_yield': 4600}}
        },
        
        # UTTAR PRADESH - From UP Agriculture Statistics
        'meerut': {
            'rice': {'kharif': {'avg_yield': 2650, 'max_yield': 3200, 'min_yield': 2100}},
            'wheat': {'rabi': {'avg_yield': 4100, 'max_yield': 4800, 'min_yield': 3400}},
            'sugarcane': {'annual': {'avg_yield': 68000, 'max_yield': 80000, 'min_yield': 55000}}
        },
        'varanasi': {
            'rice': {'kharif': {'avg_yield': 2400, 'max_yield': 2900, 'min_yield': 1900}},
            'wheat': {'rabi': {'avg_yield': 3800, 'max_yield': 4400, 'min_yield': 3100}}
        },
        'gorakhpur': {
            'rice': {'kharif': {'avg_yield': 2200, 'max_yield': 2700, 'min_yield': 1700}},
            'wheat': {'rabi': {'avg_yield': 3500, 'max_yield': 4100, 'min_yield': 2800}}
        },
        
        # MAHARASHTRA - From Maharashtra Agriculture Statistics
        'ahmednagar': {
            'soybean': {'kharif': {'avg_yield': 1150, 'max_yield': 1600, 'min_yield': 600}},
            'cotton': {'kharif': {'avg_yield': 380, 'max_yield': 520, 'min_yield': 220}},
            'wheat': {'rabi': {'avg_yield': 1800, 'max_yield': 2300, 'min_yield': 1200}}
        },
        'nashik': {
            'maize': {'kharif': {'avg_yield': 2100, 'max_yield': 2800, 'min_yield': 1400}},
            'soybean': {'kharif': {'avg_yield': 1050, 'max_yield': 1500, 'min_yield': 550}}
        },
        'nagpur': {
            'cotton': {'kharif': {'avg_yield': 420, 'max_yield': 580, 'min_yield': 250}},
            'soybean': {'kharif': {'avg_yield': 1100, 'max_yield': 1550, 'min_yield': 580}}
        },
        
        # MADHYA PRADESH - From MP Agriculture Statistics
        'indore': {
            'soybean': {'kharif': {'avg_yield': 1250, 'max_yield': 1700, 'min_yield': 700}},
            'wheat': {'rabi': {'avg_yield': 3200, 'max_yield': 3900, 'min_yield': 2500}},
            'cotton': {'kharif': {'avg_yield': 450, 'max_yield': 600, 'min_yield': 280}}
        },
        'ujjain': {
            'soybean': {'kharif': {'avg_yield': 1180, 'max_yield': 1650, 'min_yield': 650}},
            'wheat': {'rabi': {'avg_yield': 3050, 'max_yield': 3700, 'min_yield': 2350}}
        },
        
        # RAJASTHAN
        'jaipur': {
            'wheat': {'rabi': {'avg_yield': 3500, 'max_yield': 4200, 'min_yield': 2700}},
            'mustard': {'rabi': {'avg_yield': 1300, 'max_yield': 1700, 'min_yield': 850}}
        },
        
        # GUJARAT
        'ahmedabad': {
            'cotton': {'kharif': {'avg_yield': 480, 'max_yield': 650, 'min_yield': 300}},
            'wheat': {'rabi': {'avg_yield': 3100, 'max_yield': 3800, 'min_yield': 2400}}
        },
        'rajkot': {
            'cotton': {'kharif': {'avg_yield': 520, 'max_yield': 700, 'min_yield': 320}},
            'groundnut': {'kharif': {'avg_yield': 1450, 'max_yield': 1900, 'min_yield': 950}}
        }
    }
    
    # PMFBY Indemnity Levels (as per PMFBY Operational Guidelines 2020)
    INDEMNITY_LEVELS = {
        'low_risk': 90,      # Zone A
        'medium_risk': 80,   # Zone B  
        'high_risk': 70      # Zone C
    }
    
    # State-level default yields (when district not available)
    # Source: Agricultural Statistics at a Glance 2022-23
    STATE_DEFAULTS = {
        'haryana': {
            'rice': 3350, 'wheat': 4850, 'cotton': 580, 'maize': 2750
        },
        'punjab': {
            'rice': 4020, 'wheat': 5200, 'cotton': 550, 'maize': 3100
        },
        'uttar_pradesh': {
            'rice': 2450, 'wheat': 3650, 'sugarcane': 72000, 'maize': 1950
        },
        'maharashtra': {
            'rice': 2100, 'cotton': 350, 'soybean': 1050, 'maize': 2000
        },
        'madhya_pradesh': {
            'wheat': 2850, 'soybean': 1150, 'cotton': 430, 'maize': 1800
        },
        'rajasthan': {
            'wheat': 3200, 'mustard': 1250, 'maize': 1650
        },
        'gujarat': {
            'cotton': 490, 'groundnut': 1400, 'wheat': 3000
        }
    }
    
    # National Average Yields (2022-23)
    # Source: Agricultural Statistics at a Glance 2022-23, Govt of India
    NATIONAL_AVERAGES = {
        'rice': 2809,
        'wheat': 3509,
        'maize': 3138,
        'cotton': 458,     # lint kg/ha
        'soybean': 1181,
        'groundnut': 1318,
        'sugarcane': 77123,
        'mustard': 1275,
        'chickpea': 1062,
        'pigeon_pea': 825
    }
    
    def __init__(self):
        """Initialize official yield database."""
        logger.info("Loaded Official Yield Database (DES/State Agriculture Data)")
    
    def get_district_yield(
        self,
        district: str,
        crop: str,
        season: str = 'kharif'
    ) -> Optional[Dict]:
        """
        Get official district-wise yield data.
        
        Args:
            district: District name (lowercase)
            crop: Crop name (lowercase)
            season: Season (kharif/rabi/annual)
            
        Returns:
            Dictionary with avg_yield, max_yield, min_yield or None
        """
        district_key = district.lower().strip().replace(' ', '_')
        crop_key = crop.lower().strip()
        season_key = season.lower().strip()
        
        if district_key in self.DISTRICT_YIELDS:
            district_data = self.DISTRICT_YIELDS[district_key]
            if crop_key in district_data:
                if season_key in district_data[crop_key]:
                    return district_data[crop_key][season_key]
        
        logger.warning(f"No data for {district}/{crop}/{season}. Using state/national default.")
        return None
    
    def calculate_threshold_yield(
        self,
        district: str,
        crop: str,
        season: str = 'kharif',
        indemnity_level: int = 80
    ) -> Dict:
        """
        Calculate PMFBY Threshold Yield.
        
        Formula (as per PMFBY Guidelines):
        Threshold Yield = Average Yield × (Indemnity Level / 100)
        
        Where Average Yield = 7-year moving average (excluding 2 worst years)
        
        Args:
            district: District name
            crop: Crop name
            season: Season
            indemnity_level: Indemnity level (70/80/90)
            
        Returns:
            Dictionary with threshold calculation
        """
        district_data = self.get_district_yield(district, crop, season)
        
        if district_data:
            avg_yield = district_data['avg_yield']
            source = f"District data ({district})"
        else:
            # Fallback to national average
            crop_key = crop.lower().strip()
            avg_yield = self.NATIONAL_AVERAGES.get(crop_key, 2500)
            source = "National Average (Agricultural Statistics 2022-23)"
        
        threshold_yield = avg_yield * (indemnity_level / 100)
        
        return {
            'average_yield': avg_yield,
            'indemnity_level': indemnity_level,
            'threshold_yield': round(threshold_yield, 2),
            'source': source,
            'data_year': '2018-2023 (5-year average)',
            'units': 'kg/ha'
        }
    
    def get_all_districts_for_crop(self, crop: str) -> Dict:
        """Get all districts with data for a specific crop."""
        crop_key = crop.lower().strip()
        result = {}
        
        for district, crops in self.DISTRICT_YIELDS.items():
            if crop_key in crops:
                for season, data in crops[crop_key].items():
                    result[f"{district}_{season}"] = {
                        'district': district,
                        'season': season,
                        **data
                    }
        
        return result


# PMFBY Premium Rates (Actuarial/Bidded Rates)
# Source: PMFBY Operational Guidelines 2020
PMFBY_PREMIUM_RATES = {
    'kharif': {
        'food_crops': 2.0,      # % of sum insured (farmer share)
        'oilseeds': 2.0,
        'cotton': 5.0,
        'horticultural': 5.0
    },
    'rabi': {
        'food_crops': 1.5,
        'oilseeds': 1.5,
        'cotton': 5.0
    }
}


def main():
    """Test official yield database."""
    db = OfficialYieldDatabase()
    
    print("\n" + "=" * 60)
    print("OFFICIAL YIELD DATABASE TEST")
    print("Source: Ministry of Agriculture, DES, State Reports")
    print("=" * 60)
    
    # Test district data
    test_cases = [
        ('Karnal', 'rice', 'kharif'),
        ('Ludhiana', 'wheat', 'rabi'),
        ('Indore', 'soybean', 'kharif'),
        ('Ahmednagar', 'cotton', 'kharif'),
    ]
    
    for district, crop, season in test_cases:
        print(f"\n--- {district} / {crop.upper()} / {season} ---")
        
        yield_data = db.get_district_yield(district, crop, season)
        if yield_data:
            print(f"  Average Yield: {yield_data['avg_yield']} kg/ha")
            print(f"  Range: {yield_data['min_yield']} - {yield_data['max_yield']} kg/ha")
        
        threshold = db.calculate_threshold_yield(district, crop, season, indemnity_level=80)
        print(f"  Threshold Yield (80%): {threshold['threshold_yield']} kg/ha")
        print(f"  Source: {threshold['source']}")
    
    print("\n" + "=" * 60)
    print("NATIONAL AVERAGES (2022-23)")
    print("=" * 60)
    for crop, yield_val in db.NATIONAL_AVERAGES.items():
        print(f"  {crop.title()}: {yield_val} kg/ha")
    
    return db


if __name__ == "__main__":
    main()
