"""
PMFBY Yield Prediction Engine
Soil Data Fetcher Module

Provides soil data from ICAR district-level profiles.
Includes soil type, pH, nutrients, and crop suitability assessment.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SoilFetcher:
    """
    Soil data provider with ICAR district-level data.
    
    Features:
    - District-level soil profiles
    - Crop-specific suitability assessment
    - Nutrient and pH analysis
    - Water holding capacity estimation
    """
    
    # ICAR Soil Database (district-level averages)
    SOIL_DATABASE = {
        # Haryana
        "karnal": {
            "state": "Haryana",
            "soil_type": "Alluvial",
            "texture": "Loamy",
            "ph": 7.8,
            "ec_ds_m": 0.3,
            "organic_carbon_pct": 0.52,
            "nitrogen_kg_ha": 185,
            "phosphorus_kg_ha": 22,
            "potassium_kg_ha": 280,
            "water_holding_pct": 45,
            "drainage": "good",
            "depth_cm": 150,
            "cec_meq_100g": 12.5
        },
        "kurukshetra": {
            "state": "Haryana",
            "soil_type": "Alluvial",
            "texture": "Loamy to Clay Loam",
            "ph": 7.9,
            "ec_ds_m": 0.35,
            "organic_carbon_pct": 0.48,
            "nitrogen_kg_ha": 175,
            "phosphorus_kg_ha": 20,
            "potassium_kg_ha": 260,
            "water_holding_pct": 48,
            "drainage": "good",
            "depth_cm": 140,
            "cec_meq_100g": 13.0
        },
        "hisar": {
            "state": "Haryana",
            "soil_type": "Alluvial (semi-arid)",
            "texture": "Sandy Loam",
            "ph": 8.2,
            "ec_ds_m": 0.5,
            "organic_carbon_pct": 0.35,
            "nitrogen_kg_ha": 150,
            "phosphorus_kg_ha": 18,
            "potassium_kg_ha": 220,
            "water_holding_pct": 35,
            "drainage": "moderate",
            "depth_cm": 120,
            "cec_meq_100g": 10.0
        },
        # Punjab
        "ludhiana": {
            "state": "Punjab",
            "soil_type": "Alluvial",
            "texture": "Loamy",
            "ph": 7.6,
            "ec_ds_m": 0.25,
            "organic_carbon_pct": 0.55,
            "nitrogen_kg_ha": 195,
            "phosphorus_kg_ha": 25,
            "potassium_kg_ha": 290,
            "water_holding_pct": 46,
            "drainage": "good",
            "depth_cm": 160,
            "cec_meq_100g": 13.5
        },
        "amritsar": {
            "state": "Punjab",
            "soil_type": "Alluvial",
            "texture": "Loamy to Clay Loam",
            "ph": 7.7,
            "ec_ds_m": 0.28,
            "organic_carbon_pct": 0.50,
            "nitrogen_kg_ha": 185,
            "phosphorus_kg_ha": 23,
            "potassium_kg_ha": 275,
            "water_holding_pct": 47,
            "drainage": "good",
            "depth_cm": 155,
            "cec_meq_100g": 12.8
        },
        # Maharashtra
        "ahmednagar": {
            "state": "Maharashtra",
            "soil_type": "Black Cotton (Vertisol)",
            "texture": "Clay",
            "ph": 7.9,
            "ec_ds_m": 0.4,
            "organic_carbon_pct": 0.65,
            "nitrogen_kg_ha": 210,
            "phosphorus_kg_ha": 15,
            "potassium_kg_ha": 350,
            "water_holding_pct": 55,
            "drainage": "poor",
            "depth_cm": 100,
            "cec_meq_100g": 35.0
        },
        "nashik": {
            "state": "Maharashtra",
            "soil_type": "Red Lateritic",
            "texture": "Sandy Clay Loam",
            "ph": 6.5,
            "ec_ds_m": 0.2,
            "organic_carbon_pct": 0.45,
            "nitrogen_kg_ha": 165,
            "phosphorus_kg_ha": 12,
            "potassium_kg_ha": 180,
            "water_holding_pct": 38,
            "drainage": "good",
            "depth_cm": 80,
            "cec_meq_100g": 8.5
        },
        "nagpur": {
            "state": "Maharashtra",
            "soil_type": "Black Cotton (Vertisol)",
            "texture": "Heavy Clay",
            "ph": 8.0,
            "ec_ds_m": 0.45,
            "organic_carbon_pct": 0.70,
            "nitrogen_kg_ha": 220,
            "phosphorus_kg_ha": 14,
            "potassium_kg_ha": 380,
            "water_holding_pct": 58,
            "drainage": "poor",
            "depth_cm": 110,
            "cec_meq_100g": 38.0
        },
        # Madhya Pradesh
        "indore": {
            "state": "Madhya Pradesh",
            "soil_type": "Black Cotton (Vertisol)",
            "texture": "Clay",
            "ph": 7.8,
            "ec_ds_m": 0.35,
            "organic_carbon_pct": 0.60,
            "nitrogen_kg_ha": 200,
            "phosphorus_kg_ha": 16,
            "potassium_kg_ha": 340,
            "water_holding_pct": 52,
            "drainage": "moderate",
            "depth_cm": 95,
            "cec_meq_100g": 32.0
        },
        "ujjain": {
            "state": "Madhya Pradesh",
            "soil_type": "Black Cotton (Vertisol)",
            "texture": "Clay Loam",
            "ph": 7.7,
            "ec_ds_m": 0.32,
            "organic_carbon_pct": 0.58,
            "nitrogen_kg_ha": 195,
            "phosphorus_kg_ha": 17,
            "potassium_kg_ha": 320,
            "water_holding_pct": 50,
            "drainage": "moderate",
            "depth_cm": 90,
            "cec_meq_100g": 30.0
        },
        # Uttar Pradesh
        "meerut": {
            "state": "Uttar Pradesh",
            "soil_type": "Alluvial",
            "texture": "Loamy",
            "ph": 7.5,
            "ec_ds_m": 0.25,
            "organic_carbon_pct": 0.50,
            "nitrogen_kg_ha": 180,
            "phosphorus_kg_ha": 20,
            "potassium_kg_ha": 265,
            "water_holding_pct": 44,
            "drainage": "good",
            "depth_cm": 145,
            "cec_meq_100g": 12.0
        },
        "varanasi": {
            "state": "Uttar Pradesh",
            "soil_type": "Alluvial (Gangetic)",
            "texture": "Silty Clay Loam",
            "ph": 7.8,
            "ec_ds_m": 0.3,
            "organic_carbon_pct": 0.55,
            "nitrogen_kg_ha": 190,
            "phosphorus_kg_ha": 22,
            "potassium_kg_ha": 280,
            "water_holding_pct": 50,
            "drainage": "moderate",
            "depth_cm": 130,
            "cec_meq_100g": 14.0
        },
        # Rajasthan
        "jaipur": {
            "state": "Rajasthan",
            "soil_type": "Aridisol (Desert)",
            "texture": "Sandy Loam",
            "ph": 8.3,
            "ec_ds_m": 0.6,
            "organic_carbon_pct": 0.25,
            "nitrogen_kg_ha": 120,
            "phosphorus_kg_ha": 12,
            "potassium_kg_ha": 180,
            "water_holding_pct": 28,
            "drainage": "excessive",
            "depth_cm": 100,
            "cec_meq_100g": 7.5
        },
        # Gujarat
        "ahmedabad": {
            "state": "Gujarat",
            "soil_type": "Alluvial to Black",
            "texture": "Clay Loam",
            "ph": 7.9,
            "ec_ds_m": 0.4,
            "organic_carbon_pct": 0.45,
            "nitrogen_kg_ha": 160,
            "phosphorus_kg_ha": 18,
            "potassium_kg_ha": 240,
            "water_holding_pct": 42,
            "drainage": "moderate",
            "depth_cm": 110,
            "cec_meq_100g": 15.0
        },
        "rajkot": {
            "state": "Gujarat",
            "soil_type": "Black Cotton",
            "texture": "Clay",
            "ph": 8.0,
            "ec_ds_m": 0.5,
            "organic_carbon_pct": 0.55,
            "nitrogen_kg_ha": 175,
            "phosphorus_kg_ha": 15,
            "potassium_kg_ha": 300,
            "water_holding_pct": 48,
            "drainage": "poor",
            "depth_cm": 90,
            "cec_meq_100g": 28.0
        },
        # Default
        "default": {
            "state": "Unknown",
            "soil_type": "Mixed Alluvial",
            "texture": "Loam",
            "ph": 7.5,
            "ec_ds_m": 0.3,
            "organic_carbon_pct": 0.45,
            "nitrogen_kg_ha": 170,
            "phosphorus_kg_ha": 18,
            "potassium_kg_ha": 250,
            "water_holding_pct": 42,
            "drainage": "moderate",
            "depth_cm": 120,
            "cec_meq_100g": 12.0
        }
    }
    
    # Crop-specific soil requirements
    CROP_REQUIREMENTS = {
        "rice": {
            "ph_range": (5.5, 7.5),
            "preferred_texture": ["Clay", "Clay Loam", "Silty Clay Loam"],
            "drainage_preference": "poor",  # Rice prefers waterlogged
            "min_water_holding": 45,
            "min_organic_carbon": 0.4,
            "critical_nutrients": ["nitrogen", "phosphorus"]
        },
        "wheat": {
            "ph_range": (6.0, 8.0),
            "preferred_texture": ["Loam", "Clay Loam", "Loamy"],
            "drainage_preference": "good",
            "min_water_holding": 35,
            "min_organic_carbon": 0.35,
            "critical_nutrients": ["nitrogen", "phosphorus", "potassium"]
        },
        "cotton": {
            "ph_range": (6.0, 8.0),
            "preferred_texture": ["Clay", "Black Cotton", "Clay Loam"],
            "drainage_preference": "moderate",
            "min_water_holding": 40,
            "min_organic_carbon": 0.4,
            "critical_nutrients": ["nitrogen", "potassium"]
        },
        "soybean": {
            "ph_range": (6.0, 7.5),
            "preferred_texture": ["Loam", "Clay Loam", "Sandy Loam"],
            "drainage_preference": "good",
            "min_water_holding": 35,
            "min_organic_carbon": 0.4,
            "critical_nutrients": ["phosphorus", "potassium"]
        },
        "maize": {
            "ph_range": (5.8, 7.5),
            "preferred_texture": ["Loam", "Sandy Loam", "Clay Loam"],
            "drainage_preference": "good",
            "min_water_holding": 35,
            "min_organic_carbon": 0.35,
            "critical_nutrients": ["nitrogen", "phosphorus", "potassium"]
        }
    }
    
    def __init__(self):
        """Initialize soil fetcher."""
        logger.info("Soil data fetcher initialized with ICAR database")
    
    def get_soil_profile(self, district: str) -> Dict:
        """
        Get soil profile for a district.
        
        Args:
            district: District name (case insensitive)
            
        Returns:
            Dictionary with soil properties
        """
        district_key = district.lower().strip()
        
        if district_key in self.SOIL_DATABASE:
            profile = self.SOIL_DATABASE[district_key].copy()
        else:
            logger.warning(f"District '{district}' not found. Using default profile.")
            profile = self.SOIL_DATABASE["default"].copy()
        
        profile['district'] = district
        return profile
    
    def assess_suitability(
        self,
        district: str,
        crop: str
    ) -> Dict:
        """
        Assess soil suitability for a specific crop.
        
        Args:
            district: District name
            crop: Crop name (rice, wheat, cotton, soybean, maize)
            
        Returns:
            Dictionary with suitability assessment
        """
        soil = self.get_soil_profile(district)
        crop_key = crop.lower().strip()
        
        if crop_key not in self.CROP_REQUIREMENTS:
            logger.warning(f"Crop '{crop}' not in requirements database. Using generic assessment.")
            requirements = self.CROP_REQUIREMENTS.get("wheat")  # Default to wheat
        else:
            requirements = self.CROP_REQUIREMENTS[crop_key]
        
        # Assess each factor
        limitations = []
        recommendations = []
        score = 100
        
        # pH assessment
        ph_min, ph_max = requirements['ph_range']
        if soil['ph'] < ph_min:
            score -= 15
            limitations.append(f"Soil too acidic (pH {soil['ph']:.1f} < {ph_min})")
            recommendations.append("Apply lime to increase pH")
        elif soil['ph'] > ph_max:
            score -= 10
            limitations.append(f"Soil too alkaline (pH {soil['ph']:.1f} > {ph_max})")
            recommendations.append("Apply gypsum or sulfur to reduce pH")
        
        # Texture assessment
        texture_match = any(t.lower() in soil['texture'].lower() 
                          for t in requirements['preferred_texture'])
        if not texture_match:
            score -= 10
            limitations.append(f"Non-ideal texture: {soil['texture']}")
        
        # Drainage assessment
        drainage_pref = requirements['drainage_preference']
        if drainage_pref == 'poor' and soil['drainage'] == 'good':
            score -= 15
            limitations.append("Soil drains too quickly for this crop")
            recommendations.append("Use bunding/leveling for water retention")
        elif drainage_pref == 'good' and soil['drainage'] == 'poor':
            score -= 15
            limitations.append("Poor drainage may cause waterlogging")
            recommendations.append("Improve drainage with raised beds")
        
        # Water holding capacity
        if soil['water_holding_pct'] < requirements['min_water_holding']:
            score -= 10
            limitations.append(f"Low water holding capacity ({soil['water_holding_pct']}%)")
            recommendations.append("Add organic matter to improve water retention")
        
        # Organic carbon
        if soil['organic_carbon_pct'] < requirements['min_organic_carbon']:
            score -= 15
            limitations.append(f"Low organic carbon ({soil['organic_carbon_pct']}%)")
            recommendations.append("Apply FYM/compost to improve organic matter")
        
        # Nutrient assessment (simplified)
        if soil['nitrogen_kg_ha'] < 180:
            score -= 5
            recommendations.append("Apply nitrogen fertilizer (urea/DAP)")
        if soil['phosphorus_kg_ha'] < 18:
            score -= 5
            recommendations.append("Apply phosphorus fertilizer (SSP/DAP)")
        
        # Calculate final suitability class
        if score >= 80:
            suitability_class = "HIGHLY SUITABLE"
        elif score >= 60:
            suitability_class = "MODERATELY SUITABLE"
        elif score >= 40:
            suitability_class = "MARGINALLY SUITABLE"
        else:
            suitability_class = "NOT SUITABLE"
        
        return {
            'district': district,
            'crop': crop,
            'soil_type': soil['soil_type'],
            'texture': soil['texture'],
            'suitability_score': max(0, score),
            'suitability_class': suitability_class,
            'limitations': limitations,
            'recommendations': recommendations,
            'soil_properties': {
                'ph': soil['ph'],
                'organic_carbon': soil['organic_carbon_pct'],
                'nitrogen': soil['nitrogen_kg_ha'],
                'phosphorus': soil['phosphorus_kg_ha'],
                'potassium': soil['potassium_kg_ha'],
                'water_holding': soil['water_holding_pct'],
                'drainage': soil['drainage']
            }
        }
    
    def get_fertility_score(self, district: str) -> Dict:
        """
        Calculate overall soil fertility score.
        
        Args:
            district: District name
            
        Returns:
            Dictionary with fertility metrics
        """
        soil = self.get_soil_profile(district)
        
        # Normalize each parameter (0-100 scale)
        scores = {
            'organic_carbon': min(100, soil['organic_carbon_pct'] / 0.75 * 100),
            'nitrogen': min(100, soil['nitrogen_kg_ha'] / 280 * 100),
            'phosphorus': min(100, soil['phosphorus_kg_ha'] / 25 * 100),
            'potassium': min(100, soil['potassium_kg_ha'] / 280 * 100),
            'ph_optimal': max(0, 100 - abs(soil['ph'] - 7.0) * 20),
            'drainage': {'good': 90, 'moderate': 70, 'poor': 50, 'excessive': 60}.get(soil['drainage'], 60)
        }
        
        # Weighted average
        weights = {
            'organic_carbon': 0.25,
            'nitrogen': 0.20,
            'phosphorus': 0.15,
            'potassium': 0.15,
            'ph_optimal': 0.15,
            'drainage': 0.10
        }
        
        overall = sum(scores[k] * weights[k] for k in scores)
        
        if overall >= 75:
            fertility_class = "HIGH"
        elif overall >= 50:
            fertility_class = "MEDIUM"
        else:
            fertility_class = "LOW"
        
        return {
            'district': district,
            'overall_fertility_score': round(overall, 1),
            'fertility_class': fertility_class,
            'component_scores': {k: round(v, 1) for k, v in scores.items()},
            'limiting_factors': [k for k, v in scores.items() if v < 50]
        }


def main():
    """Test the soil fetcher."""
    fetcher = SoilFetcher()
    
    # Test soil profile
    print("\n=== Soil Profile: Karnal ===")
    profile = fetcher.get_soil_profile("Karnal")
    for key, value in profile.items():
        print(f"  {key}: {value}")
    
    # Test suitability assessment
    print("\n=== Suitability: Karnal for Rice ===")
    suitability = fetcher.assess_suitability("Karnal", "rice")
    print(f"  Score: {suitability['suitability_score']}")
    print(f"  Class: {suitability['suitability_class']}")
    print(f"  Limitations: {suitability['limitations']}")
    print(f"  Recommendations: {suitability['recommendations']}")
    
    # Test fertility score
    print("\n=== Fertility Score: Ahmednagar ===")
    fertility = fetcher.get_fertility_score("Ahmednagar")
    print(f"  Overall: {fertility['overall_fertility_score']} ({fertility['fertility_class']})")
    print(f"  Components: {fertility['component_scores']}")
    
    return profile


if __name__ == "__main__":
    main()
