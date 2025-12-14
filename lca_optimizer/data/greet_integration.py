"""
GREET Database Integration
GREET (Greenhouse gases, Regulated Emissions, and Energy use in Technologies)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging
import requests
from io import StringIO

from lca_optimizer.data.api_client import APIClient

logger = logging.getLogger(__name__)


class GREETIntegration:
    """
    Integration with GREET database.
    
    GREET provides well-to-wheel (WTW) emission factors for various fuel pathways.
    Data can be accessed via:
    1. GREET Excel files (requires download)
    2. GREET Online API (if available)
    3. Pre-processed data files
    """
    
    # Default GREET emission factors (g CO2eq/MJ)
    # These are approximate values - real data should come from GREET database
    DEFAULT_FACTORS = {
        "hydrogen_electrolysis_wind": 0.0,
        "hydrogen_electrolysis_solar": 0.0,
        "hydrogen_electrolysis_grid": 50.0,
        "hydrogen_steam_reforming": 90.0,
        "ammonia_green": 0.0,
        "ammonia_conventional": 80.0,
        "methanol_green": 0.0,
        "methanol_conventional": 70.0,
        "diesel": 95.0,
        "gasoline": 95.0,
        "jet_fuel": 90.0,
        "natural_gas": 70.0,
        "electricity_grid_us": 120.0,
        "electricity_grid_eu": 300.0,
    }
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize GREET integration.
        
        Args:
            data_path: Path to GREET data files (Excel or CSV)
        """
        self.data_path = Path(data_path) if data_path else None
        self.factors = self.DEFAULT_FACTORS.copy()
        
        if self.data_path and self.data_path.exists():
            self._load_greet_data()
        
        logger.info("GREET integration initialized")
    
    def _load_greet_data(self):
        """Load GREET data from files"""
        try:
            if self.data_path.suffix == '.xlsx':
                # Load from Excel (requires openpyxl)
                df = pd.read_excel(self.data_path, sheet_name='WTW')
                # Process and extract factors
                # This is a placeholder - actual implementation depends on GREET file structure
                logger.info("GREET data loaded from Excel")
            elif self.data_path.suffix == '.csv':
                df = pd.read_csv(self.data_path)
                # Process CSV data
                logger.info("GREET data loaded from CSV")
        except Exception as e:
            logger.warning(f"Failed to load GREET data: {e}. Using default factors.")
    
    def get_wtw_emissions(
        self,
        fuel_pathway: str,
        unit: str = "g_CO2eq_MJ"
    ) -> Dict[str, float]:
        """
        Get well-to-wheel emissions for a fuel pathway.
        
        Args:
            fuel_pathway: Fuel pathway name
            unit: Unit for emissions
        
        Returns:
            Dictionary with WTW, WTT, and TTW emissions
        """
        # Get base emission factor
        base_factor = self.factors.get(fuel_pathway, 95.0)  # Default to diesel
        
        # Split into WTT and TTW (simplified)
        # WTT typically 20-30% of WTW for conventional fuels
        # For green fuels, WTT is the main component
        if "green" in fuel_pathway or "electrolysis" in fuel_pathway:
            wtt = base_factor
            ttw = 0.0
        else:
            wtt = base_factor * 0.25
            ttw = base_factor * 0.75
        
        return {
            "well_to_wheel": base_factor,
            "well_to_tank": wtt,
            "tank_to_wheel": ttw,
            "unit": unit
        }
    
    def compare_pathways(
        self,
        pathways: List[str]
    ) -> pd.DataFrame:
        """
        Compare multiple fuel pathways.
        
        Args:
            pathways: List of pathway names
        
        Returns:
            DataFrame with comparison
        """
        results = []
        
        for pathway in pathways:
            wtw_data = self.get_wtw_emissions(pathway)
            results.append({
                "pathway": pathway,
                "wtw_emissions": wtw_data["well_to_wheel"],
                "wtt_emissions": wtw_data["well_to_tank"],
                "ttw_emissions": wtw_data["tank_to_wheel"]
            })
        
        df = pd.DataFrame(results)
        df = df.sort_values("wtw_emissions")
        
        return df
    
    def update_factor(
        self,
        pathway: str,
        wtw_emission: float,
        wtt_emission: Optional[float] = None,
        ttw_emission: Optional[float] = None
    ):
        """
        Update emission factor for a pathway.
        
        Args:
            pathway: Pathway name
            wtw_emission: Well-to-wheel emission (g CO2eq/MJ)
            wtt_emission: Well-to-tank emission (optional)
            ttw_emission: Tank-to-wheel emission (optional)
        """
        self.factors[pathway] = wtw_emission
        logger.info(f"Updated factor for {pathway}: {wtw_emission} g CO2eq/MJ")
    
    def load_from_url(self, url: str):
        """
        Load GREET data from URL.
        
        Args:
            url: URL to GREET data file
        """
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Try to parse as CSV
            df = pd.read_csv(StringIO(response.text))
            # Process data
            logger.info(f"GREET data loaded from URL: {url}")
            
        except Exception as e:
            logger.error(f"Failed to load GREET data from URL: {e}")

