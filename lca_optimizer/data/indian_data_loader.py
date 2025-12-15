"""
Indian Industrial Data Loader
Extracts process data from Indian industrial portals and data.gov.in
"""

import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from io import StringIO

import numpy as np
import pandas as pd
import requests

from lca_optimizer.data.grid_data import GridDataLoader, GridCarbonIntensity

logger = logging.getLogger(__name__)


class IndianDataLoader:
    """
    Loader for Indian industrial data from various sources:
    - data.gov.in: Industrial emissions, energy consumption datasets
    - CEA (Central Electricity Authority): Grid carbon intensity by state
    - CPCB (Central Pollution Control Board): Industrial emissions data
    - Ministry of Steel/Cement: Sector-specific process data
    """
    
    def __init__(self, data_dir: str = "data/raw/indian"):
        """
        Initialize Indian data loader.
        
        Args:
            data_dir: Directory to store downloaded Indian data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.data_cache = {}
        self._load_cached_data()
        
        # Indian state codes for regional data
        self.state_codes = {
            "Maharashtra": "MH", "Gujarat": "GJ", "Tamil Nadu": "TN",
            "Karnataka": "KA", "West Bengal": "WB", "Odisha": "OD",
            "Jharkhand": "JH", "Chhattisgarh": "CH", "Rajasthan": "RJ",
            "Madhya Pradesh": "MP", "Uttar Pradesh": "UP", "Punjab": "PB",
            "Haryana": "HR", "Delhi": "DL", "Andhra Pradesh": "AP",
            "Telangana": "TG", "Kerala": "KL", "Bihar": "BR"
        }
        
        logger.info("Indian Data Loader initialized")
    
    def _load_cached_data(self):
        """Load previously downloaded data from cache"""
        cache_file = self.data_dir / "cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    self.data_cache = json.load(f)
                logger.info("Loaded cached Indian data")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
    
    def _save_cache(self):
        """Save data cache to disk"""
        cache_file = self.data_dir / "cache.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(self.data_cache, f, default=str)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def download_data_gov_in_dataset(
        self,
        dataset_id: str,
        resource_id: Optional[str] = None,
        force_download: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        Download dataset from data.gov.in.
        
        Args:
            dataset_id: Dataset identifier from data.gov.in
            resource_id: Specific resource ID (optional)
            force_download: Force re-download even if cached
        
        Returns:
            DataFrame with downloaded data or None
        """
        cache_key = f"datagov_{dataset_id}"
        
        if not force_download and cache_key in self.data_cache:
            logger.info(f"Using cached data for dataset {dataset_id}")
            return pd.read_json(StringIO(self.data_cache[cache_key]))
        
        try:
            api_key = os.getenv("DATAGOV_API_KEY")
            if not api_key:
                logger.error("DATAGOV_API_KEY environment variable not set. Cannot download data.gov.in dataset.")
                return None
            
            # data.gov.in API endpoint
            base_url = "https://api.data.gov.in/resource"
            url = f"{base_url}/{dataset_id}"
            
            params = {"api-key": api_key, "format": "json", "limit": 10000}
            if resource_id:
                params["resource_id"] = resource_id
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if "records" in data:
                df = pd.DataFrame(data["records"])
                
                # Cache the data
                self.data_cache[cache_key] = df.to_json()
                self._save_cache()
                
                # Save to file
                output_file = self.data_dir / f"datagov_{dataset_id}.csv"
                df.to_csv(output_file, index=False)
                
                logger.info(f"Downloaded dataset {dataset_id}: {len(df)} records")
                return df
            else:
                logger.warning(f"No records found in dataset {dataset_id}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download dataset {dataset_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error processing dataset {dataset_id}: {e}")
            return None
    
    def get_industrial_emissions_data(
        self,
        sector: Optional[str] = None,
        state: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get industrial emissions data for Indian industries.
        
        Args:
            sector: Filter by sector (steel, cement, aluminium, chemicals, refining)
            state: Filter by state
        
        Returns:
            DataFrame with emissions data
        """
        # Try to load from local file first
        emissions_file = self.data_dir / "industrial_emissions.csv"
        if emissions_file.exists():
            df = pd.read_csv(emissions_file)
            if sector:
                df = df[df["sector"].str.lower() == sector.lower()]
            if state:
                df = df[df["state"].str.lower() == state.lower()]

            # If we have matching data, return it; otherwise fall through to sample generation
            if not df.empty:
                return df
        
        # If not available, try to download from data.gov.in
        # Common dataset IDs for industrial emissions (these would need to be updated)
        dataset_ids = {
            "industrial_emissions": "industrial-emissions-india",
            "cement_emissions": "cement-industry-emissions",
            "steel_emissions": "steel-industry-emissions"
        }
        
        # For now, generate sample data structure
        logger.warning("Industrial emissions data not found. Generating sample structure.")
        return self._generate_sample_emissions_data(sector, state)
    
    def _generate_sample_emissions_data(
        self,
        sector: Optional[str] = None,
        state: Optional[str] = None
    ) -> pd.DataFrame:
        """Generate sample emissions data structure for Indian industries"""
        sectors = ["steel", "cement", "aluminium", "chemicals", "refining"]
        states = list(self.state_codes.keys())
        
        if sector:
            sectors = [sector]
        if state:
            states = [state]
        
        data = []
        for sec in sectors:
            for st in states:
                for year in range(2015, 2024):
                    # Sample emissions data (would be replaced with real data)
                    base_emissions = {
                        "steel": 2.0,  # tCO2/t steel
                        "cement": 0.8,  # tCO2/t cement
                        "aluminium": 12.0,  # tCO2/t aluminium
                        "chemicals": 1.5,  # tCO2/t chemicals
                        "refining": 0.3  # tCO2/t refined products
                    }
                    
                    data.append({
                        "sector": sec,
                        "state": st,
                        "year": year,
                        "emissions_tco2": base_emissions.get(sec, 1.0) * np.random.uniform(0.8, 1.2),
                        "production_tonnes": np.random.uniform(100000, 10000000),
                        "energy_consumption_mwh": np.random.uniform(50000, 5000000),
                        "grid_carbon_intensity": np.random.uniform(0.7, 1.2)  # kg CO2/kWh
                    })
        
        df = pd.DataFrame(data)
        
        # Save sample data
        output_file = self.data_dir / "industrial_emissions.csv"
        df.to_csv(output_file, index=False)
        
        return df
    
    def get_state_grid_carbon_intensity(
        self,
        state: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get grid carbon intensity for Indian state.
        
        Args:
            state: Indian state name
            start_date: Start date for time series
            end_date: End date for time series
        
        Returns:
            DataFrame with carbon intensity time series
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365)
        if end_date is None:
            end_date = datetime.now()
        
        # Try to load from CEA data or local file
        state_file = self.data_dir / f"grid_ci_{state.replace(' ', '_')}.csv"
        if state_file.exists():
            df = pd.read_csv(state_file, parse_dates=["timestamp"])
            df = df[(df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)]
            return df
        
        # Generate sample data based on Indian state characteristics
        logger.info(f"Generating sample grid CI data for {state}")
        return self._generate_state_grid_ci(state, start_date, end_date)
    
    def _generate_state_grid_ci(
        self,
        state: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Generate sample grid carbon intensity for Indian state"""
        # Indian state grid characteristics (approximate)
        state_ci_baselines = {
            "Maharashtra": 0.85, "Gujarat": 0.80, "Tamil Nadu": 0.75,
            "Karnataka": 0.70, "West Bengal": 0.90, "Odisha": 0.95,
            "Jharkhand": 1.00, "Chhattisgarh": 0.95, "Rajasthan": 0.85,
            "Delhi": 0.90, "Punjab": 0.80, "Haryana": 0.85
        }
        
        baseline_ci = state_ci_baselines.get(state, 0.85)  # kg CO2/kWh
        
        # Generate hourly time series
        timestamps = pd.date_range(start_date, end_date, freq='H')
        
        # Add seasonal and daily variations
        data = []
        for ts in timestamps:
            # Seasonal variation (higher in summer due to AC demand)
            seasonal_factor = 1.0 + 0.15 * np.sin(2 * np.pi * ts.dayofyear / 365)
            
            # Daily variation (higher during peak hours)
            daily_factor = 1.0 + 0.2 * np.sin(2 * np.pi * ts.hour / 24)
            
            # Random variation
            random_factor = np.random.uniform(0.9, 1.1)
            
            ci = baseline_ci * seasonal_factor * daily_factor * random_factor
            
            data.append({
                "timestamp": ts,
                "carbon_intensity": ci,  # kg CO2/kWh
                "state": state,
                "state_code": self.state_codes.get(state, "XX")
            })
        
        df = pd.DataFrame(data)
        
        # Save to file
        state_file = self.data_dir / f"grid_ci_{state.replace(' ', '_')}.csv"
        df.to_csv(state_file, index=False)
        
        return df
    
    def get_steel_process_data(self, state: Optional[str] = None) -> pd.DataFrame:
        """
        Get Indian steel process data (DRI, BF-BOF characteristics).
        
        Args:
            state: Filter by state
        
        Returns:
            DataFrame with steel process parameters
        """
        file_path = self.data_dir / "steel_process_data.csv"
        if file_path.exists():
            df = pd.read_csv(file_path)
            if state:
                df = df[df['state'] == state]
            return df
        
        # Generate sample data for Indian steel processes
        logger.info("Generating sample Indian steel process data")
        return self._generate_steel_process_data(state)
    
    def _generate_steel_process_data(self, state: Optional[str] = None) -> pd.DataFrame:
        """Generate sample Indian steel process data"""
        states = [state] if state else list(self.state_codes.keys())[:10]  # Top 10 states
        
        data = []
        for st in states:
            # Indian steel plants typically use DRI (Direct Reduced Iron) processes
            # Mixed with BF-BOF (Blast Furnace - Basic Oxygen Furnace)
            
            # DRI process characteristics
            data.append({
                "state": st,
                "process_type": "DRI",
                "iron_ore_source": "Odisha/Jharkhand",
                "coal_type": "Indian coal",
                "dri_capacity_tonnes_per_year": np.random.uniform(500000, 5000000),
                "specific_energy_consumption_mwh_per_tonne": np.random.uniform(2.5, 4.0),
                "emission_factor_tco2_per_tonne": np.random.uniform(1.8, 2.5),
                "grid_dependency": np.random.uniform(0.6, 0.9),  # % of energy from grid
                "renewable_share": np.random.uniform(0.05, 0.25)  # % renewable in grid
            })
            
            # BF-BOF process characteristics
            data.append({
                "state": st,
                "process_type": "BF-BOF",
                "iron_ore_source": "Odisha/Jharkhand",
                "coal_type": "Indian coking coal",
                "dri_capacity_tonnes_per_year": np.random.uniform(1000000, 10000000),
                "specific_energy_consumption_mwh_per_tonne": np.random.uniform(3.5, 5.0),
                "emission_factor_tco2_per_tonne": np.random.uniform(2.0, 2.8),
                "grid_dependency": np.random.uniform(0.3, 0.6),
                "renewable_share": np.random.uniform(0.05, 0.20)
            })
        
        df = pd.DataFrame(data)
        
        # Save to file
        file_path = self.data_dir / "steel_process_data.csv"
        df.to_csv(file_path, index=False)
        
        return df
    
    def get_cement_process_data(self, state: Optional[str] = None) -> pd.DataFrame:
        """Get Indian cement process data"""
        file_path = self.data_dir / "cement_process_data.csv"
        if file_path.exists():
            df = pd.read_csv(file_path)
            if state:
                df = df[df['state'] == state]
            return df
        
        logger.info("Generating sample Indian cement process data")
        return self._generate_cement_process_data(state)
    
    def _generate_cement_process_data(self, state: Optional[str] = None) -> pd.DataFrame:
        """Generate sample Indian cement process data"""
        states = [state] if state else ["Rajasthan", "Madhya Pradesh", "Andhra Pradesh", "Karnataka", "Tamil Nadu"]
        
        data = []
        for st in states:
            # Indian cement plants with clinker substitution variations
            data.append({
                "state": st,
                "clinker_ratio": np.random.uniform(0.65, 0.85),  # Indian average
                "fly_ash_substitution": np.random.uniform(0.15, 0.35),
                "slag_substitution": np.random.uniform(0.05, 0.15),
                "production_capacity_tonnes_per_year": np.random.uniform(1000000, 10000000),
                "specific_energy_mwh_per_tonne": np.random.uniform(0.08, 0.12),
                "emission_factor_tco2_per_tonne": np.random.uniform(0.6, 0.9),
                "coal_share": np.random.uniform(0.7, 0.9),  # High coal dependency
                "petcoke_share": np.random.uniform(0.1, 0.3),
                "alternative_fuel_share": np.random.uniform(0.0, 0.1)
            })
        
        df = pd.DataFrame(data)
        file_path = self.data_dir / "cement_process_data.csv"
        df.to_csv(file_path, index=False)
        return df
    
    def get_aluminium_process_data(self, state: Optional[str] = None) -> pd.DataFrame:
        """Get Indian aluminium smelting process data"""
        file_path = self.data_dir / "aluminium_process_data.csv"
        if file_path.exists():
            df = pd.read_csv(file_path)
            if state:
                df = df[df['state'] == state]
            return df
        
        logger.info("Generating sample Indian aluminium process data")
        return self._generate_aluminium_process_data(state)
    
    def _generate_aluminium_process_data(self, state: Optional[str] = None) -> pd.DataFrame:
        """Generate sample Indian aluminium process data"""
        # Major aluminium producing states
        states = [state] if state else ["Odisha", "Chhattisgarh", "Jharkhand", "Andhra Pradesh"]
        
        data = []
        for st in states:
            data.append({
                "state": st,
                "smelting_technology": "Pre-baked anode",
                "specific_energy_consumption_mwh_per_tonne": np.random.uniform(13.5, 15.5),  # Indian average
                "emission_factor_tco2_per_tonne": np.random.uniform(10.0, 14.0),
                "grid_carbon_intensity_kg_co2_per_kwh": np.random.uniform(0.85, 1.05),
                "production_capacity_tonnes_per_year": np.random.uniform(100000, 2000000),
                "power_source": "State grid + captive power",
                "renewable_share": np.random.uniform(0.05, 0.20)
            })
        
        df = pd.DataFrame(data)
        file_path = self.data_dir / "aluminium_process_data.csv"
        df.to_csv(file_path, index=False)
        return df

