"""
Enhanced Grid Data Loader with real API integration examples
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import requests
import logging
from dataclasses import dataclass
import json

from lca_optimizer.data.grid_data import GridDataLoader, GridCarbonIntensity
from lca_optimizer.data.api_client import APIClient
from lca_optimizer.data.local_data_loader import LocalGridDataLoader
from lca_optimizer.config.settings import get_settings

logger = logging.getLogger(__name__)


class ElectricityMapsLoader(GridDataLoader):
    """
    Enhanced loader for Electricity Maps API.
    
    API Documentation: https://www.electricitymaps.com/docs/api
    """
    
    def __init__(self, api_key: Optional[str] = None, cache_dir: Optional[str] = None):
        """
        Initialize Electricity Maps loader.
        
        Args:
            api_key: Electricity Maps API key (get from https://www.electricitymaps.com/)
            cache_dir: Directory for caching responses
        """
        super().__init__(api_key=api_key, source="electricity_maps")
        
        # Get API key from settings if not provided
        if not api_key:
            settings = get_settings()
            api_key = settings.electricity_maps_api_key
        
        # Initialize API client
        self.client = APIClient(
            base_url="https://api.electricitymap.org/v3",
            api_key=api_key,
            headers={"auth-token": api_key} if api_key else {},
            rate_limit=10.0,  # 10 requests per second
            cache_dir=cache_dir or "data/cache/electricity_maps"
        )
    
    def get_current_carbon_intensity(
        self,
        location: str
    ) -> GridCarbonIntensity:
        """
        Get current carbon intensity from Electricity Maps.
        
        Args:
            location: Zone code (e.g., "DE", "FR", "US-CA")
        
        Returns:
            Grid carbon intensity data
        """
        cache_key = f"{location}_{datetime.now().strftime('%Y%m%d%H')}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            data = self.client.get(
                endpoint="carbon-intensity/latest",
                params={"zone": location},
                use_cache=True,
                cache_max_age_hours=1
            )
            
            ci_data = GridCarbonIntensity(
                location=location,
                timestamp=datetime.fromisoformat(data["datetime"].replace("Z", "+00:00")),
                carbon_intensity=data.get("carbonIntensity", 300.0),
                renewable_share=data.get("renewablePercentage", 0.4),
                source="electricity_maps"
            )
            
            self.cache[cache_key] = ci_data
            return ci_data
            
        except Exception as e:
            logger.warning(f"Failed to fetch from Electricity Maps: {e}. Using default.")
            return self._get_default_ci(location)
    
    def get_historical_carbon_intensity(
        self,
        location: str,
        start_date: datetime,
        end_date: datetime,
        frequency: str = "hourly"
    ) -> pd.DataFrame:
        """
        Get historical carbon intensity from Electricity Maps.
        
        Args:
            location: Zone code
            start_date: Start date
            end_date: End date
            frequency: "hourly" or "daily"
        
        Returns:
            DataFrame with carbon intensity over time
        """
        try:
            url = f"{self.base_url}/carbon-intensity/history"
            params = {
                "zone": location,
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            }
            
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Convert to DataFrame
            records = []
            for entry in data.get("history", []):
                records.append({
                    "timestamp": datetime.fromisoformat(entry["datetime"].replace("Z", "+00:00")),
                    "carbon_intensity": entry.get("carbonIntensity", 300.0),
                    "renewable_share": entry.get("renewablePercentage", 0.4)
                })
            
            return pd.DataFrame(records)
            
        except requests.RequestException as e:
            logger.warning(f"Failed to fetch historical data: {e}. Generating synthetic data.")
            return self._generate_synthetic_data(location, start_date, end_date, frequency)
    
    def _get_default_ci(self, location: str) -> GridCarbonIntensity:
        """Get default carbon intensity when API fails"""
        # Default values by region
        defaults = {
            "DE": 350.0,  # Germany
            "FR": 50.0,   # France (nuclear)
            "US-CA": 200.0,  # California
            "GB": 250.0,  # UK
        }
        
        ci = defaults.get(location, 300.0)
        
        return GridCarbonIntensity(
            location=location,
            timestamp=datetime.now(),
            carbon_intensity=ci,
            renewable_share=0.4,
            source="default"
        )
    
    def _generate_synthetic_data(
        self,
        location: str,
        start_date: datetime,
        end_date: datetime,
        frequency: str
    ) -> pd.DataFrame:
        """Generate synthetic data when API is unavailable"""
        if frequency == "hourly":
            dates = pd.date_range(start_date, end_date, freq="H")
        else:
            dates = pd.date_range(start_date, end_date, freq="D")
        
        # Generate realistic patterns (lower during night, higher during day)
        base_ci = 300.0
        patterns = np.sin(np.arange(len(dates)) * 2 * np.pi / 24) * 50 + base_ci
        
        data = pd.DataFrame({
            "timestamp": dates,
            "carbon_intensity": np.maximum(patterns + np.random.normal(0, 20, len(dates)), 0),
            "renewable_share": 0.3 + np.random.normal(0, 0.1, len(dates))
        })
        
        return data


class WattTimeLoader(GridDataLoader):
    """
    Enhanced loader for WattTime API.
    
    API Documentation: https://www.watttime.org/api-documentation/
    """
    
    def __init__(self, username: Optional[str] = None, password: Optional[str] = None):
        """
        Initialize WattTime loader.
        
        Args:
            username: WattTime username
            password: WattTime password
        """
        super().__init__(api_key=None, source="watt_time")
        self.username = username
        self.password = password
        self.base_url = "https://api.watttime.org/v2"
        self.token = None
        
        if username and password:
            self._authenticate()
    
    def _authenticate(self):
        """Authenticate and get access token"""
        try:
            response = requests.get(
                f"{self.base_url}/login",
                auth=(self.username, self.password),
                timeout=10
            )
            response.raise_for_status()
            self.token = response.json()["token"]
            self.headers = {"Authorization": f"Bearer {self.token}"}
            logger.info("WattTime authentication successful")
        except requests.RequestException as e:
            logger.warning(f"WattTime authentication failed: {e}")
            self.token = None
    
    def get_current_carbon_intensity(
        self,
        location: str
    ) -> GridCarbonIntensity:
        """Get current carbon intensity from WattTime"""
        if not self.token:
            return self._get_default_ci(location)
        
        cache_key = f"{location}_{datetime.now().strftime('%Y%m%d%H')}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            url = f"{self.base_url}/index"
            params = {"ba": location}  # Balancing Authority
            
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            ci_data = GridCarbonIntensity(
                location=location,
                timestamp=datetime.fromisoformat(data["point_time"].replace("Z", "+00:00")),
                carbon_intensity=data.get("moer", 300.0),  # Marginal Operating Emission Rate
                renewable_share=None,
                source="watt_time"
            )
            
            self.cache[cache_key] = ci_data
            return ci_data
            
        except requests.RequestException as e:
            logger.warning(f"Failed to fetch from WattTime: {e}. Using default.")
            return self._get_default_ci(location)
    
    def _get_default_ci(self, location: str) -> GridCarbonIntensity:
        """Get default carbon intensity"""
        return GridCarbonIntensity(
            location=location,
            timestamp=datetime.now(),
            carbon_intensity=300.0,
            renewable_share=None,
            source="default"
        )

