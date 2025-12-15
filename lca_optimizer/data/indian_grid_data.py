"""
Indian Regional Grid Carbon Intensity Data Loader
Integrates state-specific grid carbon intensity for Indian power grid
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import logging

from lca_optimizer.data.grid_data import GridDataLoader, GridCarbonIntensity
from lca_optimizer.data.indian_data_loader import IndianDataLoader

logger = logging.getLogger(__name__)


class IndianGridDataLoader(GridDataLoader):
    """
    Indian regional grid carbon intensity loader.
    
    Provides state-specific carbon intensity data for Indian power grid,
    accounting for regional variations in power generation mix.
    """
    
    # Indian state grid characteristics (approximate baseline values)
    STATE_GRID_BASELINES = {
        "Maharashtra": {"ci_kg_co2_per_kwh": 0.85, "renewable_share": 0.15},
        "Gujarat": {"ci_kg_co2_per_kwh": 0.80, "renewable_share": 0.20},
        "Tamil Nadu": {"ci_kg_co2_per_kwh": 0.75, "renewable_share": 0.25},
        "Karnataka": {"ci_kg_co2_per_kwh": 0.70, "renewable_share": 0.30},
        "Rajasthan": {"ci_kg_co2_per_kwh": 0.85, "renewable_share": 0.20},
        "Andhra Pradesh": {"ci_kg_co2_per_kwh": 0.80, "renewable_share": 0.22},
        "Telangana": {"ci_kg_co2_per_kwh": 0.82, "renewable_share": 0.18},
        "Odisha": {"ci_kg_co2_per_kwh": 0.95, "renewable_share": 0.10},
        "Jharkhand": {"ci_kg_co2_per_kwh": 1.00, "renewable_share": 0.08},
        "Chhattisgarh": {"ci_kg_co2_per_kwh": 0.95, "renewable_share": 0.12},
        "West Bengal": {"ci_kg_co2_per_kwh": 0.90, "renewable_share": 0.12},
        "Delhi": {"ci_kg_co2_per_kwh": 0.90, "renewable_share": 0.10},
        "Punjab": {"ci_kg_co2_per_kwh": 0.80, "renewable_share": 0.15},
        "Haryana": {"ci_kg_co2_per_kwh": 0.85, "renewable_share": 0.15},
        "Madhya Pradesh": {"ci_kg_co2_per_kwh": 0.88, "renewable_share": 0.18},
        "Uttar Pradesh": {"ci_kg_co2_per_kwh": 0.92, "renewable_share": 0.12},
        "Kerala": {"ci_kg_co2_per_kwh": 0.75, "renewable_share": 0.30},
        "Bihar": {"ci_kg_co2_per_kwh": 0.95, "renewable_share": 0.10}
    }
    
    def __init__(self, data_dir: str = "data/raw/indian"):
        """
        Initialize Indian grid data loader.
        
        Args:
            data_dir: Directory containing Indian data files
        """
        super().__init__(api_key=None, source="indian_regional")
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.indian_loader = IndianDataLoader(data_dir)
        
        logger.info("Indian Grid Data Loader initialized")
    
    def get_carbon_intensity(
        self,
        location: str,
        timestamp: Optional[datetime] = None
    ) -> Optional[GridCarbonIntensity]:
        """
        Get carbon intensity for Indian state.
        
        Args:
            location: Indian state name
            timestamp: Timestamp (defaults to now)
        
        Returns:
            GridCarbonIntensity object or None
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Normalize state name
        state = self._normalize_state_name(location)
        
        if state not in self.STATE_GRID_BASELINES:
            logger.warning(f"Unknown Indian state: {location}, using default")
            state = "Maharashtra"  # Default
        
        # Get historical data for this state
        state_data = self.indian_loader.get_state_grid_carbon_intensity(
            state, 
            start_date=timestamp - timedelta(days=1),
            end_date=timestamp
        )
        
        if state_data is not None and not state_data.empty:
            # Get most recent value
            latest = state_data.iloc[-1]
            ci_value = latest['carbon_intensity']
        else:
            # Use baseline with time variation
            baseline = self.STATE_GRID_BASELINES[state]["ci_kg_co2_per_kwh"]
            ci_value = self._apply_time_variation(baseline, timestamp)
        
        return GridCarbonIntensity(
            location=state,
            carbon_intensity=ci_value,  # kg CO2/kWh
            timestamp=timestamp,
            source="indian_regional",
            unit="kg_CO2_per_kWh"
        )
    
    def get_historical_carbon_intensity(
        self,
        location: str,
        start_date: datetime,
        end_date: datetime,
        frequency: str = "hourly"
    ) -> pd.DataFrame:
        """
        Get historical carbon intensity for Indian state.
        
        Args:
            location: Indian state name
            start_date: Start date
            end_date: End date
            frequency: "hourly", "daily", "monthly"
        
        Returns:
            DataFrame with historical carbon intensity
        """
        state = self._normalize_state_name(location)
        
        # Get data from Indian data loader
        state_data = self.indian_loader.get_state_grid_carbon_intensity(
            state, start_date, end_date
        )
        
        if state_data is None or state_data.empty:
            # Generate data if not available
            logger.info(f"Generating historical grid CI for {state}")
            state_data = self.indian_loader._generate_state_grid_ci(state, start_date, end_date)
        
        # Resample based on frequency
        if frequency == "daily":
            state_data = state_data.set_index('timestamp').resample('D').mean().reset_index()
        elif frequency == "monthly":
            state_data = state_data.set_index('timestamp').resample('M').mean().reset_index()
        # hourly is default, no resampling needed
        
        return state_data
    
    def get_state_statistics(self, location: str) -> Dict[str, float]:
        """
        Get statistics for Indian state grid.
        
        Args:
            location: Indian state name
        
        Returns:
            Dictionary with grid statistics
        """
        state = self._normalize_state_name(location)
        
        if state not in self.STATE_GRID_BASELINES:
            return {}
        
        baseline = self.STATE_GRID_BASELINES[state]
        
        # Get recent historical data for statistics
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        hist_data = self.get_historical_carbon_intensity(
            state, start_date, end_date, frequency="hourly"
        )
        
        if hist_data is not None and not hist_data.empty:
            ci_values = hist_data['carbon_intensity'].values
            return {
                "mean": float(np.mean(ci_values)),
                "min": float(np.min(ci_values)),
                "max": float(np.max(ci_values)),
                "std": float(np.std(ci_values)),
                "renewable_share": baseline["renewable_share"],
                "baseline_ci": baseline["ci_kg_co2_per_kwh"]
            }
        else:
            return {
                "mean": baseline["ci_kg_co2_per_kwh"],
                "renewable_share": baseline["renewable_share"]
            }
    
    def _normalize_state_name(self, location: str) -> str:
        """Normalize state name to match keys"""
        location_lower = location.lower()
        
        for state in self.STATE_GRID_BASELINES.keys():
            if state.lower() == location_lower or state.lower().replace(' ', '_') == location_lower:
                return state
        
        # Try partial match
        for state in self.STATE_GRID_BASELINES.keys():
            if location_lower in state.lower() or state.lower() in location_lower:
                return state
        
        return location  # Return as-is if no match
    
    def _apply_time_variation(self, baseline: float, timestamp: datetime) -> float:
        """
        Apply time-based variation to baseline carbon intensity.
        
        Accounts for:
        - Seasonal variations (higher in summer due to AC demand)
        - Daily variations (peak hours)
        - Random fluctuations
        """
        # Seasonal variation (summer peak)
        day_of_year = timestamp.timetuple().tm_yday
        seasonal_factor = 1.0 + 0.15 * np.sin(2 * np.pi * day_of_year / 365 - np.pi/2)
        
        # Daily variation (peak hours 6-10 AM, 6-10 PM)
        hour = timestamp.hour
        if 6 <= hour <= 10 or 18 <= hour <= 22:
            daily_factor = 1.0 + 0.2
        else:
            daily_factor = 1.0 - 0.1
        
        # Random variation
        random_factor = np.random.uniform(0.95, 1.05)
        
        return baseline * seasonal_factor * daily_factor * random_factor

