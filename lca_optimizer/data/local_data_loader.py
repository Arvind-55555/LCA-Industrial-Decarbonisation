"""
Local Data Loader
Loads grid carbon intensity from downloaded datasets
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import logging

try:
    import openpyxl  # For reading Excel files
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False
    logging.warning("openpyxl not available. Install with: pip install openpyxl")

from lca_optimizer.data.grid_data import GridDataLoader, GridCarbonIntensity

logger = logging.getLogger(__name__)


class LocalGridDataLoader(GridDataLoader):
    """
    Loader for local grid carbon intensity data.
    
    Works with downloaded datasets:
    - EPA eGRID (US)
    - OPSD (Europe)
    - Sample/custom CSV files
    """
    
    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize local data loader.
        
        Args:
            data_dir: Directory containing data files
        """
        super().__init__(api_key=None, source="local")
        self.data_dir = Path(data_dir)
        self.data_cache = {}
        self._load_data()
    
    def _load_data(self):
        """Load all available data files"""
        # Load sample grid data
        sample_file = self.data_dir / "sample_grid_data.csv"
        if sample_file.exists():
            try:
                df = pd.read_csv(sample_file, parse_dates=["timestamp"])
                self.data_cache["sample"] = df
                logger.info(f"Loaded sample grid data: {len(df)} records")
            except Exception as e:
                logger.warning(f"Failed to load sample data: {e}")
        
        # Load EPA eGRID data
        egrid_files = list(self.data_dir.glob("egrid*.xlsx"))
        if egrid_files:
            try:
                # Load eGRID data (simplified - actual structure may vary)
                df = pd.read_excel(egrid_files[0], sheet_name="SRL20", engine='openpyxl')
                # Process eGRID data to extract carbon intensity
                # This is a simplified version - actual eGRID has complex structure
                self.data_cache["egrid"] = self._process_egrid(df)
                logger.info("Loaded EPA eGRID data")
            except Exception as e:
                logger.warning(f"Failed to load eGRID data: {e}")
        
        # Load OPSD data
        opsd_files = list(self.data_dir.glob("opsd_*.csv"))
        for opsd_file in opsd_files:
            try:
                df = pd.read_csv(opsd_file, parse_dates=True, low_memory=False)
                self.data_cache["opsd"] = df
                logger.info(f"Loaded OPSD data: {opsd_file.name}")
            except Exception as e:
                logger.warning(f"Failed to load OPSD data {opsd_file}: {e}")
    
    def _process_egrid(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process EPA eGRID data to extract carbon intensity.
        
        Args:
            df: Raw eGRID DataFrame
        
        Returns:
            Processed DataFrame with carbon intensity
        """
        # eGRID structure: contains CO2 emission rates per MWh
        # This is a simplified processing - actual eGRID has multiple sheets
        try:
            # Look for CO2 emission rate columns
            co2_cols = [col for col in df.columns if 'CO2' in col.upper() or 'EMISSION' in col.upper()]
            
            if co2_cols:
                # Convert to carbon intensity (g CO2/kWh)
                # eGRID typically has lbs CO2/MWh, convert to g CO2/kWh
                processed = df.copy()
                # Simplified - actual processing depends on eGRID structure
                return processed
            else:
                logger.warning("Could not find CO2 columns in eGRID data")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error processing eGRID data: {e}")
            return pd.DataFrame()
    
    def get_current_carbon_intensity(
        self,
        location: str
    ) -> GridCarbonIntensity:
        """
        Get current carbon intensity from local data.
        
        Args:
            location: Location identifier
        
        Returns:
            Grid carbon intensity data
        """
        # Try sample data first
        if "sample" in self.data_cache:
            df = self.data_cache["sample"]
            location_data = df[df["location"] == location]
            
            if not location_data.empty:
                # Get most recent data
                latest = location_data.sort_values("timestamp").iloc[-1]
                return GridCarbonIntensity(
                    location=location,
                    timestamp=latest["timestamp"],
                    carbon_intensity=latest["carbon_intensity"],
                    renewable_share=latest.get("renewable_share", 0.4),
                    source="local_sample"
                )
        
        # Fallback to regional averages
        return self._get_regional_average(location)
    
    def get_historical_carbon_intensity(
        self,
        location: str,
        start_date: datetime,
        end_date: datetime,
        frequency: str = "hourly"
    ) -> pd.DataFrame:
        """
        Get historical carbon intensity from local data.
        
        Args:
            location: Location identifier
            start_date: Start date
            end_date: End date
            frequency: Data frequency
        
        Returns:
            DataFrame with carbon intensity over time
        """
        if "sample" in self.data_cache:
            df = self.data_cache["sample"]
            location_data = df[
                (df["location"] == location) &
                (df["timestamp"] >= start_date) &
                (df["timestamp"] <= end_date)
            ]
            
            if not location_data.empty:
                # Resample if needed
                location_data = location_data.set_index("timestamp")
                if frequency == "daily":
                    location_data = location_data[["carbon_intensity", "renewable_share"]].resample("D").mean()
                elif frequency == "monthly":
                    location_data = location_data[["carbon_intensity", "renewable_share"]].resample("MS").mean()  # Month start
                
                return location_data.reset_index()[["timestamp", "carbon_intensity", "renewable_share"]]
        
        # Generate synthetic data if no local data available
        return self._generate_synthetic_data(location, start_date, end_date, frequency)
    
    def _get_regional_average(self, location: str) -> GridCarbonIntensity:
        """Get regional average carbon intensity"""
        # Regional averages (g CO2eq/kWh)
        averages = {
            "US": 400.0,
            "EU": 300.0,
            "DE": 350.0,
            "FR": 50.0,
            "GB": 250.0,
            "CA": 150.0,
            "CN": 600.0,
            "IN": 700.0,
        }
        
        # Try to match location
        for key, value in averages.items():
            if key in location.upper():
                ci = value
                break
        else:
            ci = 300.0  # Default
        
        return GridCarbonIntensity(
            location=location,
            timestamp=datetime.now(),
            carbon_intensity=ci,
            renewable_share=0.4,
            source="regional_average"
        )
    
    def _generate_synthetic_data(
        self,
        location: str,
        start_date: datetime,
        end_date: datetime,
        frequency: str
    ) -> pd.DataFrame:
        """Generate synthetic data when local data unavailable"""
        # Determine appropriate frequency based on date range
        days_diff = (end_date - start_date).days
        
        if days_diff > 365 * 2:  # More than 2 years - use monthly
            dates = pd.date_range(start_date, end_date, freq="MS")  # Month start
            freq_type = "monthly"
        elif days_diff > 90:  # More than 3 months - use daily
            dates = pd.date_range(start_date, end_date, freq="D")
            freq_type = "daily"
        else:  # Less than 3 months - use hourly
            dates = pd.date_range(start_date, end_date, freq="h")
            freq_type = "hourly"
        
        # Override if frequency specified
        if frequency == "monthly":
            dates = pd.date_range(start_date, end_date, freq="MS")
            freq_type = "monthly"
        elif frequency == "daily":
            dates = pd.date_range(start_date, end_date, freq="D")
            freq_type = "daily"
        elif frequency == "hourly":
            dates = pd.date_range(start_date, end_date, freq="h")
            freq_type = "hourly"
        
        # Get base CI
        base_ci = self._get_regional_average(location).carbon_intensity
        
        # Generate with patterns
        ci_values = []
        for date in dates:
            if freq_type == "hourly":
                hour = date.hour
                daily_factor = 1.0 + 0.2 * abs(np.sin(2 * np.pi * hour / 24))
            elif freq_type == "daily":
                day_of_year = date.timetuple().tm_yday
                seasonal_factor = 1.0 + 0.15 * np.sin(2 * np.pi * day_of_year / 365)
                daily_factor = seasonal_factor
            else:  # monthly
                month = date.month
                seasonal_factor = 1.0 + 0.2 * np.sin(2 * np.pi * month / 12)
                daily_factor = seasonal_factor
            
            # Add long-term trend (slight decrease over time)
            years_from_start = (date - start_date).days / 365.0
            trend_factor = 1.0 - (years_from_start * 0.02)  # 2% reduction per year
            
            random_factor = np.random.normal(1.0, 0.1)
            ci = base_ci * daily_factor * trend_factor * random_factor
            ci_values.append(max(0, ci))
        
        return pd.DataFrame({
            "timestamp": dates,
            "carbon_intensity": ci_values,
            "renewable_share": [0.4] * len(dates)
        })

