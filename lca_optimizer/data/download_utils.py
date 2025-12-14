"""
Data Download Utilities
Downloads open datasets for grid carbon intensity
"""

import requests
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Optional, Dict, List
import logging
from datetime import datetime
import zipfile
import io

logger = logging.getLogger(__name__)


class DatasetDownloader:
    """Download and manage open datasets"""
    
    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize dataset downloader.
        
        Args:
            data_dir: Directory to store downloaded data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Dataset downloader initialized: {self.data_dir}")
    
    def download_epa_egrid(
        self,
        year: int = 2022,
        force: bool = False
    ) -> Optional[Path]:
        """
        Download EPA eGRID data.
        
        EPA eGRID provides US grid carbon intensity data.
        Source: https://www.epa.gov/egrid
        
        Args:
            year: Data year (2020, 2021, 2022)
            force: Force re-download even if file exists
        
        Returns:
            Path to downloaded file
        """
        filename = f"egrid{year}_data.xlsx"
        filepath = self.data_dir / filename
        
        if filepath.exists() and not force:
            logger.info(f"eGRID {year} data already exists: {filepath}")
            return filepath
        
        # EPA eGRID download URLs (these may need to be updated)
        base_url = "https://www.epa.gov/system/files/documents/2024-01"
        urls = {
            2022: f"{base_url}/egrid2022_data.xlsx",
            2021: f"{base_url}/egrid2021_data.xlsx",
            2020: f"{base_url}/egrid2020_data.xlsx"
        }
        
        url = urls.get(year)
        if not url:
            logger.warning(f"eGRID data for year {year} not available")
            return None
        
        try:
            logger.info(f"Downloading EPA eGRID {year} data...")
            response = requests.get(url, timeout=300, stream=True)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Downloaded eGRID data to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to download eGRID data: {e}")
            # Try alternative URL format
            alt_url = f"https://www.epa.gov/sites/default/files/2024-01/egrid{year}_data.xlsx"
            try:
                response = requests.get(alt_url, timeout=300, stream=True)
                response.raise_for_status()
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                logger.info(f"Downloaded eGRID data (alternative URL) to {filepath}")
                return filepath
            except Exception as e2:
                logger.error(f"Alternative download also failed: {e2}")
                return None
    
    def download_iea_data(
        self,
        dataset: str = "electricity",
        force: bool = False
    ) -> Optional[Path]:
        """
        Download IEA (International Energy Agency) data.
        
        Source: https://www.iea.org/data-and-statistics
        
        Args:
            dataset: Dataset name
            force: Force re-download
        
        Returns:
            Path to downloaded file
        """
        # IEA data is typically available through their data portal
        # This is a placeholder - actual URLs need to be obtained from IEA
        logger.info("IEA data download - manual download may be required")
        logger.info("Visit: https://www.iea.org/data-and-statistics/data-tools")
        return None
    
    def download_opsd_data(
        self,
        dataset: str = "renewable_power_plants",
        force: bool = False
    ) -> Optional[Path]:
        """
        Download Open Power System Data (OPSD).
        
        Source: https://open-power-system-data.org/
        
        Args:
            dataset: Dataset name
            force: Force re-download
        
        Returns:
            Path to downloaded file
        """
        # OPSD provides European electricity data
        base_url = "https://data.open-power-system-data.org"
        
        datasets = {
            "renewable_power_plants": f"{base_url}/renewable_power_plants/2020-08-25/renewable_power_plants_DE.csv",
            "time_series": f"{base_url}/time_series/2020-10-06/time_series_60min_singleindex.csv"
        }
        
        url = datasets.get(dataset)
        if not url:
            logger.warning(f"OPSD dataset '{dataset}' not available")
            return None
        
        filename = f"opsd_{dataset}.csv"
        filepath = self.data_dir / filename
        
        if filepath.exists() and not force:
            logger.info(f"OPSD {dataset} data already exists: {filepath}")
            return filepath
        
        try:
            logger.info(f"Downloading OPSD {dataset} data...")
            response = requests.get(url, timeout=300, stream=True)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Downloaded OPSD data to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to download OPSD data: {e}")
            return None
    
    def create_sample_grid_data(
        self,
        locations: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> Path:
        """
        Create sample grid carbon intensity data.
        
        Generates realistic synthetic data based on regional averages.
        
        Args:
            locations: List of location codes
            start_date: Start date
            end_date: End date
        
        Returns:
            Path to created file
        """
        # Regional average carbon intensities (g CO2eq/kWh)
        regional_averages = {
            "US": 400.0,
            "EU": 300.0,
            "DE": 350.0,  # Germany
            "FR": 50.0,   # France (nuclear)
            "GB": 250.0,  # UK
            "CA": 150.0,  # Canada
            "CN": 600.0,  # China
            "IN": 700.0,  # India
        }
        
        # Generate hourly data
        dates = pd.date_range(start_date, end_date, freq="h")
        data = []
        
        for location in locations:
            base_ci = regional_averages.get(location, 300.0)
            
            for date in dates:
                # Add daily pattern (lower at night, higher during day)
                hour = date.hour
                daily_factor = 1.0 + 0.2 * abs(np.sin(2 * np.pi * hour / 24))
                
                # Add weekly pattern (lower on weekends)
                weekday_factor = 0.9 if date.weekday() >= 5 else 1.0
                
                # Add seasonal pattern (lower in summer for some regions)
                month = date.month
                seasonal_factor = 1.0 + 0.1 * np.sin(2 * np.pi * month / 12)
                
                # Add random variation
                random_factor = np.random.normal(1.0, 0.1)
                
                ci = base_ci * daily_factor * weekday_factor * seasonal_factor * random_factor
                ci = max(0, ci)  # Ensure non-negative
                
                data.append({
                    "timestamp": date,
                    "location": location,
                    "carbon_intensity": ci,
                    "renewable_share": max(0, min(1, 1 - (ci / 800)))
                })
        
        df = pd.DataFrame(data)
        filepath = self.data_dir / "sample_grid_data.csv"
        df.to_csv(filepath, index=False)
        
        logger.info(f"Created sample grid data: {filepath}")
        return filepath


def download_all_datasets(data_dir: str = "data/raw"):
    """
    Download all available datasets.
    
    Args:
        data_dir: Directory to store data
    """
    downloader = DatasetDownloader(data_dir)
    
    print("Downloading datasets...")
    print("=" * 60)
    
    # Download EPA eGRID
    print("\n1. EPA eGRID (US Grid Data)")
    egrid_path = downloader.download_epa_egrid(year=2022)
    if egrid_path:
        print(f"   ✅ Downloaded: {egrid_path}")
    else:
        print("   ⚠️  Download failed - check URL or download manually")
        print("   Visit: https://www.epa.gov/egrid/download-data")
    
    # Download OPSD
    print("\n2. Open Power System Data (European Grid Data)")
    opsd_path = downloader.download_opsd_data("time_series")
    if opsd_path:
        print(f"   ✅ Downloaded: {opsd_path}")
    else:
        print("   ⚠️  Download failed - check URL or download manually")
        print("   Visit: https://open-power-system-data.org/")
    
    # Create sample data
    print("\n3. Sample Grid Data (Synthetic)")
    from datetime import datetime, timedelta
    sample_path = downloader.create_sample_grid_data(
        locations=["US", "EU", "DE", "FR", "GB"],
        start_date=datetime.now() - timedelta(days=365),
        end_date=datetime.now()
    )
    print(f"   ✅ Created: {sample_path}")
    
    print("\n" + "=" * 60)
    print("Download complete!")
    print(f"Data stored in: {data_dir}")

