#!/usr/bin/env python3
"""
Test Local Data Loader
Demonstrates using downloaded datasets instead of API keys
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timedelta
from lca_optimizer.data.local_data_loader import LocalGridDataLoader
from lca_optimizer.core.engine import LCAEngine
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_local_data():
    """Test local data loader"""
    print("=" * 60)
    print("Testing Local Data Loader (No API Keys Required!)")
    print("=" * 60)
    
    # Initialize local loader
    loader = LocalGridDataLoader(data_dir="data/raw")
    
    # Test locations
    locations = ["US", "EU", "DE", "FR", "GB"]
    
    print("\n1. Current Carbon Intensity:")
    print("-" * 60)
    for location in locations:
        ci = loader.get_current_carbon_intensity(location)
        print(f"   {location:4s}: {ci.carbon_intensity:6.1f} g CO2eq/kWh "
              f"(Source: {ci.source})")
    
    # Test historical data
    print("\n2. Historical Data (Last 7 days):")
    print("-" * 60)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    for location in ["US", "DE"]:
        historical = loader.get_historical_carbon_intensity(
            location, start_date, end_date, frequency="hourly"
        )
        if not historical.empty:
            print(f"\n   {location}:")
            print(f"     Records: {len(historical)}")
            print(f"     Average CI: {historical['carbon_intensity'].mean():.1f} g CO2eq/kWh")
            print(f"     Min CI: {historical['carbon_intensity'].min():.1f} g CO2eq/kWh")
            print(f"     Max CI: {historical['carbon_intensity'].max():.1f} g CO2eq/kWh")
    
    # Test with LCA Engine
    print("\n3. Integration with LCA Engine:")
    print("-" * 60)
    engine = LCAEngine()
    ci = engine.get_grid_carbon_intensity("US", datetime.now())
    print(f"   LCA Engine using local data: {ci} g CO2eq/kWh")
    
    print("\n" + "=" * 60)
    print("✅ Local data loader test complete!")
    print("=" * 60)
    print("\nNo API keys required - all data from local files!")


if __name__ == "__main__":
    test_local_data()

