#!/usr/bin/env python3
"""
Example: Using Real API Integrations
Demonstrates how to use real APIs with the LCA Optimizer
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timedelta
from lca_optimizer.data.grid_data_enhanced import ElectricityMapsLoader, WattTimeLoader
from lca_optimizer.data.greet_integration import GREETIntegration
from lca_optimizer.config.settings import get_settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_electricity_maps():
    """Example: Using Electricity Maps API"""
    print("=" * 60)
    print("Electricity Maps API Example")
    print("=" * 60)
    
    # Get API key from settings or environment
    settings = get_settings()
    api_key = settings.electricity_maps_api_key
    
    if not api_key:
        print("\n⚠️  No API key found. Set ELECTRICITY_MAPS_API_KEY in .env file")
        print("   Get your API key from: https://www.electricitymaps.com/")
        return
    
    # Initialize loader
    loader = ElectricityMapsLoader(api_key=api_key)
    
    # Get current carbon intensity for Germany
    print("\n1. Current Carbon Intensity (Germany):")
    ci = loader.get_current_carbon_intensity("DE")
    print(f"   Location: {ci.location}")
    print(f"   Carbon Intensity: {ci.carbon_intensity} g CO2eq/kWh")
    print(f"   Renewable Share: {ci.renewable_share * 100:.1f}%")
    print(f"   Timestamp: {ci.timestamp}")
    
    # Get historical data
    print("\n2. Historical Data (Last 24 hours):")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1)
    
    try:
        historical = loader.get_historical_carbon_intensity(
            "DE", start_date, end_date, frequency="hourly"
        )
        print(f"   Retrieved {len(historical)} data points")
        print(f"   Average CI: {historical['carbon_intensity'].mean():.1f} g CO2eq/kWh")
        print(f"   Min CI: {historical['carbon_intensity'].min():.1f} g CO2eq/kWh")
        print(f"   Max CI: {historical['carbon_intensity'].max():.1f} g CO2eq/kWh")
    except Exception as e:
        print(f"   Error: {e}")


def example_watttime():
    """Example: Using WattTime API"""
    print("\n" + "=" * 60)
    print("WattTime API Example")
    print("=" * 60)
    
    settings = get_settings()
    username = settings.watttime_username
    password = settings.watttime_password
    
    if not username or not password:
        print("\n⚠️  No credentials found. Set WATTTIME_USERNAME and WATTTIME_PASSWORD in .env")
        print("   Sign up at: https://www.watttime.org/")
        return
    
    # Initialize loader
    loader = WattTimeLoader(username=username, password=password)
    
    # Get current carbon intensity
    print("\n1. Current Carbon Intensity (CAISO - California):")
    ci = loader.get_current_carbon_intensity("CAISO")
    print(f"   Location: {ci.location}")
    print(f"   Carbon Intensity: {ci.carbon_intensity} g CO2eq/kWh")
    print(f"   Timestamp: {ci.timestamp}")


def example_greet():
    """Example: Using GREET Database"""
    print("\n" + "=" * 60)
    print("GREET Database Example")
    print("=" * 60)
    
    # Initialize GREET integration
    greet = GREETIntegration()
    
    # Compare fuel pathways
    print("\n1. Fuel Pathway Comparison:")
    pathways = [
        "hydrogen_electrolysis_wind",
        "hydrogen_electrolysis_grid",
        "hydrogen_steam_reforming",
        "diesel",
        "electricity_grid_eu"
    ]
    
    comparison = greet.compare_pathways(pathways)
    print(comparison.to_string(index=False))
    
    # Get specific pathway emissions
    print("\n2. Green Hydrogen Pathway:")
    h2_data = greet.get_wtw_emissions("hydrogen_electrolysis_wind")
    print(f"   Well-to-Wheel: {h2_data['well_to_wheel']} g CO2eq/MJ")
    print(f"   Well-to-Tank: {h2_data['well_to_tank']} g CO2eq/MJ")
    print(f"   Tank-to-Wheel: {h2_data['tank_to_wheel']} g CO2eq/MJ")


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("Real API Integration Examples")
    print("=" * 60)
    print("\nThis script demonstrates how to use real APIs with the LCA Optimizer.")
    print("Make sure to set up your API keys in the .env file.\n")
    
    # Run examples
    example_electricity_maps()
    example_watttime()
    example_greet()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

