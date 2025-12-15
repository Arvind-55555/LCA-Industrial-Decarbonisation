#!/usr/bin/env python3
"""
Example: Indian Industrial Decarbonisation LCA Platform
Demonstrates usage of Indian-specific sector optimizers and ML models
"""

import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from datetime import datetime

from lca_optimizer.core.engine import LCAEngine
from lca_optimizer.sectors.indian_steel import IndianSteelOptimizer, IndianSteelProcessConfig
from lca_optimizer.sectors.indian_cement import IndianCementOptimizer, IndianCementProcessConfig
from lca_optimizer.sectors.indian_aluminium import IndianAluminiumOptimizer, IndianAluminiumProcessConfig
from lca_optimizer.data.indian_grid_data import IndianGridDataLoader
from lca_optimizer.config.indian_settings import get_indian_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_indian_steel():
    """Example: Indian Steel Sector LCA"""
    logger.info("=" * 60)
    logger.info("Example: Indian Steel Sector LCA")
    logger.info("=" * 60)
    
    engine = LCAEngine()
    optimizer = IndianSteelOptimizer(engine, state="Maharashtra")
    
    config = IndianSteelProcessConfig(
        process_type="DRI",
        state="Maharashtra",
        iron_ore_source="Odisha",
        coal_type="Indian coal",
        production_capacity=2000000.0,  # 2 million tonnes/year
        grid_dependency=0.75
    )
    
    result = optimizer.optimize(config)
    
    logger.info(f"\nResults:")
    logger.info(f"  Initial LCA: {result['initial_lca']:,.0f} kg CO2eq")
    logger.info(f"  Optimal LCA: {result['optimal_lca']:,.0f} kg CO2eq")
    logger.info(f"  Baseline: {result['baseline_emissions']:,.0f} kg CO2eq")
    logger.info(f"  Emission Reduction: {result['emission_reduction']:.1f}%")
    logger.info(f"  Grid CI: {result['grid_ci']:.2f} kg CO2/kWh")
    logger.info(f"  Grid Reliability: {result['indian_specific_factors']['grid_reliability']:.2f}")
    logger.info(f"  Backup Power Required: {result['indian_specific_factors']['backup_power_required']:.2f} MW")
    
    return result


def example_indian_cement():
    """Example: Indian Cement Sector LCA"""
    logger.info("\n" + "=" * 60)
    logger.info("Example: Indian Cement Sector LCA")
    logger.info("=" * 60)
    
    engine = LCAEngine()
    optimizer = IndianCementOptimizer(engine, state="Rajasthan")
    
    config = IndianCementProcessConfig(
        state="Rajasthan",
        production_capacity=5000000.0,  # 5 million tonnes/year
        clinker_ratio=0.75,
        fly_ash_substitution=0.25,
        capture_technology="post-combustion",
        capture_rate=0.90
    )
    
    result = optimizer.optimize(config)
    
    logger.info(f"\nResults:")
    logger.info(f"  Initial LCA: {result['initial_lca']:,.0f} kg CO2eq")
    logger.info(f"  Optimal LCA: {result['optimal_lca']:,.0f} kg CO2eq")
    logger.info(f"  Emission Reduction: {result['emission_reduction']:.1f}%")
    logger.info(f"  Effective Clinker Ratio: {result['indian_factors']['effective_clinker_ratio']:.2f}")
    logger.info(f"  Fly Ash Substitution: {result['indian_factors']['fly_ash_substitution']:.2f}")
    
    return result


def example_indian_aluminium():
    """Example: Indian Aluminium Sector LCA"""
    logger.info("\n" + "=" * 60)
    logger.info("Example: Indian Aluminium Sector LCA")
    logger.info("=" * 60)
    
    engine = LCAEngine()
    optimizer = IndianAluminiumOptimizer(engine, state="Odisha")
    
    config = IndianAluminiumProcessConfig(
        state="Odisha",
        production_capacity=500000.0,  # 500k tonnes/year
        smelting_technology="pre_baked",
        grid_dependency=0.85
    )
    
    result = optimizer.optimize(config)
    
    logger.info(f"\nResults:")
    logger.info(f"  Initial LCA: {result['initial_lca']:,.0f} kg CO2eq")
    logger.info(f"  Optimal LCA: {result['optimal_lca']:,.0f} kg CO2eq")
    logger.info(f"  Emission Reduction: {result['emission_reduction']:.1f}%")
    logger.info(f"  Grid CI: {result['grid_ci']:.2f} kg CO2/kWh")
    logger.info(f"  Specific Energy: {result['indian_factors']['specific_energy_mwh_per_tonne']:.2f} MWh/t")
    logger.info(f"  Grid Dependency: {result['indian_factors']['grid_dependency']:.2f}")
    
    return result


def example_indian_grid_data():
    """Example: Indian Grid Carbon Intensity Data"""
    logger.info("\n" + "=" * 60)
    logger.info("Example: Indian Grid Carbon Intensity")
    logger.info("=" * 60)
    
    grid_loader = IndianGridDataLoader()
    
    states = ["Maharashtra", "Gujarat", "Tamil Nadu", "Odisha"]
    
    for state in states:
        ci = grid_loader.get_carbon_intensity(state)
        stats = grid_loader.get_state_statistics(state)
        
        logger.info(f"\n{state}:")
        logger.info(f"  Current CI: {ci.carbon_intensity:.2f} kg CO2/kWh")
        if stats:
            logger.info(f"  Mean CI: {stats.get('mean', 0):.2f} kg CO2/kWh")
            logger.info(f"  Renewable Share: {stats.get('renewable_share', 0):.1%}")


def main():
    """Run all examples"""
    logger.info("Indian Industrial Decarbonisation LCA Platform - Examples")
    logger.info("=" * 60)
    
    # Show Indian settings
    settings = get_indian_settings()
    logger.info(f"\nDefault State: {settings.default_state}")
    logger.info(f"Indian Steel DRI Share: {settings.indian_steel_dri_share:.1%}")
    logger.info(f"Indian Cement Clinker Ratio: {settings.indian_cement_clinker_ratio:.2f}")
    logger.info(f"Indian Aluminium Grid Dependency: {settings.indian_aluminium_grid_dependency:.1%}")
    
    # Run examples
    try:
        example_indian_steel()
        example_indian_cement()
        example_indian_aluminium()
        example_indian_grid_data()
        
        logger.info("\n" + "=" * 60)
        logger.info("All examples completed successfully!")
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"Error running examples: {e}", exc_info=True)


if __name__ == "__main__":
    main()

