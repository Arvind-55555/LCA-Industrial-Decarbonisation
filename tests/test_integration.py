"""
Integration tests for LCA Optimizer
"""

import pytest
from datetime import datetime, timedelta
from lca_optimizer.core.engine import LCAEngine
from lca_optimizer.sectors.steel import SteelH2DRIOptimizer, SteelProcessConfig
from lca_optimizer.data.local_data_loader import LocalGridDataLoader
from lca_optimizer.data.validation import DataValidator


def test_end_to_end_steel_optimization():
    """Test complete steel optimization workflow"""
    engine = LCAEngine()
    optimizer = SteelH2DRIOptimizer(engine)
    
    config = SteelProcessConfig(
        h2_pathway="electrolysis",
        electrolyzer_type="PEM",
        renewable_mix={"wind": 0.6, "solar": 0.4},
        iron_ore_source="Australia",
        process_heat_source="electric",
        location="US",
        production_capacity=1000000.0
    )
    
    result = optimizer.optimize(config)
    
    assert "initial_lca" in result
    assert "optimal_lca" in result
    assert "emission_reduction" in result
    assert result["optimal_lca"] >= 0
    assert result["emission_reduction"] >= 0


def test_local_data_loader():
    """Test local data loader"""
    loader = LocalGridDataLoader(data_dir="data/raw")
    
    ci = loader.get_current_carbon_intensity("US")
    assert ci.carbon_intensity > 0
    assert ci.location == "US"
    
    # Test historical data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1)
    
    historical = loader.get_historical_carbon_intensity(
        "US", start_date, end_date
    )
    assert not historical.empty
    assert "carbon_intensity" in historical.columns


def test_data_validation():
    """Test data validation"""
    validator = DataValidator()
    
    # Create test data
    import pandas as pd
    test_data = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=10, freq="h"),
        "carbon_intensity": [300.0] * 10
    })
    
    is_valid, errors = validator.validate_grid_data(test_data)
    assert is_valid
    assert len(errors) == 0
    
    # Test with invalid data
    invalid_data = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=10, freq="h"),
        "carbon_intensity": [-100.0] * 10
    })
    
    is_valid, errors = validator.validate_grid_data(invalid_data)
    assert not is_valid
    assert len(errors) > 0

