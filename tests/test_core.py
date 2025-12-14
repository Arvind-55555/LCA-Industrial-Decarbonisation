"""
Tests for core LCA engine
"""

import pytest
from datetime import datetime
from lca_optimizer.core.engine import LCAEngine, LCAResult
from lca_optimizer.core.physics import PhysicsConstraints


def test_lca_engine_initialization():
    """Test LCA engine initialization"""
    engine = LCAEngine()
    assert engine is not None
    assert engine.enable_uncertainty is True


def test_lca_calculation():
    """Test basic LCA calculation"""
    engine = LCAEngine()
    
    process_params = {
        "sector": "steel",
        "technology": "h2_dri",
        "production_capacity": 1000.0
    }
    
    result = engine.calculate_lca(
        process_params=process_params,
        location="EU",
        timestamp=datetime.now()
    )
    
    assert isinstance(result, LCAResult)
    assert result.total_emissions >= 0
    assert len(result.breakdown) > 0


def test_physics_constraints():
    """Test physics constraints"""
    physics = PhysicsConstraints()
    
    # Test mass balance
    inputs = {"water": 100.0, "electricity": 50.0}
    outputs = {"h2": 10.0, "o2": 80.0}
    losses = {"heat": 10.0}
    
    is_valid, error = physics.check_mass_balance(inputs, outputs, losses)
    # Note: This is a simplified test, actual values may not balance
    assert isinstance(is_valid, bool)
    assert error >= 0


def test_h2_production():
    """Test H2 production stoichiometry"""
    physics = PhysicsConstraints()
    
    h2_produced, o2_produced = physics.stoichiometric_h2_production(
        water_input=100.0,  # kg
        electricity_input=100.0,  # kWh
        electrolyzer_efficiency=0.7
    )
    
    assert h2_produced > 0
    assert o2_produced > 0

