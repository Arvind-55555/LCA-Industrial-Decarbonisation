"""
Indian Aluminium Sector: Smelting LCA Optimizer
Accounts for high grid dependency and state-specific carbon intensity
"""

import numpy as np
from typing import Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import logging

from lca_optimizer.core.engine import LCAEngine, LCAResult
from lca_optimizer.core.indian_physics import IndianPhysicsConstraints
from lca_optimizer.data.indian_grid_data import IndianGridDataLoader
from lca_optimizer.data.indian_data_loader import IndianDataLoader
from lca_optimizer.config.indian_settings import get_indian_settings

logger = logging.getLogger(__name__)


@dataclass
class IndianAluminiumProcessConfig:
    """Configuration for Indian aluminium smelting"""
    state: str
    production_capacity: float  # tonnes/year
    smelting_technology: str = "pre_baked"
    grid_dependency: Optional[float] = None
    renewable_share: Optional[float] = None


class IndianAluminiumOptimizer:
    """
    Optimizer for Indian aluminium smelting LCA.
    
    Accounts for:
    - High grid dependency (85%+)
    - State-specific grid carbon intensity variations
    - Higher specific energy consumption (13.5-15.5 MWh/t)
    - Limited captive renewable capacity
    """
    
    def __init__(self, lca_engine: LCAEngine, state: Optional[str] = None):
        self.engine = lca_engine
        self.settings = get_indian_settings()
        self.state = state or self.settings.default_state
        self.physics = IndianPhysicsConstraints(state=self.state)
        self.grid_loader = IndianGridDataLoader()
        self.data_loader = IndianDataLoader()
        
        logger.info(f"Indian Aluminium Optimizer initialized for state: {self.state}")
    
    def optimize(self, config: IndianAluminiumProcessConfig) -> Dict[str, Any]:
        """Optimize Indian aluminium smelting"""
        lca_result = self.calculate_lca(config)
        
        # Get grid CI
        grid_ci = self.grid_loader.get_carbon_intensity(config.state)
        grid_ci_value = grid_ci.carbon_intensity if grid_ci else 0.90
        
        # Apply Indian aluminium constraints
        aluminium_params = self.physics.indian_aluminium_smelting_constraint(
            grid_carbon_intensity=grid_ci_value,
            smelting_technology=config.smelting_technology
        )
        
        # Optimize (reduce grid dependency, increase renewable)
        optimal_config = IndianAluminiumProcessConfig(
            state=config.state,
            production_capacity=config.production_capacity,
            smelting_technology=config.smelting_technology,
            grid_dependency=max(0.70, (config.grid_dependency or 0.85) - 0.15),
            renewable_share=min(0.40, (config.renewable_share or 0.10) + 0.20)
        )
        
        optimal_lca = self.calculate_lca(optimal_config)
        
        baseline_emissions = config.production_capacity * 12.0 * 1000  # tCO2 -> kg CO2eq
        emission_reduction = (
            (baseline_emissions - optimal_lca.total_emissions) / baseline_emissions * 100
        )
        
        return {
            "initial_lca": lca_result.total_emissions,
            "optimal_lca": optimal_lca.total_emissions,
            "emission_reduction": emission_reduction,
            "breakdown": optimal_lca.breakdown,
            "grid_ci": grid_ci_value,
            "indian_factors": aluminium_params
        }
    
    def calculate_lca(self, config: IndianAluminiumProcessConfig) -> LCAResult:
        """Calculate LCA for Indian aluminium smelting"""
        # Get process data
        process_data = self.data_loader.get_aluminium_process_data(state=config.state)
        
        if not process_data.empty:
            plant_data = process_data.iloc[0]
            sec = plant_data.get('specific_energy_consumption_mwh_per_tonne', 14.5)
            grid_ci = plant_data.get('grid_carbon_intensity_kg_co2_per_kwh', 0.90)
        else:
            sec = 14.5
            grid_ci_obj = self.grid_loader.get_carbon_intensity(config.state)
            grid_ci = grid_ci_obj.carbon_intensity if grid_ci_obj else 0.90
        
        # Apply Indian constraints
        aluminium_params = self.physics.indian_aluminium_smelting_constraint(
            grid_carbon_intensity=grid_ci,
            smelting_technology=config.smelting_technology
        )
        
        # Calculate emissions
        grid_dependency = config.grid_dependency or self.settings.indian_aluminium_grid_dependency
        electricity_consumption = config.production_capacity * sec  # MWh/year
        
        grid_emissions = electricity_consumption * grid_ci * grid_dependency * 1000
        captive_emissions = (
            electricity_consumption * (1 - grid_dependency) *
            aluminium_params["captive_emissions"] / config.production_capacity * 1000
        )
        
        total_emissions = grid_emissions + captive_emissions
        
        from datetime import datetime as _dt
        return LCAResult(
            total_emissions=total_emissions,
            uncertainty=(total_emissions * 0.82, total_emissions * 1.18),
            breakdown={
                "grid_emissions": grid_emissions,
                "captive_emissions": captive_emissions
            },
            metadata={
                "sector": "aluminium",
                "state": config.state,
                "smelting_technology": config.smelting_technology,
                "grid_dependency": grid_dependency
            },
            timestamp=_dt.now()
        )

