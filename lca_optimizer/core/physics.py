"""
Physics Constraints: Thermodynamic and stoichiometric constraints for LCA
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class MassBalance:
    """Mass balance constraint"""
    inputs: Dict[str, float]  # Input materials (kg)
    outputs: Dict[str, float]  # Output materials (kg)
    losses: Dict[str, float]  # Process losses (kg)


@dataclass
class EnergyBalance:
    """Energy balance constraint"""
    inputs: Dict[str, float]  # Input energy (MJ)
    outputs: Dict[str, float]  # Output energy (MJ)
    efficiency: float  # Process efficiency


class PhysicsConstraints:
    """
    Physics-informed constraints for LCA modeling.
    
    Ensures predictions obey:
    - Mass balance equations
    - Energy balance equations
    - Stoichiometric constraints
    - Thermodynamic limits
    """
    
    # Thermodynamic constants
    H2_LHV = 120.0  # MJ/kg (Lower Heating Value)
    H2_HHV = 142.0  # MJ/kg (Higher Heating Value)
    CO2_MOLAR_MASS = 44.01  # g/mol
    H2_MOLAR_MASS = 2.016  # g/mol
    
    def __init__(self):
        """Initialize physics constraints"""
        self.constraints = []
        logger.info("Physics constraints initialized")
    
    def check_mass_balance(
        self,
        inputs: Dict[str, float],
        outputs: Dict[str, float],
        losses: Optional[Dict[str, float]] = None
    ) -> Tuple[bool, float]:
        """
        Check mass balance: Σ inputs = Σ outputs + Σ losses
        
        Returns:
            (is_valid, error)
        """
        total_input = sum(inputs.values())
        total_output = sum(outputs.values())
        total_loss = sum(losses.values()) if losses else 0.0
        
        error = abs(total_input - (total_output + total_loss))
        is_valid = error < 1e-6
        
        return is_valid, error
    
    def check_energy_balance(
        self,
        energy_input: float,
        energy_output: float,
        efficiency: float,
        tolerance: float = 0.01
    ) -> Tuple[bool, float]:
        """
        Check energy balance: output = input * efficiency
        
        Returns:
            (is_valid, error)
        """
        expected_output = energy_input * efficiency
        error = abs(energy_output - expected_output) / energy_input
        is_valid = error < tolerance
        
        return is_valid, error
    
    def stoichiometric_h2_production(
        self,
        water_input: float,  # kg H2O
        electricity_input: float,  # kWh
        electrolyzer_efficiency: float = 0.7
    ) -> Tuple[float, float]:
        """
        Calculate H2 production from water electrolysis.
        
        Reaction: 2H2O -> 2H2 + O2
        Theoretical: 1 kg H2 requires 9 kg H2O and ~33.3 kWh
        
        Returns:
            (h2_produced_kg, oxygen_produced_kg)
        """
        # Theoretical H2 from water
        h2_from_water = water_input / 9.0  # kg H2
        
        # Actual H2 considering electrolyzer efficiency
        h2_produced = h2_from_water * electrolyzer_efficiency
        
        # Oxygen production (stoichiometric)
        oxygen_produced = h2_produced * 8.0  # kg O2
        
        return h2_produced, oxygen_produced
    
    def h2_combustion_emissions(
        self,
        h2_amount: float  # kg H2
    ) -> float:
        """
        Calculate CO2 emissions from H2 combustion.
        
        Note: Pure H2 combustion produces no CO2, only H2O.
        This is for H2-derived fuels or H2 with impurities.
        
        Returns:
            CO2 emissions (kg CO2)
        """
        # Pure H2: zero CO2 emissions
        return 0.0
    
    def co2_capture_rate(
        self,
        co2_input: float,  # kg CO2
        capture_technology: str,
        capture_efficiency: Optional[float] = None
    ) -> float:
        """
        Calculate captured CO2 based on technology.
        
        Args:
            co2_input: Input CO2 (kg)
            capture_technology: "post-combustion", "oxy-fuel", "pre-combustion"
            capture_efficiency: Override default efficiency
        
        Returns:
            Captured CO2 (kg)
        """
        default_efficiencies = {
            "post-combustion": 0.90,
            "oxy-fuel": 0.95,
            "pre-combustion": 0.85
        }
        
        efficiency = capture_efficiency or default_efficiencies.get(
            capture_technology, 0.90
        )
        
        return co2_input * efficiency
    
    def steel_dri_reaction(
        self,
        iron_ore: float,  # kg Fe2O3
        h2_input: float,  # kg H2
        reduction_efficiency: float = 0.95
    ) -> Tuple[float, float, float]:
        """
        Calculate H2-DRI reaction products.
        
        Reaction: Fe2O3 + 3H2 -> 2Fe + 3H2O
        
        Returns:
            (iron_produced_kg, water_produced_kg, h2_consumed_kg)
        """
        # Stoichiometric ratios
        # Fe2O3 (159.7 g/mol) + 3H2 (6.048 g/mol) -> 2Fe (111.7 g/mol) + 3H2O
        
        # Theoretical H2 requirement: 3 * 2.016 / 159.7 = 0.0379 kg H2 per kg Fe2O3
        h2_required = iron_ore * 0.0379
        
        # Actual H2 consumption considering efficiency
        h2_consumed = min(h2_input, h2_required / reduction_efficiency)
        
        # Iron produced
        iron_produced = (h2_consumed * reduction_efficiency) / 0.0379 * (111.7 / 159.7)
        
        # Water produced
        water_produced = h2_consumed * 9.0  # 1 kg H2 -> 9 kg H2O
        
        return iron_produced, water_produced, h2_consumed
    
    def apply_constraints_to_prediction(
        self,
        prediction: torch.Tensor,
        constraints: List[callable]
    ) -> torch.Tensor:
        """
        Apply physics constraints to neural network predictions.
        
        Args:
            prediction: Raw NN prediction
            constraints: List of constraint functions
        
        Returns:
            Constrained prediction
        """
        constrained = prediction.clone()
        
        for constraint_fn in constraints:
            constrained = constraint_fn(constrained)
        
        return constrained

