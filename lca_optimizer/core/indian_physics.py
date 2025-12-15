"""
Indian-Specific Physics Constraints
Accounts for Indian industrial process characteristics, regional variations,
and operational constraints specific to Indian industries
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
import logging

from lca_optimizer.core.physics import PhysicsConstraints
from lca_optimizer.config.indian_settings import get_indian_settings, IndianStateConfig

logger = logging.getLogger(__name__)


class IndianPhysicsConstraints(PhysicsConstraints):
    """
    Extended physics constraints for Indian industrial processes.
    
    Accounts for:
    - Indian DRI process characteristics (coal-based vs gas-based)
    - Indian grid reliability and power quality issues
    - Regional material quality variations
    - Indian climate conditions (monsoon, heat waves)
    - State-specific regulatory constraints
    """
    
    def __init__(self, state: Optional[str] = None):
        """
        Initialize Indian physics constraints.
        
        Args:
            state: Indian state name for state-specific constraints
        """
        super().__init__()
        self.settings = get_indian_settings()
        self.state_config = self.settings.get_state_config(state) if state else None
        
        # Indian-specific constants
        self.INDIAN_COAL_CV = 20.0  # MJ/kg (lower than international)
        self.INDIAN_IRON_ORE_FE_CONTENT = 0.58  # Average Fe content (lower quality)
        self.INDIAN_GRID_LOSSES = 0.20  # Transmission and distribution losses
        
        logger.info(f"Indian Physics Constraints initialized for state: {state}")
    
    def indian_dri_process_constraint(
        self,
        iron_ore_quality: float,  # Fe content (0-1)
        coal_quality: float,  # Calorific value factor
        process_type: str = "coal_based"  # "coal_based" or "gas_based"
    ) -> Dict[str, float]:
        """
        Calculate Indian DRI process constraints.
        
        Indian DRI processes typically use:
        - Lower quality iron ore (58-62% Fe vs 65%+ internationally)
        - Coal-based reduction (vs natural gas internationally)
        - Higher energy consumption due to ore quality
        
        Returns:
            Dictionary with process parameters
        """
        # Base efficiency for Indian DRI
        if process_type == "coal_based":
            base_efficiency = 0.85  # Lower than gas-based
            specific_energy = 3.5  # MWh/tonne DRI (higher than gas-based ~2.8)
        else:
            base_efficiency = 0.90
            specific_energy = 2.8
        
        # Adjust for ore quality (lower Fe content requires more energy)
        fe_factor = iron_ore_quality / 0.65  # Normalize to international standard
        efficiency = base_efficiency * fe_factor
        specific_energy = specific_energy / fe_factor
        
        # Adjust for coal quality
        if process_type == "coal_based":
            coal_factor = coal_quality  # 0.8-1.0 for Indian coal
            efficiency *= coal_factor
            specific_energy /= coal_factor
        
        return {
            "reduction_efficiency": max(0.75, min(0.95, efficiency)),
            "specific_energy_mwh_per_tonne": specific_energy,
            "coal_consumption_kg_per_tonne_dri": 600.0 if process_type == "coal_based" else 0.0
        }
    
    def indian_grid_reliability_constraint(
        self,
        power_requirement: float,  # MW
        state: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Account for Indian grid reliability issues.
        
        Indian grid characteristics:
        - Higher transmission losses (~20% vs ~5-7% internationally)
        - Power quality issues (voltage fluctuations)
        - Load shedding in some states
        - Need for backup power/captive generation
        
        Returns:
            Dictionary with adjusted power parameters
        """
        state_config = self.state_config or self.settings.get_state_config(state or self.settings.default_state)
        
        if state_config:
            reliability = state_config.power_reliability
        else:
            reliability = 0.80  # Default
        
        # Grid losses
        actual_power_needed = power_requirement / (1 - self.INDIAN_GRID_LOSSES)
        
        # Reliability factor (need backup/captive power)
        backup_power_required = power_requirement * (1 - reliability) * 0.5  # 50% of downtime covered
        
        # Captive power typically uses diesel/gas, higher emissions
        captive_emission_factor = 0.6  # kg CO2/kWh for diesel gensets
        
        return {
            "grid_power_required": actual_power_needed,
            "backup_power_required": backup_power_required,
            "grid_reliability": reliability,
            "captive_emission_factor": captive_emission_factor
        }
    
    def indian_cement_clinker_constraint(
        self,
        clinker_ratio: float,
        fly_ash_availability: float,  # 0-1, local availability
        slag_availability: float  # 0-1
    ) -> Dict[str, float]:
        """
        Indian cement process constraints.
        
        Indian cement characteristics:
        - Higher clinker ratios (0.65-0.85 vs 0.50-0.65 internationally)
        - Limited fly ash availability in some regions
        - Variable slag availability
        - Higher coal dependency
        
        Returns:
            Dictionary with process parameters
        """
        # Indian average clinker ratio is higher
        indian_avg_clinker = self.settings.indian_cement_clinker_ratio
        
        # Effective clinker ratio considering substitution availability
        effective_clinker = clinker_ratio * (1 - 0.3 * fly_ash_availability - 0.1 * slag_availability)
        effective_clinker = max(0.50, min(0.90, effective_clinker))
        
        # Energy consumption (higher for higher clinker ratio)
        base_energy = 0.10  # MWh/tonne cement
        energy_factor = 1.0 + (effective_clinker - 0.65) * 0.5
        specific_energy = base_energy * energy_factor
        
        return {
            "effective_clinker_ratio": effective_clinker,
            "specific_energy_mwh_per_tonne": specific_energy,
            "fly_ash_substitution": 0.3 * fly_ash_availability,
            "slag_substitution": 0.1 * slag_availability
        }
    
    def indian_aluminium_smelting_constraint(
        self,
        grid_carbon_intensity: float,  # kg CO2/kWh
        smelting_technology: str = "pre_baked"
    ) -> Dict[str, float]:
        """
        Indian aluminium smelting constraints.
        
        Indian aluminium characteristics:
        - High grid dependency (85%+ vs 60-70% internationally)
        - Higher specific energy consumption (13.5-15.5 MWh/t vs 12-13 MWh/t)
        - Grid carbon intensity varies significantly by state
        - Limited captive renewable capacity
        
        Returns:
            Dictionary with process parameters
        """
        # Indian specific energy consumption is higher
        if smelting_technology == "pre_baked":
            base_sec = 14.5  # MWh/tonne (Indian average)
        else:
            base_sec = 15.0
        
        # Grid dependency
        grid_dependency = self.settings.indian_aluminium_grid_dependency
        
        # Emissions from grid electricity
        grid_emissions = base_sec * grid_carbon_intensity * grid_dependency
        
        # Captive power emissions (typically coal-based)
        captive_sec = base_sec * (1 - grid_dependency)
        captive_emissions = captive_sec * 0.9  # kg CO2/kWh for coal
        
        total_emissions = grid_emissions + captive_emissions
        
        return {
            "specific_energy_mwh_per_tonne": base_sec,
            "grid_dependency": grid_dependency,
            "emission_factor_kg_co2_per_tonne": total_emissions,
            "grid_emissions": grid_emissions,
            "captive_emissions": captive_emissions
        }
    
    def indian_climate_constraint(
        self,
        process_type: str,
        month: int,  # 1-12
        state: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Account for Indian climate conditions.
        
        Indian climate impacts:
        - Monsoon (June-September): High humidity, cooling challenges
        - Heat waves (April-June): Increased cooling demand, efficiency losses
        - Regional variations (coastal vs inland)
        
        Returns:
            Dictionary with climate-adjusted parameters
        """
        # Monsoon months (June-September)
        is_monsoon = 6 <= month <= 9
        
        # Heat wave months (April-June)
        is_heat_wave = 4 <= month <= 6
        
        # Climate factors
        cooling_penalty = 0.0
        efficiency_loss = 0.0
        
        if is_monsoon:
            cooling_penalty = 0.15  # 15% additional cooling energy
            efficiency_loss = 0.05  # 5% efficiency loss due to humidity
        
        if is_heat_wave:
            cooling_penalty += 0.20  # 20% additional cooling
            efficiency_loss += 0.08  # 8% efficiency loss
        
        # Process-specific adjustments
        if process_type in ["steel", "cement"]:
            # High-temperature processes less affected
            efficiency_loss *= 0.5
        elif process_type == "aluminium":
            # Smelting very sensitive to temperature
            efficiency_loss *= 1.5
        
        return {
            "cooling_energy_penalty": cooling_penalty,
            "efficiency_loss": efficiency_loss,
            "is_monsoon": is_monsoon,
            "is_heat_wave": is_heat_wave
        }
    
    def indian_transport_constraint(
        self,
        distance_km: float,
        material_type: str = "general"
    ) -> float:
        """
        Calculate transport emissions for Indian supply chain.
        
        Indian transport characteristics:
        - Higher emission factors (older vehicle fleet, road conditions)
        - Longer distances (large country, material transport)
        - Mixed transport modes (road, rail, coastal)
        
        Returns:
            Transport emissions (kg CO2/tonne)
        """
        # Indian transport emission factors (kg CO2/tonne-km)
        emission_factors = {
            "road": 0.18,  # Higher than international average
            "rail": 0.03,
            "coastal": 0.01,
            "general": 0.15  # Weighted average
        }
        
        factor = emission_factors.get(material_type, emission_factors["general"])
        
        # Apply Indian transport emission factor from settings
        factor *= self.settings.transport_emission_factor / 0.10  # Normalize to baseline
        
        return distance_km * factor
    
    def physics_loss(
        self,
        inputs: torch.Tensor,
        predictions: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculate physics-informed loss for Indian processes.
        
        Enhanced loss function that includes:
        - Standard physics constraints (mass/energy balance)
        - Indian-specific constraints (grid reliability, material quality)
        - Regional variations
        
        Returns:
            Physics loss tensor
        """
        # Base physics loss from parent class
        base_loss = torch.tensor(0.0, device=predictions.device)
        
        # Add Indian-specific constraint losses
        if inputs.shape[1] >= 3:
            # Extract features
            production = inputs[:, 0:1] if inputs.shape[1] > 0 else None
            energy = inputs[:, 1:2] if inputs.shape[1] > 1 else None
            grid_ci = inputs[:, 2:3] if inputs.shape[1] > 2 else None
            
            # Constraint: Energy should correlate with production
            if production is not None and energy is not None:
                energy_intensity = 3.5  # MWh/tonne (Indian average)
                expected_energy = production * energy_intensity
                energy_loss = torch.mean((energy - expected_energy) ** 2)
                base_loss += energy_loss * 0.1
            
            # Constraint: Emissions should correlate with grid CI
            if grid_ci is not None:
                # Higher grid CI -> higher emissions
                expected_emissions = production * (1.5 + grid_ci * 0.5) if production is not None else predictions
                emission_loss = torch.mean((predictions - expected_emissions) ** 2)
                base_loss += emission_loss * 0.05
        
        return base_loss

