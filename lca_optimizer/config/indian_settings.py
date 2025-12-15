"""
Indian-Specific Configuration Settings
Regional constraints, policies, and industrial characteristics
"""

import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


@dataclass
class IndianStateConfig:
    """Configuration for Indian state industrial characteristics"""
    state_name: str
    grid_carbon_intensity_baseline: float  # kg CO2/kWh
    renewable_share: float  # 0-1
    industrial_policy_strictness: float  # 0-1, regulatory strictness
    power_reliability: float  # 0-1, grid reliability factor
    material_availability: Dict[str, float]  # Local material availability factors


class IndianSettings(BaseSettings):
    """Indian-specific application settings"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )
    
    # Indian Data Sources
    indian_data_dir: str = "data/raw/indian"
    datagov_api_key: Optional[str] = None
    enable_indian_data_download: bool = True
    
    # Indian Regional Settings
    default_state: str = "Maharashtra"
    enable_state_specific_ci: bool = True
    
    # Indian Industrial Characteristics
    indian_steel_dri_share: float = 0.35  # % of steel from DRI process
    indian_cement_clinker_ratio: float = 0.75  # Average clinker ratio
    indian_aluminium_grid_dependency: float = 0.85  # Grid dependency for aluminium
    
    # Indian Policy Constraints
    enable_policy_constraints: bool = True
    perform_achievement_trajectory: bool = True  # PAT scheme compliance
    carbon_tax_rate_inr_per_tco2: float = 0.0  # Current carbon tax (if any)
    
    # Indian Supply Chain Characteristics
    local_material_preference: float = 0.7  # Preference for local materials
    transport_emission_factor: float = 0.15  # kg CO2/tonne-km (Indian average)
    
    # Indian Climate Conditions
    account_for_monsoon: bool = True
    account_for_heat_waves: bool = True
    
    # ML Model Settings for Indian Data
    enable_transfer_learning: bool = True
    indian_data_augmentation: bool = True
    physics_regularization_weight: float = 0.15  # Higher for data-limited scenarios
    
    # State Configurations
    # Use default_factory to satisfy Pydantic validation and populate in model_post_init
    state_configs: Dict[str, IndianStateConfig] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        """
        Populate state_configs after initial validation if not provided via env.
        """
        if not self.state_configs:
            self.state_configs = self._initialize_state_configs()
    
    def _initialize_state_configs(self) -> Dict[str, IndianStateConfig]:
        """Initialize state-specific configurations"""
        configs = {}
        
        # Major industrial states
        states_data = {
            "Maharashtra": {
                "grid_ci": 0.85,
                "renewable": 0.15,
                "policy_strictness": 0.7,
                "power_reliability": 0.85,
                "materials": {"iron_ore": 0.3, "coal": 0.4, "limestone": 0.6}
            },
            "Gujarat": {
                "grid_ci": 0.80,
                "renewable": 0.20,
                "policy_strictness": 0.75,
                "power_reliability": 0.90,
                "materials": {"iron_ore": 0.2, "coal": 0.3, "limestone": 0.5}
            },
            "Odisha": {
                "grid_ci": 0.95,
                "renewable": 0.10,
                "policy_strictness": 0.65,
                "power_reliability": 0.75,
                "materials": {"iron_ore": 0.95, "coal": 0.90, "limestone": 0.7}
            },
            "Jharkhand": {
                "grid_ci": 1.00,
                "renewable": 0.08,
                "policy_strictness": 0.60,
                "power_reliability": 0.70,
                "materials": {"iron_ore": 0.90, "coal": 0.95, "limestone": 0.6}
            },
            "Tamil Nadu": {
                "grid_ci": 0.75,
                "renewable": 0.25,
                "policy_strictness": 0.70,
                "power_reliability": 0.80,
                "materials": {"iron_ore": 0.1, "coal": 0.2, "limestone": 0.4}
            },
            "Karnataka": {
                "grid_ci": 0.70,
                "renewable": 0.30,
                "policy_strictness": 0.72,
                "power_reliability": 0.82,
                "materials": {"iron_ore": 0.4, "coal": 0.3, "limestone": 0.5}
            }
        }
        
        for state, data in states_data.items():
            configs[state] = IndianStateConfig(
                state_name=state,
                grid_carbon_intensity_baseline=data["grid_ci"],
                renewable_share=data["renewable"],
                industrial_policy_strictness=data["policy_strictness"],
                power_reliability=data["power_reliability"],
                material_availability=data["materials"]
            )
        
        return configs
    
    def get_state_config(self, state: str) -> Optional[IndianStateConfig]:
        """Get configuration for specific state"""
        return self.state_configs.get(state, self.state_configs.get(self.default_state))


_indian_settings: Optional[IndianSettings] = None


def get_indian_settings() -> IndianSettings:
    """Get Indian-specific settings (singleton)"""
    global _indian_settings
    if _indian_settings is None:
        _indian_settings = IndianSettings()
    return _indian_settings

