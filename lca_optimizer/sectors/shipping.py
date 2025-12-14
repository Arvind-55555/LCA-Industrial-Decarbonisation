"""
Shipping & Aviation Sector: Hydrogen Derivative Fuels LCA Comparator
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import logging

from lca_optimizer.core.engine import LCAEngine, LCAResult
from lca_optimizer.core.physics import PhysicsConstraints

logger = logging.getLogger(__name__)


class FuelType(Enum):
    """Fuel types for comparison"""
    GREEN_AMMONIA = "green_ammonia"
    GREEN_METHANOL = "green_methanol"
    E_FUELS = "e_fuels"
    MARINE_FUEL = "marine_fuel"
    JET_A1 = "jet_a1"


@dataclass
class FuelConfig:
    """Configuration for fuel production"""
    fuel_type: FuelType
    feedstock_source: str  # "biogenic", "captured_co2", "atmospheric"
    renewable_electricity_source: Dict[str, float]  # {"wind": 0.6, "solar": 0.4}
    location: str
    production_capacity: float  # t/year


@dataclass
class RouteConfig:
    """Configuration for shipping/aviation route"""
    origin: str
    destination: str
    distance: float  # km
    payload: float  # t
    vessel_type: Optional[str] = None  # For shipping
    aircraft_type: Optional[str] = None  # For aviation


class ShippingFuelComparator:
    """
    Comparator for well-to-wake emissions of hydrogen derivative fuels.
    
    Compares:
    - Green ammonia
    - Green methanol
    - E-fuels (PtL)
    - Traditional marine fuel / Jet A-1
    """
    
    def __init__(
        self,
        lca_engine: LCAEngine,
        physics_constraints: Optional[PhysicsConstraints] = None
    ):
        """
        Initialize fuel comparator.
        
        Args:
            lca_engine: Core LCA engine
            physics_constraints: Physics constraints validator
        """
        self.engine = lca_engine
        self.physics = physics_constraints or PhysicsConstraints()
        
        # Fuel-specific constants (LHV in MJ/kg)
        self.fuel_lhv = {
            FuelType.GREEN_AMMONIA: 18.6,
            FuelType.GREEN_METHANOL: 19.9,
            FuelType.E_FUELS: 42.0,
            FuelType.MARINE_FUEL: 40.0,
            FuelType.JET_A1: 43.0
        }
        
        # Combustion efficiency
        self.combustion_efficiency = {
            FuelType.GREEN_AMMONIA: 0.85,
            FuelType.GREEN_METHANOL: 0.90,
            FuelType.E_FUELS: 0.95,
            FuelType.MARINE_FUEL: 0.92,
            FuelType.JET_A1: 0.95
        }
        
        logger.info("Shipping Fuel Comparator initialized")
    
    def compare_fuels(
        self,
        fuels: List[FuelConfig],
        route: RouteConfig
    ) -> Dict[str, Any]:
        """
        Compare well-to-wake emissions for different fuels on a route.
        
        Args:
            fuels: List of fuel configurations to compare
            route: Route configuration
        
        Returns:
            Comparison results with emissions per fuel
        """
        results = []
        
        for fuel_config in fuels:
            # Calculate well-to-tank (WTT) emissions
            wtt_emissions = self._calculate_wtt_emissions(fuel_config)
            
            # Calculate tank-to-wake (TTW) emissions
            ttw_emissions = self._calculate_ttw_emissions(fuel_config, route)
            
            # Total well-to-wake (WTW)
            wtw_emissions = wtt_emissions + ttw_emissions
            
            # Energy required for route
            energy_required = self._calculate_energy_required(route)
            
            # Fuel consumption
            fuel_consumption = energy_required / (
                self.fuel_lhv[fuel_config.fuel_type] *
                self.combustion_efficiency[fuel_config.fuel_type]
            )
            
            # Total emissions for route
            total_emissions = wtw_emissions * fuel_consumption
            
            results.append({
                "fuel_type": fuel_config.fuel_type.value,
                "wtt_emissions": wtt_emissions,
                "ttw_emissions": ttw_emissions,
                "wtw_emissions": wtw_emissions,
                "fuel_consumption": fuel_consumption,
                "total_route_emissions": total_emissions
            })
        
        # Find optimal fuel
        optimal = min(results, key=lambda x: x["total_route_emissions"])
        
        return {
            "route": {
                "origin": route.origin,
                "destination": route.destination,
                "distance": route.distance
            },
            "fuel_comparison": results,
            "optimal_fuel": optimal["fuel_type"],
            "emission_savings": {
                fuel["fuel_type"]: (
                    fuel["total_route_emissions"] - optimal["total_route_emissions"]
                ) / fuel["total_route_emissions"] * 100
                for fuel in results
            }
        }
    
    def _calculate_wtt_emissions(self, fuel_config: FuelConfig) -> float:
        """
        Calculate well-to-tank emissions (production and transport).
        
        Returns:
            WTT emissions (kg CO2eq/kg fuel)
        """
        process_params = {
            "sector": "shipping" if fuel_config.fuel_type != FuelType.JET_A1 else "aviation",
            "fuel_type": fuel_config.fuel_type.value,
            "feedstock_source": fuel_config.feedstock_source,
            "renewable_electricity_source": fuel_config.renewable_electricity_source
        }
        
        lca_result = self.engine.calculate_lca(
            process_params=process_params,
            location=fuel_config.location,
            timestamp=datetime.now()
        )
        
        # Convert to per kg fuel
        return lca_result.total_emissions / fuel_config.production_capacity
    
    def _calculate_ttw_emissions(
        self,
        fuel_config: FuelConfig,
        route: RouteConfig
    ) -> float:
        """
        Calculate tank-to-wake emissions (combustion).
        
        Returns:
            TTW emissions (kg CO2eq/kg fuel)
        """
        fuel_type = fuel_config.fuel_type
        
        # Pure H2-derived fuels have zero CO2 from combustion
        if fuel_type in [FuelType.GREEN_AMMONIA, FuelType.GREEN_METHANOL, FuelType.E_FUELS]:
            # Only non-CO2 effects (NOx, etc.)
            return 0.05  # Placeholder for non-CO2 effects
        
        # Traditional fuels
        elif fuel_type == FuelType.MARINE_FUEL:
            return 3.15  # kg CO2/kg fuel
        
        elif fuel_type == FuelType.JET_A1:
            # Include non-CO2 effects (contrails, NOx)
            return 3.15 * 1.9  # CO2 + non-CO2 effects
        
        return 0.0
    
    def _calculate_energy_required(self, route: RouteConfig) -> float:
        """
        Calculate energy required for route (MJ).
        
        Args:
            route: Route configuration
        
        Returns:
            Energy required (MJ)
        """
        # Simplified energy calculation
        # Energy = distance * specific_energy_consumption * payload
        
        if route.vessel_type:
            # Shipping: ~0.5 MJ/t/km
            specific_energy = 0.5
        elif route.aircraft_type:
            # Aviation: ~2.0 MJ/t/km
            specific_energy = 2.0
        else:
            specific_energy = 1.0  # Default
        
        return route.distance * specific_energy * route.payload
    
    def optimize_fuel_pathway(
        self,
        route: RouteConfig,
        available_fuels: List[FuelType],
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Optimize fuel pathway for a specific route.
        
        Args:
            route: Route configuration
            available_fuels: List of available fuel types
            constraints: Additional constraints (cost, availability, etc.)
        
        Returns:
            Optimal fuel configuration
        """
        # Create fuel configurations for each available type
        fuel_configs = []
        
        for fuel_type in available_fuels:
            config = FuelConfig(
                fuel_type=fuel_type,
                feedstock_source="captured_co2" if fuel_type in [
                    FuelType.GREEN_METHANOL, FuelType.E_FUELS
                ] else "atmospheric",
                renewable_electricity_source={"wind": 0.6, "solar": 0.4},
                location=route.origin,
                production_capacity=1000.0  # Default
            )
            fuel_configs.append(config)
        
        # Compare fuels
        comparison = self.compare_fuels(fuel_configs, route)
        
        return {
            "optimal_fuel": comparison["optimal_fuel"],
            "emissions": comparison["fuel_comparison"],
            "recommendation": self._generate_recommendation(comparison)
        }
    
    def _generate_recommendation(self, comparison: Dict[str, Any]) -> str:
        """Generate human-readable recommendation"""
        optimal = comparison["optimal_fuel"]
        savings = comparison["emission_savings"]
        
        return (
            f"Optimal fuel: {optimal}. "
            f"Emission savings vs. alternatives: "
            f"{', '.join([f'{k}: {v:.1f}%' for k, v in savings.items() if k != optimal])}"
        )

