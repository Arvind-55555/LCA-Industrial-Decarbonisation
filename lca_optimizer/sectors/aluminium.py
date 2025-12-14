"""
Aluminium & Trucking Sector: Electrification & Recycling LCA
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging

from lca_optimizer.core.engine import LCAEngine, LCAResult
from lca_optimizer.core.physics import PhysicsConstraints

logger = logging.getLogger(__name__)


class BatteryType(Enum):
    """Battery types for electrification"""
    NMC = "nmc"
    LFP = "lfp"
    NEXT_GEN = "next_gen"


@dataclass
class AluminiumProcessConfig:
    """Configuration for aluminium production"""
    smelting_technology: str  # "electrolysis", "inert_anode"
    grid_carbon_intensity_source: str  # "realtime", "average"
    location: str
    production_capacity: float  # t/year
    recycling_rate: float  # 0.0 to 1.0
    recycling_loop_efficiency: float  # 0.0 to 1.0


@dataclass
class TruckingFleetConfig:
    """Configuration for trucking fleet electrification"""
    fleet_size: int
    battery_type: BatteryType
    daily_distance: float  # km
    payload_capacity: float  # t
    location: str
    charging_schedule: Optional[List[Tuple[int, int]]] = None  # [(hour_start, hour_end), ...]


class AluminiumElectrificationOptimizer:
    """
    Optimizer for electrified aluminium production and trucking fleets.
    
    Features:
    - Dynamic grid carbon intensity integration
    - Recycling loop efficiency modeling
    - Battery production emissions
    - Optimal charging schedules
    """
    
    def __init__(
        self,
        lca_engine: LCAEngine,
        physics_constraints: Optional[PhysicsConstraints] = None
    ):
        """
        Initialize aluminium electrification optimizer.
        
        Args:
            lca_engine: Core LCA engine
            physics_constraints: Physics constraints validator
        """
        self.engine = lca_engine
        self.physics = physics_constraints or PhysicsConstraints()
        
        # Aluminium production constants
        self.primary_aluminium_emissions = 16.0  # kg CO2/kg Al (global average)
        self.recycled_aluminium_emissions = 0.5  # kg CO2/kg Al (recycling)
        self.smelting_energy = 13.5  # kWh/kg Al
        
        # Battery production emissions (kg CO2/kWh capacity)
        self.battery_emissions = {
            BatteryType.NMC: 100.0,
            BatteryType.LFP: 80.0,
            BatteryType.NEXT_GEN: 60.0
        }
        
        # Trucking constants
        self.truck_energy_consumption = 1.5  # kWh/km (loaded)
        self.battery_capacity = 500.0  # kWh per truck
        
        logger.info("Aluminium Electrification Optimizer initialized")
    
    def optimize_aluminium_production(
        self,
        config: AluminiumProcessConfig,
        time_horizon: int = 24  # hours
    ) -> Dict[str, Any]:
        """
        Optimize aluminium production with dynamic grid carbon intensity.
        
        Args:
            config: Process configuration
            time_horizon: Time horizon for optimization (hours)
        
        Returns:
            Optimization results with optimal production schedule
        """
        # Get hourly grid carbon intensity
        grid_ci = self._get_hourly_grid_ci(config.location, time_horizon)
        
        # Calculate emissions for different production schedules
        baseline_emissions = self._calculate_baseline_emissions(config)
        
        # Optimize production schedule
        optimal_schedule = self._optimize_production_schedule(
            config, grid_ci, time_horizon
        )
        
        optimal_emissions = self._calculate_scheduled_emissions(
            config, optimal_schedule, grid_ci
        )
        
        return {
            "baseline_emissions": baseline_emissions,
            "optimal_emissions": optimal_emissions,
            "emission_reduction": (
                (baseline_emissions - optimal_emissions) / baseline_emissions * 100
            ),
            "optimal_schedule": optimal_schedule,
            "recycling_contribution": self._calculate_recycling_contribution(config)
        }
    
    def optimize_trucking_fleet(
        self,
        config: TruckingFleetConfig,
        time_horizon: int = 24  # hours
    ) -> Dict[str, Any]:
        """
        Optimize trucking fleet electrification with optimal charging.
        
        Args:
            config: Fleet configuration
            time_horizon: Time horizon for optimization (hours)
        
        Returns:
            Optimization results with optimal charging schedule
        """
        # Get hourly grid carbon intensity
        grid_ci = self._get_hourly_grid_ci(config.location, time_horizon)
        
        # Calculate baseline (diesel) emissions
        baseline_emissions = self._calculate_diesel_emissions(config)
        
        # Calculate battery production emissions
        battery_emissions = self._calculate_battery_emissions(config)
        
        # Optimize charging schedule
        optimal_charging = self._optimize_charging_schedule(
            config, grid_ci, time_horizon
        )
        
        # Calculate electric fleet emissions
        electric_emissions = self._calculate_electric_emissions(
            config, optimal_charging, grid_ci
        )
        
        total_electric_emissions = electric_emissions + battery_emissions
        
        return {
            "baseline_diesel_emissions": baseline_emissions,
            "electric_emissions": electric_emissions,
            "battery_emissions": battery_emissions,
            "total_electric_emissions": total_electric_emissions,
            "emission_reduction": (
                (baseline_emissions - total_electric_emissions) / baseline_emissions * 100
            ),
            "optimal_charging_schedule": optimal_charging,
            "payback_period": self._calculate_payback_period(
                baseline_emissions, total_electric_emissions, battery_emissions
            )
        }
    
    def _get_hourly_grid_ci(
        self,
        location: str,
        hours: int
    ) -> pd.Series:
        """Get hourly grid carbon intensity"""
        now = datetime.now()
        timestamps = [now + timedelta(hours=i) for i in range(hours)]
        
        ci_values = []
        for ts in timestamps:
            ci = self.engine.get_grid_carbon_intensity(location, ts)
            ci_values.append(ci)
        
        return pd.Series(ci_values, index=timestamps)
    
    def _calculate_baseline_emissions(
        self,
        config: AluminiumProcessConfig
    ) -> float:
        """Calculate baseline emissions (average grid CI)"""
        avg_ci = 300.0  # g CO2/kWh (placeholder)
        energy_emissions = (
            self.smelting_energy * config.production_capacity * avg_ci / 1000
        )
        
        # Primary production emissions
        primary_emissions = (
            self.primary_aluminium_emissions *
            config.production_capacity *
            (1 - config.recycling_rate)
        )
        
        # Recycling emissions
        recycling_emissions = (
            self.recycled_aluminium_emissions *
            config.production_capacity *
            config.recycling_rate *
            (1 - config.recycling_loop_efficiency)
        )
        
        return energy_emissions + primary_emissions + recycling_emissions
    
    def _optimize_production_schedule(
        self,
        config: AluminiumProcessConfig,
        grid_ci: pd.Series,
        time_horizon: int
    ) -> Dict[int, float]:
        """
        Optimize production schedule to minimize emissions.
        
        Returns:
            Schedule: {hour: production_rate}
        """
        # Simple optimization: produce more when grid CI is lower
        total_production = config.production_capacity / 365 * (time_horizon / 24)
        
        # Sort hours by carbon intensity
        ci_sorted = grid_ci.sort_values()
        
        schedule = {}
        remaining = total_production
        
        for hour_idx, ci_value in ci_sorted.items():
            if remaining <= 0:
                schedule[hour_idx.hour] = 0.0
            else:
                # Allocate more production to low-CI hours
                allocation = min(remaining, total_production / time_horizon * 2)
                schedule[hour_idx.hour] = allocation
                remaining -= allocation
        
        return schedule
    
    def _calculate_scheduled_emissions(
        self,
        config: AluminiumProcessConfig,
        schedule: Dict[int, float],
        grid_ci: pd.Series
    ) -> float:
        """Calculate emissions for a given production schedule"""
        total_emissions = 0.0
        
        for hour, production in schedule.items():
            if production > 0:
                ci = grid_ci.iloc[hour] if hour < len(grid_ci) else 300.0
                energy_emissions = (
                    self.smelting_energy * production * ci / 1000
                )
                total_emissions += energy_emissions
        
        return total_emissions
    
    def _calculate_recycling_contribution(
        self,
        config: AluminiumProcessConfig
    ) -> Dict[str, float]:
        """Calculate recycling contribution to emission reduction"""
        primary_emissions = (
            self.primary_aluminium_emissions *
            config.production_capacity *
            (1 - config.recycling_rate)
        )
        
        recycled_emissions = (
            self.recycled_aluminium_emissions *
            config.production_capacity *
            config.recycling_rate
        )
        
        savings = primary_emissions - recycled_emissions
        
        return {
            "recycling_rate": config.recycling_rate,
            "primary_emissions_avoided": savings,
            "reduction_percentage": (savings / primary_emissions) * 100 if primary_emissions > 0 else 0
        }
    
    def _calculate_diesel_emissions(
        self,
        config: TruckingFleetConfig
    ) -> float:
        """Calculate diesel truck emissions"""
        # Diesel: ~2.68 kg CO2/L, ~35 L/100 km
        diesel_consumption = (
            config.fleet_size *
            config.daily_distance *
            365 *
            0.35  # L/km
        )
        emissions = diesel_consumption * 2.68
        
        return emissions
    
    def _calculate_battery_emissions(
        self,
        config: TruckingFleetConfig
    ) -> float:
        """Calculate battery production emissions"""
        total_capacity = config.fleet_size * self.battery_capacity
        emissions_factor = self.battery_emissions[config.battery_type]
        
        return total_capacity * emissions_factor / 1000  # Convert to kg CO2
    
    def _optimize_charging_schedule(
        self,
        config: TruckingFleetConfig,
        grid_ci: pd.Series,
        time_horizon: int
    ) -> Dict[int, float]:
        """
        Optimize charging schedule to minimize emissions.
        
        Returns:
            Schedule: {hour: charging_power_kW}
        """
        # Energy required per day
        daily_energy = (
            config.fleet_size *
            config.daily_distance *
            self.truck_energy_consumption
        )
        
        # Charging time (assume 2 hours for full charge)
        charging_hours = 2
        charging_power = daily_energy / charging_hours
        
        # Find hours with lowest CI
        ci_sorted = grid_ci.sort_values()
        optimal_hours = ci_sorted.head(charging_hours).index
        
        schedule = {}
        for hour in range(time_horizon):
            if hour in [h.hour for h in optimal_hours]:
                schedule[hour] = charging_power
            else:
                schedule[hour] = 0.0
        
        return schedule
    
    def _calculate_electric_emissions(
        self,
        config: TruckingFleetConfig,
        charging_schedule: Dict[int, float],
        grid_ci: pd.Series
    ) -> float:
        """Calculate electric fleet emissions"""
        total_emissions = 0.0
        
        for hour, power in charging_schedule.items():
            if power > 0:
                ci = grid_ci.iloc[hour] if hour < len(grid_ci) else 300.0
                energy = power * 1.0  # 1 hour
                emissions = energy * ci / 1000  # Convert to kg CO2
                total_emissions += emissions * 365  # Annual
        
        return total_emissions
    
    def _calculate_payback_period(
        self,
        diesel_emissions: float,
        electric_emissions: float,
        battery_emissions: float
    ) -> float:
        """Calculate emission payback period (years)"""
        annual_savings = diesel_emissions - electric_emissions
        if annual_savings <= 0:
            return float('inf')
        
        return battery_emissions / annual_savings

