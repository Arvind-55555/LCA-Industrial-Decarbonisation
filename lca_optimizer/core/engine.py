"""
Core LCA Engine: Dynamic, high-resolution LCA modeling
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class LCAResult:
    """Container for LCA calculation results"""
    total_emissions: float  # kg CO2eq
    uncertainty: Tuple[float, float]  # (lower, upper) bounds
    breakdown: Dict[str, float]  # Emissions by category
    metadata: Dict[str, Any]  # Additional metadata
    timestamp: datetime


class LCAEngine:
    """
    Core LCA Engine for dynamic, high-resolution life cycle assessment.
    
    Features:
    - Real-time grid carbon intensity integration
    - Dynamic allocation for co-products
    - Uncertainty quantification
    - Multi-stage process modeling
    """
    
    def __init__(
        self,
        grid_data_source: Optional[str] = None,
        lci_database: Optional[str] = None,
        enable_uncertainty: bool = True
    ):
        """
        Initialize LCA Engine.
        
        Args:
            grid_data_source: Source for real-time grid carbon data
            lci_database: Life Cycle Inventory database path
            enable_uncertainty: Enable uncertainty quantification
        """
        self.grid_data_source = grid_data_source
        self.lci_database = lci_database
        self.enable_uncertainty = enable_uncertainty
        self.grid_cache = {}
        self.lci_cache = {}
        
        logger.info("LCA Engine initialized")
    
    def calculate_lca(
        self,
        process_params: Dict[str, Any],
        location: str,
        timestamp: Optional[datetime] = None,
        include_uncertainty: Optional[bool] = None
    ) -> LCAResult:
        """
        Calculate LCA for a given process configuration.
        
        Args:
            process_params: Process parameters (technology, inputs, etc.)
            location: Geographic location for grid data
            timestamp: Time for grid carbon intensity lookup
            include_uncertainty: Override default uncertainty setting
            
        Returns:
            LCAResult with emissions, uncertainty, and breakdown
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        if include_uncertainty is None:
            include_uncertainty = self.enable_uncertainty
        
        # Get grid carbon intensity
        grid_ci = self.get_grid_carbon_intensity(location, timestamp)
        
        # Calculate upstream emissions
        upstream = self._calculate_upstream_emissions(process_params, location)
        
        # Calculate process emissions
        process = self._calculate_process_emissions(
            process_params, grid_ci, location
        )
        
        # Calculate downstream emissions
        downstream = self._calculate_downstream_emissions(process_params, location)
        
        # Total emissions
        total = upstream + process + downstream
        
        # Uncertainty quantification
        uncertainty = (total * 0.9, total * 1.1)  # Placeholder
        if include_uncertainty:
            uncertainty = self._quantify_uncertainty(
                upstream, process, downstream
            )
        
        breakdown = {
            "upstream": upstream,
            "process": process,
            "downstream": downstream
        }
        
        return LCAResult(
            total_emissions=total,
            uncertainty=uncertainty,
            breakdown=breakdown,
            metadata={
                "location": location,
                "grid_carbon_intensity": grid_ci,
                "timestamp": timestamp.isoformat()
            },
            timestamp=timestamp
        )
    
    def get_grid_carbon_intensity(
        self,
        location: str,
        timestamp: datetime
    ) -> float:
        """
        Get real-time grid carbon intensity.
        
        Args:
            location: Geographic location
            timestamp: Time for lookup
            
        Returns:
            Grid carbon intensity (g CO2eq/kWh)
        """
        # Check cache
        cache_key = f"{location}_{timestamp.date()}_{timestamp.hour}"
        if cache_key in self.grid_cache:
            return self.grid_cache[cache_key]
        
        # Priority 1: Try local data (no API key required)
        try:
            from lca_optimizer.data.local_data_loader import LocalGridDataLoader
            local_loader = LocalGridDataLoader()
            ci_data = local_loader.get_current_carbon_intensity(location)
            intensity = ci_data.carbon_intensity
            self.grid_cache[cache_key] = intensity
            logger.debug(f"Using local data for {location}: {intensity} g CO2eq/kWh")
            return intensity
        except Exception as e:
            logger.debug(f"Local data not available: {e}")
        
        # Priority 2: Try real API if available
        try:
            from lca_optimizer.data.grid_data_enhanced import ElectricityMapsLoader
            from lca_optimizer.config.settings import get_settings
            
            settings = get_settings()
            if settings.electricity_maps_api_key:
                loader = ElectricityMapsLoader(api_key=settings.electricity_maps_api_key)
                ci_data = loader.get_current_carbon_intensity(location)
                intensity = ci_data.carbon_intensity
                self.grid_cache[cache_key] = intensity
                return intensity
        except Exception as e:
            logger.debug(f"Real API not available: {e}. Using default.")
        
        # Fallback: return average EU grid intensity
        default_intensity = 300.0  # g CO2eq/kWh
        
        self.grid_cache[cache_key] = default_intensity
        return default_intensity
    
    def _calculate_upstream_emissions(
        self,
        process_params: Dict[str, Any],
        location: str
    ) -> float:
        """Calculate upstream (cradle-to-gate) emissions"""
        # Simplified calculation based on sector
        sector = process_params.get("sector", "generic")
        
        # Base upstream emissions by sector (kg CO2eq per unit)
        sector_emissions = {
            "steel": 500.0,  # Per ton
            "cement": 200.0,
            "shipping": 50.0,  # Per ton-km
            "aluminium": 800.0,
        }
        
        base = sector_emissions.get(sector, 300.0)
        capacity = process_params.get("production_capacity", 1.0)
        
        return base * capacity / 1000.0  # Scale appropriately
    
    def _calculate_process_emissions(
        self,
        process_params: Dict[str, Any],
        grid_ci: float,
        location: str
    ) -> float:
        """Calculate process emissions"""
        sector = process_params.get("sector", "generic")
        technology = process_params.get("technology", "conventional")
        capacity = process_params.get("production_capacity", 1.0)
        
        # Base process emissions by sector and technology (kg CO2eq per unit)
        process_emissions_map = {
            "steel": {
                "h2_dri": 200.0,  # Green H2-DRI
                "bf_bof": 1800.0,  # Traditional
            },
            "cement": {
                "ccus_integrated": 100.0,
                "conventional": 900.0,
            },
            "shipping": {
                "green_ammonia": 0.0,
                "marine_fuel": 3000.0,
            },
            "aluminium": {
                "electrolysis": 50.0,
                "conventional": 16000.0,
            }
        }
        
        base = process_emissions_map.get(sector, {}).get(technology, 500.0)
        
        # Adjust for grid carbon intensity (for electrified processes)
        if technology in ["h2_dri", "electrolysis", "ccus_integrated"]:
            # Scale by grid CI (normalize to 300 g CO2/kWh)
            grid_factor = grid_ci / 300.0
            base = base * (0.5 + 0.5 * grid_factor)  # Partial grid dependence
        
        return base * capacity / 1000.0
    
    def _calculate_downstream_emissions(
        self,
        process_params: Dict[str, Any],
        location: str
    ) -> float:
        """Calculate downstream (gate-to-grave) emissions"""
        # Downstream emissions are typically small compared to upstream/process
        # This includes transport, use phase, end-of-life
        sector = process_params.get("sector", "generic")
        capacity = process_params.get("production_capacity", 1.0)
        
        # Downstream emissions (typically 5-10% of total)
        downstream_factors = {
            "steel": 50.0,
            "cement": 30.0,
            "shipping": 100.0,
            "aluminium": 200.0,
        }
        
        base = downstream_factors.get(sector, 50.0)
        return base * capacity / 1000.0
    
    def _quantify_uncertainty(
        self,
        upstream: float,
        process: float,
        downstream: float
    ) -> Tuple[float, float]:
        """
        Quantify uncertainty using Monte Carlo or Bayesian methods.
        
        Returns:
            (lower_bound, upper_bound) in kg CO2eq
        """
        # Placeholder: simple uncertainty propagation
        total = upstream + process + downstream
        uncertainty_factor = 0.1  # 10% uncertainty
        
        return (
            total * (1 - uncertainty_factor),
            total * (1 + uncertainty_factor)
        )
    
    def update_grid_data(self, location: str, data: Dict[str, float]):
        """Update grid carbon intensity data"""
        self.grid_cache.update({
            f"{location}_{k}": v for k, v in data.items()
        })

