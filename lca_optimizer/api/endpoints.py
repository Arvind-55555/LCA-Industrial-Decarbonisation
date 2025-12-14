"""
API Endpoints for LCA optimization
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from lca_optimizer.core.engine import LCAEngine
from lca_optimizer.sectors.steel import SteelH2DRIOptimizer, SteelProcessConfig
from lca_optimizer.sectors.shipping import (
    ShippingFuelComparator, FuelConfig, FuelType, RouteConfig
)
from lca_optimizer.sectors.cement import (
    CementCCUSOptimizer, CementProcessConfig, CCUSTechnology
)
from lca_optimizer.sectors.aluminium import (
    AluminiumElectrificationOptimizer,
    AluminiumProcessConfig,
    TruckingFleetConfig,
    BatteryType
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize engines
lca_engine = LCAEngine()


# Request/Response models
class SteelH2DRIRequest(BaseModel):
    h2_pathway: str = Field(..., description="H2 production pathway")
    electrolyzer_type: str = Field(..., description="Electrolyzer type")
    renewable_mix: Dict[str, float] = Field(..., description="Renewable energy mix")
    iron_ore_source: str = Field(..., description="Iron ore source location")
    process_heat_source: str = Field(..., description="Process heat source")
    location: str = Field(..., description="Plant location")
    production_capacity: float = Field(..., description="Production capacity (t/year)")


class ShippingFuelRequest(BaseModel):
    fuels: List[Dict[str, Any]] = Field(..., description="Fuel configurations")
    origin: str = Field(..., description="Origin location")
    destination: str = Field(..., description="Destination location")
    distance: float = Field(..., description="Distance (km)")
    payload: float = Field(..., description="Payload (t)")


class CementCCUSRequest(BaseModel):
    capture_technology: str = Field(..., description="CCUS technology")
    capture_rate: float = Field(..., ge=0.0, le=1.0, description="Capture rate")
    clinker_substitution: Dict[str, float] = Field(default={}, description="Clinker substitution")
    alternative_raw_materials: Dict[str, float] = Field(default={}, description="Alternative materials")
    location: str = Field(..., description="Plant location")
    production_capacity: float = Field(..., description="Production capacity (t/year)")


class AluminiumElectrificationRequest(BaseModel):
    smelting_technology: str = Field(..., description="Smelting technology")
    location: str = Field(..., description="Plant location")
    production_capacity: float = Field(..., description="Production capacity (t/year)")
    recycling_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Recycling rate")


@router.post("/lca/steel_h2_dri")
async def steel_h2_dri_optimization(request: SteelH2DRIRequest):
    """
    Optimize H2-DRI steel production LCA.
    
    Returns optimal configuration and emission reduction.
    """
    try:
        optimizer = SteelH2DRIOptimizer(lca_engine)
        
        config = SteelProcessConfig(
            h2_pathway=request.h2_pathway,
            electrolyzer_type=request.electrolyzer_type,
            renewable_mix=request.renewable_mix,
            iron_ore_source=request.iron_ore_source,
            process_heat_source=request.process_heat_source,
            location=request.location,
            production_capacity=request.production_capacity
        )
        
        result = optimizer.optimize(config)
        
        return {
            "status": "success",
            "result": result
        }
    except Exception as e:
        logger.error(f"Error in steel H2-DRI optimization: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/lca/shipping_fuel_comparison")
async def shipping_fuel_comparison(request: ShippingFuelRequest):
    """
    Compare well-to-wake emissions for different shipping fuels.
    
    Returns comparison results and optimal fuel recommendation.
    """
    try:
        comparator = ShippingFuelComparator(lca_engine)
        
        # Convert fuel configs
        fuel_configs = []
        for fuel_data in request.fuels:
            fuel_type = FuelType(fuel_data["fuel_type"])
            config = FuelConfig(
                fuel_type=fuel_type,
                feedstock_source=fuel_data.get("feedstock_source", "atmospheric"),
                renewable_electricity_source=fuel_data.get("renewable_electricity_source", {"wind": 0.6, "solar": 0.4}),
                location=request.origin,
                production_capacity=1000.0
            )
            fuel_configs.append(config)
        
        route = RouteConfig(
            origin=request.origin,
            destination=request.destination,
            distance=request.distance,
            payload=request.payload
        )
        
        result = comparator.compare_fuels(fuel_configs, route)
        
        return {
            "status": "success",
            "result": result
        }
    except Exception as e:
        logger.error(f"Error in shipping fuel comparison: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/lca/cement_ccus_optimization")
async def cement_ccus_optimization(request: CementCCUSRequest):
    """
    Optimize CCUS-integrated cement production LCA.
    
    Returns optimal configuration and emission reduction.
    """
    try:
        optimizer = CementCCUSOptimizer(lca_engine)
        
        capture_tech = CCUSTechnology(request.capture_technology)
        
        config = CementProcessConfig(
            capture_technology=capture_tech,
            capture_rate=request.capture_rate,
            clinker_substitution=request.clinker_substitution,
            alternative_raw_materials=request.alternative_raw_materials,
            location=request.location,
            production_capacity=request.production_capacity
        )
        
        result = optimizer.optimize(config)
        
        return {
            "status": "success",
            "result": result
        }
    except Exception as e:
        logger.error(f"Error in cement CCUS optimization: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/lca/aluminium_electrification")
async def aluminium_electrification(request: AluminiumElectrificationRequest):
    """
    Optimize electrified aluminium production LCA.
    
    Returns optimal production schedule and emission reduction.
    """
    try:
        optimizer = AluminiumElectrificationOptimizer(lca_engine)
        
        config = AluminiumProcessConfig(
            smelting_technology=request.smelting_technology,
            grid_carbon_intensity_source="realtime",
            location=request.location,
            production_capacity=request.production_capacity,
            recycling_rate=request.recycling_rate,
            recycling_loop_efficiency=0.95
        )
        
        result = optimizer.optimize_aluminium_production(config)
        
        return {
            "status": "success",
            "result": result
        }
    except Exception as e:
        logger.error(f"Error in aluminium electrification optimization: {e}")
        raise HTTPException(status_code=500, detail=str(e))

