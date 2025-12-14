"""
Dynamic Allocation: Time-based allocation for co-products in LCA
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class AllocationMethod(Enum):
    """Allocation methods for co-products"""
    MASS = "mass"
    ENERGY = "energy"
    ECONOMIC = "economic"
    SYSTEM_EXPANSION = "system_expansion"
    DYNAMIC = "dynamic"


class DynamicAllocation:
    """
    Dynamic allocation factors for co-products in LCA.
    
    Handles time-varying allocation for:
    - Slag in steel production
    - Fly ash in cement production
    - By-products in chemical processes
    """
    
    def __init__(self, method: AllocationMethod = AllocationMethod.DYNAMIC):
        """
        Initialize dynamic allocation.
        
        Args:
            method: Allocation method to use
        """
        self.method = method
        self.allocation_history = {}
        logger.info(f"Dynamic allocation initialized with method: {method.value}")
    
    def calculate_allocation_factors(
        self,
        main_product: str,
        co_products: List[str],
        quantities: Dict[str, float],
        timestamp: Optional[datetime] = None,
        market_prices: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Calculate allocation factors for co-products.
        
        Args:
            main_product: Name of main product
            co_products: List of co-product names
            quantities: Quantities of each product (kg)
            timestamp: Time for dynamic allocation
            market_prices: Market prices for economic allocation (USD/kg)
        
        Returns:
            Allocation factors (sum to 1.0)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        if self.method == AllocationMethod.MASS:
            return self._mass_allocation(quantities)
        
        elif self.method == AllocationMethod.ENERGY:
            return self._energy_allocation(quantities)
        
        elif self.method == AllocationMethod.ECONOMIC:
            if market_prices is None:
                logger.warning("Market prices not provided, falling back to mass allocation")
                return self._mass_allocation(quantities)
            return self._economic_allocation(quantities, market_prices)
        
        elif self.method == AllocationMethod.SYSTEM_EXPANSION:
            return self._system_expansion_allocation(main_product, co_products)
        
        elif self.method == AllocationMethod.DYNAMIC:
            return self._dynamic_allocation(
                main_product, co_products, quantities, timestamp, market_prices
            )
        
        else:
            raise ValueError(f"Unknown allocation method: {self.method}")
    
    def _mass_allocation(self, quantities: Dict[str, float]) -> Dict[str, float]:
        """Mass-based allocation"""
        total = sum(quantities.values())
        return {k: v / total for k, v in quantities.items()}
    
    def _energy_allocation(self, quantities: Dict[str, float]) -> Dict[str, float]:
        """Energy-based allocation (requires energy content data)"""
        # Placeholder: use mass allocation as default
        # TODO: Integrate energy content database
        return self._mass_allocation(quantities)
    
    def _economic_allocation(
        self,
        quantities: Dict[str, float],
        market_prices: Dict[str, float]
    ) -> Dict[str, float]:
        """Economic allocation based on market value"""
        values = {
            k: quantities[k] * market_prices.get(k, 0.0)
            for k in quantities.keys()
        }
        total_value = sum(values.values())
        
        if total_value == 0:
            return self._mass_allocation(quantities)
        
        return {k: v / total_value for k, v in values.items()}
    
    def _system_expansion_allocation(
        self,
        main_product: str,
        co_products: List[str]
    ) -> Dict[str, float]:
        """
        System expansion allocation.
        Main product gets full burden, co-products get credit.
        """
        factors = {main_product: 1.0}
        for co_product in co_products:
            factors[co_product] = 0.0  # No burden, only credit
        
        return factors
    
    def _dynamic_allocation(
        self,
        main_product: str,
        co_products: List[str],
        quantities: Dict[str, float],
        timestamp: datetime,
        market_prices: Optional[Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Dynamic allocation that varies with time and market conditions.
        
        Uses historical data and market trends to adjust allocation.
        """
        # Check for historical allocation
        cache_key = f"{main_product}_{timestamp.date()}"
        if cache_key in self.allocation_history:
            return self.allocation_history[cache_key]
        
        # Use economic allocation if prices available, otherwise mass
        if market_prices:
            factors = self._economic_allocation(quantities, market_prices)
        else:
            factors = self._mass_allocation(quantities)
        
        # Store in history
        self.allocation_history[cache_key] = factors
        
        return factors
    
    def update_allocation_history(
        self,
        product: str,
        timestamp: datetime,
        factors: Dict[str, float]
    ):
        """Update allocation history for learning"""
        cache_key = f"{product}_{timestamp.date()}"
        self.allocation_history[cache_key] = factors
    
    def get_temporal_allocation(
        self,
        product: str,
        start_date: datetime,
        end_date: datetime,
        frequency: str = "daily"
    ) -> pd.DataFrame:
        """
        Get allocation factors over time period.
        
        Returns:
            DataFrame with allocation factors over time
        """
        dates = pd.date_range(start_date, end_date, freq=frequency)
        factors_list = []
        
        for date in dates:
            cache_key = f"{product}_{date.date()}"
            if cache_key in self.allocation_history:
                factors_list.append(self.allocation_history[cache_key])
            else:
                # Default allocation
                factors_list.append({product: 1.0})
        
        return pd.DataFrame(factors_list, index=dates)

