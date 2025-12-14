"""
Policy Simulation Module
Simulates impact of green H₂ mandates, CCUS subsidies, carbon border adjustments
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

from lca_optimizer.core.engine import LCAEngine
from lca_optimizer.sectors.steel import SteelH2DRIOptimizer
from lca_optimizer.sectors.cement import CementCCUSOptimizer

logger = logging.getLogger(__name__)


@dataclass
class Policy:
    """Policy configuration"""
    name: str
    policy_type: str  # "h2_mandate", "ccus_subsidy", "carbon_border", "renewable_target"
    parameters: Dict[str, float]
    start_date: datetime
    end_date: Optional[datetime] = None
    sector: Optional[str] = None  # "steel", "cement", "all"


class PolicySimulator:
    """
    Simulates policy impacts on LCA emissions.
    
    Policies:
    - Green H₂ mandates
    - CCUS subsidies
    - Carbon border adjustments (CBAM)
    - Renewable energy targets
    """
    
    def __init__(self, lca_engine: LCAEngine):
        """
        Initialize policy simulator.
        
        Args:
            lca_engine: Core LCA engine
        """
        self.engine = lca_engine
        self.policies = []
        
        logger.info("Policy Simulator initialized")
    
    def add_policy(self, policy: Policy):
        """Add a policy to simulate"""
        self.policies.append(policy)
        logger.info(f"Added policy: {policy.name}")
    
    def simulate_policy_impact(
        self,
        baseline_emissions: float,
        sector: str,
        policy: Policy
    ) -> Dict[str, float]:
        """
        Simulate impact of a single policy.
        
        Args:
            baseline_emissions: Baseline emissions (kg CO2eq)
            sector: Sector name
            policy: Policy to simulate
        
        Returns:
            Impact results
        """
        if policy.policy_type == "h2_mandate":
            return self._simulate_h2_mandate(baseline_emissions, sector, policy)
        
        elif policy.policy_type == "ccus_subsidy":
            return self._simulate_ccus_subsidy(baseline_emissions, sector, policy)
        
        elif policy.policy_type == "carbon_border":
            return self._simulate_carbon_border(baseline_emissions, sector, policy)
        
        elif policy.policy_type == "renewable_target":
            return self._simulate_renewable_target(baseline_emissions, sector, policy)
        
        else:
            raise ValueError(f"Unknown policy type: {policy.policy_type}")
    
    def _simulate_h2_mandate(
        self,
        baseline_emissions: float,
        sector: str,
        policy: Policy
    ) -> Dict[str, float]:
        """Simulate green H₂ mandate impact"""
        mandate_percentage = policy.parameters.get("h2_share", 0.5)
        
        # Estimate emission reduction from H2 adoption
        # Steel: ~70% reduction with green H2-DRI vs BF-BOF
        # Shipping: ~90% reduction with green ammonia vs marine fuel
        
        reduction_factors = {
            "steel": 0.70,
            "shipping": 0.90,
            "aviation": 0.85,
            "trucking": 0.80
        }
        
        reduction_factor = reduction_factors.get(sector, 0.70)
        effective_reduction = reduction_factor * mandate_percentage
        
        new_emissions = baseline_emissions * (1 - effective_reduction)
        emission_reduction = baseline_emissions - new_emissions
        
        return {
            "baseline_emissions": baseline_emissions,
            "new_emissions": new_emissions,
            "emission_reduction": emission_reduction,
            "reduction_percentage": (emission_reduction / baseline_emissions) * 100,
            "policy_cost": self._estimate_h2_cost(baseline_emissions, mandate_percentage)
        }
    
    def _simulate_ccus_subsidy(
        self,
        baseline_emissions: float,
        sector: str,
        policy: Policy
    ) -> Dict[str, float]:
        """Simulate CCUS subsidy impact"""
        subsidy_rate = policy.parameters.get("subsidy_per_ton_co2", 50.0)  # USD/t CO2
        adoption_rate = policy.parameters.get("adoption_rate", 0.3)
        
        # CCUS can capture 85-95% of emissions
        capture_rate = policy.parameters.get("capture_rate", 0.90)
        effective_capture = capture_rate * adoption_rate
        
        new_emissions = baseline_emissions * (1 - effective_capture)
        emission_reduction = baseline_emissions - new_emissions
        
        # Calculate subsidy cost
        total_captured = emission_reduction
        subsidy_cost = total_captured * subsidy_rate / 1000  # Convert to USD
        
        return {
            "baseline_emissions": baseline_emissions,
            "new_emissions": new_emissions,
            "emission_reduction": emission_reduction,
            "reduction_percentage": (emission_reduction / baseline_emissions) * 100,
            "subsidy_cost": subsidy_cost,
            "cost_per_ton_reduced": subsidy_cost / (emission_reduction / 1000)
        }
    
    def _simulate_carbon_border(
        self,
        baseline_emissions: float,
        sector: str,
        policy: Policy
    ) -> Dict[str, float]:
        """Simulate carbon border adjustment (CBAM) impact"""
        carbon_price = policy.parameters.get("carbon_price", 50.0)  # USD/t CO2
        import_share = policy.parameters.get("import_share", 0.3)
        
        # CBAM incentivizes domestic production with lower emissions
        # Assumes 20% shift from high-emission imports to low-emission domestic
        
        shift_percentage = policy.parameters.get("shift_percentage", 0.2)
        domestic_emission_factor = 0.7  # Domestic production is cleaner
        
        import_emissions = baseline_emissions * import_share
        shifted_emissions = import_emissions * shift_percentage * domestic_emission_factor
        
        new_emissions = baseline_emissions - (import_emissions * shift_percentage) + shifted_emissions
        emission_reduction = baseline_emissions - new_emissions
        
        # Carbon border tax revenue
        border_tax_revenue = import_emissions * (1 - shift_percentage) * carbon_price / 1000
        
        return {
            "baseline_emissions": baseline_emissions,
            "new_emissions": new_emissions,
            "emission_reduction": emission_reduction,
            "reduction_percentage": (emission_reduction / baseline_emissions) * 100,
            "border_tax_revenue": border_tax_revenue
        }
    
    def _simulate_renewable_target(
        self,
        baseline_emissions: float,
        sector: str,
        policy: Policy
    ) -> Dict[str, float]:
        """Simulate renewable energy target impact"""
        target_percentage = policy.parameters.get("renewable_target", 0.5)
        current_renewable = policy.parameters.get("current_renewable", 0.2)
        
        # Grid carbon intensity reduction
        current_ci = 300.0  # g CO2/kWh
        renewable_ci = 50.0  # g CO2/kWh (wind/solar)
        fossil_ci = 800.0  # g CO2/kWh (coal/gas)
        
        new_ci = (target_percentage * renewable_ci) + ((1 - target_percentage) * fossil_ci)
        current_ci_avg = (current_renewable * renewable_ci) + ((1 - current_renewable) * fossil_ci)
        
        ci_reduction = (current_ci_avg - new_ci) / current_ci_avg
        
        # Estimate electricity share in emissions
        electricity_share = policy.parameters.get("electricity_share", 0.3)
        effective_reduction = ci_reduction * electricity_share
        
        new_emissions = baseline_emissions * (1 - effective_reduction)
        emission_reduction = baseline_emissions - new_emissions
        
        return {
            "baseline_emissions": baseline_emissions,
            "new_emissions": new_emissions,
            "emission_reduction": emission_reduction,
            "reduction_percentage": (emission_reduction / baseline_emissions) * 100,
            "grid_ci_reduction": ci_reduction * 100
        }
    
    def _estimate_h2_cost(self, baseline_emissions: float, h2_share: float) -> float:
        """Estimate cost of green H2 production"""
        # Simplified cost estimation
        # Green H2: ~$3-5/kg, traditional: ~$1-2/kg
        h2_cost_premium = 3.0  # USD/kg
        h2_requirement = baseline_emissions * h2_share * 0.01  # Simplified
        return h2_requirement * h2_cost_premium
    
    def compare_policies(
        self,
        baseline_emissions: float,
        sector: str,
        policies: List[Policy]
    ) -> pd.DataFrame:
        """
        Compare multiple policies.
        
        Returns:
            DataFrame with policy comparison
        """
        results = []
        
        for policy in policies:
            impact = self.simulate_policy_impact(baseline_emissions, sector, policy)
            results.append({
                "policy_name": policy.name,
                "policy_type": policy.policy_type,
                "emission_reduction": impact["emission_reduction"],
                "reduction_percentage": impact["reduction_percentage"],
                "cost": impact.get("subsidy_cost", impact.get("policy_cost", 0.0))
            })
        
        return pd.DataFrame(results)

