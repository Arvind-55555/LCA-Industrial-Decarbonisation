#!/usr/bin/env python3
"""
Demo: Policy Simulation for Industrial Decarbonisation
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime
from lca_optimizer.core.engine import LCAEngine
from lca_optimizer.policy.simulator import PolicySimulator, Policy

def main():
    """Run policy simulation demo"""
    
    # Initialize
    engine = LCAEngine()
    simulator = PolicySimulator(engine)
    
    # Define policies
    policies = [
        Policy(
            name="Green H2 Mandate 2030",
            policy_type="h2_mandate",
            parameters={"h2_share": 0.5},  # 50% green H2 by 2030
            start_date=datetime(2025, 1, 1),
            end_date=datetime(2030, 12, 31),
            sector="steel"
        ),
        Policy(
            name="CCUS Subsidy Program",
            policy_type="ccus_subsidy",
            parameters={
                "subsidy_per_ton_co2": 50.0,  # $50/t CO2
                "adoption_rate": 0.3,  # 30% adoption
                "capture_rate": 0.90  # 90% capture
            },
            start_date=datetime(2025, 1, 1),
            sector="cement"
        ),
        Policy(
            name="EU Carbon Border Adjustment",
            policy_type="carbon_border",
            parameters={
                "carbon_price": 50.0,  # $50/t CO2
                "import_share": 0.3,  # 30% imports
                "shift_percentage": 0.2  # 20% shift to domestic
            },
            start_date=datetime(2026, 1, 1),
            sector="all"
        ),
        Policy(
            name="Renewable Energy Target",
            policy_type="renewable_target",
            parameters={
                "renewable_target": 0.6,  # 60% renewable
                "current_renewable": 0.3,  # Current 30%
                "electricity_share": 0.4  # 40% of emissions from electricity
            },
            start_date=datetime(2025, 1, 1),
            sector="all"
        )
    ]
    
    # Baseline emissions (example: 1 Mt steel production)
    baseline_emissions = 1800000.0  # kg CO2eq (1.8 t CO2/t steel * 1 Mt)
    
    print("=" * 60)
    print("Policy Simulation for Industrial Decarbonisation")
    print("=" * 60)
    print(f"\nBaseline Emissions: {baseline_emissions:,.0f} kg CO2eq")
    print(f"                    {baseline_emissions/1e6:.2f} Mt CO2eq\n")
    
    # Simulate each policy
    print("\nPolicy Impacts:")
    print("-" * 60)
    
    for policy in policies:
        impact = simulator.simulate_policy_impact(
            baseline_emissions,
            policy.sector or "steel",
            policy
        )
        
        print(f"\n{policy.name} ({policy.policy_type}):")
        print(f"  Emission Reduction: {impact['emission_reduction']:,.0f} kg CO2eq")
        print(f"  Reduction Percentage: {impact['reduction_percentage']:.1f}%")
        print(f"  New Emissions: {impact['new_emissions']:,.0f} kg CO2eq")
        
        if "cost" in impact and impact["cost"] > 0:
            print(f"  Estimated Cost: ${impact['cost']:,.0f}")
    
    # Compare policies
    print("\n" + "=" * 60)
    print("Policy Comparison:")
    print("=" * 60)
    
    comparison = simulator.compare_policies(
        baseline_emissions,
        "steel",
        policies
    )
    
    print(comparison.to_string(index=False))
    
    # Find best policy
    best_policy = comparison.loc[comparison['emission_reduction'].idxmax()]
    print(f"\nBest Policy (by emission reduction): {best_policy['policy_name']}")
    print(f"  Reduction: {best_policy['reduction_percentage']:.1f}%")

if __name__ == "__main__":
    main()

