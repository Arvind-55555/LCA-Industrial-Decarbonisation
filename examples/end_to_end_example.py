#!/usr/bin/env python3
"""
End-to-End Example: Complete LCA Optimization Workflow
Demonstrates the full pipeline from data to optimization to visualization
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timedelta
import logging

from lca_optimizer.core.engine import LCAEngine
from lca_optimizer.sectors.steel import SteelH2DRIOptimizer, SteelProcessConfig
from lca_optimizer.sectors.cement import CementCCUSOptimizer, CementProcessConfig, CCUSTechnology
from lca_optimizer.policy.simulator import PolicySimulator, Policy
from lca_optimizer.data.local_data_loader import LocalGridDataLoader
from lca_optimizer.visualization.plots import (
    plot_lca_results, plot_policy_impact, plot_time_series_lca
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Complete end-to-end workflow"""
    print("=" * 70)
    print("End-to-End LCA Optimization Workflow")
    print("=" * 70)
    
    # Step 1: Initialize components
    print("\n1. Initializing Components...")
    engine = LCAEngine()
    grid_loader = LocalGridDataLoader(data_dir="data/raw")
    
    # Step 2: Steel H2-DRI Optimization
    print("\n2. Steel H2-DRI Optimization...")
    print("-" * 70)
    
    steel_optimizer = SteelH2DRIOptimizer(engine)
    steel_config = SteelProcessConfig(
        h2_pathway="electrolysis",
        electrolyzer_type="PEM",
        renewable_mix={"wind": 0.6, "solar": 0.4},
        iron_ore_source="Australia",
        process_heat_source="electric",
        location="US",
        production_capacity=1000000.0  # 1 Mt/year
    )
    
    steel_result = steel_optimizer.optimize(steel_config)
    print(f"   Initial LCA: {steel_result['initial_lca']:,.0f} kg CO2eq/t steel")
    print(f"   Optimal LCA: {steel_result['optimal_lca']:,.0f} kg CO2eq/t steel")
    print(f"   Emission Reduction: {steel_result['emission_reduction']:.1f}%")
    
    # Step 3: Cement CCUS Optimization
    print("\n3. Cement CCUS Optimization...")
    print("-" * 70)
    
    cement_optimizer = CementCCUSOptimizer(engine)
    cement_config = CementProcessConfig(
        capture_technology=CCUSTechnology.POST_COMBUSTION,
        capture_rate=0.90,
        clinker_substitution={"calcined_clay": 0.3, "fly_ash": 0.2},
        alternative_raw_materials={"slag": 0.1},
        location="EU",
        production_capacity=500000.0  # 0.5 Mt/year
    )
    
    cement_result = cement_optimizer.optimize(cement_config)
    print(f"   Baseline: {cement_result['baseline_emissions']:,.0f} kg CO2eq")
    print(f"   Optimal: {cement_result['optimal_emissions']:,.0f} kg CO2eq")
    print(f"   Reduction: {cement_result['emission_reduction_vs_baseline']:.1f}%")
    
    # Step 4: Grid Carbon Intensity Analysis
    print("\n4. Grid Carbon Intensity Analysis...")
    print("-" * 70)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    historical = grid_loader.get_historical_carbon_intensity(
        "US", start_date, end_date, frequency="hourly"
    )
    
    if not historical.empty:
        print(f"   Records: {len(historical)}")
        print(f"   Average CI: {historical['carbon_intensity'].mean():.1f} g CO2eq/kWh")
        print(f"   Min CI: {historical['carbon_intensity'].min():.1f} g CO2eq/kWh")
        print(f"   Max CI: {historical['carbon_intensity'].max():.1f} g CO2eq/kWh")
    
    # Step 5: Policy Simulation
    print("\n5. Policy Impact Simulation...")
    print("-" * 70)
    
    simulator = PolicySimulator(engine)
    
    policies = [
        Policy(
            name="Green H2 Mandate 2030",
            policy_type="h2_mandate",
            parameters={"h2_share": 0.5},
            start_date=datetime(2025, 1, 1),
            sector="steel"
        ),
        Policy(
            name="CCUS Subsidy Program",
            policy_type="ccus_subsidy",
            parameters={
                "subsidy_per_ton_co2": 50.0,
                "adoption_rate": 0.3,
                "capture_rate": 0.90
            },
            start_date=datetime(2025, 1, 1),
            sector="cement"
        ),
        Policy(
            name="Renewable Energy Target",
            policy_type="renewable_target",
            parameters={
                "renewable_target": 0.6,
                "current_renewable": 0.3,
                "electricity_share": 0.4
            },
            start_date=datetime(2025, 1, 1),
            sector="all"
        )
    ]
    
    baseline_emissions = 1800000.0  # kg CO2eq
    policy_impacts = {}
    
    for policy in policies:
        impact = simulator.simulate_policy_impact(
            baseline_emissions,
            policy.sector or "steel",
            policy
        )
        policy_impacts[policy.name] = impact
        print(f"   {policy.name}:")
        print(f"     Reduction: {impact['reduction_percentage']:.1f}%")
        print(f"     New Emissions: {impact['new_emissions']:,.0f} kg CO2eq")
    
    # Step 6: Visualization
    print("\n6. Generating Visualizations...")
    print("-" * 70)
    
    output_dir = Path("outputs/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot LCA breakdown
    plot_lca_results(
        steel_result['breakdown'],
        title="Steel H2-DRI LCA Breakdown",
        save_path=str(output_dir / "steel_lca_breakdown.png")
    )
    print("   ✅ Created: steel_lca_breakdown.png")
    
    # Plot policy impact
    plot_policy_impact(
        baseline_emissions,
        {name: {
            "new_emissions": impact["new_emissions"],
            "reduction_percentage": impact["reduction_percentage"]
        } for name, impact in policy_impacts.items()},
        title="Policy Impact Comparison",
        save_path=str(output_dir / "policy_impact.png")
    )
    print("   ✅ Created: policy_impact.png")
    
    # Plot time series
    if not historical.empty:
        plot_time_series_lca(
            historical,
            location="US",
            title="Grid Carbon Intensity - US (Last 7 Days)",
            save_path=str(output_dir / "grid_ci_timeseries.png")
        )
        print("   ✅ Created: grid_ci_timeseries.png")
    
    # Step 7: Summary Report
    print("\n7. Summary Report...")
    print("=" * 70)
    print("\nOPTIMIZATION RESULTS:")
    print(f"  Steel H2-DRI: {steel_result['emission_reduction']:.1f}% reduction")
    print(f"  Cement CCUS: {cement_result['emission_reduction_vs_baseline']:.1f}% reduction")
    
    print("\nPOLICY IMPACTS:")
    for name, impact in policy_impacts.items():
        print(f"  {name}: {impact['reduction_percentage']:.1f}% reduction")
    
    print(f"\n📊 Visualizations saved to: {output_dir}/")
    print("\n" + "=" * 70)
    print("✅ End-to-end workflow completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()

