#!/usr/bin/env python3
"""
Generate comprehensive visualization summary
Creates all charts, images, and summary reports
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timedelta
from lca_optimizer.core.engine import LCAEngine
from lca_optimizer.sectors.steel import SteelH2DRIOptimizer, SteelProcessConfig
from lca_optimizer.sectors.cement import CementCCUSOptimizer, CementProcessConfig, CCUSTechnology
from lca_optimizer.sectors.shipping import ShippingFuelComparator, FuelConfig, FuelType, RouteConfig
from lca_optimizer.sectors.aluminium import AluminiumElectrificationOptimizer, AluminiumProcessConfig
# Policy module removed - focusing on LCA process improvement
from lca_optimizer.data.local_data_loader import LocalGridDataLoader
from lca_optimizer.visualization.summary_report import VisualizationSummary
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Generate comprehensive visualization summary"""
    print("=" * 70)
    print("Generating Comprehensive Visualization Summary")
    print("=" * 70)
    
    # Initialize components
    engine = LCAEngine()
    grid_loader = LocalGridDataLoader(data_dir="data/raw")
    summary_gen = VisualizationSummary(output_dir="outputs")
    
    # Collect LCA results
    print("\n1. Collecting LCA Results...")
    lca_results = {}
    
    # Steel
    steel_optimizer = SteelH2DRIOptimizer(engine)
    steel_config = SteelProcessConfig(
        h2_pathway="electrolysis",
        electrolyzer_type="PEM",
        renewable_mix={"wind": 0.6, "solar": 0.4},
        iron_ore_source="Australia",
        process_heat_source="electric",
        location="US",
        production_capacity=1000000.0
    )
    steel_result = steel_optimizer.optimize(steel_config)
    lca_results["steel"] = {
        "total_emissions": steel_result["optimal_lca"],
        "emission_reduction": steel_result["emission_reduction"],
        "breakdown": steel_result["breakdown"]
    }
    print(f"   ✅ Steel: {steel_result['emission_reduction']:.1f}% reduction")
    
    # Cement
    cement_optimizer = CementCCUSOptimizer(engine)
    cement_config = CementProcessConfig(
        capture_technology=CCUSTechnology.POST_COMBUSTION,
        capture_rate=0.90,
        clinker_substitution={"calcined_clay": 0.3},
        alternative_raw_materials={},
        location="EU",
        production_capacity=500000.0
    )
    cement_result = cement_optimizer.optimize(cement_config)
    lca_results["cement"] = {
        "total_emissions": cement_result["optimal_emissions"],
        "emission_reduction": cement_result["emission_reduction_vs_baseline"],
        "breakdown": cement_result["breakdown"]
    }
    print(f"   ✅ Cement: {cement_result['emission_reduction_vs_baseline']:.1f}% reduction")
    
    # Collect grid data
    print("\n2. Collecting Grid Carbon Intensity Data...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    grid_data_list = []
    for location in ["US", "EU", "DE", "FR"]:
        historical = grid_loader.get_historical_carbon_intensity(
            location, start_date, end_date, frequency="hourly"
        )
        if not historical.empty:
            grid_data_list.append(historical)
            print(f"   ✅ {location}: {len(historical)} records")
    
    grid_data = None
    if grid_data_list:
        import pandas as pd
        grid_data = pd.concat(grid_data_list, ignore_index=True)
    
    # Generate summary report
    print("\n3. Generating Visualization Summary...")
    report_path = summary_gen.generate_summary_report(
        lca_results=lca_results,
        grid_data=grid_data
    )
    
    print("\n" + "=" * 70)
    print("✅ Visualization Summary Generated!")
    print("=" * 70)
    print(f"\n📄 HTML Report: {report_path}")
    print(f"📊 Plots Directory: {summary_gen.plots_dir}")
    print(f"\nTo view the dashboard, run:")
    print(f"  python -m lca_optimizer.visualization.results_dashboard run_results_dashboard")
    print("=" * 70)


if __name__ == "__main__":
    main()

