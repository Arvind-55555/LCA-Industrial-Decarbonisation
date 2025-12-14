#!/usr/bin/env python3
"""
Complete ML-Enhanced Workflow
Runs LCA calculations with ML models and generates visualizations
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timedelta
import logging
from lca_optimizer.core.engine import LCAEngine
from lca_optimizer.core.ml_enhanced_engine import MLEnhancedLCAEngine
from lca_optimizer.sectors.steel import SteelH2DRIOptimizer, SteelProcessConfig
from lca_optimizer.sectors.cement import CementCCUSOptimizer, CementProcessConfig, CCUSTechnology
from lca_optimizer.sectors.shipping import ShippingFuelComparator, FuelConfig, FuelType, RouteConfig
from lca_optimizer.sectors.aluminium import AluminiumElectrificationOptimizer, AluminiumProcessConfig
from lca_optimizer.data.local_data_loader import LocalGridDataLoader
from lca_optimizer.visualization.plots import plot_lca_results, plot_sector_comparison, plot_time_series_lca
from lca_optimizer.visualization.ml_results_visualization import plot_ml_model_comparison, plot_ml_model_performance
from lca_optimizer.visualization.summary_report import VisualizationSummary
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_complete_workflow():
    """Run complete ML-enhanced workflow"""
    print("=" * 70)
    print("ML-Enhanced LCA Workflow")
    print("=" * 70)
    
    # Initialize both engines for comparison
    print("\n1. Initializing LCA Engines...")
    rule_engine = LCAEngine()  # Rule-based for comparison
    ml_engine = MLEnhancedLCAEngine(use_ml_models=True)  # ML-enhanced
    print("   ✅ Rule-based engine initialized")
    print("   ✅ ML-enhanced engine initialized")
    
    # Store results for comparison
    rule_based_results = {}
    ml_enhanced_results = {}
    
    # Initialize grid data loader
    grid_loader = LocalGridDataLoader(data_dir="data/raw")
    
    # Initialize visualization
    summary_gen = VisualizationSummary(output_dir="outputs")
    
    # Collect LCA results with ML enhancement
    print("\n2. Running LCA Calculations with ML Models...")
    lca_results = {}
    
    # Steel
    print("\n   📊 Steel H2-DRI Optimization...")
    steel_config = SteelProcessConfig(
        h2_pathway="electrolysis",
        electrolyzer_type="PEM",
        renewable_mix={"wind": 0.6, "solar": 0.4},
        iron_ore_source="Australia",
        process_heat_source="electric",
        location="US",
        production_capacity=1000000.0
    )
    
    # Rule-based
    rule_steel_optimizer = SteelH2DRIOptimizer(rule_engine)
    rule_steel_result = rule_steel_optimizer.optimize(steel_config)
    rule_emissions = rule_steel_result.get("optimal_lca", {}).total_emissions if hasattr(rule_steel_result.get("optimal_lca", {}), 'total_emissions') else rule_steel_result.get("optimal_emissions", 0)
    rule_based_results["steel"] = rule_emissions
    
    # ML-enhanced
    ml_steel_optimizer = SteelH2DRIOptimizer(ml_engine)
    ml_steel_result = ml_steel_optimizer.optimize(steel_config)
    ml_emissions = ml_steel_result.get("optimal_lca", {}).total_emissions if hasattr(ml_steel_result.get("optimal_lca", {}), 'total_emissions') else ml_steel_result.get("optimal_emissions", 0)
    ml_enhanced_results["steel"] = ml_emissions
    
    lca_results["steel"] = {
        "total_emissions": ml_emissions,
        "emission_reduction": ml_steel_result.get("emission_reduction", 0),
        "breakdown": ml_steel_result.get("breakdown", {})
    }
    print(f"      ✅ Steel: {ml_steel_result.get('emission_reduction', 0):.1f}% reduction")
    print(f"      📈 Rule-Based: {rule_emissions:,.0f} kg CO2eq")
    print(f"      📈 ML-Enhanced: {ml_emissions:,.0f} kg CO2eq")
    
    # Cement
    print("\n   📊 Cement CCUS Optimization...")
    cement_config = CementProcessConfig(
        capture_technology=CCUSTechnology.POST_COMBUSTION,
        capture_rate=0.90,
        clinker_substitution={"calcined_clay": 0.3},
        alternative_raw_materials={},
        location="EU",
        production_capacity=500000.0,
        co2_storage_location="North Sea"
    )
    
    # Rule-based
    rule_cement_optimizer = CementCCUSOptimizer(rule_engine)
    rule_cement_result = rule_cement_optimizer.optimize(cement_config)
    rule_emissions = rule_cement_result.get("optimal_emissions", 0)
    rule_based_results["cement"] = rule_emissions
    
    # ML-enhanced
    ml_cement_optimizer = CementCCUSOptimizer(ml_engine)
    ml_cement_result = ml_cement_optimizer.optimize(cement_config)
    ml_emissions = ml_cement_result.get("optimal_emissions", 0)
    ml_enhanced_results["cement"] = ml_emissions
    
    lca_results["cement"] = {
        "total_emissions": ml_emissions,
        "emission_reduction": ml_cement_result.get("emission_reduction_vs_baseline", 0),
        "breakdown": ml_cement_result.get("breakdown", {})
    }
    print(f"      ✅ Cement: {ml_cement_result.get('emission_reduction_vs_baseline', 0):.1f}% reduction")
    print(f"      📈 Rule-Based: {rule_emissions:,.0f} kg CO2eq")
    print(f"      📈 ML-Enhanced: {ml_emissions:,.0f} kg CO2eq")
    
    # Shipping
    print("\n   📊 Shipping Fuel Comparison...")
    try:
        shipping_comparator = ShippingFuelComparator(engine)
        route = RouteConfig(
            origin="Rotterdam",
            destination="Shanghai",
            distance=18000,
            payload=10000.0,  # Required parameter
            vessel_type="container"
        )
        fuels = [
            FuelConfig(
                fuel_type=FuelType.GREEN_AMMONIA,
                feedstock_source="atmospheric",
                renewable_electricity_source={"wind": 0.6, "solar": 0.4},
                location="EU",
                production_capacity=100000.0
            ),
            FuelConfig(
                fuel_type=FuelType.MARINE_FUEL,
                feedstock_source="fossil",
                renewable_electricity_source={},
                location="EU",
                production_capacity=100000.0
            )
        ]
        shipping_result = shipping_comparator.compare_fuels(fuels, route)
    except Exception as e:
        logger.warning(f"Shipping comparison failed: {e}")
        shipping_result = None
    if shipping_result and isinstance(shipping_result, dict):
        if "optimal_fuel" in shipping_result:
            optimal = shipping_result["optimal_fuel"]
            if isinstance(optimal, dict):
                lca_results["shipping"] = {
                    "total_emissions": optimal.get("total_route_emissions", 0),
                    "emission_reduction": optimal.get("reduction_vs_baseline", 0),
                    "breakdown": {
                        "wtt_emissions": optimal.get("wtt_emissions", 0),
                        "ttw_emissions": optimal.get("ttw_emissions", 0)
                    }
                }
                print(f"      ✅ Shipping: Best fuel - {optimal.get('fuel_type', 'Unknown')}")
                print(f"      📈 ML-Enhanced Emissions: {lca_results['shipping']['total_emissions']:,.0f} kg CO2eq")
            else:
                print(f"      ⚠️ Shipping: Optimal fuel format unexpected")
        else:
            print(f"      ⚠️ Shipping: Results format unexpected")
    
    # Aluminium
    print("\n   📊 Aluminium Electrification...")
    try:
        aluminium_optimizer = AluminiumElectrificationOptimizer(engine)
        aluminium_config = AluminiumProcessConfig(
            smelting_technology="electrolysis",
            grid_renewable_share=0.8,
            recycling_rate=0.5,
            location="US",
            production_capacity=500000.0
        )
        aluminium_result = aluminium_optimizer.optimize(aluminium_config)
    except Exception as e:
        logger.warning(f"Aluminium optimization failed: {e}")
        aluminium_result = {"optimal_emissions": 0, "emission_reduction": 0, "breakdown": {}}
    lca_results["aluminium"] = {
        "total_emissions": aluminium_result.get("optimal_emissions", 0),
        "emission_reduction": aluminium_result.get("emission_reduction", 0),
        "breakdown": aluminium_result.get("breakdown", {})
    }
    print(f"      ✅ Aluminium: {aluminium_result.get('emission_reduction', 0):.1f}% reduction")
    print(f"      📈 ML-Enhanced Emissions: {lca_results['aluminium']['total_emissions']:,.0f} kg CO2eq")
    
    # Generate ML comparison visualization
    print("\n3. Generating ML Model Visualizations...")
    plots_dir = Path("outputs/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # ML Model Comparison
    if rule_based_results and ml_enhanced_results:
        print("   📊 Generating ML model comparison...")
        plot_ml_model_comparison(
            rule_based_results,
            ml_enhanced_results,
            title="ML Model Enhancement: Rule-Based vs ML-Enhanced LCA",
            save_path=str(plots_dir / "ml_model_comparison.png")
        )
    
    # Generate visualizations
    print("\n4. Generating LCA Visualizations...")
    
    # Steel LCA breakdown
    if "steel" in lca_results and lca_results["steel"]["breakdown"]:
        print("   📊 Generating steel LCA breakdown...")
        plot_lca_results(
            lca_results["steel"]["breakdown"],
            title="Steel H2-DRI LCA Breakdown (ML-Enhanced)",
            save_path=str(plots_dir / "steel_lca_breakdown.png")
        )
    
    # Cement LCA breakdown (with improved readability)
    if "cement" in lca_results and lca_results["cement"]["breakdown"]:
        print("   📊 Generating cement LCA breakdown...")
        plot_lca_results(
            lca_results["cement"]["breakdown"],
            title="Cement CCUS LCA Breakdown (ML-Enhanced)",
            save_path=str(plots_dir / "cement_lca_breakdown.png")
        )
    
    # Sector comparison
    print("   📊 Generating sector comparison...")
    sector_data = {
        sector: {
            "total_emissions": results.get("total_emissions", 0),
            "reduction": results.get("emission_reduction", 0)
        }
        for sector, results in lca_results.items()
    }
    plot_sector_comparison(
        sector_data,
        title="Sector Comparison (ML-Enhanced)",
        save_path=str(plots_dir / "sector_comparison.png")
    )
    
    # Grid CI time series
    print("   📊 Generating grid CI time series...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    grid_data_list = []
    for location in ["US", "EU", "DE", "FR"]:
        historical = grid_loader.get_historical_carbon_intensity(
            location, start_date, end_date, frequency="hourly"
        )
        if not historical.empty:
            historical['location'] = location
            grid_data_list.append(historical)
    
    if grid_data_list:
        grid_data = pd.concat(grid_data_list, ignore_index=True)
        for location in grid_data['location'].unique():
            location_data = grid_data[grid_data['location'] == location]
            plot_time_series_lca(
                location_data,
                location=location,
                title=f"Grid Carbon Intensity - {location} (ML-Enhanced)",
                save_path=str(plots_dir / f"grid_ci_{location.lower()}_ml.png")
            )
    
    # Generate summary report
    print("\n5. Generating Summary Report...")
    report_path = summary_gen.generate_summary_report(
        lca_results=lca_results,
        grid_data=grid_data if grid_data_list else None
    )
    
    print("\n" + "=" * 70)
    print("✅ ML-Enhanced Workflow Complete!")
    print("=" * 70)
    print(f"\n📄 HTML Report: {report_path}")
    print(f"📊 Plots Directory: {plots_dir}")
    print(f"\n📈 ML Model Status:")
    print(f"   - PINN: {'✅ Active' if ml_engine.pinn_model else '❌ Not loaded'}")
    print(f"   - Transformer: {'✅ Active' if ml_engine.transformer_model else '❌ Not loaded'}")
    print(f"\n🚀 Launch Dashboard:")
    print(f"   python run_dashboard.py results")
    print("=" * 70)


if __name__ == "__main__":
    run_complete_workflow()

