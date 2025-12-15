# Indian Industrial Decarbonisation LCA Platform

**A machine learning-enhanced Life Cycle Assessment platform for Indian heavy industries**

---

An advanced AI-driven system specifically designed for Indian industrial decarbonisation, featuring physics-informed neural networks (PINNs), Transformer models, and Graph Neural Networks (GNNs) trained on Indian industrial data. The platform accounts for Indian-specific characteristics including regional grid carbon intensity variations, DRI process technologies, state-specific industrial policies, and local material supply chains to enable accurate LCA optimization for India's top heavy industries.

---
## Indian Industrial Focus

This platform is specifically adapted for **Indian heavy industries**, with:

- **Indian Data Integration**: Extracts process data from data.gov.in, CEA, CPCB, and Indian industrial portals
- **Regional Grid Carbon Intensity**: State-specific carbon intensity tracking for Indian power grid
- **Indian Process Technologies**: DRI (coal-based), Indian cement clinker ratios, aluminium smelting with high grid dependency
- **State-Specific Constraints**: Accounts for regional variations, material availability, and policy constraints
- **Transfer Learning**: Adapts global models to Indian context with limited data using physics-based regularization

---

## Key Features

- **Indian ML Model Training**: PINN, Transformer, and GNN models trained on Indian industrial data with data augmentation
- **Indian Regional Grid Data**: State-specific carbon intensity for 18+ Indian states with seasonal variations
- **Sector-Specific Indian Optimizers**: Steel (DRI), Cement, Aluminium, Chemicals, Refining with Indian process characteristics
- **Physics-Informed Indian Constraints**: Accounts for grid reliability, material quality variations, climate conditions
- **Transfer Learning & Data Augmentation**: Compensates for limited Indian data using physics-based regularization
- **State-Specific Modeling**: Regional variations in material sourcing, grid reliability, and policy constraints

---
## Live Dashboard

**Access the live dashboard on GitHub Pages:**
🔗 [Dashboard](https://arvind-55555.github.io/LCA-Industrial-Decarbonisation/)

The dashboard is automatically deployed when changes are pushed to the `main` branch.

### Dashboard Features
- **LCA Results**: ML-enhanced baseline vs optimized emissions
- **Grid Carbon Intensity**: Interactive time-series analysis
- **Sector Comparison**: Cross-sector optimization potential
- **Charts & Images**: Generated LCA breakdown visualizations

---
## Project Overview

This system provides dynamic, high-resolution LCA modeling for:
- **Green Hydrogen & Derivatives**: Ammonia, methanol, e-fuels
- **CCUS**: Carbon capture, utilization, and storage effectiveness
- **Electrification**: System-wide impacts and optimization
- **Recycling**: Circular economy integration

## Indian Industrial Sectors Covered

1. **Steel**: Indian DRI processes (coal-based), BF-BOF, regional iron ore sourcing (Odisha/Jharkhand)
2. **Cement**: Indian clinker ratios (0.65-0.85), fly ash/slag substitution, regional material availability
3. **Aluminium**: High grid dependency (85%+), state-specific carbon intensity, captive power considerations
4. **Chemicals**: Indian petrochemical processes with local feedstock variations
5. **Refining**: Indian refinery operations with regional grid characteristics

### Indian-Specific Characteristics Modeled

- **Steel**: Lower quality iron ore (58-62% Fe), coal-based DRI dominance, grid reliability issues
- **Cement**: Higher clinker ratios, limited fly ash availability in some regions, high coal dependency
- **Aluminium**: Higher specific energy consumption (13.5-15.5 MWh/t), state grid variations, limited renewable capacity
- **Grid**: 20% transmission losses, power quality issues, regional reliability variations
- **Climate**: Monsoon impacts, heat wave effects, regional temperature variations

## Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    LCA Optimizer System                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────────┐
        │         Data Layer                          │
        │  - Local Data Loaders                       │
        │  - API Clients (optional)                   │
        │  - LCI Databases                            │
        └─────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────────┐
        │         Core Engine                         │
        │  - LCA Calculations                         │
        │  - Physics Constraints                      │
        │  - Grid CI Integration                      │
        └─────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────────┐
        │         ML Models                           │
        │  - PINN (Physical Validation)               │
        │  - GNN (Process Flows)                      │
        │  - Transformer (Time-Series)                │
        └─────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────────┐
        │         Sector Optimizers                   │
        │  - Steel H2-DRI                             │
        │  - Cement CCUS                              │
        │  - Shipping Fuels                           │
        │  - Aluminium Electrification                │
        └─────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────────┐
        │         Optimization Layer                  │
        │  - Multi-Objective Optimization             │
        │  - RL Agents                                │
        └─────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────────┐
        │         Visualization & Output              │
        │  - Dashboard (Dash/Plotly)                  │
        │  - Static Plots (Matplotlib)                │
        │  - Reports (HTML)                           │
        └─────────────────────────────────────────────┘
```

### Data Flow

```
INPUT    →    Data Ingestion → Core Processing → Optimization → Visualization
  │           │                │                │              │
  │           │                │                │              │
  ├─► User    ├─► Local Data   ├─► LCA Engine   ├─► Multi-     ├─► Dashboard
  │   Input   │   Loaders      │                │   Objective  │   (Port 8050)
  │           │                │                │   Opt        │
  ├─► Config  ├─► API Clients  ├─► ML Models    ├─► RL Agents  ├─► Static Plots
  │   Files   │   (Optional)   │                │              │   (PNG)
  │           │                │                │              │
  └─► Data    └─► LCI DBs      └─► Sector       └─► Results    └─► HTML Reports
      Sources                      Optimizers                    (outputs/)
```

**Detailed Data Flow**: See [docs/DATAFLOW.md](docs/DATAFLOW.md)  
**System Architecture**: See [docs/SYSTEM_ARCHITECTURE.md](docs/SYSTEM_ARCHITECTURE.md)

### Core Components

- **Dynamic LCA Engine**: Physics-informed neural networks for real-time LCA calculations
- **Sector-Specific Optimizers**: Specialized models for each industrial sector
- **Data Pipeline**: Integration with LCI databases, real-time grid data, and process data
- **Optimization Engine**: Multi-objective optimization with RL for operational control
- **API Layer**: LCA-as-a-Service endpoints
- **Visualization Dashboard**: Interactive Dash/Plotly dashboard for results visualization

## Installation

### Basic Installation

```bash
# Clone the repository
git clone <repository-url>
cd Industrial-Decarbonisation

# Install core dependencies
pip install -r requirements.txt
```

### Optional Dependencies

Some advanced features require additional packages:

- **Physics-Informed ML**: `deepxde` (requires Python <3.12)
- **NVIDIA Modulus**: For advanced PINN features (requires NVIDIA GPU setup)
- **Visualization**: `dash` and `plotly` for interactive dashboards

See `requirements-optional.txt` for details.

**Note**: The core functionality works without these optional dependencies. The ML models (PINN and Transformer) use PyTorch directly and are fully functional out of the box.

## Usage

### Complete ML-Enhanced Workflow (Recommended)

Run the complete workflow with ML models integrated:

```bash
python scripts/run_complete_ml_workflow.py
```

This will:
1. Initialize ML-enhanced LCA engine (loads PINN & Transformer models)
2. Run LCA calculations for all sectors (steel, cement, shipping, aluminium)
3. Compare rule-based vs ML-enhanced results
4. Generate all visualizations (LCA breakdowns, sector comparison, ML comparisons)
5. Create comprehensive summary reports

### Step-by-Step Workflow

1. **Download Local Datasets** (No API keys needed!):
   ```bash
   python scripts/download_datasets.py --dataset sample
   ```

2. **Test Local Data**:
   ```bash
   python examples/test_local_data.py
   ```

3. **Run Complete Workflow**:
   ```bash
   python scripts/run_complete_ml_workflow.py
   ```

4. **Launch Interactive Dashboard** (Final Step):
   ```bash
   python run_dashboard.py results
   ```
   Access at: **http://localhost:8050**

### Additional Tools

- **Scrape Indian Data Automatically**: `python scripts/scrape_indian_data.py --source all` (downloads from open sources)
- **Train Indian ML Models**: `python scripts/train_indian_models.py --sectors steel cement aluminium --states Maharashtra Gujarat`
- **Prepare Manually Downloaded Data**: `python scripts/prepare_indian_data.py emissions <downloaded_file.csv>`
- **Download Datasets**: `python scripts/download_datasets.py --dataset sample`
- **Train Global Models**: `python train_models.py --data-dir data/sample --output-dir models/trained`
- **Run API Server**: `python run_api.py`
- **Generate Summary Report**: `python scripts/generate_visualization_summary.py`

### API Endpoints

- `/lca/steel_h2_dri` - Steel H₂-DRI LCA optimization
- `/lca/shipping_fuel_comparison` - Shipping fuel comparison
- `/lca/cement_ccus_optimization` - Cement CCUS optimization
- `/lca/aluminium_electrification` - Aluminium electrification LCA

### Example Usage: Indian Steel Sector

```python
from lca_optimizer.core.engine import LCAEngine
from lca_optimizer.sectors.indian_steel import IndianSteelOptimizer, IndianSteelProcessConfig

engine = LCAEngine()
optimizer = IndianSteelOptimizer(engine, state="Maharashtra")

config = IndianSteelProcessConfig(
    process_type="DRI",
    state="Maharashtra",
    iron_ore_source="Odisha",
    coal_type="Indian coal",
    production_capacity=2000000.0,  # 2 million tonnes/year
    grid_dependency=0.75
)

result = optimizer.optimize(config)
print(f"Emission reduction: {result['emission_reduction']:.1f}%")
print(f"Grid CI: {result['grid_ci']:.2f} kg CO2/kWh")
print(f"Grid reliability: {result['indian_specific_factors']['grid_reliability']:.2f}")
```

### Example: Indian Cement Sector

```python
from lca_optimizer.sectors.indian_cement import IndianCementOptimizer, IndianCementProcessConfig

optimizer = IndianCementOptimizer(engine, state="Rajasthan")

config = IndianCementProcessConfig(
    state="Rajasthan",
    production_capacity=5000000.0,  # 5 million tonnes/year
    clinker_ratio=0.75,
    fly_ash_substitution=0.25,
    capture_technology="post-combustion",
    capture_rate=0.90
)

result = optimizer.optimize(config)
print(f"Emission reduction: {result['emission_reduction']:.1f}%")
```

### Example: Cement CCUS Optimization

```python
from lca_optimizer.core.ml_enhanced_engine import MLEnhancedLCAEngine
from lca_optimizer.sectors.cement import CementCCUSOptimizer, CementProcessConfig, CCUSTechnology

engine = MLEnhancedLCAEngine(use_ml_models=True)
optimizer = CementCCUSOptimizer(engine)

config = CementProcessConfig(
    capture_technology=CCUSTechnology.POST_COMBUSTION,
    capture_rate=0.90,
    clinker_substitution={"calcined_clay": 0.3},
    alternative_raw_materials={},
    location="EU",
    production_capacity=500000.0,
    co2_storage_location="North Sea"
)

result = optimizer.optimize(config)
print(f"Emission reduction: {result['emission_reduction_vs_baseline']:.1f}%")
print(f"Captured CO₂: {result['breakdown']['captured_co2']:,.0f} kg CO₂eq")
```

## Project Structure

```
Industrial-Decarbonisation/
├── lca_optimizer/
│   ├── core/           # Core LCA engine and physics models
│   ├── sectors/        # Sector-specific implementations
│   ├── data/           # Data pipeline and LCI integration
│   ├── models/         # ML models (PINNs, GNNs, Transformers)
│   ├── optimization/   # Optimization and RL engines
│   ├── api/            # API layer and endpoints
│   ├── utils/          # Utilities and helpers
│   └── config/         # Configuration management
├── tests/              # Unit and integration tests
├── notebooks/          # Jupyter notebooks for exploration
├── docs/               # Documentation
└── deployment/         # Deployment configurations
```

## Key Features

- **Dynamic LCA Modeling**: Real-time updates based on grid carbon intensity
- **Physics-Informed ML**: Ensures thermodynamic and stoichiometric constraints
- **Uncertainty Quantification**: Bayesian methods for LCA confidence intervals
- **Multi-Objective Optimization**: Balance emissions, cost, and feasibility
- **Real-Time Grid Data**: Integration with local data and optional APIs
- **ML Model Training**: Training utilities for PINN, GNN, and Transformer models
- **Interactive Visualization Dashboard**: Complete visualization of all results (Final Step)

## Complete Workflow

The system follows a complete workflow ending with visualization:

1. **Data Preparation** → Load and validate data
2. **LCA Calculation** → Calculate emissions with physics validation
3. **ML Model Enhancement** → Enhance predictions with PINN and Transformer models
4. **Optimization** → Optimize for minimal emissions
5. **Visualization Dashboard** → Interactive visualization of all results

**ML Models are integrated by default** - The workflow uses ML-enhanced LCA engine for improved accuracy.

See [docs/COMPLETE_WORKFLOW.md](docs/COMPLETE_WORKFLOW.md) for detailed workflow documentation.

## Next Steps

### Immediate Next Steps

1. **Integrate Real LCI Databases**:
   - Connect to Ecoinvent database
   - Integrate GREET fuel pathways
   - Add sector-specific databases (WorldSteel, GCCA, IAI)

2. **Connect Real-Time APIs**:
   - Set up Electricity Maps API key
   - Configure WattTime authentication
   - Add ENTSO-E data integration

3. **Train Models on Real Data**:
   - Collect historical LCA data
   - Train PINN models on process data
   - Fine-tune Transformer models on time-series data

4. **Extend to Additional Sectors**:
   - Aviation fuel optimization
   - Primary chemicals decarbonisation
   - Oil & gas sector modeling

### Visualization Dashboard

The visualization dashboard is the **final step** in the LCA optimization workflow, providing comprehensive visualization of all results.

**Launch Dashboard**:
```bash
python run_dashboard.py results
```

**Dashboard Features**:
- **LCA Results**: Baseline vs optimized emissions with improvement opportunities
- **Grid Carbon Intensity**: Time period analysis (7d, 30d, 1y, 3y, 5y) with location selection
- **Sector Comparison**: Cross-sector optimization potential and detailed comparison
- **Charts & Images**: Gallery of generated LCA breakdown charts

**Dashboard Tabs**:
1. **LCA Results**: Shows emission reductions, improvement potential, and opportunities
2. **Grid Carbon Intensity**: Interactive time-series with statistics (Mean, Min, Max, Std Dev)
3. **Sector Comparison**: Baseline vs optimized comparison across all sectors
4. **Charts & Images**: Displays generated PNG charts (cement, steel, sector comparison)

**Complete Workflow**:
```
1. Data Preparation → 2. LCA Calculation → 3. Optimization → 4. Visualization Dashboard
```

See [docs/VISUALIZATION_GUIDE.md](docs/VISUALIZATION_GUIDE.md) for detailed visualization documentation.

### Advanced Features

- **Digital Twin Integration**: Connect to plant SCADA systems
- **Production Deployment**: Deploy API to cloud infrastructure
- **Model Serving**: Set up model inference service



See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for deployment details.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License

