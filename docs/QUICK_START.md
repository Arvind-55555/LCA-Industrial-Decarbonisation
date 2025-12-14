# Quick Start Guide

Get started with LCA Optimizer in 5 minutes!

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Download Sample Data (No API Keys!)

```bash
python scripts/download_datasets.py --dataset sample
```

This creates `data/raw/sample_grid_data.csv` with realistic data.

## Step 3: Test the System

```bash
# Test local data loader
python examples/test_local_data.py

# Run end-to-end example
python examples/end_to_end_example.py
```

## Step 4: Start API Server

```bash
python run_api.py
```

Visit http://localhost:8000/docs for API documentation.

## Step 5: Try It Out

### Python Example

```python
from lca_optimizer.core.engine import LCAEngine
from lca_optimizer.sectors.steel import SteelH2DRIOptimizer, SteelProcessConfig

# Initialize
engine = LCAEngine()
optimizer = SteelH2DRIOptimizer(engine)

# Configure
config = SteelProcessConfig(
    h2_pathway="electrolysis",
    electrolyzer_type="PEM",
    renewable_mix={"wind": 0.6, "solar": 0.4},
    iron_ore_source="Australia",
    process_heat_source="electric",
    location="US",
    production_capacity=1000000.0
)

# Optimize
result = optimizer.optimize(config)
print(f"Emission reduction: {result['emission_reduction']:.1f}%")
```

### API Example

```bash
curl -X POST "http://localhost:8000/lca/steel_h2_dri" \
  -H "Content-Type: application/json" \
  -d '{
    "h2_pathway": "electrolysis",
    "electrolyzer_type": "PEM",
    "renewable_mix": {"wind": 0.6, "solar": 0.4},
    "iron_ore_source": "Australia",
    "process_heat_source": "electric",
    "location": "US",
    "production_capacity": 1000000.0
  }'
```

## What's Next?

- **Download real datasets**: See [DATASET_ALTERNATIVES.md](DATASET_ALTERNATIVES.md)
- **Set up APIs** (optional): See [API_INTEGRATION_GUIDE.md](API_INTEGRATION_GUIDE.md)
- **Run policy simulation**: `python examples/demo_policy_simulation.py`
- **Train ML models**: `python train_models.py`

## Troubleshooting

**Import errors?**
- Make sure you're in the project directory
- Check all dependencies are installed: `pip install -r requirements.txt`

**No data found?**
- Run: `python scripts/download_datasets.py --dataset sample`
- Check `data/raw/` directory exists

**API not starting?**
- Check port 8000 is available
- Verify no syntax errors: `python -m lca_optimizer.api.main`

## Need Help?

- Check [README.md](README.md) for overview
- See [PROJECT_STATUS.md](PROJECT_STATUS.md) for current capabilities
- Review examples in `examples/` directory

