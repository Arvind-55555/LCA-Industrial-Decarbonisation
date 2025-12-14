# Local Data Setup - Quick Guide

## ✅ No API Keys Required!

The system now works with **downloaded datasets** instead of requiring API keys.

## Quick Setup (2 minutes)

### Step 1: Download Sample Data

```bash
python scripts/download_datasets.py --dataset sample
```

This creates `data/raw/sample_grid_data.csv` with realistic data for 8 regions.

### Step 2: Test It Works

```bash
python examples/test_local_data.py
```

You should see carbon intensity values for different regions!

### Step 3: Use in Your Code

```python
from lca_optimizer.core.engine import LCAEngine

engine = LCAEngine()
# Automatically uses local data - no API keys needed!
ci = engine.get_grid_carbon_intensity("US", datetime.now())
print(f"Carbon Intensity: {ci} g CO2eq/kWh")
```

## That's It! 🎉

The system now:
- ✅ Uses local data automatically
- ✅ No API keys required
- ✅ Works offline
- ✅ No rate limits
- ✅ Free to use

## Download Real Datasets (Optional)

If you want real historical data:

### EPA eGRID (US Data)

```bash
python scripts/download_datasets.py --dataset egrid
```

Or manually:
1. Visit https://www.epa.gov/egrid/download-data
2. Download latest eGRID file
3. Save to `data/raw/egrid2022_data.xlsx`

### OPSD (European Data)

```bash
python scripts/download_datasets.py --dataset opsd
```

Or manually:
1. Visit https://open-power-system-data.org/
2. Download time series data
3. Save to `data/raw/opsd_time_series.csv`

## How It Works

The system checks for data in this order:

1. **Local files** (from `data/raw/`) ← **You are here!**
2. APIs (if keys configured)
3. Regional averages (fallback)

## Troubleshooting

**"No data found"**:
- Run: `python scripts/download_datasets.py --dataset sample`
- Check `data/raw/` directory exists

**"Excel files not reading"**:
- Install: `pip install openpyxl`

## More Information

- Full guide: [DATASET_ALTERNATIVES.md](DATASET_ALTERNATIVES.md)
- Data directory: [data/README.md](data/README.md)

