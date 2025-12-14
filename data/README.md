# Data Directory

This directory contains downloaded datasets for grid carbon intensity and LCA data.

## Available Datasets

### 1. EPA eGRID (US Grid Data)
- **Source**: https://www.epa.gov/egrid/download-data
- **Format**: Excel (.xlsx)
- **Content**: US electricity grid carbon intensity by region
- **Download**: Run `python scripts/download_datasets.py --dataset egrid`

### 2. Open Power System Data (OPSD)
- **Source**: https://open-power-system-data.org/
- **Format**: CSV
- **Content**: European electricity grid data
- **Download**: Run `python scripts/download_datasets.py --dataset opsd`

### 3. Sample Grid Data (Synthetic)
- **Source**: Generated locally
- **Format**: CSV
- **Content**: Synthetic grid carbon intensity data for multiple regions
- **Download**: Run `python scripts/download_datasets.py --dataset sample`

## Directory Structure

```
data/
├── raw/              # Downloaded raw datasets
│   ├── egrid2022_data.xlsx
│   ├── opsd_time_series.csv
│   └── sample_grid_data.csv
├── processed/        # Processed/cleaned data (if needed)
└── cache/            # API response cache
```

## Downloading Datasets

### Quick Start

```bash
# Download all datasets
python scripts/download_datasets.py

# Download specific dataset
python scripts/download_datasets.py --dataset egrid
python scripts/download_datasets.py --dataset opsd
python scripts/download_datasets.py --dataset sample
```

### Manual Download

If automatic download fails, you can manually download:

1. **EPA eGRID**:
   - Visit: https://www.epa.gov/egrid/download-data
   - Download the latest eGRID data file
   - Place in `data/raw/` directory

2. **OPSD**:
   - Visit: https://open-power-system-data.org/
   - Download time series data
   - Place in `data/raw/` directory

## Using Local Data

The system automatically uses local data when available:

```python
from lca_optimizer.data.local_data_loader import LocalGridDataLoader

loader = LocalGridDataLoader(data_dir="data/raw")
ci = loader.get_current_carbon_intensity("US")
print(f"Carbon Intensity: {ci.carbon_intensity} g CO2eq/kWh")
```

## Data Sources

### EPA eGRID
- **Coverage**: United States
- **Update Frequency**: Annual
- **License**: Public domain
- **Data**: Grid carbon intensity by region, power plant emissions

### OPSD
- **Coverage**: Europe
- **Update Frequency**: Regular updates
- **License**: Open data
- **Data**: Time series of electricity generation, consumption, prices

### Sample Data
- **Coverage**: Multiple regions (US, EU, DE, FR, GB, CA, CN, IN)
- **Update Frequency**: Generated on demand
- **License**: Synthetic data for testing
- **Data**: Hourly carbon intensity with realistic patterns

## Notes

- Local data takes priority over API calls
- If local data is unavailable, system falls back to APIs or defaults
- Sample data is useful for testing and development
- Real datasets (eGRID, OPSD) provide actual historical data

