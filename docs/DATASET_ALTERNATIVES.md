# Dataset Alternatives Guide

This guide provides alternatives to WattTime and ENTSO-E APIs by using downloadable open datasets.

## ✅ Solution: Local Data Loaders

Instead of relying on API keys, the system now supports:

1. **Downloaded datasets** stored locally
2. **Automatic data loading** from local files
3. **Fallback to synthetic data** when datasets unavailable
4. **No API keys required** for basic functionality

## Available Data Sources

### 1. EPA eGRID (US Grid Data) ⭐ Recommended

**Source**: U.S. Environmental Protection Agency  
**URL**: https://www.epa.gov/egrid/download-data  
**Format**: Excel (.xlsx)  
**Coverage**: United States  
**Update**: Annual  
**License**: Public domain

**Download**:
```bash
python scripts/download_datasets.py --dataset egrid
```

**Manual Download**:
1. Visit https://www.epa.gov/egrid/download-data
2. Download latest eGRID data file
3. Place in `data/raw/` directory

### 2. Open Power System Data (OPSD) (Europe)

**Source**: Open Power System Data  
**URL**: https://open-power-system-data.org/  
**Format**: CSV  
**Coverage**: Europe  
**Update**: Regular  
**License**: Open data

**Download**:
```bash
python scripts/download_datasets.py --dataset opsd
```

**Manual Download**:
1. Visit https://open-power-system-data.org/
2. Download time series data
3. Place in `data/raw/` directory

### 3. Sample Grid Data (Synthetic)

**Source**: Generated locally  
**Format**: CSV  
**Coverage**: Multiple regions (US, EU, DE, FR, GB, CA, CN, IN)  
**Update**: On demand  
**License**: Synthetic for testing

**Generate**:
```bash
python scripts/download_datasets.py --dataset sample
```

## Quick Start

### 1. Download Sample Data (No Internet Required)

```bash
python scripts/download_datasets.py --dataset sample
```

This creates `data/raw/sample_grid_data.csv` with realistic synthetic data.

### 2. Test Local Data Loader

```bash
python examples/test_local_data.py
```

### 3. Use in Your Code

```python
from lca_optimizer.data.local_data_loader import LocalGridDataLoader

# No API keys needed!
loader = LocalGridDataLoader(data_dir="data/raw")
ci = loader.get_current_carbon_intensity("US")
print(f"Carbon Intensity: {ci.carbon_intensity} g CO2eq/kWh")
```

## How It Works

### Priority Order

The system uses data in this order:

1. **Local Data** (from downloaded files) ← **No API keys needed!**
2. **API Data** (if keys configured)
3. **Regional Averages** (fallback)
4. **Synthetic Data** (if nothing else available)

### Automatic Integration

The LCA Engine automatically uses local data:

```python
from lca_optimizer.core.engine import LCAEngine

engine = LCAEngine()
# Automatically uses local data if available
ci = engine.get_grid_carbon_intensity("US", datetime.now())
```

## Data Structure

### Sample Grid Data Format

```csv
timestamp,location,carbon_intensity,renewable_share
2024-01-01 00:00:00,US,420.5,0.35
2024-01-01 01:00:00,US,380.2,0.40
...
```

### Supported Locations

- **US**: United States
- **EU**: European Union average
- **DE**: Germany
- **FR**: France
- **GB**: United Kingdom
- **CA**: Canada
- **CN**: China
- **IN**: India

## Downloading Real Datasets

### EPA eGRID

```bash
# Automatic download
python scripts/download_datasets.py --dataset egrid

# If automatic fails, manual download:
# 1. Visit https://www.epa.gov/egrid/download-data
# 2. Download "eGRID2022 Data File" (or latest)
# 3. Save to data/raw/egrid2022_data.xlsx
```

### OPSD

```bash
# Automatic download
python scripts/download_datasets.py --dataset opsd

# If automatic fails, manual download:
# 1. Visit https://open-power-system-data.org/
# 2. Download time series data
# 3. Save to data/raw/opsd_time_series.csv
```

## Benefits

✅ **No API Keys Required**: Works offline with downloaded data  
✅ **No Rate Limits**: No API quota concerns  
✅ **Faster**: Local data loads instantly  
✅ **Reliable**: No dependency on external APIs  
✅ **Privacy**: Data stays local  
✅ **Cost**: Free (no API costs)

## Comparison

| Feature | API-Based | Local Data |
|---------|-----------|------------|
| API Keys | Required | Not needed |
| Internet | Required | Not needed |
| Rate Limits | Yes | No |
| Cost | May have costs | Free |
| Speed | Network dependent | Instant |
| Reliability | API dependent | Always available |

## Migration from APIs

If you were using WattTime or ENTSO-E:

1. **Download datasets**:
   ```bash
   python scripts/download_datasets.py
   ```

2. **Remove API keys** from `.env` (optional):
   ```bash
   # You can keep them, but they're not required
   # WATTTIME_USERNAME=
   # WATTTIME_PASSWORD=
   # ENTSOE_SECURITY_TOKEN=
   ```

3. **Test local data**:
   ```bash
   python examples/test_local_data.py
   ```

4. **Use as before** - the system automatically uses local data!

## Troubleshooting

### No Data Available

If local data isn't available:
1. Run: `python scripts/download_datasets.py --dataset sample`
2. Check `data/raw/` directory exists
3. Verify files are readable

### Data Outdated

- **Sample data**: Regenerate with download script
- **eGRID**: Download latest from EPA website
- **OPSD**: Download latest from OPSD website

### Excel Files Not Reading

Install openpyxl:
```bash
pip install openpyxl
```

## Next Steps

1. ✅ Download sample data: `python scripts/download_datasets.py --dataset sample`
2. ✅ Test local loader: `python examples/test_local_data.py`
3. ⬜ Download real datasets (eGRID, OPSD) if needed
4. ⬜ Use in your LCA calculations - no API keys needed!

## Resources

- **EPA eGRID**: https://www.epa.gov/egrid
- **OPSD**: https://open-power-system-data.org/
- **IEA Data**: https://www.iea.org/data-and-statistics
- **Data.gov**: https://catalog.data.gov/dataset/?organization=doe-gov

