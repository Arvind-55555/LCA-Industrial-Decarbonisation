# Automated Indian Data Collection

The platform includes an automated web scraping system that downloads Indian industrial data from open sources and organizes it in the `data/raw/indian/` folder.

## Quick Start

```bash
# Scrape all available Indian data sources
python scripts/scrape_indian_data.py --source all

# Scrape specific source
python scripts/scrape_indian_data.py --source cpcb
python scripts/scrape_indian_data.py --source cea
python scripts/scrape_indian_data.py --source datagov
```

## What Gets Scraped

### 1. Industrial Emissions Data
- **Source**: CPCB (Central Pollution Control Board), data.gov.in
- **Output**: `data/raw/indian/industrial_emissions.csv`
- **Contains**: Sector-wise emissions (steel, cement, aluminium, chemicals, refining) by state and year
- **Columns**: `sector`, `state`, `year`, `emissions_tco2`, `production_tonnes`, `energy_consumption_mwh`, `grid_carbon_intensity`

### 2. Grid Carbon Intensity Data
- **Source**: CEA (Central Electricity Authority)
- **Output**: 
  - `data/raw/indian/grid_ci_all_states.csv` (combined)
  - `data/raw/indian/grid_ci_<State>.csv` (individual state files)
- **Contains**: Monthly grid carbon intensity time series for each Indian state
- **Columns**: `timestamp`, `state`, `carbon_intensity`, `source`

### 3. Public Research Datasets
- **Source**: GitHub repositories, research paper data
- **Output**: Various CSV files based on found datasets

## Data Sources

### Primary Sources

1. **data.gov.in**
   - Public datasets on industrial emissions
   - Grid emissions data
   - Sector-wise statistics
   - **Note**: Some datasets may require API key for full access

2. **CEA (Central Electricity Authority)**
   - Monthly power sector reports
   - State-wise generation data
   - Grid emission factors
   - Website: https://cea.nic.in/

3. **CPCB (Central Pollution Control Board)**
   - Industrial emissions inventory
   - Annual pollution reports
   - Sector-wise emissions data
   - Website: https://cpcb.nic.in/

4. **Public Research Repositories**
   - GitHub repositories with Indian industrial data
   - Research paper supplementary data
   - Open data repositories

## How It Works

1. **Web Scraping**: Uses BeautifulSoup to parse HTML pages
2. **API Access**: Attempts to access public APIs (data.gov.in)
3. **Data Generation**: If scraping fails, generates realistic sample data based on:
   - Indian industry averages
   - State-specific characteristics
   - Regional variations
   - Historical patterns

4. **Data Organization**: Automatically:
   - Normalizes column names
   - Formats data types
   - Saves to appropriate locations
   - Creates both combined and state-specific files

## Data Quality

### Real Data (When Available)
- Scraped from official government sources
- Based on actual reports and datasets
- Includes temporal variations

### Sample Data (Fallback)
- Generated when scraping fails or data unavailable
- Based on Indian industry averages and state characteristics
- Realistic but synthetic
- Useful for testing and development

## Usage After Scraping

Once data is scraped:

```bash
# Train models on scraped data
python scripts/train_indian_models.py --sectors steel cement aluminium

# Use in examples
python examples/indian_industrial_example.py
```

## Troubleshooting

### SSL Certificate Errors
- The scraper disables SSL verification for some Indian government sites
- This is safe for public data but use with caution
- If you prefer, you can enable verification in the code

### Connection Errors
- Some government websites may be slow or temporarily unavailable
- The scraper will fall back to generating sample data
- Try running again later if you need real data

### No Data Found
- Some sources may require authentication
- The scraper will generate sample data as fallback
- You can manually download and use `prepare_indian_data.py`

## Data Updates

To update the data:

```bash
# Re-scrape all sources
python scripts/scrape_indian_data.py --source all --force

# Or scrape specific source
python scripts/scrape_indian_data.py --source cpcb --force
```

## File Structure After Scraping

```
data/raw/indian/
├── industrial_emissions.csv          # Sector-wise emissions
├── grid_ci_all_states.csv            # Combined grid CI
├── grid_ci_Maharashtra.csv          # State-specific grid CI
├── grid_ci_Gujarat.csv
├── grid_ci_Tamil_Nadu.csv
└── ... (one file per state)
```

## Next Steps

1. **Review Scraped Data**: Check `data/raw/indian/` for downloaded files
2. **Verify Quality**: Ensure data looks reasonable
3. **Train Models**: Use scraped data for ML model training
4. **Update Regularly**: Re-scrape periodically for fresh data

## Manual Override

If you have manually downloaded data:

```bash
# Prepare manually downloaded file
python scripts/prepare_indian_data.py emissions <your_file.csv>
```

This will format your file to match the expected structure.

