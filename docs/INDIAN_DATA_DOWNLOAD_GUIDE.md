# Indian Industrial Data - Manual Download Guide

Since the data.gov.in API key may not be working, here's a guide to manually download the required Indian industrial datasets.

## Required Datasets

### 1. Industrial Emissions Data

**Search Keywords:**
- `Indian industrial emissions data`
- `India industrial CO2 emissions by sector`
- `CPCB industrial emissions data`
- `India industrial pollution data`
- `Indian manufacturing emissions dataset`
- `India sector-wise emissions data`

**What to Look For:**
- **Format**: CSV or Excel
- **Required Columns**:
  - `sector` or `industry_type` (steel, cement, aluminium, chemicals, refining)
  - `state` or `state_name`
  - `year` (2015-2024 preferred)
  - `emissions` or `co2_emissions` or `emissions_tco2` (in tonnes CO2)
  - `production` or `production_tonnes` (optional but helpful)
  - `energy_consumption` or `energy_mwh` (optional but helpful)

**Sources to Check:**
- **CPCB (Central Pollution Control Board)**: https://cpcb.nic.in/
  - Look for: Annual reports, industrial emissions inventory
- **data.gov.in**: https://data.gov.in/
  - Search: "industrial emissions", "CO2 emissions", "pollution data"
- **Ministry of Environment, Forest and Climate Change**: https://moef.gov.in/
  - Look for: National inventory reports, emissions data
- **NITI Aayog**: https://www.niti.gov.in/
  - Search: "industrial emissions", "decarbonisation data"

**File to Save As:**
```
data/raw/indian/industrial_emissions.csv
```

---

### 2. Grid Carbon Intensity by State

**Search Keywords:**
- `Indian state-wise grid carbon intensity`
- `India electricity grid CO2 emissions by state`
- `CEA state grid carbon intensity`
- `India power sector emissions by state`
- `Indian state electricity carbon factor`
- `India grid emission factor by state`

**What to Look For:**
- **Format**: CSV or Excel
- **Required Columns**:
  - `state` or `state_name`
  - `timestamp` or `date` or `year` (time series preferred)
  - `carbon_intensity` or `emission_factor` or `grid_ci` (in kg CO2/kWh or g CO2/kWh)
  - `renewable_share` or `renewable_percentage` (optional but helpful)

**Sources to Check:**
- **CEA (Central Electricity Authority)**: https://cea.nic.in/
  - Look for: Monthly/annual reports, power sector statistics
  - Search: "grid emission factor", "carbon intensity", "state-wise generation"
- **data.gov.in**: https://data.gov.in/
  - Search: "electricity", "power sector", "grid emissions", "state electricity"
- **Power System Operation Corporation (POSOCO)**: https://posoco.in/
  - Look for: Grid data, state-wise generation mix
- **Ministry of Power**: https://powermin.gov.in/
  - Look for: Power sector reports, state-wise data

**File to Save As:**
```
data/raw/indian/grid_ci_Maharashtra.csv
data/raw/indian/grid_ci_Gujarat.csv
data/raw/indian/grid_ci_Tamil_Nadu.csv
... (one file per state, or a combined file with state column)
```

**Alternative**: If you find a combined file with all states:
```
data/raw/indian/grid_ci_all_states.csv
```

---

### 3. Steel Sector Process Data

**Search Keywords:**
- `Indian steel industry data`
- `India DRI process data`
- `Indian steel production by state`
- `India steel industry emissions`
- `Indian steel plant data`
- `India iron and steel sector data`

**What to Look For:**
- **Format**: CSV or Excel
- **Required Columns**:
  - `state` or `plant_location`
  - `process_type` (DRI, BF-BOF, EAF)
  - `production_capacity` or `production_tonnes`
  - `specific_energy_consumption` or `energy_mwh_per_tonne`
  - `emission_factor` or `emissions_tco2_per_tonne`
  - `iron_ore_source` (optional)
  - `coal_type` (optional)

**Sources to Check:**
- **Ministry of Steel**: https://steel.gov.in/
  - Look for: Annual reports, plant-wise data, sector statistics
- **Steel Authority of India Limited (SAIL)**: https://sail.co.in/
  - Look for: Annual reports, sustainability reports
- **JSW Steel, Tata Steel**: Annual sustainability reports
- **data.gov.in**: Search "steel", "iron and steel"

**File to Save As:**
```
data/raw/indian/steel_process_data.csv
```

---

### 4. Cement Sector Process Data

**Search Keywords:**
- `Indian cement industry data`
- `India cement production by state`
- `Indian cement clinker ratio`
- `India cement industry emissions`
- `Indian cement plant data`
- `India cement sector statistics`

**What to Look For:**
- **Format**: CSV or Excel
- **Required Columns**:
  - `state` or `plant_location`
  - `clinker_ratio` (0-1)
  - `fly_ash_substitution` (optional)
  - `slag_substitution` (optional)
  - `production_capacity` or `production_tonnes`
  - `specific_energy` or `energy_mwh_per_tonne`
  - `emission_factor` or `emissions_tco2_per_tonne`

**Sources to Check:**
- **Cement Manufacturers' Association (CMA)**: https://www.cmaindia.org/
  - Look for: Annual reports, sector statistics
- **Ministry of Commerce & Industry**: Industry reports
- **data.gov.in**: Search "cement", "cement industry"

**File to Save As:**
```
data/raw/indian/cement_process_data.csv
```

---

### 5. Aluminium Sector Process Data

**Search Keywords:**
- `Indian aluminium industry data`
- `India aluminium smelting data`
- `Indian aluminium production by state`
- `India aluminium industry emissions`
- `Indian aluminium smelter data`
- `India aluminium sector statistics`

**What to Look For:**
- **Format**: CSV or Excel
- **Required Columns**:
  - `state` or `plant_location`
  - `smelting_technology` (pre-baked, Soderberg)
  - `production_capacity` or `production_tonnes`
  - `specific_energy_consumption` or `energy_mwh_per_tonne`
  - `grid_dependency` (0-1, % of energy from grid)
  - `emission_factor` or `emissions_tco2_per_tonne`
  - `grid_carbon_intensity` (optional)

**Sources to Check:**
- **Aluminium Association of India**: Industry reports
- **Hindalco, NALCO, Vedanta**: Annual sustainability reports
- **data.gov.in**: Search "aluminium", "aluminum"

**File to Save As:**
```
data/raw/indian/aluminium_process_data.csv
```

---

## Data Format Requirements

### For `industrial_emissions.csv`:

```csv
sector,state,year,emissions_tco2,production_tonnes,energy_consumption_mwh,grid_carbon_intensity
steel,Maharashtra,2020,4000000,2000000,7000000,0.85
steel,Gujarat,2020,3500000,1750000,6000000,0.80
cement,Rajasthan,2020,3750000,5000000,500000,0.88
aluminium,Odisha,2020,6000000,500000,7250000,0.95
...
```

**Column Descriptions:**
- `sector`: One of: `steel`, `cement`, `aluminium`, `chemicals`, `refining`
- `state`: Full state name (e.g., `Maharashtra`, `Gujarat`, `Tamil Nadu`)
- `year`: Integer year (2015-2024)
- `emissions_tco2`: Total CO2 emissions in tonnes CO2
- `production_tonnes`: Production in tonnes (optional but recommended)
- `energy_consumption_mwh`: Energy consumption in MWh (optional but recommended)
- `grid_carbon_intensity`: Grid carbon intensity in kg CO2/kWh (0.7-1.2 typical range)

### For State Grid CI Files:

**Option 1: One file per state** (`grid_ci_Maharashtra.csv`):
```csv
timestamp,carbon_intensity,state,state_code
2020-01-01 00:00:00,0.85,Maharashtra,MH
2020-01-01 01:00:00,0.87,Maharashtra,MH
...
```

**Option 2: Combined file** (`grid_ci_all_states.csv`):
```csv
timestamp,state,carbon_intensity,renewable_share
2020-01-01 00:00:00,Maharashtra,0.85,0.15
2020-01-01 00:00:00,Gujarat,0.80,0.20
...
```

---

## Quick Search Strategy

1. **Start with data.gov.in**:
   - Go to https://data.gov.in/
   - Use these search terms:
     - "industrial emissions"
     - "CO2 emissions by sector"
     - "electricity grid emissions"
     - "state-wise emissions"
     - "industrial pollution data"

2. **Check Government Ministry Websites**:
   - Ministry of Environment: https://moef.gov.in/
   - Ministry of Power: https://powermin.gov.in/
   - Ministry of Steel: https://steel.gov.in/
   - CEA: https://cea.nic.in/
   - CPCB: https://cpcb.nic.in/

3. **Industry Association Reports**:
   - Steel: SAIL, JSW, Tata Steel annual reports
   - Cement: CMA reports
   - Aluminium: Industry association reports

4. **Academic/Research Sources**:
   - Search: "India industrial emissions dataset" + "research"
   - Check: IIT research papers, TERI reports

---

## After Downloading

1. **Place files in correct location**:
   ```
   data/raw/indian/
   ├── industrial_emissions.csv
   ├── steel_process_data.csv
   ├── cement_process_data.csv
   ├── aluminium_process_data.csv
   └── grid_ci_*.csv (one per state or combined)
   ```

2. **Verify column names match** the format above

3. **Run the training script**:
   ```bash
   python scripts/train_indian_models.py --sectors steel cement aluminium
   ```

4. **If column names differ**, you can either:
   - Rename columns in the CSV files to match expected names
   - Or modify `IndianDataLoader` to map your column names

---

## Minimum Data Requirements

**For PINN Training:**
- At least 50-100 samples per sector
- Columns: `sector`, `state`, `year`, `emissions_tco2`
- Additional columns (`production_tonnes`, `energy_consumption_mwh`) improve accuracy

**For Transformer Training:**
- Time series data with at least 12+ time points per state
- Monthly or quarterly data preferred (annual data works but needs adjustment)
- Same columns as PINN training

---

## Alternative: Use Sample Data Structure

If you can't find real data immediately, the system will generate sample data based on the structure. However, for production use, real Indian data is essential.

The sample data generator creates realistic Indian industrial data based on:
- Indian industry averages
- State-specific variations
- Regional grid characteristics

But this is only for testing - real data is needed for actual model training.

