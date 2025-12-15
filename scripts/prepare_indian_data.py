#!/usr/bin/env python3
"""
Script to help prepare manually downloaded Indian data
Converts various formats to the expected structure
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import argparse
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_industrial_emissions(
    input_file: str,
    output_file: str = "data/raw/indian/industrial_emissions.csv",
    sector_column: Optional[str] = None,
    state_column: Optional[str] = None,
    year_column: Optional[str] = None,
    emissions_column: Optional[str] = None
):
    """
    Prepare industrial emissions data from manually downloaded file.
    
    Args:
        input_file: Path to downloaded CSV/Excel file
        output_file: Output file path
        sector_column: Name of sector column (if different from 'sector')
        state_column: Name of state column (if different from 'state')
        year_column: Name of year column (if different from 'year')
        emissions_column: Name of emissions column (if different from 'emissions_tco2')
    """
    logger.info(f"Reading input file: {input_file}")
    
    # Read file
    if input_file.endswith('.xlsx') or input_file.endswith('.xls'):
        df = pd.read_excel(input_file)
    else:
        df = pd.read_csv(input_file)
    
    logger.info(f"Loaded {len(df)} rows, columns: {list(df.columns)}")
    
    # Map columns if needed
    column_mapping = {}
    if sector_column and sector_column != 'sector':
        column_mapping[sector_column] = 'sector'
    if state_column and state_column != 'state':
        column_mapping[state_column] = 'state'
    if year_column and year_column != 'year':
        column_mapping[year_column] = 'year'
    if emissions_column and emissions_column != 'emissions_tco2':
        column_mapping[emissions_column] = 'emissions_tco2'
    
    if column_mapping:
        df = df.rename(columns=column_mapping)
        logger.info(f"Mapped columns: {column_mapping}")
    
    # Normalize sector names
    if 'sector' in df.columns:
        df['sector'] = df['sector'].str.lower().str.strip()
        # Map common variations
        sector_mapping = {
            'iron and steel': 'steel',
            'iron & steel': 'steel',
            'steel industry': 'steel',
            'cement industry': 'cement',
            'aluminium industry': 'aluminium',
            'aluminum': 'aluminium',
            'chemical industry': 'chemicals',
            'petrochemicals': 'chemicals',
            'refinery': 'refining',
            'oil refining': 'refining'
        }
        df['sector'] = df['sector'].replace(sector_mapping)
    
    # Normalize state names
    if 'state' in df.columns:
        df['state'] = df['state'].str.strip()
    
    # Ensure required columns exist
    required_columns = ['sector', 'state', 'year', 'emissions_tco2']
    missing = [col for col in required_columns if col not in df.columns]
    
    if missing:
        logger.error(f"Missing required columns: {missing}")
        logger.info(f"Available columns: {list(df.columns)}")
        logger.info("\nPlease specify column mappings:")
        logger.info("  --sector-column COLUMN_NAME")
        logger.info("  --state-column COLUMN_NAME")
        logger.info("  --year-column COLUMN_NAME")
        logger.info("  --emissions-column COLUMN_NAME")
        return False
    
    # Add optional columns with defaults if missing
    if 'production_tonnes' not in df.columns:
        logger.warning("'production_tonnes' column missing, will be estimated")
        # Estimate based on emissions (rough)
        df['production_tonnes'] = df['emissions_tco2'] / 2.0  # Rough estimate
    
    if 'energy_consumption_mwh' not in df.columns:
        logger.warning("'energy_consumption_mwh' column missing, will be estimated")
        # Estimate based on production (rough)
        df['energy_consumption_mwh'] = df['production_tonnes'] * 3.5  # Rough estimate
    
    if 'grid_carbon_intensity' not in df.columns:
        logger.warning("'grid_carbon_intensity' column missing, will use state defaults")
        # Use state defaults (will be filled by grid data loader)
        df['grid_carbon_intensity'] = 0.85  # Default
    
    # Ensure year is integer
    if 'year' in df.columns:
        df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
        df = df.dropna(subset=['year'])
    
    # Ensure emissions is numeric
    if 'emissions_tco2' in df.columns:
        df['emissions_tco2'] = pd.to_numeric(df['emissions_tco2'], errors='coerce')
        df = df.dropna(subset=['emissions_tco2'])
    
    # Save
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    logger.info(f"✅ Prepared data saved to: {output_path}")
    logger.info(f"   Rows: {len(df)}")
    logger.info(f"   Sectors: {df['sector'].unique().tolist()}")
    logger.info(f"   States: {df['state'].nunique()}")
    logger.info(f"   Years: {df['year'].min()} - {df['year'].max()}")
    
    return True


def prepare_grid_ci(
    input_file: str,
    output_file: str = "data/raw/indian/grid_ci_all_states.csv",
    state_column: Optional[str] = None,
    timestamp_column: Optional[str] = None,
    ci_column: Optional[str] = None
):
    """
    Prepare grid carbon intensity data from manually downloaded file.
    
    Args:
        input_file: Path to downloaded CSV/Excel file
        output_file: Output file path
        state_column: Name of state column
        timestamp_column: Name of timestamp/date column
        ci_column: Name of carbon intensity column
    """
    logger.info(f"Reading input file: {input_file}")
    
    # Read file
    if input_file.endswith('.xlsx') or input_file.endswith('.xls'):
        df = pd.read_excel(input_file)
    else:
        df = pd.read_csv(input_file)
    
    logger.info(f"Loaded {len(df)} rows, columns: {list(df.columns)}")
    
    # Map columns
    column_mapping = {}
    if state_column and state_column != 'state':
        column_mapping[state_column] = 'state'
    if timestamp_column and timestamp_column not in ['timestamp', 'date']:
        column_mapping[timestamp_column] = 'timestamp'
    if ci_column and ci_column not in ['carbon_intensity', 'grid_ci', 'emission_factor']:
        column_mapping[ci_column] = 'carbon_intensity'
    
    if column_mapping:
        df = df.rename(columns=column_mapping)
        logger.info(f"Mapped columns: {column_mapping}")
    
    # Normalize state names
    if 'state' in df.columns:
        df['state'] = df['state'].str.strip()
    
    # Parse timestamp
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
    elif 'date' in df.columns:
        df['timestamp'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        df = df.drop(columns=['date'])
    
    # Normalize carbon intensity (convert to kg CO2/kWh if needed)
    if 'carbon_intensity' in df.columns:
        ci = pd.to_numeric(df['carbon_intensity'], errors='coerce')
        # If values are very large (>100), likely in g CO2/kWh, convert to kg
        if ci.max() > 100:
            ci = ci / 1000.0
        df['carbon_intensity'] = ci
        df = df.dropna(subset=['carbon_intensity'])
    
    # Save
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    logger.info(f"✅ Prepared grid CI data saved to: {output_path}")
    logger.info(f"   Rows: {len(df)}")
    logger.info(f"   States: {df['state'].nunique()}")
    logger.info(f"   Date range: {df['timestamp'].min()} - {df['timestamp'].max()}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Prepare manually downloaded Indian industrial data"
    )
    parser.add_argument(
        "type",
        choices=["emissions", "grid"],
        help="Type of data to prepare"
    )
    parser.add_argument(
        "input_file",
        help="Path to downloaded CSV/Excel file"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path (default: data/raw/indian/...)"
    )
    
    # Column mapping options
    parser.add_argument("--sector-column", help="Name of sector column")
    parser.add_argument("--state-column", help="Name of state column")
    parser.add_argument("--year-column", help="Name of year column")
    parser.add_argument("--emissions-column", help="Name of emissions column")
    parser.add_argument("--timestamp-column", help="Name of timestamp column")
    parser.add_argument("--ci-column", help="Name of carbon intensity column")
    
    args = parser.parse_args()
    
    if args.type == "emissions":
        output = args.output or "data/raw/indian/industrial_emissions.csv"
        success = prepare_industrial_emissions(
            args.input_file,
            output,
            args.sector_column,
            args.state_column,
            args.year_column,
            args.emissions_column
        )
    else:  # grid
        output = args.output or "data/raw/indian/grid_ci_all_states.csv"
        success = prepare_grid_ci(
            args.input_file,
            output,
            args.state_column,
            args.timestamp_column,
            args.ci_column
        )
    
    if success:
        logger.info("\n✅ Data preparation complete!")
        logger.info("You can now run: python scripts/train_indian_models.py --sectors steel cement aluminium")
    else:
        logger.error("\n❌ Data preparation failed. Check column mappings.")


if __name__ == "__main__":
    main()

