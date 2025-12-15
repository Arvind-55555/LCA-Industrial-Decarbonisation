#!/usr/bin/env python3
"""
Download Open Datasets for Grid Carbon Intensity
Downloads publicly available datasets instead of relying on API keys
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lca_optimizer.data.download_utils import download_all_datasets, DatasetDownloader
import argparse


def main():
    """Main download script"""
    parser = argparse.ArgumentParser(description="Download open datasets for grid carbon intensity")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Directory to store downloaded data"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["all", "egrid", "opsd", "sample", "indian"],
        default="all",
        help="Dataset to download"
    )
    parser.add_argument(
        "--datagov-dataset-id",
        type=str,
        default=None,
        help="data.gov.in dataset_id for Indian emissions (required for --dataset indian)"
    )
    parser.add_argument(
        "--datagov-resource-id",
        type=str,
        default=None,
        help="Optional data.gov.in resource_id for Indian emissions"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files exist"
    )
    
    args = parser.parse_args()
    
    downloader = DatasetDownloader(data_dir=args.data_dir)
    
    if args.dataset == "all":
        download_all_datasets(args.data_dir)
    elif args.dataset == "egrid":
        print("Downloading EPA eGRID data...")
        path = downloader.download_epa_egrid(year=2022, force=args.force)
        if path:
            print(f"✅ Downloaded: {path}")
        else:
            print("❌ Download failed")
            print("   Manual download: https://www.epa.gov/egrid/download-data")
    elif args.dataset == "opsd":
        print("Downloading OPSD data...")
        path = downloader.download_opsd_data("time_series", force=args.force)
        if path:
            print(f"✅ Downloaded: {path}")
        else:
            print("❌ Download failed")
            print("   Manual download: https://open-power-system-data.org/")
    elif args.dataset == "sample":
        print("Creating sample grid data...")
        from datetime import datetime, timedelta
        path = downloader.create_sample_grid_data(
            locations=["US", "EU", "DE", "FR", "GB", "CA", "CN", "IN"],
            start_date=datetime.now() - timedelta(days=365),
            end_date=datetime.now()
        )
        print(f"✅ Created: {path}")
    elif args.dataset == "indian":
        print("Downloading Indian industrial emissions data from data.gov.in...")
        if not args.datagov_dataset_id:
            print("❌ --datagov-dataset-id is required for --dataset indian")
            print("   Example: --datagov-dataset-id your_dataset_id_here")
            return
        path = downloader.download_indian_emissions_from_datagov(
            dataset_id=args.datagov_dataset_id,
            resource_id=args.datagov_resource_id,
            force=args.force
        )
        if path:
            print(f"✅ Downloaded raw Indian emissions data to: {path}")
            print("   Please inspect and, if needed, transform it into 'data/raw/indian/industrial_emissions.csv'")
        else:
            print("❌ Download failed or returned no data. Check DATAGOV_API_KEY and dataset IDs.")


if __name__ == "__main__":
    main()

