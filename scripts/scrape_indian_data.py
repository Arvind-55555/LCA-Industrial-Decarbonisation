#!/usr/bin/env python3
"""
Script to scrape Indian industrial data from open sources
Automatically downloads and organizes data in data/raw/indian/
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
from lca_optimizer.data.indian_data_scraper import IndianDataScraper

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Scrape Indian industrial data from open sources"
    )
    parser.add_argument(
        "--data-dir",
        default="data/raw/indian",
        help="Directory to save scraped data (default: data/raw/indian)"
    )
    parser.add_argument(
        "--source",
        choices=["all", "datagov", "cea", "cpcb", "research"],
        default="all",
        help="Data source to scrape (default: all)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-scraping even if data exists"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Indian Industrial Data Scraper")
    logger.info("=" * 60)
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Source: {args.source}")
    logger.info("")
    
    scraper = IndianDataScraper(data_dir=args.data_dir)
    
    if args.source == "all":
        # Run full scrape
        scraped_data = scraper.run_full_scrape()
    else:
        # Scrape specific source
        if args.source == "datagov":
            logger.info("Searching data.gov.in...")
            results = scraper.scrape_datagov_in("industrial emissions")
            logger.info(f"Found {len(results)} datasets")
        elif args.source == "cea":
            logger.info("Scraping CEA data...")
            data = scraper.scrape_cea_reports()
            if data is not None:
                output_file = Path(args.data_dir) / "grid_ci_all_states.csv"
                data.to_csv(output_file, index=False)
                logger.info(f"✅ Saved: {output_file}")
        elif args.source == "cpcb":
            logger.info("Scraping CPCB data...")
            data = scraper.scrape_cpcb_emissions()
            if data is not None:
                output_file = Path(args.data_dir) / "industrial_emissions.csv"
                data.to_csv(output_file, index=False)
                logger.info(f"✅ Saved: {output_file}")
        elif args.source == "research":
            logger.info("Searching public research datasets...")
            scraper.scrape_public_research_datasets()
    
    logger.info("\n✅ Scraping process completed!")
    logger.info("\nNext steps:")
    logger.info("1. Review scraped data in: data/raw/indian/")
    logger.info("2. Run: python scripts/train_indian_models.py --sectors steel cement aluminium")


if __name__ == "__main__":
    main()

