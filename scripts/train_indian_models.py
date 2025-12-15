#!/usr/bin/env python3
"""
Main training script for Indian industrial ML models
"""

import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging

from lca_optimizer.training.train_indian_models import train_all_indian_models
from lca_optimizer.utils.logging import setup_logging

# Setup logging
setup_logging(level="INFO")
logger = logging.getLogger(__name__)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Train ML models for Indian industrial decarbonisation"
    )
    parser.add_argument(
        "--sectors",
        nargs="+",
        default=["steel", "cement", "aluminium"],
        help="Industrial sectors to train (default: steel, cement, aluminium)"
    )
    parser.add_argument(
        "--states",
        nargs="+",
        default=None,
        help="Indian states to include (default: all)"
    )
    parser.add_argument(
        "--output-dir",
        default="models/trained/indian",
        help="Output directory for trained models"
    )
    parser.add_argument(
        "--transfer-from",
        default=None,
        help="Path to pre-trained model for transfer learning"
    )
    
    args = parser.parse_args()
    
    logger.info("Starting Indian ML model training")
    logger.info(f"Sectors: {args.sectors}")
    logger.info(f"States: {args.states}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Train all models
    train_all_indian_models(
        sectors=args.sectors,
        states=args.states,
        output_dir=args.output_dir
    )
    
    logger.info("Training completed successfully")


if __name__ == "__main__":
    main()

