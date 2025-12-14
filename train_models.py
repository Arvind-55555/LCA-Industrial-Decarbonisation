#!/usr/bin/env python3
"""
Main training script for LCA Optimizer ML models
"""

import argparse
import logging
from pathlib import Path
import torch

from lca_optimizer.training.train_pinn import train_pinn_model, save_model
from lca_optimizer.training.train_transformer import train_transformer_model
from lca_optimizer.utils.logging import setup_logging

# Setup logging
setup_logging(level="INFO")
logger = logging.getLogger(__name__)


def train_all_models(data_dir: str = "data/sample", output_dir: str = "models/trained"):
    """
    Train all ML models.
    
    Args:
        data_dir: Directory containing training data
        output_dir: Directory to save trained models
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Train PINN
    logger.info("Training PINN model...")
    try:
        train_data = torch.load(data_path / "pinn_train.pt")
        val_data = torch.load(data_path / "pinn_val.pt")
        
        model, history = train_pinn_model(train_data, val_data)
        save_model(model, str(output_path / "pinn_model.pt"))
        logger.info("PINN training completed")
    except FileNotFoundError:
        logger.warning(f"PINN training data not found in {data_path}")
    
    # Train Transformer
    logger.info("Training Transformer model...")
    try:
        train_data = torch.load(data_path / "transformer_train.pt")
        val_data = torch.load(data_path / "transformer_val.pt")
        
        model, history = train_transformer_model(train_data, val_data)
        torch.save({
            "model_state_dict": model.state_dict(),
            "model_config": {
                "input_dim": model.input_dim,
                "d_model": model.d_model
            }
        }, str(output_path / "transformer_model.pt"))
        logger.info("Transformer training completed")
    except FileNotFoundError:
        logger.warning(f"Transformer training data not found in {data_path}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Train LCA Optimizer ML models")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/sample",
        help="Directory containing training data"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/trained",
        help="Directory to save trained models"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["pinn", "transformer", "all"],
        default="all",
        help="Model to train"
    )
    
    args = parser.parse_args()
    
    if args.model == "all":
        train_all_models(args.data_dir, args.output_dir)
    else:
        logger.info(f"Training {args.model} model...")
        # Individual model training can be added here


if __name__ == "__main__":
    main()

