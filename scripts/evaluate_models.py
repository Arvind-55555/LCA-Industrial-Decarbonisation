#!/usr/bin/env python3
"""
Evaluate trained ML models and generate metrics
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import json
from typing import Dict
import logging

from lca_optimizer.training.train_pinn import load_model
try:
    from lca_optimizer.models.transformer import LCATransformer
except ImportError:
    LCATransformer = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_pinn_model(model_path: str, test_data_path: str) -> Dict:
    """Evaluate PINN model"""
    logger.info(f"Evaluating PINN model: {model_path}")
    
    try:
        # Load model
        model = load_model(model_path)
        model.eval()
        
        # Load test data
        test_data = torch.load(test_data_path)
        X_test = test_data["X"]
        y_test = test_data["y"]
        
        # Predictions
        with torch.no_grad():
            y_pred = model(X_test, apply_constraints=True)
        
        # Metrics
        mse = torch.nn.functional.mse_loss(y_pred, y_test).item()
        mae = torch.nn.functional.l1_loss(y_pred, y_test).item()
        
        # Relative error
        relative_error = (torch.abs(y_pred - y_test) / (y_test + 1e-6)).mean().item() * 100
        
        # Physics loss
        physics_loss = model.physics_loss(X_test, y_pred, y_test).item()
        
        metrics = {
            "mse": float(mse),
            "mae": float(mae),
            "relative_error_percent": float(relative_error),
            "physics_loss": float(physics_loss),
            "n_samples": len(X_test)
        }
        
        logger.info(f"PINN Metrics: MSE={mse:.4f}, MAE={mae:.4f}, Rel Error={relative_error:.2f}%")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error evaluating PINN: {e}")
        return {"error": str(e)}


def evaluate_transformer_model(model_path: str, test_data_path: str) -> Dict:
    """Evaluate Transformer model"""
    logger.info(f"Evaluating Transformer model: {model_path}")
    
    try:
        # Load model
        checkpoint = torch.load(model_path, map_location='cpu')
        config = checkpoint["model_config"]
        
        model = LCATransformer(
            input_dim=config["input_dim"],
            d_model=config.get("d_model", 64)
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        
        # Load test data
        test_data = torch.load(test_data_path)
        X_test = test_data["X"]
        y_test = test_data["y"]
        
        # Predictions
        with torch.no_grad():
            y_pred = model(X_test)
        
        # Metrics
        mse = torch.nn.functional.mse_loss(y_pred, y_test).item()
        mae = torch.nn.functional.l1_loss(y_pred, y_test).item()
        relative_error = (torch.abs(y_pred - y_test) / (y_test + 1e-6)).mean().item() * 100
        
        metrics = {
            "mse": float(mse),
            "mae": float(mae),
            "relative_error_percent": float(relative_error),
            "n_samples": len(X_test),
            "sequence_length": X_test.shape[1]
        }
        
        logger.info(f"Transformer Metrics: MSE={mse:.4f}, MAE={mae:.4f}, Rel Error={relative_error:.2f}%")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error evaluating Transformer: {e}")
        return {"error": str(e)}


def main():
    """Evaluate all trained models"""
    print("=" * 70)
    print("ML Model Evaluation")
    print("=" * 70)
    
    metrics_dir = Path("models/trained")
    metrics_file = metrics_dir / "metrics.json"
    
    all_metrics = {}
    
    # Evaluate PINN
    pinn_model = metrics_dir / "pinn_model.pt"
    pinn_test = Path("data/sample/pinn_val.pt")  # Use validation as test
    
    if pinn_model.exists() and pinn_test.exists():
        print("\n1. Evaluating PINN...")
        all_metrics["pinn"] = evaluate_pinn_model(str(pinn_model), str(pinn_test))
    else:
        print("\n1. PINN: Model or test data not found")
        all_metrics["pinn"] = {"status": "not_found"}
    
    # Evaluate Transformer
    transformer_model = metrics_dir / "transformer_model.pt"
    transformer_test = Path("data/sample/transformer_val.pt")
    
    if transformer_model.exists() and transformer_test.exists():
        print("\n2. Evaluating Transformer...")
        all_metrics["transformer"] = evaluate_transformer_model(
            str(transformer_model), str(transformer_test)
        )
    else:
        print("\n2. Transformer: Model or test data not found")
        all_metrics["transformer"] = {"status": "not_found"}
    
    # GNN - not trained yet
    print("\n3. GNN: Not trained yet")
    all_metrics["gnn"] = {"status": "not_trained"}
    
    # Save metrics
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    print("\n" + "=" * 70)
    print("✅ Evaluation Complete!")
    print(f"📊 Metrics saved to: {metrics_file}")
    print("=" * 70)
    
    # Print summary
    print("\nSummary:")
    for model_name, metrics in all_metrics.items():
        if "error" not in metrics and "status" not in metrics:
            print(f"\n{model_name.upper()}:")
            print(f"  MSE: {metrics.get('mse', 'N/A'):.4f}")
            print(f"  MAE: {metrics.get('mae', 'N/A'):.4f}")
            print(f"  Relative Error: {metrics.get('relative_error_percent', 'N/A'):.2f}%")
            if 'physics_loss' in metrics:
                print(f"  Physics Loss: {metrics['physics_loss']:.4f}")


if __name__ == "__main__":
    main()

