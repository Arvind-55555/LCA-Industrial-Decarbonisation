"""
Training script for Physics-Informed Neural Network (PINN)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging
from datetime import datetime

from lca_optimizer.models.pinn import PhysicsInformedNN
from lca_optimizer.core.physics import PhysicsConstraints

logger = logging.getLogger(__name__)


def train_pinn_model(
    train_data: Dict[str, torch.Tensor],
    val_data: Optional[Dict[str, torch.Tensor]] = None,
    config: Optional[Dict] = None
) -> Tuple[PhysicsInformedNN, Dict[str, List[float]]]:
    """
    Train a PINN model for LCA prediction.
    
    Args:
        train_data: Training data with 'X' (inputs) and 'y' (targets)
        val_data: Validation data (optional)
        config: Training configuration
    
    Returns:
        (trained_model, training_history)
    """
    if config is None:
        config = {
            "input_dim": train_data["X"].shape[1],
            "hidden_dims": [64, 128, 64],
            "output_dim": 1,
            "learning_rate": 1e-3,
            "epochs": 100,
            "batch_size": 32,
            "physics_weight": 0.1
        }
    
    # Initialize model
    model = PhysicsInformedNN(
        input_dim=config["input_dim"],
        hidden_dims=config["hidden_dims"],
        output_dim=config["output_dim"],
        physics_constraints=PhysicsConstraints()
    )
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.MSELoss()
    
    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "physics_loss": []
    }
    
    # Training loop
    X_train = train_data["X"]
    y_train = train_data["y"]
    n_samples = len(X_train)
    batch_size = config["batch_size"]
    
    model.train()
    for epoch in range(config["epochs"]):
        epoch_loss = 0.0
        epoch_physics_loss = 0.0
        
        # Mini-batch training
        indices = torch.randperm(n_samples)
        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]
            
            # Forward pass
            optimizer.zero_grad()
            y_pred = model(X_batch, apply_constraints=True)
            
            # Data loss
            data_loss = criterion(y_pred, y_batch)
            
            # Physics loss
            physics_loss = model.physics_loss(X_batch, y_pred, y_batch)
            
            # Total loss
            total_loss = data_loss + config["physics_weight"] * physics_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += data_loss.item()
            epoch_physics_loss += physics_loss.item()
        
        avg_loss = epoch_loss / (n_samples / batch_size)
        avg_physics_loss = epoch_physics_loss / (n_samples / batch_size)
        history["train_loss"].append(avg_loss)
        history["physics_loss"].append(avg_physics_loss)
        
        # Validation
        if val_data is not None:
            model.eval()
            with torch.no_grad():
                y_val_pred = model(val_data["X"], apply_constraints=True)
                val_loss = criterion(y_val_pred, val_data["y"]).item()
                history["val_loss"].append(val_loss)
            model.train()
        
        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch {epoch + 1}/{config['epochs']}: "
                f"Train Loss: {avg_loss:.4f}, "
                f"Physics Loss: {avg_physics_loss:.4f}"
            )
    
    model.eval()
    logger.info("PINN training completed")
    
    return model, history


def save_model(model: PhysicsInformedNN, path: str):
    """Save trained model"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": {
            "input_dim": model.input_dim,
            "output_dim": model.output_dim
        }
    }, path)
    logger.info(f"Model saved to {path}")


def load_model(path: str) -> PhysicsInformedNN:
    """Load trained model"""
    checkpoint = torch.load(path)
    config = checkpoint["model_config"]
    
    model = PhysicsInformedNN(
        input_dim=config["input_dim"],
        hidden_dims=[64, 128, 64],  # Default, should be saved in checkpoint
        output_dim=config["output_dim"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    logger.info(f"Model loaded from {path}")
    return model

