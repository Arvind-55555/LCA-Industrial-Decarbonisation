"""
Training script for Transformer model (time-series LCA)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

from lca_optimizer.models.transformer import LCATransformer

logger = logging.getLogger(__name__)


def train_transformer_model(
    train_data: Dict[str, torch.Tensor],
    val_data: Optional[Dict[str, torch.Tensor]] = None,
    config: Optional[Dict] = None
) -> Tuple[LCATransformer, Dict[str, List[float]]]:
    """
    Train a Transformer model for time-series LCA prediction.
    
    Args:
        train_data: Training data with 'X' (sequences) and 'y' (targets)
        val_data: Validation data (optional)
        config: Training configuration
    
    Returns:
        (trained_model, training_history)
    """
    if config is None:
        config = {
            "input_dim": train_data["X"].shape[2],
            "d_model": 64,
            "nhead": 4,
            "num_layers": 3,
            "dim_feedforward": 256,
            "dropout": 0.1,
            "learning_rate": 1e-4,
            "epochs": 100,
            "batch_size": 32
        }
    
    # Initialize model
    model = LCATransformer(
        input_dim=config["input_dim"],
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        dim_feedforward=config["dim_feedforward"],
        dropout=config["dropout"]
    )
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.MSELoss()
    
    # Training history
    history = {
        "train_loss": [],
        "val_loss": []
    }
    
    # Training loop
    X_train = train_data["X"]  # (batch, seq_len, features)
    y_train = train_data["y"]   # (batch, seq_len, 1)
    n_samples = len(X_train)
    batch_size = config["batch_size"]
    
    model.train()
    for epoch in range(config["epochs"]):
        epoch_loss = 0.0
        
        # Mini-batch training
        indices = torch.randperm(n_samples)
        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]
            
            # Forward pass
            optimizer.zero_grad()
            y_pred = model(X_batch)  # (batch, seq_len, 1)
            
            # For forecasting, we predict the next timestep
            # Use last timestep of prediction if target is single value
            if y_batch.dim() == 2 and y_batch.shape[1] == 1:
                # Target is (batch, 1), use last timestep of prediction
                y_pred_last = y_pred[:, -1, :]  # (batch, 1)
                loss = criterion(y_pred_last, y_batch)
            else:
                # Target matches prediction shape
                loss = criterion(y_pred, y_batch)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / (n_samples / batch_size)
        history["train_loss"].append(avg_loss)
        
        # Validation
        if val_data is not None:
            model.eval()
            with torch.no_grad():
                y_val_pred = model(val_data["X"])  # (batch, seq_len, 1)
                y_val = val_data["y"]
                # Use last timestep if target is single value
                if y_val.dim() == 2 and y_val.shape[1] == 1:
                    y_val_pred_last = y_val_pred[:, -1, :]  # (batch, 1)
                    val_loss = criterion(y_val_pred_last, y_val).item()
                else:
                    val_loss = criterion(y_val_pred, y_val).item()
                history["val_loss"].append(val_loss)
            model.train()
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch + 1}/{config['epochs']}: Train Loss: {avg_loss:.4f}")
    
    model.eval()
    logger.info("Transformer training completed")
    
    return model, history

