"""
Training script for Graph Neural Network (GNN)
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

from lca_optimizer.models.gnn import ProcessGNN

logger = logging.getLogger(__name__)


def train_gnn_model(
    train_graphs: List[Data],
    val_graphs: Optional[List[Data]] = None,
    config: Optional[Dict] = None
) -> Tuple[ProcessGNN, Dict[str, List[float]]]:
    """
    Train a GNN model for process flow LCA prediction.
    
    Args:
        train_graphs: List of training graph data
        val_graphs: List of validation graph data (optional)
        config: Training configuration
    
    Returns:
        (trained_model, training_history)
    """
    if config is None:
        # Infer dimensions from first graph
        sample_graph = train_graphs[0]
        config = {
            "node_features": sample_graph.x.shape[1],
            "edge_features": sample_graph.edge_attr.shape[1] if sample_graph.edge_attr is not None else 0,
            "hidden_dim": 64,
            "num_layers": 3,
            "gnn_type": "GCN",
            "learning_rate": 1e-3,
            "epochs": 100,
            "batch_size": 32
        }
    
    # Initialize model
    model = ProcessGNN(
        node_features=config["node_features"],
        edge_features=config["edge_features"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        gnn_type=config["gnn_type"]
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
    model.train()
    for epoch in range(config["epochs"]):
        epoch_loss = 0.0
        
        # Batch graphs
        for i in range(0, len(train_graphs), config["batch_size"]):
            batch_graphs = train_graphs[i:i + config["batch_size"]]
            batch = Batch.from_data_list(batch_graphs)
            
            # Forward pass
            optimizer.zero_grad()
            emissions_pred = model(batch)
            
            # Get targets (assuming y is stored in graph data)
            emissions_true = torch.cat([g.y for g in batch_graphs], dim=0)
            
            # Loss
            loss = criterion(emissions_pred, emissions_true)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / (len(train_graphs) / config["batch_size"])
        history["train_loss"].append(avg_loss)
        
        # Validation
        if val_graphs is not None:
            model.eval()
            with torch.no_grad():
                val_batch = Batch.from_data_list(val_graphs)
                val_pred = model(val_batch)
                val_true = torch.cat([g.y for g in val_graphs], dim=0)
                val_loss = criterion(val_pred, val_true).item()
                history["val_loss"].append(val_loss)
            model.train()
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch + 1}/{config['epochs']}: Train Loss: {avg_loss:.4f}")
    
    model.eval()
    logger.info("GNN training completed")
    
    return model, history

