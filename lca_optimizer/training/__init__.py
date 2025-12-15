"""Training utilities for ML models"""

from lca_optimizer.training.train_pinn import train_pinn_model
from lca_optimizer.training.train_transformer import train_transformer_model

# GNN training requires torch_geometric, make it optional
try:
    from lca_optimizer.training.train_gnn import train_gnn_model
    GNN_TRAINING_AVAILABLE = True
except ImportError:
    GNN_TRAINING_AVAILABLE = False
    train_gnn_model = None

__all__ = [
    "train_pinn_model",
    "train_transformer_model",
]

if GNN_TRAINING_AVAILABLE:
    __all__.append("train_gnn_model")

