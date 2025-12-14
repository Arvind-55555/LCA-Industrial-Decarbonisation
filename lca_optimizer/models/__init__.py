"""ML Models: Physics-Informed Neural Networks, GNNs, Transformers"""

from lca_optimizer.models.pinn import PhysicsInformedNN
from lca_optimizer.models.transformer import LCATransformer

# GNN requires torch_geometric, make it optional
try:
    from lca_optimizer.models.gnn import ProcessGNN
    __all__ = ["PhysicsInformedNN", "ProcessGNN", "LCATransformer"]
except ImportError:
    __all__ = ["PhysicsInformedNN", "LCATransformer"]

