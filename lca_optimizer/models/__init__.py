"""ML Models: Physics-Informed Neural Networks, GNNs, Transformers"""

from lca_optimizer.models.pinn import PhysicsInformedNN
from lca_optimizer.models.transformer import LCATransformer

# GNN requires torch_geometric, make it optional
try:
    from lca_optimizer.models.gnn import ProcessGNN
    GNN_AVAILABLE = True
except ImportError:
    GNN_AVAILABLE = False
    ProcessGNN = None

# Indian-specific models
try:
    from lca_optimizer.models.indian_pinn import IndianPhysicsInformedNN
    INDIAN_MODELS_AVAILABLE = True
except ImportError:
    INDIAN_MODELS_AVAILABLE = False
    IndianPhysicsInformedNN = None

__all__ = ["PhysicsInformedNN", "LCATransformer"]

if GNN_AVAILABLE:
    __all__.append("ProcessGNN")

if INDIAN_MODELS_AVAILABLE:
    __all__.append("IndianPhysicsInformedNN")

