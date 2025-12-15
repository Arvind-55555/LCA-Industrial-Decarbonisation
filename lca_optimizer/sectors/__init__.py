"""Sector-specific LCA optimizers"""

from lca_optimizer.sectors.steel import SteelH2DRIOptimizer
from lca_optimizer.sectors.shipping import ShippingFuelComparator
from lca_optimizer.sectors.cement import CementCCUSOptimizer
from lca_optimizer.sectors.aluminium import AluminiumElectrificationOptimizer

# Indian sector optimizers
try:
    from lca_optimizer.sectors.indian_steel import IndianSteelOptimizer, IndianSteelProcessConfig
    from lca_optimizer.sectors.indian_cement import IndianCementOptimizer, IndianCementProcessConfig
    from lca_optimizer.sectors.indian_aluminium import IndianAluminiumOptimizer, IndianAluminiumProcessConfig
    INDIAN_SECTORS_AVAILABLE = True
except ImportError:
    INDIAN_SECTORS_AVAILABLE = False
    IndianSteelOptimizer = None
    IndianCementOptimizer = None
    IndianAluminiumOptimizer = None

__all__ = [
    "SteelH2DRIOptimizer",
    "ShippingFuelComparator",
    "CementCCUSOptimizer",
    "AluminiumElectrificationOptimizer",
]

if INDIAN_SECTORS_AVAILABLE:
    __all__.extend([
        "IndianSteelOptimizer",
        "IndianSteelProcessConfig",
        "IndianCementOptimizer",
        "IndianCementProcessConfig",
        "IndianAluminiumOptimizer",
        "IndianAluminiumProcessConfig",
    ])

