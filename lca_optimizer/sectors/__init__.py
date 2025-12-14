"""Sector-specific LCA optimizers"""

from lca_optimizer.sectors.steel import SteelH2DRIOptimizer
from lca_optimizer.sectors.shipping import ShippingFuelComparator
from lca_optimizer.sectors.cement import CementCCUSOptimizer
from lca_optimizer.sectors.aluminium import AluminiumElectrificationOptimizer

__all__ = [
    "SteelH2DRIOptimizer",
    "ShippingFuelComparator",
    "CementCCUSOptimizer",
    "AluminiumElectrificationOptimizer"
]

