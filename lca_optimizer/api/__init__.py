"""API layer for LCA-as-a-Service"""

from lca_optimizer.api.main import app
from lca_optimizer.api.endpoints import router

__all__ = ["app", "router"]

