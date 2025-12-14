"""Visualization utilities for LCA results"""

from lca_optimizer.visualization.plots import plot_lca_results, plot_time_series_lca, plot_sector_comparison

try:
    from lca_optimizer.visualization.dashboard import create_dashboard
    __all__ = ["create_dashboard", "plot_lca_results", "plot_time_series_lca", "plot_sector_comparison"]
except ImportError:
    __all__ = ["plot_lca_results", "plot_time_series_lca", "plot_sector_comparison"]

