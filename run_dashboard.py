#!/usr/bin/env python3
"""
Run LCA Optimizer Dashboards
"""

import sys

if len(sys.argv) > 1 and sys.argv[1] == "results":
    # Run results dashboard
    try:
        from lca_optimizer.visualization.results_dashboard import run_results_dashboard
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 8050
        print(f"Starting Results Dashboard on http://localhost:{port}")
        run_results_dashboard(port=port)
    except ImportError:
        print("Dash not available. Install with: pip install dash plotly")
        sys.exit(1)
else:
    # Run standard dashboard
    try:
        from lca_optimizer.visualization.dashboard import run_dashboard
        port = int(sys.argv[1]) if len(sys.argv) > 1 else 8050
        print(f"Starting Dashboard on http://localhost:{port}")
        run_dashboard(port=port, debug=True)
    except ImportError:
        print("Dash not available. Install with: pip install dash plotly")
        sys.exit(1)

