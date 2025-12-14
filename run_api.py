#!/usr/bin/env python3
"""
Run LCA Optimizer API server
"""

import uvicorn
from lca_optimizer.config.settings import get_settings
from lca_optimizer.utils.logging import setup_logging

if __name__ == "__main__":
    settings = get_settings()
    setup_logging(settings.log_level)
    
    uvicorn.run(
        "lca_optimizer.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload
    )

