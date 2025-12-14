"""
FastAPI application for LCA-as-a-Service
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from lca_optimizer.api.endpoints import router

logger = logging.getLogger(__name__)

app = FastAPI(
    title="LCA Optimizer API",
    description="Deep Learning for LCA Optimization in Hard-to-Abate Industrial Decarbonisation",
    version="0.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "LCA Optimizer API",
        "version": "0.1.0",
        "endpoints": [
            "/lca/steel_h2_dri",
            "/lca/shipping_fuel_comparison",
            "/lca/cement_ccus_optimization",
            "/lca/aluminium_electrification"
        ]
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}

