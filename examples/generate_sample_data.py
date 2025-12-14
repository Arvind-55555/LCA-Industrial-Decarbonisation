#!/usr/bin/env python3
"""
Generate sample data for training ML models
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime, timedelta

def generate_pinn_data(n_samples: int = 1000) -> dict:
    """
    Generate sample data for PINN training.
    
    Features:
    - H2 production rate
    - Electrolyzer efficiency
    - Renewable energy mix
    - Grid carbon intensity
    - Process parameters
    """
    np.random.seed(42)
    
    # Input features
    X = np.random.rand(n_samples, 10)
    X[:, 0] = np.random.uniform(0.5, 1.0, n_samples)  # H2 production rate
    X[:, 1] = np.random.uniform(0.6, 0.8, n_samples)  # Electrolyzer efficiency
    X[:, 2] = np.random.uniform(0.3, 0.9, n_samples)  # Renewable mix
    X[:, 3] = np.random.uniform(50, 800, n_samples)  # Grid CI (g CO2/kWh)
    X[:, 4:] = np.random.rand(n_samples, 6)  # Other process parameters
    
    # Target emissions (simplified model)
    # Emissions = base_emissions * (1 - renewable_mix) * grid_ci_factor
    base_emissions = 1000.0
    grid_ci_factor = X[:, 3] / 300.0  # Normalize to average
    renewable_factor = 1 - X[:, 2]
    
    y = base_emissions * renewable_factor * grid_ci_factor
    y += np.random.normal(0, 50, n_samples)  # Add noise
    y = np.maximum(y, 0)  # Ensure non-negative
    
    return {
        "X": torch.FloatTensor(X),
        "y": torch.FloatTensor(y).unsqueeze(1)
    }


def generate_transformer_data(n_samples: int = 100, seq_length: int = 24) -> dict:
    """
    Generate time-series data for Transformer training.
    
    Features:
    - Hourly grid carbon intensity
    - Production rate
    - Energy consumption
    - Process parameters
    """
    np.random.seed(42)
    
    # Generate sequences
    X = []
    y = []
    
    for _ in range(n_samples):
        # Grid CI with daily pattern
        hours = np.arange(seq_length)
        base_ci = 300.0
        daily_pattern = 50 * np.sin(2 * np.pi * hours / 24 - np.pi/2)
        ci = base_ci + daily_pattern + np.random.normal(0, 20, seq_length)
        ci = np.maximum(ci, 0)
        
        # Production rate (varies by hour)
        production = 0.8 + 0.2 * np.sin(2 * np.pi * hours / 24)
        
        # Energy consumption
        energy = production * 100 + np.random.normal(0, 10, seq_length)
        
        # Features: [grid_ci, production, energy, ...]
        features = np.column_stack([
            ci,
            production,
            energy,
            np.random.rand(seq_length, 2)  # Additional features
        ])
        
        X.append(features)
        
        # Target: emissions (simplified)
        emissions = (ci / 1000) * energy * 0.5  # kg CO2
        y.append(emissions.reshape(-1, 1))
    
    return {
        "X": torch.FloatTensor(np.array(X)),
        "y": torch.FloatTensor(np.array(y))
    }


def save_sample_data():
    """Generate and save sample data"""
    output_dir = Path("data/sample")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate PINN data
    print("Generating PINN training data...")
    pinn_data = generate_pinn_data(n_samples=1000)
    torch.save(pinn_data, output_dir / "pinn_train.pt")
    
    # Split for validation
    val_size = 200
    val_data = {
        "X": pinn_data["X"][:val_size],
        "y": pinn_data["y"][:val_size]
    }
    torch.save(val_data, output_dir / "pinn_val.pt")
    
    # Generate Transformer data
    print("Generating Transformer training data...")
    transformer_data = generate_transformer_data(n_samples=100, seq_length=24)
    torch.save(transformer_data, output_dir / "transformer_train.pt")
    
    val_transformer = {
        "X": transformer_data["X"][:20],
        "y": transformer_data["y"][:20]
    }
    torch.save(val_transformer, output_dir / "transformer_val.pt")
    
    print(f"\nSample data saved to {output_dir}/")
    print("Files created:")
    print("  - pinn_train.pt")
    print("  - pinn_val.pt")
    print("  - transformer_train.pt")
    print("  - transformer_val.pt")


if __name__ == "__main__":
    save_sample_data()

