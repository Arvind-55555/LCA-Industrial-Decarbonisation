"""
Training scripts for Indian industrial data
Implements transfer learning, data augmentation, and physics-based regularization
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
from datetime import datetime

from lca_optimizer.models.pinn import PhysicsInformedNN
from lca_optimizer.models.transformer import LCATransformer
from lca_optimizer.core.indian_physics import IndianPhysicsConstraints

# GNN is optional
try:
    from lca_optimizer.models.gnn import ProcessGNN
    GNN_AVAILABLE = True
except ImportError:
    GNN_AVAILABLE = False
    ProcessGNN = None
from lca_optimizer.data.indian_data_loader import IndianDataLoader
from lca_optimizer.config.indian_settings import get_indian_settings

logger = logging.getLogger(__name__)


def prepare_indian_training_data(
    sector: str,
    states: Optional[List[str]] = None,
    augment: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Prepare training data from Indian industrial data.
    
    Args:
        sector: Industrial sector (steel, cement, aluminium, chemicals, refining)
        states: List of states to include (None for all)
        augment: Enable data augmentation
    
    Returns:
        Dictionary with training data tensors
    """
    data_loader = IndianDataLoader()
    settings = get_indian_settings()
    
    # Get industrial emissions data
    emissions_df = data_loader.get_industrial_emissions_data(sector=sector)
    
    if states:
        emissions_df = emissions_df[emissions_df['state'].isin(states)]
    
    # Prepare features and targets
    features: List[List[float]] = []
    targets: List[List[float]] = []
    
    if not emissions_df.empty:
        # Build a numeric mapping for states to avoid using string codes directly
        state_list = list(data_loader.state_codes.keys())
        num_states = max(len(state_list), 1)
        state_index_map = {name: idx for idx, name in enumerate(state_list)}
        
        for _, row in emissions_df.iterrows():
            state_name = str(row["state"])
            state_idx = float(state_index_map.get(state_name, 0))
            
            # Features: [production, energy_consumption, grid_ci, state_code_norm, year_norm]
            features.append([
                float(row["production_tonnes"]) / 1e6,          # Normalize
                float(row["energy_consumption_mwh"]) / 1e6,     # Normalize
                float(row["grid_carbon_intensity"]),
                state_idx / float(num_states),                  # Normalize state index
                float(row["year"] - 2015) / 10.0                # Normalize year
            ])
            targets.append([float(row["emissions_tco2"]) / 1e6])  # Normalize
    
    if not features:
        raise ValueError(f"No training samples available for sector '{sector}'.")
    
    X = torch.FloatTensor(features)
    y = torch.FloatTensor(targets)
    
    # Data augmentation if enabled
    if augment and settings.indian_data_augmentation:
        X, y = augment_indian_data(X, y, sector)
    
    # Train/val split
    n_samples = len(X)
    n_train = int(0.8 * n_samples)
    indices = torch.randperm(n_samples)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    return {
        "X_train": X[train_indices],
        "y_train": y[train_indices],
        "X_val": X[val_indices],
        "y_val": y[val_indices]
    }


def augment_indian_data(
    X: torch.Tensor,
    y: torch.Tensor,
    sector: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Augment Indian industrial data to compensate for limited data.
    
    Techniques:
    - Gaussian noise injection
    - Synthetic data generation based on physics constraints
    - Regional variations
    """
    augmented_X = [X]
    augmented_y = [y]
    
    # 1. Gaussian noise injection
    noise_X = X + torch.randn_like(X) * 0.05
    augmented_X.append(noise_X)
    augmented_y.append(y)
    
    # 2. Physics-based synthetic data
    physics = IndianPhysicsConstraints()
    
    # Generate synthetic samples based on physics constraints
    n_synthetic = len(X) // 2
    synthetic_X = []
    synthetic_y = []
    
    for _ in range(n_synthetic):
        # Sample from existing data
        idx = torch.randint(0, len(X), (1,)).item()
        base_X = X[idx].clone()
        base_y = y[idx].clone()
        
        # Apply physics-based variations
        # Sample random scalar factors in given ranges using torch.rand().item()
        prod_factor = 0.8 + 0.4 * torch.rand(1, device=X.device).item()
        energy_factor = 0.9 + 0.2 * torch.rand(1, device=X.device).item()
        ci_factor = 0.85 + 0.30 * torch.rand(1, device=X.device).item()

        # Vary production capacity
        base_X[0] *= prod_factor
        # Vary energy consumption (correlated with production)
        base_X[1] = base_X[0] * energy_factor
        # Vary grid CI (regional variation)
        base_X[2] = torch.clamp(base_X[2] * ci_factor, 0.5, 1.5)
        
        # Recalculate emissions based on physics
        if sector == "steel":
            # Steel: emissions = production * emission_factor
            emission_factor = 2.0 + base_X[2] * 0.5  # Grid CI dependent
            synthetic_y_val = base_X[0] * emission_factor
        elif sector == "cement":
            emission_factor = 0.75 + base_X[2] * 0.2
            synthetic_y_val = base_X[0] * emission_factor
        elif sector == "aluminium":
            emission_factor = 12.0 + base_X[2] * 2.0
            synthetic_y_val = base_X[0] * emission_factor
        else:
            synthetic_y_val = base_y
        
        synthetic_X.append(base_X)
        # Ensure synthetic_y has same shape and dtype as original targets (1D or 2D with last dim=1)
        synthetic_y.append(synthetic_y_val.view(1, -1))
    
    if synthetic_X:
        synthetic_X = torch.stack(synthetic_X)
        synthetic_y = torch.cat(synthetic_y, dim=0)
        augmented_X.append(synthetic_X)
        augmented_y.append(synthetic_y)
    
    # Combine all augmented data
    final_X = torch.cat(augmented_X, dim=0)
    final_y = torch.cat(augmented_y, dim=0)
    
    logger.info(f"Data augmentation: {len(X)} -> {len(final_X)} samples")
    
    return final_X, final_y


def train_indian_pinn(
    sector: str,
    states: Optional[List[str]] = None,
    output_dir: str = "models/trained/indian",
    transfer_from: Optional[str] = None
) -> Tuple[PhysicsInformedNN, Dict]:
    """
    Train PINN model on Indian industrial data.
    
    Args:
        sector: Industrial sector
        states: States to include
        transfer_from: Path to pre-trained model for transfer learning
        output_dir: Output directory
    
    Returns:
        (trained_model, training_history)
    """
    # Prepare data
    data = prepare_indian_training_data(sector, states, augment=True)
    
    train_data = {
        "X": data["X_train"],
        "y": data["y_train"]
    }
    val_data = {
        "X": data["X_val"],
        "y": data["y_val"]
    }
    
    # Model configuration
    config = {
        "input_dim": train_data["X"].shape[1],
        "hidden_dims": [64, 128, 64],
        "output_dim": 1,
        "learning_rate": 1e-3,
        "epochs": 200,  # More epochs for limited data
        "batch_size": 32,
        "physics_weight": 0.15  # Higher physics weight for data-limited scenario
    }
    
    # Initialize model
    physics_constraints = IndianPhysicsConstraints()
    model = PhysicsInformedNN(
        input_dim=config["input_dim"],
        hidden_dims=config["hidden_dims"],
        output_dim=config["output_dim"],
        physics_constraints=physics_constraints,
        activation="tanh"
    )
    
    # Transfer learning if specified
    if transfer_from and Path(transfer_from).exists():
        logger.info(f"Loading pre-trained model from {transfer_from}")
        checkpoint = torch.load(transfer_from, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        logger.info("Transfer learning: loaded pre-trained weights")
    
    # Train model
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.MSELoss()
    
    history = {
        "train_loss": [],
        "val_loss": [],
        "physics_loss": []
    }
    
    model.train()
    for epoch in range(config["epochs"]):
        # Training
        optimizer.zero_grad()
        y_pred = model(train_data["X"], apply_constraints=True)
        data_loss = criterion(y_pred, train_data["y"])
        
        # Physics loss with Indian constraints
        physics_loss = physics_constraints.physics_loss(
            train_data["X"], y_pred, train_data["y"]
        )
        
        total_loss = data_loss + config["physics_weight"] * physics_loss
        total_loss.backward()
        optimizer.step()
        
        history["train_loss"].append(data_loss.item())
        history["physics_loss"].append(physics_loss.item())
        
        # Validation
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                y_val_pred = model(val_data["X"], apply_constraints=True)
                val_loss = criterion(y_val_pred, val_data["y"]).item()
                history["val_loss"].append(val_loss)
            model.train()
            
            logger.info(
                f"Epoch {epoch + 1}/{config['epochs']}: "
                f"Train Loss: {data_loss.item():.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Physics Loss: {physics_loss.item():.4f}"
            )
    
    # Save model
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = output_path / f"indian_{sector}_pinn.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "sector": sector,
        "states": states,
        "training_history": history
    }, model_path)
    
    logger.info(f"Model saved to {model_path}")
    
    return model, history


def train_indian_transformer(
    sector: str,
    states: Optional[List[str]] = None,
    output_dir: str = "models/trained/indian"
) -> Tuple[LCATransformer, Dict]:
    """
    Train Transformer model on Indian time-series emissions data.
    
    Args:
        sector: Industrial sector
        states: States to include
        output_dir: Output directory
    
    Returns:
        (trained_model, training_history)
    """
    from lca_optimizer.training.train_transformer import train_transformer_model
    
    # Prepare time-series data
    data_loader = IndianDataLoader()
    emissions_df = data_loader.get_industrial_emissions_data(sector=sector)
    
    if states:
        emissions_df = emissions_df[emissions_df['state'].isin(states)]
    
    # Create sequences (e.g., 12 months of data)
    sequences: List[List[List[float]]] = []
    targets: List[List[float]] = []
    seq_length = 12  # 12 months
    
    if not emissions_df.empty:
        # Check if we have monthly data
        has_monthly = 'month' in emissions_df.columns and emissions_df['month'].notna().any()
        
        for state in emissions_df['state'].unique():
            state_data = emissions_df[emissions_df['state'] == state].copy()
            
            if has_monthly:
                # Use monthly data - sort by year and month
                state_data = state_data[state_data['month'].notna()].sort_values(['year', 'month'])
            else:
                # Use annual data - sort by year
                state_data = state_data.sort_values('year')
            
            if len(state_data) <= seq_length:
                continue
            
            for i in range(len(state_data) - seq_length):
                seq = state_data.iloc[i:i+seq_length]
                
                # Features: [production, energy, grid_ci, year, month]
                seq_features: List[List[float]] = []
                for _, row in seq.iterrows():
                    month_norm = float((row.get('month', 6) - 1) / 11.0) if pd.notna(row.get('month')) else 0.5
                    seq_features.append([
                        float(row['production_tonnes']) / 1e6,
                        float(row['energy_consumption_mwh']) / 1e6,
                        float(row['grid_carbon_intensity']),
                        float(row['year'] - 2015) / 10.0,
                        month_norm
                    ])
                
                # Target: next period emissions
                target_row = state_data.iloc[i + seq_length]
                target = float(target_row['emissions_tco2']) / 1e6
                
                sequences.append(seq_features)
                targets.append([target])
    
    # If we still have no sequences, skip training gracefully
    if not sequences:
        logger.warning(f"No time-series sequences generated for sector '{sector}'. Skipping Transformer training.")
        dummy_model = LCATransformer(input_dim=4)
        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
        return dummy_model, history
    
    X = torch.FloatTensor(sequences)  # (batch, seq_len, features)
    y = torch.FloatTensor(targets)  # (batch, 1)
    
    # Train/val split
    n_samples = len(X)
    n_train = int(0.8 * n_samples)
    indices = torch.randperm(n_samples)
    
    train_data = {
        "X": X[indices[:n_train]],
        "y": y[indices[:n_train]]
    }
    val_data = {
        "X": X[indices[n_train:]],
        "y": y[indices[n_train:]]
    }
    
    # Train
    if len(sequences) > 0:
        model, history = train_transformer_model(train_data, val_data)
    else:
        logger.warning(f"No time-series sequences generated for sector '{sector}'. Skipping Transformer training.")
        # Return dummy model
        from lca_optimizer.models.transformer import LCATransformer
        model = LCATransformer(input_dim=5, d_model=64)
        history = {"train_loss": [], "val_loss": []}
    
    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = output_path / f"indian_{sector}_transformer.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": {
            "input_dim": model.input_dim,
            "d_model": model.d_model
        },
        "sector": sector,
        "states": states,
        "training_history": history
    }, model_path)
    
    logger.info(f"Transformer model saved to {model_path}")
    
    return model, history


def train_all_indian_models(
    sectors: List[str] = ["steel", "cement", "aluminium"],
    states: Optional[List[str]] = None,
    output_dir: str = "models/trained/indian"
):
    """
    Train all ML models for Indian industries.
    
    Args:
        sectors: List of sectors to train
        states: States to include (None for all)
        output_dir: Output directory
    """
    logger.info("Starting training for Indian industrial models")
    
    for sector in sectors:
        logger.info(f"Training models for {sector} sector...")
        
        # Train PINN
        try:
            train_indian_pinn(sector, states, output_dir=output_dir)
        except Exception as e:
            logger.error(f"Failed to train PINN for {sector}: {e}")
        
        # Train Transformer
        try:
            train_indian_transformer(sector, states, output_dir=output_dir)
        except Exception as e:
            logger.error(f"Failed to train Transformer for {sector}: {e}")
    
    logger.info("Training completed for all Indian models")

