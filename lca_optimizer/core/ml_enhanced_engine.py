"""
ML-Enhanced LCA Engine - Integrates ML models into LCA calculations
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import logging
from pathlib import Path

from lca_optimizer.core.engine import LCAEngine, LCAResult
from lca_optimizer.core.physics import PhysicsConstraints

logger = logging.getLogger(__name__)

# Try to import models, but don't fail if torch_geometric is missing
try:
    from lca_optimizer.models.pinn import PhysicsInformedNN
    from lca_optimizer.models.transformer import LCATransformer
    MODELS_AVAILABLE = True
except ImportError as e:
    MODELS_AVAILABLE = False
    logger.warning(f"ML models not fully available: {e}")


class MLEnhancedLCAEngine(LCAEngine):
    """
    LCA Engine enhanced with ML models for improved predictions.
    
    Integrates:
    - PINN: Physical validation
    - Transformer: Time-series predictions
    - GNN: Process flow modeling (when available)
    """
    
    def __init__(
        self,
        grid_data_source: Optional[str] = None,
        lci_database: Optional[str] = None,
        enable_uncertainty: bool = True,
        use_ml_models: bool = True
    ):
        """
        Initialize ML-enhanced LCA engine.
        
        Args:
            grid_data_source: Source for grid data
            lci_database: LCI database path
            enable_uncertainty: Enable uncertainty quantification
            use_ml_models: Enable ML model integration
        """
        super().__init__(grid_data_source, lci_database, enable_uncertainty)
        self.use_ml_models = use_ml_models
        self.pinn_model = None
        self.transformer_model = None
        self.physics = PhysicsConstraints()
        
        if use_ml_models:
            self._load_ml_models()
    
    def _load_ml_models(self):
        """Load trained ML models"""
        try:
            if not MODELS_AVAILABLE:
                logger.warning("ML models not available, skipping model loading")
                return
            
            # Load PINN model
            pinn_path = Path("models/trained/pinn_model.pt")
            if pinn_path.exists():
                try:
                    import torch
                    checkpoint = torch.load(pinn_path, map_location='cpu')
                    config = checkpoint["model_config"]
                    self.pinn_model = PhysicsInformedNN(
                        input_dim=config["input_dim"],
                        hidden_dims=[64, 128, 64],
                        output_dim=config["output_dim"]
                    )
                    self.pinn_model.load_state_dict(checkpoint["model_state_dict"])
                    self.pinn_model.eval()
                    logger.info("✅ PINN model loaded")
                except Exception as e:
                    logger.warning(f"PINN model load failed: {e}, using rule-based")
                    self.pinn_model = None
            else:
                logger.warning("PINN model not found, using rule-based calculations")
            
            # Load Transformer model
            transformer_path = Path("models/trained/transformer_model.pt")
            if transformer_path.exists():
                try:
                    import torch
                    checkpoint = torch.load(transformer_path, map_location='cpu')
                    config = checkpoint["model_config"]
                    self.transformer_model = LCATransformer(
                        input_dim=config["input_dim"],
                        d_model=config.get("d_model", 64)
                    )
                    self.transformer_model.load_state_dict(checkpoint["model_state_dict"])
                    self.transformer_model.eval()
                    logger.info("✅ Transformer model loaded")
                except Exception as e:
                    logger.warning(f"Transformer model load failed: {e}, using rule-based")
                    self.transformer_model = None
            else:
                logger.warning("Transformer model not found, using rule-based calculations")
        except Exception as e:
            logger.error(f"Error loading ML models: {e}")
            # Continue with rule-based if ML fails
            self.pinn_model = None
            self.transformer_model = None
    
    def calculate_lca(
        self,
        process_params: Dict[str, Any],
        location: str,
        timestamp: Optional[datetime] = None,
        include_uncertainty: Optional[bool] = None
    ) -> LCAResult:
        """
        Calculate LCA with ML model enhancement.
        
        Args:
            process_params: Process parameters
            location: Geographic location
            timestamp: Time for grid CI lookup
            include_uncertainty: Override uncertainty setting
        
        Returns:
            ML-enhanced LCAResult
        """
        # Get base LCA result
        base_result = super().calculate_lca(process_params, location, timestamp, include_uncertainty)
        
        if not self.use_ml_models:
            return base_result
        
        # Enhance with ML models
        enhanced_result = self._enhance_with_ml(base_result, process_params, location, timestamp)
        
        return enhanced_result
    
    def _enhance_with_ml(
        self,
        base_result: LCAResult,
        process_params: Dict[str, Any],
        location: str,
        timestamp: datetime
    ) -> LCAResult:
        """
        Enhance LCA result with ML model predictions.
        
        Args:
            base_result: Base LCA result
            process_params: Process parameters
            location: Location
            timestamp: Timestamp
        
        Returns:
            Enhanced LCAResult
        """
        # PINN validation
        if self.pinn_model:
            validated_emissions = self._validate_with_pinn(base_result, process_params)
            base_result.total_emissions = validated_emissions
        
        # Transformer time-series enhancement
        if self.transformer_model:
            time_series_prediction = self._predict_with_transformer(process_params, location, timestamp)
            if time_series_prediction is not None:
                # Adjust emissions based on time-series prediction
                base_result.total_emissions = time_series_prediction
        
        # Ensure physics constraints
        base_result = self._apply_physics_constraints(base_result, process_params)
        
        return base_result
    
    def _validate_with_pinn(
        self,
        result: LCAResult,
        process_params: Dict[str, Any]
    ) -> float:
        """
        Validate emissions with PINN model.
        
        Args:
            result: LCA result
            process_params: Process parameters
        
        Returns:
            Validated emissions
        """
        try:
            # Prepare input features for PINN
            features = self._extract_pinn_features(process_params, result)
            X = torch.FloatTensor([features])
            
            # Get PINN prediction
            with torch.no_grad():
                pinn_prediction = self.pinn_model(X, apply_constraints=True)
                pinn_emissions = pinn_prediction.item()
            
            # Blend rule-based and PINN predictions
            # Weight: 70% rule-based, 30% PINN (PINN provides validation)
            validated = 0.7 * result.total_emissions + 0.3 * pinn_emissions
            
            logger.debug(f"PINN validation: {result.total_emissions:.2f} -> {validated:.2f}")
            
            return max(0, validated)  # Ensure non-negative
        except Exception as e:
            logger.warning(f"PINN validation failed: {e}, using base result")
            return result.total_emissions
    
    def _predict_with_transformer(
        self,
        process_params: Dict[str, Any],
        location: str,
        timestamp: datetime
    ) -> Optional[float]:
        """
        Predict emissions with Transformer model.
        
        Args:
            process_params: Process parameters
            location: Location
            timestamp: Timestamp
        
        Returns:
            Predicted emissions or None
        """
        try:
            # Get historical grid CI for sequence
            from lca_optimizer.data.local_data_loader import LocalGridDataLoader
            loader = LocalGridDataLoader()
            
            end_date = timestamp
            start_date = timestamp.replace(hour=0)  # Start of day
            
            historical = loader.get_historical_carbon_intensity(
                location, start_date, end_date, frequency="hourly"
            )
            
            if historical.empty or len(historical) < 24:
                return None
            
            # Prepare sequence (last 24 hours)
            seq_data = historical.tail(24)
            
            # Extract features: [grid_ci, production_rate, energy_consumption, ...]
            features = []
            for _, row in seq_data.iterrows():
                grid_ci = row['carbon_intensity'] / 1000.0  # Normalize
                production = process_params.get("production_capacity", 1.0) / 1000000.0  # Normalize
                energy = production * 100.0  # Estimated energy
                features.append([grid_ci, production, energy, 0.5, 0.5])  # Last 2 are placeholders
            
            # Convert to tensor
            X = torch.FloatTensor([features])  # (batch=1, seq_len=24, features=5)
            
            # Predict with Transformer
            with torch.no_grad():
                predictions = self.transformer_model(X)
                # Use last prediction
                predicted_emissions = predictions[0, -1, 0].item()
            
            # Denormalize (rough estimate)
            predicted_emissions = predicted_emissions * 1000.0  # Scale back
            
            logger.debug(f"Transformer prediction: {predicted_emissions:.2f}")
            
            return max(0, predicted_emissions)
        except Exception as e:
            logger.warning(f"Transformer prediction failed: {e}")
            return None
    
    def _extract_pinn_features(
        self,
        process_params: Dict[str, Any],
        result: LCAResult
    ) -> list:
        """
        Extract features for PINN model.
        
        Returns:
            List of 10 features
        """
        # Feature extraction based on PINN training data format
        # [H2_rate, efficiency, renewable_mix, grid_ci, ...]
        
        grid_ci = result.metadata.get("grid_carbon_intensity", 300.0)
        production = process_params.get("production_capacity", 1.0)
        
        # Extract features (matching training data format)
        features = [
            min(production / 1000000.0, 1.0),  # H2 production rate (normalized)
            0.7,  # Electrolyzer efficiency (default)
            0.6,  # Renewable mix (default)
            grid_ci / 1000.0,  # Grid CI (normalized)
            0.5, 0.5, 0.5, 0.5, 0.5, 0.5  # Other process parameters
        ]
        
        return features[:10]  # Ensure 10 features
    
    def _apply_physics_constraints(
        self,
        result: LCAResult,
        process_params: Dict[str, Any]
    ) -> LCAResult:
        """
        Apply physics constraints to result.
        
        Args:
            result: LCA result
            process_params: Process parameters
        
        Returns:
            Constrained result
        """
        # Ensure non-negative
        result.total_emissions = max(0, result.total_emissions)
        
        # Apply mass/energy balance checks
        breakdown = result.breakdown
        total_breakdown = sum(breakdown.values())
        
        # If breakdown doesn't match total, adjust
        if abs(total_breakdown - result.total_emissions) > 0.01 * result.total_emissions:
            # Scale breakdown to match total
            if total_breakdown > 0:
                scale_factor = result.total_emissions / total_breakdown
                result.breakdown = {k: v * scale_factor for k, v in breakdown.items()}
        
        return result

