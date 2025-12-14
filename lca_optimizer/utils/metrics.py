"""Evaluation metrics for LCA models"""

import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_uncertainty: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calculate evaluation metrics.
    
    Metrics:
    - MAE: Mean Absolute Error
    - RMSE: Root Mean Squared Error
    - R²: Coefficient of determination
    - ECE: Expected Calibration Error (if uncertainty provided)
    
    Args:
        y_true: True values
        y_pred: Predicted values
        y_uncertainty: Uncertainty estimates (optional)
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # MAE
    mae = np.mean(np.abs(y_true - y_pred))
    metrics["MAE"] = float(mae)
    
    # RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    metrics["RMSE"] = float(rmse)
    
    # R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    metrics["R2"] = float(r2)
    
    # ECE (Expected Calibration Error)
    if y_uncertainty is not None:
        ece = calculate_ece(y_true, y_pred, y_uncertainty)
        metrics["ECE"] = float(ece)
    
    return metrics


def calculate_ece(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_uncertainty: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Calculate Expected Calibration Error.
    
    Measures how well-calibrated uncertainty estimates are.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        y_uncertainty: Uncertainty estimates
        n_bins: Number of bins for calibration
    
    Returns:
        ECE score
    """
    # Calculate prediction intervals
    lower = y_pred - y_uncertainty
    upper = y_pred + y_uncertainty
    
    # Check if true values fall within intervals
    in_interval = (y_true >= lower) & (y_true <= upper)
    
    # Bin by confidence level
    confidence_levels = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        # Get predictions in this confidence bin
        # Simplified: use uncertainty as proxy for confidence
        # In practice, you'd use actual confidence intervals
        mask = (y_uncertainty >= confidence_levels[i]) & (y_uncertainty < confidence_levels[i + 1])
        
        if np.sum(mask) > 0:
            accuracy = np.mean(in_interval[mask])
            confidence = confidence_levels[i] + (confidence_levels[i + 1] - confidence_levels[i]) / 2
            ece += np.abs(accuracy - confidence) * np.sum(mask)
    
    ece /= len(y_true)
    
    return ece

