"""
Transformer for Time-Series Operational Sensor Data
LSTM with attention for real-time LCA updating
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


class LCATransformer(nn.Module):
    """
    Transformer model for time-series LCA prediction.
    
    Uses attention mechanism to capture temporal dependencies
    in operational sensor data for real-time LCA updating.
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_seq_length: int = 100
    ):
        """
        Initialize LCA Transformer.
        
        Args:
            input_dim: Input feature dimension
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout rate
            max_seq_length: Maximum sequence length
        """
        super(LCATransformer, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_length)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, 1)  # Emission prediction
        )
        
        logger.info(f"LCA Transformer initialized: {input_dim} -> {d_model}")
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, input_dim)
            mask: Attention mask (optional)
        
        Returns:
            Emission predictions (batch, seq_len, 1)
        """
        # Project input
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.transformer(x, mask=mask)
        
        # Output prediction
        emissions = self.output_layer(x)
        
        return emissions
    
    def predict_next(
        self,
        x: torch.Tensor,
        n_steps: int = 1
    ) -> torch.Tensor:
        """
        Predict next n_steps emissions.
        
        Args:
            x: Input sequence (batch, seq_len, input_dim)
            n_steps: Number of steps to predict ahead
        
        Returns:
            Future emissions (batch, n_steps, 1)
        """
        predictions = []
        current_seq = x
        
        for _ in range(n_steps):
            # Predict next step
            next_pred = self.forward(current_seq)
            predictions.append(next_pred[:, -1:, :])
            
            # Append prediction to sequence (for autoregressive prediction)
            # In practice, you'd need to construct full input features
            # This is a simplified version
            current_seq = torch.cat([current_seq, next_pred], dim=1)
        
        return torch.cat(predictions, dim=1)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 5000
    ):
        super(PositionalEncoding, self).__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
        
        Returns:
            Input with positional encoding
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

