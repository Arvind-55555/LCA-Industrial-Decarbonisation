"""
Graph Neural Network (GNN) for Process Flow Diagrams and Value Chain Networks
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
from typing import Dict, List, Tuple, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ProcessGNN(nn.Module):
    """
    Graph Neural Network for modeling process flow diagrams (PFDs) and value chains.
    
    Nodes: Process units, materials, energy streams
    Edges: Material/energy flows
    """
    
    def __init__(
        self,
        node_features: int,
        edge_features: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        gnn_type: str = "GCN"
    ):
        """
        Initialize Process GNN.
        
        Args:
            node_features: Number of node features
            edge_features: Number of edge features
            hidden_dim: Hidden dimension
            num_layers: Number of GNN layers
            gnn_type: GNN type ("GCN", "GAT")
        """
        super(ProcessGNN, self).__init__()
        
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input projection
        self.node_encoder = nn.Linear(node_features, hidden_dim)
        self.edge_encoder = nn.Linear(edge_features, hidden_dim)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            if gnn_type == "GCN":
                self.gnn_layers.append(GCNConv(hidden_dim, hidden_dim))
            elif gnn_type == "GAT":
                self.gnn_layers.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=False))
            else:
                raise ValueError(f"Unknown GNN type: {gnn_type}")
        
        # Output layers
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # Emission prediction
        )
        
        logger.info(f"Process GNN initialized: {gnn_type}, {num_layers} layers")
    
    def forward(
        self,
        data: Data
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            data: Graph data (nodes, edges, edge_index)
        
        Returns:
            Node-level or graph-level emissions
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Encode nodes and edges
        x = self.node_encoder(x)
        if edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)
        
        # GNN layers
        for layer in self.gnn_layers:
            x = layer(x, edge_index)
            x = torch.relu(x)
        
        # Graph-level prediction (pooling)
        if hasattr(data, 'batch') and data.batch is not None:
            # Batch of graphs
            x = global_mean_pool(x, data.batch)
        else:
            # Single graph
            x = x.mean(dim=0, keepdim=True)
        
        # Output
        emissions = self.output_layer(x)
        
        return emissions
    
    def predict_node_emissions(
        self,
        data: Data
    ) -> torch.Tensor:
        """
        Predict emissions at node level.
        
        Args:
            data: Graph data
        
        Returns:
            Node-level emissions
        """
        x, edge_index = data.x, data.edge_index
        
        # Encode
        x = self.node_encoder(x)
        
        # GNN layers
        for layer in self.gnn_layers:
            x = layer(x, edge_index)
            x = torch.relu(x)
        
        # Node-level output
        node_emissions = self.output_layer(x)
        
        return node_emissions
    
    @staticmethod
    def create_graph_from_process(
        nodes: List[Dict[str, float]],
        edges: List[Tuple[int, int, Dict[str, float]]]
    ) -> Data:
        """
        Create graph data from process description.
        
        Args:
            nodes: List of node features
            edges: List of (source, target, edge_features) tuples
        
        Returns:
            Graph data object
        """
        # Node features
        node_features = torch.tensor([list(node.values()) for node in nodes], dtype=torch.float)
        
        # Edge indices and features
        edge_index = torch.tensor([[e[0], e[1]] for e in edges], dtype=torch.long).t().contiguous()
        edge_features = torch.tensor([list(e[2].values()) for e in edges], dtype=torch.float)
        
        return Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_features
        )

