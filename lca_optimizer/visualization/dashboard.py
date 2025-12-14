"""
Interactive Dashboard for LCA Results
Uses Dash/Plotly for web-based visualization
"""

try:
    import dash
    from dash import dcc, html, Input, Output
    import plotly.graph_objs as go
    import plotly.express as px
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# Plotly config to disable Plotly Cloud and remove unnecessary buttons
PLOTLY_CONFIG = {
    'displayModeBar': True,
    'displaylogo': False,
    'modeBarButtonsToRemove': ['sendDataToCloud', 'lasso2d', 'select2d'],
    'toImageButtonOptions': {
        'format': 'png',
        'filename': 'lca_chart',
        'height': 500,
        'width': 700,
        'scale': 1
    }
}


def create_dashboard(
    data: Optional[pd.DataFrame] = None,
    port: int = 8050
):
    """
    Create interactive dashboard for LCA results.
    
    Args:
        data: DataFrame with LCA data
        port: Port to run dashboard on
    
    Returns:
        Dash app
    """
    if not DASH_AVAILABLE:
        raise ImportError(
            "Dash not available. Install with: pip install dash plotly"
        )
    
    app = dash.Dash(__name__)
    
    app.layout = html.Div([
        html.H1("LCA Optimizer Dashboard", style={'textAlign': 'center'}),
        
        html.Div([
            dcc.Dropdown(
                id='location-dropdown',
                options=[
                    {'label': 'US', 'value': 'US'},
                    {'label': 'EU', 'value': 'EU'},
                    {'label': 'Germany', 'value': 'DE'},
                    {'label': 'France', 'value': 'FR'},
                    {'label': 'UK', 'value': 'GB'}
                ],
                value='US',
                style={'width': '200px'}
            ),
            dcc.Dropdown(
                id='sector-dropdown',
                options=[
                    {'label': 'Steel', 'value': 'steel'},
                    {'label': 'Cement', 'value': 'cement'},
                    {'label': 'Shipping', 'value': 'shipping'},
                    {'label': 'Aluminium', 'value': 'aluminium'}
                ],
                value='steel',
                style={'width': '200px', 'marginLeft': '20px'}
            )
        ], style={'display': 'flex', 'padding': '20px'}),
        
        dcc.Graph(id='carbon-intensity-chart', config=PLOTLY_CONFIG),
        dcc.Graph(id='emission-breakdown-chart', config=PLOTLY_CONFIG)
    ])
    
    @app.callback(
        [Output('carbon-intensity-chart', 'figure'),
         Output('emission-breakdown-chart', 'figure')],
        [Input('location-dropdown', 'value'),
         Input('sector-dropdown', 'value')]
    )
    def update_charts(location, sector):
        # Placeholder - would load real data
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=list(range(24)),
            y=[300 + 50 * abs(np.sin(i * np.pi / 12)) for i in range(24)],
            mode='lines',
            name='Carbon Intensity'
        ))
        fig1.update_layout(
            title=f'Grid Carbon Intensity - {location}',
            xaxis_title='Hour',
            yaxis_title='g CO2eq/kWh'
        )
        
        fig2 = go.Figure(data=[
            go.Bar(x=['Upstream', 'Process', 'Downstream'],
                  y=[100, 200, 50],
                  name='Emissions')
        ])
        fig2.update_layout(
            title=f'Emission Breakdown - {sector}',
            yaxis_title='kg CO2eq'
        )
        
        return fig1, fig2
    
    logger.info(f"Dashboard created on port {port}")
    return app


def run_dashboard(port: int = 8050, debug: bool = True):
    """Run the dashboard server"""
    app = create_dashboard()
    app.run(port=port, debug=debug)

