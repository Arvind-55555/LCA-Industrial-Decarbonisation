"""
Results Dashboard - Focused on LCA Process Improvement
Shows LCA results, grid carbon intensity, sector comparison, and improvement opportunities
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta

try:
    import dash
    from dash import dcc, html, Input, Output, dash_table
    import plotly.graph_objs as go
    import plotly.express as px
    import pandas as pd
    import numpy as np
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False
    logging.warning("Dash/Plotly not available. Install with: pip install dash plotly")

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


class ResultsDashboard:
    """
    Comprehensive dashboard focused on LCA process improvement.
    
    Features:
    - LCA results with improvement opportunities
    - Grid carbon intensity with time period selection
    - Sector comparison with optimization potential
    - Charts and images gallery
    """
    
    def __init__(self, results_dir: str = "outputs", data_dir: str = "data/raw"):
        """
        Initialize results dashboard.
        
        Args:
            results_dir: Directory containing results and plots
            data_dir: Directory containing data files
        """
        self.results_dir = Path(results_dir)
        self.data_dir = Path(data_dir)
        self.plots_dir = self.results_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Results dashboard initialized: {self.results_dir}")
    
    def create_dashboard(self, port: int = 8050):
        """
        Create comprehensive results dashboard.
        
        Args:
            port: Port to run dashboard on
        
        Returns:
            Dash app
        """
        if not DASH_AVAILABLE:
            logger.error("Dash not available. Install with: pip install dash plotly")
            return None
        
        # Custom CSS for better styling
        external_stylesheets = [
            'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap',
            {
                'href': 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css',
                'rel': 'stylesheet'
            }
        ]
        
        app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
        
        # Get available images
        images = self._get_available_images()
        
        app.layout = html.Div([
            # Header
            html.Div([
                html.Div([
                    html.H1(
                        "LCA Optimizer Dashboard",
                        style={
                            'color': '#1a1a1a',
                            'fontSize': '42px',
                            'fontWeight': '700',
                            'margin': '0',
                            'fontFamily': 'Inter, sans-serif'
                        }
                    ),
                    html.P(
                        "Life Cycle Assessment Process Improvement & Optimization",
                        style={
                            'color': '#666',
                            'fontSize': '18px',
                            'margin': '10px 0 0 0',
                            'fontFamily': 'Inter, sans-serif'
                        }
                    )
                ], style={'flex': '1'}),
                html.Div([
                    html.Div([
                        html.I(className="fas fa-chart-line", style={'marginRight': '8px'}),
                        html.Span("Real-time Analysis", style={'fontSize': '14px'})
                    ], style={
                        'padding': '10px 20px',
                        'background': '#f0f4f8',
                        'borderRadius': '8px',
                        'color': '#2c3e50',
                        'fontFamily': 'Inter, sans-serif'
                    })
                ])
            ], style={
                'display': 'flex',
                'justifyContent': 'space-between',
                'alignItems': 'center',
                'padding': '30px 40px',
                'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                'color': 'white',
                'marginBottom': '30px',
                'borderRadius': '0 0 20px 20px',
                'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'
            }),
            
            # Main Content
            html.Div([
                # Navigation Tabs
                dcc.Tabs(
                    id="main-tabs",
                    value='lca',
                    children=[
                        dcc.Tab(
                            label='📊 LCA Results',
                            value='lca',
                            style={'fontFamily': 'Inter, sans-serif', 'fontWeight': '500'},
                            selected_style={'fontFamily': 'Inter, sans-serif', 'fontWeight': '600'}
                        ),
                        dcc.Tab(
                            label='🌍 Grid Carbon Intensity',
                            value='grid',
                            style={'fontFamily': 'Inter, sans-serif', 'fontWeight': '500'},
                            selected_style={'fontFamily': 'Inter, sans-serif', 'fontWeight': '600'}
                        ),
                        dcc.Tab(
                            label='🏭 Sector Comparison',
                            value='sectors',
                            style={'fontFamily': 'Inter, sans-serif', 'fontWeight': '500'},
                            selected_style={'fontFamily': 'Inter, sans-serif', 'fontWeight': '600'}
                        ),
                        dcc.Tab(
                            label='🖼️ Charts & Images',
                            value='images',
                            style={'fontFamily': 'Inter, sans-serif', 'fontWeight': '500'},
                            selected_style={'fontFamily': 'Inter, sans-serif', 'fontWeight': '600'}
                        ),
                    ],
                    style={
                        'fontFamily': 'Inter, sans-serif',
                        'marginBottom': '30px'
                    }
                ),
                
                html.Div(id='tab-content', style={'padding': '0 20px'})
            ], style={'maxWidth': '1400px', 'margin': '0 auto', 'padding': '20px'}),
            
            # Footer
            html.Div([
                html.Hr(style={'border': 'none', 'borderTop': '1px solid #e0e0e0', 'margin': '40px 0 20px'}),
                html.P(
                    "LCA Optimizer - Deep Learning for Industrial Decarbonisation | Focused on Process Improvement",
                    style={
                        'textAlign': 'center',
                        'color': '#999',
                        'fontSize': '14px',
                        'fontFamily': 'Inter, sans-serif',
                        'margin': '20px 0'
                    }
                )
            ])
        ], style={'fontFamily': 'Inter, sans-serif', 'backgroundColor': '#f8f9fa', 'minHeight': '100vh'})
        
        @app.callback(
            Output('tab-content', 'children'),
            Input('main-tabs', 'value')
        )
        def render_tab(tab):
            if tab == 'lca':
                return self._create_lca_tab()
            elif tab == 'grid':
                return self._create_grid_tab()
            elif tab == 'sectors':
                return self._create_sectors_tab()
            elif tab == 'images':
                return self._create_images_tab(images)
            return html.Div("Tab content")
        
        # Grid carbon intensity callback
        @app.callback(
            Output('grid-content', 'children'),
            [Input('time-period-dropdown', 'value'),
             Input('location-dropdown', 'value')]
        )
        def update_grid_charts(time_period, location):
            return self._create_grid_charts(time_period, location)
        
        logger.info(f"Dashboard created on port {port}")
        return app
    
    def _create_lca_tab(self):
        """Create LCA results tab with improvement opportunities"""
        # Sample data - in real app, load from results
        sectors = ['Steel', 'Cement', 'Shipping', 'Aluminium']
        baseline = [1800, 900, 3000, 16000]
        optimized = [550, 575, 300, 500]
        reduction = [(b - o) / b * 100 for b, o in zip(baseline, optimized)]
        improvement_potential = [r for r in reduction]
        
        # Create main comparison chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Baseline Emissions',
            x=sectors,
            y=baseline,
            marker_color='#e74c3c',
            text=[f'{b:,.0f}' for b in baseline],
            textposition='outside',
            textfont=dict(size=12, color='#2c3e50')
        ))
        fig.add_trace(go.Bar(
            name='Optimized Emissions',
            x=sectors,
            y=optimized,
            marker_color='#27ae60',
            text=[f'{o:,.0f}' for o in optimized],
            textposition='outside',
            textfont=dict(size=12, color='#2c3e50')
        ))
        fig.update_layout(
            title={
                'text': 'LCA Results: Baseline vs Optimized Emissions',
                'font': {'size': 24, 'family': 'Inter, sans-serif', 'color': '#2c3e50'},
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title='Sector',
            yaxis_title='Emissions (kg CO₂eq/t)',
            barmode='group',
            height=500,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family='Inter, sans-serif', size=12),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        
        # Improvement potential chart
        fig2 = go.Figure()
        colors = ['#27ae60' if r > 50 else '#f39c12' if r > 25 else '#e74c3c' for r in improvement_potential]
        fig2.add_trace(go.Bar(
            x=sectors,
            y=improvement_potential,
            marker_color=colors,
            text=[f'{r:.1f}%' for r in improvement_potential],
            textposition='outside',
            textfont=dict(size=14, color='#2c3e50', weight='bold')
        ))
        fig2.update_layout(
            title={
                'text': 'Improvement Potential by Sector',
                'font': {'size': 24, 'family': 'Inter, sans-serif', 'color': '#2c3e50'},
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title='Sector',
            yaxis_title='Emission Reduction Potential (%)',
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family='Inter, sans-serif', size=12),
            yaxis=dict(range=[0, max(improvement_potential) * 1.2])
        )
        
        # Breakdown by category
        breakdown_data = {
            'Steel': {'upstream': 200, 'process': 300, 'downstream': 50},
            'Cement': {'upstream': 150, 'process': 350, 'downstream': 75},
            'Shipping': {'upstream': 100, 'process': 150, 'downstream': 50},
            'Aluminium': {'upstream': 200, 'process': 250, 'downstream': 50}
        }
        
        fig3 = go.Figure()
        categories = ['Upstream', 'Process', 'Downstream']
        for i, category in enumerate(categories):
            values = [breakdown_data[s][category.lower()] for s in sectors]
            fig3.add_trace(go.Bar(
                name=category,
                x=sectors,
                y=values,
                marker_color=['#3498db', '#9b59b6', '#e67e22'][i]
            ))
        fig3.update_layout(
            title={
                'text': 'Emission Breakdown by Category',
                'font': {'size': 24, 'family': 'Inter, sans-serif', 'color': '#2c3e50'},
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title='Sector',
            yaxis_title='Emissions (kg CO₂eq/t)',
            barmode='stack',
            height=450,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family='Inter, sans-serif', size=12),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        
        # Improvement opportunities table
        opportunities = pd.DataFrame({
            'Sector': sectors,
            'Current Emissions': baseline,
            'Optimized Emissions': optimized,
            'Reduction (%)': [f'{r:.1f}%' for r in reduction],
            'Improvement Priority': ['High' if r > 50 else 'Medium' if r > 25 else 'Low' for r in reduction],
            'Key Opportunity': [
                'H₂-DRI process optimization',
                'CCUS integration & material substitution',
                'Fuel pathway switching',
                'Electrification & recycling'
            ]
        })
        
        return html.Div([
            html.Div([
                html.H2("📊 LCA Results & Improvement Opportunities", style={
                    'color': '#2c3e50',
                    'fontSize': '32px',
                    'fontWeight': '700',
                    'marginBottom': '30px',
                    'fontFamily': 'Inter, sans-serif'
                }),
                html.P("Analysis of current emissions and optimization potential across sectors", style={
                    'color': '#666',
                    'fontSize': '16px',
                    'marginBottom': '30px',
                    'fontFamily': 'Inter, sans-serif'
                })
            ]),
            
            dcc.Graph(figure=fig, style={'marginBottom': '40px'}, config=PLOTLY_CONFIG),
            
            html.Div([
                html.Div([
                    dcc.Graph(figure=fig2, style={'height': '100%'}, config=PLOTLY_CONFIG)
                ], className='six columns'),
                html.Div([
                    dcc.Graph(figure=fig3, style={'height': '100%'}, config=PLOTLY_CONFIG)
                ], className='six columns')
            ], className='row', style={'marginBottom': '40px'}),
            
            html.Div([
                html.H3("Improvement Opportunities", style={
                    'color': '#2c3e50',
                    'fontSize': '24px',
                    'fontWeight': '600',
                    'marginBottom': '20px',
                    'fontFamily': 'Inter, sans-serif'
                }),
                dash_table.DataTable(
                    data=opportunities.to_dict('records'),
                    columns=[{"name": i, "id": i} for i in opportunities.columns],
                    style_cell={
                        'textAlign': 'left',
                        'padding': '12px',
                        'fontFamily': 'Inter, sans-serif',
                        'fontSize': '14px'
                    },
                    style_header={
                        'backgroundColor': '#667eea',
                        'color': 'white',
                        'fontWeight': '600',
                        'fontSize': '15px',
                        'textAlign': 'center'
                    },
                    style_data_conditional=[
                        {
                            'if': {'filter_query': '{Improvement Priority} = High'},
                            'backgroundColor': '#fee',
                            'fontWeight': '500'
                        },
                        {
                            'if': {'filter_query': '{Improvement Priority} = Medium'},
                            'backgroundColor': '#fff9e6'
                        }
                    ],
                    style_as_list_view=True
                )
            ])
        ])
    
    def _create_grid_tab(self):
        """Create grid carbon intensity tab with time period selection"""
        return html.Div([
            html.Div([
                html.H2("🌍 Grid Carbon Intensity Analysis", style={
                    'color': '#2c3e50',
                    'fontSize': '32px',
                    'fontWeight': '700',
                    'marginBottom': '20px',
                    'fontFamily': 'Inter, sans-serif'
                }),
                html.P("Select time period and location to analyze grid carbon intensity trends", style={
                    'color': '#666',
                    'fontSize': '16px',
                    'marginBottom': '30px',
                    'fontFamily': 'Inter, sans-serif'
                })
            ]),
            
            html.Div([
                html.Div([
                    html.Label("Time Period:", style={
                        'fontWeight': '600',
                        'marginBottom': '10px',
                        'fontFamily': 'Inter, sans-serif',
                        'color': '#2c3e50'
                    }),
                    dcc.Dropdown(
                        id='time-period-dropdown',
                        options=[
                            {'label': 'Last 7 Days', 'value': '7d'},
                            {'label': 'Last 30 Days (Monthly)', 'value': '30d'},
                            {'label': 'Last Year', 'value': '1y'},
                            {'label': 'Last 3 Years', 'value': '3y'},
                            {'label': 'Last 5 Years', 'value': '5y'}
                        ],
                        value='30d',
                        style={'fontFamily': 'Inter, sans-serif'}
                    )
                ], className='four columns', style={'padding': '10px'}),
                
                html.Div([
                    html.Label("Location:", style={
                        'fontWeight': '600',
                        'marginBottom': '10px',
                        'fontFamily': 'Inter, sans-serif',
                        'color': '#2c3e50'
                    }),
                    dcc.Dropdown(
                        id='location-dropdown',
                        options=[
                            {'label': 'United States (US)', 'value': 'US'},
                            {'label': 'European Union (EU)', 'value': 'EU'},
                            {'label': 'Germany (DE)', 'value': 'DE'},
                            {'label': 'France (FR)', 'value': 'FR'},
                            {'label': 'United Kingdom (GB)', 'value': 'GB'}
                        ],
                        value='US',
                        style={'fontFamily': 'Inter, sans-serif'}
                    )
                ], className='four columns', style={'padding': '10px'}),
            ], className='row', style={'marginBottom': '30px'}),
            
            html.Div(id='grid-content')
        ])
    
    def _create_grid_charts(self, time_period: str, location: str):
        """Create grid carbon intensity charts based on time period"""
        try:
            from lca_optimizer.data.local_data_loader import LocalGridDataLoader
            loader = LocalGridDataLoader(data_dir=str(self.data_dir))
            
            # Calculate date range based on time period
            end_date = datetime.now()
            if time_period == '7d':
                start_date = end_date - timedelta(days=7)
                freq = 'hourly'
            elif time_period == '30d':
                start_date = end_date - timedelta(days=30)
                freq = 'hourly'
            elif time_period == '1y':
                start_date = end_date - timedelta(days=365)
                freq = 'daily'
            elif time_period == '3y':
                start_date = end_date - timedelta(days=365*3)
                freq = 'monthly'
            elif time_period == '5y':
                start_date = end_date - timedelta(days=365*5)
                freq = 'monthly'
            else:
                start_date = end_date - timedelta(days=30)
                freq = 'hourly'
            
            # Get historical data
            historical = loader.get_historical_carbon_intensity(
                location, start_date, end_date, frequency=freq
            )
            
            if historical.empty:
                return html.Div([
                    html.P("No data available for selected period.", style={
                        'color': '#999',
                        'fontSize': '16px',
                        'fontFamily': 'Inter, sans-serif'
                    })
                ])
            
            # Resample for monthly/yearly views
            if freq == 'monthly' and 'timestamp' in historical.columns:
                historical = historical.set_index('timestamp')
                historical = historical.resample('M').mean()
                historical = historical.reset_index()
            
            # Create time series chart
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=historical['timestamp'],
                y=historical['carbon_intensity'],
                mode='lines',
                name='Carbon Intensity',
                line=dict(color='#667eea', width=2),
                fill='tozeroy',
                fillcolor='rgba(102, 126, 234, 0.1)'
            ))
            
            # Add mean line
            mean_ci = historical['carbon_intensity'].mean()
            fig1.add_hline(
                y=mean_ci,
                line_dash="dash",
                line_color="#e74c3c",
                annotation_text=f"Mean: {mean_ci:.1f} g CO₂eq/kWh",
                annotation_position="right"
            )
            
            fig1.update_layout(
                title={
                    'text': f'Grid Carbon Intensity - {location} ({time_period.upper()})',
                    'font': {'size': 24, 'family': 'Inter, sans-serif', 'color': '#2c3e50'},
                    'x': 0.5,
                    'xanchor': 'center'
                },
                xaxis_title='Time',
                yaxis_title='Carbon Intensity (g CO₂eq/kWh)',
                height=500,
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family='Inter, sans-serif', size=12),
                hovermode='x unified'
            )
            
            # Statistics cards
            stats = {
                'Mean': f"{historical['carbon_intensity'].mean():.1f}",
                'Min': f"{historical['carbon_intensity'].min():.1f}",
                'Max': f"{historical['carbon_intensity'].max():.1f}",
                'Std Dev': f"{historical['carbon_intensity'].std():.1f}"
            }
            
            stats_cards = []
            for label, value in stats.items():
                stats_cards.append(
                    html.Div([
                        html.Div(value, style={
                            'fontSize': '32px',
                            'fontWeight': '700',
                            'color': '#667eea',
                            'fontFamily': 'Inter, sans-serif'
                        }),
                        html.Div(label, style={
                            'fontSize': '14px',
                            'color': '#666',
                            'marginTop': '5px',
                            'fontFamily': 'Inter, sans-serif'
                        })
                    ], style={
                        'padding': '20px',
                        'backgroundColor': 'white',
                        'borderRadius': '8px',
                        'textAlign': 'center',
                        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                        'margin': '10px'
                    })
                )
            
            return html.Div([
                html.Div(stats_cards, style={'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': '30px'}),
                dcc.Graph(figure=fig1, style={'marginBottom': '30px'}, config=PLOTLY_CONFIG)
            ])
            
        except Exception as e:
            logger.error(f"Error loading grid data: {e}")
            return html.Div([
                html.P(f"Error loading data: {str(e)}", style={
                    'color': '#e74c3c',
                    'fontSize': '16px',
                    'fontFamily': 'Inter, sans-serif'
                })
            ])
    
    def _create_sectors_tab(self):
        """Create sector comparison tab"""
        sectors_data = {
            'Sector': ['Steel', 'Cement', 'Shipping', 'Aluminium'],
            'Baseline (t CO₂/t)': [1.8, 0.9, 3.0, 16.0],
            'Optimized (t CO₂/t)': [0.55, 0.575, 0.3, 0.5],
            'Reduction %': [69.4, 36.1, 90.0, 96.9],
            'Improvement Priority': ['High', 'Medium', 'High', 'High']
        }
        
        df = pd.DataFrame(sectors_data)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Baseline',
            x=df['Sector'],
            y=df['Baseline (t CO₂/t)'],
            marker_color='#e74c3c',
            text=[f'{v:.1f}' for v in df['Baseline (t CO₂/t)']],
            textposition='outside'
        ))
        fig.add_trace(go.Bar(
            name='Optimized',
            x=df['Sector'],
            y=df['Optimized (t CO₂/t)'],
            marker_color='#27ae60',
            text=[f'{v:.2f}' for v in df['Optimized (t CO₂/t)']],
            textposition='outside'
        ))
        fig.update_layout(
            title={
                'text': 'Sector Comparison: Baseline vs Optimized Emissions',
                'font': {'size': 24, 'family': 'Inter, sans-serif', 'color': '#2c3e50'},
                'x': 0.5,
                'xanchor': 'center'
            },
            barmode='group',
            height=500,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family='Inter, sans-serif', size=12),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        
        return html.Div([
            html.H2("🏭 Sector Comparison & Optimization Potential", style={
                'color': '#2c3e50',
                'fontSize': '32px',
                'fontWeight': '700',
                'marginBottom': '30px',
                'fontFamily': 'Inter, sans-serif'
            }),
            dcc.Graph(figure=fig, style={'marginBottom': '30px'}, config=PLOTLY_CONFIG),
            html.H3("Detailed Comparison", style={
                'color': '#2c3e50',
                'fontSize': '24px',
                'fontWeight': '600',
                'marginBottom': '20px',
                'fontFamily': 'Inter, sans-serif'
            }),
            dash_table.DataTable(
                data=df.to_dict('records'),
                columns=[{"name": i, "id": i} for i in df.columns],
                style_cell={
                    'textAlign': 'left',
                    'padding': '12px',
                    'fontFamily': 'Inter, sans-serif',
                    'fontSize': '14px'
                },
                style_header={
                    'backgroundColor': '#667eea',
                    'color': 'white',
                    'fontWeight': '600',
                    'fontSize': '15px',
                    'textAlign': 'center'
                },
                style_data_conditional=[
                    {
                        'if': {'filter_query': '{Improvement Priority} = High'},
                        'backgroundColor': '#fee',
                        'fontWeight': '500'
                    }
                ]
            )
        ])
    
    def _create_images_tab(self, images: List[str]):
        """Create images gallery tab"""
        if not images:
            return html.Div([
                html.H2("🖼️ Charts & Images", style={
                    'color': '#2c3e50',
                    'fontSize': '32px',
                    'fontWeight': '700',
                    'marginBottom': '20px',
                    'fontFamily': 'Inter, sans-serif'
                }),
                html.P("No images found. Run examples to generate plots.", style={
                    'color': '#666',
                    'fontSize': '16px',
                    'fontFamily': 'Inter, sans-serif'
                }),
                html.P("Try: python examples/end_to_end_example.py", style={
                    'color': '#999',
                    'fontSize': '14px',
                    'fontFamily': 'Inter, sans-serif'
                })
            ])
        
        image_elements = []
        for img in images:
            img_path = self.plots_dir / img
            if img_path.exists():
                import base64
                with open(img_path, 'rb') as f:
                    img_data = base64.b64encode(f.read()).decode('utf-8')
                
                image_elements.append(
                    html.Div([
                        html.H4(
                            img.replace('.png', '').replace('_', ' ').title(),
                            style={
                                'color': '#2c3e50',
                                'marginBottom': '15px',
                                'fontFamily': 'Inter, sans-serif',
                                'fontSize': '20px',
                                'fontWeight': '600'
                            }
                        ),
                        html.Img(
                            src=f"data:image/png;base64,{img_data}",
                            style={
                                'width': '100%',
                                'maxWidth': '1000px',
                                'margin': '10px 0',
                                'border': '2px solid #e0e0e0',
                                'borderRadius': '8px',
                                'boxShadow': '0 2px 8px rgba(0,0,0,0.1)'
                            }
                        ),
                        html.P(
                            f"File: {img}",
                            style={
                                'color': '#999',
                                'fontSize': '12px',
                                'fontFamily': 'Inter, sans-serif',
                                'marginTop': '10px'
                            }
                        )
                    ], style={
                        'margin': '30px 0',
                        'padding': '30px',
                        'backgroundColor': 'white',
                        'borderRadius': '10px',
                        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
                    })
                )
        
        return html.Div([
            html.H2("🖼️ Generated Charts & Images", style={
                'color': '#2c3e50',
                'fontSize': '32px',
                'fontWeight': '700',
                'marginBottom': '20px',
                'fontFamily': 'Inter, sans-serif'
            }),
            html.Div([
                html.P(
                    f"Found {len(images)} image(s) in {self.plots_dir}",
                    style={
                        'fontSize': '16px',
                        'color': '#666',
                        'fontFamily': 'Inter, sans-serif',
                        'marginBottom': '10px'
                    }
                ),
                html.P(
                    "These visualizations show LCA results, grid carbon intensity, and sector comparisons.",
                    style={
                        'fontSize': '14px',
                        'color': '#999',
                        'fontFamily': 'Inter, sans-serif'
                    }
                )
            ], style={'marginBottom': '30px'}),
            html.Div(image_elements)
        ])
    
    def _get_available_images(self) -> List[str]:
        """Get list of available plot images - LCA breakdown, sector comparison, and ML results"""
        if not self.plots_dir.exists():
            return []
        
        # Allowed charts: LCA breakdowns, sector comparison, and ML model results
        allowed_charts = [
            "cement_lca_breakdown.png",
            "sector_comparison.png",
            "steel_lca_breakdown.png",
            "ml_model_comparison.png",  # ML model comparison
            "ml_model_performance.png"  # ML model performance
        ]
        
        all_images = [f.name for f in self.plots_dir.glob("*.png")]
        all_images.extend([f.name for f in self.plots_dir.glob("*.jpg")])
        all_images.extend([f.name for f in self.plots_dir.glob("*.jpeg")])
        
        # Filter to only allowed charts
        images = [img for img in all_images if img in allowed_charts]
        
        return sorted(images)
    
    def run(self, port: int = 8050, debug: bool = True):
        """Run the dashboard server"""
        app = self.create_dashboard(port)
        if app:
            app.run(port=port, debug=debug)
        else:
            logger.error("Failed to create dashboard")


def run_results_dashboard(port: int = 8050):
    """Run the results dashboard"""
    dashboard = ResultsDashboard()
    dashboard.run(port=port)
