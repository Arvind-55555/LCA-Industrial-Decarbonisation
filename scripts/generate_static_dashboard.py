#!/usr/bin/env python3
"""
Generate static HTML dashboard for GitHub Pages deployment
"""

import json
import base64
from pathlib import Path
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_static_dashboard(output_dir: str = "docs"):
    """
    Generate static HTML dashboard for GitHub Pages.
    
    Args:
        output_dir: Output directory (default: docs for GitHub Pages)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    plots_dir = Path("outputs/plots")
    data_dir = Path("data/raw")
    
    # Get available images
    allowed_charts = [
        "cement_lca_breakdown.png",
        "sector_comparison.png",
        "steel_lca_breakdown.png",
        "ml_model_comparison.png",
        "ml_model_performance.png"
    ]
    
    all_images = []
    if plots_dir.exists():
        all_images = [f.name for f in plots_dir.glob("*.png") if f.name in allowed_charts]
    
    # Sample data for charts
    sectors = ['Steel', 'Cement', 'Shipping', 'Aluminium']
    baseline = [1800, 900, 3000, 16000]
    optimized = [550, 575, 300, 500]
    reduction = [(b - o) / b * 100 for b, o in zip(baseline, optimized)]
    
    # Generate HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI-Enhanced LCA Dashboard - Industrial Decarbonisation</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 36px;
            font-weight: 700;
            margin-bottom: 10px;
        }}
        
        .header p {{
            font-size: 18px;
            opacity: 0.9;
        }}
        
        .tabs {{
            display: flex;
            background: #f8f9fa;
            border-bottom: 2px solid #e0e0e0;
        }}
        
        .tab {{
            flex: 1;
            padding: 20px;
            text-align: center;
            cursor: pointer;
            font-weight: 600;
            font-size: 16px;
            color: #666;
            transition: all 0.3s;
            border-bottom: 3px solid transparent;
        }}
        
        .tab:hover {{
            background: #e9ecef;
        }}
        
        .tab.active {{
            color: #667eea;
            border-bottom-color: #667eea;
            background: white;
        }}
        
        .tab-content {{
            display: none;
            padding: 40px;
        }}
        
        .tab-content.active {{
            display: block;
        }}
        
        .chart-container {{
            margin-bottom: 40px;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}
        
        .stat-value {{
            font-size: 36px;
            font-weight: 700;
            margin-bottom: 10px;
        }}
        
        .stat-label {{
            font-size: 14px;
            opacity: 0.9;
        }}
        
        .image-gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 30px;
            margin-top: 30px;
        }}
        
        .image-card {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        .image-card img {{
            width: 100%;
            border-radius: 8px;
            margin-bottom: 10px;
        }}
        
        .image-card h4 {{
            color: #2c3e50;
            margin-bottom: 10px;
            font-size: 18px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }}
        
        th {{
            background: #667eea;
            color: white;
            font-weight: 600;
        }}
        
        tr:hover {{
            background: #f8f9fa;
        }}
        
        .footer {{
            text-align: center;
            padding: 30px;
            background: #f8f9fa;
            color: #666;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 AI-Enhanced LCA Dashboard</h1>
            <p>Industrial Decarbonisation - Life Cycle Assessment Optimization</p>
        </div>
        
        <div class="tabs">
            <div class="tab active" onclick="showTab('lca')">📊 LCA Results</div>
            <div class="tab" onclick="showTab('grid')">🌍 Grid Carbon Intensity</div>
            <div class="tab" onclick="showTab('sectors')">🏭 Sector Comparison</div>
            <div class="tab" onclick="showTab('charts')">🖼️ Charts & Images</div>
        </div>
        
        <!-- LCA Results Tab -->
        <div id="lca" class="tab-content active">
            <h2 style="color: #2c3e50; margin-bottom: 30px; font-size: 28px;">LCA Results & Improvement Opportunities</h2>
            
            <div class="chart-container">
                <div id="lca-comparison-chart"></div>
            </div>
            
            <div class="chart-container">
                <div id="improvement-chart"></div>
            </div>
            
            <div class="chart-container">
                <div id="breakdown-chart"></div>
            </div>
            
            <h3 style="color: #2c3e50; margin: 30px 0 20px 0;">Improvement Opportunities</h3>
            <table>
                <thead>
                    <tr>
                        <th>Sector</th>
                        <th>Current Emissions</th>
                        <th>Optimized Emissions</th>
                        <th>Reduction (%)</th>
                        <th>Improvement Priority</th>
                        <th>Key Opportunity</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Steel</td>
                        <td>1,800</td>
                        <td>550</td>
                        <td>69.4%</td>
                        <td>High</td>
                        <td>H₂-DRI process optimization</td>
                    </tr>
                    <tr>
                        <td>Cement</td>
                        <td>900</td>
                        <td>575</td>
                        <td>36.1%</td>
                        <td>Medium</td>
                        <td>CCUS integration & material substitution</td>
                    </tr>
                    <tr>
                        <td>Shipping</td>
                        <td>3,000</td>
                        <td>300</td>
                        <td>90.0%</td>
                        <td>High</td>
                        <td>Fuel pathway switching</td>
                    </tr>
                    <tr>
                        <td>Aluminium</td>
                        <td>16,000</td>
                        <td>500</td>
                        <td>96.9%</td>
                        <td>High</td>
                        <td>Electrification & recycling</td>
                    </tr>
                </tbody>
            </table>
        </div>
        
        <!-- Grid Carbon Intensity Tab -->
        <div id="grid" class="tab-content">
            <h2 style="color: #2c3e50; margin-bottom: 30px; font-size: 28px;">Grid Carbon Intensity Analysis</h2>
            
            <div style="margin-bottom: 30px;">
                <label style="display: block; margin-bottom: 10px; font-weight: 600; color: #2c3e50;">Time Period:</label>
                <select id="time-period" onchange="updateGridChart()" style="padding: 10px; width: 200px; border-radius: 4px; border: 1px solid #ddd;">
                    <option value="7d">Last 7 Days</option>
                    <option value="30d" selected>Last 30 Days</option>
                    <option value="1y">Last Year</option>
                    <option value="3y">Last 3 Years</option>
                    <option value="5y">Last 5 Years</option>
                </select>
                
                <label style="display: block; margin: 20px 0 10px 0; font-weight: 600; color: #2c3e50;">Location:</label>
                <select id="location" onchange="updateGridChart()" style="padding: 10px; width: 200px; border-radius: 4px; border: 1px solid #ddd;">
                    <option value="US" selected>United States (US)</option>
                    <option value="EU">European Union (EU)</option>
                    <option value="DE">Germany (DE)</option>
                    <option value="FR">France (FR)</option>
                    <option value="GB">United Kingdom (GB)</option>
                </select>
            </div>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value" id="stat-mean">400</div>
                    <div class="stat-label">Mean (g CO₂eq/kWh)</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="stat-min">250</div>
                    <div class="stat-label">Min (g CO₂eq/kWh)</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="stat-max">550</div>
                    <div class="stat-label">Max (g CO₂eq/kWh)</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="stat-std">75</div>
                    <div class="stat-label">Std Dev</div>
                </div>
            </div>
            
            <div class="chart-container">
                <div id="grid-chart"></div>
            </div>
        </div>
        
        <!-- Sector Comparison Tab -->
        <div id="sectors" class="tab-content">
            <h2 style="color: #2c3e50; margin-bottom: 30px; font-size: 28px;">Sector Comparison & Optimization Potential</h2>
            
            <div class="chart-container">
                <div id="sector-chart"></div>
            </div>
            
            <h3 style="color: #2c3e50; margin: 30px 0 20px 0;">Detailed Comparison</h3>
            <table>
                <thead>
                    <tr>
                        <th>Sector</th>
                        <th>Baseline (t CO₂/t)</th>
                        <th>Optimized (t CO₂/t)</th>
                        <th>Reduction %</th>
                        <th>Improvement Priority</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Steel</td>
                        <td>1.8</td>
                        <td>0.55</td>
                        <td>69.4%</td>
                        <td>High</td>
                    </tr>
                    <tr>
                        <td>Cement</td>
                        <td>0.9</td>
                        <td>0.575</td>
                        <td>36.1%</td>
                        <td>Medium</td>
                    </tr>
                    <tr>
                        <td>Shipping</td>
                        <td>3.0</td>
                        <td>0.3</td>
                        <td>90.0%</td>
                        <td>High</td>
                    </tr>
                    <tr>
                        <td>Aluminium</td>
                        <td>16.0</td>
                        <td>0.5</td>
                        <td>96.9%</td>
                        <td>High</td>
                    </tr>
                </tbody>
            </table>
        </div>
        
        <!-- Charts & Images Tab -->
        <div id="charts" class="tab-content">
            <h2 style="color: #2c3e50; margin-bottom: 30px; font-size: 28px;">Generated Charts & Images</h2>
            
            <div class="image-gallery">
"""
    
    # Add images if available
    if all_images:
        for img in sorted(all_images):
            img_path = plots_dir / img
            if img_path.exists():
                # Convert image to base64
                with open(img_path, 'rb') as f:
                    img_data = base64.b64encode(f.read()).decode('utf-8')
                
                img_title = img.replace('.png', '').replace('_', ' ').title()
                html_content += f"""
                <div class="image-card">
                    <h4>{img_title}</h4>
                    <img src="data:image/png;base64,{img_data}" alt="{img_title}">
                    <p style="color: #999; font-size: 12px; margin-top: 10px;">File: {img}</p>
                </div>
"""
    else:
        html_content += """
                <div style="text-align: center; padding: 40px; color: #666;">
                    <p>No images found. Run the workflow to generate charts.</p>
                    <p style="margin-top: 10px; font-size: 14px; color: #999;">Try: python scripts/run_complete_ml_workflow.py</p>
                </div>
"""
    
    html_content += """
            </div>
        </div>
        
        <div class="footer">
            <p>AI-Enhanced Life Cycle Assessment for Industrial Decarbonisation</p>
            <p style="margin-top: 10px; font-size: 12px;">Powered by ML-Enhanced LCA Engine (PINN, Transformer, GNN)</p>
        </div>
    </div>
    
    <script>
        // Tab switching
        function showTab(tabName) {
            // Hide all tabs
            document.querySelectorAll('.tab-content').forEach(tab => {
                tab.classList.remove('active');
            });
            document.querySelectorAll('.tab').forEach(tab => {
                tab.classList.remove('active');
            });
            
            // Show selected tab
            document.getElementById(tabName).classList.add('active');
            event.target.classList.add('active');
            
            // Update charts if needed
            if (tabName === 'grid') {
                updateGridChart();
            }
        }
        
        // LCA Comparison Chart
        var lcaComparisonData = [
            {
                x: """ + json.dumps(sectors) + """,
                y: """ + json.dumps(baseline) + """,
                name: 'Baseline Emissions',
                type: 'bar',
                marker: {color: '#e74c3c'}
            },
            {
                x: """ + json.dumps(sectors) + """,
                y: """ + json.dumps(optimized) + """,
                name: 'Optimized Emissions',
                type: 'bar',
                marker: {color: '#27ae60'}
            }
        ];
        
        var lcaComparisonLayout = {
            title: {
                text: 'LCA Results: Baseline vs Optimized Emissions',
                font: {size: 24, family: 'Inter, sans-serif', color: '#2c3e50'}
            },
            xaxis: {title: 'Sector'},
            yaxis: {title: 'Emissions (kg CO₂eq/t)'},
            barmode: 'group',
            plot_bgcolor: 'white',
            paper_bgcolor: 'white',
            font: {family: 'Inter, sans-serif', size: 12}
        };
        
        Plotly.newPlot('lca-comparison-chart', lcaComparisonData, lcaComparisonLayout, {
            displayModeBar: true,
            displaylogo: false,
            modeBarButtonsToRemove: ['sendDataToCloud', 'lasso2d', 'select2d']
        });
        
        // Improvement Chart
        var improvementData = [{
            x: """ + json.dumps(sectors) + """,
            y: """ + json.dumps(reduction) + """,
            type: 'bar',
            marker: {
                color: """ + json.dumps(['#27ae60' if r > 50 else '#f39c12' if r > 25 else '#e74c3c' for r in reduction]) + """
            }
        }];
        
        var improvementLayout = {
            title: {
                text: 'Improvement Potential by Sector',
                font: {size: 24, family: 'Inter, sans-serif', color: '#2c3e50'}
            },
            xaxis: {title: 'Sector'},
            yaxis: {title: 'Emission Reduction Potential (%)'},
            plot_bgcolor: 'white',
            paper_bgcolor: 'white',
            font: {family: 'Inter, sans-serif', size: 12}
        };
        
        Plotly.newPlot('improvement-chart', improvementData, improvementLayout, {
            displayModeBar: true,
            displaylogo: false,
            modeBarButtonsToRemove: ['sendDataToCloud', 'lasso2d', 'select2d']
        });
        
        // Breakdown Chart
        var breakdownData = [
            {
                x: """ + json.dumps(sectors) + """,
                y: [200, 150, 100, 200],
                name: 'Upstream',
                type: 'bar',
                marker: {color: '#3498db'}
            },
            {
                x: """ + json.dumps(sectors) + """,
                y: [300, 350, 150, 250],
                name: 'Process',
                type: 'bar',
                marker: {color: '#9b59b6'}
            },
            {
                x: """ + json.dumps(sectors) + """,
                y: [50, 75, 50, 50],
                name: 'Downstream',
                type: 'bar',
                marker: {color: '#e67e22'}
            }
        ];
        
        var breakdownLayout = {
            title: {
                text: 'Emission Breakdown by Category',
                font: {size: 24, family: 'Inter, sans-serif', color: '#2c3e50'}
            },
            xaxis: {title: 'Sector'},
            yaxis: {title: 'Emissions (kg CO₂eq/t)'},
            barmode: 'stack',
            plot_bgcolor: 'white',
            paper_bgcolor: 'white',
            font: {family: 'Inter, sans-serif', size: 12}
        };
        
        Plotly.newPlot('breakdown-chart', breakdownData, breakdownLayout, {
            displayModeBar: true,
            displaylogo: false,
            modeBarButtonsToRemove: ['sendDataToCloud', 'lasso2d', 'select2d']
        });
        
        // Sector Comparison Chart
        var sectorData = [
            {
                x: """ + json.dumps(sectors) + """,
                y: """ + json.dumps(baseline) + """,
                name: 'Baseline',
                type: 'bar',
                marker: {color: '#e74c3c'}
            },
            {
                x: """ + json.dumps(sectors) + """,
                y: """ + json.dumps(optimized) + """,
                name: 'Optimized',
                type: 'bar',
                marker: {color: '#27ae60'}
            }
        ];
        
        var sectorLayout = {
            title: {
                text: 'Sector Comparison: Baseline vs Optimized Emissions',
                font: {size: 24, family: 'Inter, sans-serif', color: '#2c3e50'}
            },
            barmode: 'group',
            xaxis: {title: 'Sector'},
            yaxis: {title: 'Emissions (t CO₂/t)'},
            plot_bgcolor: 'white',
            paper_bgcolor: 'white',
            font: {family: 'Inter, sans-serif', size: 12}
        };
        
        Plotly.newPlot('sector-chart', sectorData, sectorLayout, {
            displayModeBar: true,
            displaylogo: false,
            modeBarButtonsToRemove: ['sendDataToCloud', 'lasso2d', 'select2d']
        });
        
        // Grid Chart (sample data)
        function updateGridChart() {
            var timePeriod = document.getElementById('time-period').value;
            var location = document.getElementById('location').value;
            
            // Generate sample time series data
            var hours = [];
            var values = [];
            var numPoints = timePeriod === '7d' ? 168 : timePeriod === '30d' ? 720 : timePeriod === '1y' ? 365 : 36;
            
            for (var i = 0; i < numPoints; i++) {
                hours.push(new Date(Date.now() - (numPoints - i) * 3600000).toISOString());
                values.push(300 + Math.random() * 200 + 50 * Math.sin(i * 0.1));
            }
            
            var mean = values.reduce((a, b) => a + b, 0) / values.length;
            var min = Math.min(...values);
            var max = Math.max(...values);
            var std = Math.sqrt(values.reduce((sq, n) => sq + Math.pow(n - mean, 2), 0) / values.length);
            
            document.getElementById('stat-mean').textContent = Math.round(mean);
            document.getElementById('stat-min').textContent = Math.round(min);
            document.getElementById('stat-max').textContent = Math.round(max);
            document.getElementById('stat-std').textContent = Math.round(std);
            
            var gridData = [{
                x: hours,
                y: values,
                type: 'scatter',
                mode: 'lines',
                name: 'Carbon Intensity',
                line: {color: '#667eea', width: 2},
                fill: 'tozeroy',
                fillcolor: 'rgba(102, 126, 234, 0.1)'
            }];
            
            var gridLayout = {
                title: {
                    text: 'Grid Carbon Intensity - ' + location + ' (' + timePeriod.toUpperCase() + ')',
                    font: {size: 24, family: 'Inter, sans-serif', color: '#2c3e50'}
                },
                xaxis: {title: 'Time'},
                yaxis: {title: 'Carbon Intensity (g CO₂eq/kWh)'},
                plot_bgcolor: 'white',
                paper_bgcolor: 'white',
                font: {family: 'Inter, sans-serif', size: 12},
                shapes: [{
                    type: 'line',
                    x0: hours[0],
                    y0: mean,
                    x1: hours[hours.length - 1],
                    y1: mean,
                    line: {
                        color: '#e74c3c',
                        width: 2,
                        dash: 'dash'
                    }
                }],
                annotations: [{
                    x: hours[hours.length - 1],
                    y: mean,
                    text: 'Mean: ' + Math.round(mean) + ' g CO₂eq/kWh',
                    showarrow: true,
                    arrowhead: 2,
                    ax: 0,
                    ay: -40
                }]
            };
            
            Plotly.newPlot('grid-chart', gridData, gridLayout, {
                displayModeBar: true,
                displaylogo: false,
                modeBarButtonsToRemove: ['sendDataToCloud', 'lasso2d', 'select2d']
            });
        }
        
        // Initialize grid chart
        updateGridChart();
    </script>
</body>
</html>
"""
    
    # Write HTML file
    output_file = output_path / "index.html"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"✅ Static dashboard generated: {output_file}")
    logger.info(f"📊 Charts included: {len(all_images)}")
    
    return output_file


if __name__ == "__main__":
    generate_static_dashboard()

