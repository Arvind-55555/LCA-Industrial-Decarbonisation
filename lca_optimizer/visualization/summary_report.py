"""
Generate comprehensive visualization summary report
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import logging

from lca_optimizer.visualization.plots import (
    plot_lca_results, plot_time_series_lca, plot_sector_comparison
)

logger = logging.getLogger(__name__)


class VisualizationSummary:
    """Generate comprehensive visualization summary"""
    
    def __init__(self, output_dir: str = "outputs"):
        """
        Initialize visualization summary generator.
        
        Args:
            output_dir: Output directory for reports and plots
        """
        self.output_dir = Path(output_dir)
        self.plots_dir = self.output_dir / "plots"
        self.reports_dir = self.output_dir / "reports"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Visualization summary initialized: {self.output_dir}")
    
    def generate_summary_report(
        self,
        lca_results: Dict[str, Dict],
        grid_data: Optional[pd.DataFrame] = None
    ) -> Path:
        """
        Generate comprehensive summary report with all visualizations.
        
        Args:
            lca_results: Dictionary of LCA results by sector
            grid_data: Grid carbon intensity data
        
        Returns:
            Path to generated report
        """
        logger.info("Generating visualization summary report...")
        
        # Generate all plots
        plots_created = []
        
        # 1. LCA Results by Sector
        if lca_results:
            for sector, results in lca_results.items():
                if 'breakdown' in results:
                    fig = plot_lca_results(
                        results['breakdown'],
                        title=f"{sector.title()} LCA Breakdown",
                        save_path=str(self.plots_dir / f"{sector}_lca_breakdown.png")
                    )
                    plots_created.append(f"{sector}_lca_breakdown.png")
        
        # 2. Grid Carbon Intensity Time Series
        if grid_data is not None and not grid_data.empty:
            for location in grid_data.get('location', pd.Series()).unique():
                location_data = grid_data[grid_data['location'] == location]
                if not location_data.empty:
                    fig = plot_time_series_lca(
                        location_data,
                        location=location,
                        title=f"Grid Carbon Intensity - {location}",
                        save_path=str(self.plots_dir / f"grid_ci_{location.lower()}.png")
                    )
                    plots_created.append(f"grid_ci_{location.lower()}.png")
        
        # 4. Sector Comparison
        if lca_results:
            sector_data = {
                sector: {
                    "total_emissions": results.get("total_emissions", 0),
                    "reduction": results.get("emission_reduction", 0)
                }
                for sector, results in lca_results.items()
            }
            
            fig = plot_sector_comparison(
                sector_data,
                title="Sector Comparison: Total Emissions",
                save_path=str(self.plots_dir / "sector_comparison.png")
            )
            plots_created.append("sector_comparison.png")
        
        # Generate HTML report
        report_path = self._generate_html_report(plots_created, lca_results)
        
        logger.info(f"Summary report generated: {report_path}")
        logger.info(f"Plots created: {len(plots_created)}")
        
        return report_path
    
    def _generate_html_report(
        self,
        plots: List[str],
        lca_results: Dict
    ) -> Path:
        """Generate HTML summary report"""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>LCA Optimizer - Visualization Summary</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .section {{
            background: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .plot-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .plot-item {{
            text-align: center;
        }}
        .plot-item img {{
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        .metrics {{
            display: flex;
            justify-content: space-around;
            flex-wrap: wrap;
        }}
        .metric {{
            text-align: center;
            padding: 20px;
            margin: 10px;
            background: #f8f9fa;
            border-radius: 5px;
            min-width: 150px;
        }}
        .metric-value {{
            font-size: 36px;
            font-weight: bold;
            color: #667eea;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>LCA Optimizer - Visualization Summary Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="section">
        <h2>📊 Summary Metrics</h2>
        <div class="metrics">
            <div class="metric">
                <div class="metric-value">{len(lca_results)}</div>
                <div>Sectors Analyzed</div>
            </div>
            <div class="metric">
                <div class="metric-value">{len(plots)}</div>
                <div>Visualizations</div>
            </div>
            <div class="metric">
                <div class="metric-value">100%</div>
                <div>Focus on LCA Improvement</div>
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>📈 Generated Visualizations</h2>
        <div class="plot-grid">
"""
        
        for plot in plots:
            plot_path = self.plots_dir / plot
            if plot_path.exists():
                html_content += f"""
            <div class="plot-item">
                <h3>{plot.replace('.png', '').replace('_', ' ').title()}</h3>
                <img src="plots/{plot}" alt="{plot}">
            </div>
"""
        
        html_content += """
        </div>
    </div>
    
    <div class="section">
        <h2>📋 Results Summary</h2>
        <h3>LCA Results by Sector</h3>
        <table style="width: 100%; border-collapse: collapse;">
            <tr style="background-color: #667eea; color: white;">
                <th style="padding: 10px; text-align: left;">Sector</th>
                <th style="padding: 10px; text-align: right;">Total Emissions (kg CO2eq)</th>
                <th style="padding: 10px; text-align: right;">Reduction (%)</th>
            </tr>
"""
        
        for sector, results in lca_results.items():
            emissions = results.get("total_emissions", 0)
            reduction = results.get("emission_reduction", 0)
            html_content += f"""
            <tr>
                <td style="padding: 10px; border-bottom: 1px solid #ddd;">{sector.title()}</td>
                <td style="padding: 10px; text-align: right; border-bottom: 1px solid #ddd;">{emissions:,.0f}</td>
                <td style="padding: 10px; text-align: right; border-bottom: 1px solid #ddd;">{reduction:.1f}%</td>
            </tr>
"""
        
        html_content += """
        </table>
    </div>
    
    <div class="section">
        <h2>📁 Files Generated</h2>
        <ul>
"""
        
        for plot in plots:
            html_content += f"            <li>{plot}</li>\n"
        
        html_content += """
        </ul>
    </div>
</body>
</html>
"""
        
        report_path = self.reports_dir / "visualization_summary.html"
        report_path.write_text(html_content)
        
        return report_path

