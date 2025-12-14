"""
Plotting utilities for LCA results
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def plot_lca_results(
    results: Dict[str, float],
    title: str = "LCA Results",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot LCA emission breakdown with improved readability.
    
    Args:
        results: Dictionary with emission categories and values
        title: Plot title
        save_path: Path to save figure
    
    Returns:
        Matplotlib figure
    """
    # Separate captured CO2 from emissions for better visualization
    emissions_only = {k: v for k, v in results.items() if k != "captured_co2" and v > 0}
    captured_co2 = results.get("captured_co2", 0)
    
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Bar chart for emissions (excluding captured CO2)
    ax1 = fig.add_subplot(gs[0, 0])
    if emissions_only:
        categories = list(emissions_only.keys())
        values = list(emissions_only.values())
        
        # Improve category names for readability
        category_labels = []
        for cat in categories:
            # Convert snake_case to Title Case with spaces
            label = cat.replace('_', ' ').title()
            # Special formatting for common terms
            label = label.replace('Co2', 'CO₂')
            label = label.replace('Ci', 'CI')
            category_labels.append(label)
        
        # Use better colors and larger fonts
        colors = ['#e74c3c', '#f39c12', '#3498db', '#9b59b6', '#1abc9c']
        bars = ax1.bar(category_labels, values, color=colors[:len(categories)], alpha=0.85, 
                      edgecolor='black', linewidth=2.0)
        
        ax1.set_xlabel("Emission Category", fontsize=13, fontweight='bold', labelpad=10)
        ax1.set_ylabel("Emissions (kg CO₂eq)", fontsize=13, fontweight='bold', labelpad=10)
        ax1.set_title("Emissions Breakdown", fontsize=15, fontweight='bold', pad=20)
        ax1.tick_params(axis='x', rotation=30, labelsize=11)
        ax1.tick_params(axis='y', labelsize=11)
        ax1.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.8)
        
        # Add value labels with better formatting and positioning
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                # Position label above bar with padding
                label_y = height + max(values) * 0.02
                ax1.text(bar.get_x() + bar.get_width()/2., label_y,
                        f'{height:,.0f}',
                        ha='center', va='bottom', fontsize=11, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray', linewidth=0.5))
    else:
        ax1.text(0.5, 0.5, 'No emissions data', ha='center', va='center', transform=ax1.transAxes, fontsize=12)
        ax1.set_title(f"{title} - Emissions Breakdown", fontsize=14, fontweight='bold')
    
    # Pie chart for emissions distribution (excluding captured CO2)
    ax2 = fig.add_subplot(gs[0, 1])
    if emissions_only and sum(emissions_only.values()) > 0:
        categories = list(emissions_only.keys())
        values = list(emissions_only.values())
        colors = ['#e74c3c', '#f39c12', '#3498db', '#9b59b6', '#1abc9c']
        
        # Improve category names for readability
        category_labels = []
        for cat in categories:
            label = cat.replace('_', ' ').title()
            label = label.replace('Co2', 'CO₂')
            label = label.replace('Ci', 'CI')
            category_labels.append(label)
        
        # Only show slices > 1%
        filtered_data = [(cat, val, label) for cat, val, label in zip(categories, values, category_labels) 
                         if val / sum(values) > 0.01]
        if filtered_data:
            cats, vals, labels = zip(*filtered_data)
            wedges, texts, autotexts = ax2.pie(vals, labels=labels, autopct='%1.1f%%', 
                                               startangle=90, colors=colors[:len(cats)],
                                               textprops={'fontsize': 11, 'fontweight': 'bold'},
                                               pctdistance=0.85, labeldistance=1.1)
            # Improve text visibility
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(12)
            for text in texts:
                text.set_fontsize(11)
                text.set_fontweight('bold')
        ax2.set_title("Emissions Distribution", fontsize=15, fontweight='bold', pad=20)
    else:
        ax2.text(0.5, 0.5, 'No emissions data', ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title(f"{title} - Emissions Distribution", fontsize=14, fontweight='bold')
    
    # Captured CO2 visualization
    ax3 = fig.add_subplot(gs[1, :])
    if captured_co2 > 0:
        # Show captured CO2 as a separate metric
        total_emissions = sum(emissions_only.values()) if emissions_only else 0
        total_co2_generated = total_emissions + captured_co2
        net_emissions = total_emissions - captured_co2
        
        categories = ['Total CO₂ Generated', 'Captured CO₂', 'Net Emissions']
        values = [total_co2_generated, captured_co2, net_emissions]
        colors = ['#34495e', '#27ae60', '#e74c3c' if net_emissions >= 0 else '#2ecc71']
        
        bars = ax3.bar(categories, values, color=colors, alpha=0.85, edgecolor='black', linewidth=2.0)
        ax3.set_ylabel("CO₂ (kg CO₂eq)", fontsize=13, fontweight='bold', labelpad=10)
        ax3.set_title("CO₂ Capture Analysis", fontsize=15, fontweight='bold', pad=20)
        ax3.tick_params(axis='x', labelsize=12)
        ax3.tick_params(axis='y', labelsize=11)
        ax3.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.8)
        
        # Add zero line for reference
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=1.0, alpha=0.5)
        
        # Add value labels with better positioning
        max_abs_value = max(abs(v) for v in values)
        for bar in bars:
            height = bar.get_height()
            # Position label above or below bar based on sign
            if height >= 0:
                label_y = height + max_abs_value * 0.02
                va_pos = 'bottom'
            else:
                label_y = height - max_abs_value * 0.02
                va_pos = 'top'
            
            ax3.text(bar.get_x() + bar.get_width()/2., label_y,
                    f'{height:,.0f}',
                    ha='center', va=va_pos, fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, 
                             edgecolor='gray', linewidth=0.5))
        
        # Add capture rate annotation with better styling
        if total_co2_generated > 0:
            capture_rate = (captured_co2 / total_co2_generated) * 100
            reduction_rate = (captured_co2 / total_co2_generated) * 100 if total_co2_generated > 0 else 0
            
            # Create info box with multiple metrics
            info_text = f'Capture Rate: {capture_rate:.1f}%\n'
            if net_emissions < 0:
                info_text += f'Net Reduction: {abs(net_emissions):,.0f} kg CO₂eq'
            else:
                info_text += f'Remaining Emissions: {net_emissions:,.0f} kg CO₂eq'
            
            ax3.text(0.5, 0.97, info_text, 
                    ha='center', va='top', transform=ax3.transAxes,
                    fontsize=12, fontweight='bold', 
                    bbox=dict(boxstyle='round,pad=0.8', facecolor='#ecf0f1', alpha=0.95,
                             edgecolor='#34495e', linewidth=2.0),
                    family='monospace')
    else:
        ax3.text(0.5, 0.5, 'No CO2 capture data', ha='center', va='center', 
                transform=ax3.transAxes, fontsize=12)
        ax3.set_title(f"{title} - CO2 Capture Analysis", fontsize=14, fontweight='bold')
    
    # Improved main title
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.99, 
                family='sans-serif', color='#2c3e50')
    
    # Adjust layout for better spacing - use subplots_adjust instead of tight_layout
    fig.subplots_adjust(top=0.93, bottom=0.08, left=0.08, right=0.95, 
                       hspace=0.35, wspace=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', 
                   edgecolor='none', pad_inches=0.2)
        logger.info(f"Plot saved to {save_path}")
    
    return fig


def plot_time_series_lca(
    data: pd.DataFrame,
    location: str,
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot time series of carbon intensity.
    
    Args:
        data: DataFrame with timestamp and carbon_intensity columns
        location: Location name
        title: Plot title
        save_path: Path to save figure
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    if 'timestamp' in data.columns:
        data = data.set_index('timestamp')
    
    ax.plot(data.index, data['carbon_intensity'], linewidth=2, alpha=0.7)
    ax.fill_between(data.index, data['carbon_intensity'], alpha=0.3)
    ax.set_xlabel("Time")
    ax.set_ylabel("Carbon Intensity (g CO2eq/kWh)")
    ax.set_title(title or f"Grid Carbon Intensity - {location}")
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    mean_ci = data['carbon_intensity'].mean()
    ax.axhline(y=mean_ci, color='r', linestyle='--', 
               label=f'Mean: {mean_ci:.1f} g CO2eq/kWh')
    ax.legend()
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    return fig


def plot_sector_comparison(
    sector_results: Dict[str, Dict[str, float]],
    title: str = "Sector Comparison",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot LCA comparison across sectors.
    
    Args:
        sector_results: Dictionary of {sector: {metric: value}}
        title: Plot title
        save_path: Path to save figure
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sectors = list(sector_results.keys())
    emissions = [sector_results[s].get("total_emissions", 0) for s in sectors]
    
    bars = ax.bar(sectors, emissions, color=sns.color_palette("Set2", len(sectors)))
    ax.set_xlabel("Sector")
    ax.set_ylabel("Total Emissions (kg CO2eq)")
    ax.set_title(title)
    ax.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    return fig

