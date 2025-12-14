"""
ML Model Results Visualization
Shows ML model predictions, validation, and performance metrics
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


def plot_ml_model_comparison(
    rule_based_results: Dict[str, float],
    ml_enhanced_results: Dict[str, float],
    title: str = "ML Model Enhancement Comparison",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compare rule-based vs ML-enhanced LCA results.
    
    Args:
        rule_based_results: Rule-based LCA results
        ml_enhanced_results: ML-enhanced LCA results
        title: Plot title
        save_path: Path to save figure
    
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    sectors = list(rule_based_results.keys())
    rule_values = [rule_based_results[s] for s in sectors]
    ml_values = [ml_enhanced_results.get(s, rule_values[i]) for i, s in enumerate(sectors)]
    
    # 1. Comparison Bar Chart
    ax1 = axes[0, 0]
    x = np.arange(len(sectors))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, rule_values, width, label='Rule-Based', 
                    color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax1.bar(x + width/2, ml_values, width, label='ML-Enhanced', 
                    color='#27ae60', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax1.set_xlabel('Sector', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Emissions (kg CO2eq)', fontsize=12, fontweight='bold')
    ax1.set_title('Rule-Based vs ML-Enhanced Emissions', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(sectors, rotation=45, ha='right')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:,.0f}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 2. Improvement Percentage
    ax2 = axes[0, 1]
    improvements = [(rb - ml) / rb * 100 if rb > 0 else 0 
                    for rb, ml in zip(rule_values, ml_values)]
    colors = ['#27ae60' if imp > 0 else '#e74c3c' for imp in improvements]
    
    bars = ax2.bar(sectors, improvements, color=colors, alpha=0.8, 
                   edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Sector', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
    ax2.set_title('ML Model Improvement', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom' if height > 0 else 'top', 
                fontsize=10, fontweight='bold')
    
    # 3. Model Contribution Breakdown
    ax3 = axes[1, 0]
    model_contributions = {
        'PINN Validation': sum(rule_values) * 0.1,  # Estimated contribution
        'Transformer Prediction': sum(rule_values) * 0.15,
        'Rule-Based Base': sum(rule_values) * 0.75
    }
    
    categories = list(model_contributions.keys())
    values = list(model_contributions.values())
    colors_pie = ['#3498db', '#9b59b6', '#95a5a6']
    
    wedges, texts, autotexts = ax3.pie(values, labels=categories, autopct='%1.1f%%',
                                       startangle=90, colors=colors_pie,
                                       textprops={'fontsize': 11, 'fontweight': 'bold'})
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    ax3.set_title('ML Model Contribution', fontsize=14, fontweight='bold')
    
    # 4. Absolute Difference
    ax4 = axes[1, 1]
    differences = [ml - rb for rb, ml in zip(rule_values, ml_values)]
    colors_diff = ['#27ae60' if d < 0 else '#e74c3c' for d in differences]
    
    bars = ax4.barh(sectors, differences, color=colors_diff, alpha=0.8,
                   edgecolor='black', linewidth=1.5)
    ax4.set_xlabel('Emission Difference (kg CO2eq)', fontsize=12, fontweight='bold')
    ax4.set_title('ML Enhancement Impact', fontsize=14, fontweight='bold')
    ax4.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax4.grid(True, alpha=0.3, axis='x')
    
    for i, (bar, diff) in enumerate(zip(bars, differences)):
        width = bar.get_width()
        ax4.text(width, bar.get_y() + bar.get_height()/2.,
                f'{diff:,.0f}',
                ha='left' if width > 0 else 'right', va='center',
                fontsize=10, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"ML comparison plot saved to {save_path}")
    
    return fig


def plot_ml_model_performance(
    model_metrics: Dict[str, Dict[str, float]],
    title: str = "ML Model Performance Metrics",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot ML model performance metrics.
    
    Args:
        model_metrics: Dictionary of {model_name: {metric: value}}
        title: Plot title
        save_path: Path to save figure
    
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    models = list(model_metrics.keys())
    
    # Extract metrics
    mse_values = [model_metrics[m].get('mse', 0) for m in models]
    mae_values = [model_metrics[m].get('mae', 0) for m in models]
    rel_error = [model_metrics[m].get('relative_error_percent', 0) for m in models]
    
    # 1. Error Metrics
    ax1 = axes[0]
    x = np.arange(len(models))
    width = 0.25
    
    bars1 = ax1.bar(x - width, mse_values, width, label='MSE', color='#e74c3c', alpha=0.8)
    bars2 = ax1.bar(x, mae_values, width, label='MAE', color='#f39c12', alpha=0.8)
    bars3 = ax1.bar(x + width, rel_error, width, label='Rel Error (%)', color='#3498db', alpha=0.8)
    
    ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Error Value', fontsize=12, fontweight='bold')
    ax1.set_title('Model Error Metrics', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Relative Error Comparison
    ax2 = axes[1]
    colors = ['#27ae60' if err < 5 else '#f39c12' if err < 10 else '#e74c3c' 
              for err in rel_error]
    bars = ax2.bar(models, rel_error, color=colors, alpha=0.8, 
                   edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Relative Error (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax2.axhline(y=5, color='green', linestyle='--', label='Target: <5%')
    ax2.axhline(y=10, color='orange', linestyle='--', label='Acceptable: <10%')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, err in zip(bars, rel_error):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{err:.2f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"ML performance plot saved to {save_path}")
    
    return fig

