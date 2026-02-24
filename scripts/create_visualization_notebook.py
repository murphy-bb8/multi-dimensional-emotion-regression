#!/usr/bin/env python3
"""
Create Jupyter notebook for results visualization and analysis.

This script generates a comprehensive Jupyter notebook that provides
interactive visualization and analysis tools for experimental results.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List


class VisualizationNotebookGenerator:
    """Generate Jupyter notebooks for results visualization."""
    
    def __init__(self, results_dir: str):
        """
        Initialize notebook generator.
        
        Args:
            results_dir: Directory containing experimental results
        """
        self.results_dir = Path(results_dir)
        self.notebook_cells = []
    
    def generate_notebook(self, output_file: str) -> str:
        """
        Generate complete Jupyter notebook.
        
        Args:
            output_file: Output notebook file path
            
        Returns:
            Generated notebook JSON string
        """
        # Build notebook structure
        self.add_header()
        self.add_imports()
        self.add_configuration()
        self.add_data_loading()
        self.add_performance_comparison()
        self.add_per_dimension_analysis()
        self.add_visualization_gallery()
        self.add_statistical_analysis()
        self.add_interactive_tools()
        self.add_export_tools()
        
        # Create notebook structure
        notebook = {
            "cells": self.notebook_cells,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "codemirror_mode": {
                        "name": "ipython",
                        "version": 3
                    },
                    "file_extension": ".py",
                    "mimetype": "text/x-python",
                    "name": "python",
                    "nbconvert_exporter": "python",
                    "pygments_lexer": "ipython3",
                    "version": "3.8.0"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }
        
        # Save notebook
        with open(output_file, 'w') as f:
            json.dump(notebook, f, indent=2)
        
        return json.dumps(notebook, indent=2)
    
    def add_header(self):
        """Add notebook header and introduction."""
        self.notebook_cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Multi-dimensional Emotion Regression - Results Visualization\n",
                "\n",
                "This interactive notebook provides comprehensive visualization and analysis tools for the experimental results of multi-dimensional emotion regression models.\n",
                "\n",
                "## Contents\n",
                "1. [Data Loading](#data-loading)\n",
                "2. [Performance Comparison](#performance-comparison)\n",
                "3. [Per-Dimension Analysis](#per-dimension-analysis)\n",
                "4. [Visualization Gallery](#visualization-gallery)\n",
                "5. [Statistical Analysis](#statistical-analysis)\n",
                "6. [Interactive Tools](#interactive-tools)\n",
                "7. [Export Tools](#export-tools)\n",
                "\n",
                "**Generated on:** " + str(Path.cwd()) + "\n",
                "**Results Directory:** `" + str(self.results_dir) + "`"
            ]
        })
    
    def add_imports(self):
        """Add import statements."""
        self.notebook_cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Core imports\n",
                "import os\n",
                "import json\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "from pathlib import Path\n",
                "from typing import Dict, List, Any, Optional\n",
                "import warnings\n",
                "warnings.filterwarnings('ignore')\n",
                "\n",
                "# Interactive widgets\n",
                "import ipywidgets as widgets\n",
                "from ipywidgets import interact, interactive, fixed\n",
                "import plotly.graph_objects as go\n",
                "import plotly.express as px\n",
                "from plotly.subplots import make_subplots\n",
                "\n",
                "# Statistical analysis\n",
                "from scipy import stats\n",
                "from sklearn.metrics import mean_squared_error, mean_absolute_error\n",
                "\n",
                "# Set style\n",
                "plt.style.use('seaborn-v0_8')\n",
                "sns.set_palette(\"husl\")\n",
                "plt.rcParams['figure.figsize'] = (12, 8)\n",
                "plt.rcParams['font.size'] = 12\n",
                "\n",
                "print(\"✅ All imports loaded successfully!\")"
            ]
        })
    
    def add_configuration(self):
        """Add configuration cell."""
        self.notebook_cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Configuration\n",
                "RESULTS_DIR = Path('" + str(self.results_dir) + "')\n",
                "EXPERIMENTS = []\n",
                "METRICS_DATA = {}\n",
                "PREDICTIONS_DATA = {}\n",
                "\n",
                "# Load available experiments\n",
                "for exp_dir in RESULTS_DIR.iterdir():\n",
                "    if exp_dir.is_dir():\n",
                "        exp_name = exp_dir.name\n",
                "        EXPERIMENTS.append(exp_name)\n",
                "        \n",
                "        # Load metrics if available\n",
                "        metrics_file = exp_dir / 'evaluation_results.json'\n",
                "        if metrics_file.exists():\n",
                "            with open(metrics_file, 'r') as f:\n",
                "                METRICS_DATA[exp_name] = json.load(f)\n",
                "        \n",
                "        # Load predictions if available\n",
                "        pred_file = exp_dir / 'predictions.npy'\n",
                "        if pred_file.exists():\n",
                "            PREDICTIONS_DATA[exp_name] = {\n",
                "                'predictions': np.load(pred_file),\n",
                "                'targets': np.load(exp_dir / 'targets.npy')\n",
                "            }\n",
                "\n",
                "print(f\"📊 Found {len(EXPERIMENTS)} experiments:\")\n",
                "for exp in EXPERIMENTS:\n",
                "    status = \"✅\" if exp in METRICS_DATA else \"❌\"\n",
                "    print(f\"  {status} {exp}\")"
            ]
        })
    
    def add_data_loading(self):
        """Add data loading and preparation cells."""
        self.notebook_cells.extend([
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Data Loading\n",
                    "\n",
                    "Load and prepare experimental results for analysis."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Create comprehensive results dataframe\n",
                    "results_data = []\n",
                    "\n",
                    "for exp_name, metrics in METRICS_DATA.items():\n",
                    "    row = {'Experiment': exp_name}\n",
                    "    \n",
                    "    # Overall metrics\n",
                    "    if 'overall_metrics' in metrics:\n",
                    "        overall = metrics['overall_metrics']\n",
                    "        row.update({\n",
                    "            'RMSE': overall.get('rmse', np.nan),\n",
                    "            'MAE': overall.get('mae', np.nan),\n",
                    "            'Pearson': overall.get('pearson', np.nan),\n",
                    "            'Spearman': overall.get('spearman', np.nan),\n",
                    "            'R²': overall.get('r2_score', np.nan)\n",
                    "        })\n",
                    "    \n",
                    "    # Model info\n",
                    "    if 'model_info' in metrics:\n",
                    "        info = metrics['model_info']\n",
                    "        row.update({\n",
                    "            'Model Type': info.get('model_type', 'Unknown'),\n",
                    "            'Samples': info.get('n_samples', 0),\n",
                    "            'Dimensions': info.get('n_dimensions', 0)\n",
                    "        })\n",
                    "    \n",
                    "    results_data.append(row)\n",
                    "\n",
                    "# Create dataframe\n",
                    "results_df = pd.DataFrame(results_data)\n",
                    "print(\"📋 Results DataFrame:\")\n",
                    "display(results_df.round(6))"
                ]
            }
        ])
    
    def add_performance_comparison(self):
        """Add performance comparison visualizations."""
        self.notebook_cells.extend([
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Performance Comparison\n",
                    "\n",
                    "Compare performance metrics across all experiments."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Performance metrics comparison\n",
                    "fig, axes = plt.subplots(2, 2, figsize=(16, 12))\n",
                    "fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')\n",
                    "\n",
                    "# RMSE comparison\n",
                    "ax1 = axes[0, 0]\n",
                    "if 'RMSE' in results_df.columns:\n",
                    "    bars = ax1.bar(results_df['Experiment'], results_df['RMSE'])\n",
                    "    ax1.set_title('RMSE (Lower is Better)')\n",
                    "    ax1.set_ylabel('RMSE')\n",
                    "    ax1.tick_params(axis='x', rotation=45)\n",
                    "    # Add value labels on bars\n",
                    "    for bar, value in zip(bars, results_df['RMSE']):\n",
                    "        if not np.isnan(value):\n",
                    "            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,\n",
                    "                    f'{value:.4f}', ha='center', va='bottom')\n",
                    "\n",
                    "# MAE comparison\n",
                    "ax2 = axes[0, 1]\n",
                    "if 'MAE' in results_df.columns:\n",
                    "    bars = ax2.bar(results_df['Experiment'], results_df['MAE'])\n",
                    "    ax2.set_title('MAE (Lower is Better)')\n",
                    "    ax2.set_ylabel('MAE')\n",
                    "    ax2.tick_params(axis='x', rotation=45)\n",
                    "    for bar, value in zip(bars, results_df['MAE']):\n",
                    "        if not np.isnan(value):\n",
                    "            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,\n",
                    "                    f'{value:.4f}', ha='center', va='bottom')\n",
                    "\n",
                    "# Pearson correlation comparison\n",
                    "ax3 = axes[1, 0]\n",
                    "if 'Pearson' in results_df.columns:\n",
                    "    bars = ax3.bar(results_df['Experiment'], results_df['Pearson'])\n",
                    "    ax3.set_title('Pearson Correlation (Higher is Better)')\n",
                    "    ax3.set_ylabel('Pearson r')\n",
                    "    ax3.tick_params(axis='x', rotation=45)\n",
                    "    for bar, value in zip(bars, results_df['Pearson']):\n",
                    "        if not np.isnan(value):\n",
                    "            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,\n",
                    "                    f'{value:.3f}', ha='center', va='bottom')\n",
                    "\n",
                    "# R² score comparison\n",
                    "ax4 = axes[1, 1]\n",
                    "if 'R²' in results_df.columns:\n",
                    "    bars = ax4.bar(results_df['Experiment'], results_df['R²'])\n",
                    "    ax4.set_title('R² Score (Higher is Better)')\n",
                    "    ax4.set_ylabel('R²')\n",
                    "    ax4.tick_params(axis='x', rotation=45)\n",
                    "    for bar, value in zip(bars, results_df['R²']):\n",
                    "        if not np.isnan(value):\n",
                    "            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,\n",
                    "                    f'{value:.3f}', ha='center', va='bottom')\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.show()\n",
                    "\n",
                    "# Print best performing models\n",
                    "print(\"🏆 Best Performing Models:\")\n",
                    "if 'RMSE' in results_df.columns:\n",
                    "    best_rmse = results_df.loc[results_df['RMSE'].idxmin()]\n",
                    "    print(f\"  Best RMSE: {best_rmse['Experiment']} ({best_rmse['RMSE']:.6f})\")\n",
                    "if 'Pearson' in results_df.columns:\n",
                    "    best_pearson = results_df.loc[results_df['Pearson'].idxmax()]\n",
                    "    print(f\"  Best Pearson: {best_pearson['Experiment']} ({best_pearson['Pearson']:.6f})\")"
                ]
            }
        ])
    
    def add_per_dimension_analysis(self):
        """Add per-dimension analysis cells."""
        self.notebook_cells.extend([
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Per-Dimension Analysis\n",
                    "\n",
                    "Analyze performance across different emotion dimensions."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Per-dimension performance analysis\n",
                    "def plot_per_dimension_metrics():\n",
                    "    \"\"\"Create per-dimension performance visualization.\"\"\"\n",
                    "    if not METRICS_DATA:\n",
                    "        print(\"No metrics data available\")\n",
                    "        return\n",
                    "    \n",
                    "    # Collect per-dimension metrics\n",
                    "    per_dim_data = []\n",
                    "    \n",
                    "    for exp_name, metrics in METRICS_DATA.items():\n",
                    "        if 'per_dimension_metrics' in metrics:\n",
                    "            per_dim = metrics['per_dimension_metrics']\n",
                    "            \n",
                    "            if 'rmse_per_dim' in per_dim:\n",
                    "                for dim, rmse in enumerate(per_dim['rmse_per_dim']):\n",
                    "                    per_dim_data.append({\n",
                    "                        'Experiment': exp_name,\n",
                    "                        'Dimension': f'Dim {dim}',\n",
                    "                        'RMSE': rmse\n",
                    "                    })\n",
                    "    \n",
                    "    if not per_dim_data:\n",
                    "        print(\"No per-dimension metrics available\")\n",
                    "        return\n",
                    "    \n",
                    "    per_dim_df = pd.DataFrame(per_dim_data)\n",
                    "    \n",
                    "    # Create heatmap\n",
                    "    pivot_df = per_dim_df.pivot(index='Experiment', columns='Dimension', values='RMSE')\n",
                    "    \n",
                    "    plt.figure(figsize=(14, 8))\n",
                    "    sns.heatmap(pivot_df, annot=True, fmt='.4f', cmap='RdYlBu_r',\n",
                    "               cbar_kws={'label': 'RMSE'})\n",
                    "    plt.title('Per-Dimension RMSE Heatmap')\n",
                    "    plt.tight_layout()\n",
                    "    plt.show()\n",
                    "    \n",
                    "    # Bar plot comparison\n",
                    "    plt.figure(figsize=(16, 8))\n",
                    "    sns.barplot(data=per_dim_df, x='Dimension', y='RMSE', hue='Experiment')\n",
                    "    plt.title('Per-Dimension RMSE Comparison')\n",
                    "    plt.ylabel('RMSE')\n",
                    "    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')\n",
                    "    plt.tight_layout()\n",
                    "    plt.show()\n",
                    "\n",
                    "plot_per_dimension_metrics()"
                ]
            }
        ])
    
    def add_visualization_gallery(self):
        """Add visualization gallery cells."""
        self.notebook_cells.extend([
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Visualization Gallery\n",
                    "\n",
                    "Interactive visualizations for detailed analysis."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Interactive prediction vs true visualization\n",
                    "@interact\n",
                    "def plot_predictions(experiment=EXPERIMENTS if EXPERIMENTS else ['None']):\n",
                    "    \"\"\"Interactive prediction vs true values plot.\"\"\"\n",
                    "    if experiment not in PREDICTIONS_DATA:\n",
                    "        print(f\"No prediction data available for {experiment}\")\n",
                    "        return\n",
                    "    \n",
                    "    data = PREDICTIONS_DATA[experiment]\n",
                    "    y_true = data['targets']\n",
                    "    y_pred = data['predictions']\n",
                    "    \n",
                    "    # Overall plot\n",
                    "    fig, axes = plt.subplots(2, 4, figsize=(20, 10))\n",
                    "    axes = axes.flatten()\n",
                    "    \n",
                    "    for dim in range(min(8, y_true.shape[1])):\n",
                    "        ax = axes[dim]\n",
                    "        ax.scatter(y_true[:, dim], y_pred[:, dim], alpha=0.6, s=20)\n",
                    "        \n",
                    "        # Perfect prediction line\n",
                    "        min_val = min(y_true[:, dim].min(), y_pred[:, dim].min())\n",
                    "        max_val = max(y_true[:, dim].max(), y_pred[:, dim].max())\n",
                    "        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)\n",
                    "        \n",
                    "        # Calculate metrics\n",
                    "        rmse = np.sqrt(mean_squared_error(y_true[:, dim], y_pred[:, dim]))\n",
                    "        pearson = np.corrcoef(y_true[:, dim], y_pred[:, dim])[0, 1]\n",
                    "        \n",
                    "        ax.set_xlabel('True')\n",
                    "        ax.set_ylabel('Predicted')\n",
                    "        ax.set_title(f'Dim {dim}\\nRMSE: {rmse:.4f}\\nρ: {pearson:.3f}')\n",
                    "        ax.grid(True, alpha=0.3)\n",
                    "    \n",
                    "    plt.suptitle(f'Prediction vs True Values - {experiment}', fontsize=16)\n",
                    "    plt.tight_layout()\n",
                    "    plt.show()"
                ]
            }
        ])
    
    def add_statistical_analysis(self):
        """Add statistical analysis cells."""
        self.notebook_cells.extend([
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Statistical Analysis\n",
                    "\n",
                    "Statistical tests and confidence intervals."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Statistical significance testing\n",
                    "def perform_statistical_tests():\n",
                    "    \"\"\"Perform statistical tests between models.\"\"\"\n",
                    "    if len(PREDICTIONS_DATA) < 2:\n",
                    "        print(\"Need at least 2 models for statistical comparison\")\n",
                    "        return\n",
                    "    \n",
                    "    experiments = list(PREDICTIONS_DATA.keys())\n",
                    "    \n",
                    "    # Compare first two models\n",
                    "    exp1, exp2 = experiments[0], experiments[1]\n",
                    "    \n",
                    "    y_true1 = PREDICTIONS_DATA[exp1]['targets']\n",
                    "    y_pred1 = PREDICTIONS_DATA[exp1]['predictions']\n",
                    "    y_true2 = PREDICTIONS_DATA[exp2]['targets']\n",
                    "    y_pred2 = PREDICTIONS_DATA[exp2]['predictions']\n",
                    "    \n",
                    "    # Calculate errors\n",
                    "    errors1 = np.abs(y_pred1 - y_true1)\n",
                    "    errors2 = np.abs(y_pred2 - y_true2)\n",
                    "    \n",
                    "    # Paired t-test\n",
                    "    t_stat, p_value = stats.ttest_rel(errors1.flatten(), errors2.flatten())\n",
                    "    \n",
                    "    # Wilcoxon signed-rank test\n",
                    "    wilcoxon_stat, wilcoxon_p = stats.wilcoxon(errors1.flatten(), errors2.flatten())\n",
                    "    \n",
                    "    print(f\"📊 Statistical Comparison: {exp1} vs {exp2}\")\n",
                    "    print(f\"  Paired t-test: t={t_stat:.4f}, p={p_value:.6f}\")\n",
                    "    print(f\"  Wilcoxon test: W={wilcoxon_stat:.4f}, p={wilcoxon_p:.6f}\")\n",
                    "    print(f\"  Significant difference (α=0.05): {'Yes' if p_value < 0.05 else 'No'}\")\n",
                    "    \n",
                    "    # Effect size (Cohen's d)\n",
                    "    pooled_std = np.sqrt(((len(errors1) - 1) * np.var(errors1.flatten()) + \n",
                    "                          (len(errors2) - 1) * np.var(errors2.flatten())) / \n",
                    "                         (len(errors1) + len(errors2) - 2))\n",
                    "    cohens_d = (np.mean(errors1.flatten()) - np.mean(errors2.flatten())) / pooled_std\n",
                    "    print(f\"  Effect size (Cohen's d): {cohens_d:.4f}\")\n",
                    "\n",
                    "perform_statistical_tests()"
                ]
            }
        ])
    
    def add_interactive_tools(self):
        """Add interactive analysis tools."""
        self.notebook_cells.extend([
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Interactive Tools\n",
                    "\n",
                    "Interactive widgets for custom analysis."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Interactive error analysis\n",
                    "@interact\n",
                    "def analyze_errors(\n",
                    "    experiment=EXPERIMENTS if EXPERIMENTS else ['None'],\n",
                    "    dimension=widgets.IntSlider(min=0, max=7, value=0, description='Dimension:')\n",
                    "):\n",
                    "    \"\"\"Interactive error analysis for specific experiment and dimension.\"\"\"\n",
                    "    if experiment not in PREDICTIONS_DATA:\n",
                    "        print(f\"No data available for {experiment}\")\n",
                    "        return\n",
                    "    \n",
                    "    data = PREDICTIONS_DATA[experiment]\n",
                    "    y_true = data['targets'][:, dimension]\n",
                    "    y_pred = data['predictions'][:, dimension]\n",
                    "    \n",
                    "    errors = y_pred - y_true\n",
                    "    abs_errors = np.abs(errors)\n",
                    "    \n",
                    "    # Create subplots\n",
                    "    fig, axes = plt.subplots(2, 2, figsize=(15, 10))\n",
                    "    \n",
                    "    # Error distribution\n",
                    "    axes[0, 0].hist(errors, bins=30, alpha=0.7, edgecolor='black')\n",
                    "    axes[0, 0].set_title('Error Distribution')\n",
                    "    axes[0, 0].set_xlabel('Prediction - True')\n",
                    "    axes[0, 0].set_ylabel('Frequency')\n",
                    "    axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=2)\n",
                    "    \n",
                    "    # Absolute error distribution\n",
                    "    axes[0, 1].hist(abs_errors, bins=30, alpha=0.7, edgecolor='black')\n",
                    "    axes[0, 1].set_title('Absolute Error Distribution')\n",
                    "    axes[0, 1].set_xlabel('|Prediction - True|')\n",
                    "    axes[0, 1].set_ylabel('Frequency')\n",
                    "    \n",
                    "    # Error vs True value\n",
                    "    axes[1, 0].scatter(y_true, errors, alpha=0.6)\n",
                    "    axes[1, 0].axhline(0, color='red', linestyle='--', linewidth=2)\n",
                    "    axes[1, 0].set_title('Error vs True Value')\n",
                    "    axes[1, 0].set_xlabel('True Value')\n",
                    "    axes[1, 0].set_ylabel('Error')\n",
                    "    \n",
                    "    # Error vs Predicted value\n",
                    "    axes[1, 1].scatter(y_pred, errors, alpha=0.6)\n",
                    "    axes[1, 1].axhline(0, color='red', linestyle='--', linewidth=2)\n",
                    "    axes[1, 1].set_title('Error vs Predicted Value')\n",
                    "    axes[1, 1].set_xlabel('Predicted Value')\n",
                    "    axes[1, 1].set_ylabel('Error')\n",
                    "    \n",
                    "    plt.suptitle(f'Error Analysis - {experiment}, Dimension {dimension}', fontsize=14)\n",
                    "    plt.tight_layout()\n",
                    "    plt.show()\n",
                    "    \n",
                    "    # Print statistics\n",
                    "    print(f\"📈 Error Statistics for {experiment}, Dimension {dimension}:\")\n",
                    "    print(f\"  Mean Error: {np.mean(errors):.6f}\")\n",
                    "    print(f\"  Std Error: {np.std(errors):.6f}\")\n",
                    "    print(f\"  Mean Absolute Error: {np.mean(abs_errors):.6f}\")\n",
                    "    print(f\"  Max Error: {np.max(abs_errors):.6f}\")\n",
                    "    print(f\"  RMSE: {np.sqrt(np.mean(errors**2)):.6f}\")"
                ]
            }
        ])
    
    def add_export_tools(self):
        """Add export and reporting tools."""
        self.notebook_cells.extend([
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Export Tools\n",
                    "\n",
                    "Export results and generate reports."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Export results to CSV\n",
                    "def export_results():\n",
                    "    \"\"\"Export all results to CSV files.\"\"\"\n",
                    "    export_dir = RESULTS_DIR / 'exports'\n",
                    "    export_dir.mkdir(exist_ok=True)\n",
                    "    \n",
                    "    # Export main results\n",
                    "    if not results_df.empty:\n",
                    "        results_df.to_csv(export_dir / 'performance_summary.csv', index=False)\n",
                    "        print(\"✅ Performance summary exported\")\n",
                    "    \n",
                    "    # Export per-dimension metrics\n",
                    "    per_dim_data = []\n",
                    "    for exp_name, metrics in METRICS_DATA.items():\n",
                    "        if 'per_dimension_metrics' in metrics:\n",
                    "            per_dim = metrics['per_dimension_metrics']\n",
                    "            for metric_name, values in per_dim.items():\n",
                    "                if isinstance(values, list):\n",
                    "                    for dim, value in enumerate(values):\n",
                    "                        per_dim_data.append({\n",
                    "                            'Experiment': exp_name,\n",
                    "                            'Dimension': f'Dim {dim}',\n",
                    "                            'Metric': metric_name,\n",
                    "                            'Value': value\n",
                    "                        })\n",
                    "    \n",
                    "    if per_dim_data:\n",
                    "        per_dim_df = pd.DataFrame(per_dim_data)\n",
                    "        per_dim_df.to_csv(export_dir / 'per_dimension_metrics.csv', index=False)\n",
                    "        print(\"✅ Per-dimension metrics exported\")\n",
                    "    \n",
                    "    # Export predictions\n",
                    "    for exp_name, data in PREDICTIONS_DATA.items():\n",
                    "        pred_dir = export_dir / 'predictions' / exp_name\n",
                    "        pred_dir.mkdir(parents=True, exist_ok=True)\n",
                    "        np.save(pred_dir / 'predictions.npy', data['predictions'])\n",
                    "        np.save(pred_dir / 'targets.npy', data['targets'])\n",
                    "    \n",
                    "    print(f\"✅ All results exported to {export_dir}\")\n",
                    "\n",
                    "export_results()\n",
                    "\n",
                    "print(\"\\n🎉 Analysis complete! Use the interactive tools above to explore the results.\")\n",
                    "print(f\"📁 Results directory: {RESULTS_DIR}\")"
                ]
            }
        ])


def main():
    """Main function to generate visualization notebook."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create visualization notebook")
    parser.add_argument("--results_dir", type=str, required=True,
                       help="Directory containing experimental results")
    parser.add_argument("--output_file", type=str, required=True,
                       help="Output notebook file path")
    
    args = parser.parse_args()
    
    # Generate notebook
    generator = VisualizationNotebookGenerator(args.results_dir)
    notebook_json = generator.generate_notebook(args.output_file)
    
    print(f"✅ Visualization notebook generated: {args.output_file}")
    print("📓 Open the notebook in Jupyter to start interactive analysis!")


if __name__ == "__main__":
    main()
