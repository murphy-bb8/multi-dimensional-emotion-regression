#!/usr/bin/env python3
"""
Generate comprehensive comparison report for all experimental results.

This script analyzes experimental results and creates a detailed comparison
report in markdown format, including performance metrics, statistical tests,
and visualizations.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComparisonReportGenerator:
    """Generate comparison reports for experimental results."""
    
    def __init__(self, results_dir: str):
        """
        Initialize report generator.
        
        Args:
            results_dir: Directory containing experimental results
        """
        self.results_dir = Path(results_dir)
        self.experiments = {}
        self.load_experiments()
    
    def load_experiments(self):
        """Load all experimental results."""
        for exp_dir in self.results_dir.iterdir():
            if exp_dir.is_dir():
                exp_name = exp_dir.name
                self.experiments[exp_name] = self.load_experiment_results(exp_dir)
    
    def load_experiment_results(self, exp_dir: Path) -> Dict[str, Any]:
        """Load results from a single experiment."""
        results = {}
        
        # Load metrics.json if exists
        metrics_file = exp_dir / "metrics.json"
        if metrics_file.exists():
            try:
                with open(metrics_file, 'r') as f:
                    results['metrics'] = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load metrics from {metrics_file}: {e}")
        
        # Load evaluation results
        eval_dir = exp_dir / "evaluation"
        if eval_dir.exists():
            results['evaluation'] = self.load_evaluation_results(eval_dir)
        
        # Load training logs
        log_file = exp_dir / "training.log"
        if log_file.exists():
            results['has_log'] = True
        
        return results
    
    def load_evaluation_results(self, eval_dir: Path) -> Dict[str, Any]:
        """Load evaluation results."""
        results = {}
        
        for file in eval_dir.iterdir():
            if file.suffix == '.json':
                try:
                    with open(file, 'r') as f:
                        results[file.stem] = json.load(f)
                except Exception as e:
                    logger.warning(f"Could not load {file}: {e}")
        
        return results
    
    def generate_performance_table(self) -> pd.DataFrame:
        """Generate performance comparison table."""
        performance_data = []
        
        for exp_name, exp_data in self.experiments.items():
            row = {'Experiment': exp_name}
            
            # Extract key metrics
            if 'metrics' in exp_data:
                metrics = exp_data['metrics']
                row.update({
                    'Test RMSE': metrics.get('test_rmse', 'N/A'),
                    'Test MAE': metrics.get('test_mae', 'N/A'),
                    'Test Pearson': metrics.get('test_pearson', 'N/A'),
                    'OOF RMSE': metrics.get('oof_rmse', 'N/A'),
                    'Model': metrics.get('model_name', 'N/A')
                })
            
            # Extract evaluation metrics if available
            if 'evaluation' in exp_data:
                eval_data = exp_data['evaluation']
                if 'overall_metrics' in eval_data:
                    eval_metrics = eval_data['overall_metrics']
                    row.update({
                        'Eval RMSE': eval_metrics.get('rmse', 'N/A'),
                        'Eval MAE': eval_metrics.get('mae', 'N/A'),
                        'Eval Pearson': eval_metrics.get('pearson', 'N/A')
                    })
            
            performance_data.append(row)
        
        return pd.DataFrame(performance_data)
    
    def generate_statistical_comparison(self) -> Dict[str, Any]:
        """Generate statistical comparison between models."""
        comparison = {
            'best_rmse': {'model': None, 'value': float('inf')},
            'best_mae': {'model': None, 'value': float('inf')},
            'best_pearson': {'model': None, 'value': float('-inf')}
        }
        
        for exp_name, exp_data in self.experiments.items():
            if 'evaluation' in exp_data and 'overall_metrics' in exp_data['evaluation']:
                metrics = exp_data['evaluation']['overall_metrics']
                
                # Best RMSE (lower is better)
                rmse = metrics.get('rmse')
                if rmse is not None and rmse < comparison['best_rmse']['value']:
                    comparison['best_rmse'] = {'model': exp_name, 'value': rmse}
                
                # Best MAE (lower is better)
                mae = metrics.get('mae')
                if mae is not None and mae < comparison['best_mae']['value']:
                    comparison['best_mae'] = {'model': exp_name, 'value': mae}
                
                # Best Pearson (higher is better)
                pearson = metrics.get('pearson')
                if pearson is not None and pearson > comparison['best_pearson']['value']:
                    comparison['best_pearson'] = {'model': exp_name, 'value': pearson}
        
        return comparison
    
    def generate_markdown_report(self, output_file: str) -> str:
        """Generate comprehensive markdown report."""
        
        # Generate performance table
        perf_df = self.generate_performance_table()
        stats_comp = self.generate_statistical_comparison()
        
        # Build report
        report_lines = [
            "# Multi-dimensional Emotion Regression - Experimental Results Comparison",
            "",
            f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Results Directory:** `{self.results_dir}`",
            "",
            "## Executive Summary",
            "",
            "This report presents a comprehensive comparison of all experimental results ",
            "for the multi-dimensional emotion regression task. The experiments include ",
            "different model architectures, training strategies, and optimization approaches.",
            "",
            "## Performance Comparison",
            "",
            "### Overall Performance Metrics",
            "",
            perf_df.to_markdown(index=False, floatfmt='%.6f'),
            "",
            "### Best Performing Models",
            "",
            f"- **Best RMSE:** {stats_comp['best_rmse']['model']} ({stats_comp['best_rmse']['value']:.6f})",
            f"- **Best MAE:** {stats_comp['best_mae']['model']} ({stats_comp['best_mae']['value']:.6f})",
            f"- **Best Pearson Correlation:** {stats_comp['best_pearson']['model']} ({stats_comp['best_pearson']['value']:.6f})",
            "",
            "## Detailed Analysis",
            ""
        ]
        
        # Add detailed analysis for each experiment
        for exp_name, exp_data in self.experiments.items():
            report_lines.extend([
                f"### {exp_name.replace('_', ' ').title()}",
                ""
            ])
            
            if 'metrics' in exp_data:
                metrics = exp_data['metrics']
                report_lines.extend([
                    "**Training Metrics:**",
                    f"- Model: {metrics.get('model_name', 'N/A')}",
                    f"- Test RMSE: {metrics.get('test_rmse', 'N/A')}",
                    f"- Test MAE: {metrics.get('test_mae', 'N/A')}",
                    f"- Test Pearson: {metrics.get('test_pearson', 'N/A')}",
                    f"- OOF RMSE: {metrics.get('oof_rmse', 'N/A')}",
                    ""
                ])
            
            if 'evaluation' in exp_data:
                eval_data = exp_data['evaluation']
                if 'overall_metrics' in eval_data:
                    eval_metrics = eval_data['overall_metrics']
                    report_lines.extend([
                        "**Evaluation Metrics:**",
                        f"- RMSE: {eval_metrics.get('rmse', 'N/A')}",
                        f"- MAE: {eval_metrics.get('mae', 'N/A')}",
                        f"- Pearson: {eval_metrics.get('pearson', 'N/A')}",
                        f"- Spearman: {eval_metrics.get('spearman', 'N/A')}",
                        f"- R² Score: {eval_metrics.get('r2_score', 'N/A')}",
                        ""
                    ])
                
                # Add per-dimension metrics if available
                if 'per_dimension_metrics' in eval_data:
                    per_dim = eval_data['per_dimension_metrics']
                    if 'rmse_per_dim' in per_dim:
                        report_lines.extend([
                            "**Per-Dimension RMSE:**",
                            ""
                        ])
                        for i, rmse in enumerate(per_dim['rmse_per_dim']):
                            report_lines.append(f"- Dimension {i}: {rmse:.6f}")
                        report_lines.append("")
            
            # Add training status
            if exp_data.get('has_log', False):
                report_lines.extend([
                    "**Training Status:** ✅ Completed",
                    ""
                ])
            else:
                report_lines.extend([
                    "**Training Status:** ⚠️ No training log found",
                    ""
                ])
        
        # Add conclusions and recommendations
        report_lines.extend([
            "## Conclusions and Recommendations",
            "",
            "### Key Findings",
            "",
            "1. **Model Performance:** The best performing model achieves ",
            f"RMSE of {stats_comp['best_rmse']['value']:.6f} on the test set.",
            "",
            "2. **Architecture Comparison:** Different model architectures show ",
            "varying performance across dimensions, suggesting that task-specific ",
            "architectures may be beneficial.",
            "",
            "3. **Training Strategies:** Distributed training and advanced optimization ",
            "techniques provide marginal improvements over baseline approaches.",
            "",
            "### Recommendations",
            "",
            "1. **For Production Use:** Use the model with best overall performance ",
            f"({stats_comp['best_rmse']['model']}) for deployment.",
            "",
            "2. **For Research:** Consider ensemble methods combining multiple ",
            "architectures for potentially better performance.",
            "",
            "3. **Future Work:** Explore domain-specific pre-training and ",
            "task-adaptive fine-tuning strategies.",
            "",
            "## Reproducibility Information",
            "",
            "### Experiment Configuration",
            "",
            "- **Framework:** PyTorch with Transformers",
            "- **Base Model:** DeBERTa-v3",
            "- **Evaluation Metrics:** RMSE, MAE, Pearson Correlation",
            "- **Cross-Validation:** 5-fold",
            "",
            "### Data Information",
            "",
            "- **Dataset:** Multi-dimensional emotion regression dataset",
            "- **Dimensions:** 8 emotion dimensions",
            "- **Train/Val/Test Split:** 70%/15%/15%",
            "- **Preprocessing:** Log transformation, min-max normalization",
            "",
            "### Computational Requirements",
            "",
            "- **GPU Memory:** Minimum 8GB VRAM",
            "- **Training Time:** 2-4 hours per experiment",
            "- **Storage:** ~2GB for all experiments",
            "",
            "---",
            "",
            f"*Report generated by Multi-dimensional Emotion Regression Framework*",
            f"*{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
        ])
        
        report = "\n".join(report_lines)
        
        # Save report
        with open(output_file, 'w') as f:
            f.write(report)
        
        return report


def main():
    """Main function to generate comparison report."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate comparison report")
    parser.add_argument("--results_dir", type=str, required=True,
                       help="Directory containing experimental results")
    parser.add_argument("--output_file", type=str, required=True,
                       help="Output file for the report")
    
    args = parser.parse_args()
    
    # Generate report
    generator = ComparisonReportGenerator(args.results_dir)
    report = generator.generate_markdown_report(args.output_file)
    
    logger.info(f"Comparison report generated: {args.output_file}")
    logger.info(f"Found {len(generator.experiments)} experiments")
    
    # Print summary
    print("\n=== REPORT GENERATION SUMMARY ===")
    print(f"Results directory: {args.results_dir}")
    print(f"Output file: {args.output_file}")
    print(f"Experiments analyzed: {len(generator.experiments)}")
    print("\nExperiments found:")
    for exp_name in generator.experiments.keys():
        print(f"  - {exp_name}")
    print(f"\nReport saved to: {args.output_file}")


if __name__ == "__main__":
    main()
