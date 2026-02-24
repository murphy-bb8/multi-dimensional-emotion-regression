#!/usr/bin/env python3
"""
Comprehensive evaluation script for multi-dimensional emotion regression models.

This script provides:
- Model loading and evaluation
- Metrics computation
- Result visualization
- Statistical analysis
- Report generation
"""

import os
import json
import argparse
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns

from ..models.bert_regressor import BertForMultiRegression, create_model
from ..data.preprocessing import EmotionDataProcessor
from .metrics import RegressionMetrics, ModelComparator, compute_confidence_intervals

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluator."""
    
    def __init__(self, 
                 model_path: str,
                 model_type: str = "base",
                 device: Optional[str] = None):
        """
        Initialize evaluator.
        
        Args:
            model_path: Path to trained model
            model_type: Type of model ('base', 'gated', 'moe')
            device: Device to run evaluation on
        """
        self.model_path = Path(model_path)
        self.model_type = model_type
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = self.load_model()
        self.model.eval()
        
        # Initialize metrics calculator
        self.metrics_calculator = RegressionMetrics()
        
        logger.info(f"Initialized evaluator for {model_type} model on {self.device}")
    
    def load_model(self) -> nn.Module:
        """Load trained model."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        # Load model state
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Determine model configuration
        if 'config' in checkpoint:
            config = checkpoint['config']
        else:
            # Default configuration
            config = {
                'model_type': self.model_type,
                'model_name': 'microsoft/deberta-v3-base',
                'output_dim': 8,
                'dropout': 0.3
            }
        
        # Create model
        model = create_model(config)
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        return model
    
    def load_data(self, data_path: str) -> tuple:
        """Load and prepare evaluation data."""
        processor = EmotionDataProcessor()
        
        # Load data
        df = processor.load_data(data_path)
        
        # Tokenize
        tokenized_data = processor.tokenize_texts(df)
        
        return df, tokenized_data
    
    def evaluate_model(self, 
                      data_path: str,
                      output_dir: str,
                      batch_size: int = 32,
                      create_visualizations: bool = True) -> Dict[str, Any]:
        """
        Evaluate model on test data.
        
        Args:
            data_path: Path to test data
            output_dir: Directory to save results
            batch_size: Batch size for evaluation
            create_visualizations: Whether to create visualizations
            
        Returns:
            Dictionary with evaluation results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        df, tokenized_data = self.load_data(data_path)
        
        # Create dataset and dataloader
        from ..data.preprocessing import EmotionDatasetSingleDim
        
        # For multi-dimensional evaluation, we need to handle all dimensions
        all_predictions = []
        all_targets = []
        
        # Evaluate each dimension separately
        for dim in range(8):  # Assuming 8 dimensions
            dataset = EmotionDatasetSingleDim(tokenized_data, dim)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            
            dim_predictions = []
            dim_targets = []
            
            with torch.no_grad():
                for batch in dataloader:
                    inputs = {
                        'input_ids': batch['input_ids'].to(self.device),
                        'attention_mask': batch['attention_mask'].to(self.device)
                    }
                    targets = batch['label'].to(self.device)
                    
                    outputs = self.model(**inputs)
                    
                    dim_predictions.extend(outputs.cpu().numpy())
                    dim_targets.extend(targets.cpu().numpy())
            
            all_predictions.append(np.array(dim_predictions))
            all_targets.append(np.array(dim_targets))
        
        # Combine predictions and targets
        y_pred = np.stack(all_predictions, axis=1)  # [n_samples, n_dims]
        y_true = np.stack(all_targets, axis=1)    # [n_samples, n_dims]
        
        # Compute metrics
        overall_metrics = self.metrics_calculator.compute_all_metrics(y_true, y_pred)
        per_dim_metrics = self.metrics_calculator.compute_per_dimension_metrics(y_true, y_pred)
        
        # Compute confidence intervals
        confidence_intervals = compute_confidence_intervals(y_true, y_pred)
        
        # Prepare results
        results = {
            'overall_metrics': overall_metrics,
            'per_dimension_metrics': per_dim_metrics,
            'confidence_intervals': confidence_intervals,
            'predictions': y_pred.tolist(),
            'targets': y_true.tolist(),
            'model_info': {
                'model_path': str(self.model_path),
                'model_type': self.model_type,
                'device': self.device,
                'n_samples': len(y_true),
                'n_dimensions': y_true.shape[1]
            }
        }
        
        # Save results
        self.save_results(results, output_dir)
        
        # Create visualizations
        if create_visualizations:
            self.create_visualizations(y_true, y_pred, output_dir)
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_dir: str):
        """Save evaluation results."""
        # Save main results
        with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save metrics as CSV
        metrics_df = self.metrics_calculator.create_metrics_dataframe(results['overall_metrics'])
        metrics_df.to_csv(os.path.join(output_dir, 'overall_metrics.csv'))
        
        # Save per-dimension metrics
        per_dim_df = pd.DataFrame(results['per_dimension_metrics'])
        per_dim_df.index = [f'Dim_{i}' for i in range(len(per_dim_df))]
        per_dim_df.to_csv(os.path.join(output_dir, 'per_dimension_metrics.csv'))
        
        # Save predictions and targets
        np.save(os.path.join(output_dir, 'predictions.npy'), np.array(results['predictions']))
        np.save(os.path.join(output_dir, 'targets.npy'), np.array(results['targets']))
        
        logger.info(f"Results saved to {output_dir}")
    
    def create_visualizations(self, y_true: np.ndarray, y_pred: np.ndarray, output_dir: str):
        """Create evaluation visualizations."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Overall prediction vs true scatter plot
        plt.figure(figsize=(10, 8))
        plt.scatter(y_true.flatten(), y_pred.flatten(), alpha=0.5, s=20)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title('Overall Prediction vs True Values')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'overall_prediction_vs_true.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Per-dimension scatter plots
        n_dims = y_true.shape[1]
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for dim in range(n_dims):
            ax = axes[dim]
            ax.scatter(y_true[:, dim], y_pred[:, dim], alpha=0.5, s=15)
            
            # Perfect prediction line
            min_val = min(y_true[:, dim].min(), y_pred[:, dim].min())
            max_val = max(y_true[:, dim].max(), y_pred[:, dim].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
            
            ax.set_xlabel('True')
            ax.set_ylabel('Predicted')
            ax.set_title(f'Dimension {dim}')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'per_dimension_predictions.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Residual distribution
        residuals = y_pred - y_true
        
        plt.figure(figsize=(15, 5))
        
        # Overall residuals
        plt.subplot(1, 3, 1)
        plt.hist(residuals.flatten(), bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Residual (Predicted - True)')
        plt.ylabel('Frequency')
        plt.title('Overall Residual Distribution')
        plt.grid(True, alpha=0.3)
        
        # Per-dimension residuals boxplot
        plt.subplot(1, 3, 2)
        plt.boxplot(residuals, labels=[f'Dim {i}' for i in range(n_dims)])
        plt.xlabel('Dimension')
        plt.ylabel('Residual')
        plt.title('Per-Dimension Residual Distribution')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Residual vs predicted
        plt.subplot(1, 3, 3)
        plt.scatter(y_pred.flatten(), residuals.flatten(), alpha=0.5, s=10)
        plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residuals vs Predicted Values')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'residual_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Metrics heatmap
        per_dim_metrics = self.metrics_calculator.compute_per_dimension_metrics(y_true, y_pred)
        
        # Create metrics matrix for heatmap
        metrics_for_heatmap = ['rmse', 'mae', 'pearson', 'r2_score']
        metrics_matrix = []
        
        for metric in metrics_for_heatmap:
            if metric in per_dim_metrics:
                metrics_matrix.append(per_dim_metrics[metric])
        
        if metrics_matrix:
            metrics_matrix = np.array(metrics_matrix)
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(metrics_matrix, 
                       xticklabels=[f'Dim {i}' for i in range(n_dims)],
                       yticklabels=metrics_for_heatmap,
                       annot=True, 
                       fmt='.4f',
                       cmap='RdYlBu_r',
                       center=0)
            plt.title('Per-Dimension Performance Metrics Heatmap')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'metrics_heatmap.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 5. Error analysis by value ranges
        plt.figure(figsize=(15, 5))
        
        # Create value bins
        n_bins = 10
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        # Create bins based on true values
        bins = np.linspace(y_true_flat.min(), y_true_flat.max(), n_bins + 1)
        bin_indices = np.digitize(y_true_flat, bins) - 1
        
        bin_errors = []
        bin_centers = []
        
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                errors = np.abs(y_pred_flat[mask] - y_true_flat[mask])
                bin_errors.append(errors.mean())
                bin_centers.append((bins[i] + bins[i + 1]) / 2)
        
        if bin_errors:
            plt.subplot(1, 2, 1)
            plt.bar(bin_centers, bin_errors, width=(bins[1] - bins[0]) * 0.8, alpha=0.7)
            plt.xlabel('True Value Range')
            plt.ylabel('Mean Absolute Error')
            plt.title('Error Analysis by Value Range')
            plt.grid(True, alpha=0.3)
            
            # Error distribution by dimension
            plt.subplot(1, 2, 2)
            dim_errors = np.mean(np.abs(residuals), axis=0)
            plt.bar(range(n_dims), dim_errors, alpha=0.7)
            plt.xlabel('Dimension')
            plt.ylabel('Mean Absolute Error')
            plt.title('Average Error by Dimension')
            plt.xticks(range(n_dims), [f'Dim {i}' for i in range(n_dims)])
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'error_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to {output_dir}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate multi-dimensional emotion regression model")
    
    # Required arguments
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model")
    parser.add_argument("--test_data", type=str, required=True,
                       help="Path to test data CSV file")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save evaluation results")
    
    # Optional arguments
    parser.add_argument("--model_type", type=str, default="base",
                       choices=["base", "gated", "moe"],
                       help="Type of model architecture")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to run evaluation on (auto-detected if None)")
    parser.add_argument("--no_visualizations", action="store_true",
                       help="Skip creating visualizations")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        model_path=args.model_path,
        model_type=args.model_type,
        device=args.device
    )
    
    # Run evaluation
    logger.info("Starting model evaluation...")
    results = evaluator.evaluate_model(
        data_path=args.test_data,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        create_visualizations=not args.no_visualizations
    )
    
    # Print summary
    print("\n=== EVALUATION SUMMARY ===")
    print(f"Model: {args.model_path}")
    print(f"Test Data: {args.test_data}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Number of Samples: {results['model_info']['n_samples']}")
    print(f"Number of Dimensions: {results['model_info']['n_dimensions']}")
    
    print("\nOverall Metrics:")
    overall = results['overall_metrics']
    print(f"  RMSE: {overall.get('rmse', 'N/A'):.6f}")
    print(f"  MAE: {overall.get('mae', 'N/A'):.6f}")
    print(f"  Pearson: {overall.get('pearson', 'N/A'):.6f}")
    print(f"  R² Score: {overall.get('r2_score', 'N/A'):.6f}")
    
    print("\nPer-Dimension RMSE:")
    per_dim_rmse = results['per_dimension_metrics'].get('rmse', [])
    for i, rmse in enumerate(per_dim_rmse):
        print(f"  Dimension {i}: {rmse:.6f}")
    
    print(f"\nDetailed results saved to: {args.output_dir}")
    logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
