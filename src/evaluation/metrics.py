"""
Comprehensive evaluation metrics for multi-dimensional emotion regression.

This module provides:
- Standard regression metrics (MSE, RMSE, MAE)
- Correlation metrics (Pearson, Spearman)
- Multi-dimensional evaluation utilities
- Statistical significance testing
- Visualization helpers
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging

logger = logging.getLogger(__name__)


class RegressionMetrics:
    """
    Comprehensive metrics calculator for regression tasks.
    
    Provides standard and advanced metrics for evaluating multi-dimensional
    regression predictions with proper statistical validation.
    """
    
    def __init__(self, n_dimensions: int = 8):
        """
        Initialize metrics calculator.
        
        Args:
            n_dimensions: Number of output dimensions
        """
        self.n_dimensions = n_dimensions
        self.metric_names = [
            'mse', 'rmse', 'mae', 'pearson', 'spearman', 
            'r2_score', 'mape', 'explained_variance'
        ]
    
    def compute_basic_metrics(self, 
                             y_true: np.ndarray, 
                             y_pred: np.ndarray) -> Dict[str, float]:
        """
        Compute basic regression metrics.
        
        Args:
            y_true: Ground truth values [n_samples, n_dims]
            y_pred: Predicted values [n_samples, n_dims]
            
        Returns:
            Dictionary of computed metrics
        """
        # Ensure arrays are numpy arrays
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        
        # Flatten for overall metrics
        y_true_flat = y_true.reshape(-1)
        y_pred_flat = y_pred.reshape(-1)
        
        metrics = {}
        
        # Mean Squared Error
        mse = mean_squared_error(y_true_flat, y_pred_flat)
        metrics['mse'] = float(mse)
        
        # Root Mean Squared Error
        metrics['rmse'] = float(np.sqrt(mse))
        
        # Mean Absolute Error
        metrics['mae'] = float(mean_absolute_error(y_true_flat, y_pred_flat))
        
        # R² Score
        ss_res = np.sum((y_true_flat - y_pred_flat) ** 2)
        ss_tot = np.sum((y_true_flat - np.mean(y_true_flat)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        metrics['r2_score'] = float(r2)
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_true_flat - y_pred_flat) / (y_true_flat + 1e-8))) * 100
        metrics['mape'] = float(mape)
        
        # Explained Variance Score
        explained_var = 1 - np.var(y_true_flat - y_pred_flat) / (np.var(y_true_flat) + 1e-8)
        metrics['explained_variance'] = float(explained_var)
        
        return metrics
    
    def compute_correlation_metrics(self,
                                  y_true: np.ndarray,
                                  y_pred: np.ndarray) -> Dict[str, float]:
        """
        Compute correlation-based metrics.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            
        Returns:
            Dictionary of correlation metrics
        """
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        
        metrics = {}
        
        # Overall correlations (flattened)
        y_true_flat = y_true.reshape(-1)
        y_pred_flat = y_pred.reshape(-1)
        
        # Pearson correlation
        try:
            pearson_corr, pearson_p = stats.pearsonr(y_true_flat, y_pred_flat)
            metrics['pearson'] = float(pearson_corr)
            metrics['pearson_p_value'] = float(pearson_p)
        except Exception as e:
            logger.warning(f"Could not compute Pearson correlation: {e}")
            metrics['pearson'] = float('nan')
            metrics['pearson_p_value'] = float('nan')
        
        # Spearman correlation
        try:
            spearman_corr, spearman_p = stats.spearmanr(y_true_flat, y_pred_flat)
            metrics['spearman'] = float(spearman_corr)
            metrics['spearman_p_value'] = float(spearman_p)
        except Exception as e:
            logger.warning(f"Could not compute Spearman correlation: {e}")
            metrics['spearman'] = float('nan')
            metrics['spearman_p_value'] = float('nan')
        
        return metrics
    
    def compute_per_dimension_metrics(self,
                                     y_true: np.ndarray,
                                     y_pred: np.ndarray) -> Dict[str, List[float]]:
        """
        Compute metrics for each dimension separately.
        
        Args:
            y_true: Ground truth values [n_samples, n_dims]
            y_pred: Predicted values [n_samples, n_dims]
            
        Returns:
            Dictionary with per-dimension metric lists
        """
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        
        if y_true.shape[1] != self.n_dimensions:
            raise ValueError(f"Expected {self.n_dimensions} dimensions, got {y_true.shape[1]}")
        
        per_dim_metrics = {
            'mse': [],
            'rmse': [],
            'mae': [],
            'pearson': [],
            'spearman': [],
            'r2_score': []
        }
        
        for dim in range(self.n_dimensions):
            y_true_dim = y_true[:, dim]
            y_pred_dim = y_pred[:, dim]
            
            # Basic metrics
            mse = mean_squared_error(y_true_dim, y_pred_dim)
            per_dim_metrics['mse'].append(float(mse))
            per_dim_metrics['rmse'].append(float(np.sqrt(mse)))
            per_dim_metrics['mae'].append(float(mean_absolute_error(y_true_dim, y_pred_dim)))
            
            # R² score
            ss_res = np.sum((y_true_dim - y_pred_dim) ** 2)
            ss_tot = np.sum((y_true_dim - np.mean(y_true_dim)) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-8))
            per_dim_metrics['r2_score'].append(float(r2))
            
            # Correlations
            try:
                pearson_corr, _ = stats.pearsonr(y_true_dim, y_pred_dim)
                per_dim_metrics['pearson'].append(float(pearson_corr))
            except:
                per_dim_metrics['pearson'].append(float('nan'))
            
            try:
                spearman_corr, _ = stats.spearmanr(y_true_dim, y_pred_dim)
                per_dim_metrics['spearman'].append(float(spearman_corr))
            except:
                per_dim_metrics['spearman'].append(float('nan'))
        
        return per_dim_metrics
    
    def compute_all_metrics(self,
                           y_true: np.ndarray,
                           y_pred: np.ndarray,
                           include_per_dimension: bool = True) -> Dict[str, Union[float, List[float]]]:
        """
        Compute all available metrics.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            include_per_dimension: Whether to include per-dimension metrics
            
        Returns:
            Dictionary with all computed metrics
        """
        # Basic metrics
        metrics = self.compute_basic_metrics(y_true, y_pred)
        
        # Correlation metrics
        correlation_metrics = self.compute_correlation_metrics(y_true, y_pred)
        metrics.update(correlation_metrics)
        
        # Per-dimension metrics
        if include_per_dimension:
            per_dim_metrics = self.compute_per_dimension_metrics(y_true, y_pred)
            for key, values in per_dim_metrics.items():
                metrics[f'{key}_per_dim'] = values
        
        return metrics
    
    def statistical_significance_test(self,
                                    y_true: np.ndarray,
                                    y_pred1: np.ndarray,
                                    y_pred2: np.ndarray,
                                    alpha: float = 0.05) -> Dict[str, float]:
        """
        Perform statistical significance test between two prediction sets.
        
        Args:
            y_true: Ground truth values
            y_pred1: First set of predictions
            y_pred2: Second set of predictions
            alpha: Significance level
            
        Returns:
            Dictionary with test results
        """
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred1 = np.asarray(y_pred, dtype=np.float64)
        y_pred2 = np.asarray(y_pred, dtype=np.float64)
        
        # Compute errors
        errors1 = np.abs(y_true - y_pred1)
        errors2 = np.abs(y_true - y_pred2)
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(errors1.flatten(), errors2.flatten())
        
        # Wilcoxon signed-rank test (non-parametric alternative)
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(errors1.flatten(), errors2.flatten())
        
        return {
            't_statistic': float(t_stat),
            't_p_value': float(p_value),
            'wilcoxon_statistic': float(wilcoxon_stat),
            'wilcoxon_p_value': float(wilcoxon_p),
            'significant_difference': p_value < alpha,
            'significant_difference_wilcoxon': wilcoxon_p < alpha
        }
    
    def create_metrics_dataframe(self, metrics: Dict[str, Union[float, List[float]]]) -> pd.DataFrame:
        """
        Create a pandas DataFrame from metrics dictionary.
        
        Args:
            metrics: Metrics dictionary
            
        Returns:
            Formatted DataFrame
        """
        # Separate scalar and per-dimension metrics
        scalar_metrics = {}
        per_dim_metrics = {}
        
        for key, value in metrics.items():
            if isinstance(value, list):
                per_dim_metrics[key] = value
            else:
                scalar_metrics[key] = [value]
        
        # Create DataFrames
        scalar_df = pd.DataFrame(scalar_metrics, index=['Overall'])
        
        if per_dim_metrics:
            per_dim_df = pd.DataFrame(per_dim_metrics)
            per_dim_df.index = [f'Dim_{i}' for i in range(len(per_dim_df))]
            
            # Combine
            combined_df = pd.concat([scalar_df, per_dim_df])
        else:
            combined_df = scalar_df
        
        return combined_df.round(6)


class ModelComparator:
    """
    Utility class for comparing multiple models.
    
    Provides statistical comparison and ranking of different models
    based on their performance metrics.
    """
    
    def __init__(self, metrics_calculator: RegressionMetrics):
        """
        Initialize model comparator.
        
        Args:
            metrics_calculator: Metrics calculator instance
        """
        self.metrics_calculator = metrics_calculator
        self.model_results = {}
    
    def add_model_results(self,
                         model_name: str,
                         y_true: np.ndarray,
                         y_pred: np.ndarray) -> Dict[str, float]:
        """
        Add results for a model.
        
        Args:
            model_name: Name of the model
            y_true: Ground truth values
            y_pred: Model predictions
            
        Returns:
            Computed metrics for the model
        """
        metrics = self.metrics_calculator.compute_all_metrics(y_true, y_pred)
        self.model_results[model_name] = metrics
        
        logger.info(f"Added results for model: {model_name}")
        return metrics
    
    def rank_models(self, metric: str = 'rmse', ascending: bool = True) -> List[Tuple[str, float]]:
        """
        Rank models based on a specific metric.
        
        Args:
            metric: Metric to rank by
            ascending: Whether lower values are better
            
        Returns:
            List of (model_name, metric_value) tuples
        """
        rankings = []
        for model_name, metrics in self.model_results.items():
            if metric in metrics and not isinstance(metrics[metric], list):
                rankings.append((model_name, metrics[metric]))
        
        # Sort by metric value
        rankings.sort(key=lambda x: x[1], reverse=not ascending)
        return rankings
    
    def create_comparison_table(self) -> pd.DataFrame:
        """
        Create a comparison table for all models.
        
        Returns:
            DataFrame with model comparison
        """
        if not self.model_results:
            raise ValueError("No model results available for comparison")
        
        # Extract scalar metrics for each model
        comparison_data = {}
        for model_name, metrics in self.model_results.items():
            model_metrics = {}
            for key, value in metrics.items():
                if not isinstance(value, list):
                    model_metrics[key] = value
            comparison_data[model_name] = model_metrics
        
        return pd.DataFrame(comparison_data).T.round(6)


def compute_confidence_intervals(y_true: np.ndarray,
                               y_pred: np.ndarray,
                               confidence_level: float = 0.95,
                               n_bootstrap: int = 1000) -> Dict[str, Tuple[float, float]]:
    """
    Compute confidence intervals for metrics using bootstrap.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        confidence_level: Confidence level (0.0-1.0)
        n_bootstrap: Number of bootstrap samples
        
    Returns:
        Dictionary with confidence intervals for each metric
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
    n_samples = len(y_true)
    alpha = 1 - confidence_level
    
    # Bootstrap samples
    bootstrap_metrics = {
        'rmse': [],
        'mae': [],
        'pearson': []
    }
    
    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        
        # Compute metrics
        bootstrap_metrics['rmse'].append(np.sqrt(mean_squared_error(y_true_boot, y_pred_boot)))
        bootstrap_metrics['mae'].append(mean_absolute_error(y_true_boot, y_pred_boot))
        
        try:
            pearson_corr, _ = stats.pearsonr(y_true_boot.flatten(), y_pred_boot.flatten())
            bootstrap_metrics['pearson'].append(pearson_corr)
        except:
            bootstrap_metrics['pearson'].append(np.nan)
    
    # Compute confidence intervals
    confidence_intervals = {}
    for metric, values in bootstrap_metrics.items():
        values = np.array(values)
        values = values[~np.isnan(values)]  # Remove NaN values
        
        if len(values) > 0:
            lower = np.percentile(values, 100 * alpha / 2)
            upper = np.percentile(values, 100 * (1 - alpha / 2))
            confidence_intervals[metric] = (float(lower), float(upper))
        else:
            confidence_intervals[metric] = (float('nan'), float('nan'))
    
    return confidence_intervals


def main():
    """Example usage of metrics calculation."""
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    n_dims = 8
    
    y_true = np.random.randn(n_samples, n_dims)
    y_pred = y_true + 0.1 * np.random.randn(n_samples, n_dims)
    
    # Initialize metrics calculator
    calculator = RegressionMetrics(n_dimensions=n_dims)
    
    # Compute all metrics
    metrics = calculator.compute_all_metrics(y_true, y_pred)
    
    # Create DataFrame
    df = calculator.create_metrics_dataframe(metrics)
    print("Metrics DataFrame:")
    print(df)
    
    # Compute confidence intervals
    ci = compute_confidence_intervals(y_true, y_pred)
    print("\nConfidence Intervals (95%):")
    for metric, (lower, upper) in ci.items():
        print(f"{metric}: [{lower:.4f}, {upper:.4f}]")


if __name__ == "__main__":
    main()
