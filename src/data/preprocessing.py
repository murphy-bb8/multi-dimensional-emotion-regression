"""
Data preprocessing utilities for multi-dimensional emotion regression.

This module provides functions for:
- Data loading and validation
- Label preprocessing (log transform, normalization, outlier removal)
- Text preprocessing and tokenization
- Data augmentation
- Dataset splitting
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import ast
from typing import Tuple, Dict, List, Optional
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class EmotionDataProcessor:
    """
    Comprehensive data processor for multi-dimensional emotion regression.
    
    Handles loading, preprocessing, and validation of emotion datasets
    with multi-dimensional labels.
    """
    
    def __init__(self, 
                 model_name: str = "microsoft/deberta-v3-base",
                 max_length: int = 128,
                 random_state: int = 42):
        """
        Initialize the data processor.
        
        Args:
            model_name: HuggingFace model name for tokenization
            max_length: Maximum sequence length for tokenization
            random_state: Random seed for reproducibility
        """
        self.model_name = model_name
        self.max_length = max_length
        self.random_state = random_state
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        
        # Store preprocessing statistics
        self.label_stats = {}
        self.is_fitted = False
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load and validate dataset from CSV file.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Loaded and validated DataFrame
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        try:
            df = pd.read_csv(file_path, engine='python', on_bad_lines='skip')
        except Exception as e:
            raise ValueError(f"Error loading CSV file: {e}")
        
        # Validate required columns
        required_columns = ['content', 'label']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Remove rows with missing values
        initial_count = len(df)
        df = df.dropna(subset=required_columns).reset_index(drop=True)
        final_count = len(df)
        
        if initial_count != final_count:
            logger.warning(f"Removed {initial_count - final_count} rows with missing values")
        
        # Parse label strings to lists if needed
        if isinstance(df['label'].iloc[0], str):
            try:
                df['label'] = df['label'].apply(ast.literal_eval)
            except Exception as e:
                raise ValueError(f"Error parsing label strings: {e}")
        
        # Validate label format
        self._validate_labels(df['label'])
        
        logger.info(f"Loaded {len(df)} samples from {file_path}")
        return df
    
    def _validate_labels(self, labels: pd.Series) -> None:
        """
        Validate label format and values.
        
        Args:
            labels: Series containing label lists
        """
        for idx, label in enumerate(labels):
            if not isinstance(label, (list, np.ndarray)):
                raise ValueError(f"Label at index {idx} is not a list/array: {type(label)}")
            
            if len(label) != 8:
                raise ValueError(f"Label at index {idx} has {len(label)} dimensions, expected 8")
            
            try:
                label_floats = [float(x) for x in label]
                if any(np.isnan(x) or np.isinf(x) for x in label_floats):
                    raise ValueError(f"Label at index {idx} contains NaN or Inf values")
            except (ValueError, TypeError) as e:
                raise ValueError(f"Label at index {idx} contains non-numeric values: {e}")
    
    def remove_outliers(self, 
                       df: pd.DataFrame, 
                       low_quantile: float = 0.01,
                       high_quantile: float = 0.99,
                       bins: int = 20) -> pd.DataFrame:
        """
        Remove outliers based on label distributions.
        
        Args:
            df: Input DataFrame
            low_quantile: Lower quantile threshold
            high_quantile: Upper quantile threshold
            bins: Number of bins for distribution analysis
            
        Returns:
            DataFrame with outliers removed
        """
        labels = np.array(df['label'].tolist())
        keep = np.ones(len(df), dtype=bool)
        
        for dim in range(labels.shape[1]):
            col = labels[:, dim]
            q_low, q_high = np.quantile(col, [low_quantile, high_quantile])
            keep &= (col >= q_low) & (col <= q_high)
        
        df_filtered = df[keep].reset_index(drop=True)
        removed_count = len(df) - len(df_filtered)
        
        if removed_count > 0:
            logger.info(f"Removed {removed_count} outliers ({removed_count/len(df)*100:.1f}%)")
        
        return df_filtered
    
    def apply_log_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply log1p transformation to labels.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with log-transformed labels
        """
        def log1p_transform(labels):
            try:
                return (np.log1p(np.array(labels))).tolist()
            except Exception as e:
                logger.error(f"Error in log transform: {e}")
                return labels
        
        df = df.copy()
        df['label'] = df['label'].apply(log1p_transform)
        logger.info("Applied log1p transformation to labels")
        return df
    
    def normalize_labels(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Normalize labels to [0, 1] range using min-max scaling.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit normalization parameters (True for training, False for validation/test)
            
        Returns:
            DataFrame with normalized labels
        """
        df = df.copy()
        labels = np.array(df['label'].tolist())
        
        if fit:
            # Fit normalization parameters
            self.label_stats['min'] = labels.min(axis=0)
            self.label_stats['max'] = labels.max(axis=0)
            self.is_fitted = True
            logger.info("Fitted label normalization parameters")
        
        if not self.is_fitted:
            raise ValueError("Normalizer not fitted. Call with fit=True on training data first.")
        
        # Apply normalization
        min_vals = self.label_stats['min']
        max_vals = self.label_stats['max']
        
        def normalize_row(label):
            return ((np.array(label) - min_vals) / (max_vals - min_vals + 1e-8)).tolist()
        
        df['label'] = df['label'].apply(normalize_row)
        logger.info("Applied min-max normalization to labels")
        return df
    
    def denormalize_labels(self, normalized_labels: np.ndarray) -> np.ndarray:
        """
        Denormalize labels back to original scale.
        
        Args:
            normalized_labels: Normalized label array
            
        Returns:
            Denormalized label array
        """
        if not self.is_fitted:
            raise ValueError("Normalizer not fitted")
        
        min_vals = self.label_stats['min']
        max_vals = self.label_stats['max']
        
        return normalized_labels * (max_vals - min_vals + 1e-8) + min_vals
    
    def tokenize_texts(self, df: pd.DataFrame) -> List[Dict]:
        """
        Tokenize text data using the specified tokenizer.
        
        Args:
            df: DataFrame with 'content' column
            
        Returns:
            List of tokenized examples
        """
        tokenized_data = []
        
        for idx, row in df.iterrows():
            content = row['content']
            
            # Tokenize
            encoded = self.tokenizer(
                content,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            tokenized_data.append({
                'input_ids': encoded['input_ids'].squeeze(0),
                'attention_mask': encoded['attention_mask'].squeeze(0),
                'label': torch.tensor(row['label'], dtype=torch.float)
            })
        
        logger.info(f"Tokenized {len(tokenized_data)} samples")
        return tokenized_data
    
    def split_data(self, 
                   df: pd.DataFrame,
                   train_ratio: float = 0.7,
                   val_ratio: float = 0.15,
                   test_ratio: float = 0.15,
                   stratify: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            df: Input DataFrame
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            test_ratio: Ratio for test set
            stratify: Whether to use stratified splitting
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Train, validation, and test ratios must sum to 1.0")
        
        # First split: train + val vs test
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_ratio,
            random_state=self.random_state,
            stratify=df['label'].apply(lambda x: str(x)) if stratify else None
        )
        
        # Second split: train vs val
        val_size_adjusted = val_ratio / (train_ratio + val_ratio)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size_adjusted,
            random_state=self.random_state,
            stratify=train_val_df['label'].apply(lambda x: str(x)) if stratify else None
        )
        
        logger.info(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)
    
    def preprocess_pipeline(self,
                          input_path: str,
                          output_dir: str,
                          apply_outlier_removal: bool = True,
                          apply_log_transform: bool = True,
                          split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)) -> Dict[str, pd.DataFrame]:
        """
        Complete preprocessing pipeline.
        
        Args:
            input_path: Path to input CSV file
            output_dir: Directory to save processed files
            apply_outlier_removal: Whether to remove outliers
            apply_log_transform: Whether to apply log transformation
            split_ratios: Tuple of (train, val, test) ratios
            
        Returns:
            Dictionary containing processed DataFrames
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        df = self.load_data(input_path)
        logger.info(f"Loaded {len(df)} samples")
        
        # Remove outliers
        if apply_outlier_removal:
            df = self.remove_outliers(df)
            logger.info(f"After outlier removal: {len(df)} samples")
        
        # Apply log transformation
        if apply_log_transform:
            df = self.apply_log_transform(df)
        
        # Split data
        train_df, val_df, test_df = self.split_data(df, *split_ratios)
        
        # Normalize labels (fit on training data only)
        train_df = self.normalize_labels(train_df, fit=True)
        val_df = self.normalize_labels(val_df, fit=False)
        test_df = self.normalize_labels(test_df, fit=False)
        
        # Save processed data
        train_df.to_csv(os.path.join(output_dir, 'train_data.csv'), index=False)
        val_df.to_csv(os.path.join(output_dir, 'val_data.csv'), index=False)
        test_df.to_csv(os.path.join(output_dir, 'test_data.csv'), index=False)
        
        # Save normalization parameters
        with open(os.path.join(output_dir, 'normalization_params.json'), 'w') as f:
            json.dump(self.label_stats, f, indent=2)
        
        logger.info(f"Preprocessed data saved to {output_dir}")
        
        return {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }
    
    def save_preprocessor(self, path: str) -> None:
        """Save preprocessor state."""
        state = {
            'model_name': self.model_name,
            'max_length': self.max_length,
            'random_state': self.random_state,
            'label_stats': self.label_stats,
            'is_fitted': self.is_fitted
        }
        
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Preprocessor state saved to {path}")
    
    def load_preprocessor(self, path: str) -> None:
        """Load preprocessor state."""
        with open(path, 'r') as f:
            state = json.load(f)
        
        self.model_name = state['model_name']
        self.max_length = state['max_length']
        self.random_state = state['random_state']
        self.label_stats = state['label_stats']
        self.is_fitted = state['is_fitted']
        
        # Reinitialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        
        logger.info(f"Preprocessor state loaded from {path}")


def main():
    """Example usage of the data processor."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess emotion regression dataset")
    parser.add_argument("--input", type=str, required=True, help="Input CSV file path")
    parser.add_argument("--output", type=str, required=True, help="Output directory path")
    parser.add_argument("--model", type=str, default="microsoft/deberta-v3-base", 
                       help="Model name for tokenization")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = EmotionDataProcessor(
        model_name=args.model,
        max_length=args.max_length
    )
    
    # Run preprocessing pipeline
    processed_data = processor.preprocess_pipeline(
        input_path=args.input,
        output_dir=args.output
    )
    
    print("Preprocessing completed successfully!")
    print(f"Train: {len(processed_data['train'])} samples")
    print(f"Val: {len(processed_data['val'])} samples")
    print(f"Test: {len(processed_data['test'])} samples")


if __name__ == "__main__":
    main()
