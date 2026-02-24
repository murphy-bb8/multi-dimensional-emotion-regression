"""
Multi-dimensional Emotion Regression Framework

This package provides comprehensive tools for multi-dimensional emotion regression
using advanced deep learning techniques.

Modules:
- models: Neural network architectures
- training: Training scripts and utilities
- evaluation: Evaluation metrics and tools
- data: Data processing and loading
- utils: Utility functions
"""

__version__ = "1.0.0"
__author__ = "Wenxuan Wang"
__email__ = "maytheforce806@gmail.com"

from .models import *
from .training import *
from .evaluation import *
from .data import *
from .utils import *

__all__ = [
    "models",
    "training", 
    "evaluation",
    "data",
    "utils"
]
