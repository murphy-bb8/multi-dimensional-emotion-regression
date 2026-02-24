# Contributing to Multi-dimensional Emotion Regression

Thank you for your interest in contributing to our multi-dimensional emotion regression framework! This document provides guidelines and information for contributors.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Workflow](#development-workflow)
4. [Coding Standards](#coding-standards)
5. [Testing Guidelines](#testing-guidelines)
6. [Documentation](#documentation)
7. [Submitting Changes](#submitting-changes)
8. [Community](#community)

## Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic knowledge of deep learning and PyTorch

### Setup Development Environment

1. **Fork the repository**
   ```bash
   # Fork on GitHub, then clone your fork
   git clone https://github.com/yourusername/multi-dimensional-emotion-regression.git
   cd multi-dimensional-emotion-regression
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -e .  # Install in development mode
   ```

4. **Install development dependencies**
   ```bash
   pip install -r requirements-dev.txt
   ```

5. **Verify setup**
   ```bash
   python -c "import src; print('Setup successful!')"
   ```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Changes

- Write clean, well-documented code
- Follow the coding standards outlined below
- Add tests for new functionality
- Update documentation as needed

### 3. Test Your Changes

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_models.py

# Run with coverage
pytest --cov=src tests/
```

### 4. Commit Changes

```bash
git add .
git commit -m "feat: add new model architecture"
```

### 5. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub with a clear description of your changes.

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some additional guidelines:

#### Code Formatting

- Use 4 spaces for indentation (no tabs)
- Maximum line length: 88 characters
- Use [Black](https://black.readthedocs.io/) for automatic formatting
- Use [isort](https://isort.readthedocs.io/) for import sorting

```bash
# Format code
black src/ tests/
# Sort imports
isort src/ tests/
```

#### Naming Conventions

- **Variables and functions**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private members**: `_leading_underscore`

```python
class EmotionRegressor:
    def __init__(self, model_name: str):
        self._model_name = model_name
        self.MAX_SEQUENCE_LENGTH = 128
    
    def predict_emotions(self, text: str) -> np.ndarray:
        # Implementation
        pass
```

#### Type Hints

Use type hints for all function signatures and important variables:

```python
from typing import Dict, List, Optional, Union
import numpy as np

def process_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    """Process predictions and compute metrics."""
    pass
```

#### Documentation Strings

Use Google-style docstrings:

```python
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics.
    
    Args:
        y_true: Ground truth values with shape (n_samples, n_dims).
        y_pred: Predicted values with shape (n_samples, n_dims).
    
    Returns:
        Dictionary containing computed metrics:
        - 'rmse': Root mean squared error
        - 'mae': Mean absolute error
        - 'pearson': Pearson correlation coefficient
    
    Raises:
        ValueError: If input arrays have incompatible shapes.
    """
    pass
```

### Code Organization

#### Directory Structure

```
src/
├── models/           # Model architectures
├── training/         # Training scripts
├── evaluation/       # Evaluation metrics
├── data/            # Data processing
└── utils/           # Utility functions

tests/                # Test files
docs/                 # Documentation
scripts/              # Utility scripts
```

#### Module Structure

Each module should have:

1. **Module docstring**: Purpose and overview
2. **Imports**: Standard library, third-party, local
3. **Constants**: Module-level constants
4. **Classes**: Main functionality
5. **Functions**: Helper functions
6. **Main block**: `if __name__ == "__main__":`

```python
"""
Model architectures for multi-dimensional emotion regression.

This module provides various neural network architectures including
BERT-based regressors, mixture of experts models, and gated networks.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel

logger = logging.getLogger(__name__)

# Constants
DEFAULT_MODEL_NAME = "microsoft/deberta-v3-base"
DEFAULT_DROPOUT = 0.3

class BertRegressor(nn.Module):
    """BERT-based regression model."""
    
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        # Implementation
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        # Implementation

def create_model(config: dict) -> nn.Module:
    """Factory function for model creation."""
    # Implementation

if __name__ == "__main__":
    # Example usage
    pass
```

## Testing Guidelines

### Test Structure

- **Unit tests**: Test individual functions and classes
- **Integration tests**: Test component interactions
- **End-to-end tests**: Test complete workflows

### Writing Tests

```python
import pytest
import torch
import numpy as np
from src.models.bert_regressor import BertRegressor

class TestBertRegressor:
    """Test cases for BertRegressor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.model = BertRegressor()
        self.batch_size = 4
        self.seq_len = 128
    
    def test_forward_pass(self):
        """Test forward pass produces correct output shape."""
        input_ids = torch.randint(0, 1000, (self.batch_size, self.seq_len))
        attention_mask = torch.ones(self.batch_size, self.seq_len)
        
        output = self.model(input_ids, attention_mask)
        
        assert output.shape == (self.batch_size, 8), f"Expected shape (4, 8), got {output.shape}"
    
    def test_gradient_flow(self):
        """Test that gradients flow properly."""
        input_ids = torch.randint(0, 1000, (self.batch_size, self.seq_len))
        attention_mask = torch.ones(self.batch_size, self.seq_len)
        targets = torch.randn(self.batch_size, 8)
        
        output = self.model(input_ids, attention_mask)
        loss = torch.nn.functional.mse_loss(output, targets)
        loss.backward()
        
        # Check that some parameters have gradients
        has_gradients = any(p.grad is not None for p in self.model.parameters())
        assert has_gradients, "No gradients found in model parameters"
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test
pytest tests/test_models.py::TestBertRegressor::test_forward_pass

# Run with verbose output
pytest -v
```

## Documentation

### Types of Documentation

1. **Code Documentation**: Docstrings and comments
2. **API Documentation**: Generated from docstrings
3. **User Documentation**: Tutorials and guides
4. **Developer Documentation**: Architecture and design

### Writing Documentation

- Use clear, concise language
- Include code examples
- Add diagrams where helpful
- Keep documentation up to date

### Building Documentation

```bash
# Install documentation dependencies
pip install -r requirements-docs.txt

# Build documentation
cd docs/
make html

# View documentation
open _build/html/index.html
```

## Submitting Changes

### Pull Request Process

1. **Update Documentation**: Ensure all changes are documented
2. **Add Tests**: Include tests for new functionality
3. **Run Tests**: Ensure all tests pass
4. **Update Changelog**: Add entry to CHANGELOG.md
5. **Create Pull Request**: With clear description

### Pull Request Template

```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] All tests pass
- [ ] New tests added
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Changelog updated
```

### Review Process

1. **Automated Checks**: CI/CD pipeline runs tests
2. **Code Review**: Maintainers review code quality
3. **Discussion**: Address feedback and suggestions
4. **Approval**: Merge after approval

## Community

### Getting Help

- **GitHub Issues**: Report bugs and request features
- **Discussions**: Ask questions and share ideas
- **Email**: Contact maintainers directly

### Communication Channels

- **GitHub Discussions**: General questions and ideas
- **Issues**: Bug reports and feature requests
- **Email**: Private or sensitive matters

### Recognition

Contributors are recognized in:
- README.md contributors section
- Release notes
- Documentation acknowledgments

## Development Tips

### Debugging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Use pdb for debugging
import pdb; pdb.set_trace()

# Use print statements for quick debugging
print(f"Debug: variable={variable}")
```

### Performance Profiling

```python
import time
import cProfile

# Simple timing
start_time = time.time()
# ... code ...
print(f"Execution time: {time.time() - start_time:.2f}s")

# Detailed profiling
cProfile.run('your_function()')
```

### Memory Management

```python
# Clear GPU cache
torch.cuda.empty_cache()

# Monitor memory usage
import psutil
print(f"Memory usage: {psutil.virtual_memory().percent}%")
```

## Release Process

### Version Management

We use [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

1. **Update Version**: Update version in setup.py
2. **Update Changelog**: Add release notes
3. **Tag Release**: Create git tag
4. **Build Package**: Create distribution packages
5. **Upload to PyPI**: Publish package
6. **Update Documentation**: Deploy docs

## Questions?

If you have any questions about contributing, please:

1. Check existing documentation
2. Search existing issues and discussions
3. Create a new issue or discussion
4. Contact maintainers directly

Thank you for contributing to our project! 🎉
