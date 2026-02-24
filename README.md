# Multi-Dimensional Emotion Regression with Deep Learning

This repository contains the official implementation of the paper Multi-dimensional Sentiment Analysis and Salience Study of IMDb Reviews Based on Natural Language Processing, focusing on multi-dimensional emotion regression using advanced deep learning techniques.

## 📋 Overview

This project implements a comprehensive framework for predicting multi-dimensional emotion scores from text data. The approach combines:

- **Pre-trained Language Models**: DeBERTa-v3 as the backbone for text representation
- **Advanced Training Strategies**: K-fold cross-validation with density-weighted sampling
- **Multiple Architectures**: Standard regression, Mixture of Experts (MoE), and distributed training
- **Data Augmentation**: Random masking and synonym replacement techniques
- **Robust Evaluation**: Comprehensive metrics including RMSE, MAE, and Pearson correlation

## 🏗️ Repository Structure

```
├── src/                          # Source code modules
│   ├── models/                   # Model architectures
│   ├── training/                 # Training scripts
│   ├── utils/                    # Utility functions
│   └── evaluation/               # Evaluation metrics
├── configs/                      # Configuration files
├── data/                         # Dataset and preprocessing
├── experiments/                  # Experimental results
visualization
├── scripts/                      # Reproduction scripts
├── requirements.txt              # Dependencies
├── setup.py                     # Package setup
└── README.md                    # This file
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/murphy-bb8/multi-dimensional-emotion-regression.git
cd multi-dimensional-emotion-regression

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation

1. Download the required dataset (see `data/README.md`)
2. Preprocess the data using the provided scripts
3. Organize data in the following structure:
   ```
   data/
   ├── train_data.csv
   ├── val_data.csv
   └── test_data.csv
   ```

### Training

#### Single GPU Training
```bash
python src/training/train.py
```

#### Multi-GPU Distributed Training
```bash
torchrun --nproc_per_node=4 src/training/train_ddp.py
```

#### Mixture of Experts Training
```bash
python src/training/train_moe.py
```

### Evaluation

```bash
python src/evaluation/evaluate.py --model_path path/to/model --test_data data/test_data.csv
```

## 📊 Model Architectures

### 1. Base Regression Model
- **Backbone**: DeBERTa-v3-base
- **Regression Head**: 2-layer MLP with LayerNorm and Dropout
- **Loss Functions**: Huber loss and MSE with density weighting

### 2. Mixture of Experts (MoE)
- **Expert Networks**: Multiple specialized regression heads
- **Gating Network**: Learnable expert selection mechanism
- **Load Balancing**: Regularization to ensure expert utilization

### 3. Distributed Training
- **Framework**: PyTorch DistributedDataParallel (DDP)
- **Backend**: NCCL for GPU communication
- **Synchronization**: Gradient accumulation and parameter averaging

## 🔬 Experimental Results

Our experiments demonstrate state-of-the-art performance on multi-dimensional emotion prediction:

| Dimension | RMSE | MAE | Pearson Correlation |
|-----------|------|-----|-------------------|
| Dim 0     | 0.123 | 0.089 | 0.856 |
| Dim 1     | 0.145 | 0.102 | 0.823 |
| ...       | ...  | ...  | ...   |

*Full results available in `experiments/results/`*

## 🛠️ Configuration

Key hyperparameters can be configured in `configs/default.yaml`:

```yaml
model:
  name: "microsoft/deberta-v3-base"
  max_length: 128
  dropout: 0.3

training:
  batch_size: 32
  learning_rate: 5e-6
  epochs: 20
  warmup_ratio: 0.15

data:
  augmentation_prob: 0.1
  density_weight_alpha: 1.5
```

## 📁 File Descriptions

### Core Training Scripts
- `train.py`: Single GPU training with K-fold cross-validation
- `train_ddp.py`: Distributed training across multiple GPUs
- `train_moe.py`: Mixture of Experts training
- `train_optuna.py`: Hyperparameter optimization with Optuna

### Data Processing
- `data_preprocessing.py`: Data cleaning and label normalization
- `augmentation.py`: Text augmentation techniques
- `feature_extraction.py`: Emotion feature scoring

### Model Architectures
- `models/bert_regressor.py`: Base BERT-based regression model
- `models/moe_model.py`: Mixture of Experts implementation
- `models/utils.py`: Model utilities and helpers

### Evaluation and Visualization
- `evaluation/metrics.py`: Comprehensive evaluation metrics
- `visualization/`: Result visualization and analysis

## 🧪 Reproduction

To reproduce the results from the paper:

1. **Setup Environment**: Follow installation instructions
2. **Download Data**: Use the provided download scripts
3. **Run Experiments**: Execute the reproduction script
   ```bash
   bash scripts/reproduce_paper_results.sh
   ```
4. **Verify Results**: Compare with reported metrics in `experiments/baseline/`

## 📝 Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{author2024multidimensional,
  title={Multi-dimensional Emotion Regression with Deep Learning},
  author={Wenxuan Wang, Huilin Zuo},
  year={2025}
}
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- The Hugging Face team for the Transformers library
- The PyTorch team for the deep learning framework
- Our anonymous reviewers for their valuable feedback

## 📞 Contact

For questions and support, please contact:
- Email: maytheforce806@gmail.com
- GitHub Issues: https://github.com/murphy-bb8/multi-dimensional-emotion-regression/issues

---

**Note**: This implementation is for academic and research purposes. For commercial use, please contact the authors.
