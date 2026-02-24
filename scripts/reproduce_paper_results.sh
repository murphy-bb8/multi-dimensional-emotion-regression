#!/bin/bash

# Reproduction Script for IEEE Access Paper Results
# Multi-dimensional Emotion Regression with Deep Learning

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Configuration
EXPERIMENT_NAME="paper_reproduction_$(date +%Y%m%d_%H%M%S)"
BASE_DIR="experiments/${EXPERIMENT_NAME}"
CONFIG_FILE="configs/default.yaml"
DATA_DIR="data"
RESULTS_DIR="${BASE_DIR}/results"

print_status "Starting reproduction of paper results"
print_status "Experiment name: ${EXPERIMENT_NAME}"

# Create experiment directory
mkdir -p "${BASE_DIR}"
mkdir -p "${RESULTS_DIR}"

# Check if required files exist
if [ ! -f "${CONFIG_FILE}" ]; then
    print_error "Configuration file not found: ${CONFIG_FILE}"
    exit 1
fi

if [ ! -d "${DATA_DIR}" ]; then
    print_error "Data directory not found: ${DATA_DIR}"
    print_status "Please download the dataset first using:"
    print_status "  python scripts/download_local.py"
    exit 1
fi

# Check if data files exist
if [ ! -f "${DATA_DIR}/train_data.csv" ] || [ ! -f "${DATA_DIR}/val_data.csv" ] || [ ! -f "${DATA_DIR}/test_data.csv" ]; then
    print_error "Data files not found in ${DATA_DIR}"
    print_status "Please ensure train_data.csv, val_data.csv, and test_data.csv are present"
    exit 1
fi

print_status "All required files found. Starting experiments..."

# Experiment 1: Base Model Training
print_status "=== Experiment 1: Base Model Training ==="

BASE_MODEL_OUTPUT="${RESULTS_DIR}/base_model"
mkdir -p "${BASE_MODEL_OUTPUT}"

print_status "Training base BERT regression model..."
python src/training/train.py \
    --config "${CONFIG_FILE}" \
    --output_dir "${BASE_MODEL_OUTPUT}" \
    --model_type "base" \
    --experiment_name "base_model" \
    2>&1 | tee "${BASE_MODEL_OUTPUT}/training.log"

if [ $? -eq 0 ]; then
    print_status "Base model training completed successfully"
else
    print_error "Base model training failed"
    exit 1
fi

# Experiment 2: Distributed Training (if multiple GPUs available)
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    if [ $GPU_COUNT -gt 1 ]; then
        print_status "=== Experiment 2: Distributed Training ==="
        
        DDP_OUTPUT="${RESULTS_DIR}/distributed_training"
        mkdir -p "${DDP_OUTPUT}"
        
        print_status "Training with ${GPU_COUNT} GPUs using DDP..."
        torchrun --nproc_per_node=${GPU_COUNT} src/training/train_ddp.py \
            --config "${CONFIG_FILE}" \
            --output_dir "${DDP_OUTPUT}" \
            --experiment_name "distributed_training" \
            2>&1 | tee "${DDP_OUTPUT}/training.log"
        
        if [ $? -eq 0 ]; then
            print_status "Distributed training completed successfully"
        else
            print_warning "Distributed training failed, continuing with other experiments"
        fi
    else
        print_warning "Only one GPU detected, skipping distributed training experiment"
    fi
else
    print_warning "nvidia-smi not found, skipping distributed training experiment"
fi

# Experiment 3: Mixture of Experts
print_status "=== Experiment 3: Mixture of Experts ==="

MOE_OUTPUT="${RESULTS_DIR}/moe_model"
mkdir -p "${MOE_OUTPUT}"

print_status "Training Mixture of Experts model..."
python src/training/train_moe.py \
    --config "${CONFIG_FILE}" \
    --output_dir "${MOE_OUTPUT}" \
    --experiment_name "moe_model" \
    --moe_enabled true \
    2>&1 | tee "${MOE_OUTPUT}/training.log"

if [ $? -eq 0 ]; then
    print_status "MoE model training completed successfully"
else
    print_error "MoE model training failed"
    exit 1
fi

# Experiment 4: Hyperparameter Optimization
print_status "=== Experiment 4: Hyperparameter Optimization ==="

OPTUNA_OUTPUT="${RESULTS_DIR}/hyperparameter_optimization"
mkdir -p "${OPTUNA_OUTPUT}"

print_status "Running hyperparameter optimization with Optuna..."
python src/training/train_optuna.py \
    --config "${CONFIG_FILE}" \
    --output_dir "${OPTUNA_OUTPUT}" \
    --experiment_name "hyperparameter_optimization" \
    --optuna_enabled true \
    --optuna_trials 50 \
    2>&1 | tee "${OPTUNA_OUTPUT}/optimization.log"

if [ $? -eq 0 ]; then
    print_status "Hyperparameter optimization completed successfully"
else
    print_warning "Hyperparameter optimization failed, continuing with evaluation"
fi

# Experiment 5: Model Evaluation and Comparison
print_status "=== Experiment 5: Model Evaluation ==="

EVAL_OUTPUT="${RESULTS_DIR}/evaluation"
mkdir -p "${EVAL_OUTPUT}"

print_status "Evaluating all trained models..."

# Evaluate base model
if [ -d "${BASE_MODEL_OUTPUT}" ]; then
    print_status "Evaluating base model..."
    python src/evaluation/evaluate.py \
        --model_path "${BASE_MODEL_OUTPUT}/best_model.pth" \
        --test_data "${DATA_DIR}/test_data.csv" \
        --output_dir "${EVAL_OUTPUT}/base_model" \
        --model_type "base" \
        2>&1 | tee "${EVAL_OUTPUT}/base_model_evaluation.log"
fi

# Evaluate MoE model
if [ -d "${MOE_OUTPUT}" ]; then
    print_status "Evaluating MoE model..."
    python src/evaluation/evaluate.py \
        --model_path "${MOE_OUTPUT}/best_model.pth" \
        --test_data "${DATA_DIR}/test_data.csv" \
        --output_dir "${EVAL_OUTPUT}/moe_model" \
        --model_type "moe" \
        2>&1 | tee "${EVAL_OUTPUT}/moe_model_evaluation.log"
fi

# Generate comparison report
print_status "Generating comparison report..."
python scripts/generate_comparison_report.py \
    --results_dir "${RESULTS_DIR}" \
    --output_file "${BASE_DIR}/comparison_report.md" \
    2>&1 | tee "${BASE_DIR}/report_generation.log"

# Create visualization notebook
print_status "Creating visualization notebook..."
python scripts/create_visualization_notebook.py \
    --results_dir "${RESULTS_DIR}" \
    --output_file "${BASE_DIR}/results_visualization.ipynb"

# Experiment 6: Statistical Analysis
print_status "=== Experiment 6: Statistical Analysis ==="

STATS_OUTPUT="${RESULTS_DIR}/statistical_analysis"
mkdir -p "${STATS_OUTPUT}"

print_status "Performing statistical analysis..."
python scripts/statistical_analysis.py \
    --results_dir "${RESULTS_DIR}" \
    --output_dir "${STATS_OUTPUT}" \
    2>&1 | tee "${STATS_OUTPUT}/statistical_analysis.log"

# Create final summary
print_status "=== Creating Final Summary ==="

cat > "${BASE_DIR}/experiment_summary.txt" << EOF
Paper Reproduction Summary
=========================
Experiment Name: ${EXPERIMENT_NAME}
Date: $(date)
Configuration: ${CONFIG_FILE}

Completed Experiments:
1. Base Model Training: $([ -d "${BASE_MODEL_OUTPUT}" ] && echo "✅ SUCCESS" || echo "❌ FAILED")
2. Distributed Training: $([ -d "${DDP_OUTPUT}" ] && echo "✅ SUCCESS" || echo "⏭️ SKIPPED")
3. Mixture of Experts: $([ -d "${MOE_OUTPUT}" ] && echo "✅ SUCCESS" || echo "❌ FAILED")
4. Hyperparameter Optimization: $([ -d "${OPTUNA_OUTPUT}" ] && echo "✅ SUCCESS" || echo "⚠️ PARTIAL")
5. Model Evaluation: $([ -d "${EVAL_OUTPUT}" ] && echo "✅ SUCCESS" || echo "❌ FAILED")
6. Statistical Analysis: $([ -d "${STATS_OUTPUT}" ] && echo "✅ SUCCESS" || echo "❌ FAILED")

Results Location: ${RESULTS_DIR}
Comparison Report: ${BASE_DIR}/comparison_report.md
Visualization Notebook: ${BASE_DIR}/results_visualization.ipynb

Next Steps:
1. Review the comparison report for performance metrics
2. Open the visualization notebook for detailed analysis
3. Check individual experiment logs for any issues
4. Compare results with those reported in the paper

EOF

print_status "Reproduction completed!"
print_status "Results saved in: ${BASE_DIR}"
print_status "Main results directory: ${RESULTS_DIR}"
print_status "Comparison report: ${BASE_DIR}/comparison_report.md"
print_status "Visualization notebook: ${BASE_DIR}/results_visualization.ipynb"

# Display summary
echo ""
echo "=== REPRODUCTION SUMMARY ==="
cat "${BASE_DIR}/experiment_summary.txt"

# Optional: Open results in browser (uncomment if desired)
# if command -v xdg-open &> /dev/null; then
#     xdg-open "${BASE_DIR}/comparison_report.md"
# elif command -v open &> /dev/null; then
#     open "${BASE_DIR}/comparison_report.md"
# fi

print_status "To run individual experiments, use the following commands:"
echo "  Base model: python src/training/train.py --config ${CONFIG_FILE}"
echo "  Distributed: torchrun --nproc_per_node=N src/training/train_ddp.py --config ${CONFIG_FILE}"
echo "  MoE model: python src/training/train_moe.py --config ${CONFIG_FILE} --moe_enabled true"
echo "  Evaluation: python src/evaluation/evaluate.py --model_path PATH --test_data ${DATA_DIR}/test_data.csv"

print_status "Thank you for reproducing our paper results!"
