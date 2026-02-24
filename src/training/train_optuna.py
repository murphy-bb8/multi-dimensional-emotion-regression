import os, json, shutil, math, ast, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import KFold
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import logging
import seaborn as sns
import optuna

# Core functions and classes copied from train_ddp.py

def log1p_label(series):
    try:
        return series.apply(lambda x: (np.log1p(np.array(x))).tolist())
    except Exception as e:
        logging.error(f"Error in log1p_label: {e}", exc_info=True)
        return series

def get_min_max(series):
    a = np.array(series.tolist()); return a.min(axis=0), a.max(axis=0)

def normalize(series, mn, mx):
    try:
        normed = series.apply(lambda x: ((np.array(x) - mn) / (mx - mn + 1e-8)).tolist())
        arr = np.vstack(normed.values)
        assert np.isfinite(arr).all(), 'Non-finite values in normalized label'
        return normed
    except Exception as e:
        logging.error(f"Error in normalize: {e}", exc_info=True)
        return series

def denorm(v, mn, mx):
    v = np.asarray(v); return v * (mx - mn + 1e-8) + mn

def metrics_np(y_true, y_pred):
    y_true = np.asarray(y_true, np.float64); y_pred = np.asarray(y_pred, np.float64)
    mse = float(np.mean((y_pred - y_true) ** 2)); rmse = float(np.sqrt(mse)); mae = float(np.mean(np.abs(y_pred - y_true)))
    try: pearson = float(np.corrcoef(y_true.reshape(-1), y_pred.reshape(-1))[0,1])
    except Exception: pearson = float('nan')
    return dict(mse=mse, rmse=rmse, mae=mae, pearson=pearson)

def compute_dim_density_weights_from_encoded(encoded_list, dim, bins=None, alpha=1.5):
    n = max(10, len(encoded_list)); bins = bins or int(np.clip(n//200, 40, 80))
    vals = [(it['label'][dim].item() if isinstance(it['label'], torch.Tensor) else float(it['label'][dim])) for it in encoded_list]
    vals = np.asarray(vals, np.float64)
    hist, edges = np.histogram(vals, bins=bins, density=True)
    idx = np.clip(np.digitize(vals, edges[:-1], right=True), 0, bins-1)
    dens = np.maximum(hist[idx], 1e-8); w = (1.0/(dens))**alpha
    return (w/(w.mean()+1e-12)).astype(np.float32)

class EmotionDatasetSingleDim(Dataset):
    def __init__(self, data, dim, ext_weights=None): self.data, self.dim, self.ext = data, dim, ext_weights
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        b = self.data[i]; lbl = b['label']
        y = lbl[self.dim].view(1) if isinstance(lbl, torch.Tensor) else torch.tensor([lbl[self.dim]], dtype=torch.float)
        item = {'input_ids': b['input_ids'], 'attention_mask': b['attention_mask'], 'label': y}
        if self.ext is not None: item['weight'] = torch.tensor(self.ext[i], dtype=torch.float)
        return item

def weighted_huber_loss(pred, target, weight, delta=0.5):
    if weight.ndim == 1: weight = weight.unsqueeze(1)
    diff = pred - target; ad = torch.abs(diff)
    return (torch.where(ad <= delta, 0.5*diff.pow(2), delta*(ad - 0.5*delta)) * weight).mean()

def weighted_mse_loss(pred, target, weight):
    if weight.ndim == 1: weight = weight.unsqueeze(1)
    return ((pred - target) ** 2 * weight).mean()

class BertForMultiRegression(nn.Module):
    def __init__(self, pretrained_model_name, output_dim=1, dropout=0.5):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(pretrained_model_name)
        hidden = self.backbone.config.hidden_size
        self.has_pooler = hasattr(self.backbone, 'pooler') and (self.backbone.pooler is not None)
        self.regressor_ = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim)
        )
    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        h = out.pooler_output if self.has_pooler and getattr(out, 'pooler_output', None) is not None else out.last_hidden_state[:, 0]
        return self.regressor_(h)

def encode(df, tokenizer, low_freq_indices=None, dim=None):
    out=[]
    for idx, r in df.iterrows():
        content = r['content']
        if low_freq_indices and idx in low_freq_indices:
            if random.random() < 0.5:
                content = random_mask_text(content, tokenizer)
            if aug_syn is not None and random.random() < 0.5:
                content = synonym_augment(content)
        elif AUGMENT_DATA and random.random() < AUGMENT_PROB:
            content = random_mask_text(content, tokenizer)
        enc = tokenizer(content, truncation=True, padding='max_length', max_length=MAX_LEN, return_tensors='pt')
        out.append({'input_ids': enc['input_ids'].squeeze(0),
                    'attention_mask': enc['attention_mask'].squeeze(0),
                    'label': torch.tensor(r['label'], dtype=torch.float)})
    return out

def save_checkpoint(state, filename, rank=0):
    if rank == 0:
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            torch.save(state, filename)
        except Exception as e:
            print(f"[Checkpoint save failed] {filename}: {e}")

def load_checkpoint(filename, device):
    return torch.load(filename, map_location=device)

def setup_distributed():
    import torch.distributed as dist
    rank = int(os.environ['RANK']) if 'RANK' in os.environ else 0
    world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    local_rank = int(os.environ['LOCAL_RANK']) if 'LOCAL_RANK' in os.environ else 0
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    if world_size > 1:
        dist.init_process_group(backend=backend)
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank

# fit_one_fold, run_dimension_kfold, print_label_stats, get_low_freq_indices directly copied from train_ddp.py with same names

# ... (Please copy these functions completely from train_ddp.py here) ...

# ... rest of the main flow ...

# 1. Load and preprocess data
train_df = pd.read_csv("data__/train_data.csv", engine='python', on_bad_lines='skip')
val_df   = pd.read_csv("data__/val_data.csv",   engine='python', on_bad_lines='skip')
test_df  = pd.read_csv("data__/test_data.csv",  engine='python', on_bad_lines='skip')
for df_ in (train_df, val_df, test_df):
    df_.dropna(subset=['label'], inplace=True)
    df_.reset_index(drop=True, inplace=True)
    if isinstance(df_['label'].iloc[0], str):
        df_['label'] = df_['label'].apply(ast.literal_eval)
train_df = remove_outliers(train_df)
val_df = remove_outliers(val_df)
train_df['label'] = log1p_label(train_df['label'])
val_df['label'] = log1p_label(val_df['label'])
test_df['label'] = log1p_label(test_df['label'])
y_min, y_max = get_min_max(train_df['label'])
np.save('y_min.npy', y_min); np.save('y_max.npy', y_max)
y_min = np.load('y_min.npy'); y_max = np.load('y_max.npy')
train_df['label'] = normalize(train_df['label'], y_min, y_max)
val_df['label']   = normalize(val_df['label'],   y_min, y_max)
test_df['label']  = normalize(test_df['label'],  y_min, y_max)

# 2. Load tokenizer
MODEL_NAME = "/root/autodl-tmp/WorkPlace/models/microsoft--deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

# 3. Distributed setup
rank, world_size, local_rank = setup_distributed()
device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
use_amp = True
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

# 4. Optuna hyperparameter optimization objective function
def suggest_params(trial):
    return {
        'loss': trial.suggest_categorical('loss', ['mse', 'huber']),
        'huber_delta': trial.suggest_float('huber_delta', 0.3, 1.0),
        'epochs': trial.suggest_int('epochs', 10, 25),
        'batch': trial.suggest_categorical('batch', [32, 48, 64, 96]),
        'lr_bert': trial.suggest_loguniform('lr_bert', 1e-6, 1e-5),
        'lr_head': trial.suggest_loguniform('lr_head', 1e-6, 5e-5),
        'wd': trial.suggest_loguniform('wd', 1e-3, 1.0),
        'warmup': trial.suggest_float('warmup', 0.05, 0.2),
        'clip': trial.suggest_float('clip', 0.5, 2.0),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5),
        'freeze_epochs': trial.suggest_int('freeze_epochs', 3, 7)
    }

def objective(trial, d):
    params = suggest_params(trial)
    low_freq_indices_train = get_low_freq_indices(train_df, d)
    low_freq_indices_val = get_low_freq_indices(val_df, d)
    low_freq_indices_test = get_low_freq_indices(test_df, d)
    encoded_data_train = encode(train_df, tokenizer, low_freq_indices_train, d)
    encoded_data_val   = encode(val_df, tokenizer, low_freq_indices_val, d)
    encoded_data_test  = encode(test_df, tokenizer, low_freq_indices_test, d)
    res = safe_run(run_dimension_kfold, d, encoded_data_train, encoded_data_val, encoded_data_test, device, use_amp, scaler, rank, world_size)
    if res is None:
        return float('inf')
    return res['oof_rmse']

# 5. Automatic hyperparameter optimization and formal training
if rank == 0:
    for d in range(8):
        study = optuna.create_study(direction='minimize', study_name=f'dim_{d}_study')
        study.optimize(lambda trial: objective(trial, d), n_trials=20)
        best_params = study.best_params
        with open(f'best_params_dim{d}.json', 'w') as f:
            json.dump(best_params, f, indent=2)
        print(f"Dimension {d} best parameters:", best_params)

    # Formal training with best parameters
    results = []
    for d in range(8):
        with open(f'best_params_dim{d}.json') as f:
            best_cfg = json.load(f)
        low_freq_indices_train = get_low_freq_indices(train_df, d)
        low_freq_indices_val = get_low_freq_indices(val_df, d)
        low_freq_indices_test = get_low_freq_indices(test_df, d)
        encoded_data_train = encode(train_df, tokenizer, low_freq_indices_train, d)
        encoded_data_val   = encode(val_df, tokenizer, low_freq_indices_val, d)
        encoded_data_test  = encode(test_df, tokenizer, low_freq_indices_test, d)
        res = safe_run(run_dimension_kfold, d, encoded_data_train, encoded_data_val, encoded_data_test, device, use_amp, scaler, rank, world_size, cfg=best_cfg)
        if res is not None:
            results.append(res)
    with open('outputs_8x1/summary_kfold_optuna.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print('✅ Optuna automatic hyperparameter optimization and formal training completed!')

    # Copy visualization aggregation part from train_ddp.py
    import seaborn as sns
    OUT_ROOT = 'outputs_8x1'
    if len(results) > 0:
        all_test_pred = []
        all_test_true = []
        for res in results:
            test_pred_path = os.path.join(res['out_dir'], 'test_pred.npy')
            test_true_path = os.path.join(res['out_dir'], 'test_true.npy')
            if os.path.exists(test_pred_path) and os.path.exists(test_true_path):
                all_test_pred.append(np.load(test_pred_path))
                all_test_true.append(np.load(test_true_path))
        if len(all_test_pred) == 0 or len(all_test_true) == 0:
            print('❌ No valid prediction results generated, training process interrupted. Please check the logs above.')
            exit(1)
        all_test_pred = np.stack(all_test_pred, axis=1)
        all_test_true = np.stack(all_test_true, axis=1)
        n_dims = all_test_pred.shape[1]
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        for i in range(n_dims):
            ax = axes[i//4, i%4]
            ax.scatter(all_test_true[:, i], all_test_pred[:, i], alpha=0.5)
            ax.plot([all_test_true[:, i].min(), all_test_true[:, i].max()], [all_test_true[:, i].min(), all_test_true[:, i].max()], 'r--')
            ax.set_xlabel('True'); ax.set_ylabel('Predicted'); ax.set_title(f'Dim {i}')
        plt.suptitle('Scatter Plots of True vs Predicted (Per Dimension)'); plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(OUT_ROOT, 'scatter_true_vs_pred_optuna.png')); plt.close()
        # mse bar
        mse_list = [r['oof_rmse']**2 for r in results]
        plt.figure(figsize=(8, 6)); plt.bar(range(1, n_dims+1), mse_list, color='skyblue', alpha=0.8)
        plt.xlabel('Dimension'); plt.ylabel('MSE'); plt.title('Per-Dimension MSE'); plt.savefig(os.path.join(OUT_ROOT, 'mse_per_dim_optuna.png')); plt.close()
        # correlation heatmap
        corr = np.corrcoef(all_test_pred.T)
        plt.figure(figsize=(8, 6)); sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', square=True)
        plt.title('Predicted Emotion Dimensions Correlation Heatmap'); plt.savefig(os.path.join(OUT_ROOT, 'correlation_heatmap_optuna.png')); plt.close()
        # error distribution
        fig, axes = plt.subplots(2, 4, figsize=(20, 8))
        for i in range(n_dims):
            ax = axes[i//4, i%4]
            err = all_test_true[:, i] - all_test_pred[:, i]
            ax.hist(err, bins=50, alpha=0.5, color='skyblue'); ax.set_title(f'Dim {i} Error Distribution')
            ax.set_xlabel('True - Predicted'); ax.set_ylabel('Sample Count')
        plt.tight_layout(); plt.savefig(os.path.join(OUT_ROOT, 'error_distribution_optuna.png')); plt.close()
        # sample comparison
        plt.figure(figsize=(10, 12))
        for idx in range(min(5, all_test_true.shape[0])):
            plt.subplot(5, 1, idx+1)
            plt.bar(np.arange(n_dims)-0.2, all_test_true[idx], width=0.4, label='True', color='skyblue')
            plt.bar(np.arange(n_dims)+0.2, all_test_pred[idx], width=0.4, label='Predicted', color='salmon')
            plt.xticks(np.arange(n_dims), [f'Dim {i}' for i in range(n_dims)]); plt.ylim(0, 1); plt.title(f'Sample #{idx} Emotion Prediction Comparison')
            if idx == 0: plt.legend()
        plt.tight_layout(); plt.savefig(os.path.join(OUT_ROOT, 'sample_comparison_optuna.png')); plt.close()
        print('✅ Optuna visualization output generated!')
