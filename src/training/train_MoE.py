# Usage:
#   torchrun --nproc_per_node=2 train_ddp.py
# Or:
#   python -m torch.distributed.launch --nproc_per_node=2 train_ddp.py

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

plt.rcParams.update({'font.size': 14, 'axes.labelweight': 'bold', 'axes.titlesize': 16, 
                     'axes.labelsize': 14, 'xtick.labelsize': 12, 'ytick.labelsize': 12, 
                     'legend.fontsize': 12})
logging.basicConfig(filename='train_error.log', level=logging.ERROR)

def safe_run(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logging.error(f"Error in {func.__name__}: {e}", exc_info=True)
        return None

try:
    import nlpaug.augmenter.word as naw
    aug_syn = naw.SynonymAug(aug_src='wordnet')
except Exception:
    aug_syn = None

warnings.filterwarnings("ignore")

# Checkpoint resumption related
CHECKPOINT_DIR = 'checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Configuration
MODEL_NAME = "deberta-v3-base"
MAX_LEN = 128
K = 5
PATIENCE = 3
ALPHA_DENS = 1.5
OUT_ROOT = "outputs"

AUGMENT_DATA = True
AUGMENT_PROB = 0.1
MASK_TOKEN = '[MASK]'
AUGMENT_USE_SYNONYM = False

def synonym_augment(text):
    if aug_syn is not None:
        return aug_syn.augment(text)
    return text

def random_mask_text(text, tokenizer, prob=0.15):
    words = text.split()
    n = len(words)
    if n < 3: return text
    mask_num = max(1, int(n * prob))
    mask_idx = random.sample(range(n), mask_num)
    for idx in mask_idx:
        words[idx] = MASK_TOKEN
    return ' '.join(words)

# 1. Label log(x+1) transformation
# Apply log1p transformation before data reading and normalization
def log1p_label(series):
    try:
        return series.apply(lambda x: (np.log1p(np.array(x))).tolist())
    except Exception as e:
        logging.error(f"Error in log1p_label: {e}", exc_info=True)
        return series

DIM_CFG = {
    i: dict(loss='huber', huber_delta=0.6, epochs=20, batch=32, lr_bert=5e-6, lr_head=1e-5, wd=0.5, warmup=0.15, clip=1.0, freeze_epochs=5, dropout=0.3)
    for i in range(8)
}

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

def denorm(v, mn, mx): v = np.asarray(v); return v * (mx - mn + 1e-8) + mn

def metrics_np(y_true, y_pred):
    y_true = np.asarray(y_true, np.float64); y_pred = np.asarray(y_pred, np.float64)
    mse = float(np.mean((y_pred - y_true) ** 2)); rmse = float(np.sqrt(mse)); mae = float(np.mean(np.abs(y_pred - y_true)))
    try: pearson = float(np.corrcoef(y_true.reshape(-1), y_pred.reshape(-1))[0,1])
    except Exception: pearson = float('nan')
    return dict(mse=mse, rmse=rmse, mae=mae, pearson=pearson)

def compute_dim_density_weights_from_encoded(encoded_list, dim, bins=None, alpha=ALPHA_DENS):
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

# 1. MoE regression head
class MoERegressionHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        self.low_expert = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.high_expert = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, h):
        g = self.gate(h)  # (B, 1)
        y_low = self.low_expert(h)
        y_high = self.high_expert(h)
        y_pred = (1 - g) * y_low + g * y_high
        return y_pred, y_low, y_high, g

# 2. MoE loss function

def moe_loss(y_pred, y_true, y_low, y_high, g, low_q, high_q, margin=0.2, alpha=1.0, beta=0.1):
    # Main MSE
    main_loss = F.mse_loss(y_pred, y_true)
    # Soft label z
    z = ((y_true > low_q + margin) & (y_true < high_q - margin)).float() * 0.5 + (y_true >= high_q - margin).float()
    # Focus
    focus_loss = (1-z) * F.mse_loss(y_low, y_true) + z * F.mse_loss(y_high, y_true)
    # Entropy regularization
    entropy = - (g * torch.log(g + 1e-8) + (1-g) * torch.log(1-g + 1e-8)).mean()
    return main_loss + alpha * focus_loss.mean() - beta * entropy

class BertForMultiRegression(nn.Module):
    def __init__(self, pretrained_model_name=MODEL_NAME, output_dim=1, dropout=0.5):
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
        # Low-frequency interval sample enhancement
        if low_freq_indices and idx in low_freq_indices:
            if random.random() < 0.5:  # Enhancement probability adjustable
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
    """
    Initialize distributed process group.
    Expect to be launched by torchrun or torch.distributed.launch.
    """
    import torch.distributed as dist
    rank = int(os.environ['RANK']) if 'RANK' in os.environ else 0
    world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    local_rank = int(os.environ['LOCAL_RANK']) if 'LOCAL_RANK' in os.environ else 0
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    if world_size > 1:
        dist.init_process_group(backend=backend)
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank

def fit_one_fold(dim, fold_id, train_idx, val_idx, base_list, cfg, dim_out, device, use_amp, scaler, rank=0, world_size=1, local_rank=0, resume=True):
    """
    Note: we now expect each process to run this and use DistributedSampler for training.
    Only rank 0 will save best model and write visualizations.
    """
    try:
        fold_dir = os.path.join(dim_out, f'fold_{fold_id}')
        best_path = os.path.join(fold_dir, 'best_model.pth')
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f'dim{dim}_fold{fold_id}_ckpt.pth')
        # Checkpoint detection, skip if already completed
        if resume and os.path.exists(best_path) and os.path.exists(checkpoint_path):
            print(f"[Resume] Fold {fold_id} of dim {dim} already completed, skipping.")
            return best_path, None
        # compute sample weights (safe to compute on each rank since base_list identical)
        w_all = compute_dim_density_weights_from_encoded(base_list, dim)
        ds_full  = EmotionDatasetSingleDim(base_list, dim)
        ds_train = Subset(EmotionDatasetSingleDim(base_list, dim, ext_weights=w_all), train_idx.tolist())
        ds_val   = Subset(ds_full,  val_idx.tolist())

        # Distributed samplers
        train_sampler = DistributedSampler(ds_train, num_replicas=world_size, rank=rank, shuffle=True) if world_size>1 else None
        val_sampler = DistributedSampler(ds_val, num_replicas=world_size, rank=rank, shuffle=False) if world_size>1 else None

        train_loader = DataLoader(ds_train, batch_size=cfg['batch'], sampler=train_sampler, shuffle=(train_sampler is None), num_workers=8, pin_memory=True)
        val_loader   = DataLoader(ds_val,   batch_size=cfg['batch'], sampler=val_sampler,   shuffle=False, num_workers=8, pin_memory=True)

        # model to device
        model = BertForMultiRegression(pretrained_model_name=MODEL_NAME, output_dim=1, dropout=cfg['dropout']).to(device)
        # Only freeze first 10 layers after initialization
        try:
            backbone = model.backbone
            for name, param in backbone.named_parameters():
                if any([f'layer.{i}.' in name for i in range(10)]):
                    param.requires_grad = False
        except Exception as e:
            print('Failed to freeze first 10 layers:', e)
        # wrap model with DDP if distributed
        if world_size > 1:
            torch.cuda.set_device(local_rank)
            model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

        # parameter groups
        bb_params, head_params = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad: continue
            # When using DDP wrapper, parameter names include 'module.' prefix;
            keyname = name
            if keyname.startswith('module.'):
                keyname = keyname[len('module.'):]
            (bb_params if keyname.startswith('backbone.') else head_params).append(p)

        optimizer = torch.optim.AdamW([
            {'params': bb_params, 'lr': cfg['lr_bert'], 'weight_decay': cfg['wd']},
            {'params': head_params,'lr': cfg['lr_head'], 'weight_decay': cfg['wd']},
        ], betas=(0.9, 0.98), eps=1e-8)

        total_steps = len(train_loader) * max(1, cfg['epochs'])
        warmup_steps = int(total_steps * cfg['warmup'])
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: min((step+1)/max(1, warmup_steps), 1.0) * max(0.0, (total_steps-step)/max(1,total_steps))
        )

        best_val = float('inf'); no_improve = 0
        if rank==0: os.makedirs(fold_dir, exist_ok=True)
        start_epoch = 0

        # resume if rank 0 has ckpt - broadcast info to others
        if resume and os.path.exists(checkpoint_path):
            ckpt = load_checkpoint(checkpoint_path, device)
            # load model state for local model
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            scaler.load_state_dict(ckpt['scaler'])
            try: scheduler.load_state_dict(ckpt['scheduler'])
            except Exception: pass
            start_epoch = ckpt.get('epoch', 0) + 1
            best_val = ckpt.get('best_val', best_val)
            no_improve = ckpt.get('no_improve', 0)

        train_losses, val_losses = [], []

        # Remove/comment out freezing/unfreezing logic within epoch loop
        # for epoch in range(start_epoch, cfg['epochs']):
        #     if epoch < cfg['freeze_epochs']:
        #         backbone = model.module.backbone if hasattr(model, 'module') else model.backbone
        #         for p in backbone.parameters(): p.requires_grad = False
        #     else:
        #         backbone = model.module.backbone if hasattr(model, 'module') else model.backbone
        #         for p in backbone.parameters(): p.requires_grad = True

        for epoch in range(start_epoch, cfg['epochs']):
            # set epoch for sampler to shuffle per-epoch
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            model.train()
            pbar = tqdm(train_loader, desc=f"Dim {dim} | Fold {fold_id} | Epoch {epoch+1}/{cfg['epochs']}", leave=False, ascii=True)
            epoch_train_loss = 0.0

            for batch in pbar:
                optimizer.zero_grad(set_to_none=True)
                inputs = { 'input_ids': batch['input_ids'].to(device),
                           'attention_mask': batch['attention_mask'].to(device) }
                labels = batch['label'].to(device)
                weight = batch.get('weight', torch.ones_like(labels)).to(device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = model(**inputs)
                    # outputs = torch.clamp(outputs, 0.0, 1.0)  # [Commented] Predictions no longer forced to range
                    loss = weighted_huber_loss(outputs, labels, weight, delta=cfg['huber_delta']) if cfg['loss']=='huber' else weighted_mse_loss(outputs, labels, weight)
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['clip'])
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                pbar.set_postfix(loss=f"{loss.detach().float().item():.4f}")
                epoch_train_loss += loss.detach().float().item()
                del outputs, loss, labels, weight, inputs

            train_losses.append(epoch_train_loss / max(1, len(train_loader)))

            # validation: perform on each rank on its val subset; aggregate RMSE only on rank 0 by gathering across ranks
            model.eval()
            val_loss_total = 0.0
            with torch.no_grad():
                for batch in DataLoader(ds_val, batch_size=cfg['batch'], shuffle=False):
                    outputs = model(input_ids=batch['input_ids'].to(device),
                                    attention_mask=batch['attention_mask'].to(device))
                    # outputs = torch.clamp(outputs, 0.0, 1.0)  # [Commented] Predictions no longer forced to range
                    val_loss_total += F.mse_loss(outputs, batch['label'].to(device)).item()
            # compute val_loss averaged over all samples on this process, then reduce across processes to get global val_loss
            local_count = max(1, math.ceil(len(ds_val)/cfg['batch']))
            local_sum = torch.tensor(val_loss_total, device=device)
            local_count_t = torch.tensor(local_count, device=device)
            if world_size > 1:
                # sum across ranks
                dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(local_count_t, op=dist.ReduceOp.SUM)
            val_loss = (local_sum.item() / (local_count_t.item()+1e-12))
            val_losses.append(val_loss)

            # save checkpoint only on rank 0
            if rank == 0:
                try:
                    save_checkpoint({
                        'model': model.module.state_dict() if hasattr(model,'module') else model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scaler': scaler.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch,
                        'best_val': best_val,
                        'no_improve': no_improve
                    }, checkpoint_path, rank)
                except Exception as e:
                    print(f"[Checkpoint save failed] {checkpoint_path}: {e}")

            if val_loss + 1e-6 < best_val:
                best_val = val_loss; no_improve = 0
                if rank == 0:
                    try:
                        torch.save(model.module.state_dict() if hasattr(model, 'module') else model.state_dict(), best_path)
                    except Exception as e:
                        print(f"[Best model save failed] {best_path}: {e}")
            else:
                no_improve += 1
                if no_improve >= PATIENCE:
                    break

        # ensure at least one saved model exists (only rank 0 writes)
        if rank == 0 and not os.path.exists(best_path):
            try:
                torch.save(model.module.state_dict() if hasattr(model,'module') else model.state_dict(), best_path)
            except Exception as e:
                print(f"[Best model save failed] {best_path}: {e}")

        # Save loss curve on rank 0
        if rank == 0:
            plt.figure()
            plt.plot(train_losses, label='Train Loss')
            plt.plot(val_losses, label='Val Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Dim {dim} Fold {fold_id} Loss Curve')
            plt.legend()
            plt.savefig(os.path.join(fold_dir, 'loss_curve.png'))
            plt.close()

        return best_path, best_val if rank==0 else (best_path, best_val)
    except Exception as e:
        logging.error(f"Error in fit_one_fold (dim={dim}, fold={fold_id}): {e}", exc_info=True)
        return None, None

def run_dimension_kfold(dim, encoded_data_train, encoded_data_val, encoded_data_test, device, use_amp, scaler, rank=0, world_size=1, local_rank=0):
    try:
        best_loss = None
        best_cfg = None
        best_metrics = None
        # Try loss types - for cost, only rank 0 orchestrates selection, but all ranks run training for reproducibility
        for loss_type in ['mse', 'huber']:
            cfg = DIM_CFG[dim].copy()
            cfg['loss'] = loss_type
            if rank==0:
                print(f'Dimension {dim} trying loss: {loss_type}')
            # run single fold for quick evaluation (use entire sets indices for simple run)
            best_path, _ = safe_run(fit_one_fold, dim, 1, np.arange(len(encoded_data_train)), np.arange(len(encoded_data_val)), encoded_data_train+encoded_data_val, cfg, f'tmp_{dim}', device, use_amp, scaler, rank, world_size, local_rank, resume=False)
            if best_path is None:
                continue
            # evaluate on val set only by rank 0 to save work
            if rank==0:
                m = BertForMultiRegression(pretrained_model_name=MODEL_NAME, output_dim=1).to(device)
                m.load_state_dict(torch.load(best_path, map_location=device)); m.eval()
                val_loader_local = DataLoader(EmotionDatasetSingleDim(encoded_data_val, dim), batch_size=cfg['batch'], shuffle=False)
                preds=[]
                with torch.no_grad():
                    for b in val_loader_local:
                        o = m(input_ids=b['input_ids'].to(device), attention_mask=b['attention_mask'].to(device))
                        preds.append(torch.clamp(o, 0.0, 1.0).cpu())
                pred = torch.cat(preds, 0).view(-1).numpy()
                y_true = np.array([it['label'][dim].item() if isinstance(it['label'], torch.Tensor) else float(it['label'][dim]) for it in encoded_data_val])
                metrics = metrics_np(y_true, pred)
                if best_loss is None or metrics['rmse'] < best_loss:
                    best_loss = metrics['rmse']; best_cfg = cfg; best_metrics = metrics

        if rank==0:
            print(f'Dimension {dim} adopts optimal loss: {best_cfg["loss"]}, validation RMSE={best_loss:.4f}')
        # K-Fold training with best_cfg (only rank 0 saves artifacts)
        cfg = best_cfg
        out_dir = os.path.join(OUT_ROOT, f"dim_{dim}_kfold"); 
        if rank==0: os.makedirs(out_dir, exist_ok=True)
        base_list = encoded_data_train + encoded_data_val
        n = len(base_list)
        kf = KFold(n_splits=K, shuffle=True, random_state=42)
        oof = np.zeros(n, dtype=np.float32)
        fold_paths = []
        for fold_id, (tr_idx, va_idx) in enumerate(kf.split(np.arange(n)), start=1):
            fold_dir = os.path.join(out_dir, f'fold_{fold_id}')
            best_path = os.path.join(fold_dir, 'best_model.pth')
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f'dim{dim}_fold{fold_id}_ckpt.pth')
            # Checkpoint detection, skip if already completed
            if os.path.exists(best_path) and os.path.exists(checkpoint_path):
                print(f"[Resume] Fold {fold_id} of dim {dim} already completed, skipping.")
                fold_paths.append(best_path)
                if rank==0:
                    ds_val = Subset(EmotionDatasetSingleDim(base_list, dim), va_idx.tolist())
                    val_loader = DataLoader(ds_val, batch_size=cfg['batch'], shuffle=False)
                    m = BertForMultiRegression(pretrained_model_name=MODEL_NAME, output_dim=1).to(device)
                    m.load_state_dict(torch.load(best_path, map_location=device)); m.eval()
                    preds=[]
                    with torch.no_grad():
                        for b in val_loader:
                            o = m(input_ids=b['input_ids'].to(device), attention_mask=b['attention_mask'].to(device))
                            preds.append(torch.clamp(o, 0.0, 1.0).cpu())
                    oof[va_idx] = torch.cat(preds, 0).view(-1).numpy()
                continue
            try:
                best_path, _ = safe_run(fit_one_fold, dim, fold_id, tr_idx, va_idx, base_list, cfg, out_dir, device, use_amp, scaler, rank, world_size, local_rank)
            except Exception as e:
                print(f"Skip fold {fold_id} of dim {dim} due to error: {e}")
                logging.error(f"Skip fold {fold_id} of dim {dim} due to error.")
                continue
            if best_path is None:
                logging.error(f"Skip fold {fold_id} of dim {dim} due to error.")
                continue
            fold_paths.append(best_path)
            # only rank 0 loads model and compute fold val preds for OOF assembly
            if rank==0:
                ds_val = Subset(EmotionDatasetSingleDim(base_list, dim), va_idx.tolist())
                val_loader = DataLoader(ds_val, batch_size=cfg['batch'], shuffle=False)
                m = BertForMultiRegression(pretrained_model_name=MODEL_NAME, output_dim=1).to(device)
                m.load_state_dict(torch.load(best_path, map_location=device)); m.eval()
                preds=[]
                with torch.no_grad():
                    for b in val_loader:
                        o = m(input_ids=b['input_ids'].to(device), attention_mask=b['attention_mask'].to(device))
                        preds.append(torch.clamp(o, 0.0, 1.0).cpu())
                oof[va_idx] = torch.cat(preds, 0).view(-1).numpy()
        # Only rank 0 continues with evaluation/visualization
        if rank != 0:
            return None
        # OOF metrics
        y_true = np.vstack([(it['label'] if not isinstance(it['label'], torch.Tensor) else it['label'].cpu().numpy()) for it in base_list])[:, dim]
        oof_metrics = metrics_np(y_true, oof)
        # Test set prediction averaging across folds
        test_loader = DataLoader(EmotionDatasetSingleDim(encoded_data_test, dim), batch_size=cfg['batch'], shuffle=False)
        test_preds_all=[]
        for p in fold_paths:
            m = BertForMultiRegression(pretrained_model_name=MODEL_NAME, output_dim=1).to(device)
            m.load_state_dict(torch.load(p, map_location=device)); m.eval()
            preds=[]
            with torch.no_grad():
                for b in test_loader:
                    o = m(input_ids=b['input_ids'].to(device), attention_mask=b['attention_mask'].to(device))
                    preds.append(torch.clamp(o, 0.0, 1.0).cpu())
            test_preds_all.append(torch.cat(preds, 0).view(-1).numpy())
        test_pred_norm = np.mean(np.stack(test_preds_all, 0), 0) if len(test_preds_all)>0 else np.zeros(len(encoded_data_test))
        # Denorm and metrics
        y_min, y_max = np.load('y_min.npy'), np.load('y_max.npy')
        vmin, vmax = float(y_min[dim]), float(y_max[dim])
        test_true_norm = np.array([(it['label'][dim].item() if isinstance(it['label'], torch.Tensor) else float(it['label'][dim])) for it in encoded_data_test], dtype=np.float32)
        test_pred = denorm(test_pred_norm, vmin, vmax)
        test_true = denorm(test_true_norm, vmin, vmax)
        test_metrics = metrics_np(test_true, test_pred)
        # Visuals
        out_vis_dir = os.path.join(out_dir, 'visualization'); os.makedirs(out_vis_dir, exist_ok=True)
        plt.figure(); plt.scatter(test_true, test_pred, alpha=0.5)
        plt.xlabel('True'); plt.ylabel('Predicted'); plt.title(f'Dim {dim} Test: Predicted vs True')
        # Add: draw diagonal line
        min_v = min(test_true.min(), test_pred.min())
        max_v = max(test_true.max(), test_pred.max())
        plt.plot([min_v, max_v], [min_v, max_v], 'r--', linewidth=2)
        plt.savefig(os.path.join(out_vis_dir, 'pred_vs_true.png')); plt.close()
        plt.figure(); plt.hist(test_pred - test_true, bins=40, alpha=0.7); plt.xlabel('Residual (Pred - True)')
        plt.ylabel('Count'); plt.title(f'Dim {dim} Test: Residual Distribution')
        plt.savefig(os.path.join(out_vis_dir, 'residual_hist.png')); plt.close()
        # Save arrays and metrics
        np.save(os.path.join(out_dir,'oof_pred_norm.npy'),  oof)
        np.save(os.path.join(out_dir,'oof_true_norm.npy'),  y_true)
        np.save(os.path.join(out_dir,'test_pred_norm.npy'), test_pred_norm)
        np.save(os.path.join(out_dir,'test_true_norm.npy'), test_true_norm)
        np.save(os.path.join(out_dir,'test_pred.npy'),      test_pred)
        np.save(os.path.join(out_dir,'test_true.npy'),      test_true)
        with open(os.path.join(out_dir,'metrics.json'),'w',encoding='utf-8') as f:
            json.dump({'oof_norm': oof_metrics, 'test': test_metrics, 'K': K,
                       'model_name': MODEL_NAME, **cfg}, f, ensure_ascii=False, indent=2)
        try: shutil.make_archive(os.path.join(OUT_ROOT, f'dim_{dim}_kfold_artifacts'), 'zip', out_dir)
        except Exception: pass
        return {'dim': dim, 'oof_rmse': float(oof_metrics['rmse']),
                'test_rmse': float(test_metrics['rmse']), 'out_dir': out_dir}
    except Exception as e:
        logging.error(f"Error in run_dimension_kfold (dim={dim}): {e}", exc_info=True)
        return None

# Add: label distribution statistics function
def print_label_stats(df, name):
    arr = np.array(df['label'].tolist())
    print(f"[Label Distribution] {name} shape={arr.shape}")
    for d in range(arr.shape[1]):
        col = arr[:, d]
        print(f"  Dim {d}: min={np.nanmin(col):.4f}, max={np.nanmax(col):.4f}, mean={np.nanmean(col):.4f}, nan={np.isnan(col).sum()}")

def balance_high_value_samples(df, tokenizer, dim=0, bins=10, target_count=None, enhance_prob=0.5):
    """
    Oversample and enhance samples in high True value intervals to make label distribution more uniform.
    - df: Input DataFrame
    - tokenizer: Tokenizer for text enhancement
    - dim: Target dimension
    - bins: Number of intervals
    - target_count: Target sample count per interval (default: max interval sample count)
    - enhance_prob: Copy/enhancement probability
    Return new DataFrame
    """
    arr = np.array(df['label'].tolist())
    col = arr[:, dim]
    hist, edges = np.histogram(col, bins=bins)
    if target_count is None:
        target_count = hist.max()
    dfs = [df]
    for i in range(bins):
        idx = (col >= edges[i]) & (col < edges[i+1] if i < bins-1 else col <= edges[i+1])
        sub = df[idx]
        n = len(sub)
        if n == 0: continue
        # Only enhance high-value intervals (e.g., last 1/3)
        if i >= bins * 2 // 3:
            repeat = int(np.ceil(target_count / n))
            # Copy and enhance
            for _ in range(repeat-1):
                sub_aug = sub.copy()
                # Mark enhancement
                sub_aug['content'] = sub_aug['content'].apply(lambda x: random_mask_text(x, tokenizer) if random.random() < enhance_prob else x)
                if aug_syn is not None:
                    sub_aug['content'] = sub_aug['content'].apply(lambda x: synonym_augment(x) if random.random() < enhance_prob else x)
                dfs.append(sub_aug)
    df_new = pd.concat(dfs, ignore_index=True).sample(frac=1).reset_index(drop=True)
    return df_new


# 3. Small-scale experiment main flow
if __name__ == "__main__":
    print("Script started")
    rank, world_size, local_rank = setup_distributed()
    print(f"[Rank {rank}] Distributed setup done. world_size={world_size}, local_rank={local_rank}")

    # 1. Load full data
    print("Loading data...")
    df = pd.read_csv("data__/train_data.csv", engine='python', on_bad_lines='skip')
    if isinstance(df['label'].iloc[0], str):
        import ast
        df['label'] = df['label'].apply(ast.literal_eval)
    print(f"Loaded {len(df)} samples.")

    # 2. Load model
    MODEL_NAME = "/root/autodl-tmp/WorkPlace/models/microsoft--deberta-v3-base"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    bert = AutoModel.from_pretrained(MODEL_NAME)
    hidden_size = bert.config.hidden_size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bert = bert.to(device)
    for p in bert.parameters():
        p.requires_grad = False  # Freeze BERT

    # 3. Multi-dimensional loop
    for dim in range(8):
        print(f"=== Training Dim {dim} ===")

        # 3.1 Data augmentation and balancing
        print(f"[Dim {dim}] Applying data augmentation and balancing...")
        df_aug = balance_high_value_samples(df, tokenizer, dim=dim, bins=10, enhance_prob=0.3)
        # Add is_synth column to mark original and enhanced samples
        df_aug['is_synth'] = 0  # Mark original samples as 0
        # Mark enhanced samples as 1 (simplified here, should be more precise in practice)
        original_len = len(df)
        if len(df_aug) > original_len:
            df_aug.loc[original_len:, 'is_synth'] = 1
        print(f"[Dim {dim}] Data augmentation done. Original: {len(df)}, Augmented: {len(df_aug)}")

        # 3.2 Encoding
        print(f"[Dim {dim}] Encoding data...")
        inputs = tokenizer(df_aug['content'].tolist(), truncation=True, padding='max_length', max_length=128, return_tensors='pt')
        y_tensor = torch.tensor([l[dim] for l in df_aug['label']], dtype=torch.float).unsqueeze(1)
        is_synth = df_aug['is_synth'].values.astype(np.float32)
        print(f"[Dim {dim}] Encoding done.")

        # 3.3 Train MoE Head
        print(f"[Dim {dim}] Start MoE head training (soft split, weighted loss, all samples)...")
        moe_head = MoERegressionHead(hidden_size).to(device)
        optimizer = torch.optim.AdamW(list(moe_head.parameters()), lr=1e-4)
        batch_size = 8
        losses, gate_means, gate_stds, all_gates = [], [], [], []
        y = np.array([l[dim] for l in df_aug['label']])
        y = np.clip(y, 0, 3)
        low_q = np.quantile(y, 0.3)
        high_q = np.quantile(y, 0.7)
        for epoch in range(8):
            print(f"[Dim {dim}][Epoch {epoch+1}] MoE head training...")
            idx = np.random.permutation(len(df_aug))
            for i in range(0, len(df_aug), batch_size):
                bidx = idx[i:i+batch_size]
                input_ids = inputs['input_ids'][bidx].to(device)
                attn_mask = inputs['attention_mask'][bidx].to(device)
                y_true = y_tensor[bidx].to(device)
                synth_flag = torch.tensor(is_synth[bidx], dtype=torch.float, device=device)
                with torch.no_grad():
                    h = bert(input_ids=input_ids, attention_mask=attn_mask, return_dict=True).last_hidden_state[:, 0]
                y_pred, y_low, y_high, g = moe_head(h)
                y_true_flat = y_true.view(-1)
                synth_flag_flat = synth_flag.view(-1)
                weight = torch.ones_like(y_true_flat)
                mask_extreme = synth_flag_flat > 0.5
                weight[mask_extreme] = 30.0
                y_pred_flat = y_pred.view(-1)
                y_low_flat = y_low.view(-1)
                y_high_flat = y_high.view(-1)
                g_flat = g.view(-1, 1)
                main_loss = (F.mse_loss(y_pred_flat, y_true_flat, reduction='none') * weight).mean()
                z = ((y_true_flat > low_q + 0.2) & (y_true_flat < high_q - 0.2)).float() * 0.5 + (y_true_flat >= high_q - 0.2).float()
                focus_loss = ((1-z) * F.mse_loss(y_low_flat, y_true_flat, reduction='none') + z * F.mse_loss(y_high_flat, y_true_flat, reduction='none')) * weight
                focus_loss = focus_loss.mean()
                entropy = - (g_flat * torch.log(g_flat + 1e-8) + (1-g_flat) * torch.log(1-g_flat + 1e-8)).mean()
                loss = main_loss + 5.0 * focus_loss - 2.0 * entropy
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(moe_head.parameters(), max_norm=1.0)
                optimizer.step()
                losses.append(loss.item())
                gate_means.append(g.mean().item())
                gate_stds.append(g.std().item())
                all_gates.extend(g.detach().cpu().numpy().flatten().tolist())
            print(f"[Dim {dim}][Epoch {epoch+1}] last batch loss={loss.item():.4f}, gate_mean={g.mean().item():.4f}, gate_std={g.std().item():.4f}")
        print(f"[Dim {dim}] MoE head training finished.")

        # 3.4 Evaluation and visualization (main process only)
        if rank == 0:
            print(f"[Dim {dim}] Evaluating and saving plots...")
            with torch.no_grad():
                h = bert(input_ids=inputs['input_ids'].to(device), attention_mask=inputs['attention_mask'].to(device), return_dict=True).last_hidden_state[:, 0]
                y_pred, _, _, _ = moe_head(h)
                y_pred = y_pred.cpu().numpy().flatten()
                y_true = np.array([l[dim] for l in df_aug['label']])
                residual = y_true - y_pred
                import matplotlib.pyplot as plt
                # Residual plot
                plt.figure(figsize=(7,6))
                plt.scatter(y_true, residual, alpha=0.5)
                plt.axhline(0, color='r', linestyle='--', linewidth=2)
                plt.xlabel('True')
                plt.ylabel('Residual (True - Pred)')
                plt.title(f'MoE Residual vs True (Dim {dim})')
                plt.tight_layout()
                plt.savefig(f'moe_residual_vs_true_dim{dim}.png')
                plt.close()
                # Training curve
                plt.figure(figsize=(12,4))
                plt.subplot(1,3,1)
                plt.plot(losses)
                plt.title('Loss')
                plt.subplot(1,3,2)
                plt.plot(gate_means)
                plt.title('Gate Mean')
                plt.subplot(1,3,3)
                plt.plot(gate_stds)
                plt.title('Gate Std')
                plt.tight_layout()
                plt.savefig(f'moe_training_curve_dim{dim}.png')
                plt.close()
                # Gate distribution
                plt.figure(figsize=(5,4))
                plt.hist(all_gates, bins=30, alpha=0.7)
                plt.xlabel('Gate Value')
                plt.ylabel('Count')
                plt.title('Gate Distribution (All Batches)')
                plt.savefig(f'moe_gate_hist_dim{dim}.png')
                plt.close()
                # Log
                import pandas as pd
                pd.DataFrame({'loss': losses, 'gate_mean': gate_means, 'gate_std': gate_stds}).to_csv(f'moe_training_log_dim{dim}.csv', index=False)
            print(f"[Dim {dim}] All results saved.")
        torch.cuda.empty_cache()

    print("All dimensions finished.")
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        print("Destroying process group...")
        dist.destroy_process_group()
        print("Process group destroyed。")

