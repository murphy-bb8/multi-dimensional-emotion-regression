import os, json, shutil, math, ast, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import KFold
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import logging
import seaborn as sns
plt.rcParams.update({'font.size': 14, 'axes.labelweight': 'bold', 'axes.titlesize': 16, 'axes.labelsize': 14, 'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12})
logging.basicConfig(filename='train_error.log', level=logging.ERROR)

def safe_run(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logging.error(f"Error in {func.__name__}: {e}", exc_info=True)
        return None

# For synonym augmentation, install: pip install nlpaug
try:
    import nlpaug.augmenter.word as naw
    aug_syn = naw.SynonymAug(aug_src='wordnet')
except ImportError:
    aug_syn = None

warnings.filterwarnings("ignore")

# Checkpoint resumption related
CHECKPOINT_DIR = 'checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Configuration
MODEL_NAME = "/root/autodl-tmp/WorkPlace/models/microsoft--deberta-v3-base"
MAX_LEN = 128
K = 5
PATIENCE = 3
ALPHA_DENS = 1.5
OUT_ROOT = "outputs_8x1"

# Data augmentation switch
AUGMENT_DATA = True
AUGMENT_PROB = 0.1  # Reduce augmentation probability
MASK_TOKEN = '[MASK]'
AUGMENT_USE_SYNONYM = False  # Only keep random masking

# English synonym replacement augmentation

def synonym_augment(text):
    if aug_syn is not None:
        return aug_syn.augment(text)
    return text

# Simple random masking augmentation

def random_mask_text(text, tokenizer, prob=0.15):
    words = text.split()
    n = len(words)
    if n < 3: return text
    mask_num = max(1, int(n * prob))
    mask_idx = random.sample(range(n), mask_num)
    for idx in mask_idx:
        words[idx] = MASK_TOKEN
    return ' '.join(words)

# Optimized hyperparameters
DIM_CFG = {
    i: dict(loss='huber', huber_delta=0.6, epochs=15, batch=16, lr_bert=5e-6, lr_head=1e-5, wd=0.5, warmup=0.15, clip=1.0, freeze_epochs=5)
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

def encode(df, tokenizer):
    out=[]
    for _, r in df.iterrows():
        content = r['content']
        # Data augmentation
        if AUGMENT_DATA and random.random() < AUGMENT_PROB:
            content = random_mask_text(content, tokenizer)
        enc = tokenizer(content, truncation=True, padding='max_length', max_length=MAX_LEN, return_tensors='pt')
        out.append({'input_ids': enc['input_ids'].squeeze(0),
                    'attention_mask': enc['attention_mask'].squeeze(0),
                    'label': torch.tensor(r['label'], dtype=torch.float)})
    return out

def save_checkpoint(state, filename):
    torch.save(state, filename)
def load_checkpoint(filename, device):
    return torch.load(filename, map_location=device)

def fit_one_fold(dim, fold_id, train_idx, val_idx, base_list, cfg, dim_out, device, use_amp, scaler, resume=True):
    try:
        w_all = compute_dim_density_weights_from_encoded(base_list, dim)
        ds_full  = EmotionDatasetSingleDim(base_list, dim)
        ds_train = Subset(EmotionDatasetSingleDim(base_list, dim, ext_weights=w_all), train_idx.tolist())
        ds_val   = Subset(ds_full,  val_idx.tolist())
        train_loader = DataLoader(ds_train, batch_size=cfg['batch'], shuffle=True, num_workers=4, pin_memory=True)
        val_loader   = DataLoader(ds_val,   batch_size=cfg['batch'], shuffle=False, num_workers=4, pin_memory=True)

        model = BertForMultiRegression(pretrained_model_name=MODEL_NAME, output_dim=1).to(device)
        # Multi-GPU support
        if torch.cuda.device_count() > 1:
            print(f'Using {torch.cuda.device_count()} GPUs for DataParallel parallel training')
            model = torch.nn.DataParallel(model)
        # Freeze first 10 layers
        try:
            backbone = model.module.backbone if hasattr(model, 'module') else model.backbone
            for name, param in backbone.named_parameters():
                if any([f'layer.{i}.' in name for i in range(10)]):
                    param.requires_grad = False
        except Exception as e:
            print('Failed to freeze first 10 layers:', e)

        bb_params, head_params = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad: continue
            (bb_params if name.startswith('backbone.') else head_params).append(p)
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
        fold_dir = os.path.join(dim_out, f'fold_{fold_id}'); os.makedirs(fold_dir, exist_ok=True)
        best_path = os.path.join(fold_dir, 'best_model.pth')
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f'dim{dim}_fold{fold_id}_ckpt.pth')
        start_epoch = 0
        # Checkpoint resumption
        if resume and os.path.exists(checkpoint_path):
            print(f'Checkpoint detected, resuming from {checkpoint_path}')
            ckpt = load_checkpoint(checkpoint_path, device)
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            scaler.load_state_dict(ckpt['scaler'])
            scheduler.load_state_dict(ckpt['scheduler'])
            start_epoch = ckpt['epoch'] + 1
            best_val = ckpt.get('best_val', best_val)
            no_improve = ckpt.get('no_improve', 0)
        train_losses, val_losses = [], []
        for epoch in range(start_epoch, cfg['epochs']):
            # Print GPU utilization and memory usage
            try:
                import subprocess
                gpu_info = subprocess.check_output('nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv', shell=True).decode()
                print(f"[GPU INFO] Epoch {epoch+1}:\n" + gpu_info)
            except Exception as e:
                print(f"[GPU INFO] Failed to get info: {e}")
            if epoch < cfg['freeze_epochs']:
                backbone = model.module.backbone if hasattr(model, 'module') else model.backbone
                for p in backbone.parameters(): p.requires_grad = False
            else:
                backbone = model.module.backbone if hasattr(model, 'module') else model.backbone
                for p in backbone.parameters(): p.requires_grad = True
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
                    outputs = torch.clamp(outputs, 0.0, 1.0)
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
            # Validation
            model.eval()
            val_loss_total = 0.0
            with torch.no_grad():
                for batch in DataLoader(ds_val, batch_size=cfg['batch'], shuffle=False):
                    outputs = model(input_ids=batch['input_ids'].to(device),
                                    attention_mask=batch['attention_mask'].to(device))
                    outputs = torch.clamp(outputs, 0.0, 1.0)
                    val_loss_total += F.mse_loss(outputs, batch['label'].to(device)).item()
            val_loss = val_loss_total / max(1, math.ceil(len(ds_val)/cfg['batch']))
            val_losses.append(val_loss)
            # Save checkpoint
            save_checkpoint({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'best_val': best_val,
                'no_improve': no_improve
            }, checkpoint_path)
            # Save/load model compatibility with DataParallel
            # Save: torch.save(model.module.state_dict() if hasattr(model, 'module') else model.state_dict(), best_path)
            # Load: model.module.load_state_dict(...) if hasattr(model, 'module') else model.load_state_dict(...)
            if val_loss + 1e-6 < best_val:
                best_val = val_loss; no_improve = 0; torch.save(model.module.state_dict() if hasattr(model, 'module') else model.state_dict(), best_path)
            else:
                no_improve += 1
                if no_improve >= PATIENCE: break
        # Fallback save, ensure at least one model is saved
        if not os.path.exists(best_path):
            torch.save(model.module.state_dict() if hasattr(model, 'module') else model.state_dict(), best_path)
        # Save loss curve
        plt.figure()
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Dim {dim} Fold {fold_id} Loss Curve')
        plt.legend()
        plt.savefig(os.path.join(fold_dir, 'loss_curve.png'))
        plt.close()
        return best_path, best_val
    except Exception as e:
        logging.error(f"Error in fit_one_fold (dim={dim}, fold={fold_id}): {e}", exc_info=True)
        return None, None

def run_dimension_kfold(dim, encoded_data_train, encoded_data_val, encoded_data_test, device, use_amp, scaler):
    try:
        # Automatically select optimal loss function
        best_loss = None
        best_cfg = None
        best_metrics = None
        for loss_type in ['mse', 'huber']:
            cfg = DIM_CFG[dim].copy()
            cfg['loss'] = loss_type
            print(f'Dimension {dim} trying loss: {loss_type}')
            best_path, _ = safe_run(fit_one_fold, dim, 1, np.arange(len(encoded_data_train)), np.arange(len(encoded_data_val)), encoded_data_train+encoded_data_val, cfg, f'tmp_{dim}', device, use_amp, scaler, resume=False)
            # Validation set evaluation
            m = BertForMultiRegression(pretrained_model_name=MODEL_NAME, output_dim=1).to(device)
            # At model loading:
            # m = BertForMultiRegression(...)
            # if torch.cuda.device_count() > 1: m = torch.nn.DataParallel(m)
            # m.load_state_dict(torch.load(...))
            m.load_state_dict(torch.load(best_path, map_location=device)); m.eval()
            val_loader = DataLoader(EmotionDatasetSingleDim(encoded_data_val, dim), batch_size=cfg['batch'], shuffle=False)
            preds=[]
            with torch.no_grad():
                for b in val_loader:
                    o = m(input_ids=b['input_ids'].to(device), attention_mask=b['attention_mask'].to(device))
                    preds.append(torch.clamp(o, 0.0, 1.0).cpu())
            pred = torch.cat(preds, 0).view(-1).numpy()
            y_true = np.array([it['label'][dim].item() if isinstance(it['label'], torch.Tensor) else float(it['label'][dim]) for it in encoded_data_val])
            metrics = metrics_np(y_true, pred)
            if best_loss is None or metrics['rmse'] < best_loss:
                best_loss = metrics['rmse']
                best_cfg = cfg
                best_metrics = metrics
        print(f'Dimension {dim} adopts optimal loss: {best_cfg["loss"]}, validation RMSE={best_loss:.4f}')
        # Use optimal loss for re-KFold
        cfg = best_cfg
        out_dir = os.path.join(OUT_ROOT, f"dim_{dim}_kfold"); os.makedirs(out_dir, exist_ok=True)
        base_list = encoded_data_train + encoded_data_val
        n = len(base_list)
        kf = KFold(n_splits=K, shuffle=True, random_state=42)
        oof = np.zeros(n, dtype=np.float32)
        fold_paths = []
        for fold_id, (tr_idx, va_idx) in enumerate(kf.split(np.arange(n)), start=1):
            best_path, _ = safe_run(fit_one_fold, dim, fold_id, tr_idx, va_idx, base_list, cfg, out_dir, device, use_amp, scaler)
            if best_path is None:
                logging.error(f"Skip fold {fold_id} of dim {dim} due to error.")
                continue
            fold_paths.append(best_path)
            ds_val = Subset(EmotionDatasetSingleDim(base_list, dim), va_idx.tolist())
            val_loader = DataLoader(ds_val, batch_size=cfg['batch'], shuffle=False)
            m = BertForMultiRegression(pretrained_model_name=MODEL_NAME, output_dim=1).to(device)
            # At model loading:
            # m = BertForMultiRegression(...)
            # if torch.cuda.device_count() > 1: m = torch.nn.DataParallel(m)
            # m.load_state_dict(torch.load(...))
            m.load_state_dict(torch.load(best_path, map_location=device)); m.eval()
            preds=[]
            with torch.no_grad():
                for b in val_loader:
                    o = m(input_ids=b['input_ids'].to(device), attention_mask=b['attention_mask'].to(device))
                    preds.append(torch.clamp(o, 0.0, 1.0).cpu())
            oof[va_idx] = torch.cat(preds, 0).view(-1).numpy()
        # OOF metrics
        y_true = np.vstack([(it['label'] if not isinstance(it['label'], torch.Tensor) else it['label'].cpu().numpy()) for it in base_list])[:, dim]
        oof_metrics = metrics_np(y_true, oof)
        # Test set fold averaging
        test_loader = DataLoader(EmotionDatasetSingleDim(encoded_data_test, dim), batch_size=cfg['batch'], shuffle=False)
        test_preds_all=[]
        for p in fold_paths:
            m = BertForMultiRegression(pretrained_model_name=MODEL_NAME, output_dim=1).to(device)
            # At model loading:
            # m = BertForMultiRegression(...)
            # if torch.cuda.device_count() > 1: m = torch.nn.DataParallel(m)
            # m.load_state_dict(torch.load(...))
            m.load_state_dict(torch.load(p, map_location=device)); m.eval()
            preds=[]
            with torch.no_grad():
                for b in test_loader:
                    o = m(input_ids=b['input_ids'].to(device), attention_mask=b['attention_mask'].to(device))
                    preds.append(torch.clamp(o, 0.0, 1.0).cpu())
            test_preds_all.append(torch.cat(preds, 0).view(-1).numpy())
        test_pred_norm = np.mean(np.stack(test_preds_all, 0), 0)
        # Denormalize and evaluate
        y_min, y_max = np.load('y_min.npy'), np.load('y_max.npy')
        vmin, vmax = float(y_min[dim]), float(y_max[dim])
        test_true_norm = np.array([(it['label'][dim].item() if isinstance(it['label'], torch.Tensor) else float(it['label'][dim])) for it in encoded_data_test], dtype=np.float32)
        test_pred = denorm(test_pred_norm, vmin, vmax)
        test_true = denorm(test_true_norm, vmin, vmax)
        test_metrics = metrics_np(test_true, test_pred)
        # Visualization: predicted vs true, residual distribution
        out_vis_dir = os.path.join(out_dir, 'visualization'); os.makedirs(out_vis_dir, exist_ok=True)
        # Predicted vs true scatter plot
        plt.figure()
        plt.scatter(test_true, test_pred, alpha=0.5)
        plt.xlabel('True')
        plt.ylabel('Predicted')
        plt.title(f'Dim {dim} Test: Predicted vs True')
        plt.savefig(os.path.join(out_vis_dir, 'pred_vs_true.png'))
        plt.close()
        # Residual distribution
        plt.figure()
        plt.hist(test_pred - test_true, bins=40, alpha=0.7)
        plt.xlabel('Residual (Pred - True)')
        plt.ylabel('Count')
        plt.title(f'Dim {dim} Test: Residual Distribution')
        plt.savefig(os.path.join(out_vis_dir, 'residual_hist.png'))
        plt.close()
        # Save
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

if __name__ == "__main__":
    # Load data
    train_df = pd.read_csv("data__/train_data.csv", engine='python', on_bad_lines='skip')
    val_df   = pd.read_csv("data__/val_data.csv",   engine='python', on_bad_lines='skip')
    test_df  = pd.read_csv("data__/test_data.csv",  engine='python', on_bad_lines='skip')
    for df_ in (train_df, val_df, test_df):
        df_.dropna(subset=['label'], inplace=True)
        df_.reset_index(drop=True, inplace=True)
        if isinstance(df_['label'].iloc[0], str):
            df_['label'] = df_['label'].apply(ast.literal_eval)
    # Label distribution analysis and outlier removal
    def remove_outliers(df, bins=20, low_q=0.01, high_q=0.99):
        try:
            labels = np.array(df['label'].tolist())
            keep = np.ones(len(df), dtype=bool)
            for dim in range(labels.shape[1]):
                col = labels[:, dim]
                q_low, q_high = np.quantile(col, [low_q, high_q])
                keep &= (col >= q_low) & (col <= q_high)
            df = df[keep].reset_index(drop=True)
            assert not df['label'].isnull().any(), 'NaN in label after outlier removal'
            return df
        except Exception as e:
            logging.error(f"Error in remove_outliers: {e}", exc_info=True)
            return df
    train_df = remove_outliers(train_df)
    val_df = remove_outliers(val_df)
    # Normalization
    y_min, y_max = get_min_max(train_df['label'])
    np.save('y_min.npy', y_min); np.save('y_max.npy', y_max)
    train_df['label'] = normalize(train_df['label'], y_min, y_max)
    val_df['label']   = normalize(val_df['label'],   y_min, y_max)
    test_df['label']  = normalize(test_df['label'],  y_min, y_max)
    print("✅ Data normalization and outlier removal completed:", len(train_df), len(val_df), len(test_df))
    # Encoding
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    encoded_data_train = encode(train_df, tokenizer)
    encoded_data_val   = encode(val_df, tokenizer)
    encoded_data_test  = encode(test_df, tokenizer)
    print("✅ Encoding completed:", len(encoded_data_train), len(encoded_data_val), len(encoded_data_test))
    # Training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    os.makedirs(OUT_ROOT, exist_ok=True)
    results=[]
    all_test_pred = []
    all_test_true = []
    try:
        for d in range(8): 
            try:
                res = safe_run(run_dimension_kfold, d, encoded_data_train, encoded_data_val, encoded_data_test, device, use_amp, scaler)
                if res is not None:
                    results.append(res)
                    print(f"Dimension {d}: OOF_RMSE={res['oof_rmse']:.6f} | TEST_RMSE={res['test_rmse']:.6f} | {res['out_dir']}")
            except Exception as e:
                logging.error(f"Error in main loop for dim {d}: {e}", exc_info=True)
                continue
        # Only keep summary_kfold.json
        with open(os.path.join(OUT_ROOT, 'summary_kfold.json'), 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        # Generate main visualization images
        # 1. Scatter plot (True vs Predicted)
        all_test_pred = []
        all_test_true = []
        for res in results:
            test_pred_path = os.path.join(res['out_dir'], 'test_pred.npy')
            test_true_path = os.path.join(res['out_dir'], 'test_true.npy')
            all_test_pred.append(np.load(test_pred_path))
            all_test_true.append(np.load(test_true_path))
        # Main flow add non-empty check before np.stack
        if len(all_test_pred) == 0 or len(all_test_true) == 0:
            print('❌ No valid prediction results generated, training process interrupted. Please check the logs above.')
            exit(1)  # or raise SystemExit(1)
        all_test_pred = np.stack(all_test_pred, axis=1)
        all_test_true = np.stack(all_test_true, axis=1)
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        for i in range(8):
            ax = axes[i//4, i%4]
            ax.scatter(all_test_true[:, i], all_test_pred[:, i], alpha=0.5)
            ax.plot([all_test_true[:, i].min(), all_test_true[:, i].max()], [all_test_true[:, i].min(), all_test_true[:, i].max()], 'r--')
            ax.set_xlabel('True')
            ax.set_ylabel('Predicted')
            ax.set_title(f'Dim {i}')
        plt.suptitle('Scatter Plots of True vs Predicted (Per Dimension)')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(OUT_ROOT, 'scatter_true_vs_pred.png'))
        plt.close()
        # 2. Loss curve (using last fold as example)
        loss_curve_path = os.path.join(results[-1]['out_dir'], 'loss_curve.png')
        if os.path.exists(loss_curve_path):
            import shutil
            shutil.copy(loss_curve_path, os.path.join(OUT_ROOT, 'loss_curve.png'))
        # 3. MSE bar chart
        mse_list = [r['oof_rmse']**2 for r in results]
        plt.figure(figsize=(8, 6))
        plt.bar(range(1, 9), mse_list, color='skyblue', alpha=0.8)
        plt.xlabel('Dimension')
        plt.ylabel('MSE')
        plt.title('Per-Dimension MSE')
        plt.savefig(os.path.join(OUT_ROOT, 'mse_per_dim.png'))
        plt.close()
        # 4. Correlation heatmap
        corr = np.corrcoef(all_test_pred.T)
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', square=True)
        plt.title('Predicted Emotion Dimensions Correlation Heatmap')
        plt.savefig(os.path.join(OUT_ROOT, 'correlation_heatmap.png'))
        plt.close()
        # 5. Error distribution
        fig, axes = plt.subplots(2, 4, figsize=(20, 8))
        for i in range(8):
            ax = axes[i//4, i%4]
            err = all_test_true[:, i] - all_test_pred[:, i]
            ax.hist(err, bins=50, alpha=0.5, color='skyblue')
            ax.set_title(f'Dim {i} Error Distribution')
            ax.set_xlabel('True - Predicted')
            ax.set_ylabel('Sample Count')
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_ROOT, 'error_distribution.png'))
        plt.close()
        # 6. Sample comparison bar chart (first 5 samples)
        plt.figure(figsize=(10, 12))
        for idx in range(5):
            plt.subplot(5, 1, idx+1)
            plt.bar(np.arange(8)-0.2, all_test_true[idx], width=0.4, label='True', color='skyblue')
            plt.bar(np.arange(8)+0.2, all_test_pred[idx], width=0.4, label='Predicted', color='salmon')
            plt.xticks(np.arange(8), [f'Dim {i}' for i in range(8)])
            plt.ylim(0, 1)
            plt.title(f'Sample #{idx} Emotion Prediction Comparison')
            if idx == 0:
                plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_ROOT, 'sample_comparison.png'))
        plt.close()
        print('✅ Only keep summary_kfold.json and main visualization images, output directory:', OUT_ROOT)
    except Exception as e:
        logging.error(f"Error in main: {e}", exc_info=True)
