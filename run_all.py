"""
Run all 32 TSM2 experiments efficiently.
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler

from tsm2_model import TSM2
from data_loader import get_dataloaders, DATASET_INFO


def train_and_eval(dataset_name, pred_len, device, max_time=600):
    """Train one experiment with time budget."""
    print(f"\n{'='*60}")
    print(f"Training: {dataset_name}, H={pred_len}")
    print(f"{'='*60}")
    
    num_variates = DATASET_INFO[dataset_name]['num_variates']
    
    # Dataset-specific configs - tuned for good generalization
    # Key insight: these datasets need careful regularization to avoid overfitting
    configs = {
        'ETTh1':         {'d_model': 64, 'layers': 2, 'expand': 1, 'lr': 5e-4, 'bs': 32, 'epochs': 30, 'patience': 10, 'wd': 0.01, 'dropout': 0.3},
        'ETTh2':         {'d_model': 64, 'layers': 2, 'expand': 1, 'lr': 5e-4, 'bs': 32, 'epochs': 30, 'patience': 10, 'wd': 0.01, 'dropout': 0.3},
        'ETTm1':         {'d_model': 64, 'layers': 2, 'expand': 1, 'lr': 5e-4, 'bs': 64, 'epochs': 20, 'patience': 7, 'wd': 0.01, 'dropout': 0.3},
        'ETTm2':         {'d_model': 64, 'layers': 2, 'expand': 1, 'lr': 5e-4, 'bs': 64, 'epochs': 20, 'patience': 7, 'wd': 0.01, 'dropout': 0.3},
        'exchange_rate':  {'d_model': 64, 'layers': 2, 'expand': 1, 'lr': 3e-4, 'bs': 32, 'epochs': 30, 'patience': 10, 'wd': 0.01, 'dropout': 0.3},
        'weather':       {'d_model': 64, 'layers': 2, 'expand': 1, 'lr': 5e-4, 'bs': 64, 'epochs': 15, 'patience': 5, 'wd': 0.01, 'dropout': 0.2},
        'electricity':   {'d_model': 32, 'layers': 2, 'expand': 1, 'lr': 3e-4, 'bs': 16, 'epochs': 10, 'patience': 4, 'wd': 0.01, 'dropout': 0.2},
        'traffic':       {'d_model': 32, 'layers': 2, 'expand': 1, 'lr': 3e-4, 'bs': 8,  'epochs': 10, 'patience': 4, 'wd': 0.01, 'dropout': 0.2},
    }
    
    cfg = configs[dataset_name]
    
    # Data
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset_name, seq_len=512, pred_len=pred_len,
        batch_size=cfg['bs'], num_workers=4,
    )
    
    # Model
    model = TSM2(
        num_variates=num_variates, seq_len=512, pred_len=pred_len,
        patch_len=16, stride=8, d_model=cfg['d_model'],
        d_state=16, d_conv=4, expand=cfg['expand'],
        num_layers=cfg['layers'], dropout=cfg['dropout'],
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}, Batches: {len(train_loader)}")
    
    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg['epochs'], eta_min=1e-6)
    scaler = GradScaler()
    
    ckpt_path = f'checkpoints/{dataset_name}_H{pred_len}.pt'
    os.makedirs('checkpoints', exist_ok=True)
    
    best_val = float('inf')
    patience_counter = 0
    start_time = time.time()
    
    for epoch in range(cfg['epochs']):
        # Training
        model.train()
        train_loss = 0
        n_batches = 0
        for bx, by in train_loader:
            bx = bx.to(device, non_blocking=True)
            by = by.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                pred = model(bx)
                loss = criterion(pred, by)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            n_batches += 1
        
        train_loss /= max(n_batches, 1)
        scheduler.step()
        
        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                bx = bx.to(device, non_blocking=True)
                by = by.to(device, non_blocking=True)
                with autocast():
                    pred = model(bx)
                val_preds.append(pred.float().cpu())
                val_targets.append(by.cpu())
        
        val_preds = torch.cat(val_preds)
        val_targets = torch.cat(val_targets)
        val_mse = ((val_preds - val_targets)**2).mean().item()
        
        elapsed = time.time() - start_time
        print(f"  Epoch {epoch+1:2d}/{cfg['epochs']} | Train: {train_loss:.6f} | Val MSE: {val_mse:.6f} | Best: {best_val:.6f} | {elapsed:.0f}s")
        
        if val_mse < best_val:
            best_val = val_mse
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_counter += 1
        
        if patience_counter >= cfg['patience']:
            print(f"  Early stopping at epoch {epoch+1}")
            break
        
        if elapsed > max_time:
            print(f"  Time limit reached ({max_time}s)")
            break
    
    # Test
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    test_preds, test_targets = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device, non_blocking=True)
            by = by.to(device, non_blocking=True)
            with autocast():
                pred = model(bx)
            test_preds.append(pred.float().cpu())
            test_targets.append(by.cpu())
    
    test_preds = torch.cat(test_preds)
    test_targets = torch.cat(test_targets)
    mse = ((test_preds - test_targets)**2).mean().item()
    mae = (test_preds - test_targets).abs().mean().item()
    
    total_time = time.time() - start_time
    print(f"  >>> Test MSE: {mse:.6f}, MAE: {mae:.6f} (total: {total_time:.0f}s)")
    
    return {'mse': mse, 'mae': mae, 'time': total_time}


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    results_path = 'results/results.json'
    os.makedirs('results', exist_ok=True)
    
    if os.path.exists(results_path):
        with open(results_path) as f:
            all_results = json.load(f)
    else:
        all_results = {}
    
    # All experiments - small datasets first
    experiments = []
    for ds in ['ETTh1', 'ETTh2', 'exchange_rate', 'ETTm1', 'ETTm2', 'weather', 'electricity', 'traffic']:
        for h in [96, 192, 336, 720]:
            experiments.append((ds, h))
    
    # Time budgets per dataset
    time_budgets = {
        'ETTh1': 600, 'ETTh2': 600, 'exchange_rate': 600,
        'ETTm1': 600, 'ETTm2': 600, 'weather': 900,
        'electricity': 1200, 'traffic': 1200,
    }
    
    for ds, h in experiments:
        key = str(h)
        if ds in all_results and key in all_results[ds] and all_results[ds][key].get('mse', 0) > 0:
            print(f"\nSkipping {ds} H={h} (MSE={all_results[ds][key]['mse']:.6f})")
            continue
        
        try:
            metrics = train_and_eval(ds, h, device, max_time=time_budgets[ds])
            if ds not in all_results:
                all_results[ds] = {}
            all_results[ds][key] = {'mse': round(metrics['mse'], 6), 'mae': round(metrics['mae'], 6)}
        except Exception as e:
            import traceback
            print(f"ERROR: {ds} H={h}: {e}")
            traceback.print_exc()
            if ds not in all_results:
                all_results[ds] = {}
            all_results[ds][key] = {'mse': -1, 'mae': -1}
        
        # Save after each experiment
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
    
    # Print summary
    print_summary(all_results)


def print_summary(all_results):
    targets = {
        'ETTh1': {96: 0.375, 192: 0.398, 336: 0.419, 720: 0.422},
        'ETTh2': {96: 0.253, 192: 0.334, 336: 0.347, 720: 0.401},
        'ETTm1': {96: 0.322, 192: 0.349, 336: 0.366, 720: 0.407},
        'ETTm2': {96: 0.173, 192: 0.230, 336: 0.279, 720: 0.388},
        'electricity': {96: 0.142, 192: 0.153, 336: 0.175, 720: 0.209},
        'exchange_rate': {96: 0.163, 192: 0.229, 336: 0.383, 720: 0.999},
        'traffic': {96: 0.396, 192: 0.408, 336: 0.427, 720: 0.449},
        'weather': {96: 0.161, 192: 0.208, 336: 0.252, 720: 0.337},
    }
    
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"{'Dataset':<15} {'H=96':>15} {'H=192':>15} {'H=336':>15} {'H=720':>15}")
    print("-"*75)
    for ds in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'electricity', 'exchange_rate', 'traffic', 'weather']:
        row = f"{ds:<15}"
        for h in [96, 192, 336, 720]:
            mse = all_results.get(ds, {}).get(str(h), {}).get('mse', float('nan'))
            target = targets.get(ds, {}).get(h, float('nan'))
            if mse > 0:
                row += f" {mse:.3f}({target:.3f})"
            else:
                row += f"     -({target:.3f})"
        print(row)


if __name__ == '__main__':
    main()
