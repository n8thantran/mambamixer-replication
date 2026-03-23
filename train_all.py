"""
Training script for TSM2 on all 8 datasets × 4 horizons = 32 experiments.
Designed for efficiency with dataset-specific hyperparameters tuned for generalization.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from tsm2_model import TSM2
from data_loader import get_dataloaders, DATASET_INFO


# Dataset-specific configs, tuned to balance quality and training time
CONFIGS = {
    'ETTh1': {
        'd_model': 16, 'num_layers': 2, 'd_state': 16, 'd_conv': 4, 'expand': 2,
        'lr': 5e-4, 'weight_decay': 0.05, 'batch_size': 64, 'epochs': 20, 'patience': 5,
        'dropout': 0.5, 'patch_len': 16, 'stride': 8,
    },
    'ETTh2': {
        'd_model': 16, 'num_layers': 2, 'd_state': 16, 'd_conv': 4, 'expand': 2,
        'lr': 5e-4, 'weight_decay': 0.05, 'batch_size': 64, 'epochs': 20, 'patience': 5,
        'dropout': 0.5, 'patch_len': 16, 'stride': 8,
    },
    'ETTm1': {
        'd_model': 16, 'num_layers': 2, 'd_state': 16, 'd_conv': 4, 'expand': 2,
        'lr': 1e-3, 'weight_decay': 0.05, 'batch_size': 128, 'epochs': 15, 'patience': 5,
        'dropout': 0.3, 'patch_len': 16, 'stride': 8,
    },
    'ETTm2': {
        'd_model': 16, 'num_layers': 2, 'd_state': 16, 'd_conv': 4, 'expand': 2,
        'lr': 1e-3, 'weight_decay': 0.05, 'batch_size': 128, 'epochs': 15, 'patience': 5,
        'dropout': 0.3, 'patch_len': 16, 'stride': 8,
    },
    'electricity': {
        'd_model': 32, 'num_layers': 2, 'd_state': 16, 'd_conv': 4, 'expand': 2,
        'lr': 5e-4, 'weight_decay': 0.01, 'batch_size': 16, 'epochs': 10, 'patience': 3,
        'dropout': 0.2, 'patch_len': 16, 'stride': 8,
    },
    'exchange_rate': {
        'd_model': 16, 'num_layers': 2, 'd_state': 16, 'd_conv': 4, 'expand': 2,
        'lr': 5e-4, 'weight_decay': 0.05, 'batch_size': 32, 'epochs': 30, 'patience': 8,
        'dropout': 0.5, 'patch_len': 16, 'stride': 8,
    },
    'traffic': {
        'd_model': 32, 'num_layers': 2, 'd_state': 16, 'd_conv': 4, 'expand': 1,
        'lr': 5e-4, 'weight_decay': 0.01, 'batch_size': 4, 'epochs': 10, 'patience': 3,
        'dropout': 0.2, 'patch_len': 16, 'stride': 8,
    },
    'weather': {
        'd_model': 32, 'num_layers': 2, 'd_state': 16, 'd_conv': 4, 'expand': 2,
        'lr': 1e-3, 'weight_decay': 0.01, 'batch_size': 64, 'epochs': 15, 'patience': 5,
        'dropout': 0.3, 'patch_len': 16, 'stride': 8,
    },
}


def train_epoch(model, train_loader, optimizer, criterion, device, max_grad_norm=1.0):
    """Train one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0
    
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, data_loader, device):
    """Evaluate model, return MSE and MAE."""
    model.eval()
    all_preds = []
    all_targets = []
    
    for batch_x, batch_y in data_loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)
        
        pred = model(batch_x)
        all_preds.append(pred.cpu())
        all_targets.append(batch_y.cpu())
    
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    mse = ((all_preds - all_targets) ** 2).mean().item()
    mae = (all_preds - all_targets).abs().mean().item()
    
    return {'mse': mse, 'mae': mae}


def train_experiment(dataset_name, pred_len, device, data_dir='./data', save_dir='./checkpoints'):
    """Train one experiment. Returns test metrics."""
    print(f"\n{'='*60}")
    print(f"Training TSM2: {dataset_name}, H={pred_len}")
    print(f"{'='*60}")
    
    cfg = CONFIGS[dataset_name]
    num_variates = DATASET_INFO[dataset_name]['num_variates']
    
    # Data
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset_name, seq_len=512, pred_len=pred_len,
        batch_size=cfg['batch_size'], data_dir=data_dir, num_workers=2,
    )
    
    # Model
    model = TSM2(
        num_variates=num_variates, seq_len=512, pred_len=pred_len,
        patch_len=cfg['patch_len'], stride=cfg['stride'],
        d_model=cfg['d_model'], d_state=cfg['d_state'],
        d_conv=cfg['d_conv'], expand=cfg['expand'],
        num_layers=cfg['num_layers'], dropout=cfg['dropout'],
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    
    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg['epochs'], eta_min=1e-6)
    
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, f'{dataset_name}_H{pred_len}.pt')
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(cfg['epochs']):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()
        elapsed = time.time() - t0
        
        improved = ""
        if val_metrics['mse'] < best_val_loss:
            best_val_loss = val_metrics['mse']
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
            improved = " *"
        else:
            patience_counter += 1
        
        print(f"  Epoch {epoch+1:3d}/{cfg['epochs']} | "
              f"Train: {train_loss:.4f} | Val MSE: {val_metrics['mse']:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e} | {elapsed:.1f}s{improved}")
        
        if patience_counter >= cfg['patience']:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    # Test with best model
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    test_metrics = evaluate(model, test_loader, device)
    print(f"  >>> Test MSE: {test_metrics['mse']:.6f}, MAE: {test_metrics['mae']:.6f}")
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    
    return test_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--pred_len', type=int, default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--results_dir', type=str, default='./results')
    args = parser.parse_args()
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.results_dir, exist_ok=True)
    
    results_path = os.path.join(args.results_dir, 'results.json')
    if os.path.exists(results_path):
        with open(results_path) as f:
            all_results = json.load(f)
    else:
        all_results = {}
    
    if args.dataset and args.pred_len:
        datasets = [args.dataset]
        horizons = [args.pred_len]
    elif args.dataset:
        datasets = [args.dataset]
        horizons = [96, 192, 336, 720]
    else:
        # Run all 32 experiments - small datasets first
        datasets = ['ETTh1', 'ETTh2', 'exchange_rate', 'ETTm1', 'ETTm2', 
                     'weather', 'electricity', 'traffic']
        horizons = [96, 192, 336, 720]
    
    for dataset in datasets:
        if dataset not in all_results:
            all_results[dataset] = {}
        for horizon in horizons:
            key = str(horizon)
            if key in all_results.get(dataset, {}) and all_results[dataset][key].get('mse', 0) > 0:
                print(f"\nSkipping {dataset} H={horizon} (already done: MSE={all_results[dataset][key]['mse']:.6f})")
                continue
            
            try:
                metrics = train_experiment(dataset, horizon, device, args.data_dir, args.save_dir)
                all_results[dataset][key] = {'mse': round(metrics['mse'], 6), 'mae': round(metrics['mae'], 6)}
            except Exception as e:
                import traceback
                print(f"ERROR: {dataset} H={horizon}: {e}")
                traceback.print_exc()
                all_results[dataset][key] = {'mse': -1, 'mae': -1, 'error': str(e)}
            
            # Save after each experiment
            with open(results_path, 'w') as f:
                json.dump(all_results, f, indent=2)
    
    # Print summary table
    print_results_table(all_results, results_path)


def print_results_table(all_results, results_path=None):
    """Print a nice results summary table."""
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
    
    print("\n" + "="*90)
    print("TSM2 RESULTS (MSE / MAE) — Ours vs Paper Target")
    print("="*90)
    
    datasets_order = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'electricity', 'exchange_rate', 'traffic', 'weather']
    horizons = [96, 192, 336, 720]
    
    header = f"{'Dataset':<15}"
    for h in horizons:
        header += f"  {'H='+str(h):>20s}"
    print(header)
    print("-" * 95)
    
    for dataset in datasets_order:
        row = f"{dataset:<15}"
        for h in horizons:
            mse = all_results.get(dataset, {}).get(str(h), {}).get('mse', float('nan'))
            target = targets.get(dataset, {}).get(h, float('nan'))
            if mse and mse > 0:
                ratio = mse / target if target > 0 else 0
                row += f"  {mse:.3f}({target:.3f}){ratio:.1f}x"
            else:
                row += f"      -  ({target:.3f})    "
        print(row)
    
    # Save table to file
    if results_path:
        table_path = results_path.replace('.json', '_table.txt')
        with open(table_path, 'w') as f:
            f.write("TSM2 Results: MSE\n")
            f.write(f"{'Dataset':<15} {'H=96':>10} {'H=192':>10} {'H=336':>10} {'H=720':>10}\n")
            f.write("-" * 55 + "\n")
            for dataset in datasets_order:
                row = f"{dataset:<15}"
                for h in horizons:
                    mse = all_results.get(dataset, {}).get(str(h), {}).get('mse', float('nan'))
                    if mse and mse > 0:
                        row += f" {mse:>9.4f}"
                    else:
                        row += f" {'—':>9s}"
                f.write(row + "\n")
            
            f.write("\nTSM2 Results: MAE\n")
            f.write(f"{'Dataset':<15} {'H=96':>10} {'H=192':>10} {'H=336':>10} {'H=720':>10}\n")
            f.write("-" * 55 + "\n")
            for dataset in datasets_order:
                row = f"{dataset:<15}"
                for h in horizons:
                    mae = all_results.get(dataset, {}).get(str(h), {}).get('mae', float('nan'))
                    if mae and mae > 0:
                        row += f" {mae:>9.4f}"
                    else:
                        row += f" {'—':>9s}"
                f.write(row + "\n")
            
            f.write("\nPaper Target (MSE):\n")
            f.write(f"{'Dataset':<15} {'H=96':>10} {'H=192':>10} {'H=336':>10} {'H=720':>10}\n")
            f.write("-" * 55 + "\n")
            for dataset in datasets_order:
                row = f"{dataset:<15}"
                for h in horizons:
                    target = targets.get(dataset, {}).get(h)
                    row += f" {target:>9.3f}" if target else f" {'—':>9s}"
                f.write(row + "\n")
        
        print(f"\nResults table saved to {table_path}")


if __name__ == '__main__':
    main()
