"""
Run all TSM2 experiments for the 8 benchmark datasets.
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from tsm2_model import TSM2
from data_loader import get_dataloaders, DATASET_INFO


def train_epoch(model, train_loader, optimizer, criterion, device, max_grad_norm=1.0):
    model.train()
    total_loss = 0
    n_batches = 0
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        loss.backward()
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def evaluate(model, data_loader, criterion, device):
    model.eval()
    preds_list = []
    targets_list = []
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            pred = model(batch_x)
            preds_list.append(pred.cpu())
            targets_list.append(batch_y.cpu())
    
    preds = torch.cat(preds_list, dim=0)
    targets = torch.cat(targets_list, dim=0)
    
    mse = ((preds - targets) ** 2).mean().item()
    mae = (preds - targets).abs().mean().item()
    
    return {'mse': mse, 'mae': mae}


# Hyperparameter configurations per dataset
# These are tuned to match paper results
CONFIGS = {
    'ETTh1': {
        'd_model': 128, 'num_layers': 4, 'd_state': 16, 'd_conv': 4, 'expand': 2,
        'patch_len': 16, 'stride': 8, 'dropout': 0.1,
        'lr': 1e-3, 'weight_decay': 1e-4, 'batch_size': 32, 'epochs': 50, 'patience': 10,
    },
    'ETTh2': {
        'd_model': 128, 'num_layers': 4, 'd_state': 16, 'd_conv': 4, 'expand': 2,
        'patch_len': 16, 'stride': 8, 'dropout': 0.1,
        'lr': 1e-3, 'weight_decay': 1e-4, 'batch_size': 32, 'epochs': 50, 'patience': 10,
    },
    'ETTm1': {
        'd_model': 128, 'num_layers': 4, 'd_state': 16, 'd_conv': 4, 'expand': 2,
        'patch_len': 16, 'stride': 8, 'dropout': 0.1,
        'lr': 1e-3, 'weight_decay': 1e-4, 'batch_size': 32, 'epochs': 30, 'patience': 7,
    },
    'ETTm2': {
        'd_model': 128, 'num_layers': 4, 'd_state': 16, 'd_conv': 4, 'expand': 2,
        'patch_len': 16, 'stride': 8, 'dropout': 0.1,
        'lr': 1e-3, 'weight_decay': 1e-4, 'batch_size': 32, 'epochs': 30, 'patience': 7,
    },
    'electricity': {
        'd_model': 64, 'num_layers': 3, 'd_state': 16, 'd_conv': 4, 'expand': 2,
        'patch_len': 16, 'stride': 8, 'dropout': 0.1,
        'lr': 5e-4, 'weight_decay': 1e-4, 'batch_size': 16, 'epochs': 30, 'patience': 7,
    },
    'exchange_rate': {
        'd_model': 128, 'num_layers': 4, 'd_state': 16, 'd_conv': 4, 'expand': 2,
        'patch_len': 16, 'stride': 8, 'dropout': 0.1,
        'lr': 5e-4, 'weight_decay': 1e-4, 'batch_size': 32, 'epochs': 50, 'patience': 10,
    },
    'traffic': {
        'd_model': 64, 'num_layers': 3, 'd_state': 16, 'd_conv': 4, 'expand': 2,
        'patch_len': 16, 'stride': 8, 'dropout': 0.1,
        'lr': 5e-4, 'weight_decay': 1e-4, 'batch_size': 8, 'epochs': 30, 'patience': 7,
    },
    'weather': {
        'd_model': 128, 'num_layers': 4, 'd_state': 16, 'd_conv': 4, 'expand': 2,
        'patch_len': 16, 'stride': 8, 'dropout': 0.1,
        'lr': 1e-3, 'weight_decay': 1e-4, 'batch_size': 32, 'epochs': 30, 'patience': 7,
    },
}


def run_experiment(dataset_name, pred_len, device, data_dir='./data', save_dir='./checkpoints'):
    """Run a single experiment."""
    print(f"\n{'='*60}")
    print(f"Training TSM2: {dataset_name}, H={pred_len}")
    print(f"{'='*60}")
    
    cfg = CONFIGS[dataset_name].copy()
    num_variates = DATASET_INFO[dataset_name]['num_variates']
    
    # Get data
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset_name, seq_len=512, pred_len=pred_len,
        batch_size=cfg['batch_size'], data_dir=data_dir, num_workers=4,
    )
    
    # Build model
    model = TSM2(
        num_variates=num_variates,
        seq_len=512,
        pred_len=pred_len,
        patch_len=cfg['patch_len'],
        stride=cfg['stride'],
        d_model=cfg['d_model'],
        d_state=cfg['d_state'],
        d_conv=cfg['d_conv'],
        expand=cfg['expand'],
        num_layers=cfg['num_layers'],
        dropout=cfg['dropout'],
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg['epochs'], eta_min=1e-6)
    
    best_val_mse = float('inf')
    patience_counter = 0
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, f'{dataset_name}_H{pred_len}.pt')
    
    for epoch in range(cfg['epochs']):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0
        
        print(f"  Epoch {epoch+1:3d}/{cfg['epochs']} | "
              f"Train: {train_loss:.6f} | Val MSE: {val_metrics['mse']:.6f} | "
              f"Val MAE: {val_metrics['mae']:.6f} | {elapsed:.1f}s")
        
        if val_metrics['mse'] < best_val_mse:
            best_val_mse = val_metrics['mse']
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= cfg['patience']:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    # Load best and test
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_metrics = evaluate(model, test_loader, criterion, device)
    
    print(f"\n  >>> Test MSE: {test_metrics['mse']:.6f}, MAE: {test_metrics['mae']:.6f}")
    return test_metrics


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Datasets to run (default: all)')
    parser.add_argument('--horizons', nargs='+', type=int, default=None,
                        help='Horizons to run (default: all)')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--results_dir', type=str, default='./results')
    args = parser.parse_args()
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    all_datasets = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 
                     'electricity', 'exchange_rate', 'traffic', 'weather']
    all_horizons = [96, 192, 336, 720]
    
    datasets = args.datasets if args.datasets else all_datasets
    horizons = args.horizons if args.horizons else all_horizons
    
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Load existing results if any
    results_path = os.path.join(args.results_dir, 'results.json')
    if os.path.exists(results_path):
        with open(results_path) as f:
            all_results = json.load(f)
    else:
        all_results = {}
    
    for dataset in datasets:
        if dataset not in all_results:
            all_results[dataset] = {}
        for horizon in horizons:
            key = str(horizon)
            # Skip if already done
            if key in all_results[dataset] and not np.isnan(all_results[dataset][key].get('mse', float('nan'))):
                print(f"\nSkipping {dataset} H={horizon} (already done: MSE={all_results[dataset][key]['mse']:.4f})")
                continue
            
            try:
                metrics = run_experiment(dataset, horizon, device, args.data_dir, args.save_dir)
                all_results[dataset][key] = {
                    'mse': float(metrics['mse']),
                    'mae': float(metrics['mae']),
                }
            except Exception as e:
                import traceback
                print(f"ERROR: {dataset} H={horizon}: {e}")
                traceback.print_exc()
                all_results[dataset][key] = {'mse': float('nan'), 'mae': float('nan')}
            
            # Save after each experiment
            with open(results_path, 'w') as f:
                json.dump(all_results, f, indent=2)
            
            # Clear GPU cache
            torch.cuda.empty_cache()
    
    # Print summary table
    print("\n" + "="*80)
    print("RESULTS SUMMARY (MSE)")
    print("="*80)
    
    # Paper targets
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
    
    header = f"{'Dataset':<15}"
    for h in all_horizons:
        header += f" {'H='+str(h):>12}"
    print(header)
    print("-" * 65)
    
    for dataset in all_datasets:
        if dataset not in all_results:
            continue
        row = f"{dataset:<15}"
        for h in all_horizons:
            key = str(h)
            if key in all_results[dataset]:
                mse = all_results[dataset][key].get('mse', float('nan'))
                target = targets.get(dataset, {}).get(h, float('nan'))
                row += f" {mse:>5.3f}({target:.3f})"
            else:
                row += f" {'N/A':>12}"
        print(row)
    
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
