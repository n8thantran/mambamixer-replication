"""
Training script for TSM2 time series forecasting.
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
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

from tsm2_model import TSM2
from data_loader import get_dataloaders, DATASET_INFO


def train_epoch(model, train_loader, optimizer, criterion, device, max_grad_norm=1.0, scheduler=None):
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
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / max(n_batches, 1)


def evaluate(model, data_loader, criterion, device):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            pred = model(batch_x)
            all_preds.append(pred.cpu())
            all_targets.append(batch_y.cpu())
    
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    mse = ((all_preds - all_targets) ** 2).mean().item()
    mae = (all_preds - all_targets).abs().mean().item()
    
    return {
        'loss': mse,
        'mse': mse,
        'mae': mae,
    }


def get_model_config(dataset_name, pred_len):
    """Get model hyperparameters based on dataset."""
    num_variates = DATASET_INFO[dataset_name]['num_variates']
    
    # Base config
    config = {
        'num_variates': num_variates,
        'seq_len': 512,
        'pred_len': pred_len,
        'patch_len': 16,
        'stride': 8,
        'd_model': 128,
        'd_state': 16,
        'd_conv': 4,
        'expand': 2,
        'num_layers': 4,
        'dropout': 0.1,
    }
    
    # Dataset-specific adjustments for memory
    if dataset_name == 'traffic':
        config['d_model'] = 64
        config['num_layers'] = 3
        config['expand'] = 1
    elif dataset_name == 'electricity':
        config['d_model'] = 64
        config['num_layers'] = 3
    
    return config


def get_training_config(dataset_name, pred_len):
    """Get training hyperparameters."""
    config = {
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'batch_size': 32,
        'epochs': 50,
        'patience': 10,
        'max_grad_norm': 1.0,
    }
    
    # Dataset-specific adjustments
    if dataset_name in ['traffic']:
        config['batch_size'] = 8
        config['lr'] = 5e-4
        config['epochs'] = 30
    elif dataset_name in ['electricity']:
        config['batch_size'] = 16
        config['lr'] = 5e-4
        config['epochs'] = 30
    
    if dataset_name == 'exchange_rate':
        config['lr'] = 5e-4
        config['epochs'] = 50
        config['patience'] = 15
    
    return config


def train_model(dataset_name, pred_len, device, data_dir='./data', save_dir='./checkpoints'):
    """Train TSM2 model for a specific dataset and prediction horizon."""
    
    print(f"\n{'='*60}")
    print(f"Training TSM2: {dataset_name}, H={pred_len}")
    print(f"{'='*60}")
    
    # Get configs
    model_config = get_model_config(dataset_name, pred_len)
    train_config = get_training_config(dataset_name, pred_len)
    
    print(f"Model config: {model_config}")
    print(f"Training config: {train_config}")
    
    # Data
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset_name, 
        seq_len=model_config['seq_len'],
        pred_len=pred_len,
        batch_size=train_config['batch_size'],
        data_dir=data_dir,
        num_workers=4,
    )
    
    # Model
    model = TSM2(**model_config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=train_config['lr'], 
                      weight_decay=train_config['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=train_config['epochs'], eta_min=1e-6)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f'{dataset_name}_H{pred_len}.pt')
    
    for epoch in range(train_config['epochs']):
        t0 = time.time()
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device,
                                 max_grad_norm=train_config['max_grad_norm'])
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        elapsed = time.time() - t0
        
        print(f"  Epoch {epoch+1:3d}/{train_config['epochs']} | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val MSE: {val_metrics['mse']:.6f} | "
              f"Val MAE: {val_metrics['mae']:.6f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
              f"Time: {elapsed:.1f}s")
        
        if val_metrics['mse'] < best_val_loss:
            best_val_loss = val_metrics['mse']
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= train_config['patience']:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_metrics = evaluate(model, test_loader, criterion, device)
    
    print(f"\n  Test MSE: {test_metrics['mse']:.6f}")
    print(f"  Test MAE: {test_metrics['mae']:.6f}")
    
    return test_metrics, model_config, train_config


def main():
    parser = argparse.ArgumentParser(description='Train TSM2')
    parser.add_argument('--dataset', type=str, default='ETTh1',
                        choices=['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 
                                 'electricity', 'exchange_rate', 'traffic', 'weather'])
    parser.add_argument('--pred_len', type=int, default=96,
                        choices=[96, 192, 336, 720])
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--run_all', action='store_true', help='Run all datasets and horizons')
    args = parser.parse_args()
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(args.results_dir, exist_ok=True)
    
    if args.run_all:
        datasets = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 
                     'electricity', 'exchange_rate', 'traffic', 'weather']
        horizons = [96, 192, 336, 720]
    else:
        datasets = [args.dataset]
        horizons = [args.pred_len]
    
    # Load existing results
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
            # Skip if already done
            if str(horizon) in all_results.get(dataset, {}):
                mse = all_results[dataset][str(horizon)].get('mse', float('nan'))
                if not np.isnan(mse):
                    print(f"\nSkipping {dataset} H={horizon} (already done, MSE={mse:.6f})")
                    continue
            
            try:
                test_metrics, model_config, train_config = train_model(
                    dataset, horizon, device, args.data_dir, args.save_dir
                )
                all_results[dataset][str(horizon)] = {
                    'mse': test_metrics['mse'],
                    'mae': test_metrics['mae'],
                }
                
                # Save results incrementally
                with open(results_path, 'w') as f:
                    json.dump(all_results, f, indent=2)
                    
            except Exception as e:
                import traceback
                print(f"ERROR training {dataset} H={horizon}: {e}")
                traceback.print_exc()
                all_results[dataset][str(horizon)] = {'mse': float('nan'), 'mae': float('nan')}
    
    # Print summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY (MSE)")
    print("="*80)
    print(f"{'Dataset':<15} {'H=96':>10} {'H=192':>10} {'H=336':>10} {'H=720':>10}")
    print("-"*55)
    for dataset in datasets:
        row = f"{dataset:<15}"
        for h in horizons:
            mse = all_results.get(dataset, {}).get(str(h), {}).get('mse', float('nan'))
            row += f" {mse:>9.4f}"
        print(row)
    
    # Save final results
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
