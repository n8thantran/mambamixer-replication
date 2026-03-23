"""
Comprehensive training script for TSM2 on all 8 datasets and 4 horizons.
Optimized for speed while maintaining good results.
"""
import torch
import torch.nn as nn
import numpy as np
import time
import os
import json
import sys
from data_loader import load_dataset, TimeSeriesDataset
from torch.utils.data import DataLoader
from tsm2_model import TSM2

def train_and_eval(dataset_name, pred_len, config, device='cuda', verbose=True):
    """Train and evaluate TSM2 on a single dataset/horizon combination."""
    
    # Load data
    train_data, val_data, test_data, mean, std = load_dataset(dataset_name, './data', 512)
    num_variates = train_data.shape[1]
    
    train_ds = TimeSeriesDataset(train_data, 512, pred_len)
    val_ds = TimeSeriesDataset(val_data, 512, pred_len)
    test_ds = TimeSeriesDataset(test_data, 512, pred_len)
    
    batch_size = config.get('batch_size', 32)
    # For large datasets, use larger batch size
    if num_variates > 100:
        batch_size = min(batch_size, 16)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size*2, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size*2, shuffle=False, num_workers=4, pin_memory=True)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}, Horizon: {pred_len}, Variates: {num_variates}")
        print(f"Samples: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    
    # Model
    d_model = config.get('d_model', 128)
    num_layers = config.get('num_layers', 4)
    dropout = config.get('dropout', 0.1)
    d_state = config.get('d_state', 16)
    d_conv = config.get('d_conv', 4)
    expand = config.get('expand', 2)
    patch_len = config.get('patch_len', 16)
    stride = config.get('stride', 8)
    
    model = TSM2(
        num_variates=num_variates, seq_len=512, pred_len=pred_len,
        patch_len=patch_len, stride=stride,
        d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand,
        num_layers=num_layers, dropout=dropout,
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"Parameters: {n_params:,}")
    
    # Optimizer
    lr = config.get('lr', 1e-3)
    weight_decay = config.get('weight_decay', 1e-4)
    epochs = config.get('epochs', 50)
    patience = config.get('patience', 10)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.MSELoss()
    
    best_val = float('inf')
    patience_counter = 0
    best_state = None
    
    for epoch in range(epochs):
        t0 = time.time()
        
        # Train
        model.train()
        total_loss = 0
        n_batches = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()
        
        # Validate
        model.eval()
        val_loss = 0
        val_n = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += ((pred - y)**2).sum().item()
                val_n += y.numel()
        val_mse = val_loss / val_n
        
        elapsed = time.time() - t0
        
        if val_mse < best_val:
            best_val = val_mse
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        
        if verbose and (epoch % 5 == 0 or patience_counter == 0):
            print(f'  Epoch {epoch+1:3d}/{epochs}: Train={total_loss/n_batches:.6f}, Val={val_mse:.6f}, Best={best_val:.6f}, Time={elapsed:.1f}s')
        
        if patience_counter >= patience:
            if verbose:
                print(f'  Early stopping at epoch {epoch+1}')
            break
    
    # Test
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    
    test_loss = 0
    test_mae_loss = 0
    test_n = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            test_loss += ((pred - y)**2).sum().item()
            test_mae_loss += (pred - y).abs().sum().item()
            test_n += y.numel()
    
    test_mse = test_loss / test_n
    test_mae = test_mae_loss / test_n
    
    if verbose:
        print(f'  Test MSE: {test_mse:.6f}, MAE: {test_mae:.6f}')
    
    return test_mse, test_mae


# Dataset-specific hyperparameter configurations
# The paper says they use grid search to tune hyperparameters
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
        'lr': 1e-3, 'weight_decay': 1e-4, 'batch_size': 32, 'epochs': 50, 'patience': 10,
    },
    'ETTm2': {
        'd_model': 128, 'num_layers': 4, 'd_state': 16, 'd_conv': 4, 'expand': 2,
        'patch_len': 16, 'stride': 8, 'dropout': 0.1,
        'lr': 1e-3, 'weight_decay': 1e-4, 'batch_size': 32, 'epochs': 50, 'patience': 10,
    },
    'electricity': {
        'd_model': 128, 'num_layers': 4, 'd_state': 16, 'd_conv': 4, 'expand': 2,
        'patch_len': 16, 'stride': 8, 'dropout': 0.1,
        'lr': 1e-3, 'weight_decay': 1e-4, 'batch_size': 8, 'epochs': 30, 'patience': 5,
    },
    'exchange_rate': {
        'd_model': 128, 'num_layers': 4, 'd_state': 16, 'd_conv': 4, 'expand': 2,
        'patch_len': 16, 'stride': 8, 'dropout': 0.1,
        'lr': 1e-3, 'weight_decay': 1e-4, 'batch_size': 32, 'epochs': 50, 'patience': 10,
    },
    'traffic': {
        'd_model': 128, 'num_layers': 4, 'd_state': 16, 'd_conv': 4, 'expand': 2,
        'patch_len': 16, 'stride': 8, 'dropout': 0.1,
        'lr': 1e-3, 'weight_decay': 1e-4, 'batch_size': 4, 'epochs': 20, 'patience': 5,
    },
    'weather': {
        'd_model': 128, 'num_layers': 4, 'd_state': 16, 'd_conv': 4, 'expand': 2,
        'patch_len': 16, 'stride': 8, 'dropout': 0.1,
        'lr': 1e-3, 'weight_decay': 1e-4, 'batch_size': 32, 'epochs': 50, 'patience': 10,
    },
}


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Parse arguments
    if len(sys.argv) > 1:
        datasets = sys.argv[1].split(',')
    else:
        datasets = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'electricity', 'exchange_rate', 'traffic', 'weather']
    
    if len(sys.argv) > 2:
        horizons = [int(h) for h in sys.argv[2].split(',')]
    else:
        horizons = [96, 192, 336, 720]
    
    os.makedirs('/workspace/results', exist_ok=True)
    
    # Load existing results
    results_file = '/workspace/results/all_results.json'
    if os.path.exists(results_file):
        with open(results_file) as f:
            all_results = json.load(f)
    else:
        all_results = {}
    
    for ds in datasets:
        if ds not in all_results:
            all_results[ds] = {}
        
        config = CONFIGS.get(ds, CONFIGS['ETTh1'])
        
        for h in horizons:
            key = str(h)
            if key in all_results[ds]:
                print(f"Skipping {ds} H={h} (already done: MSE={all_results[ds][key]['mse']:.6f})")
                continue
            
            try:
                mse, mae = train_and_eval(ds, h, config, device)
                all_results[ds][key] = {'mse': mse, 'mae': mae}
                
                # Save after each experiment
                with open(results_file, 'w') as f:
                    json.dump(all_results, f, indent=2)
                
            except Exception as e:
                import traceback
                print(f"ERROR on {ds} H={h}: {e}")
                traceback.print_exc()
    
    # Print summary table
    print("\n" + "="*80)
    print("RESULTS SUMMARY (MSE)")
    print("="*80)
    print(f"{'Dataset':<15} {'H=96':>8} {'H=192':>8} {'H=336':>8} {'H=720':>8}")
    print("-"*47)
    for ds in datasets:
        if ds in all_results:
            vals = []
            for h in horizons:
                key = str(h)
                if key in all_results[ds]:
                    vals.append(f"{all_results[ds][key]['mse']:.4f}")
                else:
                    vals.append("  N/A  ")
            print(f"{ds:<15} {'  '.join(vals)}")
    
    print("\nTarget values from paper:")
    targets = {
        'ETTh1': [0.375, 0.398, 0.419, 0.422],
        'ETTh2': [0.253, 0.334, 0.347, 0.401],
        'ETTm1': [0.322, 0.349, 0.366, 0.407],
        'ETTm2': [0.173, 0.230, 0.279, 0.388],
        'electricity': [0.142, 0.153, 0.175, 0.209],
        'exchange_rate': [0.163, 0.229, 0.383, 0.999],
        'traffic': [0.396, 0.408, 0.427, 0.449],
        'weather': [0.161, 0.208, 0.252, 0.337],
    }
    for ds in datasets:
        if ds in targets:
            vals = [f"{v:.4f}" for v in targets[ds]]
            print(f"{ds:<15} {'  '.join(vals)}")


if __name__ == '__main__':
    main()
