"""
Fast training for large datasets (electricity, traffic).
Uses gradient accumulation and fewer epochs to fit within time constraints.
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


def train_epoch(model, train_loader, optimizer, criterion, device, max_grad_norm=1.0, accum_steps=1):
    model.train()
    total_loss = 0
    n_batches = 0
    optimizer.zero_grad(set_to_none=True)
    for i, (batch_x, batch_y) in enumerate(train_loader):
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)
        pred = model(batch_x)
        loss = criterion(pred, batch_y) / accum_steps
        loss.backward()
        total_loss += loss.item() * accum_steps
        n_batches += 1
        if (i + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    # Handle remaining gradients
    if n_batches % accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    total_mse = 0
    total_mae = 0
    total_samples = 0
    for batch_x, batch_y in data_loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)
        pred = model(batch_x)
        bs = batch_x.shape[0]
        total_mse += ((pred - batch_y) ** 2).mean().item() * bs
        total_mae += (pred - batch_y).abs().mean().item() * bs
        total_samples += bs
    return {'mse': total_mse / total_samples, 'mae': total_mae / total_samples}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--pred_len', type=int, required=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Configs for large datasets
    if args.dataset == 'electricity':
        cfg = {
            'd_model': 32, 'num_layers': 2, 'd_state': 16, 'd_conv': 4, 'expand': 2,
            'lr': 1e-4, 'weight_decay': 0.01, 'batch_size': 16, 'epochs': 5, 'patience': 3,
            'dropout': 0.2, 'patch_len': 16, 'stride': 8, 'accum_steps': 2,
        }
    elif args.dataset == 'traffic':
        cfg = {
            'd_model': 32, 'num_layers': 2, 'd_state': 16, 'd_conv': 4, 'expand': 1,
            'lr': 1e-4, 'weight_decay': 0.01, 'batch_size': 8, 'epochs': 5, 'patience': 3,
            'dropout': 0.2, 'patch_len': 16, 'stride': 8, 'accum_steps': 2,
        }
    else:
        raise ValueError(f"Use train_all.py for {args.dataset}")

    num_variates = DATASET_INFO[args.dataset]['num_variates']
    seq_len = 512
    pred_len = args.pred_len

    print(f"\n{'='*60}")
    print(f"Training TSM2: {args.dataset}, H={pred_len}")
    print(f"{'='*60}")

    train_loader, val_loader, test_loader = get_dataloaders(
        args.dataset, seq_len=seq_len, pred_len=pred_len,
        batch_size=cfg['batch_size'], num_workers=4
    )

    model = TSM2(
        num_variates=num_variates, seq_len=seq_len, pred_len=pred_len,
        patch_len=cfg['patch_len'], stride=cfg['stride'],
        d_model=cfg['d_model'], d_state=cfg['d_state'],
        d_conv=cfg['d_conv'], expand=cfg['expand'],
        num_layers=cfg['num_layers'], dropout=cfg['dropout'],
    ).to(device)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg['epochs'], eta_min=1e-6)
    criterion = nn.MSELoss()

    os.makedirs('./checkpoints', exist_ok=True)
    ckpt_path = f'./checkpoints/{args.dataset}_H{pred_len}.pt'

    best_val = float('inf')
    patience_counter = 0

    for epoch in range(1, cfg['epochs'] + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, 
                                  accum_steps=cfg['accum_steps'])
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()
        elapsed = time.time() - t0

        improved = val_metrics['mse'] < best_val
        if improved:
            best_val = val_metrics['mse']
            torch.save(model.state_dict(), ckpt_path)
            patience_counter = 0
        else:
            patience_counter += 1

        print(f"  E {epoch:2d}/{cfg['epochs']} | Train: {train_loss:.4f} | Val: {val_metrics['mse']:.4f} | {elapsed:.1f}s {'*' if improved else ''}")

        if patience_counter >= cfg['patience']:
            print(f"  Early stopping at epoch {epoch}")
            break

    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    test_metrics = evaluate(model, test_loader, device)
    print(f"  >>> Test MSE: {test_metrics['mse']:.6f}, MAE: {test_metrics['mae']:.6f}")

    # Save results
    results_file = './results/results.json'
    os.makedirs('./results', exist_ok=True)
    if os.path.exists(results_file):
        with open(results_file) as f:
            all_results = json.load(f)
    else:
        all_results = {}

    if args.dataset not in all_results:
        all_results[args.dataset] = {}
    all_results[args.dataset][str(pred_len)] = {
        'mse': round(test_metrics['mse'], 6),
        'mae': round(test_metrics['mae'], 6),
    }

    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {results_file}")


if __name__ == '__main__':
    main()
