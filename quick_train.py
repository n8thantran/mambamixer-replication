"""Quick training script for testing TSM2 on ETTh1."""
import torch
import torch.nn as nn
import time
from data_loader import load_dataset, TimeSeriesDataset
from torch.utils.data import DataLoader
from tsm2_model import TSM2

def train_and_eval(dataset_name, pred_len, d_model=64, num_layers=2, dropout=0.2, 
                   lr=5e-4, epochs=30, batch_size=32, device='cuda'):
    train_data, val_data, test_data, mean, std = load_dataset(dataset_name, './data', 512)
    num_variates = train_data.shape[1]
    
    train_ds = TimeSeriesDataset(train_data, 512, pred_len)
    val_ds = TimeSeriesDataset(val_data, 512, pred_len)
    test_ds = TimeSeriesDataset(test_data, 512, pred_len)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)
    
    model = TSM2(
        num_variates=num_variates, seq_len=512, pred_len=pred_len,
        patch_len=16, stride=8,
        d_model=d_model, d_state=16, d_conv=4, expand=2,
        num_layers=num_layers, dropout=dropout,
    ).to(device)
    
    print(f'Params: {sum(p.numel() for p in model.parameters()):,}')
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.MSELoss()
    
    best_val = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        total_loss = 0
        n = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n += 1
        scheduler.step()
        
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                preds.append(pred.cpu())
                targets.append(y.cpu())
        preds = torch.cat(preds)
        targets = torch.cat(targets)
        val_mse = ((preds - targets)**2).mean().item()
        
        elapsed = time.time() - t0
        
        if val_mse < best_val:
            best_val = val_mse
            patience_counter = 0
            torch.save(model.state_dict(), f'/tmp/best_{dataset_name}_{pred_len}.pt')
        else:
            patience_counter += 1
        
        print(f'Epoch {epoch+1:3d}/{epochs}: Train={total_loss/n:.6f}, Val MSE={val_mse:.6f}, Best={best_val:.6f}, Time={elapsed:.1f}s')
        
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    # Test
    model.load_state_dict(torch.load(f'/tmp/best_{dataset_name}_{pred_len}.pt'))
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            preds.append(pred.cpu())
            targets.append(y.cpu())
    preds = torch.cat(preds)
    targets = torch.cat(targets)
    test_mse = ((preds - targets)**2).mean().item()
    test_mae = (preds - targets).abs().mean().item()
    print(f'Test MSE: {test_mse:.6f}, MAE: {test_mae:.6f}')
    return test_mse, test_mae

if __name__ == '__main__':
    import sys
    dataset = sys.argv[1] if len(sys.argv) > 1 else 'ETTh1'
    pred_len = int(sys.argv[2]) if len(sys.argv) > 2 else 96
    d_model = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    num_layers = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    
    train_and_eval(dataset, pred_len, d_model=d_model, num_layers=num_layers)
