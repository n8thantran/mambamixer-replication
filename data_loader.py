"""
Data loading utilities for TSM2 time series forecasting.
Supports 8 benchmark datasets with standard train/val/test splits.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


DATASET_INFO = {
    'ETTh1': {'num_variates': 7, 'freq': 'h',
              'borders': [(0, 8640), (8640-512, 11520), (11520-512, 14400)]},
    'ETTh2': {'num_variates': 7, 'freq': 'h',
              'borders': [(0, 8640), (8640-512, 11520), (11520-512, 14400)]},
    'ETTm1': {'num_variates': 7, 'freq': '15min',
              'borders': [(0, 34560), (34560-512, 46080), (46080-512, 57600)]},
    'ETTm2': {'num_variates': 7, 'freq': '15min',
              'borders': [(0, 34560), (34560-512, 46080), (46080-512, 57600)]},
    'electricity': {'num_variates': 321, 'freq': 'h'},
    'exchange_rate': {'num_variates': 8, 'freq': 'd'},
    'traffic': {'num_variates': 862, 'freq': 'h'},
    'weather': {'num_variates': 21, 'freq': '10min'},
}


def download_ett_datasets(data_dir='./data'):
    """Download ETT datasets."""
    os.makedirs(data_dir, exist_ok=True)
    ett_base_url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/"
    for name in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2']:
        filepath = os.path.join(data_dir, f'{name}.csv')
        if not os.path.exists(filepath) or os.path.getsize(filepath) < 100:
            print(f"Downloading {name}...")
            url = ett_base_url + f'{name}.csv'
            os.system(f'wget -q -O {filepath} "{url}"')


def load_ett_data(dataset_name, data_dir='./data'):
    """Load ETT dataset from CSV."""
    filepath = os.path.join(data_dir, f'{dataset_name}.csv')
    if not os.path.exists(filepath) or os.path.getsize(filepath) < 100:
        download_ett_datasets(data_dir)
    
    df = pd.read_csv(filepath)
    df = df.drop('date', axis=1)
    return df.values.astype(np.float32)


def load_lh_data(dataset_name, data_dir='./data'):
    """Load dataset from datasetsforecast long horizon format."""
    # Map dataset names to LH directory names
    name_map = {
        'electricity': ('ECL', 'M'),
        'exchange_rate': ('Exchange', 'M'),
        'traffic': ('traffic', 'M'),
        'weather': ('weather', 'M'),
    }
    
    lh_name, kind = name_map[dataset_name]
    lh_base = os.path.join(data_dir, 'lh_data', 'longhorizon', 'datasets')
    csv_path = os.path.join(lh_base, lh_name, kind, 'df_y.csv')
    
    if not os.path.exists(csv_path):
        # Try to download using datasetsforecast
        print(f"Downloading {dataset_name} via datasetsforecast...")
        from datasetsforecast.long_horizon import LongHorizon
        Y_df, _, _ = LongHorizon.load(directory=os.path.join(data_dir, 'lh_data'), 
                                       group=lh_name)
    
    # Read and pivot from long to wide format
    print(f"Loading {dataset_name} from {csv_path}...")
    df = pd.read_csv(csv_path)
    # Pivot: rows=timestamps, columns=unique_id, values=y
    pivot_df = df.pivot(index='ds', columns='unique_id', values='y')
    pivot_df = pivot_df.sort_index()
    
    return pivot_df.values.astype(np.float32)


def load_dataset(dataset_name, data_dir='./data', seq_len=512):
    """Load a dataset and return train/val/test splits as numpy arrays.
    
    Uses standard splits:
    - ETT: 12/4/4 months
    - Others: 70/10/20 split
    
    Normalization: per-variate using training set statistics.
    """
    if dataset_name.startswith('ETT'):
        data = load_ett_data(dataset_name, data_dir)
    else:
        data = load_lh_data(dataset_name, data_dir)
    
    n = len(data)
    info = DATASET_INFO[dataset_name]
    
    if dataset_name.startswith('ETT'):
        borders = info['borders']
        train_slice = slice(borders[0][0], borders[0][1])
        val_slice = slice(borders[1][0], borders[1][1])
        test_slice = slice(borders[2][0], borders[2][1])
    else:
        train_end = int(n * 0.7)
        val_end = int(n * 0.8)
        # For val/test, we need seq_len lookback
        train_slice = slice(0, train_end)
        val_slice = slice(train_end - seq_len, val_end)
        test_slice = slice(val_end - seq_len, n)
    
    train_data = data[train_slice]
    val_data = data[val_slice]
    test_data = data[test_slice]
    
    # Compute normalization stats from training data
    train_mean = train_data.mean(axis=0)
    train_std = train_data.std(axis=0) + 1e-5
    
    # Normalize all splits
    train_data = (train_data - train_mean) / train_std
    val_data = (val_data - train_mean) / train_std
    test_data = (test_data - train_mean) / train_std
    
    return train_data, val_data, test_data, train_mean, train_std


class TimeSeriesDataset(Dataset):
    """Time series dataset for sliding window forecasting."""
    
    def __init__(self, data, seq_len=512, pred_len=96):
        self.data = torch.FloatTensor(data)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.total_len = seq_len + pred_len
        self.n_samples = len(data) - self.total_len + 1
    
    def __len__(self):
        return max(0, self.n_samples)
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + self.seq_len:idx + self.total_len]
        return x, y


def get_dataloaders(dataset_name, seq_len=512, pred_len=96, batch_size=32, 
                    data_dir='./data', num_workers=4):
    """Get train/val/test dataloaders for a dataset."""
    train_data, val_data, test_data, mean, std = load_dataset(
        dataset_name, data_dir, seq_len
    )
    
    train_dataset = TimeSeriesDataset(train_data, seq_len, pred_len)
    val_dataset = TimeSeriesDataset(val_data, seq_len, pred_len)
    test_dataset = TimeSeriesDataset(test_data, seq_len, pred_len)
    
    print(f"Dataset: {dataset_name}")
    print(f"  Data shapes: train={train_data.shape}, val={val_data.shape}, test={test_data.shape}")
    print(f"  Samples: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    for ds_name in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 
                     'electricity', 'exchange_rate', 'traffic', 'weather']:
        try:
            train_loader, val_loader, test_loader = get_dataloaders(
                ds_name, seq_len=512, pred_len=96, batch_size=32, num_workers=0
            )
            x, y = next(iter(train_loader))
            print(f"  Batch: x={x.shape}, y={y.shape}")
            print()
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
            print()
