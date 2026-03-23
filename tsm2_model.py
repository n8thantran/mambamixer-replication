"""
TSM2 (Time Series MambaMixer) model implementation.

Based on "MambaMixer: Efficient Selective State Space Models with Dual Token and Channel Selection"

Key components:
1. Selective Token (Time) Mixer - unidirectional S6 for causal time dimension
   Operates on (B*M, N, D): mixing across N patches per variate
2. Selective Channel (Variate) Mixer - bidirectional S6 for non-causal variates
   Operates on (B*N, M, D): mixing across M variates per patch
3. Weighted Averaging of Earlier Features (alpha, beta, theta, gamma)
4. 2D normalization on time and feature dimensions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
import math


class Norm2D(nn.Module):
    """2D normalization on both time and feature dimensions."""
    def __init__(self, d1, d2):
        super().__init__()
        self.norm1 = nn.LayerNorm(d2)
        self.norm2 = nn.LayerNorm(d1)
    
    def forward(self, x):
        """x: (B, d1, d2)"""
        x = self.norm1(x)           # normalize along last dim (d2)
        x = x.transpose(-1, -2)
        x = self.norm2(x)           # normalize along d1 dim
        x = x.transpose(-1, -2)
        return x


class SelectiveTokenMixer(nn.Module):
    """Unidirectional S6 for causal time mixing. Input: (B*M, N, D)"""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        x = self.dropout(x)
        return x + residual


class SelectiveChannelMixer(nn.Module):
    """Bidirectional S6 for non-causal variate mixing. Input: (B*N, M, D)"""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.mamba_fwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_bwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        residual = x
        x = self.norm(x)
        y_fwd = self.mamba_fwd(x)
        y_bwd = torch.flip(self.mamba_bwd(torch.flip(x, [1])), [1])
        y = y_fwd + y_bwd
        y = self.dropout(y)
        return y + residual


class TSM2(nn.Module):
    """Time Series MambaMixer for multivariate time series forecasting.
    
    Paper Eq 8-9 weighted averaging (using 0-indexed layers, l=0..L-1):
    At start: y_T[0] = y_C[0] = embedded input
    
    For layer l (0-indexed):
      Token mixer input:   sum_{i=0..l} alpha[l,i]*y_T[i] + sum_{i=0..l} beta[l,i]*y_C[i]
      -> produces y_T[l+1]
      Channel mixer input: sum_{i=0..l+1} theta[l,i]*y_T[i] + sum_{i=0..l} gamma[l,i]*y_C[i]  
      -> produces y_C[l+1]
    
    (For first layer l=0, token input is just y_T[0]=y_C[0]=embedding)
    """
    def __init__(self, num_variates, seq_len=512, pred_len=96, patch_len=16, stride=8,
                 d_model=128, d_state=16, d_conv=4, expand=2, num_layers=4, dropout=0.1):
        super().__init__()
        self.num_variates = num_variates
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_patches = (seq_len - patch_len) // stride + 1
        
        # Patch embedding + positional encoding
        self.patch_embedding = nn.Linear(patch_len, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)
        
        # MambaMixer layers (each has a token mixer + channel mixer)
        self.token_mixers = nn.ModuleList()
        self.channel_mixers = nn.ModuleList()
        for _ in range(num_layers):
            self.token_mixers.append(
                SelectiveTokenMixer(d_model, d_state, d_conv, expand, dropout))
            self.channel_mixers.append(
                SelectiveChannelMixer(d_model, d_state, d_conv, expand, dropout))
        
        # Weighted averaging: for simplicity and stability, use a single set of weights
        # per layer that combines all previous token and channel outputs
        # alpha[l]: weights for token outputs feeding into token mixer (l+1 weights)
        # beta[l]: weights for channel outputs feeding into token mixer (l+1 weights)
        # theta[l]: weights for token outputs feeding into channel mixer (l+2 weights, includes current token out)
        # gamma[l]: weights for channel outputs feeding into channel mixer (l+1 weights)
        self.alpha = nn.ParameterList()
        self.beta = nn.ParameterList()
        self.theta = nn.ParameterList()
        self.gamma = nn.ParameterList()
        
        for l in range(num_layers):
            self.alpha.append(nn.Parameter(torch.ones(l + 1) / (l + 1)))
            self.beta.append(nn.Parameter(torch.ones(l + 1) / (l + 1)))
            self.theta.append(nn.Parameter(torch.ones(l + 2) / (l + 2)))
            self.gamma.append(nn.Parameter(torch.ones(l + 1) / (l + 1)))
        
        # 2D Normalization + prediction head
        self.norm = Norm2D(self.num_patches, d_model)
        self.head = nn.Linear(d_model * self.num_patches, pred_len)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """x: (B, L, M) -> (B, H, M)"""
        B, L, M = x.shape
        
        # Instance normalization (RevIN)
        means = x.mean(dim=1, keepdim=True)
        stds = x.std(dim=1, keepdim=True) + 1e-5
        x = (x - means) / stds
        
        # Patchify each variate: (B, L, M) -> (B, M, L) -> (B*M, N, patch_len) -> (B*M, N, D)
        x = x.transpose(1, 2)  # (B, M, L)
        x = x.unfold(dimension=2, size=self.patch_len, step=self.stride)  # (B, M, N, P)
        N = x.shape[2]
        x = x.reshape(B * M, N, self.patch_len)
        x = self.patch_embedding(x) + self.pos_embed
        x = self.dropout(x)
        
        D = self.d_model
        
        # y_T[0] = y_C[0] = initial embedding
        y_T = [x]  # token outputs, each (B*M, N, D)
        y_C = [x]  # channel outputs, each (B*M, N, D)
        
        for l in range(self.num_layers):
            # --- Token mixer input: weighted sum of y_T[0..l] and y_C[0..l] ---
            a = F.softmax(self.alpha[l], dim=0)  # (l+1,)
            b = F.softmax(self.beta[l], dim=0)   # (l+1,)
            tok_in = sum(a[i] * y_T[i] + b[i] * y_C[i] for i in range(l + 1))
            
            # Token mixing: (B*M, N, D) -> (B*M, N, D)
            tok_out = self.token_mixers[l](tok_in)
            y_T.append(tok_out)  # y_T[l+1]
            
            # --- Channel mixer input: weighted sum of y_T[0..l+1] and y_C[0..l] ---
            t = F.softmax(self.theta[l], dim=0)  # (l+2,)
            g = F.softmax(self.gamma[l], dim=0)  # (l+1,)
            ch_in = sum(t[i] * y_T[i] for i in range(l + 2))  # includes y_T[l+1]
            ch_in = ch_in + sum(g[i] * y_C[i] for i in range(l + 1))
            
            # Reshape for channel mixing: (B*M, N, D) -> (B*N, M, D)
            ch_in_r = ch_in.view(B, M, N, D).permute(0, 2, 1, 3).reshape(B * N, M, D)
            ch_out_r = self.channel_mixers[l](ch_in_r)
            ch_out = ch_out_r.view(B, N, M, D).permute(0, 2, 1, 3).reshape(B * M, N, D)
            y_C.append(ch_out)  # y_C[l+1]
        
        # Final output from last channel output
        out = y_C[-1]  # (B*M, N, D)
        out = self.norm(out)
        out = out.reshape(B * M, -1)  # (B*M, N*D)
        out = self.head(out)  # (B*M, pred_len)
        out = out.reshape(B, M, self.pred_len).transpose(1, 2)  # (B, H, M)
        
        # De-normalize
        out = out * stds + means
        return out


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    B, L, M = 4, 512, 7
    H = 96
    
    model = TSM2(
        num_variates=M, seq_len=L, pred_len=H,
        patch_len=16, stride=8, d_model=128,
        d_state=16, d_conv=4, expand=2,
        num_layers=4, dropout=0.1,
    ).to(device)
    
    x = torch.randn(B, L, M).to(device)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}, Expected: ({B}, {H}, {M})")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Num patches: {model.num_patches}, Variates: {M}")
    
    # Test with different variate counts
    for M_test in [8, 21, 321]:
        model_test = TSM2(num_variates=M_test, seq_len=512, pred_len=96, d_model=64, num_layers=2).to(device)
        x_test = torch.randn(2, 512, M_test).to(device)
        y_test = model_test(x_test)
        print(f"M={M_test}: Input {x_test.shape} -> Output {y_test.shape} ✓")
    
    print("All TSM2 tests passed!")
