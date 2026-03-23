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
5. RevIN (reversible instance normalization)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
import math


class Norm2D(nn.Module):
    """2D normalization on both time and feature dimensions (Section 3.3)."""
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


class TSM2(nn.Module):
    """Time Series MambaMixer for multivariate time series forecasting.
    
    Architecture per paper Section 3.4:
    - Patch embedding with positional encoding
    - L layers of: Token Mixer (unidirectional Mamba) + Channel Mixer (bidirectional Mamba)
    - Weighted averaging of earlier features per Eq. 8-9
    - 2D Normalization
    - Flatten prediction head
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
        self.input_dropout = nn.Dropout(dropout)
        
        # MambaMixer layers
        self.token_mixers = nn.ModuleList()
        self.channel_mixers_fwd = nn.ModuleList()
        self.channel_mixers_bwd = nn.ModuleList()
        self.token_norms = nn.ModuleList()
        self.channel_norms = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        for _ in range(num_layers):
            self.token_mixers.append(
                Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand))
            self.channel_mixers_fwd.append(
                Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand))
            self.channel_mixers_bwd.append(
                Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand))
            self.token_norms.append(nn.LayerNorm(d_model))
            self.channel_norms.append(nn.LayerNorm(d_model))
            self.dropout_layers.append(nn.Dropout(dropout))
        
        # Weighted averaging parameters (Eq 8-9)
        # alpha[l]: weights for y_T[0..l], beta[l]: weights for y_C[0..l]  -> token mixer input
        # theta[l]: weights for y_T[0..l+1], gamma[l]: weights for y_C[0..l] -> channel mixer input
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
        self.norm_2d = Norm2D(self.num_patches, d_model)
        self.head = nn.Linear(d_model * self.num_patches, pred_len)
        self.head_dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """x: (B, L, M) -> (B, H, M)"""
        B, L, M = x.shape
        N = self.num_patches
        D = self.d_model
        
        # RevIN: Instance normalization (per-sample, per-variate)
        means = x.mean(dim=1, keepdim=True)
        stds = x.std(dim=1, keepdim=True) + 1e-5
        x = (x - means) / stds
        
        # Patchify: (B, L, M) -> (B*M, N, D)
        x = x.transpose(1, 2)  # (B, M, L)
        x = x.unfold(dimension=2, size=self.patch_len, step=self.stride)  # (B, M, N, P)
        x = x.reshape(B * M, N, self.patch_len)
        x = self.patch_embedding(x) + self.pos_embed
        x = self.input_dropout(x)
        
        # Track outputs for weighted averaging
        y_T = [x]  # token mixer outputs
        y_C = [x]  # channel mixer outputs
        
        for l in range(self.num_layers):
            # --- Token mixer input: weighted avg of y_T[0..l] and y_C[0..l] (Eq 8) ---
            a = F.softmax(self.alpha[l], dim=0)
            b = F.softmax(self.beta[l], dim=0)
            tok_in = sum(a[i] * y_T[i] + b[i] * y_C[i] for i in range(l + 1))
            
            # Token mixing: unidirectional Mamba along N patches per variate
            residual = tok_in
            tok_in = self.token_norms[l](tok_in)
            tok_out = self.token_mixers[l](tok_in)
            tok_out = self.dropout_layers[l](tok_out) + residual
            y_T.append(tok_out)
            
            # --- Channel mixer input: weighted avg of y_T[0..l+1] and y_C[0..l] (Eq 9) ---
            t = F.softmax(self.theta[l], dim=0)
            g = F.softmax(self.gamma[l], dim=0)
            ch_in = sum(t[i] * y_T[i] for i in range(l + 2))
            ch_in = ch_in + sum(g[i] * y_C[i] for i in range(l + 1))
            
            # Channel mixing: bidirectional Mamba along M variates per patch
            residual = ch_in
            ch_r = ch_in.view(B, M, N, D).permute(0, 2, 1, 3).reshape(B * N, M, D)
            ch_r = self.channel_norms[l](ch_r)
            fwd = self.channel_mixers_fwd[l](ch_r)
            bwd = torch.flip(self.channel_mixers_bwd[l](torch.flip(ch_r, [1])), [1])
            ch_out = fwd + bwd
            ch_out = self.dropout_layers[l](ch_out)
            ch_out = ch_out.view(B, N, M, D).permute(0, 2, 1, 3).reshape(B * M, N, D) + residual
            y_C.append(ch_out)
        
        # Final output from last channel output
        out = y_C[-1]  # (B*M, N, D)
        out = self.norm_2d(out)
        out = self.head_dropout(out.reshape(B * M, -1))
        out = self.head(out)  # (B*M, pred_len)
        out = out.reshape(B, M, self.pred_len).transpose(1, 2)  # (B, H, M)
        
        # De-normalize (reverse RevIN)
        out = out * stds + means
        return out


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test with typical ETT-like config
    B, L, M, H = 4, 512, 7, 96
    model = TSM2(
        num_variates=M, seq_len=L, pred_len=H,
        patch_len=16, stride=8, d_model=16,
        d_state=16, d_conv=4, expand=2,
        num_layers=2, dropout=0.3,
    ).to(device)
    
    x = torch.randn(B, L, M).to(device)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test with different variate counts
    for M_test in [8, 21, 321]:
        model_test = TSM2(
            num_variates=M_test, seq_len=512, pred_len=96,
            d_model=16, num_layers=2, dropout=0.3
        ).to(device)
        x_test = torch.randn(2, 512, M_test).to(device)
        y_test = model_test(x_test)
        print(f"M={M_test}: Input {x_test.shape} -> Output {y_test.shape}, "
              f"Params: {sum(p.numel() for p in model_test.parameters()):,}")
    
    print("All TSM2 tests passed!")
