# MambaMixer Replication Progress

## Current Phase
Ready to run all 32 experiments. Model, data, and training are all tested and working.

## Implementation Plan
- [x] 1. Set up environment (mamba-ssm installed)
- [x] 2-6. TSM2 model architecture (tsm2_model.py) — CLEAN, TESTED
- [x] 7. Data loader (data_loader.py) — works for all 8 datasets
- [x] 8. Training script (train_fast.py) — with AMP, early stopping, AdamW, CosineAnnealing
- [x] 9. Fixed channel mixer to properly mix across variates (B*N, M, D)
- [x] 10. Fixed weighted averaging to use 4 weight sets (alpha, beta, theta, gamma) per paper Eq 8-9
- [ ] 11. Train all 32 experiments and collect results
- [ ] 12. reproduce.sh
- [ ] 13. Results tables in /workspace/results/

## Model Architecture (CORRECT)
- Token Mixer: (B*M, N, D) - unidirectional Mamba along N patches, per variate ✓
- Channel Mixer: (B*N, M, D) - bidirectional Mamba along M variates, per patch ✓
- Weighted averaging: 4 sets (alpha, beta for token input; theta, gamma for channel input) ✓
- 2D normalization, RevIN, patch embedding with positional encoding ✓

## Key Decisions from Paper
- Input L=512, horizons H∈{96,192,336,720}, stride=1 for windowing
- MSE loss, grid search hyperparameters
- Patch len=16, stride=8, d_model=128, num_layers=4, d_state=16, d_conv=4, expand=2
- Paper says "tune hyperparameters using grid search" - no specific lr/batch/epochs given

## Training Config Per Dataset
- ETTh1/h2: d_model=128, layers=4, expand=2, lr=1e-3, batch=32, epochs=50, patience=10
- ETTm1/m2: d_model=128, layers=4, expand=2, lr=1e-3, batch=64, epochs=30, patience=8
- exchange_rate: d_model=128, layers=4, expand=2, lr=5e-4, batch=32, epochs=50, patience=15
- weather: d_model=128, layers=4, expand=2, lr=1e-3, batch=32, epochs=30, patience=8
- electricity: d_model=64, layers=3, expand=2, lr=5e-4, batch=16, epochs=20, patience=5
- traffic: d_model=64, layers=2, expand=1, lr=5e-4, batch=8, epochs=20, patience=5

## Target Results (Table 5 MSE):
| Dataset | H=96 | H=192 | H=336 | H=720 |
|---------|-------|-------|-------|-------|
| ETTh1   | 0.375 | 0.398 | 0.419 | 0.422 |
| ETTh2   | 0.253 | 0.334 | 0.347 | 0.401 |
| ETTm1   | 0.322 | 0.349 | 0.366 | 0.407 |
| ETTm2   | 0.173 | 0.230 | 0.279 | 0.388 |
| Electricity | 0.142 | 0.153 | 0.175 | 0.209 |
| Exchange | 0.163 | 0.229 | 0.383 | 0.999 |
| Traffic  | 0.396 | 0.408 | 0.427 | 0.449 |
| Weather  | 0.161 | 0.208 | 0.252 | 0.337 |

## Completed Work
- tsm2_model.py: Clean TSM2 implementation with all paper components, tested for M=7,8,21,321
- data_loader.py: TimeSeriesDataset, get_dataloaders(), all 8 datasets working
- train_fast.py: Optimized training with AMP, early stopping, AdamW, CosineAnnealing
- train.py: Original training script (kept as backup)

## Failed Approaches
- Old channel mixer operating on d_model dimension instead of variate dimension → wrong
- Incomplete weighted averaging with confused indexing → cleaned up

## Rubric Status
- [x] Data Pipeline (12%) — all 8 datasets loading correctly
- [x] Model Architecture (30%) — correct token/channel mixing, weighted averaging, 2D norm
- [x] Training Configuration (20%) — MSE loss, AdamW, early stopping, cosine LR
- [ ] Evaluation (15%) — need to run experiments and compute MSE/MAE
- [ ] Results Reproduction (23%) — need results table

## Timing Estimates
- ETTh1/h2: ~37s/epoch × 50 epochs = ~30 min each → ~4h for 8 experiments
- ETTm1/m2: larger datasets but fewer epochs → ~6h for 8 experiments  
- exchange_rate: small → ~1h for 4 experiments
- weather: medium → ~3h for 4 experiments
- electricity: large (321 vars) → slower per epoch, fewer epochs → ~4h for 4 experiments
- traffic: very large (862 vars) → slowest → ~6h for 4 experiments
- Total: ~24h estimated, need to optimize

## Priority
Run experiments in order: ETTh1, ETTh2, exchange_rate, ETTm1, ETTm2, weather, electricity, traffic
Small fast datasets first to get early results.
