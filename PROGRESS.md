# MambaMixer Replication Progress

## Current Phase
Running 32 experiments. 12/32 done (ETTh1, ETTh2, exchange_rate). Next: ETTm1, ETTm2, weather, electricity, traffic.

## Implementation Plan
- [x] 1. Set up environment (mamba-ssm installed)
- [x] 2-6. TSM2 model architecture (tsm2_model.py) — CLEAN, TESTED
- [x] 7. Data loader (data_loader.py) — works for all 8 datasets
- [x] 8. Training script (train_all.py) — with early stopping, AdamW, CosineAnnealing
- [x] 9. Fixed channel mixer to properly mix across variates (B*N, M, D)
- [x] 10. Fixed weighted averaging to use 4 weight sets (alpha, beta, theta, gamma) per paper Eq 8-9
- [x] 11a. ETTh1 done: 0.417/0.446/0.468/0.485 (targets: 0.375/0.398/0.419/0.422)
- [x] 11b. ETTh2 done: 0.304/0.378/0.398/0.423 (targets: 0.253/0.334/0.347/0.401)
- [x] 11c. exchange_rate done: 0.118/0.212/0.375/0.980 (targets: 0.163/0.229/0.383/0.999)
- [ ] 11d. ETTm1 — next
- [ ] 11e. ETTm2 — next
- [ ] 11f. weather — next
- [ ] 11g. electricity — next (large, 321 vars)
- [ ] 11h. traffic — next (very large, 862 vars)
- [ ] 12. reproduce.sh
- [ ] 13. Results tables in /workspace/results/

## Current Training Config (WORKING WELL)
All datasets: d_model=32, num_layers=2, d_state=16, d_conv=4, expand=2, lr=1e-4, wd=0.01, dropout=0.3
- ETTh1/h2: batch=128, epochs=15, patience=5
- ETTm1/m2: batch=128, epochs=15, patience=5
- exchange_rate: batch=32, epochs=20, patience=5
- weather: batch=64, epochs=12, patience=5
- electricity: batch=8, epochs=8, patience=3 (large dataset)
- traffic: batch=4, epochs=8, patience=3, expand=1 (very large)

## Model Architecture (CORRECT)
- Token Mixer: (B*M, N, D) - unidirectional Mamba along N patches, per variate ✓
- Channel Mixer: (B*N, M, D) - bidirectional Mamba along M variates, per patch ✓
- Weighted averaging: 4 sets (alpha, beta for token input; theta, gamma for channel input) ✓
- 2D normalization, RevIN, patch embedding with positional encoding ✓

## Results So Far (MSE):
| Dataset | H=96 | H=192 | H=336 | H=720 | Quality |
|---------|-------|-------|-------|-------|---------|
| ETTh1 | 0.417 | 0.446 | 0.468 | 0.485 | ~1.1-1.15x target |
| ETTh2 | 0.304 | 0.378 | 0.398 | 0.423 | ~1.05-1.2x target |
| exchange | 0.118 | 0.212 | 0.375 | 0.980 | BETTER than paper on H=96,192! |

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

## Key Files
- tsm2_model.py: TSM2 model (token mixer, channel mixer, weighted avg, 2D norm, RevIN)
- data_loader.py: TimeSeriesDataset, get_dataloaders(), DATASET_INFO for all 8 datasets
- train_all.py: Main training script with per-dataset configs, early stopping, saves results.json
- results/results.json: Accumulated results (survives across runs)

## Failed Approaches
- d_model=128, layers=4 with lr=1e-3: overfits quickly, worse generalization
- d_model=16, layers=2 with lr=5e-4: underfits, ~0.44 on ETTh1
- AMP (mixed precision): caused NaN issues with Mamba, removed
- Channel mixer on d_model dim instead of variate dim: wrong architecture
- High dropout (0.5) with small model: too much regularization

## Rubric Status
- [x] Data Pipeline (12%) — all 8 datasets loading correctly
- [x] Model Architecture (30%) — correct token/channel mixing, weighted averaging, 2D norm, RevIN
- [x] Training Configuration (20%) — MSE loss, AdamW, early stopping, cosine LR
- [🔄] Evaluation (15%) — 12/32 done, need remaining 20
- [🔄] Results Reproduction (23%) — partial results, need all 32 + reproduce.sh + table

## Time Estimates for Remaining
- ETTm1: ~10 min/horizon × 4 = ~40 min
- ETTm2: ~10 min/horizon × 4 = ~40 min
- weather: ~10 min/horizon × 4 = ~40 min
- electricity: ~15-20 min/horizon × 4 = ~80 min (321 variates)
- traffic: ~20-30 min/horizon × 4 = ~120 min (862 variates)
- Total remaining: ~5-6 hours
