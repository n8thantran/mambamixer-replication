# MambaMixer (TSM2) Replication Progress

## Current Phase
ALL 32 EXPERIMENTS COMPLETE. Creating final deliverables (reproduce.sh, results/).

## Implementation Plan
- [x] 1. Set up environment (mamba-ssm installed)
- [x] 2-6. TSM2 model architecture (tsm2_model.py) — CLEAN, TESTED
- [x] 7. Data loader (data_loader.py) — works for all 8 datasets
- [x] 8. Training script (train_all.py) — with early stopping, AdamW, CosineAnnealing
- [x] 9. Fixed channel mixer to properly mix across variates (B*N, M, D)
- [x] 10. Fixed weighted averaging: 4 weight sets (alpha, beta, theta, gamma) per Eq 8-9
- [x] 11a. ETTh1: 0.417/0.446/0.468/0.485 (targets: 0.375/0.398/0.419/0.422)
- [x] 11b. ETTh2: 0.304/0.378/0.398/0.423 (targets: 0.253/0.334/0.347/0.401)
- [x] 11c. exchange_rate: 0.118/0.212/0.375/0.980 (targets: 0.163/0.229/0.383/0.999)
- [x] 11d. ETTm1: 0.310/0.350/0.379/0.439 (targets: 0.322/0.349/0.366/0.407)
- [x] 11e. ETTm2: 0.180/0.237/0.304/0.380 (targets: 0.173/0.230/0.279/0.388)
- [x] 11f. weather: 0.153/0.196/0.248/0.321 (targets: 0.161/0.208/0.252/0.337)
- [x] 11g. electricity: 0.147/0.159/0.175/0.215 (targets: 0.142/0.153/0.175/0.209)
- [x] 11h. traffic: 0.412/0.421/0.431/0.470 (targets: 0.396/0.408/0.427/0.449)
- [x] 12. reproduce.sh — CREATED
- [x] 13. Results tables in /workspace/results/ — CREATED

## Final Results Summary
- **Overall MSE ratio: 1.029x** (within 3% of paper on average)
- **25/32 within 10%** of paper
- **31/32 within 15%** of paper
- **11/32 better than paper** (especially exchange_rate, weather)
- **MAE ratio: 1.041x**, 29/32 within 10%

## Model Architecture (CORRECT)
- Token Mixer: (B*M, N, D) - unidirectional Mamba along N patches, per variate ✓
- Channel Mixer: (B*N, M, D) - bidirectional Mamba along M variates, per patch ✓
- Weighted averaging: 4 sets (alpha, beta for token input; theta, gamma for channel input) ✓
- 2D normalization, RevIN, patch embedding with positional encoding ✓

## Training Config
All datasets: d_model=32, num_layers=2, d_state=16, d_conv=4, expand=2, lr=1e-4, wd=0.01
- ETTh1/h2: batch=128, epochs=15, patience=5, dropout=0.3
- ETTm1/m2: batch=128, epochs=15, patience=5, dropout=0.3
- exchange_rate: batch=32, epochs=20, patience=5, dropout=0.3
- weather: batch=64, epochs=12, patience=5, dropout=0.3
- electricity: batch=8, epochs=8, patience=3, dropout=0.2
- traffic: batch=4, epochs=8, patience=3, dropout=0.2, expand=1

## Key Files
- tsm2_model.py: TSM2 model (token mixer, channel mixer, weighted avg, 2D norm, RevIN)
- data_loader.py: TimeSeriesDataset, get_dataloaders(), DATASET_INFO for all 8 datasets
- train_all.py: Main training script with per-dataset configs and early stopping
- train_large.py: Variant for large datasets (not needed in final flow)
- generate_results.py: Generate comparison tables and CSV
- reproduce.sh: Master script to regenerate all results
- results/results.json: All 32 MSE/MAE results
- results/results.csv: CSV format results with paper comparison
- results/results_table.txt: Formatted comparison table

## Rubric Status
- [x] Model architecture matches paper (dual Mamba SSM, token+channel mixing, weighted avg)
- [x] All 8 datasets × 4 horizons = 32 experiments completed
- [x] Results close to paper (avg 1.029x MSE ratio)
- [x] reproduce.sh created and tested
- [x] Results saved in /workspace/results/
- [x] Code is clean and runnable

## Failed Approaches (for reference)
- Initial attempt with d_model=64, num_layers=4 was too large/overfit
- Channel mixer initially processed wrong dimension (per-patch instead of per-variate)
- Weighted averaging initially had only 2 weight sets instead of 4 per paper
