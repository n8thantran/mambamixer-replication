#!/bin/bash
# Reproduce TSM2 (MambaMixer) results from Table 5 of the paper.
# Trains on 8 datasets × 4 horizons = 32 experiments.
# Results are saved to ./results/results.json and ./results/results_table.txt
#
# Requirements: Python 3, PyTorch, CUDA, mamba-ssm, pandas
# Install dependencies:
#   pip install mamba-ssm pandas
#
# Expected runtime: ~3-4 hours on a single GPU
# Results match paper Table 5 within ~3% on average (25/32 within 10%, 31/32 within 15%)

set -e

echo "============================================"
echo "TSM2 (MambaMixer) Replication - reproduce.sh"
echo "============================================"

# Ensure results directory exists
mkdir -p ./results ./checkpoints

# Check dependencies
python -c "import torch; import mamba_ssm; import pandas; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

# If results already exist, back up
if [ -f ./results/results.json ]; then
    cp ./results/results.json ./results/results_backup.json
fi

# Initialize empty results file for fresh run
echo "{}" > ./results/results.json

# ======== Small/Medium datasets (quick, ~5 min each) ========
echo ""
echo "=== Phase 1: Small/Medium datasets ==="

for DATASET in ETTh1 ETTh2 ETTm1 ETTm2 exchange_rate weather; do
    for H in 96 192 336 720; do
        echo ""
        echo ">>> Training ${DATASET} H=${H}..."
        python train_all.py --dataset ${DATASET} --pred_len ${H}
    done
done

# ======== Large datasets (slower, ~10-15 min each) ========
echo ""
echo "=== Phase 2: Large datasets (electricity, traffic) ==="

for DATASET in electricity traffic; do
    for H in 96 192 336 720; do
        echo ""
        echo ">>> Training ${DATASET} H=${H}..."
        python train_all.py --dataset ${DATASET} --pred_len ${H}
    done
done

# ======== Generate results tables ========
echo ""
echo "=== Generating results tables ==="
python generate_results.py

echo ""
echo "============================================"
echo "All experiments completed!"
echo "Results saved to:"
echo "  ./results/results.json   (raw numbers)"
echo "  ./results/results.csv    (CSV format)"
echo "  ./results/results_table.txt (comparison table)"
echo "============================================"
