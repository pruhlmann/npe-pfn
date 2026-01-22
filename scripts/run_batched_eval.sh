#!/bin/bash
#OAR -n npe_pfn_batched_eval
#OAR -l /nodes=1/gpunum=1,walltime=6:00:00
#OAR -p gpumodel='V100'
#OAR --stdout npe_pfn_batched_eval_%jobid%.out
#OAR --stderr npe_pfn_batched_eval_%jobid%.err

# NPE-PFN Batched Evaluation on RoPEFM Tasks
# Uses sample_batched for efficient joint metric computation
#
# Submit with: oarsub -S ./scripts/run_batched_eval.sh

set -e

echo "========================================"
echo "NPE-PFN Batched Evaluation"
echo "========================================"
echo "Job ID: $OAR_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "========================================"

# Load conda environment
source ~/.bashrc
conda activate npe-fn

# Navigate to project directory
cd /home/pruhlman/Doctorat/project/npe-pfn

# Data paths on edgar
DATA_PATH="/scratch/clear/pruhlman/npe-pfn/data"
OUTPUT_DIR="/scratch/clear/pruhlman/npe-pfn/results/batched_eval_$(date +%Y%m%d_%H%M%S)"

echo "Data path: $DATA_PATH"
echo "Output dir: $OUTPUT_DIR"

# Check that data exists
echo ""
echo "Checking data availability..."
for task in pendulum wind_tunnel light_tunnel; do
    if [ -d "${DATA_PATH}/${task}" ]; then
        echo "  ✓ ${task} data found"
    else
        echo "  ✗ ${task} data NOT found at ${DATA_PATH}/${task}"
    fi
done

# Run evaluation
echo ""
echo "Starting evaluation..."
python scripts/evaluate_ropefm_batched.py \
    --data_path "${DATA_PATH}" \
    --tasks pendulum wind_tunnel light_tunnel \
    --num_cal 10 50 200 1000 \
    --seeds 0 1 2 3 4 \
    --max_test_samples 2000 \
    --projection_dim 128 \
    --output_dir "${OUTPUT_DIR}"

echo ""
echo "========================================"
echo "Evaluation complete!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "========================================"
