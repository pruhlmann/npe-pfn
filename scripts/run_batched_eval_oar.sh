#!/bin/bash
#OAR -l walltime=6:00:00

# NPE-PFN Batched Evaluation - Single Task
# This script is designed to be submitted 3 times in parallel, once per task
#
# Usage:
#   ./scripts/launch_batched_eval.sh
# Or manually:
#   oarsub -S "./scripts/run_batched_eval_oar.sh pendulum"

set -e

# Get task from argument
TASK=$1

if [ -z "$TASK" ]; then
    echo "Error: No task specified. Pass task as argument."
    exit 1
fi

echo "========================================"
echo "NPE-PFN Batched Evaluation - $TASK"
echo "========================================"
echo "Job ID: $OAR_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Task: $TASK"
echo "========================================"

# Load conda environment
source /home/pruhlman/miniconda3/etc/profile.d/conda.sh
conda activate npe-fn

# Navigate to project directory
cd /home/pruhlman/project/npe-pfn

# Data paths on edgar
DATA_PATH="/scratch/clear/pruhlman/npe-pfn/data"
OUTPUT_DIR="/scratch/clear/pruhlman/npe-pfn/results/batched_eval_$(date +%Y%m%d)"

echo "Data path: $DATA_PATH"
echo "Output dir: $OUTPUT_DIR"

# Create output directory
mkdir -p $OUTPUT_DIR

# Check that task data exists
echo ""
echo "Checking data availability..."
if [ -d "${DATA_PATH}/${TASK}" ]; then
    echo "  ✓ ${TASK} data found"
else
    echo "  ✗ ${TASK} data NOT found at ${DATA_PATH}/${TASK}"
    exit 1
fi

# Run evaluation for this task
echo ""
echo "Starting evaluation for ${TASK}..."
python scripts/evaluate_ropefm_batched.py \
    --data_path "${DATA_PATH}" \
    --tasks ${TASK} \
    --num_cal 10 50 200 1000 \
    --seeds 0 1 2 3 4 \
    --max_test_samples 5000 \
    --projection_dim 128 \
    --output_dir "${OUTPUT_DIR}"

echo ""
echo "========================================"
echo "Evaluation complete for ${TASK}!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "========================================"
