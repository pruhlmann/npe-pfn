#!/bin/bash
#OAR -n npe_pfn_eval
#OAR -l /nodes=1/gpunum=1,walltime=12:00:00
#OAR -t besteffort
#OAR -t idempotent
#OAR --array-param-file /home/pruhlman/project/npe-pfn/scripts/tasks.txt

# NPE-PFN Evaluation Script
# Launch with: oarsub --scanscript ./run.sh

set -e

# Configuration
PROJECT_DIR="/home/pruhlman/project/npe-pfn"
DATA_PATH="/scratch/clear/pruhlman/npe-pfn/data"
OUTPUT_DIR="/scratch/clear/pruhlman/npe-pfn/results"
CONDA_ENV="npe-pfn"

# Create output directory
mkdir -p $OUTPUT_DIR

# Activate conda
source /home/pruhlman/miniconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV

# Change to project directory
cd $PROJECT_DIR

# Task is passed as array parameter
TASK=$1

echo "=============================================="
echo "Running NPE-PFN evaluation"
echo "Task: $TASK"
echo "Data path: $DATA_PATH"
echo "Output dir: $OUTPUT_DIR"
echo "=============================================="

# Run evaluation
python scripts/evaluate_ropefm.py \
    --data_path $DATA_PATH \
    --tasks $TASK \
    --num_cal 10 50 200 1000 \
    --seeds 0 \
    --max_test_samples 2000 \
    --batch_size 10 \
    --gpu_num 0 \
    --output_dir $OUTPUT_DIR

echo "=============================================="
echo "Evaluation complete for task: $TASK"
echo "=============================================="
