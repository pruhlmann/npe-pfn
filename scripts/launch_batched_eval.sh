#!/bin/bash
# Launch NPE-PFN Batched Evaluation for all 3 tasks in parallel
#
# Usage:
#   ./scripts/launch_batched_eval.sh
#
# This will submit 3 OAR jobs, one for each task:
#   - pendulum
#   - wind_tunnel
#   - light_tunnel

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_SCRIPT="${SCRIPT_DIR}/run_batched_eval_oar.sh"

echo "========================================"
echo "Launching NPE-PFN Batched Evaluation"
echo "========================================"
echo "Submitting 3 parallel jobs..."
echo ""

# Submit jobs for each task
for TASK in pendulum wind_tunnel light_tunnel; do
    echo "Submitting job for: ${TASK}"
    JOB_ID=$(oarsub -n "npe_pfn_${TASK}" \
        -l walltime=6:00:00 \
        -O "npe_pfn_${TASK}_%jobid%.out" \
        -E "npe_pfn_${TASK}_%jobid%.err" \
        -S "${JOB_SCRIPT} ${TASK}" 2>&1 | grep -oP 'OAR_JOB_ID=\K\d+' || echo "submitted")
    echo "  -> Job ID: ${JOB_ID}"
done

echo ""
echo "========================================"
echo "All jobs submitted!"
echo "Check status with: oarstat -u \$USER"
echo "========================================"
