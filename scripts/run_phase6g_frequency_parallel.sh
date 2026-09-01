#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PYTHON=/home/liulei/miniconda3/envs/gnn311/bin/python
RUNNER=scripts/tune_phase6g_intervention_rate.py
OUT=outputs/phase6g/frequency_study

echo "PHASE6G_PARALLEL_BASELINE_START"
baseline_pids=()
for shard in 0 1 2; do
  "$PYTHON" "$RUNNER" --device cuda --methods ALNS \
    --shard-count 3 --shard-index "$shard" \
    > "$OUT/worker_ALNS_${shard}.log" 2>&1 &
  baseline_pids+=("$!")
done
for pid in "${baseline_pids[@]}"; do
  wait "$pid"
done

echo "PHASE6G_PARALLEL_RATES_START"
rate_pids=()
rates=(R20 R50 R100)
for rate in "${rates[@]}"; do
  "$PYTHON" "$RUNNER" --device cuda --methods "$rate" \
    > "$OUT/worker_${rate}.log" 2>&1 &
  rate_pids+=("$!")
done
for pid in "${rate_pids[@]}"; do
  wait "$pid"
done

"$PYTHON" "$RUNNER" --summarize-only
echo "PHASE6G_PARALLEL_COMPLETE"
