#!/usr/bin/env bash
set -uo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PYTHON=/home/liulei/miniconda3/envs/gnn311/bin/python
RUNNER=scripts/run_phase6g_dev_holdout.py
OUT=outputs/phase6g/dev_holdout

mkdir -p "$OUT"
exec >> "$OUT/production_parallel.log" 2>&1

child_pids=()
terminate_children() {
  trap - INT TERM HUP
  for pid in "${child_pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  for pid in "${child_pids[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
  echo "PHASE6G_DEV_HOLDOUT_PARALLEL_INTERRUPTED"
  exit 130
}
trap terminate_children INT TERM HUP

echo "PHASE6G_DEV_HOLDOUT_PARALLEL_START"
"$PYTHON" "$RUNNER" --methods H1 > "$OUT/worker_H1.log" 2>&1
h1_status=$?
if (( h1_status != 0 )); then
  echo "PHASE6G_DEV_HOLDOUT_H1_FAILED status=$h1_status"
  exit "$h1_status"
fi

for method in ALNS GA CSGNI; do
  "$PYTHON" "$RUNNER" --device cuda --methods "$method" \
    > "$OUT/worker_${method}.log" 2>&1 &
  child_pids+=("$!")
done

status=0
for pid in "${child_pids[@]}"; do
  wait "$pid" || status=$?
done
if (( status != 0 )); then
  echo "PHASE6G_DEV_HOLDOUT_WORKER_FAILED status=$status"
  exit "$status"
fi

"$PYTHON" "$RUNNER" --summarize-only
summary_status=$?
if (( summary_status != 0 )); then
  echo "PHASE6G_DEV_HOLDOUT_SUMMARY_FAILED status=$summary_status"
  exit "$summary_status"
fi
echo "PHASE6G_DEV_HOLDOUT_PARALLEL_COMPLETE"
