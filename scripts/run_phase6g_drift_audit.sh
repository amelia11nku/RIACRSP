#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p outputs/phase6g/drift_audit
exec > outputs/phase6g/drift_audit/production.log 2>&1
exec /home/liulei/miniconda3/envs/gnn311/bin/python scripts/run_phase6g_drift_audit.py --device cuda
