#!/usr/bin/env bash
set -euo pipefail

conda run -n gnn311 python scripts/train_downstream_ppo.py \
  --config configs/phase5b_final.json --seed 520101 --updates 25 --device cuda \
  --out-dir outputs/phase5b/downstream_seed_1

conda run -n gnn311 python scripts/train_downstream_ppo.py \
  --config configs/phase5b_final.json --seed 520102 --updates 25 --device cuda \
  --out-dir outputs/phase5b/downstream_seed_2

conda run -n gnn311 python scripts/train_downstream_ppo.py \
  --config configs/phase5b_final.json --seed 520103 --updates 25 --device cuda \
  --out-dir outputs/phase5b/downstream_seed_3
