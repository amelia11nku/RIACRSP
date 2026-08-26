# Tiny 01 Exact Schedule Validation

Solver: `exhaustive-active-schedule-bnb`

Status: `OPTIMAL`

Optimal makespan: `157.0`

## J1

- `o11 @ M3`; config `C1`; process `[19.0, 26.0]`.
  W: `WH → M3 by W1 [0.0, 19.0]`.
  F: `WH → M3 → WH` by `F1`; arrival `13.0`, return `26.0`.
- `o12 @ M2`; config `C2`; process `[41.0, 51.0]`.
  W: `M3 → M2 by W1 [26.0, 29.0]`.
  F: `WH → M2 → WH` by `F1`; arrival `41.0`, return `56.0`.
- `o13 @ M3`; config `C8`; process `[121.0, 135.0]`.
  W: `M2 → M3 by W1 [62.0, 65.0]`.
  F: `WH → M3 → WH` by `F1`; arrival `121.0`, return `134.0`.

## J2

- `o22 @ M3`; config `C4`; process `[69.0, 78.0]`.
  W: `WH → M3 by W1 [41.0, 60.0]`.
  F: `WH → M3 → WH` by `F1`; arrival `69.0`, return `82.0`.
- `o21 @ M3`; config `C4`; process `[95.0, 105.0]`.
  W: `same island — NONE`.
  F: `WH → M3 → WH` by `F1`; arrival `95.0`, return `108.0`.
- `o23 @ M2`; config `C2`; process `[149.0, 157.0]`.
  W: `M3 → M2 by W1 [105.0, 108.0]`.
  F: `WH → M2 → WH` by `F1`; arrival `149.0`, return `164.0`.

## Metrics

- Makespan: 157.0
- Reconfiguration count/time: 3 / 11.0
- W loaded/empty travel: 49.0 / 26.0
- F travel: 252.0
- Total cost: 81.12299999999999
- Feasible: TRUE

`TINY_EXACT_VALIDATED = TRUE`
