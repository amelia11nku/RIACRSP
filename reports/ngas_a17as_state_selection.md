# NGAS A1.7A-S deterministic state selection

The frozen supplemental set contains **27 existing clean non-R12 states**. Each
S/M/L × EARLY/MIDDLE/LATE cell contains exactly three states: two governed TRAIN
states and one governed VALIDATION state. No 20–40% or 60–80% state is included.

Selection is deterministic. Within each cell, TRAIN is selected before VALIDATION;
each choice maximizes newly covered condition tags, then the sum of reciprocal tag
frequencies in the eligible scale-stage pool, then uses lexicographic `state_key`.
All 27 state keys and payload hashes are unique. Supplemental labels are audit-only
and ineligible for future training loaders.

- Selection manifest: `artifacts/ngas_a17as/supplemental_protocol_manifest.json`
- Selected rows: `reports/ngas_a17as_state_selection.csv`
- Coverage table: `reports/ngas_a17as_state_selection_coverage.csv`
