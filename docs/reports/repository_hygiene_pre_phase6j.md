# Repository hygiene before Phase 6J

## Decision

The maintenance pass is complete at the working-tree level. It removes proven
duplicate payloads and runtime artifacts without changing experiment outcomes,
frozen data boundaries, candidate-bank semantics, decoder behavior,
feasibility behavior, or Phase 6I-MR conclusions. Phase 6J may start only from
the local maintenance checkpoint created after the validations in this report.

No Git history was rewritten and no `git push` was executed. Deleting files in
the new maintenance commit does not reduce the size of blobs retained by older
Git history.

## Baseline

- Branch: `main`
- Starting HEAD: `fa97fec6cac2a94cd1bdc39cbb8c0001f411fa2c`
- Upstream: `upstream/main` at the same commit
- `origin/main`: fetched successfully at the same commit
- Additional `SOFJSPT` remote: fetch required unavailable credentials; this did
  not affect RI-ACRSP baseline verification
- Initial worktree: clean, with no untracked user files
- Active jobs: none; `phase6i-mr-r11-v2.service` was absent/inactive and no
  Phase 6 process was running

The initial full-suite command was:

```text
/home/liulei/miniconda3/envs/gnn311/bin/python -m pytest -q
```

It stopped during collection with three errors because
`rcias_clgri/exact/general_gurobi.py` supplied the identical keyword argument
`h1_mip_start_used=bool(use_h1_mip_start)` twice in one constructor call. The
starting commit contains the same duplicate. The maintenance pass removed only
the second occurrence; this is a syntax-only repair of an otherwise
unparseable module, not an algorithm change.

## Inventory and classification

The detailed pre-cleanup inventory is
[`repository_inventory_pre_phase6j.csv`](repository_inventory_pre_phase6j.csv),
with the top 50 tracked/local/history files in
[`repository_largest_files_pre_phase6j.csv`](repository_largest_files_pre_phase6j.csv).
It distinguishes filesystem size from reachable Git-history/object size.

Every identified tracked cleanup candidate, historical report, root-level
document and phase script was assigned a class before removal in
[`repository_cleanup_classification_pre_phase6j.csv`](repository_cleanup_classification_pre_phase6j.csv):

| Classification | Files | Applied policy |
|---|---:|---|
| `KEEP_TRACKED_ACTIVE` | 70 | Active source, paper export/QA contract, or compact derived artifact |
| `KEEP_TRACKED_PROVENANCE` | 916 | Frozen phase logic, reports, canonical evidence, or non-duplicated raw provenance |
| `KEEP_LOCAL_IGNORED` | 379 | Generated Parquet live logs retained locally at their existing paths |
| `ARCHIVE_POINTER_ONLY` | 92 | 90 byte-identical raw copies and 2 superseded root documents |
| `DELETE_SAFE` | 5 | Zero-byte stale `.progress.lock` files from completed/superseded jobs |

Ambiguous historical scripts and reports were retained. An unreferenced script
was not treated as obsolete merely because no current command named it.

## Changes

### Raw-result deduplication

The 90 files below
`paper_experiments/ablation/raw_results/full_reference/runs/` occupied
19,260,270 bytes and matched the retained Phase 6H Core45 files one-for-one by
SHA-256. The pre-existing replay audit covers all pairs. The copies were
removed, while their original paths, hashes, retained canonical paths and
recovery commands were recorded in
[`archive_manifest.csv`](../history/archive_manifest.csv).

`paper_experiments/ablation/analyze_p1_ablation.py` now reads the canonical
Phase 6H source directly and validates it against the new 180-row
[`canonical_reference_manifest.csv`](../../paper_experiments/ablation/audit/canonical_reference_manifest.csv).
`prepare_p1_ablation.py` no longer creates a self-contained duplicate result
tree.

The 90 `no_ni_alns_h1_equivalence` files were retained. Their earlier source
exists only below ignored `outputs/`, so deleting the tracked paper copy would
remove the only repository-visible reproduction payload.

### Runtime artifacts

- 379 `live_logs/*.parquet` files, 236,829,962 bytes, were removed from Git
  tracking but retained locally at the same paths for optional diagnostic
  regeneration.
- Five zero-byte `.progress.lock` files were deleted after confirming no active
  process or service used them.
- `.gitignore` now covers `artifacts/`, `.progress.lock`, PID files,
  `live_logs/`, `.mypy_cache/`, and `.ruff_cache/` without globally ignoring
  JSON, CSV, SVG, EPS, or Parquet.

### Historical material and root cleanup

`notes.txt` and `implementation_validation_report.md` (10,011 bytes together)
explicitly identified
[`phase2_final_validation_report.md`](phase2_final_validation_report.md) as
their current replacement, so they were removed from active HEAD and indexed
for recovery. No other historical report was deleted.

[`ALGORITHM_PHASE_HISTORY.md`](../history/ALGORITHM_PHASE_HISTORY.md) now gives
one concise phase/status/report/commit/successor index. No phase-specific
script was removed or mechanically renamed because the retained scripts encode
reproduction logic and a safe superseding implementation was not established.

### Phase 6I-MR evidence preservation

Thirty-one compact authoritative files that otherwise exist only below the
ignored `outputs/` tree were hash-copied to
`artifacts/evidence/phase6i_mr_r11/`. That archive is local and ignored; the
tracked pointer and SHA-256 list is
[`phase6i_mr_local_evidence_manifest.csv`](../history/phase6i_mr_local_evidence_manifest.csv).
The original `outputs/` copies and all raw R11 results remain untouched.

Preserved conclusions and boundaries:

- final decision: `MODEL_REVISION`;
- selected candidate: `U2_MIXED_OLD_NEW`;
- selected artifact SHA-256:
  `dc59abbe34be21480c8f855e2bd62e5253fdf3f15f947c94bc124c5cbd4ec89a`;
- R11: 288/288 complete, replay/integrity `PASS`, one access, no retuning;
- Phase 6H Core45 remains the manuscript evidence;
- the proposal bank remains 24 deterministic rules followed by target-set
  deduplication; `candidate_trials=8` remains repair/decoder trials per target.

## Before and after

| Measure | Before | After | Change |
|---|---:|---:|---:|
| Tracked files | 3,057 | 2,593 | -464 |
| Tracked active-tree bytes | 823,252,594 | 568,333,804 | -254,918,790 |
| Working-tree files, excluding `.git` | 22,452 | 22,399 | -53 |
| Working-tree bytes, excluding `.git` | 39,251,619,596 | 39,234,176,753 | -17,442,843 |
| Ignored/local files | 19,394 | 19,806 | +412 |
| Ignored/local bytes | 38,428,359,122 | 38,665,842,949 | +237,483,827 |
| `.git/objects` filesystem bytes | 1,765,825,271 | 1,766,025,729* | +200,458* |
| Removed byte-identical raw copies | 0 | 90 | +90 |
| Historical documents consolidated | 0 | 2 | +2 |
| Scripts removed/moved | 0 | 0 | 0 |

The ignored/local increase reflects the 379 preserved logs plus the 31-file
evidence archive. The active tracked tree is smaller, but reachable historical Git blobs remain.
Any remote-size reduction would require a separately approved history-rewrite
procedure; none was attempted here.

`*` The post-cleanup object-store value is the inventory snapshot immediately
before the checkpoint commit. Staging and committing the inventory itself adds
a small amount of Git metadata without changing active-tree contents.

## Validation

| Check | Result |
|---|---|
| Full Python suite | `250 passed in 13.13s` after the syntax-only repair |
| Compile all active Python | `PASS` |
| Canonical benchmark byte regeneration | `130/130 PASS` |
| Shared-decoder small validation | H1/H2/H3 feasible on all three tiny instances; exact solver `OPTIMAL` |
| Phase 6I-MR R11 audit | 288 results, 180 live logs, 900 forced states; every integrity/replay/feasibility check true |
| P1 ablation regeneration | 270/270 schedules replayed exactly and feasibly; Friedman/post-hoc statistics unchanged |
| Paper package | `13/13 PASS`, including Phase 6H Core45, figures and QA |
| Ignoring policy | local evidence and all live-log directories ignored; no live log remains tracked |

P1 compact statistical outputs retained their pre-cleanup hashes. The detailed
`ablation_runs.csv` changed only its 90 Full-arm `result_path` values from the
removed duplicate tree to the canonical Phase 6H tree; objective, feasibility,
runtime, decoder count and statistical values are unchanged.

## `REVIEW_REQUIRED`

The frozen `configs/phase6i_mr_command_manifest.json` contains two composite
stage names that have never existed as files in reachable repository history:

- `scripts/train_phase6i_mr_revision.py`;
- `scripts/run_phase6i_mr_hard_state_round.py`.

The concrete frozen implementation files (`train_phase6i_mr_heads.py`,
`train_phase6i_mr_u3.py`, `build_phase6i_mr_hard_states.py`, and
`refit_phase6i_mr_heads.py`) remain present and are hashed by the lower-level
training protocols. The completed Phase 6I-MR command manifest was retained
unchanged because inventing compatibility wrappers after R11 would falsify its
historical provenance. This is a non-blocking historical-manifest discrepancy
for Phase 6J, which must receive a new preregistration and command manifest
before any long run.
