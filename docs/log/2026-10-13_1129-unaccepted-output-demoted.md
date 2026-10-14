# GH-1129 — unaccepted output demoted (#1142)

## Round 7 (2026-10-14)

**P0 — refusal evidence.** `_positive_judge_refusal_evidence` no longer treats every
non-empty `judge_reason` as refusal. Production stamps accept-route reasons on every
accepted attempt (`faithful`, `heuristics passed`, `native_table_verifier: EXACT_PASS`
from `agentic.py`); those are excluded via `_JUDGE_ACCEPT_ROUTE_REASONS`. Refusal is
proven by `rejection_class` or a non-accept `judge_reason`.

**P1 — fixtures.** `test_ladder_resume` and `test_completed_without_refusal_evidence_*`
parametrised over the three accept reasons with production-shaped sidecars.

**P2.** (a) `rejection_class` forwarded on `BEST_ATTEMPT_FLAGGED` and `WHOLE_DOC_SECTION`;
(b) GH-161 `resume_ledger_audit_reject` recorded before the WARNING status gate;
(c) unaccepted banner idempotence replaces regex-matched forged lines with the canonical
`unaccepted_output_mark` bytes; (d) HTML tag stripper uses a tag-aware pattern so
`values <5 and >3 compared` is preserved.

**Replay (run 23117187, worktree finalize):** `sidecars_scanned=17091`;
`success_and_audit_false_before=385`; `success_and_audit_false_after_finalize=0`;
868 pages change status/text on replay (385 success→demoted, 477 warning, 6 error).
On the 385 cohort, all 385 still earn the unaccepted mark after finalize (0 accept-route-only
rows in that historical slice — every sidecar carries a non-accept `judge_reason`, plus 28
with `rejection_class`).

**Tests:** `7467 passed` (`scratch/routing-baseline/exec/1129/pytest-r7.txt`).
**Ruff:** `uvx ruff@0.16.0 format --check .` clean after format.
**Mutants:** `exec/1129/mutants-r7.txt` — m1–m4 kill their target tests (manual
`pytest -c` patches; copied-tree runner needs repair).

## Round 8 (2026-10-14)

**P0 — `judge_accepted`.** New persisted `PageOutput.judge_accepted: bool | None`
set from `att.accepted` beside `audit_passed` in `orchestrator.py` (~10075) for
every agentic attempt; never rewritten by C3 demotion. Serialized in sidecar /
blob (`to_dict` / `from_dict`), forwarded on `_WholeDoc`, `BEST_ATTEMPT_FLAGGED`,
and `WHOLE_DOC_SECTION`. Refusal evidence is `judge_accepted is False` or non-empty
`rejection_class` only; `judge_reason` is banner text. Removed `_JUDGE_ACCEPT_ROUTE_REASONS`.
`attempts_summary.accepted` prefers `judge_accepted` when set.

**P2 — resume gate.** ERROR terminal pages return before the GH-161 audit gate (no
`resume_ledger_audit_reject`); WARNING + `audit_passed=False` (#1129 demotion) still
emit the event.

**385 split (run 23117187, SUCCESS + `audit_passed=false`):** All **385** have
`attempts_summary` (written at sidecar flush via `_attempts_summary`, after routing
stamps each attempt's `audit_passed` from `att.accepted` at ~10075; per-attempt
rows are not rewritten when C3 demotes the winner's status). Every row shows
`accepted: false` → **385 judge-refused** on the journal, **0 judge-accept +
table-ladder demote** in this slice (`table_ladder_disposition` absent on all 385).
Legacy sidecars lack `judge_accepted`; R8 finalize marks only **67** of 385
(28 `rejection_class`, 39 missing-verdict `judge_outcome`); **318** keep demoted
bytes without the unaccepted banner (`judge_reason` alone is not evidence).

**Tests:** `7473 passed` (`scratch/routing-baseline/exec/1129/pytest-r8.txt`).
**Ruff:** `uvx ruff@0.16.0 format --check .` clean.
**Mutants:** `exec/1129/mutants-r8.txt`.
