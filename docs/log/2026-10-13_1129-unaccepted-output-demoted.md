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

## Round 9 (2026-10-14)

**P0-1 — empty refusal reason.** `_unaccepted_mark_reason_from_output` returns fixed
text (`the page judge refused it; no reason was recorded`) when refusal evidence exists
but neither `judge_reason` nor `rejection_class` carries text. `test_process_judge_refusal_empty_reason_still_marks`.

**P0-2 — provenance allowlist.** `_UNACCEPTED_MODEL_DISCLOSURE_PROVENANCES` adds
`FLAGGED_MODEL_KEPT` and `UNVERIFIABLE_TABLE_MODEL_KEPT`. Every `SelectionProvenance`
and refusal reachability (R9):

| Provenance | Refusal evidence can reach? | Visible mark when refused |
| --- | --- | --- |
| `CORRUPT_MATH_HYBRID` | no (hybrid lane, not judge-refused model kept) | — |
| `PASSING_BEST_OUTPUT` | no (judge accepted) | — |
| `UNVERIFIABLE_TABLE_SCANNED` | no (fail-closed marker) | page failed marker |
| `UNVERIFIABLE_TABLE_MODEL_KEPT` | yes | unaccepted banner (#1129) |
| `UNVERIFIABLE_TABLE_NATIVE` | no (native marker) | fail-closed marker |
| `ROTATED_TEXT_SHREDDED` | no | fail-closed marker |
| `INVISIBLE_SCAN_UNREAD` | no | fail-closed marker |
| `FLAGGED_MODEL_KEPT` | yes | unaccepted banner (#1129) |
| `STRUCTURE_CLASS_GRID_CORROBORATED` | no (corroboration ending) | WARNING disposition |
| `STRUCTURE_CLASS_GRID_PASSING` | no | — |
| `STRUCTURE_CLASS_GRID_FLAGGED` | soft-refusal only; ships demoted grid | status / disposition |
| `STRUCTURE_CLASS_JUDGE_TIMEOUT_*` | timeout, not completed refusal | judge-timeout note |
| `STRUCTURE_CLASS_FLOOR` / `*_FLOOR` | no model bytes | fail-closed marker |
| `NATIVE_UNTRUSTED_JUDGE_TIMEOUT` | no | native WARNING |
| `NATIVE_FALLBACK` / `NATIVE_CLEAN` | no (native lane) | native demotion only |
| `WHOLE_DOC_SECTION` | yes | unaccepted banner |
| `BEST_OUTPUT_UNVERIFIED` | yes | unaccepted banner |
| `BEST_ATTEMPT_FLAGGED` | yes | unaccepted banner |
| `NO_TEXT_MARKER` | no | failure marker |

**P1-3 — GH-161 gate.** `resume_ledger_audit_reject` only when
`judge_refusal_evidence_from_winning(winning)` and `status=SUCCESS`; event text
names the stored status and “the judge refused”. WARNING demotions (native fallback,
C3 table-unverified) emit **0** events.

**P1-4 — `judge_accepted` truth.** Stamped from `AcceptDecision.accept` in
`agentic.route_page` (after outcome typing, before append); promotion sets
`judge_accepted=True`; crop re-judge acceptance sets `bo.judge_accepted=True`.
Orchestrator no longer copies `att.accepted` into `judge_accepted` after #1124.

**P2-5 — whole-doc.** Removed whole-doc half of `test_judge_accepted_survives_*`;
production does not stamp `judge_accepted` on `whole_doc_attempts` yet — #1145 will
mark every WARNING page.

**P2-6 — fixtures.** C3 demotion tests seed `audit_passed=True` + `judge_accepted=True`
+ `TABLE_REJECTED`; C3's `if output.audit_passed` branch demotes.

**P2-7 — duplicates.** Removed F811 duplicate tests (lines 781–872).

**Legacy replay:** the **318** pages with `judge_accepted=None` on old sidecars stay
unmarked on replay until a fresh run writes `judge_accepted`; #1145 marks every WARNING
page cluster-wide.

**Tests:** `7481 passed` (`scratch/routing-baseline/exec/1129/pytest-r9.txt`).
**Ruff:** `uvx ruff@0.16.0 format --check .`
**Mutants:** `scratch/routing-baseline/exec/1129/mutants-r9.txt` (copied `src` tree,
`socr.__file__` canary).

## Round 10 (2026-10-14)

**P6 golden restored.** Reverted `prechange_assemble.json` and the emptied oracle from
round 9 (`git checkout origin/main` on the three P6 files). Measured **six** new leaf
diffs vs the frozen baseline — all on page 3 (`D3_MODEL_KEPT`, sidecar
`rejection_class: judge_only` in `prechange_assemble.json`). Added
`UNACCEPTED_OUTPUT_1129_DIFFERENCES` (six `ExpectedDiff` rows) in
`tests/p6_stage_c_oracle.py`; cause is `manifest._apply_unaccepted_output_mark` on
`UNVERIFIABLE_TABLE_MODEL_KEPT` when `_positive_judge_refusal_evidence` sees
`rejection_class`. Stage-C “unaffected surface” tests now compare against
`patched_prechange` where #1129 bytes differ (same pattern as stage A/B).

**Mutant harness.** `scratch/routing-baseline/exec/1129/run-mutants-r10.py` copies
`src/`, `tests/`, `pyproject.toml` into a temp root, runs pytest with that rootdir
(pyproject `pythonpath = ["src"]`), and installs `tests/conftest_mutant_copy_canary.py`
(asserts `socr.__file__` under the copy). Anchors use uncapped `count == 1`. Results:
`mutants-r10.txt`.

**gh262 banner.** `test_gh262_d3_marker_over_cached_grid.py` reuses
`test_gh259_flagged_model_table_wins._state`, which sets
`setattr(model_attempt, "rejection_class", rejection_class)` (default
`SOFT_REJECTION` / `ambiguous_deferred`) before `d3_floor_kept_model_output` selects
`UNVERIFIABLE_TABLE_MODEL_KEPT` — so the unaccepted blockquote is expected.

**Tests:** `7481 passed` (`scratch/routing-baseline/exec/1129/pytest-r10.txt`).
**Ruff:** `uvx ruff@0.16.0 format --check .` clean after oracle format.
**Mutants:** `scratch/routing-baseline/exec/1129/mutants-r10.txt`.

## Round 11 (2026-10-14)

Reviews of f796d561 (Astra, Fable): do not merge.

- **Fail-closed coverage.** The banner allowlist (`_UNACCEPTED_MODEL_DISCLOSURE_PROVENANCES`) kept missing endings; round 10 missed `STRUCTURE_CLASS_GRID_FLAGGED` (Astra, reproduced). It is replaced by `_UNACCEPTED_OUTPUT_MARK_EXCLUDED_PROVENANCES`: any winner with refusal evidence gets the banner unless its ending already ships an equivalent top disclosure (failure markers, floors, invisible-scan and rotated-shred stubs). A new provenance is covered by default.
- **GH-161 resume gate** reverted to origin/main's code (Fable's m4b showed main's gate passes every gh161 test). The legacy SUCCESS + audit_passed=false sidecar (demotion monkeypatched away) emits `resume_ledger_audit_reject` exactly once across two loads. A #1129-demoted WARNING page is not restored and emits nothing.
- **`_rejudge_kept_candidate`** stamps `judge_accepted=True` on an accepting resume re-judge.
- **The journal's `accepted`** keeps main's meaning (final acceptance), so GH-169's escalation trail is intact.
- **Whole-document path** reverted to main. Not stamped with `judge_accepted`; the round-9 table row "WHOLE_DOC_SECTION | yes | banner" was wrong. A refused whole-document page ships WARNING without this banner and is covered by #1145's fail-closed fallback, which marks every WARNING page. Tests that seeded verdict fields on whole-document attempts were deleted.
- **Not in this PR:** replaying outputs written before #1129 (#1153); the second disclosure on D3 model-kept pages (top banner plus in-body note), which #1145's coverage map decides.

Mutants (executed in a copied tree, with a real `conftest.py` canary and a proof-of-copy mutant): every one fails its target test. Log: `scratch/routing-baseline/exec/1129/mutants-r11-run.txt`. Full suite: 7482 passed.

## Round 12 (2026-10-14)

Reviews of 06faa300 (Argon, Fable): do not merge.

- **P1 — accept vetoes stale class.** `_positive_judge_refusal_evidence` checks `judge_accepted is True` before any `rejection_class` left on a #1048-promoted winner. Test: promoted table shape with `judge_accepted=True` + stale class → page-unchecked only, no unaccepted banner.
- **P0/P2 — exclusion set is marker-first only; scanned floor carries judge fields.** Removed `UNVERIFIABLE_TABLE_SCANNED` from `_UNACCEPTED_OUTPUT_MARK_EXCLUDED_PROVENANCES` — that ending splices or recovers prose from the refused attempt, so line 1 is model-derived unless the top banner fires. The synthetic `PageOutput` now forwards `judge_reason`, `judge_outcome`, `judge_accepted`, `rejection_class` from `p.best_output`. Exclusion audit (each member: model text above its own mark? → excluded):
  - `NO_TEXT_MARKER` — yes (marker-only)
  - `UNVERIFIABLE_TABLE_NATIVE` — yes (native floor / marker pattern)
  - `ROTATED_TEXT_SHREDDED`, `INVISIBLE_SCAN_UNREAD` — yes (stub markers)
  - `STRUCTURE_CLASS_FLOOR`, `STRUCTURE_CLASS_PAGE_JUDGE_TIMEOUT_FLOOR`, `STRUCTURE_CLASS_TEXT_TABLE_FLOOR` — yes (fail-closed marker bodies)
  - `UNVERIFIABLE_TABLE_SCANNED` — **no** (refused model prose or recovered native with inline marker; judge fields required)
  - `STRUCTURE_CLASS_GRID_FLAGGED`, credentialed timeout, `NATIVE_FALLBACK`, `BEST_*` — not in exclusion set; banner when refusal evidence present
- **P2 — finalize order.** `_apply_unaccepted_output_mark` computes `reason`, applies the provenance exclusion, then calls `_finalize_best_output_unverified` so `NATIVE_FALLBACK` lanes without refusal evidence keep `failure_mode=NONE`.
- **P2 — #713 second disclosure.** Credentialed judge-timeout winners gain the top “did not finish” unaccepted banner beside the in-body TIMED OUT note (owner wants top mark). Pinned with resume-stable test.
- **P2 tidy.** `_WholeDoc` dropped unused judge/error fields; `docs/OUTPUT.md` documents refusal-evidence rule and marker-first exclusion (no stale allowlist / `judge_reason` as evidence); `agentic` SOURCE_EVIDENCE refusal sets `judge_accepted=False`; dual-banner tests use production shape (`judge_accepted=True` on promoted winner).
- **Not in this PR:** an escalated winner the page judge never saw still ships WARNING without this banner (banner would be untrue). #1145 adds `page_judge=UNKNOWN` for that cluster; legacy `judge_accepted=None` replay unchanged (#1153).

**Tests:** `7486 passed` (`scratch/routing-baseline/exec/1129/pytest-r12.txt`).
**Ruff:** `uvx ruff@0.16.0 format --check .`
**Mutants:** `scratch/routing-baseline/exec/1129/mutants-r12.txt` (`run-mutants-r12.py`, conftest canary).

## Round 13 (2026-10-14)

Reviews of 79106d50 (Argon, Fable): do not merge.

- **P0 — revert scanned-floor banner (round 12 regression).** `UNVERIFIABLE_TABLE_SCANNED` is back in `_UNACCEPTED_OUTPUT_MARK_EXCLUDED_PROVENANCES`. Since #652 r10 that ending ships native recovery or a marker, never model prose; the round-12 banner broke `_is_restored_prose_recovery` and lost #649 prose on resume. Judge fields on the synthetic `PageOutput` stay for sidecar/metadata only. Deleted `test_scanned_table_floor_refusal_gets_unaccepted_banner`; added `test_restored_page_keeps_recovered_prose_with_judge_refusal_metadata` in gh649.
- **P1 — split status vs failure_mode.** `_warn_audit_failed_success_status` demotes every `SUCCESS` + `audit_passed=False` winner to `WARNING` before the refusal gate; `_stamp_kept_judge_rejection_failure_mode` runs only for `BEST_OUTPUT_UNVERIFIED` / `BEST_ATTEMPT_FLAGGED` when the banner path fires. Native / whole-doc lanes keep lane `failure_mode`.
- **P0 — `STRUCTURE_CLASS_GRID_CORROBORATED`.** Argon’s review quoted an old exclusion allowlist; on this branch corroborated winners are **not** excluded. Shipped bytes are model grid text — `test_corroborated_grid_refusal_gets_unaccepted_mark` pins the top banner when refusal evidence is present while `HEADER_BINDING_UNVERIFIED` is unchanged.
- **Tidy.** Removed dead `judge_refusal_evidence_from_winning`; P6 stage-C oracle mutant (`r13_p6_unenumerated_unaccepted`) confirms unenumerated #1129 drift fails `test_complete_exact_delta_matches_shared_oracle`.
- **Not in this PR:** GH-96 escalated winners ship `SUCCESS` with default `audit_passed=True` on the fresh `PageOutput` until selection demotes — pre-existing; #1145’s `page_judge=UNKNOWN` cluster handles the unjudged case. Legacy `judge_accepted=None` replay unchanged (#1153).

**Tests:** `7487 passed` (`scratch/routing-baseline/exec/1129/pytest-r13.txt`).
**Ruff:** `uvx ruff@0.16.0 format --check .`
**Mutants:** `scratch/routing-baseline/exec/1129/mutants-r13.txt` (`run-mutants-r13.py`, conftest canary).
