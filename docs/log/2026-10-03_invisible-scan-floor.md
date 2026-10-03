# 2026-10-03 - invisible-scan floor (#1027)

## Problem

A scan whose native text is an invisible OCR layer over a raster (`_invisible_text_suspect`, #961) is
routed to the model ladder. When no reading was accepted, the selector shipped the layer as
`NATIVE_FALLBACK`: WARNING, failure mode `native_minus_as_digit` (wrong cause), disposition
`demoted_native`. The layer is known garbage (one letter per line, stray axis ticks). Real case:
archive re-OCR, 1990 ECMA doc, pages 15 and 25 (page 15: a scanned chart whose gemini reading was
judge-rejected for shifted rows). Pages and counts only; the PDF is copyrighted.

## Decision (owner: a wrong number is worse than a missing one)

Ship neither the layer nor the rejected reading. Ship the existing fail-closed floor (marker + page
image).

## Changes

- `core/result.py`: `FailureMode.INVISIBLE_SCAN_UNREAD` (`invisible_scan_unread`).
- `core/manifest.py`: new branch in `_select_page_output_tagged`, after the rotated-shred floor and
  before the flagged-model branch (so a rejected reading cannot win). New
  `SelectionProvenance.INVISIBLE_SCAN_UNREAD`, `PagePrimaryReason.INVISIBLE_SCAN_UNREAD`, disposition
  `(FAIL_CLOSED_MARKER, INVISIBLE_SCAN_UNREAD)`, marker family `invisible OCR layer unread` (reason is
  recoverable from the shipped bytes). WARNING, `audit_passed=False` (resume re-OCRs).
- Failure-mode ordering: `native_invisible_text_scan` now outranks `native_minus_as_digit` in the
  native-fallback chain and in the chart lane (`orchestrator.py`).
- `pipeline/orchestrator.py`: page image rendered when the ladder accepted nothing
  (`invisible_scan_page_p<N>.png`, needs `--save-figures`, else the marker ships alone); own audit
  event `invisible_scan_unread` and own CLI line, replacing the generic `page_failed` /
  "no usable output" line for these pages. The page still counts in `failed_pages`, so the document
  is AUDIT_FAILED and `final_result.error` names it. Metadata carries the failure mode and
  disposition per page, as for every floor.
- Sidecar transparency: `attempts_summary` (engine, accepted, judge_outcome, rejection_reason
  truncated to `ATTEMPT_SUMMARY_REASON_MAX_CHARS` = 200). A skipped (resumed) page keeps the original
  record (`PageState.attempts_summary_restored`). The final `.md` is unchanged for every page not
  affected; only the sidecar gained a key.
- `core/audit_log.py`: rank for the new event. `docs/OUTPUT.md`: failure mode + event documented.

## Scope choices (deliberate; flag if wrong)

- Floor fires on `invisible_text_over_raster` only. A FAILED scan means "unknown", not "known
  garbage", so it keeps the native fallback (now with the corrected mode).
- Floor needs a non-native attempt to have run. With no provider or under `--native-only` the layer
  is the only text; #961 documents that as retained WARNING and its tests pin it. A pure provider
  outage on an invisible-layer scan therefore still ships the layer (WARNING).
- Structure-class pages are left to S1 (`_reaches_structure_class_branch`).
- Not added to `tables_trust` distrust kinds: that set is table content, this is page text.

## Tests

`tests/test_gh1027_invisible_scan_floor.py` (9): same page accepted vs rejected through `process()`,
same page with/without the invisible flag (non-invisible `needs_ocr_enhancement` unchanged), scan-failed
keeps native, no-model-attempt keeps #961 retention, ordering, summary truncation. Hermetic
(`_available_engines_for_agentic` patched, `_resolve_judge_model` -> "").

Frozen pins updated for the new member / sidecar key: `test_r7_winner_kind_tags` (22->23, returns
18->19, groups), `test_p6_disposition_contract` (23), `test_p6_disposition_persistence` (key set),
`p6_stage_c_oracle.VOLATILE_KEYS` (+`attempts_summary`), `test_gh987_judge_breaker_e2e._shape`
(the circuit-open reason text differs by design; engine/accepted/judge_outcome stay pinned).

Mutants (external copy of `src` + `tests`; `socr.__file__` canary = `test_loaded_source_is_this_checkout`
passing inside the copy; each anchor count 1 before editing):
1. branch disabled (`and _model_attempt_ran(p)` -> `and False`): 3 tests fail.
2. ordering reverted (`... and not minus_as_digit_suspect(p)`): ordering test fails.
3. sidecar summary dropped (`"attempts_summary": []`): e2e test fails.

Full suite and `uvx ruff@0.16.0 format --check .` clean before commit.
