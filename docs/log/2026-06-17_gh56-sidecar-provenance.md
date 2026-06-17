# GH-56 fix log — per-page sidecar records the shipped winner, not the rejected attempt

Date: 2026-06-17
Issue: GH-56 (CE OCR is not solved: prioritize reliable tables and figures)
Branch: fix/56-table-fallback-provenance
Found via: cell-by-cell validation on the real Consensus Economics forecaster grid
(`/Volumes/Main/.../consensus_economics/pdf/202401.pdf` page 4)

---

## What surfaced

Ran `socr --agentic --strict-local` on the dense CE forecaster grid (28 forecasters ×
10 indicators × 2 years). The numeric **values** are captured faithfully (spot-checked
~7 cells: correct, column-ordered, `na` preserved), but the page ships the **worse of
two flawed tables**:

- qwen produces a *row-major* table (readable) but **ragged** — the Goldman row has 19
  data cells where there should be 20. The native-table verifier correctly hard-fails it
  (`geometry_impossible_collapse`, `native_lane_count=27 / output_col_count=11`).
- The fallback then ships the born-digital native `find_tables()` output, which is fully
  **collapsed** (28 names in one cell, numbers newline-stacked per column).

While validating, the sidecar exposed a latent provenance/resume bug (the subject of this
fix). The table-quality problem itself (which artifact to ship, how to repair) is the
separate design fork tracked on GH-56 — NOT fixed here.

## The bug fixed here

`UnifiedPipeline._flush_page_sidecar` set the sidecar's `winning_output` from
`ps.best_output` (the raw per-page OCR attempt) instead of the selection that actually
ships — `_winning_page_output(state, page_num, whole_doc)`, the same one
`build_manifest` and `canonical_page_texts` (→ the `pages/NNN.md` fragment) use. They
diverge precisely in the fallback cases:

- a rejected OCR attempt (`status=SUCCESS` yet `audit_passed=False`) overridden by a
  flagged born-digital native-text fallback (`engine=native`, WARNING); or
- a CLI whole-doc attempt recovered into per-page text (non-agentic path).

Two consequences:

1. **Provenance lie:** the `pages/NNN.json` sidecar disagreed with its paired
   `pages/NNN.md` fragment (sidecar said `engine=qwen, status=success`; the page shipped
   `engine=native, WARNING`).
2. **Resume silent-skip (serious):** the resume gate `_load_terminal_page` skips a page
   when `winning_output.status == success`. Recording qwen's success-status attempt made
   the flagged broken-table page look clean, so a re-run would SKIP it and erase its
   `audit_failed` signal — the exact silent loss the gate's own contract (the
   non-SUCCESS-status reprocess rule) forbids.

## The change

`src/socr/pipeline/orchestrator.py` — `_flush_page_sidecar`:

- `winning_dict` now comes from `_winning_page_output(state, page_num, whole_doc)` with
  `whole_doc = _whole_doc_page_texts(state)`, computed exactly as `build_manifest`
  (manifest.py:421) and `canonical_page_texts` (manifest.py:349) do. For a passing page
  (`best_output.audit_passed=True`) this returns `best_output` unchanged — no regression,
  no churn to existing sidecars/golden tests. Only flagged-fallback / whole-doc-recovery
  pages change.
- `page_fingerprint` stays `""` for a **failure-marker** winner (via
  `is_page_failed_marker`), not just for an empty dict — a failure marker has no cached
  BlobStore entry to cross-reference, so fingerprinting its synthesized text would point
  at a blob that was never stored. A real or native-fallback winner carries genuine text
  worth fingerprinting.

## Tests (`tests/test_pp5_resume_ledger.py`)

- `test_flagged_native_fallback_sidecar_records_winner_not_attempt` — the agentic
  native-fallback case: sidecar must record `engine=native`/WARNING (not the qwen
  attempt) and `_load_terminal_page` must NOT skip the flagged page. Verified to FAIL on
  the pre-fix code (`assert 'qwen' == 'native'`).
- `test_whole_doc_recovery_sidecar_matches_manifest` — the non-agentic whole-doc case
  (caught in review): sidecar must record the recovered CLI output, not a failure marker.
  Verified to FAIL with `whole_doc=None`.

Full suite green (1075 passed), ruff clean. Verified end-to-end on the real CE page:
sidecar now reads `status=warning, engine=native, audit_passed=False`.

## Still open on GH-56 (design fork — NOT this fix)

- Verifier-guided repair re-ask (re-prompt qwen with native lane geometry as a hard
  constraint: N rows × 20 columns, `na` for blanks; re-verify). Cloud escalation in
  non-strict mode.
- Fix the native fallback itself: `find_tables()` returning lane-stacked cells should
  trigger `reconstruct_table_regions()` / a word-geometry rowizer instead of only running
  when `find_tables()` returns nothing.
- Do NOT build a generic "explode collapsed grid" post-processor — silent-row-drift trap.
