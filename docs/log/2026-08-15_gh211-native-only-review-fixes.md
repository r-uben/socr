# GH-211 (`fix/211-native-only-table-status`): close the two review majors

Branch `fix/211-native-only-table-status` was REJECTED with 2 majors. This
session closes both without relitigating the settled `--native-only` = "off,
record and surface only" ticket scope.

## MAJOR 1 — silent content loss (`src/socr/core/manifest.py`)

**Finding as reported:** demoting the native `PageOutput` makes
`_winning_page_output` fall back to `ps.native_text`, discarding anything
appended to `PageOutput.text` after native capture (GH-36b's equation LaTeX
sidecar).

**What was actually true after reading the code:** the speculative fix point
cited (`manifest.py:281`, the `p.best_output.audit_passed` branch) is dead
for this bug — both call sites that build a demoted native `PageOutput`
(`orchestrator.py` prose-extraction loop and the agentic per-page loop) set
`audit_passed=False` **at construction time**, so `p.best_output.audit_passed`
is never `True` for a `native_table_unverifiable` page. The real bug is in
the generic native-fallback block a few lines down
(`if p.is_born_digital and p.native_text:` → `native_is_fallback` branch),
which unconditionally shipped `text=p.native_text` — discarding
`p.best_output.text`, which is the SAME object GH-36b's
`_attach_equation_latex_sidecars` mutates in place
(`po.text = po.text + "\n\n" + result.sidecar_block`, orchestrator.py:5550)
after the native `PageOutput` is created and attached to `ps.best_output`
(confirmed: `bo = ps.best_output` at orchestrator.py:2716 is the same object
`_attach_equation_latex_sidecars(state, [bo])` mutates at orchestrator.py:2804,
within the same per-page loop iteration).

**Fix:** in that fallback block, prefer `p.best_output.text` over
`p.native_text` whenever `p.best_output` exists, is non-empty, and its engine
starts with `"native"` — the repo-wide 1C invariant ("append, never replace")
means `best_output.text` is always native_text plus zero or more appended
sidecars, so this is a strict superset, never data loss in the other
direction. Falls back to `p.native_text` only when there is no live
`best_output` to read from.

## MAJOR 2 — the audit log lies (`src/socr/pipeline/orchestrator.py`)

Under `--native-only`, `_is_trusted_native_without_ocr` short-circuits the
OCR ladder for essentially all born-digital pages (the narrow rotated+table
exception still routes to OCR). So a page that lands in
`native_fallback_pages` purely because `native_table_unverifiable` is set
never had an OCR attempt — but the event emitted for every page in that list
says "OCR tried and never passed on a structured/enhancement page."

**Fix:** split out `native_only_distrust_pages` — pages that are
born-digital, `--native-only`, `native_table_unverifiable` (and NOT also
`native_table_structure_failed`, which would mean the D3 floor already
routed them elsewhere), with every recorded attempt's engine starting with
`"native"` (the `all(...)` guard excludes the rotated+table case, which
genuinely did route through OCR and keeps the "OCR tried and failed"
wording). These pages are excluded from `native_fallback_pages` and get:
- their own audit event kind, `native_only_table_distrusted`, with accurate
  wording ("OCR never attempted (ladder disabled)");
- their own CLI summary line (distinct from the "fell back to native text"
  line);
- they still flip `pages_ok` to `False` (document status still surfaces as
  `AUDIT_FAILED`/partial) — this ticket is "record and surface only," not
  "stop treating a distrusted table as a document-level problem."

Note: an earlier, already-accurate `table_structure_failed` audit event
(added at extraction time in `_phase_analyze`, `orchestrator.py:663-676`,
predating this session) already carried correct wording for these pages.
This fix does not touch that event; it only stops the *second*, misleading
`native_fallback` event from also being recorded for the same page.

## Files changed

- `src/socr/core/manifest.py` — `_winning_page_output`: prefer
  `best_output.text` over `native_text` in the native-fallback path.
- `src/socr/pipeline/orchestrator.py` — `_phase_assemble`: new
  `native_only_distrust_pages` bucket, excluded from `native_fallback_pages`,
  own audit event + CLI line, still counted in `pages_ok`.
- `tests/test_manifest_native_table_demotion_gh211.py` (new) — direct unit
  test against `_winning_page_output`: a demoted native table with an
  appended equation sidecar keeps the sidecar; without one, ships native text
  unchanged.
- `tests/test_native_only_table_status_gh211.py` — extended the existing
  full-pipeline test to assert no `native_fallback` event is recorded for a
  `--native-only` distrust page, and that `native_only_table_distrusted` is.

## Test result

`~/venvs/socr/bin/pytest tests/ -q` → **1617 passed, 1 xfailed** (pre-existing
xfail, unrelated). Targeted run
(`tests/test_manifest_agentic.py tests/test_manifest_native_table_demotion_gh211.py
tests/test_manifest_replay.py tests/test_orchestrator.py
tests/test_native_only_table_status_gh211.py`): 143 passed.

`uvx ruff@0.16.0 format --check .` → clean (294 files).

## Hermeticity

All new/modified tests patch `_available_engines_for_agentic` (existing
fixture in `test_native_only_table_status_gh211.py`) or never touch the
provider ladder at all (`test_manifest_native_table_demotion_gh211.py`
constructs `DocumentState` directly and calls `_winning_page_output`, no
`process()` / agentic mode involved). No ollama, no provider.

## Scope note

Not revisited: `--native-only` semantics (still off = record-and-surface
only, never reroute — settled). The rest of the accepted #211 behavior
(D3 floor exclusion, `table_structure_failed` in `TABLE_DISTRUST_KINDS`,
the extraction-time `table_structure_failed` event) is untouched.
