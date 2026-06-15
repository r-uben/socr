# Structured Content Routing

Date: 2026-06-15
Branch: `feat/49-structured-content-routing`

## Repro

Original CE batch behavior routed every page through native PyMuPDF text:

```text
32/32 pages born-digital
32 trusted native text
All pages born-digital (no OCR needed)
```

That was wrong for born-digital table pages. The detector recorded table
metadata, but clean table pages did not set `needs_ocr_enhancement`, so both
agentic and legacy native-first routing treated PyMuPDF output as trusted prose.

## Decision

Use native text directly for born-digital prose, but route born-digital pages
with table-like structure through the OCR/VLM ladder unless `--native-only` is
set. Native text remains a fallback if all OCR attempts fail, but that fallback
is warning-status and audit-failed rather than a clean success.

External check: Antigravity/Gemini agreed the root cause and fix were sound and
flagged the provenance risk: table-page native fallback must not be masked as a
passing manifest entry.

## Implementation

- `PageState` now preserves `has_tables` from born-digital assessment.
- Native bypass policy is centralized in `UnifiedPipeline._is_trusted_native_without_ocr`.
- Agentic routing passes born-digital OCR pages with native text into the native
  fallback set, so table pages can fall back without being called trusted native.
- Page judges reject non-success outputs, preventing warning fallback text from
  satisfying the agentic ladder.
- Manifest and assembly fallback checks include structured table fallback via
  `native_table_structure_failed`.

GH-49A remains open: deterministic native table verification is still a separate
future verifier, not this routing fix.

## Verification

```text
uv run pytest -q
843 passed, 5 warnings

uv run ruff check src/socr/core/state.py src/socr/core/manifest.py \
  src/socr/pipeline/agentic.py src/socr/pipeline/orchestrator.py
All checks passed
```

Real CE smoke on `/tmp/socr-ce-202606-p1-2.pdf`:

```text
Auto-selected engine: qwen
Agentic routing
  ladder: qwen($0) -> nougat($0) -> marker($0)
  p1: qwen
  p2: qwen
```

Manifest page entries recorded `fingerprint.engine = "qwen"` for both pages.

## Provenance-masking fix (post-review)

An adversarial review identified a gap: in the agentic path, when a provider IS
available, the engine returns non-empty content with `PageStatus.SUCCESS`, but the
judge rejects ALL ladder rungs, `native_table_structure_failed` was never set. As a
result `_assemble_result` saw no fallback flag, treated the native-text fallback as
a clean pass, and the document status became `DocumentStatus.SUCCESS` — masking a
table page whose OCR was entirely rejected.

### Root cause

The no-provider branch (empty ladder) at ~line 1133 already set the flag. The
provider-present rejection path at ~line 1226 did not.

### Fix

After `ps.best_output = decision.final_output` (~line 1226 in
`src/socr/pipeline/orchestrator.py`), added:

```python
if not decision.accepted and self._page_has_tables(page_num, ps):
    ps.native_table_structure_failed = True
```

Table predicate used: `self._page_has_tables(page_num, ps)` — the existing helper
that checks `ps.has_tables` (set from `DocumentAssessment` via `apply_born_digital`).
Consistent with how the flag is set in the no-provider and legacy paths.

### Regression test

`TestNativeOnlyRouting::test_agentic_table_judge_reject_all_rungs_is_audit_failed`
in `tests/test_orchestrator.py`.

- Pre-fix: FAILED (`DocumentStatus.SUCCESS` returned instead of `AUDIT_FAILED`)
- Post-fix: PASSED

**Note on sparse-OK interaction:** `_make_bd_assessment` leaves `word_count=0` on
all `PageAssessment` objects, which makes `_sparse_page_ok` return `True` for every
page. When `sparse_ok=True`, the heuristic checker downgrades the word-count failure
from an error to a warning, so even `_bad_text()` ("short", 1 word) passes the
judge. The test therefore creates a `PageAssessment` directly with `word_count=100`
(above `audit_min_words=50`) to ensure the full heuristic gate applies.

### Verification

```text
~/venvs/socr/bin/pytest -q
844 passed, 5 warnings

~/venvs/socr/bin/ruff check src/socr/core/state.py src/socr/core/manifest.py \
  src/socr/pipeline/agentic.py src/socr/pipeline/orchestrator.py
All checks passed

~/venvs/socr/bin/ruff format --check <touched files>
5 files already formatted
```
