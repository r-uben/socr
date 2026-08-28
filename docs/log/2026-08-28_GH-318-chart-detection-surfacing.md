# GH-318 — a swallowed chart-detection failure shipped a clean SUCCESS

**Date:** 2026-08-28
**Branch:** `fix/318-chart-detection-surfacing`
**Closes:** #318 · **Follows:** #297 (closed #181) · **Same class as:** #252, #211

## The defect, measured

`_phase_agentic`'s chart-eligibility `except` (added by #297) logs a WARNING, appends
`AuditEvent(kind="chart_asset_detection_failed")`, and `continue`s to the non-chart
route. #297 deliberately deferred demotion — correct for the recursion ceiling it was
fixing. The surfacing hole stayed open.

Reproduced by running the same pipeline twice, changing only whether the detector raises:

```
detector FAILED  -> page=success audit_passed=True doc=success
detector CLEAN   -> page=success audit_passed=True doc=success
```

Identical at every surface a consumer can see. The skip was invisible unless someone
opened `audit_log.json`.

After the fix:

```
detector FAILED  -> page=warning audit_passed=True doc=audit_failed
detector CLEAN   -> page=success audit_passed=True doc=success
```

## The constraint that shaped the fix

`audit_passed` is the winner-**selection** flag, not a page-quality flag.
`core/manifest.py` states it outright: flipping it "discards the page's content, the
#252 round-1 defect". Since #318 requires the content be **kept**, the fix demotes
`PageStatus` and the document bucket, and never touches `audit_passed`.

## What changed

| Site | Change |
|---|---|
| `core/state.py` | new `PageState.chart_asset_detection_failed` field, documented as distinct from `chart_asset_render_failed` (never-decided vs decided-then-failed). Deliberately **out** of `needs_repair` — a detector crash must not force the chart lane or a repair pass. |
| `orchestrator.py` (detection site) | sets the flag alongside the existing audit event |
| `orchestrator.py` (page finalize) | demotes `bo.status` SUCCESS → WARNING; `audit_passed` untouched |
| `orchestrator.py` (`_flush_page_sidecar`) | writes the flag, next to its render-failed sibling |
| `orchestrator.py` (`_restore_terminal_page_state`) | restores it with **OR, never assignment** |
| `orchestrator.py` (assemble buckets) | `chart_detection_failed_pages` → `pages_ok = pages_ok and not …` → `AUDIT_FAILED` |
| `orchestrator.py` (`_chart_detection_failed_note`) | document-level note into `final_result.error`, so `metadata.json` carries it |

**No CLI change.** The existing `DocumentStatus.AUDIT_FAILED` branch in `cli.py` already
prints "Completed with warnings … output written", which is exactly the required
surface. Reuse over new code.

## Two decisions worth recording

**1. The resume OR (raised by the Codex consult, not by me).** Restoring the flag with a
plain assignment is wrong: OCR pages can resume *before* chart eligibility runs, so a
stale clean sidecar could erase a failure this run had already recorded. The restore ORs.
This is mutation-tested — `test_gh318_detection_flag_survives_resume_restore` fails when
the OR is weakened back to assignment, and was confirmed to do so.

**2. WARNING re-processes the page on every resume — accepted deliberately.** The resume
ledger admits terminal pages only at `SUCCESS` (`_load_terminal_page`), so a WARNING page
is never skipped. That is the correct trade: the page's chart-vs-table routing was never
decided, so re-deciding it next run is the point. The cost is real, though — a
deterministically-failing detector re-runs that page every resume. Recorded here so the
behaviour is a choice on the record rather than a surprise later.

## Tests

Both pin a **difference**, never a measured absolute tuple (CLAUDE.md: provider-dependent
machinery makes absolute pins pass locally and fail in CI).

- `test_gh318_chart_detection_failure_is_visible_at_page_and_document_status` — two runs,
  one variable; asserts content identical and `audit_passed` True in both, while page
  status, document status and `metadata.json` all differ.
- `test_gh318_detection_flag_survives_resume_restore` — exercises the real
  `_restore_terminal_page_state` against a stale clean sidecar.

Hermetic: `_available_engines_for_agentic` is patched to `[PROFILE_QWEN_LOCAL]`, so the
agentic loop routes with no provider present (the CI trap from CLAUDE.md).
