# 2026-09-07 — TICKET-E2: `table_not_scorable` scoped to detected tables (GH-655)

## Problem

`table_not_scorable` was firing on every prose page, not just pages with a
real table. Measured on the census: 3/3 pages of the ECB meeting-transcript
excerpt (`ecb-meetings-2020-transcript-p12-14`) flagged, and 400 events
across 68 Fed documents. The emitter keyed off `PageState.has_tables`, a
lane-cooccupancy heuristic (`BornDigitalDetector._detect_tables`) that
false-positives on ordinary numeric prose — two lines of `label` / `value`
pairs are enough to look like a two-column table.

## Fix

Both `table_not_scorable` emission sites live inside
`_table_page_needs_escalation` in `src/socr/pipeline/orchestrator.py` (this
is the sole emitter — grepped for the string across the whole repo, no
other site exists). Each is now gated behind a new helper,
`_page_detected_table_count`, which reads the independent, structural
table-region detector count (`PageState.detected_table_count` /
`DocumentAssessment.pages[i].detected_table_count`, GH-520) instead of the
heuristic `has_tables` flag:

```python
if self._page_detected_table_count(page_num, ps) > 0:
    # existing table_not_scorable emission (both branches)
```

The function's return value — whether the page *needs escalation* — is
unchanged in every branch; only whether the audit event is emitted for a
page with zero detected tables changed. The `elif report.unexplained_lanes`
branch, which emits a different event kind (`table_unexplained_lanes`), was
left untouched — out of scope for this ticket.

`_page_detected_table_count` mirrors the existing `_page_has_tables`
fallback pattern: prefer the in-flight `PageState`, fall back to the
document-level `_assessment_for_page` lookup. `_assessment_for_page` was
made defensive (`getattr(self, "_last_assessment", None)` instead of a
direct attribute read) because `_page_detected_table_count` unconditionally
falls through to it when the `PageState` count is zero — unlike
`_page_has_tables`, which most tests never drove into that branch. Without
this, pipeline test doubles built via `object.__new__(UnifiedPipeline)`
(skipping `__init__`) raised `AttributeError`.

`src/socr/core/tables_trust.py` needed no logic change — it is a pure
reader of whatever `AuditEvent` kinds reach it — only a comment on the
`table_not_scorable` entry in `TABLE_DISTRUST_KINDS` documenting the new
scoping.

## Downstream consumers

Grepped every reference to `table_not_scorable` in the repo. The only
consumer is `tables_trust.py`'s generic kind-based aggregation; nothing
keys off this event existing specifically for prose pages. No
architectural fork, no CONSILIUM-GATE needed.

## Why the two existing test files needed edits

`tests/test_gh95_tables_trust.py::test_a_not_scorable_page_surfaces_with_no_escalation_provider`
and three tests in `tests/test_gh96_escalation_lane.py`
(`test_prose_fragments_are_not_a_grid`, `test_a_single_wide_row_is_not_a_grid`,
`test_a_non_grid_page_costs_nothing`) built page fixtures that are the exact
shape of the census false positive — numeric prose lines with `has_tables`
true and no real table — and asserted the **old, buggy** behavior: that
`table_not_scorable` fires. After the fix those assertions are wrong by
design, since the fixtures never set `detected_table_count`, so it defaults
to 0. Fixed:

- `test_a_not_scorable_page_surfaces_with_no_escalation_provider`: fixture now
  sets `detected_table_count=0` explicitly; asserts `state.events == []`,
  `trust.untrusted_pages == []`, `payload["pages"] == {}`.
- `test_prose_fragments_are_not_a_grid`: asserts `state.events == []`.
- `test_a_non_grid_page_costs_nothing`: asserts `state.events == []`
  (unchanged: zero provider cost, `calls == []`, `state.engine_runs == ()`).
- `test_a_single_wide_row_is_not_a_grid`: needed no assertion change — it only
  checks the escalation-need boolean, not events. It started passing once the
  `_assessment_for_page` `AttributeError` fix landed.

Per the ticket's "pin the difference" requirement, each rewritten test now has
a companion asserting the corrected, still-fires case:

- `test_gh95_tables_trust.py::test_the_same_page_surfaces_once_a_table_is_actually_detected`
  — identical fixture, `detected_table_count=1`, asserts the event fires.
- `test_gh96_escalation_lane.py::test_prose_fragments_still_surface_when_a_table_is_detected`
  — same page shape, `detected_table_count=1`, asserts the event fires.

## New test file

`tests/pipeline/test_table_not_scorable_scope.py` — two synthetic tests
(prose page → 0 untrusted pages; page with `detected_table_count=1` → 1) plus
two tests against the real named census fixtures, run through
`BornDigitalDetector` directly against the source PDFs (no live model):

- `ecb-meetings-2020-transcript-p12-14.pdf` p1: `detected_table_count == 0`,
  no events, `untrusted_page_count == 0` (was 3 before the fix, confirmed via
  `~/Data/socr/census-ecb-2026-09-06/out/.../tables_trust.json`).
- `ecb-surveys-2013-ecb.blssurvey2013q1.en-p29-31.pdf` p2:
  `detected_table_count == 3` (a real table page); if it flags, still flags
  correctly, `untrusted_page_count == 1`.

Both real-fixture tests are `skipif(not path.exists())`-guarded but ran (not
skipped) on this machine, since the fixtures exist.

## Test results

- Targeted table-related test files (`test_gh95_tables_trust.py`,
  `test_gh96_escalation_lane.py`, `test_table_not_scorable_scope.py`, and
  related tables_trust suite): 86 passed.
- New file alone: 4 passed (0 skipped — real fixtures present).
- Golden / byte-identity assembly tests: 35 passed.
- Full suite: 4327 passed, 4 xfailed, 0 failed, 586.71s.

## Ruff

`uvx ruff@0.16.0 format --check .` initially flagged
`tests/pipeline/test_table_not_scorable_scope.py` (one long line). Reformatted
with `uvx ruff@0.16.0 format tests/pipeline/test_table_not_scorable_scope.py`;
re-run confirms 600/600 files clean.

## Before / after

| Fixture | detected_table_count | Before | After |
| --- | --- | --- | --- |
| ECB transcript p1–3 | 0 | `untrusted_page_count: 3`, all `table_not_scorable` | `untrusted_page_count: 0`, no events |
| ECB survey 2013 p2 | 3 | flags (real table) | unchanged — still flags |
