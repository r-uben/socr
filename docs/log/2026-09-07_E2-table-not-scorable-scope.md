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

## Round 2 — reviewer blocker: `detected_table_count == 0` is not the same as "no table"

Review rejected round 1 with a confirmed counterexample: `fed-meetings-2010-11-03-minutes`
page 11 ("Table 1. Economic projections", real numbers) has
`native_table_region_count: 2` in its persisted sidecar (`pages/00011.json`) but never
serializes a `detected_table_count` key at all — this Fed corpus run predates that field
reaching every page's `PageState` copy, so `getattr(ps, "detected_table_count", 0)` reads
0. Under the round-1 gate (`detected_table_count > 0`), this page — a real, borderless
table only the native reconstruction pass sees — loses its only distrust signal. GH-520's
own docstring documents exactly this asymmetry and treats a bare `detected_table_count ==
0` as insufficient evidence, failing closed rather than reading it as "no table" (the
structure-class floor scoping). E2's gate had reintroduced the same mistake GH-520 exists
to prevent.

**Fix:** both emission sites now call a new `_page_has_scorable_table_evidence` helper
instead of `_page_detected_table_count(...) > 0` directly:

```python
def _page_has_scorable_table_evidence(self, page_num, ps=None) -> bool:
    if self._page_detected_table_count(page_num, ps) > 0:
        return True
    # falls back through ps.native_table_region_count, then the assessment's
    return ...
```

i.e. `detected_table_count > 0 OR native_table_region_count > 0`, checking `PageState`
first and the document-level assessment second for each, mirroring the existing
`_page_detected_table_count` fallback shape. `_page_detected_table_count` itself is
unchanged and still used internally.

### New tests

- `tests/pipeline/test_table_not_scorable_scope.py::test_borderless_table_with_zero_detected_count_still_flags`
  — synthetic pin: `detected_table_count=0`, `native_table_region_count=2` — event fires.
  This is the direct "pin the difference" test for the OR-gate itself.
- `tests/pipeline/test_table_not_scorable_scope.py::test_fed_minutes_p11_borderless_table_survives_missing_detected_count`
  — real-fixture reproduction of the reviewer's exact case: reads the actual persisted
  `pages/00011.json` sidecar from
  `~/repos/research/central-bank-network/data/ocr-runs/fed-01/fed-meetings-2010-2010-11-2010-11-03-minutes/`,
  asserts the sidecar genuinely has no `detected_table_count` key and a positive
  `native_table_region_count`, drives `_table_page_needs_escalation` against the real
  source PDF page (`ocr-staging/fed-01/pdf/fed-meetings-2010-2010-11-2010-11-03-minutes.pdf`,
  page 11), and asserts `table_not_scorable` fires and `untrusted_page_count == 1`. Also
  updated `test_prose_page_yields_zero_untrusted_pages` /
  `test_detected_table_page_still_flags` to set `native_table_region_count=0` explicitly
  (previously implicit via `SimpleNamespace` attribute absence, which `getattr` already
  tolerated, but explicit is clearer given the new OR branch) and threaded
  `native_table_region_count` through `_score_one_page`'s `ps` construction for the
  existing ECB real-fixture tests.

### Reviewer's requested counts, reproduced

**ECB (12/13 suppressed correctly).** Re-scored every page of all 9 in-corpus ECB PDFs
(30 pages total) through the fixed code (`BornDigitalDetector` + `_surface_table_scoring`,
no model). Before the fix, the live census runs recorded 13 `table_not_scorable` events
across 5 of the 9 documents (`tables_trust.json` `counts_by_kind`). After the fix: 1 event
survives (`ecb-speeches-2025-speech-p21-23.pdf` page 3, `detected_table_count=2`, a real
table) — **12 of 13 suppressed**, the 1 remaining is a genuine table page, confirming the
gate does not overcorrect into losing the true positive.

**Fed (3 of 400 events rescued by the OR gate).** Scanned every `audit_log.json` under
`~/repos/research/central-bank-network/data/ocr-runs/fed-01` (767 documents) for
`table_not_scorable` events and cross-referenced each event's page against its persisted
`pages/NNNNN.json` sidecar. Total: **400 events across 68 documents** — matches the
census log's headline figure exactly. Of those 400: **0 had `detected_table_count > 0`**
recorded in the sidecar (this corpus run predates that field, which is the actual root
cause the census caught), **3 had `native_table_region_count > 0`** (rescued only by the
round-2 OR gate — the Fed p11 case above is one of the 3), and **397 had neither** (both
signals 0, correctly suppressed as false positives on numeric prose). This is the full
population, not a sample — a stronger form of the "3 of 12" figure quoted in review,
which was evidently a hand-checked subset of the same 400. Reproduction script inlined
below for the record:

```python
import json, glob, os

base = "."  # run from central-bank-network/data/ocr-runs/fed-01
total = rescued = already = neither = 0
for docdir in sorted(glob.glob(os.path.join(base, "*"))):
    audit = os.path.join(docdir, "audit_log.json")
    if not os.path.isdir(docdir) or not os.path.exists(audit):
        continue
    events = json.load(open(audit))
    events = events if isinstance(events, list) else events.get("events", [])
    for e in [e for e in events if e.get("kind") == "table_not_scorable"]:
        total += 1
        pn = e.get("page_num")
        pj = os.path.join(docdir, "pages", f"{pn:05d}.json") if pn else None
        detected = native = None
        if pj and os.path.exists(pj):
            pd = json.load(open(pj))
            detected = pd.get("detected_table_count")
            native = pd.get("native_table_region_count", 0) or 0
        if detected:
            already += 1
        elif native:
            rescued += 1
        else:
            neither += 1
print(total, already, rescued, neither)  # 400 0 3 397
```

### Updated test results (round 2)

- Targeted (`test_gh95_tables_trust.py` + `test_gh96_escalation_lane.py` +
  `test_gh96_table_exactness.py` + `test_gh96_escalation_decision.py` +
  `test_gh96_escalation_canary.py` + `test_table_not_scorable_scope.py`): 110 passed.
- New scope file alone: 6 passed (0 skipped — all real fixtures present, including the
  new Fed p11 fixture).
- Golden / byte-identity + agentic fuse (`test_pp2_agentic_fuse.py`,
  `test_p3_judged_bytes_ship.py`, `test_p6_stage_c_difference.py`,
  `test_p6_stage_ab_difference.py`): 67 passed.
- Full suite: 4329 passed, 4 xfailed, 0 failed, 607.91s (0:10:07). Foreground, waited on.

## Ruff (round 2)

`uvx ruff@0.16.0 format --check .` clean after round-2 edits (re-run before commit).

