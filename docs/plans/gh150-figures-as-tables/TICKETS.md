# TICKETS — GH-150 figures extracted as tables

Status keys: `TODO` · `WIP` · `DONE` · `BLOCKED`. `depends-on` gates dispatch.
Same wave ⇒ disjoint files. Each ticket = one implementer agent + one reviewer pass.

Context: the two lowest-recall pages in the 22,979-page corpus are figures the
table lane claimed. `has_chart_marks` returned False on a page with 932 vector
drawing operations, and even when it fires, `_is_chart_asset_page` excludes any
page where `_page_has_tables` is true — so tables win unconditionally.

## Stream A — detection recall

### TICKET-A1 — accept thin-stroke vector plots as chart marks · TODO · depends-on: none · wave 1
**Problem:** `_has_vector_data_marks` requires coloured fills or thick strokes, so a
thin-line spike plot (Heston p10, 932 drawing ops) is not seen as a chart.
**Do:** Extend the positive-signal test so a large spatially-coherent cluster of thin
strokes also qualifies. Derive the qualifying condition from the cluster (count and
bbox area relative to the page), not a tuned stroke-width constant.
**Files:** `src/socr/figures/extractor.py`
**Done when:** `~/venvs/socr/bin/python -c "import fitz; from socr.figures.extractor import has_chart_marks; d=fitz.open('<Heston>.pdf'); print(has_chart_marks(d[9]))"` prints `True`, and the same call on a prose page of the same document prints `False`.

### TICKET-A2 — pin chart detection on the two regression pages · TODO · depends-on: A1 · wave 2
**Problem:** A1's change must not be reverted by a later tuning pass, and must not
start claiming prose pages.
**Do:** Add a hermetic test building (a) a thin-stroke spike plot, (b) a filled bar
chart, (c) a dense prose page, (d) a real table; assert chart detection fires on
(a) and (b) only.
**Files:** `tests/test_chart_detection_gh150.py`
**Done when:** `~/venvs/socr/bin/pytest tests/test_chart_detection_gh150.py -q` exits 0 with ≥4 tests, and the same file fails when A1 is reverted.

## Stream B — precedence

### TICKET-B1 — arbitrate chart-vs-table instead of table-always-wins · TODO · depends-on: none · wave 1
**Problem:** `_is_chart_asset_page` excludes every page where `_page_has_tables` is
true, so a figure whose axis labels parse as rows can never reach the chart lane.
**Do:** Replace the unconditional exclusion with an arbitration: when both signals
fire, prefer the chart lane. Record the decision as an `AuditEvent` naming which
lane won and why. Rationale to encode in the comment: an image reference loses
nothing recoverable, a pipe grid of axis labels loses everything.
**Files:** `src/socr/pipeline/orchestrator.py`
**Done when:** an `AuditEvent` of kind `chart_table_arbitration` appears in `audit_log.json` for a page where both fire, and `~/venvs/socr/bin/pytest tests/ -q` still passes.

### TICKET-B2 — end-to-end: the two worst corpus pages · TODO · depends-on: A1, B1 · wave 3
**Problem:** The fix must be demonstrated on the pages that motivated it, not only
on synthetic fixtures.
**Do:** Measure word recall and emitted output for Heston p10 and Drechsler p55
before/after; assert each now emits an image reference rather than a pipe grid.
**Files:** `tests/test_chart_detection_gh150.py`
**Done when:** neither page's `native_text` contains a markdown table separator (`| --- |`), and both contain an image reference; recorded in `logs/`.
