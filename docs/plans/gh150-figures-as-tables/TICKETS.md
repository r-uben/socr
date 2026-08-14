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
**Fixture constraint (from the A1 ruling):** fixture (a), the thin-stroke spike
plot, MUST draw an enclosing axes rectangle (`page.draw_rect`) plus
`>= MIN_DRAWINGS_FOR_VECTOR` interior spike marks — not two bare axis lines —
or the framed-cluster gate correctly returns False and the test misreads an
A2 fixture bug as an A1 failure. The existing `tests/test_chart_lane.py::_make_monochrome_lineplot_pdf`
(bare `draw_line` axes) must stay a documented False under
`test_monochrome_lineplot_is_false_negative` and is out of A2's rewrite scope.

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

### TICKET-B2 — pin the residual first-hit-wins gate (fixtures + xfail) · TODO · depends-on: A1, B1 · wave 3b
**Problem:** The fix must be demonstrated on the pages that motivated it, not only on
synthetic fixtures.

⚠️ **RETARGETED 2026-08-13 by a wave-3 ruling.** The original `Done when` — "neither page's
`native_text` contains `| --- |` and both contain an image reference" — **cannot be satisfied
within this ticket's write set.** The wave-3 design pass copied both PDFs to `/tmp`, hashed
them, and measured through the installed package:

| Paper | bytes | sha256 (prefix) |
|---|---|---|
| Heston (p10) | 449,243 | `2ff4173d787d476c…` |
| Drechsler (p55) | 711,917 | `d9eba00cd208b9fe…` |

2 PDFs, 115 pages. **The defect still reproduces**: both target pages still carry a markdown
pipe separator and no image reference. But the production fix is a **merge** — chart
placeholders must be merged with non-empty `find_tables()` regions — and that lives in
`src/socr/core/born_digital.py`, which this ticket does not own. Emptying `find_tables()` is a
*labelled simulation* of reaching the existing fallback, not the after-state and not the patch
to land.

**Do:** Fixtures, tests and logs only.
1. Record the **before** measurement (installed package, sha256-keyed, on the `/tmp` copies).
2. Append a generated **mixed table-plus-chart** fixture and a **table-only negative control**.
   The generator must assert on reopen: mixed page `find_tables() >= 1` and
   `chart_region_bboxes() >= 1`; control `find_tables() >= 1` and `chart_region_bboxes() == 0`.
   State in the log that the fixture is defined *from those predicates* — it proves the code
   path, not corpus shape. **No corpus PDF or page extract in git** (public repo, copyrighted
   journal articles).
3. **Append** — do not rewrite — deterministic `BornDigitalDetector` tests. Provider-free:
   fixture self-checks; a labelled rung check that `rowize_from_words_chart_aware` already
   emits the placeholder when reached; green `detect_page` / `extract_structured` guards that a
   genuine table and its prose survive and that a table-only page gains no image reference.
4. One `xfail(strict=True)`: `extract_structured` on the mixed fixture emits both the table and
   `![chart region`. Its docstring must say it is a pin until the `born_digital.py` merge lands,
   and that the original B2 `Done when` is still open. Delete the marker in a follow-up **in
   this same file** — do not leave a permanent xfail.

**Binding constraints:** do NOT edit `born_digital.py`, `figures/extractor.py`, or
`orchestrator.py`. Log the **before** (real end-to-end) and, separately, the stubbed-`find_tables`
simulation — leave the *after* column empty until the source ticket lands. Do **not** present
245/245 or 192/192 raw-word multiset retention as the issue's 17%/41% figures. Green mixed-page
tests may only assert what is true today.

**Files:** `tests/test_chart_detection_gh150.py`, `tests/fixtures/`, `logs/`
**Done when:** the fixtures and tests above exist and pass, the strict xfail is present and
documented as a pin, and the before-measurement is recorded with sizes and checksums.
**B2 must NOT be marked done on the corpus pages** until TICKET-C1 lands and the xfail marker
is deleted after a real installed-package *after* measurement.

### TICKET-C1 — merge chart placeholders with non-empty table regions · TODO · depends-on: B2 · wave 4 · NEEDS OWNERSHIP GRANT
**Problem:** This is the actual production fix the original B2 promised. On a mixed page, the
first hit wins: a non-empty `find_tables()` result suppresses the chart placeholder entirely,
so a page with both a table and a chart ships the table and silently drops the chart.

**Do:** Merge chart region placeholders with non-empty `find_tables()` regions rather than
letting either suppress the other. The fix is **merge**, not "skip `find_tables()` when a chart
exists".

**Files:** `src/socr/core/born_digital.py`
⚠️ **BLOCKED ON OWNERSHIP.** `born_digital.py` is claimed by GH-151 B1, in flight as PR #200.
The coordinator must re-cut ownership explicitly before dispatch — this ticket cannot start
until #200 merges.
**Done when:** neither Heston p10 nor Drechsler p55 `native_text` contains a markdown table
separator where a chart belongs, both contain an image reference, measured through the
installed package with sizes and checksums recorded; and B2's strict xfail is deleted in the
same wave.

