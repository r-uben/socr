# TICKETS — GH-152 side-by-side tables merged

Status keys: `TODO` · `WIP` · `DONE` · `BLOCKED`. `depends-on` gates dispatch.

Context: `2025__haim` p31 prints TABLE A5 and TABLE A6 side by side. The rowizer
segments rows by y across the full page width, so the two tables are read as
single rows spanning both and neither survives. 54 tokens are also lost outright.
`extract_structured` already documents the x-band limitation for READING ORDER;
this is the same limitation damaging table reconstruction.

⚠️ `src/socr/tables/reconstruct.py` carries the GH-146 work, committed and open as
**PR #149**, and is then held by GH-144 A2 for the whole of wave 2. Do not dispatch A1
until both have landed.

## Stream A — column segmentation

### TICKET-A1 — segment a page into x-bands before rowizing · TODO · depends-on: none · wave 1
**Problem:** Row clustering spans the full page width, merging columns.
**Do:** Detect vertical gutters from the word x-distribution and split the page into
bands. Reuse the existing clip-then-rowize approach (`rowize_from_word_list` already
scopes to a bbox) — apply it per band. A single-column page must yield exactly one
band, so existing behaviour is unchanged.
**Files:** `src/socr/tables/reconstruct.py`
**Done when:** a synthetic two-column page yields 2 bands and a single-column page yields 1; `~/venvs/socr/bin/pytest tests/ -q` passes unchanged.

### TICKET-A2 — rowize each band independently, in reading order · TODO · depends-on: A1 · wave 2
**Problem:** Two tables need two grids, emitted left-to-right then top-to-bottom.
**Do:** Rowize each band separately and order the resulting regions by (band, y0).
**Files:** `src/socr/tables/reconstruct.py`
**Done when:** the p31 fixture emits two distinct markdown tables, and TABLE A5's correlation values do not appear interleaved with A6's means.

## Stream B — evidence

### TICKET-B1 — end-to-end on the motivating page · TODO · depends-on: A2 · wave 3
**Problem:** Must be demonstrated on the real page, not only a synthetic one.
**Do:** Measure p31 word recall before/after and assert the 54 previously-missing
tokens (`Measure`, `Correlation`, `Mean`, `SD`, `Guidance`, `Legislative`, `0.95`,
`0.78`, …) are present.
**Files:** `tests/test_side_by_side_tables_gh152.py`
**Done when:** p31 recall ≥ 95% and the named tokens are all present; recorded in `logs/`.
