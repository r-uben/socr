# TICKETS — GH-144 rowizer destroys numeric values

Status keys: `TODO` · `WIP` · `DONE` · `BLOCKED`. `depends-on` gates dispatch.

Context: the rowizer places a column boundary INSIDE a number.
`3M Treasury yield 0.67` becomes `| 3M | Treasury | yield 0 | .67 |` — the value
`0.67` ceases to exist. Nakamura & Steinsson p42 loses 49 of its 152 values this
way. A minimal synthetic reproduction is pinned as a strict xfail in
`tests/test_region_overlap_gh145.py::test_no_table_value_is_lost`.

Hypothesis to test first: row labels use single internal spaces while columns are
separated by space RUNS, so a lane detector clustering on x finds a false gutter
inside the label — and once the boundary is misplaced, every value on the row shifts.

⚠️ `src/socr/tables/reconstruct.py` carries the GH-146 work, merged 2026-08-12 as
**PR #149**. A2 and A2b both build on it.

This plan also owns the **#146 residual** (TICKET-A2b). Issue #146 stayed open after PR #149
because that PR fixed only cause 1 (a data row promoted to header); cause 2 — the region
excluding the header band — is the same class of `reconstruct.py` boundary error as #144 and
must serialize on the same file, so it is tracked here rather than in its own folder.

## Stream A — diagnose, then fix

### TICKET-A1 — characterise the misplacement · TODO · depends-on: none · wave 1
**Problem:** The fix must target the actual cause, not the first plausible one.
**Do:** For the synthetic repro and for NS p42/p17/p43, record where each lane
boundary lands relative to word bboxes, and whether lost values cluster by column
(lane-assignment bug) or by row (row-segmentation bug). Write findings to `logs/`.
No production code changes.
Also measure, on the same pages, the **y-extent** of each emitted region against the
y-extent of the header band above it, and record why the header band falls outside — this
is the #146 residual that A2b fixes, and it is cheapest to measure in the same pass.
**Files:** `docs/plans/gh144-rowizer-destroys-values/logs/`
**Done when:** a dated log states which of the two hypotheses holds, with per-value evidence for at least 10 destroyed values, and separately names the code path that excludes the header band from the region.

### TICKET-A2 — never place a lane boundary inside a token · TODO · depends-on: A1 · wave 2
**Problem:** A column boundary that splits a word or number is always wrong,
whatever the x-clustering says.
**Do:** Constrain lane boundaries to fall in whitespace between word bboxes. A
candidate boundary intersecting a word's bbox must be moved to the nearest gap.
**Files:** `src/socr/tables/reconstruct.py`
**Done when:** `~/venvs/socr/bin/pytest tests/test_region_overlap_gh145.py::test_no_table_value_is_lost -q` passes (the strict xfail flips), and the full suite still passes.

### TICKET-A2b — include the header band in the table region · TODO · depends-on: A1, A2 · wave 2
**Problem:** The #146 residual, and the reason issue #146 is still open after PR #149.
The rowizer's region excludes the column-header band sitting just above the data rows —
on NS p13 the region spans `y 129..374` while the header is at `y 112..120`. PR #149 stopped
the first *data* row being promoted to header, so the table now ships with an **empty**
header instead of a wrong one: lossless, but still schema-less. The header exists in the
PDF and is simply not in the region.
**Do:** Fix whatever A1 names as the cause. Do not assume it is the gap threshold —
`_SPLIT_GAP_MULT` (1.5) and `_SPLIT_GAP_MIN_PT` (10.0 pt) would not split a ~9 pt gap, so the
band is more likely never entering the region than being split out of it. Whatever the cause,
the region must cover the header band, and `_grid_to_markdown`'s `_is_data_row` inference must
then see a real header in `grid[0]` rather than an empty one.
**Files:** `src/socr/tables/reconstruct.py`
**Done when:** NS p13 Table I emits `| 3M Treasury yield | 0.67 |` as a body row **under a populated header row** (not an empty one), the strict-xfail test from A2 still passes, and the full suite passes.
**Note:** same file as A2 — one agent holds `reconstruct.py` and works A2 then A2b in sequence.

### TICKET-A3 — corpus regression · TODO · depends-on: A2, A2b · wave 3
**Problem:** One paper is not evidence of a general fix.
**Do:** Re-run the corpus measurement; report the change in the below-95% table-page
population and confirm no page regressed.
**Files:** `logs/`
**Done when:** a log records before/after counts for table pages below 95% recall, with no page worse than before.
