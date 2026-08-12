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

⚠️ `src/socr/tables/reconstruct.py` carries the GH-146 work, committed and open as
**PR #149**. Do not dispatch A2 until it merges (wave 0).

## Stream A — diagnose, then fix

### TICKET-A1 — characterise the misplacement · TODO · depends-on: none · wave 1
**Problem:** The fix must target the actual cause, not the first plausible one.
**Do:** For the synthetic repro and for NS p42/p17/p43, record where each lane
boundary lands relative to word bboxes, and whether lost values cluster by column
(lane-assignment bug) or by row (row-segmentation bug). Write findings to `logs/`.
No production code changes.
**Files:** `docs/plans/gh144-rowizer-destroys-values/logs/`
**Done when:** a dated log states which of the two hypotheses holds, with per-value evidence for at least 10 destroyed values.

### TICKET-A2 — never place a lane boundary inside a token · TODO · depends-on: A1 · wave 2
**Problem:** A column boundary that splits a word or number is always wrong,
whatever the x-clustering says.
**Do:** Constrain lane boundaries to fall in whitespace between word bboxes. A
candidate boundary intersecting a word's bbox must be moved to the nearest gap.
**Files:** `src/socr/tables/reconstruct.py`
**Done when:** `~/venvs/socr/bin/pytest tests/test_region_overlap_gh145.py::test_no_table_value_is_lost -q` passes (the strict xfail flips), and the full suite still passes.

### TICKET-A3 — corpus regression · TODO · depends-on: A2 · wave 3
**Problem:** One paper is not evidence of a general fix.
**Do:** Re-run the corpus measurement; report the change in the below-95% table-page
population and confirm no page regressed.
**Files:** `logs/`
**Done when:** a log records before/after counts for table pages below 95% recall, with no page worse than before.
