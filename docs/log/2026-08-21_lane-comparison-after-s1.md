# The same 21 pages, measured again after S1 landed

2026-08-21. A re-run of the 2026-08-20 lane measurement against `main@7c7f174`, the
commit that merged #269. Same 9 economics papers, same 21 pages, same runner, same
manifest. The only thing that changed is the code under test, which is the whole
design of a before/after.

Read `2026-08-20_lane-comparison.md` first: it defines the contested set, the method,
and the baseline this file is measured against.

**The page content is not here and cannot be.** The corpus is copyrighted and this
repo is public. What is committed is the method, the per-page routing verdicts, and
the statistics.

## The number

On the **8 pages where socr held two or more candidates and had to choose**, it now
ships output a reader could cite on **4**. On the baseline it shipped the better of
its two candidates on **0 of 8** — 7 worse, 1 tie, native won nothing outright.

Those two numbers are not the same measurement and this file will not pretend they
are. The baseline verdict was *relative* (which of the two candidates was better);
this one is *absolute* (is the shipped text citable at all). What is directly
comparable is the routing: **4 of the 8 moved off native to a model lane, and all 4
became citable.** Across all 21 pages the shipped engine was the local model on 11,
the cloud model on 5, native on 4, and nougat on 1 — native's share fell from 8 to 4.

## What this measures and what it does not

This run was NOT a repeat of the one quoted in the 2026-08-20 record's re-run
section. That earlier re-measurement ran against `d25b761`, **before** the review fix
`d88d01e` reworked the S1 gate by 146 lines in `pipeline/orchestrator.py`. Its
numbers never described shipped code. This file replaces them.

## The 8 contested pages, before and after

| document | page | kind | baseline | after S1 | verdict on what ships now |
|---|---|---|---|---|---|
| cochrane_piazzesi | 10 | table | native / warning | **qwen** / success | citable grid |
| cochrane_piazzesi | 12 | table | native / warning | **gemini** / success | citable grid |
| nakamura_steinsson | 13 | table | native / success | **gemini** / success | citable grid |
| pflueger_rinaldi | 34 | table | native / success | **nougat** / success | citable grid |
| kaminska_et_al | 38 | figure | native / success | native / **error** | content absent, but a silent SUCCESS became a hard failure |
| cochrane_piazzesi | 15 | table | native / error | unchanged | refusal marker; a cached extraction was discarded (#262) |
| pflueger_rinaldi | 43 | equation | native / warning | unchanged | display structure lost (#271) |
| nakamura_steinsson | 42 | table | native / warning | unchanged | every digit, no grid |

## Judging

Two judges on different vendors (an OpenAI model and an xAI model), each reading the
page image independently, with a grounding requirement: state the caption as printed
and the printed column count before giving a verdict.

They agree on **7 of 8** exactly. The single split is severity on the equation page —
one calls it WRONG, the other DEGRADED; both call it worse than the alternative that
was available. Neither the 4-of-8 headline nor the two absents differ between them.

| | faithful | degraded | wrong | absent |
|---|---|---|---|---|
| judge 1 | 4 | 1 | 1 | 2 |
| judge 2 | 4 | 2 | 0 | 2 |

## The finding that complicates the story

The page this whole line of work was argued from — nakamura_steinsson p42, the
flattened regression table — did **not** move, and that is not simply a failure.

Both judges rate the shipped native output DEGRADED but **better** than the model
alternative, because the model's grid substitutes a wrong digit: the page prints
`1.10` in one cell and `1.11` in another, and the model's grid carries `1.11` twice.
This was confirmed three ways — both judges independently, and by hand against the
page image.

So p42 is not a case of socr choosing the worse lane. It is a page where **neither
lane is citable**: native keeps every digit and loses the grid, the model builds the
grid and corrupts a digit. The same `1.11`-for-`1.10` slip is recorded in the
2026-08-20 file, so it reproduces rather than being run-to-run noise.

That is the strongest argument in this record for the binding oracle (#266): the
defect is invisible to any check that compares bags of numbers, and it is equally
invisible to a check that only asks whether a grid exists.

## What came out of it

- **#262 reproduces on merged main.** On cochrane_piazzesi p15 a 102-byte refusal
  marker shipped while a 2,546-byte cached extraction — 11 grid rows, 36 values, plus
  the page's prose — sat in the same run's cache. PR #264 is the fix and is blocked on
  the shared grid predicate (#268).
- **#271 filed.** An equation page shipped native under WARNING with
  `needs_ocr_enhancement` set, while a cached candidate with correct aligned LaTeX was
  discarded: 0 fractions and 0 display environments shipped against 11 and 6
  available. S1 does not reach it — S1 is gated on `is_structure_class()`, which means
  tables.

## What in this record cannot be checked from the repo

Marked explicitly, as in the 2026-08-20 file.

The per-page **quality** verdicts (faithful / degraded / wrong / absent) exist only as
the table above. They came from two models reading page images that cannot be
committed. The verdict JSON records which engine shipped and with what flags — not
which output was better.

Also session record rather than repo-checkable: the `1.10`/`1.11` substitution, the
byte counts on the discarded p15 and p43 candidates, and the judges' 7-of-8
agreement. The routing, flags, statuses, decimal counts and engine mix are all in
`2026-08-21_lane-comparison-after-s1-verdicts.json`.

## Re-running it

Identical to the 2026-08-20 procedure — same `select.py`, same runner, same manifest.
Point `LANE_CAMPAIGN_DIR` at a directory holding the manifest and run the committed
runner against a checkout of the commit you want to measure. Prove isolation first:
the editable install resolves `import socr` to the main checkout unless `PYTHONPATH`
points at the tree under test, and without that check every number is void.
