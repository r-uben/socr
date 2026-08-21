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

On the **baseline's 8 contested pages** — the pages where the 2026-08-20 run held two
or more candidates and socr had to choose — it now ships output a reader could cite on
**4**. That denominator is fixed to the baseline on purpose: in this run only 4 pages
had two or more surviving cached candidates, so "pages with a choice" is not a stable
set to measure across two runs, and quoting this run's own count would compare
different denominators. On the baseline it shipped the better of
its two candidates on **0 of 8** — 7 worse, 1 tie, native won nothing outright.

Those two numbers are not the same measurement and this file will not pretend they
are. The baseline verdict was *relative* (which of the two candidates was better);
this one is *absolute* (is the shipped text citable at all). What is directly
comparable is the routing: **4 of the 8 moved off native to a model lane, and all 4
became citable.** Across all 21 pages the shipped engine was the local model on 11,
the cloud model on 5, native on 4, and nougat on 1 — native's share fell from 8 to 4.

## What this measures and what it does not

This is the **first committed measurement of S1 as it actually shipped.** An earlier
re-measurement was run during the working session against `d25b761`, the S1 branch
before review; its numbers circulated in that session and are superseded here. They
were never committed, so nothing in this repo is being corrected — the 2026-08-20
record contains no re-measurement numbers, only the baseline and the method.

The distinction matters because `d25b761` is not the shipped code. Merged `7c7f174`
differs from it by 146 changed lines in `pipeline/orchestrator.py` (108 added, 38
removed). Two commits sit in that gap and only one of them is about this: `d88d01e`
reworked the S1 gate itself (96 added, 62 removed against `d25b761`), while `3cc4d9d`
(67 added, 31 removed) split a fragment flush and does not touch routing. The three
line counts do not sum, because the two commits edit overlapping regions -- which is
exactly why the 146 belongs to the branch-to-merged gap and to no single commit.

Anyone quoting the earlier session numbers as a description of shipped behaviour
would be wrong.

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

**This page is NOT an example of multiset blindness, and an earlier draft of this
file said it was.** Normalising every minus variant and comparing the two candidates
as bags of decimals: both hold 152, native holds one `1.10` the model lacks, the model
holds one extra `1.11`. A Unicode-aware multiset comparison separates them exactly.
What it cannot see is native's flattening — that bag is identical to a correctly bound
table's — and a grid-existence check catches *that* one while being blind to a changed
digit.

So the two checks are complementary here, and neither defect on this page escapes
both. What p42 actually shows is narrower and still worth the space: on one page, the
lane that preserves the digits destroys the binding, and the lane that builds the
binding alters a digit, so **shipping either one unqualified is wrong** and no
single-lane policy fixes it.

The argument for the binding oracle (#266) rests on the 2026-08-20 finding and on
GH-270 — a value that exists elsewhere on the page placed at the wrong (row, column),
where the bag really is identical and only position distinguishes right from wrong.
It does not rest on this page, and this file no longer claims it does.

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

**Everything content-level is session record, not repo-checkable.** The verdict JSON
records which engine shipped, with what flags and status, and how many decimals each
cached attempt held. It does not record what any of that text said, so no claim about
quality can be confirmed from this repo. That covers, exhaustively:

- the per-page quality verdicts (faithful / degraded / wrong / absent) and both
  judges' category counts, including the 7-of-8 agreement and the grounding procedure
  they were held to;
- the four "citable grid" readings in the before/after table;
- everything asserted about p42 — "every digit, no grid", native rated better than the
  model, neither lane citable, and the `1.10`/`1.11` substitution itself;
- the p15 figures (11 grid rows, 36 values, prose recovered) and the p43 figures
  (0 fractions and 0 display environments shipped against 11 and 6 available);
- that the two runs differ only in the code under test, and which commit each ran
  against. The verdict JSON carries no SHA; `MEASURED_AGAINST` was a session file.

The routing, flags, statuses, decimal counts, candidate lists and engine mix ARE in
`2026-08-21_lane-comparison-after-s1-verdicts.json`, and the arithmetic over them can
be rechecked from this repo alone.

## Re-running it

Identical to the 2026-08-20 procedure — same `select.py`, same runner, same manifest.
Point `LANE_CAMPAIGN_DIR` at a directory holding the manifest and run the committed
runner against a checkout of the commit you want to measure. Prove isolation first:
the editable install resolves `import socr` to the main checkout unless `PYTHONPATH`
points at the tree under test, and without that check every number is void.
