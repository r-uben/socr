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

> **CORRECTION, 2026-08-21 — two ENGINE NAMES in this record were stale.** On an
> accepted GH-96 post-route escalation, socr copies only the escalated attempt's text
> into the winning output (`bo.text = out.text` at
> `src/socr/pipeline/orchestrator.py@7c7f174:2332`). It does not replace that output or
> its `engine` / provider provenance. The escalated text therefore ships under the
> pre-escalation engine's name (GH-274).
>
> The nine document `audit_log.json` files identify exactly 2 escalations across these
> 21 pages, both via qwen: nakamura_steinsson p13 was recorded as `gemini`, and
> pflueger_rinaldi p34 as `nougat`. The verdict JSON and table below now name qwen for
> both. The recorded mix was qwen 11 / gemini 5 / native 4 / nougat 1; the corrected
> mix is **qwen 13 / gemini 4 / native 4 / nougat 0**.
>
> **What survives:** all quality findings, including "2 of 8 citable" and the per-page
> defects, because those came from reading the shipped text against page images rather
> than from trusting its engine label. The cross-run byte comparison also stands. In
> this run, "native's share fell from 8 to 4" still holds because neither escalated
> page carried a native label. That is not a structural guarantee: the caller's guard
> at `orchestrator.py@7c7f174:3387-3401` excludes `chart_asset`, not `native`, so the
> same mutation can in general put model text into a native-labelled output.
>
> This survived eight review rounds, and none of them checked whether the engine labels
> were true. An earlier revision of this note claimed a reason — that every round
> checked quality against the page images instead — which is not supported: the later
> rounds were adversarial checks of the content-free guard, of schema and manifest
> identity, of path encodings, and of wording. Why nobody checked the labels is not
> established, and inventing a cause is the same failure this note is about.

## The number

On the **baseline's 8 contested pages** — the pages where the 2026-08-20 run held two
or more candidates and socr had to choose — it now ships output a reader could cite on
**2**. The count is consumer-dependent and the split is worth naming rather than
hedging: 2 for anything that renders or parses the markdown to a conforming reader —
both Pandoc's GFM reader and markdown-it obey pflueger p34's 4-cell delimiter row and
drop its fourth regression column — and 3 only if a human reading the raw source and
repairing the table's width by hand counts as citable. If the contract is usable
markdown, it is 2.

That denominator is fixed to the baseline on purpose: in this run
only 4 pages had two or more surviving cached candidates, so "pages with a choice" is not a stable
set to measure across two runs, and quoting this run's own count would compare
different denominators.

**The baseline's own split is NOT a measured comparable and this file will not use it
as one.** The 2026-08-20 record states 7 worse, 1 tie, native won nothing outright.
That is the old panel's prose; its per-page verdicts were never COMMITTED — they exist
in that session's own reconciliation artifacts, so they were recorded, just not here —
and the new panel contradicts it on the one page where a direct comparison is possible
(see
"How much of this is the code" below). The defensible before/after is the routing,
which is recorded on both sides.

**An earlier revision of this file said 4, and said all four newly-routed pages became
citable. Both judges said so too. A hand check against the page images disproved it**
— see "What the judges missed" below. Two of the four carry defects that leave the
values right and the binding wrong, which is the failure this whole record is about.

Note also that the two panels were not even asked the same question. The baseline
verdict was *relative* — which of two candidates is better. This one is *absolute* —
is the shipped text citable at all. A page can move on one and not the other.

What is directly comparable, and recorded on both sides rather than judged: **4 of the
8 moved off native to a model lane.** Across all 21 pages the shipped engine was the
qwen model on 13, the gemini model on 4, native on 4, and nougat on 0. Native's share
fell from 8 to 4 in this run. That routing change is real. What the model lanes then
produced is the question the rest of this file is about, and the answer is mixed.

## What this measures and what it does not

This is the **first committed measurement of S1 as it actually shipped.** An earlier
re-measurement was run during the working session against `d25b761`, the S1 branch
before review; its numbers circulated in that session and are superseded here. They
were never committed, so nothing in this repo is being corrected — the 2026-08-20
record contains no re-measurement numbers, only the baseline and the method.

The distinction matters because `d25b761` is not the shipped code. Merged `7c7f174`
differs from it by 146 changed lines in `pipeline/orchestrator.py` (108 added, 38
removed). That is a source-tree comparison, not an ancestry walk: #269 was
squash-merged, so `d25b761`, `d88d01e` and `3cc4d9d` are not ancestors of `7c7f174`
and `git merge-base --is-ancestor` will say so. Two follow-up commits on the
pre-squash branch produced that gap, and only one is about routing: `d88d01e`
reworked the S1 gate itself (96 added, 62 removed against `d25b761`), while `3cc4d9d`
(67 added, 31 removed against `d88d01e`) moved fragment/stitch flushing and left
winner selection alone. The three counts do not sum, because the two commits edit
overlapping regions -- which is exactly why the 146 belongs to the
branch-to-merged gap and to no single commit.

Anyone quoting the earlier session numbers as a description of shipped behaviour
would be wrong.

## The 8 contested pages, before and after

| document | page | kind | baseline | after S1 | verdict on what ships now |
|---|---|---|---|---|---|
| cochrane_piazzesi | 10 | table | native / warning | **qwen** / success | grid, but the `Large T` label sits on the coefficient row |
| cochrane_piazzesi | 12 | table | native / warning | **gemini** / success | citable grid (checked by hand) |
| nakamura_steinsson | 13 | table | native / success | **qwen** / success | citable grid (checked by hand) |
| pflueger_rinaldi | 34 | table | native / success | **qwen** / success | grid malformed: 4-column header, 10 body rows of 5 |
| kaminska_et_al | 38 | figure | native / success | native / **error** | content absent, but a silent SUCCESS became a hard failure |
| cochrane_piazzesi | 15 | table | native / error | unchanged | refusal marker; a cached extraction was discarded (#262) |
| pflueger_rinaldi | 43 | equation | native / warning | unchanged | display structure lost (#271) |
| nakamura_steinsson | 42 | table | native / warning | unchanged | every digit, no grid |

## How much of this is the code and how much is the judges

The obvious threat to the headline is that two different judge panels scored the two
runs, so some of the movement could be re-judging rather than S1. That is testable,
because the shipped text either changed between runs or it did not.

Comparing shipped bytes on the baseline's 8 contested pages:

| | pages | what a verdict difference could mean |
|---|---|---|
| shipped text CHANGED | 5 | the code did something |
| shipped text BYTE-IDENTICAL | 3 | only the judges differ |

The safe claim, and the only one this evidence carries: **both currently citable
outputs sit in the byte-changed group, and none of the three byte-identical outputs is
citable under this panel.**

It is tempting to call that an improvement count and this file will not, for the reason
given above — the two panels answered different questions, and the baseline's per-page
verdicts were never committed, so "how many pages got better" has no value this repo
can check.
What byte-identity establishes is narrower: where the shipped text did not change, any
difference in verdict is the panel, not the code.

The 3 byte-identical pages are p15, p42 and p43, and they are where the panels can be
compared directly. On p42 they disagree outright: the baseline records native winning
nothing on any contested page, while both judges here rate the shipped native output
better than the model alternative — on the same image, the same two candidate files
byte for byte, and the same shipped output. **Judge disagreement on identical evidence
is therefore demonstrated, not hypothetical**, which is the reason the baseline split
is quoted above as prose rather than used as an anchor.

## Judging

Two judges on different vendors (an OpenAI model and an xAI model), each reading the
page image independently, with a grounding requirement: state the caption as printed
and the printed column count before giving a verdict.

They agree on **7 of 8** exactly. The single split is severity on the equation page —
one calls it WRONG, the other DEGRADED; both call it worse than the alternative that
was available. The two absents do not differ between them.

Their agreement is not the reassurance it looks like: both passed the same two pages
that a hand check fails, missing a different defect on each — see below.

| | faithful | degraded | wrong | absent |
|---|---|---|---|---|
| judge 1 | 4 | 1 | 1 | 2 |
| judge 2 | 4 | 2 | 0 | 2 |

**Both faithful counts are too high by two.** See below.

## What the judges missed, and why it is the most useful thing here

Both judges rated cochrane p10 FAITHFUL. Both cited its coefficient values matching
the page. **Neither detected that those values were bound to the wrong rows** — and
they were. What a judge internally checked is not observable; what is observable is
that both justifications cite matching values, neither mentions the binding, and the
binding is wrong. This file claims the miss, not the reasoning behind it.

On the page, the coefficient row of the top panel is **unlabelled**, and the row below
it is labelled `Large T`. The shipped grid moves that one label up: `Large T` lands on
the coefficient row, and the real Large-T standard-error row is left unlabelled.
`Small T` and `EH` stay on their correct rows — an earlier revision of this file said
every label in the panel had shifted, which is not true and overstated the defect.

One label is enough. A reader taking `Large T` from this grid gets the coefficient row,
and the standard errors it names are orphaned on the row beneath. Every number is
present and correct.

pflueger p34 fails differently and no judge mentioned it either: its delimiter row
declares 4 columns while 10 body rows carry 5, so a standard markdown parser drops the
excess cell from each — the entire fourth regression column. The source text holds the
values in the right places; anything that renders or parses the file does not.

So of the 4 pages S1 newly routed to a model, **2 produce citable output, 1 mislabels
its coefficient row, and 1 is structurally malformed** — the last being citable only to
a reader who repairs the table width by hand, since a conforming parser drops a column.

Two consequences, and the second is the reason this section exists:

1. **Model output is not citable merely because a grid exists and the numbers are
   right.** That is the assumption S1 ships on, and it does not hold on half the pages
   S1 newly routed here.
2. **A structure rubric did not produce a structure check.** Both judges, on different
   vendors, were given a rubric naming row and column binding explicitly plus a
   grounding requirement. Both passed a shifted label on one page and a malformed grid
   on another, and every justification they wrote cites values rather than bindings.
   Whether they looked at the structure and misread it, or never looked, is not
   something their outputs can settle — and it does not matter for the conclusion: two
   instruments, two different misses, and asking for structure in the prompt was not
   sufficient to get it. Not a law about judges in general; enough to stop relying on
   them alone here. The 2026-08-20 record lists
   "one row-label shift" among the model defects it found, so this class was known and
   still walked past. Any future measurement here needs a mechanical binding check —
   the judges cannot be the only instrument, because they fail the same way the
   pipeline does.

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
grid and corrupts a digit.

The 2026-08-20 file records the same `1.11`-for-`1.10` slip, but that is **not** a
second independent observation and an earlier draft here wrongly implied it was: the
model's candidate file for this page is byte-identical between the two runs, as is the
page image and the shipped output. The two runs used separate output directories and
separate caches and no shared cache was found, so the likely explanation is that
decoding is deterministic on this page — but that is not established here, and until
it is, the slip is one observation seen twice rather than two.

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
**p10, the shifted-label page filed as #273** — which is the genuine identical-bag case
and is measured, not asserted: baseline native and the qwen output shipped today each
hold **50 decimals with identical sign-aware multisets**. They are not otherwise alike,
and an earlier revision of this file wrongly said they differed only in a label:
native is a 109-line flattened stream with no table at all, the qwen output a 46-line
grid. The precise claim is that their numeric bags are identical while the qwen grid
misbinds `Large T`, and only the page image settles which binding is right. A bag
comparison cannot see that defect at all. GH-270 is
NOT such a case despite an earlier revision of this file citing it as one — its
fabricating output holds 176 decimals against the source's 152, so the bags differ
substantially and a multiset check would notice. It is a fabrication case, not a
binding-blindness case.

So the record's own strongest evidence for #266 is p10, and it took a hand check to
find it: it is a page where every number is right, the bag is identical, a grid exists,
and the output is still uncitable.

## What came out of it

- **#262 reproduces on merged main.** On cochrane_piazzesi p15 a 102-byte refusal
  marker shipped while a 2,546-byte cached extraction — 11 grid rows, 36 values, plus
  the page's prose — sat in the same run's cache. PR #264 is the fix and is blocked on
  the shared grid predicate (#268).
- **#271 filed.** An equation page shipped native under WARNING with
  `needs_ocr_enhancement` set, while a cached candidate with correct aligned LaTeX was
  discarded: 0 fractions and 0 display environments shipped, against 33 `\frac`
  commands and 3 `aligned` environments available. S1 does not reach it — S1 is gated
  on `is_structure_class()`, which means tables.

## What in this record cannot be checked from the repo

Marked explicitly, as in the 2026-08-20 file.

**Everything content-level is session record, not repo-checkable.** The verdict JSON
records which engine shipped, with what flags and status, and how many decimals each
cached attempt held. It does not record what any of that text said, so no claim about
quality can be confirmed from this repo. That covers, exhaustively:

- the per-page quality verdicts (faithful / degraded / wrong / absent) and both
  judges' category counts, including the 7-of-8 agreement and the grounding procedure
  they were held to;
- every "verdict on what ships now" cell in the before/after table: the p10 shifted
  label, the p34 width defect, and the hand checks that found p12 and p13 clean;
- everything asserted about p42 — "every digit, no grid", native rated better than the
  model, neither lane citable, and the `1.10`/`1.11` substitution itself;
- the p15 figures (11 grid rows, 36 values, prose recovered) and the p43 figures
  (0 fractions and 0 display environments shipped, against 33 `\frac` commands and 3
  `aligned` environments in the discarded candidate);
- that the two runs differ only in the code under test, and which commit each ran
  against. The verdict JSON carries no SHA; `MEASURED_AGAINST` was a session file;
- the accepted qwen escalations on p13 and p34 that justify correcting their recorded
  producers — the source `audit_log.json` files are not committed;
- the entire cross-run byte comparison — that shipped text changed on 5 of the 8 pages
  and was byte-identical on 3, which 3 those were, that p42's two candidate files and
  page image are identical across runs, and that the runs used separate output
  directories with no shared cache found. None of it is derivable from the committed
  JSON, and the whole "how much of this is the code" section rests on it;
- the judges' exact rubric and their written justifications, and the check that a
  conforming markdown parser drops p34's fourth column.

The routing, flags, statuses, decimal counts, candidate lists and engine mix ARE in
`2026-08-21_lane-comparison-after-s1-verdicts.json`, and the arithmetic over them can
be rechecked from this repo alone.

## Re-running it

Identical to the 2026-08-20 procedure — same `select.py`, same runner, same manifest.
Point `LANE_CAMPAIGN_DIR` at a directory holding the manifest and run the committed
runner against a checkout of the commit you want to measure. Prove isolation first:
the editable install resolves `import socr` to the main checkout unless `PYTHONPATH`
points at the tree under test, and without that check every number is void.
