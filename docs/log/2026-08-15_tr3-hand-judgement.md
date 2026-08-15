# TR-3 hand judgement — what the 62 firings actually are

Date: 2026-08-15
Status: **measurement complete, sampling stopped early.** Input to the #200 decision.
Companion: `2026-08-14_gh151-b1-escalation-decision.md` (the decision this informs),
`2026-08-14_gh151-b1-predicate-design.md` (the corpus measurement),
`docs/plans/fake-native-pages/` (the A1 exclusion applied to the sample).

## Why this exists

The escalation decision recorded TR-3 (`has_unverifiable_table_region`) as the best available
structural signal: *"62 of 245 pages (25.3%) carry a detected geometry hard-fail that nothing
surfaces… a verified geometric failure, not a shape heuristic — better evidence than anything B1
invented."*

That was an inference from the firing count. Nobody had looked at the pages. This is that look.

## Method

Review set: the 62 TR-3 firings, minus 6 fake-native pages (per the A1 definition:
`raster_coverage >= 0.90 AND char_count > 0 AND is_born_digital`), capped at 34 and shuffled with
seed 20260814 so no single document dominates the head of the queue. Set lives outside the repo at
`~/.local/share/socr/tr3-judge/` — the corpus is copyrighted and this is a public repo.

Each page judged side-by-side, rendered page against socr's markdown, by the owner.

**Sampling stopped at 7 of 34.** Every real table was damaged, each with multiple independent
defects. Continuing would refine a rate that does not change any decision below.

## Result

| # | page | verdict |
|---|---|---|
| 1 | `021_2017__ozdagli` p38 | damaged |
| 2 | `039_2021__nagel__ml_ap` p157 | **not a table** (book index) |
| 3 | `028_2018__herskovic__JF` p25 | damaged |
| 4 | `018_2016__ramey__shocks` p122 | **not a table** (figure) |
| 5 | `011_2003__woodford` p799 | **not a table** (book index) |
| 6 | `037_2020__tsukioka_yamasaki` p25 | damaged |
| 7 | `015_2015__Hameed_Morck_Shen_Yeung` p49 | damaged |

- **4 of 4 real tables damaged.**
- **3 of 7 firings are not tables at all** — two book back-matter indexes, one figure.

## Finding 1 — TR-3 is blind to the defects that actually ruin the table

Every numeral was individually correct on all four damaged pages. Not one numeral was reliably
attributable to both its row and its column.

| defect | seen in | visible to TR-3? |
|---|---|---|
| header band destroyed / detached from columns | 4/4 | **no** — no digits change |
| row labels detached or off-by-one from their values | 3/4 | **no** |
| star-only row deleted entirely (`DWH test`) | 1/4 | **no** — the row holds no numerals |
| coefficient row torn across two output rows | 2/4 | yes |
| table content re-emitted as loose text after the table | 1/4 | no |

TR-3 is a numeric-token multiset check over reconstruction geometry. It compares numbers to numbers.
A destroyed header changes no number; a deleted significance-star row changes no number. TR-3 caught
these pages, but on the torn rows — not on the damage that makes the output unusable.

Concrete: on `037_2020__tsukioka_yamasaki` p25 the entire `DWH test` row (`*** *** ** *** ***`) is
absent from the output. socr can delete a whole row of a regression table and TR-3 registers nothing.

**This directly contradicts the premise the escalation decision rested on.** TR-3 is not a
better-evidenced version of the structural signal. It is a signal on a different axis, and the axis
it measures is not where the failures are.

## Finding 2 — TR-3 fires on pages with no table on them

Three of seven. Two are book back-matter indexes (Nagel p157, Woodford p799), one is a figure
(Ramey p122).

The mechanism is not mysterious for the indexes: a two-column index is short dense lines with page
numbers trailing each line. That right-hand run of page numbers is a numeric lane. TR-3 is a
numeric-lane geometry check. It fires by construction.

An escalation signal built on TR-3 would therefore spend model calls on pages the native path
handles correctly — the precise cost the owner's routing principle exists to avoid (*use native
where there is text; use OCR for tables, figures, formulas*).

Note this is upstream of TR-3: the **table detector** is wrong on these pages before reconstruction
runs at all. Nothing downstream can be right on a page with no table.

## Finding 3 — this is an argument for B1's predicate, not against it

`structural_gate_fires()` is `ragged OR detached_label_rows`. **Detached label rows is defect (b) in
the table above — the second most common failure, 3 of 4 pages.** B1's predicate looks directly at
the thing that breaks.

The escalation decision criticised the predicate for firing on 26.9% of native table pages, with
reviewers naming four legitimate shapes it flags (textual column headers, panel sub-headings, units
rows, alternative numbering).

**Hypothesis, not measured:** those four shapes are also the shapes that co-occur with real damage —
every damaged page here is a multi-panel or multi-header table, and the panel/header band is exactly
what got destroyed. If so, part of the 26.9% is not false positives. This has NOT been tested and
must not be treated as established.

## What this changes

- The escalation decision's **Option 1 ("land the plumbing, replace the predicate with TR-3")** is
  now the weakest option, not the strongest. TR-3 has ~57% precision as a table signal (4/7) and is
  structurally blind to 3 of the 5 observed defect classes.
- **Follow-up 2** in that log — *"62/245 table pages carry an unsurfaced TR-3 geometry hard-fail…
  arguably more valuable than B1 itself"* — is **retracted**. The 62 are not 62 broken tables. On
  this sample they are roughly 57% tables, all broken, and 43% pages that should never have been
  assessed as tables.
- **Issue #211** (`--native-only` ships table pages with a hardcoded `PageStatus.SUCCESS` and no
  attempt recorded) is confirmed as the one live content-integrity hole reachable today: it is the
  only default-reachable path on which a table like these ships labelled clean.

## What is NOT claimed

These 62 pages are a **selected** sample — selected because TR-3 already found something anomalous.
"When a native-reconstructed table page fires TR-3, the table is broken" is supported. **"Native
table reconstruction is always broken" is not measured** and does not follow from this. The base rate
over all 245 native table pages is unknown.

Standing lesson, applied here rather than learned again: *a vivid single-page failure is a hypothesis
about a population, not a measurement of one.* This document is the measurement; it is also a
measurement of a selected subpopulation, and says so.

## Open questions for the #200 session

1. Given TR-3's blindness to header and star-row loss, does the escalation predicate need a
   **header-attribution** check — every data column resolves to a header cell — as a third term
   alongside `ragged` and `detached_label_rows`?
2. Is the 26.9% B1 firing rate partly real damage? Testable directly: judge a sample of the pages B1
   fires on and TR-3 does not.
3. Should the **table detector** be gated ahead of any of this? Indexes and figures reaching table
   reconstruction is a defect in its own right, independent of which structural gate ships.

## Filed

Both bugs this measurement surfaced were unfiled; filed 2026-08-15:

- **#212** — `bug(agentic): EXACT_PASS accepts a model table with no structural check`.
  `native_verifier.py:1055-1058` sets `EXACT_PASS` on purely numeric-multiset
  conditions; `agentic.py:595-612` returns `accept=True, confidence=1.0` without ever
  calling the inner visual/structural judge. Cross-referenced against #151, #162, #205.
- **#213** — `bug(tables): book indexes and figures are routed to table reconstruction`.
  3 of 7 TR-3 firings in this sample were not tables (two book back-matter indexes, one
  figure); a two-column index's trailing page-number run is a numeric lane that a
  numeric-lane geometry check fires on by construction. Cross-referenced against #150
  and the closed #113.
