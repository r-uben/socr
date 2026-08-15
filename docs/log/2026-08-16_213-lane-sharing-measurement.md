# #213 — does lane SHARING separate a book index from a real table?

Date: 2026-08-16
Issue: <https://github.com/r-uben/socr/issues/213>
Predecessor: `2026-08-15_213-branch-identification.md` (established that `has_numeric_columns`
is the branch that admits a book index; `find_tables()` never fires)

**Verdict: no. The idea is dead as a gate. Killed before any code was written.**

## The hypothesis

`2026-08-15_213-branch-identification.md` established *why* index pages pass the numeric-lane
gate: an index entry carries a *list* of page numbers, so one wrapped prose line populates many
lanes at once. That log closed by observing no value of `_MIN_LANES_PER_ROW` separates the two
populations.

The follow-on idea was that the populations differ in *shape*, not in count:

> Index pages open MANY lanes but each row uses only a FEW of them, so rows barely overlap.
> Real numeric tables should be the opposite: few lanes, and most rows populate the SAME lanes.

If true, some lane-*sharing* statistic separates them where a lane-*count* threshold cannot.

## Statistic

**Mean pairwise Jaccard of lane-sets over qualifying rows** (rows populating `>= _MIN_LANES_PER_ROW`
lanes). Chosen over per-lane occupancy because it measures the hypothesis directly — "do two rows
use the *same* lanes" — and separated better on the judged pages. Occupancy
(`sum(lanes_per_row) / (lanes_touched * qualifying_rows)`, the fill fraction of the row-by-lane slot
matrix) is reported alongside as a cross-check; it tells the same story slightly less sharply.

Lane clustering reuses `reconstruct.py`'s own `_LANE_X_TOL_PT`, `_NUM_TOKEN_RE`, `_NUMERIC_RE` and
the exact clustering loop from `has_numeric_columns`. Nothing was re-invented, so the numbers below
describe the gate as it actually ships.

## Population A — hand-judged book indexes (n=3)

| page | jaccard | occupancy |
|---|---|---|
| `2003__woodford` p798 | 0.111 | 0.238 |
| `2003__woodford` p799 | 0.176 | 0.308 |
| `2021__nagel__ml_ap` p157 | **0.323** | 0.272 |

Range: jaccard `[0.111, 0.323]`, occupancy `[0.238, 0.308]`.

## Population B — hand-judged real tables (n=8, 7 documents)

Verdicts from `2026-08-15_tr3-hand-judgement.md` and `2026-08-15_b1-hand-judgement.md`; a "damaged"
verdict means a real table was present.

| page | jaccard | occupancy |
|---|---|---|
| `2013__Snowberg_Wolfers_Zitzewitz` p15 | **0.600** | 0.500 |
| `2018__herskovic__JF` p25 | 0.609 | 0.471 |
| `2020__tsukioka_yamasaki` p25 | 0.742 | 0.429 |
| `2017__ozdagli` p38 | 0.758 | 0.429 |
| `2010__Menzly_Ozbas` p20 | 0.778 | 0.500 |
| `2000__romer_romer` p21 | 0.790 | 0.479 |
| `2013__Snowberg_Wolfers_Zitzewitz` p16 | 1.000 | 1.000 |
| `2015__Hameed_Morck_Shen_Yeung` p49 | 1.000 | 1.000 |

Range: jaccard `[0.600, 1.000]`, occupancy `[0.429, 1.000]`.

## A vs B: no overlap — and that is the trap

A and B do not overlap on either statistic. Gap width: **0.277** jaccard (0.323 → 0.600), 0.121
occupancy. Taken alone this reads as a viable discriminator.

**It is an artifact of n = 8.** B spans 8 pages across 7 documents. Its clean floor of 0.600 is
where the smallest of eight samples happened to land, not a boundary of the population.

## Population C — the background that kills it

All native table pages across the 40-paper probe list (`/tmp/b1probe/list.txt`): **403 pages**.
Contaminated by construction — some are the very indexes being excluded.

Splitting C by whether a `Table N` caption appears in the page text (a regex proxy for "a real table
is here", not a judgement):

| | n | min | p10 | median | max |
|---|---|---|---|---|---|
| captioned | 186 | 0.030 | 0.360 | **0.732** | 1.000 |
| uncaptioned | 217 | 0.000 | 0.056 | **0.354** | 1.000 |

**The signal is real.** A two-to-one median separation on 403 pages is not noise, and the low tail
is dominated by exactly the pages that should not be tables — spot-checking the bottom of the
distribution found prose, displayed equations, figure pages, and bibliographies, not tables.

**The signal is not separable.**

- **10 of 186 captioned table pages (5.4%)** sit at or below A's ceiling of 0.323 — including
  `2000__romer_romer` p23, `2018__herskovic` p28, `2016__ramey` p112. Captioned minimum is 0.030,
  far below every index page.
- **92 pages (22.8%) sit inside the A–B gap.** Hand-checking a spread across it found captioned
  regression tables throughout: `1994__christiano_eichenbaum_evans` p52 (`Table 5.3`) scores
  **0.329** — six thousandths above `nagel` p157.
- The jaccard histogram has **no valley anywhere below 0.95**. It is a continuum with a spike at
  1.0, not two modes.

So a threshold set to catch all three known index pages costs roughly 5% of real captioned table
pages, and it is chosen from a distribution offering no natural cut.

## What this does NOT establish

- B is too small to establish a range. Its floor is a sample artifact; do not quote 0.600 as a
  boundary.
- The C captioned/uncaptioned split is a **regex proxy**, not hand judgement. It bounds the
  direction of the effect, not its precision.
- The statistic is **blind to non-index non-tables**. `2016__ramey` p122 — hand-judged "not a table"
  (a figure) in `2026-08-15_tr3-hand-judgement.md` — scores **1.000 / 1.000**, the maximum. Even a
  working index discriminator would not have caught the other non-table in the same judged sample.

## Recommendation

Do not build a gate on lane sharing. It may survive as one weak term among several; it cannot be
the discriminator.

If it is ever revived, the missing input is **~20 more hand-judged real tables drawn specifically
from the 0.3–0.6 band** — the disputed region. More statistics on these 11 pages will not settle
anything.

Raw results: `/tmp/claude_213_lanes/results.json` (403 C rows + A and B per-page stats). 0 pages
evicted, 0 unreadable; `2020__tsukioka_yamasaki` was a 0-byte iCloud placeholder and was recovered
from its ProtonDrive twin rather than counted as clean.
