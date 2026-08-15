# #213 — the lane-sharing discriminator is dead

Date: 2026-08-16
Supersedes the Task 2 proposal in `docs/log/2026-08-15_213-branch-identification.md`
Issue: <https://github.com/r-uben/socr/issues/213>

## What was proposed

Task 1 established that book indexes reach table reconstruction through the numeric-lane gate
(`has_numeric_columns`), not through PyMuPDF's `find_tables`. That log then proposed a
replacement signal: **lane sharing**. Index pages open many lanes but each row uses only a few,
so rows barely overlap; a real table should be the opposite — few lanes, most rows sharing them.
It closed by flagging that this was a deduction with no measurement of the true-table side.

## What was measured

Session `b219b696` measured it. Statistic: **mean pairwise Jaccard of lane-sets over qualifying
rows** (rows spanning >= 3 lanes), reusing `reconstruct.py`'s `_LANE_X_TOL_PT`, `_NUM_TOKEN_RE`
and `_NUMERIC_RE` unchanged. Per-lane occupancy reported alongside.

| set | n | jaccard range |
|---|---|---|
| A — index pages | 3 | 0.111 – 0.323 |
| B — hand-judged real tables | 8 (7 docs) | 0.600 – 1.000 |
| C — native table pages, background | 403 | 0.030 – 1.000 |

On A vs B alone the signal looks excellent: no overlap, a gap of 0.277.

**The gap does not survive contact with C.** Against A's ceiling of 0.323:

- **112 of 403** background pages (27.8%) score at or below it.
- **92 of 403** (22.8%) fall inside the A–B gap.
- Splitting C by a "Table N" caption regex, captioned median 0.732 vs uncaptioned 0.354 — so the
  signal is *directionally real* — but **10 of 186 captioned pages (5.4%)** sit at or below A's
  ceiling, captioned minimum 0.030.
- Hand-checked examples inside the gap are genuine captioned regression tables:
  `christiano_eichenbaum` p52 ("Table 5.3") scores **0.329** — six thousandths above the
  `nagel` p157 index at 0.323.
- The histogram over C has **no valley below 0.95**. It is a flat continuum, not two modes.

## Verdict

**Do not build a gate on this statistic.** A threshold that catches all three index pages costs
roughly 5% of real captioned tables, and there is no natural place to cut. B's clean floor of
0.600 is an artifact of n=8, not a boundary.

It may survive as one weak term among several. It cannot be the discriminator.

## Independent verification

This log's author did not take the result on trust. Re-implementing the statistic from scratch
and recomputing it:

- **13 of 13** individually checkable pages reproduce to within 0.002 — all of A, all reachable
  of B, both hand-checked C pages, and the figure control.
- The 92-page in-gap count and the 27.8% below-ceiling count reproduce exactly from the raw data.
- The absent valley is visible directly in a 20-bin histogram of C.
- The A-set lane and row counts also match the independent Task 1 measurement.

One B page, `2020__tsukioka_yamasaki` p25, is a 0-byte iCloud placeholder locally and could not
be rechecked; the measuring session reported recovering it from ProtonDrive. It does not affect
the verdict either way.

Raw data: `/tmp/claude_213_lanes/results.json` (not committed — derived from corpus content).

## Known limits of the measurement itself

- **B is 8 pages over 7 documents.** Too small to establish a range.
- **The C captioned/uncaptioned split is a regex proxy**, not human judgement.
- **The statistic is blind to non-index non-tables.** A confirmed figure, `ramey` p122, scores
  1.000/1.000 — indistinguishable from the cleanest real table. Relevant to #150, not to #213.

## What Task 2 needs now

Not more statistics on these 11 pages. To resurrect lane sharing at all, someone needs roughly
**20 more hand-judged real tables drawn specifically from the 0.3–0.6 band** — the region where
the claim actually fails. Otherwise Task 2 needs a different signal, and the geometric-ratio
family should be treated as measured-and-rejected rather than untried.

The Task 1 findings are unaffected: the branch identification stands, and the two remedies it
ruled out (PyMuPDF arbitration, threshold tuning on `_MIN_LANES_PER_ROW`) stay ruled out.

No source touched. No corpus content committed.
