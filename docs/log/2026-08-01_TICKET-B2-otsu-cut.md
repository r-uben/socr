# TICKET-B2 reopen (second fix) — Otsu-style lane cut replaces ratio-jump

Fixes the two RED gates pinned at `14e7f55` on top of `ca90cd4` (the dedup fix from
`docs/log/2026-08-01_TICKET-B2-reopen-fix.md`), per STATUS.md's "Next action" and the
task-lead brief. Scope: `src/socr/tables/native_rows.py` only.

## Root cause (restated with evidence)

`_gap_cut_threshold` chose the lane boundary as "the largest ratio jump between
consecutive **distinct** positive gap magnitudes." That is structurally unsound: it
picks one distinguished pair of magnitudes and ignores how many gaps actually produced
each one.

Two noise floors exist in real glyph geometry — exact zero (tokens sharing an anchor)
and small non-zero jitter (digit widths, float rounding). The two previous attempts
bracket the failure:

- **Excluding zero** (the shipped rule at `14e7f55`): on the offset-230 regular-grid
  fixture, glyph rendering produces distinct positive gaps `{50.0, 50.000015}` — the
  same lane boundary measured twice with sub-point float noise. With zero excluded,
  that meaningless `50.0 → 50.000015` transition (ratio `1.0000003`) is the *only*
  candidate elbow, so the cut lands at `50.0000075`d and only one of five columns
  clears it — `(0,1,1,1,1)`, 40% faithful score.
- **Including zero** (rejected in the prior fix, per its log): a transition away from
  exact zero is *always* an infinite ratio, so it always wins regardless of whether the
  next value up is real spacing or itself noise — it picked a 5pt digit-jitter gap over
  a real 55pt spacing on the battery fixture (6 regressions).

Neither formulation can separate a noise floor from real spacing by looking at a single
ratio between two neighbours: ratio comparison has no notion of *how much evidence*
supports each magnitude, only their relative size.

## Fix

Replaced the max-ratio search with a 1-D Otsu/Jenks-style partition
(`_gap_cut_threshold` + new `_between_group_variance`, both in `native_rows.py`):

1. Take every gap in the flattened token list, **including zero, undeduplicated** —
   the actual list the caller thresholds against.
2. For every candidate cut (midpoint between two consecutive *distinct* gap
   magnitudes), split the full weighted list into "at or below" / "above" and score
   the split by Otsu's between-class variance: `w_low * w_high * (mean_high -
   mean_low)^2`, weighted by how many gaps land on each side.
3. Take the candidate maximising that score.

This is still parameter-free (no constant, no tolerance, no epsilon) — the cut is
selected by an objective computed from the data. It differs from the ratio rule in
exactly the property that was missing: **weight by evidence, not just relative size.**
On the offset-230 fixture, splitting `{0.0}` from `{50.0, 50.000015}` has enormous
between-group variance (means ~50pt apart, weighted by however many gaps support each
side); splitting `{...,50.0}` from `{50.000015}` has almost none (means nearly
identical, and the high group has support of 1). Otsu's search finds the first split
regardless of what other magnitudes are nearby, which the max-ratio search could not.

With at most one distinct gap magnitude, there's nothing to partition — cut at its own
midpoint against the implied zero floor, same fallback as the previous rule
(`len(positive) == 1`), generalised to also correctly return `0.0` for an all-zero gap
list (draw no boundary).

## Verification beyond pytest

`native_rows_from_page` lane matrix, `~/venvs/socr/bin/python` heredoc probe, widths
2–6 × offsets 210/230/250/270 (even spacing), plus three uneven-spacing geometries:

| geometry | lanes (all 4 rows) |
|---|---|
| width 2–6, offset 210/230/250/270 | `(0,1,...,width-1)` for every combination — 20/20 correct |
| uneven gaps (50,70,90), offset 250 | `(0,1,2,3)` — **fixed**, was `(0,0,1,2)` |
| uneven gaps (50,80,120), offset 250 | `(0,1,2,3)` — **fixed**, was `(0,0,1,2)` |
| uneven gaps (50,70,90), offset 230 | `(0,1,2,3)` — **fixed** |

**Uneven spacing is fixed.** The prior fix's log flagged this as an open, un-gated gap
("the ratio-elbow rule finds the *single* largest jump ... an uneven grid ... has no
signal to distinguish 'the smallest one is noise' from 'all three are signal'"). Otsu
resolves it because the objective is not "is this the single biggest jump" but "does
this split separate two evidence-weighted groups" — the near-zero within-lane cluster
and the {50,70,90} between-lane values are far enough apart in mean that any cut
between them scores far higher than any cut *among* 50/70/90, so all three real
boundaries clear the threshold and every column gets its own lane.

## Test results

- `~/venvs/socr/bin/pytest tests/ -q` — **1392 passed, 2 xfailed** (1394 collected),
  0 failed. Before this fix (`14e7f55`): 2 failed
  (`test_a_perfect_transcription_of_a_regular_grid_scores_100`,
  `test_lane_assignment_is_invariant_to_page_offset` at offsets 230/240).
- Explicitly re-ran the protected guard tests
  (`test_a_perfect_transcription_scores_100`,
  `test_dropping_a_column_never_beats_keeping_the_gap`,
  `test_padding_with_low_entropy_columns_never_beats_a_faithful_transcription`,
  `test_same_shift_plus_spurious_column_never_beats_the_incumbent`,
  `test_wrapped_label_is_scored_the_same_as_unwrapped`, and both newly-green gates):
  13 passed, 1 xfailed (the wrapped-label test, correctly still `strict=True` xfail —
  TICKET-B5, untouched).
- `uvx ruff@0.16.0 format --check .` — `246 files already formatted`, clean.

## Files changed

- `src/socr/tables/native_rows.py` — `_gap_cut_threshold` rewritten (Otsu partition
  instead of max-ratio search), new `_between_group_variance` helper. No other file
  touched; no test file touched.
