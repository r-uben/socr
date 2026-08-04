# TICKET-B2 reopen (third fix) — paired columns, left anchor, zero-floor cut

Fixes the RED gate pinned in STATUS.md's "Next action"
(`test_paired_columns_do_not_collapse_into_one_lane`) on top of `52d26ea`, without
touching any test file. Scope: `src/socr/tables/native_rows.py` only.

## Root cause (two separate defects, not one)

The prior Otsu-cut fix (`docs/log/2026-08-01_TICKET-B2-otsu-cut.md`) correctly weighted
candidate cuts by evidence, fixing regular grids, offset sensitivity, and uneven
spacing. Two things remained broken, and both had to be fixed for the paired-column
gate to go green without regressing the battery fixture:

1. **The battery fixture's own digit-jitter counterexample was never really solved,
   only avoided.** `test_metric_corruption_battery.py`'s base fixture is
   left-aligned with digit-count-dependent right edges (`"42.8"` vs `"3.8"` → 5.0pt
   right-edge gap, genuine non-zero noise, not float rounding). Any `_gap_cut_threshold`
   change that unconditionally isolates the lowest-magnitude gap reproduces the
   documented 6-failure regression from `docs/log/2026-08-01_TICKET-B2-reopen-fix.md`
   (`Errors and fixes` section), because that 5pt gap is real evidence, not noise —
   the *magnitude* alone cannot distinguish it from a genuine paired-column boundary.

2. **A single 2-way Otsu split still assumes exactly two gap-magnitude groups.**
   Paired-year tables produce three: ~0 within a lane, a tight within-pair gap, and a
   wide between-pair gap. The variance-maximising cut picks whichever single boundary
   is most bimodal, which merges the within-pair gap into the zero-lane and collapses
   each pair into one lane — `(0,0,1,1)`, 50%.

## Fix

Two changes, each addressing one defect:

**(a) A third, left-edge anchor in `_assign_lanes`.** The anchor-selection step
already tried right edges and centres, picking whichever gives lower total within-lane
dispersion. Added `_anchor_left` and a three-way comparison. Left-aligned text (the
battery fixture's insertion style) has an *exactly constant* left edge regardless of
digit count — zero dispersion — so it wins outright over right (10.7pt dispersion from
digit-width jitter) and centre (2.68pt). This resolves defect 1 **without any change
to the cut algorithm**: by the time `_gap_cut_threshold` sees the gap list for the
winning anchor, the digit-jitter gap simply isn't there.

Tie-breaking three anchors that all achieve equal (often exactly `0.0`) dispersion
needed its own fix: dispersion alone under-determines the winner once part (b) below
lets a zero-floor anchor over-split into many trivially-zero-variance clusters (every
singleton cluster has 0 variance by construction, so finer partitions are *free* to tie
without evidence). Tie-break now also compares cluster count — the more parsimonious
partition wins — before falling back to the fixed anchor-preference order
(right > centre > left).

**(b) An unconditional zero-floor split in `_gap_cut_threshold`.** When the sorted,
deduplicated candidate list's smallest magnitude is exact `0.0`, the cut is placed at
`candidates[1] / 2` (just above zero) without running the variance search at all;
every distinct positive magnitude, however many there are, becomes a lane boundary.
The Otsu search is unchanged and still runs whenever no exact zero is present.

This is defensible as *not* a tuned/magic constant because `0.0` is not a measured
quantity but an identity: two anchors land at *exactly* the same coordinate (IEEE-754
equality) only when two tokens genuinely share an anchor — real, distinct token
positions cannot produce that by accident, unlike any positive gap, however small,
which is always a real measured distance. Because defect 1 is now resolved upstream by
anchor selection (point a), the anchor `_gap_cut_threshold` actually receives a gap
list for never has a "fake" near-zero gap masquerading as noise — every zero it sees is
real, and every positive magnitude above it is real column spacing. That collapses
"how many classes are there above zero" into an irrelevant question for the *cut value*
itself: `_cluster_by_anchor` only checks `gap > cut`, so as long as the cut sits between
`0` and the smallest positive magnitude, every distinct positive magnitude — one, two,
or ten — correctly starts a new lane, regardless of how many further sub-classes exist
among the positives.

### How the class count is actually selected (precise, checkable)

There is exactly one data-derived binary decision made by `_gap_cut_threshold`, made
freshly for every gap list it is called with: **is an exact-zero magnitude present in
the candidate set (`candidates[0] == 0.0`), yes or no** — that is read directly off the
data (`sorted(set(gaps))`), not assumed. If yes, the cut is pinned to isolate it
(`candidates[1] / 2`) and every positive magnitude is treated as spacing. If no, the
existing weighted Otsu/Jenks 2-way search (unchanged, `_between_group_variance`) picks
the cut among the positive-only candidates exactly as before. No constant, tolerance,
or epsilon is introduced anywhere in this function; the only literal compared against
is `0.0`, the exact float identity a same-lane pair produces by construction. This is
**not** a general N-class Otsu extension — it recognises that this function's output is
always a single scalar cut applied uniformly to a sorted sequence, so the only question
that can ever change that cut's *placement* is "does a zero floor exist," and treats
positive magnitudes above it uniformly as signal once that upstream disambiguation
(anchor selection) has already run.

**Known scope limit, stated rather than hidden**: if a table's `_gap_cut_threshold` is
ever called on an anchor whose gap list contains *both* an exact-zero floor *and*
positive noise indistinguishable in magnitude from a real spacing (i.e., anchor
selection failed to find a clean anchor for that table), this branch has no mechanism
to separate them — it isolates zero and treats everything else as signal, full stop.
No fixture in this ticket's required sweep, nor in the wider suite, exercises that
shape; it is a residual limit of the anchor-selection precondition, not a gap in this
function's own logic, and is recorded here rather than left implicit.

## Verification beyond pytest

`native_rows_from_page` lane matrix, `~/venvs/socr/bin/python` heredoc probes,
synthetic fixtures matching the required sweep, checked before reporting:

| sweep | combinations | result |
|-------|--------------|--------|
| even spacing, widths 2-6 × offsets {207.3,210,230,250,268.9} × spacings {42,50,55.5,60,73} | 100 | **clean** |
| uneven spacing: (50,70,90), (40,90,55), (35,120,60), (80,45,95), (50,52,54) | 5 | **clean** |
| paired columns: within {30,35,40} × between {85,105,120} × {2,3} pairs, right-aligned | 18 | **clean** |
| mixed (one tight pair + two evenly-spaced singletons, right-aligned) | 3 | **clean** |
| right-aligned varying digit counts (1-4 digits, orders of magnitude, sign toggling) | 9 | **clean** |
| font sizes 6-14 | 9 | **clean** |

All 144 combinations produce the exact expected per-column lane assignment
(`(0,1,...,width-1)` for every row). Additionally re-ran
`tests/test_metric_corruption_battery.py` directly (not just via the full suite) to
confirm the battery fixture and its corrupting-transform variants are unaffected: 28
passed, 1 xfailed (the wrapped-label test, correctly still `strict=True` — TICKET-B5,
untouched).

## Test results

- `~/venvs/socr/bin/pytest tests/ -q` — **1395 passed, 2 xfailed** (1397 collected),
  0 failed. Before this fix (`52d26ea`): 1392 passed, 3 failed (the three paired-column
  parametrizations), 2 xfailed.
- Explicitly re-ran every protected/gate test by name: `test_a_perfect_transcription_of_a_regular_grid_scores_100`,
  `test_lane_assignment_is_invariant_to_page_offset`, `test_a_perfect_transcription_scores_100`,
  `test_dropping_a_column_never_beats_keeping_the_gap`,
  `test_padding_with_low_entropy_columns_never_beats_a_faithful_transcription`,
  `test_same_shift_plus_spurious_column_never_beats_the_incumbent`,
  `test_wrapped_label_is_scored_the_same_as_unwrapped`, and
  `test_paired_columns_do_not_collapse_into_one_lane` — **15 passed, 1 xfailed** (the
  wrapped-label test, correctly still xfail).
- `uvx ruff@0.16.0 format --check .` — `247 files already formatted`, clean.

## Files changed

- `src/socr/tables/native_rows.py` only:
  - `_assign_lanes` — added `_anchor_left`, three-way (right/centre/left) dispersion
    comparison with a cluster-count tie-break before anchor-preference order,
    docstring updated.
  - `_gap_cut_threshold` — added the unconditional zero-floor branch, docstring
    extended to explain both the paired-column fix and its scope limit.
- No test file touched.
