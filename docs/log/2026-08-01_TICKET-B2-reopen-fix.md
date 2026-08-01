# TICKET-B2 reopen fix — lane splitter collapses regular grids

Fixes the REJECT-level defect recorded in `docs/plans/metric-blind-spots/STATUS.md`
("Next action") on top of `27aa432` (TICKET-B2 attempt 2) and `5a961af` (the failing
gate test committed to pin the defect).

## Root cause

`_gap_cut_threshold` (`src/socr/tables/native_rows.py`) built its candidate list as
`sorted(g for g in gaps if g > 0)` — **positive-filtered but not deduplicated**. On a
regular, evenly-spaced grid every between-lane gap is the identical magnitude, so
that list is `[60, 60, 60, ...]`: every *consecutive* ratio reads `60/60 == 1.0`, no
ratio jump exists anywhere, and the caller's strict `gaps[i-1] > cut` (with
`cut == 60`) never fires. Every column collapsed into lane `0`.

The pre-existing `len(positive) == 1` special case (added when B2 first landed) was
meant to cover exactly this — "a single distinct within-lane-vs-between-lane
magnitude" — but it only triggered when there was literally **one raw gap value in
the list**, not one *distinct* value repeated many times. A dense table with more
than 2 rows almost always repeats its between-lane gap, so the branch was
unreachable on any real grid wider than 2 columns.

## Fix

One-line root cause, one-line fix: deduplicate before sorting —
`sorted({g for g in gaps if g > 0})` instead of `sorted(g for g in gaps if g > 0)`.
This makes the `len(positive) == 1` branch (unchanged) reachable whenever there is
**one distinct** lane-boundary magnitude, regardless of how many rows repeat it. The
now-dead `if positive[i] > 0 else inf` guard in the ratio comprehension was removed
along with it (no zeros can appear in a set built from `g > 0` in the first place —
that branch was already unreachable, harmless, but it read as if it were live).

## Alternatives tried and rejected

**Including zero gaps directly in the elbow search** (i.e. treating the literal
"tokens share an anchor exactly" gap as a first-class data point and looking for the
largest ratio jump across the *whole* sorted gap list, zeros included) was the first
attempt, following the STATUS.md hint that "zero is the within-lane magnitude."
It fixes the regular-grid collapse, but **breaks a real, already-passing case**:
`tests/test_metric_corruption_battery.py`'s base fixture uses **left-aligned**
insertion with digit-count-dependent right edges (`"42.8"` vs `"3.8"` right-edge
gap = 5.0pt), producing a *non-zero* but genuinely within-lane noise floor. Because a
transition **away from exact zero is always an infinite ratio**, mathematically, zero
inclusion **always** wins the "largest ratio jump" search regardless of whether the
next value up is real between-lane spacing or itself still noise — it picked the
0→5pt jump as the elbow (cut ≈ 2.5) instead of the real 5pt→55pt jump (cut ≈ 30),
splitting each lane into two on nothing but digit-width jitter. Reproduced directly:
`test_base_fixture_is_scorable`, two `test_corrupting_transform_makes_score_strictly_worse`
cases, and three others regressed (6 failures) before this was caught and reverted in
favour of the dedup fix, which passes the same fixture unchanged.

## Verification — lane table, every geometry in the ticket

Measured directly against `native_rows_from_page` (`~/venvs/socr/bin/python`
heredoc probes; synthetic fixtures matching the ones in
`tests/test_metric_corruption_battery.py`), before (`5a961af`, unpatched) and after:

| geometry | before | after |
|----------|--------|-------|
| 2 cols, evenly spaced | `(0,1)` | `(0,1)` — unchanged, already correct |
| 3 cols, evenly spaced | `(0,0,0)` | `(0,1,2)` — **fixed** |
| 4 cols, evenly spaced | `(0,0,0,0)` | `(0,1,2,3)` — **fixed** |
| 5 cols, evenly spaced | `(0,0,0,0,0)` | `(0,1,2,3,4)` — **fixed** |
| 4 cols, uneven (50/70/90pt gaps) | `(0,0,1,2)` | `(0,0,1,2)` — **unchanged, still partial** |
| 4 cols, very uneven (50/80/120pt gaps) | `(0,0,1,2)` | `(0,0,1,2)` — **unchanged, still partial** |
| battery fixture (left-aligned, digit-width noise, 2 real cols) | `(0,1)` | `(0,1)` — unchanged, already correct |

The uneven cases are a **pre-existing, separate limitation, not something this
change touches in either direction** — the "before" column above was measured
against the unpatched `5a961af` tree, not against some earlier working state; both
before and after produce the identical partial split. Root cause: the ratio-elbow
rule finds the *single* largest jump in the sorted distinct-gap list, but an uneven
grid with **no true noise floor at all** (every within-lane gap is exact zero, no
digit-width jitter in this fixture) has *multiple* distinct between-lane magnitudes
(50, 70, 90) that are all legitimate lane boundaries — the algorithm has no signal in
this shape to distinguish "the smallest one is noise" from "all three are signal,"
since ratios among 50/70/90 are mild (1.4, 1.29) and comparably sized to each other.
Distinguishing that case from the battery fixture's real noise-vs-signal split (ratio
11, one dominant jump) requires more than a single global max-ratio search; every
formulation explored during this fix that tried to fold zero into the elbow search to
help the uneven case either regressed the battery fixture (see above) or required an
additional magic constant to decide "is this ratio big enough to be a real elbow,"
which the ticket's constraint rules out. **Flagging this rather than shipping a
speculative extension**: no test in the suite exercises the uneven case (only the
regular-grid parametrization is gated), so it is not a regression from this change,
but it is a real, open gap in lane assignment on non-regular column spacing that a
future ticket should address with a different technique (e.g. per-lane dispersion
comparison across candidate cluster counts, not a single global gap threshold).

## Test results

- `~/venvs/socr/bin/pytest tests/ -q` — **1388 passed, 2 xfailed** (1390 collected),
  up from `5a961af`'s pinned-red baseline of 1385 passed / 3 failed / 1 xfailed
  (1389 collected — the extra collected test is the width-2 case of the newly
  parametrized `test_a_perfect_transcription_of_a_regular_grid_scores_100`, which
  was already passing and is now counted alongside its 3/4/5 siblings).
- `uvx ruff@0.16.0 format --check .` — `245 files already formatted`, clean.

## Files changed

- `src/socr/tables/native_rows.py` — `_gap_cut_threshold` only (dedup fix +
  docstring rewrite explaining why dedup, not zero-inclusion).
