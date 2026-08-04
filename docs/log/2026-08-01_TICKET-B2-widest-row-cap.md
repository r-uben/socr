# TICKET-B2 reopen (fourth fix) — best-first split, widest-row cap

Fixes the RED corpus gate pinned in STATUS.md's "Next action" (16/18 scorable pages
producing 2x-5x more lanes than the widest row has values) on top of `c78ccd1`. Scope:
`src/socr/tables/native_rows.py` only.

## Root cause (confirmed, then found insufficient to fix with the first design tried)

The team-lead's diagnosis was correct: `_gap_cut_threshold`'s zero-floor branch (from
the paired-columns fix, `docs/log/2026-08-01_TICKET-B2-paired-columns-fix.md`) isolates
an exact-zero gap and then treats *every* positive gap magnitude, however tiny, as a
lane boundary. Real pages have both a coincidental shared edge (so the branch fires) and
pervasive sub-point-to-~1.2pt rendering jitter (so every jitter gap becomes a spurious
lane) — exactly the scope limit that fix's own log disclosed and flagged as untested.

## First attempt at the fix (this session) — failed, kept honest here

Replaced the single global cut with recursive, row-support-validated bisection: find the
split maximising `_between_group_variance`, accept it only if both sides keep >=2
distinct rows (GH-113's existing rule), recurse depth-first into whichever side
validated, otherwise try the next-best candidate. This passed all 1396 non-corpus
tests cleanly but **made the corpus gate measurably worse** — e.g. page 13 went from a
buggy 24 lanes to 56 lanes.

Root cause of the regression, found by tracing page 13 token-by-token: **>=2-row
support is too weak a bar once a table has 20+ rows.** Almost any split leaves >=2 rows
on each side by pure chance — page 13's rows include a "Memo:" footnote block whose
column right edges sit ~1-1.2pt left of the main table body's, and with 23 rows in the
table, that kerning-scale offset alone produces enough coincidental 2-row groupings to
validate as "real" splits at every level of recursion. Depth-first recursion never runs
out of validated splits to make, so it fragments single logical columns into 10+
spurious micro-clusters.

## Second attempt — the one that shipped

Kept row-support validation (it correctly rejects a lone-value split — that part of the
prior design was right) but added the one bound the data itself implies:

**A page cannot have more real lanes than its widest row has values.** That count is
read directly off the token list already in hand (the largest number of tokens any
single row contributes), not supplied externally — it is the same `widest_row`
invariant `tests/test_corpus_rescore_gate.py` already uses as ground truth for "how many
lanes should exist." Not a tunable constant: it is a fact about this page's own rows,
computed fresh per anchor from the tokens `_cluster_by_anchor` is given.

Combined with **best-first** ordering instead of depth-first: at each step, every
currently-open cluster proposes its own best row-supported split (or none); the single
highest-scoring proposal across *all* open clusters is applied, and the process repeats
until either no cluster has a valid split left, or the cluster count reaches the
widest-row cap. This spends the bounded "budget" of splits on the largest, most
evidence-backed boundaries first: real 20-70pt column gaps score far higher on
`_between_group_variance` than any 1-1.2pt kerning-scale sub-grid offset, so the true
column boundaries are exhausted — and the cap is hit — before the search ever needs to
look inside the noise band.

No new constant, tolerance, or epsilon: the only literal in the new code is the
mechanical `>= 2` from the pre-existing GH-113 rule (unchanged, not introduced by this
fix) and the loop condition `len(clusters) < max_lanes`, where `max_lanes` is computed,
not chosen.

## Known scope limit (stated rather than hidden)

- The cap assumes the widest row on the page has no positionally-skipped column — the
  same assumption the corpus gate's own `widest_row` ground truth already makes.
- This does not solve duplicated page content. A page whose rows are themselves
  duplicated (the same row rendered twice, byte-identical) makes every noise-scale split
  look exactly as well-supported as a real one *and* does not raise the widest single
  row's value count, so the cap cannot separate real repeated structure from evidence
  manufactured by duplication. That is a content-identity question, not a position one,
  and out of this function's scope (page 55's near-duplicate-row rows were the leading
  suspect during debugging; the corpus measurement below shows page 55 is now correct
  at 6 lanes / 50.0%, so this limit was not actually exercised on this run — noted as
  residual risk, not a demonstrated failure).

## Corpus measurement (before -> after, real preserved OCR run)

| page | lanes before | lanes after | widest row | pct before | pct after |
|------|--------------|-------------|------------|------------|-----------|
| 13 | 24 | **6** | 6 | 69.6% | **100.0%** |
| 24 | 3 | 2 | 2 | 0.0% | 0.0% |
| 39 | 19 | **7** | 7 | 54.1% | **98.6%** |
| 45 | 8 | **7** | 7 | 89.8% | **98.0%** |
| 46 | 21 | **7** | 7 | 38.7% | **74.7%** |
| 48 | 15 | **7** | 7 | 70.3% | **100.0%** |
| 51 | 21 | **7** | 7 | 55.6% | **100.0%** |
| 53 | 8 | 4 | 4 | 0.0% | 0.0% |
| 55 | 30 | **6** | 6 | 13.6% | **50.0%** |
| 59 | 18 | **7** | 7 | 78.3% | **100.0%** |
| 60 | 12 | **6** | 6 | 63.8% | **100.0%** |
| 61 | 31 | **8** | 8 | 47.3% | **100.0%** |
| 62 | 27 | **6** | 6 | 50.6% | **100.0%** |
| 63 | 33 | **7** | 7 | 42.9% | **100.0%** |
| 64 | 28 | **7** | 7 | 45.2% | **100.0%** |
| 65 | 16 | **6** | 6 | 65.2% | **100.0%** |
| 66 | 28 | **7** | 7 | 46.3% | **100.0%** |
| 67 | 18 | **6** | 6 | 52.5% | **100.0%** |

Lane count now equals `widest_row` on **all 16 previously over-splitting pages**, and
pct rose to 100.0% on 12 of them. Pages 24 and 53 are not table pages in the meaningful
sense (widest_row 2 and 4, low base pct) and are unchanged in shape — lane count did not
grow, pct did not regress, exactly the gate's requirement. **No page still over-splits.**
No synthetic gate had to give ground: all seven protected tests
(`test_a_perfect_transcription_of_a_regular_grid_scores_100`,
`test_lane_assignment_is_invariant_to_page_offset`,
`test_paired_columns_do_not_collapse_into_one_lane`,
`test_a_perfect_transcription_scores_100`,
`test_dropping_a_column_never_beats_keeping_the_gap`,
`test_padding_with_low_entropy_columns_never_beats_a_faithful_transcription`,
`test_same_shift_plus_spurious_column_never_beats_the_incumbent`) still pass, and
`test_wrapped_label_is_scored_the_same_as_unwrapped` remains `xfail` (TICKET-B5,
untouched).

`tests/data/obr_efo_2022_11_baseline.json` re-recorded to the "after" column above —
strictly downward on every lane count, strictly upward or equal on every pct, per the
module's own re-recording rule.

## Test results

- `~/venvs/socr/bin/pytest tests/ -q` — **1398 passed, 2 xfailed**, 0 failed (before this
  fix, `c78ccd1`: 1398 passed, 2 xfailed, but with the corpus gate failing when the
  preserved run is present locally — the gate skips in the collected/CI count either
  way, so the pass count itself does not move; the failure surfaced only when the
  preserved corpus was available, which is the point of that gate).
- `uvx ruff@0.16.0 format --check .` — `249 files already formatted`, clean.

## Files changed

- `src/socr/tables/native_rows.py` — `_cluster_by_anchor` rewritten: dropped the
  recursive/depth-first row-support-only design in favour of best-first split ordering
  (`_split_bounded`, `_best_proposal`) bounded by a widest-row cap computed from the
  tokens themselves. `_split_validated` removed (superseded by `_best_proposal`).
  `_split_candidates`, `_row_support`, `_between_group_variance` (operating on raw
  anchor values, not gap magnitudes) kept from the same-session first attempt. Docstring
  in `_assign_lanes` updated to reference the new algorithm by name.
- `tests/data/obr_efo_2022_11_baseline.json` — re-recorded downward per the corpus
  measurement above.
- No test file touched.
