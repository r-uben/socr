# GH-936: `prose_in_header`, a DEFER-only ship-gate predicate (2026-10-01)

Branch `fix/936-prose-in-header` from origin/main 0567718 (#935; ancestry verified). Design and
measurement: `~/.local/state/socr-housekeeping/gh936/design.md` (strict form, GO).

## Change

`src/socr/tables/ship_gate.py`: `prose_in_header_faults`, constant `PROSE_IN_HEADER`, one
`faults += ...` line in `native_ship_gate`. Per block with geometry, header rows = output rows above
the first core paired row. A source row above the first core row is *carried* when every one of its
words is a token of those header rows (counted, NFKC). Fires when a carried row is ONE run of 2+ words
(no gap over `ALIGNED_RUN_GAP_MAX_WORD_SPACES` x `reconstruct._median_word_gap(words)`; the gate is
handed upright-frame words on rotated pages). No new constant, no font-size clause (design: zero
census gain, brittle float comparison), no panel-row exemption (would save only Stock-Watson 44).
Abstains when the page has no measurable word space.

Known holes (design, by construction): a one-word caption, and a caption with a gap over the bound.

## Resume / audit

No new kind. `SHIP_GATE_KIND` (`native_ship_gate_deferred`) is emitted from
`_plan_native_table_first` and carries the faults generically; the orchestrator's
resume-exemption table keys on the kind, and no source outside `ship_gate.py` names a predicate.

## Measurement (frozen inputs.pkl, 127 pages; main = origin/main 0567718 via `git archive`, branch = worktree)

- Main's predicate sets equal the design's `pr935.json` on all 127 pages (cross-check).
- Branch minus `prose_in_header` equals main on all 127 pages.
- `prose_in_header` fires on 31 pages. Predicate-set changes: 31, every one is the added
  `prose_in_header`; **0 removals** (asserted).
- SHIP to DEFER on 11 pages: 7 upright census SHIPs (below) and 4 Fama lift pages (368, 562, 782, 792)
  that already carry `action=defer` from the rowizer, so the gate does not decide them.
- The 7: Fama 733, Herskovic 29, Mendoza-Fernandez 60 (real, renders viewed this session: caption
  plus notes spread over header cells; spanning heading fused with values); Fama 728 (broken page,
  deferrable); Binsbergen 57, Kim-Muhn-Nikolaev 52, Stock-Watson 44 (false: a one-run spanning
  heading, a panel title, an in-table panel label; renders from the design). 3 false DEFERs cost 3 reads.

## Tests (`tests/test_gh936_prose_in_header.py`, 12)

Synthetic difference pins (grid with vs without the caption; plan SHIP with the gate off, DEFER with
it on; same caption laid out over lanes does not fire; gap at the bound fires, just over does not;
two-word row fires, one-word does not) and must-not-fire controls (clean grid, caption dropped by the
grid, partly carried, counted words, a source run below the first data row, no measurable word space).

Mutations in an external copy (src + tests + pyproject, plus a `socr.__file__` canary; each anchor
asserted uncapped `count == 1`; baseline control passes): M1 `len < 2` to `< 1`, M2 all to any, M3
counted to presence, M4 K unbounded, M5 K halved, M6 wiring dropped, M7 rows at/below the first data
row scanned, M8 run test inverted. All 8 killed. M3 and M7 first survived and prompted two tests
(a word printed twice, a run below the data).

## Round 2 (Astra, PR #944): spacing-evidence policy

Round 1 took the page's median same-line gap from every word on the page. That is wrong twice over.
A PDF that prints whole table rows as one line, or a table-only page, makes the column pitch the
"word space", so a real multi-column header reads as one run. And the gh916/gh917 synthetic grids
(no prose) fired on 88 tests, which round 1 hid behind a blanket opt-out fixture. Round 1's claim
that a real page gets its word space from body prose was an assumption, not a measurement. It is
removed.

Policy (`_page_word_space`, a helper any predicate can use for the page's ordinary word space):
- the median same-line gap is measured ONLY on text lines outside every table's vertical extent
  (core rows less/plus the outward reach of `_PANEL_GAP_ROWS` pitches, the reach `data_row_missing`
  and `header_band_missing` already use);
- ABSTAIN unless at least `_MIN_SPACING_LINES` such lines carry a gap. That constant is
  `_PLACEHOLDER_MIN_ROWS` (2): a repeat is evidence, one line is not. No new empirical constant;
- words without block/line indices (5-tuples) are skipped, so the predicate abstains and never turns
  a page into `gate_error` (round 1 had replaced `direction_unavailable` with `gate_error` there).

The blanket `no_prose_in_header` fixture is deleted and no test opts out of the predicate. The
gh916/gh917 modules pass UNMOCKED (162 passed with `test_gh936`), because those grids have no text
outside the table.

### Measured exposure (frozen inputs.pkl, 127 pages; trees extracted from the commits)

- Spacing evidence on 65 pages; 59 abstain for lack of it; 3 have no table geometry. The abstain
  rate is large: most of the census is table-heavy extracts or the Fama appendix.
- `prose_in_header` fires on 16 pages (round 1: 31). Every other predicate's set is identical to
  main on all 127 pages, so 0 removals, and identical to main d8dc9b1 (#942's
  `header_band_missing` included).
- SHIP to DEFER, same 6 against main 0567718 and d8dc9b1 (round 1: 11):
  - real, kept: Fama 733, Herskovic 29, Mendoza-Fernandez 60;
  - deferrable, kept: Fama 728;
  - false, kept: Kim-Muhn-Nikolaev 52, Stock-Watson 44.
- Lost versus round 1: Binsbergen 57 (false); Fama lift pages 368, 562, 782, 792 (already
  `action=defer`, not gate decisions); ten pages that already DEFER on another predicate (Fama
  46/427/561/570/590/592/780, Bybee 67, Theodoridis 371/1203). All lost for lack of evidence.
- The policy keeps the three real catches and drops one false DEFER. The exposure it accepts: a
  defective header on a page with fewer than two outside-table text lines ships, as it did before
  this ticket.

### #942 composition

Rebased onto d8dc9b1. #942's `header_band_missing` is untouched: it keeps its all-words
`_median_word_gap` (the first commit had dropped that import in a silent merge; restored). Its census
set is identical. Moving its run clause to `_page_word_space` would stop whole-row-line pages from
over-splitting there, but that changes #942's behaviour and is not done here.

### Tests (`tests/test_gh936_prose_in_header.py`, 17)

Round 1's 12 plus `TestSpacingEvidence`: a table-only page with a caption run abstains, and the same
page with prose fires; whole-row-line layout is quiet with and without a caption, and is judged by the
outside prose when there is some; one prose line is not evidence and two are; 5-field words abstain
without `gate_error`. Rotated integration case: NOT added. The existing rotated native-table tests in
gh916 run unmocked and pass (table-only words, so the abstain path), but no test drives a rotated
page with prose plus a header caption through `plan_native_table`.

Mutations (external copy of src, tests, pyproject, `socr.__file__` canary, uncapped anchor
`count == 1`, clean baseline): round 1's eight plus M9 no evidence floor, M10 in-table lines count as
evidence, M11 five-field words not skipped. All 11 killed.

## Round 3 (Astra P2): the yardstick must not be calibrated by header rows

Two carried header lines above the excluded extent could each supply their own gap g. The median
was g, and a near header row with the same wide gap passed g <= 2g (circular). Raising the line floor
does not fix that. Fix: a line whose every word is a grid token (NFKC, any block) is excluded from
the calibration, as is everything inside the extent; with no independent line left the predicate
abstains (`_page_word_space(words, extents, carried)`).

Tests (20 in the file): two wide-gap header lines outside the extent and no prose abstain (killed by
M12, carried lines calibrate); the same layout plus two prose lines uses the prose spacing (quiet for
the wide near row, fires for a tight one); two uncarried note lines inside the extent are not evidence
(killed by M10, which stopped dying once the carried rule landed, hence the new test). 13 mutants
now, all killed.

Census, same 127 pages: fires on 15 pages (was 16), 59 abstain, 0 removals, every other predicate's
set identical to main d8dc9b1, SHIP to DEFER the same 6 (real Fama 733, Herskovic 29, Mendoza 60;
deferrable Fama 728; false Kim 52, Stock-Watson 44). The page that stopped firing already DEFERs on
another predicate. The docstring's "+7 / 3 false" is corrected to the measured 6 / 2.

## Suite

Full suite, default OLLAMA_HOST, nohup, one run on the rebased head: 6111 passed, 2 skipped,
4 xfailed, 0 failed. Round 1's single failure (`test_gh713_round2...`) did not recur.
`uvx ruff@0.16.0 format --check .` clean.

Full suite, default OLLAMA_HOST, nohup, one run on the round-3 head: 6114 passed, 2 skipped, 4 xfailed, 0 failed. Ruff format check clean.
