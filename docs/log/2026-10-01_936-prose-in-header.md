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

## Existing gate fixtures (deviation)

First full run: 88 failures in `test_gh916_native_ship_gate`, `test_gh917_gate_direction_header`,
`test_gh917_text_in_numeric_column`. Those synthetic pages put each row on one text line and carry no
prose, so the column pitch is the only gap `_median_word_gap` can measure and every header row reads as
one run. A real page's word space comes from its body text (the 127-page census confirms the
predicate behaves on real pages). Giving every cell its own line broke the sign tests (they pair by
`(block, line)`). Fix: a `no_prose_in_header` fixture in `tests/native_table_fixtures.py`, applied
through `pytestmark` to those three modules; `prose_in_header` has its own file with a prose baseline.

## Suite

Full suite, default OLLAMA_HOST, one run: 6089 passed, 2 skipped, 4 xfailed, 1 failed
(`test_gh713_round2_credential_lifecycle::test_restored_credentialed_page_reassembles_byte_identically`,
`assert None is not None`). It passes alone (11 passed) and the file does not touch the ship gate;
treated as an order/time flake of the 75-minute run, not rerun in full. `uvx ruff@0.16.0 format
--check .` clean.
