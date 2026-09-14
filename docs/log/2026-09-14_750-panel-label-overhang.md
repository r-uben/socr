# #750: a heading overhanging the top tick left every panel unlabelled

## What changed

- `src/socr/figures/chart_reader.py` (`_panel_label`): the vertical
  "sits above the highest tick" test now reads the candidate row's CENTRE
  (`row.cy`), not its bottom edge (`row.y1`). Comparing against the edge
  worked everywhere the axis happened to be drawn a little further down the
  page (`row.y1 - top_tick` ranged -0.34pt to -9.38pt across the other 21
  corpus releases), but PyMuPDF's word bbox carries the font's own
  ascent/descent padding below the glyph ink, and on `sep-20250319-p09` /
  `sep-20250618-p09` that padding alone pushes the bottom edge 0.45pt past
  the tick even though the glyphs — and the row's centre — sit clearly above
  it. Comparing centres removes the font-padding noise instead of adding a
  tolerance constant to absorb it: no threshold is introduced.
- `read_chart_page`: a panel whose heading cannot be resolved (`_panel_label`
  returns `""`) is now recorded as a **refusal** naming why, instead of being
  published as a `PanelReading` with `label=""`. Previously an unlabelled
  panel's readings still flowed through to every downstream consumer with no
  identity to attach to — indistinguishable from an absent reading at every
  surface that mattered, and (see Findings) silently corrupting a sibling
  panel's data in one specific downstream consumer.
- `tests/test_gh750_panel_label_overhang.py` (new, 6 tests): two geometry
  pins built from `WordRow`/`Frame` values measured directly off
  `sep-20250319-p09` region 1 (not invented) — the heading-row overhang
  itself, and a negative control (a legend row, horizontally inside the
  frame just like a heading, whose centre sits well below the tick and must
  stay excluded); one pin for the `shared`-set finding below; the two real
  corpus pages (skipped when the corpus is absent) as the authoritative
  check; and one synthetic-page test for the refusal path, which no real
  corpus page exercises.

## Findings that corrected the brief

The brief and the issue both diagnosed this as the **horizontal** x0/x1
containment check (`frame.x0 <= row.x0 and row.x1 <= frame.x1`) rejecting a
heading that "starts a few points to the left of the frame". Measuring the
actual geometry on both affected pages shows this is wrong: the heading's
`x0`/`x1` are comfortably inside the frame on every panel of both documents
(e.g. `'2025'` at x0=115.22 against frame x0=107.62/x1=502.60). What fails is
the **vertical** test, `row.y1 > top_tick` — the row's padded bbox bottom
edge sits 0.45pt below the top tick, a rendering-padding artifact unrelated
to horizontal position. The horizontal containment check was never the
failure mode on these two releases, so the fix does not touch it.

This also answers the orchestrator's `shared`-set question directly, with a
measurement rather than a guess: across all 23 corpus documents, removing the
horizontal containment check entirely (with the *old* vertical check still
in place) never changed a single resolved label. `shared` — the set of rows
every panel of a figure draws identically — already excludes the one axis
title case the containment check's docstring names, on every document this
corpus has. The containment check is not dead code, though: it is the only
thing that would stop a single-panel chart (where `shared` is empty by
construction, since nothing repeats across panels) from picking up its own
axis title as a heading. This corpus has none, so that case is untested here
and the check is left in place, undisturbed.

## Downstream effect measured (not predicted)

`socr-score-sep-ground-truth` against the 23-document SEP corpus,
`main@374fb93` baseline vs. this fix:

| | baseline | after |
|---|---|---|
| reader cells | 2271 | 2391 |
| exact | 2221 | 2381 |
| no_ground_truth | 50 | 10 |
| wrong_count / wrong_bin / fabricated / missing | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |

The residual 10 `no_ground_truth` are `sep-20250917-p09`'s `2028` panel, a
column the Fed's own published table does not carry (out of scope here).

The brief predicted `no_ground_truth` 50→10 and `exact` 2221→2261 (total
cells unchanged at 2271). `no_ground_truth` landed exactly as predicted;
`exact` landed 120 cells higher than predicted, because total scored cells
rose from 2271 to 2391. Root cause, found while reconciling the numbers: the
harness's `_reader_readings` keys reader panels by `panel.label` in a plain
dict (`out.setdefault(panel.label, {})[series.name] = counts`). With all four
panels on a broken page sharing `label=""`, three of the four panels'
readings were being silently **overwritten**, not merely miscounted — only
one panel's worth of data (the 40 cells the brief cites) ever reached
`no_ground_truth`; the other ~120 cells were dropped from the measurement
entirely, neither counted nor visible as a defect. Refusing an unlabelled
panel instead of publishing `label=""` closes this collision as a side
effect (a refused panel never reaches `reading.panels`, so it cannot collide
under an empty-string key), but the harness's dict-keyed grouping remains a
latent trap for any other reader defect that produces two same-labelled
panels — out of this ticket's `Write ownership` (`chart_reader.py` +
tests), flagged here rather than fixed silently.

No document outside the two affected releases moved: verified per-document,
not just in aggregate — all 21 unaffected documents score identically to the
baseline (checked with `_score_side` called directly, not just diffed
against the aggregate total).

## Verification

- `uvx ruff@0.16.0 format --check .` — clean (whole repo).
- `PYTHONPATH=$PWD/src ~/venvs/socr/bin/pytest -q` — 5366 passed, 4 xfailed,
  0 failed.
- Each of the two guards (the `cy` comparison, the refusal branch) was
  independently mutated in a scratch copy outside this tree and shown to
  fail exactly the test(s) that target it, with every other new test still
  green (load-bearing evidence, not just "the suite is green").
