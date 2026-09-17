# 2026-09-17 — GH-752: panel key collision in the SEP ground-truth scorer

## What was wrong

`src/socr/figures/score_sep_ground_truth.py`'s `_reader_readings` and
`_model_readings` each built a `dict[str, ...]` keyed by panel label
(`panel.label` on the reader side, a Markdown heading's text on the model
side). A panel label is corpus content, not an identity guarantee: two panels
sharing a label (a layout change, a genuinely repeated heading, a new corpus
with duplicate years) silently overwrote each other via `dict.setdefault`.
Measured on a real release: all four panels of one page collapsed onto one
key, three were overwritten, and 120 cells vanished from the report entirely
— not even as `no_ground_truth`. The reported denominator looked complete
and was wrong.

Two sites had this shape:
- `_reader_readings` (was line 167): `out.setdefault(panel.label, {})[series.name] = counts`
- `_model_readings` (was line 210): `out.setdefault(panel_label, {}).setdefault(header, {})[bin_key] = value`

`_score_side` consumed both via `.items()`, so the loss was already baked in
by the time scoring ran — no fix at the `_score_side` level alone could
recover it.

## Fix

Changed the return type of both `_reader_readings` and `_model_readings` from
`dict[str, dict[str, dict[str, int]]]` (keyed by label) to `PanelReadings =
list[tuple[str, dict[str, dict[str, int]]]]` — one entry per panel/grid
found, in discovery order, never merged across a shared label. `_score_side`
now iterates the list directly instead of `.items()`. This is the "key by
something genuinely unique" branch from the ticket (position in the list, not
the label, is the identity) rather than the "detect and refuse" branch:
nothing is ever dropped, so there is no loss to detect. Two panels that
genuinely share a label still both score against the shared ground-truth
panel (`ReleaseTable.panel` is itself keyed by label) — that is correct: no
cell that either side actually read is discarded.

Both sites in the ticket were in scope and both were changed together; a
fix to only one would still let the other corrupt its side's denominator.

## Files changed

- `src/socr/figures/score_sep_ground_truth.py` — `PanelReadings` type alias;
  `_reader_readings`, `_model_readings`, `_score_side` changed from dict to
  list.
- `tests/test_gh747_sep_ground_truth.py` — two existing `_model_readings`
  tests updated for the new list return shape (no behavioural change to
  those tests, same assertions against the new shape).
- `tests/test_gh752_panel_key_collision.py` — new. Three tests, synthetic
  fixtures only (no corpus content):
  - `_reader_readings` keeps two `PanelReading`s that share a label (built
    directly via `chart_reader` dataclasses + monkeypatched `open_pdf` /
    `chart_region_bboxes` / `read_chart_page`, so it needs no real PDF).
  - `_model_readings` keeps two grids under the same Markdown heading.
  - `_score_side` pins the DIFFERENCE: the same two-panel input scored once
    with distinct labels and once with colliding labels produces the same
    cell count (2) either way — the collision costs nothing instead of
    losing a panel.

## Test results (measured, this session)

- `tests/test_gh752_panel_key_collision.py` + `tests/test_gh747_sep_ground_truth.py`:
  17 passed.
- Full suite (`PYTHONPATH=/tmp/wt-752/src ~/venvs/socr/bin/pytest -q`):
  5603 passed, 4 xfailed — no regressions from the baseline (issue #752's
  worktree started from `main@87df2ab`; this ticket did not touch any other
  file, so no separate pre-change baseline run was needed to know the delta
  is exactly this commit).
- Lint: `uvx ruff@0.16.0 format --check .` from the repo root — 727 files
  already formatted, clean.

## Mutation proof (three guards, each reverted independently)

Mutant copy at `/tmp/wt-752-mutant` (src + tests + pyproject.toml, so
`pythonpath = ["src"]` in the copied `pyproject.toml` cannot shadow the
external `PYTHONPATH`); a canary test
(`tests/test_canary_752.py::test_socr_resolves_to_the_mutant_copy_not_the_worktree`)
compares `os.path.realpath(socr.__file__)` against the mutant tree's own
realpath (`/tmp` symlinks to `/private/tmp` on this machine, so a bare prefix
check would false-fail on correct mutant source) — passed on every round,
confirming the mutant's own source was under test.

1. **`_reader_readings` reverted** to `dict.setdefault(panel.label, {})[...]`
   (the original site): `test_reader_readings_keeps_two_panels_that_share_a_label`
   redenned — `assert ['2020'] == ['2020', '2020']` (1 failed, 3 passed).
2. **`_model_readings` reverted** to nested `dict.setdefault` (the original
   site): `test_model_readings_keeps_two_grids_under_the_same_heading`
   redenned — same shape, `assert ['2020'] == ['2020', '2020']` (1 failed,
   17 passed; the other #752 tests build their list fixtures directly and
   don't go through `_model_readings`, so they stayed green as expected).
3. **`_score_side` reverted** to `dict(side_readings).items()` (re-collapsing
   the list back into a dict before iterating, proving `_score_side` itself
   — not just its two callers — is part of the fix): `test_score_side_scores_every_panel_even_when_labels_collide`
   redenned — `assert 1 == 2` on `len(colliding_scores)` (the second panel's
   cell vanished exactly as #752 described) (1 failed, 17 passed).

Mutant copy deleted after the third round; the real worktree was never
touched by any mutation.

## Does this change what the harness reports on the corpus?

Not measured — this ticket's own scope excludes re-scoring the corpus (see
`TICKET-752.md`, "Out of scope"), and I did not run
`socr-score-sep-ground-truth` against the real corpus. What is known from the
fix's shape: on any corpus page where every panel already has a distinct
label — every page checked in #747/#750's prior work — this change is
behaviourally inert; `list(dict.items())` and the new list-of-tuples iterate
the same (label, by_series) pairs in the same order, so `_score_side`
produces byte-identical `CellScore`s. It changes behaviour ONLY on a page
where two panels or two model grids genuinely share a label — a shape #750's
refusal path already prevents for the specific "empty/unidentifiable label"
case, and the wider corpus (23-24 real releases with year+"Longer run"
panels) is not known to contain a genuine same-label collision. If one exists
undetected in the current corpus, this fix would surface cells the old
scorer was silently discarding, which would only ever raise `documents_scored`
cell counts and could turn previously-invisible mistakes visible (e.g. a
duplicate panel scoring `wrong_count` against the shared truth panel) — never
the reverse. No number is asserted here because none was measured.

## Where I disagreed with the framing — nowhere

The dispatch's framing was accurate on inspection: both named sites are real,
both use the same collision shape, and `reading.panels` itself is keyed by
an already-unique `region_index` (`dict[int, PanelReading]`) — the bug was
entirely in `_reader_readings` throwing that identity away and re-keying by
`panel.label`, which confirms the "key by something genuinely unique" fix is
the natural one rather than a workaround. No disagreement to report.
