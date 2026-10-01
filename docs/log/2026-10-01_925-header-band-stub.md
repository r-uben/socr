# GH-925: header band whose first word is the row stub (WIP, CONSILIUM-GATE)

Status: NOT READY TO MERGE. The census shows the stub exemption widens into caption/prose absorption (#921 territory).

## Change

`src/socr/tables/reconstruct.py`
- `_is_label_region_word(word, lane_centers)`: the single left-of-first-lane predicate (`x < lane_centers[0] - snap radius`, not a
  folded margin note). `_rowize_segment` now uses it for the three label tests it used to inline.
- `_header_row_eligible(row_ws, lane_centers, snaps, allow_label)`: header rows may contain label-region words, but need at least one
  word that snaps to a lane (a caption lying wholly in the label region is not absorbed). Second revision: label words are allowed
  only on the row nearest the data (`allow_label` is true for the first row of the upward walk).
- `_prepend_header_band` and `_extend_scope_for_header` both use it; absorbed label words go to the header row's label cell.

`tests/test_gh925_header_band_stub.py`: 8 tests (stub-first absorbed, no-stub unchanged, data rows identical, difference pin against
the old rule, caption not absorbed in two shapes, label-only row not absorbed, source canary). All pass.

## Census (main 0eb6121 vs branch, frozen src copies, `socr.__file__` asserted)

127 unique pages (35 rotated q917, 92 upright; the 13 lift pages are all inside the q917 set). Rotated via
`attempt_rotated_native_table`, upright via `_phase_analyze` native_text. Artefacts: `~/.local/state/socr-housekeeping/gh925/`
(`changed/` with index.json, png, before/after md; scripts `census.py`, `compare.py`).

- Revision 1 (label words allowed on any absorbed row): 16 pages change grid; 0 verdict-only changes.
  `header_band_missing` fires 18 -> 15. Gate verdict changes: Fama 561 loses `header_band_missing`; upright Ayivodji p43 and
  Gong p53 lose it; two pages move defer -> refuse in `plan.action`.
- Numerics: the multiset of ALL numeric tokens is identical before/after on all 16 pages (nothing added, nothing removed). A
  "data rows only" multiset flags 10 pages only because adding a header row shifts which row is the first row; not a number move.
- Revision 2 (label words only on the nearest row): 4 of the 16 grids differ from revision 1 (Boukus p38, Fama 728, Ayivodji 43,
  Beckmann 79). The other 12 are unchanged. Not enough.

## Finding: the rule widens

Of the 16 changed pages only a few are the intended recovery (Fama 561 gets its `Country | rho1..` header band; Ayivodji 43 and
Gong 53 clear `header_band_missing`). The rest absorb captions and footnote prose into the grid as header rows, e.g. Fama 728
(a Table 3 footnote paragraph), Fama 733, Bybee 78 (`Table M.1: ...` caption), Hansen 28, Perico-Ortiz 41 (`Table B.5: ...`),
Beckmann 79. Cause: on dense-lane pages the lane-snap test is nearly vacuous, so a prose or caption line passes whenever a
single word falls near a lane and the rest sit left of the first lane. The old rule rejected these because the left-of-lane words
did not snap. No numbers move, but text that was prose is now table cells.

## Decision needed

Is a stub-first header distinguishable from a caption or prose line by geometry alone? Options:
1. Require lane-shaped structure on the row (one word per lane, words within the lane span) plus the data-derived x-range.
2. Restrict to the segment-adjacent row AND require a mechanical caption test shared with #921 (shared work with that ticket).
3. Drop the rowizer change; keep the gate net (`header_band_missing` DEFER) and fix Fama 561 only via its own header geometry.

## Not done

Full suite (one run was started detached and was still running when this was written; it began before revision 2, so it is not
a valid run), mutation runs, whole-repo ruff format check beyond the touched files.

## Outcome: measured and REJECTED (2026-10-01)

The WIP is kept unmerged on `fix/925-header-band-stub` (9aa2c40). Do not re-implement it.

- **Net loss under the cost model.** Three pages lose a DEFER: Fama 561, Ayivodji 43 and Gong 53. Only Gong 53 is a clean
  recovery; the other two also absorb a title or stray glyphs.
- **New silent ships.** On at least 9 pages a footnote paragraph, caption or panel title becomes header rows, and the gate
  predicates are byte-identical before and after. These are pages that shipped a correct header on main; nothing DEFERs them:
  Fama 728, 733, 753; Phillips-Zhdanov 52; Lopez-Lira 51; Bybee 78; Hansen 28; Beckmann 79; Perico-Ortiz 41;
  Eskildsen 70; De Fiore 26.
- **Why tuning cannot fix it.**
  - A lane-shape test fails: a 5-word caption over a 5-lane table is lane-shaped, while real header cells ("Std. Dev.",
    spanning headings) are not one word per lane.
  - The "nearest row only" rule (revision 2) fails because the band is built at two sites, `_extend_scope_for_header` and
    then `_prepend_header_band`. The second pass sees the caption as its own nearest row; Bybee 78 lands the caption in the
    label cell.
- **Status quo is safe.** On main a dropped stub-first header is caught by the gate's `header_band_missing`, so the page goes
  to a model read.
- **Prerequisite for any retry:** a line-level prose/caption predicate (sentence punctuation, full-width span, no alignment
  with the data rows below). Use the 9 pages above plus #921's pages as its fixture. #921 consumes the same predicate. A
  retry of #925 must build the band at a single site.

Census artefacts: `~/.local/state/socr-housekeeping/gh925/` (`changed/index.json`, before/after markdown, renders).
Consultant: Fable.
