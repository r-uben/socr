# 2026-08-20 — #263: rotation refusal without a table

## What was wrong

`born_digital.py::_assess_page_signals` consulted `text_direction_is_rotated`
only inside `if has_tables:`. A rotated page with no detected table therefore
shipped its native layer untouched, and on a rotated *figure* page that layer is
character-level confetti under a clean `SUCCESS`.

Reference: Kaminska–Mumtaz–Sustek p38, 177 chars over 47 lines, 32 of them two
characters or fewer — `MC / O / F / round / a / ields / y / n / i / anges …`.

## The mechanism, read off the page's own geometry

`page.get_text("dict")` on p38 shows every fragment as its own PyMuPDF *line*,
`dir=(0.0, -1.0)`, all with x-extent `491.8 .. 503.7` (one column), y-extents
butting directly against each other (`94.5..113.7`, `113.7..122.8`,
`122.5..130.1`, …). PyMuPDF starts a new line whenever the next glyph run is not
ahead of the current one along the writing direction; this caption's runs are
placed behind each other, so it is cut at every run boundary and emitted in
y-order, i.e. reversed.

## The test, and why it is not a short-line count

`rotated_text_is_shredded(blocks, direction)` compares each line break against
the page's own type size: two lines are compared only when their extents
perpendicular to the text direction overlap (same baseline column); the break is
*spurious* when their separation along the text direction is non-negative and no
larger than the thinner line's own perpendicular extent — that line's glyph
height on this page. Verdict is `spurious > genuine`: two counts the page itself
produces, no threshold.

A fraction-of-short-lines test was measured and rejected. Over the 11 rotated
born-digital pages in the 27-paper reference corpus it cannot separate the defect
from a chart: Pflueger–Rinaldi p37 has 67 short lines out of 95 and is perfectly
sound (axis tick labels `0 5 10 15 20`; its caption reads back verbatim). The
geometric predicate fires on 1 of those 11 pages, and on exactly 1 page across
all 27 papers — p38.

## Remedy

`needs_ocr_enhancement = True` (the page routes to OCR), plus a fail-closed floor
in `manifest.py::_winning_page_output`: when nothing was accepted, ship
`[page N failed: rotated text extraction shredded — see image]` with a full-page
PNG ref rather than the fragments. Reversing and re-joining them is a repair this
does not attempt — a wrong reading is worse than a missing one.

Surfaced at page status (ERROR), document status, the `rotated_text_shredded`
audit event, the sidecar flag, and the CLI failure marker.

## Scope note

The check runs on the no-table branch only. That is not a re-narrowing: rotation
is now consulted on both branches, and on a rotated page *with* a table the
GH-147 branch already refuses unconditionally, which is strictly stronger. It is
also the honest limit of the predicate — a rotated table's cells legitimately
tile along the reading axis, the exact geometry the shred test reads as damage
(measured on `_rotated_ruled_grid_pdf`: spurious 9, genuine 5).
