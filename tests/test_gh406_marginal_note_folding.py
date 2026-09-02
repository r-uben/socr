"""GH-406: a rotated marginal note must not erase the table.

Measured on the synthetic fixture: one 19-character note set sideways in the
margin took the page from ONE reconstructed region to ZERO. The whole table
vanished with no signal -- the native path simply reports no table.

The mechanism was isolated by experiment, not inferred:

- the page's dominant direction is NOT flipped, so it is not an orientation bug
- the note contributes single-word y-bands that interleave with the table rows,
  halving the median row gap
- removing the note words restores the table, AND so does keeping them but
  snapping them onto existing rows -- so the extra BANDS are the cause

Hence `_fold_marginal_bands`: keep the words, attach them to the nearest real
row. Dropping them would trade a table for a silent word loss (#418).
"""

from __future__ import annotations

import pytest

fitz = pytest.importorskip("fitz")

from socr.tables.reconstruct import rowize_from_word_list  # noqa: E402

NOTE = "Running Header 2026"
# GH-459: the note above contains a NUMERIC token (`2026`), which seeds a lane at
# its own x and so let the alphabetic words snap to it. That green-washed the
# keep test below: an alphabetic-only note at the same position still vanished.
# Every keep assertion is now parametrised over both.
ALPHA_NOTE = "Running Header"


def _page(with_note: bool, note: str = NOTE):
    """The ticket's own reproduction: a 4-column table plus a notes line.

    A hand-rolled word list does NOT reproduce this -- the first version of
    these tests used one and passed with the fix removed. The defect depends on
    the real fixture's row pitch and the note's word spacing, so the fixture has
    to be the measured one.
    """
    doc = fitz.open()
    page = doc.new_page(width=600, height=800)
    data = [
        ["Model", "Beta", "SE", "t-stat"],
        ["OLS", "1.25", "0.05", "25.0"],
        ["IV", "1.80", "0.12", "15.0"],
        ["GMM", "1.45", "0.08", "18.1"],
    ]
    cols = [100, 220, 340, 460]
    y = 100
    for row in data:
        for c, cell in enumerate(row):
            page.insert_text((cols[c], y), cell, fontsize=10)
        y += 25
    page.insert_text((100, y + 20), "Note: Standard errors clustered, N = 500.", fontsize=9)
    if with_note:
        page.insert_text((550, 200), note, fontsize=8, rotate=90)
    return doc, page


def _regions(with_note: bool, note: str = NOTE):
    doc, page = _page(with_note, note)
    try:
        return rowize_from_word_list(list(page.get_text("words")))
    finally:
        doc.close()


def test_a_marginal_note_does_not_erase_the_table() -> None:
    """The measured defect: 1 region without the note, 0 with it."""
    without = _regions(False)
    with_note = _regions(True)

    assert without, "fixture must reconstruct without the note, or it pins nothing"
    assert with_note, "the marginal note erased the table"
    assert len(with_note) == len(without)


@pytest.mark.parametrize("note", [NOTE, ALPHA_NOTE], ids=["with-numeric", "alphabetic-only"])
def test_the_notes_words_are_kept_not_dropped(note: str) -> None:
    """The fix must not become the loss it prevents (#418).

    GH-459: parametrised because the original note carries `2026`. That numeric
    token seeds a lane at the note's x, so the alphabetic words snapped to it
    and this test passed for the wrong reason -- an alphabetic-only note at the
    same position was still dropped at lane assignment. The `alphabetic-only`
    case is the one that fails without the folded-marginal keep path.
    """
    emitted = " ".join(md for _rect, md in _regions(True, note))
    tokens = emitted.replace("|", " ").split()
    for word in note.split():
        assert word in tokens, f"{word!r} was dropped instead of folded: {emitted}"
        assert tokens.count(word) >= 1


def test_the_table_itself_is_unchanged_by_the_note() -> None:
    """Scope: EVERY table cell, stub included, is untouched by the note.

    GH-461: the previous version of this test compared `row[1:]` only, and so
    encoded a real defect -- #459 put the note's words in the LABEL cell, which
    made row stubs read `| OLS 2026 |`. That is silent corruption of row
    identity, the same class of wrongness as attaching the note to a value.

    The comparison now includes column 0. The note lives in a column of its own,
    so it can be kept without claiming to be either a stub or a datum.
    """

    def cells(md: str) -> list[list[str]]:
        return [
            [c.strip() for c in line.strip().strip("|").split("|")]
            for line in md.splitlines()
            if line.lstrip().startswith("|") and "---" not in line
        ]

    without = cells(_regions(False)[0][1])
    with_note = cells(_regions(True)[0][1])

    assert len(with_note) == len(without), (
        f"the note changed the ROW count: {len(with_note)} vs {len(without)}"
    )

    width = len(without[0])
    for got, expected in zip(with_note, without):
        assert got[:width] == expected, (
            f"the note disturbed a table cell -- stub included: {got[:width]} != {expected}"
        )

    note_cells = [row[width:] for row in with_note]
    seen = {tok for row in note_cells for cell in row for tok in cell.split()}
    assert seen, f"the note is not in a column of its own: {with_note}"
    assert seen <= set(NOTE.split()), (
        f"the note column picked up something that is not the note: {seen}"
    )


def _left_margin_page(with_note: bool):
    """The same table, with the note in the LEFT margin instead of the right."""
    doc = fitz.open()
    page = doc.new_page(width=600, height=800)
    data = [
        ["Model", "Beta", "SE", "t-stat"],
        ["OLS", "1.25", "0.05", "25.0"],
        ["IV", "1.80", "0.12", "15.0"],
        ["GMM", "1.45", "0.08", "18.1"],
    ]
    cols = [140, 250, 360, 470]
    y = 100
    for row in data:
        for c, cell in enumerate(row):
            page.insert_text((cols[c], y), cell, fontsize=10)
        y += 25
    page.insert_text((140, y + 20), "Note: Standard errors clustered, N = 500.", fontsize=9)
    if with_note:
        page.insert_text((40, 200), ALPHA_NOTE, fontsize=8, rotate=90)
    return doc, page


def _left_cells(with_note: bool) -> list[list[str]]:
    doc, page = _left_margin_page(with_note)
    try:
        regions = rowize_from_word_list(list(page.get_text("words")))
    finally:
        doc.close()
    assert regions, "fixture must reconstruct"
    return [
        [c.strip() for c in line.strip().strip("|").split("|")]
        for line in regions[0][1].splitlines()
        if line.lstrip().startswith("|") and "---" not in line
    ]


def test_a_left_margin_note_does_not_corrupt_the_stub() -> None:
    """#464 review: the label branch consumed a LEFT-margin note first.

    The dedicated note column only ever helped notes to the RIGHT of the data.
    A note in the left margin sits left of the data boundary, so it was
    classified as label text before the folded-marginal check ran, and the row
    stub stayed corrupted -- the exact defect GH-461 fixed on the other side.
    """
    without = _left_cells(False)
    with_note = _left_cells(True)

    assert len(with_note) == len(without), "the note changed the row count"
    width = len(without[0])
    for got, expected in zip(with_note, without):
        assert got[:width] == expected, (
            f"the left-margin note corrupted a table cell: {got[:width]} != {expected}"
        )

    seen = {tok for row in with_note for cell in row[width:] for tok in cell.split()}
    assert seen, f"the left-margin note was dropped entirely: {with_note}"
    assert seen <= set(ALPHA_NOTE.split())
