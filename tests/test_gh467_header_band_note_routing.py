"""GH-467: pin the folded-note routing inside `_prepend_header_band`.

#464 added the tag check to that function's own cell-assignment loop, but no
fixture reached it -- reverting the routing left the whole suite green, and the
PR said so rather than claiming coverage.

The obstacle was reaching `_prepend_header_band` at all. It only prepends when
the header sits in a PREVIOUS y-segment whose words all snap to the current
segment's lanes, so a header carrying a stub-column word (`n`, at the label x)
stops the walk immediately. A header of nothing but lane-aligned words, one
segment above the data, does prepend -- and a marginal note whose y is nearest
the HEADER row folds onto it rather than onto a data row.

Without the routing the note snaps into the nearest header cell:

    |  | Est | SE | R2 Hdr |      <- header corrupted

With it, the header band gets the same dedicated note column the data rows use:

    |  | Est | SE | R2 | Hdr |
"""

from __future__ import annotations

from socr.tables.reconstruct import rowize_from_word_list

NOTE = "Hdr"


def _w(x0: float, y0: float, text: str, width: float = 26.0) -> tuple:
    return (x0, y0, x0 + width, y0 + 10, text, 0, 0, 0)


def _words(with_note: bool) -> list:
    words: list = [
        # Header band: ONLY lane-aligned words, so the prepend walk accepts it.
        _w(250.0, 100.0, "Est"),
        _w(330.0, 100.0, "SE"),
        _w(410.0, 100.0, "R2"),
    ]
    if with_note:
        # Right margin, y nearest the HEADER row so the fold attaches it there.
        words.append(_w(545.0, 102.0, NOTE))
    y = 140.0
    for i in range(4):
        words += [
            _w(60.0, y, str(i + 2)),
            _w(120.0, y, "Treasury"),
            _w(250.0, y, f"{i}.11"),
            _w(330.0, y, f"({i}.02)"),
            _w(410.0, y, f"0.{i}5"),
        ]
        y += 16.0
    return words


def _rows(with_note: bool) -> list[list[str]]:
    regions = rowize_from_word_list(_words(with_note))
    assert regions, "fixture must reconstruct, or nothing is measured"
    return [
        [c.strip() for c in line.strip().strip("|").split("|")]
        for line in regions[0][1].splitlines()
        if line.lstrip().startswith("|") and "---" not in line
    ]


def test_the_header_band_is_reached_at_all() -> None:
    """Anchor: without this, the pin below could pass on a page with no band.

    Every earlier attempt at this fixture failed here -- the header was never
    prepended, so the routing was never exercised and a green test proved
    nothing.
    """
    header = _rows(with_note=False)[0]
    assert [c for c in header if c] == ["Est", "SE", "R2"], (
        f"the header band was not prepended, so this file pins nothing: {header}"
    )


def test_a_note_folded_onto_the_header_row_does_not_corrupt_a_header_cell() -> None:
    """The defect: without the routing the note snaps into `R2`, giving `R2 Hdr`."""
    without = _rows(with_note=False)
    with_note = _rows(with_note=True)

    width = len(without[0])
    assert with_note[0][:width] == without[0], (
        f"the note corrupted a header cell: {with_note[0][:width]} != {without[0]}"
    )

    extra = [c for c in with_note[0][width:] if c.strip()]
    assert extra == [NOTE], f"the note is not in the header band's own note column: {with_note[0]}"


def test_the_data_rows_are_untouched_by_a_header_note() -> None:
    """Control: folding onto the header must not disturb the data rows."""
    without = _rows(with_note=False)
    with_note = _rows(with_note=True)

    assert len(with_note) == len(without), "the note changed the row count"
    width = len(without[0])
    for got, expected in zip(with_note[1:], without[1:]):
        assert got[:width] == expected, f"a data row changed: {got[:width]} != {expected}"
