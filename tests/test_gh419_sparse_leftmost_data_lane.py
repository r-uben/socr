"""GH-419: a sparse leftmost DATA column is not a stub.

#416 stopped #342's shape by requiring the promoted lane to be populated on
fewer data rows than its neighbour, and capped the blast radius to one leading
promotion. Neither refuses a false FIRST promotion, and this shape supplies one:
a real leftmost data column that happens to be sparse, plus the recurring
gutter marks ("n.a.", a dagger) every table has.

Pinned as a DIFFERENCE against the same geometry with the markers removed. The
markers live in a gutter that carries no data, so they must not change the
table's shape at all -- and the sparse column's values must land in their own
CELLS, not merely appear somewhere in the row (GH-343: a swallowed value still
sits inside the label string, so a substring check passes while the column is
gone).
"""

from __future__ import annotations

from socr.tables.reconstruct import rowize_from_word_list


def _w(x0: float, y0: float, text: str, width: float = 24.0) -> tuple:
    return (x0, y0, x0 + width, y0 + 10, text, 0, 0, 0)


def _rows(*, markers: bool) -> list:
    """Four rows: a label, a SPARSE real data lane, a gutter, three full lanes.

    The leftmost data lane is populated on 2 of 4 rows -- fewer than its
    neighbour, which is what makes #416's sparsity comparison classify it as a
    stub. Only the gutter marks differ between the two variants.
    """
    words: list = []
    y = 100.0
    for r in range(4):
        words.append(_w(50.0, y, f"Label{r}"))
        if r < 2:
            words.append(_w(150.0, y, f"{r}.77"))
        if markers:
            words.append(_w(230.0, y, "n.a."))
        words.append(_w(320.0, y, f"{r}.11"))
        words.append(_w(390.0, y, f"{r}.22"))
        words.append(_w(460.0, y, f"{r}.33"))
        y += 16.0
    return words


def _cells(words: list) -> list[list[str]]:
    regions = rowize_from_word_list(words)
    assert regions, "fixture must reconstruct to a table at all"
    return [
        [c.strip() for c in line.strip().strip("|").split("|")]
        for line in regions[0][1].splitlines()
        if line.lstrip().startswith("|") and "---" not in line
    ]


def test_gutter_marks_do_not_change_the_table_shape() -> None:
    with_marks = _cells(_rows(markers=True))
    without = _cells(_rows(markers=False))
    assert len(with_marks[0]) == len(without[0]), (
        f"recurring gutter marks swallowed a column: {len(without[0])} -> {len(with_marks[0])}"
    )


def test_the_sparse_columns_values_keep_their_own_cells() -> None:
    """Not a substring check: a swallowed value still sits inside the label."""
    rows = _cells(_rows(markers=True))
    data = [r for r in rows if any(c.startswith("Label") for c in r)]
    assert len(data) == 4, f"expected four data rows, got {data}"

    for row in data[:2]:
        # GH-418: the rescued gutter marker now shares this cell ("0.77 n.a."),
        # so compare TOKENS within a data cell rather than the whole cell text.
        # The claim is unchanged -- the value must be in a data column, not
        # swallowed into the label -- and index >= 1 is what enforces that.
        assert any({"0.77", "1.77"} & set(c.split()) for c in row[1:]), (
            f"the sparse column's value is not in a data cell: {row}"
        )
    for row in data:
        assert row[0].startswith("Label") and len(row[0].split()) == 1, (
            f"the label cell absorbed the sparse column or the marker: {row[0]!r}"
        )


def test_a_genuine_stub_is_still_promoted() -> None:
    """Control: the GH-331 shape must keep working.

    A genuine stub carries the row identifier on BLOCK-START rows only, so it is
    sparse -- that is what #416's sparsity comparison keys on -- and the row's
    label text sits to its RIGHT, with nothing to its left. The sparse DATA
    column in the tests above has its label already to the left, and that is the
    only thing separating the two shapes. If this control stopped passing, the
    fix would have traded #419's loss for #331's.
    """
    words: list = []
    y = 100.0
    for r in range(4):
        if r % 2 == 0:  # block-start rows only
            words.append(_w(50.0, y, str(r + 2)))
        words.append(_w(110.0, y, "Treasury"))
        words.append(_w(320.0, y, f"{r}.11"))
        words.append(_w(390.0, y, f"{r}.22"))
        words.append(_w(460.0, y, f"{r}.33"))
        y += 16.0
    rows = _cells(words)
    data = [r for r in rows if "Treasury" in " ".join(r)]
    assert len(data) == 4, f"expected four rows carrying the label: {rows}"
    for row in data:
        assert "Treasury" in row[0], (
            f"the stub was not promoted, so the label stayed out of the label cell: {row!r}"
        )


def test_a_label_reaching_into_the_snap_margin_still_counts() -> None:
    """#438 review 1: the guard must key on the LEFT edge, like the label rule.

    ``_rowize_segment`` decides label membership with
    ``w[0] < data_start_x - snap_margin``. A label that STARTS left of the
    candidate lane but reaches into the snap margin is label text there, so it
    must block promotion here too. Keying on the right edge missed exactly
    these, and the sparse column was swallowed anyway.
    """
    words: list = []
    y = 100.0
    for r in range(4):
        # starts well left of the 150 lane, but its right edge runs past 144
        words.append(_w(60.0, y, f"LongLabel{r}", width=90.0))
        if r < 2:
            words.append(_w(150.0, y, f"{r}.77"))
        words.append(_w(230.0, y, "n.a."))
        words.append(_w(320.0, y, f"{r}.11"))
        words.append(_w(390.0, y, f"{r}.22"))
        words.append(_w(460.0, y, f"{r}.33"))
        y += 16.0
    rows = _cells(words)
    data = [r for r in rows if r[0].strip()]
    assert len(data) == 4, f"expected four data rows: {rows}"
    for row in data[:2]:
        assert any({"0.77", "1.77"} & set(c.split()) for c in row[1:]), (
            f"a label overlapping the snap margin let the column be swallowed: {row}"
        )


def test_a_lone_margin_marker_does_not_block_a_genuine_stub() -> None:
    """#438 review 2: one stray left-margin token is not a label column.

    Treating ANY non-numeric word to the left as proof of a data column would
    refuse a genuine stub whenever a footnote or significance marker sits in
    the left margin -- handing back the GH-331 label loss the promotion exists
    to prevent. The evidence has to recur, the same standard the gap test uses.
    """
    words: list = []
    y = 100.0
    for r in range(4):
        if r == 0:  # a single stray marker in the left margin
            words.append(_w(20.0, y, "*"))
        if r % 2 == 0:  # genuine stub: block-start rows only
            words.append(_w(50.0, y, str(r + 2)))
        words.append(_w(110.0, y, "Treasury"))
        words.append(_w(320.0, y, f"{r}.11"))
        words.append(_w(390.0, y, f"{r}.22"))
        words.append(_w(460.0, y, f"{r}.33"))
        y += 16.0
    rows = _cells(words)
    data = [r for r in rows if "Treasury" in " ".join(r)]
    assert len(data) == 4, f"expected four rows carrying the label: {rows}"
    for row in data:
        assert "Treasury" in row[0], f"a lone margin marker refused a genuine stub: {row!r}"
