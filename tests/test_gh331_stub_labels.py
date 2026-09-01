"""GH-331: a numeric stub column must not eat the row labels.

``_rowize_segment`` set the label boundary from the leftmost NUMERIC lane, which
assumes that lane is the first data column. On a table with a numeric stub —
Cochrane's ``n`` column holding 2, 3, 4, 5 — it is not, so every row label sits to
its right, gets snapped into the first data lane, and displaces the row.

Measured on the corpus: 18/18 orphaned-stub rows on one page, 37/51 on another,
the signature on 5 separate papers.

Hermetic: synthetic word geometry, no PDF read and no provider.
"""

from __future__ import annotations

from socr.tables.reconstruct import rowize_from_word_list


def _w(x: float, y: float, text: str) -> tuple:
    return (x, y, x + 22.0, y + 8.0, text, 0, 0, 0)


def _stub_table(with_labels: bool = True) -> list:
    """A numeric stub at x=60, text labels at x=100, data lanes from x=200.

    Shaped like the real failure: the stub carries row identifiers on block-start
    rows only, and the labels sit BETWEEN the stub and the first data column.
    """
    words: list = []
    y = 100.0
    for block, ident in enumerate(("2", "3", "4")):
        words.append(_w(60.0, y, ident))  # numeric stub
        for c, val in enumerate(("1.11", "2.22", "3.33")):
            words.append(_w(200.0 + c * 70.0, y, val))
        y += 16.0
        for label in ("Large", "Small"):
            if with_labels:
                words.append(_w(100.0, y, label))
                words.append(_w(140.0, y, "T"))
            for c, val in enumerate(("0.44", "0.55", "0.66")):
                words.append(_w(200.0 + c * 70.0, y, val))
            y += 16.0
    return words


def _cells(md: str) -> list[list[str]]:
    return [
        [c.strip() for c in line.strip().strip("|").split("|")]
        for line in md.splitlines()
        if line.lstrip().startswith("|") and "---" not in line
    ]


def test_row_labels_survive_a_numeric_stub_column():
    """The regression: labels were snapped into the first data lane and lost."""
    regions = rowize_from_word_list(_stub_table())
    assert regions, "fixture must produce a table region"
    grid = _cells(regions[0][1])

    labels = [row[0] for row in grid]
    assert any("Large" in c for c in labels), f"'Large T' never reached a label cell: {labels}"
    assert any("Small" in c for c in labels), f"'Small T' never reached a label cell: {labels}"


def test_no_word_is_silently_dropped():
    """The house rule, pinned. Today's failure mode is silent loss, not misplacement.

    GH-343: matched on CELL TOKENS, not ``in`` against a joined string. The stub
    ids are ``"2"``/``"3"``/``"4"``, which occur inside the data values
    ``"2.22"``/``"3.33"``/``"4.44"`` -- so a substring check stayed green with
    the stub ids gone entirely, which is exactly the silent loss this test
    exists to catch.
    """
    words = _stub_table()
    grid = _cells(rowize_from_word_list(words)[0][1])
    emitted_tokens: list[str] = []
    for row in grid:
        for cell in row:
            emitted_tokens.extend(cell.split())

    for w in words:
        assert w[4] in emitted_tokens, f"token {w[4]!r} vanished from the grid"


def test_promotion_touches_only_the_label_column(monkeypatch):
    """Pinned as PROMOTED vs NOT, which is what the claim is about.

    GH-343: this used to compare the with-labels fixture against a stub-only
    one and drop column 0. Both paths land the stub in the label cell, so the
    data sub-grid matched even if promotion never fired -- the test was
    tautological and stayed green with the production line reverted.

    The difference that actually tests the claim is the helper on versus off.
    Promotion must change the LABEL column (that is its whole job) and leave
    every data cell untouched.
    """
    from socr.tables import reconstruct

    promoted = _cells(rowize_from_word_list(_stub_table(with_labels=True))[0][1])

    monkeypatch.setattr(
        reconstruct, "_promote_stub_lanes", lambda lane_centers, *_a, **_k: lane_centers
    )
    unpromoted = _cells(rowize_from_word_list(_stub_table(with_labels=True))[0][1])

    # What differs is not that labels MOVE -- without promotion they are lost
    # outright: the row-label words fall outside every lane's snap radius once
    # the stub lane is still in the list, so they are dropped from the grid
    # entirely. That is GH-331's loss, and it is why both grids are the same
    # WIDTH (#456 review): the stub stays in the label cell either way, so
    # there is no extra data column to skip when comparing.
    labels_promoted = [row[0] for row in promoted]
    labels_unpromoted = [row[0] for row in unpromoted]
    assert len(promoted[0]) == len(unpromoted[0]), (
        f"the two grids are different widths ({len(promoted[0])} vs "
        f"{len(unpromoted[0])}), so the column-wise comparison below is not "
        "aligned and this test would be comparing different columns"
    )
    assert labels_promoted != labels_unpromoted, (
        "promotion changed nothing, so this fixture cannot tell the two apart "
        f"and the assertion below is vacuous: {labels_promoted}"
    )

    data_promoted = [row[1:] for row in promoted]
    data_unpromoted = [row[1:] for row in unpromoted]
    assert data_promoted == data_unpromoted, (
        f"promotion moved a DATA cell, not just the label:\n"
        f"  promoted:   {data_promoted}\n  unpromoted: {data_unpromoted}"
    )


def test_a_table_with_no_intervening_labels_is_untouched():
    """Inertness. The rule fires only on the evidence of the bug — label text
    between two numeric lanes — so an ordinary table must be unaffected."""
    words: list = []
    y = 100.0
    for _ in range(4):
        for c, val in enumerate(("1.11", "2.22", "3.33")):
            words.append(_w(200.0 + c * 70.0, y, val))
        y += 16.0

    regions = rowize_from_word_list(words)
    assert regions
    grid = _cells(regions[0][1])
    # Every data value still present, and no value displaced into a label cell.
    flat = " ".join(c for row in grid for c in row)
    for val in ("1.11", "2.22", "3.33"):
        assert val in flat


def test_a_single_stray_word_does_not_promote_a_data_column():
    """Recurrence, not one sighting.

    A lone non-numeric word between two data lanes -- a footnote marker, a loose
    glyph -- must not move the boundary, or a real data column is swallowed into
    the label. Reuses `_MIN_TABLE_ROWS`, the existing minimum evidence for a
    table, rather than a new constant.

    Pinned as a DIFFERENCE: the same geometry with and without the stray word must
    produce the same shape.
    """

    def _plain(stray: bool):
        words: list = []
        y = 100.0
        for i in range(5):
            for c, val in enumerate(("1.11", "2.22", "3.33")):
                words.append(_w(200.0 + c * 70.0, y, val))
            if stray and i == 0:
                words.append(_w(245.0, y, "a"))
            y += 16.0
        return _cells(rowize_from_word_list(words)[0][1])

    with_stray, without = _plain(True), _plain(False)
    widths_with = {len(r) for r in with_stray}
    widths_without = {len(r) for r in without}
    assert widths_with == widths_without, (
        f"one stray word changed the column count: {widths_with} vs {widths_without}"
    )
