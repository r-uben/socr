"""GH-360: column lanes must describe the tokens that actually ship.

``_merge_unclosed_bracket_words`` rejoins ``[0.01,`` + ``0.35]`` into one cell,
but lanes used to be clustered on the RAW words -- so the closer contributed a
lane of its own, and the merged token (which sits at the OPENER's x) left that
lane empty on every repaired row.

Every assertion here is a DIFFERENCE between two runs of the same rowizer, not
a value measured on one machine.
"""

from socr.tables.reconstruct import _OPENERS, _is_numeric_word, rowize_from_word_list


def _w(x0: float, y0: float, text: str, width: float = 26.0) -> tuple:
    return (x0, y0, x0 + width, y0 + 10, text, 0, 0, 0)


def _cells(words: list) -> list[list[str]]:
    regions = rowize_from_word_list(words)
    assert regions, "fixture must reconstruct to a table at all"
    return [
        [c.strip() for c in line.strip().strip("|").split("|")]
        for line in regions[0][1].splitlines()
        if line.lstrip().startswith("|") and "---" not in line
    ]


def _rows(closer_x: float, *, torn: bool = True) -> list:
    """Geometry taken from a measured corpus page (cochrane_piazzesi p20).

    Four data rows whose second column is a bracketed span PyMuPDF tears in two
    (``(1964`` + ``-1989)``), so the closer's x is a lane of its own. A fifth
    row puts a NON-numeric closer near that lane, which is what stops
    ``_clean_grid`` from dropping the column: it is blank on every data row but
    not on every row.

    ``torn`` renders the span as the single word it prints as, for the control.
    """
    words: list = []
    y = 100.0
    for i in range(4):
        words += [_w(116.0, y, str(i + 1))]
        if torn:
            words += [_w(127.0, y, "(1964"), _w(159.0, y, f"-198{i})")]
        else:
            words += [_w(127.0, y, f"(1964 -198{i})", width=54.0)]
        words += [
            _w(205.0, y, f"{i}.51"),
            _w(238.0, y, f"({i}.18)"),
            _w(458.0, y, f"0.0{i}"),
        ]
        y += 14.0
    words += [
        _w(133.0, y, "8"),
        _w(143.0, y, "(all"),
        _w(closer_x, y, "f)"),
        _w(458.0, y, "0.13"),
    ]
    return words


def test_the_closers_lane_does_not_become_a_column() -> None:
    """The tear must not manufacture a column that every data row leaves blank."""
    torn = _cells(_rows(146.0, torn=True))
    whole = _cells(_rows(146.0, torn=False))
    assert len(torn[0]) == len(whole[0]), (
        f"the tear manufactured a column: torn={len(torn[0])} whole={len(whole[0])}"
    )
    data = [r for r in torn if r and r[0].strip().isdigit()]
    assert len(data) == 4, f"fixture must yield four data rows, got {data}"
    for row in data:
        assert "" not in row, f"phantom blank cell left by the closer's lane: {row}"


def test_the_words_the_phantom_column_split_are_reunited() -> None:
    """Removing the phantom lane must put ``(all f)`` back in one cell."""
    cells = [c for row in _cells(_rows(146.0)) for c in row]
    assert any("(all f)" in c for c in cells), f"still split across columns: {cells}"


def test_a_lane_something_else_occupies_is_kept() -> None:
    """The prune must not delete a lane a non-numeric word legitimately sits in.

    Measured on the corpus (cochrane_piazzesi p10): a lone header ``T`` sits
    above a column whose lane exists only because interval closers land there.
    Pruning every merged-away lane deleted ``T`` -- a phantom column traded for
    lost content, which is not a fix. Words outside every lane's snap radius are
    discarded (the pre-existing drop tracked by #418), so the lane has to stay.
    """
    words: list = []
    y = 100.0
    words += [_w(60.0, y, "Var"), _w(150.0, y, "Est"), _w(250.0, y, "R2"), _w(320.0, y, "T")]
    y += 16.0
    for r in range(5):
        words += [
            _w(60.0, y, f"Row{r}"),
            _w(150.0, y, f"{r}.10"),
            _w(250.0, y, "[0.01,"),
            _w(320.0, y, "0.35]"),
        ]
        y += 16.0
    flat = [c for row in _cells(words) for c in row]
    assert "T" in flat, f"a header word in the closer's lane was dropped: {flat}"


def test_openers_advertises_only_pairs_that_can_merge() -> None:
    """A run extends across ``_is_numeric_word`` words only.

    So an opener whose bracketed form that predicate rejects can never merge
    anything, and listing it promises a capability the code lacks.
    """
    for opener, closer in _OPENERS.items():
        sample = f"{opener}0.35{closer}"
        assert _is_numeric_word(_w(0.0, 0.0, sample)), (
            f"{opener}{closer} is advertised but {sample!r} can never join a run"
        )
