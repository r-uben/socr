"""GH-342: gutter marks must not move the label boundary.

``_promote_stub_lanes`` advanced on "there is recurring non-numeric text in the
gap between two lanes". Recurrence stops a single stray glyph, but it does not
distinguish a stub column from ordinary gutter marks: a ``n.a.`` or a dagger
footnote appearing on three data rows in a wide data-to-data gutter satisfies
exactly the same test. ``data_start_x`` moved and a real data column was
swallowed into the label cell.

Pinned as a DIFFERENCE at ``rowize_from_word_list``, the production caller: the
same geometry with and without the markers must produce the same grid. A count
asserted against a literal would pass for the wrong reason if lane detection
changed underneath.
"""

from __future__ import annotations

from socr.tables.reconstruct import rowize_from_word_list


def _w(x: float, y: float, text: str, w: float = 26.0, h: float = 10.0):
    return (x, y, x + w, y + h, text, 0, 0, 0)


def _four_data_lanes(*, markers: bool, marker: str = "n.a.") -> list:
    """Four full data lanes, a label column, and a WIDE gutter after lane 1.

    The gutter is 120pt, comfortably over twice the snap radius, which is what
    makes it eligible for promotion at all. Markers sit on three data rows --
    enough to satisfy the recurrence test the old code relied on.
    """
    words: list = []
    y = 100.0
    lanes = [80.0, 200.0, 320.0, 440.0]
    for r in range(4):
        words.append(_w(40.0, y, f"Row{r}"))
        for c, x in enumerate(lanes):
            words.append(_w(x, y, f"{r}{c}.5"))
        if markers and r < 3:
            words.append(_w(150.0, y, marker))
        y += 16.0
    return words


def _grid(words: list) -> list[list[str]]:
    regions = rowize_from_word_list(words)
    assert regions, "fixture must produce a table region"
    return [
        [c.strip() for c in line.strip().strip("|").split("|")]
        for line in regions[0][1].splitlines()
        if line.lstrip().startswith("|") and "---" not in line
    ]


class TestGutterMarksDoNotSwallowAColumn:
    def test_markers_do_not_change_the_column_count(self) -> None:
        without = _grid(_four_data_lanes(markers=False))
        with_marks = _grid(_four_data_lanes(markers=True))

        assert len(with_marks[0]) == len(without[0]), (
            f"a recurring gutter mark moved the label boundary: "
            f"{len(without[0])} columns became {len(with_marks[0])}"
        )

    def test_no_data_value_is_lost_into_the_label(self) -> None:
        """The consequence that matters: a swallowed column takes real numbers
        with it, and this is a citation corpus."""
        with_marks = _grid(_four_data_lanes(markers=True))
        emitted = {tok for row in with_marks for cell in row for tok in cell.split()}

        for r in range(4):
            for c in range(4):
                assert f"{r}{c}.5" in emitted, f"data value {r}{c}.5 was lost"

    def test_a_dagger_footnote_behaves_the_same_as_n_a(self) -> None:
        """The ticket names both shapes; neither is numeric, so neither should
        be read as a stub column."""
        dagger = _grid(_four_data_lanes(markers=True, marker="†"))
        without = _grid(_four_data_lanes(markers=False))

        assert len(dagger[0]) == len(without[0])
