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
        # GH-418 step 2 retarget: a captured marker now legitimately gains a
        # trailing column of its own (GH-461's `orphan_marginals`, emitted on
        # every row and dropped by `_clean_grid` only when no row uses it) --
        # so the WITH-markers grid is one column WIDER than the without-
        # markers grid by design, and the total-column-count proxy this test
        # used to assert no longer holds. What it actually protects -- the
        # marker does not move the label boundary or swallow a data lane --
        # is restated against the data-lane cells only (label + numeric
        # lanes), excluding that trailing column.
        without = _grid(_four_data_lanes(markers=False))
        with_marks = _grid(_four_data_lanes(markers=True))

        data_width = len(without[0])  # no markers -> trailing column is empty, dropped
        assert [row[:data_width] for row in with_marks] == without, (
            f"a recurring gutter mark moved the label boundary or a data lane: "
            f"{[row[:data_width] for row in with_marks]} != {without}"
        )

    def test_no_data_value_is_lost_into_the_label(self) -> None:
        """The consequence that matters: a swallowed column takes real numbers
        with it, and this is a citation corpus."""
        with_marks = _grid(_four_data_lanes(markers=True))
        emitted = {tok for row in with_marks for cell in row for tok in cell.split()}

        for r in range(4):
            for c in range(4):
                assert f"{r}{c}.5" in emitted, f"data value {r}{c}.5 was lost"

        # GH-418 step 2 retarget -- the intended flip: this row populates all
        # 4 numeric lanes, so the marker is now CAPTURED into the trailing
        # column instead of deleted, and no drop event fires for it. Before
        # this ticket the marker was silently dropped; asserting its absence
        # was pinning that loss as correct. "Absence-of-token is not a
        # behaviour to preserve" (panel ruling, docs/log/2026-09-16_418-design.md).
        assert "n.a." in emitted, "the marker should now be captured, not dropped"

        drops: list[dict] = []
        regions = rowize_from_word_list(_four_data_lanes(markers=True), orphan_drops=drops)
        assert regions, "fixture must produce a table region"
        assert drops == [], f"a captured marker must not also be reported as a drop: {drops}"

    def test_a_dagger_footnote_behaves_the_same_as_n_a(self) -> None:
        """The ticket names both shapes; neither is numeric, so neither should
        be read as a stub column."""
        # GH-418 step 2 retarget: same reasoning as
        # test_markers_do_not_change_the_column_count -- restated against the
        # data-lane cells, since the dagger is now captured into its own
        # trailing column too.
        dagger = _grid(_four_data_lanes(markers=True, marker="†"))
        without = _grid(_four_data_lanes(markers=False))

        data_width = len(without[0])
        assert [row[:data_width] for row in dagger] == without
