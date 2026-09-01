"""GH-352: the reported flag must describe the walk that actually ran.

``column_binding_unverifiable`` was computed from the PROJECTED column count
while the 1:1 cell walk gates on the PHYSICAL one.
``_project_candidate_data_columns`` can make those disagree — and then the walk
is skipped while the scoreboard reports columns verified. That is the 1/13 in
the GH-332 table.

It does not stamp SUCCESS today, because ``fully_checked`` can still be 0 via
``ambiguous_count``. GH-326's gate will read this flag, and then it would.
"""

from __future__ import annotations

from socr.tables.binding import bind


def _w(x0, y0, x1, y1, text):
    return (x0, y0, x1, y1, text, 0, 0, 0)


def _two_lane_page():
    """A page with exactly TWO numeric lanes."""
    return [
        _w(50, 100, 90, 110, "Coef"),
        _w(150, 100, 180, 110, "1.10"),
        _w(250, 100, 280, 110, "1.11"),
        _w(50, 130, 90, 140, "SE"),
        _w(150, 130, 180, 140, "0.05"),
        _w(250, 130, 280, 140, "0.06"),
    ]


#: FOUR physical data columns, two of them empty. The projection collapses them
#: to two, which matches the page's two lanes -- so the projected count agrees
#: with the lanes while the physical count does not. That is the divergence.
_WIDER_THAN_THE_PAGE = "\n".join(
    [
        "|      | OLS  | note | IV   | note2 |",
        "| --- | --- | --- | --- | --- |",
        "| Coef | 1.10 |      | 1.11 |       |",
        "| SE   | 0.05 |      | 0.06 |       |",
    ]
)

#: The same page and the same values, with no columns to project away.
_MATCHING = "\n".join(
    [
        "|      | OLS  | IV   |",
        "| --- | --- | --- |",
        "| Coef | 1.10 | 1.11 |",
        "| SE   | 0.05 | 0.06 |",
    ]
)


class TestTheFlagFollowsTheWalk:
    def test_a_projected_table_is_not_reported_as_column_verifiable(self) -> None:
        """The bug: projection made the flag say verified while the walk that
        would verify it was skipped."""
        result = bind(_two_lane_page(), _WIDER_THAN_THE_PAGE)

        assert result.column_binding_unverifiable is True, (
            "the 1:1 walk is skipped for this table (physical columns != lanes), "
            "so it must not be reported as column-verifiable"
        )

    def test_a_table_whose_columns_match_the_page_is_still_verifiable(self) -> None:
        """Difference control: same page, same values, no projection. Without
        this, a change that marked everything unverifiable would pass the test
        above while destroying the signal."""
        result = bind(_two_lane_page(), _MATCHING)

        assert result.column_binding_unverifiable is False

    def test_the_two_tables_differ_only_in_projected_columns(self) -> None:
        """Pinned as a difference so neither case can pass for an unrelated
        reason: the values are identical, only the empty columns differ."""
        projected = bind(_two_lane_page(), _WIDER_THAN_THE_PAGE)
        matching = bind(_two_lane_page(), _MATCHING)

        assert projected.column_binding_unverifiable != matching.column_binding_unverifiable
