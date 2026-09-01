"""GH-349: a right-aligned numeric column must not miss the gate.

``has_numeric_columns`` clustered lanes on word **x0**. A right-aligned column
has a stable **x1** and a varying x0 -- ``1`` against ``1000.00`` differ by far
more than ``_LANE_X_TOL_PT`` -- so its tokens split into separate lanes, none
reached the reuse threshold, and a real 3-column borderless table returned
False. That is exactly the ``find_tables() == 0`` booktabs page this gate exists
for.

The fix keys lanes on either edge. It does NOT weaken reuse: #334 measured that
dropping the threshold to 2 rows readmits 2 of the 4 Glaeser noise pages, so the
same recurrence requirement is kept and only the anchor changes.
"""

from __future__ import annotations

from socr.tables.reconstruct import has_numeric_columns


class _Page:
    def __init__(self, words):
        self._words = words

    def get_text(self, kind):
        assert kind == "words"
        return self._words


def _w(x0: float, y0: float, text: str, width: float):
    return (x0, y0, x0 + width, y0 + 10, text, 0, 0, 0)


#: Widths differ by well over _LANE_X_TOL_PT (6.0): "1" is 5pt, "3000.25" is 35pt.
_VALUES = [
    ("1", "1000.00", "2"),
    ("-1.96", "3", "250.5"),
    ("1000.00", "2", "1"),
    ("4", "-1.96", "3000.25"),
]


def _right_aligned() -> _Page:
    """Stable RIGHT edges at 200/300/400; x0 varies with token width."""
    words = []
    for row, triple in enumerate(_VALUES):
        y = 100.0 + row * 16.0
        for right, token in zip((200.0, 300.0, 400.0), triple):
            width = len(token) * 5.0
            words.append(_w(right - width, y, token, width))
    return _Page(words)


def _left_aligned() -> _Page:
    """The same values with stable LEFT edges -- the case that always worked."""
    words = []
    for row, triple in enumerate(_VALUES):
        y = 100.0 + row * 16.0
        for left, token in zip((150.0, 250.0, 350.0), triple):
            words.append(_w(left, y, token, len(token) * 5.0))
    return _Page(words)


def _scatter(rows: int = 10, per_row: int = 4) -> _Page:
    """Corrupt-layer noise: as many tokens, each in its own lane, neither edge
    recurring. The case reuse was added to reject."""
    words = []
    x = 40.0
    for row in range(rows):
        y = 100.0 + row * 14.0
        for i in range(per_row):
            token = f"{row}{i}.5"
            words.append(_w(x, y, token, len(token) * 5.0))
            x += 37.0
    return _Page(words)


class TestAlignmentDoesNotDecideDetection:
    def test_a_right_aligned_table_is_detected(self) -> None:
        assert has_numeric_columns(_right_aligned()) is True

    def test_the_same_table_left_aligned_is_also_detected(self) -> None:
        """Difference pin: identical values and identical column count -- only
        the alignment differs. If the two ever disagree again, that is the bug."""
        assert has_numeric_columns(_right_aligned()) == has_numeric_columns(_left_aligned())


class TestReuseIsNotWeakened:
    def test_scatter_is_still_rejected(self) -> None:
        """The control that matters. #334 measured that lowering the reuse
        threshold readmits real noise pages; keying on a second edge must not
        achieve the same thing by another route."""
        assert has_numeric_columns(_scatter()) is False
