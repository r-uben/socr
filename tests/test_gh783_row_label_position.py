"""GH-783: ``_has_row_labels`` must be POSITIONAL, not just "any non-numeric
token in this row".

Measured on ``main`` before this fix: a value-only band whose cells carry a
significance marker (``***``) attached to the number -- a token like
``0.33***`` -- fails ``_NUM_TOKEN_RE`` (which is anchored and does not allow
trailing star characters) and so registered as a "label" no matter where it
sat. That let a wide single table's own inter-value-group gap read as if the
right-hand group had its own label column, defeating GH-152's false-positive
guard (``_has_row_labels``) on a corpus-realistic case: academic tables
routinely carry ``***`` in value columns.

The fix requires the non-numeric token to sit strictly LEFT of the band's
own leftmost RECURRING numeric lane (the GH-152 plan's TICKET-A1 ruling).
Reusing ``_adjacent_lane_of`` for clustering and gating lane recurrence on
``_MIN_TABLE_ROWS`` -- not ``has_numeric_columns`` / ``_MIN_LANES_PER_ROW``,
which the plan forbids reusing as the per-band gate.

Hermetic: pure synthetic word geometry (PyMuPDF ``get_text("words")`` tuple
shape), no PDF read, no provider, no corpus content.
"""

from __future__ import annotations

import pytest

fitz = pytest.importorskip("fitz", reason="PyMuPDF not installed")

from socr.tables.reconstruct import _has_row_labels, rowize_from_word_list  # noqa: E402

CHAR_W = 6.0
ROW_H = 14.0


def _w(x: float, y: float, text: str, block: int, line: int, word_no: int = 0) -> tuple:
    x1 = x + CHAR_W * len(text)
    return (x, y, x1, y + 9.0, text, block, line, word_no)


class TestStarredValueColumnDoesNotFakeALabelColumn:
    """The headline defect: a decorated-value column must not register as a
    label column just because its tokens fail ``_NUM_TOKEN_RE``."""

    def test_unit_starred_value_only_band_has_no_row_labels(self):
        """Direct unit reproduction of the defect, exactly as measured on
        ``main``: a value-only band (no label word at all) whose values
        carry ``***`` must NOT read as having a label column."""
        words: list = []
        y = 100.0
        for i in range(5):
            row_y = y + i * ROW_H
            words.append(_w(350.0, row_y, f"{i}.33", block=1, line=i))
            words.append(_w(400.0, row_y, f"{i}.44***", block=1, line=i))

        assert _has_row_labels(words) is False

    def test_wide_single_table_with_starred_value_group_is_not_split(self):
        """End-to-end reproduction through ``rowize_from_word_list``: a
        SINGLE table with two value-column groups (label + Mean group + an
        SD group whose cells carry significance stars) must stay one
        region. Pre-fix, the starred group's tokens registered as its own
        label column and the false-positive guard let the split through --
        confirmed by mutation (reverting the positional clause reproduces a
        2-region split on this exact fixture)."""
        words: list = []
        y = 100.0
        for i in range(5):
            row_y = y + i * ROW_H
            words.append(_w(60.0, row_y, f"Row{i}", block=0, line=i))
            words.append(_w(110.0, row_y, f"{i}.11", block=0, line=i))
            words.append(_w(150.0, row_y, f"{i}.22", block=0, line=i))
            # Wide gap between the two value-column groups, mirroring
            # test_wide_single_table_between_value_column_groups_is_not_split
            # -- but here the SD group has TWO clean numeric columns (so it
            # still forms a valid grid on its own) plus a THIRD, starred
            # column -- the starred column alone must not fake a label.
            words.append(_w(350.0, row_y, f"{i}.33", block=0, line=i))
            words.append(_w(400.0, row_y, f"{i}.44", block=0, line=i))
            words.append(_w(450.0, row_y, f"{i}.55***", block=0, line=i))

        regions = rowize_from_word_list(words)
        assert len(regions) == 1, (
            f"a starred value-column group must not fake a label column and "
            f"split a single table into {len(regions)} regions"
        )


class TestGenuineSideBySideTablesStillSplit:
    """Load-bearing: tightening the guard must not defeat GH-152 itself."""

    def _two_tables(self, n_rows: int = 5) -> list:
        words: list = []
        y = 100.0
        for i in range(n_rows):
            row_y = y + i * ROW_H
            words.append(_w(60.0, row_y, f"LeftLab{i}", block=0, line=i))
            words.append(_w(140.0, row_y, f"{i}.11", block=0, line=i))
            words.append(_w(190.0, row_y, f"{i}.22", block=0, line=i))
            words.append(_w(350.0, row_y, f"RightLab{i}", block=1, line=i))
            words.append(_w(430.0, row_y, f"{i}.33", block=1, line=i))
            words.append(_w(480.0, row_y, f"{i}.44", block=1, line=i))
            words.append(_w(530.0, row_y, f"{i}.55", block=1, line=i))
        return words

    def test_unit_both_bands_still_have_row_labels(self):
        words = self._two_tables()
        left_words = [w for w in words if w[5] == 0]
        right_words = [w for w in words if w[5] == 1]
        assert _has_row_labels(left_words) is True
        assert _has_row_labels(right_words) is True

    def test_two_genuine_tables_are_still_split(self):
        regions = rowize_from_word_list(self._two_tables())
        assert len(regions) == 2, (
            f"tightening _has_row_labels must not defeat GH-152's own split; "
            f"got {len(regions)} regions"
        )


class TestIncidentalNumericTokenInARealLabelColumn:
    """A label column with one row's incidental numeric label token (e.g. a
    stray ID number) must still be recognised as a label column -- the
    accidental numeric token must not found its own one-off "lane" and drag
    the positional boundary into the label column itself."""

    def test_one_off_numeric_label_does_not_blind_the_positional_check(self):
        words: list = []
        y = 100.0
        labels = ["Alpha", "Beta", "3", "Delta", "Epsilon"]
        for i, lab in enumerate(labels):
            row_y = y + i * ROW_H
            words.append(_w(60.0, row_y, lab, block=0, line=i))
            words.append(_w(140.0, row_y, f"{i}.11", block=0, line=i))
            words.append(_w(190.0, row_y, f"{i}.22", block=0, line=i))

        assert _has_row_labels(words) is True


class TestMinTableRowsFloorIsEnforced:
    """The clause's row-count part (``_MIN_TABLE_ROWS``), not just the
    positional part, must be exercised: a band whose labeled rows clear the
    ``_MIN_DATA_ROW_FRAC`` FRACTION but fall short of the absolute
    ``_MIN_TABLE_ROWS`` floor must not read as having a label column, even
    though every labeled row's label sits correctly to the left of the
    numeric lane. Row counts are chosen so the fraction alone (without the
    floor) would pass -- isolating the floor as the reason for False."""

    def test_below_min_table_rows_labeled_is_not_recognised(self):
        from socr.tables.reconstruct import _MIN_TABLE_ROWS

        assert _MIN_TABLE_ROWS >= 2, "fixture assumes _MIN_TABLE_ROWS > 1"
        n_labeled = _MIN_TABLE_ROWS - 1
        n_rows = 2 * n_labeled  # exactly _MIN_DATA_ROW_FRAC (0.5) labeled

        words: list = []
        y = 100.0
        for i in range(n_rows):
            row_y = y + i * ROW_H
            if i < n_labeled:
                words.append(_w(60.0, row_y, f"Lab{i}", block=0, line=i))
            words.append(_w(140.0, row_y, f"{i}.11", block=0, line=i))
            words.append(_w(190.0, row_y, f"{i}.22", block=0, line=i))

        assert _has_row_labels(words) is False
