"""GH-355: two mechanical holes in the presence oracle.

``native_text_value_counts`` feeds ``presence_verdict_from_text``, which decides
the GH-322 disposition. Both bugs make a CORRECT candidate look wrong.

1. The grid path iterated a ``Counter`` directly, which yields keys, so every
   native token's count collapsed to 1 -- while the regex fallback kept
   multiplicity. Two surfaces, two answers, from a function whose docstring says
   "with multiplicity". This module uses Counters rather than sets precisely so
   one occurrence becoming two reads as invented (GH-270 substitution); with
   counts flattened, a candidate correctly repeating a repeated value fails the
   multiset check and falls back to flagged native.

2. The fallback regex required a digit before the decimal point, so ``.75``
   matched from the ``7`` and became ``75`` -- an order of magnitude out,
   silently, in a presence oracle. Econ tables write ``.75`` and ``.05``
   constantly.

Both pinned as DIFFERENCES between the two surfaces, which is what the bug
actually was: the same page read two ways must give the same counts.
"""

from __future__ import annotations

from socr.tables.escalation_canary import native_text_value_counts

_GRID = "| a | b |\n| --- | --- |\n| 5.0 | 5.0 |\n| 5.0 | 1.0 |\n"
_PROSE = "values 5.0 and 5.0 and 5.0 and 1.0 here"


class TestMultiplicitySurvivesBothPaths:
    def test_the_grid_path_keeps_multiplicity(self) -> None:
        counts = native_text_value_counts(_GRID)
        assert counts["5.0"] == 3, f"three occurrences collapsed to {counts['5.0']}"

    def test_both_surfaces_agree(self) -> None:
        """The real defect: the same values, read two ways, disagreed. Pinned as
        a difference so neither path can drift alone."""
        assert native_text_value_counts(_GRID) == native_text_value_counts(_PROSE)

    def test_distinct_values_are_still_counted_separately(self) -> None:
        """Control: a fix that returned inflated counts for everything would
        satisfy the tests above."""
        counts = native_text_value_counts(_GRID)
        assert counts["1.0"] == 1


class TestLeadingDecimalsAreNotMangled:
    def test_a_leading_decimal_is_not_read_as_a_whole_number(self) -> None:
        counts = native_text_value_counts("the value .75 appears")
        assert "75" not in counts, "'.75' was read as 75 -- an order of magnitude out"
        assert counts["0.75"] == 1

    def test_a_negative_leading_decimal_keeps_its_sign(self) -> None:
        counts = native_text_value_counts("coefficient -.05 here")
        assert counts["-0.05"] == 1
        assert "05" not in counts

    def test_ordinary_decimals_are_unchanged(self) -> None:
        """Control: the widened pattern must not disturb the common case."""
        counts = native_text_value_counts("values 0.75 and 12.5 here")
        assert counts["0.75"] == 1
        assert counts["12.5"] == 1


class TestTheOracleCoversThePageNotJustTheGrid:
    """GH-355 hole 3, and the reason it matters most of the three.

    This caller's input is ``native_table_structure_failed`` -- the grid is
    exactly what is broken. Narrowing the oracle to grid cells meant a correct
    model value sitting OUTSIDE the ragged native grid read as invented, and the
    good table was discarded.
    """

    _RAGGED = "| a | b |\n| --- | --- |\n| 5.0 | 1.0 |\nstray value 9.9 outside the grid\n"

    def test_a_value_outside_the_ragged_grid_is_seen(self) -> None:
        counts = native_text_value_counts(self._RAGGED)
        assert counts["9.9"] == 1, (
            "a page value outside the broken grid never entered the oracle, so a "
            "model that read it correctly looked like it invented it"
        )

    def test_grid_values_keep_their_counts(self) -> None:
        """Control: the union must not cost the grid its own signal."""
        counts = native_text_value_counts(self._RAGGED)
        assert counts["5.0"] == 1
        assert counts["1.0"] == 1

    def test_a_value_inside_the_grid_is_not_double_counted(self) -> None:
        """The union takes the MAX of the two passes, not the sum: a grid value
        appears in both, and adding would invent an occurrence -- which is
        exactly the GH-270 substitution signal this module exists to preserve."""
        counts = native_text_value_counts(_GRID)
        assert counts["5.0"] == 3, f"three occurrences became {counts['5.0']}"
        assert counts["1.0"] == 1
