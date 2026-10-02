"""GH-924: ``split_phrase``, a DEFER-only predicate in the native-first ship gate.

The source prints ``word number`` (``March 2001``) at the page's ordinary word space and the grid
puts the two in different cells: the rowizer left the word in the label cell and snapped the number
to a numeric lane. Every test pins a DIFFERENCE: the same words with the phrase in one cell and in
two, the same markdown with the gate off, the same gap at and just over the word space. Words are
synthetic PyMuPDF-shaped tuples; nothing reads a corpus file and nothing needs a provider.
"""

from __future__ import annotations

from native_table_fixtures import CHAR_W, COL_XS, HEADER, PITCH, ROWS, UNCHECKED, Y0
from test_gh916_native_ship_gate import _md, _plan, _predicates
from test_gh936_prose_in_header import WORD_SPACE, _grid_words, _w

from socr.tables import ship_gate
from socr.tables.native_first import DEFER, SHIP

SP = ship_gate.SPLIT_PHRASE
NUMS = ["0.179", "0.211", "0.301"]


def _words(gap: float, *, label: str = "March", prose_lines: int = 2) -> list[tuple]:
    """The grid with its first data row replaced by ``GDP <label> <2001>`` then three lane numbers.

    The phrase word and the number sit ``gap`` apart on one text line; the other rows are the
    shared fixture.
    """
    base = _grid_words(prose_lines=prose_lines)
    y = Y0 + PITCH  # first data row
    keep = [w for w in base if round(w[1]) != round(y)]
    x = COL_XS[0]
    row = [_w(x, y, "GDP", 7, 0, 0)]
    x += CHAR_W * 3 + WORD_SPACE
    row.append(_w(x, y, label, 7, 0, 1))
    x += CHAR_W * len(label) + gap
    row.append(_w(x, y, "2001", 7, 0, 2))
    for k, num in enumerate(NUMS):
        row.append(_w(COL_XS[2 + k], y, num, 7, 0, 3 + k))
    return keep + row


def _md_rows(first: list[str]) -> str:
    return _md(list(HEADER), [first] + [list(r) for r in ROWS[1:]])


SPLIT = ["GDP March", "2001", *NUMS]
JOINED = ["GDP March 2001", "", *NUMS]


def _fired(words, md) -> set[str]:
    return {f["predicate"] for f in ship_gate.native_ship_gate(words, md, line_dirs=UNCHECKED)}


class TestFaultAndNoFault:
    def test_difference_pin_split_phrase_defers_and_joined_ships(self) -> None:
        words = _words(WORD_SPACE)
        assert _plan(words, _md_rows(SPLIT), gate=False).action == SHIP, "the exact-pass exists"
        assert _plan(words, _md_rows(JOINED)).action == SHIP
        plan = _plan(words, _md_rows(SPLIT))
        assert plan.action == DEFER and _predicates(plan) == {SP}
        assert plan.reason.startswith(ship_gate.SHIP_GATE_REASON_PREFIX)

    def test_any_word_not_a_month_vocabulary(self) -> None:
        assert _fired(
            _words(WORD_SPACE, label="Zorblax"), _md_rows(["GDP Zorblax", "2001", *NUMS])
        ) == {SP}

    def test_gap_at_the_word_space_fires_and_just_over_does_not(self) -> None:
        assert _fired(_words(WORD_SPACE), _md_rows(SPLIT)) == {SP}
        assert _fired(_words(WORD_SPACE + 0.5), _md_rows(SPLIT)) == set()


class TestMustNotFire:
    def test_clean_grid_is_quiet(self) -> None:
        assert _fired(_grid_words(), _md(list(HEADER), [list(r) for r in ROWS])) == set()

    def test_label_value_pair_at_column_spacing_is_a_real_cell(self) -> None:
        # "Model 2": the number is a value in its own lane, printed a column away.
        words = _words(COL_XS[1] - COL_XS[0], label="Model")  # a column-wide gap
        assert _fired(words, _md_rows(["GDP Model", "2001", *NUMS])) == set()

    def test_words_without_line_indices_abstain_and_never_raise(self) -> None:
        words = [w[:5] for w in _words(WORD_SPACE)]
        fired = _fired(words, _md_rows(SPLIT))
        assert ship_gate.GATE_ERROR not in fired and SP not in fired

    def test_table_only_page_abstains_for_lack_of_spacing_evidence(self) -> None:
        # Per-cell lines inside the table are column pitch, not a word space.
        assert SP not in _fired(_words(WORD_SPACE, prose_lines=0), _md_rows(SPLIT))

    def test_a_number_without_a_letter_word_before_it_is_not_a_phrase(self) -> None:
        words = _words(WORD_SPACE, label="1999")
        assert SP not in _fired(words, _md_rows(["GDP 1999", "2001", *NUMS]))

    def test_a_word_the_grid_dropped_is_not_this_predicate(self) -> None:
        assert SP not in _fired(_words(WORD_SPACE), _md_rows(["GDP", "2001", *NUMS]))

    def test_a_symbol_before_the_number_is_not_a_word(self) -> None:
        # A bare "$" has no letter; a sign glyph split from its number is sign_detached's case.
        words = _words(WORD_SPACE, label="$")
        assert SP not in _fired(words, _md_rows(["GDP $", "2001", *NUMS]))

    def test_two_words_split_across_cells_are_not_a_number_phrase(self) -> None:
        # The second word must be a number: "GDP March" in two cells is a label problem, not this one.
        words = _words(WORD_SPACE, label="March")
        assert SP not in _fired(words, _md_rows(["GDP", "March 2001", *NUMS]))
