"""GH-951: a caption or notes word cut in two across neighbouring cells of one grid row.

Each test pins a DIFFERENCE between two runs that change only the thing under test. The
gate is pure (no provider, no ollama), so nothing here depends on the CI environment.
"""

from __future__ import annotations

from native_table_fixtures import HEADER, ROWS
from test_gh916_native_ship_gate import _gate, _md, _words

from socr.tables import ship_gate

SPLIT = ship_gate.WORD_SPLIT_ACROSS_CELLS
#: The source: the table, then a notes line below it.
NOTES = ["Notes", "Standard", "errors", "in", "brackets"]
#: The same line with the word "Standard" cut across two cells.
CUT = ["Notes", "Standa", "rd", "errors", "in", "brackets"]
SOURCE = _words([HEADER] + ROWS + [NOTES])


def _faults(block: list[list[str]], words=SOURCE) -> list:
    """The predicate alone, on one block."""
    return ship_gate.word_split_across_cells_faults([block], ship_gate._source_rows(words))


def _grid(notes: list[str]) -> str:
    return _md(HEADER, ROWS + [notes])


def test_caption_split_mid_word_fires_and_whole_words_are_quiet() -> None:
    assert _faults([CUT])
    assert not _faults([NOTES])


def test_end_to_end_difference_on_a_full_grid() -> None:
    split = _gate(SOURCE, _grid(["Notes", "Standa", "rd errors", "in", "brackets"]))
    whole = _gate(SOURCE, _grid(["Notes", "Standard", "errors in", "brackets", ""]))
    assert SPLIT in split
    assert SPLIT not in whole


def test_hyphenated_compound_cut_at_the_hyphen_fires() -> None:
    """Decision: ``well-`` + ``known`` joins to the source word ``well-known`` and fires.

    The source keeps the compound as one word, and a table cell does not end in a hyphen
    for a reason other than a wrapped word.
    """
    words = _words([["A", "well-known", "result"]])
    assert _faults([["A", "well-", "known", "result"]], words)
    assert not _faults([["A", "well-known", "result"]], words)


def test_numeric_split_is_quiet() -> None:
    """Difference: the same cut on a number is quiet; the numeric predicates own it."""
    assert not _faults([["Total", "12.", "50", "x"]], _words([["Total", "12.50", "x"]]))
    assert _faults([["Total", "ab", "cd", "x"]], _words([["Total", "abcd", "x"]]))


def test_word_present_elsewhere_in_the_grid_is_quiet_known_hole() -> None:
    """Known hole: a split word that also occurs whole in the block cannot be told from a
    correct cell pair, so the predicate abstains. Pinned so widening it is deliberate."""
    assert _faults([CUT])
    assert not _faults([["Standard", "errors"], CUT])


def test_ligature_in_the_source_is_folded() -> None:
    """The PDF's ``Staff`` with U+FB00 and the grid's ``Sta`` + ``ff`` are the same word."""
    words = _words([["Total", "Staﬀ", "x"]])
    assert _faults([["Total", "Sta", "ff", "x"]], words)
    assert not _faults([["Total", "Staﬀ", "x"]], words)


def test_cells_of_different_rows_never_join() -> None:
    assert not _faults([["Notes", "Standa"], ["rd", "errors"]])
    # positive counterpart: the same cells in ONE row, over the line that holds the word
    assert _faults([CUT])


def _lines(*rows: list[str]) -> list:
    """A source with each row of words on a line of its own."""
    return _words(list(rows))


def test_word_elsewhere_on_the_page_is_not_evidence() -> None:
    """Fixed-source control: the header's own line reads ``Pre tax``; ``Pretax`` sits on
    another line. Only the source line the row sits on may vouch for the join."""
    header = [["Pre", "tax", "Income"]]
    collision = _lines(["Pre", "tax", "Income"], ["Pretax", "profit", "grew"])
    assert not _faults(header, collision)
    # control: the SAME row over a line that really holds one word `Pretax` fires
    assert _faults(header, _lines(["Pretax", "Income"], ["Pretax", "profit", "grew"]))
    # unrelated perturbation of the source leaves the verdict unchanged
    perturbed = collision + _words([["Zebra", "stripes"]], start_line=9, y_start=900.0)
    assert not _faults(header, perturbed)
    assert _faults(
        header, _lines(["Pretax", "Income"]) + _words([["Zebra"]], start_line=9, y_start=900.0)
    )


def test_a_word_spanning_an_empty_cell_fires_only_when_one_word_spans_it() -> None:
    row = [["non", "", "linear"]]
    assert _faults(row, _lines(["nonlinear"]))
    assert not _faults(row, _lines(["non", "linear"], ["x", "nonlinear"]))


def test_number_plus_unit_never_fires() -> None:
    assert not _faults([["5", "kg"]], _lines(["5kg"]))
    assert _faults([["ab", "cd"]], _lines(["abcd"]))


def test_number_as_right_fragment_never_fires() -> None:
    assert not _faults([["kg", "5"]], _lines(["kg5"]))


def test_tokens_inside_one_cell_are_not_a_cell_split() -> None:
    assert not _faults([["ab cd"]], _lines(["abcd"]))
    assert _faults([["ab", "cd"]], _lines(["abcd"]))


def test_only_the_cut_the_line_places_fires() -> None:
    """Row ``ab|cd|ab|cd`` over the line ``ab cd abcd``: the join sits at the third and
    fourth cells only, so exactly one fault, not one per pair that spells ``abcd``."""
    row = [["ab", "cd", "ab", "cd"]]
    assert len(_faults(row, _lines(["ab", "cd", "abcd"]))) == 1


def test_a_row_must_cover_its_whole_source_line() -> None:
    """The cut word on a line that continues past the row is not that row's evidence."""
    assert not _faults([["Standa", "rd"]], _lines(["Standard", "x"]))
    assert _faults([["Standa", "rd"]], _lines(["Standard"]))


def test_hyphen_split_needs_the_positioned_word_to_be_the_compound() -> None:
    row = [["A", "well-", "known"]]
    assert _faults(row, _lines(["A", "well-known"]))
    assert not _faults(row, _lines(["A", "well-", "known"], ["well-known"]))


def test_short_tuples_and_empty_input_abstain_and_never_raise() -> None:
    short = [(0.0, 0.0, 1.0, 1.0, "Standard"), (0.0, 0.0, 1.0, 1.0, "x")]
    assert _faults([["Standa", "rd", "x"]], short)
    assert _faults([["only"]], short) == []
    assert ship_gate.word_split_across_cells_faults([], {}) == []
    assert ship_gate.word_split_across_cells_faults([[[]]], {0: [(0, 0, 1, 1, "a")]}) == []


def test_only_adds_faults(monkeypatch) -> None:
    """A fault another predicate already raised is kept alongside the new one."""
    md = _grid(["Notes", "Standa", "rd errors", "in", "brackets"])
    full = ship_gate.native_ship_gate(SOURCE, md, line_dirs=None)
    monkeypatch.setattr(ship_gate, "word_split_across_cells_faults", lambda *a, **k: [])
    without = ship_gate.native_ship_gate(SOURCE, md, line_dirs=None)
    assert ship_gate.DIRECTION_UNAVAILABLE in {f["predicate"] for f in without}
    assert all(f in full for f in without)
    assert {f["predicate"] for f in full if f not in without} == {SPLIT}
