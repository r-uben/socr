"""Tests for the pure binding oracle (``socr.tables.binding``).

Hermetic: synthetic ``page.get_text("words")``-shaped tuples and literal
markdown strings only. No PDFs, no corpus, no provider.

The module does not exist on ``origin/main`` at all, so every test here would
fail with an ``ImportError`` there — that is not what makes these tests
meaningful. Each test also carries a comment stating the BEHAVIOURAL claim it
pins, so the non-vacuity case is that a body-swapped or logic-stripped
version of this same module (imports intact) fails the assertion, not merely
that the module is absent.
"""

from __future__ import annotations

from collections import Counter

import pytest

from socr.tables.binding import bind, parse_grid


def w(x0: float, y0: float, x1: float, y1: float, text: str) -> tuple:
    """One ``page.get_text("words")`` tuple: (x0, y0, x1, y1, text, block, line, word)."""
    return (x0, y0, x1, y1, text, 0, 0, 0)


def nums(markdown: str) -> list[str]:
    """All bare numeric-looking tokens in a markdown table, cell order irrelevant."""
    out = []
    for line in markdown.splitlines():
        if "|" not in line:
            continue
        for cell in line.strip().strip("|").split("|"):
            cell = cell.strip()
            if cell and cell.replace(".", "", 1).replace("-", "", 1).isdigit():
                out.append(cell)
    return out


# ---------------------------------------------------------------------------
# 1. THE MULTISET-KILLER
# ---------------------------------------------------------------------------


def test_multiset_killer_separates_flattened_from_correct():
    """A multiset-based implementation cannot pass this test by construction:
    ``correct`` and ``flattened`` share an identical numeric multiset (proven
    first), yet the binder must agree with one and contradict the other,
    because the flattening swaps which COLUMN each Coef-row value sits in."""
    words = [
        w(140, 70, 170, 80, "OLS"),
        w(240, 70, 270, 80, "IV"),
        w(50, 100, 90, 110, "Coef"),
        w(150, 100, 180, 110, "1.10"),
        w(250, 100, 280, 110, "1.11"),
        w(50, 130, 90, 140, "SE"),
        w(150, 130, 180, 140, "0.05"),
        w(250, 130, 280, 140, "0.06"),
    ]
    correct = """
|      | OLS  | IV   |
|------|------|------|
| Coef | 1.10 | 1.11 |
| SE   | 0.05 | 0.06 |
"""
    flattened = """
|      | OLS  | IV   |
|------|------|------|
| Coef | 1.11 | 1.10 |
| SE   | 0.05 | 0.06 |
"""
    assert Counter(nums(correct)) == Counter(nums(flattened)), (
        "fixture is not honest: correct/flattened must share a multiset"
    )

    result_correct = bind(words, correct)
    result_flattened = bind(words, flattened)

    assert result_correct.structural_agreement is True
    assert result_correct.contradicted_cells == []

    assert result_flattened.structural_agreement is False
    assert len(result_flattened.contradicted_cells) == 2
    tokens = {(c.native_token, c.model_token) for c in result_flattened.contradicted_cells}
    assert tokens == {("1.10", "1.11"), ("1.11", "1.10")}


# ---------------------------------------------------------------------------
# 2. Empty cell is a slot, not an absence
# ---------------------------------------------------------------------------


def test_empty_cell_is_a_first_class_matched_slot():
    """An empty native lane bound to an empty candidate cell must be recorded
    as a matched (empty) slot, not silently dropped from the result the way
    ``LabeledRow.values`` would compact it away."""
    words = [
        w(140, 70, 170, 80, "OLS"),
        w(240, 70, 270, 80, "IV"),
        w(50, 100, 90, 110, "Coef"),
        w(150, 100, 180, 110, "1.10"),
        w(250, 100, 280, 110, "1.11"),
        w(50, 130, 90, 140, "Div"),
        w(150, 130, 180, 140, "0.02"),
        # no word at all in the IV lane for the Div row: a genuine empty cell
    ]
    md = """
|      | OLS  | IV   |
|------|------|------|
| Coef | 1.10 | 1.11 |
| Div  | 0.02 |      |
"""
    result = bind(words, md)
    assert result.structural_agreement is True
    empty_matches = [m for m in result.matched_cells if m.row_path[-1] == "Div" and m.value is None]
    assert len(empty_matches) == 1
    assert result.native_unbound == []
    assert result.model_unbound == []


# ---------------------------------------------------------------------------
# 3. Dropped digit surfaces as native-unbound
# ---------------------------------------------------------------------------


def test_dropped_digit_is_native_unbound():
    """A native number with no corresponding candidate value is C4's
    dropped-digit signal: it must land in ``native_unbound``, never silently
    treated as a match and never as an ordinary value contradiction."""
    words = [
        w(140, 70, 170, 80, "OLS"),
        w(240, 70, 270, 80, "IV"),
        w(50, 100, 90, 110, "Coef"),
        w(150, 100, 180, 110, "1.10"),
        w(250, 100, 280, 110, "1.11"),
        w(50, 130, 90, 140, "SE"),
        w(150, 130, 180, 140, "0.05"),
        w(250, 130, 280, 140, "0.03"),  # present natively...
    ]
    md = """
|      | OLS  | IV   |
|------|------|------|
| Coef | 1.10 | 1.11 |
| SE   | 0.05 |      |
"""
    # ...but dropped from the candidate's IV column for the SE row.
    result = bind(words, md)
    assert result.structural_agreement is False
    assert len(result.native_unbound) == 1
    dropped = result.native_unbound[0]
    assert dropped.token == "0.03"
    assert dropped.row_path[-1] == "SE"


# ---------------------------------------------------------------------------
# 4. Ambiguous geometry yields zero convictions
# ---------------------------------------------------------------------------


def test_ambiguous_lanes_never_convict():
    """Two numeric x-positions 10pt apart cluster as distinct lanes (> the
    6pt cluster tolerance) but are NOT well-separated (< the 18pt
    well-separated gap) — exactly the geometry C3 says must defer, not
    convict, even though the candidate's second value is wrong."""
    words = [
        w(50, 100, 90, 110, "X"),
        w(150, 100, 180, 110, "1.23"),
        w(160, 100, 190, 110, "4.56"),
    ]
    md = """
|   | A    | B    |
|---|------|------|
| X | 1.23 | 9.99 |
"""
    result = bind(words, md)
    assert result.contradicted_cells == []
    assert result.native_unbound == []
    assert result.model_unbound == []
    assert result.matched_cells == []
    assert result.ambiguous_count >= 1


# ---------------------------------------------------------------------------
# 5. Two "Total" rows on one page do not cross-bind
# ---------------------------------------------------------------------------


def test_two_total_rows_do_not_cross_bind():
    """Panel A's Total and Panel B's Total share a leaf label. Row paths
    prefixed by their panel keep them distinct, and a contradiction in one
    must report ITS OWN native value, never the other panel's."""
    words = [
        w(150, 70, 180, 80, "OLS"),
        w(250, 70, 280, 80, "IV"),
        w(50, 100, 100, 110, "Panel"),
        w(105, 100, 115, 110, "A"),
        w(50, 130, 90, 140, "Total"),
        w(150, 130, 180, 140, "10.0"),
        w(250, 130, 280, 140, "20.0"),
        w(50, 160, 100, 170, "Panel"),
        w(105, 160, 115, 170, "B"),
        w(50, 190, 90, 200, "Total"),
        w(150, 190, 180, 200, "30.0"),
        w(250, 190, 280, 200, "40.0"),
    ]
    md = """
|         | OLS  | IV   |
|---------|------|------|
| Panel A |      |      |
| Total   | 10.0 | 99.9 |
| Panel B |      |      |
| Total   | 30.0 | 40.0 |
"""
    result = bind(words, md)
    assert len(result.contradicted_cells) == 1
    bad = result.contradicted_cells[0]
    assert bad.row_path == ("Panel A", "Total")
    assert bad.native_token == "20.0"
    assert bad.model_token == "99.9"
    # Panel B's Total is untouched by Panel A's contradiction.
    panel_b_matches = [m for m in result.matched_cells if m.row_path[0] == "Panel B"]
    assert len(panel_b_matches) == 2
    assert not any(m.row_path[0] == "Panel B" for c in result.contradicted_cells for m in [c])


# ---------------------------------------------------------------------------
# 6. Spanning header paths expand with provenance
# ---------------------------------------------------------------------------


def test_spanning_header_expands_to_both_lanes_with_provenance():
    """A native header word whose bbox covers both lane intervals, confirmed
    by the candidate's matching (blank-followed) header cell, must appear in
    BOTH lanes' column paths with ``spans_lanes == 2`` recorded."""
    words = [
        # "Model" spans both lanes: x0=130..270 covers both 150 and 250 centres.
        w(130, 60, 270, 70, "Model"),
        w(140, 80, 170, 90, "OLS"),
        w(240, 80, 270, 90, "IV"),
        w(50, 110, 90, 120, "Coef"),
        w(150, 110, 180, 120, "1.10"),
        w(250, 110, 280, 120, "1.11"),
    ]
    md = """
|      | Model |      |
|      | OLS   | IV   |
|------|-------|------|
| Coef | 1.10  | 1.11 |
"""
    result = bind(words, md)
    paths = {chp.lane: chp for chp in result.column_header_paths}
    assert paths[0].path == ("Model", "OLS")
    assert paths[1].path == ("Model", "IV")
    assert paths[0].spans_lanes == 2
    assert paths[1].spans_lanes == 2
    assert paths[0].unverifiable is False
    assert paths[1].unverifiable is False


# ---------------------------------------------------------------------------
# A1 — strict grid parser: pipe-bearing prose is not a phantom table
# ---------------------------------------------------------------------------


def test_strict_parser_rejects_pipe_bearing_prose_without_a_real_separator():
    prose = """
Revenue growth | margin expansion | headcount reduction all contributed to
the quarter's result | which beat consensus | by a wide margin.
"""
    assert parse_grid(prose) is None


def test_strict_parser_accepts_a_real_table():
    md = """
| a | b |
|---|---|
| 1 | 2 |
"""
    grid = parse_grid(md)
    assert grid is not None
    assert grid.rows == (("1", "2"),)


# ---------------------------------------------------------------------------
# 7. A dropped NATIVE row must not vanish silently (BLOCKING 1)
# ---------------------------------------------------------------------------


def test_dropped_native_row_is_surfaced_not_silently_agreed():
    """Native has three rows (A, B, C); the candidate only carries A and C —
    row B was dropped entirely. The anchors either side of the gap (A, C)
    still line up fine, so a binder that only checks "did every CANDIDATE
    row get bound" sees nothing wrong. That is exactly the blind spot this
    module exists to close: B's values must surface as ``native_unbound``
    and the result must not claim ``structural_agreement``."""
    words = [
        w(140, 70, 170, 80, "OLS"),
        w(240, 70, 270, 80, "IV"),
        w(50, 100, 90, 110, "A"),
        w(150, 100, 180, 110, "1.0"),
        w(250, 100, 280, 110, "2.0"),
        w(50, 130, 90, 140, "B"),
        w(150, 130, 180, 140, "3.0"),
        w(250, 130, 280, 140, "4.0"),
        w(50, 160, 90, 170, "C"),
        w(150, 160, 180, 170, "5.0"),
        w(250, 160, 280, 170, "6.0"),
    ]
    md = """
|   | OLS | IV  |
|---|-----|-----|
| A | 1.0 | 2.0 |
| C | 5.0 | 6.0 |
"""
    result = bind(words, md)
    assert result.structural_agreement is False
    assert result.row_binding_unverifiable is True
    dropped_tokens = {u.token for u in result.native_unbound}
    assert dropped_tokens == {"3.0", "4.0"}
    assert all(u.row_path[-1] == "B" for u in result.native_unbound)
    # A and C themselves are untouched: both still bind and match cleanly.
    assert result.contradicted_cells == []
    a_c_matches = {(m.row_path[-1], m.value) for m in result.matched_cells}
    assert a_c_matches == {("A", "1.0"), ("A", "2.0"), ("C", "5.0"), ("C", "6.0")}


# ---------------------------------------------------------------------------
# 8. A3 — decimal precision is not normalised away (MAJOR 3)
# ---------------------------------------------------------------------------


def test_trailing_zero_precision_is_a_contradiction_not_a_match():
    """Native '1.10' and candidate '1.1' are numerically equal but
    typographically distinct precision claims (A3): the binder must treat
    them as a CONTRADICTION, not silently normalise '1.10' down to '1.1'
    and call it a match."""
    words = [
        w(50, 100, 90, 110, "Coef"),
        w(150, 100, 180, 110, "1.10"),
    ]
    md = """
|      | OLS |
|------|-----|
| Coef | 1.1 |
"""
    result = bind(words, md)
    assert result.structural_agreement is False
    assert len(result.contradicted_cells) == 1
    bad = result.contradicted_cells[0]
    assert bad.native_token == "1.10"
    assert bad.model_token == "1.1"
    assert result.matched_cells == []


# ---------------------------------------------------------------------------
# 9. A numeric row-label stub does not inflate the lane count (MAJOR 4)
# ---------------------------------------------------------------------------


def test_numeric_row_label_is_not_mistaken_for_a_data_lane():
    """A year used as the row's own stub label ('2020', '2021') is numeric
    by ``is_numeric_token`` and would cluster into its own lane exactly like
    a genuine data column — but it is always the LEFTMOST word in its row,
    which a real data lane never is (the row's label text — even a numeric
    one — always precedes the data). The lane count must stay at 2 (OLS,
    IV), not 3, and the year itself must show up as the row's label, not be
    silently dropped from the native multiset used for row anchoring."""
    words = [
        w(140, 70, 170, 80, "OLS"),
        w(240, 70, 270, 80, "IV"),
        w(50, 100, 90, 110, "2020"),
        w(150, 100, 180, 110, "1.10"),
        w(250, 100, 280, 110, "1.11"),
        w(50, 130, 90, 140, "2021"),
        w(150, 130, 180, 140, "2.20"),
        w(250, 130, 280, 140, "2.21"),
    ]
    md = """
|      | OLS  | IV   |
|------|------|------|
| 2020 | 1.10 | 1.11 |
| 2021 | 2.20 | 2.21 |
"""
    result = bind(words, md)
    assert result.column_binding_unverifiable is False
    assert result.row_binding_unverifiable is False
    assert result.structural_agreement is True
    matches = {(m.row_path[-1], m.value) for m in result.matched_cells}
    assert matches == {
        ("2020", "1.10"),
        ("2020", "1.11"),
        ("2021", "2.20"),
        ("2021", "2.21"),
    }


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
