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

import json
from collections import Counter
from pathlib import Path

import pytest

from socr.tables.binding import BindingResult, bind, parse_grid


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


def _assert_bidirectional_coverage(result, native_total: int, candidate_total: int) -> None:
    """I1 BIDIRECTIONALITY: every native data cell must be accounted for as
    matched, contradicted, native_unbound, or ambiguous -- and likewise
    every candidate data cell as matched, contradicted, model_unbound, or
    ambiguous. A cell that is neither bound NOR reported unbound (NOR
    ambiguous) has been dropped invisibly -- the exact defect shape
    BLOCKING 1 (round 1), HIGH 1, and HIGH 2 each independently turned out
    to be.

    Applies even to tables where `column_binding_unverifiable` is True (the
    lane/column-mismatch path HIGH 1's fix salvages): a lane/column the DP
    salvage maps to a counterpart is not claimed as a binding (never
    matched/contradicted -- the table stays column_binding_unverifiable
    regardless), but it is not left unreported either. It is counted as
    ambiguous, same as any other "known geometry, not confidently
    convictable either way" cell -- an earlier version of this helper
    excluded that branch from the invariant entirely, which was itself an
    unreported third state (bound, reported-unbound, or silently mapped)
    hiding behind the very branch I1 was written to fix."""
    matched = len(result.matched_cells)
    contradicted = len(result.contradicted_cells)
    native_seen = matched + contradicted + len(result.native_unbound) + result.ambiguous_count
    candidate_seen = matched + contradicted + len(result.model_unbound) + result.ambiguous_count
    assert native_seen == native_total, (
        f"native side: accounted for {native_seen}, expected {native_total} "
        f"(matched={matched} contradicted={contradicted} "
        f"native_unbound={len(result.native_unbound)} ambiguous={result.ambiguous_count})"
    )
    assert candidate_seen == candidate_total, (
        f"candidate side: accounted for {candidate_seen}, expected {candidate_total} "
        f"(matched={matched} contradicted={contradicted} "
        f"model_unbound={len(result.model_unbound)} ambiguous={result.ambiguous_count})"
    )


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


def test_unparseable_or_empty_binding_fails_closed():
    """No evidence is not structural agreement. A malformed candidate must
    fail closed, and the public result's empty/default state must not be a
    pass if a future early-return path constructs one directly."""
    malformed = bind([], "not a markdown table")

    assert malformed.structural_agreement is False
    assert BindingResult().structural_agreement is False


# ---------------------------------------------------------------------------
# GH-273: numeric row anchors must verify the candidate row-label stub
# ---------------------------------------------------------------------------


def test_numeric_row_anchors_reject_shifted_row_labels():
    """GH-273: correct and shifted grids contain the exact same six numbers.
    Numeric row anchoring therefore maps each candidate row to the right native
    band in both cases; the candidate stub must then be checked against that
    anchored native row rather than being trusted or used as the anchor itself.
    """
    words = [
        w(140, 70, 170, 80, "OLS"),
        w(240, 70, 270, 80, "IV"),
        # The coefficient row is genuinely unlabelled.
        w(150, 100, 180, 110, "1.10"),
        w(250, 100, 280, 110, "1.11"),
        w(50, 130, 100, 140, "Large T"),
        w(150, 130, 180, 140, "0.05"),
        w(250, 130, 280, 140, "0.06"),
        w(50, 160, 100, 170, "Small T"),
        w(150, 160, 180, 170, "0.07"),
        w(250, 160, 280, 170, "0.08"),
    ]
    correct = """
|         | OLS  | IV   |
|---------|------|------|
|         | 1.10 | 1.11 |
| Large T | 0.05 | 0.06 |
| Small T | 0.07 | 0.08 |
"""
    shifted = """
|         | OLS  | IV   |
|---------|------|------|
| Large T | 1.10 | 1.11 |
|         | 0.05 | 0.06 |
| Small T | 0.07 | 0.08 |
"""
    assert Counter(nums(correct)) == Counter(nums(shifted))

    correct_result = bind(words, correct)
    shifted_result = bind(words, shifted)

    assert correct_result.structural_agreement is True
    assert correct_result.row_label_contradictions == []
    assert len(correct_result.matched_cells) == 6

    assert shifted_result.structural_agreement is False
    assert len(shifted_result.matched_cells) == 6
    assert [(m.row_path, m.candidate_label) for m in shifted_result.row_label_contradictions] == [
        (("",), "Large T"),
        (("Large T",), ""),
    ]


def test_symbolic_row_labels_fail_closed_as_unverifiable():
    """The shared prose-label normalizer cannot prove mathematical notation
    equivalent (native ``β`` versus model ``$\\beta$``). Do not falsely report
    a contradiction, but do not let an unchecked label pass either."""
    words = [
        w(140, 70, 170, 80, "OLS"),
        w(50, 100, 90, 110, "β"),
        w(150, 100, 180, 110, "10.0"),
    ]
    candidate = r"""
|         | OLS  |
|---------|------|
| $\beta$ | 10.0 |
"""

    result = bind(words, candidate)

    assert result.row_label_contradictions == []
    assert result.row_label_unverifiable is True
    assert result.structural_agreement is False


def test_gh585_sibling_latex_greek_and_word_command_labels_are_not_contradictions():
    """GH-585: a Greek-letter command (``\\Delta``) and a plain alphabetic
    word command (``\\log``) are sibling presentation classes to the GH-582
    ``\\text{}``/``^`` wrap — the ladder-corpus doc05 held pair. On ``main``
    (GH-582 fix only) this label is convicted as a shifted/invented label
    because ``\\Delta`` survives the shared normalizer as the spelled-out
    ASCII word ``delta``, which the native ``∆`` side never carries."""
    words = [
        w(140, 70, 170, 80, "OLS"),
        w(50, 100, 140, 110, "∆log Comm. price (3m)"),
        w(150, 100, 180, 110, "0.95"),
    ]
    candidate = r"""
|                                          | OLS  |
|------------------------------------------|------|
| $\Delta \log \text{ Comm. price (3m)}$   | 0.95 |
"""

    result = bind(words, candidate)

    assert result.row_label_contradictions == []
    assert result.row_label_unverifiable is False
    assert result.structural_agreement is True


def test_gh585_unsupported_word_command_does_not_falsely_agree_through_bind():
    """GH-585 review round 3: retaining the backslash at the
    ``strip_math_presentation`` helper is not, by itself, evidence that
    ``\\logx`` stays distinct through the actual binder compare --
    ``normalize_label`` strips any leftover backslash from every label
    unconditionally, so a naive fix could still let ``\\logx`` silently
    agree with the bare word ``logx`` (or, worse, with the real operator
    ``\\log`` on the shorter label). Neither must agree: the native row is
    a genuine contradiction in both cases."""
    words = [
        w(140, 70, 170, 80, "OLS"),
        w(50, 100, 140, 110, "logx y"),
        w(150, 100, 180, 110, "0.95"),
    ]
    against_bare_word = r"""
|            | OLS  |
|------------|------|
| $\logx y$  | 0.95 |
"""
    against_real_operator = r"""
|           | OLS  |
|-----------|------|
| $\log y$  | 0.95 |
"""

    for candidate in (against_bare_word, against_real_operator):
        result = bind(words, candidate)
        assert len(result.row_label_contradictions) == 1
        assert result.row_label_unverifiable is False
        assert result.structural_agreement is False


# ---------------------------------------------------------------------------
# GH-582: inline-math wrapping is presentation, not a value/label change
# ---------------------------------------------------------------------------


def test_inline_math_wrapped_numeric_cells_bind_identically_to_plain():
    """GH-582: wrapping every numeric cell in ``$...$`` must not change a
    single matched/contradicted cell. Same native geometry, same candidate
    values, differing only in whether each cell carries inline-math
    delimiters -- the two ``BindingResult``s must be identical on the
    fields that record what was compared, and the wrapped run must convict
    zero cells."""
    words = [
        w(140, 70, 170, 80, "OLS"),
        w(50, 100, 100, 110, "Large T"),
        w(150, 100, 180, 110, "0.05"),
        w(50, 130, 100, 140, "Small T"),
        w(150, 130, 180, 140, "0.07"),
    ]
    plain = """
|         | OLS  |
|---------|------|
| Large T | 0.05 |
| Small T | 0.07 |
"""
    wrapped = """
|         | OLS  |
|---------|------|
| Large T | $0.05$ |
| Small T | $0.07$ |
"""

    plain_result = bind(words, plain)
    wrapped_result = bind(words, wrapped)

    assert wrapped_result.contradicted_cells == []
    assert [(m.row_path, m.value) for m in plain_result.matched_cells] == [
        (m.row_path, m.value) for m in wrapped_result.matched_cells
    ]
    assert plain_result.contradicted_cells == wrapped_result.contradicted_cells
    assert plain_result.row_label_contradictions == wrapped_result.row_label_contradictions
    assert plain_result.structural_agreement == wrapped_result.structural_agreement is True


def test_inline_math_wrapped_numeric_cell_is_bound_then_compared_not_convicted():
    """GH-582: the previous fixture wraps every numeric cell, so with the
    three source edits reverted, ``_candidate_row_multiset`` sees no
    numeric candidate row at all and row anchoring fails before the cell
    walk ever runs -- it does not exercise the reported defect
    (``binding.py`` ~1572: an already-BOUND ``$-0.06$`` cell convicted as
    "not a numeric token at all"). Here the OLS lane is plain, so the row
    binds through it on both the reverted and the fixed source; only the
    IV lane is wrapped, so the cell walk reaches the actual comparison.
    Reverting the three source edits reproduces the issue's exact
    contradiction pair (``−0.06``, ``$-0.06$``); the fix clears it."""
    words = [
        w(140, 70, 170, 80, "OLS"),
        w(240, 70, 270, 80, "IV"),
        w(50, 100, 100, 110, "Large T"),
        w(150, 100, 180, 110, "0.05"),
        w(250, 100, 280, 110, "−0.06"),
    ]
    candidate = """
|         | OLS  | IV      |
|---------|------|---------|
| Large T | 0.05 | $-0.06$ |
"""

    result = bind(words, candidate)

    assert not result.row_binding_unverifiable
    assert result.contradicted_cells == []
    assert [(m.row_path, m.value) for m in result.matched_cells] == [
        (("Large T",), "0.05"),
        (("Large T",), "−0.06"),
    ]
    assert result.structural_agreement is True


def test_text_wrapped_header_style_row_label_is_not_a_contradiction():
    """GH-582: a row label re-typeset as ``\\text{}``/``^`` math (the
    Pflueger-Rinaldi corpus shape: native ``Adjusted R2`` vs. model
    ``Adjusted $\\text{R}^2$``) must fold to the same key as its plain-text
    rendering, not convict as a shifted/invented label."""
    words = [
        w(140, 70, 170, 80, "OLS"),
        w(50, 100, 140, 110, "Adjusted R2"),
        w(150, 100, 180, 110, "0.95"),
    ]
    candidate = r"""
|                        | OLS  |
|------------------------|------|
| Adjusted $\text{R}^2$  | 0.95 |
"""

    result = bind(words, candidate)

    assert result.row_label_contradictions == []
    assert result.contradicted_cells == []
    assert result.structural_agreement is True


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
    # I1: this is precisely the fixture round 1 shipped one-sided -- had the
    # bidirectionality invariant been checked here then, it would have
    # failed before the round-2/round-3 findings ever needed a reviewer.
    _assert_bidirectional_coverage(result, native_total=6, candidate_total=4)


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


# ---------------------------------------------------------------------------
# 10. Decorated numerics (bold, significance stars, currency) still cluster
#     into lanes and bind normally — the lane predicate must match the row/
#     header predicate (is_numeric_token), not the raw undecorated regex.
# ---------------------------------------------------------------------------


def test_decorated_native_numerics_yield_a_non_zero_lane_count_and_bind():
    """Markdown bold (``**1.2**``) and a Unicode significance star
    (``0.05∗∗``) are the ordinary shape of a typeset econometrics table
    (GH-103, GH-206). If lane clustering uses a predicate that does not
    strip presentation the way row/header parsing does, every decorated
    native value fails to cluster into any lane, lane_count collapses to
    zero, and the whole table becomes ``column_binding_unverifiable`` —
    silently abstaining everywhere instead of binding normally."""
    words = [
        w(140, 70, 170, 80, "OLS"),
        w(50, 100, 90, 110, "Coef"),
        w(150, 100, 180, 110, "**1.2**"),
        w(50, 130, 90, 140, "SE"),
        w(150, 130, 180, 140, "0.05∗∗"),
    ]
    md = """
|      | OLS  |
|------|------|
| Coef | 1.2  |
| SE   | 0.05 |
"""
    result = bind(words, md)
    assert result.column_binding_unverifiable is False
    assert result.structural_agreement is True
    matches = {(m.row_path[-1], m.value) for m in result.matched_cells}
    # The reported value is the native token AS PRINTED (evidence stays
    # verbatim); the comparison that decided it matched normalizes both
    # sides internally.
    assert matches == {("Coef", "**1.2**"), ("SE", "0.05∗∗")}


def test_decorated_native_numeric_contradicted_by_candidate_is_a_contradiction():
    """A decorated native value that disagrees with the candidate must
    produce a real CONTRADICTION, not a vacuous abstention. This is the
    test that proves the oracle actually works on the table shape it was
    built for, not just that it fails safe on it."""
    words = [
        w(140, 70, 170, 80, "OLS"),
        w(50, 100, 90, 110, "Coef"),
        w(150, 100, 180, 110, "$1.10"),
        w(50, 130, 90, 140, "SE"),
        w(150, 130, 180, 140, "0.05∗∗"),
    ]
    md = """
|      | OLS    |
|------|--------|
| Coef | 1.10   |
| SE   | 999.99 |
"""
    result = bind(words, md)
    assert result.column_binding_unverifiable is False
    assert result.structural_agreement is False
    assert len(result.contradicted_cells) == 1
    bad = result.contradicted_cells[0]
    assert bad.row_path[-1] == "SE"
    assert bad.native_token == "0.05∗∗"
    assert bad.model_token == "999.99"
    # Coef's own decorated value still matches cleanly (value reported
    # verbatim, as printed natively, not normalized).
    assert any(m.row_path[-1] == "Coef" and m.value == "$1.10" for m in result.matched_cells)


# ---------------------------------------------------------------------------
# 11. A label-less data matrix must not have its first column mistaken for
#     a stub — "always leftmost in its row" alone is not sufficient
#     evidence (MAJOR 4 follow-up: caught by an independent reviewer's own
#     adversarial fixture while re-checking the stub-lane fix).
# ---------------------------------------------------------------------------


def test_unlabeled_data_matrix_keeps_its_first_column():
    """A pure numeric matrix with no row-label text at all makes its first
    (leftmost) data column trivially "always leftmost in its row" too —
    exactly the same shape a genuine numeric row-label stub has. Removing
    it as a stub would silently turn a fully verifiable table into a
    completely unverifiable one. Stub removal must only fire when the
    candidate's own column count confirms it is needed, not on the
    "always leftmost" geometry signal alone."""
    words = [
        w(150, 70, 180, 80, "OLS"),
        w(250, 70, 280, 80, "IV"),
        w(150, 100, 180, 110, "1.10"),
        w(250, 100, 280, 110, "1.11"),
        w(150, 130, 180, 140, "2.20"),
        w(250, 130, 280, 140, "2.21"),
    ]
    md = """
|   | OLS  | IV   |
|---|------|------|
|   | 1.10 | 1.11 |
|   | 2.20 | 2.21 |
"""
    result = bind(words, md)
    assert result.column_binding_unverifiable is False
    assert result.structural_agreement is True
    assert result.contradicted_cells == []
    matches = {(m.col_path[-1], m.value) for m in result.matched_cells}
    assert matches == {("OLS", "1.10"), ("IV", "1.11"), ("OLS", "2.20"), ("IV", "2.21")}


# ---------------------------------------------------------------------------
# 12. I1 BIDIRECTIONALITY -- round-2 review: both fixes above still went in
#     one-sided. Round 1 fixed the native side (BLOCKING 1); the candidate
#     side was never implemented (HIGH 2), and the column-mismatch early
#     return dropped BOTH sides' cell signal for the whole table (HIGH 1).
#     MEDIUM 3 is a third instance of the same root cause: a row-level "no
#     candidate at all" fact getting demoted to vague in-row ambiguity.
# ---------------------------------------------------------------------------


def test_lane_column_mismatch_still_surfaces_invented_column_as_model_unbound():
    """HIGH 1: on a lane/column-count mismatch `bind` used to return
    immediately with ZERO cell signal for the whole table -- an invented
    candidate column vanished exactly as completely as a genuinely
    unknowable one. Reproduced with 1 native lane against 2 candidate
    columns, the second wholly invented: the invented value must surface as
    `model_unbound` even though the table stays `column_binding_unverifiable`
    (never claiming a binding for it, only refusing to hide it)."""
    words = [
        w(50, 100, 90, 110, "Coef"),
        w(150, 100, 180, 110, "1.23"),
    ]
    md = """
|      | Val  | Extra |
|------|------|-------|
| Coef | 1.23 | 5.55  |
"""
    result = bind(words, md)
    assert result.column_binding_unverifiable is True
    # The one native lane DOES have a plausible candidate counterpart (their
    # values agree), so it must not also be reported native_unbound.
    assert result.native_unbound == []
    unbound = {(u.row_path[-1], u.token) for u in result.model_unbound}
    assert unbound == {("Coef", "5.55")}
    # I1: the DP-mapped pair (lane 0 <-> Val) is not claimed as a binding,
    # but is not silently dropped either -- it must land in ambiguous_count,
    # or coverage below fails. native_total=1 ("1.23"); candidate_total=2
    # ("1.23" and "5.55").
    _assert_bidirectional_coverage(result, native_total=1, candidate_total=2)


def test_mapped_lane_disagreement_is_not_silently_dropped():
    """Coordinator follow-up to HIGH 1: a lane the DP salvage maps to a
    candidate column is never claimed as a binding (the table stays
    `column_binding_unverifiable`), but a REAL disagreement at that mapped
    position must not vanish with no signal at all either -- that would be
    the exact silent-third-state gap the coordinator flagged: neither bound,
    nor reported unbound, nor even counted as ambiguous.

    Native has one lane (A=1.0, B=2.0). Candidate has two columns: "Val"
    agrees with native for row A (1.0) but DISAGREES for row B (Val=9.0,
    native says 2.0) -- a genuine wrong-digit case hidden behind a
    lane/column-count mismatch. "Extra" never agrees with native at all, so
    the DP maps lane 0 -> Val (score 1) and leaves Extra unmapped (score 0),
    deterministically, not by tie-break.

    Row B's disagreement must not be silently eaten: it cannot be asserted
    `contradicted_cells` (the fix never claims a binding under a lane/column
    mismatch), but it must show up as `ambiguous_count`, not nothing."""
    words = [
        w(50, 100, 90, 110, "A"),
        w(150, 100, 180, 110, "1.0"),
        w(50, 130, 90, 140, "B"),
        w(150, 130, 180, 140, "2.0"),
    ]
    md = """
|   | Val | Extra |
|---|-----|-------|
| A | 1.0 | 5.0   |
| B | 9.0 | 5.0   |
"""
    result = bind(words, md)
    assert result.column_binding_unverifiable is True
    # Never a false conviction either way for the mapped (disagreeing) pair.
    assert result.matched_cells == []
    assert result.contradicted_cells == []
    assert result.native_unbound == []
    unbound = {(u.row_path[-1], u.token) for u in result.model_unbound}
    assert unbound == {("A", "5.0"), ("B", "5.0")}
    # One ambiguous cell per row for the mapped Val<->lane-0 pair -- this is
    # the count that would be 0 (i.e. the disagreement plain vanished) before
    # this fix.
    assert result.ambiguous_count == 2
    _assert_bidirectional_coverage(result, native_total=2, candidate_total=4)


def test_invented_candidate_row_surfaces_its_values_as_model_unbound():
    """HIGH 2: the candidate-side mirror of BLOCKING 1 (round 1's fix for a
    dropped NATIVE row). An invented candidate row that `_bind_rows` cannot
    anchor to anything used to never enter the per-cell walk at all, so its
    values vanished with nothing but `row_binding_unverifiable` to show for
    it -- C4's invented-digit signal, silently dropped instead of
    surfaced."""
    words = [
        w(50, 100, 90, 110, "A"),
        w(150, 100, 180, 110, "1.0"),
        w(50, 130, 90, 140, "B"),
        w(150, 130, 180, 140, "2.0"),
    ]
    md = """
|          | Val  |
|----------|------|
| A        | 1.0  |
| B        | 2.0  |
| Invented | 99.0 |
"""
    result = bind(words, md)
    assert result.row_binding_unverifiable is True
    unbound = {(u.row_path[-1], u.token) for u in result.model_unbound}
    assert unbound == {("Invented", "99.0")}
    assert result.native_unbound == []
    matched = {(m.row_path[-1], m.value) for m in result.matched_cells}
    assert matched == {("A", "1.0"), ("B", "2.0")}
    _assert_bidirectional_coverage(result, native_total=2, candidate_total=3)


def test_invented_digits_on_parent_heading_row_are_model_unbound():
    """A candidate row interpolated onto a native parent (a section heading
    such as "Panel A", carrying no numbers of its own) used to be present
    in ``row_binding`` — so HIGH 2 did not report it — and then skipped by
    both cell walks (``if native_row.is_parent: continue``). Invented
    digits on that row landed in no bucket and ``structural_agreement``
    stayed True. Those numbers are C4 invented-digit ``model_unbound``.

    Pin a difference, not only the bad case: the same native geometry with
    an empty Panel A heading still agrees. ``_assert_bidirectional_coverage``
    on the invented table is the invariant no parent-row fixture applied
    before — it would have been ``candidate_seen 2 vs 4`` on 8bba5ff."""
    words = [
        w(150, 70, 180, 80, "OLS"),
        w(250, 70, 280, 80, "IV"),
        w(50, 100, 100, 110, "Panel"),
        w(105, 100, 115, 110, "A"),
        w(50, 130, 90, 140, "Total"),
        w(150, 130, 180, 140, "10.0"),
        w(250, 130, 280, 140, "20.0"),
    ]
    honest = """
|         | OLS  | IV   |
|---------|------|------|
| Panel A |      |      |
| Total   | 10.0 | 20.0 |
"""
    invented = """
|         | OLS  | IV   |
|---------|------|------|
| Panel A | 9.9  | 8.8  |
| Total   | 10.0 | 20.0 |
"""
    result_honest = bind(words, honest)
    assert result_honest.structural_agreement is True
    assert result_honest.model_unbound == []
    _assert_bidirectional_coverage(result_honest, native_total=2, candidate_total=2)

    result = bind(words, invented)
    assert result.structural_agreement is False
    unbound = {(u.row_path[-1], u.token) for u in result.model_unbound}
    assert unbound == {("Panel A", "9.9"), ("Panel A", "8.8")}
    assert result.contradicted_cells == []
    matched = {(m.row_path[-1], m.value) for m in result.matched_cells}
    assert matched == {("Total", "10.0"), ("Total", "20.0")}
    _assert_bidirectional_coverage(result, native_total=2, candidate_total=4)


def test_invented_middle_row_interpolated_onto_parent_is_model_unbound():
    """Second shape of the same hole: a native parent sits BETWEEN two data
    rows. Equal-length interpolation binds an invented candidate row onto
    that parent, and both cell walks skip it, so the invented digits vanish
    even though every candidate row is 'bound'."""
    words = [
        w(150, 70, 180, 80, "OLS"),
        w(250, 70, 280, 80, "IV"),
        w(50, 100, 90, 110, "Coef"),
        w(150, 100, 180, 110, "1.0"),
        w(250, 100, 280, 110, "2.0"),
        w(50, 130, 100, 140, "Panel"),
        w(105, 130, 115, 140, "A"),
        w(50, 160, 90, 170, "Total"),
        w(150, 160, 180, 170, "10.0"),
        w(250, 160, 280, 170, "20.0"),
    ]
    md = """
|          | OLS  | IV   |
|----------|------|------|
| Coef     | 1.0  | 2.0  |
| Invented | 9.9  | 8.8  |
| Total    | 10.0 | 20.0 |
"""
    result = bind(words, md)
    assert result.structural_agreement is False
    unbound_tokens = {u.token for u in result.model_unbound}
    assert unbound_tokens == {"9.9", "8.8"}
    assert result.contradicted_cells == []
    matched = {(m.row_path[-1], m.value) for m in result.matched_cells}
    assert matched == {("Coef", "1.0"), ("Coef", "2.0"), ("Total", "10.0"), ("Total", "20.0")}
    _assert_bidirectional_coverage(result, native_total=4, candidate_total=6)


def test_parent_row_inventions_use_native_lane_headers_in_salvage_walk():
    """Salvage (lane_count != n_cand_cols) must not treat a candidate column
    index as a native lane index when attributing invented digits on a
    parent-bound row. Native has three lanes Alpha/Beta/Gamma; the
    candidate drops Alpha. DP maps Beta->col0, Gamma->col1, so 9.9 (under
    the candidate's Beta column) is ('Beta',) and 8.8 is ('Gamma',) -- not
    the off-by-one ('Alpha',)/('Beta',) you get from
    ``header_paths_by_lane.get(col_idx)``. Detection without the right
    header is the same defect a prior round already ruled out elsewhere
    in this file."""
    words = [
        w(150, 70, 180, 80, "Alpha"),
        w(250, 70, 280, 80, "Beta"),
        w(350, 70, 380, 80, "Gamma"),
        w(50, 100, 100, 110, "Panel"),
        w(105, 100, 115, 110, "A"),
        w(50, 130, 90, 140, "Total"),
        w(150, 130, 180, 140, "10.0"),
        w(250, 130, 280, 140, "20.0"),
        w(350, 130, 380, 140, "30.0"),
    ]
    md = """
|         | Beta | Gamma |
|---------|------|-------|
| Panel A | 9.9  | 8.8   |
| Total   | 20.0 | 30.0  |
"""
    result = bind(words, md)
    assert result.column_binding_unverifiable is True
    assert result.structural_agreement is False
    unbound = {(u.token, u.col_path) for u in result.model_unbound}
    assert unbound == {("9.9", ("Beta",)), ("8.8", ("Gamma",))}
    _assert_bidirectional_coverage(result, native_total=3, candidate_total=4)


def test_dropped_row_ambiguity_does_not_hide_behind_ambiguous_count():
    """MEDIUM 3: a native row with NO candidate counterpart at all is a
    STRONGER, more specific fact than in-row lane jitter -- the ambiguity
    gate exists to guard the per-cell walk (where a wrong lane assignment
    could misattribute a value to the wrong column), not to swallow a
    row that was never even reachable from the candidate side. Reproduced
    with a duplicated row 15pt from its twin (inside `_WELL_SEPARATED_GAP_PT`,
    so both copies are band-ambiguous): the duplication must surface as
    `native_unbound`, not vanish into `ambiguous_count`."""
    words = [
        w(50, 100, 90, 110, "X"),
        w(150, 100, 180, 110, "1.0"),
        w(50, 115, 90, 125, "X"),
        w(150, 115, 180, 125, "1.0"),
    ]
    md = """
|   | Val |
|---|-----|
| X | 1.0 |
"""
    result = bind(words, md)
    assert result.row_binding_unverifiable is True
    tokens = [u.token for u in result.native_unbound]
    assert tokens == ["1.0", "1.0"]
    _assert_bidirectional_coverage(result, native_total=2, candidate_total=1)


# ---------------------------------------------------------------------------
# 8. GH-270: diagonal-slide fabrication into genuinely empty native cells
# ---------------------------------------------------------------------------


def test_diagonal_slide_into_empty_cells_is_model_unbound_not_matched():
    """GH-270: the real-world defect this module exists to catch. On
    Nakamura-Steinsson page 42 the local VLM filled genuinely empty "Real"
    column cells with values borrowed from the NEXT row's "Nominal" column
    -- a diagonal slide -- and shipped the page SUCCESS. A plain
    value-multiset diff missed it: one value (1.04) disappearing from its
    real position while another (1.02) gains a spurious occurrence nets to
    zero, because the diff counts values, not positions.

    This is the hardest version of the fabrication class: every fabricated
    value IS genuine page content -- it's just the wrong row's Nominal
    value smeared into this row's blank Real cell. A naive "does this value
    appear anywhere on the page" check passes every fabricated cell. Only a
    check that binds by (row, lane) position, not by whole-table value
    membership, can catch it.

    Fixture: 3M/6M/1Y have a real Nominal estimate and a genuinely EMPTY
    Real cell (mirroring the paper page); a fourth "Baseline" row has both
    lanes filled, legitimately establishing the Real lane's geometry so
    ``lane_count == n_cand_cols`` (this exercises the direct per-cell walk,
    not the HIGH 1 lane-mismatch salvage path). The candidate fills each
    blank Real cell with the NEXT row's Nominal value -- 1.19 and 2.05 each
    legitimately appear once elsewhere in native geometry and now ALSO
    appear (fabricated) one row up, exactly the "value appears twice in
    adjacent rows" shape the coordinator's own diff missed."""
    words = [
        w(140, 70, 170, 80, "Nominal"),
        w(240, 70, 270, 80, "Real"),
        w(50, 100, 90, 110, "3M"),
        w(150, 100, 180, 110, "1.11"),
        # Real column for 3M is genuinely empty: no native word here at all.
        w(50, 130, 90, 140, "6M"),
        w(150, 130, 180, 140, "1.19"),
        # Real column for 6M is genuinely empty too.
        w(50, 160, 90, 170, "1Y"),
        w(150, 160, 180, 170, "2.05"),
        # Real column for 1Y is genuinely empty too.
        w(50, 190, 100, 200, "Baseline"),
        w(150, 190, 180, 200, "1.04"),
        w(250, 190, 280, 200, "0.88"),  # only this row's Real is real native content
    ]
    md = """
|          | Nominal | Real |
|----------|---------|------|
| 3M       | 1.11    | 1.19 |
| 6M       | 1.19    | 2.05 |
| 1Y       | 2.05    | 1.04 |
| Baseline | 1.04    | 0.88 |
"""
    result = bind(words, md)
    # Column geometry IS verifiable here (Baseline's Real establishes the
    # lane) -- this must go through the direct per-cell walk, not HIGH 1's
    # degraded salvage path, to prove the base mechanism alone catches this.
    assert result.column_binding_unverifiable is False
    assert result.row_binding_unverifiable is False
    assert result.row_label_contradictions == []

    fabricated = {(u.row_path[-1], u.token) for u in result.model_unbound}
    assert fabricated == {("3M", "1.19"), ("6M", "2.05"), ("1Y", "1.04")}
    # None of the diagonally-slid values are ever counted as a legitimate
    # match, even though each one IS a real value elsewhere on the page.
    matched = {(m.row_path[-1], m.value) for m in result.matched_cells}
    assert matched == {
        ("3M", "1.11"),
        ("6M", "1.19"),
        ("1Y", "2.05"),
        ("Baseline", "1.04"),
        ("Baseline", "0.88"),
    }
    assert result.contradicted_cells == []
    assert result.native_unbound == []
    _assert_bidirectional_coverage(result, native_total=5, candidate_total=8)


# ---------------------------------------------------------------------------
# 9. GH-330 Task 2: _assign_bands groups without chaining tight rows
# ---------------------------------------------------------------------------


def test_assign_bands_groups_by_round_y0_without_chaining_adjacent_rows():
    """GH-330 Task 2: _assign_bands must group by round(y0) and avoid fusing adjacent rows.

    Two numeric rows at 7 pt pitch (y0=100.0 and y0=107.0) must form 2 distinct
    bands rather than being collapsed by a 6 pt x-lane chaining heuristic.
    """
    from socr.tables.binding import _assign_bands

    words = [
        w(50, 100, 90, 106, "RowA"),
        w(150, 100, 180, 106, "1.0"),
        w(50, 107, 90, 113, "RowB"),
        w(150, 107, 180, 113, "2.0"),
    ]
    centers, _ = _assign_bands(words)
    assert len(centers) == 2

    md = """
|      | Val |
|------|-----|
| RowA | 1.0 |
| RowB | 2.0 |
"""
    result = bind(words, md)
    assert result.row_binding_unverifiable is False
    assert len(result.matched_cells) == 2


def test_superscript_same_line_marker_folds_into_numeric_group_using_metadata():
    """GH-330 Task 2: A displaced marker on the same line folds into its numeric row.

    A superscript marker '*' with identical block/line identity folds into Row B,
    yielding 2 bands. A marker with distinct line identity remains separate.
    """
    from socr.tables.binding import _assign_bands

    words_with_same_line_marker = [
        (50, 100, 90, 106, "RowA", 0, 0, 0),
        (150, 100, 180, 106, "1.0", 0, 0, 1),
        (50, 107, 90, 113, "RowB", 0, 1, 0),
        (150, 107, 180, 113, "2.0", 0, 1, 1),
        (185, 104, 190, 108, "*", 0, 1, 2),  # same line_no=1 as Row B
    ]
    words_clean = [
        (50, 100, 90, 106, "RowA", 0, 0, 0),
        (150, 100, 180, 106, "1.0", 0, 0, 1),
        (50, 107, 90, 113, "RowB", 0, 1, 0),
        (150, 107, 180, 113, "2.0", 0, 1, 1),
    ]
    centers_marker, _ = _assign_bands(words_with_same_line_marker)
    centers_clean, _ = _assign_bands(words_clean)

    assert len(centers_marker) == 2
    assert len(centers_clean) == 2

    md = """
|      | Val |
|------|-----|
| RowA | 1.0 |
| RowB | 2.0 |
"""
    result_marker = bind(words_with_same_line_marker, md)
    result_clean = bind(words_clean, md)
    assert result_marker.row_binding_unverifiable is False
    assert result_clean.row_binding_unverifiable is False
    assert len(result_marker.matched_cells) == len(result_clean.matched_cells)


def test_marker_with_distinct_line_identity_remains_separate_row():
    """GH-330 Task 2: A marker with distinct line identity must not be swallowed."""
    from socr.tables.binding import _assign_bands

    words_distinct_line = [
        (50, 100, 90, 106, "RowA", 0, 0, 0),
        (150, 100, 180, 106, "1.0", 0, 0, 1),
        (50, 104, 90, 108, "Note", 0, 99, 0),  # distinct line 99
        (50, 107, 90, 113, "RowB", 0, 1, 0),
        (150, 107, 180, 113, "2.0", 0, 1, 1),
    ]
    centers, _ = _assign_bands(words_distinct_line)
    assert len(centers) == 3


# ---------------------------------------------------------------------------
# 10. GH-330 Task 3: Vertical band ambiguity from word extents
# ---------------------------------------------------------------------------


def test_vertical_band_ambiguity_from_word_extents_not_lane_gap_constant():
    """GH-330 Task 3: Vertical band ambiguity uses adjacent word vertical overlap.

    1. Rows at 10 pt pitch with 8 pt tall non-overlapping boxes have ambiguous_count == 0.
    2. Rows with vertically overlapping bounding boxes have ambiguous_count > 0.
    """
    # Non-overlapping rows (pitch 10 pt, height 8 pt)
    words_non_overlapping = [
        w(150, 70, 180, 78, "Val"),
        w(50, 100, 90, 108, "RowA"),
        w(150, 100, 180, 108, "1.0"),
        w(50, 110, 90, 118, "RowB"),
        w(150, 110, 180, 118, "2.0"),
    ]
    md = """
|      | Val |
|------|-----|
| RowA | 1.0 |
| RowB | 2.0 |
"""
    result_clean = bind(words_non_overlapping, md)
    assert result_clean.ambiguous_count == 0
    assert result_clean.column_binding_unverifiable is False
    assert result_clean.row_binding_unverifiable is False

    # Overlapping rows (y spans [100, 112] and [108, 120] overlap by 4 pt)
    words_overlapping = [
        w(150, 70, 180, 78, "Val"),
        w(50, 100, 90, 112, "RowA"),
        w(150, 100, 180, 112, "1.0"),
        w(50, 108, 90, 120, "RowB"),
        w(150, 108, 180, 120, "2.0"),
    ]
    result_overlapping = bind(words_overlapping, md)
    assert result_overlapping.ambiguous_count > 0


# ---------------------------------------------------------------------------
# 11. GH-330 Task 4: Numeric vs value-less rows binding
# ---------------------------------------------------------------------------


def test_candidate_valueless_units_row_absorbed_in_header_preserves_numeric_row_binding():
    """GH-330 Task 4: Value-less candidate rows do not invalidate numeric row binding."""
    words = [
        w(150, 70, 180, 80, "Col1"),
        w(250, 70, 280, 80, "Col2"),
        w(50, 100, 90, 110, "Row1"),
        w(150, 100, 180, 110, "1.0"),
        w(250, 100, 280, 110, "2.0"),
        w(50, 130, 90, 140, "Row2"),
        w(150, 130, 180, 140, "3.0"),
        w(250, 130, 280, 140, "4.0"),
    ]
    md = """
|              | Col1 | Col2 |
|--------------|------|------|
| Row1         | 1.0  | 2.0  |
| ($ Millions) |      |      |
| Row2         | 3.0  | 4.0  |
"""
    result = bind(words, md)
    assert result.row_binding_unverifiable is False
    assert result.candidate_valueless_unbound == 1
    assert len(result.matched_cells) == 4


def test_unmatched_native_parent_row_reports_native_valueless_unbound():
    """GH-330 Task 4: Value-less native parent row surfaces without invalidating numeric binding."""
    words = [
        w(150, 70, 180, 80, "Col1"),
        w(250, 70, 280, 80, "Col2"),
        w(50, 100, 100, 110, "Panel A"),
        w(50, 130, 90, 140, "Row1"),
        w(150, 130, 180, 140, "1.0"),
        w(250, 130, 280, 140, "2.0"),
        w(50, 160, 90, 170, "Row2"),
        w(150, 160, 180, 170, "3.0"),
        w(250, 160, 280, 170, "4.0"),
    ]
    md = """
|      | Col1 | Col2 |
|------|------|------|
| Row1 | 1.0  | 2.0  |
| Row2 | 3.0  | 4.0  |
"""
    result = bind(words, md)
    assert result.row_binding_unverifiable is False
    assert result.native_valueless_unbound == 1
    assert len(result.matched_cells) == 4


def test_valueless_candidate_opposite_numeric_native_row_never_binds():
    """GH-330 Task 4: Value-less candidate opposite a numeric native row does not bind."""
    words = [
        w(150, 70, 180, 80, "Col1"),
        w(250, 70, 280, 80, "Col2"),
        w(50, 100, 90, 110, "Row1"),
        w(150, 100, 180, 110, "1.0"),
        w(250, 100, 280, 110, "2.0"),
        w(50, 130, 90, 140, "Row2"),
        w(150, 130, 180, 140, "3.0"),
        w(250, 130, 280, 140, "4.0"),
    ]
    md = """
|      | Col1 | Col2 |
|------|------|------|
| Row1 | 1.0  | 2.0  |
| Row2 |      |      |
"""
    result = bind(words, md)
    native_tokens = {u.token for u in result.native_unbound}
    assert native_tokens == {"3.0", "4.0"}
    assert result.row_binding_unverifiable is True


def test_bound_parent_row_increments_row_labels_checked():
    """GH-330 Task 4: A bound parent row verifies its label and increments row_labels_checked."""
    words = [
        w(150, 70, 180, 80, "Col1"),
        w(250, 70, 280, 80, "Col2"),
        w(50, 100, 100, 110, "Panel A"),
        w(50, 130, 90, 140, "Row1"),
        w(150, 130, 180, 140, "1.0"),
        w(250, 130, 280, 140, "2.0"),
    ]
    md = """
|         | Col1 | Col2 |
|---------|------|------|
| Panel A |      |      |
| Row1    | 1.0  | 2.0  |
"""
    result = bind(words, md)
    assert result.row_label_contradictions == []
    assert result.row_labels_checked >= 2


# ---------------------------------------------------------------------------
# 12. GH-330 Task 6: Reconcile binder data-lane geometry with rowizer columns
# ---------------------------------------------------------------------------


def test_numeric_stub_column_excluded_from_data_lanes_when_candidate_has_matching_data_cols():
    """GH-330 Task 6: Numeric stub column (e.g. Years) is not counted as a data lane."""
    words = [
        w(150, 70, 180, 80, "ColA"),
        w(250, 70, 280, 80, "ColB"),
        w(50, 100, 90, 110, "1990"),
        w(150, 100, 180, 110, "10.5"),
        w(250, 100, 280, 110, "20.5"),
        w(50, 130, 90, 140, "1991"),
        w(150, 130, 180, 140, "11.5"),
        w(250, 130, 280, 140, "21.5"),
    ]
    md = """
| Year | ColA | ColB |
|------|------|------|
| 1990 | 10.5 | 20.5 |
| 1991 | 11.5 | 21.5 |
"""
    result = bind(words, md)
    assert result.column_binding_unverifiable is False
    assert result.row_binding_unverifiable is False
    assert len(result.matched_cells) == 4


@pytest.mark.xfail(
    strict=True,
    reason="GH-330 Task 6: label-free table geometry reconciliation not implemented. "
    "strict=True so this turns RED when it starts passing, instead of sitting green.",
)
def test_label_free_table_first_numeric_lane_not_dropped_as_stub():
    """GH-330 Task 6: In a table with no stub column, first data lane must not be dropped."""
    words = [
        w(50, 70, 80, 80, "ColA"),
        w(150, 70, 180, 80, "ColB"),
        w(50, 100, 80, 110, "1.0"),
        w(150, 100, 180, 110, "2.0"),
        w(50, 130, 80, 140, "3.0"),
        w(150, 130, 180, 140, "4.0"),
    ]
    md = """
| ColA | ColB |
|------|------|
| 1.0  | 2.0  |
| 3.0  | 4.0  |
"""
    result = bind(words, md)
    assert result.column_binding_unverifiable is False
    assert len(result.matched_cells) == 4


def test_raw_lane_outside_region_does_not_break_scoped_column_binding():
    """GH-330 Task 6: Words outside the table region do not pollute lane geometry."""
    words = [
        # Table region (0, 0, 300, 200)
        w(150, 70, 180, 80, "ColA"),
        w(250, 70, 280, 80, "ColB"),
        w(50, 100, 90, 110, "Row1"),
        w(150, 100, 180, 110, "1.0"),
        w(250, 100, 280, 110, "2.0"),
        # Extraneous numbers in prose outside table
        w(400, 100, 450, 110, "999.0"),
        w(400, 130, 450, 140, "888.0"),
    ]
    # Candidate emits flat header
    md = """
|      | ColA | ColB |
|------|------|------|
| Row1 | 1.0  | 2.0  |
"""
    result_scoped = bind(words, md, region=(0, 0, 300, 200))
    result_unscoped = bind(words, md)

    assert result_scoped.column_binding_unverifiable is False
    assert result_unscoped.column_binding_unverifiable is True


# ---------------------------------------------------------------------------
# 13. GH-330 Task 7: Zero false model_unbound on self-bind grids
# ---------------------------------------------------------------------------


def test_self_bind_header_depth_discrepancy_zero_model_unbound():
    """GH-330 Task 7: Self-bind with header depth difference reports zero model_unbound."""
    words = [
        w(150, 50, 280, 60, "Spanning Header"),
        w(150, 70, 180, 80, "ColA"),
        w(250, 70, 280, 80, "ColB"),
        w(50, 100, 90, 110, "Row1"),
        w(150, 100, 180, 110, "1.0"),
        w(250, 100, 280, 110, "2.0"),
    ]
    md = """
|      | ColA | ColB |
|------|------|------|
| Row1 | 1.0  | 2.0  |
"""
    result = bind(words, md)
    assert result.model_unbound == []


def test_self_bind_parent_row_labels_zero_model_unbound():
    """GH-330 Task 7: Self-bind with panel parent rows reports zero model_unbound."""
    words = [
        w(150, 70, 180, 80, "ColA"),
        w(250, 70, 280, 80, "ColB"),
        w(50, 100, 110, 110, "Panel Header"),
        w(50, 130, 90, 140, "Row1"),
        w(150, 130, 180, 140, "1.0"),
        w(250, 130, 280, 140, "2.0"),
    ]
    md = """
|              | ColA | ColB |
|--------------|------|------|
| Panel Header |      |      |
| Row1         | 1.0  | 2.0  |
"""
    result = bind(words, md)
    assert result.model_unbound == []


def test_genuine_invented_value_in_cell_still_convicted():
    """GH-330 Task 7: Genuine invented candidate rows remain detected in model_unbound."""
    words = [
        w(150, 70, 180, 80, "ColA"),
        w(250, 70, 280, 80, "ColB"),
        w(50, 100, 90, 110, "Row1"),
        w(150, 100, 180, 110, "1.0"),
        w(250, 100, 280, 110, "2.0"),
    ]
    md = """
|          | ColA | ColB |
|----------|------|------|
| Row1     | 1.0  | 2.0  |
| Invented | 99.9 | 88.8 |
"""
    result = bind(words, md)
    unbound_tokens = {u.token for u in result.model_unbound}
    assert unbound_tokens == {"99.9", "88.8"}


# ---------------------------------------------------------------------------
# VI-A2: binder row-label repair (GH-331 / #418 / #146)
# ---------------------------------------------------------------------------

_A2_CONTROLS = Path(__file__).parent / "fixtures" / "replay_binding" / "controls"


def _a2_words(payload: dict) -> list:
    return [tuple(word) for word in payload["words"]]


def _a2_load(name: str) -> dict:
    return json.loads((_A2_CONTROLS / name).read_text())


def test_region_edge_stub_is_kept_whether_or_not_x0_sits_just_outside():
    """VI-A2 shape 1. Measured drop is 8e-4–2e-3 pt of top-left overflow, not a
    missing stub column. Two word lists that differ only in whether ``3Y``'s
    x0 is on the region edge or 0.002 pt left of it must produce the SAME
    native label — the difference that used to exist is the bug."""
    from socr.tables.binding import _native_rows, _words_in_region

    region = (100.0, 50.0, 400.0, 200.0)
    header = [
        w(200, 70, 230, 80, "OLS"),
        w(300, 70, 330, 80, "IV"),
    ]
    rest = [
        w(120, 100, 180, 110, "Treasury"),
        w(200, 100, 230, 110, "0.50"),
        w(300, 100, 330, 110, "0.51"),
    ]
    on_edge = header + [w(100.0, 100, 112, 110, "3Y")] + rest
    just_out = header + [w(99.998, 100, 111.998, 110, "3Y")] + rest

    def _label(words):
        scoped = _words_in_region(words, region)
        rows, _, _ = _native_rows(scoped)
        data = [row for row in rows if not row.is_parent]
        assert data, "fixture produced no data row"
        return data[0].row_path[-1]

    assert _label(on_edge) == _label(just_out)
    assert _label(just_out) == "3Y Treasury"


def test_numeric_free_text_groups_are_not_folded_together():
    """Abstain: overlapping ``1t`` under ``Rotated PCs`` stays its own group.

    On the measured fixture no page-derived test separates a subscript from
    a short annotation, so overlapping vs parked-below must keep the same
    number of bands — the parent label is unchanged either way.
    """
    from socr.tables.binding import _assign_bands, _native_rows

    header = [
        w(200, 70, 230, 80, "3-M"),
        w(300, 70, 330, 80, "2-YR"),
    ]
    rotated = [
        (50, 100, 110, 111, "Rotated", 0, 0, 0),
        (115, 100, 145, 111, "PCs", 0, 0, 1),
    ]
    subscript_overlap = [
        (80, 107, 95, 115, "1t", 1, 0, 0),
    ]
    subscript_below = [
        (80, 130, 95, 138, "1t", 1, 0, 0),
    ]
    data = [
        w(50, 160, 90, 170, "Action"),
        w(200, 160, 230, 170, "1.48"),
        w(300, 160, 330, 170, "1.00"),
    ]

    n_overlap, _ = _assign_bands(header + rotated + subscript_overlap + data)
    n_below, _ = _assign_bands(header + rotated + subscript_below + data)
    assert len(n_overlap) == len(n_below)

    overlap_rows, _, _ = _native_rows(header + rotated + subscript_overlap + data)
    without_rows, _, _ = _native_rows(header + rotated + data)
    overlap_rotated = [row.row_path[-1] for row in overlap_rows if "Rotated" in row.row_path[-1]]
    without_rotated = [row.row_path[-1] for row in without_rows if "Rotated" in row.row_path[-1]]
    assert overlap_rotated == without_rotated
    assert any(row.row_path[-1].strip() == "1t" for row in overlap_rows)


def test_text_fold_does_not_hop_into_a_numeric_row():
    """A numeric-free marker already remapped onto a numeric row must not
    become a hop for a later text group. ``*`` shares RowB's line and
    folds into it; a shorter ``a`` under the star would follow that hop
    unless the resolved destination is required to be numeric-free."""
    from socr.tables.binding import _assign_bands

    words = [
        (50, 100, 90, 106, "RowA", 0, 0, 0),
        (150, 100, 180, 106, "1.0", 0, 0, 1),
        (50, 110, 90, 116, "RowB", 0, 1, 0),
        (150, 110, 180, 116, "2.0", 0, 1, 1),
        (185, 104, 190, 111, "*", 0, 1, 2),
        (186, 107, 189, 110, "a", 0, 99, 0),
    ]
    _centers, y_to_band = _assign_bands(words)
    assert y_to_band[round(104)] == y_to_band[round(110)]
    assert y_to_band[round(107)] != y_to_band[round(110)]


def test_annotation_under_label_is_not_folded():
    """Negative control: 'see note a' under a label. Native label with vs
    without the annotation is identical."""
    from socr.tables.binding import _native_rows, _words_in_region

    payload = _a2_load("annotation_under_label.json")
    words = _a2_words(payload)
    region = tuple(payload["region"])
    annotation = {"see", "note", "a"}
    without = [word for word in words if word[4] not in annotation]

    def _label(word_list):
        rows, _, _ = _native_rows(_words_in_region(word_list, region))
        data = [row for row in rows if not row.is_parent]
        assert data
        return data[0].row_path[-1]

    assert _label(words) == _label(without)
    assert _label(words) == "Large T"
    assert "see" not in _label(words)
    assert "note" not in _label(words)


def test_short_annotation_under_numeric_free_parent_is_not_folded():
    """Negative control that actually reaches the text-band path: a pure
    label row (no numeric tokens) with a shorter ``(a)`` wholly contained
    under one parent word, y-boxes overlapping. Required: the annotation
    stays separate — parent label identical with vs without ``(a)``."""
    from socr.tables.binding import _assign_bands, _native_rows, _words_in_region
    from socr.tables.native_verifier import is_numeric_token

    payload = _a2_load("short_annotation_under_text_parent.json")
    words = _a2_words(payload)
    region = tuple(payload["region"])
    parent_y = round(100.0)
    assert not any(is_numeric_token(word[4]) for word in words if round(word[1]) == parent_y), (
        "fixture parent row must be numeric-free or this control never hits the fold"
    )

    without = [word for word in words if word[4] != "(a)"]
    _centers, y_to_band = _assign_bands(_words_in_region(words, region))
    assert y_to_band[round(107.0)] != y_to_band[parent_y]

    def _rotated_label(word_list):
        rows, _, _ = _native_rows(_words_in_region(word_list, region))
        for row in rows:
            if row.row_path and "Rotated" in row.row_path[-1]:
                return row.row_path[-1]
        raise AssertionError(f"no Rotated row in {[row.row_path for row in rows]}")

    assert _rotated_label(words) == _rotated_label(without) == "Rotated"
    assert "(a)" not in _rotated_label(words)


def test_shape2_numeric_inside_label_is_not_swallowed_by_widening():
    """Autopsy shape 2 control: ``500`` at a data-lane x must stay a value.
    Pin the difference between a candidate that matches the cut native label
    and one that claims the printed ``500`` — the latter still contradicts."""
    payload = _a2_load("shape2_numeric_in_label.json")
    words = _a2_words(payload)
    region = tuple(payload["region"])
    cut = """
|            | OLS  | IV   |
|------------|------|------|
| slope      | 1.10 | 1.11 |
| dlog S&P   | 0.50 | 0.51 |
| other      | 0.20 | 0.21 |
"""
    printed = """
|                   | OLS  | IV   |
|-------------------|------|------|
| slope             | 1.10 | 1.11 |
| dlog S&P 500 (3m) | 0.50 | 0.51 |
| other             | 0.20 | 0.21 |
"""
    cut_result = bind(words, cut, region=region)
    printed_result = bind(words, printed, region=region)
    cut_labels = [c.candidate_label for c in cut_result.row_label_contradictions]
    printed_labels = [c.candidate_label for c in printed_result.row_label_contradictions]
    assert cut_labels != printed_labels
    assert not any("500" in label for label in cut_labels)
    assert any("500" in label for label in printed_labels)


def test_shape3_nonnumeric_cells_stay_in_the_native_label():
    """Autopsy shape 3 control: date-range cells are not values. The native
    label with the date ranges present must differ from the same geometry
    with those cells deleted — they were absorbed, and A2 must not start
    treating them as a value lane."""
    from socr.tables.binding import _native_rows, _words_in_region

    payload = _a2_load("shape3_nonnumeric_values.json")
    words = _a2_words(payload)
    region = tuple(payload["region"])
    without_dates = [word for word in words if ":" not in word[4]]

    def _sample_label(word_list):
        rows, _, _ = _native_rows(_words_in_region(word_list, region))
        for row in rows:
            if row.row_path and "Sample" in row.row_path[-1]:
                return row.row_path[-1]
        raise AssertionError(f"no Sample row in {[row.row_path for row in rows]}")

    with_dates = _sample_label(words)
    stripped = _sample_label(without_dates)
    assert with_dates != stripped
    assert "1988:1-2019:12" in with_dates
    assert stripped == "Sample"


def test_row_swap_control_still_contradicts():
    """VI-A2 must not pass by accepting swapped stubs. Correct vs swapped
    markdown on the same words: one agrees, the other contradicts."""
    payload = _a2_load("row_swap.json")
    words = _a2_words(payload)
    region = tuple(payload["region"])
    correct = bind(words, payload["markdown_correct"], region=region)
    swapped = bind(words, payload["markdown_swapped"], region=region)
    assert correct.row_label_contradictions == []
    assert swapped.row_label_contradictions != []
    assert {c.candidate_label for c in swapped.row_label_contradictions} == {"Small", "Large"}


def test_neighbouring_label_outside_the_region_is_not_captured():
    """A neighbour stub whose centroid is outside the region must not
    enter the label. Same table with vs without that word: labels identical."""
    from socr.tables.binding import _native_rows, _words_in_region

    payload = _a2_load("neighbouring_label.json")
    words = _a2_words(payload)
    region = tuple(payload["region"])
    without = [word for word in words if word[4] != "Neighbour"]

    def _label(word_list):
        rows, _, _ = _native_rows(_words_in_region(word_list, region))
        data = [row for row in rows if not row.is_parent]
        assert data
        return data[0].row_path[-1]

    assert _label(words) == _label(without)
    assert _label(words) == "Treasury"
    assert "Neighbour" not in _label(words)


def test_grazing_caption_and_footnote_are_excluded_as_on_main():
    """A caption box that dips a sub-point into the region from ABOVE and a
    footnote box from BELOW must stay out. Top-left containment (main)
    excludes the caption graze; centroid excludes both. Pin: with vs
    without those two words, the native label is identical — the
    difference is zero."""
    from socr.tables.binding import _native_rows, _words_in_region

    payload = _a2_load("grazing_caption_footnote.json")
    words = _a2_words(payload)
    region = tuple(payload["region"])
    grazers = {"Caption", "Footnote"}
    without = [word for word in words if word[4] not in grazers]
    grazer_words = [word for word in words if word[4] in grazers]
    scoped = _words_in_region(words, region)
    scoped_texts = {word[4] for word in scoped}
    assert grazers.isdisjoint(scoped_texts)

    def _top_left(word: tuple) -> bool:
        rx0, ry0, rx1, ry1 = region
        return rx0 <= word[0] <= rx1 and ry0 <= word[1] <= ry1

    def _intersects(word: tuple) -> bool:
        rx0, ry0, rx1, ry1 = region
        return min(word[2], rx1) > max(word[0], rx0) and min(word[3], ry1) > max(word[1], ry0)

    caption = next(word for word in words if word[4] == "Caption")
    assert _top_left(caption) is False
    assert all(_intersects(word) for word in grazer_words)

    def _label(word_list):
        rows, _, _ = _native_rows(_words_in_region(word_list, region))
        data = [row for row in rows if not row.is_parent]
        assert data
        return data[0].row_path[-1]

    assert _label(words) == _label(without)
    assert _label(words) == "Treasury"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
