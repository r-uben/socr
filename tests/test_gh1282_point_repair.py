"""#1282: repair a dropped decimal point from the page's own text layer.

Part 1 is the 20 attack tests on synthetic word geometry (the pure function).  Part 2 is the
page helper on real fitz pages at rotation 0/90/270.  Differences are pinned, never a value a
machine happened to produce.

Words are fitz tuples (x0, y0, x1, y1, text).  One line per printed row; a label of one or two
words at x=10, values at x=200, 260, 320...  Heights 10, line pitch 20.
"""

from __future__ import annotations

import pytest

from socr.tables import row_binding_check as rbc
from socr.tables.row_binding_check import (
    REFUSE_AMBIGUOUS,
    REFUSE_CLAIMED,
    REFUSE_CONTRADICTED,
    REFUSE_DECORATED,
    REFUSE_LABEL,
    REFUSE_NO_AGREEING,
    REFUSE_ORDER,
    RowVerdict,
    TableOutcome,
    check_row_binding,
    point_dropped,
    propose_point_repairs,
)

H = 10.0


def line(y, label, values):
    out = []
    x = 10.0
    for w in label.split():
        out.append((x, y, x + 8.0 * len(w), y + H, w))
        x += 8.0 * len(w) + 6.0
    x = 200.0
    for v in values:
        out.append((x, y, x + 6.0 * len(v), y + H, v))
        x += 60.0
    return out


def page(*rows):
    words = []
    for i, (label, values) in enumerate(rows):
        words.extend(line(20.0 * i, label, values))
    return words


def md(*rows):
    head = "|  |  |  |  |\n| --- | --- | --- | --- |"  # an empty heading row: nothing to judge
    body = "\n".join("| " + " | ".join([label, *values]) + " |" for label, values in rows)
    return head + "\n" + body


def run(words, markdown):
    return propose_point_repairs(words, None, markdown)


GDP = ("Real GDP", ["1.0", "2.0", "3.0"])


# ---------------------------------------------------------------------------
# Part 1: the 20 attacks
# ---------------------------------------------------------------------------


def test_baseline_repairs_the_dropped_point():
    words = page(GDP, ("Net exports", ["-.7", "1.2", "3.4"]))
    shipped = md(GDP, ("Net exports", ["-7", "1.2", "3.4"]))
    new, reps, refs = run(words, shipped)
    assert [(r.label, r.old, r.new) for r in reps] == [("Net exports", "-7", "-.7")], (reps, refs)
    assert "| Net exports | -.7 | 1.2 | 3.4 |" in new
    # the difference: the unrepaired table is UNVERIFIABLE on that row, the repaired one is BOUND
    before = check_row_binding(words, None, shipped)
    after = check_row_binding(words, None, new)
    assert before.outcome is TableOutcome.UNVERIFIABLE
    assert after.outcome is TableOutcome.BOUND


def test_two_drops_on_one_line_both_repaired():
    words = page(GDP, ("Net exports", ["-.7", "1.2", "-.4", "5.5"]))
    _, reps, refs = run(words, md(GDP, ("Net exports", ["-7", "1.2", "-4", "5.5"])))
    assert [(r.old, r.new) for r in reps] == [("-7", "-.7"), ("-4", "-.4")], (reps, refs)


def test_right_number_is_never_changed():
    # the page really prints -7: the row has an exact line, the rule does not run (condition 1)
    words = page(GDP, ("Net exports", ["-7", "1.2", "3.4"]))
    shipped = md(GDP, ("Net exports", ["-7", "1.2", "3.4"]))
    new, reps, refs = run(words, shipped)
    assert reps == [] and refs == [] and new == shipped


def test_a_row_with_an_exact_line_is_not_moved_to_a_point_dropped_sibling_line():
    # condition 1: the page prints `Previous -7 ...` AND `Previous -.7 ...`.  The shipped row equals
    # the first exactly, so it is right as it stands; a sibling line that differs by a point must
    # not pull it.
    words = page(GDP, ("Previous", ["-7", "1.2", "3.4"]), ("Previous", ["-.7", "1.2", "3.4"]))
    shipped = md(GDP, ("Previous", ["-7", "1.2", "3.4"]))
    new, reps, refs = run(words, shipped)
    assert reps == [] and new == shipped, (reps, refs)


def test_same_values_under_another_label_is_the_binders_business_not_a_repair():
    words = page(("Net exports", ["-.7", "1.2", "3.4"]), ("Imports", ["-7", "1.2", "3.4"]))
    _, reps, _ = run(words, md(("Net exports", ["-7", "1.2", "3.4"]), ("Imports", ["5", "6", "7"])))
    assert reps == []


def test_minus_seven_printed_elsewhere_does_not_mislead():
    words = page(("Net exports", ["-.7", "1.2", "3.4"]), ("Farm", ["2", "-7", "3"]))
    new, reps, refs = run(
        words, md(("Net exports", ["-7", "1.2", "3.4"]), ("Farm", ["2", "-7", "3"]))
    )
    assert [(r.label, r.old, r.new) for r in reps] == [("Net exports", "-7", "-.7")], (reps, refs)
    assert "| Farm | 2 | -7 | 3 |" in new


def test_identical_previous_lines_are_ambiguous_and_refused():
    words = page(("Previous", ["-.7", "1.2", "3.4"]), ("Previous", ["-.7", "1.2", "3.4"]))
    _, reps, refs = run(words, md(("Previous", ["-7", "1.2", "3.4"])))
    assert reps == [] and refs and refs[0].why == (REFUSE_AMBIGUOUS,), (reps, refs)


def test_previous_lines_told_apart_by_the_other_values():
    """The other `Previous` line must be told apart by more than MAX_RIVAL_MISMATCHES values
    (review fix N): differing in the point and one value, it could be the row's own line with
    one more model slip, and the row is refused; differing in three, line 1 is repaired."""
    assert rbc.MAX_RIVAL_MISMATCHES == 2
    words = page(("Previous", ["-.7", "1.2", "3.4"]), ("Previous", ["-.7", "1.2", "3.5"]))
    _, reps, refs = run(words, md(("Previous", ["-7", "1.2", "3.5"])))
    assert reps == [] and refs[0].why == (rbc.REFUSE_RIVAL_LINE,), (reps, refs)
    words = page(("Previous", ["-.7", "1.9", "3.4"]), ("Previous", ["-.7", "1.2", "3.5"]))
    _, reps, refs = run(words, md(("Previous", ["-7", "1.2", "3.5"])))
    assert [(r.old, r.new, r.native_line) for r in reps] == [("-7", "-.7", 1)], (reps, refs)


def test_label_must_be_exact_footnote_mark_refuses():
    words = page(GDP, ("Receipts2", ["-.7", "1.2", "3.4"]))
    for shipped in ("Receipts", "Receipts²", "Receipts3"):
        _, reps, refs = run(words, md(GDP, (shipped, ["-7", "1.2", "3.4"])))
        assert reps == [] and refs and refs[0].why == (REFUSE_LABEL,), (shipped, reps, refs)


def test_label_mismatch_refuses_even_with_all_other_values_equal():
    words = page(GDP, ("Net exports", ["-.7", "1.2", "3.4"]))
    _, reps, refs = run(words, md(GDP, ("Exports", ["-7", "1.2", "3.4"])))
    assert reps == [] and refs[0].why == (REFUSE_LABEL,), (reps, refs)


def test_digit_drop_is_not_a_point_drop():
    words = page(GDP, ("Euro area", ["-1.1", "1.2", "3.4"]))
    _, reps, refs = run(words, md(GDP, ("Euro area", ["-.1", "1.2", "3.4"])))
    assert reps == [] and refs[0].why == ("pair:digit_dropped",), (reps, refs)


def test_sign_difference_refuses():
    words = page(GDP, ("Japan", ["-.7", "1.2", "3.4"]))
    _, reps, refs = run(words, md(GDP, ("Japan", [".7", "1.2", "3.4"])))
    assert reps == [] and refs[0].why == ("pair:sign_differs",), (reps, refs)


def test_mixed_point_drop_and_other_difference_refuses():
    words = page(GDP, ("Japan", ["-.7", "1.2", "3.4"]))
    _, reps, refs = run(words, md(GDP, ("Japan", ["-7", "1.3", "3.4"])))
    assert reps == [], (reps, refs)


def test_no_agreeing_value_refuses_and_the_floor_is_live(monkeypatch):
    words = page(("Real GDP", ["1.0"]), ("Net exports", ["-.7"]))
    shipped = "| Item | a |\n| --- | --- |\n| Real GDP | 1.0 |\n| Net exports | -7 |"
    new, reps, refs = run(words, shipped)
    assert reps == [] and refs[0].why == (REFUSE_NO_AGREEING,), (reps, refs)
    # the difference: with the floor removed the same input IS repaired
    monkeypatch.setattr(rbc, "MIN_AGREEING_VALUES", 0)
    _, reps0, _ = run(words, shipped)
    assert [(r.old, r.new) for r in reps0] == [("-7", "-.7")]


def test_reading_order_refuses_a_line_above_the_bound_row_before_it():
    words = page(("Net exports", ["-.7", "1.2", "3.4"]), GDP, ("Imports", ["4", "5", "6"]))
    _, reps, refs = run(
        words, md(GDP, ("Net exports", ["-7", "1.2", "3.4"]), ("Imports", ["4", "5", "6"]))
    )
    assert reps == [] and refs[0].why == (REFUSE_ORDER,), (reps, refs)


def test_line_claimed_by_another_row_refuses():
    words = page(GDP, ("Net exports", ["-.7", "1.2", "3.4"]))
    _, reps, refs = run(
        words,
        md(
            GDP,
            ("Net exports", ["-7", "1.2", "3.4"]),
            ("Net exports", ["-.7", "1.2", "3.4"]),
        ),
    )
    assert reps == [] and refs[0].why == (REFUSE_CLAIMED,), (reps, refs)


def test_decorated_cell_refuses():
    words = page(GDP, ("Net exports", ["-.7", "1.2", "3.4"]))
    _, reps, refs = run(words, md(GDP, ("Net exports", ["**-7**", "1.2", "3.4"])))
    assert reps == [] and refs[0].why == (REFUSE_DECORATED,), (reps, refs)


def test_positive_point_drop_and_two_digit_forms():
    words = page(GDP, ("Services", [".4", "1.2", "3.4"]), ("Durables", ["1.2", "2.0", "3.0"]))
    _, reps, refs = run(
        words, md(GDP, ("Services", ["4", "1.2", "3.4"]), ("Durables", ["12", "2.0", "3.0"]))
    )
    assert [(r.old, r.new) for r in reps] == [("4", ".4"), ("12", "1.2")], (reps, refs)


def test_trailing_point_form_is_a_point_drop():
    words = page(GDP, ("Nonres. structures", ["-.2", "1.2", "3.4"]))
    _, reps, refs = run(words, md(GDP, ("Nonres. structures", ["-2.", "1.2", "3.4"])))
    assert [(r.old, r.new) for r in reps] == [("-2.", "-.2")], (reps, refs)


def test_value_count_mismatch_refuses():
    words = page(GDP, ("Net exports", ["-.7", "1.2", "3.4", "9.9"]))
    _, reps, refs = run(words, md(GDP, ("Net exports", ["-7", "1.2", "3.4"])))
    assert reps == [], (reps, refs)


def test_repair_is_idempotent():
    words = page(GDP, ("Net exports", ["-.7", "1.2", "3.4"]))
    once, reps1, _ = run(words, md(GDP, ("Net exports", ["-7", "1.2", "3.4"])))
    twice, reps2, _ = run(words, once)
    assert reps1 and not reps2 and once == twice


# ---------------------------------------------------------------------------
# Owner choices of 2026-11-07
# ---------------------------------------------------------------------------


def test_the_pdfs_own_string_ships_not_a_zero_padded_one():
    words = page(GDP, ("Net exports", ["-.7", "1.2", "3.4"]))
    new, reps, _ = run(words, md(GDP, ("Net exports", ["-7", "1.2", "3.4"])))
    assert reps[0].new == "-.7" and "-0.7" not in new


def test_bare_number_row_labels_are_kept():
    # 2011-11-02 p105: rows labelled by a bare quarter number, exact label + agreeing values
    # the bare label sits one gap before the values, as in a printed quarter column
    # (a far-away label would make the line a second panel, which is another test)
    words = page(GDP) + [(170.0, 20.0, 178.0, 30.0, "2")] + line(20.0, "", ["-.7", "1.2", "3.4"])
    new, reps, refs = run(words, md(GDP, ("2", ["-7", "1.2", "3.4"])))
    assert [(r.label, r.old, r.new) for r in reps] == [("2", "-7", "-.7")], (reps, refs)
    assert "| 2 | -.7 | 1.2 | 3.4 |" in new


def _contradicted_case():
    """A displaced row (the model slid the values of 'Final sales' one label down) plus a
    point-drop row that, on its own, WOULD be repaired."""
    words = page(
        ("Real GDP", ["1.1", "2.2", "3.3"]),
        ("Final sales", ["4.4", "5.5", "6.6"]),
        ("Priv. dom. final purchases", ["7.7", "8.8", "9.9"]),
        ("Net exports", ["-.7", "1.2", "3.4"]),
    )
    shipped = md(
        ("Real GDP", ["1.1", "2.2", "3.3"]),
        ("Final sales", []),
        ("Priv. dom. final purchases", ["4.4", "5.5", "6.6"]),
        ("", ["7.7", "8.8", "9.9"]),
        ("Net exports", ["-7", "1.2", "3.4"]),
    )
    return words, shipped


def test_contradicted_table_is_not_repaired():
    words, shipped = _contradicted_case()
    assert check_row_binding(words, None, shipped).outcome is TableOutcome.CONTRADICTED
    new, reps, refs = run(words, shipped)
    assert new == shipped and reps == []
    assert [r.why for r in refs] == [(REFUSE_CONTRADICTED,)]
    # the difference: the same point-drop row, in a table that is not contradicted, IS repaired
    ok_words = page(GDP, ("Net exports", ["-.7", "1.2", "3.4"]))
    _, reps_ok, _ = run(ok_words, md(GDP, ("Net exports", ["-7", "1.2", "3.4"])))
    assert reps_ok


def test_point_dropped_shapes():
    assert point_dropped("-7", "-0.7") and point_dropped("4", "0.4") and point_dropped("12", "1.2")
    assert not point_dropped("-0.1", "-1.1") and not point_dropped("7", "-0.7")
    assert not point_dropped("1.0", "1.0") and not point_dropped("7", "7")


def test_condition_6_the_gate_refuses_when_a_bound_row_would_be_lost(monkeypatch):
    """Condition 6 is live: with the production gate's answer forced to 'not bound', a row that
    passes conditions 1-5 is refused, and without the force it is repaired."""
    words = page(GDP, ("Net exports", ["-.7", "1.2", "3.4"]))
    shipped = md(GDP, ("Net exports", ["-7", "1.2", "3.4"]))
    _, reps, _ = run(words, shipped)
    assert reps
    real = rbc.check_row_binding
    calls = {"n": 0}

    def fake(words, region, markdown):
        calls["n"] += 1
        res = real(words, region, markdown)
        if calls["n"] == 1:
            return res  # the base check
        from dataclasses import replace

        rows = tuple(
            replace(e, verdict=RowVerdict.UNVERIFIABLE) if "Net exports" in e.shipped_label else e
            for e in res.rows
        )
        return rbc.RowBindingResult(res.outcome, rows)

    monkeypatch.setattr(rbc, "check_row_binding", fake)
    new, reps2, refs2 = run(words, shipped)
    assert reps2 == [] and new == shipped
    assert refs2[0].why == (rbc.REFUSE_NOT_BOUND,)


# ---------------------------------------------------------------------------
# Review of #1282 (Fable): whole-page condition 1, the agreeing floor, affixes
# ---------------------------------------------------------------------------


def test_condition_1_looks_at_the_whole_page_not_the_tables_box():
    """Case B: the block's box misses the row's own line (`Real GDP 2.5 1.0 7`, y=40) and holds
    another table's line with the same label and a point-drop difference (y=0).  The model's row is
    RIGHT; with the box alone it would become `.7`.  An exact line anywhere on the page forbids it."""
    words = page(
        ("Real GDP", ["2.5", "1.0", ".7"]),
        ("Imports", ["1.1", "1.2", "1.3"]),
        ("Real GDP", ["2.5", "1.0", "7"]),
        ("Imports", ["3.1", "3.2", "3.3"]),
    )
    shipped = md(("Real GDP", ["2.5", "1.0", "7"]), ("Imports", ["3.1", "3.2", "3.3"]))
    box_of_the_first_table = (0.0, -5.0, 400.0, 35.0)
    new, reps, refs = propose_point_repairs(words, box_of_the_first_table, shipped)
    assert reps == [] and new == shipped, (reps, refs)
    # the difference: with the row's own line gone from the page, the same box DOES repair it
    gone = page(("Real GDP", ["2.5", "1.0", ".7"]), ("Imports", ["1.1", "1.2", "1.3"]))
    _, reps2, _ = propose_point_repairs(
        gone,
        box_of_the_first_table,
        md(("Real GDP", ["2.5", "1.0", "7"]), ("Imports", ["1.1", "1.2", "1.3"])),
    )
    assert [(r.old, r.new) for r in reps2] == [("7", ".7")]


def test_a_two_value_row_with_one_agreeing_value_is_refused():
    """Case A: `-7 | 1.5` against `-.7 | 1.5` has one agreeing value; two are required."""
    words = page(("Real GDP", ["1.0", "2.0"]), ("Net exports", ["-.7", "1.5"]))
    shipped = md(("Real GDP", ["1.0", "2.0"]), ("Net exports", ["-7", "1.5"]))
    new, reps, refs = run(words, shipped)
    assert reps == [] and new == shipped and refs[0].why == (REFUSE_NO_AGREEING,), (reps, refs)
    assert rbc.MIN_AGREEING_VALUES == 2


def test_a_native_token_with_an_affix_the_shipped_cell_lacks_is_refused():
    """Case E: the page prints `-.7%`, the model shipped `-7`."""
    words = page(GDP, ("Rate", ["1.0", "-.7%", "3.0", "4.0"]))
    shipped = md(GDP, ("Rate", ["1.0", "-7", "3.0", "4.0"]))
    new, reps, refs = run(words, shipped)
    assert reps == [] and new == shipped, (reps, refs)
    assert refs and refs[0].why == (rbc.REFUSE_AFFIX,), refs
    # the difference: the same affix on BOTH sides is a plain point repair
    new2, reps2, _ = run(words, md(GDP, ("Rate", ["1.0", "-7%", "3.0", "4.0"])))
    assert [(r.old, r.new) for r in reps2] == [("-7%", "-.7%")], reps2
