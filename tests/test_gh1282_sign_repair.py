"""#1282, plan A2 step 2 (owner ruling D1): repair a flipped minus from the PDF's own string.

The sign repair is an extension of the point repair: same binding, same refusals (conditions 0-7,
B, Z, N), same identity and commit machinery.  Extra conditions, each with a named refusal and a
test that fails when the condition is removed (mutation record: docs/log/2026-11-07_1282-point-repair.md):

- S1 one difference per cell: digits and point identical, only the sign differs (``pair:point_and_sign``);
- S2 the value is not zero (``pair:sign_of_zero``);
- S3 the minus is attached to its number by GEOMETRY (``REFUSE_SIGN_GEOMETRY``);
- S4 the layer encodes minus signs in this table (``REFUSE_SIGN_NO_MINUS_EVIDENCE``);
- S5 no mirror row prints the same absolute values (``REFUSE_SIGN_MIRROR``);
- S6 notation is equal, not a contradiction: handled by normalisation, never by repair.

Part 1 is the pure function on synthetic word geometry, part 2 is the page entry at rotation
0/90/270, part 3 is ``process()`` as a pair of arms in both provider states.  Differences are
pinned, never a value a machine happened to produce.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from socr.tables import row_binding_check as rbc
from socr.tables.row_binding_check import (
    REFUSE_RIVAL_LINE,
    REFUSE_SIGN_GEOMETRY,
    REFUSE_SIGN_MIRROR,
    REFUSE_SIGN_NO_MINUS_EVIDENCE,
    REFUSE_TOKEN_MAP,
    check_row_binding,
    propose_point_repairs,
    sign_flipped,
)

H = 10.0
NEG = ("Real GDP", ["1.0", "-2.0", "3.0"])  # a bound row that prints a minus: the table's witness
POS = ("Real GDP", ["1.0", "2.0", "3.0"])


# vertical extents of a detached glyph, relative to a number box [y, y + H]
EXTENTS = {
    "tall": lambda y: (y, y + H),  # as tall as the digits: the line grouping keeps it in the line
    "thin": lambda y: (y + 1.0, y + 3.0),  # inside the band, on a line of its own
    "above": lambda y: (
        y - 6.0,
        y - 2.0,
    ),  # ends 2 pt above the number's top: outside the strict band
    "far": lambda y: (y - 14.0, y - 10.0),  # 10 pt above: further than the band's slack
}


def lay(y, label, values):
    """One printed line.  A value "~1.5" is a detached minus glyph just left of 1.5 (a word of its
    own, 2 pt from the number); "1.5~" is a detached minus just right of 1.5 (2 pt from 1.5 and
    34 pt from the next number).  "^1.5" is the same glyph as "~1.5" but as tall as the digits.
    The "~" glyph is thin and sits in the upper part of the number's band: the line grouping puts
    it on a line of its own (the way a rule-drawn minus is extracted), so no other check sees it;
    "^" lands in the number's own line and in the printed label when it precedes the first value."""
    out = []
    x = 10.0
    for w in label.split():
        out.append((x, y, x + 8.0 * len(w), y + H, w))
        x += 8.0 * len(w) + 6.0
    x = 200.0
    for v in values:
        if isinstance(
            v, tuple
        ):  # (glyph, value, extent, side): a glyph word of the caller's choosing
            glyph, v, extent, side = v
            top, bottom = EXTENTS[extent](y)
            gx = (
                x - 8.0 if side == "number" else x - 60.0 + 6.0 * 3 + 2.0
            )  # 2 pt after the previous value
            out.append((gx, top, gx + 6.0, bottom, glyph))
        elif v[0] in "~^":
            top, bottom = (y + 1.0, y + 3.0) if v[0] == "~" else (y, y + H)
            out.append((x - 8.0, top, x - 2.0, bottom, "-"))
            v = v[1:]
        if v.endswith("~"):
            v = v[:-1]
            out.append((x + 6.0 * len(v) + 2.0, y + 1.0, x + 6.0 * len(v) + 8.0, y + 3.0, "-"))
        out.append((x, y, x + 6.0 * len(v), y + H, v))
        x += 60.0
    return out


def layer(*rows):
    words = []
    for i, (label, values) in enumerate(rows):
        words.extend(lay(20.0 * i, label, values))
    return words


def table(*rows):
    head = "|  |  |  |  |\n| --- | --- | --- | --- |"
    body = "\n".join("| " + " | ".join([label, *values]) + " |" for label, values in rows)
    return head + "\n" + body


def go(words, markdown):
    return propose_point_repairs(words, None, markdown)


def why(refusals):
    return {w for f in refusals for w in f.why}


# ---------------------------------------------------------------------------
# The classifier
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shipped,native,expected",
    [
        ("1.5", "-1.5", True),
        ("-1.5", "1.5", True),
        ("-3.0", "3.0", True),
        ("-1.5", "-1.5", False),  # no sign difference
        ("-3", "0.3", False),  # the point is off as well (S1)
        ("-3", "3.0", False),  # a different digit string
        ("0.0", "-0.0", False),  # zero carries no sign (S2)
        ("0", "-0", False),
    ],
)
def test_sign_flipped_is_one_difference_and_never_zero(shipped, native, expected):
    assert sign_flipped(shipped, native) is expected


# ---------------------------------------------------------------------------
# Part 1: positive cases
# ---------------------------------------------------------------------------


def test_a_dropped_minus_is_restored_from_the_layers_string():
    words = layer(NEG, ("Net exports", ["-1.5", "2.5", "3.5"]))
    shipped = table(NEG, ("Net exports", ["1.5", "2.5", "3.5"]))
    new, reps, refs = go(words, shipped)
    assert [(r.old, r.new, r.kind) for r in reps] == [("1.5", "-1.5", "sign")], (reps, refs)
    assert "| Net exports | -1.5 | 2.5 | 3.5 |" in new
    # the difference: the unrepaired row has no line, the repaired one is BOUND
    assert check_row_binding(words, None, new).outcome.value == "bound"
    assert check_row_binding(words, None, shipped).outcome.value == "unverifiable"


def test_a_spurious_minus_is_removed():
    words = layer(NEG, ("Net exports", ["1.5", "2.5", "3.5"]))
    shipped = table(NEG, ("Net exports", ["-1.5", "2.5", "3.5"]))
    new, reps, refs = go(words, shipped)
    assert [(r.old, r.new, r.kind) for r in reps] == [("-1.5", "1.5", "sign")], (reps, refs)
    assert "| Net exports | 1.5 | 2.5 | 3.5 |" in new


def test_the_layers_own_string_is_written_verbatim_with_its_unicode_minus():
    words = layer(NEG, ("Net exports", ["−1.5", "2.5", "3.5"]))
    new, reps, _ = go(words, table(NEG, ("Net exports", ["1.5", "2.5", "3.5"])))
    assert [(r.old, r.new) for r in reps] == [("1.5", "−1.5")]
    assert "| −1.5 |" in new


def test_a_point_drop_and_a_sign_flip_in_one_row_are_each_one_difference_in_their_cell():
    # the flipped minus is the row's first value: a minus after another number is range-shaped
    words = layer(NEG, ("Net exports", ["-2.5", "-.7", "3.5", "4.5"]))
    shipped = table(NEG, ("Net exports", ["2.5", "-7", "3.5", "4.5"]))
    _, reps, refs = go(words, shipped)
    assert sorted((r.old, r.new, r.kind) for r in reps) == [
        ("-7", "-.7", "point"),
        ("2.5", "-2.5", "sign"),
    ], (reps, refs)


def test_a_point_only_repair_still_says_point():
    words = layer(POS, ("Net exports", ["-.7", "1.2", "3.4"]))
    _, reps, _ = go(words, table(POS, ("Net exports", ["-7", "1.2", "3.4"])))
    assert [(r.old, r.new, r.kind) for r in reps] == [("-7", "-.7", "point")]


# ---------------------------------------------------------------------------
# Part 1: refusals
# ---------------------------------------------------------------------------


def test_s1_point_and_sign_both_off_is_refused():
    # the layer prints .7, the model wrote -7: the point AND the sign differ
    words = layer(NEG, ("Net exports", [".7", "2.5", "3.5"]))
    new, reps, refs = go(words, table(NEG, ("Net exports", ["-7", "2.5", "3.5"])))
    assert reps == [] and why(refs) == {"pair:point_and_sign"}, (reps, refs)
    assert "-7" in new


@pytest.mark.parametrize("shipped,printed", [("0.0", "-0.0"), ("-0.0", "0.0")])
def test_s2_zero_is_never_sign_repaired(shipped, printed):
    words = layer(NEG, ("Net exports", [printed, "2.5", "3.5"]))
    _, reps, refs = go(words, table(NEG, ("Net exports", [shipped, "2.5", "3.5"])))
    assert reps == [] and why(refs) == {"pair:sign_of_zero"}, (reps, refs)


def test_s3_a_detached_minus_glyph_beside_the_number_means_the_layer_says_negative():
    # layer: `- 1.5` (a thin minus glyph of its own, 2 pt left of the number, on a line of its
    # own).  The model's -1.5 is right; removing its minus would write a wrong number, and nothing
    # else (label gate, binding) can see the glyph.
    words = layer(NEG, ("Net exports", ["4.5", "~1.5", "3.5"]))
    shipped = table(NEG, ("Net exports", ["4.5", "-1.5", "3.5"]))
    new, reps, refs = go(words, shipped)
    assert reps == [] and why(refs) == {REFUSE_SIGN_GEOMETRY}, (reps, refs)
    assert new == shipped


def test_s3_a_detached_minus_before_the_first_value_is_stopped_earlier_by_the_label_rule():
    # the glyph sits in the printed label (`Net exports -`), so the strict label gate refuses it
    words = layer(NEG, ("Net exports", ["^1.5", "2.5", "3.5"]))
    _, reps, refs = go(words, table(NEG, ("Net exports", ["-1.5", "2.5", "3.5"])))
    assert reps == [] and "label_not_exact" in why(refs), (reps, refs)


def test_s3_control_the_glyph_that_belongs_to_the_previous_number_does_not_attach():
    # a minus glyph sits right after the PREVIOUS number (2 pt) and 34 pt before this one: it
    # belongs to the left, so this number is positive and the model's `-1.5` is the flip.
    words = layer(NEG, ("Net exports", ["9.5~", "1.5", "3.5"]))
    _, reps, refs = go(words, table(NEG, ("Net exports", ["9.5", "-1.5", "3.5"])))
    assert [(r.old, r.new) for r in reps] == [("-1.5", "1.5")], (reps, refs)


def test_s3_a_layer_value_negative_only_by_parentheses_is_notation_not_evidence():
    words = layer(NEG, ("Net exports", ["(1.5)", "2.5", "3.5"]))
    shipped = table(NEG, ("Net exports", ["1.5", "2.5", "3.5"]))
    new, reps, refs = go(words, shipped)
    assert reps == [] and new == shipped, (reps, refs)
    assert REFUSE_SIGN_GEOMETRY in why(refs)


def test_s4_a_table_whose_layer_shows_no_minus_does_not_vouch_for_a_sign():
    words = layer(POS, ("Net exports", ["-1.5", "2.5", "3.5"]))
    shipped = table(POS, ("Net exports", ["1.5", "2.5", "3.5"]))
    new, reps, refs = go(words, shipped)
    assert reps == [] and why(refs) == {REFUSE_SIGN_NO_MINUS_EVIDENCE}, (reps, refs)
    assert new == shipped
    # control: the same table with one OTHER negative value that agrees is repaired
    _, reps, _ = go(
        layer(NEG, ("Net exports", ["-1.5", "2.5", "3.5"])),
        table(NEG, ("Net exports", ["1.5", "2.5", "3.5"])),
    )
    assert [(r.old, r.new) for r in reps] == [("1.5", "-1.5")]


def test_s4_the_negative_value_may_sit_on_the_row_itself():
    words = layer(POS, ("Net exports", ["-1.5", "-2.5", "3.5"]))
    _, reps, refs = go(words, table(POS, ("Net exports", ["1.5", "-2.5", "3.5"])))
    assert [(r.old, r.new) for r in reps] == [("1.5", "-1.5")], (reps, refs)


@pytest.mark.parametrize("far", [1, 9], ids=["adjacent", "far_away"])
def test_s5_a_mirror_row_with_the_same_absolute_values_refuses_at_any_distance(far):
    filler = [(f"Filler {i}", [f"{i}.1", f"{i}.2", f"{i}.3"]) for i in range(far - 1)]
    own = ("Net exports", ["-1.5", "2.5", "3.5"])
    mirror = ("Previous", ["1.5", "-2.5", "-3.5"])
    words = layer(NEG, own, *filler, mirror)
    shipped = table(NEG, ("Net exports", ["1.5", "2.5", "3.5"]), *filler)
    new, reps, refs = go(words, shipped)
    assert reps == [] and why(refs) == {REFUSE_SIGN_MIRROR}, (reps, refs)
    assert new == shipped
    # control: the other line differs in one absolute value, so it mirrors nothing
    words = layer(NEG, own, *filler, ("Previous", ["1.6", "-2.5", "-3.5"]))
    _, reps, refs = go(words, shipped)
    assert [(r.old, r.new) for r in reps] == [("1.5", "-1.5")], (reps, refs)


def test_a_same_label_line_close_to_the_row_is_refused_by_the_existing_rival_rule():
    words = layer(NEG, ("Previous", ["-1.5", "2.5", "3.5"]), ("Previous", ["-1.5", "2.6", "3.5"]))
    _, reps, refs = go(words, table(NEG, ("Previous", ["1.5", "2.5", "3.5"])))
    assert reps == [] and REFUSE_RIVAL_LINE in why(refs), (reps, refs)


def test_a_cell_holding_a_detached_minus_token_is_not_edited_by_token():
    words = layer(NEG, ("Net exports", ["-1.5", "2.5", "3.5"]))
    _, reps, refs = go(words, table(NEG, ("Net exports", ["- 1.5", "2.5", "3.5"])))
    assert reps == [] and REFUSE_TOKEN_MAP in why(refs), (reps, refs)


def test_a_row_whose_values_all_differ_in_sign_is_refused_by_the_agreeing_floor():
    # both values differ, so none agrees: the existing MIN_AGREEING_VALUES floor refuses it
    words = layer(NEG, ("Net exports", ["-1.5", "-2.5"]))
    _, reps, refs = go(words, table(NEG, ("Net exports", ["1.5", "2.5"])))
    assert reps == [] and why(refs) == {rbc.REFUSE_NO_AGREEING}, (reps, refs)


@pytest.mark.parametrize(
    "printed",
    ["−1.5", "–1.5", "-1.5", "(1.5)"],
    ids=["unicode_minus", "en_dash", "hyphen", "parentheses"],
)
def test_s6_notation_is_equal_not_a_contradiction(printed):
    # the model wrote -1.5; the page prints the same number in another notation: no repair, no refusal
    words = layer(NEG, ("Net exports", [printed, "2.5", "3.5"]))
    shipped = table(NEG, ("Net exports", ["-1.5", "2.5", "3.5"]))
    new, reps, refs = go(words, shipped)
    assert (new, reps, refs) == (shipped, [], [])


def test_s6_a_footnote_digit_glued_to_the_printed_value_is_not_a_contradiction():
    words = layer(NEG, ("Net exports", ["1.5¹", "2.5", "3.5"]))
    shipped = table(NEG, ("Net exports", ["-1.5", "2.5", "3.5"]))
    new, reps, refs = go(words, shipped)
    assert (new, reps) == (shipped, [])


# ---------------------------------------------------------------------------
# Review round (PR #1304): range dash, glyph outside the band, glyph overlapping the number
# ---------------------------------------------------------------------------

RANGE = ("Central tendency", ["2.3", "\u20132.6", "4.4"])
RANGE_SHIPPED = ("Central tendency", ["2.3", "2.6", "4.4"])


DASHES = ["-", "\u2212", "\u2013", "\u2014", "\u2012"]
DASH_IDS = ["hyphen_minus", "u2212", "en", "em", "figure"]


@pytest.mark.parametrize("dash", DASHES, ids=DASH_IDS)
def test_a_range_shaped_dash_after_a_number_is_not_added_as_a_minus(dash):
    # `2.3 -2.6`: the dash glued to the second number may separate two numbers, whatever its glyph.
    # Another row of the table carries a real negative (NEG): it says nothing about this dash.
    words = layer(NEG, ("Central tendency", ["2.3", dash + "2.6", "4.4"]))
    shipped = table(NEG, RANGE_SHIPPED)
    new, reps, refs = go(words, shipped)
    assert reps == [] and why(refs) == {REFUSE_SIGN_GEOMETRY}, (reps, refs)
    assert new == shipped


@pytest.mark.parametrize("dash", ["-", "\u2013"], ids=["hyphen_minus", "en"])
def test_a_row_of_two_ranges_is_refused_even_though_each_range_looks_like_a_negative(dash):
    # each dash is evidence of a minus for the other: the strict rule takes no such support
    words = layer(NEG, ("Central tendency", ["2.3", dash + "2.6", "3.1", dash + "3.4"]))
    shipped = table(NEG, ("Central tendency", ["2.3", "2.6", "3.1", "3.4"]))
    new, reps, refs = go(words, shipped)
    assert reps == [] and why(refs) == {REFUSE_SIGN_GEOMETRY}, (reps, refs)
    assert new == shipped


def test_a_range_dash_is_refused_even_when_the_row_has_an_agreeing_negative():
    # `-0.3 \u20130.1 1.2`: a negative lower bound then a range dash.  The agreeing `-0.3` is no
    # evidence about the second dash.
    words = layer(NEG, ("Central tendency", ["-0.3", "\u20130.1", "1.2"]))
    shipped = table(NEG, ("Central tendency", ["-0.3", "0.1", "1.2"]))
    new, reps, refs = go(words, shipped)
    assert reps == [] and why(refs) == {REFUSE_SIGN_GEOMETRY}, (reps, refs)
    assert new == shipped


@pytest.mark.parametrize("dash", DASHES, ids=DASH_IDS)
def test_a_unary_dash_still_repairs(dash):
    # first value of the row (nothing on its left, or only its label: a word that is not a number)
    for label, values, shipped_values in (
        ("Central tendency", [dash + "2.6", "2.3", "4.4"], ["2.6", "2.3", "4.4"]),
        ("Central tendency x", [dash + "2.6", "4.4", "5.5"], ["2.6", "4.4", "5.5"]),
    ):
        words = layer(NEG, (label, values))
        _, reps, refs = go(words, table(NEG, (label, shipped_values)))
        assert [r.kind for r in reps] == ["sign"], (values, reps, refs)


@pytest.mark.parametrize("extent", ["above", "thin"])
def test_a_detached_minus_outside_the_numbers_strict_band_is_still_seen(extent):
    # layer `-3.0` as a glyph word whose box ends 2 pt above the number ("above") or sits inside
    # its band ("thin"): the model's -3.0 is right, removing the minus would be a wrong number.
    words = layer(NEG, ("Net exports", ["4.5", (("-", "3.0", extent, "number")), "3.5"]))
    shipped = table(NEG, ("Net exports", ["4.5", "-3.0", "3.5"]))
    new, reps, refs = go(words, shipped)
    assert reps == [] and why(refs) == {REFUSE_SIGN_GEOMETRY}, (reps, refs)
    assert new == shipped


def test_a_detached_minus_further_than_the_band_slack_is_not_attached():
    # control: the glyph is 10 pt above the number (slack is SIGN_BAND_TOL of its height, 5 pt)
    words = layer(NEG, ("Net exports", ["4.5", ("-", "3.0", "far", "number"), "3.5"]))
    _, reps, refs = go(words, table(NEG, ("Net exports", ["4.5", "-3.0", "3.5"])))
    assert [(r.old, r.new) for r in reps] == [("-3.0", "3.0")], (reps, refs)


def test_a_minus_box_that_overlaps_the_number_box_counts_as_attached():
    # the glyph box starts at the number's own left edge: the strict left-edge test used to drop it
    class T:  # the number word, at x 260-278, y 20-30
        x0, x1, y0, y1, text = 260.0, 278.0, 20.0, 30.0, "3.0"

    overlapping = [(260.0, 21.0, 264.0, 23.0, "-")]
    assert rbc._detached_minus_attached(overlapping, T) is True
    assert rbc._detached_minus_attached([(270.0, 21.0, 276.0, 23.0, "-")], T) is True  # inside it
    assert (
        rbc._detached_minus_attached([(100.0, 21.0, 104.0, 23.0, "-")], T) is True
    )  # no word on its left
    far = [(220.0, 21.0, 224.0, 23.0, "-"), (200.0, 20.0, 218.0, 30.0, "9.5")]
    assert rbc._detached_minus_attached(far, T) is False  # nearer the previous value, control


# The glyph is kept in the number's line by the grouping (tall), so the label and binding gates see
# an extra word.  Which refusal wins is pinned: it is not the geometry rule that stops these, and
# a later change to the binding condition would open them.
ODD_GLYPHS = [
    "\u2010",  # hyphen
    "\uf02d",  # private-use minus of a symbol font
    "\ufffd",  # replacement character
    "\u00ad",  # soft hyphen
    "\uff0d",  # fullwidth hyphen-minus
    "~",
]


def test_a_tall_glyph_nearer_the_previous_value_is_refused_by_the_binding_gate_not_by_geometry():
    words = layer(NEG, ("Net exports", ["4.5", ("-", "1.5", "tall", "previous"), "3.5"]))
    shipped = table(NEG, ("Net exports", ["4.5", "-1.5", "3.5"]))
    new, reps, refs = go(words, shipped)
    assert reps == [] and why(refs) == {rbc.REFUSE_NOT_BOUND}, (reps, refs)
    assert new == shipped


def test_the_same_tall_glyph_as_a_hyphen_minus_is_refused_by_geometry_instead():
    words = layer(NEG, ("Net exports", ["4.5", ("-", "1.5", "tall", "number"), "3.5"]))
    _, reps, refs = go(words, table(NEG, ("Net exports", ["4.5", "-1.5", "3.5"])))
    assert reps == [] and why(refs) == {REFUSE_SIGN_GEOMETRY}, (reps, refs)


@pytest.mark.parametrize("glyph", ODD_GLYPHS)
def test_an_odd_glyph_beside_the_number_is_refused_by_the_binding_gate_not_by_geometry(glyph):
    words = layer(NEG, ("Net exports", ["4.5", (glyph, "1.5", "tall", "number"), "3.5"]))
    shipped = table(NEG, ("Net exports", ["4.5", "-1.5", "3.5"]))
    new, reps, refs = go(words, shipped)
    assert reps == [], (reps, refs)
    assert why(refs) == {rbc.REFUSE_NOT_BOUND}, refs
    assert new == shipped


# ---------------------------------------------------------------------------
# Part 2: the page entry at rotation 0 / 90 / 270
# ---------------------------------------------------------------------------

fitz = pytest.importorskip("fitz")

from test_gh1267_row_binding_check import ROTATIONS, build_page  # noqa: E402
from test_gh1282_point_repair_review_fixes import block, cell, live, why as page_why  # noqa: E402

PAGE_NATIVE = [
    ("Real GDP", ["1.1", "-2.2", "3.3"]),
    ("Net exports", ["-1.5", "2.5", "3.5"]),
    ("Exports", ["2.1", "2.2", "2.3"]),
    ("Imports", ["3.1", "3.2", "3.3"]),
]


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_page_a_dropped_minus_is_restored(rotation):
    shipped = block([PAGE_NATIVE[0], ("Net exports", ["1.5", "2.5", "3.5"]), *PAGE_NATIVE[2:]])
    new, reps, refs = live(build_page(PAGE_NATIVE, rotation), shipped + "\n")
    assert cell(new, 0, "Net exports", 0) == "-1.5", (reps, refs)
    assert [(r.old, r.new, r.kind) for _, rs in reps for r in rs] == [("1.5", "-1.5", "sign")]


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_page_a_spurious_minus_is_removed_and_a_detached_one_is_not(rotation):
    native = [PAGE_NATIVE[0], ("Net exports", ["1.5", "2.5", "3.5"]), *PAGE_NATIVE[2:]]
    shipped = block([PAGE_NATIVE[0], ("Net exports", ["-1.5", "2.5", "3.5"]), *PAGE_NATIVE[2:]])
    new, reps, refs = live(build_page(native, rotation), shipped + "\n")
    assert cell(new, 0, "Net exports", 0) == "1.5", (reps, refs)
    # the layer prints `- 1.5` (an inline minus glyph beside the number): the same shipped cell
    # is right, and the repair must leave it (here the binding gate sees the extra word)
    detached = [PAGE_NATIVE[0], ("Net exports", ["4.5", "- 1.5", "3.5"]), *PAGE_NATIVE[2:]]
    shipped2 = block([PAGE_NATIVE[0], ("Net exports", ["4.5", "-1.5", "3.5"]), *PAGE_NATIVE[2:]])
    new2, reps2, refs2 = live(build_page(detached, rotation), shipped2 + "\n")
    assert cell(new2, 0, "Net exports", 1) == "-1.5", (reps2, refs2)


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_page_a_mirror_row_refuses(rotation):
    native = [
        PAGE_NATIVE[0],
        ("Net exports", ["-1.5", "2.5", "3.5"]),
        ("Exports", ["2.1", "2.2", "2.3"]),
        ("Revision", ["1.5", "-2.5", "-3.5"]),
    ]
    shipped = block([PAGE_NATIVE[0], ("Net exports", ["1.5", "2.5", "3.5"]), *native[2:]])
    new, reps, refs = live(build_page(native, rotation), shipped + "\n")
    assert cell(new, 0, "Net exports", 0) == "1.5", (reps, refs)
    assert REFUSE_SIGN_MIRROR in page_why(refs)


# A minus DRAWN as a vector segment is absent from the page words: the layer prints `1.5`, the page
# shows `-1.5`, and the model's minus is right.  Removing it is refused when such a segment sits
# attached to the number (DRAWN_MINUS_MAX_NUMBER_WIDTHS) or when the drawings cannot be read.
DRAWN_POS = ("Net exports", ["4.5", "3.0", "3.5"])
DRAWN_SHIPPED = ("Net exports", ["4.5", "-3.0", "3.5"])
NUMBER_X = 260.0  # the layer helper puts the second value here (y = 20 * row, height H)


def _drawn_case(drawn):
    words = layer(NEG, DRAWN_POS)
    shipped = table(NEG, DRAWN_SHIPPED)
    new, reps, refs = propose_point_repairs(words, None, shipped, drawn)
    return shipped, new, reps, refs


def test_a_drawn_minus_attached_to_the_number_refuses_the_removal():
    short = (NUMBER_X - 7.0, 24.5, NUMBER_X - 3.0, 24.5)  # 20 + 4.5: inside the number's band
    shipped, new, reps, refs = _drawn_case([short])
    assert reps == [] and why(refs) == {REFUSE_SIGN_GEOMETRY}, (reps, refs)
    assert new == shipped


def test_the_same_page_without_the_drawn_minus_removes_it():  # control: the guard is the drawing
    _, _, reps, refs = _drawn_case([])
    assert [(r.old, r.new) for r in reps] == [("-3.0", "3.0")], (reps, refs)


@pytest.mark.parametrize(
    "box,reason",
    [
        ((NUMBER_X - 80.0, 24.5, NUMBER_X - 3.0, 24.5), "a table rule is longer than the number"),
        ((NUMBER_X - 7.0, 60.0, NUMBER_X - 3.0, 60.0), "outside the number's band"),
        ((NUMBER_X - 7.0, 24.0, NUMBER_X - 6.0, 40.0), "vertical, not horizontal"),
        ((222.0, 24.5, 226.0, 24.5), "attached to the previous value, not to the number"),
    ],
    ids=["long_rule", "other_band", "vertical", "previous_value"],
)
def test_a_drawing_that_is_not_a_minus_before_the_number_does_not_block_the_removal(box, reason):
    _, _, reps, refs = _drawn_case([box])
    assert [(r.old, r.new) for r in reps] == [("-3.0", "3.0")], (reason, reps, refs)


def test_drawings_that_cannot_be_read_refuse_the_removal_but_not_the_addition():
    shipped, new, reps, refs = _drawn_case(None)
    assert reps == [] and why(refs) == {REFUSE_SIGN_GEOMETRY}, (reps, refs)
    assert new == shipped
    words = layer(NEG, ("Net exports", ["-1.5", "2.5", "3.5"]))
    _, reps, refs = propose_point_repairs(
        words, None, table(NEG, ("Net exports", ["1.5", "2.5", "3.5"])), None
    )
    assert [r.kind for r in reps] == ["sign"], (reps, refs)


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_page_a_drawn_minus_refuses_the_removal(rotation):
    from native_table_fixtures import place

    from test_gh1267_row_binding_check import HEIGHT, LABEL_U, V0, VALUE_U0, WIDTH  # noqa: F401
    from test_gh1267_row_binding_check import LINE_PITCH

    native = [PAGE_NATIVE[0], ("Net exports", ["1.5", "2.5", "3.5"]), *PAGE_NATIVE[2:]]
    shipped = block([PAGE_NATIVE[0], ("Net exports", ["-1.5", "2.5", "3.5"]), *PAGE_NATIVE[2:]])
    page = build_page(native, rotation)
    v = V0 + 1 * LINE_PITCH - 2.5  # mid-height of the digits of the "Net exports" row
    a = place(VALUE_U0 - 6.0, v, rotation, WIDTH, HEIGHT)
    b = place(VALUE_U0 - 2.0, v, rotation, WIDTH, HEIGHT)
    page.draw_line(a, b, width=0.5)
    new, reps, refs = live(page, shipped + "\n")
    assert cell(new, 0, "Net exports", 0) == "-1.5", (reps, refs)
    assert REFUSE_SIGN_GEOMETRY in page_why(refs)


# ---------------------------------------------------------------------------
# Part 3: process(), the arms differ exactly by the step, in both provider states
# ---------------------------------------------------------------------------

from test_gh1267_row_binding_pipeline import _events as _events_of  # noqa: E402
from test_gh1267_row_binding_pipeline import _fragment, _run, _sidecar  # noqa: E402
from test_gh1282_point_repair_pipeline import KIND, _noop  # noqa: E402

from socr.pipeline.orchestrator import UnifiedPipeline  # noqa: E402

P_NATIVE = [
    ("Real GDP", ["1.1", "-2.2", "3.3"]),
    ("Net exports", ["-1.5", "2.5", "3.5"]),
    ("Exports", ["2.1", "2.2", "2.3"]),
    ("Imports", ["3.1", "3.2", "3.3"]),
]
# the model dropped the minus of -1.5, and slipped a label so the page stays marked in BOTH arms
P_DROPPED = [
    P_NATIVE[0],
    ("Net exports", ["1.5", "2.5", "3.5"]),
    P_NATIVE[2],
    ("Imxorts", P_NATIVE[3][1]),
]


def _events(out_dir, kind):
    """The events of *kind*; a run with no event at all may write no audit log."""
    try:
        return _events_of(out_dir, kind)
    except StopIteration:
        return []


def _pair(tmp_path, rows, rotation, providers, native=P_NATIVE):
    kw = dict(check_runs=True, native=native, providers=providers)
    _, _, _, on, pdf = _run(tmp_path, rows, rotation, out_name="on", **kw)
    with patch.object(UnifiedPipeline, "_repair_dropped_points", _noop):
        _, _, _, off, _ = _run(tmp_path, rows, rotation, out_name="off", pdf=pdf, **kw)
    return on, off


@pytest.mark.parametrize("providers", [True, False], ids=["ladder", "no_ladder"])
@pytest.mark.parametrize("rotation", ROTATIONS)
def test_process_the_step_restores_exactly_the_minus_and_records_its_kind(
    tmp_path, rotation, providers
):
    on, off = _pair(tmp_path, P_DROPPED, rotation, providers)
    if not providers:
        # no model reads the page: the step has nothing to repair and the arms cannot differ
        assert _fragment(on) == _fragment(off)
        assert _events(on, KIND) == _events(off, KIND) == []
        return
    f_on, f_off = _fragment(on), _fragment(off)
    assert "| -1.5 |" in f_on and "| -1.5 |" not in f_off
    assert "| 1.5 |" in f_off and "| 1.5 |" not in f_on
    assert _events(off, KIND) == []
    (event,) = _events(on, KIND)
    (row,) = event["data"]["rows"]
    assert (row["label"], row["old"], row["new"], row["kind"]) == (
        "Net exports",
        "1.5",
        "-1.5",
        "sign",
    )
    assert "flipped the minus sign" in event["detail"]
    assert _sidecar(on)["row_binding"]["p1-t0"]["repairs"] == event["data"]["rows"]
    assert "repairs" not in _sidecar(off)["row_binding"]["p1-t0"]


@pytest.mark.parametrize("providers", [True, False], ids=["ladder", "no_ladder"])
@pytest.mark.parametrize("rotation", ROTATIONS)
def test_process_a_mirror_row_leaves_the_page_exactly_as_read(tmp_path, rotation, providers):
    """The scope guard, as a DIFFERENCE: with the mirror row on the page the arms are identical
    (the step has a refusal to make and makes it).  The control, where the step does repair a
    dropped minus and the arms differ, is the test above."""
    # six lines: the harness sends a four-line page to the native lane at rotation 90, where no
    # model table exists and the test would be vacuous
    tail = [
        ("Previous", ["1.5", "-2.5", "-3.5"]),
        ("Other", ["7.1", "7.2", "7.3"]),
        ("More", ["8.1", "8.2", "8.3"]),
    ]
    mirrored = [*P_NATIVE[:3], *tail]
    shipped = [P_NATIVE[0], ("Net exports", ["1.5", "2.5", "3.5"]), P_NATIVE[2], *tail]
    on, off = _pair(tmp_path, shipped, rotation, providers, native=mirrored)
    assert _fragment(on) == _fragment(off)
    assert _events(on, KIND) == _events(off, KIND) == []
