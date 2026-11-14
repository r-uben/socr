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
        if v[0] in "~^":
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
    words = layer(NEG, ("Net exports", ["-.7", "-2.5", "3.5", "4.5"]))
    shipped = table(NEG, ("Net exports", ["-7", "2.5", "3.5", "4.5"]))
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


def test_a_row_with_one_agreeing_value_is_refused_by_the_existing_floor():
    words = layer(NEG, ("Net exports", ["-1.5", "-2.5"]))
    _, reps, _ = go(words, table(NEG, ("Net exports", ["1.5", "2.5"])))
    assert reps == []


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
