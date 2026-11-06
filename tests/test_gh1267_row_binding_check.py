"""#1267: the pure row-binding check, on synthetic fitz PDFs at rotation 0, 90 and 270.

Differences are pinned where possible: the same page is checked against the faithful
markdown and against a shifted one, and only the verdicts of the shifted rows change.
"""

from __future__ import annotations

import fitz
import pytest
from native_table_fixtures import place

from socr.tables.row_binding_check import (
    RowBindingResult,
    RowVerdict,
    TableOutcome,
    check_page_row_binding,
)

ROTATIONS = (0, 90, 270)
WIDTH, HEIGHT = 612, 792
LABEL_U = 72.0
VALUE_U0 = 330.0
VALUE_STEP = 45.0
LINE_PITCH = 18.0
V0 = 100.0
FONT = 9


def build_page(lines: list[tuple[str, list[str]]], rotation: int) -> fitz.Page:
    """One text line per ``(label, values)``: label at LABEL_U, values to the right, upright frame."""
    doc = fitz.open()
    page = doc.new_page(width=WIDTH, height=HEIGHT)

    def text(u: float, v: float, s: str) -> None:
        page.insert_text(
            place(u, v, rotation, WIDTH, HEIGHT), s, fontsize=FONT, fontname="helv", rotate=rotation
        )

    for i, (label, values) in enumerate(lines):
        v = V0 + i * LINE_PITCH
        if label:
            text(LABEL_U, v, label)
        for k, val in enumerate(values):
            text(VALUE_U0 + k * VALUE_STEP, v, val)
    return page


def md(rows: list[tuple[str, list[str]]], header: str = "| Item | A | B | C |") -> str:
    out = [header, "|---|---|---|---|"]
    for label, values in rows:
        cells = list(values) + [""] * (3 - len(values)) if values else ["", "", ""]
        out.append("| " + " | ".join([label, *cells]) + " |")
    return "\n".join(out) + "\n"


def run(native, rows, rotation) -> RowBindingResult:
    page = build_page(native, rotation)
    results, rot = check_page_row_binding(page, md(rows))
    assert rot in (0, 90, 270)
    assert len(results) == 1
    return results[0]


def verdicts(res: RowBindingResult) -> list[RowVerdict]:
    return [r.verdict for r in res.rows]


GROUP = [
    ("Real GDP", ["1.1", "2.2", "3.3"]),
    ("Final sales", ["4.4", "5.5", "6.6"]),
    ("Priv. dom. final purchases", ["7.7", "8.8", "9.9"]),
    ("Net exports", ["1.5", "2.5", "3.5"]),
]


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_correct_table_is_bound(rotation):
    res = run(GROUP, GROUP, rotation)
    assert res.outcome is TableOutcome.BOUND
    assert verdicts(res) == [RowVerdict.BOUND] * 4


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_group_shifted_one_row_down_is_contradicted(rotation):
    """Parent emptied, its values under the next label, the last row unlabelled (canary shape)."""
    shifted = [
        ("Real GDP", GROUP[0][1]),
        ("Final sales", []),
        ("Priv. dom. final purchases", GROUP[1][1]),
        ("", GROUP[2][1]),
        ("Net exports", GROUP[3][1]),
    ]
    clean = run(GROUP, GROUP, rotation)
    bad = run(GROUP, shifted, rotation)
    assert clean.outcome is TableOutcome.BOUND
    assert bad.outcome is TableOutcome.CONTRADICTED
    by_label = {r.shipped_label: r.verdict for r in bad.rows}
    assert by_label["Final sales"] is RowVerdict.EMPTIED
    assert by_label["Priv. dom. final purchases"] is RowVerdict.DISPLACED
    assert by_label[""] is RowVerdict.LABEL_LOST
    # the rows the shift did not touch keep the verdict they had on the faithful table
    assert by_label["Real GDP"] is RowVerdict.BOUND
    assert by_label["Net exports"] is RowVerdict.BOUND


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_shift_without_empty_parent_is_displaced_only(rotation):
    """Labels slid one row down without an emptied parent: no EMPTIED, still contradicted."""
    swapped = [
        ("Real GDP", GROUP[0][1]),
        ("Priv. dom. final purchases", GROUP[1][1]),
        ("Final sales", GROUP[2][1]),
        ("Net exports", GROUP[3][1]),
    ]
    res = run(GROUP, swapped, rotation)
    assert res.outcome is TableOutcome.CONTRADICTED
    assert res.count(RowVerdict.DISPLACED) == 2
    assert res.count(RowVerdict.EMPTIED) == 0


ZEROS = [
    ("Alpha loans", ["0", "0", "0"]),
    ("Beta loans", ["0", "0", "0"]),
    ("Gamma loans", ["0", "0", "0"]),
]


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_all_zero_rows_in_order_are_bound(rotation):
    assert run(ZEROS, ZEROS, rotation).outcome is TableOutcome.BOUND


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_all_zero_row_dropped_is_unverifiable_not_contradicted(rotation):
    """Alpha and Gamma ship, Beta is missing: the order cannot say which zero line Gamma is."""
    res = run(ZEROS, [ZEROS[0], ZEROS[2]], rotation)
    assert res.outcome is TableOutcome.UNVERIFIABLE
    assert RowVerdict.UNVERIFIABLE in verdicts(res)
    assert not {RowVerdict.DISPLACED, RowVerdict.LABEL_LOST, RowVerdict.EMPTIED} & set(
        verdicts(res)
    )


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_identical_value_rows_with_right_labels_are_bound(rotation):
    native = [
        ("One", ["3.1", "3.1", "3.1"]),
        ("Two", ["4.0", "4.1", "4.2"]),
        ("Three", ["3.1", "3.1", "3.1"]),
    ]
    assert run(native, native, rotation).outcome is TableOutcome.BOUND


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_wrapped_two_line_label_is_bound(rotation):
    native = [
        ("Cash operating balance,", []),
        ("end of period", ["13", "24", "30"]),
        ("Receipts", ["1775", "1825", "1853"]),
    ]
    rows = [
        ("Cash operating balance, end of period", ["13", "24", "30"]),
        ("Receipts", ["1775", "1825", "1853"]),
    ]
    assert run(native, rows, rotation).outcome is TableOutcome.BOUND


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_split_label_band_previous_is_bound(rotation):
    """'Previous' printed on its own baseline above its values (Greenbook p34 shape)."""
    native = [
        ("CPI", ["1.7", "1.8", "1.9"]),
        ("Previous", []),
        ("", ["1.6", "1.7", "1.8"]),
        ("Core CPI", ["2.0", "2.1", "2.2"]),
    ]
    rows = [
        ("CPI", ["1.7", "1.8", "1.9"]),
        ("Previous", ["1.6", "1.7", "1.8"]),
        ("Core CPI", ["2.0", "2.1", "2.2"]),
    ]
    res = run(native, rows, rotation)
    assert res.outcome is TableOutcome.BOUND, [
        (r.shipped_label, r.verdict, r.reason) for r in res.rows
    ]


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_valueless_group_header_is_bound(rotation):
    native = [
        ("Exports", []),
        ("Core goods", ["1.0", "2.0", "3.0"]),
        ("Services", ["4.0", "5.0", "6.0"]),
    ]
    rows = [
        ("Exports", []),
        ("Core goods", ["1.0", "2.0", "3.0"]),
        ("Services", ["4.0", "5.0", "6.0"]),
    ]
    res = run(native, rows, rotation)
    assert res.outcome is TableOutcome.BOUND
    assert res.count(RowVerdict.EMPTIED) == 0


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_categorical_row_displaced_onto_numeric_label(rotation):
    """p68: native 'Incentive to refinance | Small Moderate Large' then 'Percent of loans | 25 21 54'."""
    native = [
        ("Incentive to refinance", ["Small", "Moderate", "Large"]),
        ("Percent of loans", ["25", "21", "54"]),
    ]
    good = [
        ("Incentive to refinance", ["Small", "Moderate", "Large"]),
        ("Percent of loans", ["25", "21", "54"]),
    ]
    bad = [("Percent of loans", ["Small", "Moderate", "Large"]), ("", ["25", "21", "54"])]
    assert run(native, good, rotation).outcome is TableOutcome.BOUND
    res = run(native, bad, rotation)
    assert res.outcome is TableOutcome.CONTRADICTED
    assert verdicts(res) == [RowVerdict.DISPLACED, RowVerdict.LABEL_LOST]


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_emptied_row_with_values_printed(rotation):
    native = [
        ("US CURRENT ACCOUNT BALANCE", ["-100", "-110", "-120"]),
        ("Net exports", ["1.5", "2.5", "3.5"]),
    ]
    rows = [("US CURRENT ACCOUNT BALANCE", []), ("Net exports", ["1.5", "2.5", "3.5"])]
    res = run(native, rows, rotation)
    assert res.outcome is TableOutcome.CONTRADICTED
    assert verdicts(res) == [RowVerdict.EMPTIED, RowVerdict.BOUND]


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_leading_dot_values_bind_and_shift_is_seen(rotation):
    native = [("Fiscal impetus", [".6", "-.4", ".9"]), ("Output gap", ["-.1", ".2", "-.3"])]
    assert run(native, native, rotation).outcome is TableOutcome.BOUND
    shifted = [("Fiscal impetus", [".6", "-.4", ".9"]), ("Output gap", [".6", "-.4", ".9"])]
    # the second row carries the first row's values under the second label: no line carries them
    # under that label, so the leading-dot values still find their line and the label is rejected.
    res = run(native, shifted, rotation)
    assert RowVerdict.BOUND != res.rows[1].verdict


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_leading_dot_equals_zero_dot_form(rotation):
    native = [("Fiscal impetus", [".6", "-.4", ".9"])]
    rows = [("Fiscal impetus", ["0.6", "-0.4", "0.9"])]
    assert run(native, rows, rotation).outcome is TableOutcome.BOUND


def test_no_native_line_is_unverifiable_never_bound():
    res = run([("Other", ["9", "9", "9"])], [("Real GDP", ["1.1", "2.2", "3.3"])], 0)
    assert res.outcome is TableOutcome.UNVERIFIABLE
    assert verdicts(res) == [RowVerdict.UNVERIFIABLE]


def test_mirror_rotations_give_identical_verdicts():
    shifted = [
        ("Real GDP", GROUP[0][1]),
        ("Final sales", []),
        ("Priv. dom. final purchases", GROUP[1][1]),
        ("", GROUP[2][1]),
    ]
    outs = {
        rot: [(r.verdict, r.shipped_label) for r in run(GROUP, shifted, rot).rows]
        for rot in ROTATIONS
    }
    assert outs[0] == outs[90] == outs[270]


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_repeated_section_label_is_told_apart_by_position(rotation):
    """'Developing Countries' is printed in two blocks; only the second one lost its values."""
    native = [
        ("Developing Countries", ["1.0", "2.0", "3.0"]),
        ("Mexico", ["4.0", "5.0", "6.0"]),
        ("Previous forecast", []),
        ("Developing Countries", ["7.0", "8.0", "9.0"]),
        ("Brazil", ["1.5", "2.5", "3.5"]),
    ]
    intact = [
        ("Developing Countries", ["1.0", "2.0", "3.0"]),
        ("Mexico", ["4.0", "5.0", "6.0"]),
        ("Previous forecast", []),
        ("Developing Countries", ["7.0", "8.0", "9.0"]),
        ("Brazil", ["1.5", "2.5", "3.5"]),
    ]
    emptied = [*intact[:3], ("Developing Countries", []), intact[4]]
    assert run(native, intact, rotation).outcome is TableOutcome.BOUND
    res = run(native, emptied, rotation)
    assert res.outcome is TableOutcome.CONTRADICTED
    assert [r.verdict for r in res.rows if r.shipped_label == "Developing Countries"] == [
        RowVerdict.BOUND,
        RowVerdict.EMPTIED,
    ]


def test_twin_panels_on_shared_baselines_bind_in_reading_order():
    from socr.tables.row_binding_check import check_row_binding

    def word(x, y, t):
        return (x, y, x + 12 * len(t), y + 9, t, 0, 0, 0)

    words = []
    for i, (left, right) in enumerate([("Treasury", "Agency"), ("Bills", "Notes")]):
        y = 100 + i * 14
        words.append(word(40, y, left))
        words += [word(160 + 40 * k, y, f"{i + 1}.{k + 1}") for k in range(3)]
        words.append(word(400, y, right))
        words += [word(520 + 40 * k, y, f"{i + 5}.{k + 1}") for k in range(3)]
    good = "| a | b | c | d |\n|---|---|---|---|\n| Treasury | 1.1 | 1.2 | 1.3 |\n| Agency | 5.1 | 5.2 | 5.3 |\n"
    swapped = "| a | b | c | d |\n|---|---|---|---|\n| Agency | 1.1 | 1.2 | 1.3 |\n| Treasury | 5.1 | 5.2 | 5.3 |\n"
    assert check_row_binding(words, None, good).outcome is TableOutcome.BOUND
    assert check_row_binding(words, None, swapped).outcome is not TableOutcome.BOUND


# ---------------------------------------------------------------------------
# Argon's adversarial cases (scratch/army/1267/advice-argon.md): what the check does TODAY.
# Safe = a real defect is not BOUND, a clean layout is not CONTRADICTED.  Unsafe cases are
# xfail(strict=True) with the reason; they are not fixed here.
# ---------------------------------------------------------------------------

CONVICT = {RowVerdict.DISPLACED, RowVerdict.LABEL_LOST, RowVerdict.EMPTIED}
V3 = ["1", "2", "3"]


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_argon_a_wrong_label_from_far_away_is_not_bound(rotation):
    """The label of a row three lines away is stolen: unadjacent, so UNVERIFIABLE (safe, not convicted)."""
    native = [
        ("Alpha", V3),
        ("Beta", ["4", "5", "6"]),
        ("Gamma", ["7", "8", "9"]),
        ("Delta", ["1.5", "2.5", "3.5"]),
    ]
    rows = [native[0], native[1], native[2], ("Alpha", native[3][1])]
    res = run(native, rows, rotation)
    assert res.outcome is not TableOutcome.BOUND
    assert res.rows[3].verdict is RowVerdict.UNVERIFIABLE


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_argon_b_header_shifted_onto_first_data_row(rotation):
    native = [
        ("Exports", []),
        ("Core goods", ["1.0", "2.0", "3.0"]),
        ("Services", ["4.0", "5.0", "6.0"]),
    ]
    rows = [("Exports", ["1.0", "2.0", "3.0"]), ("Services", ["4.0", "5.0", "6.0"])]
    res = run(native, rows, rotation)
    assert res.outcome is TableOutcome.CONTRADICTED
    assert res.rows[0].verdict is RowVerdict.DISPLACED


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_argon_c_categorical_shift_with_correct_numeric_row(rotation):
    """p68 shape with the numeric row correctly labelled: the categorical row alone is convicted."""
    native = [
        ("Incentive to refinance", ["Small", "Moderate", "Large"]),
        ("Percent of loans", ["25", "21", "54"]),
    ]
    rows = [
        ("Percent of loans", ["Small", "Moderate", "Large"]),
        ("Percent of loans", ["25", "21", "54"]),
    ]
    res = run(native, rows, rotation)
    assert res.outcome is TableOutcome.CONTRADICTED
    assert [r.verdict for r in res.rows] == [RowVerdict.DISPLACED, RowVerdict.BOUND]


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_argon_c_categorical_row_label_dropped(rotation):
    native = [
        ("Incentive to refinance", ["Small", "Moderate", "Large"]),
        ("Percent of loans", ["25", "21", "54"]),
    ]
    rows = [("", ["Small", "Moderate", "Large"]), ("Percent of loans", ["25", "21", "54"])]
    res = run(native, rows, rotation)
    assert res.rows[0].verdict is RowVerdict.UNVERIFIABLE  # never BOUND, never convicted
    assert res.outcome is TableOutcome.UNVERIFIABLE


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_argon_d_one_value_row_emptied(rotation):
    native = [("Total", ["5.0"]), ("Net", ["1.0", "2.0", "3.0"])]
    rows = [("Total", []), ("Net", ["1.0", "2.0", "3.0"])]
    res = run(native, rows, rotation)
    assert res.rows[0].verdict is RowVerdict.UNVERIFIABLE  # never BOUND, never convicted
    assert res.outcome is TableOutcome.UNVERIFIABLE


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_argon_e_numeric_row_labels_never_convict(rotation):
    """Years, quarters and `2018: Q2` as the first cell: a clean table is never CONTRADICTED.
    Today the year rows come out UNVERIFIABLE (a whole-line and a values-only panel both match), not BOUND."""
    native = [
        ("2005", ["1.0", "2.0", "3.0"]),
        ("2006", ["4.0", "5.0", "6.0"]),
        ("2018: Q2", ["7", "8", "9"]),
        ("3", ["1.5", "2.5", "3.5"]),
    ]
    res = run(native, native, rotation)
    assert res.outcome is not TableOutcome.CONTRADICTED
    assert not CONVICT & set(verdicts(res))
    assert [r.verdict for r in res.rows[2:]] == [RowVerdict.BOUND, RowVerdict.BOUND]


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_argon_e_swapped_year_labels_are_not_bound(rotation):
    native = [("2005", ["1.0", "2.0", "3.0"]), ("2006", ["4.0", "5.0", "6.0"])]
    swapped = [("2006", ["1.0", "2.0", "3.0"]), ("2005", ["4.0", "5.0", "6.0"])]
    assert run(native, swapped, rotation).outcome is not TableOutcome.BOUND


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_argon_f_spaced_footnote_digit_never_convicts(rotation):
    """`prod. 4` (digit separated from its label) reads as a leading value; the clean row is not convicted.
    Today it is UNVERIFIABLE (the values are found only as a label-less panel), not BOUND."""
    native = [("Manufacturing prod. 4", ["-5.0", "-8.7", "1.2"]), ("Other", V3)]
    rows = [("Manufacturing prod. 4", ["-5.0", "-8.7", "1.2"]), ("Other", V3)]
    res = run(native, rows, rotation)
    assert not CONVICT & set(verdicts(res))
    assert res.rows[1].verdict is RowVerdict.BOUND


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_argon_g_footer_on_a_data_baseline_is_not_a_value(rotation):
    native = [("Defense", ["0.1", "0.2", "0.6", "I-19"]), ("Other", V3)]
    rows = [("Defense", ["0.1", "0.2", "0.6"]), ("Other", V3)]
    assert run(native, rows, rotation).outcome is TableOutcome.BOUND


@pytest.mark.xfail(
    strict=True,
    reason="UNSAFE: a printed row whose values appear nowhere in the shipped table is never looked for; "
    "the rows that did ship bind, so the table is BOUND",
)
@pytest.mark.parametrize("rotation", ROTATIONS)
def test_argon_h_dropped_printed_row(rotation):
    native = [("Alpha", V3), ("Beta", ["4", "5", "6"]), ("Gamma", ["7", "8", "9"])]
    assert run(native, [native[0], native[2]], rotation).outcome is not TableOutcome.BOUND


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_wrapped_label_split_over_two_markdown_rows_is_bound(rotation):
    """2004-12-14 greenbook p28: the markdown puts the wrapped label on an all-empty row and the values
    on the unlabelled row below.  The same shape with values on the row above is the canary's tail row."""
    native = [
        ("Gross saving less gross", []),
        ("investment3", ["-347", "-391", "-402"]),
        ("Net exports", ["1.5", "2.5", "3.5"]),
    ]
    split = [
        ("Gross saving less gross investment3", []),
        ("", ["-347", "-391", "-402"]),
        ("Net exports", ["1.5", "2.5", "3.5"]),
    ]
    res = run(native, split, rotation)
    assert RowVerdict.LABEL_LOST not in verdicts(res)
    assert res.rows[1].verdict is RowVerdict.BOUND


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_group_tail_row_stays_label_lost_when_the_row_above_has_values(rotation):
    shifted = [
        ("Real GDP", GROUP[0][1]),
        ("Final sales", GROUP[1][1]),
        ("", GROUP[2][1]),
    ]
    res = run(GROUP[:3], shifted, rotation)
    assert res.rows[2].verdict is RowVerdict.LABEL_LOST


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_duplicate_shipped_row_against_one_native_line_is_never_bound(rotation):
    """One printed row, shipped twice: no one-to-one assignment exists, so nothing may be BOUND."""
    native = [("Alpha", ["1", "2"])]
    rows = [("Alpha", ["1", "2"]), ("Alpha", ["1", "2"])]
    page = build_page(native, rotation)
    markdown = "| Item | A | B |\n|---|---|---|\n| Alpha | 1 | 2 |\n| Alpha | 1 | 2 |\n"
    results, _ = check_page_row_binding(page, markdown, None)
    res = results[0]
    assert res.outcome is not TableOutcome.BOUND
    assert RowVerdict.BOUND not in verdicts(res)
    assert {r.reason for r in res.rows} == {"no one-to-one order-preserving assignment"}
    del rows


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_labelled_empty_row_with_no_native_line_is_unverifiable(rotation):
    """A label with no values that no native line carries: nothing can bind it (never BOUND)."""
    native = [("Alpha", ["1", "2", "3"])]
    rows = [("Alpha", ["1", "2", "3"]), ("Unprinted label", [])]
    res = run(native, rows, rotation)
    assert res.rows[1].verdict is RowVerdict.UNVERIFIABLE
    assert res.rows[1].reason == "no native line carries this label"
    assert res.outcome is TableOutcome.UNVERIFIABLE


def test_a_table_with_no_data_rows_has_nothing_to_bind():
    from socr.tables.row_binding_check import check_row_binding

    res = check_row_binding([], None, "| a | b |\n|---|---|\n")
    assert res.rows == () and res.outcome is TableOutcome.BOUND


@pytest.mark.parametrize(
    "kind", ["table_row_binding_contradicted", "table_row_binding_unverifiable"]
)
def test_each_row_binding_kind_alone_makes_a_page_untrusted_and_blocks_the_credential(kind):
    """The KIND is pinned, not just the page: a page whose only event is this kind is untrusted.

    (A withheld page is also listed through ``table_ladder_withheld``, which would mask a kind
    dropped from ``TABLE_DISTRUST_KINDS``.)"""
    from socr.core.audit_log import AuditEvent
    from socr.core.manifest import CREDENTIAL_BLOCKING_EVENT_KINDS
    from socr.core.tables_trust import (
        NON_RESOLVABLE_DISTRUST_KINDS,
        TABLE_DISTRUST_KINDS,
        build_tables_trust,
    )

    assert kind in TABLE_DISTRUST_KINDS
    assert kind in NON_RESOLVABLE_DISTRUST_KINDS
    assert kind in CREDENTIAL_BLOCKING_EVENT_KINDS
    trust = build_tables_trust(
        "doc.pdf",
        [AuditEvent(page_num=3, kind=kind, detail="table p3-t0", data={"table_id": "p3-t0"})],
    )
    assert trust.untrusted_pages == [3]
    assert trust.pages[3].reasons == [kind]


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_wrap_split_exemption_needs_a_real_continuation(rotation):
    """1B: `Goods and services` is a valueless header, `Services` a data line. The shipped header row
    followed by an unlabelled value row is NOT a wrapped label: the line's own label (`services`) adds
    nothing the band does not carry, so the values may belong to either. UNVERIFIABLE, never BOUND."""
    native = [("Goods and services", []), ("Services", ["1.0", "2.0", "3.0"])]
    rows = [("Goods and services", []), ("", ["1.0", "2.0", "3.0"])]
    res = run(native, rows, rotation)
    assert res.rows[1].verdict is RowVerdict.UNVERIFIABLE
    assert res.outcome is not TableOutcome.BOUND
    assert not CONVICT & set(verdicts(res))


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_argon_1a_partial_label_on_a_longer_native_label(rotation):
    """Merge gate P1 (#1271): `Government` is only a subset of `Federal government` and no line of the
    page prints `Government` as its own label. Nothing proves the label wrong, nothing proves it right:
    UNVERIFIABLE (marked), never BOUND. The faithful label on the same page stays BOUND."""
    native = [("Federal government", ["2.1", "2.2", "2.3"]), ("Net exports", ["1.5", "2.5", "3.5"])]
    rows = [("Government", ["2.1", "2.2", "2.3"]), ("Net exports", ["1.5", "2.5", "3.5"])]
    res = run(native, rows, rotation)
    assert res.rows[0].verdict is RowVerdict.UNVERIFIABLE
    assert res.outcome is TableOutcome.UNVERIFIABLE
    faithful = run(native, native, rotation)
    assert faithful.outcome is TableOutcome.BOUND


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_short_categorical_label_lost_is_not_vacuously_bound(rotation):
    """Merge gate P1 (#1271): the native label `US` is short. Dropping it from the shipped
    categorical row must not leave an empty token list that `all()` accepts."""
    native = [("US", ["Small", "Moderate", "Large"]), ("Percent of loans", ["25", "21", "54"])]
    dropped = [("", ["Small", "Moderate", "Large"]), ("Percent of loans", ["25", "21", "54"])]
    res = run(native, dropped, rotation)
    assert res.rows[0].verdict is RowVerdict.UNVERIFIABLE
    assert res.outcome is TableOutcome.UNVERIFIABLE
    faithful = run(native, native, rotation)
    assert faithful.outcome is TableOutcome.BOUND


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_argon_1a_one_shipped_label_for_two_native_labels(rotation):
    native = [
        ("Total domestic demand", ["2.1", "2.2", "2.3"]),
        ("Total foreign demand", ["1.5", "2.5", "3.5"]),
    ]
    rows = [("Total demand", ["2.1", "2.2", "2.3"]), ("Total demand", ["1.5", "2.5", "3.5"])]
    assert run(native, rows, rotation).outcome is not TableOutcome.BOUND


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_best_label_home_a_partial_label_whose_own_line_exists_is_displaced(rotation):
    """The canary shape: `Exports` shipped on the `Net exports` line while the page prints an
    `Exports` line of its own, next to it. That line is the label's home."""
    native = [("Exports", ["2.1", "2.2", "2.3"]), ("Net exports", ["1.5", "2.5", "3.5"])]
    rows = [("Exports", ["2.1", "2.2", "2.3"]), ("Exports", ["1.5", "2.5", "3.5"])]
    res = run(native, rows, rotation)
    assert res.rows[0].verdict is RowVerdict.BOUND
    assert res.rows[1].verdict is RowVerdict.DISPLACED
    assert res.outcome is TableOutcome.CONTRADICTED


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_best_label_home_a_distant_home_is_unverifiable_not_convicted(rotation):
    native = [
        ("Exports", ["2.1", "2.2", "2.3"]),
        ("Imports", ["3.1", "3.2", "3.3"]),
        ("Services", ["4.1", "4.2", "4.3"]),
        ("Net exports", ["1.5", "2.5", "3.5"]),
    ]
    rows = [native[0], native[1], native[2], ("Exports", ["1.5", "2.5", "3.5"])]
    res = run(native, rows, rotation)
    assert res.rows[3].verdict is RowVerdict.UNVERIFIABLE
    assert not CONVICT & {res.rows[3].verdict}


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_best_label_home_an_equal_label_stays_bound(rotation):
    native = [("Exports", ["2.1", "2.2", "2.3"]), ("Net exports", ["1.5", "2.5", "3.5"])]
    assert run(native, native, rotation).outcome is TableOutcome.BOUND


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_short_qualifiers_are_part_of_the_label(rotation):
    """Merge gate round 2 (Astra P0): `US government` and `UK government` swapped must not bind."""
    native = [("US government", ["2.1", "2.2", "2.3"]), ("UK government", ["1.5", "2.5", "3.5"])]
    swapped = [("UK government", ["2.1", "2.2", "2.3"]), ("US government", ["1.5", "2.5", "3.5"])]
    assert run(native, native, rotation).outcome is TableOutcome.BOUND
    res = run(native, swapped, rotation)
    assert res.outcome is TableOutcome.CONTRADICTED
    assert verdicts(res) == [RowVerdict.DISPLACED, RowVerdict.DISPLACED]


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_a_bare_digit_in_a_label_never_lets_a_swap_bind(rotation):
    """`Tier 1` / `Tier 2` swapped. The digit is also a number on the text-layer line, so the check
    cannot place these rows: UNVERIFIABLE (marked), never BOUND."""
    native = [("Tier 1 capital", ["2.1", "2.2", "2.3"]), ("Tier 2 capital", ["1.5", "2.5", "3.5"])]
    swapped = [("Tier 2 capital", ["2.1", "2.2", "2.3"]), ("Tier 1 capital", ["1.5", "2.5", "3.5"])]
    assert run(native, swapped, rotation).outcome is not TableOutcome.BOUND


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_a_footnote_mark_is_not_part_of_the_label(rotation):
    """`(4)`, `¹` and `$^{2}$` after a label are footnote marks the layer prints apart or not at all;
    the label still binds, and a swap under footnote marks is still convicted."""
    native = [("Consumer prices", ["2.1", "2.2", "2.3"]), ("Real GDP", ["1.5", "2.5", "3.5"])]
    footnoted = [
        ("Consumer prices (4)", ["2.1", "2.2", "2.3"]),
        ("Real GDP¹", ["1.5", "2.5", "3.5"]),
    ]
    assert run(native, footnoted, rotation).outcome is TableOutcome.BOUND
    latex = [
        ("**Consumer prices**$^{2}$", ["2.1", "2.2", "2.3"]),
        ("Real GDP", ["1.5", "2.5", "3.5"]),
    ]
    assert run(native, latex, rotation).outcome is TableOutcome.BOUND
    swapped = [("Real GDP (4)", ["2.1", "2.2", "2.3"]), ("Consumer prices¹", ["1.5", "2.5", "3.5"])]
    assert run(native, swapped, rotation).outcome is TableOutcome.CONTRADICTED


@pytest.mark.parametrize("rotation", ROTATIONS)
@pytest.mark.parametrize(
    "label, cells",
    [("US", ["Business", "Moderate", "Large"]), ("Total", ["Subtotal", "Moderate", "Large"])],
)
def test_a_lost_label_inside_a_word_cell_is_not_bound(rotation, label, cells):
    """Merge gate round 2 (Astra P1 / Fable P2): `us` inside `business`, `total` inside `subtotal` are
    substrings, not the label. Token equality only."""
    native = [(label, cells), ("Percent of loans", ["25", "21", "54"])]
    dropped = [("", cells), ("Percent of loans", ["25", "21", "54"])]
    res = run(native, dropped, rotation)
    assert res.rows[0].verdict is RowVerdict.UNVERIFIABLE
    assert run(native, native, rotation).outcome is TableOutcome.BOUND


def _v(a: int) -> list[str]:
    return [f"{a}.1", f"{a}.2", f"{a}.3"]


@pytest.mark.parametrize("rotation", ROTATIONS)
@pytest.mark.parametrize("label", ["", "Net exports"])
def test_a_data_row_in_the_header_position_is_checked(rotation, label):
    """Adversarial pass F1: the model put the first data row in the markdown header position, with a
    lost or wrong label. A header row with decimal values is judged; never BOUND."""
    first = "| " + " | ".join([label, *GROUP[0][1]]) + " |"
    body = "\n".join("| " + " | ".join([lab, *vals]) + " |" for lab, vals in GROUP[1:])
    markdown = first + "\n|---|---|---|---|\n" + body + "\n"
    results, _ = check_page_row_binding(build_page(GROUP, rotation), markdown)
    assert results[0].outcome is not TableOutcome.BOUND
    faithful = "| " + " | ".join([GROUP[0][0], *GROUP[0][1]]) + " |"
    results, _ = check_page_row_binding(
        build_page(GROUP, rotation), faithful + "\n|---|---|---|---|\n" + body + "\n"
    )
    assert results[0].outcome is TableOutcome.BOUND


def test_non_latin_labels_take_part_in_label_identity():
    """Adversarial pass F2: an ASCII-only filter left Greek labels with no tokens, and a label with no
    tokens bound anything. Words given directly (no PDF font needed), upright frame."""
    from socr.tables.row_binding_check import check_row_binding

    labels = ["Εξαγωγές", "Εισαγωγές", "Ισοζύγιο"]

    def words(rows):
        out = []
        for i, (lab, vals) in enumerate(rows):
            y = V0 + i * LINE_PITCH
            out.append((LABEL_U, y, LABEL_U + 60, y + FONT, lab))
            for k, v in enumerate(vals):
                x = VALUE_U0 + k * VALUE_STEP
                out.append((x, y, x + 18, y + FONT, v))
        return out

    native = [(lab, _v(i + 1)) for i, lab in enumerate(labels)]
    assert check_row_binding(words(native), None, md(native)).outcome is TableOutcome.BOUND
    shifted = [(labels[1], _v(1)), (labels[2], _v(2)), ("", _v(3))]
    assert check_row_binding(words(native), None, md(shifted)).outcome is not TableOutcome.BOUND


SPLIT_NATIVE = [
    ("Gross saving less gross investment", _v(1)),
    ("Net lending", _v(2)),
    ("Other", _v(3)),
]


@pytest.mark.parametrize("rotation", ROTATIONS)
@pytest.mark.parametrize(
    "rows",
    [
        [("Gross saving less gross", []), ("investment", _v(1))],
        [("Gross saving less gross", _v(1)), ("investment", [])],
        [("Gross", []), ("saving less gross investment", _v(1))],
        [("Gross saving less gross investment", []), ("Gross saving less gross investment", _v(1))],
    ],
    ids=["values_on_second", "values_on_first", "first_word_row", "label_repeated"],
)
def test_a_label_split_over_two_rows_is_not_convicted(rotation, rows):
    """Adversarial pass F3: one printed row shipped as two markdown rows, values under the right
    label. Not CONTRADICTED (UNVERIFIABLE is fine)."""
    res = run(SPLIT_NATIVE, [*rows, *SPLIT_NATIVE[1:]], rotation)
    assert res.outcome is not TableOutcome.CONTRADICTED


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_a_parent_emptied_under_a_shorter_child_label_is_still_convicted(rotation):
    """The frame-1 shape the split rule must not excuse: `Net exports` ships empty and the next row,
    `Exports` (a shorter piece of the parent's label), carries the parent's values."""
    native = [("Exports", _v(1)), ("Imports", _v(2)), ("Net exports", _v(3)), ("Other", _v(4))]
    rows = [("Exports", _v(1)), ("Imports", _v(2)), ("Net exports", []), ("Exports", _v(3))]
    res = run(native, rows, rotation)
    assert res.outcome is TableOutcome.CONTRADICTED
    assert res.rows[2].verdict is RowVerdict.EMPTIED
