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


# Round 5 (R2): every markdown row with content is judged, the header row included, and these synthetic
# pages print no column-heading line. The default header is therefore an empty heading row (nothing to
# judge); tests about header rows pass their own header.
EMPTY_HEADER = "|  |  |  |  |"


def md(rows: list[tuple[str, list[str]]], header: str = EMPTY_HEADER) -> str:
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
def test_wrapped_two_line_label_is_never_convicted(rotation):
    """Round 4: a label wrapped onto a band above its values is not printed before the values on their
    line, so it is UNVERIFIABLE (marked), never BOUND, and never convicted."""
    native = [
        ("Cash operating balance,", []),
        ("end of period", ["13", "24", "30"]),
        ("Receipts", ["1775", "1825", "1853"]),
    ]
    rows = [
        ("Cash operating balance, end of period", ["13", "24", "30"]),
        ("Receipts", ["1775", "1825", "1853"]),
    ]
    res = run(native, rows, rotation)
    assert res.outcome is TableOutcome.UNVERIFIABLE
    assert res.rows[0].verdict is RowVerdict.UNVERIFIABLE


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_split_label_band_previous_is_never_convicted(rotation):
    """'Previous' printed on its own baseline above its values (Greenbook p34 shape). Round 4: band
    evidence never grants BOUND; the row is UNVERIFIABLE and the table is not convicted."""
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
    assert res.outcome is TableOutcome.UNVERIFIABLE, [
        (r.shipped_label, r.verdict, r.reason) for r in res.rows
    ]
    assert res.rows[1].verdict is RowVerdict.UNVERIFIABLE


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
    good = "|  |  |  |  |\n|---|---|---|---|\n| Treasury | 1.1 | 1.2 | 1.3 |\n| Agency | 5.1 | 5.2 | 5.3 |\n"
    swapped = "|  |  |  |  |\n|---|---|---|---|\n| Agency | 1.1 | 1.2 | 1.3 |\n| Treasury | 5.1 | 5.2 | 5.3 |\n"
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
    # Round 5 (Fable P2): a band label on a line whose values are right may be a wrapped label's first
    # line, so the band rule marks instead of convicting (no false withhold). Never BOUND.
    assert res.outcome is TableOutcome.UNVERIFIABLE
    assert res.rows[0].verdict is RowVerdict.UNVERIFIABLE


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
    The year rows and the one-digit row come out UNVERIFIABLE (a whole-line and a values-only panel both
    match: two readings of the label), `2018: Q2` is printed exactly before its values: BOUND."""
    native = [
        ("2005", ["1.0", "2.0", "3.0"]),
        ("2006", ["4.0", "5.0", "6.0"]),
        ("2018: Q2", ["7", "8", "9"]),
        ("3", ["1.5", "2.5", "3.5"]),
    ]
    res = run(native, native, rotation)
    assert res.outcome is not TableOutcome.CONTRADICTED
    assert not CONVICT & set(verdicts(res))
    assert res.rows[2].verdict is RowVerdict.BOUND


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
def test_wrapped_label_split_over_two_markdown_rows_is_never_convicted(rotation):
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
    assert not CONVICT & set(verdicts(res))
    assert res.rows[1].verdict is RowVerdict.UNVERIFIABLE  # round 4: never BOUND by a wrap


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
    markdown = "|  |  |  |\n|---|---|---|\n| Alpha | 1 | 2 |\n| Alpha | 1 | 2 |\n"
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

    res = check_row_binding([], None, "|  |  |\n|---|---|\n")
    assert res.rows == () and res.outcome is TableOutcome.BOUND
    # Round 5 (R2): a header row with content is a row to judge; one the layer does not print is not BOUND
    headed = check_row_binding([], None, "| a | b |\n|---|---|\n")
    assert headed.outcome is TableOutcome.UNVERIFIABLE


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
    # round 6: any footnote difference is UNVERIFIABLE (marked), never BOUND, never convicted
    assert run(native, footnoted, rotation).outcome is TableOutcome.UNVERIFIABLE
    latex = [
        ("**Consumer prices**$^{2}$", ["2.1", "2.2", "2.3"]),
        ("Real GDP", ["1.5", "2.5", "3.5"]),
    ]
    assert run(native, latex, rotation).outcome is TableOutcome.UNVERIFIABLE
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


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_footnote_digits_on_both_sides_must_agree(rotation):
    """Merge gate round 3 (Fable P2): `Previous¹` / `Previous²` differ only by the footnote digit.
    Stripping the marks made a swap between them bind; the digits are part of the label when both
    sides carry one."""
    native = [("Previous¹", ["2.1", "2.2", "2.3"]), ("Previous²", ["1.5", "2.5", "3.5"])]
    swapped = [("Previous²", ["2.1", "2.2", "2.3"]), ("Previous¹", ["1.5", "2.5", "3.5"])]
    assert run(native, native, rotation).outcome is TableOutcome.BOUND
    assert run(native, swapped, rotation).outcome is TableOutcome.CONTRADICTED


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_a_parenthesised_footnote_on_the_text_layer_is_not_a_value(rotation):
    """Merge gate round 3 (Fable P2): the layer prints `Model (1)` as two words and `(1)` reads as -1,
    so a faithful table was CONTRADICTED. The mark joins the label word on the native side too."""
    native = [("Model (1)", ["2.1", "2.2", "2.3"]), ("Model (2)", ["1.5", "2.5", "3.5"])]
    swapped = [("Model (2)", ["2.1", "2.2", "2.3"]), ("Model (1)", ["1.5", "2.5", "3.5"])]
    assert run(native, native, rotation).outcome is TableOutcome.BOUND
    assert run(native, swapped, rotation).outcome is TableOutcome.CONTRADICTED
    uk = [("United Kingdom (4)", ["2.1", "2.2", "2.3"]), ("Japan", ["1.5", "2.5", "3.5"])]
    assert run(uk, uk, rotation).outcome is TableOutcome.BOUND


def _words(rows):
    """fitz-style word tuples in the upright frame for (label, values) lines; no PDF font needed."""
    out = []
    for i, (lab, vals) in enumerate(rows):
        y = V0 + i * LINE_PITCH
        if lab:
            out.append((LABEL_U, y, LABEL_U + 60, y + FONT, lab))
        for k, v in enumerate(vals):
            x = VALUE_U0 + k * VALUE_STEP
            out.append((x, y, x + 18, y + FONT, v))
    return out


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_superscript_footnotes_never_merge_numbered_labels(rotation):
    """Merge gate round 3 (Astra P0): native `Q1` / `Q2`, shipped `Q²` / `Q¹` swapped."""
    native = [("Q1", ["1.1", "2.2", "3.3"]), ("Q2", ["4.4", "5.5", "6.6"])]
    swapped = [("Q²", ["1.1", "2.2", "3.3"]), ("Q¹", ["4.4", "5.5", "6.6"])]
    assert run(native, swapped, rotation).outcome is not TableOutcome.BOUND


def test_combining_marks_are_part_of_the_label():
    """Merge gate round 3 (Astra P0): `कुल` and `काल` differ only in a vowel sign."""
    from socr.tables.row_binding_check import check_row_binding

    native = [("कुल", ["1.1", "1.2", "1.3"]), ("काल", ["2.1", "2.2", "2.3"])]
    swapped = [("काल", ["1.1", "1.2", "1.3"]), ("कुल", ["2.1", "2.2", "2.3"])]
    assert check_row_binding(_words(native), None, md(native)).outcome is TableOutcome.BOUND
    assert check_row_binding(_words(native), None, md(swapped)).outcome is not TableOutcome.BOUND


def _with_header(first, body):
    def cells(lab, vals):
        return "| " + " | ".join([lab, *vals]) + " |"

    return cells(*first) + "\n|---|---|---|---|\n" + "\n".join(cells(*r) for r in body) + "\n"


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_a_whole_number_data_row_in_the_header_is_judged(rotation):
    """Merge gate round 3 (Astra P0): `Imports | 11 | 12 | 13` in the header carries the Exports
    values, over the correct Imports row. Judged like a body row: CONTRADICTED."""
    native = [("Exports", ["11", "12", "13"]), ("Imports", ["21", "22", "23"])]
    page = build_page(native, rotation)
    bad, _ = check_page_row_binding(page, _with_header(("Imports", native[0][1]), [native[1]]))
    assert bad[0].outcome is TableOutcome.CONTRADICTED
    good, _ = check_page_row_binding(page, _with_header(native[0], [native[1]]))
    assert good[0].outcome is TableOutcome.BOUND
    # a year heading is not a data row: it is not judged
    years = [("Item", ["2008", "2009", "2010"]), *native]
    heading, _ = check_page_row_binding(
        build_page(years, rotation), _with_header(years[0], years[1:])
    )
    assert heading[0].outcome is TableOutcome.BOUND


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_a_displaced_row_in_the_header_is_convicted(rotation):
    """Merge gate round 3 (Astra P1): the same mislabelled row is CONTRADICTED in the body and in
    the header position."""
    native = [("Exports", ["1.1", "1.2", "1.3"]), ("Imports", ["2.1", "2.2", "2.3"])]
    page = build_page(native, rotation)
    body, _ = check_page_row_binding(page, md([("Imports", native[0][1]), native[1]]))
    head, _ = check_page_row_binding(page, _with_header(("Imports", native[0][1]), [native[1]]))
    assert body[0].outcome is TableOutcome.CONTRADICTED
    assert head[0].outcome is TableOutcome.CONTRADICTED


ADVERSARIAL_LABEL_PAIRS = [
    ("Q1", "Q2"),
    ("Q¹", "Q²"),
    ("Q1", "Q²"),
    ("CO²", "CO³"),
    ("कुल", "काल"),
    ("US government", "UK government"),
    ("Line 1", "Line 2"),
    ("Model (1)", "Model (2)"),
    ("Previous¹", "Previous²"),
    ("Tier 1 capital", "Tier 2 capital"),
    ("Café", "Cafe"),
    ("Εξαγωγές", "Εισαγωγές"),
    ("Total", "Subtotal"),
    ("Net exports", "Exports"),
    ("Inflation", "Infation"),
    ("2019:Q2", "2019:Q3"),
    ("Exports less imports", "Imports less exports"),
    ("Projected", "Projected Projected"),
]


@pytest.mark.parametrize("a, b", ADVERSARIAL_LABEL_PAIRS)
def test_label_identity_never_merges_labels_that_differ_in_a_letter_digit_or_mark(a, b):
    """Merge gate round 3 (Astra): the CLASS rule. Labels whose letters, digits or combining marks
    differ (after case folding and NFKC) are never the same label, in either direction."""
    import unicodedata

    from socr.tables.row_binding_check import _label_tokens, _same_label

    def key(s):
        s = unicodedata.normalize("NFKC", s).lower()
        return "".join(c for c in s if unicodedata.category(c)[0] in "LNM")

    assert key(a) != key(b)
    ta, tb = _label_tokens(a), _label_tokens(b)
    assert ta != tb
    assert not _same_label(ta, tb)


@pytest.mark.parametrize(
    "shipped, native",
    [
        ("Receipts", "Receipts2"),
        ("United Kingdom", "United Kingdom (4)"),
        ("Real GDP¹", "Real GDP"),
    ],
)
def test_a_footnote_on_one_side_only_still_matches(shipped, native):
    from socr.tables.row_binding_check import _label_tokens, _same_label

    assert _same_label(_label_tokens(shipped), _label_tokens(native))


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_word_order_is_part_of_the_label(rotation):
    """Corpus label property check (round 3): set equality merged `Exports less imports` with
    `Imports less exports`. Swapped, they must not bind."""
    native = [
        ("Exports less imports", ["1.1", "1.2", "1.3"]),
        ("Imports less exports", ["2.1", "2.2", "2.3"]),
    ]
    swapped = [
        ("Imports less exports", ["1.1", "1.2", "1.3"]),
        ("Exports less imports", ["2.1", "2.2", "2.3"]),
    ]
    assert run(native, native, rotation).outcome is TableOutcome.BOUND
    assert run(native, swapped, rotation).outcome is not TableOutcome.BOUND


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_a_heading_with_numbers_inside_words_is_not_a_data_row(rotation):
    """Frame-1 live boxes: a heading `Institution Name | (March 24 - | April 24)` was judged as a data
    row and convicted. A value cell with words in it is a heading cell."""
    native = [
        ("Institution Name", ["(March", "24", "-", "April", "24)"]),
        ("Exports", ["1.1", "1.2", "1.3"]),
        ("Imports", ["2.1", "2.2", "2.3"]),
    ]
    markdown = _with_header(("Institution Name", ["(March 24 -", "April 24)", ""]), native[1:])
    res, _ = check_page_row_binding(build_page(native, rotation), markdown)
    # never convicted; under R2 (round 5) it is judged and binds only exactly, so it may be marked
    assert res[0].outcome is not TableOutcome.CONTRADICTED
    assert res[0].rows[0].verdict not in CONVICT


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_an_empty_row_is_not_emptied_by_a_line_differing_in_a_short_token(rotation):
    """Short tokens in the EMPTIED search too: an empty `UK government` row is not the `US government`
    line's row, so the table is not convicted."""
    native = [("US government", ["1.1", "1.2", "1.3"]), ("Other", ["2.1", "2.2", "2.3"])]
    rows = [("UK government", []), ("Other", ["2.1", "2.2", "2.3"])]
    res = run(native, rows, rotation)
    assert res.rows[0].verdict is not RowVerdict.EMPTIED
    assert res.outcome is not TableOutcome.CONTRADICTED


# ---------------------------------------------------------------------------
# Round 4: BOUND needs exact evidence. Every reviewer pair from rounds 1-4.
# ---------------------------------------------------------------------------
REVIEWER_PAIRS = [
    # round 1
    ("Government", "Federal government"),
    ("", "US"),
    # round 2
    ("US government", "UK government"),
    ("", "Business"),
    ("Total", "Subtotal"),
    # round 3
    ("Previous¹", "Previous²"),
    ("Q¹", "Q²"),
    ("Q1", "Q²"),
    ("CO²", "CO³"),
    ("कुल", "काल"),
    ("Model (1)", "Model (2)"),
    ("Line 1", "Line 2"),
    ("Exports less imports", "Imports less exports"),
    ("Projected", "Projected Projected"),
    # round 4
    ("Rate (%)", "Rate ($)"),
    ("2003", "2003:1"),
    ("1", "10"),
    ("2.00", "2000"),
    ("Q1", "Q10"),
    ("A1", "A10"),
    ("M", "M1"),
    ("CO", "CO₂"),
    ("Phase", "Phase2"),
    ("FY2019", "FY201"),
    ("Total¹", "Total1"),
    ("Exports less", "Exports less imports"),
    # round 5
    ("A**", "A*"),
    ("A`", "A"),
    ("Total*", "Total**"),
    ("", "(1)"),
    ("", "¹"),
    ("", "*"),
    ("Series_A", "Series A"),
    ("A*B", "A B"),
]


@pytest.mark.parametrize("a, b", REVIEWER_PAIRS)
def test_reviewer_pairs_are_never_the_same_label(a, b):
    from socr.tables.row_binding_check import _labels_exact

    assert not _labels_exact(a, b) and not _labels_exact(b, a)


@pytest.mark.parametrize(
    "shipped, printed",
    [
        ("**Real GDP**", "Real GDP"),
        ("Receipts:", "Receipts"),
        ("Receipts¹", "Receipts"),
        ("United Kingdom (4)", "United Kingdom"),
        ("Consumer prices$^{2}$", "Consumer prices"),
        ("Equipment &amp; software", "Equipment & software"),
        ("real  GDP", "Real GDP"),
    ],
)
def test_minimal_normalisation_keeps_the_same_label(shipped, printed):
    from socr.tables.row_binding_check import _labels_exact

    assert _labels_exact(shipped, printed)
    # round 6: the BOUND path compares strictly; only the footnote cases differ there
    footnote = shipped in ("Receipts¹", "United Kingdom (4)", "Consumer prices$^{2}$")
    assert _labels_exact(shipped, printed, strict=True) is (not footnote)


SWAP_PAIRS = [
    ("Rate (%)", "Rate ($)"),
    ("Q1", "Q10"),
    ("A1", "A10"),
    ("M", "M1"),
    ("Phase", "Phase2"),
    ("FY2019", "FY201"),
    ("Previous¹", "Previous²"),
]


@pytest.mark.parametrize("rotation", ROTATIONS)
@pytest.mark.parametrize("a, b", SWAP_PAIRS)
def test_swapped_reviewer_labels_never_bind(rotation, a, b):
    native = [(a, ["1.1", "1.2", "1.3"]), (b, ["2.1", "2.2", "2.3"])]
    swapped = [(b, ["1.1", "1.2", "1.3"]), (a, ["2.1", "2.2", "2.3"])]
    assert run(native, swapped, rotation).outcome is not TableOutcome.BOUND


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_year_rows_swapped_never_bind(rotation):
    native = [("2003", ["1.1", "1.2", "1.3"]), ("2004", ["2.1", "2.2", "2.3"])]
    swapped = [("2004", ["1.1", "1.2", "1.3"]), ("2003", ["2.1", "2.2", "2.3"])]
    assert run(native, swapped, rotation).outcome is not TableOutcome.BOUND


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_a_label_wrapped_onto_a_band_never_binds(rotation):
    """Round 4 (Astra P0): `Exports less` / `imports` printed over two lines, `Imports less exports`
    on the next; both shipped rows labelled `Imports less exports`. Pools may not grant BOUND."""
    native = [
        ("Exports less", []),
        ("imports", ["1.1", "1.2", "1.3"]),
        ("Imports less exports", ["2.1", "2.2", "2.3"]),
    ]
    rows = [
        ("Imports less exports", ["1.1", "1.2", "1.3"]),
        ("Imports less exports", ["2.1", "2.2", "2.3"]),
    ]
    assert run(native, rows, rotation).outcome is not TableOutcome.BOUND


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_a_four_digit_data_row_in_the_header_never_binds(rotation):
    """Round 4 (Astra P0): header `Imports | 1011 | 1012 | 1013` carries the Exports values."""
    native = [("Exports", ["1011", "1012", "1013"]), ("Imports", ["2021", "2022", "2023"])]
    res, _ = check_page_row_binding(
        build_page(native, rotation), _with_header(("Imports", native[0][1]), [native[1]])
    )
    assert res[0].outcome is not TableOutcome.BOUND


def test_exact_label_rule_over_all_pairs_of_reviewer_labels():
    """Round 4 property, ALL pairs (no bucketing) of every label the four review rounds used plus
    their normalisation variants: two labels are the same only when they differ by case, whitespace,
    markup, a footnote mark on one side, or one trailing period/colon (independent residue test)."""
    import html as _html
    import itertools
    import re as _re
    import unicodedata as _ud

    from socr.tables.row_binding_check import _labels_exact

    labels = sorted(
        {x for pair in [*REVIEWER_PAIRS, *ADVERSARIAL_LABEL_PAIRS] for x in pair if x}
        | {"**Real GDP**", "Real GDP", "Receipts:", "Receipts", "Receipts¹", "real  gdp"}
        | {"United Kingdom (4)", "United Kingdom", "Consumer prices$^{2}$", "Consumer prices"}
    )
    foot = _re.compile(r"(\(\d\)|\$\^\{?\d\}?\$|[¹²³⁰⁴-⁹]|\*+)$")

    def base(s, shipped):
        # round 5: only PAIRED markup, only on the shipped side; a lone * or ` is a character
        if shipped:
            s = _re.sub(r"<br\s*/?>", " ", s)
            s = _re.sub(r"\*\*(\S.*?\S|\S)\*\*", r"\1", s)
            s = _re.sub(r"(?<![*\w])\*([^\s*].*?[^\s*]|[^\s*])\*(?![*\w])", r"\1", s)
            s = _html.unescape(_re.sub(r"`([^`]+)`", r"\1", s))
        return _re.sub(r"\s+", "", _ud.normalize("NFC", s).casefold())

    def allowed(a, b):
        x, y = base(a, True), base(b, False)
        for xs in {x, x[:-1] if x[-1:] in ".:" else x}:
            for ys in {y, y[:-1] if y[-1:] in ".:" else y}:
                fx, fy = foot.search(xs), foot.search(ys)
                if xs == ys:
                    return True
                if (
                    fx
                    and not fy
                    and xs[: fx.start()]
                    and xs[: fx.start()].rstrip(".:") == ys.rstrip(".:")
                ):
                    return True
                if (
                    fy
                    and not fx
                    and ys[: fy.start()]
                    and ys[: fy.start()].rstrip(".:") == xs.rstrip(".:")
                ):
                    return True
        return False

    bad = [
        (a, b)
        for a, b in itertools.permutations(labels, 2)
        if _labels_exact(a, b) and not allowed(a, b)
    ]
    assert bad == []


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_an_empty_group_row_binds_only_on_an_exact_valueless_line(rotation):
    """Round 4: a valueless group row is BOUND only when a valueless native line prints exactly its
    label; `Goods` over a `Goods and services` line is UNVERIFIABLE."""
    native = [("Goods and services", []), ("Durables", ["1.1", "1.2", "1.3"])]
    exact = run(native, native, rotation)
    assert exact.rows[0].verdict is RowVerdict.BOUND
    partial = run(native, [("Goods", []), native[1]], rotation)
    assert partial.rows[0].verdict is RowVerdict.UNVERIFIABLE
    assert partial.outcome is TableOutcome.UNVERIFIABLE


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_a_header_row_binds_only_on_its_exact_printed_label(rotation):
    """Round 4: a header-position data row whose label comes from a band above its line (pool
    evidence) is UNVERIFIABLE, not BOUND."""
    native = [
        ("Real GDP", []),
        ("growth", ["1.1", "1.2", "1.3"]),
        ("Net exports", ["2.1", "2.2", "2.3"]),
    ]
    res, _ = check_page_row_binding(
        build_page(native, rotation), _with_header(("Real GDP growth", native[1][1]), [native[2]])
    )
    assert res[0].outcome is TableOutcome.UNVERIFIABLE
    assert res[0].rows[0].verdict is RowVerdict.UNVERIFIABLE


# Round 5 (Astra / Fable), each pinned at the check level.
@pytest.mark.parametrize("rotation", ROTATIONS)
@pytest.mark.parametrize("a, b", [("A**", "A*"), ("Total*", "Total**"), ("Total", "Total*")])
def test_literal_asterisks_never_let_a_swap_bind(rotation, a, b):
    native = [
        (a, ["1.1", "1.2", "1.3"]),
        (b, ["2.1", "2.2", "2.3"]),
        ("Other", ["3.1", "3.2", "3.3"]),
    ]
    swapped = [(b, native[0][1]), (a, native[1][1]), native[2]]
    assert run(native, swapped, rotation).outcome is not TableOutcome.BOUND


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_a_mixed_header_cell_is_judged(rotation):
    """Round 5 (Astra P0): header `Imports | 1.1 million | 1.2 | 1.3` (Exports' values) over the
    faithful Imports row; a mixed text/number header was skipped and the table bound."""
    native = [("Exports", ["1.1", "1.2", "1.3"]), ("Imports", ["2.1", "2.2", "2.3"])]
    md_text = (
        "| Imports | 1.1 million | 1.2 | 1.3 |\n|---|---|---|---|\n| Imports | 2.1 | 2.2 | 2.3 |\n"
    )
    res, _ = check_page_row_binding(build_page(native, rotation), md_text)
    assert res[0].outcome is not TableOutcome.BOUND


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_a_categorical_header_row_is_judged(rotation):
    """Round 5 (Fable P2): `| UK | Small | Moderate | Large |` (US's cells) in the header position."""
    native = [
        ("US", ["Small", "Moderate", "Large"]),
        ("UK", ["Large", "Small", "Moderate"]),
        ("EA", ["Moderate", "Large", "Small"]),
    ]
    res, _ = check_page_row_binding(
        build_page(native, rotation), _with_header(("UK", native[0][1]), native[1:])
    )
    assert res[0].outcome is not TableOutcome.BOUND


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_a_label_that_is_only_a_footnote_mark_is_never_erased(rotation):
    """Round 5 (Astra P0): native row `(1)` shipped with an empty label is not BOUND."""
    native = [("(1)", ["1.1", "1.2", "1.3"]), ("(2)", ["2.1", "2.2", "2.3"])]
    res = run(native, [("", native[0][1]), ("", native[1][1])], rotation)
    assert res.outcome is not TableOutcome.BOUND


@pytest.mark.parametrize("rotation", ROTATIONS)
@pytest.mark.parametrize("a, b", [("Series_A", "Series A"), ("A*B", "A B")])
def test_punctuation_inside_a_label_never_convicts_a_faithful_table(rotation, a, b):
    """Round 5 (Astra P2): shipped and native labels are tokenised the same way."""
    native = [(a, ["1.1", "1.2", "1.3"]), (b, ["2.1", "2.2", "2.3"])]
    assert run(native, native, rotation).outcome is not TableOutcome.CONTRADICTED


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_an_emptied_row_is_not_bound_by_a_valueless_twin_line(rotation):
    """Round 5 (Fable P2): a valueless `Total` line and a `Total` line with values both in range;
    shipping the second Total EMPTY must not bind."""
    native = [
        ("Receipts", []),
        ("Individual", ["1.1", "1.2", "1.3"]),
        ("Total", []),
        ("Total", ["2.1", "2.2", "2.3"]),
        ("Outlays", []),
        ("Defense", ["3.1", "3.2", "3.3"]),
    ]
    rows = [native[0], native[1], native[2], ("Total", []), native[4], native[5]]
    assert run(native, rows, rotation).outcome is not TableOutcome.BOUND


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_a_wrapped_label_first_line_on_correct_values_is_not_withheld(rotation):
    """Round 5 (Fable P2): `Exports` printed above `of goods 1.1 2.2 3.3`; shipping `Exports` with the
    right values is a mark, not a withhold."""
    native = [
        ("Exports", []),
        ("of goods", ["1.1", "2.2", "3.3"]),
        ("Imports", ["4.1", "4.2", "4.3"]),
        ("Other", ["5.1", "5.2", "5.3"]),
    ]
    rows = [("Exports", native[1][1]), native[2], native[3]]
    assert run(native, rows, rotation).outcome is TableOutcome.UNVERIFIABLE


def test_a_numeric_leading_footnote_pair_never_binds_swapped():
    """Round 6 (Astra P0): `2020 Total` / `2020 Total*` swapped was BOUND (footnote home blind to a
    numeric-leading label). Label and values set tightly, as one panel. The BOUND path now has no
    footnote tolerance at all."""
    from socr.tables.row_binding_check import check_row_binding

    def line(y, words):
        out, x = [], 40.0
        for w in words:
            out.append((x, y, x + 6 * len(w), y + 9, w))
            x += 6 * len(w) + 4
        return out

    native = [("2020 Total", ["1.1", "1.2", "1.3"]), ("2020 Total*", ["2.1", "2.2", "2.3"])]
    words = [
        w for i, (lab, vals) in enumerate(native) for w in line(100 + 14 * i, [*lab.split(), *vals])
    ]
    swapped = [("2020 Total*", native[0][1]), ("2020 Total", native[1][1])]
    assert check_row_binding(words, None, md(native)).outcome is TableOutcome.BOUND
    assert check_row_binding(words, None, md(swapped)).outcome is not TableOutcome.BOUND


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_an_emptied_header_row_is_judged(rotation):
    """Round 6 (Astra P0): `| Exports | | | |` in the header position while the page prints Exports'
    values: a labelled header row without values is judged; never BOUND."""
    native = [("Exports", ["1.1", "1.2", "1.3"]), ("Imports", ["2.1", "2.2", "2.3"])]
    res, _ = check_page_row_binding(
        build_page(native, rotation), _with_header(("Exports", ["", "", ""]), [native[1]])
    )
    assert res[0].outcome is not TableOutcome.BOUND


def _set_line(y, words):
    out, x = [], 40.0
    for w in words:
        out.append((x, y, x + 6 * len(w), y + 9, w))
        x += 6 * len(w) + 4
    return out


def test_a_baseline_with_two_stubs_flattened_into_one_row_never_binds():
    """Round 7 (Fable P1): `Exports 1.1 2.2 3.3 Imports 4.4 5.5 6.6` on one baseline, shipped as ONE
    row under `Exports` with all six values."""
    from socr.tables.row_binding_check import check_row_binding

    words = _set_line(100, ["Exports", "1.1", "2.2", "3.3", "Imports", "4.4", "5.5", "6.6"])
    flat = "|  |  |  |  |  |  |  |\n|---|---|---|---|---|---|---|\n"
    flat += "| Exports | 1.1 | 2.2 | 3.3 | 4.4 | 5.5 | 6.6 |\n"
    assert check_row_binding(words, None, flat).outcome is not TableOutcome.BOUND
    # a word the shipped row carries itself (`n.a.`) is not another stub
    words2 = _set_line(100, ["Exports", "1.1", "n.a.", "3.3"])
    ok = "|  |  |  |  |\n|---|---|---|---|\n| Exports | 1.1 | n.a. | 3.3 |\n"
    assert check_row_binding(words2, None, ok).outcome is TableOutcome.BOUND


def test_symbol_only_labels_are_judged_not_blank():
    """Round 7 (Astra P1): `%` and `$` labels had no tokens and were skipped as blank rows."""
    from socr.tables.row_binding_check import check_row_binding

    words = _set_line(100, ["%", "1.1", "1.2", "1.3"]) + _set_line(114, ["$", "2.1", "2.2", "2.3"])
    faithful = md([("%", ["1.1", "1.2", "1.3"]), ("$", ["2.1", "2.2", "2.3"])])
    swapped = md([("$", ["1.1", "1.2", "1.3"]), ("%", ["2.1", "2.2", "2.3"])])
    assert check_row_binding(words, None, faithful).outcome is TableOutcome.BOUND
    res = check_row_binding(words, None, swapped)
    assert res.outcome is not TableOutcome.BOUND
    assert len(res.rows) == 2  # neither row skipped as blank
    # a valueless symbol row the page never prints was skipped as a blank row (table BOUND)
    words3 = _set_line(100, ["Exports", "1.1", "1.2", "1.3"]) + _set_line(
        114, ["Imports", "2.1", "2.2", "2.3"]
    )
    spurious = md(
        [("Exports", ["1.1", "1.2", "1.3"]), ("%", []), ("Imports", ["2.1", "2.2", "2.3"])]
    )
    assert check_row_binding(words3, None, spurious).outcome is not TableOutcome.BOUND


@pytest.mark.parametrize(
    "header, extra",
    [
        ("|  | $ | % |  |", []),
        ("| Item | $ | % |  |", []),
        (EMPTY_HEADER, [("", ["$", "%"])]),
        (EMPTY_HEADER, [("Units", ["$", "%"])]),
    ],
    ids=["heading-no-stub", "heading-with-stub", "body-no-stub", "body-with-stub"],
)
def test_symbol_only_cells_are_judged_not_skipped(header, extra):
    """Round 10 (Astra P1): `$` and `%` cells counted as neither numbers nor words, so these rows
    were skipped and the table came out BOUND with units the layer does not print."""
    from socr.tables.row_binding_check import check_row_binding

    # The layer prints the stubs alone, so without this rule a labelled row binds as valueless.
    words = (
        _set_line(86, ["Item"])
        + _set_line(100, ["Exports", "1.1", "1.2", "1.3"])
        + _set_line(114, ["Imports", "2.1", "2.2", "2.3"])
        + _set_line(128, ["Units"])
    )
    rows = [("Exports", ["1.1", "1.2", "1.3"]), ("Imports", ["2.1", "2.2", "2.3"])]
    assert check_row_binding(words, None, md(rows)).outcome is TableOutcome.BOUND
    res = check_row_binding(words, None, md([*rows, *extra], header=header))
    assert res.outcome is TableOutcome.UNVERIFIABLE
    assert [e for e in res.rows if e.verdict is RowVerdict.UNVERIFIABLE]


@pytest.mark.parametrize("dash", ["—", "-", "..."])
def test_symbols_where_the_page_prints_numbers_stay_a_conviction(dash):
    """Round 12 (Fable P2): the symbol rule must only stop a pass, never soften EMPTIED."""
    from socr.tables.row_binding_check import check_row_binding

    words = _set_line(100, ["Final", "sales", "4.4", "5.5", "6.6"]) + _set_line(
        114, ["Exports", "1.1", "1.2", "1.3"]
    )
    res = check_row_binding(
        words, None, md([("Final sales", [dash, dash, dash]), ("Exports", ["1.1", "1.2", "1.3"])])
    )
    assert res.outcome is TableOutcome.CONTRADICTED
    assert RowVerdict.EMPTIED in verdicts(res)


@pytest.mark.parametrize("heading", ["|  | - | - |  |", "| - | - | - | - |"])
def test_a_dash_only_heading_above_the_separator_is_judged(heading):
    """Round 12 (Astra P1): the parser reads this line as a separator, so it was never judged."""
    from socr.tables.row_binding_check import check_row_binding

    words = _set_line(100, ["Exports", "1.1", "1.2", "1.3"]) + _set_line(
        114, ["Imports", "2.1", "2.2", "2.3"]
    )
    rows = [("Exports", ["1.1", "1.2", "1.3"]), ("Imports", ["2.1", "2.2", "2.3"])]
    assert check_row_binding(words, None, md(rows)).outcome is TableOutcome.BOUND
    assert check_row_binding(words, None, md(rows, header=heading)).outcome is (
        TableOutcome.UNVERIFIABLE
    )


def _hdr(n: int) -> str:
    return "|" + "  |" * n + "\n|" + "---|" * n + "\n"


@pytest.mark.parametrize(
    "line, row",
    [
        # round 8 (Astra P0): a symbol second stub
        (
            ["Exports", "1.1", "2.2", "3.3", "%", "4.4", "5.5", "6.6"],
            ["Exports", "1.1", "2.2", "3.3", "4.4", "5.5", "6.6"],
        ),
        (
            ["Exports", "1.1", "2.2", "3.3", "$", "4.4", "5.5", "6.6"],
            ["Exports", "1.1", "2.2", "3.3", "4.4", "5.5", "6.6"],
        ),
        # round 8 (Fable P1): letter-suffixed cells, n.a., categorical words in the right panel
        (
            ["Exports", "1.1", "2.2", "Imports", "3.3a", "4.4a"],
            ["Exports", "1.1", "2.2", "3.3a", "4.4a"],
        ),
        (
            ["Exports", "1.1", "2.2", "Imports", "n.a.", "n.a."],
            ["Exports", "1.1", "2.2", "n.a.", "n.a."],
        ),
        (
            ["US", "Small", "Moderate", "UK", "Large", "Weak"],
            ["US", "Small", "Moderate", "Large", "Weak"],
        ),
        # round 8 (Fable P2): the right stub carried as a TRAILING cell
        (
            ["Exports", "1.1", "2.2", "Imports", "3.3", "4.4"],
            ["Exports", "1.1", "2.2", "3.3", "4.4", "Imports"],
        ),
    ],
)
def test_round8_flattened_second_stub_never_binds(line, row):
    """Round 8 positional rule: between the first and last matched word of the line no native word may
    go unmatched by a shipped cell in that position."""
    from socr.tables.row_binding_check import check_row_binding

    markdown = _hdr(len(row)) + "| " + " | ".join(row) + " |\n"
    assert check_row_binding(_set_line(100, line), None, markdown).outcome is not TableOutcome.BOUND


@pytest.mark.parametrize(
    "line, row",
    [
        (
            ["Exports", "1.1", "to", "2.2", "3.3"],
            ["Exports", "1.1 to 2.2", "3.3"],
        ),  # full cell text
        (["Exports", "1.1", "n.a.", "3.3"], ["Exports", "1.1", "n.a.", "3.3"]),
        # trailing native words after the last matched value (a running head) are allowed
        (
            ["Exports", "1.1", "2.2", "3.3", "continued"],
            ["Exports", "1.1", "2.2", "3.3"],
        ),
    ],
)
def test_round8_positional_rule_keeps_faithful_rows_bound(line, row):
    from socr.tables.row_binding_check import check_row_binding

    markdown = _hdr(len(row)) + "| " + " | ".join(row) + " |\n"
    assert check_row_binding(_set_line(100, line), None, markdown).outcome is TableOutcome.BOUND


def test_round8_a_symbol_label_on_a_word_label_line_is_label_lost():
    """Round 8 (Fable P2): `•` in place of `Exports` is judged like an empty label."""
    from socr.tables.row_binding_check import check_row_binding

    words = _set_line(100, ["Exports", "1.1", "2.2"]) + _set_line(114, ["Imports", "3.3", "4.4"])
    bullet = _hdr(3) + "| • | 1.1 | 2.2 |\n| Imports | 3.3 | 4.4 |\n"
    res = check_row_binding(words, None, bullet)
    assert res.rows[0].verdict is RowVerdict.LABEL_LOST
    assert res.outcome is TableOutcome.CONTRADICTED
