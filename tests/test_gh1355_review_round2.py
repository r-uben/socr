"""#1355 review round 2: four fail-open inputs a reviewer measured. Each arm pins a difference."""

from __future__ import annotations

from collections import Counter

import pytest

fitz = pytest.importorskip("fitz")

from socr.core.result import PageOutput, PageStatus  # noqa: E402
from socr.pipeline.agentic import winner_accounts_for_marked_tables  # noqa: E402
from socr.tables.source_evidence import (  # noqa: E402
    SecondReading,
    _witness_may_stand_for,
    collect_table_tokens,
    verify_scanned_table,
)

# --- 1. header-label containment ---------------------------------------------------------------


def _out(text: str, **kw) -> PageOutput:
    return PageOutput(page_num=1, text=text, status=PageStatus.SUCCESS, engine="x", **kw)


def _marked(text: str) -> PageOutput:
    return _out(text, table_disputed="x", table_disputed_values=("-9.9",))


def _md(header: list[str], rows: list[list[str]], *, separator: bool = True) -> str:
    lines = ["| " + " | ".join(header) + " |"]
    if separator:
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def _html(rows: list[list[str]], *, th: bool) -> str:
    def cells(r, tag):
        return "".join(f"<{tag}>{c}</{tag}>" for c in r)

    body = f"<tr>{cells(rows[0], 'th' if th else 'td')}</tr>"
    body += "".join(f"<tr>{cells(r, 'td')}</tr>" for r in rows[1:])
    return f"<table>{body}</table>"


def test_p009_shape_still_lifts_and_a_different_header_number_does_not() -> None:
    rows = [["Corn", "5.5"], ["Soy", "6.5"]]
    marked = _marked(_md(["item", "2001"], rows))
    assert winner_accounts_for_marked_tables(
        _out(_md(["item", "2001 (Percent change)"], rows)), marked
    )
    # m10: a header carrying a DIFFERENT number covers nothing
    assert not winner_accounts_for_marked_tables(
        _out(_md(["item", "2002 (Percent change)"], rows)), marked
    )


def test_a_changed_data_value_is_not_hidden_by_header_numbers() -> None:
    marked = _marked(_md(["item", "Model 1", "Model 2"], [["Corn", "2", "5.5"]]))
    same = _out(_md(["item", "Model 1", "Model 2"], [["Corn", "2", "5.5"]]))
    changed = _out(_md(["item", "Model 1", "Model 2"], [["Corn", "3", "5.5"]]))
    assert winner_accounts_for_marked_tables(same, marked)
    assert not winner_accounts_for_marked_tables(changed, marked)


def test_a_row_label_number_is_not_covered_by_a_header_that_mentions_it() -> None:
    marked = _marked(_md(["item", "Level"], [["2001", "5.5"], ["Soy", "6.5"]]))
    winner = _out(_md(["item", "Level, 2001 dollars"], [["2010", "5.5"], ["Soy", "6.5"]]))
    assert not winner_accounts_for_marked_tables(winner, marked)


def test_header_numbers_of_another_table_do_not_cover_a_loss() -> None:
    t1_marked = _md(["a", "b"], [["x", "12"], ["y", "7.5"]])
    t2 = _md(["Top 12 firms", "c"], [["z", "3.5"]])
    t2_plain = _md(["Top firms", "c"], [["z", "3.5"]])
    marked = _marked(t1_marked + "\n" + t2_plain)
    lost = _out(_md(["a", "b"], [["x", "13"], ["y", "7.5"]]) + "\n" + t2)
    assert not winner_accounts_for_marked_tables(lost, marked)
    kept = _out(t1_marked + "\n" + t2)
    assert winner_accounts_for_marked_tables(kept, marked)


def test_no_th_in_html_means_row_zero_is_data() -> None:
    marked_rows = [["Banks", "2", "7"], ["Corn", "5.5", "6.5"]]
    win_rows = [["Banks, 2 of 7", "5.5", "6.5"], ["Corn", "5.5", "6.5"]]
    marked = _marked(_html(marked_rows, th=False))
    assert not winner_accounts_for_marked_tables(_out(_html(win_rows, th=False)), marked)
    # control: the same shapes with a real header (th on both) lift
    assert winner_accounts_for_marked_tables(
        _out(_html(win_rows, th=True)), _marked(_html(marked_rows, th=True))
    )


def test_markdown_without_a_separator_row_has_no_header() -> None:
    marked = _marked(_md(["Banks", "2", "7"], [["Corn", "5.5", "6.5"]], separator=False))
    win = _out(_md(["Banks, 2 of 7", "x", "y"], [["Corn", "5.5", "6.5"]], separator=False))
    assert not winner_accounts_for_marked_tables(win, marked)
    marked_h = _marked(_md(["Banks", "2", "7"], [["Corn", "5.5", "6.5"]]))
    win_h = _out(_md(["Banks, 2 of 7", "x", "y"], [["Corn", "5.5", "6.5"]]))
    assert winner_accounts_for_marked_tables(win_h, marked_h)


# --- 2. the stand-in is the emitted token with marks DELETED -----------------------------------


@pytest.mark.parametrize(
    ("witness", "emitted", "expected"),
    [
        ("41", "-4.1", True),
        ("11", "1.1", True),
        ("4.1", "-4.1", True),
        ("-41", "-4.1", True),
        ("07", "0.7", True),  # the zero survived, only the point was lost
        ("1.25", "12.5", False),  # a point in another place
        ("0.41", "4.1", False),
        ("7", "0.07", False),  # a leading zero is not a lost mark
        ("7", "0.7", False),  # decided conservatively: the zero is not a mark either
        ("-4.1", "4.1", False),  # the witness holds a sign the emitted token lacks
        ("4.1", "41", False),
    ],
)
def test_stand_in_is_the_emitted_token_with_marks_deleted(
    witness: str, emitted: str, expected: bool
) -> None:
    assert _witness_may_stand_for(witness, emitted) is expected


# --- 3 and 4: page-level ----------------------------------------------------------------------

TABLE = "| item | a | b |\n|---|---|---|\n| Corn | -4.1 | 2.0 |\n| Soy | 1.5 | 12 |\n"
TABLE_POS = "| item | a | b |\n|---|---|---|\n| Corn | 4.1 | 2.0 |\n| Soy | 1.5 | 12 |\n"


def _second(table: str = TABLE) -> SecondReading:
    t = collect_table_tokens(table)
    assert t is not None
    return SecondReading(engine="qwen", numeric=t.numeric)


def _verify(layer: str, ocr, table: str = TABLE):
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 100), layer, fontsize=11)
    try:
        return verify_scanned_table(
            page,
            table,
            ocr_image_fn=ocr,
            native_trusted=False,
            layer_is_evidence=True,
            second_reading=_second(table),
            engine="gemini",
        )
    finally:
        doc.close()


LOSSY = lambda pix: "Corn 41 2.0 Soy 1.5 12"  # noqa: E731


@pytest.mark.parametrize("layer_token", ["–41", "—4.1", "41bp", "x41", "­41", "4,1", "–4.1"])
def test_the_layer_veto_sees_numbers_the_tokeniser_drops(layer_token: str) -> None:
    assert _verify("Corn Soy item", LOSSY).passed is True
    vetoed = _verify(f"Corn Soy item {layer_token}", LOSSY)
    assert vetoed.passed is False and vetoed.corroborated_values == ()


def test_a_rung_reading_that_holds_a_sign_the_emitted_token_lacks_vetoes() -> None:
    def reader(lossy_rungs: bool):
        calls = {"n": 0}

        def read(pix):
            calls["n"] += 1
            if calls["n"] == 1 or lossy_rungs:  # call 1 is the base 150 dpi reading
                return "Corn 41 2.0 Soy 1.5 12"
            return "Corn -4.1 2.0 Soy 1.5 12"

        return read

    # emitted 4.1; base reads 41; the finer readings read -4.1 and do not clear the table
    refused = _verify("Corn Soy item", reader(False), TABLE_POS)
    control = _verify("Corn Soy item", reader(True), TABLE_POS)
    assert control.passed is True and control.corroborated_values == ("4.1",)
    assert refused.passed is False and refused.corroborated_values == ()


def test_a_header_spread_over_two_rows_covers_its_numbers_but_a_stub_row_does_not() -> None:
    # the p009 shape: the free reading writes the years in a second header row (empty stub cell)
    marked = _marked(
        "| | Price | Percent change |\n|---|---|---|\n| | | 2001 |\n| Corn | 5.5 | 6.5 |\n"
    )
    paid = _out("| | Price | 2001 (Percent change) |\n|---|---|---|\n| Corn | 5.5 | 6.5 |\n")
    assert winner_accounts_for_marked_tables(paid, marked)
    # the same number one row lower, behind a stub label, is data and earns no credit
    as_data = _marked(
        "| | Price | Percent change |\n|---|---|---|\n| Corn | 5.5 | 6.5 |\n| Soy | | 2001 |\n"
    )
    assert not winner_accounts_for_marked_tables(paid, as_data)
