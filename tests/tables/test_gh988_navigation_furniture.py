"""#988 M2: a navigation bar the model writes as a table is not judged as one.

Coca-Cola's 2018-2021 sustainability reports print a website menu, a section
sub-menu and drawn rules across the top of every page. Gemini writes the menu
as a Markdown table, and the gate refused complete readings of the real table
below it: as an empty table (``table_content_empty``), a ragged one
(``grid_shape``), or because the menu's rules became the header cut
(``header_unattributed``). Furniture is found by repetition across the
document's pages (``socr.tables.furniture``).

Each test pins a difference in the same process.
"""

from __future__ import annotations

import json
from pathlib import Path

import fitz

from socr.core.result import PageOutput, PageStatus
from socr.pipeline.agentic import AcceptDecision, NativeTableVerifierJudge
from socr.tables import furniture
from socr.tables.furniture import (
    DocumentFurniture,
    document_furniture,
    furniture_from_pages,
    strip_furniture_runs,
)
from socr.tables.reconcile import TABLE_CONTENT_EMPTY, table_content_defect
from socr.tables.structure_check import (
    DEFECT_GRID_SHAPE,
    DEFECT_HEADER_UNATTRIBUTED,
    table_output_defect,
)

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "gh988_coke_2021_p74"

TABLE = (
    "| Item | 2018 | 2019 | 2020 |\n"
    "|---|---|---|---|\n"
    "| Sales | 1.5 | 2.5 | 3.5 |\n"
    "| Costs | 4.5 | 5.5 | 6.5 |\n"
    "| Profit | 7.5 | 8.5 | 9.5 |\n"
)
MENU = "| Home | Data | Reports |\n|---|---|---|\n| Overview | Water |\n"


# --------------------------------------------------------------------------
# The real page: Coca-Cola 2021 Business & ESG Report p74, job 687398's answer
# --------------------------------------------------------------------------


def _coke_p74() -> tuple[str, list[tuple], list[tuple], DocumentFurniture]:
    words = [tuple(w) for w in json.loads((FIXTURE / "words.json").read_text())]
    words75 = [tuple(w) for w in json.loads((FIXTURE / "words_p75.json").read_text())]
    rules = {
        int(page): [tuple(r) for r in page_rules]
        for page, page_rules in json.loads((FIXTURE / "rules.json").read_text()).items()
    }
    # Words of p74 and p75 (the menu repeats on both); rules of all 86 pages.
    pages = [(words if n == 74 else words75 if n == 75 else [], rules[n]) for n in sorted(rules)]
    answer = (FIXTURE / "cluster_answer.txt").read_text()
    return answer, words, rules[74], furniture_from_pages(pages)


def test_coke_p74_menu_table_and_menu_rules_are_both_left_out() -> None:
    """The answer the cluster gated on p74 (its table complete and correct)
    writes the menu and sub-menu as one ragged pipe run, and the page draws
    two full-width rules under the menu on most pages of the report. Each
    half of the fix is needed: with only the run left out, the menu rules cut
    the header; with only the rules left out, the run is ragged."""
    answer, words, rules, fur = _coke_p74()
    gated, kept = strip_furniture_runs(answer, fur.page_words(words)), fur.keep_rules(rules)

    assert table_output_defect(answer, words, rules) == DEFECT_GRID_SHAPE
    assert table_output_defect(gated, words, rules) == DEFECT_HEADER_UNATTRIBUTED
    assert table_output_defect(answer, words, kept) == DEFECT_GRID_SHAPE
    assert table_output_defect(gated, words, kept) == ""
    assert "Executive Summary" in answer and "Executive Summary" not in gated
    assert "| Year ended December 31," in gated


# --------------------------------------------------------------------------
# Finding furniture on a PDF
# --------------------------------------------------------------------------


def _report_pdf(n_pages: int = 3) -> fitz.Document:
    """Each page: a menu line and a sub-menu line (pages 1-2 only), each with a
    full-width rule under it, and a page-specific title. The sub-menu items sit
    over the table's data columns. Page 1 has a table
    whose own rule spans the stub column only, so the nearest full-width rules
    above it are the menu's (as on Coca-Cola 2021 p74). Page 2 repeats one
    table word at another position."""
    doc = fitz.open()
    for i in range(n_pages):
        page = doc.new_page(width=600, height=800)
        page.insert_text((40, 30), "Home   Data   Reports", fontsize=10)
        page.draw_line((40, 36), (560, 36))
        if i < 2:
            page.insert_text((200, 50), "Overview", fontsize=10)
            page.insert_text((300, 50), "Water", fontsize=10)
        page.draw_line((40, 56), (560, 56))
        page.insert_text((40, 80), f"Title{i + 1}", fontsize=14)
        if i == 1:
            page.insert_text((300, 300), "Sales", fontsize=10)  # page 1 prints it elsewhere
        if i == 0:
            for x, text in ((40, "Item"), (200, "2018"), (300, "2019"), (400, "2020")):
                page.insert_text((x, 120), text, fontsize=10)
            page.draw_line((40, 126), (150, 126))  # under the stub column only
            for row, line in enumerate(TABLE.splitlines()[2:]):
                cells = [c.strip() for c in line.strip("|").split("|")]
                for x, text in zip((40, 200, 300, 400), cells, strict=True):
                    page.insert_text((x, 145 + row * 20), text, fontsize=10)
    return doc


def test_repeated_words_and_majority_rules_are_furniture() -> None:
    """A word printed at the same place on two or more pages is furniture (so
    a sub-menu repeated within its section counts); a page's own words are
    not, even when another page prints the same word elsewhere. A rule is
    furniture only when drawn on most pages: the table's own rule on page 1
    is kept."""
    doc = _report_pdf()
    fur = document_furniture(doc)
    page1 = fur.page_words(doc[0].get_text("words"))
    page3 = fur.page_words(doc[2].get_text("words"))

    assert {"home", "data", "reports", "overview", "water"} <= page1
    assert {"title1", "item", "sales", "1", "5"}.isdisjoint(page1)
    assert {"overview", "water"}.isdisjoint(page3)
    assert [round(r[0]) for r in fur.keep_rules(fitz_rules(doc[0]))] == [126]


def fitz_rules(page) -> list[tuple[float, float, float]]:
    from socr.tables.locate import _horizontal_rules

    return _horizontal_rules(page)


def test_a_rule_on_half_the_pages_or_fewer_is_kept() -> None:
    """Two pages of four draw the same rule: not a majority, not furniture."""
    rule = (100.0, 40.0, 560.0)
    fur = furniture_from_pages([([], [rule]), ([], [rule]), ([], []), ([], [])])
    assert fur.keep_rules([rule]) == [rule]
    fur = furniture_from_pages([([], [rule]), ([], [rule]), ([], [rule]), ([], [])])
    assert fur.keep_rules([rule]) == []


# --------------------------------------------------------------------------
# Leaving furniture runs out of what is gated
# --------------------------------------------------------------------------

MENU_WORDS = frozenset({"home", "data", "reports", "overview", "water", "2020", "goals"})


def test_menu_run_is_left_out_and_the_table_kept() -> None:
    gated = strip_furniture_runs(MENU + "\n" + TABLE, MENU_WORDS)
    assert "Home" not in gated
    assert TABLE.strip() in gated


def test_a_run_with_one_word_of_its_own_is_kept() -> None:
    """Every word of a run must be furniture; one page word keeps it."""
    run = "| Home | Data | Sales |\n|---|---|---|\n"
    assert strip_furniture_runs(run + "\n" + TABLE, MENU_WORDS).startswith(run)


def test_years_printed_in_the_menu_do_not_keep_it() -> None:
    """Sub-menus carry years ("2020 Sustainability Goals"). Numbers printed at
    furniture positions are furniture like any other word."""
    run = "| Overview | **2020 Goals** | Water |\n|---|---|---|\n"
    assert "Goals" not in strip_furniture_runs(run + "\n" + TABLE, MENU_WORDS)


def test_an_answer_made_only_of_furniture_is_judged_as_written() -> None:
    """With nothing but the menu, the answer has no table of its own; the gate
    must see the menu rather than nothing."""
    assert strip_furniture_runs(MENU, MENU_WORDS) == MENU
    assert table_output_defect(strip_furniture_runs(MENU, MENU_WORDS), None) == DEFECT_GRID_SHAPE


# --------------------------------------------------------------------------
# table_content_empty and header-only runs
# --------------------------------------------------------------------------

HEADER_ONLY = "| Home | Data | Reports |\n|---|---|---|\n"


def test_header_only_run_beside_a_populated_table_is_not_empty() -> None:
    assert table_content_defect(HEADER_ONLY + "\n" + TABLE) == ""
    assert table_content_defect(TABLE + "\n" + HEADER_ONLY) == ""


def test_header_only_run_on_its_own_is_still_empty() -> None:
    assert table_content_defect(HEADER_ONLY) == TABLE_CONTENT_EMPTY
    assert table_content_defect(HEADER_ONLY + "\nSome prose.\n\n" + HEADER_ONLY) == (
        TABLE_CONTENT_EMPTY
    )


def test_a_placeholder_body_is_empty_beside_a_populated_table() -> None:
    placeholders = "| A | B |\n|---|---|\n| - | - |\n| — | — |\n"
    assert table_content_defect(placeholders + "\n" + TABLE) == TABLE_CONTENT_EMPTY


def test_known_limit_a_header_only_data_table_beside_a_populated_one_passes() -> None:
    """Disclosed limit (#988): the content term cannot tell a menu from a data
    table whose whole body was dropped, so a second table written header-only
    beside a populated one is no longer refused as empty. Its numbers are then
    missing for the value guard and the row-shortfall term to find. Pinned so
    a change to it is a decision."""
    dropped = "| Region | 2019 | 2020 |\n|---|---|---|\n"
    assert table_content_defect(dropped + "\n" + TABLE) == ""


# --------------------------------------------------------------------------
# The production gate (NativeTableVerifierJudge) on a PDF with a menu
# --------------------------------------------------------------------------


class _Accept:
    def assess(self, output, provider):
        return AcceptDecision(accept=True, reason="inner judge", confidence=1.0)


def _judge(doc: fitz.Document) -> NativeTableVerifierJudge:
    return NativeTableVerifierJudge(
        inner=_Accept(), get_fitz_page=lambda n: doc[n - 1], is_table_page=lambda n: True
    )


def _assess(doc: fitz.Document, text: str) -> AcceptDecision:
    out = PageOutput(page_num=1, text=text, status=PageStatus.SUCCESS, confidence=0.9)
    return _judge(doc).assess(out, object())


def test_gate_leaves_the_menu_out_and_fails_closed_when_the_scan_fails(monkeypatch) -> None:
    """Same page, same answer: with the document scanned, the menu run is not
    gated and the answer passes; when the scan raises, the gate behaves as
    before #988 M2 and the ragged menu run refuses the page."""
    doc = _report_pdf()
    answer = MENU + "\n" + TABLE

    assert _assess(doc, answer).accept is True

    def _raise(_doc):
        raise RuntimeError("scan failed")

    monkeypatch.setattr(furniture, "document_furniture", _raise)
    refused = _assess(doc, answer)
    assert refused.accept is False
    assert DEFECT_GRID_SHAPE in refused.reason


def test_gate_leaves_the_menu_rules_out_of_the_header_cut(monkeypatch) -> None:
    """The answer writes only the table. The two full-width rules nearest above
    it are the menu's, so with every rule kept the header cut owes the
    sub-menu words to the table's header and refuses it; with the document
    scanned, the menu rules are dropped and the answer passes."""
    doc = _report_pdf()

    assert _assess(doc, TABLE).accept is True

    def _raise(_doc):
        raise RuntimeError("scan failed")

    monkeypatch.setattr(furniture, "document_furniture", _raise)
    refused = _assess(doc, TABLE)
    assert refused.accept is False
    assert DEFECT_HEADER_UNATTRIBUTED in refused.reason
