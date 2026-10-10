"""#988 M2: a navigation bar the model writes as a table is not judged as one.

Coca-Cola's 2018-2021 sustainability reports print a website menu, a section
sub-menu and drawn rules across the top of every page. Gemini writes the menu
as a Markdown table, and the gate refused complete readings of the real table
below it: as an empty table (``table_content_empty``), a ragged one
(``grid_shape``), or because the sub-menu between the menu's two rules was
owed to the table's header (``header_unattributed``). Furniture is found by
repetition across the document's pages (``socr.tables.furniture``).

Each test pins a difference in the same process.
"""

from __future__ import annotations

import json
from pathlib import Path

import fitz

from socr.core.manifest import _apply_table_emission_guard
from socr.core.result import PageOutput, PageStatus
from socr.pipeline.agentic import AcceptDecision, NativeTableVerifierJudge
from socr.pipeline.orchestrator import UnifiedPipeline
from socr.tables import furniture
from socr.tables.furniture import (
    TABLE_FURNITURE_REMOVED_KIND,
    document_furniture,
    furniture_from_pages,
    strip_furniture_runs,
)
from socr.tables.locate import _horizontal_rules
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
HEADER_ONLY_MENU = "| Home | Data | Reports |\n|---|---|---|\n"


# --------------------------------------------------------------------------
# The real page: Coca-Cola 2021 Business & ESG Report p74, job 687398's answer
# --------------------------------------------------------------------------


def test_coke_p74_menu_run_and_menu_band_are_both_left_out() -> None:
    """The answer the cluster gated on p74 (its table complete and correct)
    writes the menu and sub-menu as one ragged pipe run, and the table draws no
    full-width rule of its own, so the nearest two are the menu's. Each half of
    the fix is needed: with only the run left out, the sub-menu between the
    menu's rules is owed to the header; with only the band excused, the run is
    ragged. The menu repeats on p75, and the section's overview (p65, prose
    with no table) prints it too, which is all the scan needs. Without p65 the
    band is owed: p74 and p75 both carry a table."""
    words = [tuple(w) for w in json.loads((FIXTURE / "words.json").read_text())]
    words75 = [tuple(w) for w in json.loads((FIXTURE / "words_p75.json").read_text())]
    words65 = [tuple(w) for w in json.loads((FIXTURE / "words_p65.json").read_text())]
    rules = [tuple(r) for r in json.loads((FIXTURE / "rules_p74.json").read_text())]
    answer = (FIXTURE / "cluster_answer.txt").read_text()
    fur = furniture_from_pages([words, words75, words65])
    gated, removed = strip_furniture_runs(answer, fur.page_words(words))
    is_menu_band = fur.is_menu_band

    assert table_output_defect(answer, words, rules) == DEFECT_GRID_SHAPE
    assert table_output_defect(gated, words, rules) == DEFECT_HEADER_UNATTRIBUTED
    assert table_output_defect(answer, words, rules, is_menu_band) == DEFECT_GRID_SHAPE
    assert table_output_defect(gated, words, rules, is_menu_band) == ""
    without_overview = furniture_from_pages([words, words75]).is_menu_band
    assert table_output_defect(gated, words, rules, without_overview) == DEFECT_HEADER_UNATTRIBUTED
    assert len(removed) == 1 and "Executive Summary" in removed[0]
    assert "Executive Summary" not in gated
    assert "| Year ended December 31," in gated


# --------------------------------------------------------------------------
# Finding furniture on a PDF
# --------------------------------------------------------------------------


def _report_pdf(
    n_pages: int = 3, sub_menu: tuple[str, str] = ("Overview", "Water")
) -> fitz.Document:
    """Each page: a menu line and a sub-menu line (pages 1-2 only), each with a
    full-width rule under it, and a page-specific title. The sub-menu items sit
    over the table's data columns. Page 1 has a table whose own rule spans the
    stub column only, so the nearest full-width rules above it are the menu's
    (as on Coca-Cola 2021 p74). Page 2 repeats one table word at another
    position."""
    doc = fitz.open()
    for i in range(n_pages):
        page = doc.new_page(width=600, height=800)
        page.insert_text((40, 30), "Home   Data   Reports", fontsize=10)
        page.draw_line((40, 36), (560, 36))
        if i < 2:
            page.insert_text((200, 50), sub_menu[0], fontsize=10)
            page.insert_text((300, 50), sub_menu[1], fontsize=10)
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


def test_repeated_words_are_furniture() -> None:
    """A word printed at the same place on two or more pages is furniture (so
    a sub-menu repeated within its section counts); a page's own words are
    not, even when another page prints the same word elsewhere."""
    doc = _report_pdf()
    fur = document_furniture(doc)
    page1 = fur.page_words(doc[0].get_text("words"))
    page3 = fur.page_words(doc[2].get_text("words"))

    assert {"home", "data", "reports", "overview", "water"} <= page1
    assert {"title1", "item", "sales", "1.5"}.isdisjoint(page1)
    assert {"overview", "water"}.isdisjoint(page3)


def test_numbers_are_compared_whole() -> None:
    """PR #1042 review: ``0.32`` is not furniture because the page prints a
    ``0`` and a ``32`` (a page number, a year fragment) at furniture positions."""
    fur = furniture_from_pages([[(10, 10, 20, 20, "0.32")], [(10, 10, 20, 20, "0.32")]])
    assert fur.page_words([(10, 10, 20, 20, "0.32")]) == {"0.32"}

    run = "| Rate | 0.32 |\n|---|---|\n"
    assert strip_furniture_runs(run + "\n" + TABLE, frozenset({"rate", "0", "32"}))[1] == []
    assert strip_furniture_runs(run + "\n" + TABLE, frozenset({"rate", "0.32"}))[1] == [run.strip()]


def test_a_menu_band_is_also_printed_on_a_page_with_no_table() -> None:
    """PR #1042 review: a band repeated word for word is a menu only when a
    word of it is also printed, at the same place, on a page with no data row.
    A table's header is printed only above its table. An ``&`` there does not
    count: it carries no identity of its own."""
    menu = [(200, 50, 240, 60, "Overview"), (300, 50, 330, 60, "Water")]
    amp = [(250, 50, 255, 60, "&")]
    data_row = [(40, 150, 60, 160, "Sales")] + [
        (x, 150, x + 20, 160, v) for x, v in ((200, "1.5"), (300, "2.5"), (400, "3.5"))
    ]
    table_page = menu + amp + data_row

    assert not furniture_from_pages([table_page, table_page]).is_menu_band(menu)
    assert furniture_from_pages([table_page, table_page, menu]).is_menu_band(menu + amp)
    assert not furniture_from_pages([table_page, table_page, amp]).is_menu_band(menu + amp)


# --------------------------------------------------------------------------
# Leaving furniture runs out
# --------------------------------------------------------------------------

MENU_WORDS = frozenset({"home", "data", "reports", "overview", "water", "2020", "goals"})


def test_menu_run_is_left_out_and_the_table_kept() -> None:
    gated, removed = strip_furniture_runs(MENU + "\n" + TABLE, MENU_WORDS)
    assert "Home" not in gated
    assert TABLE in gated
    assert removed == [MENU.strip()]


def test_a_run_with_one_word_of_its_own_is_kept() -> None:
    """Every word of a run must be furniture; one page word keeps it."""
    run = "| Home | Data | Sales |\n|---|---|---|\n"
    gated, removed = strip_furniture_runs(run + "\n" + TABLE, MENU_WORDS)
    assert gated.startswith(run) and removed == []


def test_years_printed_in_the_menu_do_not_keep_it() -> None:
    """Sub-menus carry years ("2020 Sustainability Goals"). Numbers printed at
    furniture positions are furniture like any other word."""
    run = "| Overview | **2020 Goals** | Water |\n|---|---|---|\n"
    assert "Goals" not in strip_furniture_runs(run + "\n" + TABLE, MENU_WORDS)[0]


def test_an_answer_made_only_of_furniture_is_judged_as_written() -> None:
    """With nothing but the menu, the answer has no table of its own; the gate
    must see the menu rather than nothing."""
    assert strip_furniture_runs(MENU, MENU_WORDS) == (MENU, [])
    assert table_output_defect(MENU, None) == DEFECT_GRID_SHAPE


def test_a_header_only_run_beside_a_populated_table_is_still_empty() -> None:
    """PR #1042 review: checks with no document (the manifest backstop,
    native-first, the born-digital native check) cannot tell a menu from a
    data table whose body was dropped, so a header-only run stays a defect
    there, whatever else the page carries."""
    dropped = "| Region | 2019 | 2020 |\n|---|---|---|\n"
    assert table_content_defect(dropped + "\n" + TABLE) == TABLE_CONTENT_EMPTY
    assert table_content_defect(TABLE + "\n" + dropped) == TABLE_CONTENT_EMPTY


# --------------------------------------------------------------------------
# The production gate (NativeTableVerifierJudge) on a PDF with a menu
# --------------------------------------------------------------------------


class _Accept:
    def assess(self, output, provider):
        return AcceptDecision(accept=True, reason="inner judge", confidence=1.0)


def _assess(doc: fitz.Document, text: str, events: list | None = None):
    out = PageOutput(page_num=1, text=text, status=PageStatus.SUCCESS, confidence=0.9)
    judge = NativeTableVerifierJudge(
        inner=_Accept(),
        get_fitz_page=lambda n: doc[n - 1],
        is_table_page=lambda n: True,
        record_event=None if events is None else events.append,
    )
    return judge.assess(out, object()), out


def _scan_fails(monkeypatch) -> None:
    def _raise(_doc):
        raise RuntimeError("scan failed")

    monkeypatch.setattr(furniture, "document_furniture", _raise)


def test_gate_ships_the_answer_without_the_menu_and_records_it(monkeypatch) -> None:
    """Same page, same answer: with the document scanned, the menu run is not
    gated, the answer passes, and the text that ships no longer carries the
    menu; the removal is an audit event. When the scan raises, the gate
    behaves as before #988 M2: the ragged menu run refuses the page and the
    text is left as written."""
    doc = _report_pdf()
    answer = MENU + "\n" + TABLE
    events: list = []

    decision, shipped = _assess(doc, answer, events)
    assert decision.accept is True
    assert "Home" not in shipped.text and TABLE in shipped.text
    removed = [e for e in events if e.kind == TABLE_FURNITURE_REMOVED_KIND]
    assert len(removed) == 1 and removed[0].data["runs"] == [MENU.strip()]

    _scan_fails(monkeypatch)
    refused, kept = _assess(doc, answer)
    assert refused.accept is False
    assert DEFECT_GRID_SHAPE in refused.reason
    assert kept.text == answer


def test_backstop_passes_what_the_gate_ships() -> None:
    """The manifest backstop has no document, so it judges a header-only menu
    run as an empty table. The gate removes the run from the text it accepts,
    so the backstop sees what the gate judged. Left in, the run demotes the
    page."""
    doc = _report_pdf()
    answer = HEADER_ONLY_MENU + "\n" + TABLE

    decision, shipped = _assess(doc, answer)
    assert decision.accept is True
    assert _apply_table_emission_guard(shipped, 1).status == PageStatus.SUCCESS

    as_written = PageOutput(page_num=1, text=answer, status=PageStatus.SUCCESS, confidence=0.9)
    assert _apply_table_emission_guard(as_written, 1).status == PageStatus.ERROR


def test_a_header_band_of_menu_words_is_not_owed(monkeypatch) -> None:
    """The answer writes only the table. The two full-width rules nearest
    above it are the menu's, with only sub-menu words between them. With the
    document scanned that band is not taken for the header and the answer
    passes; when the scan raises, the sub-menu words are owed and the header
    cut refuses it."""
    doc = _report_pdf()

    assert _assess(doc, TABLE)[0].accept is True

    _scan_fails(monkeypatch)
    refused = _assess(doc, TABLE)[0]
    assert refused.accept is False
    assert DEFECT_HEADER_UNATTRIBUTED in refused.reason


def test_a_shared_ampersand_does_not_make_a_menu_band_owed() -> None:
    """Sub-menus print ``&`` ("GHG & Waste"), and so do table headers. An ``&``
    in both is not the answer writing the band: only a token with a letter or
    digit counts as written."""
    doc = _report_pdf(sub_menu=("Overview", "&"))
    answer = TABLE.replace("| Item |", "| Item & unit |", 1)

    assert _assess(doc, answer)[0].accept is True


# --------------------------------------------------------------------------
# PR #1042 review: a table layout repeated on every page keeps its header cut
# --------------------------------------------------------------------------

APPENDIX_HEADER = ("Item", "Alpha", "Beta")


def _appendix_pdf(last_headers: tuple[str, ...], notes_page: bool = False) -> fitz.Document:
    """One booktabs table per page, at the same place on every page: toprule,
    header row, midrule, three data rows. The rules repeat on every page; the
    last column's header is ``last_headers[i]`` on page ``i + 1``. With
    ``notes_page``, a last page prints the header's shared columns (``Item
    Alpha Beta``) with notes under them and no table."""
    doc = fitz.open()
    for i, last in enumerate(last_headers):
        page = doc.new_page(width=600, height=800)
        page.insert_text((40, 60), "Online Appendix", fontsize=10)
        page.draw_line((40, 100), (560, 100))
        for x, text in zip((40, 200, 300, 400), (*APPENDIX_HEADER, last), strict=True):
            page.insert_text((x, 115), text, fontsize=10)
        page.draw_line((40, 122), (560, 122))
        for r in range(3):
            page.insert_text((40, 140 + r * 20), f"Row{r}", fontsize=10)
            for c, x in enumerate((200, 300, 400)):
                page.insert_text((x, 140 + r * 20), f"{10 * i + r}.{c}5", fontsize=10)
        page.draw_line((40, 190), (560, 190))
    if notes_page:
        page = doc.new_page(width=600, height=800)
        for x, text in zip((40, 200, 300), APPENDIX_HEADER, strict=True):
            page.insert_text((x, 115), text, fontsize=10)
        page.insert_text((40, 140), "Notes: standard errors in parentheses.", fontsize=10)
    return doc


def _appendix_answer(last_header: str) -> str:
    rows = "".join(
        f"| Row{r} | " + " | ".join(f"{r}.{c}5" for c in range(3)) + " |\n" for r in range(3)
    )
    return f"| {' | '.join(APPENDIX_HEADER)} | {last_header} |\n|---|---|---|---|\n" + rows


def test_a_repeated_table_layout_keeps_its_header_cut() -> None:
    """Every page draws the table's rules at the same coordinates. The header
    band holds one word of the page's own (``Gamma1``), so it is still the
    header: an answer that drops that word is refused, the full answer passes."""
    doc = _appendix_pdf(("Gamma1", "Gamma2", "Gamma3"))

    assert _assess(doc, _appendix_answer("Gamma1"))[0].accept is True
    refused = _assess(doc, _appendix_answer(""))[0]
    assert refused.accept is False
    assert DEFECT_HEADER_UNATTRIBUTED in refused.reason


def _appendix_page(last_headers: tuple[str, ...], notes_page: bool = False):
    doc = _appendix_pdf(last_headers, notes_page)
    page = doc[0]
    return page.get_text("words"), _horizontal_rules(page), document_furniture(doc)


def _promoted(answer: str) -> str:
    """*answer* with its header row dropped and its first body row promoted into it."""
    _header, separator, first, *rest = answer.splitlines()
    return "\n".join([first, separator, *rest]) + "\n"


def test_a_repeated_header_band_the_answer_writes_none_of_is_owed() -> None:
    """PR #1042 review: every word of the header band is printed at the same
    place on every page, so every word is furniture, and the answer writes none
    of them: a blank header row, or the header row dropped and the first body
    row promoted into it. That is what a menu the model left out looks like.
    The band is printed only above the table, so it is the header and is owed;
    the gate refuses the promoted answer and passes the full one."""
    words, rules, fur = _appendix_page(("Gamma", "Gamma", "Gamma"))
    doc = _appendix_pdf(("Gamma", "Gamma", "Gamma"))
    full = _appendix_answer("Gamma")
    blank = "|  |  |  |  |\n" + full.split("\n", 1)[1]

    for answer in (blank, _promoted(full)):
        assert table_output_defect(answer, words, rules, fur.is_menu_band) == (
            DEFECT_HEADER_UNATTRIBUTED
        )
    refused = _assess(doc, _promoted(full))[0]
    assert refused.accept is False
    assert DEFECT_HEADER_UNATTRIBUTED in refused.reason
    assert _assess(doc, full)[0].accept is True


def test_a_header_band_on_a_notes_page_is_owed_to_a_partial_header() -> None:
    """PR #1042 re-review: the header is also printed on a page with no table (a
    last page of notes), so the band passes for a menu. The answer still wrote
    part of the band (``Item Alpha Beta``), so the band is the table's header
    and the dropped ``Gamma`` is owed."""
    words, rules, fur = _appendix_page(("Gamma", "Gamma", "Gamma"), notes_page=True)
    dropped = _appendix_answer("")

    assert fur.is_menu_band([w for w in words if w[4] in (*APPENDIX_HEADER, "Gamma")])
    assert table_output_defect(dropped, words, rules, fur.is_menu_band) == (
        DEFECT_HEADER_UNATTRIBUTED
    )
    assert table_output_defect(_appendix_answer("Gamma"), words, rules, fur.is_menu_band) == ""


def test_known_limit_a_header_band_on_a_notes_page_is_not_owed_to_a_blank_header() -> None:
    """Disclosed limit (#988): when the header band is repeated word for word,
    is also printed on a page with no table, and the answer writes none of it
    (a blank header row, or the first body row promoted into it), the band
    looks exactly like a menu the model left out, and the header cut abstains.
    Pinned so a change to it is a decision."""
    words, rules, fur = _appendix_page(("Gamma", "Gamma", "Gamma"), notes_page=True)
    full = _appendix_answer("Gamma")
    blank = "|  |  |  |  |\n" + full.split("\n", 1)[1]

    for answer in (blank, _promoted(full)):
        assert table_output_defect(answer, words, rules) == DEFECT_HEADER_UNATTRIBUTED
        assert table_output_defect(answer, words, rules, fur.is_menu_band) == ""


def test_a_band_with_a_page_specific_word_is_owed_even_to_a_blank_header() -> None:
    """The limit above needs EVERY band word to be furniture. With one word the
    page prints only here (``Gamma1``), the band is the table's header even
    when it is also printed on a page with no table and the answer writes none
    of it, and the blank header is refused."""
    words, rules, fur = _appendix_page(("Gamma1", "Gamma2", "Gamma3"), notes_page=True)
    blank = "|  |  |  |  |\n" + _appendix_answer("Gamma1").split("\n", 1)[1]

    assert table_output_defect(blank, words, rules, fur.is_menu_band) == (
        DEFECT_HEADER_UNATTRIBUTED
    )


# --------------------------------------------------------------------------
# PR #1042 re-review: a run with a number never leaves the shipped text
# --------------------------------------------------------------------------


def test_only_a_furniture_run_with_a_numeric_cell_stays_in_the_shipped_text() -> None:
    """A data table repeated at the same place on two pages has cells that are
    numbers. Such a run is left out of what the gate checks but stays in the
    shipped text. A number inside a text cell does not keep a run: Coca-Cola's
    sub-menus print "2020 Sustainability Goals"."""
    text_year = "| Overview | 2030 Goals |\n|---|---|\n"
    numeric = "| Overview | 2030 |\n|---|---|\n"
    words = MENU_WORDS | {"2030"}
    answer = text_year + "\n" + numeric + "\n" + TABLE

    checked, _ = strip_furniture_runs(answer, words)
    shipped, removed = strip_furniture_runs(answer, words, keep_numbered=True)
    assert "2030" not in checked
    assert removed == [text_year.strip()]
    assert numeric in shipped and "Goals" not in shipped


def test_gate_keeps_a_menu_run_with_a_number_in_the_shipped_text() -> None:
    """Same gate, a sub-menu item that is a number: the answer is judged
    without the run and passes, and the run stays in the text that ships."""
    doc = _report_pdf(sub_menu=("Overview", "2030"))
    answer = "| Home | Data | Reports |\n|---|---|---|\n| Overview | 2030 |\n\n" + TABLE
    events: list = []

    decision, shipped = _assess(doc, answer, events)
    assert decision.accept is True
    assert shipped.text == answer
    assert not [e for e in events if e.kind == TABLE_FURNITURE_REMOVED_KIND]


def test_the_removal_record_survives_a_resume() -> None:
    """cubic P2 on #1042: a resumed terminal page skips the gate, so its
    removal record must be replayed from the sidecar or it disappears from the
    audit log while the edited text stays."""
    assert TABLE_FURNITURE_REMOVED_KIND in UnifiedPipeline.resume_restore_kinds()
