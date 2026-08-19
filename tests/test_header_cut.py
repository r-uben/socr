"""GH-212: header attribution proved from the drawn rule, not from token content.

Hermetic — synthetic ``fitz`` pages built with ``insert_text`` plus REAL drawn
rules. That last part is the point: this predicate cuts the page at the rule
above the numeric anchor, so a fixture that only inserts text (the pattern in
``tests/test_agentic.py:282-295``) abstains by construction and can never prove
a HARD. Every HARD case below draws a booktabs-style toprule + midrule.

The six cases are the ones the ratified spec
(``docs/log/2026-08-19_212-header-attribution-design.md``) requires. Four of
them assert a table is NOT rejected — that is the failure direction that got the
previous four predicates reverted, and it is measured: parenthesised column
numbers appear on 11.1% of table pages in the library.
"""

from __future__ import annotations

import fitz

from socr.tables.header_attribution import HeaderVerdict
from socr.tables.header_cut import header_cut_verdict
from socr.tables.locate import _horizontal_rules
from socr.tables.reconcile import find_table_blocks

_LABEL_X = 40.0
_LANES = [150.0, 210.0, 270.0]
_TOP_Y = 90.0  # toprule
_HDR_Y = 105.0  # header band
_MID_Y = 120.0  # midrule
_DATA_YS = [140.0, 160.0, 180.0]

_ROWS = [
    ("Alpha", ["0.12", "0.34", "0.56"]),
    ("Beta", ["1.20", "3.40", "5.60"]),
    ("Gamma", ["2.10", "4.30", "6.50"]),
]


def _md(header: list[str], body: list[list[str]]) -> str:
    sep = "| " + " | ".join(["---"] * len(header)) + " |"
    out = ["| " + " | ".join(header) + " |", sep]
    out += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(out)


def _body_md() -> list[list[str]]:
    return [[label] + vals for label, vals in _ROWS]


def _page(
    header_tokens: list[str] | None,
    *,
    draw_top: bool = True,
    top_as_thick_rect: bool = False,
    second_tier: list[str] | None = None,
) -> fitz.Page:
    """Label + 3 numeric lanes, with real booktabs rules above the data.

    ``top_as_thick_rect`` emits the toprule the way LaTeX often does -- a filled
    rectangle rather than a stroked line. That distinction is load-bearing:
    ``_horizontal_rules`` measures the GEOMETRIC flatness of an "l" item and
    ignores its stroke width, so a visually heavy stroked line survives while an
    equally heavy filled bar is discarded.
    """
    doc = fitz.open()
    page = doc.new_page(width=700, height=400)
    if draw_top:
        if top_as_thick_rect:
            page.draw_rect(fitz.Rect(30, _TOP_Y, 400, _TOP_Y + 1.5), fill=(0, 0, 0), width=0)
        else:
            page.draw_line(fitz.Point(30, _TOP_Y), fitz.Point(400, _TOP_Y), width=0.6)
    if header_tokens is not None:
        for x, tok in zip(_LANES, header_tokens):
            page.insert_text((x, _HDR_Y), tok, fontsize=9)
    if second_tier is not None:
        for x, tok in zip(_LANES, second_tier):
            page.insert_text((x, _HDR_Y + 12), tok, fontsize=9)
    page.draw_line(fitz.Point(30, _MID_Y), fitz.Point(400, _MID_Y), width=0.4)
    for y, (label, vals) in zip(_DATA_YS, _ROWS):
        page.insert_text((_LABEL_X, y), label, fontsize=9)
        for x, v in zip(_LANES, vals):
            page.insert_text((x, y), v, fontsize=9)
    return page


def _verdict(page, md: str) -> HeaderVerdict:
    blocks = find_table_blocks(md)
    assert blocks, "fixture markdown produced no table block"
    return header_cut_verdict(blocks[0].grid, page.get_text("words"), _horizontal_rules(page))


class TestRejectsDestroyedHeaders:
    def test_blank_header_row_is_hard(self):
        """The 4-of-4 case: native header band present, emitted header blank."""
        page = _page(["1997", "2002", "2007"])
        md = _md(["", "", "", ""], _body_md())
        assert _verdict(page, md) is HeaderVerdict.HARD

    def test_header_tokens_dumped_into_a_body_row_is_hard(self):
        """Page 021's shape: header row blank, its tokens landed in the body.

        Tokens being *somewhere* in the markdown is not attribution — in a body
        row they bind as data, and the columns stay unnamed.
        """
        page = _page(["1997", "2002", "2007"])
        body = _body_md() + [["", "1997", "2002", "2007"]]
        md = _md(["", "", "", ""], body)
        assert _verdict(page, md) is HeaderVerdict.HARD


class TestDoesNotRejectCorrectTables:
    def test_intact_header_is_ok(self):
        page = _page(["1997", "2002", "2007"])
        md = _md(["", "1997", "2002", "2007"], _body_md())
        assert _verdict(page, md) is HeaderVerdict.OK

    def test_parenthesised_column_numbers_emitted_bare_is_ok(self):
        """11.1% of library table pages: native "(1)" vs emitted "1".

        ``strip_presentation`` deliberately never folds parentheses (they mark
        negatives on the numeric path), so without a bracket fold on THIS path
        alone the predicate would reject roughly one table page in nine.
        """
        page = _page(["(1)", "(2)", "(3)"])
        md = _md(["", "1", "2", "3"], _body_md())
        assert _verdict(page, md) is HeaderVerdict.OK

    def test_star_and_na_body_rows_are_never_owed(self):
        """Significance stars and n.a. sit BELOW the midrule, so they are out.

        This is the exact construction that reverted the positional attempt.
        """
        page = _page(["1997", "2002", "2007"])
        page.insert_text((_LABEL_X, 200.0), "n.a.", fontsize=9)
        for x in _LANES:
            page.insert_text((x, 200.0), "***", fontsize=9)
        md = _md(["", "1997", "2002", "2007"], _body_md())
        assert _verdict(page, md) is HeaderVerdict.OK

    def test_two_tier_header_emitted_across_two_rows_is_ok(self):
        """A faithful emission may put the second tier in grid[1]."""
        page = _page(["Model", "Model", "Model"], second_tier=["(1)", "(2)", "(3)"])
        md = _md(["", "Model", "Model", "Model"], [["", "1", "2", "3"]] + _body_md())
        assert _verdict(page, md) is HeaderVerdict.OK


class TestAbstains:
    def test_no_toprule_abstains_rather_than_owing_the_page(self):
        """Without a rule above the cut the header has no upper bound.

        Falling back to the top of the table neighbourhood would owe the
        caption, the running head, and any prose numeral near a lane.
        """
        page = _page(["1997", "2002", "2007"], draw_top=False)
        md = _md(["", "", "", ""], _body_md())
        assert _verdict(page, md) is HeaderVerdict.UNVERIFIABLE

    def test_thick_toprule_is_discarded_upstream_so_abstains(self):
        """1.6% of library table pages; page 021's toprule is 1.494pt.

        ``_RULE_FLATNESS_PT`` is 1.0, so a heavy rule never reaches us.
        """
        page = _page(["1997", "2002", "2007"], top_as_thick_rect=True)
        md = _md(["", "", "", ""], _body_md())
        assert _verdict(page, md) is HeaderVerdict.UNVERIFIABLE

    def test_two_panels_sharing_a_data_row_abstain(self):
        """No unique anchor: the cut would land on the wrong panel's midrule."""
        doc = fitz.open()
        page = doc.new_page(width=700, height=700)
        for base in (0.0, 300.0):
            page.draw_line(fitz.Point(30, _TOP_Y + base), fitz.Point(400, _TOP_Y + base), width=0.6)
            for x, tok in zip(_LANES, ["1997", "2002", "2007"]):
                page.insert_text((x, _HDR_Y + base), tok, fontsize=9)
            page.draw_line(fitz.Point(30, _MID_Y + base), fitz.Point(400, _MID_Y + base), width=0.4)
            for y, (label, vals) in zip(_DATA_YS, _ROWS):
                page.insert_text((_LABEL_X, y + base), label, fontsize=9)
                for x, v in zip(_LANES, vals):
                    page.insert_text((x, y + base), v, fontsize=9)
        md = _md(["", "", "", ""], _body_md())
        assert _verdict(page, md) is HeaderVerdict.UNVERIFIABLE

    def test_no_rules_at_all_abstains(self):
        doc = fitz.open()
        page = doc.new_page(width=700, height=400)
        for x, tok in zip(_LANES, ["1997", "2002", "2007"]):
            page.insert_text((x, _HDR_Y), tok, fontsize=9)
        for y, (label, vals) in zip(_DATA_YS, _ROWS):
            page.insert_text((_LABEL_X, y), label, fontsize=9)
            for x, v in zip(_LANES, vals):
                page.insert_text((x, y), v, fontsize=9)
        md = _md(["", "", "", ""], _body_md())
        assert _verdict(page, md) is HeaderVerdict.UNVERIFIABLE
