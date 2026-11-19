"""#1378: a page that prints no number is not doubted by the number check.

Pins (a) the Roman-numeral page references the two sides of the comparison tokenised
differently (``VIII-1``), and (b) the exit for a page with nothing to check, which needs the
layer to hold no further number AND independent evidence (the #1050 ink coverage) that the
layer holds all of the page. Each guard is a DIFFERENCE between two runs of the same clause
that differ in one input, so no outcome measured on one machine is pinned.
"""

from __future__ import annotations

import pytest

from socr.core.marks import CheckResult
from socr.core.result import PageOutput, PageStatus
from socr.core.state import PageState
from socr.core.verdict import (
    NO_NUMBER_ON_PAGE_REASON,
    text_layer_numbers_clause,
    text_layer_numbers_outcome,
)
from socr.tables.reading_agreement import PROVENANCE_OWN_CHARACTERS, compare_numbers

CONTENTS = "\n".join(
    f"{name} ..... {ref}"
    for name, ref in [
        ("First", "I-1"),
        ("Second", "II-1"),
        ("Seventh", "VII-1"),
        ("Eighth", "VIII-1"),
        ("Thirteenth", "XIII-12"),
    ]
)


def _page(layer: str, *, coverage: str = "passed", born_digital: bool = False, shape: bool = True):
    """``coverage`` is the strict ink fact: "passed" -> accounted for, anything else -> not."""
    p = PageState(page_num=1)
    p.own_layer_text = layer
    p.is_born_digital = born_digital
    p.layer_refused_for_shape_alone = shape
    p.text_ink_fully_accounted = coverage == "passed"
    return p


def _read(text: str) -> PageOutput:
    return PageOutput(page_num=1, text=text, status=PageStatus.SUCCESS, engine="qwen")


def _result(output: PageOutput, p: PageState):
    clause = text_layer_numbers_clause(output, p)
    return None if clause is None else clause.result


# --- tokenisation --------------------------------------------------------------------


def test_roman_page_references_tokenise_the_same_on_both_sides() -> None:
    """The layer's ``VIII-1`` frees a ``1``; the reading must claim it, as it does ``VII-1``."""
    record = compare_numbers(CONTENTS, CONTENTS, provenance=PROVENANCE_OWN_CHARACTERS)
    assert record.compared == 0
    assert record.reference_only == 0
    assert record.disputed == ()


def test_roman_reference_claim_does_not_hide_a_dropped_value() -> None:
    """A label digit claims ONE layer number: a layer value the reading lacks stays counted."""
    layer = CONTENTS + "\nTotal 41"
    record = compare_numbers(layer, CONTENTS, provenance=PROVENANCE_OWN_CHARACTERS)
    assert record.reference_only == 1


def test_a_word_hyphen_digit_is_still_not_a_label() -> None:
    """Only a well-formed Roman numeral is widened: ``Korea-8`` keeps its old treatment."""
    base = compare_numbers("Korea 8", "Korea-8", provenance=PROVENANCE_OWN_CHARACTERS)
    widened = compare_numbers("Korea-8", "Korea-8", provenance=PROVENANCE_OWN_CHARACTERS)
    assert base.compared == widened.compared == 0
    assert widened.reference_only == 1  # the layer's 8 is not claimed by the reading's word


# --- the exit: only with the ink-coverage evidence -----------------------------------


@pytest.mark.parametrize(
    ("layer", "reading"),
    [
        ("International Developments", "# International Developments"),
        ("(This page intentionally blank.)", "(This page intentionally blank.)"),
        ("(Page I-58 is intentionally blank.)", "(Page I-58 is intentionally blank.)"),
        (CONTENTS, CONTENTS),
    ],
)
def test_a_page_with_no_number_is_not_doubted_only_when_coverage_passed(layer, reading) -> None:
    covered = text_layer_numbers_outcome(_read(reading), _page(layer))
    uncovered = text_layer_numbers_outcome(_read(reading), _page(layer, coverage=""))
    assert covered.clause is None
    assert covered.entry.reason == NO_NUMBER_ON_PAGE_REASON
    assert uncovered.clause is not None
    assert uncovered.clause.result is CheckResult.UNKNOWN


@pytest.mark.parametrize("verdict", ["flagged", "unknown", ""])
def test_coverage_that_did_not_pass_keeps_the_page_warned(verdict) -> None:
    p = _page("International Developments", coverage=verdict)
    assert _result(_read("# International Developments"), p) is CheckResult.UNKNOWN


def test_guard_layer_prints_numbers_and_reading_ships_none_is_still_warned() -> None:
    p = _page("Growth was 2.1 and 3.4 in 2017.", born_digital=True, shape=False)
    assert _result(_read("Growth was steady."), p) is CheckResult.UNKNOWN
    # the difference: the same reading on a layer with no number is not doubted
    p_none = _page("Growth was steady.", born_digital=True, shape=False)
    assert _result(_read("Growth was steady."), p_none) is None


def test_guard_scan_without_layer_is_still_warned_whatever_the_coverage() -> None:
    for coverage in ("passed", "flagged", ""):
        p = _page("", coverage=coverage)
        assert _result(_read("# International Developments"), p) is CheckResult.UNKNOWN


def test_guard_layer_not_shown_clean_keeps_the_page_warned() -> None:
    """A refused layer that is not shape-refused-clean (p011's dot leaders) confirms nothing."""
    refused = _page("Summary", shape=False)
    assert _result(_read("Summary"), refused) is CheckResult.UNKNOWN
    shape_ok = _page("Summary", shape=True)
    assert _result(_read("Summary"), shape_ok) is None
    damaged = _page("Summary", born_digital=True, shape=False)
    damaged.minus_as_digit_hits = True
    assert _result(_read("Summary"), damaged) is CheckResult.UNKNOWN


def test_guard_numbers_in_the_reading_are_still_compared() -> None:
    """The exit is for ``compared == 0`` only: a number the layer lacks is still a FAIL/UNKNOWN."""
    p = _page("Rate", born_digital=True, shape=False)
    assert _result(_read("Rate was 9.31"), p) is CheckResult.FAIL


# --- the words check confirms a clean short layer (second commit) -------------------------

from socr.core.marks import Check  # noqa: E402
from socr.core.verdict import (  # noqa: E402
    WORDS_LAYER_NOT_EVIDENCE_REASON,
    independent_page_evidence,
    text_layer_words_outcome,
)


def _with_words(p: PageState, words: str) -> PageState:
    p.own_layer_word_reference = words
    return p


def _words(text: str, p: PageState):
    return text_layer_words_outcome(_read(text), p).entry


def test_words_agree_on_a_clean_short_layer_is_a_pass() -> None:
    p = _with_words(_page("International Developments"), "international developments")
    entry = _words("# International Developments", p)
    assert entry.result.value == "PASS"
    p_open = _with_words(
        _page("International Developments", coverage=""), "international developments"
    )
    other = _words("# International Developments", p_open)
    assert other.result.value == "NOT_RUN"
    assert other.reason == WORDS_LAYER_NOT_EVIDENCE_REASON


def test_one_word_differs_keeps_the_page_warned() -> None:
    p = _with_words(_page("International Developments"), "international developments")
    outcome = text_layer_words_outcome(_read("# International Development"), p)
    assert outcome.clause is not None
    assert outcome.clause.result is CheckResult.UNKNOWN


@pytest.mark.parametrize("verdict", ["flagged", "unknown", ""])
def test_a_short_layer_with_coverage_not_passed_is_not_evidence(verdict) -> None:
    """A short running head over a scanned body: the body's ink is uncovered, so ``flagged``."""
    p = _with_words(_page("Running head", coverage=verdict), "running head")
    assert _words("Running head", p).result.value == "NOT_RUN"


def test_a_layer_not_shown_clean_is_not_evidence_even_with_coverage() -> None:
    p = _with_words(_page("Summary", shape=False), "summary")
    assert _words("Summary", p).result.value == "NOT_RUN"
    sym = _with_words(_page("Summary"), "summary")
    sym.has_unrecovered_symbol_glyphs = True
    assert _words("Summary", sym).result.value == "NOT_RUN"


def test_a_divider_page_is_independently_confirmed_only_with_a_words_pass() -> None:
    """Numbers N/A alone is no confirmation (the page stays uncorroborated); with a words PASS
    on the same text it is."""
    from socr.core.verdict import stamp_text_digest

    text = "# International Developments"
    for coverage, confirmed in (("passed", True), ("", False)):
        p = _with_words(
            _page("International Developments", coverage=coverage), "international developments"
        )
        out = _read(text)
        entries = [
            stamp_text_digest(text_layer_words_outcome(out, p).entry, text),
            stamp_text_digest(text_layer_numbers_outcome(out, p).entry, text),
        ]
        assert independent_page_evidence(entries, text) is confirmed
    assert Check.TEXT_LAYER_WORDS.value == "text_layer_words"


# --- the strict ink fact, built from real PDFs (blocking review finding) --------------------

import fitz  # noqa: E402

from socr.core.text_coverage import analyze_page  # noqa: E402

TITLE = "International Developments"


def _title_page(doc: fitz.Document, title: str = TITLE) -> fitz.Page:
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 80), title, fontsize=18)
    return page


def _raster_of(draw, *, dpi: int = 150, size=(400, 260)) -> fitz.Pixmap:
    scratch = fitz.open()
    sp = scratch.new_page(width=size[0], height=size[1])
    draw(sp)
    return sp.get_pixmap(dpi=dpi)


def _with_raster(draw, *, dpi: int = 150) -> fitz.Page:
    doc = fitz.open()
    page = _title_page(doc)
    page.insert_image(fitz.Rect(72, 200, 472, 460), pixmap=_raster_of(draw, dpi=dpi))
    return page


def _grid(values, fontsize, color=(0, 0, 0)):
    def draw(sp):
        for r, row in enumerate(values):
            for c, v in enumerate(row):
                sp.insert_text((20 + 70 * c, 30 + 40 * r), v, fontsize=fontsize, color=color)

    return draw


DECIMALS = [[f"{r}.{c}{r + c}" for c in range(5)] for r in range(1, 5)]
INTEGERS = [[str(10 + r * 5 + c) for c in range(5)] for r in range(4)]


def _single(text, fontsize):
    def draw(sp):
        sp.insert_text((60, 120), text, fontsize=fontsize)

    return draw


def _bars(sp):
    for i, v in enumerate((30, 80, 55, 100)):
        sp.draw_rect(fitz.Rect(40 + 80 * i, 220 - v, 90 + 80 * i, 220), fill=(0, 0, 0))
        sp.insert_text((50 + 80 * i, 235), str(10 * (i + 3)), fontsize=10)


def _ticks(sp):
    sp.draw_line((30, 200), (380, 200))
    for i, y in enumerate(range(2015, 2024, 2)):
        sp.insert_text((30 + 70 * i, 218), str(y), fontsize=9)
    for j in range(5):
        sp.insert_text((4, 200 - 40 * j), str(j * 10), fontsize=9)


REVIEWER_RASTERS = {
    "decimals_10pt": (_grid(DECIMALS, 10), 150),
    "integers_10pt": (_grid(INTEGERS, 10), 150),
    "decimals_grey": (_grid(DECIMALS, 10, (0.75, 0.75, 0.75)), 150),
    "decimals_72dpi": (_grid(DECIMALS, 10), 72),
    "chart_ticks_years": (_ticks, 150),
    "bar_value_labels": (_bars, 150),
    "single_value": (_single("3.25", 12), 150),
    "page_number": (_single("17", 12), 150),
    "year_48pt": (_single("2023", 48), 150),
}


def test_the_covered_title_page_is_accounted_for() -> None:
    doc = fitz.open()
    cov = analyze_page(_title_page(doc))
    assert cov.verdict == "passed"
    assert cov.ink_fully_accounted is True


@pytest.mark.parametrize("name", sorted(REVIEWER_RASTERS))
def test_a_raster_of_numbers_under_a_title_layer_is_never_accounted_for(name) -> None:
    """The reviewer's repro: ``passed`` holds for most of these (no word-like run), the strict
    fact never does, and so the page stays warned end to end."""
    draw, dpi = REVIEWER_RASTERS[name]
    page = _with_raster(draw, dpi=dpi)
    cov = analyze_page(page)
    assert cov.ink_fully_accounted is False
    p = _page(TITLE)
    p.text_coverage_verdict = cov.verdict
    p.text_ink_fully_accounted = cov.ink_fully_accounted
    assert _result(_read(f"# {TITLE}"), p) is CheckResult.UNKNOWN
    # the difference: the same page with the fact granted would have exited
    p.text_ink_fully_accounted = True
    assert _result(_read(f"# {TITLE}"), p) is None


def test_the_verdict_alone_is_not_the_fact() -> None:
    """At least one reviewer input is ``passed`` yet not accounted for: the tests above are not
    vacuous against the old predicate."""
    passed = [
        n
        for n, (draw, dpi) in REVIEWER_RASTERS.items()
        if analyze_page(_with_raster(draw, dpi=dpi)).verdict == "passed"
    ]
    assert passed


def test_a_vector_drawing_is_never_accounted_for() -> None:
    doc = fitz.open()
    page = _title_page(doc)
    page.draw_rect(fitz.Rect(72, 200, 100, 220), fill=(0, 0, 0))
    assert analyze_page(page).ink_fully_accounted is False


def test_glyph_sized_ink_outside_the_layer_is_never_accounted_for(monkeypatch) -> None:
    """The ink rule alone (no raster, no drawing): the number line is real ink, and the layer the
    analysis is given omits it, as a layer short of the page's glyphs would."""
    from socr.core import text_coverage

    doc = fitz.open()
    page = _title_page(doc)
    page.insert_text((72, 300), "3.25", fontsize=12)
    real = text_coverage.visible_char_boxes

    def title_only(pg):
        vis, inv = real(pg)
        return [b for b in vis if b[1] < 150], inv

    assert analyze_page(page).ink_fully_accounted is True
    monkeypatch.setattr(text_coverage, "visible_char_boxes", title_only)
    assert analyze_page(page).ink_fully_accounted is False


# --- the exit does not depend on the number tokeniser ---------------------------------------

GLUED = [
    "USD25",
    "EUR50",
    "No.12",
    "Dec.31",
    "Vol.12",
    "bp25",
    "T-10",
    "Q4",
    "M2-1",
    "VI-1.5",
    "VIII-100",
    "MMIV-12",
    "$5bn",
    "3.2pp",
    "1.5x",
    "FY2023",
    "up2.3%",
]


@pytest.mark.parametrize("glued", GLUED)
def test_a_glued_number_in_the_reading_stays_warned(glued) -> None:
    p = _page("Selected figures")
    assert _result(_read(f"Selected figures {glued}"), p) is CheckResult.UNKNOWN
    # the difference: the same page and reading without the glued number exits
    assert _result(_read("Selected figures"), p) is None


def test_the_same_digits_on_both_sides_still_exit() -> None:
    p = _page("(Page I-58 is intentionally blank.)")
    assert _result(_read("(Page I-58 is intentionally blank.)"), p) is None


def test_an_en_dash_layer_value_is_not_claimed_by_a_reading_label() -> None:
    p = _page("Section VIII\u20131 Total 1", shape=True)
    assert _result(_read("Section VIII-1 Total"), p) is CheckResult.UNKNOWN


def test_a_layer_number_fused_to_a_dot_leader_is_not_dropped() -> None:
    p = _page("Summary.....41")
    assert _result(_read("Summary"), p) is CheckResult.UNKNOWN


def test_a_cosmetic_encoding_flag_on_a_born_digital_layer_keeps_the_page_warned() -> None:
    p = _page("Summary", born_digital=True, shape=False)
    assert _result(_read("Summary"), p) is None
    p.has_encoding_hygiene_suspect = True
    assert _result(_read("Summary"), p) is CheckResult.UNKNOWN


# --- the refinalize seam (the line the review found untested) -------------------------------


def test_refinalize_reads_the_strict_fact_from_the_sidecar() -> None:
    from socr.core.born_digital import PageAssessment

    from socr.pipeline.refinalize_own_text import _page_state

    sidecar = {"page_num": 1, "text_ink_fully_accounted": True, "text_coverage": "passed"}
    pa = PageAssessment(page_num=1, is_born_digital=False, native_text="", confidence=0.0)
    assert _page_state(sidecar, pa).text_ink_fully_accounted is True
    old = {"page_num": 1, "text_coverage": "passed"}
    assert _page_state(old, pa).text_ink_fully_accounted is False


def test_a_thin_rule_does_not_stop_a_page_being_accounted_for() -> None:
    """p029 and p043 carry two hairline rules; a rule cannot hold a glyph."""
    doc = fitz.open()
    page = _title_page(doc)
    page.draw_rect(fitz.Rect(111, 103, 500, 104), fill=(0, 0, 0))
    assert analyze_page(page).ink_fully_accounted is True


# --- review round 2: short tokens, two-way agreement, drawn digits, run-side wiring ----------

SHORT_TOKEN_PAIRS = [
    ("Part II / International Developments", "Part I / International Developments", False),
    ("Part II", "Part III", False),
    ("Chapter XII", "Chapter XIII", False),
    ("Chapter XIII", "Chapter XVII", False),
    ("Chapter IV", "Chapter", False),
    ("Part Two", "Part Ten", False),
    ("Annex C", "Annex G", False),
    ("Rate ½", "Rate", False),
    ("Part Ⅷ", "Part", False),
    ("Tier II capital " * 5, "Tier I capital " * 5, True),
]


@pytest.mark.parametrize(("layer", "reading", "born_digital"), SHORT_TOKEN_PAIRS)
def test_a_short_token_that_differs_keeps_the_page_warned(layer, reading, born_digital) -> None:
    p = _page(layer, born_digital=born_digital, shape=not born_digital)
    assert _result(_read(reading), p) is CheckResult.UNKNOWN
    # the difference: the same page and layer with the reading unchanged exits
    assert _result(_read(layer), p) is None


def _evidence(layer: str, reading: str) -> str:
    p = _with_words(_page(layer), layer.casefold())
    return text_layer_words_outcome(_read(reading), p).entry.result.value


def test_a_reading_that_adds_words_is_not_confirmed_by_a_short_layer() -> None:
    layer = "International Developments"
    assert _evidence(layer, "# International Developments") == "PASS"
    assert (
        _evidence(layer, "# International Developments\n\nAn invented closing sentence.") != "PASS"
    )
    table = "| Region | Share |\n|---|---|\n| Alpha | beta |"
    assert _evidence(layer, f"# International Developments\n\n{table}") != "PASS"


def test_a_reading_that_adds_a_short_token_is_not_confirmed() -> None:
    layer = "International Developments"
    assert _evidence(layer, "# International Developments II") != "PASS"
    assert _evidence(layer, "# International Developments") == "PASS"


def test_markup_is_not_an_added_word() -> None:
    assert _evidence("International Developments", "## **International** _Developments_") == "PASS"


def _lcd_page(vertical_only: bool = False) -> fitz.Page:
    """The digits ``1741`` drawn as 60 pt strokes of width 1: larger than any glyph, so the
    glyph-sized residual test alone ignores them, and an axis-aligned stroke has an empty rect."""
    doc = fitz.open()
    page = _title_page(doc)
    top = 200.0
    if vertical_only:
        page.draw_line((100, top), (100, top + 60), width=1)
        return page
    for k, strokes in enumerate(("v", "hv", "v", "v")):
        x = 100 + 50 * k
        if "v" in strokes:
            page.draw_line((x, top), (x, top + 60), width=1)
        if "h" in strokes:
            page.draw_line((x, top), (x + 30, top), width=1)
    return page


def test_digits_drawn_as_strokes_are_never_accounted_for() -> None:
    assert analyze_page(_lcd_page()).ink_fully_accounted is False
    assert analyze_page(_lcd_page(vertical_only=True)).ink_fully_accounted is False


def test_a_long_horizontal_stroke_is_a_rule_and_is_allowed() -> None:
    doc = fitz.open()
    page = _title_page(doc)
    page.draw_line((100, 400), (460, 400), width=1)
    assert analyze_page(page).ink_fully_accounted is True


def test_the_run_side_records_the_strict_fact(tmp_path) -> None:
    """The orchestrator sets the fact on the page and writes it to the sidecar only when true:
    guards the wiring a verdict-level test cannot see (the setting, and the sidecar write). Drives
    the two real methods directly, so no page is routed and no provider is reached."""
    import json
    from unittest.mock import patch

    from test_gh1050_text_coverage import _pipeline

    from socr.core.state import DocumentHandle, DocumentState

    def run(name: str, *, raster: bool, check: bool = True):
        doc = fitz.open()
        page = _title_page(doc)
        if raster:
            page.insert_image(
                fitz.Rect(72, 200, 472, 460), pixmap=_raster_of(_grid(DECIMALS, 10), dpi=150)
            )
        pdf = tmp_path / f"{name}.pdf"
        doc.save(pdf)
        with patch.object(DocumentHandle, "__post_init__", lambda self: None):
            handle = DocumentHandle(path=pdf, page_count=1)
        state = DocumentState(handle=handle)
        state.pages[1] = PageState(page_num=1)
        pipeline = _pipeline(check_text_coverage=check)
        out_dir = tmp_path / f"out_{name}"
        pipeline._prepare_text_coverage(state, out_dir, set(), set())
        output = _read(f"# {TITLE}")
        state.pages[1].attempts.append(output)
        state.pages[1].best_output = output
        sidecar = json.loads(pipeline._flush_page_sidecar(state, 1, out_dir).read_text())
        return state.pages[1].text_ink_fully_accounted, sidecar

    on, rec_on = run("on", raster=False)
    raster, rec_raster = run("raster", raster=True)
    off, rec_off = run("off", raster=False, check=False)
    assert on is True and rec_on.get("text_ink_fully_accounted") is True
    assert raster is False and "text_ink_fully_accounted" not in rec_raster
    assert off is False and "text_ink_fully_accounted" not in rec_off


def test_an_empty_rect_line_is_measured_from_its_points() -> None:
    from socr.core.text_coverage import _drawing_extent

    page = _lcd_page(vertical_only=True)
    (drawing,) = page.get_drawings()
    assert drawing["rect"].is_empty
    x0, y0, x1, y1 = _drawing_extent(drawing)
    assert (y1 - y0) == pytest.approx(60)


def test_each_defence_catches_the_drawn_digits_on_its_own(monkeypatch) -> None:
    """The drawing rule and the residual-ink rule both fail the page; neither leans on the other."""
    from socr.core import text_coverage

    page = _lcd_page()
    assert analyze_page(page).ink_fully_accounted is False
    # drawings hidden: the residual ink (components far larger than a glyph) must still fail it
    monkeypatch.setattr(type(page), "get_drawings", lambda self: [])
    assert analyze_page(page).ink_fully_accounted is False
    monkeypatch.undo()
    # residual ink hidden: the drawings must still fail it
    real = text_coverage._residual_components

    def nothing_residual(ink, layer):
        labels, stats, resid, covered, glyph = real(ink, layer)
        return labels, stats, resid[:0], covered, glyph[:0]

    monkeypatch.setattr(text_coverage, "_residual_components", nothing_residual)
    assert analyze_page(page).ink_fully_accounted is False
