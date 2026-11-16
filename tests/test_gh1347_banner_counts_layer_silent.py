"""#1347: the page banner counts layer-silent numbers and names contradictions.

A layer-silent dispute ("reading says B, not in the text layer") is counted in one phrase in
the shipped banner; a contradiction ("text layer says A, reading says B") is named. The
sidecar's clause reason keeps every value. The rule is by kind, not by length.
"""

from __future__ import annotations

import re
from dataclasses import replace
from types import SimpleNamespace

from socr.benchmark.scoreboard import _is_marked
from socr.core.manifest import _apply_verdict_marks, restate_legacy_banners
from socr.core.marks import Check, CheckResult, page_unchecked_clause_mark
from socr.core.result import PageOutput, PageStatus
from socr.core.state import PageState
from socr.core.verdict import (
    DAMAGED_LAYER_PREFIX,
    SHAPE_REFUSED_LAYER_PREFIX,
    PageVerdict,
    text_layer_numbers_clause,
)
from socr.tables.reading_agreement import (
    LAYER_SILENT_COUNT_TEMPLATE,
    summarise_disputes_for_banner,
)

LAYER = "Korea\n-.8\n-.9\n4.0"
#: layer number -> the minus-dropped writing the reading ships for it (one contradiction each)
DROPPED = [("-.8", "-8"), ("-.9", "-9")]


def _reading(silent: int, contradictions: int) -> str:
    """One layer number kept, ``contradictions`` minus-dropped writings of layer numbers,
    ``silent`` numbers the layer lacks."""
    cells = (
        [read if i < contradictions else ref for i, (ref, read) in enumerate(DROPPED)]
        + ["4.0"]
        + [str(1000 + 7 * i) for i in range(silent)]
    )
    return "| Korea | " + " | ".join(cells) + " |"


def _page(damaged: bool = True) -> PageState:
    p = PageState(page_num=1)
    p.own_layer_text = LAYER
    p.is_born_digital = not damaged
    return p


def _out(text: str) -> PageOutput:
    return PageOutput(page_num=1, text=text, status=PageStatus.SUCCESS, engine="qwen")


def _banner(out: PageOutput) -> str:
    return out.text.splitlines()[0]


def _shipped(silent: int, contradictions: int, damaged: bool = True):
    return _apply_verdict_marks(_out(_reading(silent, contradictions)), _page(damaged), 1)


def test_banner_counts_silent_and_names_contradictions_sidecar_keeps_all() -> None:
    n = 5
    out, verdict = _shipped(n, 1)
    banner = _banner(out)
    assert LAYER_SILENT_COUNT_TEMPLATE.format(count=n) in banner
    assert "text layer says -.8, reading says -8" in banner
    assert "not in the text layer" not in banner
    assert "1000" not in banner
    clause = next(c for c in verdict.clauses if c.check is Check.TEXT_LAYER_NUMBERS)
    for i in range(n):
        assert f"reading says {1000 + 7 * i}, not in the text layer" in clause.reason
    assert "text layer says -.8, reading says -8" in clause.reason
    assert clause.reason.count(" / ") + 1 == n + 1


def test_only_silent_entries() -> None:
    out, verdict = _shipped(4, 0)
    banner = _banner(out)
    assert banner.endswith(DAMAGED_LAYER_PREFIX + LAYER_SILENT_COUNT_TEMPLATE.format(count=4) + "]")
    assert "text layer says" not in banner
    assert _is_marked(out.text)


def test_only_contradictions_is_unchanged() -> None:
    clause = text_layer_numbers_clause(_out(_reading(0, 2)), _page())
    assert clause is not None
    assert summarise_disputes_for_banner(clause.reason) == clause.reason
    assert "text layer says -.8, reading says -8" in clause.mark(1)
    assert "text layer says -.9, reading says -9" in clause.mark(1)
    assert "number(s) in the reading" not in clause.mark(1)


def test_failing_layer_case_counts_too() -> None:
    """The FAIL case (a clean born-digital layer) takes the same treatment."""
    out, _ = _shipped(3, 1, damaged=False)
    banner = _banner(out)
    assert "FAIL" in banner
    assert LAYER_SILENT_COUNT_TEMPLATE.format(count=3) in banner
    assert "not in the text layer" not in banner


def test_banner_is_independent_of_the_number_of_silent_entries() -> None:
    """No length threshold: the banner differs only in the count."""
    small = _banner(_shipped(2, 1)[0])
    large = _banner(_shipped(40, 1)[0])
    assert re.sub(r"\d+ number", "N number", small) == re.sub(r"\d+ number", "N number", large)


def _old_format_page(silent: int, contradictions: int, damaged: bool = True, page=None):
    """The page as a run from before #1347 stored it: the clause's line rendered from the FULL
    reason, the body the live text's. Returns ``(stored, live, verdict)``."""
    page = page or _page(damaged)
    live, verdict = _apply_verdict_marks(_out(_reading(silent, contradictions)), page, 1)
    clause = next(c for c in verdict.clauses if c.check is Check.TEXT_LAYER_NUMBERS)
    old = page_unchecked_clause_mark(1, clause.check, clause.result, clause.scope, clause.reason)
    assert old != _banner(live), "premise: the two renderings differ"
    stored = replace(live, text=old + "\n" + live.text.split("\n", 1)[1])
    return stored, live, verdict


def test_a_stored_old_format_page_refinalizes_to_the_live_banner() -> None:
    stored, live, verdict = _old_format_page(6, 1)
    assert "not in the text layer" in _banner(stored)
    for published in (None, PageVerdict.from_dict(verdict.to_dict())):
        again, _ = _apply_verdict_marks(stored, _page(), 1, published=published)
        assert again.text == live.text
        assert "not in the text layer" not in _banner(again)
        assert _is_marked(again.text)
    twice, _ = _apply_verdict_marks(again, _page(), 1, published=verdict)
    assert twice.text == again.text


def test_the_refinalize_tool_rewrites_a_stored_old_format_page() -> None:
    """The tool leaves a page whose verdict is unchanged alone UNLESS its banner is the old long
    form; then it writes the live text. A page already in the new form is still left alone."""
    from socr.pipeline.refinalize_own_text import refinalize_page

    stored, live, verdict = _old_format_page(6, 1)

    def sidecar(out):
        return {
            "page_num": 1,
            "terminal": True,
            "status": out.status.value,
            "winning_output": out.to_dict(),
            "verdict": verdict.to_dict(),
        }

    pa = SimpleNamespace(
        own_layer_text=LAYER,
        is_born_digital=False,
        control_byte_digit_hits=0,
        control_byte_scan_failed=False,
        minus_as_digit_hits=0,
        minus_as_digit_scan_failed=False,
        has_unrecovered_symbol_glyphs=False,
        has_encoding_hygiene_suspect=False,
        invisible_text_over_raster=False,
        invisible_text_scan_failed=False,
        prior_ocr_layer=False,
        prior_ocr_layer_scan_failed=False,
    )
    assert refinalize_page(sidecar(live), pa) is None
    done = refinalize_page(sidecar(stored), pa)
    assert done is not None
    assert done[0].text == live.text


def test_a_clause_a_different_reason_wrote_is_not_restated() -> None:
    """Only the exact long-form render of the clause's own reason is rewritten."""
    stored, live, verdict = _old_format_page(6, 1)
    foreign = page_unchecked_clause_mark(
        1, Check.TEXT_LAYER_NUMBERS, CheckResult.UNKNOWN, "page", "a reason a person wrote"
    )
    text = foreign + "\n" + live.text.split("\n", 1)[1]
    assert restate_legacy_banners(text, verdict, 1) == text


def test_one_layer_silent_entry_is_counted_not_listed() -> None:
    """No length threshold: a single layer-silent entry takes the same form as forty."""
    out, _ = _shipped(1, 1)
    banner = _banner(out)
    assert LAYER_SILENT_COUNT_TEMPLATE.format(count=1) in banner
    assert "not in the text layer" not in banner
    assert "1000" not in banner


def test_shape_refused_and_cosmetic_layers_count_too() -> None:
    shape = _page()
    shape.layer_refused_for_shape_alone = True
    cosmetic = _page(damaged=False)
    cosmetic.has_encoding_hygiene_suspect = True
    for page, prefix in ((shape, SHAPE_REFUSED_LAYER_PREFIX), (cosmetic, DAMAGED_LAYER_PREFIX)):
        out, _ = _apply_verdict_marks(_out(_reading(5, 1)), page, 1)
        banner = _banner(out)
        assert prefix in banner
        assert "text layer says -.8, reading says -8" in banner
        assert LAYER_SILENT_COUNT_TEMPLATE.format(count=5) in banner
        assert "not in the text layer" not in banner
        stored, live, _v = _old_format_page(5, 1, page=page)
        again, _ = _apply_verdict_marks(stored, page, 1)
        assert again.text == live.text


def test_record_form_is_lossless() -> None:
    clause = text_layer_numbers_clause(_out(_reading(3, 1)), _page())
    assert clause is not None
    assert clause.reason.startswith(DAMAGED_LAYER_PREFIX)
    assert clause.reason.count("not in the text layer") == 3


def test_unparseable_reason_is_returned_unchanged() -> None:
    for reason in ("", "no number to check", "reading says 1, not in the text layer / free text"):
        assert summarise_disputes_for_banner(reason) == reason
