"""Model preamble loops with no page content (#JCT audit)."""

from __future__ import annotations

import pytest

pytest.importorskip("rich")

from socr.audit.boilerplate import (
    is_boilerplate_only_output,
    strip_leading_ocr_boilerplate,
    substantive_ocr_text,
)
from socr.audit.heuristics import HeuristicsChecker
from socr.core.config import EngineType
from socr.core.providers import provider_ladder
from socr.core.result import PageOutput, PageStatus
from socr.pipeline.agentic import HeuristicPageJudge, route_page

# Real JCT corpus sidecars (JCS-10-59 p2, JCX-16-81 p14), pinned verbatim.
JCS_P2_TEXT = (
    "The following is a transcription of the text in the image, rendered in clean Markdown format:\n\n"
    "---\n\n"
    "The following is a transcription of the text in the image, rendered in clean Markdown format:\n\n"
    "---\n\n"
    "The following is a transcription of the text in the image, rendered in clean Markdown format:\n\n"
    "---"
)
JCX_P14_TEXT = (
    "The following is a transcription of the text in the image into clean Markdown format, "
    "preserving the document structure:\n\n"
    "---\n\n"
    "The following is a transcription of the text in the image into clean Markdown format, "
    "preserving the document structure:\n\n"
    "---\n\n"
    "The following is a transcription of the text in the image into clean Markdown format, "
    "preserving the document structure:\n\n"
    "---"
)

_PREAMBLE_LINE = (
    "The following is a transcription of the text in the image, rendered in clean Markdown format:"
)


def test_jcs_fixture_is_boilerplate_only() -> None:
    assert is_boilerplate_only_output(JCS_P2_TEXT)
    assert not HeuristicsChecker().check(JCS_P2_TEXT).passed


def test_jcx_fixture_is_boilerplate_only() -> None:
    assert is_boilerplate_only_output(JCX_P14_TEXT)
    assert not HeuristicsChecker().check(JCX_P14_TEXT).passed


def test_strip_preamble_preserves_body() -> None:
    body = "Section 1\n\nOutlays total 100 billion in fiscal year 2026."
    raw = f"{_PREAMBLE_LINE}\n\n---\n\n{body}"
    assert strip_leading_ocr_boilerplate(raw) == body
    assert substantive_ocr_text(raw) == body
    assert not is_boilerplate_only_output(raw)
    assert HeuristicsChecker(min_word_count=5).check(strip_leading_ocr_boilerplate(raw)).passed


def test_legitimate_repeated_header_not_flagged() -> None:
    text = (
        "Budget Committee\n\n"
        "Budget Committee\n\n"
        "Outlays in fiscal year 2026 total one trillion dollars."
    )
    assert not is_boilerplate_only_output(text)
    assert HeuristicsChecker(min_word_count=5).check(text).passed


def test_heuristic_judge_strips_leading_preamble_on_accept() -> None:
    body = " ".join(["word"] * 40)
    raw = f"{_PREAMBLE_LINE}\n\n{body}"
    judge = HeuristicPageJudge(HeuristicsChecker(min_word_count=10))
    out = PageOutput(page_num=1, text=raw, status=PageStatus.SUCCESS)
    decision = judge.assess(out, provider=type("P", (), {"engine": EngineType.GLM})())
    assert decision.accept
    assert out.text == body


def test_route_page_escalates_past_boilerplate_to_next_engine() -> None:
    ladder = provider_ladder({EngineType.GLM, EngineType.GEMINI}, include_ineligible=True)
    good = " ".join(["policy"] * 40)

    def run(profile, page_num: int) -> PageOutput:
        engine = profile.engine
        text = JCS_P2_TEXT if engine == EngineType.GLM else good
        return PageOutput(
            page_num=page_num,
            text=text,
            status=PageStatus.SUCCESS,
            engine=engine.value,
        )

    judge = HeuristicPageJudge(HeuristicsChecker(min_word_count=10))
    decision = route_page(1, ladder[:2], run, judge)
    assert decision.accepted
    assert decision.final_output.engine == EngineType.GEMINI.value
    assert decision.final_output.text == good
    assert len(decision.attempts) == 2
    assert not decision.attempts[0].accepted
    assert decision.attempts[1].accepted
