"""Replay projection must use the same paid-Gemini block predicate as live routing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from socr.routing.cascade_policy import (
    PAID_BLOCK_REASON_SOURCE_EVIDENCE,
    PAID_BLOCK_REASON_SPARSE_WORD_COUNT,
    paid_gemini_block_after_reject,
    sparse_signals_from_manifest_entry,
    sparse_signals_from_page_assessment,
)


@dataclass
class _FixturePage:
    page_num: int
    is_born_digital: bool
    word_count: int
    has_figures: bool = False
    has_images: bool = False
    has_chart_asset: bool = False


FIXTURES: list[tuple[str, _FixturePage, str, str]] = [
    (
        "born_digital_sparse_caption",
        _FixturePage(1, True, 12, has_figures=False),
        "Word count 12 < 40",
        PAID_BLOCK_REASON_SPARSE_WORD_COUNT,
    ),
    (
        "scanned_with_figure_not_sparse_paid_block",
        _FixturePage(2, False, 18, has_figures=True),
        "Word count 18 < 40",
        "",
    ),
    (
        "scanned_not_image_dominant",
        _FixturePage(3, False, 120, has_figures=False),
        "Word count 35 < 40",
        "",
    ),
    (
        "chart_asset_sparse_paid_block",
        _FixturePage(4, False, 8, has_chart_asset=True),
        "Word count 8 < 40",
        PAID_BLOCK_REASON_SPARSE_WORD_COUNT,
    ),
    (
        "source_evidence_table",
        _FixturePage(5, False, 200, has_figures=True),
        "source_evidence_table: numeric tokens unsupported",
        PAID_BLOCK_REASON_SOURCE_EVIDENCE,
    ),
    (
        "table_structure_still_escalates",
        _FixturePage(6, False, 200),
        "table_structure_failed: grid_shape",
        "",
    ),
]


@pytest.mark.parametrize(
    "name, page, judge_reason, expected_tag", FIXTURES, ids=[f[0] for f in FIXTURES]
)
def test_live_and_replay_share_block_tag(
    name: str,
    page: _FixturePage,
    judge_reason: str,
    expected_tag: str,
) -> None:
    audit_min_words = 40
    live_signals = sparse_signals_from_page_assessment(page)
    if page.has_chart_asset:
        from socr.routing.cascade_policy import SparsePageSignals

        live_signals = SparsePageSignals(
            is_born_digital=page.is_born_digital,
            native_word_count=page.word_count,
            has_figures=page.has_figures,
            has_chart_asset=True,
        )
    live_tag = paid_gemini_block_after_reject(
        judge_reason, live_signals, audit_min_words=audit_min_words
    )
    assert live_tag == expected_tag, name

    entry = {
        "page_num": page.page_num,
        "fingerprint": {
            "engine": "chart_asset" if page.has_chart_asset else "gemini",
        },
        "routing_meta": {
            "is_born_digital": page.is_born_digital,
            "word_count": page.word_count,
            "has_figures": page.has_figures or page.has_images,
            "has_chart_asset": page.has_chart_asset,
        },
        "journal": [
            {"engine": "qwen", "accepted": False, "reason": judge_reason},
            {"engine": "gemini", "accepted": True, "reason": ""},
        ],
    }
    replay_signals = sparse_signals_from_manifest_entry(entry, entry["journal"])
    replay_tag = paid_gemini_block_after_reject(
        judge_reason, replay_signals, audit_min_words=audit_min_words
    )
    assert replay_tag == live_tag == expected_tag, name


def test_scanned_photo_page_heuristic_judge_not_sparse(tmp_path: Path) -> None:
    """Scanned page with embedded image: judge must not accept low word count."""
    import fitz

    from socr.audit.heuristics import HeuristicsChecker
    from socr.core.born_digital import DocumentAssessment, PageAssessment
    from socr.pipeline.agentic import HeuristicPageJudge
    from socr.pipeline.orchestrator import UnifiedPipeline

    doc = fitz.open()
    page = doc.new_page(width=400, height=500)
    page.insert_text((50, 480), "Caption only", fontsize=10)
    from PIL import Image

    img_path = tmp_path / "photo.png"
    Image.new("RGB", (64, 64), color=(180, 120, 60)).save(img_path)
    rect = fitz.Rect(40, 40, 360, 420)
    page.insert_image(rect, filename=str(img_path))
    pdf = tmp_path / "scan_photo.pdf"
    doc.save(pdf)
    doc.close()

    pipeline = UnifiedPipeline.__new__(UnifiedPipeline)
    pipeline.config = type("C", (), {"audit_min_words": 40})()
    pipeline._last_assessment = DocumentAssessment(
        path=pdf,
        pages=[
            PageAssessment(
                page_num=1,
                is_born_digital=False,
                native_text="Caption only",
                confidence=0.5,
                word_count=2,
                has_figures=True,
                has_images=True,
            )
        ],
    )
    assert pipeline._sparse_page_ok(1) is False

    judge = HeuristicPageJudge(HeuristicsChecker(), sparse_ok=pipeline._sparse_page_ok)
    from socr.core.providers import PROFILE_QWEN_LOCAL
    from socr.core.result import PageOutput, PageStatus

    out = PageOutput(
        page_num=1,
        text="short",
        status=PageStatus.SUCCESS,
        engine="qwen",
    )
    decision = judge.assess(out, PROFILE_QWEN_LOCAL)
    assert decision.accept is False
