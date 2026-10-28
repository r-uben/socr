"""Replay projection must use the same paid-Gemini block predicate as live routing."""

from __future__ import annotations

from dataclasses import dataclass
import pytest

from socr.routing.cascade_policy import (
    paid_gemini_block_tag,
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
    invisible_text_over_raster: bool = False
    prior_ocr_layer: bool = False


FIXTURES: list[tuple[str, _FixturePage, str, str]] = [
    (
        "born_digital_sparse_caption",
        _FixturePage(1, True, 12, has_figures=False),
        "Word count 12 < 40",
        "sparse_word_count",
    ),
    (
        "scanned_figure_dominant",
        _FixturePage(
            2,
            False,
            18,
            has_figures=True,
            invisible_text_over_raster=False,
        ),
        "Word count 18 < 40",
        "sparse_word_count",
    ),
    (
        "scanned_not_image_dominant",
        _FixturePage(3, False, 120, has_figures=False),
        "Word count 35 < 40",
        "",
    ),
    (
        "prior_ocr_sparse",
        _FixturePage(4, False, 8, prior_ocr_layer=True),
        "Word count 8 < 40",
        "sparse_word_count",
    ),
    (
        "source_evidence_table",
        _FixturePage(5, False, 200, has_figures=True),
        "source_evidence_table: numeric tokens unsupported",
        "source_evidence_table",
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
    live_tag = paid_gemini_block_tag(judge_reason, live_signals, audit_min_words=audit_min_words)
    assert live_tag == expected_tag, name

    entry = {
        "page_num": page.page_num,
        "fingerprint": {"engine": "gemini"},
        "routing_meta": {
            "is_born_digital": page.is_born_digital,
            "word_count": page.word_count,
            "has_figures": page.has_figures or page.has_images,
            "invisible_text_over_raster": page.invisible_text_over_raster,
            "prior_ocr_layer": page.prior_ocr_layer,
        },
        "journal": [
            {"engine": "qwen", "accepted": False, "reason": judge_reason},
            {"engine": "gemini", "accepted": True, "reason": ""},
        ],
    }
    replay_signals = sparse_signals_from_manifest_entry(entry, entry["journal"])
    replay_tag = paid_gemini_block_tag(
        judge_reason, replay_signals, audit_min_words=audit_min_words
    )
    assert replay_tag == live_tag == expected_tag, name


def test_scanned_sparse_blocks_paid_gemini_in_route_page() -> None:
    from unittest.mock import MagicMock

    from socr.core.providers import PROFILE_GEMINI, PROFILE_MARKER, PROFILE_QWEN_LOCAL
    from socr.core.result import PageOutput, PageStatus
    from socr.pipeline.agentic import AcceptDecision, route_page
    from socr.routing.cascade_policy import SparsePageSignals

    ladder = [PROFILE_QWEN_LOCAL, PROFILE_MARKER, PROFILE_GEMINI]

    def run_provider(prof, page_num: int) -> PageOutput:
        return PageOutput(
            page_num=page_num,
            text="![fig](x.png)",
            status=PageStatus.SUCCESS,
            engine=prof.engine.value,
        )

    judge = MagicMock()
    judge.assess.side_effect = [
        AcceptDecision(accept=False, reason="Word count 15 < 40"),
        AcceptDecision(accept=False, reason="empty/error output"),
        AcceptDecision(accept=True, reason="must not run"),
    ]
    signals = SparsePageSignals(
        is_born_digital=False,
        native_word_count=15,
        has_figures=True,
        image_dominant=True,
    )
    decision = route_page(
        1,
        ladder,
        run_provider,
        judge,
        sparse_signals=signals,
        audit_min_words=40,
    )
    assert judge.assess.call_count == 2
    gemini_attempts = [a for a in decision.attempts if a.engine.value == "gemini"]
    assert len(gemini_attempts) == 1
    assert "sparse_word_count" in (gemini_attempts[0].reason or "")
