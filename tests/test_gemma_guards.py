"""Hermetic tests for Gemma OCR output guards and crop engine wiring."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import fitz
import pytest

from socr.core.config import PipelineConfig
from socr.core.result import FailureMode, PageStatus
from socr.engines.gemma_cloud import GemmaCloudEngine
from socr.routing.cascade_policy import parse_token_usage
from socr.routing.gemma_guards import (
    apply_gemma_output_guards,
    has_image_url_loop,
    has_repeated_text_loop,
    truncate_to_token_budget,
)


def test_truncate_to_token_budget() -> None:
    long_text = "word " * 5000
    out, truncated = truncate_to_token_budget(long_text, max_tokens=100)
    assert truncated is True
    assert "[truncated: gemma output token cap]" in out
    assert len(out) < len(long_text)


def test_truncation_is_rejected_by_guards() -> None:
    long_text = "word " * 5000
    out = apply_gemma_output_guards(long_text, page_num=1, max_output_tokens=100)
    assert out.status is PageStatus.ERROR
    assert out.failure_mode is FailureMode.TRUNCATED
    assert out.audit_passed is False


def test_rejects_image_url_loop() -> None:
    url = "https://example.com/x.png"
    text = "\n".join(f"![a]({url})" for _ in range(5))
    assert has_image_url_loop(text)
    out = apply_gemma_output_guards(text, page_num=1, max_output_tokens=4096)
    assert out.status is PageStatus.ERROR
    assert "image URL loop" in (out.error or "")


def test_rejects_repeated_text_loop() -> None:
    text = "\n".join(["same line"] * 4)
    assert has_repeated_text_loop(text)
    out = apply_gemma_output_guards(text, page_num=2, max_output_tokens=4096)
    assert out.status is PageStatus.ERROR


def test_equations_flagged_unverified() -> None:
    text = "Energy $E=mc^2$ here."
    out = apply_gemma_output_guards(text, page_num=3, max_output_tokens=4096)
    assert out.status is PageStatus.SUCCESS
    assert "gemma equations unverified" in out.text
    assert out.audit_notes


def test_parse_token_usage_ignores_total_tokens() -> None:
    stderr = "prompt_tokens: 100\ntotal_tokens: 9999\ncompletion_tokens: 42"
    inp, out = parse_token_usage(stderr, "")
    assert inp == 100
    assert out == 42


def test_gemma_process_pages_one_crop_fails(tmp_path: Path) -> None:
    engine = GemmaCloudEngine()
    config = PipelineConfig(
        enable_gemma_cloud=True,
        gemma_max_output_tokens=4096,
        gemma_crop_dpi=150,
    )
    pdf = tmp_path / "doc.pdf"
    doc = fitz.open()
    page = doc.new_page(width=400, height=600)
    page.insert_text((50, 80), "Upper block " * 8, fontsize=11)
    page.insert_text((50, 420), "Lower block " * 8, fontsize=11)
    doc.save(pdf)
    doc.close()

    with patch.object(
        engine,
        "_run_cli_on_images",
        side_effect=[
            ("left ok", 10, 20, ""),
            ("", None, None, "CLI exited 1: boom"),
        ],
    ):
        outputs = engine.process_pages(pdf, [1], config)

    assert len(outputs) == 1
    assert outputs[0].status is PageStatus.ERROR
    assert outputs[0].audit_passed is False
    assert "boom" in (outputs[0].error or "")


def test_gemma_process_pages_real_render_one_half_errors(tmp_path: Path) -> None:
    """Drive real ``process_pages`` (real crop render); stub only CLI."""
    engine = GemmaCloudEngine()
    config = PipelineConfig(
        enable_gemma_cloud=True,
        gemma_max_output_tokens=4096,
        gemma_crop_dpi=72,
    )
    pdf = tmp_path / "doc.pdf"
    doc = fitz.open()
    page = doc.new_page(width=400, height=600)
    page.insert_text((50, 100), "Top section text here.", fontsize=12)
    page.insert_text((50, 400), "Bottom section continues.", fontsize=12)
    doc.save(pdf)
    doc.close()

    calls = {"n": 0}

    def _cli_side_effect(*_args, **_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return "Top section text here.", 5, 6, ""
        return "", None, None, "CLI exited 1: second crop failed"

    with patch.object(engine, "_run_cli_on_images", side_effect=_cli_side_effect):
        outputs = engine.process_pages(pdf, [1], config)

    assert len(outputs) == 1
    assert outputs[0].status is PageStatus.ERROR
    assert outputs[0].audit_passed is False
    assert calls["n"] == 2
