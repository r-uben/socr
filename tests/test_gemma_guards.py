"""Hermetic tests for Gemma OCR output guards and crop engine wiring."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from socr.core.config import PipelineConfig
from socr.core.result import PageStatus
from socr.engines.gemma_cloud import GemmaCloudEngine
from socr.routing.gemma_guards import (
    apply_gemma_output_guards,
    has_image_url_loop,
    has_repeated_text_loop,
    truncate_to_token_budget,
)


def test_truncate_to_token_budget() -> None:
    long_text = "word " * 5000
    out = truncate_to_token_budget(long_text, max_tokens=100)
    assert "[truncated: gemma output token cap]" in out
    assert len(out) < len(long_text)


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


def test_gemma_process_pages_uses_crops_and_guards(tmp_path: Path) -> None:
    engine = GemmaCloudEngine()
    config = PipelineConfig(gemma_max_output_tokens=4096, gemma_crop_dpi=300)
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    with (
        patch(
            "socr.engines.gemma_cloud.render_page_half_crops",
            return_value=[("page_0001_L", tmp_path / "a.png"), ("page_0001_R", tmp_path / "b.png")],
        ),
        patch.object(
            engine,
            "_run_cli_on_images",
            side_effect=[
                ("Left column text.", 10, 20, ""),
                ("Right column text.", 11, 21, ""),
            ],
        ),
    ):
        (tmp_path / "a.png").write_bytes(b"x")
        (tmp_path / "b.png").write_bytes(b"y")
        outputs = engine.process_pages(pdf, [1], config)

    assert len(outputs) == 1
    assert outputs[0].status is PageStatus.SUCCESS
    assert outputs[0].engine == "gemma"
    assert "Left column" in outputs[0].text
    assert "Right column" in outputs[0].text
