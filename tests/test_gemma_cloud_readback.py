"""Gemma half-crop aggregated read-back (qwen-ocr ``## Page N`` shape)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from ocr_output_contract import assemble_pages, doc_dir_for, markdown_path_for, relative_key
from PIL import Image

from socr.core.config import PipelineConfig
from socr.core.result import PageStatus
from socr.engines.gemma_cloud import GemmaCloudEngine

pytest.importorskip("fitz")


def _write_min_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(255, 255, 255)).save(path)


def _gemma_process_subprocess(
    page_text_by_index: dict[int, str], *, section_count: int | None = None
):
    """Fake ``qwen-ocr process <dir> -o <out>`` writing one aggregated markdown doc."""

    def _run(cmd, *args, **kwargs):
        images_dir = Path(cmd[2])
        out_dir = Path(cmd[cmd.index("-o") + 1])
        n_imgs = len(list(images_dir.glob("*.png")))
        n = section_count if section_count is not None else n_imgs
        pages = [page_text_by_index.get(i + 1, f"section {i + 1}") for i in range(n)]
        rel_key = relative_key(images_dir, images_dir.parent)
        doc_dir = doc_dir_for(out_dir, rel_key)
        doc_dir.mkdir(parents=True, exist_ok=True)
        md_path = markdown_path_for(doc_dir, rel_key)
        md_path.write_text(assemble_pages(pages), encoding="utf-8")
        result = type("R", (), {})()
        result.returncode = 0
        result.stdout = ""
        result.stderr = "prompt_tokens: 11\ncompletion_tokens: 22"
        return result

    return _run


def test_half_crop_single_section_read_as_success(tmp_path: Path) -> None:
    """Regression: empty page map used to yield mismatch (1 section vs 0 images)."""
    engine = GemmaCloudEngine()
    config = PipelineConfig(enable_gemma_cloud=True, quiet=True)
    half_dir = tmp_path / "half_gemma_p1_clip0"
    half_dir.mkdir()
    stem = "gemma_p1_clip0"
    _write_min_png(half_dir / f"{stem}.png")

    expected = "Revenue grew 12% year over year in the segment."
    fake = _gemma_process_subprocess({1: expected})

    with patch("socr.engines.gemma_cloud.subprocess.run", side_effect=fake):
        text, inp, out, err = engine._run_cli_on_images(half_dir, config, 60.0)

    assert err == ""
    assert expected in text
    assert inp == 11
    assert out == 22


def test_half_crop_count_mismatch_still_fails_closed(tmp_path: Path) -> None:
    engine = GemmaCloudEngine()
    config = PipelineConfig(enable_gemma_cloud=True, quiet=True)
    half_dir = tmp_path / "half_only"
    half_dir.mkdir()
    _write_min_png(half_dir / "crop_a.png")

    fake = _gemma_process_subprocess({1: "one", 2: "two"}, section_count=2)
    with patch("socr.engines.gemma_cloud.subprocess.run", side_effect=fake):
        text, _inp, _out, err = engine._run_cli_on_images(half_dir, config, 60.0)

    assert not text.strip()
    assert "mismatch" in err.lower()
    assert "2 section" in err.lower()
    assert "1 image" in err.lower()


def test_process_pages_single_crop_success(tmp_path: Path) -> None:
    engine = GemmaCloudEngine()
    config = PipelineConfig(
        enable_gemma_cloud=True,
        gemma_max_output_tokens=4096,
        gemma_crop_dpi=72,
        quiet=True,
    )
    pdf = tmp_path / "doc.pdf"
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=400, height=400)
    page.insert_text((40, 80), "Single column paragraph for Gemma.", fontsize=11)
    doc.save(pdf)
    doc.close()

    body = "Single column paragraph for Gemma."
    with patch.object(
        engine,
        "_run_cli_on_images",
        return_value=(body, 5, 6, ""),
    ):
        outputs = engine.process_pages(pdf, [1], config)

    assert len(outputs) == 1
    assert outputs[0].status is PageStatus.SUCCESS
    assert body in outputs[0].text
