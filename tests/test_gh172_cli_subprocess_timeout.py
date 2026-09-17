"""GH-172, CLI half: the CLI-engine subprocess kill is bound by the SOFT
per-provider deadline, not the whole-document timeout.

``engines/base.py`` already runs CLI engines as killable subprocesses
(``subprocess.run(..., timeout=...)``); it was bounded by
``config.timeout`` (the 1800s whole-document budget) rather than the
per-page deadline the agentic loop already reports as "timed out" in its own
audit trail. A wedged CLI (e.g. a hung ``qwen-ocr``) used to survive up to
1800s after ``route_page`` had already abandoned its wrapper thread and moved
on to the next page. This pins the tightened bound.

Hermetic: ``subprocess.run`` is stubbed; no real CLI, network, GPU, or
Ollama daemon is touched.
"""

from __future__ import annotations

import subprocess

import fitz
import pytest

from socr.core.config import PipelineConfig
from socr.engines.qwen import QwenEngine


def _make_pdf(path, n_pages=1):
    doc = fitz.open()
    for i in range(n_pages):
        page = doc.new_page()
        page.insert_text((72, 72), f"native text page {i + 1}")
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture
def stub_subprocess_run(monkeypatch):
    """Replace subprocess.run with one that records `timeout=` and raises TimeoutExpired."""
    calls: list[float | None] = []

    def _fake_run(cmd, **kwargs):
        calls.append(kwargs.get("timeout"))
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout"))

    monkeypatch.setattr("socr.engines.base.subprocess.run", _fake_run)
    return calls


def test_process_pages_defaults_to_document_timeout(tmp_path, stub_subprocess_run):
    """No override: existing whole-document behaviour is unchanged."""
    pdf_path = _make_pdf(tmp_path / "doc.pdf")
    config = PipelineConfig(timeout=1800.0)
    engine = QwenEngine()

    outputs = engine.process_pages(pdf_path=pdf_path, page_nums=[1], config=config)

    assert stub_subprocess_run == [1800.0]
    assert outputs[0].error == "Timeout after 1800.0s"


def test_process_pages_honours_a_tighter_soft_deadline(tmp_path, stub_subprocess_run):
    """GH-172: an explicit subprocess_timeout overrides config.timeout."""
    pdf_path = _make_pdf(tmp_path / "doc.pdf")
    config = PipelineConfig(timeout=1800.0)
    engine = QwenEngine()

    outputs = engine.process_pages(
        pdf_path=pdf_path, page_nums=[1], config=config, subprocess_timeout=42.0
    )

    assert stub_subprocess_run == [42.0]
    assert outputs[0].error == "Timeout after 42.0s"
