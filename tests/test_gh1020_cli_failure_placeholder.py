"""GH-1020: a CLI per-page failure placeholder is a failure, not a SUCCESS page.

qwen-ocr-cli exits 1 and writes ``*[OCR failed for page N]*`` for a page whose backend
call errored (qwen_ocr/processor.py::_ocr_pages). socr used to read that back as
SUCCESS with ``audit_passed=True``, spend a judge call on the marker, and could ship it.

**Pin a DIFFERENCE, not a value** (CLAUDE.md, #257). The agentic test runs the same
production path twice in one process, changing ONLY what the rung-one CLI wrote, and
asserts the judge saw the real text in one leg and never saw the marker in the other,
and that the next rung ran only in the failure leg. No absolute page/document status
is pinned, so the CI no-provider divergence cannot reach the assertions.

Hermetic: ladder patched, judge model "", ``get_engine`` stubbed, subprocess stubbed.
"""

from __future__ import annotations

import pathlib
from typing import Any
from unittest.mock import MagicMock

import pytest

fitz = pytest.importorskip("fitz", reason="PyMuPDF not installed")

from ocr_output_contract import (  # noqa: E402
    assemble_pages,
    doc_dir_for,
    markdown_path_for,
    relative_key,
)

from socr.core.config import EngineType, PipelineConfig  # noqa: E402
from socr.core.providers import PROFILE_GEMINI, PROFILE_QWEN_LOCAL  # noqa: E402
from socr.core.result import FailureMode, PageOutput, PageStatus  # noqa: E402
from socr.engines.base import BaseEngine, is_cli_failure_placeholder  # noqa: E402
from socr.pipeline.agentic import AcceptDecision  # noqa: E402
from socr.pipeline.orchestrator import UnifiedPipeline  # noqa: E402

PLACEHOLDER = "*[OCR failed for page 1]*"
REAL_TEXT = "Estimated coefficient 0.082 significant"


class _QwenLikeEngine(BaseEngine):
    """Real BaseEngine read-back; only the subprocess boundary is faked."""

    @property
    def name(self) -> str:
        return "qwen"

    @property
    def cli_command(self) -> str:
        return "qwen-ocr"

    def _build_command(self, pdf_path, output_dir, config):
        return ["qwen-ocr", str(pdf_path), "-o", str(output_dir)]


def _fake_run(text: str, returncode: int):
    def _run(cmd, *args, **kwargs):
        if "-o" not in cmd:  # e.g. a ``--version`` availability probe
            probe = MagicMock()
            probe.returncode = 0
            probe.stdout = probe.stderr = ""
            return probe
        images_dir = pathlib.Path(cmd[1])
        out_dir = pathlib.Path(cmd[cmd.index("-o") + 1])
        rel_key = relative_key(images_dir, images_dir.parent)
        doc_dir = doc_dir_for(out_dir, rel_key)
        doc_dir.mkdir(parents=True, exist_ok=True)
        markdown_path_for(doc_dir, rel_key).write_text(assemble_pages([text]), encoding="utf-8")
        result = MagicMock()
        result.returncode = returncode
        result.stdout = ""
        result.stderr = "Ollama 500" if returncode else ""
        return result

    return _run


def _pdf(path: pathlib.Path) -> pathlib.Path:
    doc = fitz.open()
    page = doc.new_page()
    y = 80
    for _ in range(14):
        page.insert_text((60, y), REAL_TEXT, fontsize=9)
        y += 16
    doc.save(str(path))
    doc.close()
    return path


# --- engine level -----------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "*[OCR failed for page 1]*",
        "*[OCR failed for page 17]*\n",
        "\n  *[OCR Failed]*  \n",
    ],
)
def test_exact_cli_markers_are_recognised(text: str) -> None:
    assert is_cli_failure_placeholder(text)


@pytest.mark.parametrize(
    "text",
    [
        "",
        None,
        "Real text.\n\n*[OCR failed for page 1]*",
        "The CLI prints *[OCR failed for page 1]* on error.",
        "*[OCR failed]*",
        "[OCR failed for page 1]",
    ],
)
def test_legitimate_text_is_never_a_marker(text: str | None) -> None:
    assert not is_cli_failure_placeholder(text)


def _engine_page(tmp_path, monkeypatch, text: str, returncode: int) -> PageOutput:
    monkeypatch.setattr("socr.engines.base.subprocess.run", _fake_run(text, returncode))
    pdf = _pdf(tmp_path / "d.pdf")
    return _QwenLikeEngine().process_pages(pdf, [1], PipelineConfig(timeout=30))[0]


def test_process_pages_placeholder_differs_from_real_text(tmp_path, monkeypatch) -> None:
    ok = _engine_page(tmp_path, monkeypatch, REAL_TEXT, 0)
    bad = _engine_page(tmp_path, monkeypatch, PLACEHOLDER, 1)

    assert ok.status == PageStatus.SUCCESS and ok.audit_passed
    assert bad.status == PageStatus.ERROR
    assert bad.failure_mode == FailureMode.CLI_ERROR
    assert not bad.audit_passed
    assert PLACEHOLDER not in (bad.text or "")
    assert "Ollama 500" in (bad.error or "")


def test_placeholder_with_exit_zero_is_still_a_failure(tmp_path, monkeypatch) -> None:
    bad = _engine_page(tmp_path, monkeypatch, PLACEHOLDER, 0)
    assert bad.status == PageStatus.ERROR
    assert bad.failure_mode == FailureMode.CLI_ERROR


# --- agentic loop: no judge call, next rung runs ----------------------------------


class _RecordingJudge:
    """Accepts every SUCCESS page, so only the engine read-back can keep the marker out.

    Mirrors the shipped judges, which refuse a non-SUCCESS or empty output without a
    model call (agentic.py: "empty/error output"). ``seen`` records the text of every
    output the judge was handed.
    """

    def __init__(self) -> None:
        self.seen: list[str] = []

    def assess(self, output: Any, provider: Any) -> AcceptDecision:
        self.seen.append(output.text or "")
        if output.status != PageStatus.SUCCESS or not (output.text or "").strip():
            return AcceptDecision(accept=False, reason="empty/error output")
        return AcceptDecision(accept=True, reason="stub accepts all")


class _NextRung:
    name = "gemini"

    def __init__(self) -> None:
        self.calls = 0

    def is_available(self) -> bool:
        return True

    def process_pages(self, pdf_path, page_nums, config, dpi, **_kw) -> list[PageOutput]:
        self.calls += 1
        return [
            PageOutput(page_num=n, text="rung two text", status=PageStatus.SUCCESS, engine="gemini")
            for n in page_nums
        ]


def _run_agentic(tmp_path, monkeypatch, leg: str, cli_text: str, returncode: int):
    from socr.pipeline import orchestrator as orch

    monkeypatch.setattr("socr.engines.base.subprocess.run", _fake_run(cli_text, returncode))
    qwen, gemini, judge = _QwenLikeEngine(), _NextRung(), _RecordingJudge()
    monkeypatch.setattr(orch, "get_engine", lambda et: qwen if et == EngineType.QWEN else gemini)
    pipe = UnifiedPipeline(
        PipelineConfig(
            agentic=True,
            quiet=True,
            primary_engine=EngineType.QWEN,
            local_engine=EngineType.QWEN,
            enabled_engines=[EngineType.QWEN, EngineType.GEMINI],
            write_manifest=False,
            judge_backend="heuristic",
            dual_pass_tables=False,
            detect_equations=False,
            save_figures=False,
        )
    )
    pipe._available_engines_for_agentic = lambda: [PROFILE_QWEN_LOCAL, PROFILE_GEMINI]
    pipe._build_page_judge = lambda state: judge
    pipe._resolve_crop_vlm_model = lambda: None
    pipe._resolve_judge_model = lambda *a, **k: ""
    _detect = pipe.bd_detector.detect

    def _needs_ocr(path):
        assessment = _detect(path)
        assessment.pages[0].needs_ocr_enhancement = True
        return assessment

    pipe.bd_detector.detect = _needs_ocr
    out = tmp_path / f"out-{leg}"
    pipe.process(_pdf(tmp_path / f"{leg}.pdf"), output_dir=out)
    written = "\n".join(p.read_text() for p in out.rglob("*.md"))
    return judge, gemini, written


def test_placeholder_is_not_judged_and_next_rung_runs(tmp_path, monkeypatch) -> None:
    ok_judge, ok_next, ok_md = _run_agentic(tmp_path, monkeypatch, "ok", REAL_TEXT, 0)
    bad_judge, bad_next, bad_md = _run_agentic(tmp_path, monkeypatch, "bad", PLACEHOLDER, 1)

    # Control leg: rung one's real text was judged, the second rung never needed.
    assert any(REAL_TEXT in t for t in ok_judge.seen)
    assert ok_next.calls == 0
    assert "OCR failed" not in ok_md

    # Failure leg: the marker never reached the judge or the output, rung two ran.
    assert not any("OCR failed" in t for t in bad_judge.seen), bad_judge.seen
    assert "OCR failed" not in bad_md
    assert bad_next.calls == 1
