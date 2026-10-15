"""An empty model answer is a failed call: socr repeats it once and counts it.

Rorqual job 23180840, Coca-Cola 2018 p.60: Gemini answered HTTP 200 with no text,
gemini-ocr-cli logged ``Error processing .../page_0060.png: Empty response`` and wrote
its failure placeholder, socr read that as an ordinary CLI error, made no second call,
and the page shipped the fail-closed floor. An answer of only whitespace was worse: it
read back as a SUCCESS page with no text.

Pinned as DIFFERENCES (CLAUDE.md, #257): the same read-back with only the stderr line
or the answer's bytes changed, and the same engine runner with only the first answer
changed. Hermetic: the subprocess and the engine are stubbed; no provider is reached.
"""

from __future__ import annotations

import pathlib
from unittest.mock import MagicMock

import pytest

fitz = pytest.importorskip("fitz", reason="PyMuPDF not installed")

from socr.core.audit_log import PROVIDER_EMPTY_RESPONSE_KIND  # noqa: E402
from socr.core.config import EngineType, PipelineConfig  # noqa: E402
from socr.core.document import DocumentHandle  # noqa: E402
from socr.core.result import FailureMode, PageOutput, PageStatus  # noqa: E402
from socr.core.state import DocumentState, PageState  # noqa: E402
from socr.engines.base import BaseEngine  # noqa: E402
from socr.pipeline.orchestrator import UnifiedPipeline  # noqa: E402

REAL_TEXT = "Scope 3 emissions 412,000 tonnes"
EMPTY_STDERR = (
    "2026-10-14 10:14:57 - gemini_ocr.processor - ERROR - Error processing "
    "/tmp/x/images/page_0001.png: Empty response: finish_reason"
)
OTHER_STDERR = (
    "2026-10-14 10:14:57 - gemini_ocr.processor - ERROR - Error processing "
    "/tmp/x/images/page_0001.png: 503 UNAVAILABLE"
)


class _GeminiLikeEngine(BaseEngine):
    """Real BaseEngine read-back; only the subprocess boundary is faked."""

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def cli_command(self) -> str:
        return "gemini-ocr"

    def _build_command(self, pdf_path, output_dir, config):
        return ["gemini-ocr", str(pdf_path), "-o", str(output_dir)]


def _fake_run(text: str | None, returncode: int, stderr: str):
    """Write ``text`` where gemini-ocr writes a page (``<out>/<stem>/<stem>.md``)."""

    def _run(cmd, *args, **kwargs):
        out_dir = pathlib.Path(cmd[cmd.index("-o") + 1])
        for image in sorted(pathlib.Path(cmd[1]).glob("*.png")):
            if text is not None:
                page_dir = out_dir / image.stem
                page_dir.mkdir(parents=True, exist_ok=True)
                (page_dir / f"{image.stem}.md").write_text(text, encoding="utf-8")
        result = MagicMock()
        result.returncode = returncode
        result.stdout = ""
        result.stderr = stderr
        return result

    return _run


def _pdf(path: pathlib.Path) -> pathlib.Path:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((60, 80), REAL_TEXT, fontsize=9)
    doc.save(str(path))
    doc.close()
    return path


def _read_back(tmp_path, monkeypatch, text: str | None, returncode: int, stderr: str):
    monkeypatch.setattr("socr.engines.base.subprocess.run", _fake_run(text, returncode, stderr))
    pdf = _pdf(tmp_path / "d.pdf")
    return _GeminiLikeEngine().process_pages(pdf, [1], PipelineConfig(timeout=30))[0]


def test_whitespace_answer_is_an_empty_response_not_a_success(tmp_path, monkeypatch) -> None:
    real = _read_back(tmp_path, monkeypatch, REAL_TEXT, 0, "")
    blank = _read_back(tmp_path, monkeypatch, " \n\t\n", 0, "")

    assert real.status == PageStatus.SUCCESS
    assert blank.status == PageStatus.ERROR
    assert blank.failure_mode is FailureMode.EMPTY_RESPONSE
    assert not blank.audit_passed


def test_cli_reported_empty_answer_differs_from_other_cli_errors(tmp_path, monkeypatch) -> None:
    """Same placeholder and exit code; only the stderr line for the page differs."""
    placeholder = "*[OCR failed for page 1]*"
    empty = _read_back(tmp_path, monkeypatch, placeholder, 1, EMPTY_STDERR)
    other = _read_back(tmp_path, monkeypatch, placeholder, 1, OTHER_STDERR)

    assert empty.failure_mode is FailureMode.EMPTY_RESPONSE
    assert other.failure_mode is FailureMode.CLI_ERROR
    assert empty.status == other.status == PageStatus.ERROR


def test_a_refusal_reported_as_empty_response_is_not_an_empty_answer(tmp_path, monkeypatch) -> None:
    """gemini-ocr logs a RECITATION refusal with the same words; it stays a refusal."""
    refusal = EMPTY_STDERR + "=FinishReason.RECITATION, len(parts)=0"
    stop = EMPTY_STDERR + "=FinishReason.STOP, len(parts)=0"
    refused = _read_back(tmp_path, monkeypatch, None, 1, refusal)
    empty = _read_back(tmp_path, monkeypatch, None, 1, stop)
    assert refused.failure_mode is FailureMode.RECITATION
    assert empty.failure_mode is FailureMode.EMPTY_RESPONSE


def test_empty_answer_naming_another_page_does_not_mark_this_one(tmp_path, monkeypatch) -> None:
    stderr = EMPTY_STDERR.replace("page_0001", "page_0002")
    out = _read_back(tmp_path, monkeypatch, None, 1, stderr)
    assert out.failure_mode is FailureMode.CLI_ERROR


# --- the engine runner: one more call, counted ------------------------------------


class _ScriptedEngine:
    """Answers each ``process_pages`` call with the next scripted failure mode."""

    name = "gemini"

    def __init__(self, script: list[FailureMode]) -> None:
        self.script = list(script)
        self.calls: list[tuple[list[int], float | None]] = []

    def is_available(self) -> bool:
        return True

    def process_pages(self, pdf_path, page_nums, config, dpi=200, subprocess_timeout=None):
        self.calls.append((list(page_nums), subprocess_timeout))
        mode = self.script.pop(0)
        if mode is FailureMode.NONE:
            return [
                PageOutput(page_num=p, text=REAL_TEXT, status=PageStatus.SUCCESS, engine=self.name)
                for p in page_nums
            ]
        return [
            PageOutput(page_num=p, status=PageStatus.ERROR, engine=self.name, failure_mode=mode)
            for p in page_nums
        ]


def _run_engine(tmp_path, monkeypatch, script, subprocess_timeout_sec=None):
    engine = _ScriptedEngine(script)
    monkeypatch.setattr("socr.pipeline.orchestrator.get_engine", lambda _t: engine)
    pdf = _pdf(tmp_path / "r.pdf")
    state = DocumentState(handle=DocumentHandle(path=pdf, page_count=1))
    state.pages[1] = PageState(page_num=1)
    pipeline = UnifiedPipeline(PipelineConfig(quiet=True, timeout=30))
    out = pipeline._run_engine_on_pages(
        state,
        [1],
        set(),
        EngineType.GEMINI,
        "agentic",
        subprocess_timeout_sec=subprocess_timeout_sec,
    )
    counted = [e for e in state.events if e.kind == PROVIDER_EMPTY_RESPONSE_KIND]
    return engine, out[0], counted


def test_empty_answer_is_retried_once_and_counted(tmp_path, monkeypatch) -> None:
    empty_engine, empty_out, empty_events = _run_engine(
        tmp_path, monkeypatch, [FailureMode.EMPTY_RESPONSE, FailureMode.NONE]
    )
    error_engine, error_out, error_events = _run_engine(
        tmp_path, monkeypatch, [FailureMode.CLI_ERROR, FailureMode.NONE]
    )

    # Only the empty answer earns a second call, and the second call's text ships.
    assert len(empty_engine.calls) == 2
    assert len(error_engine.calls) == 1
    assert empty_out.status == PageStatus.SUCCESS
    assert error_out.status == PageStatus.ERROR
    assert [e.data["retried"] for e in empty_events] == [True]
    assert error_events == []


def test_second_empty_answer_is_counted_and_not_retried_again(tmp_path, monkeypatch) -> None:
    engine, out, events = _run_engine(
        tmp_path, monkeypatch, [FailureMode.EMPTY_RESPONSE, FailureMode.EMPTY_RESPONSE]
    )
    assert len(engine.calls) == 2
    assert out.failure_mode is FailureMode.EMPTY_RESPONSE
    assert [e.data["attempt"] for e in events] == [1, 2]


def test_retry_shares_the_first_calls_deadline(tmp_path, monkeypatch) -> None:
    engine, _, _ = _run_engine(
        tmp_path,
        monkeypatch,
        [FailureMode.EMPTY_RESPONSE, FailureMode.NONE],
        subprocess_timeout_sec=600.0,
    )
    (_, first), (_, second) = engine.calls
    assert first == 600.0
    assert second is not None and 0 < second <= first
