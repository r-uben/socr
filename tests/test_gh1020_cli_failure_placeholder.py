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
from socr.core.result import DocumentStatus, FailureMode, PageOutput, PageStatus  # noqa: E402
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
        # A marker line anywhere means part of the page is missing.
        "Real text.\n\n*[OCR failed for page 1]*",
        "*[OCR failed for page 1]*\n\nReal text after it.",
        "## Page 1\n\nfine\n\n## Page 2\n\n*[OCR failed for page 2]*",
    ],
)
def test_exact_cli_markers_are_recognised(text: str) -> None:
    assert is_cli_failure_placeholder(text)


@pytest.mark.parametrize(
    "text",
    [
        "",
        None,
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
    bad_judge, bad_next, bad_md = _run_agentic(tmp_path, monkeypatch, "bad", PLACEHOLDER, 0)

    # Control leg: rung one's real text was judged, the second rung never needed.
    assert any(REAL_TEXT in t for t in ok_judge.seen)
    assert ok_next.calls == 0
    assert "OCR failed" not in ok_md

    # Failure leg: the marker never reached the judge or the output, rung two ran.
    assert not any("OCR failed" in t for t in bad_judge.seen), bad_judge.seen
    assert "OCR failed" not in bad_md
    assert bad_next.calls == 1


# --- GH-1020 review fixes ----------------------------------------------------------

MIXED = f"{REAL_TEXT}\n\n{PLACEHOLDER}\n\nmore real text"


def test_mixed_content_page_is_a_failure_not_a_partial_success(tmp_path, monkeypatch) -> None:
    page = _engine_page(tmp_path, monkeypatch, MIXED, 1)
    assert page.status == PageStatus.ERROR
    assert page.failure_mode == FailureMode.CLI_ERROR
    assert not page.audit_passed
    assert not page.text  # the readable part is not shipped as if it were the page


def test_fence_toggling_cannot_hide_a_real_marker() -> None:
    """Astra: ``` / ~~~ / ``` followed by a real marker. No fence exemption, fail closed."""
    toggled = f"```\n~~~\n```\n{PLACEHOLDER}\n"
    assert is_cli_failure_placeholder(toggled)
    assert is_cli_failure_placeholder(f"Example:\n\n```\n{PLACEHOLDER}\n```\n")


def test_exit_zero_whole_document_marker_is_an_error(tmp_path, monkeypatch) -> None:
    def _run(cmd, *args, **kwargs):
        input_path = pathlib.Path(cmd[1])
        out_dir = pathlib.Path(cmd[cmd.index("-o") + 1])
        rel_key = relative_key(input_path, input_path.parent)
        doc_dir = doc_dir_for(out_dir, rel_key)
        doc_dir.mkdir(parents=True, exist_ok=True)
        markdown_path_for(doc_dir, rel_key).write_text("*[OCR Failed]*\n", encoding="utf-8")
        result = MagicMock()
        result.returncode = 0
        result.stdout = result.stderr = ""
        return result

    monkeypatch.setattr("socr.engines.base.subprocess.run", _run)
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    result = _QwenLikeEngine().process_document(pdf, tmp_path / "out", PipelineConfig(timeout=30))
    assert result.status == DocumentStatus.ERROR
    assert result.failure_mode == FailureMode.CLI_ERROR
    assert not result.pages


def test_quoted_marker_inside_a_sentence_stays_success(tmp_path, monkeypatch) -> None:
    quoted = f"The CLI prints {PLACEHOLDER} when it fails."
    page = _engine_page(tmp_path, monkeypatch, quoted, 0)
    assert page.status == PageStatus.SUCCESS and quoted in page.text


def _two_page_pdf(path: pathlib.Path) -> pathlib.Path:
    doc = fitz.open()
    for _ in range(2):
        doc.new_page().insert_text((60, 80), REAL_TEXT, fontsize=9)
    doc.save(str(path))
    doc.close()
    return path


def test_aggregate_section_failure_fails_only_that_page(tmp_path, monkeypatch) -> None:
    """qwen folds a dir of images into ONE '## Page N' doc: one bad section, one bad page."""

    def _run(cmd, *args, **kwargs):
        if "-o" not in cmd:
            probe = MagicMock()
            probe.returncode = 0
            probe.stdout = probe.stderr = ""
            return probe
        images_dir = pathlib.Path(cmd[1])
        out_dir = pathlib.Path(cmd[cmd.index("-o") + 1])
        rel_key = relative_key(images_dir, images_dir.parent)
        doc_dir = doc_dir_for(out_dir, rel_key)
        doc_dir.mkdir(parents=True, exist_ok=True)
        markdown_path_for(doc_dir, rel_key).write_text(
            assemble_pages([REAL_TEXT, "*[OCR failed for page 2]*"]), encoding="utf-8"
        )
        result = MagicMock()
        result.returncode = 1
        result.stdout = ""
        result.stderr = "Ollama 500"
        return result

    monkeypatch.setattr("socr.engines.base.subprocess.run", _run)
    pdf = _two_page_pdf(tmp_path / "two.pdf")
    p1, p2 = _QwenLikeEngine().process_pages(pdf, [1, 2], PipelineConfig(timeout=30))
    assert p1.status == PageStatus.SUCCESS and REAL_TEXT in p1.text
    assert p2.status == PageStatus.ERROR and p2.failure_mode == FailureMode.CLI_ERROR
    assert "OCR failed" not in (p2.text or "")


def test_exit_zero_aggregate_document_with_a_failed_section_is_an_error(
    tmp_path, monkeypatch
) -> None:
    def _run(cmd, *args, **kwargs):
        input_path = pathlib.Path(cmd[1])
        out_dir = pathlib.Path(cmd[cmd.index("-o") + 1])
        rel_key = relative_key(input_path, input_path.parent)
        doc_dir = doc_dir_for(out_dir, rel_key)
        doc_dir.mkdir(parents=True, exist_ok=True)
        markdown_path_for(doc_dir, rel_key).write_text(
            assemble_pages([REAL_TEXT, "*[OCR failed for page 2]*"]), encoding="utf-8"
        )
        result = MagicMock()
        result.returncode = 0
        result.stdout = result.stderr = ""
        return result

    monkeypatch.setattr("socr.engines.base.subprocess.run", _run)
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    result = _QwenLikeEngine().process_document(pdf, tmp_path / "out", PipelineConfig(timeout=30))
    assert result.failure_mode == FailureMode.CLI_ERROR
    assert not result.pages


# --- the real judges: the marker never reaches a model -----------------------------


def test_real_judges_refuse_a_failed_page_without_calling_a_model(tmp_path, monkeypatch) -> None:
    from socr.pipeline.agentic import HeuristicPageJudge, VLMPageJudge

    bad = _engine_page(tmp_path, monkeypatch, MIXED, 1)
    model = MagicMock()
    render = MagicMock()
    vlm = VLMPageJudge(model, render)

    decision = vlm.assess(bad, PROFILE_QWEN_LOCAL)
    assert not decision.accept
    model.judge.assert_not_called()  # no model call
    render.assert_not_called()  # not even a page render
    checker = MagicMock()
    assert not HeuristicPageJudge(checker).assess(bad, PROFILE_QWEN_LOCAL).accept
    checker.check.assert_not_called()

    # Control: the same judge DOES call the model for a good page.
    ok = _engine_page(tmp_path, monkeypatch, REAL_TEXT, 0)
    model.judge.return_value = MagicMock(faithful=True, issues=[], confidence=0.9)
    vlm.assess(ok, PROFILE_QWEN_LOCAL)
    model.judge.assert_called_once()


# --- _best_effort and the table ladder ----------------------------------------------


def test_best_effort_never_selects_a_failed_candidate() -> None:
    from socr.pipeline.agentic import ProviderAttempt, _best_effort

    failed = PageOutput(
        page_num=1,
        text=PLACEHOLDER,
        status=PageStatus.ERROR,
        failure_mode=FailureMode.CLI_ERROR,
        engine="qwen",
        audit_passed=True,  # the strongest selection key, still must lose
        confidence=1.0,
    )
    weak = PageOutput(
        page_num=1, text="a", status=PageStatus.SUCCESS, engine="gemini", audit_passed=False
    )
    attempts = [
        ProviderAttempt(engine=EngineType.QWEN, output=failed, cost_usd=0.0, accepted=False),
        ProviderAttempt(engine=EngineType.GEMINI, output=weak, cost_usd=0.0, accepted=False),
    ]
    assert _best_effort(attempts, 1).output is weak
    # Only failures: whatever is returned is still a failure, never promoted.
    only = _best_effort(attempts[:1], 1).output
    assert only.status == PageStatus.ERROR


def test_table_ladder_is_never_entered_for_a_failed_candidate(monkeypatch) -> None:
    import contextlib

    from socr.judge import table_ladder
    from socr.tables import witness

    entered: list[str] = []

    @contextlib.contextmanager
    def _witnesses(*a, **k):
        entered.append("witnesses")
        yield []

    monkeypatch.setattr(witness, "prepare_table_witnesses", _witnesses)
    monkeypatch.setattr(table_ladder, "run_table_ladder", lambda *a, **k: entered.append("ladder"))
    pipe = UnifiedPipeline(PipelineConfig(quiet=True, agentic=True))
    table_md = "| a | b |\n|---|---|\n| 1 | 2 |\n"
    state = MagicMock()

    def _gate(output: PageOutput) -> None:
        pipe._run_table_judge_gate(state, 1, MagicMock(), output, [MagicMock()])

    failed = PageOutput(
        page_num=1,
        text=table_md,
        status=PageStatus.ERROR,
        failure_mode=FailureMode.CLI_ERROR,
        engine="qwen",
    )
    _gate(failed)
    assert entered == []  # no rung, and no witness preparation, for a failed candidate

    # Control: the same text as a SUCCESS page does reach the ladder machinery.
    _gate(PageOutput(page_num=1, text=table_md, status=PageStatus.SUCCESS, engine="qwen"))
    assert entered, "control: a good page must reach the table ladder, or this test proves nothing"


def _pre_pr_best_effort(attempts, page_num):
    """The selection exactly as it was on main before GH-1020."""
    from socr.pipeline.agentic import ProviderAttempt, _error_output

    usable = [a for a in attempts if a.output.text.strip()]
    pool = usable or attempts
    if not pool:
        return ProviderAttempt(
            engine=EngineType.AUTO,
            output=_error_output(page_num, "no provider produced output"),
            cost_usd=0.0,
            accepted=False,
            reason="all providers failed",
        )
    return max(
        pool,
        key=lambda a: (a.output.audit_passed, a.output.confidence, a.output.word_count),
    )


def _attempt(text, status, mode, passed, conf, engine=EngineType.QWEN):
    from socr.pipeline.agentic import ProviderAttempt

    out = PageOutput(
        page_num=1,
        text=text,
        status=status,
        failure_mode=mode,
        engine=engine.value,
        audit_passed=passed,
        confidence=conf,
    )
    return ProviderAttempt(engine=engine, output=out, cost_usd=0.0, accepted=False)


def test_best_effort_all_failed_matches_the_pre_pr_selection() -> None:
    """With no healthy candidate the choice is the one main made, ranking and all."""
    from socr.pipeline.agentic import _best_effort

    err, cli = PageStatus.ERROR, FailureMode.CLI_ERROR
    cases = [
        # every attempt empty: audit_passed / confidence decide, as on main
        [_attempt("", err, cli, False, 0.1), _attempt("", err, None, True, 0.2)],
        [_attempt("", err, cli, True, 0.9), _attempt("", err, cli, False, 0.99)],
        # only failed candidates carry text
        [_attempt("one two", err, cli, False, 0.5), _attempt("", err, None, True, 0.9)],
        [],
    ]
    for attempts in cases:
        assert _best_effort(attempts, 1) is not None
        got, want = _best_effort(attempts, 1), _pre_pr_best_effort(attempts, 1)
        assert (got.output.text, got.engine, got.reason) == (
            want.output.text,
            want.engine,
            want.reason,
        )
        if attempts:
            assert got is want


def test_best_effort_differs_from_pre_pr_only_by_dropping_failed_for_healthy() -> None:
    from socr.pipeline.agentic import _best_effort

    err, cli, ok = PageStatus.ERROR, FailureMode.CLI_ERROR, PageStatus.SUCCESS
    failed = _attempt(PLACEHOLDER, err, cli, True, 1.0)
    weak = _attempt("a", ok, None, False, 0.0, EngineType.GEMINI)
    attempts = [failed, weak]
    assert _pre_pr_best_effort(attempts, 1) is failed  # main would have shipped it
    assert _best_effort(attempts, 1) is weak
    # Among healthy candidates the ranking is untouched.
    strong = _attempt("a b c", ok, None, True, 0.9, EngineType.MARKER)
    assert _best_effort([failed, weak, strong], 1) is strong
    assert _best_effort([weak, strong], 1) is _pre_pr_best_effort([weak, strong], 1)


def test_per_page_file_with_fence_wrapped_marker_is_a_failure(tmp_path, monkeypatch) -> None:
    """A per-page file whose marker is wrapped in a fence: it fails either way, and the
    cleaned text (fence unwrapped) is checked as well as the raw."""
    text = f"```markdown\n{PLACEHOLDER}\n```\n"

    def _run(cmd, *args, **kwargs):
        if "-o" not in cmd:
            probe = MagicMock()
            probe.returncode = 0
            probe.stdout = probe.stderr = ""
            return probe
        images_dir = pathlib.Path(cmd[1])
        out_dir = pathlib.Path(cmd[cmd.index("-o") + 1])
        rel_key = relative_key(images_dir / "page_0001.png", images_dir)
        doc_dir = doc_dir_for(out_dir, rel_key)
        doc_dir.mkdir(parents=True, exist_ok=True)
        markdown_path_for(doc_dir, rel_key).write_text(text, encoding="utf-8")
        result = MagicMock()
        result.returncode = 0
        result.stdout = result.stderr = ""
        return result

    monkeypatch.setattr("socr.engines.base.subprocess.run", _run)
    pdf = _pdf(tmp_path / "d.pdf")
    page = _QwenLikeEngine().process_pages(pdf, [1], PipelineConfig(timeout=30))[0]
    assert page.status == PageStatus.ERROR and page.failure_mode == FailureMode.CLI_ERROR


def test_page_with_a_fenced_marker_line_fails_closed(tmp_path, monkeypatch) -> None:
    text = f"{REAL_TEXT}\n\n```markdown\n{PLACEHOLDER}\n```\n"
    page = _engine_page(tmp_path, monkeypatch, text, 0)
    assert page.status == PageStatus.ERROR and page.failure_mode == FailureMode.CLI_ERROR


FRONTMATTER_HIDDEN = f"---\n{{marker}}\n---\n{REAL_TEXT}"


def test_frontmatter_hidden_marker_fails_the_document(tmp_path, monkeypatch) -> None:
    """Astra: cleaning strips the frontmatter, so only the RAW file shows the marker."""
    raw = FRONTMATTER_HIDDEN.format(marker="*[OCR Failed]*")
    assert not is_cli_failure_placeholder(BaseEngine._clean_output(raw, "qwen")), (
        "setup: cleaning must hide the marker, or this test cannot tell raw from cleaned"
    )

    def _run(cmd, *args, **kwargs):
        input_path = pathlib.Path(cmd[1])
        out_dir = pathlib.Path(cmd[cmd.index("-o") + 1])
        rel_key = relative_key(input_path, input_path.parent)
        doc_dir = doc_dir_for(out_dir, rel_key)
        doc_dir.mkdir(parents=True, exist_ok=True)
        markdown_path_for(doc_dir, rel_key).write_text(raw, encoding="utf-8")
        result = MagicMock()
        result.returncode = 0
        result.stdout = result.stderr = ""
        return result

    monkeypatch.setattr("socr.engines.base.subprocess.run", _run)
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    result = _QwenLikeEngine().process_document(pdf, tmp_path / "out", PipelineConfig(timeout=30))
    assert result.status == DocumentStatus.ERROR
    assert result.failure_mode == FailureMode.CLI_ERROR


def test_frontmatter_hidden_marker_fails_the_aggregate_page(tmp_path, monkeypatch) -> None:
    hidden = f"---\n*[OCR failed for page 1]*\n---\n{assemble_pages([REAL_TEXT, REAL_TEXT])}"
    assert not is_cli_failure_placeholder(BaseEngine._clean_output(hidden, "qwen")), (
        "setup: cleaning must hide the marker"
    )

    def _run(cmd, *args, **kwargs):
        if "-o" not in cmd:
            probe = MagicMock()
            probe.returncode = 0
            probe.stdout = probe.stderr = ""
            return probe
        images_dir = pathlib.Path(cmd[1])
        out_dir = pathlib.Path(cmd[cmd.index("-o") + 1])
        rel_key = relative_key(images_dir, images_dir.parent)
        doc_dir = doc_dir_for(out_dir, rel_key)
        doc_dir.mkdir(parents=True, exist_ok=True)
        markdown_path_for(doc_dir, rel_key).write_text(hidden, encoding="utf-8")
        result = MagicMock()
        result.returncode = 0
        result.stdout = result.stderr = ""
        return result

    monkeypatch.setattr("socr.engines.base.subprocess.run", _run)
    pdf = _two_page_pdf(tmp_path / "two.pdf")
    pages = _QwenLikeEngine().process_pages(pdf, [1, 2], PipelineConfig(timeout=30))
    assert any(
        p.status == PageStatus.ERROR and p.failure_mode == FailureMode.CLI_ERROR for p in pages
    )
    assert all("OCR failed" not in (p.text or "") for p in pages)
