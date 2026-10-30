"""``--retry-failed-pages``: a page that failed its first pass gets one more, in the run.

Coca-Cola 2018 (72 pp.), same socr version and flags: Bocconi job 693808 failed
p.10, 42, 54; Rorqual job 23180840 failed p.42, 56, 60. A failed page was never given
a second try: its sidecar is terminal and the run fingerprint blocks a selective
rerun. The option re-runs each failed page once after the page loop, ladder reversed,
inside its own document (the #1042 navigation-bar filter needs the whole document
around a page), and records the first attempt on the page.

Pinned as DIFFERENCES (CLAUDE.md, #257): the same six-page document twice, changing
only the option. Hermetic: the ladder, the engines, the judge and the judge model are
stubbed; no provider is reached.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import pytest

fitz = pytest.importorskip("fitz", reason="PyMuPDF not installed")

from socr.core.audit_log import PAGE_RETRY_KIND  # noqa: E402
from socr.core.config import EngineType, PipelineConfig  # noqa: E402
from socr.core.providers import PROFILE_GEMINI, PROFILE_QWEN_LOCAL  # noqa: E402
from socr.core.result import FailureMode, PageOutput, PageStatus  # noqa: E402
from socr.pipeline.agentic import AcceptDecision  # noqa: E402
from socr.pipeline.orchestrator import UnifiedPipeline  # noqa: E402

PAGES = 6
FLAKY_PAGE = 3


class _Judge:
    def assess(self, output: Any, provider: Any) -> AcceptDecision:
        if output.failure_mode is FailureMode.TIMEOUT:
            # route_page words a provider timeout this way; the cascade halt reads it.
            return AcceptDecision(accept=False, reason="provider timeout")
        if output.status != PageStatus.SUCCESS or not (output.text or "").strip():
            return AcceptDecision(accept=False, reason="empty/error output")
        return AcceptDecision(accept=True, reason="stub accepts all")


def _flaky_once(name: str, page: int, call_no: int) -> FailureMode | None:
    """Default script: ``FLAKY_PAGE`` fails the first time each engine reads it."""
    return FailureMode.CLI_ERROR if page == FLAKY_PAGE and call_no == 1 else None


class _Engine:
    """Reads every page; ``script(engine, page, call_no)`` names a failure mode or None.

    ``call_no`` counts this engine's calls on that page from 1. ``log`` is shared by
    both engines: ``(engine, page, pdf_path)`` per call, in order.
    """

    def __init__(self, name: str, log: list[tuple[str, int, pathlib.Path]], script) -> None:
        self.name = name
        self.log = log
        self.script = script

    def is_available(self) -> bool:
        return True

    def process_pages(self, pdf_path, page_nums, config, dpi=200, **_kw) -> list[PageOutput]:
        outs = []
        for n in page_nums:
            self.log.append((self.name, n, pathlib.Path(pdf_path)))
            call_no = sum(1 for e, p, _ in self.log if e == self.name and p == n)
            mode = self.script(self.name, n, call_no)
            if mode is not None:
                outs.append(
                    PageOutput(
                        page_num=n,
                        status=PageStatus.ERROR,
                        engine=self.name,
                        failure_mode=mode,
                        error=f"Timeout after 1s ({mode.value})",
                    )
                )
            else:
                outs.append(
                    PageOutput(
                        page_num=n,
                        text=f"{self.name} reading of page {n}: revenue 1,204",
                        status=PageStatus.SUCCESS,
                        engine=self.name,
                    )
                )
        return outs


def _pdf(path: pathlib.Path) -> pathlib.Path:
    doc = fitz.open()
    for n in range(1, PAGES + 1):
        page = doc.new_page()
        for line in range(12):
            page.insert_text(
                (60, 80 + 16 * line), f"Page {n} line {line} revenue 1,204", fontsize=9
            )
    doc.save(str(path))
    doc.close()
    return path


def _run(tmp_path, monkeypatch, leg: str, retry: bool, script=_flaky_once, backend_idle=True):
    from socr.pipeline import orchestrator as orch

    log: list[tuple[str, int, pathlib.Path]] = []
    qwen, gemini = _Engine("qwen", log, script), _Engine("gemini", log, script)
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
            native_first=False,
            table_judge_ladder=False,
            retry_failed_pages=retry,
        )
    )
    pipe._available_engines_for_agentic = lambda: [PROFILE_QWEN_LOCAL, PROFILE_GEMINI]
    pipe._build_page_judge = lambda state: _Judge()
    pipe._resolve_crop_vlm_model = lambda: None
    pipe._resolve_judge_model = lambda *a, **k: ""
    pipe._probe_backend_idle = lambda *a, **k: backend_idle
    _detect = pipe.bd_detector.detect

    def _needs_ocr(path):
        assessment = _detect(path)
        for page in assessment.pages:
            page.needs_ocr_enhancement = True
        return assessment

    pipe.bd_detector.detect = _needs_ocr
    pdf = _pdf(tmp_path / f"{leg}.pdf")
    out = tmp_path / f"out-{leg}"
    pipe.process(pdf, output_dir=out)
    sidecars = {
        int(p.stem): json.loads(p.read_text(encoding="utf-8")) for p in out.rglob("pages/*.json")
    }
    return pdf, log, sidecars


def test_failed_page_is_retried_once_with_the_ladder_reversed(tmp_path, monkeypatch) -> None:
    _, off_log, off = _run(tmp_path, monkeypatch, "off", retry=False)
    pdf, on_log, on = _run(tmp_path, monkeypatch, "on", retry=True)

    # Off: the flaky page fails both rungs once and ships failed.
    assert [(e, p) for e, p, _ in off_log if p == FLAKY_PAGE] == [
        ("qwen", FLAKY_PAGE),
        ("gemini", FLAKY_PAGE),
    ]
    assert off[FLAKY_PAGE]["status"] == "error"
    assert off[FLAKY_PAGE]["page_retry"] is None

    # On: one more pass over that page only; paid Gemini is at-most-once, so retry
    # succeeds via the free rung only.
    assert [(e, p) for e, p, _ in on_log if p == FLAKY_PAGE] == [
        ("qwen", FLAKY_PAGE),
        ("gemini", FLAKY_PAGE),
        ("qwen", FLAKY_PAGE),
    ]
    assert on[FLAKY_PAGE]["status"] != "error"
    # Pages that succeeded are not run again.
    assert [(e, p) for e, p, _ in on_log if p != FLAKY_PAGE] == [
        (e, p) for e, p, _ in off_log if p != FLAKY_PAGE
    ]


def test_retry_is_recorded_on_the_page_and_in_the_audit(tmp_path, monkeypatch) -> None:
    _, _, on = _run(tmp_path, monkeypatch, "on", retry=True)
    record = on[FLAKY_PAGE]["page_retry"]

    assert record["first_attempt"]["status"] == "error"
    assert record["retry_ladder"][0] == PROFILE_GEMINI.id
    retry_events = [e for e in on[FLAKY_PAGE]["audit_events"] if e["kind"] == PAGE_RETRY_KIND]
    assert len(retry_events) == 1
    assert retry_events[0]["data"]["first_status"] == "error"
    assert retry_events[0]["data"]["retry_status"] != "error"
    assert all(s["page_retry"] is None for n, s in on.items() if n != FLAKY_PAGE)


def test_retry_reads_the_page_inside_its_whole_document(tmp_path, monkeypatch) -> None:
    pdf, log, _ = _run(tmp_path, monkeypatch, "doc", retry=True)
    retry_call = [path for e, p, path in log if p == FLAKY_PAGE][-1]
    assert retry_call == pdf
    with fitz.open(str(retry_call)) as doc:
        assert doc.page_count == PAGES


# --- review fixes (PR #1160) --------------------------------------------------------


def _empty_twice_on(pages: set[int]):
    """Each engine answers ``pages`` empty on its first two calls (the call and its
    empty-answer retry), so the first pass fails them with provider_empty_response
    events, and reads them on any later call."""

    def script(name: str, page: int, call_no: int) -> FailureMode | None:
        return FailureMode.EMPTY_RESPONSE if page in pages and call_no <= 2 else None

    return script


def test_each_retried_page_keeps_exactly_its_own_first_attempt_events(tmp_path, monkeypatch):
    """Two failed pages: resetting the first must not shift where the second's events are
    looked for (the list offset recorded before the loop no longer holds after a removal)."""
    _, _, on = _run(tmp_path, monkeypatch, "two", retry=True, script=_empty_twice_on({2, 4}))

    for page in (2, 4):
        kept = on[page]["page_retry"]["first_attempt"]["audit_events"]
        # Qwen still retries empty answers; paid Gemini is at-most-once (one event).
        assert [e["kind"] for e in kept].count("provider_empty_response") == 2, (page, kept)
        assert all(e["page_num"] == page for e in kept)
        # Kept whole, not as a count: the detail of the first attempt survives.
        assert all(e["detail"] for e in kept)
        # And the replaced attempt's events left the page's live audit stream.
        live = [e["kind"] for e in on[page]["audit_events"]]
        assert "provider_empty_response" not in live, (page, live)
        assert on[page]["status"] != "error"


def test_a_halt_during_the_retry_pass_resets_no_further_page_and_records_the_retry(
    tmp_path, monkeypatch
):
    """The retry of page 2 times out on every rung while the backend probe says busy:
    that arms the cascade halt. Page 4, next in the retry queue, must keep its first
    attempt (no reset, no retry), and page 2's completed retry must still be recorded."""

    def script(name: str, page: int, call_no: int) -> FailureMode | None:
        if page in (2, 4) and call_no == 1:
            return FailureMode.CLI_ERROR
        if page == 2 and call_no == 2:
            return FailureMode.TIMEOUT
        return None

    _, log, on = _run(tmp_path, monkeypatch, "halt", retry=True, script=script, backend_idle=False)

    assert on[2]["page_retry"] is not None
    assert [e["kind"] for e in on[2]["audit_events"]].count(PAGE_RETRY_KIND) == 1
    assert on[4]["page_retry"] is None
    assert [(e, p) for e, p, _ in log if p == 4] == [("qwen", 4), ("gemini", 4)]
