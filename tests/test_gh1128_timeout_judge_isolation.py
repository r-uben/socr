"""#1128: the page-judge timeout worker must not mutate the live PageOutput.

Measured on current main (2026-10-13), worktree at 5567dbda: 15 of 15
``process()`` runs with the reviewer's 0.1 s deadline and the contradicted
table 99.9 / 88.8 vs witness 41.5 / 17.2 shipped SUCCESS unmarked.
``_LATE_MUTATION_REPEATS`` is that batch.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import fitz
import pytest

from socr.core.config import EngineType, PipelineConfig
from socr.core.daemon_call import submit_daemon
from socr.core.providers import PROFILE_QWEN_LOCAL
from socr.core.result import FailureMode, PageOutput, PageStatus
from socr.judge.judge import PageJudgeTimeoutError
from socr.pipeline.agentic import (
    AcceptDecision,
    CircuitBreakerPageJudge,
    HeuristicPageJudge,
    SourceEvidenceTableJudge,
)
from socr.pipeline import orchestrator as orch
from socr.pipeline.orchestrator import UnifiedPipeline
from socr.tables.source_evidence import SourceEvidenceResult

# See module docstring: counted on current main before the isolation change.
_LATE_MUTATION_REPEATS = 15
_DEADLINE = 0.05
_SLEEP = 0.2
_LATE_TEXT = "LATE MUTATION WIPED THE TABLE"
_MODEL = "| Item | Value |\n| --- | --- |\n| alpha | 99.9 |\n| beta | 88.8 |\n"
_LABEL_NOTE = "table label unverified by page evidence: alpha"


def test_loaded_source_is_this_checkout() -> None:
    import socr

    assert Path(socr.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1] / "src")


class _Handshake:
    """One barrier per assess so a second call (promotion) cannot run on-time."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.mutated = threading.Event()

    def wait_until_past_deadline(self) -> None:
        assert self.release.wait(5)
        self.release.clear()

    def finish_mutation(self) -> None:
        self.mutated.set()

    def let_worker_mutate(self) -> None:
        self.mutated.clear()
        self.release.set()
        assert self.mutated.wait(5)
        self.mutated.clear()


class _LateMutateJudge:
    """Waits past the deadline, then mutates the object it was given."""

    def __init__(self, handshake: _Handshake | None = None) -> None:
        self._handshake = handshake

    def assess(self, output, provider):
        if self._handshake is not None:
            self._handshake.wait_until_past_deadline()
        else:
            time.sleep(_SLEEP)
        output.status = PageStatus.SUCCESS
        output.failure_mode = FailureMode.NONE
        output.table_disputed = ""
        output.table_disputed_values = ()
        output.audit_passed = True
        output.text = _LATE_TEXT
        if self._handshake is not None:
            self._handshake.finish_mutation()
        return AcceptDecision(accept=True, reason="late accept")


class _OnTimeMutateJudge:
    def assess(self, output, provider):
        output.status = PageStatus.WARNING
        output.failure_mode = FailureMode.HALLUCINATION
        output.text = (output.text or "") + "\nON_TIME"
        return AcceptDecision(accept=True, reason="on-time mutate")


class _BrokenLeaf:
    def assess(self, output, provider):
        raise RuntimeError("judge HTTP 404")


class _InnerTimeoutLeaf:
    """Finishes inside the wrapper deadline, then raises builtin TimeoutError."""

    def assess(self, output, provider):
        output.status = PageStatus.WARNING
        output.failure_mode = FailureMode.HALLUCINATION
        output.text = (output.text or "") + "\nINNER_TIMEOUT"
        raise TimeoutError("leaf deadline")


class _SuccessEngine:
    def __init__(self) -> None:
        self.name = "qwen"

    def is_available(self) -> bool:
        return True

    def process_pages(self, pdf_path, page_nums, config, dpi, subprocess_timeout=None, **_kw):
        return [
            PageOutput(
                page_num=n,
                text="Results.\n\n" + _MODEL,
                status=PageStatus.SUCCESS,
                engine=self.name,
            )
            for n in page_nums
        ]


def _scan_pdf(path: Path) -> Path:
    doc = fitz.open()
    page = doc.new_page()
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 200, 200), False)
    pix.set_rect(pix.irect, (235, 235, 235))
    page.insert_image(page.rect, pixmap=pix)
    doc.save(path)
    doc.close()
    return path


class _Engine:
    def __init__(self) -> None:
        self.name = "qwen"

    def is_available(self) -> bool:
        return True

    def process_pages(self, pdf_path, page_nums, config, dpi, subprocess_timeout=None, **_kw):
        return [
            PageOutput(
                page_num=n,
                text="Results.\n\n" + _MODEL,
                status=PageStatus.ERROR,
                engine=self.name,
                audit_passed=False,
                failure_mode=FailureMode.HALLUCINATION,
                table_disputed="99.9, 88.8",
                table_disputed_values=("99.9", "88.8"),
            )
            for n in page_nums
        ]


def _pipeline() -> UnifiedPipeline:
    return UnifiedPipeline(
        PipelineConfig(
            agentic=True,
            quiet=True,
            primary_engine=EngineType.QWEN,
            local_engine=EngineType.QWEN,
            enabled_engines=[EngineType.QWEN],
            native_first=True,
            write_manifest=False,
            judge_backend="heuristic",
            dual_pass_tables=False,
            table_judge_ladder=False,
            detect_equations=False,
            save_figures=False,
        )
    )


def _ship(tmp_path: Path, tag: str, pipe: UnifiedPipeline, monkeypatch) -> str:
    """Run process() with the late wipe forced before assemble.

    The judge waits until the caller has already taken the deadline, mutates,
    then signals back. ``process()`` cannot assemble until that handshake
    finishes, so ``probe = output`` reaches the shipped fragment.
    """
    handshake = _Handshake()
    late = _LateMutateJudge(handshake)
    timeout_cls = UnifiedPipeline._TimeoutJudge

    class _SyncedTimeout(timeout_cls):
        def assess(self, output, provider):
            try:
                return super().assess(output, provider)
            except PageJudgeTimeoutError:
                handshake.let_worker_mutate()
                raise

    pdf = _scan_pdf(tmp_path / f"{tag}.pdf")
    out = tmp_path / f"out-{tag}"
    monkeypatch.setattr(orch, "get_engine", lambda engine_type: _Engine())
    pipe._available_engines_for_agentic = lambda: [PROFILE_QWEN_LOCAL]
    pipe._build_page_judge = lambda state: _SyncedTimeout(late, _DEADLINE, owner=pipe)
    pipe._resolve_crop_vlm_model = lambda: None
    pipe._resolve_judge_model = lambda *a, **k: ""
    pipe.process(pdf, output_dir=out)
    return next(iter(out.rglob("pages/00001.md"))).read_text()


def test_late_mutation_never_reaches_shipped_page(tmp_path: Path, monkeypatch) -> None:
    """N process() runs: late wipe never appears; shipped bytes are identical."""
    pipe = _pipeline()
    texts = [_ship(tmp_path, f"r{i}", pipe, monkeypatch) for i in range(_LATE_MUTATION_REPEATS)]
    assert len(texts) == _LATE_MUTATION_REPEATS
    assert all(t == texts[0] for t in texts)
    assert _LATE_TEXT not in texts[0]
    assert "99.9" in texts[0]


def test_on_time_result_is_adopted() -> None:
    """A judge that finishes in time still mutates the live object (vs passthrough)."""
    pipe = _pipeline()
    provider = MagicMock()
    live = PageOutput(page_num=1, text="body", status=PageStatus.SUCCESS, engine="qwen")
    direct = PageOutput(page_num=1, text="body", status=PageStatus.SUCCESS, engine="qwen")
    wrapped = pipe._TimeoutJudge(_OnTimeMutateJudge(), timeout_sec=5.0)
    passthrough = pipe._TimeoutJudge(_OnTimeMutateJudge(), timeout_sec=None)
    w = wrapped.assess(live, provider)
    p = passthrough.assess(direct, provider)
    assert w.accept is p.accept is True
    assert live.text == direct.text
    assert "ON_TIME" in live.text
    assert live.status is PageStatus.WARNING
    assert live.failure_mode is FailureMode.HALLUCINATION


def test_inner_timeout_error_is_wrapped_and_adopted() -> None:
    """#1138: an inner builtin TimeoutError is wrapped; on_timeout fires; writes stay."""
    fired: list[int] = []
    live = PageOutput(page_num=1, text="body", status=PageStatus.SUCCESS, engine="qwen")
    judge = UnifiedPipeline._TimeoutJudge(
        _InnerTimeoutLeaf(),
        timeout_sec=5.0,
        on_timeout=lambda page_num: fired.append(page_num),
    )
    with pytest.raises(PageJudgeTimeoutError) as excinfo:
        judge.assess(live, MagicMock())
    assert str(excinfo.value).startswith("page judge timeout:")
    assert "after " not in str(excinfo.value)
    assert fired == [1]
    assert "INNER_TIMEOUT" in live.text
    assert live.status is PageStatus.WARNING
    assert live.failure_mode is FailureMode.HALLUCINATION


def test_late_discard_is_journaled() -> None:
    pipe = _pipeline()
    out = PageOutput(page_num=1, text=_MODEL, status=PageStatus.SUCCESS, engine="qwen")
    judge = pipe._TimeoutJudge(_LateMutateJudge(), timeout_sec=_DEADLINE)
    with pytest.raises(PageJudgeTimeoutError):
        judge.assess(out, MagicMock())
    time.sleep(_SLEEP + 0.05)
    assert "late page-judge result discarded" in out.audit_notes
    assert out.text == _MODEL
    assert _LATE_TEXT not in out.text


def _ship_write_then_raise(
    tmp_path: Path, tag: str, pipe: UnifiedPipeline, *, breaker: bool
) -> dict:
    pdf = _scan_pdf(tmp_path / f"{tag}.pdf")
    out = tmp_path / f"out-{tag}"
    pipe._available_engines_for_agentic = lambda: [PROFILE_QWEN_LOCAL]
    pipe._resolve_crop_vlm_model = lambda: None
    pipe._resolve_judge_model = lambda *a, **k: ""

    def build(state):
        inner = CircuitBreakerPageJudge(_BrokenLeaf(), lambda: True) if breaker else _BrokenLeaf()
        return SourceEvidenceTableJudge(
            inner,
            lambda n: object(),
            record_event=lambda ev: pipe._record_judge_event(state, ev),
            native_trusted=lambda n: False,
        )

    pipe._build_page_judge = build
    evidence = SourceEvidenceResult(
        verifiable=True,
        passed=True,
        reason="numeric tokens supported",
        content_unverified="alpha",
    )
    with (
        patch.object(orch, "get_engine", lambda typ: _SuccessEngine()),
        patch("socr.tables.source_evidence.verify_scanned_table", return_value=evidence),
    ):
        pipe.process(pdf, output_dir=out)
    side = json.loads(next(out.rglob("pages/00001.json")).read_text())
    win = side.get("winning_output") or {}
    return {
        "status": win.get("status"),
        "label": win.get("table_label_unverified"),
        "audit_notes": list(win.get("audit_notes") or []),
    }


@pytest.mark.parametrize("breaker", [False, True])
def test_on_time_exception_ships_as_passthrough(tmp_path: Path, breaker: bool) -> None:
    """SE writes a label, then the leaf raises: wrapped == passthrough."""

    def _run(deadline: float | None, tag: str) -> dict:
        pipe = _pipeline()
        pipe._TimeoutJudge = lambda inner, timeout_sec, **kw: UnifiedPipeline._TimeoutJudge(
            inner, deadline, **kw
        )
        return _ship_write_then_raise(tmp_path, tag, pipe, breaker=breaker)

    passthrough = _run(None, f"pass-{breaker}")
    wrapped = _run(5.0, f"wrap-{breaker}")
    assert wrapped == passthrough
    assert passthrough["status"] == "warning"
    assert passthrough["label"] == "alpha"
    assert _LABEL_NOTE in passthrough["audit_notes"]


def test_completion_between_timeout_and_handler_still_notes_discard() -> None:
    """Worker finishing after result() times out is late; the note still lands."""
    release = threading.Event()

    class _WaitThenWrite:
        def assess(self, output, provider):
            assert release.wait(5)
            output.text = "late result"
            return AcceptDecision(accept=True, reason="late")

    class _CompletionBetweenTimeoutAndHandler:
        def __init__(self, future):
            self.future = future

        def result(self, timeout):
            try:
                return self.future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                release.set()
                self.future.result(timeout=5)
                raise

        def done(self):
            return self.future.done()

        def cancel(self):
            return self.future.cancel()

    out = PageOutput(page_num=1, text="original", status=PageStatus.SUCCESS, engine="qwen")
    with patch.object(
        orch, "submit_daemon", lambda fn: _CompletionBetweenTimeoutAndHandler(submit_daemon(fn))
    ):
        with pytest.raises(PageJudgeTimeoutError):
            UnifiedPipeline._TimeoutJudge(_WaitThenWrite(), 0.01).assess(out, PROFILE_QWEN_LOCAL)
    assert out.text == "original"
    assert "late page-judge result discarded" in out.audit_notes


def test_evidence_written_before_leaf_deadline_survives_stall(tmp_path: Path, monkeypatch) -> None:
    """Astra r2 P1: SE writes on the live object, then the leaf stalls.

    Production order is SE(Timeout(leaf)). Discarding the leaf copy must
    not erase the label already recorded outside the deadline.
    """
    releases: list[threading.Event] = []
    entered: list[bool] = []
    timeout_cls = UnifiedPipeline._TimeoutJudge

    class _StalledLeaf:
        def assess(self, output, provider):
            assert output.table_label_unverified == "alpha"
            assert _LABEL_NOTE in output.audit_notes
            release = threading.Event()
            releases.append(release)
            entered.append(True)
            assert release.wait(5)
            raise RuntimeError("judge HTTP 404 after delay")

    class _ReleaseAfterTimeout(timeout_cls):
        def assess(self, output, provider):
            try:
                return super().assess(output, provider)
            except PageJudgeTimeoutError:
                releases[-1].set()
                raise

    pipe = _pipeline()
    pipe._available_engines_for_agentic = lambda: [PROFILE_QWEN_LOCAL]
    pipe._resolve_crop_vlm_model = lambda: None
    pipe._resolve_judge_model = lambda *a, **k: ""

    def build(state):
        timed = _ReleaseAfterTimeout(_StalledLeaf(), _DEADLINE, owner=pipe)
        return SourceEvidenceTableJudge(
            timed,
            lambda n: object(),
            record_event=lambda ev: pipe._record_judge_event(state, ev),
            native_trusted=lambda n: False,
        )

    pipe._build_page_judge = build
    evidence = SourceEvidenceResult(
        verifiable=True,
        passed=True,
        reason="numeric tokens supported",
        content_unverified="alpha",
    )
    pdf = _scan_pdf(tmp_path / "early-warning.pdf")
    out = tmp_path / "out-early-warning"
    with (
        patch.object(orch, "get_engine", lambda typ: _SuccessEngine()),
        patch("socr.tables.source_evidence.verify_scanned_table", return_value=evidence),
    ):
        pipe.process(pdf, output_dir=out)
    side = json.loads(next(out.rglob("pages/00001.json")).read_text())
    win = side.get("winning_output") or {}
    assert entered
    assert win.get("status") == "warning"
    assert win.get("table_label_unverified") == "alpha"
    assert _LABEL_NOTE in list(win.get("audit_notes") or [])


# The hung stub's sleep, and the probe's outer bound. The bound covers interpreter
# start plus ``import socr`` (fitz loads) on a slow CI runner, so it must be well
# above that; it must stay below the hung stub's sleep so a no-timeout mutant is
# still caught. Half the sleep satisfies both.
_HUNG_STUB_SLEEP_SEC = 30
_PROBE_OUTER_TIMEOUT_SEC = _HUNG_STUB_SLEEP_SEC / 2


def test_tesseract_witness_timeout_fails_closed_without_hanging() -> None:
    """#1128 r4 P1: a finite witness timeout returns could-not-read; no hang.

    The child injects a pytesseract stub (CI has no ``scanned`` extra) that
    raises when ``timeout`` is passed and sleeps otherwise. Production
    returns ``exec_error`` inside the outer bound (``_PROBE_OUTER_TIMEOUT_SEC``). A no-timeout mutant
    is killed with the child, so the stub never stays in this process's
    ``sys.modules``.
    """
    src = Path(__file__).resolve().parents[1] / "src"
    child = r"""
import sys
import time
import types

from socr.tables.source_evidence import (
    TESSERACT_WITNESS_TIMEOUT_SEC,
    WITNESS_EXEC_ERROR,
    classical_ocr_with_state,
)

class _Pix:
    width = 4
    height = 4
    samples = bytes(4 * 4 * 3)

def fake_image_to_string(img, timeout=0, **_kw):
    if timeout:
        raise RuntimeError("Tesseract process timeout")
    time.sleep(__HUNG_SLEEP__)
    return "hung"

stub = types.ModuleType("pytesseract")
stub.image_to_string = fake_image_to_string
stub.TesseractNotFoundError = type("TesseractNotFoundError", (OSError,), {})
sys.modules["pytesseract"] = stub
assert TESSERACT_WITNESS_TIMEOUT_SEC > 0
text, state, detail = classical_ocr_with_state(_Pix())
assert text == ""
assert state == WITNESS_EXEC_ERROR
assert "timeout" in detail.lower()
print("OK", flush=True)
"""
    child = child.replace("__HUNG_SLEEP__", str(_HUNG_STUB_SLEEP_SEC))
    env = {**os.environ, "PYTHONPATH": str(src)}
    proc = subprocess.Popen(
        [sys.executable, "-c", child],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=_PROBE_OUTER_TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        pytest.fail(
            "classical_ocr_with_state hung: image_to_string was called without a finite timeout"
        )
    assert proc.returncode == 0, stderr or stdout
    assert "OK" in stdout


def test_contradicted_table_never_ships_success_unmarked(tmp_path: Path, monkeypatch) -> None:
    """#1135: contradicted table + stalled leaf. Timeout occurred; never SUCCESS unmarked.

    Production ``_build_page_judge`` (SE outside the 0.1 s leaf deadline) plus
    a 0.12 s contradict so ``se_inside_deadline`` still misses the witness.
    The leaf is event-gated inside the existing ``_TimeoutJudge``. Same-process
    compare: a refuse-now leaf ships the same rejection without a timeout.
    """

    def _slow_contradict(*_a, **_k):
        time.sleep(0.12)
        return SourceEvidenceResult(
            verifiable=True,
            passed=False,
            reason="witness contradicts table values",
            disputed_values=("99.9", "88.8"),
        )

    def _ship(*, stall: bool, tag: str) -> dict:
        releases: list[threading.Event] = []
        entered: list[bool] = []
        timeout_hits: list[bool] = []
        timeout_cls = UnifiedPipeline._TimeoutJudge

        class _ReleaseAfterTimeout(timeout_cls):
            def assess(self, output, provider):
                try:
                    return super().assess(output, provider)
                except PageJudgeTimeoutError:
                    timeout_hits.append(True)
                    if releases:
                        releases[-1].set()
                    raise

        pipe = _pipeline()
        pipe._available_engines_for_agentic = lambda: [PROFILE_QWEN_LOCAL]
        pipe._resolve_crop_vlm_model = lambda: None
        pipe._resolve_judge_model = lambda *a, **k: ""
        pipe._TimeoutJudge = lambda inner, timeout_sec, **kw: _ReleaseAfterTimeout(inner, 0.1, **kw)

        def stalled_assess(self, output, provider):
            release = threading.Event()
            releases.append(release)
            entered.append(True)
            assert release.wait(5)
            return AcceptDecision(accept=False, reason="stalled leaf woke")

        def refuse_assess(self, output, provider):
            entered.append(True)
            return AcceptDecision(accept=False, reason="heuristic refuse")

        pdf = _scan_pdf(tmp_path / f"{tag}.pdf")
        out = tmp_path / f"out-{tag}"
        monkeypatch.setattr(orch, "get_engine", lambda typ: _SuccessEngine())
        leaf = stalled_assess if stall else refuse_assess
        with (
            patch.object(HeuristicPageJudge, "assess", leaf),
            patch("socr.tables.source_evidence.verify_scanned_table", side_effect=_slow_contradict),
        ):
            pipe.process(pdf, output_dir=out)
        side = json.loads(next(out.rglob("pages/00001.json")).read_text())
        win = side.get("winning_output") or {}
        md = next(out.rglob("pages/00001.md")).read_text()
        disputed = (win.get("table_disputed") or "").strip()
        status = win.get("status") or side.get("status")
        marked_or_floored = (
            bool(disputed)
            or "disputed" in md.lower()
            or "unverify" in md.lower()
            or "failed table" in md.lower()
            or status in {"error", "warning"}
        )
        success_unmarked = status == "success" and not disputed and "disputed" not in md.lower()
        return {
            "entered": bool(entered),
            "timeout": bool(timeout_hits),
            "status": status,
            "disputed": disputed,
            "marked_or_floored": marked_or_floored,
            "success_unmarked": success_unmarked,
        }

    stalled = _ship(stall=True, tag="stall")
    prompt = _ship(stall=False, tag="prompt")
    assert stalled["entered"], "leaf never ran — the #1135 timeout was not exercised"
    assert stalled["timeout"], "stalled leaf did not trip the page-judge deadline"
    assert stalled["marked_or_floored"]
    assert not stalled["success_unmarked"]
    assert prompt["marked_or_floored"]
    assert not prompt["success_unmarked"]
    assert stalled["status"] == prompt["status"]
    assert stalled["disputed"] == prompt["disputed"]
