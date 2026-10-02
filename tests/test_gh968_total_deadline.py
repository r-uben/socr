"""GH-968: every Ollama HTTP call has a TOTAL wall-clock deadline.

httpx's ``timeout=`` (and urllib's) is per-read/per-socket-op inactivity. A peer
that trickles a byte every few hundred ms never trips it, so a call with
``timeout=600`` can hang a whole document indefinitely. The fix wraps each call
in ``call_with_total_deadline`` (daemon thread, abandoned on overrun).

Hermetic: the only server is a loopback socket this file opens. No Ollama, no
provider, no model call. Every potentially-hanging call runs in a daemon thread
joined with a bound, so a regression cannot hang the suite.
"""

from __future__ import annotations

import socket
import threading
import time
from pathlib import Path
from unittest.mock import patch

import fitz
import httpx
import pytest

from socr.core import ollama_utils
from socr.core.audit_log import AuditEvent
from socr.core.config import EngineType, PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.ollama_utils import TotalDeadlineExceeded, call_with_total_deadline
from socr.core.result import PageOutput, PageStatus
from socr.core.state import DocumentState
from socr.engines import gemini_api
from socr.judge import table_rung_ollama
from socr.judge.table_rung_ollama import _post_chat as REAL_POST_CHAT
from socr.judge.table_rung_ollama import build_ollama_rung, ollama_rung_reachable
from socr.judge.table_verdict import TABLE_LADDER_UNVERIFIED_KIND
from socr.math import equation_latex, recover
from socr.pipeline.orchestrator import UnifiedPipeline
from socr.tables import extract
from socr.tables.binding import BindingEvidence

#: The configured per-call timeout the tests use. Small, because the point is
#: that the TOTAL deadline equals it; the trickle interval is well below it so a
#: per-read timeout can never fire.
DEADLINE = 0.5
TRICKLE_INTERVAL = 0.1


class TrickleServer:
    """Accepts a connection, sends headers, then one body byte every interval.

    Content-Length is enormous so the response never completes; each byte
    resets a per-read timeout, which is exactly the GH-968 shape.
    """

    def __init__(self, interval: float = TRICKLE_INTERVAL) -> None:
        self.interval = interval
        self.stop = threading.Event()
        self.accepted = 0
        self._sock = socket.socket()
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(8)
        self.url = f"http://127.0.0.1:{self._sock.getsockname()[1]}"
        threading.Thread(target=self._accept_loop, daemon=True).start()

    def _accept_loop(self) -> None:
        while not self.stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            self.accepted += 1
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn: socket.socket) -> None:
        try:
            conn.settimeout(2.0)
            seen = b""
            while b"\r\n\r\n" not in seen:
                chunk = conn.recv(65536)
                if not chunk:
                    return
                seen += chunk
            conn.sendall(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                b"Content-Length: 1000000000\r\n\r\n"
            )
            while not self.stop.is_set():
                conn.sendall(b" ")
                time.sleep(self.interval)
        except OSError:
            pass
        finally:
            conn.close()

    def close(self) -> None:
        self.stop.set()
        self._sock.close()


@pytest.fixture
def trickle():
    server = TrickleServer()
    try:
        yield server
    finally:
        server.close()


@pytest.fixture(autouse=True)
def _real_post_chat(monkeypatch):
    """conftest pins ``_post_chat`` to a no-daemon stub; these tests need the real one."""
    monkeypatch.setattr(table_rung_ollama, "_post_chat", REAL_POST_CHAT)


def _bounded(fn, bound: float):
    """Run *fn* in a daemon thread; ``(finished, elapsed, value_or_exc)``."""
    box: list[object] = []

    def _w() -> None:
        try:
            box.append(fn())
        except BaseException as exc:
            box.append(exc)

    start = time.monotonic()
    t = threading.Thread(target=_w, daemon=True)
    t.start()
    t.join(bound)
    return (not t.is_alive()), time.monotonic() - start, (box[0] if box else None)


PAYLOAD = {"model": "m:cloud", "messages": [], "stream": False}


# -- the wrapper ------------------------------------------------------------


def test_trickle_server_defeats_a_per_read_timeout(trickle):
    """Control / difference pin: with NO wrapper, the same call is still running
    past 3x the deadline (a per-read timeout never fires on a trickle)."""
    finished, elapsed, _ = _bounded(
        lambda: httpx.post(f"{trickle.url}/api/chat", json=PAYLOAD, timeout=DEADLINE),
        3 * DEADLINE,
    )
    assert not finished, f"raw httpx.post returned after {elapsed:.2f}s; the trickle is not biting"
    assert trickle.accepted == 1


def test_post_chat_fails_within_1_5x_the_deadline_on_a_trickle(trickle):
    finished, elapsed, value = _bounded(
        lambda: REAL_POST_CHAT(trickle.url, PAYLOAD, DEADLINE), 3 * DEADLINE
    )
    assert finished, "_post_chat is still running past 3x the deadline"
    assert isinstance(value, httpx.TimeoutException), value
    assert isinstance(value, TotalDeadlineExceeded)
    assert elapsed < 1.5 * DEADLINE, elapsed
    assert "m:cloud" in str(value), "the failure must name the call"


def test_overrun_error_is_catchable_as_httpx_and_builtin_timeouts():
    exc = TotalDeadlineExceeded("x")
    assert isinstance(exc, httpx.ReadTimeout)
    assert isinstance(exc, httpx.HTTPError)
    assert isinstance(exc, TimeoutError)  # the urllib callers' handler


def test_wrapper_returns_value_and_reraises_the_callees_exception():
    assert call_with_total_deadline(lambda: 7, 1.0) == 7

    def _boom():
        raise ValueError("inner")

    with pytest.raises(ValueError, match="inner"):
        call_with_total_deadline(_boom, 1.0)


def test_worker_thread_is_a_daemon_and_the_alias_is_kept():
    seen: list[bool] = []
    call_with_total_deadline(lambda: seen.append(threading.current_thread().daemon), 1.0)
    assert seen == [True]
    assert ollama_utils._call_within(lambda: 1, 1.0) == (True, 1)


def test_abandoned_call_does_not_block_the_caller_or_exit():
    release = threading.Event()
    start = time.monotonic()
    with pytest.raises(TotalDeadlineExceeded):
        call_with_total_deadline(lambda: release.wait(30), 0.2, label="stuck")
    assert time.monotonic() - start < 1.0
    release.set()


# -- real trickle through each remaining transport -------------------------


def test_generation_canary_gives_up_on_a_trickle(trickle):
    finished, elapsed, value = _bounded(
        lambda: extract._ollama_generation_canary(trickle.url, "m", DEADLINE), 3 * DEADLINE
    )
    assert finished and value is False
    assert elapsed < 1.5 * DEADLINE, elapsed


def test_urllib_equation_lanes_give_up_on_a_trickle(trickle, tmp_path):
    finished, elapsed, value = _bounded(
        lambda: recover.latex_for_image(b"png", host=trickle.url, timeout=DEADLINE), 3 * DEADLINE
    )
    assert finished and value == ""
    assert elapsed < 1.5 * DEADLINE, elapsed

    crop = tmp_path / "c.png"
    crop.write_bytes(b"png")
    finished, elapsed, value = _bounded(
        lambda: equation_latex.latex_for_crop(crop, host=trickle.url, timeout=DEADLINE),
        3 * DEADLINE,
    )
    assert finished and value == ""
    assert elapsed < 1.5 * DEADLINE, elapsed


# -- every wrapped site goes through call_with_total_deadline ---------------


class _Spy:
    """Stands in for the wrapper: records the call, never runs fn, overruns."""

    def __init__(self) -> None:
        self.calls: list[tuple[float, str]] = []

    def __call__(self, fn, timeout, *, label=""):
        self.calls.append((timeout, label))
        raise TotalDeadlineExceeded(f"{label} spy overrun")


@pytest.fixture
def spy():
    return _Spy()


def test_site_post_chat(monkeypatch, spy):
    monkeypatch.setattr(table_rung_ollama, "call_with_total_deadline", spy)
    with pytest.raises(TotalDeadlineExceeded):
        REAL_POST_CHAT("http://h", PAYLOAD, 7.5)
    assert [c[0] for c in spy.calls] == [7.5]


def test_site_ollama_rung_reachable(monkeypatch, spy):
    monkeypatch.setattr(table_rung_ollama, "call_with_total_deadline", spy)
    assert ollama_rung_reachable("m", "http://h", timeout=4.25) is False
    assert [c[0] for c in spy.calls] == [4.25]


def test_site_canary_and_tags_probe(monkeypatch, spy):
    monkeypatch.setattr(extract, "call_with_total_deadline", spy)
    assert extract._ollama_generation_canary("http://h", "m", 3.5) is False
    assert extract.probe_ollama_idle("http://h", timeout=2.5, model="m") is False
    # the /api/tags precondition fails first, so only it is reached here
    assert [c[0] for c in spy.calls] == [3.5, 2.5]
    assert "canary" in spy.calls[0][1] and "tags" in spy.calls[1][1]


def test_site_probe_reaches_the_canary_after_tags(monkeypatch, spy):
    calls: list[str] = []

    def _tags_ok(fn, timeout, *, label=""):
        calls.append(label)
        return httpx.Response(200, request=httpx.Request("GET", "http://h"))

    monkeypatch.setattr(extract, "call_with_total_deadline", _tags_ok)
    # the canary then runs through the same name, and "succeeds" here
    assert extract.probe_ollama_idle("http://h", timeout=1.0, model="m", generation_timeout=2.0)
    assert any("tags" in c for c in calls) and any("canary" in c for c in calls)


def test_site_recover_latex_for_image(monkeypatch, spy):
    monkeypatch.setattr(recover, "call_with_total_deadline", spy)
    assert recover.latex_for_image(b"png", timeout=6.5) == ""
    assert [c[0] for c in spy.calls] == [6.5]


def test_site_equation_latex_for_crop(monkeypatch, spy, tmp_path):
    monkeypatch.setattr(ollama_utils, "call_with_total_deadline", spy)
    crop = tmp_path / "c.png"
    crop.write_bytes(b"png")
    assert equation_latex.latex_for_crop(crop, timeout=5.5) == ""
    assert [c[0] for c in spy.calls] == [5.5]


def test_site_gemini_api_figure_engine(monkeypatch, spy):
    from PIL import Image

    monkeypatch.setattr(gemini_api, "call_with_total_deadline", spy)
    engine = gemini_api.OllamaFigureEngine(host="http://h")
    assert engine.is_available() is False
    info = engine.describe_figure(Image.new("RGB", (4, 4)))
    assert "Ollama figure error" in info.description
    assert "TotalDeadlineExceeded" in info.description
    assert [c[0] for c in spy.calls] == [3.0, 120.0]


# -- the table gate: a hanging rung makes the page UNVERIFIED --------------


def _ruled_pdf(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    page = doc.new_page()
    cols = [100, 220, 300, 380]
    rows = [100 + i * 22 for i in range(4)]
    for r, y in enumerate(rows):
        for c, x in enumerate(cols):
            page.insert_text((x + 4, y + 12), f"{r}{c}", fontsize=9)
    for yy in rows:
        page.draw_line((100, yy), (460, yy))
    for xx in cols + [460]:
        page.draw_line((xx, rows[0]), (xx, rows[-1]))
    pdf_path = tmp_path / "doc.pdf"
    doc.save(pdf_path)
    doc.close()
    return pdf_path


_TABLE_MD = (
    "| c0 | c1 | c2 | c3 |\n"
    "| --- | --- | --- | --- |\n"
    "| 10 | 11 | 12 | 13 |\n"
    "| 20 | 21 | 22 | 23 |\n"
    "| 30 | 31 | 32 | 33 |\n"
)


def _gate_run(tmp_path: Path, rung) -> tuple[DocumentState, float]:
    config = PipelineConfig(
        primary_engine=EngineType.QWEN,
        agentic=True,
        judge_backend="heuristic",
        enabled_engines=[EngineType.QWEN],
        quiet=True,
        save_figures=False,
        write_manifest=False,
        table_judge_ladder=True,
    )
    pipeline = UnifiedPipeline(config)
    pipeline._binding_evidence_for_witness = lambda *a, **kw: (None, BindingEvidence.ABSTAIN)
    pipeline._build_table_cell_adjudicator = lambda: None
    with patch.object(DocumentHandle, "__post_init__", lambda self: None):
        handle = DocumentHandle(path=_ruled_pdf(tmp_path), page_count=1)
    state = DocumentState(handle=handle)
    bo = PageOutput(
        page_num=1, text=_TABLE_MD, status=PageStatus.SUCCESS, engine="qwen", audit_passed=True
    )
    start = time.monotonic()
    pipeline._run_table_judge_gate(state, 1, state.pages[1], bo, [rung])
    return state, time.monotonic() - start


def test_gate_hanging_rung_is_unverified_not_a_hang(trickle, tmp_path):
    """Same table, same gate, same rung; only the server differs. A trickling
    server makes the page UNVERIFIED within the deadline and names the call."""
    rung = build_ollama_rung("glm-test:cloud", trickle.url, DEADLINE)
    box: list[tuple] = []
    t = threading.Thread(target=lambda: box.append(_gate_run(tmp_path, rung)), daemon=True)
    t.start()
    t.join(6 * DEADLINE)
    assert not t.is_alive(), "the table gate is hung on a trickling Ollama"
    state, elapsed = box[0]
    events: list[AuditEvent] = [e for e in state.events if e.kind == TABLE_LADDER_UNVERIFIED_KIND]
    assert len(events) == 1, state.events
    blob = repr(events[0].data) + events[0].detail
    assert "total deadline" in blob and "glm-test:cloud" in blob, blob
    assert elapsed < 3 * DEADLINE, elapsed
