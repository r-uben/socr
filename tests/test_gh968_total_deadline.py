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
from socr.core.ollama_utils import (
    TotalDeadlineExceeded,
    call_with_total_deadline,
    safe_host_label,
)
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
#: A wrapped call must fail within MARGIN x the deadline (generous: CI is loaded);
#: BOUND only stops a regression from hanging the suite.
MARGIN = 4
BOUND = 12 * DEADLINE


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
def _loopback_isolation(monkeypatch):
    """No environment proxy may intercept the loopback server; no stray from an
    earlier test may leak into this one's outstanding-call registry."""
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    ollama_utils._OUTSTANDING.clear()
    yield
    ollama_utils._OUTSTANDING.clear()


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


def test_openai_generation_canary_fails_within_the_deadline_on_a_trickle(trickle):
    """#987 (cubic P2): the vLLM/SGLang liveness canary ran a bare ``httpx.post``, so a
    trickling server held the calling (document) loop. The raw-httpx control above shows
    the trickle really defeats a per-read timeout."""
    finished, elapsed, value = _bounded(
        lambda: extract._openai_generation_canary(trickle.url, "m", DEADLINE), BOUND
    )
    assert finished, "the openai canary is still running past 12x the deadline"
    assert value is False
    assert elapsed < MARGIN * DEADLINE, elapsed


def test_site_openai_models_precondition_and_canary(monkeypatch, spy):
    """#987 (cubic P2): the openai ``/models`` precondition runs under the total
    deadline too. (Not a trickle test: the suite's hermetic stub replaces
    ``httpx.get``, so a GET cannot reach a loopback server; ``httpx.post`` can.)"""
    monkeypatch.setattr(extract, "call_with_total_deadline", spy)
    assert extract.probe_openai_server_idle("http://h/v1", timeout=2.5, model="m") is False
    assert [c[0] for c in spy.calls] == [2.5]
    assert "models" in spy.calls[0][1]
    assert extract._openai_generation_canary("http://h/v1", "m", 3.5) is False
    assert [c[0] for c in spy.calls] == [2.5, 3.5]
    assert "canary" in spy.calls[1][1]


def test_post_chat_fails_within_1_5x_the_deadline_on_a_trickle(trickle):
    finished, elapsed, value = _bounded(
        lambda: REAL_POST_CHAT(trickle.url, PAYLOAD, DEADLINE), BOUND
    )
    assert finished, "_post_chat is still running past 3x the deadline"
    assert isinstance(value, httpx.TimeoutException), value
    assert isinstance(value, TotalDeadlineExceeded)
    assert elapsed < MARGIN * DEADLINE, elapsed
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
        lambda: extract._ollama_generation_canary(trickle.url, "m", DEADLINE), BOUND
    )
    assert finished and value is False
    assert elapsed < MARGIN * DEADLINE, elapsed


def test_urllib_equation_lanes_give_up_on_a_trickle(trickle, tmp_path):
    finished, elapsed, value = _bounded(
        lambda: recover.latex_for_image(b"png", host=trickle.url, timeout=DEADLINE), BOUND
    )
    assert finished and value == ""
    assert elapsed < MARGIN * DEADLINE, elapsed

    crop = tmp_path / "c.png"
    crop.write_bytes(b"png")
    finished, elapsed, value = _bounded(
        lambda: equation_latex.latex_for_crop(crop, host=trickle.url, timeout=DEADLINE),
        BOUND,
    )
    assert finished and value == ""
    assert elapsed < MARGIN * DEADLINE, elapsed


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
    t.join(BOUND)
    assert not t.is_alive(), "the table gate is hung on a trickling Ollama"
    state, elapsed = box[0]
    events: list[AuditEvent] = [e for e in state.events if e.kind == TABLE_LADDER_UNVERIFIED_KIND]
    assert len(events) == 1, state.events
    blob = repr(events[0].data) + events[0].detail
    assert "total deadline" in blob and "glm-test:cloud" in blob, blob
    assert elapsed < MARGIN * DEADLINE, elapsed


# -- abandoned workers are bounded: one per endpoint (review P1) -----------

PASS_JSON = '{"verdict": "PASS", "confidence": "high", "findings": []}'


def test_repeated_overruns_leave_at_most_one_live_worker():
    release = threading.Event()
    started: list[int] = []

    def _stuck():
        started.append(1)
        release.wait(30)

    before = threading.active_count()
    try:
        with pytest.raises(TotalDeadlineExceeded, match="exceeded total deadline"):
            call_with_total_deadline(_stuck, 0.1, label="ep-A")
        for _ in range(6):
            with pytest.raises(TotalDeadlineExceeded, match="previous call still outstanding"):
                call_with_total_deadline(_stuck, 0.1, label="ep-A")
        assert started == [1], "a new worker was started while the stray was outstanding"
        assert threading.active_count() - before <= 1
        # a different endpoint is not blocked by ep-A's stray
        assert call_with_total_deadline(lambda: "ok", 1.0, label="ep-B") == "ok"
    finally:
        release.set()


def test_calls_resume_once_the_stray_finishes():
    release = threading.Event()
    with pytest.raises(TotalDeadlineExceeded):
        call_with_total_deadline(lambda: release.wait(30), 0.1, label="ep-R")
    with pytest.raises(TotalDeadlineExceeded, match="still outstanding"):
        call_with_total_deadline(lambda: "x", 0.1, label="ep-R")
    release.set()
    ollama_utils._OUTSTANDING["ep-R"].join(5)
    assert call_with_total_deadline(lambda: "back", 1.0, label="ep-R") == "back"


def test_next_page_recovers_after_a_hung_page(monkeypatch, tmp_path):
    """Page 1 hangs (UNVERIFIED), page 2 fails fast without a second request,
    and page 3 succeeds once the stray has finished."""
    release = threading.Event()
    posts: list[int] = []

    def _fake_post(url, **kwargs):
        posts.append(1)
        if len(posts) == 1:
            release.wait(30)
        return httpx.Response(
            200,
            json={"message": {"content": PASS_JSON}},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", _fake_post)
    rung = build_ollama_rung("glm-test:cloud", "http://fake-host:11434", 0.2)

    def _unverified(state):
        return [e for e in state.events if e.kind == TABLE_LADDER_UNVERIFIED_KIND]

    # page 1: the call hangs -> UNVERIFIED through the real table gate
    state1, _ = _gate_run(tmp_path / "p1", rung)
    assert len(_unverified(state1)) == 1
    assert state1.pages[1].table_judge_retry_pending is True

    # page 2: also through the gate; fails fast, no second request, still surfaced
    state2, elapsed2 = _gate_run(tmp_path / "p2", rung)
    events2 = _unverified(state2)
    assert len(events2) == 1, "the fail-fast page bypassed the table gate's UNVERIFIED terminal"
    assert "still outstanding" in repr(events2[0].data) + events2[0].detail
    assert state2.pages[1].table_judge_retry_pending is True
    assert len(posts) == 1, "page 2 stacked a second request on the unresponsive endpoint"
    assert elapsed2 < MARGIN * DEADLINE

    # the stray finishes: page 3 reaches the endpoint again and is not UNVERIFIED
    release.set()
    for stray in list(ollama_utils._OUTSTANDING.values()):
        stray.join(5)
    state3, _ = _gate_run(tmp_path / "p3", rung)
    assert len(posts) == 2, "recovery: the page-3 request never reached the endpoint"
    assert _unverified(state3) == []
    assert state3.pages[1].table_judge_retry_pending is False


def test_concurrent_same_label_callers_start_at_most_one_worker():
    release = threading.Event()
    started: list[int] = []
    n = 8
    barrier = threading.Barrier(n)
    outcomes: list[str] = []

    def _stuck():
        started.append(1)
        release.wait(30)

    def _caller():
        barrier.wait()
        try:
            call_with_total_deadline(_stuck, 0.3, label="ep-C")
            outcomes.append("returned")
        except TotalDeadlineExceeded as exc:
            outcomes.append("outstanding" if "still outstanding" in str(exc) else "overrun")

    before = threading.active_count()
    callers = [threading.Thread(target=_caller, daemon=True) for _ in range(n)]
    try:
        for c in callers:
            c.start()
        for c in callers:
            c.join(BOUND)
        assert started == [1], f"{len(started)} workers started for one label"
        assert sorted(outcomes) == ["outstanding"] * (n - 1) + ["overrun"], outcomes
        assert threading.active_count() - before <= 1
    finally:
        release.set()


def test_labels_isolate_endpoints_by_host_and_model(monkeypatch):
    """A hung host/model must not block a different host or model."""
    import json
    import urllib.request

    release = threading.Event()
    calls: list[str] = []

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"response": "x"}).encode()

    def _fake_urlopen(req, timeout=None):
        body = json.loads(req.data)
        key = f"{req.full_url}|{body['model']}"
        calls.append(key)
        if key == "http://hosta/api/generate|m1":
            release.wait(30)
        return _Resp()

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    try:
        assert recover.latex_for_image(b"p", model="m1", host="http://hosta", timeout=0.2) == ""
        # same endpoint again: fails fast, no second request
        assert recover.latex_for_image(b"p", model="m1", host="http://hosta", timeout=0.2) == ""
        assert calls == ["http://hosta/api/generate|m1"]
        # other model, same host / other host, same model: both unaffected
        assert recover.latex_for_image(b"p", model="m2", host="http://hosta", timeout=0.2) != ""
        assert recover.latex_for_image(b"p", model="m1", host="http://hostb", timeout=0.2) != ""
        assert len(calls) == 3
    finally:
        release.set()


def test_every_site_label_carries_host_and_model(monkeypatch, tmp_path):
    labels: list[str] = []

    def _spy(fn, timeout, *, label=""):
        labels.append(label)
        raise TotalDeadlineExceeded(label)

    monkeypatch.setattr(recover, "call_with_total_deadline", _spy)
    monkeypatch.setattr(ollama_utils, "call_with_total_deadline", _spy)
    monkeypatch.setattr(gemini_api, "call_with_total_deadline", _spy)
    monkeypatch.setattr(extract, "call_with_total_deadline", _spy)
    monkeypatch.setattr(table_rung_ollama, "call_with_total_deadline", _spy)

    recover.latex_for_image(b"p", model="MODEL", host="http://HOST")
    crop = tmp_path / "c.png"
    crop.write_bytes(b"p")
    equation_latex.latex_for_crop(crop, model="MODEL", host="http://HOST")
    engine = gemini_api.OllamaFigureEngine(model="MODEL", host="http://HOST")
    engine.is_available()
    from PIL import Image

    engine.describe_figure(Image.new("RGB", (2, 2)))
    extract._ollama_generation_canary("http://HOST", "MODEL", 1.0)
    extract.probe_ollama_idle("http://HOST", timeout=1.0, model="MODEL")
    ollama_rung_reachable("MODEL", "http://HOST", timeout=1.0)
    with pytest.raises(TotalDeadlineExceeded):
        REAL_POST_CHAT("http://HOST", {"model": "MODEL"}, 1.0)

    assert len(labels) == 8, labels
    for label in labels:
        assert "HOST" in label, f"label lacks the host: {label!r}"
    # the model-specific calls name the model too
    for label in (labels[0], labels[1], labels[3], labels[4], labels[7]):
        assert "MODEL" in label, f"label lacks the model: {label!r}"


# -- labels never carry URL userinfo (cubic P2) -----------------------------


def _site_labels(monkeypatch, tmp_path, host: str) -> list[str]:
    labels: list[str] = []

    def _spy(fn, timeout, *, label=""):
        labels.append(label)
        raise TotalDeadlineExceeded(label)

    for mod in (recover, ollama_utils, gemini_api, extract, table_rung_ollama):
        monkeypatch.setattr(mod, "call_with_total_deadline", _spy)
    recover.latex_for_image(b"p", model="MODEL", host=host)
    crop = tmp_path / "c.png"
    crop.write_bytes(b"p")
    equation_latex.latex_for_crop(crop, model="MODEL", host=host)
    from PIL import Image

    engine = gemini_api.OllamaFigureEngine(model="MODEL", host=host)
    engine.is_available()
    engine.describe_figure(Image.new("RGB", (2, 2)))
    extract._ollama_generation_canary(host, "MODEL", 1.0)
    extract.probe_ollama_idle(host, timeout=1.0, model="MODEL")
    ollama_rung_reachable("MODEL", host, timeout=1.0)
    with pytest.raises(TotalDeadlineExceeded):
        REAL_POST_CHAT(host, {"model": "MODEL"}, 1.0)
    return labels


def test_safe_host_label_strips_userinfo_keeps_scheme_host_port():
    assert safe_host_label("http://alice:s3cret@gpu1:11434/") == "http://gpu1:11434"
    assert safe_host_label("https://tok@h.example") == "https://h.example"
    assert safe_host_label("http://h:11434?token=abc#frag") == "http://h:11434"
    assert safe_host_label("http://h:11434") == "http://h:11434"
    assert safe_host_label("alice:s3cret@gpu1:11434") == "gpu1:11434"


def test_no_site_label_leaks_credentials_and_userinfo_does_not_split_endpoints(
    monkeypatch, tmp_path
):
    with_creds = _site_labels(monkeypatch, tmp_path, "http://alice:s3cret@HOST:11434")
    other_creds = _site_labels(monkeypatch, tmp_path, "http://bob:hunter2@HOST:11434")
    plain = _site_labels(monkeypatch, tmp_path, "http://HOST:11434")
    assert len(with_creds) == 8
    for label in with_creds + other_creds:
        for secret in ("alice", "s3cret", "bob", "hunter2", "@"):
            assert secret not in label, f"{secret!r} leaked into {label!r}"
    # Sites 5 and 6 go through ``resolve_ollama_host``, which (pre-existing,
    # unrelated to this ticket) mangles a userinfo host before the label is built;
    # the userinfo-invariance claim is for the label helper's own sites.
    own = [0, 1, 2, 3, 4, 7]
    assert [with_creds[i] for i in own] == [other_creds[i] for i in own]
    assert [with_creds[i] for i in own] == [plain[i] for i in own]


def test_figure_is_available_is_inconclusive_when_the_probe_fails_fast(monkeypatch):
    """A still-draining earlier probe must not turn a healthy daemon into False."""
    tags = httpx.Response(
        200,
        json={"models": [{"name": "MODEL"}]},
        request=httpx.Request("GET", "http://h/api/tags"),
    )
    gets: list[int] = []

    def _get(url, **kw):
        gets.append(1)
        return tags

    monkeypatch.setattr(httpx, "get", _get)
    engine = gemini_api.OllamaFigureEngine(model="MODEL", host="http://u:p@h")
    assert engine.is_available() is True

    release = threading.Event()
    stray = threading.Thread(target=lambda: release.wait(30), daemon=True)
    stray.start()
    ollama_utils._OUTSTANDING["ollama figure http://h/api/tags"] = stray
    try:
        assert engine.is_available() is True, "fail-fast was read as 'daemon unavailable'"
        assert len(gets) == 1, "a second probe was stacked on the draining one"
    finally:
        release.set()
