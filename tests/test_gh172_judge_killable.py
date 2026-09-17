"""GH-172: the actual production judge call is bounded, not just the primitive.

``run_killable`` is proven generically in ``test_gh172_killable_boundary.py``.
This file proves the SITE that wires it in: ``OllamaVisionJudge.judge()``
(``socr/judge/ollama_judge.py``), which is what
``UnifiedPipeline._TimeoutJudge.assess`` calls inside the agentic per-page
loop.

Two levels of evidence, per the ticket:

- ``test_judge_call_is_bounded_and_typed_in_process`` exercises the real
  method directly and checks the typed exception classification the rest of
  the pipeline already keys on (``is_page_judge_timeout``) -- this is the
  fast, reliable regression guard.
- ``test_judge_call_exits_in_a_child_process`` runs the SAME call inside a
  fresh child interpreter (mirroring ``test_gh172_abandoned_worker_exit.py``'s
  pattern) and asserts that process's own wall-clock lifetime is bounded --
  this is "process exit, observed from outside the process", the form of
  evidence #172 asked for, applied to the site this ticket actually changed.

Deferred (see docs/log/2026-09-17_172-gh172-implementation.md): a true
end-to-end ``socr`` CLI child test (stub PDF, forced local ladder + VLM judge
backend, CLI argument parsing) was judged too large for this pass. What is
covered here is the call `_TimeoutJudge.assess` -> `VLMPageJudge.assess` ->
`OllamaVisionJudge.judge()` actually reaches in production, minus the CLI
entrypoint and page routing around it.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from socr.judge.judge import is_page_judge_timeout
from socr.judge.ollama_judge import OllamaVisionJudge

_TRICKLE_INTERVAL_SEC = 0.2
_JUDGE_TIMEOUT_SEC = 1.5
_OUTER_BOUND_SEC = _JUDGE_TIMEOUT_SEC + 4.0  # judge timeout + run_killable's term/kill grace


class _TrickleServer:
    """A loopback server that answers /api/tags (available) and trickles /api/generate."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self.port = self._sock.getsockname()[1]
        self._sock.listen(8)
        self._sock.settimeout(0.5)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except (TimeoutError, socket.timeout):
                continue
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn: socket.socket) -> None:
        try:
            conn.settimeout(5.0)
            buf = b""
            while b"\r\n\r\n" not in buf:
                chunk = conn.recv(4096)
                if not chunk:
                    return
                buf += chunk
            request_line = buf.split(b"\r\n", 1)[0].decode("latin-1")
            if "/api/tags" in request_line:
                body = f'{{"models": [{{"name": "{self._model_name}"}}]}}'.encode()
                conn.sendall(
                    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                    + f"Content-Length: {len(body)}\r\n\r\n".encode()
                    + body
                )
                return
            # /api/generate: headers, then trickle forever — the defeating
            # case (a chunk faster than any read timeout).
            conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n")
            while not self._stop.is_set():
                conn.sendall(b"x")
                time.sleep(_TRICKLE_INTERVAL_SEC)
        except OSError:
            return
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def stop(self) -> None:
        self._stop.set()
        self._sock.close()


@pytest.fixture
def trickle_server():
    server = _TrickleServer(model_name="qwen2-vl:7b")
    yield server
    server.stop()


@pytest.fixture
def page_image(tmp_path) -> Path:
    # judge() only needs bytes it can base64-encode; content is irrelevant.
    img = tmp_path / "page.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\nnot a real png, judge() never decodes it")
    return img


def test_judge_call_is_bounded_and_typed_in_process(trickle_server, page_image) -> None:
    # `is_available()` is not exercised here: it calls `httpx.get`, which the
    # suite's autouse `_table_judge_rungs_are_absent` fixture patches globally
    # (module-attribute patching makes it global, not per-module) to keep the
    # rest of the suite hermetic against a real ollama daemon. `judge()` itself
    # only ever uses `httpx.post`, which that fixture leaves untouched -- the
    # call under test here.
    judge = OllamaVisionJudge(
        model="qwen2-vl:7b", host=trickle_server.url, timeout=_JUDGE_TIMEOUT_SEC
    )

    start = time.monotonic()
    with pytest.raises(TimeoutError) as excinfo:
        judge.judge(page_image, "some ocr text")
    elapsed = time.monotonic() - start

    assert elapsed < _OUTER_BOUND_SEC, (
        f"judge() took {elapsed:.2f}s against a trickling peer; expected it "
        f"bounded by ~{_JUDGE_TIMEOUT_SEC}s + kill grace"
    )
    # This is the classification the rest of the pipeline already keys on
    # (route_page's judge guard, the #713 credentialed-stand-in gate) — the
    # new exception type must satisfy it with no changes there.
    assert is_page_judge_timeout(excinfo.value)


def test_judge_call_exits_in_a_child_process(trickle_server, page_image) -> None:
    """Mirrors test_gh172_abandoned_worker_exit.py's pattern, for the real site."""
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = str(Path(__file__).resolve().parents[1] / "src")
    child_env = dict(os.environ)
    child_env["PYTHONPATH"] = os.pathsep.join([src_dir, tests_dir, child_env.get("PYTHONPATH", "")])

    child_src = textwrap.dedent(
        f"""
        from socr.judge.ollama_judge import OllamaVisionJudge
        judge = OllamaVisionJudge(
            model="qwen2-vl:7b", host={trickle_server.url!r}, timeout={_JUDGE_TIMEOUT_SEC}
        )
        try:
            judge.judge({str(page_image)!r}, "some ocr text")
        except TimeoutError:
            raise SystemExit(0)
        raise SystemExit(3)  # did not time out — unexpected, fail loudly
        """
    )
    start = time.monotonic()
    result = subprocess.run(
        [sys.executable, "-c", child_src],
        env=child_env,
        capture_output=True,
        timeout=_OUTER_BOUND_SEC + 5.0,  # outer safety net; the assertion below is the real bound
    )
    elapsed = time.monotonic() - start

    assert result.returncode == 0, (
        f"child exited {result.returncode}, stderr={result.stderr.decode(errors='replace')[-2000:]}"
    )
    assert elapsed < _OUTER_BOUND_SEC, (
        f"child process lived {elapsed:.2f}s against a trickling peer; "
        f"expected it to exit within ~{_OUTER_BOUND_SEC:.1f}s"
    )
