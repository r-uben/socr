"""GH-172: the killable process boundary actually bounds a wedged/trickling call.

Pairs with ``tests/test_gh172_abandoned_worker_exit.py`` (which pins that
THREAD abandonment cannot do this) and exercises the mechanism this ticket
adds instead: ``socr.core.killable.run_killable``.

The stub server below reproduces the panel's own measurement
(``docs/log/2026-09-17_172-design.md``): a peer that sends one byte per
``_TRICKLE_INTERVAL`` seconds, well under the read timeout, never trips
``httpx``'s per-chunk read timeout and keeps a plain HTTP call open forever.
That is the load-bearing test mode -- a silent/no-response peer is already
caught by the read timeout and would pass even without this ticket's fix.

Every callable crossing the killable boundary must be a top-level, importable
function (``CallSpec`` constraint) -- they live at module level here so a
freshly spawned child can resolve ``tests.test_gh172_killable_boundary:name``.
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

import httpx
import pytest

from socr.core.killable import CallSpec, KillableTimeoutError, run_killable

# One byte every 0.2s is comfortably faster than any read timeout used below
# (>= 1.0s), reproducing the panel's measurement (0.3s trickle vs 1.0s read
# timeout, alive at 12.2s) with margin.
_TRICKLE_INTERVAL_SEC = 0.2
# Long enough that a run_killable call which actually bounds the trickle is
# unambiguous; short enough to stay a fast test.
_OUTER_TIMEOUT_SEC = 1.5
_GRACE_SEC = 2.0  # generous upper bound on run_killable's own term/kill joins


class _TrickleServer:
    """A loopback server that sends HTTP headers, then one byte forever."""

    def __init__(self) -> None:
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
    server = _TrickleServer()
    yield server
    server.stop()


# ---------------------------------------------------------------------------
# Module-level (top-level, importable) callables for CallSpec.
# ---------------------------------------------------------------------------


def _get(url: str, timeout: float) -> str:
    resp = httpx.get(url, timeout=timeout)
    return resp.text


def _sleep_forever(seconds: float) -> str:
    time.sleep(seconds)
    return "done"


def _answer_then_leave_a_thread_running(hold_seconds: float) -> str:
    """Returns immediately but leaves a non-daemon THREAD alive in the child.

    Reproduces the answered-but-still-alive case (rev-172): the resolved
    callable sending its result over the pipe does not mean the CHILD
    PROCESS has exited -- a lingering non-daemon thread (its own connection
    pool, a background task) holds the interpreter open well past that.
    """
    thread = threading.Thread(target=time.sleep, args=(hold_seconds,), daemon=False)
    thread.start()
    return "done"


# ---------------------------------------------------------------------------
# The measurement itself: plain httpx against the trickle stub does NOT raise.
# ---------------------------------------------------------------------------


# A plain in-process httpx call against the trickle stub is EXPECTED to hang
# past _OUTER_TIMEOUT_SEC — that is the defect (re-verifying the panel's own
# measurement). It is run as a CHILD process with its own hard ceiling so a
# regression here bounds a subprocess timeout, not the whole test suite.
def test_plain_httpx_trickle_defeat_measured_in_a_child(trickle_server) -> None:
    src = textwrap.dedent(
        f"""
        import httpx, time
        start = time.monotonic()
        try:
            httpx.get({trickle_server.url!r}, timeout=httpx.Timeout({_OUTER_TIMEOUT_SEC}))
        except httpx.TimeoutException:
            raise SystemExit(1)  # would mean the trickle FAILED to defeat the timeout
        raise SystemExit(0)
        """
    )
    with pytest.raises(subprocess.TimeoutExpired):
        subprocess.run(
            [sys.executable, "-c", src],
            timeout=_OUTER_TIMEOUT_SEC + 1.0,
            capture_output=True,
        )


# ---------------------------------------------------------------------------
# run_killable bounds it.
# ---------------------------------------------------------------------------


def test_run_killable_bounds_a_trickling_call(trickle_server) -> None:
    spec = CallSpec(func=f"{__name__}:_get", args=(trickle_server.url, _OUTER_TIMEOUT_SEC + 60.0))
    start = time.monotonic()
    with pytest.raises(KillableTimeoutError):
        run_killable(spec, timeout=_OUTER_TIMEOUT_SEC)
    elapsed = time.monotonic() - start
    assert elapsed < _OUTER_TIMEOUT_SEC + _GRACE_SEC, (
        f"run_killable took {elapsed:.2f}s to raise; the deadline "
        f"({_OUTER_TIMEOUT_SEC}s) plus its own term/kill grace "
        f"({_GRACE_SEC}s) should bound it"
    )


def test_run_killable_returns_promptly_on_success() -> None:
    spec = CallSpec(func=f"{__name__}:_sleep_forever", args=(0.05,))
    start = time.monotonic()
    assert run_killable(spec, timeout=5.0) == "done"
    assert time.monotonic() - start < 5.0


# ---------------------------------------------------------------------------
# The kill is process-GROUP wide (ruling constraint 2): a grandchild the
# killable call spawns must die too, not just the direct child.
# ---------------------------------------------------------------------------


def _wedge_with_grandchild(pidfile: str, url: str, timeout: float) -> str:
    proc = subprocess.Popen(["sleep", "100"])
    with open(pidfile, "w") as f:
        f.write(str(proc.pid))
    return _get(url, timeout)


def test_kill_is_process_group_wide(trickle_server, tmp_path, monkeypatch) -> None:
    # The spawned child resolves CallSpec.func via `importlib.import_module`,
    # so it needs this test module on its own sys.path — this file's
    # directory is not on PYTHONPATH by default (only src/ is).
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    existing = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, [tests_dir, existing])))

    pidfile = tmp_path / "grandchild.pid"
    spec = CallSpec(
        func="test_gh172_killable_boundary:_wedge_with_grandchild",
        args=(str(pidfile), trickle_server.url, _OUTER_TIMEOUT_SEC + 60.0),
    )
    with pytest.raises(KillableTimeoutError):
        run_killable(spec, timeout=_OUTER_TIMEOUT_SEC)

    # Give the OS a moment to finish reaping after SIGKILL.
    deadline = time.monotonic() + _GRACE_SEC
    grandchild_pid = int(pidfile.read_text().strip())
    alive = True
    while time.monotonic() < deadline:
        try:
            os.kill(grandchild_pid, 0)
        except ProcessLookupError:
            alive = False
            break
        time.sleep(0.1)
    assert not alive, (
        f"grandchild pid {grandchild_pid} was still alive after the killable "
        "boundary's deadline + grace — the kill did not reach the process group"
    )


# ---------------------------------------------------------------------------
# The ANSWERED path escalates too (rev-172, PR #796 review): a child that
# sends its result but leaves a non-daemon thread running is not gone. This
# must be observed as PROCESS wall-clock from OUTSIDE the driving process --
# a test asserting only the return value passes even while the process hangs
# at exit, because that hang happens in `multiprocessing.util._exit_function`
# (an unconditional, unbounded join of any daemon=False child still alive at
# interpreter teardown), a path `run_killable`'s own deadline logic never
# inspects.
# ---------------------------------------------------------------------------

# Long enough that "the driving process happened to exit before the leaked
# thread finished" cannot be mistaken for the fix working.
_LEAKED_THREAD_HOLD_SEC = 30.0
# Generous bound on how long the FIX should take to notice and reap the
# still-alive child: term_grace (this test's own) + the escalation's own
# kill_grace, with margin.
_ANSWERED_PATH_OUTER_BOUND_SEC = 10.0


def test_run_killable_reaps_an_answered_but_still_alive_child() -> None:
    """The fix: `run_killable` must not let an answered child outlive it.

    Driven as a CHILD-of-the-test process (mirrors
    `test_judge_call_exits_in_a_child_process`) because the defect is that
    process's own exit that hangs, not anything observable by inspecting a
    return value from inside pytest.
    """
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = str(Path(__file__).resolve().parents[1] / "src")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([src_dir, tests_dir, env.get("PYTHONPATH", "")])

    driver_src = textwrap.dedent(
        f"""
        from socr.core.killable import CallSpec, run_killable
        spec = CallSpec(
            func="test_gh172_killable_boundary:_answer_then_leave_a_thread_running",
            args=({_LEAKED_THREAD_HOLD_SEC},),
        )
        result = run_killable(spec, timeout=5.0, term_grace=0.5, kill_grace=2.0)
        assert result == "done", result
        """
    )
    start = time.monotonic()
    result = subprocess.run(
        [sys.executable, "-c", driver_src],
        env=env,
        capture_output=True,
        timeout=_ANSWERED_PATH_OUTER_BOUND_SEC + 5.0,  # outer safety net only
        text=True,
    )
    elapsed = time.monotonic() - start

    assert result.returncode == 0, (
        f"driver exited {result.returncode}, stderr={result.stderr[-2000:]}"
    )
    assert elapsed < _ANSWERED_PATH_OUTER_BOUND_SEC, (
        f"driving process lived {elapsed:.2f}s; expected the answered-but-"
        f"still-alive child to be reaped well under the "
        f"{_LEAKED_THREAD_HOLD_SEC}s its leaked thread would otherwise hold "
        "the interpreter open for -- run_killable's answered path must "
        "escalate a still-alive child the same as its deadline path does"
    )
