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

import multiprocessing
import os
import signal
import socket
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import httpx
import pytest
from multiprocessing import connection as mp_connection

from socr.core.killable import (
    DEFAULT_KILL_GRACE_SEC,
    DEFAULT_TERM_GRACE_SEC,
    CallSpec,
    KillableTimeoutError,
    run_killable,
)

# One byte every 0.2s is comfortably faster than any read timeout used below
# (>= 1.0s), reproducing the panel's measurement (0.3s trickle vs 1.0s read
# timeout, alive at 12.2s) with margin.
_TRICKLE_INTERVAL_SEC = 0.2
# Long enough that a run_killable call which actually bounds the trickle is
# unambiguous; short enough to stay a fast test.
_OUTER_TIMEOUT_SEC = 1.5


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


def _bare_spawn_roundtrip_sec() -> float:
    """Wall time of a bare ``multiprocessing`` spawn child that imports this module.

    Deliberately NOT routed through ``run_killable`` (GH-991 review): calibrating
    with the code under test would let an overhead regression inflate its own
    allowance. Process start-up is the load-dependent part of the wall time, so a
    bound that ignores it is a coin flip on a busy host.
    """
    ctx = multiprocessing.get_context("spawn")
    start = time.monotonic()
    proc = ctx.Process(target=_sleep_forever, args=(0.0,))
    proc.start()
    proc.join()
    return time.monotonic() - start


def test_run_killable_bounds_a_trickling_call(trickle_server) -> None:
    spec = CallSpec(func=f"{__name__}:_get", args=(trickle_server.url, _OUTER_TIMEOUT_SEC + 60.0))
    # run_killable's documented worst case: the deadline, then SIGTERM and SIGKILL
    # each given their full grace; plus one independently measured process start-up.
    # This is the broad integration bound (an unbounded trickle never returns). The
    # exactness of the deadline itself is pinned by the test below, deterministically.
    bound = (
        _OUTER_TIMEOUT_SEC
        + DEFAULT_TERM_GRACE_SEC
        + DEFAULT_KILL_GRACE_SEC
        + _bare_spawn_roundtrip_sec()
    )
    start = time.monotonic()
    with pytest.raises(KillableTimeoutError):
        run_killable(spec, timeout=_OUTER_TIMEOUT_SEC)
    elapsed = time.monotonic() - start
    assert elapsed < bound, (
        f"run_killable took {elapsed:.2f}s to raise; the deadline "
        f"({_OUTER_TIMEOUT_SEC}s) plus its own term/kill grace and one process "
        f"start-up ({bound:.2f}s in all) should bound it"
    )


def test_run_killable_hands_the_requested_deadline_to_poll_unchanged(monkeypatch) -> None:
    """Deterministic pin on the deadline (GH-991 review): the wide wall-clock bound
    above tolerates a deadline stretched a few times over, so check the value that
    reaches ``Connection.poll`` instead of timing it."""
    seen: list[float] = []
    real_poll = mp_connection.Connection.poll

    def spy(self, timeout=0.0):
        seen.append(timeout)
        return real_poll(self, timeout)

    monkeypatch.setattr(mp_connection.Connection, "poll", spy)
    requested = 37.25  # distinctive, so a scaled or substituted value cannot match
    spec = CallSpec(func=f"{__name__}:_sleep_forever", args=(0.0,))
    assert run_killable(spec, timeout=requested) == "done"
    assert seen == [requested], f"run_killable polled with {seen}, requested {requested}"


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
    # Atomic publish: a reader polling for the file must never see it half-written.
    tmp = pidfile + ".tmp"
    with open(tmp, "w") as f:
        f.write(str(proc.pid))
    os.replace(tmp, pidfile)
    return _get(url, timeout)


# Upper limit for waiting on an EVENT (a file appearing, a process exiting). Not a
# performance claim: events are polled, so this only bounds how long a genuinely
# stuck test may hang; it must exceed any scheduler delay a loaded host produces.
_EVENT_DEADLINE_SEC = 60.0


def _wait_for(predicate, *, what: str, fail: bool = True) -> bool:
    deadline = time.monotonic() + _EVENT_DEADLINE_SEC
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    if fail:
        pytest.fail(f"timed out after {_EVENT_DEADLINE_SEC:g}s waiting for {what}")
    return False


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
    # GH-991: ``run_killable``'s deadline starts at spawn, and a spawned child
    # needs a load-dependent time to import this module and fork the grandchild.
    # If the deadline fired first, the kill landed before there was a grandchild
    # (no pidfile, or a group with nothing in it), so the test measured the
    # machine, not the kill. Hold the deadline's clock back until the grandchild
    # is published: ``run_killable``'s single ``parent_conn.poll(timeout)`` waits
    # for the pidfile first, then runs unchanged. The kill path is untouched.
    real_poll = mp_connection.Connection.poll

    pidfile_appeared: list[bool] = []

    def poll_once_grandchild_exists(self, timeout=0.0):
        # Never raise from here: that would skip run_killable's kill path and leak
        # the child. If the pidfile never shows, run the real poll anyway so
        # run_killable still kills the group, and fail below with a clear message.
        pidfile_appeared.append(
            _wait_for(pidfile.exists, what=f"grandchild pidfile {pidfile}", fail=False)
        )
        return real_poll(self, timeout)

    monkeypatch.setattr(mp_connection.Connection, "poll", poll_once_grandchild_exists)

    grandchild_pid = None
    try:
        with pytest.raises(KillableTimeoutError):
            run_killable(spec, timeout=_OUTER_TIMEOUT_SEC)
        assert pidfile_appeared == [True], (
            f"grandchild pidfile {pidfile} did not appear within {_EVENT_DEADLINE_SEC:g}s; "
            "the child never reached the point under test"
        )
        grandchild_pid = int(pidfile.read_text().strip())

        def gone() -> bool:
            try:
                os.kill(grandchild_pid, 0)
            except ProcessLookupError:
                return True
            return False

        # SIGKILL is synchronous but the orphaned grandchild is reaped by init
        # asynchronously: poll for that, with a deadline far past any scheduler delay.
        assert _wait_for(gone, what="grandchild exit", fail=False), (
            f"grandchild pid {grandchild_pid} was still alive after the killable "
            "boundary's deadline + grace — the kill did not reach the process group"
        )
    finally:
        # Never leak the 100s sleeper, whatever the verdict.
        if grandchild_pid is None and pidfile.exists():
            grandchild_pid = int(pidfile.read_text().strip())
        if grandchild_pid is not None:
            try:
                os.kill(grandchild_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


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


# ---------------------------------------------------------------------------
# The SIGKILL escalation is group-scoped and unconditional (PR #796 review,
# Astra): a direct child dying from SIGTERM does not mean every process in
# its GROUP did -- a descendant that ignores SIGTERM can still be alive even
# though `proc.is_alive()` is already False. This pins the deterministic
# SHAPE of the fix (unconditional SIGKILL on a pgid captured at spawn, not
# re-derived from a possibly-already-reaped pid) -- not the timing-dependent
# survival case itself, which Astra could not reliably reproduce and which
# this ticket does not claim to pin.
# ---------------------------------------------------------------------------


def test_terminate_then_kill_escalates_to_sigkill_even_if_the_direct_child_already_exited(
    monkeypatch,
) -> None:
    from socr.core import killable

    calls: list[tuple[int, int]] = []
    monkeypatch.setattr(killable.os, "killpg", lambda pgid, sig: calls.append((pgid, sig)))

    class _AlreadyExitedProc:
        def is_alive(self) -> bool:
            return False  # the direct child died from SIGTERM

        def join(self, timeout: float | None = None) -> None:
            pass

    killable._terminate_then_kill(_AlreadyExitedProc(), pgid=4242, term_grace=0.01, kill_grace=0.01)

    assert calls == [(4242, signal.SIGTERM), (4242, signal.SIGKILL)], (
        "SIGKILL must fire on the captured pgid regardless of the direct "
        f"child's own liveness (a group descendant can outlive it); got {calls}"
    )
