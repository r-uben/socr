"""GH-172: a killable process boundary for calls that can wedge an httpx thread.

The problem, measured (``docs/log/2026-09-17_172-design.md``): a soft timeout
built from ``ThreadPoolExecutor`` + ``future.cancel()`` + ``shutdown(wait=False)``
cannot actually stop a running call. Workers are non-daemon and cannot be made
daemon after they start, so the interpreter joins a wedged worker at exit
regardless of what the caller "abandoned" -- the CLI can outlive its own
reported timeout indefinitely. Bounding the client's own timeout does not
close this either: a peer that keeps a response stream open and supplies a
chunk before every read interval (a trickle, including SSE keepalives) never
trips ``httpx``'s per-chunk read timeout, and ``httpx`` has no total-request
deadline to fall back on.

The only mechanism the OS actually honors on a blocked syscall is killing the
process that issued it. This module is that boundary: ``run_killable`` runs
one external call in a fresh ``spawn``-ed child, and on timeout kills the
child's entire process GROUP -- not just the worker -- so a grandchild the
call itself launched (or left behind) dies with it (panel ruling, constraint
2). It is deliberately narrow: the child never owns durable writes (ruling
constraint 1); it computes one value and returns it over a pipe, and nothing
it did is visible to socr's ledger until this function returns that value to
a caller who persists it exactly as every other provider/judge call already
does. Constraint 4 is why the call is a ``CallSpec`` (an import path + plain
picklable args) rather than a closure or bound method: ``spawn`` cannot pickle
either, and pickling ``self`` would carry a stateful client -- open sockets, a
connection pool -- across the boundary this exists to isolate FROM.

``spawn``, not ``fork``: every caller here runs from a thread already
(``ThreadPoolExecutor`` workers, or the orchestrator's main thread), and
fork-after-threads is unsafe, notably on macOS (the platform this was
developed and measured on).

The teardown reaper (``_reap_stragglers``, registered via ``atexit``) is the
panel's absorbed dissent: a document can COMPLETE with a wedged child still
alive if a caller drops a ``run_killable`` process without waiting for it (or
crashes between start and join), and that case has no cascade-halt latch to
trigger a cleanup. ``atexit`` fires only once the interpreter's normal
shutdown sequence has begun, i.e. after every caller's ``main()`` body --
including ``_phase_agentic``'s per-page loop and ``_phase_assemble`` -- has
RETURNED. Page writes in that loop are main-thread and
``_flush_page_fragment`` writes ``.md.tmp`` then renames, so by the time this
reaper runs there is no in-flight write left to truncate. That ordering is
the ruling's load-bearing invariant for treating a forced kill as safe here.
"""

from __future__ import annotations

import atexit
import importlib
import logging
import multiprocessing
import os
import signal
import threading
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Grace periods for the terminate -> kill escalation. Named and derived
# rather than copied from ``engines/vllm_manager.py``'s bare ``timeout=10``
# / ``timeout=5`` (panel ruling: those are magic numbers). A SIGTERM needs
# enough headroom for CPython's normal interpreter teardown in the child
# (atexit handlers, socket/file close) to finish without racing the escalation
# to SIGKILL; that nominal teardown budget is on the order of a second, so the
# grace is set to a small multiple of it. SIGKILL cannot be caught or delayed,
# so its own "grace" only bounds how long the parent waits for the OS to
# actually reap the corpse, not any cooperative shutdown.
_INTERPRETER_TEARDOWN_BUDGET_SEC = 1.0
_GRACE_MULTIPLIER = 3.0
DEFAULT_TERM_GRACE_SEC = _INTERPRETER_TEARDOWN_BUDGET_SEC * _GRACE_MULTIPLIER
DEFAULT_KILL_GRACE_SEC = _INTERPRETER_TEARDOWN_BUDGET_SEC * _GRACE_MULTIPLIER

# Exception TYPE NAMES (not instances -- see ``_child_main``) that mean the
# child's own call hit an ordinary, ALREADY-bounded timeout (a merely slow
# peer unblocking at its own client timeout) rather than surviving because it
# was killed. Reclassified as ``KillableTimeoutError`` in the parent so a
# defence-in-depth client timeout inside the child is not silently downgraded
# to an opaque ``RuntimeError`` and misses the same typed-timeout handling
# (``socr.judge.judge.is_page_judge_timeout``) an outer-deadline kill gets.
_TIMEOUT_TYPE_NAMES = frozenset(
    {
        "TimeoutError",
        "TimeoutException",
        "ReadTimeout",
        "ConnectTimeout",
        "WriteTimeout",
        "PoolTimeout",
        "TimeoutExpired",
    }
)


class KillableTimeoutError(TimeoutError):
    """A killable-process call did not complete before its deadline (and was killed).

    Subclasses ``TimeoutError`` deliberately: ``is_page_judge_timeout``
    already classifies by exception TYPE against a tuple that includes
    ``TimeoutError``, so this slots into the existing typed-timeout surfacing
    (page ``judge_outcome``, document status, metadata) with no new special
    case anywhere that already handles a judge or provider timeout.
    """

    def __init__(self, spec: str, timeout: float) -> None:
        super().__init__(f"{spec!r} did not complete within {timeout:g}s (killed)")
        self.spec = spec
        self.timeout = timeout


@dataclass(frozen=True)
class CallSpec:
    """A picklable description of one external call: an import path + plain args.

    ``func`` is ``"module.path:qualname"``, resolved fresh inside the child --
    never a closure or bound method (see module docstring, ruling constraint
    4). ``args``/``kwargs`` must themselves be picklable plain values (str,
    bytes, numbers, paths as str) -- no open clients, file handles, or PIL
    images sharing memory with the parent.
    """

    func: str
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] | None = None


def _resolve(spec: str) -> Any:
    module_name, sep, qualname = spec.partition(":")
    if not sep:
        raise ValueError(f"CallSpec.func must be 'module:qualname', got {spec!r}")
    obj: Any = importlib.import_module(module_name)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def _child_main(spec: CallSpec, conn) -> None:
    """Entry point run inside the spawned child. Never imported directly."""
    # New session/process-group leader: this process and anything IT spawns
    # can be killed as one unit with `os.killpg` (ruling constraint 2).
    # `multiprocessing.Process` has no `preexec_fn` (unlike `subprocess.Popen`,
    # which `vllm_manager.py` uses for the same purpose), so this is the
    # earliest point in the child it can be done.
    try:
        os.setsid()
    except OSError:
        pass  # already a session leader; nothing to do
    try:
        fn = _resolve(spec.func)
        result = fn(*spec.args, **(spec.kwargs or {}))
        conn.send(("ok", result))
    except BaseException as exc:  # noqa: BLE001 - the child must report SOMETHING
        # Exceptions from arbitrary external calls (httpx, subprocess, ...)
        # are not reliably picklable across the spawn boundary -- they can
        # carry live sockets or request objects. Cross only the TYPE NAME and
        # message; the parent reconstructs a plain exception from that.
        conn.send(("error", type(exc).__name__, str(exc)))
    finally:
        conn.close()


_live_lock = threading.Lock()
# proc -> pgid, captured once at spawn (see `run_killable`) rather than
# re-derived from `os.getpgid(proc.pid)` at escalation time -- see
# `_terminate_then_kill` for why that re-derivation is unsafe.
_live_processes: dict[Any, int] = {}


def _terminate_then_kill(proc: Any, pgid: int | None, term_grace: float, kill_grace: float) -> None:
    """SIGTERM the process GROUP, bounded join, escalate SIGKILL to the SAME group. Best-effort.

    ``pgid`` must be captured by the caller at spawn time, not re-derived here
    via ``os.getpgid(proc.pid)``: once ``proc`` has been reaped that call
    raises ``ProcessLookupError``, so a lookup done only after
    ``proc.is_alive()`` is already False can silently return no pgid at
    exactly the moment escalation needs one (PR #796 review, Astra).

    The SIGKILL step is unconditional on ``pgid`` -- NOT gated on
    ``proc.is_alive()``. The direct child dying from SIGTERM does not mean
    every process in its GROUP did: a descendant that ignores SIGTERM can
    still be alive even though the child socr spawned is gone, and checking
    only the direct child's liveness would leave that descendant running.
    ``killpg`` on an already-empty group is a harmless ``ProcessLookupError``,
    swallowed the same as every other best-effort signal here.
    """
    if pgid is not None:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass
    proc.join(timeout=term_grace)
    if pgid is not None:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass
    proc.join(timeout=kill_grace)


def _reap_stragglers() -> None:
    """Unconditional process-teardown reaper. See module docstring."""
    with _live_lock:
        stragglers = list(_live_processes.items())
        _live_processes.clear()
    for proc, pgid in stragglers:
        if proc.is_alive():
            _terminate_then_kill(proc, pgid, DEFAULT_TERM_GRACE_SEC, DEFAULT_KILL_GRACE_SEC)


atexit.register(_reap_stragglers)


def run_killable(
    spec: CallSpec,
    *,
    timeout: float,
    term_grace: float = DEFAULT_TERM_GRACE_SEC,
    kill_grace: float = DEFAULT_KILL_GRACE_SEC,
) -> Any:
    """Run ``spec`` in a killable child process; raise past ``timeout``.

    On success, returns whatever the resolved callable returned (must be
    picklable). On the child raising, re-raises in the parent -- as
    ``KillableTimeoutError`` when the child's own exception was itself a
    timeout (see ``_TIMEOUT_TYPE_NAMES``), otherwise as a plain
    ``RuntimeError`` carrying the child's exception type and message. On the
    deadline expiring with no answer, kills the child's process group and
    raises ``KillableTimeoutError`` -- the case #172 exists for.

    The answered path escalates too, not just the deadline path: the resolved
    callable sending its answer over the pipe does not mean the CHILD PROCESS
    has exited -- it can leave a non-daemon thread running (its own connection
    pool, a lingering background task) after returning. ``multiprocessing``'s
    own ``atexit`` handler (``util._exit_function``) unconditionally,
    UNBOUNDEDLY joins any ``daemon=False`` child still alive at interpreter
    teardown, regardless of this module's own reaper -- so an answered-but-
    still-alive child reproduces the exact defect #172 exists to close, one
    layer down. ``term_grace`` bounds how long a normally-exiting child is
    given to finish; past that, it gets the identical terminate-then-kill
    escalation the deadline path uses. One escalation, both paths -- the
    result is already captured, so returning it costs nothing.
    """
    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_child_main, args=(spec, child_conn), daemon=False)
    proc.start()
    child_conn.close()  # the child owns the writable end now
    # `_child_main` calls `os.setsid()` as its first action, making itself a
    # new session AND process-group leader -- so its pgid equals its own pid,
    # captured HERE, once, while the process is known to exist. Escalation
    # code must use this captured value, not re-derive it later via
    # `os.getpgid(proc.pid)`, which raises once the process has been reaped
    # (see `_terminate_then_kill`).
    pgid = proc.pid
    with _live_lock:
        _live_processes[proc] = pgid
    try:
        if parent_conn.poll(timeout):
            outcome = parent_conn.recv()
            proc.join(term_grace)
            if proc.is_alive():
                logger.warning(
                    "killable call %r answered but its process is still alive "
                    "after %.1fs — killing child process group",
                    spec.func,
                    term_grace,
                )
                _terminate_then_kill(proc, pgid, term_grace, kill_grace)
            if outcome[0] == "ok":
                return outcome[1]
            _, type_name, message = outcome
            if type_name in _TIMEOUT_TYPE_NAMES:
                raise KillableTimeoutError(spec.func, timeout)
            raise RuntimeError(f"{spec.func} failed in killable child: {type_name}: {message}")
        # Deadline hit with no answer: presumed wedged. Kill the whole group
        # so a grandchild the call spawned (or left running) dies with it.
        logger.warning(
            "killable call %r exceeded %.1fs — killing child process group", spec.func, timeout
        )
        _terminate_then_kill(proc, pgid, term_grace, kill_grace)
        raise KillableTimeoutError(spec.func, timeout)
    finally:
        parent_conn.close()
        with _live_lock:
            _live_processes.pop(proc, None)
