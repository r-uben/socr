"""GH-172: what an abandoned timeout worker actually does to process exit.

`route_page` and `_read_with_deadline` both abandon a stalled
`ThreadPoolExecutor` worker with `shutdown(wait=False)`, and both used to
describe it as a *daemon* thread that the interpreter would discard. It is not,
and the difference decides whether a wedged provider can hold the CLI open.

This measures it in a CHILD process, because the claim is about interpreter
shutdown and cannot be observed from inside the test process.

It also pins the two escapes that look like they should work and do not, so a
future attempt does not rediscover them:

- `t.daemon = True` raises on a running thread
- forcing `_daemonic` and unregistering from `concurrent.futures.thread`'s
  atexit map still does not release exit, because `threading._shutdown` joins
  on locks captured when the thread STARTED

The real fix is bounding the client timeout by the soft one, or a killable
process boundary. Neither is done here; #172 stays open for it. This file
exists so the behaviour is measured rather than assumed, and so a change in
either direction is visible.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import time

# The child sleeps this long inside the abandoned worker. Long enough that a
# process which exits promptly is unambiguous, short enough to stay a test.
_WEDGE_SECONDS = 6.0
_PROMPT = _WEDGE_SECONDS / 2


def _run_child(body: str) -> float:
    """Run *body* in a child interpreter; return its wall-clock lifetime."""
    src = textwrap.dedent(body)
    start = time.monotonic()
    subprocess.run(
        [sys.executable, "-c", src],
        capture_output=True,
        timeout=_WEDGE_SECONDS * 4,
        check=True,
    )
    return time.monotonic() - start


_PREAMBLE = f"""
    import concurrent.futures, time
    from concurrent.futures import thread as _t
    ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(lambda: time.sleep({_WEDGE_SECONDS}))
    try:
        fut.result(timeout=0.2)
    except concurrent.futures.TimeoutError:
        fut.cancel()
"""


def test_an_abandoned_worker_holds_the_process_open() -> None:
    """The behaviour the old comments denied."""
    lifetime = _run_child(_PREAMBLE + "        ex.shutdown(wait=False)\n")

    assert lifetime >= _PROMPT, (
        f"the process exited in {lifetime:.1f}s, before its abandoned worker "
        f"finished ({_WEDGE_SECONDS}s). If that is now true, the GH-172 comments "
        "in route_page and _read_with_deadline are stale and the ticket may be "
        "closable -- check why before deleting this test."
    )


def test_daemonising_the_running_worker_does_not_release_exit() -> None:
    """Both escapes that look like they should work, pinned as not working.

    `t.daemon = True` raises on a running thread; forcing `_daemonic` and
    unregistering from the atexit map leaves `threading._shutdown` joining on
    the lock it captured at thread start.
    """
    lifetime = _run_child(
        _PREAMBLE
        + """
        raised = False
        for t in list(_t._threads_queues):
            try:
                t.daemon = True
            except RuntimeError:
                raised = True
            t._daemonic = True
            _t._threads_queues.pop(t, None)
        assert raised, "setting daemon on a running thread no longer raises"
        ex.shutdown(wait=False)
"""
    )

    assert lifetime >= _PROMPT, (
        f"the process exited in {lifetime:.1f}s after daemonising the worker. "
        "If that now works, GH-172 has a one-line fix and this test should "
        "become the fix's regression guard."
    )
