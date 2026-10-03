"""Submit a call to a DAEMON thread and get a ``concurrent.futures.Future`` (GH-974).

``ThreadPoolExecutor`` workers are never daemon threads, and cannot be made so once
started (GH-172: ``threading._shutdown`` joins on locks captured at thread start).
A deadline site that ABANDONS its worker (``future.result(timeout=...)`` then
``shutdown(wait=False)``) therefore leaves a thread that holds the interpreter open
until the hung call returns, which on a wedged socket is never.

``submit_daemon`` has the executor-free shape those sites need: one daemon thread per
call, a real ``Future`` back, so ``result(timeout=)``, ``done()`` and ``cancel()``
(a no-op once running, as with executors) keep their meaning. An abandoned call is
simply discarded at interpreter exit.
"""

from __future__ import annotations

import concurrent.futures
import threading
from collections.abc import Callable
from typing import Any


def submit_daemon(
    fn: Callable[..., Any], /, *args: Any, **kwargs: Any
) -> concurrent.futures.Future:
    """Run ``fn(*args, **kwargs)`` in a daemon thread; return its Future."""
    future: concurrent.futures.Future = concurrent.futures.Future()

    def _work() -> None:
        if not future.set_running_or_notify_cancel():
            return
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:  # handed to whoever reads the future
            future.set_exception(exc)

    threading.Thread(target=_work, daemon=True, name="socr-daemon-call").start()
    return future
