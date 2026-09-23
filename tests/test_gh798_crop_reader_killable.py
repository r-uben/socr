"""GH-798: the crop-reread reader call is now bounded by a real process kill.

``run_killable`` is proven generically in ``test_gh172_killable_boundary.py``,
and the SAME pattern applied to the judge's httpx call is proven at its own
site in ``test_gh172_judge_killable.py``. This file proves the third and
final named site: ``TableCropExtractor._read_with_deadline``
(``socr/tables/extract.py``), whose worker thread runs
``OllamaTableReader.read`` / ``VllmTableReader.read``.

Step-0 finding recorded in the decision log: of the four sites named across
GH-172/GH-798 (``route_page``, ``_TimeoutJudge.assess``,
``_escalate_table_page``, ``_read_with_deadline``), only this one was
genuinely unkillable -- the other three already bottom out in a subprocess or
(since #796) in ``run_killable`` itself. ``OllamaTableReader.read`` /
``VllmTableReader.read`` made a raw ``httpx.post`` with no process boundary
at all, so a peer that trickles bytes faster than the per-chunk read timeout
(the panel's measured defeat, ``docs/log/2026-09-17_172-design.md``) wedged
the worker thread forever, exactly as ``_read_with_deadline``'s own
pre-existing docstring already said.

Pinned as a DIFFERENCE, not a value (this repo's own testing rule) -- and
pinned at the layer that actually differs. ``_read_with_deadline``'s own
``ThreadPoolExecutor`` + ``future.result(timeout=...)`` wrapper already
raises ``_CropTimeoutError`` on ANY timeout, killable or not (see its
docstring: it only ABANDONS the thread, it does not claim to stop it) --
so asserting that exception through that wrapper would pass identically
before and after this fix, and would not be a guard at all. The thing the
fix actually changes is what happens to the CALL once the wrapper gives up:
before, the abandoned worker thread stays blocked on the peer forever
(measured below); after, ``run_killable`` kills the process making the
call, so ``OllamaTableReader.read`` itself raises within bounded time when
called directly, with no wrapper propping it up.

So this test reads the SAME trickling peer twice in one process: once
through the historical ``ThreadPoolExecutor`` + ``cancel`` +
``shutdown(wait=False)`` pattern (a reconstruction of
``OllamaTableReader.read`` as it stood before this ticket, with no killable
boundary) -- the abandoned thread must still be blocked on the peer well
past the point its own wrapper gave up waiting -- and once through the
real, current ``OllamaTableReader.read`` called DIRECTLY (not through
``_read_with_deadline``) -- it must raise a typed ``TimeoutError`` within a
bounded time on its own.
"""

from __future__ import annotations

import concurrent.futures
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

from socr.tables.extract import OllamaTableReader

# One byte every 0.2s is comfortably faster than any read timeout used below,
# reproducing the panel's own measurement (0.3s trickle vs 1.0s read timeout,
# alive at 12.2s) with margin.
_TRICKLE_INTERVAL_SEC = 0.2
# The reader's own httpx timeout (defence-in-depth only, per the ruling) and,
# for the new path, the deadline `run_killable` is actually given.
_READER_TIMEOUT_SEC = 1.0
# The outer wall-clock deadline `_read_with_deadline` is given directly (this
# repo's `crop_wall_clock_deadline` floor is 30s, too slow for a unit test --
# existing tests, e.g. `test_extractor_releases_within_deadline`, already pass
# an explicit override the same way).
_OUTER_DEADLINE_SEC = 1.5
# run_killable's own term+kill grace (3.0s + 3.0s, see core/killable.py) plus
# headroom for a loaded CI machine.
_KILLABLE_GRACE_SEC = 8.0
#: #857: headroom on the OUTER subprocess ceiling for child interpreter start-up
#: (a cold CI import of socr can take several seconds). Generous on purpose: the
#: child's own exit-code check is what enforces the bounded-time property this
#: test is about; this ceiling only turns a genuine hang into a failure instead of
#: a stuck job, and a tight one produced a naked ``TimeoutExpired`` before the
#: diagnostics could bind.
_CHILD_STARTUP_HEADROOM_SEC = 60.0


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


@pytest.fixture
def crop_image(tmp_path) -> Path:
    img = tmp_path / "crop.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\nnot a real png, the reader never decodes it")
    return img


def test_legacy_thread_wedges_new_reader_raises_typed_timeout(trickle_server, crop_image) -> None:
    # --- Historical path: a plain httpx.post with no process boundary,
    # wrapped exactly as `_read_with_deadline` wraps every reader call
    # (ThreadPoolExecutor + cancel + shutdown(wait=False)). This is
    # `OllamaTableReader.read`'s body as it stood before this ticket. ---
    completed = threading.Event()

    def _legacy_read(_img_path: Path) -> str:
        try:
            resp = httpx.post(
                f"{trickle_server.url}/api/generate",
                json={"model": "m", "prompt": "p", "images": [], "stream": False},
                timeout=_READER_TIMEOUT_SEC,
            )
            resp.raise_for_status()
            return resp.text
        finally:
            # Only reached if the trickling peer ever answers -- on this
            # server it never does, which is exactly the defect: the read
            # timeout above is per-chunk, and the peer supplies a chunk
            # before every interval.
            completed.set()

    ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = ex.submit(_legacy_read, crop_image)
    start = time.monotonic()
    with pytest.raises(concurrent.futures.TimeoutError):
        future.result(timeout=_OUTER_DEADLINE_SEC)
    elapsed_to_abandon = time.monotonic() - start
    future.cancel()
    ex.shutdown(wait=False)
    assert elapsed_to_abandon < _OUTER_DEADLINE_SEC + 1.0

    # The abandoned worker is still blocked on the trickling peer well past
    # the point its own wrapper gave up waiting -- GH-172's defect,
    # reproduced at this site rather than assumed.
    assert not completed.wait(timeout=2.0), (
        "the legacy thread finished -- it should still be wedged on the trickle"
    )

    # --- Current path: the real OllamaTableReader.read, called directly
    # (no ThreadPoolExecutor wrapper propping it up -- that wrapper raises
    # _CropTimeoutError on ANY timeout regardless of this fix, so it cannot
    # discriminate old from new; see the module docstring). This now crosses
    # run_killable, which kills the process actually making the call. ---
    # Run as a CHILD process with its own hard `subprocess.run(timeout=...)`
    # ceiling -- matching the shape `test_gh172_killable_boundary.py` already
    # uses for its expected-to-hang plain-httpx arm
    # (`test_plain_httpx_trickle_defeat_measured_in_a_child`). If `read()`'s
    # body is ever reverted to a direct in-process `httpx.post` (no
    # `run_killable` boundary), calling it in-process here would wedge on the
    # trickling peer forever: CI would hang instead of failing, which is the
    # worst failure shape. The outer ceiling below turns that into a prompt
    # test FAILURE (`subprocess.TimeoutExpired`) instead.
    src_dir = str(Path(__file__).resolve().parents[1] / "src")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([src_dir, env.get("PYTHONPATH", "")])

    driver_src = textwrap.dedent(
        f"""
        import time
        from socr.tables.extract import OllamaTableReader
        reader = OllamaTableReader(
            model="m", host={trickle_server.url!r}, timeout={_READER_TIMEOUT_SEC}
        )
        start = time.monotonic()
        try:
            reader.read({str(crop_image)!r})
        except TimeoutError:
            elapsed = time.monotonic() - start
            raise SystemExit(0 if elapsed < {_READER_TIMEOUT_SEC} + {_KILLABLE_GRACE_SEC} else 1)
        raise SystemExit(2)  # did not raise at all -- also a failure
        """
    )
    start = time.monotonic()
    result = subprocess.run(
        [sys.executable, "-c", driver_src],
        env=env,
        capture_output=True,
        # Outer safety net: the hard ceiling itself. `_KILLABLE_GRACE_SEC`
        # already covers run_killable's own term/kill grace; the extra 5.0s
        # is headroom for child interpreter startup on a loaded machine.
        timeout=_READER_TIMEOUT_SEC + _KILLABLE_GRACE_SEC + _CHILD_STARTUP_HEADROOM_SEC,
        text=True,
    )
    elapsed = time.monotonic() - start

    assert result.returncode == 0, (
        f"child exited {result.returncode} after {elapsed:.2f}s, stderr={result.stderr[-2000:]}"
    )
