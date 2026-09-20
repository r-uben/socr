"""GH-849: a merely-slow peer must not cascade-degrade the whole page.

PR #844 gave the crop-reread reader call a killable process boundary
(GH-798). ``run_killable`` (``core/killable.py``) reclassifies BOTH a real
kill (the deadline expired, no answer -- #172's case) and a peer that merely
answered its own timeout (e.g. ``httpx.ReadTimeout``, still alive) into the
SAME exception type, ``KillableTimeoutError(TimeoutError)``. Since
``concurrent.futures.TimeoutError is TimeoutError`` on py3.11+,
``TableCropExtractor._read_with_deadline``'s
``except concurrent.futures.TimeoutError`` catches both, and
``extract()`` set ``_backend_degraded`` UNCONDITIONALLY on any hit -- so one
slow read dropped every remaining LOCATED table on the page.

This test drives the SAME multi-crop page fixture twice, changing only
whether the injected reader raises a peer-style (``killed=False``) or a
kill-style (``killed=True``) ``KillableTimeoutError`` on its first call, and
asserts the two runs differ exactly as intended:

  - peer-style -> remaining crops ARE attempted (not skipped),
                  ``_backend_degraded`` is NOT set, but the failure is still
                  visible (``_timed_out`` sentinel -- no silent loss).
  - kill-style -> remaining crops are skipped as ``backend_degraded``,
                  ``_backend_degraded`` IS set.

Hermetic: no ollama, no network, no provider -- a plain injected fake
``TableReader``, no agentic path touched.
"""

from __future__ import annotations

import pytest

from socr.core.killable import KillableTimeoutError

fitz = pytest.importorskip("fitz")


def _build_two_table_page(tmp_path):
    """A page with two separately-located ruled tables, stacked vertically."""
    doc = fitz.open()
    page = doc.new_page(width=500, height=500)
    for base_y in (80, 280):
        rows = [base_y, base_y + 22, base_y + 44]
        cols = [100, 220, 300, 380]
        for r, y in enumerate(rows[:-1]):
            page.insert_text((cols[0] + 4, y + 16), f"row{r}", fontsize=9)
            page.insert_text((cols[1] + 4, y + 16), "1.0", fontsize=9)
        for yy in rows:
            page.draw_line((100, yy), (460, yy))
        for xx in cols + [460]:
            page.draw_line((xx, rows[0]), (xx, rows[-1]))
    pdf = tmp_path / "two_tables.pdf"
    doc.save(str(pdf))
    doc.close()
    return pdf


def _boxes_for(pdf):
    from socr.tables.locate import locate_tables

    boxes = locate_tables(fitz.open(pdf)[0])
    if len(boxes) < 2:
        pytest.skip(f"fixture produced {len(boxes)} located boxes, need >= 2")
    return boxes


class _FirstCallTimesOutReader:
    """Raises a KillableTimeoutError on the first `.read()`, then succeeds."""

    timeout = 5.0

    def __init__(self, *, killed: bool) -> None:
        self._killed = killed
        self.calls = 0

    def read(self, *_a, **_k) -> str:
        self.calls += 1
        if self.calls == 1:
            raise KillableTimeoutError("fake:read_crop", self.timeout, killed=self._killed)
        return "| a | b |\n| - | - |\n| 1 | 2 |\n"


def _run(pdf, boxes, *, killed: bool):
    from socr.tables.extract import TableCropExtractor

    reader = _FirstCallTimesOutReader(killed=killed)
    extractor = TableCropExtractor(reader)
    crops = extractor.extract(pdf, 1, boxes, cascade_probe=False)
    return extractor, crops, reader


def test_peer_timeout_does_not_cascade(tmp_path) -> None:
    pdf = _build_two_table_page(tmp_path)
    boxes = _boxes_for(pdf)

    extractor, crops, reader = _run(pdf, boxes, killed=False)

    assert len(crops) == len(boxes), (
        f"expected one sentinel/result per box, got {len(crops)} for {len(boxes)} boxes"
    )
    assert getattr(crops[0], "_timed_out", False), "the peer timeout must still be visible"
    assert getattr(crops[0], "_failed", "") != "backend_degraded", (
        "a peer timeout must not skip remaining crops as backend_degraded"
    )
    # The second box was actually attempted (not skipped): it got the reader's
    # successful reply, not an empty/failed sentinel.
    assert crops[1].markdown.strip(), (
        f"remaining crop was skipped after a peer timeout: {crops[1]!r}"
    )
    assert getattr(extractor, "_backend_degraded", False) is False, (
        "a peer-side timeout must not mark the backend degraded"
    )
    assert reader.calls == len(boxes), "remaining crops must still be attempted"


def test_kill_timeout_still_cascades(tmp_path) -> None:
    pdf = _build_two_table_page(tmp_path)
    boxes = _boxes_for(pdf)

    extractor, crops, _reader = _run(pdf, boxes, killed=True)

    assert len(crops) == len(boxes)
    assert getattr(crops[0], "_timed_out", False), "the kill timeout must still be visible"
    assert getattr(extractor, "_backend_degraded", False) is True, (
        "a real kill must still mark the backend degraded"
    )
    # The remaining box was skipped, not attempted.
    assert getattr(crops[1], "_failed", "") == "backend_degraded", (
        f"remaining crop should be skipped as backend_degraded, got {crops[1]!r}"
    )


def test_the_two_runs_differ_exactly_as_intended(tmp_path) -> None:
    """The core pin: same fixture, same reader shape, only `killed` differs."""
    pdf = _build_two_table_page(tmp_path)
    boxes = _boxes_for(pdf)

    peer_extractor, peer_crops, _r1 = _run(pdf, boxes, killed=False)
    kill_extractor, kill_crops, _r2 = _run(pdf, boxes, killed=True)

    assert getattr(peer_extractor, "_backend_degraded", False) != getattr(
        kill_extractor, "_backend_degraded", False
    ), "killed=True vs killed=False must produce different cascade outcomes"
    assert peer_crops[1].markdown.strip() and not kill_crops[1].markdown.strip(), (
        "only the kill-style run should skip the remaining crop"
    )
