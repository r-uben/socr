"""GH-520 part 1: a table count recorded BEFORE reconstruction runs.

`docs/log/2026-09-01_p2-structure-class-floor.md` swept for a signal that could
prove a floored page's regional splice covered every table, and found none:

| candidate | verdict |
| --- | --- |
| `find_tables()` in `_detect_tables` | reduced to a bool, count discarded |
| `find_tables()` in `extract_structured` | consumed inline, filtered by reconstruction success |
| `has_tables` | a bool |
| `native_table_region_count` / `_identities` | post-reconstruction, the circular signal |

So P2 shipped the whole-page floor and withheld the page's prose. `native_
table_region_count` is produced by the same GFM parser it would validate: a
sibling table that failed reconstruction is simply absent from it, and a
collapsed grid could ship inside text labelled "preserved prose".

This is that missing signal. What matters about it is not the number but WHERE
it comes from: `find_tables()`, at detection time, unaware of whether anything
later parsed. `test_the_count_does_not_follow_the_reconstruction_signal` is the
whole point of the ticket -- it is what `native_table_region_count` cannot do.

Consuming it (the case-(iii) regional splice) is deliberately not in this
change.
"""

from __future__ import annotations

from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")

from socr.core.born_digital import BornDigitalDetector  # noqa: E402


def _ruled_table(page, top: float, *, rows: int = 3, cols: int = 3) -> None:
    left, right = 72.0, 520.0
    height = 22.0 * rows
    page.draw_rect(fitz.Rect(left, top, right, top + height), color=(0, 0, 0), width=1.0)
    for r in range(1, rows):
        y = top + r * (height / rows)
        page.draw_line(fitz.Point(left, y), fitz.Point(right, y), color=(0, 0, 0), width=0.8)
    for c in range(1, cols):
        x = left + c * ((right - left) / cols)
        page.draw_line(fitz.Point(x, top), fitz.Point(x, top + height), color=(0, 0, 0), width=0.8)
    for r in range(rows):
        for c in range(cols):
            page.insert_text(
                (left + 8 + c * ((right - left) / cols), top + 15 + r * (height / rows)),
                f"{r}{c}.{r + c}",
                fontsize=9,
            )


def _pdf(tmp_path: Path, name: str, *, tables: int, prose: bool = True) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    pdf = tmp_path / name
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    if prose:
        for i in range(8):
            page.insert_text(
                (72, 60 + i * 14),
                "Surrounding prose that must be attributable to the page, not to a table.",
                fontsize=9,
            )
    for t in range(tables):
        _ruled_table(page, 200.0 + t * 200.0)
    doc.save(str(pdf))
    doc.close()
    return pdf


def _assess(pdf: Path):
    return BornDigitalDetector().detect(pdf).pages[0]


def test_a_ruled_table_is_counted_with_a_bbox(tmp_path: Path) -> None:
    page = _assess(_pdf(tmp_path / "one", "one.pdf", tables=1))

    assert page.detected_table_count == 1, (
        f"the detector saw {page.detected_table_count} tables on a one-table page; "
        "the signal is not measuring find_tables()"
    )
    assert len(page.detected_table_bboxes) == 1
    x0, y0, x1, y1 = page.detected_table_bboxes[0]
    assert x1 > x0 and y1 > y0, f"degenerate bbox {page.detected_table_bboxes[0]}"


def test_two_tables_are_counted_separately(tmp_path: Path) -> None:
    """The shape the floor guard needs: siblings, each with its own region."""
    page = _assess(_pdf(tmp_path / "two", "two.pdf", tables=2))

    assert page.detected_table_count == 2, (
        f"two ruled tables were detected as {page.detected_table_count}; a guard "
        "comparing parsed blocks against this count would mis-fire"
    )
    assert len(page.detected_table_bboxes) == 2
    first, second = sorted(page.detected_table_bboxes, key=lambda b: b[1])
    assert second[1] > first[3] or second[1] > first[1], (
        f"the two regions overlap or were merged: {page.detected_table_bboxes}"
    )


def test_a_page_with_no_table_counts_none(tmp_path: Path) -> None:
    """Difference control. Without it a detector that returned a constant would
    satisfy both tests above."""
    page = _assess(_pdf(tmp_path / "none", "none.pdf", tables=0))

    assert page.detected_table_count == 0
    assert page.detected_table_bboxes == []


def test_the_count_does_not_follow_the_reconstruction_signal(tmp_path: Path) -> None:
    """The point of the whole ticket, pinned as a DIVERGENCE.

    `native_table_region_count` is produced by the GFM parser, so a table that
    fails reconstruction vanishes from it -- and a coverage check built on it
    concludes the page was fully covered when it was not. On a clean fixture the
    two numbers agree, which proves nothing; what has to be shown is that they
    come from different places.

    So the reconstruction side channel is made to report nothing, exactly as it
    would for a page whose regions all failed to parse. The parser-derived count
    must follow it to zero and the detection-level count must not move at all.
    """
    pdf = _pdf(tmp_path / "indep", "indep.pdf", tables=2)

    baseline = _assess(pdf)
    assert baseline.detected_table_count == 2, "the baseline itself is wrong"
    assert baseline.native_table_region_count == 2, (
        "the parser-derived count does not start out agreeing, so making it "
        "diverge below would not be measuring a divergence"
    )

    real_signals = BornDigitalDetector._assess_page_signals

    def _no_regions_reconstructed(self, page, page_num, direction):
        assessment = real_signals(self, page, page_num, direction)
        # What a page whose every region failed to parse leaves behind.
        self._last_extraction_table_count = 0
        self._last_extraction_region_identities = []
        return assessment

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(BornDigitalDetector, "_assess_page_signals", _no_regions_reconstructed)
        diverged = _assess(pdf)

    assert diverged.native_table_region_count == 0, (
        "the parser-derived count did not follow the reconstruction signal, so "
        "the seam being patched is not the one that feeds it"
    )
    assert diverged.detected_table_count == 2, (
        "the detection-level count followed reconstruction to zero, so it is "
        "not independent of the parser and cannot validate its output"
    )
    assert len(diverged.detected_table_bboxes) == 2


def test_has_tables_is_unchanged_by_the_new_signal(tmp_path: Path) -> None:
    """`_detect_tables` now reads the same helper, so a bug there would move
    routing for every page in the corpus. Pass 1 must still say yes to a ruled
    table and the borderless pass must still be reachable."""
    ruled = _assess(_pdf(tmp_path / "ht", "ht.pdf", tables=1))
    assert ruled.has_tables is True

    plain = _assess(_pdf(tmp_path / "hf", "hf.pdf", tables=0))
    assert plain.has_tables is False, (
        "a prose page is now routed as a table page; the pass-1 rewrite changed "
        "detection, not just recording"
    )


def test_a_table_without_a_readable_bbox_still_counts_but_contributes_no_box() -> None:
    """The asymmetry, stated in the field's own docstring and pinned here.

    Dropping a bbox-less table from the COUNT would quietly change
    `has_tables`, which has nothing to do with this ticket. Counting it while
    contributing no box makes `count > len(bboxes)`, which a consumer mapping
    blocks onto regions must read as fail-closed -- a table nobody can point at.
    """

    class _Table:
        def __init__(self, bbox):
            self.bbox = bbox

    class _Page:
        def find_tables(self):
            class _R:
                tables = [_Table((0.0, 0.0, 10.0, 10.0)), _Table(None), _Table("nonsense")]

            return _R()

    count, boxes = BornDigitalDetector._detect_table_regions(_Page())
    assert count == 3, "a table with an unusable bbox stopped being a table"
    assert boxes == [(0.0, 0.0, 10.0, 10.0)]
    assert count > len(boxes), (
        "the mismatch a consumer fails closed on is not observable, so the "
        "fail-closed branch can never be reached"
    )


def test_a_detector_failure_is_no_evidence_of_tables() -> None:
    """Best effort, and empty on failure -- the absence-of-evidence precedent
    the rest of this module follows. A raising page must not become a table
    page, and must not become a page with a confident zero either: both halves
    are empty, so any consumer sees the fail-closed shape."""

    class _Page:
        def find_tables(self):
            raise RuntimeError("table detection died")

    assert BornDigitalDetector._detect_table_regions(_Page()) == (0, [])
