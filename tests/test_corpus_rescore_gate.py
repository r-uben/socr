"""#123 — the corpus gate. Real pages, not fixtures.

Every defect in TICKET-B2 was found by a probe *outside* the unit fixtures, and the
suite was green at 1384, 1388, 1392 and 1395 while the lane rule was broken
underneath. Synthetic geometry only ever tests the shapes its author already thought
of: 144 synthetic combinations passed while the reference document failed on 16 of
its 18 scorable pages, with the clusterer emitting 2x-5x more lanes than the widest
row has values.

This module scores the **preserved OCR run** — real emitted markdown against the real
PDF's native text layer — and holds the result to a committed baseline. It is a
monotone-improvement gate, deliberately not an absolute rule: there is no defensible
constant for "how many lanes a table may have", so instead nothing is allowed to get
worse than the recorded state, and the baseline is re-recorded (down, never up) as
fixes land.

The preserved run is outside the repo and CI does not have it, so these tests skip
when it is absent. That is a real limitation — this gate protects local work, which
is where every one of these regressions was actually caught. Do not "fix" the skip by
weakening the assertions.

Re-record the baseline ONLY when scores improve, and say so explicitly in the commit:

    lanes must not increase, pct must not decrease.
"""

from __future__ import annotations

import json
import pathlib

import pytest

fitz = pytest.importorskip("fitz")

from socr.benchmark.table_exactness import score_page  # noqa: E402
from socr.tables.native_rows import native_rows_from_page  # noqa: E402

_PDF = (
    pathlib.Path.home()
    / "data/fiscal-ballast/regulation/uk/obr_efo/2022-11-17_efo-november-2022.pdf"
)
_PAGES = (
    pathlib.Path.home()
    / "data/fiscal-ballast/_experiments/2026-08-01_gh96-corpus-rerun"
    / "out/2022-11-17_efo-november-2022/pages"
)
_BASELINE = pathlib.Path(__file__).parent / "data" / "obr_efo_2022_11_baseline.json"


def _corpus_available() -> bool:
    return _PDF.is_file() and _PAGES.is_dir()


pytestmark = pytest.mark.skipif(
    not _corpus_available(),
    reason=(
        "preserved OCR run not present (lives outside the repo under ~/data). "
        "This gate protects local work; see the module docstring."
    ),
)


def _measure() -> dict[str, dict[str, float]]:
    """Score every page of the preserved run against the real PDF."""
    doc = fitz.open(_PDF)
    try:
        measured: dict[str, dict[str, float]] = {}
        for md_path in sorted(_PAGES.glob("*.md")):
            index = int(md_path.stem)
            if index - 1 >= len(doc):
                continue
            page = doc[index - 1]
            report = score_page(page, md_path.read_text())
            if report.pct is None:
                continue
            ground_truth = native_rows_from_page(page)
            lanes = [
                lane for row in ground_truth for lane in getattr(row, "lanes", ()) if lane >= 0
            ]
            measured[str(index)] = {
                "pct": report.pct,
                "lanes": (max(lanes) + 1) if lanes else 0,
                "widest_row": max((len(row.values) for row in ground_truth), default=0),
            }
        return measured
    finally:
        doc.close()


@pytest.fixture(scope="module")
def measured() -> dict[str, dict[str, float]]:
    return _measure()


@pytest.fixture(scope="module")
def baseline() -> dict[str, dict[str, float]]:
    return json.loads(_BASELINE.read_text())


def test_no_page_scores_worse_than_the_baseline(measured, baseline):
    """Exactness on real pages must never regress."""
    regressions = [
        f"p{page}: {baseline[page]['pct']}% -> {measured[page]['pct']}%"
        for page in baseline
        if page in measured and measured[page]["pct"] < baseline[page]["pct"] - 0.05
    ]
    assert not regressions, "exactness regressed on real pages: " + "; ".join(regressions)


def test_no_page_gains_lanes_against_the_baseline(measured, baseline):
    """Lane count must never grow.

    Over-splitting is the failure mode that broke the reference document: the
    ground truth invents columns that are not there, so faithful OCR is scored
    against a grid it could not have produced.
    """
    regressions = [
        f"p{page}: {baseline[page]['lanes']} -> {measured[page]['lanes']} lanes"
        for page in baseline
        if page in measured and measured[page]["lanes"] > baseline[page]["lanes"]
    ]
    assert not regressions, "lane count grew on real pages: " + "; ".join(regressions)


def test_every_baseline_page_is_still_scorable(measured, baseline):
    """A page that was scorable must not silently drop out of the denominator."""
    missing = sorted(set(baseline) - set(measured), key=int)
    assert not missing, f"pages no longer scorable: {missing}"
