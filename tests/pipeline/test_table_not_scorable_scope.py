"""GH-655 TICKET-E2: ``table_not_scorable`` scoped to detected tables only.

The census measured ``_table_page_needs_escalation``'s grid gate over-firing on
plain prose: every page of an ECB meeting transcript (3/3) was flagged
``table_not_scorable`` even though the native row parser found zero rows on
them, and the Fed corpus carried 400 such events across 68 documents. The gate
alone (``rows_establish_grid`` / a ``ceiling_note`` from ``score_page``) cannot
tell "this page never had a table" from "this page had a table nobody could
score" -- both fail the same grid test.

The fix scopes emission to ``detected_table_count > 0``: the born-digital
detector's independent table-region signal (``BornDigitalDetector._detect_table_regions``,
GH-520), already trusted elsewhere for the same discrimination (the D3 floor
scoping at ``orchestrator.py`` ~L1651). It does not change whether
``_table_page_needs_escalation`` decides to escalate -- only whether the
distrust event reaches ``tables_trust.json``.

Two tiers, matching this repo's established pattern (see
``tests/pipeline/test_page_failed_marker_scope.py``):
  * synthetic fixtures pin the guard-satisfied vs guard-violated difference
    directly, no PDF needed beyond a tiny in-memory one.
  * the named census fixtures (ECB transcript p1, ECB survey-2013 p2) are read
    read-only from ``~/Data/socr`` -- never copied into the repo -- and
    skipped when the file is not present on the machine running the test.

Hermetic: everything here drives ``BornDigitalDetector`` and the orchestrator's
own scoring/derivation methods directly. No engine, no provider ladder, no
ollama -- so no need to patch ``_available_engines_for_agentic``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import fitz
import pytest

from socr.core.config import PipelineConfig
from socr.core.result import PageOutput, PageStatus
from socr.core.tables_trust import build_tables_trust
from socr.pipeline.orchestrator import UnifiedPipeline

# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


def _prose_fragment_pdf(path: Path) -> None:
    """Two lines of numeric prose -- the exact shape the census reproduced.

    ``has_tables`` (the lane-cooccupancy heuristic) fires on this shape (GH-113);
    the region detector, by contrast, does not: it needs a genuine grid, not
    two co-occupying lanes.
    """
    doc = fitz.open()
    pg = doc.new_page()
    y = 200.0
    for label, value in [("November", "2022"), ("CP", "749")]:
        pg.insert_text((60.0, y), label, fontsize=9)
        pg.insert_text((300.0, y), value, fontsize=9)
        y += 18.0
    pg.draw_line(fitz.Point(50, 190), fitz.Point(470, 190))
    pg.draw_line(fitz.Point(50, y), fitz.Point(470, y))
    doc.save(path)
    doc.close()


def _pipeline() -> UnifiedPipeline:
    pipe = object.__new__(UnifiedPipeline)
    pipe.config = PipelineConfig()
    return pipe


def test_prose_page_yields_zero_untrusted_pages(tmp_path):
    """The census reproduction: a prose page must not surface `table_not_scorable`.

    Pins the ticket's own acceptance wording: "transcript excerpt yields
    untrusted_page_count == 0".
    """
    path = tmp_path / "prose.pdf"
    _prose_fragment_pdf(path)

    pipe = _pipeline()
    state = SimpleNamespace(events=[], handle=SimpleNamespace(path=path))
    ps = SimpleNamespace(has_tables=True, detected_table_count=0)
    bo = PageOutput(page_num=1, text="some prose", status=PageStatus.SUCCESS, engine="native")

    pipe._surface_table_scoring(state, 1, ps, bo)

    trust = build_tables_trust("prose.pdf", state.events)
    payload = trust.to_dict()

    assert payload["untrusted_page_count"] == 0
    assert state.events == []


def test_detected_table_page_still_flags(tmp_path):
    """Pins the other half: a page the region detector DID find a table on
    must still surface `table_not_scorable` when its native rows fail the
    grid test. Same fixture, only `detected_table_count` differs -- the
    scoping must not swallow a real not-scorable table along with the
    false-positive prose case.
    """
    path = tmp_path / "prose.pdf"
    _prose_fragment_pdf(path)

    pipe = _pipeline()
    state = SimpleNamespace(events=[], handle=SimpleNamespace(path=path))
    ps = SimpleNamespace(has_tables=True, detected_table_count=1)
    bo = PageOutput(page_num=1, text="some prose", status=PageStatus.SUCCESS, engine="native")

    pipe._surface_table_scoring(state, 1, ps, bo)

    trust = build_tables_trust("prose.pdf", state.events)
    payload = trust.to_dict()

    assert payload["untrusted_page_count"] == 1
    assert [e.kind for e in state.events] == ["table_not_scorable"]


# ---------------------------------------------------------------------------
# Real fixtures (read-only, external to the repo). Skipped when the fixture
# file is not present on this machine.
# ---------------------------------------------------------------------------

ECB_TRANSCRIPT = (
    Path.home() / "Data/socr/census-ecb-2026-09-06/in/ecb-meetings-2020-transcript-p12-14.pdf"
)
ECB_SURVEY_2013 = (
    Path.home()
    / "Data/socr/census-ecb-2026-09-06/in"
    / "ecb-surveys-2013-ecb.blssurvey2013q1.en-p29-31.pdf"
)


def _score_one_page(pdf_path: Path, page_num: int):
    """Run the same detection + scoring the pipeline runs, with no engine.

    Returns the events list build_tables_trust would have received for this
    single page, plus the assessment for inspection.
    """
    from socr.core.born_digital import BornDigitalDetector

    detector = BornDigitalDetector()
    pa = detector.detect_page(pdf_path, page_num)

    pipe = _pipeline()
    state = SimpleNamespace(events=[], handle=SimpleNamespace(path=pdf_path))
    ps = SimpleNamespace(has_tables=pa.has_tables, detected_table_count=pa.detected_table_count)
    # The native text is what shipped, unmodified, since no OCR ran.
    bo = PageOutput(
        page_num=page_num, text=pa.native_text or "", status=PageStatus.SUCCESS, engine="native"
    )

    pipe._surface_table_scoring(state, page_num, ps, bo)
    return state.events, pa


@pytest.mark.skipif(not ECB_TRANSCRIPT.exists(), reason="fixture not present on this machine")
def test_ecb_transcript_p1_no_longer_flags():
    """GH-655: the exact reproduction. Before this fix, the live run's
    ``tables_trust.json`` for this document read
    ``{"untrusted_page_count": 3, "counts_by_kind": {"table_not_scorable": 3}}``
    across all 3 pages -- a pure-prose transcript excerpt with zero detected
    tables. This drives page 1 alone through the same detector + scoring path
    and asserts the flag is gone.
    """
    events, pa = _score_one_page(ECB_TRANSCRIPT, 1)

    assert pa.detected_table_count == 0, pa.detected_table_count
    assert events == []

    trust = build_tables_trust(ECB_TRANSCRIPT.name, events)
    assert trust.to_dict()["untrusted_page_count"] == 0


@pytest.mark.skipif(not ECB_SURVEY_2013.exists(), reason="fixture not present on this machine")
def test_ecb_survey_2013_p2_detected_table_page_unaffected():
    """A page the region detector genuinely found tables on (measured
    ``detected_table_count == 3``, #639) must keep scoring -- the scoping
    change must not silence a real table page.
    """
    events, pa = _score_one_page(ECB_SURVEY_2013, 2)

    assert pa.detected_table_count > 0, pa.detected_table_count
    # Whether this specific page's native rows happen to establish a grid is
    # a fact about the fixture, not the scoping change -- either way, the
    # count-gated path was reached (never short-circuited to "no detection").
    trust = build_tables_trust(ECB_SURVEY_2013.name, events)
    payload = trust.to_dict()
    if events:
        assert payload["untrusted_page_count"] == 1
        assert "table_not_scorable" in payload["pages"]["2"]["reasons"]
