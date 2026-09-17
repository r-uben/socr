"""GH-793: pin the production wiring PR #792 (GH-789/GH-790) left unguarded.

PR #792 (``docs/log/2026-09-16_789-790.md``) threaded ``orphan_drops`` through
``reconstruct_table_regions`` and wired it at ``born_digital.py``'s
``extract_structured`` call site (~3630-3637):

    _rtr_drops: list[dict] = []
    table_regions = reconstruct_table_regions(page, rejections=_rejections,
                                               orphan_drops=_rtr_drops)
    ...
    if _rtr_drops:
        self._last_extraction_orphan_drops.extend(_rtr_drops)

Its own regression test (``tests/test_gh789_790_dark_drop_paths.py``) calls
``reconstruct_table_regions`` directly, passing ``orphan_drops`` itself -- it
proves the helper honours the keyword, but never drives the three lines above
that wire the helper's output into the detector's own side channel. Delete any
one of them and every existing test still passes.

**Caller surface: ``UnifiedPipeline._phase_analyze``, using the REAL
``BornDigitalDetector`` (not mocked).** This is one hop past the ``extract_
structured`` call site: it is the literal production method that (a) invokes
``self.bd_detector.detect()`` -- the same ``detect() -> _assess_page() ->
extract_structured()`` chain that owns the wiring gap -- and (b) contains the
``AuditEvent(kind="orphan_word_dropped")`` emission the ticket asks this test
to reach. Stopping at ``PageAssessment.orphan_word_drops`` alone would leave
that emission code itself unguarded by this ticket; going further, to a full
``pipeline.process()`` run, would drag in the agentic OCR loop and every other
phase for no additional coverage of the three lines under test. Other tests in
this repo already mock ``pipeline.bd_detector`` to pin ``_phase_analyze``'s
event-emission logic in isolation (``test_gh205_tr3_unconditional_event.py``);
this file deliberately does NOT mock it, because the wiring gap lives inside
the detector itself.

**Reachability.** The destroyed-token fallback only fires when a page's
default-strategy ``find_tables()`` table is empty enough that
``_table_to_markdown`` ships nothing, AND a subsequent ``vertical_strategy=
"text"`` call finds a grid that boundary-splits a numeric token (GH-144). That
is real PyMuPDF table-detection behaviour, not reliably reproducible from
synthetic PDF geometry alone (see PR #792's own fixture, which fakes
``find_tables``/``get_text`` outright rather than hand-crafting a PDF that
trips the real detector). ``detect()`` opens its own ``fitz.Document`` from a
path, so there is no page instance to monkeypatch ahead of time the way #792's
fixture does -- this file monkeypatches ``fitz.Page.get_text``/``find_tables``
at the CLASS level instead (restored by the ``monkeypatch`` fixture), scoped
to one real, on-disk PDF built with genuine ``insert_text`` content so the
born-digital heuristics (char/word counts, garbage ratio, direction) pass on
real data while table geometry is controlled. ``get_text("words")`` is the
only intercepted call whose real counterpart is bypassed; ``"text"``/``"dict"``
calls fall through to the genuine PyMuPDF implementation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz", reason="PyMuPDF not installed")

from socr.core.born_digital import BornDigitalDetector  # noqa: E402
from socr.core.config import EngineType, PipelineConfig  # noqa: E402
from socr.core.document import DocumentHandle  # noqa: E402
from socr.core.state import DocumentState  # noqa: E402
from socr.pipeline.orchestrator import UnifiedPipeline  # noqa: E402

_PAGE_TEXT = (
    "Firm A B C Alpha 0.67 0.61 0.06 Beta 0.85 0.80 0.05 "
    "Gamma 1.00 0.94 0.06 Delta 1.10 1.06 0.04 Memo 0.50 n.a."
)


class _FakeRow:
    def __init__(self, bbox, cells):
        self.bbox = bbox
        self.cells = cells


class _FakeTable:
    """The text-strategy (``vertical_strategy="text"``) result: a grid whose
    ``"0.67"`` boundary-splits to ``"0"`` (GH-144), triggering the rejection
    and the destroyed-token fallback under test.
    """

    def __init__(self, bbox, grid, rows):
        self.bbox = bbox
        self._grid = grid
        self.rows = rows

    def extract(self):
        return self._grid


class _StubTable:
    """The default-strategy (lines) result: a valid bbox, no rows.

    ``_detect_table_regions``/``extract_structured``'s own early ``find_
    tables()`` call (both unqualified, i.e. default lines strategy) see this.
    A valid bbox makes ``has_tables`` True, routing ``_assess_page_signals``
    into the ``extract_structured`` branch -- but empty ``extract()`` makes
    ``_table_to_markdown`` return ``""`` and ``_is_lane_stacked`` return
    False, so ``table_regions`` stays empty and the code falls through to the
    text-strategy ``reconstruct_table_regions`` fallback this ticket targets.
    """

    def __init__(self):
        self.bbox = (0.0, 0.0, 260.0, 60.0)

    def extract(self):
        return []


class _FakeResult:
    def __init__(self, tables):
        self.tables = tables


def _destroyed_grid_words() -> tuple[_FakeTable, list[tuple]]:
    """The GH-789/790 fixture's table + word-geometry list, unchanged from
    ``tests/test_gh789_790_dark_drop_paths.py``: a 6-row, 3-numeric-lane
    table whose "Memo" row is sparse (one populated lane) and carries an
    orphan word ("n.a.") past the snap radius of every lane.
    """
    grid = [
        ["Firm", "A", "B", "C"],
        ["Alpha", "0", "0.61", "0.06"],  # destroyed: native "0.67" != raw cell "0"
        ["Beta", "0.85", "0.80", "0.05"],
        ["Gamma", "1.00", "0.94", "0.06"],
        ["Delta", "1.10", "1.06", "0.04"],
        ["Memo", "0.50", "", ""],
    ]

    def _row(y0, y1, cells):
        return _FakeRow((0.0, y0, 260.0, y1), cells)

    rows = [
        _row(
            10.0 * n,
            10.0 * (n + 1),
            [
                (0.0, 10.0 * n, 40.0, 10.0 * (n + 1)),
                (50.0, 10.0 * n, 90.0, 10.0 * (n + 1)),
                (100.0, 10.0 * n, 140.0, 10.0 * (n + 1)),
                (150.0, 10.0 * n, 190.0, 10.0 * (n + 1)),
            ],
        )
        for n in range(6)
    ]
    table = _FakeTable((0.0, 0.0, 260.0, 60.0), grid, rows)

    labels = ["Firm", "Alpha", "Beta", "Gamma", "Delta", "Memo"]
    lane_a = [None, "0.67", "0.85", "1.00", "1.10", "0.50"]
    lane_b = [None, "0.61", "0.80", "0.94", "1.06", None]
    lane_c = [None, "0.06", "0.05", "0.06", "0.04", None]

    words: list[tuple] = []
    for n, (label, a, b, c) in enumerate(zip(labels, lane_a, lane_b, lane_c)):
        y0, y1 = 2.0 + 10.0 * n, 8.0 + 10.0 * n
        col0 = "Firm" if n == 0 else label
        words.append((0.0, y0, 30.0, y1, col0, 0, n, 0))
        header_a = "A" if n == 0 else a
        header_b = "B" if n == 0 else b
        header_c = "C" if n == 0 else c
        if header_a is not None:
            words.append((55.0, y0, 80.0, y1, header_a, 0, n, 1))
        if header_b is not None:
            words.append((105.0, y0, 130.0, y1, header_b, 0, n, 2))
        if header_c is not None:
            words.append((155.0, y0, 180.0, y1, header_c, 0, n, 3))
    # The orphan: 50pt right of lane C's centre (~167), 6th row (Memo, n=5).
    words.append((230.0, 52.0, 260.0, 58.0, "n.a.", 0, 5, 4))
    return table, words


def _patch_fitz_page(
    monkeypatch: pytest.MonkeyPatch, table: _FakeTable, words: list[tuple]
) -> None:
    """Class-level ``fitz.Page`` patch: real ``get_text("text"/"dict")``
    (real inserted content drives the born-digital heuristics), fake
    ``get_text("words")`` (this fixture's controlled word-geometry), and a
    strategy-dispatched fake ``find_tables()`` (stub for the default/lines
    call, the destroyed grid for the text-strategy call).
    """
    orig_get_text = fitz.Page.get_text

    def _fake_get_text(self, *a, **k):
        if a and a[0] == "words":
            return words
        return orig_get_text(self, *a, **k)

    def _fake_find_tables(self, *a, **k):
        if k.get("vertical_strategy") == "text":
            return _FakeResult([table])
        return _FakeResult([_StubTable()])

    monkeypatch.setattr(fitz.Page, "get_text", _fake_get_text)
    monkeypatch.setattr(fitz.Page, "find_tables", _fake_find_tables)


def _build_fixture_pdf(tmp_path: Path) -> Path:
    """A real, on-disk PDF with genuine inserted text so char/word-count,
    garbage-ratio and direction gates pass on real data (table geometry is
    supplied separately by the ``fitz.Page`` patch above).
    """
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), _PAGE_TEXT, fontsize=8)
    out = tmp_path / "gh793_fixture.pdf"
    doc.save(str(out))
    doc.close()
    return out


def _clean_pdf(tmp_path: Path) -> Path:
    """A page with no destroyed-token fallback reachable: born-digital, but
    no table geometry at all -- the negative control.
    """
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Just prose. " * 10, fontsize=8)
    out = tmp_path / "gh793_clean.pdf"
    doc.save(str(out))
    doc.close()
    return out


def _make_pipeline() -> UnifiedPipeline:
    cfg = PipelineConfig(
        agentic=True,
        judge_backend="heuristic",
        quiet=True,
        write_manifest=False,
        enabled_engines=[EngineType.QWEN],
        primary_engine=EngineType.QWEN,
        dual_pass_tables=False,
        detect_equations=False,
        save_figures=False,
    )
    pipeline = UnifiedPipeline(cfg)
    # CI hermeticity: no ollama, no provider ladder, no judge backend probe.
    pipeline._available_engines_for_agentic = lambda: []
    pipeline._resolve_judge_model = lambda: ""
    return pipeline


def test_destroyed_token_fallback_reaches_page_assessment_and_audit_event(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The full production hop: a real ``BornDigitalDetector.detect()`` call
    (via ``_phase_analyze``) drives the destroyed-token fallback, and the
    drop it reports reaches BOTH ``PageAssessment.orphan_word_drops`` and an
    ``AuditEvent(kind="orphan_word_dropped")`` on ``state.events``.
    """
    table, words = _destroyed_grid_words()
    _patch_fitz_page(monkeypatch, table, words)
    pdf_path = _build_fixture_pdf(tmp_path)

    pipeline = _make_pipeline()
    state = DocumentState(handle=DocumentHandle(path=pdf_path))
    pipeline._phase_analyze(state)

    assert pipeline._last_assessment is not None
    pages = pipeline._last_assessment.pages
    assert pages, "setup: the fixture page must have been assessed"
    pa = pages[0]
    assert pa.is_born_digital, "setup: the fixture page must classify as born-digital"
    assert pa.text_grid_rejections, "setup: the text-strategy grid must actually be rejected"
    assert any(rec["word"] == "n.a." for rec in pa.orphan_word_drops), (
        f"destroyed-token fallback drop did not reach PageAssessment.orphan_word_drops: "
        f"{pa.orphan_word_drops}"
    )

    events = [e for e in state.events if e.kind == "orphan_word_dropped"]
    assert len(events) == 1, state.events
    assert "n.a." in events[0].detail
    assert events[0].data == {"dropped_count": 1, "words": ["n.a."]}


def test_clean_page_reports_no_orphan_drops(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Negative control: a page with no table geometry at all never reaches
    the fallback, so neither surface fires.
    """
    pdf_path = _clean_pdf(tmp_path)

    pipeline = _make_pipeline()
    state = DocumentState(handle=DocumentHandle(path=pdf_path))
    pipeline._phase_analyze(state)

    assert pipeline._last_assessment is not None
    pages = pipeline._last_assessment.pages
    assert pages
    assert pages[0].orphan_word_drops == []
    assert not any(e.kind == "orphan_word_dropped" for e in state.events)
