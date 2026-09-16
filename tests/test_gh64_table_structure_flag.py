"""GH-64: a tabular-looking page that falls to native must flag the loss.

PP-6 (GH-54) narrowed table routing to the lane-cooccupancy gate
(``has_numeric_columns``, ``_MIN_LANES_PER_ROW >= 3``). That gate structurally
cannot fire on a 2-column whitespace-aligned label|value table -- there is
only one numeric lane per row -- so such a page now falls to native prose
with its row x column grid unreconstructed, and nothing recorded it.

These tests pin ``PageAssessment.possible_table_structure_not_reconstructed``,
which reuses the pre-PP-6 ``_detect_columnar_numbers`` heuristic (restored as
a private, audit-only predicate) to recognise exactly the set of pages whose
ROUTING PP-6 changed -- never as a routing gate itself.
"""

from pathlib import Path

import fitz

from socr.core.born_digital import BornDigitalDetector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _create_label_value_table_pdf(path: Path, num_rows: int = 20) -> None:
    """A 2-column, whitespace-aligned, borderless label|value table.

    Single-word labels far to the left, numeric values far to the right, no
    ruled lines. PyMuPDF's own line grouping splits each row across the wide
    horizontal gap, so each row surfaces as two single-token lines -- the
    exact shape the pre-PP-6 heuristic keyed on, and that
    ``has_numeric_columns`` cannot reach (one numeric lane per row, gate
    requires >= 3).
    """
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 50), "Table 3. Key indicators by category.", fontsize=10, fontname="helv")
    labels = [
        "Alpha",
        "Beta",
        "Gamma",
        "Delta",
        "Epsilon",
        "Zeta",
        "Eta",
        "Theta",
        "Iota",
        "Kappa",
        "Lambda",
        "Mu",
        "Nu",
        "Xi",
        "Omicron",
        "Pi",
        "Rho",
        "Sigma",
        "Tau",
        "Upsilon",
        "Phi",
        "Chi",
        "Psi",
        "Omega",
    ][:num_rows]
    for i, label in enumerate(labels):
        y = 90 + i * 22
        page.insert_text((72, y), label, fontsize=9, fontname="helv")
        page.insert_text((450, y), f"{(i * 3 + 1) / 7:.3f}", fontsize=9, fontname="helv")
    doc.save(str(path))
    doc.close()


def _create_chart_axis_pdf(path: Path) -> None:
    """Chart-axis tick values, single x-lane -- the PP-6 false-positive class.

    Documented limitation, not a bug: reusing the pre-PP-6 heuristic verbatim
    (as the ticket requires -- no new threshold) inherits its known
    false-positive class. This is the SAME fixture PP-6's own test suite
    (``TestLaneCooccupancyRoutingGate`` in ``tests/test_born_digital.py``)
    uses to prove the old heuristic false-fires here.
    """
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text(
        (72, 60),
        "Figure 2. Impulse response of output to a monetary policy shock.",
        fontsize=10,
        fontname="helv",
    )
    tick_x = 50.0
    for i, val in enumerate([f"{v:.1f}" for v in [v / 10 for v in range(20)]]):
        page.insert_text((tick_x, 100 + i * 28), val, fontsize=8, fontname="helv")
    doc.save(str(path))
    doc.close()


def _create_dense_forecast_table_pdf(path: Path) -> None:
    """A genuine multi-column numeric grid -- routes to table handling."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text(
        (72, 50),
        "Table 1. GDP growth forecasts across baseline and shock scenarios.",
        fontsize=10,
        fontname="helv",
    )
    col_xs = [90.0, 180.0, 270.0, 360.0, 450.0]
    headers = ["Variable", "b", "s", "h", "q"]
    for ci, hdr in enumerate(headers):
        page.insert_text((col_xs[ci], 80), hdr, fontsize=9, fontname="helv")
    rows = [
        ["GDP", "0.253", "0.179", "0.211", "0.301"],
        ["CPI", "0.144", "0.135", "0.290", "0.188"],
        ["IP", "0.041", "0.050", "0.154", "0.099"],
        ["UR", "0.082", "0.321", "0.144", "0.211"],
        ["CB", "0.180", "0.171", "0.365", "0.244"],
        ["TR", "0.310", "0.220", "0.410", "0.188"],
    ]
    for ri, row in enumerate(rows):
        for ci, cell in enumerate(row):
            page.insert_text((col_xs[ci], 100 + ri * 22), cell, fontsize=9, fontname="helv")
    doc.save(str(path))
    doc.close()


def _create_prose_pdf(path: Path) -> None:
    """Ordinary justified prose -- no tabular shape at all."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    text = "This report summarises recent developments in the economy. " * 8
    page.insert_textbox(fitz.Rect(72, 72, 540, 700), text, fontsize=10, fontname="helv")
    doc.save(str(path))
    doc.close()


# ---------------------------------------------------------------------------
# Criterion 1: the label|value table that PP-6 stopped routing must flag
# ---------------------------------------------------------------------------


class TestPossibleTableStructureNotReconstructed:
    def test_label_value_table_flagged(self, tmp_path: Path) -> None:
        """A borderless label|value table falls to native AND is flagged.

        Confirms the defect this ticket targets: has_tables is False (PP-6's
        gate genuinely cannot reach a 1-numeric-lane-per-row table) and the
        new flag is True (the pre-PP-6 heuristic still recognises the shape).
        """
        pdf_path = tmp_path / "label_value.pdf"
        _create_label_value_table_pdf(pdf_path)

        detector = BornDigitalDetector()
        page = detector.detect(pdf_path).pages[0]

        assert page.is_born_digital, "Fixture must be born-digital"
        assert not page.has_tables, (
            "Fixture must reproduce the GH-64 defect: has_numeric_columns requires "
            ">= 3 co-occupied numeric lanes per row, a label|value table has one, "
            "so it must NOT route to table handling."
        )
        assert page.possible_table_structure_not_reconstructed, (
            "GH-64: a page shaped like a borderless label|value table (>=15 "
            "single-token lines, >50% of non-empty lines) that fell to native "
            "prose must flag the unreconstructed grid structure."
        )
        assert any(
            "grid structure not" in note or "structure not reconstructed" in note
            for note in page.notes
        ), "The flag must also surface as a human-readable note."

    def test_flag_scales_with_row_count_floor(self, tmp_path: Path) -> None:
        """Below the heuristic's own row floor (15), the flag must not fire.

        Not a new threshold -- this pins the EXISTING floor inside the reused
        predicate, at a row count (10) comfortably under it.
        """
        pdf_path = tmp_path / "short_label_value.pdf"
        _create_label_value_table_pdf(pdf_path, num_rows=6)

        detector = BornDigitalDetector()
        page = detector.detect(pdf_path).pages[0]

        assert not page.possible_table_structure_not_reconstructed, (
            "6 rows (12 single-token lines split label/value) is under the "
            "heuristic's own >=15 floor; confirm the reused predicate's "
            "threshold, not a new one, gates this."
        )


# ---------------------------------------------------------------------------
# Criterion 2: pages with no table structure at all must NOT flag
# ---------------------------------------------------------------------------


class TestNegativeNoFlagOnNonTabularPages:
    def test_ordinary_prose_not_flagged(self, tmp_path: Path) -> None:
        """Ordinary justified prose (multi-word lines) must never flag."""
        pdf_path = tmp_path / "prose.pdf"
        _create_prose_pdf(pdf_path)

        detector = BornDigitalDetector()
        page = detector.detect(pdf_path).pages[0]

        assert page.is_born_digital
        assert not page.has_tables
        assert not page.possible_table_structure_not_reconstructed, (
            "A flag that fires on ordinary prose is noise, not a signal "
            "(GH-64 acceptance criterion 2)."
        )

    def test_academic_paragraph_paper_not_flagged(self, tmp_path: Path) -> None:
        """Multi-line, multi-word academic prose (mirrors the detector's own
        standard born-digital fixture shape in test_born_digital.py) must
        never flag."""
        pdf_path = tmp_path / "paper.pdf"
        doc = fitz.open()
        lines = [
            "This is a born-digital academic paper about economic growth and monetary",
            "policy in developing countries. The author presents a comprehensive analysis",
            "of fiscal multipliers across different exchange rate regimes. The empirical",
            "evidence suggests that government spending has larger effects during recessions",
            "than during expansions, consistent with theoretical predictions from New",
            "Keynesian models with credit constraints and heterogeneous agents.",
            "The methodology combines structural vector autoregression with panel data",
            "techniques to identify causal effects of policy interventions.",
        ]
        page = doc.new_page()
        y = 72
        for line in lines:
            page.insert_text((72, y), line, fontsize=11, fontname="helv")
            y += 16
        doc.save(str(pdf_path))
        doc.close()

        detector = BornDigitalDetector()
        page = detector.detect(pdf_path).pages[0]

        assert page.is_born_digital
        assert not page.possible_table_structure_not_reconstructed


class TestKnownFalsePositiveClassInherited:
    """Documented limitation, not covered by criterion 2.

    The ticket requires reusing ``_detect_columnar_numbers`` VERBATIM ("do
    not write a new ratio, row-count cut, or lane threshold"). That heuristic
    is known (PP-6's own commit message and ``TestLaneCooccupancyRoutingGate``
    in ``test_born_digital.py``) to false-fire on chart-axis tick columns,
    which are single-token lines but share one x-lane, not a table. Reusing
    the predicate verbatim inherits that false-positive class by construction;
    this is called out explicitly here and in the decision log rather than
    silently narrowed with a new threshold this ticket forbids.
    """

    def test_chart_axis_labels_still_flag_a_known_false_positive(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "chart_axis.pdf"
        _create_chart_axis_pdf(pdf_path)

        detector = BornDigitalDetector()
        page = detector.detect(pdf_path).pages[0]

        assert not page.has_tables, "PP-6 routing gate must still reject this (unchanged)"
        assert page.possible_table_structure_not_reconstructed, (
            "Inherited false positive: verbatim reuse of the pre-PP-6 heuristic "
            "means chart-axis tick columns (its known false-positive class) still "
            "trip the audit flag. This is a documented trade-off of 'no new "
            "threshold', not a regression -- see docs/log/2026-09-16_64.md."
        )


# ---------------------------------------------------------------------------
# Criterion 3: a page routed to table handling must not double-report
# ---------------------------------------------------------------------------


class TestNoDoubleReportingWhenTableRouted:
    def test_dense_table_page_not_flagged(self, tmp_path: Path) -> None:
        """A genuine multi-column grid routes to table handling; the audit-only
        flag must stay False -- has_tables already says everything needed."""
        pdf_path = tmp_path / "forecast_table.pdf"
        _create_dense_forecast_table_pdf(pdf_path)

        detector = BornDigitalDetector()
        page = detector.detect(pdf_path).pages[0]

        assert page.has_tables, "Fixture must route to table handling (PP-6 regression guard)"
        assert not page.possible_table_structure_not_reconstructed, (
            "GH-64 criterion 3: a page already routed to table handling must not "
            "also carry the 'fell to native, structure lost' flag."
        )


# ---------------------------------------------------------------------------
# Criterion 4: no new magic threshold -- the flag is derived, not tunable
# ---------------------------------------------------------------------------


class TestReusesExistingPredicateOnly:
    def test_detect_columnar_numbers_is_byte_identical_to_pre_pp6(self, tmp_path: Path) -> None:
        """Pin the restored ``_detect_columnar_numbers`` against the exact
        pre-PP-6 heuristic values (>=15 single-token lines, >50% ratio) so a
        future edit cannot quietly turn it into a new, invented threshold."""
        pdf_path = tmp_path / "label_value.pdf"
        _create_label_value_table_pdf(pdf_path)

        with fitz.open(str(pdf_path)) as doc:
            page = doc[0]
            lines = page.get_text("text").splitlines()
            nonempty = [ln.strip() for ln in lines if ln.strip()]
            single_token = sum(1 for ln in nonempty if len(ln.split()) == 1)
            expected = single_token >= 15 and single_token / len(nonempty) > 0.50

            assert BornDigitalDetector._detect_columnar_numbers(page) == expected


# ---------------------------------------------------------------------------
# Criterion 5: routing is byte-identical to before this change
# ---------------------------------------------------------------------------


class TestRoutingByteIdentical:
    """This ticket adds a SURFACE; it must never change which lane a page
    takes. Pin ``has_tables`` on every fixture used above against the value
    it had before GH-64 (i.e. the value PP-6's own tests already pin), and
    additionally pin it against a call to ``_detect_tables`` with the new
    flag's predicate monkeypatched to always return the opposite of what it
    would -- proving ``has_tables`` never reads the new predicate at all."""

    def test_has_tables_unaffected_by_the_new_predicate(self, tmp_path: Path, monkeypatch) -> None:
        pdf_path = tmp_path / "label_value.pdf"
        _create_label_value_table_pdf(pdf_path)

        detector = BornDigitalDetector()
        baseline = detector.detect(pdf_path).pages[0].has_tables

        # Force the new predicate to always answer False -- if has_tables
        # read it even indirectly, forcing it off would change has_tables.
        monkeypatch.setattr(
            BornDigitalDetector, "_detect_columnar_numbers", staticmethod(lambda page: False)
        )
        forced_false = detector.detect(pdf_path).pages[0]
        assert forced_false.has_tables == baseline
        assert not forced_false.possible_table_structure_not_reconstructed, (
            "sanity: forcing the predicate off must turn the flag off, proving "
            "the monkeypatch took effect and this is not a vacuous assertion"
        )

        # Force it to always answer True -- has_tables must still be untouched.
        monkeypatch.setattr(
            BornDigitalDetector, "_detect_columnar_numbers", staticmethod(lambda page: True)
        )
        forced_true = detector.detect(pdf_path).pages[0]
        assert forced_true.has_tables == baseline, (
            "has_tables must be byte-identical regardless of what the new "
            "GH-64 predicate answers -- it must never feed routing."
        )

    def test_pp6_fixtures_has_tables_unchanged(self, tmp_path: Path) -> None:
        """Same two fixtures PP-6's own suite pins, same expected has_tables
        values, demonstrating this ticket did not move the routing gate."""
        chart_path = tmp_path / "chart_axis.pdf"
        _create_chart_axis_pdf(chart_path)
        forecast_path = tmp_path / "forecast.pdf"
        _create_dense_forecast_table_pdf(forecast_path)

        detector = BornDigitalDetector()
        chart_page = detector.detect(chart_path).pages[0]
        forecast_page = detector.detect(forecast_path).pages[0]

        assert not chart_page.has_tables, "PP-6: chart axis ticks must still not route to tables"
        assert forecast_page.has_tables, "PP-6: dense forecast grid must still route to tables"


# ---------------------------------------------------------------------------
# Criterion 1 (continued): the flag reaches PageState and the page sidecar
# ---------------------------------------------------------------------------


class TestReachesPageStateAndSidecar:
    """Mirrors the #136 shape (``tests/test_encoding_signature_tiers_gh136.py``):
    ``PageAssessment.notes`` alone reaches nothing the pipeline reads
    (orchestrator.py:9605-9609). ``apply_born_digital`` must copy the flag onto
    ``PageState``, and ``_agentic_native_page`` must turn it into an
    ``AuditEvent`` on ``state.events`` -- the stream the page sidecar is built
    from -- exactly like the two audit events already emitted there.
    """

    def test_flag_propagates_from_assessment_to_page_state(self, tmp_path: Path) -> None:
        from socr.core.state import DocumentState, PageState

        pdf_path = tmp_path / "label_value.pdf"
        _create_label_value_table_pdf(pdf_path)
        detector = BornDigitalDetector()
        assessment = detector.detect(pdf_path)

        state = DocumentState.__new__(DocumentState)
        state.pages = {1: PageState(page_num=1)}
        DocumentState.apply_born_digital(state, assessment)

        assert state.pages[1].possible_table_structure_not_reconstructed is True

    def test_page_state_default_is_clean(self) -> None:
        from socr.core.state import PageState

        assert PageState(page_num=1).possible_table_structure_not_reconstructed is False

    def test_agentic_native_page_emits_audit_event_when_flag_set(self) -> None:
        """The flag reaches ``state.events`` (the sidecar's audit-event source)
        via the same seam #136/#217 use, proven by calling the real method."""
        from socr.core.result import PageStatus
        from socr.core.state import DocumentState, PageState
        from socr.pipeline.orchestrator import UnifiedPipeline

        orch = UnifiedPipeline.__new__(UnifiedPipeline)
        orch.config = type("Cfg", (), {"native_only": False})()

        state = DocumentState.__new__(DocumentState)
        state.events = []
        ps = PageState(page_num=1)
        ps.native_text = "Alpha 0.143"
        ps.possible_table_structure_not_reconstructed = True

        orch._agentic_native_page(state, 1, ps)

        assert ps.best_output.status == PageStatus.SUCCESS, (
            "GH-64 report-only contract: the page must still ship SUCCESS, "
            "not be demoted for this flag alone."
        )
        matches = [
            ev for ev in state.events if ev.kind == "possible_table_structure_not_reconstructed"
        ]
        assert len(matches) == 1, (
            f"expected exactly one GH-64 audit event, got {len(matches)}: {state.events}"
        )
        assert matches[0].page_num == 1

    def test_agentic_native_page_no_event_when_flag_clear(self) -> None:
        """Criterion 2/3 at the orchestrator seam: no flag, no event."""
        from socr.core.state import DocumentState, PageState
        from socr.pipeline.orchestrator import UnifiedPipeline

        orch = UnifiedPipeline.__new__(UnifiedPipeline)
        orch.config = type("Cfg", (), {"native_only": False})()

        state = DocumentState.__new__(DocumentState)
        state.events = []
        ps = PageState(page_num=1)
        ps.native_text = "ordinary prose"
        ps.possible_table_structure_not_reconstructed = False

        orch._agentic_native_page(state, 1, ps)

        assert not any(
            ev.kind == "possible_table_structure_not_reconstructed" for ev in state.events
        )
