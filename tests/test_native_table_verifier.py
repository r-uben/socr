"""Tests for the two-tier native table verifier (GH-49A).

Covers:
- Hard-fail: born-digital table page where K native numeric lanes collapse to
  fewer output cells → verifier hard-fails, VLM NOT called.
- Warn-only: valid table with paired-year columns / spanning header / stub
  label column → verifier WARNS but does NOT hard-fail, VLM still decides.
  (Guard against false-fail — critical AC.)
- Bypass: scanned page (no native words) → verifier is skipped cleanly.
- Audit: both hard-fail and warn produce the expected audit records.
- CBO-style sparse drift (GH-46-D2 geometry-impossible kind) is caught.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import fitz

from socr.core.audit_log import AuditEvent
from socr.core.result import PageOutput, PageStatus
from socr.pipeline.agentic import AcceptDecision, NativeTableVerifierJudge
from socr.tables.native_verifier import verify_native_table

# Physical column gap used in synthetic fitz pages.
#
# Must be large enough that PyMuPDF inserts separate word tokens in
# get_text("words").  Empirically, insert_text calls at the same y are merged
# into one "word" run unless the x-distance exceeds the rendered width of the
# preceding token.  For 9pt font, "0.043" is ~45pt wide, so a gap >= 60pt
# reliably creates separate tokens.
#
# Must also exceed _WELL_SEPARATED_GAP_PT (18pt) so the verifier's
# well-separated lane predicate fires.  60pt >> 18pt satisfies both.
_PHYS_COL_GAP: float = 60.0


# --------------------------------------------------------------------------
# Helpers — build synthetic fitz pages and output markdown
# --------------------------------------------------------------------------


def _make_fitz_page_with_words(
    rows: list[list[tuple[float, str]]],
) -> fitz.Page:
    """Build a fitz page where each row is a list of (x, word) pairs.

    Words are placed at y = 100 + row_index * 30.  Only numeric tokens are
    placed here (the verifier cares only about numeric-token geometry).

    Use x positions spaced by _PHYS_COL_GAP (60pt) to guarantee separate
    word tokens in get_text("words").
    """
    doc = fitz.open()
    page = doc.new_page(width=700, height=900)
    for row_idx, cells in enumerate(rows):
        y = 100.0 + row_idx * 30
        for x, word in cells:
            page.insert_text((x, y), word, fontsize=9)
    return page


def _make_fitz_page_explicit(tokens: list[tuple[float, float, str]]) -> fitz.Page:
    """Build a fitz page from explicit (y, x, word) tuples.

    Use this instead of _make_fitz_page_with_words when you need to place a
    header row at a smaller y than data rows — i.e. when testing that the
    verifier's header-row offset logic fires correctly.  Header tokens such as
    ``(1)(2)(3)`` or ``2020 2021 2022`` match _NUM_TOKEN_RE and appear above
    the data rows on a real PDF, so they must be at lower y values.
    """
    doc = fitz.open()
    page = doc.new_page(width=700, height=900)
    for y, x, word in tokens:
        page.insert_text((x, y), word, fontsize=9)
    return page


def _make_empty_page() -> fitz.Page:
    """A fitz page with NO inserted text (simulates a scanned page)."""
    doc = fitz.open()
    return doc.new_page(width=500, height=700)


def _md_table(header: list[str], rows: list[list[str]]) -> str:
    """Build a minimal GitHub-markdown table string."""
    sep = "| " + " | ".join(["---"] * len(header)) + " |"
    lines = [
        "| " + " | ".join(header) + " |",
        sep,
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# verify_native_table unit tests
# --------------------------------------------------------------------------


class TestVerifyNativeTableHardFail:
    """Tier 1: value-guard (TR-4 row-aware multiset + label-binding)."""

    def test_no_hard_fail_year_paired_cells_all_values_present(self):
        """TR-4 acceptance case: a row with 3 native numeric tokens, but the output
        packs all three into 1 cell (e.g. year-paired "0.043 0.051 0.039").
        All values ARE present — the value-guard passes because the per-row
        numeric-token multiset matches.  This is the key fix for the real-CE
        false-positive where paired-year tables were incorrectly hard-failed."""
        native_rows = [
            [
                (100.0, "0.043"),
                (100.0 + _PHYS_COL_GAP, "0.051"),
                (100.0 + 2 * _PHYS_COL_GAP, "0.039"),
            ],
        ]
        page = _make_fitz_page_with_words(native_rows)
        # Output packs all three values into 1 cell — but all values ARE present
        output_text = _md_table(
            ["label", "values"],
            [["log wage", "0.043 0.051 0.039"]],
        )
        result = verify_native_table(page, output_text)
        assert result.hard_fail is False, (
            "All native values present per row → value-guard must PASS even if values "
            f"are packed into fewer cells (year-pair case); got: {result}"
        )
        assert result.native_lane_count >= 3

    def test_hard_fail_two_lane_collapse_to_one_cell(self):
        """Two well-separated native lanes, output data row has only 1 populated cell."""
        # Two rows: the GH-249 grid gate needs two rows at the modal width
        # before the native layer can serve as ground truth at all.
        native_rows = [
            [(100.0, "1.2"), (100.0 + _PHYS_COL_GAP, "2.3")],
            [(100.0, "3.4"), (100.0 + _PHYS_COL_GAP, "4.5")],
        ]
        page = _make_fitz_page_with_words(native_rows)
        # Output: header has 2 cols, each data row has only 1 populated cell
        output_text = _md_table(
            ["c1", "c2"],
            [["1.2", ""], ["3.4", ""]],  # 1 populated cell each; native has 2 lanes
        )
        result = verify_native_table(page, output_text)
        assert result.hard_fail is True, (
            "Two native lanes collapsed to 1 output cell must hard-fail"
        )
        assert len(result.drifted_rows) >= 1

    def test_no_hard_fail_when_cell_count_matches_native_lanes(self):
        """Verifier counts POPULATED cells, not numeric integrity.
        If output has 2 cells and native has 2 lanes, no collapse even if a cell
        has merged values.  Glued-numeric detection is the heuristics checker's job."""
        native_rows = [
            [(100.0, "1.2"), (100.0 + _PHYS_COL_GAP, "2.3")],
        ]
        page = _make_fitz_page_with_words(native_rows)
        # Output has 2 populated cells (country + "1.2 2.3" in one cell)
        output_text = _md_table(
            ["country", "value"],
            [["USA", "1.2 2.3"]],  # 2 populated cells, native has 2 lanes → no collapse
        )
        result = verify_native_table(page, output_text)
        assert result.hard_fail is False, (
            "Verifier should not hard-fail when populated-cell count matches native lane count; "
            "glued-numeric detection is the heuristics checker's job"
        )

    def test_hard_fail_dropped_value_in_one_row(self):
        """Multiple data rows; one has a dropped value — hard-fail fires for that row.

        Native row 0 has 3 values; output row 0 is missing one value (2.2 dropped).
        Native row 1 is complete.  The value-guard detects the multiset mismatch
        for row 0 and hard-fails.
        """
        native_rows = [
            [  # row 0 — 3 lanes (values 2.1, 2.2, 2.3)
                (100.0, "2.1"),
                (100.0 + _PHYS_COL_GAP, "2.2"),
                (100.0 + 2 * _PHYS_COL_GAP, "2.3"),
            ],
            [  # row 1 — 3 lanes, all present in output (3.1, 3.2, 3.3)
                (100.0, "3.1"),
                (100.0 + _PHYS_COL_GAP, "3.2"),
                (100.0 + 2 * _PHYS_COL_GAP, "3.3"),
            ],
        ]
        page = _make_fitz_page_with_words(native_rows)
        # Row 0 drops 2.2; row 1 is complete
        output_text = _md_table(
            ["label", "c1", "c2", "c3"],
            [
                ["Alpha", "2.1", "", "2.3"],  # 2.2 missing → multiset mismatch
                ["Beta", "3.1", "3.2", "3.3"],
            ],
        )
        result = verify_native_table(page, output_text)
        assert result.hard_fail is True, (
            "Dropped value (2.2 missing from row 0) must hard-fail via multiset mismatch"
        )
        assert len(result.drifted_rows) >= 1
        assert result.drifted_rows[0]["predicate"] == "multiset_mismatch"


class TestVerifyNativeTableWarnOnly:
    """Tier 2: ambiguous mismatch — warn but do NOT hard-fail.

    These are the critical false-fail guard tests.
    """

    def test_no_warn_for_valid_table_matching_lanes_and_cols(self):
        """A well-formed table whose output column count = native_lanes + 1 (label stub):
        neither hard_fail nor warn."""
        # 3 numeric lanes; 4-col output (label + 3 value cols) → col_gap = 1
        native_rows = [
            [
                (220.0, "0.043"),
                (220.0 + _PHYS_COL_GAP, "0.051"),
                (220.0 + 2 * _PHYS_COL_GAP, "0.039"),
            ],
            [
                (220.0, "1,204"),
                (220.0 + _PHYS_COL_GAP, "1,204"),
                (220.0 + 2 * _PHYS_COL_GAP, "1,204"),
            ],
        ]
        page = _make_fitz_page_with_words(native_rows)
        output_text = _md_table(
            ["label", "c1", "c2", "c3"],
            [
                ["log wage", "0.043", "0.051", "0.039"],
                ["N", "1,204", "1,204", "1,204"],
            ],
        )
        result = verify_native_table(page, output_text)
        assert result.hard_fail is False
        assert result.warn is False, (
            "A table where output_col_count == native_lane_count + 1 (label stub) must NOT warn"
        )

    def test_no_hard_fail_paired_year_columns(self):
        """Paired-year columns: two year columns share a narrow x-band and
        cluster into ONE native lane, but the output correctly shows them as
        two separate columns.  col_gap >= 2 but no row collapse → warn only."""
        # Two year columns very close together (within _LANE_X_TOL_PT = 6pt)
        # so they cluster into one native lane
        close_gap = 2.0  # well within _LANE_X_TOL_PT = 6.0
        native_rows = [
            # year tokens at almost the same x → same lane
            [(200.0, "1.1"), (200.0 + close_gap, "1.2")],
        ]
        page = _make_fitz_page_with_words(native_rows)
        # Output correctly shows 3 columns (country + 2 years)
        output_text = _md_table(
            ["country", "2023", "2024"],
            [["USA", "1.1", "1.2"]],
        )
        result = verify_native_table(page, output_text)
        # native_lane ≈ 1 (both tokens cluster together), output_cols = 3
        # data row populated cells = 3, native well-separated lanes for that row = 0
        # (within tolerance, not well-separated) → no hard-fail
        assert result.hard_fail is False, (
            "Paired-year columns must NOT hard-fail — col-count mismatch is legitimate"
        )

    def test_no_hard_fail_spanning_header(self):
        """A spanning header column: output has more cols than native lanes
        but data rows are not collapsed → must NOT hard-fail."""
        # 2 numeric lanes; output has 3 data cols
        native_rows = [
            [(200.0, "1.1"), (200.0 + _PHYS_COL_GAP, "2.2")],
        ]
        page = _make_fitz_page_with_words(native_rows)
        output_text = _md_table(
            ["country", "value_a", "value_b", "value_c"],
            [["USA", "1.1", "2.2", ""]],  # 3 populated cells (country + 2 values)
        )
        result = verify_native_table(page, output_text)
        assert result.hard_fail is False, "Spanning-header case must NOT hard-fail"

    def test_warn_large_gap_no_row_collapse(self):
        """col_gap >= 2 but output data rows have correct (non-collapsed) cell count
        → warn only, no hard-fail."""
        # 2 native numeric lanes; output has 4 columns → col_gap = |4 - 2| = 2 → warn
        # Two rows: required by the GH-249 grid gate.
        native_rows = [
            [(100.0, "1.1"), (100.0 + _PHYS_COL_GAP, "2.2")],
            [(100.0, "3.3"), (100.0 + _PHYS_COL_GAP, "4.4")],
        ]
        page = _make_fitz_page_with_words(native_rows)
        # 4-col output, data rows have 3 populated cells (label + 2 values) → no collapse
        output_text = _md_table(
            ["label", "c1", "c2", "c3"],
            [["row1", "1.1", "2.2", ""], ["row2", "3.3", "4.4", ""]],
        )
        result = verify_native_table(page, output_text)
        # col_gap = |4 - 2| = 2 → warn
        # data row populated cells = 3, native well-separated lanes = 2 → 3 >= 2 → no hard-fail
        assert result.hard_fail is False, "No row collapse → must NOT hard-fail"
        assert result.warn is True, "col_gap >= 2 with no collapse → should warn"
        assert "ambiguous_lane_count_mismatch" in result.reason

    def test_cbo_style_dropped_value_now_hard_fails(self):
        """TR-4 behaviour change: a sparse row that DROPS a value now hard-fails.

        The old verifier caught COLLAPSE (fewer cells than native lanes) but NOT
        a dropped value when the cell count happened to match.  The new value-guard
        (row-aware multiset equality) correctly catches any dropped value, including
        the CBO-style case where 2.2 is present in the native row but absent from
        the output row — the per-row numeric-token multiset does not match.

        This is CORRECT behaviour: a missing value is a genuine data-loss, not a
        legitimate layout variant.  The old no-fail was the false-negative the
        value-guard fixes.
        """
        # Two rows: required by the GH-249 grid gate. The SECOND row is complete;
        # the drop under test is confined to the first, so the hard-fail must come
        # from that row's multiset and not from a row-count gap.
        native_rows = [
            [(150.0, "2.1"), (150.0 + _PHYS_COL_GAP, "2.2")],
            [(150.0, "3.1"), (150.0 + _PHYS_COL_GAP, "3.2")],
        ]
        page = _make_fitz_page_with_words(native_rows)
        # Output: 2.2 is absent → multiset {2.1:1} != native {2.1:1, 2.2:1}
        output_text = _md_table(
            ["country", "2026", "2027"],
            [["CBO", "2.1", ""], ["CBO-alt", "3.1", "3.2"]],
        )
        result = verify_native_table(page, output_text)
        assert result.hard_fail is True, (
            "A dropped value (2.2 absent from output) must now hard-fail via the "
            "value-guard multiset check — this is the correct TR-4 behaviour."
        )
        assert "multiset_mismatch" in result.reason


class TestVerifyNativeTableBypass:
    """Scanned pages bypass the verifier cleanly."""

    def test_scan_page_bypasses(self):
        """A page with no native words (scan) → no hard_fail, no warn."""
        page = _make_empty_page()
        output_text = _md_table(
            ["country", "2026"],
            [["USA", "1.2"]],
        )
        result = verify_native_table(page, output_text)
        assert result.hard_fail is False
        assert result.warn is False
        assert result.native_lane_count == 0

    def test_no_markdown_table_in_output_bypasses(self):
        """If the output has no parseable markdown table, verifier bypasses."""
        native_rows = [[(100.0, "1.2"), (100.0 + _PHYS_COL_GAP, "2.3")]]
        page = _make_fitz_page_with_words(native_rows)
        output_text = "Just some plain text with no table."
        result = verify_native_table(page, output_text)
        assert result.hard_fail is False
        assert result.warn is False


# --------------------------------------------------------------------------
# Audit event tests
# --------------------------------------------------------------------------


class TestNativeTableVerifierAuditEvents:
    """Both hard-fail and warn produce expected audit records."""

    def _make_inner_judge(self, accept: bool) -> MagicMock:
        inner = MagicMock()
        inner.assess.return_value = AcceptDecision(accept=accept, reason="inner judge")
        return inner

    def _make_output(
        self,
        page_num: int = 1,
        text: str = "| a | b |\n| --- | --- |\n| 1 | 2 |",
    ) -> PageOutput:
        return PageOutput(
            page_num=page_num,
            text=text,
            status=PageStatus.SUCCESS,
            confidence=0.9,
        )

    def test_hard_fail_emits_audit_event(self):
        """Hard-fail emits a native_table_verifier_hard_fail event and does
        NOT call the inner judge."""
        # Two rows: required by the GH-249 grid gate.
        native_rows = [
            [
                (100.0, "0.1"),
                (100.0 + _PHYS_COL_GAP, "0.2"),
                (100.0 + 2 * _PHYS_COL_GAP, "0.3"),
            ],
            [
                (100.0, "0.4"),
                (100.0 + _PHYS_COL_GAP, "0.5"),
                (100.0 + 2 * _PHYS_COL_GAP, "0.6"),
            ],
        ]
        fitz_page = _make_fitz_page_with_words(native_rows)

        # Output rows collapse 3 native lanes into 2 cells (label + one value)
        output_text = _md_table(
            ["label", "vals"],
            # 2 populated cells each, native has 3 lanes → collapse
            [["row1", "0.1"], ["row2", "0.4"]],
        )
        output = self._make_output(page_num=1, text=output_text)

        events: list[AuditEvent] = []
        inner = self._make_inner_judge(accept=True)

        judge = NativeTableVerifierJudge(
            inner=inner,
            get_fitz_page=lambda pn: fitz_page,
            is_table_page=lambda pn: True,
            record_event=events.append,
        )
        provider = MagicMock()
        decision = judge.assess(output, provider)

        # Should hard-fail: accept=False, VLM NOT called
        assert decision.accept is False
        assert "native_table_verifier" in decision.reason
        inner.assess.assert_not_called()

        # Should emit exactly one audit event of the right kind
        assert len(events) == 1
        evt = events[0]
        assert evt.kind == "native_table_verifier_hard_fail"
        assert evt.page_num == 1
        assert "predicate" in evt.data
        # TR-4: predicate is now "multiset_mismatch" (value-guard replaced
        # geometry_impossible_collapse; the dropped value 0.2/0.3 triggers it).
        assert evt.data["predicate"] == "multiset_mismatch"
        assert "drifted_rows" in evt.data
        assert "native_lane_count" in evt.data
        assert "output_col_count" in evt.data

    def test_warn_emits_audit_event_and_delegates(self):
        """Warn tier emits a native_table_verifier_warn event and still calls
        the inner judge (VLM or heuristic)."""
        # 2 native lanes; output has 4 columns (col_gap = 2 → warn)
        # Two rows: required by the GH-249 grid gate.
        native_rows = [
            [(100.0, "1.1"), (100.0 + _PHYS_COL_GAP, "2.2")],
            [(100.0, "3.3"), (100.0 + _PHYS_COL_GAP, "4.4")],
        ]
        fitz_page = _make_fitz_page_with_words(native_rows)
        output_text = _md_table(
            ["label", "c1", "c2", "c3"],
            # 3 populated cells each — no collapse
            [["row1", "1.1", "2.2", ""], ["row2", "3.3", "4.4", ""]],
        )
        output = self._make_output(page_num=2, text=output_text)

        events: list[AuditEvent] = []
        inner = self._make_inner_judge(accept=True)

        judge = NativeTableVerifierJudge(
            inner=inner,
            get_fitz_page=lambda pn: fitz_page,
            is_table_page=lambda pn: True,
            record_event=events.append,
        )
        provider = MagicMock()
        decision = judge.assess(output, provider)

        # Warn: inner judge should have been called and its decision returned
        inner.assess.assert_called_once()
        assert decision.accept is True  # inner judge accepted

        # Audit event. GH-200: the winner-side structural gate also runs on
        # this accepting exit and may additionally record a
        # table_header_unverifiable abstain (2 native lanes -> the
        # header-attribution geometry chain has nothing to check here) --
        # filter by kind rather than asserting a fixed total.
        warn_events = [e for e in events if e.kind == "native_table_verifier_warn"]
        assert len(warn_events) == 1
        evt = warn_events[0]
        assert evt.page_num == 2
        assert "native_lane_count" in evt.data
        assert "output_col_count" in evt.data

    def test_exact_pass_ships_immediately_with_event(self):
        """TR-6 EXACT_PASS: verifier ships immediately without calling inner judge.

        When row counts match and multiset is clean (EXACT_PASS), the judge
        short-circuits: it emits a native_table_verifier_exact_pass event and
        returns accept=True immediately, without calling the inner judge.
        """
        # Two rows: required by the GH-249 grid gate.
        native_rows = [
            [
                (200.0, "0.1"),
                (200.0 + _PHYS_COL_GAP, "0.2"),
                (200.0 + 2 * _PHYS_COL_GAP, "0.3"),
            ],
            [
                (200.0, "0.4"),
                (200.0 + _PHYS_COL_GAP, "0.5"),
                (200.0 + 2 * _PHYS_COL_GAP, "0.6"),
            ],
        ]
        fitz_page = _make_fitz_page_with_words(native_rows)
        # 4-col output (label + 3 values), col_gap = 4 - 3 = 1 → no warn
        output_text = _md_table(
            ["label", "c1", "c2", "c3"],
            [["row1", "0.1", "0.2", "0.3"], ["row2", "0.4", "0.5", "0.6"]],
        )
        output = self._make_output(page_num=3, text=output_text)

        events: list[AuditEvent] = []
        inner = self._make_inner_judge(accept=True)

        judge = NativeTableVerifierJudge(
            inner=inner,
            get_fitz_page=lambda pn: fitz_page,
            is_table_page=lambda pn: True,
            record_event=events.append,
        )
        provider = MagicMock()
        decision = judge.assess(output, provider)

        # TR-6: EXACT_PASS short-circuits — inner judge is NOT called
        inner.assess.assert_not_called()
        assert decision.accept is True, "EXACT_PASS must accept"
        # GH-200: the structural gate also runs on this accepting exit and
        # may additionally record a table_header_unverifiable abstain (the
        # 3-lane fixture has no header words at all) -- filter by kind.
        exact_pass_events = [e for e in events if e.kind == "native_table_verifier_exact_pass"]
        assert len(exact_pass_events) == 1, f"expected 1 exact_pass event, got {exact_pass_events}"

    def test_scan_page_bypasses_verifier_in_judge(self):
        """Scanned page: no native words → verifier bypassed, inner judge called."""
        fitz_page = _make_empty_page()
        output_text = _md_table(
            ["a", "b"],
            [["1", "2"]],
        )
        output = self._make_output(page_num=4, text=output_text)

        events: list[AuditEvent] = []
        inner = self._make_inner_judge(accept=True)

        judge = NativeTableVerifierJudge(
            inner=inner,
            get_fitz_page=lambda pn: fitz_page,
            is_table_page=lambda pn: True,
            record_event=events.append,
        )
        provider = MagicMock()
        judge.assess(output, provider)

        inner.assess.assert_called_once()
        assert len(events) == 0, "Scan bypass must not emit events"

    def test_repairs_too_narrow_spanning_header_before_judging(self):
        tokens: list[tuple[float, float, str]] = [
            (80.0, 150.0, "Near"),
            (80.0, 175.0, "outcome"),
            (80.0, 315.0, "Far"),
            (80.0, 335.0, "outcome"),
        ]
        data_xs = [150.0, 215.0, 280.0, 345.0]
        for x, ordinal in zip(data_xs, ["(1)", "(2)", "(3)", "(4)"]):
            tokens.append((110.0, x, ordinal))
        for x, value in zip(data_xs, ["-4.8", None, "-4.1", "-0.2"]):
            if value is not None:
                tokens.append((140.0, x, value))
        for x, value in zip(data_xs, ["0.1", "0.2", "0.3", "0.4"]):
            tokens.append((170.0, x, value))
        fitz_page = _make_fitz_page_explicit(tokens)
        output_text = _md_table(
            ["", "Dependent variable:", "", ""],
            [
                ["", "Near outcome", "", "Far outcome"],
                ["", "(1)", "(2)", "(3)", "(4)"],
                ["Signal", "-4.8", "", "-4.1", "-0.2"],
                ["Control", "0.1", "0.2", "0.3", "0.4"],
            ],
        )
        output = self._make_output(page_num=4, text=output_text)
        events: list[AuditEvent] = []
        inner = self._make_inner_judge(accept=True)
        judge = NativeTableVerifierJudge(
            inner=inner,
            get_fitz_page=lambda pn: fitz_page,
            is_table_page=lambda pn: True,
            record_event=events.append,
        )

        decision = judge.assess(output, MagicMock())

        from socr.tables.reconcile import find_table_blocks

        repaired = find_table_blocks(output.text)[0].grid
        assert decision.accept is True
        assert {len(row) for row in repaired} == {5}
        assert repaired[1][-1] == "Far outcome"
        repair_events = [event for event in events if event.kind == "table_header_repair"]
        assert len(repair_events) == 1

    def test_non_table_page_bypasses_verifier(self):
        """Non-table page (is_table_page=False): verifier bypassed entirely."""
        native_rows = [[(100.0, "1.1"), (100.0 + _PHYS_COL_GAP, "2.2")]]
        fitz_page = _make_fitz_page_with_words(native_rows)
        output_text = _md_table(["a", "b"], [["1", "2"]])
        output = self._make_output(page_num=5, text=output_text)

        events: list[AuditEvent] = []
        inner = self._make_inner_judge(accept=True)

        judge = NativeTableVerifierJudge(
            inner=inner,
            get_fitz_page=lambda pn: fitz_page,
            is_table_page=lambda pn: False,  # NOT a table page
            record_event=events.append,
        )
        provider = MagicMock()
        judge.assess(output, provider)

        inner.assess.assert_called_once()
        assert len(events) == 0


# --------------------------------------------------------------------------
# False-fail guard: real tables that must NOT trigger hard_fail
# --------------------------------------------------------------------------


class TestFalseFailGuard:
    """These shapes must NEVER hard-fail.  A false-fail costs an unnecessary
    escalation and, if all rungs fail, document-level AUDIT_FAILED on a
    correct table.  These tests are load-bearing.

    All tests use _make_fitz_page_explicit with BOTH header tokens AND data
    tokens, mirroring a real born-digital PDF.  Earlier versions omitted the
    header row, making them paper tigers for the header-row pairing bug
    (reviewer finding, 2026-06-15).
    """

    def test_no_false_fail_with_spec_number_header_and_sparse_se_row(self):
        """THE CRITICAL GUARD TEST (header-row pairing bug fix).

        A regression table with spec-number header row ``(1)(2)(3)`` at y=70,
        a full data row at y=100, and a sparse SE row at y=130 with only one
        value in lane 0.  Output: 2 data rows (full + sparse).

        Before the fix: native_row_list[0] = header (3 lanes) was paired with
        output row 0 (full row, 4 cells — ok), and native_row_list[1] = data
        row (3 lanes) was paired with output row 1 (sparse, 1 cell) → FALSE
        hard-fail fired.

        After the fix: rows are paired by unique numeric-token overlap when
        native/output row counts differ. Output row 0 pairs to the full data
        row; output row 1 pairs to the sparse SE row. No hard-fail.

        This test FAILED on commit 7026dc8 and PASSES after the pairing fix.
        """
        # Header row: (1)(2)(3) at y=70 — these match _NUM_TOKEN_RE
        # Data row:   full coefficients at y=100
        # Sparse row: one SE value at y=130
        tokens: list[tuple[float, float, str]] = []
        for i, tok in enumerate(["(1)", "(2)", "(3)"]):
            tokens.append((70.0, 220.0 + i * _PHYS_COL_GAP, tok))
        for i, tok in enumerate(["0.043", "0.051", "0.039"]):
            tokens.append((100.0, 220.0 + i * _PHYS_COL_GAP, tok))
        # Sparse SE row: only one value in the first data lane
        tokens.append((130.0, 220.0, "(0.014)"))

        page = _make_fitz_page_explicit(tokens)
        output_text = _md_table(
            ["", "(1)", "(2)", "(3)"],
            [
                ["log wage", "0.043", "0.051", "0.039"],
                ["", "(0.014)", "", ""],
            ],
        )
        result = verify_native_table(page, output_text)
        assert result.hard_fail is False, (
            "Spec-number header + full data row + sparse SE row must NOT hard-fail; "
            f"got: hard_fail={result.hard_fail}, reason={result.reason!r}"
        )

    def test_no_false_fail_with_leading_header_matching_footnote(self):
        """A leading numeric header row (above data rows) whose tokens match
        the output column headers is excluded from the effective row count.

        Native rows: spec-number header (1)(2)(3) at y=70 (above data), then
        a sparse SE row at y=100.  Output header uses ``(1) (2) (3)`` as column
        labels.  The header row at y=70 is ABOVE the first data row at y=100,
        so it is identified as a leading header-like row and excluded.
        Effective native count = 1 (data row only); output numeric = 1 → PASS.

        This is the standard case for regression tables: spec-number column
        headers appear as native numeric rows above the data rows.
        """
        tokens: list[tuple[float, float, str]] = []
        # Leading header row ABOVE the data row
        for i, tok in enumerate(["(1)", "(2)", "(3)"]):
            tokens.append((70.0, 220.0 + i * _PHYS_COL_GAP, tok))
        # Sparse SE data row
        tokens.append((100.0, 220.0, "(0.014)"))

        page = _make_fitz_page_explicit(tokens)
        output_text = _md_table(
            ["", "(1)", "(2)", "(3)"],
            [["", "(0.014)", "", ""]],
        )
        result = verify_native_table(page, output_text)
        assert result.hard_fail is False, (
            "Leading spec-number header row must be excluded as header-like; "
            "only the data row remains → row count matches → no hard-fail. "
            f"Got: hard_fail={result.hard_fail}, reason={result.reason!r}"
        )

    def test_leading_header_row_with_matching_tokens_excluded_from_row_count(self):
        """A LEADING native row (above all data rows) whose tokens match the output
        column headers is excluded from the effective native row count.

        Pattern: native has spec-number header (1)(2)(3) at y=70 (ABOVE data),
        then data row (0.043 0.051 0.039) at y=100.  Output header uses (1)(2)(3)
        as column labels.  The leading header row's tokens {1,2,3} are ALL present
        in the output header token set AND the row is above the first data-like
        native row (y=70 < y=100) → it is excluded as a leading header-like row.
        Effective native = 1 row; output numeric = 1 row → row count matches → PASS.

        IMPORTANT: Only LEADING header rows (above the first data row) are excluded.
        A trailing footnote below the data rows (same tokens) is NOT excluded, since
        the verifier cannot distinguish it from a dropped data row — that is the
        conservative correct behaviour (see test_footnote_row_with_distinct_tokens_*).
        """
        tokens: list[tuple[float, float, str]] = []
        # Leading spec-number header ABOVE the data row
        for i, tok in enumerate(["(1)", "(2)", "(3)"]):
            tokens.append((70.0, 220.0 + i * _PHYS_COL_GAP, tok))
        # Data row below
        for i, tok in enumerate(["0.043", "0.051", "0.039"]):
            tokens.append((100.0, 220.0 + i * _PHYS_COL_GAP, tok))

        page = _make_fitz_page_explicit(tokens)
        # Output uses (1) (2) (3) as headers — matching the leading header row
        output_text = _md_table(
            ["label", "(1)", "(2)", "(3)"],
            [["log wage", "0.043", "0.051", "0.039"]],
        )
        result = verify_native_table(page, output_text)
        assert result.hard_fail is False, (
            "Leading spec-number header row (above data) must be excluded from "
            "effective native count; only the data row counts → row matches → PASS. "
            f"Got: hard_fail={result.hard_fail}, reason={result.reason!r}"
        )

    def test_footnote_row_with_distinct_tokens_exact_pass_via_yband(self):
        """TR-6: a trailing numeric footnote is excluded by pairing-derived y-band.

        Pattern: native has data row (0.043 0.051 0.039) at y=100 and a
        footnote (7 8 9) at y=160.  Output has one data row matching (0.043,
        0.051, 0.039).  The preliminary pairing matches the output row to the
        native data row at y=100; the derived y-band excludes y=160 (footnote).
        After scoping: 1 native row, 1 output numeric row → counts match →
        multiset matches → EXACT_PASS, hard_fail=False, row_count_warn=False.

        This is the TR-6 win: chart/prose/footnote rows are excluded by pairing
        before the row-count check, not by a magic constant.
        """
        tokens: list[tuple[float, float, str]] = []
        for i, tok in enumerate(["0.043", "0.051", "0.039"]):
            tokens.append((100.0, 220.0 + i * _PHYS_COL_GAP, tok))
        # Footnote with tokens NOT in the output header
        for i, tok in enumerate(["7", "8", "9"]):
            tokens.append((160.0, 220.0 + i * _PHYS_COL_GAP, tok))

        page = _make_fitz_page_explicit(tokens)
        output_text = _md_table(
            ["label", "(1)", "(2)", "(3)"],
            [["log wage", "0.043", "0.051", "0.039"]],
        )
        result = verify_native_table(page, output_text)
        assert result.hard_fail is False, (
            "Footnote excluded by y-band scoping must not hard-fail. "
            f"Got: hard_fail={result.hard_fail}, reason={result.reason!r}"
        )
        assert result.row_count_warn is False, (
            "Footnote excluded by y-band → row counts match → no row_count_warn. "
            f"Got: row_count_warn={result.row_count_warn}"
        )
        from socr.tables.native_verifier import VerifierState

        assert result.state == VerifierState.EXACT_PASS, (
            f"Footnote excluded by y-band → EXACT_PASS. Got: state={result.state!r}"
        )

    def test_no_false_fail_standard_4col_table_with_header(self):
        """A standard 4-column econ table WITH spec-number header tokens in the
        native layer.  native_lanes = 3 (values) + 3 (header) = 3 distinct lanes
        at the same x-positions; 4-col output (label + 3 values); col_gap = 1.
        Must not hard-fail (header-row offset applied correctly)."""
        tokens: list[tuple[float, float, str]] = []
        # Header row: (1)(2)(3) at y=70
        for i, tok in enumerate(["(1)", "(2)", "(3)"]):
            tokens.append((70.0, 220.0 + i * _PHYS_COL_GAP, tok))
        # Data rows at y=100, 130, 160
        data = [
            ["0.043", "0.051", "0.039"],
            ["(0.014)", "(0.016)", "(0.013)"],
            ["1,204", "1,204", "1,204"],
        ]
        for row_i, vals in enumerate(data):
            for col_i, tok in enumerate(vals):
                tokens.append((100.0 + row_i * 30, 220.0 + col_i * _PHYS_COL_GAP, tok))

        page = _make_fitz_page_explicit(tokens)
        output_text = _md_table(
            ["", "(1)", "(2)", "(3)"],
            [
                ["log wage", "0.043", "0.051", "0.039"],
                ["", "(0.014)", "(0.016)", "(0.013)"],
                ["N", "1,204", "1,204", "1,204"],
            ],
        )
        result = verify_native_table(page, output_text)
        assert result.hard_fail is False, (
            "Standard 4-col econ table with header must not hard-fail "
            f"(native_lanes={result.native_lane_count}, output_cols={result.output_col_count})"
        )

    def test_no_false_fail_sparse_se_row_with_header(self):
        """A table with year-label header row (``2020 2021 2022``) + one
        sparse SE row.  Year labels match _NUM_TOKEN_RE, so the header appears
        in native_row_list.  Offset must skip it."""
        tokens: list[tuple[float, float, str]] = []
        # Year header at y=70
        for i, tok in enumerate(["2020", "2021", "2022"]):
            tokens.append((70.0, 220.0 + i * _PHYS_COL_GAP, tok))
        # Sparse SE row: 3 values at y=100 (all 3 lanes populated)
        for i, tok in enumerate(["(0.014)", "(0.016)", "(0.013)"]):
            tokens.append((100.0, 220.0 + i * _PHYS_COL_GAP, tok))

        page = _make_fitz_page_explicit(tokens)
        output_text = _md_table(
            ["", "2020", "2021", "2022"],
            [["", "(0.014)", "(0.016)", "(0.013)"]],
        )
        result = verify_native_table(page, output_text)
        assert result.hard_fail is False, "Sparse SE row with year-label header must not hard-fail"

    def test_no_false_fail_stub_label_column_with_header(self):
        """A table with a stub label column and year-header tokens in the
        native layer.  col_gap = output_cols - native_lanes should still be 1
        (label stub).  Must not warn or hard-fail."""
        tokens: list[tuple[float, float, str]] = []
        # Year header at y=70
        for i, tok in enumerate(["2026", "2027"]):
            tokens.append((70.0, 200.0 + i * _PHYS_COL_GAP, tok))
        # Two full data rows at y=100, 130
        for row_i, (v1, v2) in enumerate([("1.1", "1.2"), ("2.1", "2.2")]):
            tokens.append((100.0 + row_i * 30, 200.0, v1))
            tokens.append((100.0 + row_i * 30, 200.0 + _PHYS_COL_GAP, v2))

        page = _make_fitz_page_explicit(tokens)
        output_text = _md_table(
            ["country", "2026", "2027"],
            [["USA", "1.1", "1.2"], ["UK", "2.1", "2.2"]],
        )
        result = verify_native_table(page, output_text)
        assert result.hard_fail is False
        assert result.warn is False, "col_gap == 1 (label stub) with year-header must not warn"

    def test_no_false_fail_empty_cells_in_dense_table_with_header(self):
        """A dense table with a spec-number header in native layer, where some
        output cells are empty (spanning label rows).  Must not hard-fail.

        The output uses ``(1) (2) (3)`` as column headers (matching the native
        spec-number header row), so the native header row is identified as
        header-like (its tokens are all in the output header token set) and
        excluded from the effective native row count.
        """
        tokens: list[tuple[float, float, str]] = []
        # Spec-number header at y=70
        for i, tok in enumerate(["(1)", "(2)", "(3)"]):
            tokens.append((70.0, 100.0 + i * _PHYS_COL_GAP, tok))
        # Single data row at y=100
        for i, tok in enumerate(["0.31", "0.34", "0.36"]):
            tokens.append((100.0, 100.0 + i * _PHYS_COL_GAP, tok))

        page = _make_fitz_page_explicit(tokens)
        # Output uses (1)(2)(3) as column headers — matching the native header row
        output_text = _md_table(
            ["", "(1)", "(2)", "(3)"],
            [["R2", "0.31", "0.34", "0.36"]],
        )
        result = verify_native_table(page, output_text)
        assert result.hard_fail is False, (
            "Dense row with label + spec-number header must not hard-fail; "
            "the native spec-number row is excluded as header-like. "
            f"Got: hard_fail={result.hard_fail}, reason={result.reason!r}"
        )

    def test_no_false_fail_panel_header_row_with_spec_number_header_and_data(self):
        """A label-only panel/section-header output row (zero numeric tokens),
        paired by the same-row-count ordinal fallback against a native
        spec-number header row, must NOT hard-fail. A row with no numeric output
        tokens cannot demonstrate a geometry-impossible collapse, so it is
        skipped/deferred, never compared against native header lanes.

        Reproduces a real AER/QJE multi-panel regression layout (``Panel A.``
        section rows) that false-failed before the ``output_tokens`` guard on
        the ordinal fallback.
        """
        tokens: list[tuple[float, float, str]] = []
        # Native spec-number header row at y=70
        for i, tok in enumerate(["(1)", "(2)", "(3)"]):
            tokens.append((70.0, 220.0 + i * _PHYS_COL_GAP, tok))
        # Native data row at y=100
        for i, tok in enumerate(["0.043", "0.051", "0.039"]):
            tokens.append((100.0, 220.0 + i * _PHYS_COL_GAP, tok))

        page = _make_fitz_page_explicit(tokens)
        output_text = _md_table(
            ["", "(1)", "(2)", "(3)"],
            [
                ["Outcome: log wages", "", "", ""],  # label-only row, 0 numeric tokens
                ["Coefficient", "0.043", "0.051", "0.039"],
            ],
        )
        result = verify_native_table(page, output_text)
        assert result.hard_fail is False, (
            "Panel header row with no numeric tokens must not false-fail via ordinal "
            f"fallback; got: hard_fail={result.hard_fail}, reason={result.reason!r}"
        )


# --------------------------------------------------------------------------
# TR-4 value-guard acceptance criteria tests
# --------------------------------------------------------------------------


class TestTR4ValueGuard:
    """TR-4 acceptance tests: the value-guard (row-aware multiset + label-binding).

    These tests directly verify the six acceptance criteria from the TR-4 ticket:
    1. Year-paired correct table → PASSES.
    2. Merged rows (multiple forecasters in one output row) → FAILS.
    3. Dropped number → FAILS (covered also by TestVerifyNativeTableHardFail).
    4. Name/value offset (TR-4a interleaved pattern) → FAILS via label-binding.
    5. Clean separate-column table → PASSES (regression guard).
    6. Full collapse (28 rows → 1 row) → still FAILS.
    """

    def test_year_paired_table_passes(self):
        """TR-4 AC 1: year-paired correct table PASSES.

        Real-CE case: each forecaster row has two-year values packed into cells
        (e.g. "2.3 1.9" = 2024+2025 for one indicator), but all native tokens
        ARE present per row.  The value-guard must ACCEPT this table — the old
        geometry_impossible_collapse check falsely rejected it.

        Synthetic setup: 3 forecasters × 4 numeric tokens each (two indicators,
        two years).  Output: 3 rows, each with year-paired cells but all values.
        """
        # 3 forecasters, each row at a different y, 4 numeric values per row
        # (two indicators × two years at 4 distinct x-lanes)
        tokens: list[tuple[float, float, str]] = []
        forecaster_values = [
            ["2.3", "1.9", "2.1", "2.4"],  # forecaster A
            ["2.5", "2.0", "1.8", "2.2"],  # forecaster B
            ["1.7", "2.1", "2.3", "1.6"],  # forecaster C
        ]
        x_lanes = [
            150.0,
            150.0 + _PHYS_COL_GAP,
            150.0 + 2 * _PHYS_COL_GAP,
            150.0 + 3 * _PHYS_COL_GAP,
        ]
        for row_i, vals in enumerate(forecaster_values):
            y = 100.0 + row_i * 30
            for col_i, v in enumerate(vals):
                tokens.append((y, x_lanes[col_i], v))
        page = _make_fitz_page_explicit(tokens)

        # Output: year-paired — each indicator's two years in ONE cell per row.
        # "2.3 1.9" = GDP 2024 + GDP 2025 for forecaster A.
        # This table has 3 output columns (label + 2 indicator-pair cells)
        # but native had 4 numeric lanes.  All values are present.
        output_text = _md_table(
            ["Forecaster", "GDP (2024 2025)", "CPI (2024 2025)"],
            [
                ["Forecaster A", "2.3 1.9", "2.1 2.4"],
                ["Forecaster B", "2.5 2.0", "1.8 2.2"],
                ["Forecaster C", "1.7 2.1", "2.3 1.6"],
            ],
        )
        result = verify_native_table(page, output_text)
        assert result.hard_fail is False, (
            "Year-paired table with all native values present must NOT hard-fail. "
            f"Got: hard_fail={result.hard_fail}, reason={result.reason!r}"
        )

    def test_merged_rows_fails(self):
        """TR-4 AC 2: merged rows (multiple forecasters in one output row) → FAILS.

        3 native forecaster rows; output has an interleaved pattern — 1 unlabeled
        data row, 2 label-only rows, 1 partial labeled row.  This is caught by the
        row-count check (3 native vs 2 output numeric rows) — the faster predicate
        that fires before label-binding.
        """
        # 3 native rows at distinct y-positions, each with 2 numeric tokens
        tokens: list[tuple[float, float, str]] = []
        native_values = [
            ("2.1", "1.9"),  # forecaster A
            ("3.2", "2.8"),  # forecaster B
            ("1.5", "2.3"),  # forecaster C
        ]
        for row_i, (v1, v2) in enumerate(native_values):
            y = 100.0 + row_i * 30
            tokens.append((y, 150.0, v1))
            tokens.append((y, 150.0 + _PHYS_COL_GAP, v2))
        page = _make_fitz_page_explicit(tokens)

        # Output: 3 forecasters interleaved — 2 label-only rows + 1 data-without-label row.
        # 2 output numeric rows (row 0 + row 3) vs 3 native rows → row_count_mismatch fires.
        output_text = _md_table(
            ["Forecaster", "2024", "2025"],
            [
                ["", "2.1", "1.9"],  # data-only: values but no label
                ["Forecaster A", "", ""],  # label-only: label but no values
                ["Forecaster B", "", ""],  # label-only: label but no values
                ["Forecaster C", "3.2", "2.8"],  # partial: C values misattributed
            ],
        )
        result = verify_native_table(page, output_text)
        # TR-6: pairing-derived y-band scoping excludes the unmatched 3rd native
        # row (no output row pairs with it), so row_count_warn does NOT fire.
        # The hard-fail is from label-binding (1 adjacent pair vs 1 clean labeled
        # row → 1 >= 1) — CERTAIN_FAIL.
        assert result.hard_fail is True, (
            "Merged rows with interleaved pattern must hard-fail via label-binding. "
            f"Got: hard_fail={result.hard_fail}, reason={result.reason!r}"
        )
        assert "label_binding" in result.reason, (
            f"Expected label_binding in reason, got: {result.reason!r}"
        )
        from socr.tables.native_verifier import VerifierState

        assert result.state == VerifierState.CERTAIN_FAIL

    def test_interleaved_name_value_offset_fails_via_label_binding(self):
        """TR-4 AC 4: name/value offset (TR-4a dense fixture pattern) → FAILS.

        The TR-4a failure mode: a name/value y-offset in the native page causes
        the born-digital path to produce interleaved rows — data rows with empty
        labels paired with label-only rows with no data.

        This test constructs the interleaved pattern directly and verifies the
        label-binding check rejects it.
        """
        # Native page: 2 rows, each with 2 numeric tokens
        native_rows = [
            [(150.0, "5.1"), (150.0 + _PHYS_COL_GAP, "3.2")],
            [(150.0, "2.7"), (150.0 + _PHYS_COL_GAP, "4.1")],
        ]
        page = _make_fitz_page_with_words(native_rows)

        # Interleaved output (TR-4a failure mode):
        # - Row 1: empty label, has values ("data-without-label")
        # - Row 2: has label, no values ("label-only")
        # - Row 3: empty label, has values
        # - Row 4: has label, no values
        output_text = _md_table(
            ["Name", "col_A", "col_B"],
            [
                ["", "5.1", "3.2"],  # data-without-label
                ["Alpha", "", ""],  # label-only
                ["", "2.7", "4.1"],  # data-without-label
                ["Beta", "", ""],  # label-only
            ],
        )
        result = verify_native_table(page, output_text)
        assert result.hard_fail is True, (
            "Interleaved name/value offset pattern must hard-fail via label-binding. "
            f"Got: hard_fail={result.hard_fail}, reason={result.reason!r}"
        )
        assert "label_binding" in result.reason

    def test_clean_separate_column_table_passes(self):
        """TR-4 AC 5: clean separate-column table (TR-0 fixture style) → PASSES.

        Every row has a non-empty label and all native values in separate cells.
        This is the baseline case that must not regress.
        """
        # 4 rows of 3 numeric values each
        tokens: list[tuple[float, float, str]] = []
        data = [
            ("2.1", "1.9", "3.4"),
            ("2.3", "2.0", "3.1"),
            ("1.8", "2.2", "3.6"),
            ("2.5", "1.7", "3.2"),
        ]
        for row_i, vals in enumerate(data):
            y = 100.0 + row_i * 30
            for col_i, v in enumerate(vals):
                tokens.append((y, 150.0 + col_i * _PHYS_COL_GAP, v))
        page = _make_fitz_page_explicit(tokens)

        output_text = _md_table(
            ["Forecaster", "GDP_2024", "GDP_2025", "CPI_2024"],
            [
                ["Ashford Capital", "2.1", "1.9", "3.4"],
                ["Brightwater Research", "2.3", "2.0", "3.1"],
                ["Clearview Economics", "1.8", "2.2", "3.6"],
                ["Dunmore Analytics", "2.5", "1.7", "3.2"],
            ],
        )
        result = verify_native_table(page, output_text)
        assert result.hard_fail is False, (
            "Clean separate-column table with all values and labels must PASS. "
            f"Got: hard_fail={result.hard_fail}, reason={result.reason!r}"
        )

    def test_full_collapse_via_interleaved_pattern_fails(self):
        """TR-4 AC 6: the original full-collapse (many names in one cell) → FAILS.

        The real CE full-collapse produces the INTERLEAVED pattern: a single
        merged data row (with no label) contains all forecasters' values, while
        the individual forecasters appear as label-only rows (names with no data).

        This is now caught by the row-count check (4 native rows vs 1 output
        numeric row → row_count_mismatch), which fires before label-binding.
        """
        # 4 native rows at distinct y-positions, each with 2 numeric tokens
        native_rows = [
            [(150.0, "2.1"), (150.0 + _PHYS_COL_GAP, "1.9")],
            [(150.0, "3.2"), (150.0 + _PHYS_COL_GAP, "2.8")],
            [(150.0, "1.5"), (150.0 + _PHYS_COL_GAP, "2.3")],
            [(150.0, "2.7"), (150.0 + _PHYS_COL_GAP, "1.4")],
        ]
        page = _make_fitz_page_with_words(native_rows)

        # Real collapse output: all values in ONE data-without-label row;
        # forecasters appear as separate label-only rows.
        output_text = _md_table(
            ["Forecaster", "2024", "2025"],
            [
                ["", "2.1 3.2 1.5 2.7", "1.9 2.8 2.3 1.4"],  # data-without-label
                ["Forecaster A", "", ""],  # label-only
                ["Forecaster B", "", ""],  # label-only
                ["Forecaster C", "", ""],  # label-only
                ["Forecaster D", "", ""],  # label-only
            ],
        )
        result = verify_native_table(page, output_text)
        # With Option B, row-count mismatch is a SOFT warning; the hard-fail here
        # is from label-binding (1 adjacent pair, 0 clean labeled rows → 1 >= 0).
        assert result.hard_fail is True, (
            "Full collapse (4 native rows → 1 output numeric row) must FAIL. "
            f"Got: hard_fail={result.hard_fail}, reason={result.reason!r}"
        )
        assert "label_binding" in result.reason, (
            f"Expected label_binding in reason, got: {result.reason!r}"
        )
        # Soft row-count warning should also be set (4 native vs 1 output numeric)
        assert result.row_count_warn is True

    def test_n2_normalization_thousands_separator(self):
        """N2 token normalization: '1,204' and '1204' are treated as equal."""
        from socr.tables.native_verifier import _normalize_numeric_token

        assert _normalize_numeric_token("1,204") == "1204"
        assert _normalize_numeric_token("1,234,567") == "1234567"

    def test_n2_normalization_parenthesis_wrapper(self):
        """N2 token normalization: '(0.014)' → '0.014' (econ negative-sign wrapper)."""
        from socr.tables.native_verifier import _normalize_numeric_token

        assert _normalize_numeric_token("(0.014)") == "0.014"
        assert _normalize_numeric_token("(1,204)") == "1204"

    def test_n2_normalization_preserves_decimal_precision(self):
        """N2 normalization preserves stated decimal precision (no float cast)."""
        from socr.tables.native_verifier import _normalize_numeric_token

        # 2.10 and 2.1 are DIFFERENT strings after N2 (no float cast)
        assert _normalize_numeric_token("2.10") == "2.10"
        assert _normalize_numeric_token("2.1") == "2.1"
        assert _normalize_numeric_token("2.10") != _normalize_numeric_token("2.1")

    def test_n2_normalization_percent_strip(self):
        """N2 token normalization: trailing '%' is stripped."""
        from socr.tables.native_verifier import _normalize_numeric_token

        assert _normalize_numeric_token("45%") == "45"
        assert _normalize_numeric_token("3.2%") == "3.2"

    def test_thousands_separator_table_passes(self):
        """A table with native '1,204' tokens and output '1,204' tokens: PASSES."""
        native_rows = [
            [(100.0, "1,204"), (100.0 + _PHYS_COL_GAP, "2,500")],
        ]
        page = _make_fitz_page_with_words(native_rows)
        output_text = _md_table(
            ["label", "c1", "c2"],
            [["N", "1,204", "2,500"]],
        )
        result = verify_native_table(page, output_text)
        assert result.hard_fail is False, (
            "Table with thousands-separator values must pass when all values present"
        )


# --------------------------------------------------------------------------
# TR-4 row-count tests (BLOCKING 1 fix)
# --------------------------------------------------------------------------


class TestTR4RowCount:
    """Row-count mismatch check (soft warning — Option B, panel decision).

    A row-count gap between native and output numeric rows raises a soft
    completeness WARNING (``row_count_warn=True``), NOT a hard-fail.  The table
    ships with WARNING status; only the per-row multiset and label-binding
    checks remain as hard-fails.

    Rationale: row-count gaps can be caused by non-table page numerals (chart
    tick labels, bar values, prose annotations) that are correctly excluded from
    the output table.  Silently discarding correct tables is worse than flagging
    them for human review.
    """

    def test_dropped_forecaster_row_warns_not_hard_fails(self):
        """3 native rows; output drops row B (only A and C labeled) → WARN, not hard-fail.

        Row count: 3 native vs 2 output numeric → row_count_warn=True, hard_fail=False.
        """
        tokens: list[tuple[float, float, str]] = []
        for row_i, (v1, v2) in enumerate([("2.1", "1.9"), ("3.2", "2.8"), ("1.5", "2.3")]):
            y = 100.0 + row_i * 30
            tokens.append((y, 150.0, v1))
            tokens.append((y, 150.0 + _PHYS_COL_GAP, v2))
        page = _make_fitz_page_explicit(tokens)

        # Output drops forecaster B — only A and C present
        output_text = _md_table(
            ["Forecaster", "2024", "2025"],
            [
                ["Forecaster A", "2.1", "1.9"],
                ["Forecaster C", "1.5", "2.3"],
            ],
        )
        result = verify_native_table(page, output_text)
        assert result.hard_fail is False, (
            "Dropped forecaster row (3 native → 2 output) must WARN (not hard-fail). "
            f"Got: hard_fail={result.hard_fail}, reason={result.reason!r}"
        )
        assert result.row_count_warn is True, (
            "Dropped forecaster row must set row_count_warn=True. "
            f"Got: row_count_warn={result.row_count_warn}"
        )

    def test_dropped_row_plus_corrupted_values_ships_ambiguous_not_silent(self):
        """ACCEPTED TRADEOFF — pinned so it cannot silently regress.

        When an extraction BOTH drops a row AND corrupts a remaining row's values,
        the count discrepancy (3 native vs 2 output) makes the per-row pairing
        UNRELIABLE — the verifier cannot distinguish this from the benign case
        (chart/prose numerals inflating the native count: the real-CE false-positive
        TR-6 fixes). Per the harness design, a multiset mismatch UNDER a row-count
        discrepancy is AMBIGUOUS: the table ships FLAGGED (``row_count_warn``
        surfaces it), NOT hard-failed and NOT silently.

        This is the residual "bad localization" failure mode both /consilium panelists
        named; TR-7 (the cropped VLM table-judge) is the planned backstop for
        AMBIGUOUS pages. A maintainer who tightens this back to a hard-fail will
        REOPEN the CE false-positive — hence this test pins the accepted behavior.
        """
        from socr.tables.native_verifier import VerifierState

        tokens: list[tuple[float, float, str]] = []
        for row_i, (v1, v2) in enumerate([("2.1", "1.9"), ("3.2", "2.8"), ("1.8", "2.4")]):
            y = 100.0 + row_i * 30
            tokens.append((y, 150.0, v1))
            tokens.append((y, 150.0 + _PHYS_COL_GAP, v2))
        page = _make_fitz_page_explicit(tokens)

        # Output DROPS the middle row AND CORRUPTS the third row (1.8/2.4 → 9.1/8.7).
        output_text = _md_table(
            ["Forecaster", "2024", "2025"],
            [
                ["Goldman", "2.1", "1.9"],  # correct
                ["Barclays", "9.1", "8.7"],  # corrupted; native was 1.8/2.4
            ],
        )
        result = verify_native_table(page, output_text)
        assert result.hard_fail is False, (
            "dropped-row + corrupted-value under a count discrepancy is AMBIGUOUS "
            f"(ships flagged), not a hard-fail. Got hard_fail={result.hard_fail}"
        )
        assert result.row_count_warn is True, "the discrepancy must surface as a warning"
        assert result.state == VerifierState.AMBIGUOUS, (
            f"must be AMBIGUOUS (ships flagged; TR-7 VLM judge is the backstop). Got {result.state}"
        )

    def test_invented_row_warns_not_hard_fails(self):
        """2 native rows; output invents a 3rd row → WARN, not hard-fail.

        Row count: 2 native vs 3 output numeric → row_count_warn=True, hard_fail=False.
        """
        tokens: list[tuple[float, float, str]] = []
        for row_i, (v1, v2) in enumerate([("2.1", "1.9"), ("3.2", "2.8")]):
            y = 100.0 + row_i * 30
            tokens.append((y, 150.0, v1))
            tokens.append((y, 150.0 + _PHYS_COL_GAP, v2))
        page = _make_fitz_page_explicit(tokens)

        # Output invents a 3rd forecaster row not in the native PDF
        output_text = _md_table(
            ["Forecaster", "2024", "2025"],
            [
                ["Forecaster A", "2.1", "1.9"],
                ["Forecaster B", "3.2", "2.8"],
                ["Forecaster C", "1.5", "2.3"],  # invented — not in native
            ],
        )
        result = verify_native_table(page, output_text)
        assert result.hard_fail is False, (
            "Invented output row (2 native → 3 output) must WARN (not hard-fail). "
            f"Got: hard_fail={result.hard_fail}, reason={result.reason!r}"
        )
        assert result.row_count_warn is True, (
            "Invented output row must set row_count_warn=True. "
            f"Got: row_count_warn={result.row_count_warn}"
        )

    def test_two_native_merged_into_one_labeled_output_row_warns(self):
        """2 native rows merged into 1 labeled output row → WARN, not hard-fail.

        A labeled merge changes the row count: 2 native vs 1 output numeric →
        row_count_warn=True, hard_fail=False.
        """
        tokens: list[tuple[float, float, str]] = []
        for row_i, (v1, v2) in enumerate([("2.1", "1.9"), ("3.2", "2.8")]):
            y = 100.0 + row_i * 30
            tokens.append((y, 150.0, v1))
            tokens.append((y, 150.0 + _PHYS_COL_GAP, v2))
        page = _make_fitz_page_explicit(tokens)

        # Both forecasters merged into one labeled output row
        output_text = _md_table(
            ["Forecaster", "2024", "2025"],
            [["Forecaster A & B", "2.1 3.2", "1.9 2.8"]],  # merged — 2 native into 1
        )
        result = verify_native_table(page, output_text)
        assert result.hard_fail is False, (
            "Two native rows merged into one labeled output row must WARN (not hard-fail). "
            f"Got: hard_fail={result.hard_fail}, reason={result.reason!r}"
        )
        assert result.row_count_warn is True, (
            "Merged-row output must set row_count_warn=True. "
            f"Got: row_count_warn={result.row_count_warn}"
        )

    def test_page_numerals_excluded_by_yband_exact_pass(self):
        """TR-6: non-table page numerals outside the y-band → EXACT_PASS.

        Simulates a page where:
        - 2 table rows at y=100 and y=130 (matched by the output)
        - 1 chart/prose row at y=160 (NOT in the output, outside the pairing y-band)

        With TR-6 pairing-derived y-band scoping: the preliminary pairing matches
        output rows A and B to native rows at y=100 and y=130.  The y-band is
        [85..145] (30pt gap → 15pt margin).  y=160 falls outside → excluded.
        After scoping: 2 effective native rows = 2 output numeric rows → EXACT_PASS.

        The table ships immediately via EXACT_PASS (no row_count_warn, no inner judge).
        """
        tokens: list[tuple[float, float, str]] = []
        # Table rows at y=100 and y=130 (paired to output)
        for row_i, (v1, v2) in enumerate([("2.1", "1.9"), ("3.2", "2.8")]):
            y = 100.0 + row_i * 30
            tokens.append((y, 150.0, v1))
            tokens.append((y, 150.0 + _PHYS_COL_GAP, v2))
        # Chart/prose row at y=160 (outside y-band, not in output)
        tokens.append((160.0, 150.0, "4.0"))
        tokens.append((160.0, 150.0 + _PHYS_COL_GAP, "3.5"))
        page = _make_fitz_page_explicit(tokens)

        output_text = _md_table(
            ["Forecaster", "2024", "2025"],
            [
                ["Forecaster A", "2.1", "1.9"],
                ["Forecaster B", "3.2", "2.8"],
            ],
        )
        result = verify_native_table(page, output_text)

        # Verifier: EXACT_PASS — chart row excluded by y-band scoping
        from socr.tables.native_verifier import VerifierState

        assert result.hard_fail is False, "Chart row excluded by y-band → no hard-fail"
        assert result.row_count_warn is False, (
            "Chart row excluded by y-band → row counts match → no row_count_warn"
        )
        assert result.state == VerifierState.EXACT_PASS, (
            f"Chart row excluded by y-band → EXACT_PASS. Got: state={result.state!r}"
        )

        # Judge: ships immediately (EXACT_PASS), emits exact_pass event
        events: list = []
        inner_judge = MagicMock()
        inner_judge.assess.return_value = AcceptDecision(
            accept=True, reason="inner ok", confidence=0.9
        )
        judge = NativeTableVerifierJudge(
            inner=inner_judge,
            get_fitz_page=lambda _n: page,
            is_table_page=lambda _n: True,
            record_event=events.append,
        )
        page_output = PageOutput(
            page_num=1,
            text=output_text,
            status=PageStatus.SUCCESS,
        )
        decision = judge.assess(page_output, MagicMock())
        assert decision.accept is True, (
            f"EXACT_PASS must SHIP. Got: accept={decision.accept}, reason={decision.reason!r}"
        )
        # Inner judge NOT called on EXACT_PASS
        inner_judge.assess.assert_not_called()
        event_kinds = [e.kind for e in events]
        assert "native_table_verifier_exact_pass" in event_kinds, (
            f"Expected native_table_verifier_exact_pass event. Got: {event_kinds}"
        )

    def test_row_count_warn_still_fires_when_pairing_cant_scope(self):
        """Row-count warn fires when pairing can't exclude extra native rows.

        When 2 native rows are merged into 1 output row (both native rows have
        equal token overlap with the single output row → tie → no unique match),
        the pairing can't determine a y-band.  The effective native rows stay at 2
        while output has 1 numeric row → row_count_warn=True (AMBIGUOUS).
        """
        tokens: list[tuple[float, float, str]] = []
        # 2 native rows with distinct values
        for row_i, (v1, v2) in enumerate([("5.0", "3.0"), ("2.0", "4.0")]):
            y = 100.0 + row_i * 30
            tokens.append((y, 150.0, v1))
            tokens.append((y, 150.0 + _PHYS_COL_GAP, v2))
        page = _make_fitz_page_explicit(tokens)

        # Output merges both native rows into 1 row (no unique match for pairing)
        output_text = _md_table(
            ["Forecaster", "2024", "2025"],
            [["Merged", "5.0 2.0", "3.0 4.0"]],  # both native rows' values
        )
        result = verify_native_table(page, output_text)

        # No unique pairing → no y-band scoping → 2 native vs 1 output → warn
        assert result.hard_fail is False, f"No hard-fail expected. Got: {result.reason!r}"
        assert result.row_count_warn is True, (
            f"2 native rows, 1 output row, no unique pairing → row_count_warn. "
            f"Got: row_count_warn={result.row_count_warn}"
        )
        from socr.tables.native_verifier import VerifierState

        assert result.state == VerifierState.AMBIGUOUS

    def test_legitimate_structural_rows_do_not_trip_row_count(self):
        """A table with structural rows (section header, sub-table year header,
        multi-line labels) alongside 5 clean labeled-numeric data rows → PASSES.

        This is the BLOCKING 2 false-positive guard: a real CE-style table that
        includes section headers, year sub-headers, and two-line labels (which
        produce a few label-only or unlabeled-numeric rows) must not trigger
        row_count_mismatch or label-binding when the core data rows are correct.

        Setup:
        - 5 native data rows (one per forecaster, 2 values each)
        - Output: 5 clean labeled-numeric rows + 1 section-header (label-only) +
          1 sub-table year-header (unlabeled-numeric: the year row has numbers
          but no label column) + 1 multi-line label (label-only)

        Row count: 5 native vs 6 output numeric rows... wait, the year-header
        row IS numeric (e.g. "2024 2025") so output_numeric_count would be 6.

        To avoid the row-count fire, the year-header tokens must either:
          (a) appear in the output markdown header (excluded as header-like), OR
          (b) have corresponding native rows.

        In this test we model the year header as a NATIVE row AND an output data
        row so both sides agree: 6 native rows (5 data + 1 year-header row) vs
        6 output numeric rows (5 labeled data + 1 unlabeled year header).

        The label-binding check: 1 adjacent (label-only, unlabeled-numeric) pair
        (section-header before year-header) vs 5 clean labeled rows → 1 < 5 →
        does NOT dominate → label-binding does NOT fire.

        Result: PASSES.
        """
        tokens: list[tuple[float, float, str]] = []
        # Native rows: year-header row at y=70, then 5 forecaster rows
        for i, yr in enumerate(["2024", "2025"]):
            tokens.append((70.0, 150.0 + i * _PHYS_COL_GAP, yr))
        forecaster_data = [
            ("2.1", "1.9"),
            ("3.2", "2.8"),
            ("1.5", "2.3"),
            ("2.7", "1.4"),
            ("2.0", "2.2"),
        ]
        for row_i, (v1, v2) in enumerate(forecaster_data):
            y = 100.0 + row_i * 30
            tokens.append((y, 150.0, v1))
            tokens.append((y, 150.0 + _PHYS_COL_GAP, v2))
        page = _make_fitz_page_explicit(tokens)

        # Output: section header (label-only) + year-header (unlabeled-numeric) +
        # 5 clean labeled rows + multi-line label (label-only)
        output_text = _md_table(
            ["Forecaster", "GDP_2024", "GDP_2025"],
            [
                ["Comparison Forecasts", "", ""],  # section header (label-only)
                ["", "2024", "2025"],  # year sub-header (unlabeled-numeric)
                ["Forecaster A", "2.1", "1.9"],  # clean labeled
                ["Forecaster B", "3.2", "2.8"],  # clean labeled
                ["Forecaster C", "1.5", "2.3"],  # clean labeled
                ["Auto & Light Truck", "", ""],  # multi-line label part 1 (label-only)
                ["Forecaster D", "2.7", "1.4"],  # clean labeled
                ["Forecaster E", "2.0", "2.2"],  # clean labeled
            ],
        )
        result = verify_native_table(page, output_text)
        assert result.hard_fail is False, (
            "Table with section headers, year sub-headers, and multi-line labels "
            "alongside 5 clean labeled rows must PASS — structural rows must not "
            "trip row_count_mismatch or label-binding. "
            f"Got: hard_fail={result.hard_fail}, reason={result.reason!r}"
        )


class TestGH249ChartPageAbstains:
    """GH-249: a native layer that is not a grid cannot serve as ground truth.

    On the FOMC SEP figure pages the native text layer is a column of axis tick
    labels -- one number per y-line. Nothing about a table is present, but the
    row parser reads each tick as a row and the value guard then grades the
    output against that phantom. The inversion is the point: a transcript that
    correctly says "there is no table here" sits furthest from the tick count
    and is the one flagged, while a dump of every tick matches it exactly.

    The verifier now applies ``rows_establish_grid`` -- the same predicate the
    GH-96 exactness metric and the GH-113 escalation trigger already use -- and
    abstains when it fails. Abstaining means the untouched bypass result: no
    warn, no hard-fail, nothing said about the output either way.
    """

    @staticmethod
    def _tick_label_page() -> fitz.Page:
        """An axis: one numeric token per line, all in a single x lane."""
        return _make_fitz_page_with_words(
            [[(120.0, str(n))] for n in range(2, 20, 2)],
        )

    def test_chart_tick_labels_are_not_a_grid(self):
        """The predicate itself: ticks fail, a real two-lane table passes."""
        from socr.core.table_grid import rows_establish_grid
        from socr.tables.native_verifier import _GridRow, _rows_by_y_from_words

        ticks = self._tick_label_page().get_text("words")
        tick_rows = [
            _GridRow(tuple(tok for _x, tok in row)) for row in _rows_by_y_from_words(ticks).values()
        ]
        assert tick_rows, "fixture produced no numeric rows at all"
        assert rows_establish_grid(tick_rows) is False

    def test_verifier_abstains_on_a_tick_label_page(self):
        """A correct 'no table here' output is not flagged against the ticks."""
        page = self._tick_label_page()
        # What a model that read the page correctly emits: a caption, no data.
        output_text = _md_table(
            ["Figure", "Note"],
            [["Figure 2.A", "Distribution of participants' projections"]],
        )
        result = verify_native_table(page, output_text)

        assert result.hard_fail is False, result.reason
        assert result.warn is False, result.reason
        assert result.row_count_warn is False, result.row_count_warn_reason

    def test_the_tick_dump_is_not_rewarded_over_the_correct_reading(self):
        """The inversion, pinned as a DIFFERENCE rather than as two verdicts.

        Before the gate, the transcript that dumped every tick label scored
        closer to the phantom than the one that reported no table. Both are now
        treated identically, because neither is being scored at all.
        """
        page = self._tick_label_page()
        correct = _md_table(
            ["Figure", "Note"],
            [["Figure 2.A", "Distribution of participants' projections"]],
        )
        tick_dump = _md_table(
            ["value", "count"],
            [[str(n), ""] for n in range(2, 20, 2)],
        )

        correct_result = verify_native_table(page, correct)
        dump_result = verify_native_table(page, tick_dump)

        assert (correct_result.hard_fail, correct_result.row_count_warn) == (
            dump_result.hard_fail,
            dump_result.row_count_warn,
        ), (
            "the tick dump and the correct reading must be indistinguishable to "
            "the verifier; if they differ, the phantom baseline is still being "
            f"built. correct={correct_result!r} dump={dump_result!r}"
        )

    def test_a_real_grid_is_still_verified(self):
        """Reverse regression: the gate must not silence the guard on tables.

        Same page shape as the collapse tests -- two rows, two well-separated
        lanes -- with an output that drops a value. That must still hard-fail;
        a gate that abstained here would disable the value guard outright.
        """
        page = _make_fitz_page_with_words(
            [
                [(100.0, "1.2"), (100.0 + _PHYS_COL_GAP, "2.3")],
                [(100.0, "3.4"), (100.0 + _PHYS_COL_GAP, "4.5")],
            ]
        )
        output_text = _md_table(
            ["label", "c1", "c2"],
            [["row1", "1.2", ""], ["row2", "3.4", "4.5"]],
        )
        result = verify_native_table(page, output_text)
        assert result.hard_fail is True, (
            "a dropped value on a genuine grid must still hard-fail after the "
            f"GH-249 gate. Got: {result!r}"
        )

    def test_single_row_native_now_abstains(self):
        """The narrowing this fix carries, recorded rather than hidden.

        ``rows_establish_grid`` requires TWO rows at the modal width, so a
        one-data-row native table is no longer verifiable. That is deliberate --
        a single numeric line is exactly what the predicate cannot tell apart
        from prose or a stray axis label -- but it does mean the value guard no
        longer fires on genuine one-row tables. Pinned so the cost is visible
        and any future change to it is a decision, not a surprise.
        """
        page = _make_fitz_page_with_words(
            [[(100.0, "1.2"), (100.0 + _PHYS_COL_GAP, "2.3")]],
        )
        output_text = _md_table(["c1", "c2"], [["1.2", ""]])
        result = verify_native_table(page, output_text)

        assert result.hard_fail is False
        assert result.warn is False
