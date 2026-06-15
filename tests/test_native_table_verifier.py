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
    """Tier 1: geometry-impossible row collapse."""

    def test_hard_fail_row_collapse(self):
        """A row with 3 well-separated native lanes but only 2 output cells
        (label + collapsed values) is a geometry-impossible collapse → hard_fail=True."""
        # Three native numeric tokens placed at _PHYS_COL_GAP distances
        native_rows = [
            [
                (100.0, "0.043"),
                (100.0 + _PHYS_COL_GAP, "0.051"),
                (100.0 + 2 * _PHYS_COL_GAP, "0.039"),
            ],
        ]
        page = _make_fitz_page_with_words(native_rows)
        # Output collapses all three values into 1 populated cell (2 cols total)
        output_text = _md_table(
            ["label", "values"],
            [["log wage", "0.043 0.051 0.039"]],
        )
        result = verify_native_table(page, output_text)
        assert result.hard_fail is True, f"Expected hard_fail for collapsed row; got: {result}"
        assert result.warn is False
        assert "geometry_impossible_collapse" in result.reason
        assert len(result.drifted_rows) >= 1
        assert result.native_lane_count >= 3

    def test_hard_fail_two_lane_collapse_to_one_cell(self):
        """Two well-separated native lanes, output data row has only 1 populated cell."""
        native_rows = [
            [(100.0, "1.2"), (100.0 + _PHYS_COL_GAP, "2.3")],
        ]
        page = _make_fitz_page_with_words(native_rows)
        # Output: header has 2 cols, data row has only 1 populated cell
        output_text = _md_table(
            ["c1", "c2"],
            [["1.2", ""]],  # only 1 populated cell; native has 2 lanes
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

    def test_hard_fail_multiple_rows_some_collapse(self):
        """Multiple data rows; only one collapses — hard-fail fires for that row."""
        native_rows = [
            [  # row 0 — 3 lanes
                (100.0, "0.043"),
                (100.0 + _PHYS_COL_GAP, "0.051"),
                (100.0 + 2 * _PHYS_COL_GAP, "0.039"),
            ],
            [  # row 1 — 3 lanes (same x positions, different y)
                (100.0, "1,204"),
                (100.0 + _PHYS_COL_GAP, "1,204"),
                (100.0 + 2 * _PHYS_COL_GAP, "1,204"),
            ],
        ]
        page = _make_fitz_page_with_words(native_rows)
        # Row 0 collapsed (1 populated cell); row 1 is fine (3 populated + label = 4)
        output_text = _md_table(
            ["label", "c1", "c2", "c3"],
            [
                ["log wage", "0.043 0.051 0.039", "", ""],
                ["N", "1,204", "1,204", "1,204"],
            ],
        )
        result = verify_native_table(page, output_text)
        assert result.hard_fail is True
        assert len(result.drifted_rows) >= 1
        assert result.drifted_rows[0]["row_idx"] == 0


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
        native_rows = [
            [(100.0, "1.1"), (100.0 + _PHYS_COL_GAP, "2.2")],
        ]
        page = _make_fitz_page_with_words(native_rows)
        # 4-col output, data row has 3 populated cells (label + 2 values) → no collapse
        output_text = _md_table(
            ["label", "c1", "c2", "c3"],
            [["row1", "1.1", "2.2", ""]],
        )
        result = verify_native_table(page, output_text)
        # col_gap = |4 - 2| = 2 → warn
        # data row populated cells = 3, native well-separated lanes = 2 → 3 >= 2 → no hard-fail
        assert result.hard_fail is False, "No row collapse → must NOT hard-fail"
        assert result.warn is True, "col_gap >= 2 with no collapse → should warn"
        assert "ambiguous_lane_count_mismatch" in result.reason

    def test_cbo_style_sparse_drift_no_hard_fail(self):
        """CBO-style: a sparse row has 2 well-separated native lanes, output has
        2 populated cells but with a value in the wrong column (drift).
        The verifier catches COLLAPSE (fewer cells), NOT drift to the wrong column.
        Hard-fail must NOT fire here — that would be a false-fail on sparse rows."""
        native_rows = [
            [(150.0, "2.1"), (150.0 + _PHYS_COL_GAP, "2.2")],
        ]
        page = _make_fitz_page_with_words(native_rows)
        # Output: row has 2 populated cells (CBO + 2.1) but 2.2 is missing → 1 numeric drifted
        # populated-cell count: "CBO" + "2.1" = 2 cells, native_lanes = 2 → no collapse
        output_text = _md_table(
            ["country", "2026", "2027"],
            [["CBO", "2.1", ""]],
        )
        result = verify_native_table(page, output_text)
        assert result.hard_fail is False, (
            "Column-lane drift (correct cell count, wrong column) is NOT geometry-impossible "
            "collapse; hard-fail would false-fire on sparse rows."
        )


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
        native_rows = [
            [
                (100.0, "0.1"),
                (100.0 + _PHYS_COL_GAP, "0.2"),
                (100.0 + 2 * _PHYS_COL_GAP, "0.3"),
            ],
        ]
        fitz_page = _make_fitz_page_with_words(native_rows)

        # Output row collapses 3 native lanes into 2 cells (label + one value)
        output_text = _md_table(
            ["label", "vals"],
            [["row1", "0.1"]],  # 2 populated cells, native has 3 lanes → collapse
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
        assert evt.data["predicate"] == "geometry_impossible_collapse"
        assert "drifted_rows" in evt.data
        assert "native_lane_count" in evt.data
        assert "output_col_count" in evt.data

    def test_warn_emits_audit_event_and_delegates(self):
        """Warn tier emits a native_table_verifier_warn event and still calls
        the inner judge (VLM or heuristic)."""
        # 2 native lanes; output has 4 columns (col_gap = 2 → warn)
        native_rows = [
            [(100.0, "1.1"), (100.0 + _PHYS_COL_GAP, "2.2")],
        ]
        fitz_page = _make_fitz_page_with_words(native_rows)
        output_text = _md_table(
            ["label", "c1", "c2", "c3"],
            [["row1", "1.1", "2.2", ""]],  # 3 populated cells — no collapse
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

        # Audit event
        assert len(events) == 1
        evt = events[0]
        assert evt.kind == "native_table_verifier_warn"
        assert evt.page_num == 2
        assert "native_lane_count" in evt.data
        assert "output_col_count" in evt.data

    def test_no_event_on_clean_pass(self):
        """No audit event when the verifier finds no issue."""
        native_rows = [
            [
                (200.0, "0.1"),
                (200.0 + _PHYS_COL_GAP, "0.2"),
                (200.0 + 2 * _PHYS_COL_GAP, "0.3"),
            ],
        ]
        fitz_page = _make_fitz_page_with_words(native_rows)
        # 4-col output (label + 3 values), col_gap = 4 - 3 = 1 → no warn
        output_text = _md_table(
            ["label", "c1", "c2", "c3"],
            [["row1", "0.1", "0.2", "0.3"]],
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
        judge.assess(output, provider)

        inner.assess.assert_called_once()
        assert len(events) == 0, "No events on a clean pass"

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

        After the fix (_offset = max(0, 3-2) = 1): output row 0 → native[1]
        (data, 3 lanes vs 4 cells → ok); output row 1 → native[2] (sparse, 1
        lane vs 1 cell → ok).  No hard-fail.

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
        output cells are empty (spanning label rows).  Must not hard-fail."""
        tokens: list[tuple[float, float, str]] = []
        # Spec-number header at y=70
        for i, tok in enumerate(["(1)", "(2)", "(3)"]):
            tokens.append((70.0, 100.0 + i * _PHYS_COL_GAP, tok))
        # Single data row at y=100
        for i, tok in enumerate(["0.31", "0.34", "0.36"]):
            tokens.append((100.0, 100.0 + i * _PHYS_COL_GAP, tok))

        page = _make_fitz_page_explicit(tokens)
        output_text = _md_table(
            ["", "c1", "c2", "c3"],
            [["R2", "0.31", "0.34", "0.36"]],
        )
        result = verify_native_table(page, output_text)
        assert result.hard_fail is False, (
            "Dense row with label + spec-number header must not hard-fail"
        )
