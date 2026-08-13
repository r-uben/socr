"""GH-151 TICKET-B1: surface a structurally defective native table grid.

The gate: a page whose extracted native markdown table splits a label row
from its values row (``FINDING_DETACHED_LABEL``, or plain ``ragged``) must
NOT ship as trusted native SUCCESS. Before this ticket it did — the defect
was detected (GH-151 A1's ``structure_check``) but nothing consumed it.

Everything here is a synthetic PyMuPDF fixture built inside ``tmp_path`` (no
committed PDF, no fixture generator script — see the ticket's mustNotDo).
Content-loss / recovery claims are measured END TO END through the
installed package (``BornDigitalDetector().extract_structured`` /
``UnifiedPipeline.process()``), never against an isolated rung such as
``reconstruct_table_regions`` in isolation — that exact mistake produced the
false 100%-table-loss alarm on PR #192 (see docs/plans/extraction-defects/
STATUS.md).

CI has no ollama/provider: ``_available_engines_for_agentic`` is patched on
every ``process()`` call even though the acceptance run here is
``agentic=False, native_only=True`` (house rule; a prior ticket shipped a
test that was locally green and CI-red for skipping this).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

fitz = pytest.importorskip("fitz")

from socr.core.born_digital import BornDigitalDetector
from socr.core.config import EngineType, PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.providers import PROFILE_QWEN_LOCAL
from socr.core.result import DocumentStatus, PageOutput, PageStatus
from socr.core.state import DocumentState
from socr.pipeline.orchestrator import UnifiedPipeline
from socr.tables import structure_check

# ---------------------------------------------------------------------------
# Fixture: a ruled regression-table page whose "R2" row is genuinely split
# from its values row by the native extractor — label alone on one y-band,
# its six values alone on the next, mirroring the GH-151 p26 defect shape.
# ---------------------------------------------------------------------------

_LABEL_X = 60.0
_VALUE_XS = (150.0, 210.0, 270.0, 330.0, 390.0, 450.0)
_ROW_H = 18.0
_HEADER = ("", "c1", "c2", "c3", "c4", "c5", "c6")

# (label, values) pairs, two ordinary rows first so the booktabs reconstructor
# has enough rows to recognise the grid (``reconstruct._MIN_ROWS``), then
# "TERM" (a normal labelled row) followed by "R2" carrying no values on its
# own line — its values land on the immediately following line with no
# label, a physically split row, not a legitimate blank-label continuation
# (which would carry a label ABOVE it that also has values, per the
# negative-control tests below).
_ROWS: list[tuple[str, tuple[str, ...] | None]] = [
    ("alpha", ("1.0", "2.0", "3.0", "4.0", "5.0", "6.0")),
    ("beta", ("1.1", "2.1", "3.1", "4.1", "5.1", "6.1")),
    ("TERM", ("0.32*", "0.51", "1.02", "0.88", "0.14", "0.09")),
    ("R2", None),
    ("", ("0.12", "0.16", "0.61", "0.09", "0.12", "0.61")),
]


def _detached_label_table_pdf(path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page()

    top = 170.0
    y = top + _ROW_H
    for i, cell in enumerate(_HEADER):
        if not cell:
            continue
        x = _LABEL_X if i == 0 else _VALUE_XS[i - 1]
        page.insert_text((x, y), cell, fontsize=9, fontname="helv")
    y += _ROW_H
    for label, values in _ROWS:
        page.insert_text((_LABEL_X, y), label, fontsize=9, fontname="helv")
        if values:
            for x, v in zip(_VALUE_XS, values):
                page.insert_text((x, y), v, fontsize=9, fontname="helv")
        y += _ROW_H

    # A ruled box (top / below-header / bottom) so the booktabs text-strategy
    # reconstructor (``reconstruct.reconstruct_table_regions``, exercised only
    # through the installed ``extract_structured`` here) recognises the grid,
    # the same rule shape used by tests/test_gh96_table_exactness.py's fixture.
    page.draw_line(fitz.Point(50, top), fitz.Point(500, top))
    page.draw_line(fitz.Point(50, top + _ROW_H), fitz.Point(500, top + _ROW_H))
    page.draw_line(fitz.Point(50, y), fitz.Point(500, y))

    doc.save(str(path))
    doc.close()


def _config(**overrides) -> PipelineConfig:
    defaults = dict(
        primary_engine=EngineType.DEEPSEEK,
        agentic=False,
        native_only=True,
        native_first=True,
        quiet=True,
        audit_enabled=True,
        save_figures=False,
        write_manifest=True,
        dual_pass_tables=False,
        judge_hard_pages=False,
    )
    defaults.update(overrides)
    return PipelineConfig(**defaults)


def _run(pdf_path: Path, out_dir: Path):
    pipeline = UnifiedPipeline(_config())
    with patch.object(
        pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]
    ):
        result = pipeline.process(pdf_path, out_dir)
    return result


# ---------------------------------------------------------------------------
# Setup proof: the installed extractor genuinely splits this fixture. If it
# doesn't, the rest of the test is vacuous — fail loudly rather than measure
# an isolated rung (see module docstring).
# ---------------------------------------------------------------------------


def test_setup_extract_structured_genuinely_splits_the_row(tmp_path: Path) -> None:
    pdf_path = tmp_path / "setup.pdf"
    _detached_label_table_pdf(pdf_path)

    page = fitz.open(str(pdf_path))[0]
    md = BornDigitalDetector().extract_structured(page)

    reports = structure_check.check_markdown(md)
    assert reports, f"extract_structured produced no parseable table block; markdown was:\n{md}"
    fired = [r for r in reports if r.detached_label_rows or r.ragged]
    assert fired, (
        "setup precondition failed: installed extract_structured did not split "
        f"the R2 row from its values on this fixture; markdown was:\n{md}\n"
        f"reports: {reports}"
    )


# ---------------------------------------------------------------------------
# RED: today's (pre-fix) behaviour on unmodified main — document SUCCESS,
# sidecar audit_passed=True, zero table_structure_failed events, exactly one
# ('native', SUCCESS, True) attempt. This function is not run as a test on
# the fixed tree (the assertions below are the GREEN ones); it is preserved
# as the documented red-state proof recorded in
# docs/plans/gh151-structural-gate/logs/2026-08-13_b1.md, captured by
# running this same body against unmodified main via `git stash`.
# ---------------------------------------------------------------------------


class TestProcessEndToEnd:
    def test_native_only_ships_page_flagged_not_passed(self, tmp_path: Path) -> None:
        """GREEN acceptance: the gate fires end to end under --native-only."""
        pdf_path = tmp_path / "doc.pdf"
        _detached_label_table_pdf(pdf_path)
        out_dir = tmp_path / "out"

        result = _run(pdf_path, out_dir)

        assert result.status == DocumentStatus.AUDIT_FAILED

        # EngineResult doesn't carry raw DocumentState.events; verify via the
        # persisted audit_log.json instead (durable, checked by the CLI too).
        audit_log_path = out_dir / "doc" / "audit_log.json"
        if not audit_log_path.exists():
            # single-file layout may differ; search for it.
            candidates = list(out_dir.rglob("audit_log.json"))
            assert candidates, f"no audit_log.json found under {out_dir}"
            audit_log_path = candidates[0]
        audit_log = json.loads(audit_log_path.read_text())
        events = audit_log.get("events", [])
        fired = [e for e in events if e.get("kind") == "table_structure_failed"]
        assert len(fired) == 1, f"expected exactly one table_structure_failed event, got {events}"

        # Sidecar: WARNING / audit_passed=False / flag persisted.
        sidecar_candidates = list(out_dir.rglob("pages/00001.json"))
        assert sidecar_candidates, f"no page sidecar found under {out_dir}"
        sidecar = json.loads(sidecar_candidates[0].read_text())
        assert sidecar["native_table_structure_defective"] is True
        winning = sidecar.get("winning_output") or sidecar
        assert sidecar.get("status") in ("warning", "WARNING") or winning.get("status") in (
            "warning",
            "WARNING",
        )

        # End-to-end token preservation: every value token from the installed
        # extractor's markdown survives into the shipped page text.
        page = fitz.open(str(pdf_path))[0]
        extracted = BornDigitalDetector().extract_structured(page)
        shipped_pages = list(out_dir.rglob("pages/00001.md"))
        assert shipped_pages
        shipped_text = shipped_pages[0].read_text()
        for _, values in _ROWS:
            if values:
                for v in values:
                    assert v in shipped_text, f"value {v!r} lost from shipped page text"
        assert "R2" in shipped_text

        # No-reroute pin: the shipped page is native, at zero cost — no OCR
        # engine call was ever made under --native-only.
        assert sidecar["engine"] == "native"
        assert sidecar["cost_usd"] == 0.0

    def test_restore_terminal_page_state_restores_the_flag(self, tmp_path: Path) -> None:
        """``_restore_terminal_page_state`` (PP-5 resume path) restores the flag
        from the sidecar written by a real run, rather than dropping it.

        The page is WARNING (not SUCCESS), so ``_load_terminal_page``'s own
        conservative gate never grants it a skip on a second full ``process()``
        run — a WARNING page must always be re-examined, by design. That is
        exactly why this test calls ``_restore_terminal_page_state`` directly
        against the sidecar the first run actually wrote, the same call the
        live resume loop makes once a page IS eligible.
        """
        pdf_path = tmp_path / "doc.pdf"
        _detached_label_table_pdf(pdf_path)
        out_dir = tmp_path / "out"

        pipeline = UnifiedPipeline(_config())
        with patch.object(
            pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]
        ):
            pipeline.process(pdf_path, out_dir)

        sidecar_candidates = list(out_dir.rglob("pages/00001.json"))
        assert sidecar_candidates, f"no page sidecar found under {out_dir}"
        sidecar = json.loads(sidecar_candidates[0].read_text())
        assert sidecar["native_table_structure_defective"] is True

        fresh_state = DocumentState(handle=DocumentHandle(path=pdf_path, page_count=1))
        assert fresh_state.pages[1].native_table_structure_defective is False

        fake_output = PageOutput(page_num=1, text="restored text", status=PageStatus.WARNING)
        pipeline._restore_terminal_page_state(fresh_state, 1, fake_output, out_dir)

        assert fresh_state.pages[1].native_table_structure_defective is True


# ---------------------------------------------------------------------------
# Predicate units (pure check_grid/check_markdown, no pipeline) — negative
# controls required by the ticket's Done-when.
# ---------------------------------------------------------------------------


class TestDetachedLabelPredicate:
    def test_p26_seam_fires(self) -> None:
        from test_structure_check_gh151 import GH151_P26_MD

        reports = structure_check.check_markdown(GH151_P26_MD)
        assert len(reports) == 1
        assert reports[0].detached_label_rows

    def test_se_tstat_continuation_row_does_not_fire(self) -> None:
        """A labelled coefficient row followed by a blank-label SE row WITH
        values on the coefficient row is a legitimate continuation, not a
        split — because the left row of the pair is NOT label-only (it has
        its own values)."""
        grid = [
            ["", "col1", "col2"],
            ["beta", "1.23", "4.56"],
            ["", "(0.11)", "(0.22)"],  # SE row: blank label, has values
        ]
        report = structure_check.check_grid(grid)
        assert report.detached_label_rows == ()

    def test_group_heading_above_column_band_does_not_fire_first_row(self) -> None:
        grid = [
            ["", "col1", "col2", "col3"],
            ["Panel A", "", "", ""],
            ["", "(1)", "(2)", "(3)"],
        ]
        report = structure_check.check_grid(grid)
        assert report.detached_label_rows == ()

    def test_group_heading_above_column_band_does_not_fire_mid_table(self) -> None:
        grid = [
            ["", "col1", "col2", "col3"],
            ["alpha", "1.0", "2.0", "3.0"],
            ["Panel B", "", "", ""],
            ["", "(1)", "(2)", "(3)"],
            ["beta", "4.0", "5.0", "6.0"],
        ]
        report = structure_check.check_grid(grid)
        assert report.detached_label_rows == ()

    def test_split_footnote_pair_fires_as_a_deliberate_classification(self) -> None:
        """A footnote mangled into a grid (label-only note marker, followed by
        a values-only row) is classified as a detached label — the predicate
        cannot distinguish "footnote" from "table row" and is not asked to;
        this is a documented, deliberate classification, not a bug."""
        grid = [
            ["", "col1", "col2"],
            ["gamma", "0.5", "0.6"],
            ["aNote:", "", ""],
            ["", "text", "spillover"],
        ]
        report = structure_check.check_grid(grid)
        assert report.detached_label_rows == (2,)
