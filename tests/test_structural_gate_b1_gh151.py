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
from types import SimpleNamespace
from unittest.mock import patch

import pytest

fitz = pytest.importorskip("fitz")

from socr.core.born_digital import BornDigitalDetector
from socr.core.config import EngineType, PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.manifest import _winning_page_output
from socr.core.providers import PROFILE_QWEN_LOCAL
from socr.core.result import DocumentStatus, FailureMode, PageOutput, PageStatus
from socr.core.state import DocumentState
from socr.pipeline.orchestrator import UnifiedPipeline
from socr.tables import structure_check
from socr.tables.reconcile import find_table_blocks
from socr.tables.structure_check import structural_gate_fires as _gate_fires

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
        # doneWhen: the SHIPPED page (not just the discarded attempt) carries
        # FailureMode.NATIVE_TABLE_STRUCTURE_FAILED -- the surface the manifest
        # actually freezes and ``pages/NNN.md`` is built from.
        assert winning.get("failure_mode") == "native_table_structure_failed"

        # End-to-end token preservation: every value token the installed
        # extractor actually emitted survives into the shipped page text --
        # not just the hardcoded _ROWS constants, so this genuinely compares
        # against extract_structured's own output rather than merely
        # re-asserting the fixture's own inputs.
        page = fitz.open(str(pdf_path))[0]
        extracted = BornDigitalDetector().extract_structured(page)
        extracted_blocks = find_table_blocks(extracted)
        assert extracted_blocks, f"no parseable table block in:\n{extracted}"
        extracted_tokens = {
            cell for block in extracted_blocks for row in block.grid for cell in row if cell.strip()
        }
        shipped_pages = list(out_dir.rglob("pages/00001.md"))
        assert shipped_pages
        shipped_text = shipped_pages[0].read_text()
        for token in extracted_tokens:
            assert token in shipped_text, f"token {token!r} from extract_structured lost"
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
        its own values).

        Asserts against ``structural_gate_fires`` -- the same function
        ``born_digital.py`` calls to compute ``native_table_structure_defective``
        (``ragged or detached_label_rows``) -- not just ``detached_label_rows``
        alone. The grid is uniformly 3-wide, so
        ``ragged`` is incidentally False here too; a prior version of this
        test asserted only ``detached_label_rows == ()`` and so still passed
        when the gate was (hypothetically) widened to include
        ``orphan_rows``, which DOES fire on this exact grid (row 2 is a
        blank-label row with values) -- meaning it pinned nothing the gate
        actually depends on. See
        ``test_se_control_is_load_bearing_against_a_widened_gate`` below for
        the proof.
        """
        grid = [
            ["", "col1", "col2"],
            ["beta", "1.23", "4.56"],
            ["", "(0.11)", "(0.22)"],  # SE row: blank label, has values
        ]
        report = structure_check.check_grid(grid)
        assert not _gate_fires([report])

    def test_se_control_is_load_bearing_against_a_widened_gate(self) -> None:
        """Proof that the SE-continuation control above pins the actual gate
        predicate, not merely one of its inputs: the SE grid's own
        ``orphan_rows`` finding DOES fire (row 2 is blank-label-with-values),
        so a gate hypothetically widened to ``ragged or orphan_rows or
        detached_label_rows`` — the exact regression the ticket's narrowing
        decision rules out — trips on this fixture. If this assertion fails,
        the SE control above is not exercising anything ``orphan_rows``
        would also satisfy, and stops being evidence that excluding
        ``orphan_rows`` from the gate matters.
        """
        grid = [
            ["", "col1", "col2"],
            ["beta", "1.23", "4.56"],
            ["", "(0.11)", "(0.22)"],
        ]
        report = structure_check.check_grid(grid)
        assert report.orphan_rows != ()  # the SE row is an orphan row...
        assert not _gate_fires([report])  # ...but the actual gate does not fire on it
        widened_gate_fires = bool(report.ragged or report.orphan_rows or report.detached_label_rows)
        assert widened_gate_fires  # ...while a gate that included orphan_rows would

    def test_group_heading_above_column_band_does_not_fire_first_row(self) -> None:
        """A group-heading row (label, no values) directly above a
        ``(1)...(n)`` column-number band -- the other legitimate shape the
        ticket's Done-when names -- must not trip the gate."""
        grid = [
            ["", "col1", "col2", "col3"],
            ["Panel A", "", "", ""],
            ["", "(1)", "(2)", "(3)"],
        ]
        report = structure_check.check_grid(grid)
        assert report.detached_label_rows == ()
        assert not _gate_fires([report])

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
        assert not _gate_fires([report])

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


# ---------------------------------------------------------------------------
# Wiring units (review finding BLOCKING 2) -- direct, hermetic assertions
# against each of the six surfaces the ticket's BINDING CONSTRAINT names.
# No PDF, no provider: a DocumentState/PageState built by hand, matching the
# exact shape production code leaves the page in at each point. Each was
# confirmed load-bearing by temporarily re-introducing the regression the
# review named (deleting the manifest guard / adding the flag to the
# needs_repair tuple) and observing the corresponding test fail; see the
# ticket's fix log for the transcript.
# ---------------------------------------------------------------------------


def _bare_handle(pages: int = 1) -> DocumentHandle:
    with patch.object(DocumentHandle, "__post_init__", lambda self: None):
        return DocumentHandle(path=Path("/tmp/fake-gh151-b1.pdf"), page_count=pages)


class TestWiringUnits:
    def test_score_per_page_forces_demotion_without_erasing_the_flag(self) -> None:
        """(1) ``_score_per_page`` must not erase the flag or re-promote the
        demoted attempt to ``audit_passed=True`` -- even though the grid it
        is judging is otherwise clean and would pass the heuristic scorer if
        the flag-check were skipped."""
        state = DocumentState(handle=_bare_handle())
        ps = state.pages[1]
        ps.is_born_digital = True
        clean_md = (
            "| Forecast | 2026 | 2027 |\n| --- | --- | --- |\n| A | 1.2 | 1.3 |\n| B | 2.1 | 2.2 |"
        )
        ps.native_text = clean_md
        ps.native_table_structure_defective = True
        bo = PageOutput(
            page_num=1,
            text=clean_md,
            status=PageStatus.SUCCESS,
            engine="native",
            audit_passed=True,
        )
        ps.attempts.append(bo)
        ps.best_output = bo

        pipe = UnifiedPipeline(PipelineConfig(quiet=True))
        pipe._score_per_page(state)

        assert bo.audit_passed is False
        assert bo.failure_mode == FailureMode.NATIVE_TABLE_STRUCTURE_FAILED
        assert ps.best_output is None
        assert ps.native_table_structure_defective is True  # not erased

    def test_needs_repair_stays_false_on_the_flag_alone(self) -> None:
        """(2) The flag alone must never force ``PageState.needs_repair`` --
        that would trigger a real repair pass (and OCR spend) even under
        ``--native-only``, undoing the settled ruling that the flag is
        honoured downstream, not routed on."""
        state = DocumentState(handle=_bare_handle())
        ps = state.pages[1]
        ps.is_born_digital = True
        ps.native_text = "native text"
        ps.native_table_structure_defective = True
        # Post-``_score_per_page`` shape: best_output cleared, the demoted
        # native attempt is the only attempt.
        ps.attempts.append(
            PageOutput(
                page_num=1,
                text="native text",
                status=PageStatus.WARNING,
                engine="native",
                audit_passed=False,
                failure_mode=FailureMode.NATIVE_TABLE_STRUCTURE_FAILED,
            )
        )

        assert ps.needs_repair is False

    def test_manifest_refuses_to_freeze_a_passing_native_defective_best_output(self) -> None:
        """(3a) Manifest lock #1. Even if a flagged native attempt somehow
        reached ``best_output`` with ``audit_passed=True`` (the PP-7-R1 bug
        shape), the manifest must not freeze it as the winner. Reachable
        only via this direct construction -- see the ticket fix log's note
        that every live path already demotes before the manifest runs."""
        state = DocumentState(handle=_bare_handle())
        ps = state.pages[1]
        ps.is_born_digital = True
        ps.native_text = "native text"
        ps.native_table_structure_defective = True
        bo = PageOutput(
            page_num=1,
            text="native text",
            status=PageStatus.SUCCESS,
            engine="native",
            audit_passed=True,
        )
        ps.attempts.append(bo)
        ps.best_output = bo

        out = _winning_page_output(state, 1)

        assert out is not bo
        assert out.status == PageStatus.WARNING
        assert out.audit_passed is False
        assert out.failure_mode == FailureMode.NATIVE_TABLE_STRUCTURE_FAILED

    def test_manifest_native_is_fallback_fires_on_the_flag_alone(self) -> None:
        """(3b/4) Manifest lock #2. With no ``best_output`` and no OTHER
        deficiency flag set (not ``needs_ocr_enhancement``, not
        ``native_table_structure_failed``, not ``chart_asset_render_failed``),
        the B1 flag alone must OR into ``native_is_fallback`` so the shipped
        page is WARNING, never a re-stamped SUCCESS."""
        state = DocumentState(handle=_bare_handle())
        ps = state.pages[1]
        ps.is_born_digital = True
        ps.native_text = "native text"
        ps.native_table_structure_defective = True
        ps.attempts.append(
            PageOutput(
                page_num=1,
                text="native text",
                status=PageStatus.WARNING,
                engine="native",
                audit_passed=False,
            )
        )

        out = _winning_page_output(state, 1)

        assert out.status == PageStatus.WARNING
        assert out.audit_passed is False
        assert out.failure_mode == FailureMode.NATIVE_TABLE_STRUCTURE_FAILED

    def test_manifest_passing_non_native_winner_is_unaffected_by_the_flag(self) -> None:
        """(6) The flag only gates the NATIVE attempt. A passing non-native
        (e.g. VLM) winner must return immediately, unmodified -- the flag on
        a native page must never veto an already-won non-native page."""
        state = DocumentState(handle=_bare_handle())
        ps = state.pages[1]
        ps.is_born_digital = True
        ps.native_text = "native text"
        ps.native_table_structure_defective = True
        bo = PageOutput(
            page_num=1,
            text="qwen text",
            status=PageStatus.SUCCESS,
            engine="qwen",
            audit_passed=True,
        )
        ps.attempts.append(bo)
        ps.best_output = bo

        out = _winning_page_output(state, 1)

        assert out is bo
        assert out.status == PageStatus.SUCCESS
        assert out.audit_passed is True

    def test_clear_fail_closed_flags_releases_the_flag(self) -> None:
        """(5) A measured improvement (table escalation accepted) must
        release the flag via ``_clear_fail_closed_flags``, or a page whose
        text was genuinely fixed ships WARNING forever."""
        state = DocumentState(handle=_bare_handle())
        ps = state.pages[1]
        ps.native_table_structure_defective = True
        profile = SimpleNamespace(engine=SimpleNamespace(value="qwen"))

        UnifiedPipeline._clear_fail_closed_flags(state, 1, ps, profile)

        assert ps.native_table_structure_defective is False
        cleared_events = [
            e for e in state.events if e.kind == "table_escalation_recovered_fail_closed"
        ]
        assert len(cleared_events) == 1
        assert "native_table_structure_defective" in cleared_events[0].detail


# ---------------------------------------------------------------------------
# BLOCKING review finding: a page carrying all three flags at once
# (``native_table_structure_defective``, ``native_table_structure_failed``,
# ``native_table_unverifiable``) must land in ``d3_floor_pages`` ONLY -- not
# also in ``native_fallback_pages``. The B1 disjunct was originally placed
# OUTSIDE the TR-3 D3 exclusion, double-counting this exact combination and
# producing two contradictory audit records for the same page (this ticket's
# own synthetic fixture yields ``native_table_unverifiable: true``, so the
# three-flag combination is not hypothetical -- see the fix log).
# ---------------------------------------------------------------------------


class TestThreeFlagPageIsNotDoubleCounted:
    def test_defective_and_unverifiable_without_structure_failed_lands_in_native_fallback(
        self, tmp_path: Path
    ) -> None:
        """B1's defective flag and TR-3's per-region unverifiable flag are set
        independently. A page can carry ``native_table_structure_defective``
        and ``native_table_unverifiable`` while ``native_table_structure_failed``
        stays False (short-circuited before the heuristic scorer runs). That
        page is NOT a D3 floor page (D3 requires ``native_table_structure_failed``
        too), so excluding on ``native_table_unverifiable`` alone -- rather
        than on the exact d3_floor_pages predicate -- would silently drop it
        from BOTH lists, hiding a WARNING/audit_passed=False page from every
        document-level failure surface."""
        state = DocumentState(handle=_bare_handle())
        ps = state.pages[1]
        ps.is_born_digital = True
        ps.native_text = "| a | b |\n| --- | --- |\n| 1 | 2 |"
        ps.native_table_structure_defective = True
        ps.native_table_structure_failed = False
        ps.native_table_unverifiable = True
        ps.attempts.append(
            PageOutput(
                page_num=1,
                text="ocr attempt",
                status=PageStatus.WARNING,
                engine="qwen",
                audit_passed=False,
            )
        )

        pipe = UnifiedPipeline(PipelineConfig(quiet=True))
        pipe._phase_assemble(state, tmp_path)

        kinds = [e.kind for e in state.events if e.page_num == 1]
        assert "table_region_unverifiable" not in kinds  # not a D3 floor page
        assert "native_fallback" in kinds  # must surface, not be silently dropped

    def test_three_flag_page_lands_only_in_d3_floor(self, tmp_path: Path) -> None:
        state = DocumentState(handle=_bare_handle())
        ps = state.pages[1]
        ps.is_born_digital = True
        ps.native_text = "| a | b |\n| --- | --- |\n| 1 | 2 |"
        ps.native_table_structure_defective = True
        ps.native_table_structure_failed = True
        ps.native_table_unverifiable = True
        ps.attempts.append(
            PageOutput(
                page_num=1,
                text="ocr attempt",
                status=PageStatus.WARNING,
                engine="qwen",
                audit_passed=False,
            )
        )

        pipe = UnifiedPipeline(PipelineConfig(quiet=True))
        pipe._phase_assemble(state, tmp_path)

        kinds = [e.kind for e in state.events if e.page_num == 1]
        assert kinds.count("table_region_unverifiable") == 1  # d3_floor_pages event
        assert "native_fallback" not in kinds  # NOT also in native_fallback_pages

    def test_three_flag_page_with_needs_ocr_enhancement_lands_only_in_d3_floor(
        self, tmp_path: Path
    ) -> None:
        """Review round 2: the same D3-floor page (``native_table_structure_failed``
        AND ``native_table_unverifiable``) but ALSO carrying
        ``needs_ocr_enhancement=True`` (e.g. a corrupt-math page whose table
        region separately hard-fails TR-3's per-region geometry check). The
        exclusion previously lived inside the ``native_table_structure_failed``
        disjunct only, so ``needs_ocr_enhancement`` -- the FIRST disjunct --
        satisfied ``native_fallback_pages`` unconditionally and bypassed it,
        double-counting the page. Proven load-bearing: reverting the exclusion
        to live inside the disjunct (its round-1 placement) makes this fail.
        """
        state = DocumentState(handle=_bare_handle())
        ps = state.pages[1]
        ps.is_born_digital = True
        ps.native_text = "| a | b |\n| --- | --- |\n| 1 | 2 |"
        ps.needs_ocr_enhancement = True
        ps.native_table_structure_defective = True
        ps.native_table_structure_failed = True
        ps.native_table_unverifiable = True
        ps.attempts.append(
            PageOutput(
                page_num=1,
                text="ocr attempt",
                status=PageStatus.WARNING,
                engine="qwen",
                audit_passed=False,
            )
        )

        pipe = UnifiedPipeline(PipelineConfig(quiet=True))
        pipe._phase_assemble(state, tmp_path)

        kinds = [e.kind for e in state.events if e.page_num == 1]
        assert kinds.count("table_region_unverifiable") == 1  # d3_floor_pages event
        assert "native_fallback" not in kinds  # NOT also in native_fallback_pages


# ---------------------------------------------------------------------------
# GH-200: the header-attribution term (record and surface, never reroute).
# A page whose grid is RECTANGULAR (B1's own grid-shape predicate does NOT
# fire, and every numeral is correct so TR-3 would not fire either) but whose
# header band is destroyed must still be demoted, end to end, under
# --native-only.
# ---------------------------------------------------------------------------

_HDR_LABEL_X = _LABEL_X
_HDR_VALUE_XS = _VALUE_XS
_HDR_HEADER = ("Currency", "Low%", "Mid%", "High%", "Wide%", "Total%", "Extra%")
_HDR_ROWS: list[tuple[str, tuple[str, ...]]] = [
    ("alpha", ("1.0", "2.0", "3.0", "4.0", "5.0", "6.0")),
    ("beta", ("1.1", "2.1", "3.1", "4.1", "5.1", "6.1")),
    ("gamma", ("1.2", "2.2", "3.2", "4.2", "5.2", "6.2")),
]


def _destroyed_header_table_pdf(path: Path) -> None:
    """A rectangular, non-ragged table whose native header words are real,
    but whose emitted markdown (patched in the test) blanks them entirely."""
    doc = fitz.open()
    page = doc.new_page()

    top = 170.0
    y = top + _ROW_H
    for i, cell in enumerate(_HDR_HEADER):
        x = _HDR_LABEL_X if i == 0 else _HDR_VALUE_XS[i - 1]
        page.insert_text((x, y), cell, fontsize=9, fontname="helv")
    y += _ROW_H
    for label, values in _HDR_ROWS:
        page.insert_text((_HDR_LABEL_X, y), label, fontsize=9, fontname="helv")
        for x, v in zip(_HDR_VALUE_XS, values):
            page.insert_text((x, y), v, fontsize=9, fontname="helv")
        y += _ROW_H

    page.draw_line(fitz.Point(50, top), fitz.Point(500, top))
    page.draw_line(fitz.Point(50, top + _ROW_H), fitz.Point(500, top + _ROW_H))
    page.draw_line(fitz.Point(50, y), fitz.Point(500, y))

    doc.save(str(path))
    doc.close()


def _blanked_header_markdown() -> str:
    n_cols = len(_HDR_ROWS[0][1]) + 1
    # Label cell survives (an all-blank row is dropped by the parser's own
    # separator-row blind spot, see structure_check.py's module docstring
    # and test_header_attribution.py::test_missing_header_band_is_hard);
    # every data-lane header cell is blank.
    blank_row = "| " + _HDR_HEADER[0] + " | " + " | ".join([""] * (n_cols - 1)) + " |"
    sep_row = "| " + " | ".join(["---"] * n_cols) + " |"
    rows = [blank_row, sep_row]
    for label, values in _HDR_ROWS:
        rows.append(f"| {label} | {' | '.join(values)} |")
    return "\n".join(rows)


class TestHeaderAttributionEndToEnd:
    @pytest.mark.xfail(
        strict=True,
        reason=(
            "GH-200: the header-attribution reject disjunct is parked in "
            "table_output_defect. The REQUIREMENT is unchanged -- a header "
            "defect must be recorded and surfaced under --native-only -- but "
            "every predicate tried so far also returns HARD on byte-perfect "
            "correct tables (significance-star and n.a. rows), and a false "
            "reject destroys good output. This flips to XPASS the moment a "
            "sound predicate is wired back in."
        ),
    )
    def test_native_only_records_header_defect_without_rerouting(self, tmp_path: Path) -> None:
        """process() on a generated born-digital table PDF with native_only.

        The emitted markdown (stubbed via ``extract_structured``) is
        rectangular -- B1's own grid-shape predicate must NOT fire -- but its
        header row is entirely blank while the native page carries real
        header words over every data lane, so the header-attribution HARD
        verdict must fire and demote the page without ever consulting the
        OCR ladder.
        """
        pdf_path = tmp_path / "doc.pdf"
        _destroyed_header_table_pdf(pdf_path)
        out_dir = tmp_path / "out"

        blanked_md = _blanked_header_markdown()
        reports = structure_check.check_markdown(blanked_md)
        assert not _gate_fires(reports), (
            f"setup precondition failed: the blanked-header fixture must NOT "
            f"trip the grid-shape gate on its own; reports: {reports}"
        )

        with patch.object(BornDigitalDetector, "extract_structured", return_value=blanked_md):
            pipeline = UnifiedPipeline(_config())
            with patch.object(
                pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]
            ):
                result = pipeline.process(pdf_path, out_dir)

        assert result.status == DocumentStatus.AUDIT_FAILED

        audit_log_path = out_dir / "doc" / "audit_log.json"
        if not audit_log_path.exists():
            candidates = list(out_dir.rglob("audit_log.json"))
            assert candidates, f"no audit_log.json found under {out_dir}"
            audit_log_path = candidates[0]
        audit_log = json.loads(audit_log_path.read_text())
        events = audit_log.get("events", [])
        fired = [e for e in events if e.get("kind") == "table_structure_failed"]
        assert len(fired) == 1, f"expected exactly one table_structure_failed event, got {events}"

        sidecar_candidates = list(out_dir.rglob("pages/00001.json"))
        assert sidecar_candidates, f"no page sidecar found under {out_dir}"
        sidecar = json.loads(sidecar_candidates[0].read_text())
        assert sidecar["native_table_header_unattributed"] is True
        assert sidecar.get("native_table_structure_defective") is False
        winning = sidecar.get("winning_output") or sidecar
        assert sidecar.get("status") in ("warning", "WARNING") or winning.get("status") in (
            "warning",
            "WARNING",
        )
        assert winning.get("failure_mode") == "native_table_structure_failed"

        tables_trust_candidates = list(out_dir.rglob("tables_trust.json"))
        assert tables_trust_candidates, f"no tables_trust.json found under {out_dir}"
        tables_trust = json.loads(tables_trust_candidates[0].read_text())
        assert "1" in tables_trust.get("pages", {}), (
            f"tables_trust.json carries no distrust record for page 1: {tables_trust}"
        )
        assert "table_structure_failed" in tables_trust["pages"]["1"]["reasons"]

        # No-reroute pin: exactly one attempt (native), zero cost -- the OCR
        # ladder was never consulted even though the page was demoted.
        assert sidecar["engine"] == "native"
        assert sidecar["cost_usd"] == 0.0


# ---------------------------------------------------------------------------
# GH-200: post-route recheck. The GH-56 header repair mutates
# ``ps.best_output.text`` AFTER the judge already accepted it, so the shipped
# text may not be the text the structural gate saw. If repair produces a
# ragged/detached-label grid, the page must be demoted IN PLACE (WARNING /
# audit_passed=False), with exactly one audit event for the page.
# ---------------------------------------------------------------------------


class TestPostRouteHeaderRepairRecheck:
    def test_post_route_header_repair_recheck(self, tmp_path: Path) -> None:
        from socr.core.config import EngineType, PipelineConfig
        from socr.core.providers import PROFILE_QWEN_LOCAL
        from socr.pipeline.agentic import PageDecision, ProviderAttempt

        clean_text = (
            "| Forecast | 2026 | 2027 |\n| --- | --- | --- |\n| A | 1.2 | 1.3 |\n| B | 2.1 | 2.2 |"
        )
        # A grid ``check_markdown`` genuinely reports as ragged -- rows of
        # inconsistent width, the same shape ``structural_gate_fires`` gates
        # on elsewhere in this file.
        ragged_after_repair = "| Forecast | 2026 | 2027 |\n| A | 1.2 |\n| B | 2.1 | 2.2 | 2.3 |"
        reports = structure_check.check_markdown(ragged_after_repair)
        assert _gate_fires(reports), (
            f"setup precondition failed: the post-repair fixture must trip "
            f"the grid-shape gate; reports: {reports}"
        )

        doc = fitz.open()
        doc.new_page()
        pdf_path = tmp_path / "doc.pdf"
        doc.save(str(pdf_path))
        doc.close()

        config = PipelineConfig(
            primary_engine=EngineType.QWEN,
            agentic=True,
            judge_backend="heuristic",
            enabled_engines=[EngineType.QWEN],
            quiet=True,
            audit_enabled=True,
            save_figures=False,
            write_manifest=False,
            native_first=False,
        )
        pipeline = UnifiedPipeline(config)

        def _accepted_route(page_num, ladder, run_provider, judge, **kwargs):
            out = PageOutput(
                page_num=page_num,
                text=clean_text,
                status=PageStatus.SUCCESS,
                engine="qwen",
                audit_passed=True,
            )
            prof = ladder[0]
            att = ProviderAttempt(
                engine=prof.engine,
                output=out,
                cost_usd=0.0,
                accepted=True,
                reason="stub-accept",
                provider_id=prof.id,
                model=prof.model,
                backend=prof.backend,
            )
            return PageDecision(page_num=page_num, final_output=out, attempts=[att])

        with (
            patch.object(
                pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]
            ),
            patch("socr.pipeline.orchestrator.route_page", side_effect=_accepted_route),
            patch("socr.pipeline.orchestrator.probe_ollama_idle", return_value=True),
            patch(
                "socr.tables.header_repair.repair_table_headers_on_page",
                return_value=(ragged_after_repair, 1),
            ),
            # Force this synthetic (tableless-looking) page through the
            # header-repair block, which is gated on ``_page_has_tables``.
            patch.object(UnifiedPipeline, "_page_has_tables", return_value=True),
        ):
            result = pipeline.process(pdf_path, tmp_path / "out")

        assert result.status == DocumentStatus.AUDIT_FAILED

        out_dir = tmp_path / "out"
        audit_log_path = out_dir / "doc" / "audit_log.json"
        if not audit_log_path.exists():
            candidates = list(out_dir.rglob("audit_log.json"))
            assert candidates, f"no audit_log.json found under {out_dir}"
            audit_log_path = candidates[0]
        audit_log = json.loads(audit_log_path.read_text())
        events = [e for e in audit_log.get("events", []) if e.get("page_num") == 1]
        fired = [e for e in events if e.get("kind") == "table_structure_failed"]
        assert len(fired) == 1, f"expected exactly one audit event for page 1, got {events}"
        assert fired[0].get("data", {}).get("site") == "post_route_recheck"

        sidecar_candidates = list(out_dir.rglob("pages/00001.json"))
        assert sidecar_candidates, f"no page sidecar found under {out_dir}"
        sidecar = json.loads(sidecar_candidates[0].read_text())
        winning = sidecar.get("winning_output") or sidecar
        assert sidecar.get("status") in ("warning", "WARNING") or winning.get("status") in (
            "warning",
            "WARNING",
        )
        assert sidecar.get("audit_passed") is False or winning.get("audit_passed") is False
