"""P6 stage A/B, cold review round 2: the reverted behaviour, pinned.

Four of round 2's findings were out-of-scope behaviour changes that had to go back to
HEAD's behaviour (1, 2, 5) or a genuine hole in the new public field (3). Each is
pinned here with the shape the reviewer measured, so a later stage that WANTS one of
these changes has to state it rather than reintroduce it.

Hermetic: builds ``DocumentState`` directly and drives selection / ``_phase_assemble``.
No provider ladder, no judge, no network. Every assertion is a DIFFERENCE between two
runs of the same fixture that vary in one thing, never an absolute measured tuple.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")

from socr.core.config import EngineType, PipelineConfig  # noqa: E402
from socr.core.document import DocumentHandle  # noqa: E402
from socr.core.manifest import (  # noqa: E402
    PageEnding,
    PagePrimaryReason,
    _apply_ladder_disposition_guard,
    finalized_page_record,
    is_page_failed_marker,
    page_failed_marker,
)
from socr.core.result import FailureMode, PageOutput, PageStatus  # noqa: E402
from socr.core.state import DocumentState  # noqa: E402

NATIVE_BODY = "NATIVE BODY, a real text layer with enough words to be a page."
REJECTED_MODEL_BODY = "REJECTED MODEL BODY, the reading the ladder refused."

#: A strict grid carrying a live LaTeX leak. It is simultaneously a valid
#: authored-grid candidate (``has_strict_table_grid``), a ``kept_table_grid_defect``
#: of ``table_latex_leak``, and an emission defect -- which is the whole point of the
#: finding-5 shape: the candidate is kept and flagged, then the final emission guard
#: replaces the body with a marker.
LEAKY_GRID = (
    "Prose above the table.\n\n"
    "| a | b | c |\n"
    "|---|---|---|\n"
    "| \\multicolumn{2}{c}{x} | 1 | 2 |\n"
    "| p | 3 | 4 |\n\n"
    "Prose below.\n"
)


def _pdf(tmp_path: Path, name: str = "d.pdf") -> Path:
    path = tmp_path / name
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "born digital page")
    doc.save(str(path))
    doc.close()
    return path


def _state(tmp_path: Path) -> DocumentState:
    return DocumentState(handle=DocumentHandle.from_path(_pdf(tmp_path)))


class TestFinding1SelectorIsNotWidenedByALadderDisposition:
    """A ladder disposition must not change WHICH output wins selection.

    HEAD's predicate is exactly ``p.best_output and p.best_output.audit_passed``. The
    reviewed tree added ``or p.table_ladder_disposition is not None``, which promotes
    an audit-REJECTED ``best_output`` into the passing arm -- so the shipped bytes and
    the recorded engine both changed. Stage A/B forbids selector changes.
    """

    def _page(
        self, tmp_path: Path, *, ladder: FailureMode | None, has_tables: bool = True
    ) -> DocumentState:
        state = _state(tmp_path)
        ps = state.pages[1]
        ps.is_born_digital = True
        ps.native_text = NATIVE_BODY
        # A table page routes through the structure-class branch; a prose page does
        # not. Both reviewed shapes are covered, because the widened predicate lands
        # them on different wrong answers -- native prose instead of a fail-closed
        # marker on the first, and the rejected model reading itself on the second.
        ps.has_tables = has_tables
        ps.needs_ocr_enhancement = True
        rejected = PageOutput(
            page_num=1,
            text=REJECTED_MODEL_BODY,
            status=PageStatus.SUCCESS,
            engine="qwen",
            audit_passed=False,
            failure_mode=FailureMode.TABLE_REJECTED,
        )
        ps.attempts.append(rejected)
        ps.best_output = rejected
        ps.table_ladder_disposition = ladder
        return state

    def test_the_ladder_disposition_does_not_change_which_bytes_win(self, tmp_path: Path) -> None:
        """The difference: one thing varies, and the winning TEXT must not."""
        without = finalized_page_record(self._page(tmp_path, ladder=None), 1).output
        with_ladder = finalized_page_record(
            self._page(tmp_path, ladder=FailureMode.TABLE_REJECTED), 1
        ).output

        assert with_ladder.text == without.text
        assert with_ladder.engine == without.engine
        assert with_ladder.text != REJECTED_MODEL_BODY, (
            "an audit-rejected best_output must never win selection outright"
        )

    def test_a_rejected_best_output_does_not_ship_as_success(self, tmp_path: Path) -> None:
        """The reviewer's second shape: a PROSE page, where nothing distrusts native.

        Here the widened predicate returns the rejected reading outright, so the
        sidecar engine and status flip from ``native/warning`` to ``qwen/success``.
        """
        out = finalized_page_record(
            self._page(tmp_path, ladder=FailureMode.TABLE_REJECTED, has_tables=False), 1
        ).output
        assert not (out.status is PageStatus.SUCCESS and out.engine == "qwen"), (
            "the rejected model reading shipped as a clean SUCCESS: the selector "
            "predicate has been widened again"
        )
        assert out.engine.startswith("native")
        assert out.audit_passed is False

    def test_the_prose_shape_also_keeps_its_bytes(self, tmp_path: Path) -> None:
        """Same difference, stated on the prose page: the ladder changes no bytes."""
        without = finalized_page_record(
            self._page(tmp_path, ladder=None, has_tables=False), 1
        ).output
        with_ladder = finalized_page_record(
            self._page(tmp_path, ladder=FailureMode.TABLE_REJECTED, has_tables=False), 1
        ).output
        assert with_ladder.text == without.text
        assert with_ladder.engine == without.engine

    def test_the_backfill_shape_keeps_its_sidecar_engine_and_status(self, tmp_path: Path) -> None:
        """The reviewer's second shape, end to end through ``_phase_assemble``.

        A born-digital page whose native body carries a GFM table nothing witnessed:
        ``_backfill_missing_table_ladder_terminals`` stamps TABLE_UNVERIFIED on the
        page, and the sidecar's record is finalized after that. Under the widened
        predicate the sidecar flipped from ``native/warning`` to ``qwen/success``.
        The saved body stays native either way -- ``saved_text`` overwrites the newly
        selected text -- which is exactly why the assertion is on the sidecar.
        """
        import json

        from socr.pipeline import orchestrator as orch

        state = _state(tmp_path)
        ps = state.pages[1]
        ps.is_born_digital = True
        ps.native_text = "Prose.\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\nMore prose.\n"
        ps.has_tables = False
        rejected = PageOutput(
            page_num=1,
            text=REJECTED_MODEL_BODY,
            status=PageStatus.SUCCESS,
            engine="qwen",
            audit_passed=False,
            failure_mode=FailureMode.NONE,
        )
        ps.attempts.append(rejected)
        ps.best_output = rejected

        config = PipelineConfig(
            primary_engine=EngineType.QWEN,
            agentic=False,
            enabled_engines=[EngineType.QWEN],
            quiet=True,
            save_figures=False,
            write_manifest=False,
            table_judge_ladder=True,
        )
        pipeline = orch.UnifiedPipeline(config)
        pipeline._scan_root = state.handle.path.parent
        out_dir = tmp_path / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        pipeline._phase_assemble(state, out_dir)

        # Fixture premise: the backfill really did stamp the page, else the widened
        # predicate would have had nothing to read and this would pass vacuously.
        assert ps.table_ladder_disposition is FailureMode.TABLE_UNVERIFIED

        sidecar = json.loads(next(out_dir.rglob("pages/00001.json")).read_text())
        winner = sidecar["winning_output"]
        assert winner["engine"] == "native", winner
        assert winner["status"] == PageStatus.WARNING.value, winner


class TestRound3RestoredDispositionNeverOutranksTheShippedBytes:
    """A restored disposition is a BASE, not a verdict.

    Cold review round 3. ``PageState.resumed_disposition`` exists so a resume that
    reprocessed nothing reproduces its sidecar byte for byte. It was applied LAST --
    after the saved-body replacement, the table-emission guard and the shared marker
    recogniser -- so on a page whose FINAL body tripped the guard it overwrote a
    correctly computed fail-closed disposition with the old sidecar value. The page
    shipped ``[page N failed: invalid table emission — …]`` while its published
    disposition still said ``MODEL_OUTPUT / ACCEPTED_OUTPUT``.

    The two cases below are the same document with one field changed, so they pin the
    boundary from both sides: bytes unchanged keeps the restored value, bytes rewritten
    by the guard does not.
    """

    #: Three header cells, two delimiter cells. A width-mismatched GFM table, which
    #: the final emission guard replaces with a whole-page marker.
    MALFORMED_GFM = "Prose above.\n\n| a | b | c |\n|---|---|\n| 1 | 2 | 3 |\n\nProse below.\n"
    CLEAN_BODY = "An ordinary accepted page with nothing wrong with it at all.\n"

    ACCEPTED = {"ending": "model_output", "primary_reason": "accepted_output"}

    def _assemble(self, tmp_path: Path, *, body: str, resumed: dict | None):
        """A two-page document: page 1 under test, page 2 clean so the document ships.

        Page 2 is not decoration. With a single failing page the document has no text,
        so no ``.md`` and no manifest are written and the assertion would have nothing
        to read.
        """
        import json

        from socr.pipeline import orchestrator as orch

        tmp_path.mkdir(parents=True, exist_ok=True)
        pdf = tmp_path / "resumed.pdf"
        doc = fitz.open()
        for _ in range(2):
            doc.new_page().insert_text((72, 72), "born digital page")
        doc.save(str(pdf))
        doc.close()
        state = DocumentState(handle=DocumentHandle.from_path(pdf))

        for page_num, text in ((1, body), (2, self.CLEAN_BODY)):
            ps = state.pages[page_num]
            out = PageOutput(
                page_num=page_num,
                text=text,
                status=PageStatus.SUCCESS,
                engine="qwen",
                audit_passed=True,
                failure_mode=FailureMode.NONE,
            )
            ps.attempts.append(out)
            ps.best_output = out
        state.pages[1].resumed_disposition = resumed

        config = PipelineConfig(
            primary_engine=EngineType.QWEN,
            agentic=False,
            enabled_engines=[EngineType.QWEN],
            quiet=True,
            save_figures=False,
            write_manifest=True,
            # docs/log/2026-09-03_p1-prep-latch-and-audit.md (cold review round 1)
            table_judge_ladder=False,
        )
        pipeline = orch.UnifiedPipeline(config)
        out_dir = tmp_path / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        pipeline._phase_assemble(state, out_dir)

        sidecar = json.loads(next(out_dir.rglob("pages/00001.json")).read_text())
        manifest_path = next(out_dir.rglob("manifest.json"), None)
        manifest_entry = (
            json.loads(manifest_path.read_text())["entries"]["1"] if manifest_path else None
        )
        body_md = next(p for p in out_dir.rglob("*.md") if p.parent.name != "pages").read_text()
        return sidecar, manifest_entry, body_md

    FAIL_CLOSED = {"ending": "fail_closed_marker", "primary_reason": "invalid_table_emission"}

    def test_a_guard_rewritten_body_outranks_the_restored_disposition(self, tmp_path: Path) -> None:
        sidecar, entry, body_md = self._assemble(
            tmp_path, body=self.MALFORMED_GFM, resumed=self.ACCEPTED
        )

        # Fixture premise: the page really did ship the marker on every byte surface.
        assert sidecar["winning_output"]["failure_mode"] == "table_emission_invalid"
        assert "invalid table emission" in sidecar["winning_output"]["text"]
        assert "invalid table emission" in body_md

        assert sidecar["disposition"] == self.FAIL_CLOSED, (
            "the restored disposition overwrote a fail-closed page: it is a base, and "
            "may never outrank a guard that rewrote the current shipped bytes"
        )
        assert entry is not None and entry["disposition"] == self.FAIL_CLOSED

    def test_the_same_page_without_a_restored_disposition_agrees(self, tmp_path: Path) -> None:
        """The control: restoring nothing must reach the same answer."""
        sidecar, entry, _ = self._assemble(tmp_path, body=self.MALFORMED_GFM, resumed=None)
        assert sidecar["disposition"] == self.FAIL_CLOSED
        assert entry is not None and entry["disposition"] == self.FAIL_CLOSED

    def test_an_unchanged_resume_still_keeps_its_restored_disposition(self, tmp_path: Path) -> None:
        """The other side of the boundary: no guard fired, so stability wins.

        Without this the ordering fix could be "ignore the restored value", which would
        put the resume sidecar back to rewriting itself on a run that reprocessed
        nothing.
        """
        restored = {"ending": "model_output", "primary_reason": "unaccepted_output_kept"}
        with_restore, entry, _ = self._assemble(tmp_path, body=self.CLEAN_BODY, resumed=restored)
        assert with_restore["disposition"] == restored
        assert entry is not None and entry["disposition"] == restored

        without, _, _ = self._assemble(tmp_path / "b", body=self.CLEAN_BODY, resumed=None)
        assert without["disposition"] != restored, (
            "fixture premise: the restored value must differ from the recomputed one, "
            "else this test cannot tell stabilisation from coincidence"
        )


class TestFinding2LadderGuardDoesNotOverwriteATerminal:
    """The page-level ladder disposition is stamped ONLY onto an empty failure mode.

    HEAD's condition is ``output.failure_mode is FailureMode.NONE``. Widening it to
    also accept a mode already in ``_LADDER_TERMINAL_FAILURE_MODES`` replaces one
    recorded terminal with another, changing the shipped output, the sidecar, the
    manifest blob and the retry semantics.
    """

    def _out(self, mode: FailureMode) -> PageOutput:
        return PageOutput(
            page_num=1,
            text="body",
            status=PageStatus.WARNING,
            engine="qwen",
            audit_passed=False,
            failure_mode=mode,
        )

    class _Page:
        def __init__(self, disposition: FailureMode) -> None:
            self.table_ladder_disposition = disposition

    def test_an_existing_terminal_is_not_replaced_by_a_different_one(self) -> None:
        guarded = _apply_ladder_disposition_guard(
            self._out(FailureMode.TABLE_REJECTED), 1, self._Page(FailureMode.TABLE_UNVERIFIED)
        )
        assert guarded.failure_mode is FailureMode.TABLE_REJECTED, (
            "TABLE_REJECTED was overwritten with the page's TABLE_UNVERIFIED "
            "disposition; the guard may only fill an EMPTY failure mode"
        )

    def test_an_empty_failure_mode_is_still_stamped(self) -> None:
        """The control: the guard's own job still happens."""
        guarded = _apply_ladder_disposition_guard(
            self._out(FailureMode.NONE), 1, self._Page(FailureMode.TABLE_UNVERIFIED)
        )
        assert guarded.failure_mode is FailureMode.TABLE_UNVERIFIED


class TestFinding3TheEndingIsReadFromTheShippedBytes:
    """Any recognised whole-page failure marker is FAIL_CLOSED_MARKER.

    The reviewed tree recognised only the table-emission marker, so a page whose body
    is ``[page 1 failed: timeout during extraction]`` was published as
    ``(MODEL_OUTPUT, UNACCEPTED_OUTPUT_KEPT)`` -- a page that shipped no content at
    all, described as a kept model reading.
    """

    #: One per marker family the tree can author, plus a free-form marker no builder
    #: writes today. The last one is the reviewer's shape and the reason the
    #: recogniser, not a family list, decides the ENDING.
    MARKERS = (
        page_failed_marker(1),
        "[page 1 failed: unverifiable table — see image]",
        "[page 1 failed: rotated text extraction shredded — see image]",
        "[page 1 failed: invalid table emission — table_width_mismatch]",
        "[page 1 failed: timeout during extraction]",
    )

    def _record(self, tmp_path: Path, text: str):
        state = _state(tmp_path)
        ps = state.pages[1]
        best = PageOutput(
            page_num=1,
            text=text,
            status=PageStatus.SUCCESS,
            engine="qwen",
            audit_passed=False,
            failure_mode=FailureMode.AUDIT_FAILED,
        )
        ps.attempts.append(best)
        ps.best_output = best
        return finalized_page_record(state, 1)

    @pytest.mark.parametrize("marker", MARKERS)
    def test_every_recognised_marker_is_fail_closed(self, tmp_path: Path, marker: str) -> None:
        assert is_page_failed_marker(marker), "fixture premise: the shared recogniser sees it"
        record = self._record(tmp_path, marker)
        assert record.disposition.ending is PageEnding.FAIL_CLOSED_MARKER, (
            f"{marker!r} shipped as {record.disposition}"
        )

    def test_the_reason_names_which_marker_shipped(self, tmp_path: Path) -> None:
        """Not merely fail-closed: the reason distinguishes the families."""
        reasons = {m: self._record(tmp_path, m).disposition.primary_reason for m in self.MARKERS}
        assert reasons[page_failed_marker(1)] is PagePrimaryReason.NO_USABLE_OUTPUT
        assert (
            reasons["[page 1 failed: rotated text extraction shredded — see image]"]
            is PagePrimaryReason.ROTATED_NATIVE_TEXT_SHREDDED
        )
        assert (
            reasons["[page 1 failed: invalid table emission — table_width_mismatch]"]
            is PagePrimaryReason.INVALID_TABLE_EMISSION
        )
        # A marker no builder authors cannot be attributed, and says so rather than
        # borrowing a cause it cannot read off the bytes.
        assert (
            reasons["[page 1 failed: timeout during extraction]"]
            is PagePrimaryReason.SHIPPED_FAILURE_MARKER
        )

    def test_ordinary_content_is_untouched(self, tmp_path: Path) -> None:
        """The control: recognising markers must not reclassify real pages."""
        record = self._record(tmp_path, "Ordinary page prose that is not a marker at all.")
        assert record.disposition.ending is not PageEnding.FAIL_CLOSED_MARKER


class TestFinding5TheFlaggedModelEventNamesTheCandidatesDefect:
    """``_kept_defect`` inspects ``best_output``, not the guard-rewritten record.

    The event exists to name the defect in the grid that was KEPT. Reading the
    finalized record instead reads it after the emission guard has replaced the body
    with a marker, so the defect the event is for silently became "".
    """

    def _assemble(self, tmp_path: Path):
        state = _state(tmp_path)
        ps = state.pages[1]
        ps.is_born_digital = True
        ps.native_text = "Native prose with an ungridded table region."
        ps.has_tables = True
        ps.native_table_structure_defective = True
        leaky = PageOutput(
            page_num=1,
            text=LEAKY_GRID,
            status=PageStatus.SUCCESS,
            engine="gemini",
            audit_passed=False,
            failure_mode=FailureMode.NONE,
        )
        leaky.rejection_class = "judge_only"
        ps.attempts.append(leaky)
        ps.best_output = leaky

        from socr.pipeline import orchestrator as orch

        config = PipelineConfig(
            primary_engine=EngineType.QWEN,
            agentic=False,
            enabled_engines=[EngineType.QWEN],
            quiet=True,
            save_figures=False,
            write_manifest=False,
            # docs/log/2026-09-03_p1-prep-latch-and-audit.md (cold review round 1)
            table_judge_ladder=False,
        )
        pipeline = orch.UnifiedPipeline(config)
        out_dir = tmp_path / "out"
        out_dir.mkdir(parents=True, exist_ok=True)

        buf = io.StringIO()
        from rich.console import Console

        real = orch.console
        orch.console = Console(file=buf, force_terminal=False, width=100_000, no_color=True)
        try:
            pipeline._phase_assemble(state, out_dir)
        finally:
            orch.console = real
        return state

    def test_the_kept_grids_defect_reaches_the_audit_event(self, tmp_path: Path) -> None:
        state = self._assemble(tmp_path)
        events = [e for e in state.events if getattr(e, "kind", "") == "flagged_model_table_kept"]
        assert events, [getattr(e, "kind", "") for e in state.events]
        event = events[0]
        defect = (getattr(event, "data", None) or {}).get("grid_defect", "")
        assert defect == "table_latex_leak", (
            "the event no longer names the defect in the candidate it describes; "
            "it is reading the record after the emission guard replaced the body"
        )
        assert "table_latex_leak" in (getattr(event, "detail", "") or "")
