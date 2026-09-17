"""GH-674: an in-scope equation-sidecar skip must not ship SUCCESS with an
orphan crop.

#664 made the skip AUDITED (``equation_sidecar_skipped_no_page_output``, on
the resume allowlist), but the event had no consumer -- nothing demoted on
it, so a genuine in-scope miss shipped a clean SUCCESS while the crop PNG sat
on disk with no sidecar attached. This ticket demotes the document instead of
fabricating a ``PageOutput``, which #664 already ruled out ("inventing one
would be a back-door SUCCESS path").

Every test here pins the OUTCOME (``DocumentState``/``EngineResult`` status),
not the event -- a test asserting the event exists would have passed before
this fix, per the ticket's own framing.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

fitz = pytest.importorskip("fitz")

from socr.core.audit_log import AuditEvent
from socr.core.config import PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.result import DocumentStatus, PageOutput, PageStatus
from socr.core.state import DocumentState, PageState
from socr.pipeline.orchestrator import UnifiedPipeline

SKIP_KIND = "equation_sidecar_skipped_no_page_output"


def _pdf(tmp_path: Path, name: str = "doc.pdf") -> Path:
    path = tmp_path / name
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "Real prose that must survive. " * 6)
    doc.save(str(path))
    doc.close()
    return path


def _make_pipeline(**overrides) -> UnifiedPipeline:
    cfg = PipelineConfig(
        agentic=True,
        quiet=True,
        save_figures=False,
        recover_clean_equations=True,
        detect_equations=True,
        write_manifest=False,
        **overrides,
    )
    return UnifiedPipeline(cfg)


def _region_event(page_num: int, crop: Path, *, region_index: int = 0) -> AuditEvent:
    return AuditEvent(
        page_num=page_num,
        kind="equation_region_detected",
        engine="detect_equations",
        detail="test",
        data={
            "source_bbox": [0.0, 0.0, 1.0, 1.0],
            "padded_bbox": [0.0, 0.0, 1.0, 1.0],
            "has_eq_number": False,
            "crop_path": str(crop),
            "detection_time_s": 0.001,
            "source_text": "native prose",
            "equation_label": None,
            "region_index": region_index,
        },
    )


def _state_with_content(
    pdf_path: Path, page_num: int, text: str
) -> tuple[DocumentState, PageOutput]:
    """A page that ALREADY has a shipped result -- the shape the ticket
    describes: content ships from elsewhere while a region's sidecar-attach
    call (separately) never got a ``PageOutput`` to attach to."""
    state = DocumentState(handle=DocumentHandle.from_path(pdf_path))
    out = PageOutput(page_num=page_num, text=text, status=PageStatus.SUCCESS, engine="native")
    ps = state.pages[page_num]
    ps.is_born_digital = True
    ps.native_text = text
    ps.attempts.append(out)
    ps.best_output = out
    return state, out


class TestInScopeMissDemotesTheDocument:
    def test_document_status_is_not_success(self, tmp_path: Path) -> None:
        pdf_path = _pdf(tmp_path)
        state, _out = _state_with_content(pdf_path, 1, "Real prose that must survive.")
        crop = tmp_path / "equation_0_page1.png"
        crop.write_bytes(b"fakepng")
        state.events.append(_region_event(1, crop))

        orch = _make_pipeline()
        orch._scan_root = tmp_path

        # The in-scope miss: page 1 is in scope but this call gets NO
        # PageOutput for it (the defensive GH-157 branch).
        orch._attach_equation_latex_sidecars(state, [], page_nums=[1])
        skipped = [e for e in state.events if e.kind == SKIP_KIND]
        assert len(skipped) == 1, "setup: the skip event was never produced"
        assert state.pages[1].equation_sidecar_skipped is True, "setup: flag never set"

        out_dir = tmp_path / "out"
        result = orch._phase_assemble(state, out_dir)

        assert result.status != DocumentStatus.SUCCESS, (
            "an in-scope equation-sidecar miss still shipped a clean document: "
            f"status={result.status}, error={result.error!r}"
        )
        assert result.status == DocumentStatus.AUDIT_FAILED, result.status
        assert "1" in (result.error or ""), (
            f"the demotion must name the affected page: {result.error!r}"
        )
        # And the page's real content still ships -- this is a demotion, not
        # a fabrication or a content deletion.
        saved = "\n".join(p.read_text(encoding="utf-8") for p in out_dir.rglob("*.md"))
        assert "Real prose that must survive." in saved, saved

    def test_clean_run_with_no_miss_still_succeeds(self, tmp_path: Path) -> None:
        """Reverse regression: the new bucket must not demote a clean run."""
        pdf_path = _pdf(tmp_path)
        state, _out = _state_with_content(pdf_path, 1, "Real prose that must survive.")

        orch = _make_pipeline()
        orch._scan_root = tmp_path
        out_dir = tmp_path / "out"
        result = orch._phase_assemble(state, out_dir)

        assert result.status == DocumentStatus.SUCCESS, (
            f"a clean document with no equation-sidecar miss was demoted: "
            f"status={result.status}, error={result.error!r}"
        )


class TestOutOfScopeAttachedPageIsNotDemoted:
    """#664's ``page_nums`` scoping is load-bearing: a later page's call must
    not false-fire a skip -- or this ticket's flag -- for an earlier page
    that already attached. Breaking this regresses #664."""

    def test_only_the_missing_page_is_flagged(self, tmp_path: Path) -> None:
        pdf_path = _pdf(tmp_path)
        state = DocumentState(handle=DocumentHandle.from_path(pdf_path))
        for n in (1, 2):
            state.pages[n] = PageState(page_num=n, is_born_digital=True, native_text="region")

        orch = _make_pipeline()
        orch._scan_root = tmp_path

        # Page 1: a normal per-page call that DOES have its own PageOutput --
        # attaches cleanly, no miss.
        crop1 = tmp_path / "equation_region_1.png"
        crop1.write_bytes(b"fakepng")
        state.events.append(_region_event(1, crop1))
        po1 = PageOutput(page_num=1, text="region", status=PageStatus.SUCCESS, engine="native")
        state.pages[1].best_output = po1
        state.pages[1].attempts.append(po1)
        with patch("socr.math.equation_latex.latex_for_crop", return_value=""):
            orch._attach_equation_latex_sidecars(state, [po1], page_nums=[1])

        assert state.pages[1].equation_sidecar_skipped is False, (
            "setup: page 1 must attach cleanly, not miss"
        )

        # Page 2: the in-scope miss -- scoped call gets no PageOutput.
        crop2 = tmp_path / "equation_region_2.png"
        crop2.write_bytes(b"fakepng")
        state.events.append(_region_event(2, crop2))
        state.pages[2].native_text = "region 2"
        state.pages[2].best_output = PageOutput(
            page_num=2, text="region 2", status=PageStatus.SUCCESS, engine="native"
        )
        state.pages[2].attempts.append(state.pages[2].best_output)
        orch._attach_equation_latex_sidecars(state, [], page_nums=[2])

        # Page 1 must NOT be false-flagged by page 2's call.
        assert state.pages[1].equation_sidecar_skipped is False, (
            "an out-of-scope page that already attached was wrongly demoted"
        )
        assert state.pages[2].equation_sidecar_skipped is True

        out_dir = tmp_path / "out"
        result = orch._phase_assemble(state, out_dir)
        assert result.status == DocumentStatus.AUDIT_FAILED, result.status
        assert "2" in (result.error or "")


class TestFlagSurvivesResume:
    """If the demotion needs to survive resume -- it does, an orphan crop
    does not heal -- persist and OR-restore, per #682's pattern."""

    def test_flag_is_persisted_and_restored(self, tmp_path: Path) -> None:
        pdf_path = _pdf(tmp_path)
        handle = DocumentHandle.from_path(pdf_path)
        state = DocumentState(handle=handle)

        crop = tmp_path / "equation_0_page1.png"
        crop.write_bytes(b"fakepng")
        state.pages[1] = PageState(page_num=1, is_born_digital=True, native_text="native prose")
        state.events.append(_region_event(1, crop))

        orch = _make_pipeline()
        orch._attach_equation_latex_sidecars(state, [], page_nums=[1])
        assert state.pages[1].equation_sidecar_skipped is True, "setup"

        po = PageOutput(page_num=1, text="native prose", status=PageStatus.SUCCESS, engine="native")
        state.pages[1].best_output = po

        out_dir = tmp_path / "out"
        orch._scan_root = pdf_path.parent
        orch._flush_page_sidecar(state, 1, out_dir)

        sidecar = next(out_dir.rglob("pages/00001.json"))
        import json

        meta = json.loads(sidecar.read_text())
        assert meta.get("equation_sidecar_skipped") is True, (
            f"the flag never reached the sidecar, so the restore below is vacuous: {meta}"
        )

        resumed = DocumentState(handle=DocumentHandle(path=pdf_path))
        resumed.pages[1] = PageState(page_num=1, is_born_digital=True, native_text="native prose")
        resumed_po = PageOutput(
            page_num=1, text="native prose", status=PageStatus.SUCCESS, engine="native"
        )
        orch._restore_terminal_page_state(resumed, 1, resumed_po, out_dir)

        assert resumed.pages[1].equation_sidecar_skipped is True, (
            "the flag did not survive resume -- a resumed run would ship the "
            "orphan-crop page as a clean SUCCESS again"
        )

    def test_a_flag_set_this_run_is_not_cleared_by_an_older_sidecar(self, tmp_path: Path) -> None:
        """OR-restore, not a plain assignment: a sidecar written before this
        flag existed (or on a run that never hit the miss) must not erase a
        flag THIS run has already set."""
        pdf_path = _pdf(tmp_path)
        orch = _make_pipeline()

        state = DocumentState(handle=DocumentHandle.from_path(pdf_path))
        state.pages[1] = PageState(page_num=1, is_born_digital=True, native_text="native prose")
        state.pages[1].equation_sidecar_skipped = True  # set THIS run

        out_dir = tmp_path / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        pages_dir = out_dir / "pages"
        pages_dir.mkdir(parents=True, exist_ok=True)
        import json

        # An older sidecar with no ``equation_sidecar_skipped`` key at all.
        (pages_dir / "00001.json").write_text(json.dumps({"status": "success"}))

        po = PageOutput(page_num=1, text="native prose", status=PageStatus.SUCCESS, engine="native")
        orch._restore_terminal_page_state(state, 1, po, out_dir)

        assert state.pages[1].equation_sidecar_skipped is True, (
            "an older sidecar with no key for this flag cleared a flag this run "
            "had already set -- must be OR-restored, never a plain assignment"
        )
