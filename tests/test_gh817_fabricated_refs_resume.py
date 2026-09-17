"""GH-817: the ``fabricated_image_refs`` (GH-225) demotion must survive resume.

GH-225's guard demotes the DOCUMENT to ``AUDIT_FAILED`` via the
``fabricated_ref_pages`` orthogonal bucket in ``_phase_assemble``, keyed off
``PageState.fabricated_image_refs``. That counter was never written to the
sidecar meta block and never restored in ``_restore_terminal_page_state``. A
resumed run that restores the page as terminal therefore comes back with the
counter at 0: the demotion silently disappears while the cleaned (redacted)
text -- the fabricated refs already stripped -- still ships as SUCCESS.

Every test here pins the OUTCOME (``DocumentStatus``), not the field's mere
presence in the sidecar -- a field round-trip test would pass while the
demotion stayed lost, which is exactly the bug (per the ticket's own framing).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")

from socr.core.config import PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.result import DocumentStatus, PageOutput, PageStatus
from socr.core.state import DocumentState, PageState
from socr.pipeline.orchestrator import UnifiedPipeline


def _pdf(tmp_path: Path, name: str = "doc.pdf") -> Path:
    path = tmp_path / name
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "Real prose that must survive. " * 6)
    doc.save(str(path))
    doc.close()
    return path


def _make_pipeline(**overrides) -> UnifiedPipeline:
    cfg = PipelineConfig(
        agentic=True, quiet=True, save_figures=False, write_manifest=False, **overrides
    )
    return UnifiedPipeline(cfg)


def _fabricate(
    orch: UnifiedPipeline, state: DocumentState, page_num: int, doc_dir: Path
) -> PageOutput:
    """Run the real GH-225 guard against a page whose text has an image ref
    with no provenance anywhere in the source PDF, so ``fabricated_image_refs``
    is set the way a real run sets it (not hand-assigned)."""
    text = (
        "Real prose that must survive.\n\n"
        "![invented figure](https://not-in-the-source.example/ghost.png)\n"
    )
    page_out = PageOutput(page_num=page_num, text=text, status=PageStatus.SUCCESS, engine="native")
    ps = state.pages.get(page_num) or PageState(page_num=page_num)
    ps.is_born_digital = True
    ps.native_text = text
    ps.attempts.append(page_out)
    ps.best_output = page_out
    state.pages[page_num] = ps

    orch._guard_fabricated_image_refs(state, page_num, page_out, doc_dir)
    return page_out


class TestFabricationDemotesTheDocument:
    def test_document_status_is_not_success(self, tmp_path: Path) -> None:
        pdf_path = _pdf(tmp_path)
        state = DocumentState(handle=DocumentHandle.from_path(pdf_path))
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        _fabricate(_make_pipeline(), state, 1, out_dir)
        assert state.pages[1].fabricated_image_refs == 1, "setup: the guard never fired"

        orch = _make_pipeline()
        orch._scan_root = tmp_path
        result = orch._phase_assemble(state, out_dir)

        assert result.status != DocumentStatus.SUCCESS, (
            "a page with a fabricated image ref still shipped a clean document: "
            f"status={result.status}, error={result.error!r}"
        )
        assert result.status == DocumentStatus.AUDIT_FAILED, result.status

    def test_clean_run_with_no_fabrication_still_succeeds(self, tmp_path: Path) -> None:
        """Reverse regression: a page with NO fabricated ref must resume clean."""
        pdf_path = _pdf(tmp_path)
        state = DocumentState(handle=DocumentHandle.from_path(pdf_path))
        text = "Real prose that must survive."
        page_out = PageOutput(page_num=1, text=text, status=PageStatus.SUCCESS, engine="native")
        ps = PageState(page_num=1, is_born_digital=True, native_text=text)
        ps.attempts.append(page_out)
        ps.best_output = page_out
        state.pages[1] = ps

        out_dir = tmp_path / "out"
        out_dir.mkdir()
        orch = _make_pipeline()
        orch._scan_root = tmp_path
        result = orch._phase_assemble(state, out_dir)

        assert result.status == DocumentStatus.SUCCESS, (
            f"a clean document with no fabricated ref was demoted: "
            f"status={result.status}, error={result.error!r}"
        )


class TestCounterSurvivesResume:
    """The reported bug, reproduced end to end: RUN1 demotes, a real
    flush/restore round trip must make RUN2 demote too."""

    def test_full_flush_restore_reassemble_cycle_still_demotes(self, tmp_path: Path) -> None:
        pdf_path = _pdf(tmp_path)
        handle = DocumentHandle.from_path(pdf_path)
        state = DocumentState(handle=handle)
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        orch = _make_pipeline()
        orch._scan_root = pdf_path.parent
        page_out = _fabricate(orch, state, 1, out_dir)
        assert state.pages[1].fabricated_image_refs == 1, "setup"

        # RUN1: assemble demotes, as the reverse-regression test above confirms
        # in isolation. Now flush the page as terminal (the shape a real run
        # writes before exiting) and confirm the counter reached the sidecar.
        orch._flush_page_sidecar(state, 1, out_dir)
        sidecar = next(out_dir.rglob("pages/00001.json"))
        meta = json.loads(sidecar.read_text())
        assert meta.get("fabricated_image_refs") == 1, (
            f"the counter never reached the sidecar, so the restore below is vacuous: {meta}"
        )

        # RUN2: a fresh DocumentState, as a resumed process constructs. Restore
        # the terminal page from disk exactly as the resume ledger does, then
        # re-run assemble on the restored state.
        resumed = DocumentState(handle=DocumentHandle(path=pdf_path))
        resumed.pages[1] = PageState(page_num=1, is_born_digital=True, native_text=page_out.text)
        resumed_po = PageOutput(
            page_num=1, text=page_out.text, status=page_out.status, engine="native"
        )
        orch._restore_terminal_page_state(resumed, 1, resumed_po, out_dir)
        resumed.pages[1].best_output = resumed_po
        resumed.pages[1].attempts.append(resumed_po)

        assert resumed.pages[1].fabricated_image_refs == 1, (
            "the counter did not survive resume -- RUN2 would report SUCCESS "
            "over a page that shipped a fabricated image reference"
        )

        out_dir2 = tmp_path / "out2"
        out_dir2.mkdir()
        result2 = orch._phase_assemble(resumed, out_dir2)
        assert result2.status == DocumentStatus.AUDIT_FAILED, (
            "RUN2 shipped SUCCESS after a resume that lost the fabrication demotion: "
            f"status={result2.status}, error={result2.error!r}"
        )

    @staticmethod
    def _sidecar_path(pdf_path: Path, out_dir: Path) -> Path:
        """The exact path ``_restore_terminal_page_state`` reads (doc_dir_for +
        relative_key against ``scan_root``), so writing anywhere else would let
        the restore silently miss the file and pass on a no-op read."""
        from ocr_output_contract import doc_dir_for, relative_key

        doc_dir = doc_dir_for(out_dir, relative_key(pdf_path, pdf_path.parent))
        return doc_dir / "pages" / "00001.json"

    def test_a_count_set_this_run_is_not_cleared_by_an_older_sidecar(self, tmp_path: Path) -> None:
        """The OR case: a value set THIS run must survive a sidecar written
        before the field existed (or on a run that never hit a fabrication)."""
        pdf_path = _pdf(tmp_path)
        orch = _make_pipeline()
        orch._scan_root = pdf_path.parent

        state = DocumentState(handle=DocumentHandle.from_path(pdf_path))
        state.pages[1] = PageState(page_num=1, is_born_digital=True, native_text="native prose")
        state.pages[1].fabricated_image_refs = 2  # set THIS run

        out_dir = tmp_path / "out"
        sidecar_path = self._sidecar_path(pdf_path, out_dir)
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        # An older sidecar with no ``fabricated_image_refs`` key at all.
        sidecar_path.write_text(json.dumps({"status": "success"}))

        po = PageOutput(page_num=1, text="native prose", status=PageStatus.SUCCESS, engine="native")
        orch._restore_terminal_page_state(state, 1, po, out_dir)

        assert state.pages[1].fabricated_image_refs == 2, (
            "an older sidecar with no key for this counter cleared a value this "
            "run had already set -- must be max-restored, never a plain assignment"
        )

    def test_max_restore_does_not_silently_shrink_a_lower_run_value(self, tmp_path: Path) -> None:
        """The counter-specific half of the judgement call: a sidecar with a
        HIGHER count than this run's in-memory value must win, not lose to a
        plain OR-as-bool that would floor the restore at 1."""
        pdf_path = _pdf(tmp_path)
        orch = _make_pipeline()
        orch._scan_root = pdf_path.parent

        state = DocumentState(handle=DocumentHandle.from_path(pdf_path))
        state.pages[1] = PageState(page_num=1, is_born_digital=True, native_text="native prose")
        state.pages[1].fabricated_image_refs = 1  # this run only re-detected one

        out_dir = tmp_path / "out"
        sidecar_path = self._sidecar_path(pdf_path, out_dir)
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar_path.write_text(json.dumps({"fabricated_image_refs": 3}))

        po = PageOutput(page_num=1, text="native prose", status=PageStatus.SUCCESS, engine="native")
        orch._restore_terminal_page_state(state, 1, po, out_dir)

        assert state.pages[1].fabricated_image_refs == 3, (
            "max-restore must keep the larger of the two counts, not the in-memory value alone"
        )
