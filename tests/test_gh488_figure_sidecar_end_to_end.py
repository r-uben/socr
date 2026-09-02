"""GH-488: the figure metadata contract, through a real assemble run.

GH-171 / GH-485 are pinned at the CALL SITES -- an AST check that every terminal
`_flush_page_sidecar` names a real figure source. That catches the shapes those
tickets took, but it never runs the pipeline, so it cannot show that a
completed run actually lands complete metadata in `pages/NNN.json`.

This drives `_phase_assemble` with the figure phase stubbed to attach a figure
exactly as the real one does (`result.figures = [...]`), and asserts the
sidecar. Verified to catch the GH-485 regression: with the final flush passing
`extra_figures=[]`, `figure_refs` comes back `[]`.

One honest limit. Dropping the MID-assemble re-flush's figures (GH-171's own
fix) does NOT redden this file, because on this path the later flush carries
them anyway. The two are not redundant in general -- the final flush is
conditional on `final_page_outputs is not None`, so the mid one is the only
carrier when that is None -- but this fixture does not reach that branch, and
the AST pin in `test_gh171_sidecar_carries_figures.py` is what covers it.
Recorded rather than claimed.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

fitz = pytest.importorskip("fitz")

from socr.core.config import EngineType, PipelineConfig  # noqa: E402
from socr.core.document import DocumentHandle  # noqa: E402
from socr.core.result import FigureInfo, PageOutput, PageStatus  # noqa: E402
from socr.core.state import DocumentState  # noqa: E402
from socr.pipeline.orchestrator import UnifiedPipeline  # noqa: E402

FIG = FigureInfo(
    figure_num=1,
    page_num=1,
    figure_type="chart",
    description="a described chart",
    image_path="figures/fig_p1_1.png",
    engine="qwen",
    bbox=(10.0, 20.0, 300.0, 400.0),
)


def _run(tmp_path: Path, *, figures: list[FigureInfo]) -> list[dict]:
    """Assemble one page, with the figure phase attaching *figures*."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    pdf = tmp_path / "d.pdf"
    doc = fitz.open()
    doc.new_page().insert_text(
        (72, 72), "born digital text long enough to count as a real text layer."
    )
    doc.save(str(pdf))
    doc.close()

    pipeline = UnifiedPipeline(
        PipelineConfig(
            primary_engine=EngineType.QWEN,
            enabled_engines=[EngineType.QWEN],
            quiet=True,
            save_figures=True,
            write_manifest=False,
        )
    )
    state = DocumentState(handle=DocumentHandle.from_path(pdf))
    out = PageOutput(
        page_num=1,
        text="page body",
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=True,
    )
    state.pages[1].attempts.append(out)
    state.pages[1].best_output = out

    def _embed(self, st, result, out_dir, text):
        # Exactly what the real phase does: attach to the RESULT, not to
        # state.engine_runs. That asymmetry is what GH-171 was about.
        result.figures = list(figures)
        return text

    out_dir = tmp_path / "out"
    with patch.object(UnifiedPipeline, "_describe_and_embed_figures", _embed):
        pipeline._phase_assemble(state, out_dir)

    sidecars = list(out_dir.rglob("pages/*.json"))
    assert len(sidecars) == 1, f"expected one sidecar, got {sidecars}"
    return json.loads(sidecars[0].read_text())["figure_refs"]


def test_a_completed_run_lands_complete_figure_metadata(tmp_path: Path) -> None:
    """Acceptance: final path, bbox, type and caption engine all present."""
    refs = _run(tmp_path / "happy", figures=[FIG])

    assert len(refs) == 1, f"the figure never reached the sidecar: {refs}"
    ref = refs[0]
    assert ref["image_path"] == FIG.image_path
    assert ref["bbox"] == list(FIG.bbox)
    assert ref["figure_type"] == "chart"
    assert ref["engine"] == "qwen", "the caption engine is missing from the record"
    assert ref["description"] == FIG.description


def test_a_run_with_no_figures_records_none(tmp_path: Path) -> None:
    """Control: an empty list must not be manufactured into a phantom entry.

    Without this, a sidecar writer that invented a placeholder ref would satisfy
    the test above.
    """
    assert _run(tmp_path / "none", figures=[]) == []


def test_a_second_assemble_does_not_duplicate_the_refs(tmp_path: Path) -> None:
    """Figure-phase retry / resume: re-running must not double the record.

    The sidecar is rewritten by several flushes in one assemble, and a retry
    runs them all again. De-duplication is what keeps the authoritative record
    from growing on every attempt.
    """
    refs = _run(tmp_path / "retry", figures=[FIG, FIG])
    assert len(refs) == 1, f"the same figure was recorded more than once: {refs}"
