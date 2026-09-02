"""GH-171: the authoritative sidecar must be written AFTER figure extraction.

`_flush_page_sidecar` built `figure_refs` from `state.engine_runs` only, and
`_describe_and_embed_figures` attaches its results to the returned
`EngineResult` instead -- and runs AFTER the assemble-time flush. So the file
the pipeline calls authoritative shipped with an empty `figure_refs` on every
page that had figures, while the markdown and the manifest both carried them:
three records of one page disagreeing.

This is the same shape as the defect the MAJOR 6(a) note in `_phase_assemble`
already documents for audit events -- a sidecar is a snapshot, and nothing
corrects it afterwards -- so the fix is the same one: flush again, later.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from socr.core.result import FigureInfo

FIG = FigureInfo(
    figure_num=1,
    page_num=1,
    figure_type="chart",
    description="a chart",
    image_path="figures/fig_p1_1.png",
    engine="qwen",
    bbox=(10.0, 20.0, 300.0, 400.0),
)


def _pipeline():
    from socr.core.config import EngineType, PipelineConfig
    from socr.pipeline.orchestrator import UnifiedPipeline

    return UnifiedPipeline(
        PipelineConfig(
            primary_engine=EngineType.QWEN,
            enabled_engines=[EngineType.QWEN],
            quiet=True,
        )
    )


def _state(tmp_path: Path):
    fitz = pytest.importorskip("fitz")
    from socr.core.document import DocumentHandle
    from socr.core.result import PageOutput, PageStatus
    from socr.core.state import DocumentState

    tmp_path.mkdir(parents=True, exist_ok=True)
    pdf = tmp_path / "d.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "born digital text long enough to be a text layer.")
    doc.save(str(pdf))
    doc.close()

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
    return state


def _refs(sidecar: Path) -> list[dict]:
    return json.loads(sidecar.read_text())["figure_refs"]


def test_a_figure_found_only_after_the_flush_still_reaches_the_sidecar(tmp_path: Path) -> None:
    """The defect: figures attached post-flush were invisible to the sidecar."""
    state = _state(tmp_path / "a")
    out_dir = tmp_path / "a" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline = _pipeline()
    # The pre-figure flush, exactly as `_phase_assemble` does it first.
    before = pipeline._flush_page_sidecar(state, 1, out_dir)
    assert _refs(before) == [], "fixture must start with no figures, or it pins nothing"

    # The figure phase's results live on the RESULT, not on state.engine_runs.
    after = pipeline._flush_page_sidecar(state, 1, out_dir, extra_figures=[FIG])

    refs = _refs(after)
    assert len(refs) == 1, f"the figure never reached the sidecar: {refs}"
    assert refs[0]["image_path"] == FIG.image_path
    assert refs[0]["bbox"] == list(FIG.bbox)
    assert refs[0]["figure_type"] == "chart"
    assert refs[0]["engine"] == "qwen"


def test_a_figure_is_not_recorded_twice(tmp_path: Path) -> None:
    """Control: the same figure can reach both sources on a re-flush.

    `state.engine_runs` may already carry it, so a naive concatenation would
    duplicate every figure in the authoritative record.
    """
    from socr.core.result import DocumentStatus, EngineResult

    state = _state(tmp_path / "b")
    out_dir = tmp_path / "b" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    state.engine_runs.append(
        EngineResult(
            document_path=state.handle.path,
            engine="qwen",
            status=DocumentStatus.SUCCESS,
            figures=[FIG],
        )
    )

    sidecar = _pipeline()._flush_page_sidecar(state, 1, out_dir, extra_figures=[FIG])
    assert len(_refs(sidecar)) == 1, f"the figure was recorded twice: {_refs(sidecar)}"


def test_a_figure_on_another_page_is_not_claimed(tmp_path: Path) -> None:
    """Control: page filtering must survive the second source."""
    other = FigureInfo(figure_num=1, page_num=2, figure_type="chart", image_path="x.png")
    state = _state(tmp_path / "c")
    out_dir = tmp_path / "c" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    sidecar = _pipeline()._flush_page_sidecar(state, 1, out_dir, extra_figures=[other])
    assert _refs(sidecar) == [], f"page 1's sidecar claimed page 2's figure: {_refs(sidecar)}"


def test_the_reflush_happens_after_the_figure_phase() -> None:
    """The production ordering, not just the helper's new parameter.

    Pinning `extra_figures` alone would stay green if nothing ever passed it --
    which is precisely the state this ticket describes.
    """
    import ast

    src = Path(__file__).resolve().parents[1] / "src" / "socr" / "pipeline" / "orchestrator.py"
    tree = ast.parse(src.read_text())

    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_flush_page_sidecar"
        and any(kw.arg == "extra_figures" for kw in node.keywords)
    ]
    assert calls, (
        "no caller passes extra_figures, so the figure phase's results still "
        "never reach the sidecar"
    )


def test_every_terminal_flush_carries_the_figures() -> None:
    """GH-485: fixing ONE call site was not enough -- the last writer wins.

    #484 added `extra_figures` to the mid-assemble re-flush, but a LATER
    authoritative flush (after `_rewrite_all_fragments`) still called the helper
    without it. That write scans `state.engine_runs` only, which the figure
    phase never populates, so it silently undid the fix and shipped empty
    `figure_refs` again on the happy path.

    So the pin is over ALL terminal call sites rather than one: any flush that
    is not explicitly provisional must pass the figures.
    """
    import ast

    src = Path(__file__).resolve().parents[1] / "src" / "socr" / "pipeline" / "orchestrator.py"
    tree = ast.parse(src.read_text())

    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_flush_page_sidecar"
    ]
    assert len(calls) >= 3, f"expected several flush sites, found {len(calls)}"

    offenders: list[str] = []
    for call in calls:
        kwargs = {kw.arg: kw.value for kw in call.keywords}
        # A provisional flush is the mid-run crash-recovery copy; it predates
        # the figure phase by design and is not authoritative.
        terminal = kwargs.get("terminal")
        is_provisional = isinstance(terminal, ast.Constant) and terminal.value is False
        if is_provisional:
            continue
        if "extra_figures" not in kwargs:
            offenders.append(ast.unparse(call))

    assert not offenders, (
        "these terminal sidecar writes do not carry the figure phase's results, "
        "so whichever runs last will wipe them:\n  " + "\n  ".join(offenders)
    )
