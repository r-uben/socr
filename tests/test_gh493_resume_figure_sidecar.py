"""GH-493: the figure record after a RETRY, and the three records agreeing.

#491 (via `test_gh488_figure_sidecar_end_to_end.py`) pinned the happy-path
flush: one `_phase_assemble`, figures attached, `figure_refs` complete. Two
things it left unmeasured, both named in #488's acceptance:

1. **A real figure-phase retry.** The figure phase is the one phase allowed to
   fail without failing the run -- `_phase_assemble` catches its exception and
   keeps the un-embedded markdown. So a document can reach a TERMINAL sidecar
   with no figure metadata and then be re-run. Nothing asserted that the second
   run repairs it; #491's third case was renamed away from the vacuous
   `[FIG, FIG]` single-assemble shape and never reached a second assemble.

2. **Page-local figure audit events, and Markdown / manifest agreement.** A
   figure that ships leaves a mark in three places -- the `pages/NNN.md`
   fragment, the manifest page blob that `replay` reproduces, and the sidecar's
   own frozen `winning_output`. Three records of one page are only worth having
   if they cannot disagree.

Measured at `_phase_assemble` and the real flush path, as the ticket requires,
never at an isolated helper.
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
FIG_REF = f"![Figure 1]({FIG.image_path})"


def _pdf(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    pdf = tmp_path / "d.pdf"
    doc = fitz.open()
    doc.new_page().insert_text(
        (72, 72), "born digital text long enough to count as a real text layer."
    )
    doc.save(str(pdf))
    doc.close()
    return pdf


def _assemble(
    pdf: Path,
    out_dir: Path,
    *,
    figure_phase: str,
    write_manifest: bool = False,
    cap_page: int | None = None,
) -> None:
    """One full `_phase_assemble` over a one-page document.

    ``figure_phase`` is ``"crash"`` (the phase raises, as a real caption-engine
    failure does), or ``"ok"`` (it attaches FIG and splices the ref into the
    body, which is what the real phase does).
    """
    pipeline = UnifiedPipeline(
        PipelineConfig(
            primary_engine=EngineType.QWEN,
            enabled_engines=[EngineType.QWEN],
            quiet=True,
            save_figures=True,
            write_manifest=write_manifest,
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

    def _embed(self, st, result, out_dir_, text):
        if figure_phase == "crash":
            raise RuntimeError("caption engine died")
        if cap_page is not None:
            from socr.core.audit_log import AuditEvent

            st.events.append(
                AuditEvent(
                    page_num=cap_page,
                    kind="figure_cap_reached",
                    engine="",
                    detail="Figure extraction cap reached",
                    data={"figures_max_total": 1},
                )
            )
        result.figures = [FIG]
        return text.replace("page body", f"page body\n\n{FIG_REF}")

    with patch.object(UnifiedPipeline, "_describe_and_embed_figures", _embed):
        pipeline._phase_assemble(state, out_dir)


def _sidecar(out_dir: Path) -> dict:
    paths = list(out_dir.rglob("pages/*.json"))
    assert len(paths) == 1, f"expected one sidecar, got {paths}"
    return json.loads(paths[0].read_text())


def test_a_figure_phase_retry_repairs_the_sidecar(tmp_path: Path) -> None:
    """Run 1's figure phase dies; run 2 must land COMPLETE figure metadata.

    The first run is not a hypothetical: `_phase_assemble` swallows a figure-
    phase exception on purpose, so the document ships terminal and figure-less.
    Everything a consumer needs -- path, bbox, type, caption engine, caption --
    has to be there after the retry, not merely a non-empty list.
    """
    pdf = _pdf(tmp_path / "src")
    out_dir = tmp_path / "out"

    _assemble(pdf, out_dir, figure_phase="crash")
    crashed = _sidecar(out_dir)
    assert crashed["figure_refs"] == [], (
        "the crashing run already recorded figures, so the retry below would prove nothing"
    )
    assert crashed["terminal"] is True, (
        "the crashed run must be TERMINAL -- that is why the retry matters"
    )

    _assemble(pdf, out_dir, figure_phase="ok")
    refs = _sidecar(out_dir)["figure_refs"]

    assert len(refs) == 1, f"the retry did not repair the figure record: {refs}"
    ref = refs[0]
    assert ref["image_path"] == FIG.image_path
    assert ref["bbox"] == list(FIG.bbox)
    assert ref["figure_type"] == "chart"
    assert ref["engine"] == "qwen", "the caption engine is missing from the record"
    assert ref["description"] == FIG.description


def test_a_page_local_figure_event_reaches_that_pages_sidecar(tmp_path: Path) -> None:
    """A figure-phase audit event must land on ITS page, and only there.

    `figure_cap_reached` is the phase's one durable "content may have been
    dropped" signal. The sidecar filters `state.events` by page, so an event
    raised for a page the pipeline is not flushing is silently invisible.
    """
    pdf = _pdf(tmp_path / "src")
    out_dir = tmp_path / "out"
    _assemble(pdf, out_dir, figure_phase="ok", cap_page=1)

    kinds = [ev["kind"] for ev in _sidecar(out_dir)["audit_events"]]
    assert "figure_cap_reached" in kinds, (
        f"the figure-phase cap event never reached the page sidecar: {kinds}"
    )


def test_an_event_for_another_page_does_not_leak_in(tmp_path: Path) -> None:
    """Difference control: without this, a sidecar that dumped EVERY event
    would satisfy the test above while telling the reader nothing about which
    page lost content."""
    pdf = _pdf(tmp_path / "src")
    out_dir = tmp_path / "out"
    _assemble(pdf, out_dir, figure_phase="ok", cap_page=7)

    kinds = [ev["kind"] for ev in _sidecar(out_dir)["audit_events"]]
    assert "figure_cap_reached" not in kinds, (
        f"page 1's sidecar claims a cap event raised for page 7: {kinds}"
    )


def test_fragment_manifest_and_sidecar_carry_the_same_page_body(tmp_path: Path) -> None:
    """The three records of one page must agree, figure ref included.

    - `pages/NNN.md` is what a downstream consumer reads.
    - the manifest page blob is what `replay` reproduces.
    - the sidecar's `winning_output.text` is what the resume gate restores.

    The figure phase rewrites the body AFTER the first two are written, so this
    is exactly where they can drift apart.
    """
    from socr.core.cache import BlobStore

    pdf = _pdf(tmp_path / "src")
    out_dir = tmp_path / "out"
    _assemble(pdf, out_dir, figure_phase="ok", write_manifest=True)

    sidecar_path = list(out_dir.rglob("pages/*.json"))[0]
    doc_dir = sidecar_path.parent.parent

    fragment = (sidecar_path.parent / "00001.md").read_text()
    sidecar_text = _sidecar(out_dir)["winning_output"]["text"]

    manifest = json.loads((doc_dir / "manifest.json").read_text())
    entry = manifest["entries"]["1"]
    blob_text = BlobStore(doc_dir / "cache").get(entry["blob_ref"])["text"]

    assert FIG_REF in fragment, f"the fragment lost the figure ref:\n{fragment}"
    assert FIG_REF in blob_text, (
        f"the manifest blob replay would reproduce has no figure ref:\n{blob_text}"
    )
    assert FIG_REF in sidecar_text, (
        f"the sidecar's frozen winner has no figure ref:\n{sidecar_text}"
    )
    assert fragment.strip() == blob_text.strip() == sidecar_text.strip(), (
        "the three records of page 1 disagree:\n"
        f"fragment={fragment!r}\nblob={blob_text!r}\nsidecar={sidecar_text!r}"
    )
