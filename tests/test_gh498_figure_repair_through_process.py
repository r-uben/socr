"""GH-498: what a USER's second run does to a figure-less terminal sidecar.

#497 pinned the figure-phase retry by calling ``_phase_assemble`` twice on a
fresh ``DocumentState``. That demonstrates the post-figure ``extra_figures``
flush mechanism, not the flow a user actually hits: run 1 is asserted TERMINAL
with SUCCESS, and a real ``process()`` re-run of that exact document consults
the doc-level resume gate first.

Measured here, through ``process()`` end to end:

- run 1, figure phase raises -> the document is recorded SUCCESS anyway, with a
  terminal sidecar carrying ``figure_refs == []``
- run 2, figure phase healthy -> **SKIPPED**. The repair never happens.
- run 3, same but ``--reprocess`` -> SUCCESS, complete ``figure_refs``.

So repair through the user-visible path requires ``--reprocess`` today. That is
pinned as a DIFFERENCE between runs 2 and 3 -- identical code and identical
inputs, one flag apart -- rather than as an absolute outcome, because the
absolute one is provider-dependent and this file must mean the same thing on a
machine with no ollama. (Verified: identical results with the ollama host
pointed at a closed port.)

Whether run 2 SHOULD skip is a product question, not a test one. The figure
phase's own comment says a crash there "leaves a retryable record instead of a
skipped-forever doc", and the crash handler then finalises the record anyway --
a contradiction filed separately. This file pins today's behaviour so that
change, when it comes, cannot happen silently.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

fitz = pytest.importorskip("fitz")

from socr.core.config import EngineType, PipelineConfig  # noqa: E402
from socr.core.result import DocumentStatus, FigureInfo  # noqa: E402
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


def _born_digital_pdf(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    pdf = tmp_path / "d.pdf"
    doc = fitz.open()
    doc.new_page().insert_text(
        (72, 72),
        "born digital text long enough to count as a real text layer, with several words.",
    )
    doc.save(str(pdf))
    doc.close()
    return pdf


def _process(pdf: Path, out_dir: Path, *, figure_phase: str, reprocess: bool = False):
    """One user-visible run. ``figure_phase`` is "crash" or "ok"."""
    pipeline = UnifiedPipeline(
        PipelineConfig(
            primary_engine=EngineType.QWEN,
            enabled_engines=[EngineType.QWEN],
            quiet=True,
            save_figures=True,
            write_manifest=False,
            reprocess=reprocess,
        )
    )

    def _embed(self, state, result, out, text):
        if figure_phase == "crash":
            raise RuntimeError("caption engine died")
        result.figures = [FIG]
        return text

    with patch.object(UnifiedPipeline, "_describe_and_embed_figures", _embed):
        return pipeline.process(pdf, out_dir)


def _figure_refs(out_dir: Path) -> list[dict]:
    sidecars = list(out_dir.rglob("pages/*.json"))
    assert len(sidecars) == 1, f"expected one sidecar, got {sidecars}"
    return json.loads(sidecars[0].read_text())["figure_refs"]


def _crashed_run(tmp_path: Path) -> tuple[Path, Path]:
    """Leave a document in the state this ticket is about: terminal, SUCCESS,
    no figure metadata, because the figure phase died."""
    pdf = _born_digital_pdf(tmp_path / "src")
    out_dir = tmp_path / "out"

    result = _process(pdf, out_dir, figure_phase="crash")

    assert result.status is DocumentStatus.SUCCESS, (
        "a figure-phase crash no longer ships the document as SUCCESS; the "
        "premise of this whole file has changed and it needs rewriting, not a "
        "threshold tweak"
    )
    assert _figure_refs(out_dir) == [], (
        "the crashed run already recorded figures, so nothing below measures repair"
    )
    return pdf, out_dir


def test_a_plain_rerun_does_not_repair_the_figure_record(tmp_path: Path) -> None:
    """Today's behaviour, pinned so it cannot change without someone noticing.

    The doc-level resume gate sees a completed record with a matching
    fingerprint and checksum, so the second run never reaches the figure phase.
    The user sees a successful document whose provenance is permanently empty.
    """
    pdf, out_dir = _crashed_run(tmp_path)

    result = _process(pdf, out_dir, figure_phase="ok")

    assert result.status is DocumentStatus.SKIPPED, (
        f"the plain re-run was not skipped ({result.status}); if repair is now "
        "reachable without --reprocess, this test should be inverted, not deleted"
    )
    assert _figure_refs(out_dir) == [], (
        "the figure record was repaired by a plain re-run -- good news, but the "
        "test below no longer measures a difference"
    )


def test_reprocess_reaches_the_figure_phase_and_repairs_it(tmp_path: Path) -> None:
    """The same document, the same code, one flag apart.

    Pinned as a difference rather than an absolute outcome: a page's status and
    audit verdict are provider-dependent, but "these two runs differ, and this
    is how" holds on a machine with no provider at all.
    """
    pdf, out_dir = _crashed_run(tmp_path)

    skipped = _process(pdf, out_dir, figure_phase="ok")
    assert skipped.status is DocumentStatus.SKIPPED
    without_reprocess = _figure_refs(out_dir)

    repaired = _process(pdf, out_dir, figure_phase="ok", reprocess=True)
    with_reprocess = _figure_refs(out_dir)

    assert repaired.status is not DocumentStatus.SKIPPED, (
        "--reprocess did not force a re-run, so the figure phase was never "
        "re-entered and the record cannot be repaired at all"
    )
    assert without_reprocess == [] and len(with_reprocess) == 1, (
        f"--reprocess made no difference to the figure record: "
        f"{without_reprocess} vs {with_reprocess}"
    )

    ref = with_reprocess[0]
    assert ref["image_path"] == FIG.image_path
    assert ref["bbox"] == list(FIG.bbox)
    assert ref["engine"] == "qwen", "the caption engine is missing from the repaired record"
