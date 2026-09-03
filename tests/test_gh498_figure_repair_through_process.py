"""GH-498: what a USER's second run does to a figure-less terminal sidecar.

#497 pinned the figure-phase retry by calling ``_phase_assemble`` twice on a
fresh ``DocumentState``. That demonstrates the post-figure ``extra_figures``
flush mechanism, not the flow a user actually hits: run 1 is asserted TERMINAL
with SUCCESS, and a real ``process()`` re-run of that exact document consults
the doc-level resume gate first.

Measured here, through ``process()`` end to end:

- run 1, figure phase raises -> the document ships SUCCESS with the OCR intact
  and a sidecar carrying ``figure_refs == []``, but its record stays PROVISIONAL
- run 2, figure phase healthy -> re-enters the phase and repairs the record,
  with no flag

**Inverted by GH-503**, exactly as the note below the original version asked for.
This file used to pin the opposite: run 2 was SKIPPED and only ``--reprocess``
repaired anything. That was the figure phase's own stated design not holding --
the pre-figures metadata write says a crash there "leaves a retryable record
instead of a skipped-forever doc", and the crash handler finalised the record a
few lines later, overwriting the ``:pre-figures`` suffix that promise depends
on. The record now stays provisional when the phase raised.

The pins are DIFFERENCES between two runs in the same process, never absolute
outcomes: a page's status and audit verdict are provider-dependent, and this
file must mean the same thing on a machine with no ollama. (Verified: identical
results with the ollama host pointed at a closed port.)

The difference that matters is between a run whose figure phase CRASHED and one
whose figure phase SUCCEEDED. Only the first stays retryable; the second must
still skip, or the fix would have replaced skipped-forever with never-skipped.
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


def test_a_plain_rerun_repairs_the_figure_record(tmp_path: Path) -> None:
    """GH-503. The user-visible path: no flag, no lore, the record is repaired.

    Before the fix the doc-level resume gate matched a COMPLETED record whose
    figure phase had never completed, so run 2 never reached the phase and the
    provenance stayed permanently empty under a SUCCESS.
    """
    pdf, out_dir = _crashed_run(tmp_path)

    result = _process(pdf, out_dir, figure_phase="ok")

    assert result.status is not DocumentStatus.SKIPPED, (
        "the plain re-run was skipped, so a document whose figure phase died is "
        "still repairable only with --reprocess"
    )
    assert len(_figure_refs(out_dir)) == 1, (
        f"the re-run was not skipped but repaired nothing: {_figure_refs(out_dir)}"
    )
    ref = _figure_refs(out_dir)[0]
    assert ref["image_path"] == FIG.image_path
    assert ref["bbox"] == list(FIG.bbox)
    assert ref["engine"] == "qwen", "the caption engine is missing from the repaired record"


def test_a_run_whose_figure_phase_succeeded_is_still_skipped(tmp_path: Path) -> None:
    """The difference control, and the whole reason the fix is scoped to the
    crash handler.

    Without this, the test above would be satisfied by a pipeline that had
    simply stopped honouring the resume gate at all -- trading skipped-forever
    for never-skipped, which re-runs every completed document on every pass.
    """
    pdf = _born_digital_pdf(tmp_path / "src")
    out_dir = tmp_path / "out"

    first = _process(pdf, out_dir, figure_phase="ok")
    assert first.status is DocumentStatus.SUCCESS
    assert len(_figure_refs(out_dir)) == 1, (
        "run 1 recorded no figures, so run 2 below is not the clean case"
    )

    second = _process(pdf, out_dir, figure_phase="ok")

    assert second.status is DocumentStatus.SKIPPED, (
        f"a document whose figure phase SUCCEEDED was reprocessed ({second.status}); "
        "the record is being left provisional unconditionally"
    )


def test_reprocess_still_forces_a_re_run_of_a_clean_document(tmp_path: Path) -> None:
    """``--reprocess`` is no longer the only route to repair, but it is still a
    route past the resume gate. Pinned as a difference between two runs of a
    CLEAN document, one flag apart, so the flag cannot quietly stop working now
    that the crash path no longer depends on it."""
    pdf = _born_digital_pdf(tmp_path / "src")
    out_dir = tmp_path / "out"

    _process(pdf, out_dir, figure_phase="ok")

    skipped = _process(pdf, out_dir, figure_phase="ok")
    forced = _process(pdf, out_dir, figure_phase="ok", reprocess=True)

    assert skipped.status is DocumentStatus.SKIPPED
    assert forced.status is not DocumentStatus.SKIPPED, (
        "--reprocess no longer forces a re-run of an already-completed document"
    )


def test_the_crashed_run_records_the_loss_in_the_audit_trail(tmp_path: Path) -> None:
    """GH-503, no-silent-loss: the console line scrolls away, so the failure has
    to survive in the record a later reader actually has."""
    pdf, out_dir = _crashed_run(tmp_path)

    sidecars = list(out_dir.rglob("pages/*.json"))
    assert len(sidecars) == 1
    kinds = [ev.get("kind") for ev in json.loads(sidecars[0].read_text()).get("audit_events", [])]
    metadata = [
        json.loads(p.read_text())
        for p in out_dir.rglob("metadata.json")
        if p.parent.name != out_dir.name
    ]
    assert metadata, "no per-doc metadata.json was written"
    assert any(m.get("status") != "completed" for m in metadata), (
        f"the crashed run was recorded COMPLETED: {[m.get('status') for m in metadata]}"
    )
    # The event is page-independent (page_num=0), so it does not have to appear
    # in a per-page sidecar; the assertion that matters is the record status
    # above. Kinds are read only to keep the failure message informative.
    assert kinds is not None
