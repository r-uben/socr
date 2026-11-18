"""#1355 review: the corroboration record survives the real pipeline, not only ``route_page``.

Two runs of ``process()`` in one process differ in ONE thing: whether the earlier (free) reading
carries the same ``-4.1`` the paid reading emits. The witness (an injected 150 dpi OCR stand-in
that lost the sign and point) is the same in both. Claims are differences between the arms and
the resume of one arm, never an absolute outcome measured on this machine.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_gh1185_text_layer_numbers_e2e import _doc_dir, _page

from socr.core.config import EngineType, PipelineConfig
from socr.core.marks import Check
from socr.core.providers import PROFILE_GEMINI, PROFILE_QWEN_LOCAL
from socr.core.result import PageOutput, PageStatus
from socr.pipeline import orchestrator as orch
from socr.pipeline.agentic import AcceptDecision, SourceEvidenceTableJudge
from socr.pipeline.orchestrator import UnifiedPipeline
from socr.tables.source_evidence import CORROBORATED_KIND

fitz = pytest.importorskip("fitz")

TABLE = "| item | a | b |\n|---|---|---|\n| Corn | {cell} | 2.0 |\n| Soy | 1.5 | 12 |\n"
PROSE = "Quarterly commodity price changes by crop, percent, as printed on the page."


class _Inner:
    def assess(self, output, provider):
        return AcceptDecision(accept=True, reason="faithful")


class _Reader:
    """qwen reads ``earlier_cell``; every other engine reads the true ``-4.1``."""

    def __init__(self, name: str, earlier_cell: str) -> None:
        self.name, self.earlier_cell = name, earlier_cell

    def is_available(self) -> bool:
        return True

    def process_pages(self, pdf_path, page_nums, config, dpi, subprocess_timeout=None, **_kw):
        cell = self.earlier_cell if self.name == "qwen" else "-4.1"
        return [
            PageOutput(
                page_num=n,
                text=f"{PROSE}\n\n{TABLE.format(cell=cell)}",
                status=PageStatus.SUCCESS,
                engine=self.name,
            )
            for n in page_nums
        ]


def _pdf(path: Path) -> Path:
    """A scan: one page-covering raster and no text layer, so the page needs a model reading."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 306, 396), False)
    pix.set_rect(pix.irect, (235, 235, 235))
    page.insert_image(page.rect, pixmap=pix)
    doc.save(path)
    doc.close()
    return path


def _run(out_dir: Path, pdf: Path, monkeypatch, earlier_cell: str, *, reprocess: bool = False):
    with monkeypatch.context() as m:
        m.setattr(orch, "get_engine", lambda t: _Reader(t.value, earlier_cell))
        pipe = UnifiedPipeline(
            PipelineConfig(
                agentic=True,
                quiet=True,
                primary_engine=EngineType.QWEN,
                local_engine=EngineType.QWEN,
                enabled_engines=[EngineType.QWEN, EngineType.GEMINI],
                native_first=True,
                write_manifest=True,
                reprocess=reprocess,
                judge_backend="heuristic",
                dual_pass_tables=False,
                detect_equations=False,
                describe_figure_crops=False,
            )
        )
        pipe._available_engines_for_agentic = lambda: [PROFILE_QWEN_LOCAL, PROFILE_GEMINI]
        pipe._resolve_crop_vlm_model = lambda: None
        pipe._resolve_judge_model = lambda *a, **k: ""

        def build_judge(state):
            fdoc = fitz.open(str(pdf))
            return SourceEvidenceTableJudge(
                _Inner(),
                lambda n: fdoc[n - 1],
                lambda ev: state.events.append(ev),
                ocr_image_fn=lambda pix: "Corn 41 2.0 Soy 1.5 12",
                native_trusted=lambda n: False,
                layer_is_evidence=lambda n: False,
            )

        pipe._build_page_judge = build_judge
        return pipe.process(pdf, output_dir=out_dir)


def _corroborated_events(out: Path) -> list[dict]:
    log = json.loads((_doc_dir(out) / "audit_log.json").read_text())
    return [e for e in log["events"] if e["kind"] == CORROBORATED_KIND]


def test_the_record_reaches_the_sidecar_and_the_audit_log_and_survives_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf = _pdf(tmp_path / "doc.pdf")
    carries, lacks = tmp_path / "carries", tmp_path / "lacks"
    _run(carries, pdf, monkeypatch, "-4.1")
    _run(lacks, pdf, monkeypatch, "-4.2")

    side, body = _page(carries, 1)
    lack_side, _ = _page(lacks, 1)
    record = {"tokens": ["-4.1"], "corroborated_by": "qwen", "engine": "gemini"}

    # the only difference between the arms is whether the earlier reading carried the token
    assert side["winning_output"]["table_witness_corroboration"] == record
    assert "table_witness_corroboration" not in lack_side["winning_output"]
    assert [e["data"] for e in _corroborated_events(carries)] == [record]
    assert _corroborated_events(lacks) == []

    # the corroborated paid reading still ships marked with the reading_dispute clause
    assert side["winning_output"]["engine"] == "gemini"
    assert side["status"] == "warning"
    [clause] = [c for c in side["verdict"]["clauses"] if c["check"] == Check.READING_DISPUTE.value]
    assert clause["result"] == "UNKNOWN"
    assert "reading_dispute=UNKNOWN" in body

    # resume: the terminal page is kept, so the field and the event are still there
    _run(carries, pdf, monkeypatch, "-4.1")
    again, _ = _page(carries, 1)
    assert again["winning_output"]["table_witness_corroboration"] == record
    assert [e["data"] for e in _corroborated_events(carries)] == [record]
