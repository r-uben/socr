"""Keep a model table when stored words cannot witness it (rotated-table-first).

On a rotated born-digital table page the distrusted text layer is excluded from
the evidence bundle and classical OCR on the derotated raster often reads
nothing. Row corroboration may fail on layout without finding a numeric
contradiction. The model table must ship flagged in that case; a witness that
does contradict the model's numbers must still hard-reject.

Hermetic: patch ``_available_engines_for_agentic``; no real VLM.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import fitz
import pytest

from socr.core.config import EngineType, PipelineConfig
from socr.core.providers import PROFILE_QWEN_LOCAL
from socr.core.result import PageOutput, PageStatus
from socr.pipeline.agentic import AcceptDecision, SourceEvidenceTableJudge
from socr.pipeline.orchestrator import UnifiedPipeline
from socr.tables.source_evidence import WITNESS_EMPTY_READING, verify_scanned_table

_FOMC_PDF = Path(
    "/cursor/stores/bc-f87a255a-cbe0-40f3-b811-f505faf8233d/media/"
    "fed-minutes-example/fomcminutes20190619.pdf"
)

_FOMC_MODEL_TABLE = (
    "| Variable | 2019 | 2020 | 2021 | Longer run |\n"
    "| --- | --- | --- | --- | --- |\n"
    "| Change in real GDP | 2.1 | 2.0 | 1.8 | 1.9 |\n"
    "| Unemployment rate | 3.6 | 3.7 | 3.8 | 4.2 |\n"
    "| PCE inflation | 1.5 | 1.9 | 2.0 | 2.0 |\n"
    "| Core PCE inflation | 1.8 | 1.9 | 2.0 | 2.0 |\n"
)


def _empty_pixel_witness(_pix) -> str:
    return ""


@pytest.mark.skipif(not _FOMC_PDF.is_file(), reason="FOMC fixture not in project store")
def test_fomc_rotated_page_keeps_model_table_when_stored_words_unusable() -> None:
    doc = fitz.open(_FOMC_PDF)
    try:
        page = doc[13]
        result = verify_scanned_table(
            page,
            _FOMC_MODEL_TABLE,
            ocr_image_fn=_empty_pixel_witness,
            native_trusted=False,
        )
        assert result.passed, result.reason
        assert result.content_unverified
        assert "stored_words_unverified" in result.content_unverified
    finally:
        doc.close()


def test_contradicting_stored_words_still_reject() -> None:
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    for i in range(3):
        page.insert_text((72, 100 + i * 20), f"Some Other Line {i} 999.9 888.8", fontsize=10)
    candidate = (
        "| Counterparty | Amount | Drawn |\n"
        "| --- | --- | --- |\n"
        "| Bundesbank | 62.5 | 12.5 |\n"
        "| Bank of Japan | 67.0 | 15.0 |\n"
    )
    try:
        result = verify_scanned_table(
            page,
            candidate,
            ocr_image_fn=_empty_pixel_witness,
            native_trusted=False,
        )
        assert not result.passed, result.reason
        assert not result.content_unverified
    finally:
        doc.close()


def test_judge_accepts_flagged_table_for_empty_reading_with_stored_words() -> None:
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 100), "scrambled layer 1.0 2.0", fontsize=10, rotate=90)
    table = "| row | a | b |\n| --- | --- | --- |\n| one | 1.0 | 2.0 |"
    events: list = []

    judge = SourceEvidenceTableJudge(
        inner=type(
            "Inner",
            (),
            {
                "assess": staticmethod(
                    lambda output, provider: AcceptDecision(accept=True, reason="inner ok")
                )
            },
        )(),
        get_fitz_page=lambda _pn: page,
        record_event=events.append,
        ocr_image_fn=_empty_pixel_witness,
        native_trusted=lambda _pn: False,
    )
    output = PageOutput(
        page_num=1,
        text=table,
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=True,
    )
    decision = judge.assess(output, PROFILE_QWEN_LOCAL)
    assert decision.accept is True
    assert output.table_label_unverified
    assert "stored_words_unverified" in output.table_label_unverified
    doc.close()


def _config() -> PipelineConfig:
    return PipelineConfig(
        agentic=True,
        native_first=True,
        native_only=False,
        primary_engine=EngineType.QWEN,
        enabled_engines=[EngineType.QWEN],
        tiered=False,
        dual_pass_tables=False,
        detect_equations=False,
        save_figures=False,
        quiet=True,
        table_judge_ladder=False,
    )


@pytest.mark.skipif(not _FOMC_PDF.is_file(), reason="FOMC fixture not in project store")
def test_agentic_fomc_page_14_keeps_model_table(tmp_path: Path) -> None:
    single = tmp_path / "fomc-p14.pdf"
    src = fitz.open(_FOMC_PDF)
    doc = fitz.open()
    doc.insert_pdf(src, from_page=13, to_page=13)
    doc.save(single)
    doc.close()
    src.close()

    pipeline = UnifiedPipeline(_config())

    def _route(page_num, ladder, run_provider, judge, **kwargs):
        from socr.pipeline.agentic import PageDecision, ProviderAttempt

        out = PageOutput(
            page_num=page_num,
            text=_FOMC_MODEL_TABLE,
            status=PageStatus.SUCCESS,
            engine="qwen",
            audit_passed=False,
        )
        prof = ladder[0]
        decision = judge.assess(out, prof)
        out.audit_passed = decision.accept
        att = ProviderAttempt(
            engine=prof.engine,
            output=out,
            cost_usd=0.0,
            accepted=decision.accept,
            reason=decision.reason,
            provider_id=prof.id,
            model=prof.model,
            backend=prof.backend,
        )
        return PageDecision(
            page_num=page_num,
            final_output=out,
            attempts=[att],
            accepted=decision.accept,
        )

    with (
        patch("socr.pipeline.orchestrator.route_page", side_effect=_route),
        patch.object(pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]),
        patch(
            "socr.tables.native_first.attempt_rotated_native_table",
            return_value=None,
        ),
        patch(
            "socr.tables.source_evidence.classical_ocr_with_state",
            lambda pix: ("", WITNESS_EMPTY_READING, ""),
        ),
    ):
        result = pipeline.process(single, tmp_path / "out")

    body = result.markdown or ""
    assert "2.1" in body and "3.6" in body, body[:500]
    assert "failed: unverifiable table" not in body, body
    sidecar = json.loads(
        next((tmp_path / "out").rglob("pages/00001.json")).read_text(encoding="utf-8")
    )
    assert sidecar.get("status") == "warning"
    kinds = [ev["kind"] for ev in sidecar.get("audit_events", [])]
    assert "source_evidence_table_label_unverified" in kinds
    detail = next(
        ev["detail"]
        for ev in sidecar["audit_events"]
        if ev["kind"] == "source_evidence_table_label_unverified"
    )
    assert "stored_words_unverified" in detail
