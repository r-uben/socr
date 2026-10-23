"""Model preamble loops with no page content (#JCT audit)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("rich")

from socr.audit.boilerplate import (
    is_boilerplate_only_output,
    strip_leading_ocr_boilerplate,
    substantive_ocr_text,
)
from socr.audit.heuristics import HeuristicsChecker
from socr.core.config import EngineType, PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.normalizer import OutputNormalizer
from socr.core.providers import provider_ladder
from socr.core.result import PageOutput, PageStatus
from socr.core.state import DocumentState
from socr.pipeline.agentic import HeuristicPageJudge, route_page
from socr.pipeline.orchestrator import UnifiedPipeline
from socr.pipeline.refinalize_boilerplate import classify_winning_text, refinalize_document

# Real JCT corpus sidecars (JCS-10-59 p2, JCX-16-81 p14), pinned verbatim.
JCS_P2_TEXT = (
    "The following is a transcription of the text in the image, rendered in clean Markdown format:\n\n"
    "---\n\n"
    "The following is a transcription of the text in the image, rendered in clean Markdown format:\n\n"
    "---\n\n"
    "The following is a transcription of the text in the image, rendered in clean Markdown format:\n\n"
    "---"
)
JCX_P14_TEXT = (
    "The following is a transcription of the text in the image into clean Markdown format, "
    "preserving the document structure:\n\n"
    "---\n\n"
    "The following is a transcription of the text in the image into clean Markdown format, "
    "preserving the document structure:\n\n"
    "---\n\n"
    "The following is a transcription of the text in the image into clean Markdown format, "
    "preserving the document structure:\n\n"
    "---"
)

_PREAMBLE_LINE = (
    "The following is a transcription of the text in the image, rendered in clean Markdown format:"
)

HEARING_TRANSCRIPTION = (
    "The following is a transcription of the hearing held before the "
    "Committee on Ways and Means on March 4, 2020."
)
WITNESS_REMARKS = "Here are transcribed remarks of the witness regarding tax policy."
_LEGISLATIVE_BODY = (
    "\n\nActual legislative text continues on this page with substantive detail "
    "about revenue effects and scoring."
)


def test_jcs_fixture_is_boilerplate_only() -> None:
    assert is_boilerplate_only_output(JCS_P2_TEXT)
    assert not HeuristicsChecker().check(JCS_P2_TEXT).passed


def test_jcx_fixture_is_boilerplate_only() -> None:
    assert is_boilerplate_only_output(JCX_P14_TEXT)
    assert not HeuristicsChecker().check(JCX_P14_TEXT).passed


def test_strip_preamble_preserves_body() -> None:
    body = "Section 1\n\nOutlays total 100 billion in fiscal year 2026."
    raw = f"{_PREAMBLE_LINE}\n\n---\n\n{body}"
    assert strip_leading_ocr_boilerplate(raw) == body
    assert substantive_ocr_text(raw) == body
    assert not is_boilerplate_only_output(raw)
    assert HeuristicsChecker(min_word_count=5).check(strip_leading_ocr_boilerplate(raw)).passed


def test_legitimate_repeated_header_not_flagged() -> None:
    text = (
        "Budget Committee\n\n"
        "Budget Committee\n\n"
        "Outlays in fiscal year 2026 total one trillion dollars."
    )
    assert not is_boilerplate_only_output(text)
    assert HeuristicsChecker(min_word_count=5).check(text).passed


def test_heuristic_judge_strips_leading_preamble_on_accept() -> None:
    body = " ".join(["word"] * 40)
    raw = f"{_PREAMBLE_LINE}\n\n{body}"
    judge = HeuristicPageJudge(HeuristicsChecker(min_word_count=10))
    out = PageOutput(page_num=1, text=raw, status=PageStatus.SUCCESS)
    decision = judge.assess(out, provider=type("P", (), {"engine": EngineType.GLM})())
    assert decision.accept
    assert out.text == body


@pytest.mark.parametrize(
    "engine",
    ["native_text", "qwen"],
)
def test_hearing_transcription_line_not_stripped_by_normalizer(engine: str) -> None:
    raw = HEARING_TRANSCRIPTION + _LEGISLATIVE_BODY
    normalized = OutputNormalizer().normalize(raw, engine=engine)
    assert normalized.startswith(HEARING_TRANSCRIPTION)
    assert HEARING_TRANSCRIPTION in normalized


def test_witness_remarks_line_not_stripped_for_model_or_native_engine() -> None:
    raw = WITNESS_REMARKS + _LEGISLATIVE_BODY
    for engine in ("qwen", "native_text"):
        normalized = OutputNormalizer().normalize(raw, engine=engine)
        assert normalized.startswith(WITNESS_REMARKS)


def test_refinalizer_leaves_hearing_transcription_sidecar_untouched() -> None:
    text = HEARING_TRANSCRIPTION + _LEGISLATIVE_BODY
    assert classify_winning_text(text) is None


def test_heuristic_judge_does_not_mutate_text_when_rejecting() -> None:
    raw = f"{_PREAMBLE_LINE}\n\nshort"
    judge = HeuristicPageJudge(HeuristicsChecker(min_word_count=50))
    out = PageOutput(page_num=1, text=raw, status=PageStatus.SUCCESS)
    before = out.text
    decision = judge.assess(out, provider=type("P", (), {"engine": EngineType.GLM})())
    assert not decision.accept
    assert out.text == before


def _make_pipeline() -> UnifiedPipeline:
    return UnifiedPipeline(
        PipelineConfig(
            agentic=True,
            judge_backend="heuristic",
            enabled_engines=[EngineType.QWEN],
            primary_engine=EngineType.QWEN,
            local_engine=EngineType.QWEN,
            save_figures=False,
            dual_pass_tables=False,
            detect_equations=False,
            recover_clean_equations=False,
            quiet=True,
            write_manifest=False,
        )
    )


def _one_page_pdf(tmp_path: Path) -> Path:
    """Raster-only page (no text layer) so agentic routing reaches the model."""
    fitz = pytest.importorskip("fitz")
    path = tmp_path / "doc.pdf"
    doc = fitz.open()
    page = doc.new_page()
    # Non-white pixel so the page is not treated as blank (#1100).
    page.draw_rect(fitz.Rect(72, 72, 200, 200), color=(0, 0, 0), fill=(0, 0, 0))
    doc.save(str(path))
    doc.close()
    return path


def _agentic_run(
    out_dir: Path,
    pdf_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    pages_read: list[int],
    page_text: str,
) -> None:
    from socr.core.providers import PROFILE_QWEN_LOCAL
    from socr.pipeline import orchestrator as orch
    from test_gh1156_part_a_interim import _FixedJudge

    class _StubEngine:
        name = "qwen"

        def is_available(self) -> bool:
            return True

        def process_pages(self, pdf_path, page_nums, config, dpi, subprocess_timeout=None, **_kw):
            pages_read.extend(page_nums)
            return [
                PageOutput(
                    page_num=n,
                    text=page_text,
                    status=PageStatus.SUCCESS,
                    engine="qwen",
                )
                for n in page_nums
            ]

    with monkeypatch.context() as m:
        m.setattr(orch, "get_engine", lambda engine_type: _StubEngine())
        pipe = _make_pipeline()
        pipe._scan_root = pdf_path.parent
        pipe.config.native_first = False
        pipe._available_engines_for_agentic = lambda: [PROFILE_QWEN_LOCAL]
        pipe._build_page_judge = lambda state: _FixedJudge(True)
        pipe._resolve_crop_vlm_model = lambda: None
        pipe._resolve_judge_model = lambda *a, **k: ""
        pipe.process(pdf_path, output_dir=out_dir)


def test_refinalize_apply_then_resume_rereads_boilerplate_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--apply`` flips ``winning_output.audit_passed``; resume re-OCRs via that gate."""
    pdf_path = _one_page_pdf(tmp_path)
    out_dir = tmp_path / "out"
    first_read: list[int] = []
    _agentic_run(out_dir, pdf_path, monkeypatch, pages_read=first_read, page_text=JCS_P2_TEXT)
    assert first_read == [1]

    doc_dir = next(out_dir.rglob("pages")).parent
    sidecar_path = doc_dir / "pages" / "00001.json"
    md_path = next(doc_dir.glob("*.md"))

    # Simulate the JCT corpus: terminal SUCCESS with audit_passed=true on boilerplate bytes.
    meta = json.loads(sidecar_path.read_text(encoding="utf-8"))
    meta["winning_output"]["text"] = JCS_P2_TEXT
    meta["winning_output"]["status"] = PageStatus.SUCCESS.value
    meta["winning_output"]["audit_passed"] = True
    meta["judge_rejected"] = False
    sidecar_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (doc_dir / "pages" / "00001.md").write_text(JCS_P2_TEXT, encoding="utf-8")
    md_path.write_text(f"## Page 1\n\n{JCS_P2_TEXT}", encoding="utf-8")

    changes = refinalize_document(doc_dir, apply=True)
    assert len(changes) == 1 and changes[0].action == "re_ocr"

    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["terminal"] is True
    assert sidecar["winning_output"]["status"] == PageStatus.SUCCESS.value
    assert sidecar["winning_output"]["audit_passed"] is False
    assert sidecar["judge_rejected"] is True

    pipeline = _make_pipeline()
    pipeline._scan_root = pdf_path.parent
    fresh = DocumentState(handle=DocumentHandle.from_path(pdf_path))
    assert pipeline._load_terminal_page(fresh, 1, out_dir) is None

    replacement = "Re-read after refinalize: committee report body " * 6
    resume_read: list[int] = []
    _agentic_run(out_dir, pdf_path, monkeypatch, pages_read=resume_read, page_text=replacement)

    assert 1 in resume_read, "resume must re-invoke the model for the demoted page"
    final_md = md_path.read_text(encoding="utf-8")
    assert "text in the image, rendered in clean Markdown format" not in final_md
    assert replacement.strip() in final_md


def test_route_page_escalates_past_boilerplate_to_next_engine() -> None:
    ladder = provider_ladder({EngineType.GLM, EngineType.GEMINI}, include_ineligible=True)
    good = " ".join(["policy"] * 40)

    def run(profile, page_num: int) -> PageOutput:
        engine = profile.engine
        text = JCS_P2_TEXT if engine == EngineType.GLM else good
        return PageOutput(
            page_num=page_num,
            text=text,
            status=PageStatus.SUCCESS,
            engine=engine.value,
        )

    judge = HeuristicPageJudge(HeuristicsChecker(min_word_count=10))
    decision = route_page(1, ladder[:2], run, judge)
    assert decision.accepted
    assert decision.final_output.engine == EngineType.GEMINI.value
    assert decision.final_output.text == good
    assert len(decision.attempts) == 2
    assert not decision.attempts[0].accepted
    assert decision.attempts[1].accepted
