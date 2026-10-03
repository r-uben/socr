"""#1027: a scan with an invisible OCR layer and no accepted model reading ships the floor.

The invisible layer is known garbage (one letter per line, stray axis ticks). When the model
ladder ran and accepted nothing, neither it nor the rejected reading may ship: the page ships
the fail-closed marker (``invisible_scan_unread``), ``audit_passed`` False so a resume re-OCRs.

Hermetic: ``_available_engines_for_agentic`` patched, ``_resolve_judge_model`` -> "". Every
behavioural pin is a DIFFERENCE between two runs in the same process that change one thing,
never an absolute outcome measured locally (CLAUDE.md, #257).
"""

from __future__ import annotations

import json
from pathlib import Path

import fitz
import pytest

from socr.core import manifest
from socr.core.config import EngineType, PipelineConfig
from socr.core.manifest import (
    PageEnding,
    PagePrimaryReason,
    SelectionProvenance,
    _select_page_output_tagged,
)
from socr.core.providers import PROFILE_QWEN_LOCAL
from socr.core.result import DocumentStatus, FailureMode, PageOutput, PageStatus
from socr.core.state import DocumentState, PageState
from socr.pipeline import orchestrator as orch
from socr.pipeline.orchestrator import ATTEMPT_SUMMARY_REASON_MAX_CHARS, UnifiedPipeline

_PROSE = "The estimated effect of the policy change on output is reported below. "
_OCR_MARK = "MODEL-READING-MARK"


def test_loaded_source_is_this_checkout() -> None:
    import socr

    assert Path(socr.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1] / "src")


# ---------------------------------------------------------------------------
# Selection-level pins (unit): same PageState, one flag differs.
# ---------------------------------------------------------------------------


def _page(*, over_raster: bool, scan_failed: bool = False, minus: int = 0, attempts=True):
    from socr.core.document import DocumentHandle

    state = DocumentState(handle=DocumentHandle(path=Path("x.pdf"), page_count=1))
    p = PageState(page_num=1)
    p.is_born_digital = True
    p.native_text = _PROSE * 4
    p.needs_ocr_enhancement = True
    p.invisible_text_over_raster = over_raster
    p.invisible_text_scan_failed = scan_failed
    p.minus_as_digit_hits = minus
    if attempts:
        rejected = PageOutput(
            page_num=1,
            text=_OCR_MARK,
            status=PageStatus.WARNING,
            engine="gemini",
            audit_passed=False,
            failure_mode=FailureMode.AUDIT_FAILED,
        )
        p.attempts = [rejected]
        p.best_output = rejected
    state.pages[1] = p
    return state


def test_invisible_layer_with_rejected_model_ships_floor_not_native() -> None:
    out, prov = _select_page_output_tagged(_page(over_raster=True), 1)
    assert prov is SelectionProvenance.INVISIBLE_SCAN_UNREAD
    assert out.failure_mode is FailureMode.INVISIBLE_SCAN_UNREAD
    assert out.status is PageStatus.WARNING
    assert out.audit_passed is False
    assert "estimated effect" not in out.text and _OCR_MARK not in out.text
    assert manifest.is_page_failed_marker(out.text)
    disp = manifest.provenance_to_disposition(prov)
    assert disp.ending is PageEnding.FAIL_CLOSED_MARKER
    assert disp.primary_reason is PagePrimaryReason.INVISIBLE_SCAN_UNREAD
    # The reason is also recoverable from the shipped bytes alone.
    assert manifest._shipped_marker_reason(out.text) is PagePrimaryReason.INVISIBLE_SCAN_UNREAD


def test_difference_same_page_without_the_invisible_flag_is_unchanged() -> None:
    """A non-invisible needs_ocr_enhancement page keeps the native fallback."""
    on, on_prov = _select_page_output_tagged(_page(over_raster=True), 1)
    off, off_prov = _select_page_output_tagged(_page(over_raster=False), 1)
    assert on_prov is SelectionProvenance.INVISIBLE_SCAN_UNREAD
    assert off_prov is SelectionProvenance.NATIVE_FALLBACK
    assert off.status is PageStatus.WARNING and off.audit_passed is False
    assert "estimated effect" in off.text
    assert off.failure_mode is not FailureMode.INVISIBLE_SCAN_UNREAD


def test_accepted_model_reading_is_unchanged() -> None:
    state = _page(over_raster=True)
    accepted = PageOutput(
        page_num=1,
        text=_OCR_MARK,
        status=PageStatus.SUCCESS,
        engine="gemini",
        audit_passed=True,
    )
    state.pages[1].attempts = [accepted]
    state.pages[1].best_output = accepted
    out, prov = _select_page_output_tagged(state, 1)
    assert prov is SelectionProvenance.PASSING_BEST_OUTPUT
    assert out.text == _OCR_MARK


def test_scan_failed_is_unknown_not_known_garbage_and_keeps_native_fallback() -> None:
    out, prov = _select_page_output_tagged(_page(over_raster=False, scan_failed=True), 1)
    assert prov is SelectionProvenance.NATIVE_FALLBACK
    assert out.failure_mode is FailureMode.NATIVE_INVISIBLE_TEXT_SCAN
    assert "estimated effect" in out.text


def test_no_model_attempt_keeps_the_documented_961_retention() -> None:
    out, prov = _select_page_output_tagged(_page(over_raster=True, attempts=False), 1)
    assert prov is not SelectionProvenance.INVISIBLE_SCAN_UNREAD
    assert "estimated effect" in out.text


def test_failure_mode_ordering_invisible_outranks_minus_as_digit() -> None:
    both, _ = _select_page_output_tagged(_page(over_raster=False, scan_failed=True, minus=3), 1)
    assert both.failure_mode is FailureMode.NATIVE_INVISIBLE_TEXT_SCAN
    # Difference: the same minus hits without the invisible cause keep their own mode.
    minus_only, _ = _select_page_output_tagged(_page(over_raster=False, minus=3), 1)
    assert minus_only.failure_mode is FailureMode.NATIVE_MINUS_AS_DIGIT


# ---------------------------------------------------------------------------
# End to end through process(): judge accepts vs rejects the same scan.
# ---------------------------------------------------------------------------


def _scan_pdf(path: Path) -> Path:
    doc = fitz.open()
    page = doc.new_page()
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 200, 200), False)
    pix.set_rect(pix.irect, (235, 235, 235))
    page.insert_image(page.rect, pixmap=pix)
    y = 72
    for _ in range(8):
        page.insert_text((72, y), _PROSE * 2, fontname="helv", fontsize=10, render_mode=3)
        y += 14
    doc.save(path)
    doc.close()
    return path


class _Engine:
    name = "qwen"

    def is_available(self) -> bool:
        return True

    def process_pages(self, pdf_path, page_nums, config, dpi, subprocess_timeout=None, **_kw):
        return [
            PageOutput(page_num=n, text=_OCR_MARK, status=PageStatus.SUCCESS, engine="qwen")
            for n in page_nums
        ]


class _Judge:
    def __init__(self, accept: bool, reason: str = "") -> None:
        self.accept = accept
        self.reason = reason

    def assess(self, output, provider):
        from socr.pipeline.agentic import AcceptDecision

        return AcceptDecision(accept=self.accept, reason=self.reason)


def _run(tmp_path, monkeypatch, tag, *, accept, reason=""):
    pdf = _scan_pdf(tmp_path / f"{tag}.pdf")
    with monkeypatch.context() as m:
        m.setattr(orch, "get_engine", lambda engine_type: _Engine())
        pipe = UnifiedPipeline(
            PipelineConfig(
                agentic=True,
                quiet=True,
                primary_engine=EngineType.QWEN,
                local_engine=EngineType.QWEN,
                enabled_engines=[EngineType.QWEN],
                native_first=True,
                write_manifest=False,
                judge_backend="heuristic",
                dual_pass_tables=False,
                detect_equations=False,
                save_figures=True,
            )
        )
        pipe._available_engines_for_agentic = lambda: [PROFILE_QWEN_LOCAL]
        pipe._build_page_judge = lambda state: _Judge(accept, reason)
        pipe._resolve_crop_vlm_model = lambda: None
        pipe._resolve_judge_model = lambda *a, **k: ""
        result = pipe.process(pdf, output_dir=tmp_path / f"out-{tag}")
    out = tmp_path / f"out-{tag}"
    side = json.loads(next(iter(out.rglob("pages/00001.json"))).read_text())
    text = next(iter(out.rglob("pages/00001.md"))).read_text()
    audit = json.loads(next(iter(out.rglob("audit_log.json"))).read_text())
    return result, side, text, audit["events"]


def test_e2e_rejected_vs_accepted_same_scan(tmp_path, monkeypatch) -> None:
    long_reason = "rows shifted " * 100
    acc, acc_side, acc_text, acc_events = _run(tmp_path, monkeypatch, "acc", accept=True)
    rej, rej_side, rej_text, rej_events = _run(
        tmp_path, monkeypatch, "rej", accept=False, reason=long_reason
    )

    # Accepted: the model reading ships, nothing about the floor appears.
    assert _OCR_MARK in acc_text
    assert acc_side["failure_mode"] != FailureMode.INVISIBLE_SCAN_UNREAD.value

    # Rejected: neither the layer nor the rejected reading; the floor, honestly named.
    assert _OCR_MARK not in rej_text and "estimated effect" not in rej_text
    assert "invisible OCR layer unread" in rej_text
    assert "invisible_scan_page_p1" in rej_text, "the marker carries the page image"
    assert rej_side["failure_mode"] == FailureMode.INVISIBLE_SCAN_UNREAD.value
    assert rej_side["status"] == "warning"
    assert rej_side["audit_passed"] is False
    assert rej.status is not DocumentStatus.SUCCESS
    kinds = {e["kind"] for e in rej_events}
    assert "invisible_scan_unread" in kinds
    assert "page_failed" not in kinds, "its own event replaces the generic one"
    assert "invisible_scan_unread" not in {e["kind"] for e in acc_events}

    # Sidecar transparency: the rung that ran is visible, reason truncated to the constant.
    summary = rej_side["attempts_summary"]
    assert summary and summary[0]["engine"] == "qwen"
    assert summary[0]["accepted"] is False
    assert 0 < len(summary[0]["rejection_reason"]) <= ATTEMPT_SUMMARY_REASON_MAX_CHARS
    assert acc_side["attempts_summary"][0]["accepted"] is True


def test_attempts_summary_helper_truncates_and_is_additive() -> None:
    from socr.pipeline.orchestrator import _attempts_summary

    ps = PageState(page_num=1)
    ps.attempts = [
        PageOutput(
            page_num=1,
            text="t",
            status=PageStatus.WARNING,
            engine="gemini",
            audit_passed=False,
            judge_reason="x" * (ATTEMPT_SUMMARY_REASON_MAX_CHARS * 3),
            judge_outcome="rejected",
        )
    ]
    (row,) = _attempts_summary(ps)
    assert set(row) == {"engine", "accepted", "judge_outcome", "rejection_reason"}
    assert len(row["rejection_reason"]) == ATTEMPT_SUMMARY_REASON_MAX_CHARS
    assert _attempts_summary(PageState(page_num=2)) == []


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        UnifiedPipeline, "_available_engines_for_agentic", lambda self: [PROFILE_QWEN_LOCAL]
    )
    monkeypatch.setattr(UnifiedPipeline, "_resolve_judge_model", lambda self, *a, **kw: "")


def test_empty_engine_sentinel_is_not_a_model_attempt() -> None:
    """The no-provider sentinel (empty engine) must not trigger the floor."""
    state = _page(over_raster=True)
    state.pages[1].attempts = [
        PageOutput(page_num=1, text="", status=PageStatus.ERROR, engine="", audit_passed=False)
    ]
    state.pages[1].best_output = None
    _, prov = _select_page_output_tagged(state, 1)
    assert prov is not SelectionProvenance.INVISIBLE_SCAN_UNREAD


def test_marker_points_at_image_only_when_one_exists() -> None:
    with_png = _page(over_raster=True)
    with_png.pages[1].invisible_scan_png_ref = "![p](figures/x.png)"
    out_png, _ = _select_page_output_tagged(with_png, 1)
    out_bare, _ = _select_page_output_tagged(_page(over_raster=True), 1)
    assert "see image" in out_png.text and "figures/x.png" in out_png.text
    assert "see image" not in out_bare.text and "see PDF page 1" in out_bare.text
    assert manifest.is_page_failed_marker(out_bare.text)
    assert manifest._shipped_marker_reason(out_bare.text) is PagePrimaryReason.INVISIBLE_SCAN_UNREAD
