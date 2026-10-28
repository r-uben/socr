"""Corporate harvest follow-ups: table-only floors keep prose; garbled layers skip witness."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")

from socr.core.born_digital import BornDigitalDetector
from socr.core.config import EngineType, PipelineConfig
from socr.core.manifest import is_page_failed_marker
from socr.core.result import FailureMode, PageOutput, PageStatus
from socr.core.state import DocumentState, PageState, layer_is_admissible_evidence
from socr.core.table_verification_floor import page_failed_only_table_verification
from socr.pipeline.orchestrator import UnifiedPipeline

_PROSE = "Scope 3 emissions targets for the reporting year"
_TABLE = "| Scope | 2022 |\n| --- | --- |\n| 3 | 412 |"


def _pdf_with_table_and_prose(path: Path) -> Path:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), _PROSE, fontsize=11)
    rows = [["Scope", "2022"], ["3", "412"]]
    for r, cells in enumerate(rows):
        for c, cell in enumerate(cells):
            page.insert_text((80 + c * 100, 140 + r * 22), cell, fontsize=9)
    for r in range(3):
        page.draw_line(fitz.Point(76, 128 + r * 22), fitz.Point(320, 128 + r * 22))
    for c in range(3):
        page.draw_line(fitz.Point(76 + c * 100, 128), fitz.Point(76 + c * 100, 128 + 2 * 22))
    doc.save(str(path))
    doc.close()
    return path


class _TableRejectEngine:
    name = "qwen"

    def is_available(self) -> bool:
        return True

    def process_pages(self, pdf_path, page_nums, config, dpi, subprocess_timeout=None, **_kw):
        return [
            PageOutput(
                page_num=n,
                text=f"{_PROSE}\n\n{_TABLE}",
                status=PageStatus.SUCCESS,
                engine="qwen",
            )
            for n in page_nums
        ]


class _FixedTableVerifierJudge:
    """Wraps the real chain but hard-rejects with a table-verifier reason only."""

    def __init__(self, inner):
        self._inner = inner

    def assess(self, output, provider):
        from socr.pipeline.agentic import AcceptDecision

        return AcceptDecision(
            accept=False,
            reason="native_table_verifier: numbers not backed by page text",
            confidence=0.0,
        )


class _ProseRejectJudge:
    def assess(self, output, provider):
        from socr.pipeline.agentic import AcceptDecision

        return AcceptDecision(accept=False, reason="page judge refused prose cells", confidence=0.0)


def _pipeline(table_only: bool = True):
    from socr.core.providers import PROFILE_QWEN_LOCAL

    pipe = UnifiedPipeline(
        PipelineConfig(
            agentic=True,
            quiet=True,
            primary_engine=EngineType.QWEN,
            local_engine=EngineType.QWEN,
            enabled_engines=[EngineType.QWEN],
            native_first=False,
            write_manifest=False,
            judge_backend="heuristic",
            table_judge_ladder=False,
            dual_pass_tables=False,
            detect_equations=False,
            save_figures=False,
        )
    )
    pipe._available_engines_for_agentic = lambda: [PROFILE_QWEN_LOCAL]
    pipe._resolve_judge_model = lambda *a, **k: ""
    pipe._resolve_crop_vlm_model = lambda: None
    if table_only:
        pipe._build_page_judge = lambda state: _FixedTableVerifierJudge(None)
    else:
        pipe._build_page_judge = lambda state: _ProseRejectJudge()
    return pipe


def _run(tmp_path: Path, monkeypatch, tag: str, *, table_only: bool = True):
    from socr.pipeline import orchestrator as orch

    work = tmp_path / tag
    work.mkdir(parents=True, exist_ok=True)
    pdf = _pdf_with_table_and_prose(work / "doc.pdf")
    monkeypatch.setattr(orch, "get_engine", lambda engine_type: _TableRejectEngine())
    out = work / "out"
    _pipeline(table_only=table_only).process(pdf, output_dir=out)
    body = next(iter(out.rglob("pages/00001.md"))).read_text()
    sidecar = json.loads(next(iter(out.rglob("pages/00001.json"))).read_text())
    return body, sidecar


def test_table_only_verification_keeps_prose_flagged(tmp_path: Path, monkeypatch) -> None:
    body, sidecar = _run(tmp_path, monkeypatch, "table-only", table_only=True)

    assert _PROSE in body
    assert "412" not in body or "failed: unverifiable table" in body
    assert not is_page_failed_marker(body)
    assert sidecar["status"] in ("error", "warning")
    assert sidecar.get("winning_output", {}).get("audit_passed") is False
    events = sidecar.get("audit_events") or []
    assert any(ev.get("kind") == "table_verification_prose_recovery" for ev in events)
    assert sidecar.get("withheld_table_manifest")


def test_prose_refusal_still_floors_whole_page(tmp_path: Path, monkeypatch) -> None:
    body, _sidecar = _run(tmp_path, monkeypatch, "prose-fail", table_only=False)

    assert is_page_failed_marker(body) or _PROSE not in body


def test_reverting_table_only_gate_restores_whole_page_marker(tmp_path: Path, monkeypatch) -> None:
    import socr.core.table_verification_floor as tvf

    monkeypatch.setattr(tvf, "page_failed_only_table_verification", lambda _p: False)
    body, _ = _run(tmp_path, monkeypatch, "reverted", table_only=True)
    assert is_page_failed_marker(body) or _PROSE not in body


def test_garbled_text_layer_routes_off_witness(tmp_path: Path) -> None:
    path = tmp_path / "garbled.pdf"
    doc = fitz.open()
    page = doc.new_page()
    cid_line = " ".join(f"(cid:{n})" for n in range(24))
    page.insert_text((72, 72), cid_line + " " + cid_line, fontsize=10)
    doc.save(str(path))
    doc.close()

    assessment = BornDigitalDetector().detect_page(path, 1)
    assert assessment.is_born_digital is False
    assert assessment.text_layer_witness_unusable is True

    ps = PageState(page_num=1, text_layer_witness_unusable=True)
    assert layer_is_admissible_evidence(ps) is False


def test_no_text_layer_is_not_born_digital(tmp_path: Path) -> None:
    path = tmp_path / "scan.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(str(path))
    doc.close()

    assessment = BornDigitalDetector().detect_page(path, 1)
    assert assessment.is_born_digital is False
    assert assessment.word_count < BornDigitalDetector.MIN_WORDS_SPARSE


def test_page_failed_only_table_verification_predicate() -> None:
    ps = PageState(page_num=1)
    ps.attempts.append(
        PageOutput(
            page_num=1,
            text="x",
            status=PageStatus.ERROR,
            engine="qwen",
            audit_passed=False,
            judge_reason="native_table_verifier: wrong grid shape",
        )
    )
    assert page_failed_only_table_verification(ps)
    ps.attempts.append(
        PageOutput(
            page_num=1,
            text="y",
            status=PageStatus.ERROR,
            engine="qwen",
            audit_passed=False,
            judge_reason="page judge refused prose",
        )
    )
    assert not page_failed_only_table_verification(ps)
