"""GH-995: pages a PARTIAL_SAVE_VLM_TIMEOUT halt never processed must not ship terminal SUCCESS.

Before the fix the halted loop ``break``-ed, then assemble stamped every unprocessed
page ``SUCCESS / native_prose / terminal=true`` with no events, and the resume ledger
(``_load_terminal_page``) trusted them: a re-run skipped pages no table or model pass
had ever seen.

Pinned as a DIFFERENCE, not an absolute: the same document is run once with no halt
and once with the halt forced at page 1. Provider-dependent machinery (the D3 floor,
``native_fallback``) does not fire in CI, so no locally measured status tuple is pinned.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from socr.core.providers import PROFILE_QWEN_LOCAL
from socr.core.result import FailureMode, PageOutput, PageStatus

from test_pp2_agentic_fuse import (
    EngineType,
    _make_bd_assessment,
    _make_config,
    _make_pipeline,
)

PAGES = 4
HALT_AT = 1  # page 1 times out; pages 2..4 are never processed
NATIVE = {2, 4}  # born-digital; 1 and 3 go through route_page


def _pdf(directory: Path) -> Path:
    fitz = pytest.importorskip("fitz")
    path = directory / "doc.pdf"
    doc = fitz.open()
    for i in range(PAGES):
        doc.new_page().insert_text((72, 72), f"page {i + 1} text " * 10)
    doc.save(str(path))
    doc.close()
    return path


def _decision(page_num, ladder, *, timeout: bool):
    from socr.pipeline.agentic import PageDecision, ProviderAttempt

    out = PageOutput(
        page_num=page_num,
        text="" if timeout else "recognised text " * 20,
        status=PageStatus.ERROR if timeout else PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=not timeout,
    )
    prof = ladder[0]
    att = ProviderAttempt(
        engine=prof.engine,
        output=out,
        cost_usd=0.0,
        accepted=not timeout,
        reason="provider timeout" if timeout else "ok",
        provider_id=prof.id,
        model=prof.model,
        backend=prof.backend,
    )
    return PageDecision(page_num=page_num, final_output=out, attempts=[att])


def _run(pdf: Path, out: Path, *, halt: bool, reprocess: bool = False):
    pipeline = _make_pipeline(
        _make_config(agentic=True, enabled_engines=[EngineType.QWEN], reprocess=reprocess)
    )
    pipeline.bd_detector = MagicMock()
    pipeline.bd_detector.detect.return_value = _make_bd_assessment(PAGES, born_digital_pages=NATIVE)
    routed: list[int] = []
    ledger_hits: list[int] = []
    real_load = pipeline._load_terminal_page

    def _spy_load(state, page_num, output_dir):
        hit = real_load(state, page_num, output_dir)
        if hit is not None:
            ledger_hits.append(page_num)
        return hit

    pipeline._load_terminal_page = _spy_load

    def _route(page_num, ladder, run_provider, judge, **kwargs):
        routed.append(page_num)
        return _decision(page_num, ladder, timeout=halt and page_num == HALT_AT)

    with (
        patch.object(pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]),
        patch.object(pipeline, "_resolve_judge_model", return_value=""),
        patch("socr.pipeline.orchestrator.route_page", side_effect=_route),
        patch("socr.pipeline.orchestrator.probe_ollama_idle", return_value=not halt),
    ):
        result = pipeline.process(pdf, out)
    return result, routed, ledger_hits


def _sidecars(out: Path) -> dict[int, dict]:
    pages = out / "doc" / "pages"
    return {int(p.stem): json.loads(p.read_text()) for p in sorted(pages.glob("*.json"))}


def test_halt_leaves_unprocessed_pages_non_terminal_and_not_success(tmp_path: Path) -> None:
    (tmp_path / "c").mkdir()
    (tmp_path / "h").mkdir()
    control_out = tmp_path / "control"
    halted_out = tmp_path / "halted"
    _, control_routed, _ = _run(_pdf(tmp_path / "c"), control_out, halt=False)
    result, halted_routed, _ = _run(_pdf(tmp_path / "h"), halted_out, halt=True)

    assert result.error and "PARTIAL_SAVE_VLM_TIMEOUT" in result.error
    assert halted_routed == [HALT_AT]  # the halt really cut the loop
    assert control_routed == [1, 3]

    control = _sidecars(control_out)
    halted = _sidecars(halted_out)
    # Difference: the same pages are terminal when the loop reached them ...
    assert all(control[n]["terminal"] is True for n in range(HALT_AT + 1, PAGES + 1))
    # ... and are not when the halt cut the loop before them.
    for n in range(HALT_AT + 1, PAGES + 1):
        assert halted[n]["terminal"] is not True, f"p{n} must not be skippable on resume"
        assert halted[n]["status"] != "success", f"p{n} must not read SUCCESS"
        assert halted[n]["failure_mode"] == FailureMode.PAGE_NOT_PROCESSED_AFTER_HALT.value


def test_resume_reprocesses_pages_the_halt_skipped(tmp_path: Path) -> None:
    pdf = _pdf(tmp_path)
    out = tmp_path / "out"
    _run(pdf, out, halt=True)

    # Second run, same output dir and input, no wedge. ``--reprocess`` lifts the
    # document-level skip (a PARTIAL doc with an unchanged fingerprint is skipped whole);
    # the per-page ledger is what then decides each page, and it must not trust the
    # pages the halt never reached.
    _, routed, ledger_hits = _run(pdf, out, halt=False, reprocess=True)
    assert 3 in routed
    # The native pages are never routed, so the ledger is the only thing that could
    # skip them: it must have turned every page the halt never reached away.
    assert not set(ledger_hits) & set(range(HALT_AT + 1, PAGES + 1)), ledger_hits
    resumed = _sidecars(out)
    for n in range(HALT_AT + 1, PAGES + 1):
        assert resumed[n]["terminal"] is True
        assert resumed[n]["status"] == "success"
