"""#1253 B3: a refused native table gets one model attempt, and ships only if accepted.

Every test drives ``process()`` on synthetic PDFs and pins a DIFFERENCE, never a value read
off one machine (CLAUDE.md, #253/#257): the same document is run twice in one process with
only the retry allowlist changed (the "off" arm IS main's behaviour), in both provider
states. ``route_page`` is stubbed; the provider ladder, the judge model and the engine
selection are pinned so the result does not depend on what this host has pulled.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import fitz
import pytest
from native_table_fixtures import (
    COL_XS,
    HEADER,
    ROWS,
    flush_and_restore,
    native_first_config,
    routed_decision,
)

from socr.core.born_digital import BornDigitalDetector
from socr.core.document import DocumentHandle
from socr.core.manifest import (
    REJECTION_JUDGE_ONLY,
    d3_floor_kept_model_output,
)
from socr.core.providers import PROFILE_QWEN_LOCAL
from socr.core.result import PageOutput, PageStatus
from socr.core.state import DocumentState
from socr.pipeline.agentic import PageDecision, ProviderAttempt
from socr.pipeline.orchestrator import UnifiedPipeline
from socr.tables.native_first import (
    MODEL_RETRY_KIND,
    MODEL_RETRY_REFUSAL_REASONS,
    REFUSE,
    NativeTableFirstWork,
    NativeTablePlan,
)

PROVIDER_STATES = pytest.param([PROFILE_QWEN_LOCAL], id="provider"), pytest.param([], id="none")

#: A model grid the strict-grid check recognises, so a rejected reading WOULD ship through
#: #262 if the retry exemption were missing.
MODEL_GRID = "| A | B |\n| --- | --- |\n| x | 1 |\n| y | 2 |\n"
#: Pages the 156-document run ended as "failed table, see image" with no model attempt.
MEASURED_REFUSED_PAGES = 513
REFUSAL_REASONS_NOT_RETRIED = ("header_unattributed", "label_binding", "text layer unreadable")


def _many_page_pdf(path: Path, pages: int) -> None:
    doc = fitz.open()
    for _ in range(pages):
        page = doc.new_page(width=612, height=792)
        page.insert_text((72, 50), "Table 1. GDP growth forecasts.", fontsize=10, fontname="helv")
        for ci, hdr in enumerate(HEADER):
            page.insert_text((COL_XS[ci], 80), hdr, fontsize=9, fontname="helv")
        for ri, row in enumerate(ROWS):
            for ci, cell in enumerate(row):
                page.insert_text((COL_XS[ci], 100 + ri * 22), cell, fontsize=9, fontname="helv")
    doc.save(str(path))
    doc.close()


class _DefectiveDetector(BornDigitalDetector):
    """Flags every page's native table structure as defective: a real REFUSE plan."""

    def detect(self, path):
        assessment = super().detect(path)
        for page in assessment.pages:
            page.native_table_structure_defective = True
        return assessment


def _decision(page_num: int, ladder: list, kind: str) -> PageDecision:
    if kind == "accepted":
        return routed_decision(page_num, ladder, text=MODEL_GRID)
    out = PageOutput(
        page_num=page_num,
        text=MODEL_GRID,
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=False,
        rejection_class=REJECTION_JUDGE_ONLY,
    )
    prof = ladder[0]
    attempt = ProviderAttempt(
        engine=prof.engine,
        output=out,
        cost_usd=0.0,
        accepted=False,
        reason="judge refused",
        provider_id=prof.id,
        model=prof.model,
        backend=prof.backend,
    )
    return PageDecision(page_num=page_num, final_output=out, attempts=[attempt], accepted=False)


def _run(
    tmp_path: Path,
    providers: list,
    *,
    retry: bool,
    kind: str = "accepted",
    pages: int = 1,
    tag: str = "",
    plan_reason: str | None = None,
):
    pdf = tmp_path / f"t{pages}.pdf"
    if not pdf.exists():
        _many_page_pdf(pdf, pages)
    pipeline = UnifiedPipeline(native_first_config())
    pipeline.bd_detector = _DefectiveDetector()
    routes: list[int] = []

    def _route(page_num, ladder, run_provider, judge, **kwargs):
        routes.append(page_num)
        return _decision(page_num, ladder, kind)

    allowlist = MODEL_RETRY_REFUSAL_REASONS if retry else frozenset()
    out_dir = tmp_path / f"out_{'on' if retry else 'off'}_{kind}_{len(providers)}_{tag}"
    patches = [
        patch("socr.pipeline.orchestrator.route_page", side_effect=_route),
        patch.object(pipeline, "_available_engines_for_agentic", return_value=providers),
        patch.object(pipeline, "_resolve_judge_model", return_value=""),
        patch("socr.pipeline.orchestrator.MODEL_RETRY_REFUSAL_REASONS", allowlist),
    ]
    if plan_reason is not None:
        patches.append(
            patch.object(
                pipeline,
                "_plan_native_table_first",
                return_value=NativeTableFirstWork(NativeTablePlan(REFUSE, reason=plan_reason)),
            )
        )
    for p in patches:
        p.start()
    try:
        result = pipeline.process(pdf, out_dir)
    finally:
        for p in reversed(patches):
            p.stop()
    sidecars = {
        int(f.stem): json.loads(f.read_text(encoding="utf-8"))
        for f in sorted(out_dir.rglob("pages/*.json"))
        if f.stem.isdigit()
    }
    return result, sidecars, routes


def _kinds(sidecar: dict) -> list[str]:
    return [e["kind"] for e in sidecar["audit_events"]]


def _retry_events(sidecar: dict) -> list[dict]:
    return [e for e in sidecar["audit_events"] if e["kind"] == MODEL_RETRY_KIND]


@pytest.mark.parametrize("providers", PROVIDER_STATES)
def test_retry_differs_from_main_only_when_a_provider_is_up(tmp_path, providers) -> None:
    on_result, on, on_routes = _run(tmp_path, providers, retry=True)
    off_result, off, off_routes = _run(tmp_path, providers, retry=False)
    # Both arms refuse the native grid first: the bookkeeping is unchanged.
    assert "native_table_cell_unresolved" in _kinds(on[1])
    assert "native_table_cell_unresolved" in _kinds(off[1])
    assert _retry_events(off[1]) == []
    if providers:
        assert off_routes == []
        assert on_routes == [1]
        (event,) = _retry_events(on[1])
        assert event["data"]["attempted"] is True
        assert event["data"]["accepted"] is True
        assert event["data"]["reason"] == "structure_defective"
        # An accepted reading replaces the marker.
        assert on_result.markdown != off_result.markdown
        assert "| x | 1 |" in on_result.markdown
    else:
        # No provider: the empty ladder never reaches the retry. Identical to main.
        assert on_routes == off_routes == []
        assert _retry_events(on[1]) == []
        assert on_result.markdown == off_result.markdown
        assert on[1]["status"] == off[1]["status"]
        assert _kinds(on[1]) == _kinds(off[1])


@pytest.mark.parametrize("providers", PROVIDER_STATES)
def test_a_rejected_reading_ends_as_the_same_marker_main_ships(tmp_path, providers) -> None:
    """P0-1 and P0-2 together: not the refused native grid, not the rejected model grid."""
    on_result, on, _ = _run(tmp_path, providers, retry=True, kind="rejected")
    off_result, off, _ = _run(tmp_path, providers, retry=False, kind="rejected")
    # Byte-identical to main's refusal: the refused grid is not shipped, the rejected model
    # grid is not shipped (#262 blocked), and the page's prose outcome is the same.
    assert on_result.markdown == off_result.markdown
    assert "| x | 1 |" not in on_result.markdown
    assert on[1]["status"] == off[1]["status"] == "error"
    # P0-1: the refusal flags that route a not-accepted attempt to the floor (and never to
    # NATIVE_FALLBACK) are set on the retried page exactly as on the refused one.
    for key in ("native_table_unverifiable", "native_table_structure_failed"):
        assert on[1].get(key) is True and off[1].get(key) is True, key
    assert on[1]["native_table_unverifiable"] == off[1]["native_table_unverifiable"]
    if providers:
        (event,) = _retry_events(on[1])
        assert event["data"]["attempted"] is True
        assert event["data"]["accepted"] is False
        assert event["data"]["why_not_accepted"]
        assert event["data"]["shipped_refused_native_grid"] is False
        assert event["data"]["primary_reason"] != "unaccepted_output_kept"
        assert event["data"]["selection_provenance"] != "native_fallback"


def test_rejected_reading_is_kept_by_262_without_the_exemption(tmp_path) -> None:
    """The comparison arm of P0-2: same rejected reading, ``native_table_model_retry`` cleared.

    Proves the reading really would ship over the floor, so the exemption has something to do.
    """
    pdf = tmp_path / "one.pdf"
    _many_page_pdf(pdf, 1)
    state = DocumentState(handle=DocumentHandle(path=pdf, page_count=1))
    pipeline = UnifiedPipeline(native_first_config())
    pipeline._phase_analyze(state)
    ps = state.pages[1]
    ps.native_table_structure_failed = True
    ps.native_table_unverifiable = True
    out = _decision(1, [PROFILE_QWEN_LOCAL], "rejected").final_output
    ps.attempts.append(out)
    ps.best_output = out
    assert d3_floor_kept_model_output(ps) is out
    ps.native_table_model_retry = "row_count"
    assert d3_floor_kept_model_output(ps) is None


@pytest.mark.parametrize("providers", PROVIDER_STATES)
def test_unaccepted_kept_is_counted_apart_from_accepted_recoveries(tmp_path, providers) -> None:
    _, accepted, _ = _run(tmp_path, providers, retry=True, kind="accepted", pages=3, tag="a")
    _, rejected, _ = _run(tmp_path, providers, retry=True, kind="rejected", pages=3, tag="r")
    if not providers:
        assert all(not _retry_events(s) for s in (*accepted.values(), *rejected.values()))
        return
    acc = [e["data"] for s in accepted.values() for e in _retry_events(s)]
    rej = [e["data"] for s in rejected.values() for e in _retry_events(s)]
    assert len(acc) == len(rej) == 3
    assert all(d["accepted"] for d in acc)
    assert not any(d["accepted"] for d in rej)
    assert not any(d["primary_reason"] == "unaccepted_output_kept" for d in acc + rej)
    assert not any(d["shipped_refused_native_grid"] for d in acc + rej)


def test_every_refused_page_gets_exactly_one_recorded_attempt(tmp_path) -> None:
    """The measured shape: every refused page gets one record, with a provider up.

    ``MEASURED_REFUSED_PAGES`` (513, the 156-document run) takes about a minute through
    ``process()``, so the suite runs a smaller document and ``SOCR_RETRY_FULL_SCALE=1`` runs
    the measured count. The assertion is the same: records == pages, none missing.
    """
    pages = MEASURED_REFUSED_PAGES if os.environ.get("SOCR_RETRY_FULL_SCALE") else 48
    _, sidecars, routes = _run(
        tmp_path, [PROFILE_QWEN_LOCAL], retry=True, kind="rejected", pages=pages
    )
    assert len(sidecars) == pages
    assert sorted(routes) == list(range(1, pages + 1))
    events = [e for s in sidecars.values() for e in _retry_events(s)]
    assert len(events) == pages
    assert all(e["data"]["attempted"] for e in events)
    assert not any(e["data"]["shipped_refused_native_grid"] for e in events)
    assert sum(1 for e in events if e["data"]["primary_reason"] == "unaccepted_output_kept") == 0
    assert all(s["native_table_model_retry"] == "structure_defective" for s in sidecars.values())


@pytest.mark.parametrize("reason", sorted(MODEL_RETRY_REFUSAL_REASONS))
def test_each_allowlisted_reason_is_retried(tmp_path, reason) -> None:
    _, sidecars, routes = _run(
        tmp_path, [PROFILE_QWEN_LOCAL], retry=True, plan_reason=reason, tag=reason
    )
    assert routes == [1]
    (event,) = _retry_events(sidecars[1])
    assert event["data"]["reason"] == reason


@pytest.mark.parametrize("reason", REFUSAL_REASONS_NOT_RETRIED)
def test_other_refusal_reasons_stay_refused(tmp_path, reason) -> None:
    _, sidecars, routes = _run(
        tmp_path, [PROFILE_QWEN_LOCAL], retry=True, plan_reason=reason, tag=reason
    )
    assert routes == []
    assert _retry_events(sidecars[1]) == []
    assert "native_table_cell_unresolved" in _kinds(sidecars[1])


def test_allowlist_is_the_five_named_reasons() -> None:
    assert MODEL_RETRY_REFUSAL_REASONS == {
        "row_count",
        "structure_defective",
        "unlocalizable_cells",
        "region_unverifiable",
        "numeric_orphan",
    }


def test_retry_record_and_flag_survive_a_resume(tmp_path) -> None:
    _, sidecars, _ = _run(tmp_path, [PROFILE_QWEN_LOCAL], retry=True, kind="rejected", tag="resume")
    assert sidecars[1]["native_table_model_retry"] == "structure_defective"
    assert MODEL_RETRY_KIND in UnifiedPipeline.resume_restore_kinds()
    pdf = tmp_path / "t1.pdf"
    pipeline = UnifiedPipeline(native_first_config())
    state = DocumentState(handle=DocumentHandle(path=pdf, page_count=1))
    pipeline._phase_analyze(state)
    judge_patch = patch.object(pipeline, "_resolve_judge_model", return_value="")
    judge_patch.start()
    state.pages[1].native_table_model_retry = "row_count"
    from socr.core.audit_log import AuditEvent

    state.events.append(
        AuditEvent(page_num=1, kind=MODEL_RETRY_KIND, engine="qwen", data={"reason": "row_count"})
    )
    try:
        resumed = flush_and_restore(pipeline, state, pdf, tmp_path / "out_resume_flush")
    finally:
        judge_patch.stop()
    assert resumed.pages[1].native_table_model_retry == "row_count"
    assert [e.kind for e in resumed.events].count(MODEL_RETRY_KIND) == 1
