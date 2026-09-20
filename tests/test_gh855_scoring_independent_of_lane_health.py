"""GH-855: a latched escalation lane must not also silence table SCORING.

``_escalation_degraded`` (orchestrator.py, declared once per document, outside
the page loop) is a document-scoped latch: once one page's escalation deadline
expires it stays True for the rest of the document. Before this fix it gated
BOTH downstream consumers:

* ``_escalate_table_page`` -- the recovery attempt itself. Correct: a wedged
  provider must not be retried on every remaining page.
* ``_surface_table_scoring`` -- pure reporting (``state.events`` only; see its
  docstring). Wrong: the moment recovery gives up on a document, socr also
  stopped RECORDING that a later page's table disagrees with its native text
  layer -- exactly the silent-loss shape CLAUDE.md forbids.

The AFFECTED population is exactly the pages where ``_page_has_tables`` is
False (the detector missed the table), the lane is configured, the page isn't
a chart asset, and the latch is set -- those are the pages the old expression
skipped scoring on and the new one does not. On that population neither
downstream consumer of the score is reachable (``:9148``'s dual-pass reread
also requires ``_page_has_tables``; ``:9212``'s escalation also requires
``_lane_live``, false while latched), so this ticket's own claim is that nothing
except the new AuditEvent can move on those pages.

Two independent assertions, deliberately not one comparison carrying both:

1. ``test_a_latched_page_the_detector_missed_now_gets_scored`` -- presence.
   With the latch forced SET, a page in the affected population emits a real
   scoring AuditEvent. Falsified by the mutation guard: reverting the fix's
   one-token change (``_lane_configured`` -> ``_lane_live``) makes this
   assertion fail, because the event is never emitted.
2. ``test_the_latch_does_not_move_text_audit_passed_or_status`` -- safety.
   Same fixture, latch forced SET vs forced CLEAR, escalation forced to
   REJECT in both runs (so ``bo`` never changes downstream of the score).
   Selected text, per-page ``audit_passed`` and per-page status must be
   identical between the two runs.

CI has no ollama and no provider (see CLAUDE.md): ``_available_engines_for_agentic``
is patched in every run, and ``_escalate_table_page`` is replaced with a
deterministic double -- GH-96's own real timeout/networking behaviour is
already covered by ``test_gh96_escalation_lane.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

fitz = pytest.importorskip("fitz")

from socr.core.config import EngineType, PipelineConfig  # noqa: E402
from socr.core.providers import PROFILE_GEMINI, PROFILE_QWEN_LOCAL  # noqa: E402
from socr.core.result import PageOutput, PageStatus  # noqa: E402
from socr.pipeline.agentic import AcceptDecision, PageDecision, ProviderAttempt  # noqa: E402
from socr.pipeline.orchestrator import UnifiedPipeline  # noqa: E402

_ROWS = [
    ("Total effect of decisions", ["42.8", "30.5", "2.6"]),
    ("September energy package", ["43.2", "26.8", "3.7"]),
    ("Energy price guarantee", ["24.8", "26.8", "3.7"]),
    ("Energy bill relief scheme", ["18.4", "9.1", "5.5"]),
]

# Drops two of the three native columns. `_table_page_needs_escalation` reads
# this against the native grid drawn below and (verified empirically before
# writing this test) emits a real `table_unexplained_lanes` AuditEvent -- not
# merely a bool -- so presence is a directly observable fact, not an inferred
# one.
_MISSING_COLUMNS_CANDIDATE = "\n".join(
    ["| | c1 |", "| --- | --- |"] + [f"| {label} | {values[0]} |" for label, values in _ROWS]
)


def _grid_pdf(path: Path, pages: int) -> Path:
    """A born-digital PDF with `pages` copies of the GH-96 reference grid."""
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    for _ in range(pages):
        pg = doc.new_page()
        y = 200.0
        for label, values in _ROWS:
            pg.insert_text((60.0, y), label, fontsize=9)
            for x, v in zip((300.0, 360.0, 420.0), values):
                pg.insert_text((x, y), v, fontsize=9)
            y += 18.0
        pg.draw_line(fitz.Point(50, 190), fitz.Point(470, 190))
        pg.draw_line(fitz.Point(50, y), fitz.Point(470, y))
    doc.save(str(path))
    doc.close()
    return path


def _route_fn(page_num, ladder, run_provider, judge, **kwargs):
    """Every page routes to a fixed candidate missing two native columns."""
    prof = ladder[0]
    out = PageOutput(
        page_num=page_num,
        text=_MISSING_COLUMNS_CANDIDATE,
        status=PageStatus.SUCCESS,
        # Label the output with the profile it is actually packaged under.
        # A hardcoded name here can disagree with `ladder[0]` and make the
        # fixture describe a route the run never took (GH-861).
        engine=getattr(prof.engine, "value", str(prof.engine)),
        audit_passed=True,
    )
    decision = judge.assess(out, prof)
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
        page_num=page_num, final_output=out, attempts=[att], accepted=decision.accept
    )


def _config() -> PipelineConfig:
    return PipelineConfig(
        primary_engine=EngineType.QWEN,
        agentic=True,
        judge_backend="heuristic",
        enabled_engines=[EngineType.QWEN],
        quiet=True,
        save_figures=False,
        write_manifest=True,
        native_first=False,
        dual_pass_tables=False,
        escalate_ambiguous_tables=True,
        table_judge_ladder=False,
    )


def _run(tmp_path: Path, *, page_one_degrades: bool) -> tuple[Path, list[int]]:
    """Run the fused agentic loop over a 2-page grid document.

    Returns ``(output_dir, escalated_pages)``, the second being the page
    numbers ``_escalate_table_page`` was actually called for, in order. The
    call record is what lets a test tell "escalation was skipped because the
    lane is latched" from "escalation ran and declined" -- the two are
    indistinguishable from the sidecar, because this double always returns
    the incumbent unchanged (GH-861).

    `_page_has_tables` is forced False so every page is in the AFFECTED
    population this ticket's expression change covers -- isolated from the
    (unaffected) detector-flagged first arm. Escalation is a deterministic
    double that ALWAYS rejects (returns the incumbent `bo` unchanged): page 1
    latches the lane iff this run is the "degraded" one; real GH-96
    timeout/networking behaviour is test_gh96_escalation_lane.py's concern.
    """
    pdf = _grid_pdf(tmp_path / "doc.pdf", pages=2)
    output_dir = tmp_path / "out"
    pipeline = UnifiedPipeline(_config())

    escalated_pages: list[int] = []

    def _stub_escalate(state, page_num, ps, bo, profile, run_provider, pdf_path, **kwargs):
        escalated_pages.append(page_num)
        degraded = page_one_degrades and page_num == 1
        return degraded, bo

    with (
        patch.object(
            pipeline,
            "_available_engines_for_agentic",
            return_value=[PROFILE_QWEN_LOCAL, PROFILE_GEMINI],
        ),
        patch.object(UnifiedPipeline, "_page_has_tables", return_value=False),
        patch.object(pipeline, "_escalate_table_page", side_effect=_stub_escalate),
        patch("socr.pipeline.orchestrator.route_page", side_effect=_route_fn),
        patch(
            "socr.pipeline.agentic.HeuristicPageJudge.assess",
            return_value=AcceptDecision(accept=True, reason="heuristics passed"),
        ),
        patch("socr.pipeline.orchestrator.probe_ollama_idle", return_value=True),
    ):
        pipeline.process(pdf, output_dir)

    return output_dir, escalated_pages


def _audit_events(output_dir: Path) -> list[dict]:
    paths = list(output_dir.rglob("audit_log.json"))
    if not paths:
        return []
    return json.loads(paths[0].read_text()).get("events", [])


def _page_sidecar(output_dir: Path, page_num: int) -> dict:
    path = next(output_dir.rglob(f"pages/{page_num:05d}.json"))
    return json.loads(path.read_text())["winning_output"]


# ----------------------------------------------------------------------
# 1. Presence -- falsified by the mutation guard, not by this run alone
# ----------------------------------------------------------------------


def test_a_latched_page_the_detector_missed_now_gets_scored(tmp_path: Path) -> None:
    """Latch forced SET: page 2 (affected population) still emits a scoring event.

    Pre-fix, gating this arm on `_lane_live` instead of `_lane_configured`
    meant page 2 was never scored once page 1 latched the lane -- this
    assertion is exactly what the mutation guard (see decision log) shows
    fails when that coupling is restored.
    """
    output_dir, _escalated = _run(tmp_path, page_one_degrades=True)

    events = _audit_events(output_dir)
    page_two_kinds = [e["kind"] for e in events if e.get("page_num") == 2]
    assert "table_unexplained_lanes" in page_two_kinds, (
        "GH-855 regression: page 2's table-scoring event is missing once page 1 "
        f"latched the lane -- coverage depended on lane health: {page_two_kinds!r}"
    )


# ----------------------------------------------------------------------
# 2. Safety -- the latch alone must not move text / audit_passed / status
# ----------------------------------------------------------------------


def test_the_latch_does_not_move_text_audit_passed_or_status(tmp_path: Path) -> None:
    """Paired run: only the latch state differs; escalation rejects in both.

    Neither downstream consumer of the score is reachable on the affected
    population (`:9148` also requires `_page_has_tables`, false here by
    construction; `:9212` also requires `_lane_live`, false while latched) --
    so scoring must stay observation-only regardless of whether the latch is
    set. `page_one_degrades` is the only variable between the two runs.
    """
    clear_dir, clear_escalated = _run(tmp_path / "clear", page_one_degrades=False)
    set_dir, set_escalated = _run(tmp_path / "set", page_one_degrades=True)

    clear_page_two = _page_sidecar(clear_dir, 2)
    set_page_two = _page_sidecar(set_dir, 2)

    assert clear_page_two["text"] == set_page_two["text"] == _MISSING_COLUMNS_CANDIDATE, (
        "page 2's selected text moved with the latch state"
    )
    assert clear_page_two["audit_passed"] == set_page_two["audit_passed"], (
        f"page 2's audit_passed moved with the latch state: "
        f"clear={clear_page_two['audit_passed']!r} set={set_page_two['audit_passed']!r}"
    )
    assert clear_page_two["status"] == set_page_two["status"], (
        f"page 2's status moved with the latch state: "
        f"clear={clear_page_two['status']!r} set={set_page_two['status']!r}"
    )

    # Escalation itself must still respect the latch, and only the call
    # record can show it: the double returns the incumbent unchanged, so a
    # page that was never escalated and a page that was escalated and
    # declined produce identical sidecars. Page 1 is escalated in both runs
    # (the lane is live going in); page 2 is where the runs diverge, and it
    # must be skipped in the SET run and attempted in the CLEAR one.
    assert clear_escalated == [1, 2], clear_escalated
    assert set_escalated == [1], set_escalated

    # ... while the scoring event is emitted for page 2 in BOTH runs. That is
    # the GH-855 separation: the latch stops the recovery attempt without
    # stopping the report.
    clear_events_p2 = [e["kind"] for e in _audit_events(clear_dir) if e.get("page_num") == 2]
    set_events_p2 = [e["kind"] for e in _audit_events(set_dir) if e.get("page_num") == 2]
    assert "table_unexplained_lanes" in clear_events_p2
    assert "table_unexplained_lanes" in set_events_p2
