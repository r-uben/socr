"""GH-205: the TR-3 per-region geometry hard-fail must be surfaced, always.

``has_unverifiable_table_region`` is computed on every native table page, but it
only ever reached a surface IN CONJUNCTION with something else — ``--native-only``
for the analyze-time ``table_structure_failed`` event, a failed OCR ladder for the
assemble-time D3-floor event, ``native_table_structure_failed`` for
``_winning_page_output``.  On a page where no conjunction holds the verdict
reached no page status, no document status, no metadata field and no CLI line.

SCOPE IS SURFACING ONLY.  The flag's firing rate is not a measured defect rate,
and the issue blocks any status or routing decision on hand-judging the firing
set first.

The detection has its OWN kind, ``table_region_geometry_hard_fail``, distinct
from the D3 fail-closed ``table_region_unverifiable``: one says "detected, and
nothing acted on it", the other says "acted on it, the region went to the
image-asset lane".  A consumer of ``tables_trust.json`` must be able to tell
those apart, and a D3 page must not read as one kind counted twice.

Two things about the scope guards below, both learned the hard way when the first
attempt at this change (``d490250``) passed locally and failed in CI.

* They are **differential**, never absolute.  The earlier guard pinned outcome
  tuples measured on a developer machine; CI, which has neither ollama nor a
  provider, legitimately produced different ambient values and the pin fired on
  an environment difference rather than on a scope violation.  Each guard here
  runs the pipeline **twice in the same process** and asserts what must be equal
  BETWEEN the two runs, so the ambient environment cancels out.
* Every end-to-end case is parametrised over **both provider states** — an empty
  agentic ladder (CI) and a one-entry ladder (a developer machine).  The empty
  case is listed first because it is the one CI actually runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import fitz
import pytest

from socr.core.born_digital import DocumentAssessment, PageAssessment
from socr.core.config import EngineType, PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.providers import PROFILE_QWEN_LOCAL
from socr.core.result import PageOutput, PageStatus
from socr.core.state import DocumentState
from socr.pipeline.orchestrator import UnifiedPipeline

DETECTION_KIND = "table_region_geometry_hard_fail"
DISPOSITION_KIND = "table_region_unverifiable"
# #262: the third TR-3 kind -- the D3 floor was SUPERSEDED because a ladder
# attempt authored a grid, so the page ships that reading FLAGGED rather than
# the failed-table marker. A disposition like ``DISPOSITION_KIND``, and mutually
# exclusive with it on any one page.
SUPERSEDED_KIND = "d3_floor_model_table_kept"

# CI has no ollama and no provider; a developer machine has both. Parametrising
# over the agentic ladder covers the second axis, and pinning the judge to the
# heuristic backend removes the first: ``judge_backend="auto"`` probes ollama,
# which is exactly the ambient difference that made the previous attempt's
# absolute pins unreproducible in CI (and cost 30s per run locally).
PROVIDER_STATES = [
    pytest.param([], id="no_provider"),
    pytest.param([PROFILE_QWEN_LOCAL], id="provider"),
]


def _make_pipeline(**overrides) -> UnifiedPipeline:
    cfg = PipelineConfig(
        agentic=True,
        judge_backend="heuristic",
        enabled_engines=[EngineType.GEMINI],
        primary_engine=EngineType.DEEPSEEK,
        save_figures=False,
        dual_pass_tables=False,
        detect_equations=False,
        recover_clean_equations=False,
        quiet=True,
        audit_enabled=False,
        write_manifest=False,
        **overrides,
    )
    return UnifiedPipeline(cfg)


def _page(page_num: int, *, unverifiable: bool) -> PageAssessment:
    return PageAssessment(
        page_num=page_num,
        is_born_digital=True,
        native_text=f"native table text for page {page_num} " * 5,
        confidence=0.9,
        has_tables=True,
        has_unverifiable_table_region=unverifiable,
    )


def _run_analyze(pipeline: UnifiedPipeline, pages: list[PageAssessment]) -> DocumentState:
    pdf_path = Path("/tmp/gh205.pdf")
    assessment = DocumentAssessment(path=pdf_path, pages=pages)
    pipeline.bd_detector = MagicMock()
    pipeline.bd_detector.detect.return_value = assessment
    state = DocumentState(handle=DocumentHandle(path=pdf_path, page_count=len(pages)))
    pipeline._phase_analyze(state)
    return state


def test_tr3_hard_fail_is_surfaced_without_native_only() -> None:
    """The conjunction-free case: an ordinary run, not ``--native-only``.

    At the base commit the flag is set on the assessment and nothing anywhere
    records it.
    """
    pipeline = _make_pipeline(native_only=False)
    state = _run_analyze(pipeline, [_page(1, unverifiable=True), _page(2, unverifiable=False)])

    tr3 = [e for e in state.events if e.kind == DETECTION_KIND]
    assert [e.page_num for e in tr3] == [1], (
        "The TR-3 geometry hard-fail on page 1 reached no surface at all; "
        f"recorded events were {[(e.page_num, e.kind) for e in state.events]}"
    )


def test_detection_kind_is_distinct_from_the_d3_disposition_kind() -> None:
    """The two TR-3 kinds mean different things and must stay different kinds.

    ``table_region_unverifiable`` is emitted at assemble and means the OCR ladder
    also failed, so the region was routed to the image-asset lane. The analyze-time
    detection means only that TR-3 fired and nothing acted on it — the page's
    native text may still ship. Collapsing them makes ``tables_trust.json``
    unable to distinguish a routed page from an unrouted one, and makes a D3 page
    carry the same kind twice.
    """
    from socr.core.audit_log import build_run_audit
    from socr.core.tables_trust import TABLE_DISTRUST_KINDS

    assert DETECTION_KIND != DISPOSITION_KIND
    assert DETECTION_KIND in TABLE_DISTRUST_KINDS, sorted(TABLE_DISTRUST_KINDS)
    assert DISPOSITION_KIND in TABLE_DISTRUST_KINDS, sorted(TABLE_DISTRUST_KINDS)

    pipeline = _make_pipeline(native_only=False)
    state = _run_analyze(pipeline, [_page(1, unverifiable=True)])
    kinds = [e.kind for e in state.events]
    assert kinds.count(DETECTION_KIND) == 1, kinds
    assert DISPOSITION_KIND not in kinds, (
        "analyze-time detection must not claim the D3 fail-closed disposition: "
        "nothing has been routed to the image-asset lane at this point"
    )

    # Both kinds are ranked, so a page carrying the detection AND the later D3
    # disposition reads top-to-bottom as "detected here, acted on there".
    audit = build_run_audit(state)
    ranked = [e.kind for e in audit.events if e.page_num == 1]
    assert ranked == [DETECTION_KIND], ranked


def test_clean_table_page_stays_quiet() -> None:
    """Reverse regression: no flag, no event. A clean run leaves no noise."""
    pipeline = _make_pipeline(native_only=False)
    state = _run_analyze(pipeline, [_page(1, unverifiable=False), _page(2, unverifiable=False)])

    assert not [e for e in state.events if e.kind == DETECTION_KIND], state.events


# ----------------------------------------------------------------------
# End-to-end over the whole of ``process()`` — analyze, agentic routing AND
# assemble — because page status is decided in assemble and routing in agentic,
# and a guard that stops at analyze can see neither.
# ----------------------------------------------------------------------


class _FixedDetector:
    def __init__(self, assessment: DocumentAssessment) -> None:
        self._assessment = assessment

    def detect(self, path: Path) -> DocumentAssessment:
        assert path == self._assessment.path
        return self._assessment


_FIXTURE_MARKDOWN = "\n".join(
    ["| Coefficient | Estimate | Standard error |", "| --- | --- | --- |"]
    + [f"| Coefficient {row} | {row}.12 | {row}.34 |" for row in range(10)]
)


def _native_table_pdf(path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((60, 70), "Estimate Standard error", fontsize=9)
    for row in range(10):
        y = 90 + row * 16
        page.insert_text((60, y), f"Coefficient {row}", fontsize=9)
        page.insert_text((260, y), f"{row}.12", fontsize=9)
    page.draw_line(fitz.Point(50, 60), fitz.Point(360, 60))
    page.draw_line(fitz.Point(50, 260), fitz.Point(360, 260))
    doc.save(path)
    doc.close()


def _run_process(
    tmp_path: Path,
    *,
    agentic: bool,
    unverifiable: bool,
    ladder: list,
    emit: bool = True,
    quiet: bool = True,
):
    """One full ``process()`` run, NOT ``--native-only``, with the OCR ladder watched.

    ``ladder`` is what ``_available_engines_for_agentic`` returns — ``[]`` models
    CI, a one-entry list models a developer machine.  ``emit=False`` disables the
    GH-205 emission and nothing else, which is how the scope guard isolates this
    change's effect from everything the ambient environment does.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    pdf_path = tmp_path / "paper.pdf"
    _native_table_pdf(pdf_path)
    native_text = _FIXTURE_MARKDOWN
    assessment = DocumentAssessment(
        path=pdf_path,
        pages=[
            PageAssessment(
                page_num=1,
                is_born_digital=True,
                native_text=native_text,
                confidence=1.0,
                has_tables=True,
                has_unverifiable_table_region=unverifiable,
            )
        ],
    )
    pipeline = UnifiedPipeline(
        PipelineConfig(
            agentic=agentic,
            judge_backend="heuristic",
            native_first=True,
            native_only=False,
            primary_engine=EngineType.QWEN,
            enabled_engines=[EngineType.QWEN],
            tiered=False,
            dual_pass_tables=False,
            detect_equations=False,
            save_figures=False,
            write_manifest=True,
            quiet=quiet,
        )
    )
    pipeline.bd_detector = _FixedDetector(assessment)
    pipeline._available_engines_for_agentic = lambda: list(ladder)
    pipeline._surface_table_scoring = lambda *args, **kwargs: None
    # The non-agentic path runs ``_phase_judge_hard_pages``, which builds an
    # OllamaVisionJudge and posts to it for real regardless of ``judge_backend``.
    # On a developer machine that is a ~35s live call per run; in CI it is a
    # refused connection. Resolving no judge model takes the phase out of the
    # picture entirely, in both places, so the runs being compared differ in the
    # TR-3 variable and in nothing else. This is the same ambient dependency that
    # made the previous attempt's absolute pins unreproducible in CI.
    pipeline._resolve_judge_model = lambda: ""
    if not emit:
        # ``raising=False`` semantics on purpose: at the base commit the method
        # does not exist, so this is a no-op there and both halves of the guard
        # run identical code — the guard then fails on the event assertion, which
        # is behavioural, rather than on an AttributeError for a new symbol.
        pipeline._emit_tr3_detection_events = lambda *args, **kwargs: None

    # The OCR ladder is recorded, not forbidden: this run is NOT --native-only,
    # so a table page legitimately enters it. What #205 forbids is the TR-3 flag
    # CHANGING that decision, which is why the guards compare two runs rather
    # than asserting an absolute route.
    ocr_calls: list[tuple] = []

    def _record_ocr(_state, page_nums, _fallback, engine, _phase, *args, **kwargs):
        ocr_calls.append((tuple(page_nums), engine.value))
        return [
            PageOutput(
                page_num=n,
                text=native_text,
                status=PageStatus.SUCCESS,
                engine=engine.value,
                confidence=0.95,
                audit_passed=True,
            )
            for n in page_nums
        ]

    pipeline._run_engine_on_pages = _record_ocr

    captured: dict = {}
    assemble = pipeline._phase_assemble

    def _capture_state(state, output_dir):
        captured["state"] = state
        return assemble(state, output_dir)

    pipeline._phase_assemble = _capture_state
    output_dir = tmp_path / "out"
    result = pipeline.process(pdf_path, output_dir)
    return result, captured["state"], output_dir, ocr_calls


def _outcome(result, state) -> tuple:
    """(page status, audit_passed, failure mode, document status) — the tuple #205 pins."""
    page = state.pages[1]
    assert page.attempts, "no attempt recorded for page 1"
    att = page.attempts[0]
    return (att.status, att.audit_passed, att.failure_mode, result.status)


def _only(path: Path, pattern: str) -> Path:
    matches = list(path.rglob(pattern))
    assert len(matches) == 1, f"expected exactly one {pattern} under {path}, got {matches}"
    return matches[0]


@pytest.mark.parametrize("ladder", PROVIDER_STATES)
@pytest.mark.parametrize("agentic", [False, True])
def test_tr3_detection_reaches_every_surface_end_to_end(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], agentic: bool, ladder: list
) -> None:
    """Cardinal rule: the detection must surface at page, document, metadata AND CLI.

    Each surface is checked under the DETECTION kind's own name, so a cell where
    the D3 disposition happens to fire anyway cannot make this pass vacuously.
    """
    capsys.readouterr()
    _result, state, output_dir, _ocr = _run_process(
        tmp_path, agentic=agentic, unverifiable=True, ladder=ladder, quiet=False
    )

    # 1. page level: the event is recorded against the page, exactly once
    detections = [e for e in state.events if e.kind == DETECTION_KIND]
    assert [e.page_num for e in detections] == [1], [(e.page_num, e.kind) for e in state.events]

    # 2. audit log
    audit = json.loads(_only(output_dir, "audit_log.json").read_text(encoding="utf-8"))
    assert audit["counts"].get(DETECTION_KIND) == 1, audit["counts"]

    # 3. trust sidecar, under the DETECTION kind's own name
    trust = json.loads(_only(output_dir, "tables_trust.json").read_text(encoding="utf-8"))
    assert trust["untrusted_pages"] == [1], trust
    assert DETECTION_KIND in trust["pages"]["1"]["reasons"], trust["pages"]["1"]["reasons"]

    # 4. document metadata
    doc_metadata = json.loads(
        _only(output_dir / "paper", "metadata.json").read_text(encoding="utf-8")
    )
    assert "untrusted tables" in (doc_metadata.get("error") or ""), doc_metadata.get("error")

    # 5. CLI
    cli_output = capsys.readouterr().out
    assert "untrusted tables" in cli_output, cli_output


@pytest.mark.parametrize("ladder", PROVIDER_STATES)
def test_d3_page_carries_detection_and_disposition_once_each(tmp_path: Path, ladder: list) -> None:
    """A D3 fail-closed page reads as "detected here, acted on there".

    Before this change both emissions used ``table_region_unverifiable``, so the
    same page carried one kind twice and a consumer could not tell a routed
    region from an unrouted detection. They are now two kinds, one each.
    """
    _result, state, output_dir, _ocr = _run_process(
        tmp_path, agentic=True, unverifiable=True, ladder=ladder
    )

    kinds = [e.kind for e in state.events if e.page_num == 1]
    assert kinds.count(DETECTION_KIND) == 1, kinds
    # #262: WHICH disposition depends on whether the ladder authored a grid,
    # and on this fixture that is not a free variable -- ``native_table_
    # structure_failed`` is set BY the structural gate firing on the model's
    # grid, so a grid-free ladder output leaves the page with no hard fail and
    # no disposition at all (measured). With a provider the grid exists, so the
    # floor is superseded and the page ships the model's flagged reading; with
    # the empty CI ladder nothing was produced and the floor fires. Either way
    # the guarantee this test exists for is unchanged: EXACTLY ONE disposition,
    # ranked after the detection, and never one kind counted twice.
    dispositions = [k for k in kinds if k in (DISPOSITION_KIND, SUPERSEDED_KIND)]
    assert len(dispositions) == 1, kinds
    disposition = dispositions[0]

    audit = json.loads(_only(output_dir, "audit_log.json").read_text(encoding="utf-8"))
    assert audit["counts"].get(DETECTION_KIND) == 1, audit["counts"]
    assert audit["counts"].get(disposition) == 1, audit["counts"]
    # Ranked detection-before-disposition, so the page's story reads in order.
    page_kinds = [e["kind"] for e in audit["events"] if e["page_num"] == 1]
    assert page_kinds.index(DETECTION_KIND) < page_kinds.index(disposition), page_kinds


# ----------------------------------------------------------------------
# Scope guards.  Both are DIFFERENTIAL: two runs in the same process, one
# variable changed, assert what must be equal between them.  Neither pins an
# absolute outcome, because an absolute outcome is a property of the machine.
# ----------------------------------------------------------------------


@pytest.mark.parametrize("ladder", PROVIDER_STATES)
@pytest.mark.parametrize("agentic", [False, True])
def test_surfacing_changes_nothing_but_the_event(
    tmp_path: Path, agentic: bool, ladder: list
) -> None:
    """THE scope guard: the GH-205 emission, and only it, is toggled.

    Two runs, identical in every respect — same flagged page, same ladder, same
    stubs, same process — except that one has ``_emit_tr3_detection_events``
    disabled.  Any difference in the outcome tuple or the OCR call list is
    therefore caused by this change and nothing else, whatever the ambient
    environment does.  That is the invariant #205 demands while its firing set
    is unjudged: surfacing only, no status, no routing.

    This subsumes the document-status question that the flag-differential guard
    below has to leave out, because here the flag is held constant.
    """
    with_event = _run_process(
        tmp_path / "with", agentic=agentic, unverifiable=True, ladder=ladder, emit=True
    )
    without_event = _run_process(
        tmp_path / "without", agentic=agentic, unverifiable=True, ladder=ladder, emit=False
    )

    assert _outcome(with_event[0], with_event[1]) == _outcome(without_event[0], without_event[1]), (
        "the GH-205 surfacing moved a status the issue forbids it to move "
        f"(agentic={agentic}, ladder={[p.name for p in ladder]}): "
        f"{_outcome(with_event[0], with_event[1])} != "
        f"{_outcome(without_event[0], without_event[1])}"
    )
    assert with_event[3] == without_event[3], (
        f"the GH-205 surfacing changed OCR routing: {with_event[3]} != {without_event[3]}"
    )

    # ...and the event itself really is the difference, so the equality above is
    # not the trivial equality of two identical runs.
    assert [e.kind for e in with_event[1].events].count(DETECTION_KIND) == 1, [
        e.kind for e in with_event[1].events
    ]
    assert DETECTION_KIND not in [e.kind for e in without_event[1].events]


@pytest.mark.parametrize("ladder", PROVIDER_STATES)
@pytest.mark.parametrize("agentic", [False, True])
def test_flagging_a_page_moves_no_page_status_and_no_route(
    tmp_path: Path, agentic: bool, ladder: list
) -> None:
    """Flag differential: TR-3 on vs off changes the page's fate in no way.

    Page status, ``audit_passed``, the failure mode and the OCR call list are all
    equal between a flagged and an unflagged run of the same page.

    **Document status is deliberately not compared here, and that is a measured
    fact rather than an omission.**  At the base commit ``f5b1d2a``, before any
    part of this change, flipping the flag already moves the document status in
    all four cells — SUCCESS -> AUDIT_FAILED non-agentic, AUDIT_FAILED -> ERROR
    agentic — because ``native_table_unverifiable`` is already a term in
    ``native_fallback_pages`` and in the D3 floor, and ``pages_ok`` is keyed on
    both.  So #205's "surfaced nowhere" is true of the analyze-time DETECTION,
    not of the flag in general.  Asserting document-status equality across the
    flag would assert something false about the pipeline as it already is; the
    guard above holds the flag constant instead, and covers document status there.
    """
    flagged = _run_process(tmp_path / "flagged", agentic=agentic, unverifiable=True, ladder=ladder)
    clean = _run_process(tmp_path / "clean", agentic=agentic, unverifiable=False, ladder=ladder)

    assert _outcome(*flagged[:2])[:3] == _outcome(*clean[:2])[:3], (
        "the TR-3 flag moved a PAGE-level outcome that #205 forbids it to move "
        f"(agentic={agentic}, ladder={[p.name for p in ladder]}): "
        f"{_outcome(*flagged[:2])[:3]} != {_outcome(*clean[:2])[:3]}"
    )
    assert flagged[3] == clean[3], f"the TR-3 flag changed OCR routing: {flagged[3]} != {clean[3]}"

    # The differential is real: the event fires on the flagged run only.
    assert [e.kind for e in flagged[1].events].count(DETECTION_KIND) == 1, [
        e.kind for e in flagged[1].events
    ]
    assert DETECTION_KIND not in [e.kind for e in clean[1].events]

    # The --native-only demotion event must not have leaked out of --native-only.
    assert not [e for e in flagged[1].events if e.kind == "table_structure_failed"]
