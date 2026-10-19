"""PR #1211 review round 2: Argon's four P1s on `418b2903`.

Round 1 closed the readers; these are the four ways the layer still reached a decision.
Two of them are mistakes round 1 itself made, and both are recorded as such:
`verify_scanned_table`'s deferral guard was removed on reasoning that held only for the
orchestrator's non-raising path, and the resume ASSIGNED the suspicion flag from the
sidecar instead of OR-ing it with this run's own analysis.

The rule is unchanged: on the UNION (`state.layer_is_admissible_evidence`, a FAILED
detection scan included), the suspect layer is never evidence for or against a reading.

One ruling shapes P1-2 and is pinned here rather than left in prose (owner, 2026-10-19):
the native verifier may not convict a reading the PIXEL lane has independently
confirmed, and otherwise it keeps checking. NOT a blanket bypass on the union -- on a
page with no opaque cover the pixels ARE the layer's own ink, so dropping the row check
there lets a row swap through. #1156's `test_a_transparent_cover_is_not_an_independent_witness`
and `test_a_detection_error_does_not_turn_a_swap_into_an_acceptance` measure exactly
that, both stay green, and both go red under the blanket version (trialled).

Hermetic: no provider, no model, no tesseract. The pixel witness is injected, which is
what the production caller passes.
"""

from __future__ import annotations

import json
from pathlib import Path

import fitz
import pytest
from ocr_output_contract import doc_dir_for, relative_key
from test_1205_table_chain_evidence_readers import (
    _LAYER_MD,
    _TRUE_MD,
    _flag,
    _make_config,
    _make_pipeline,
    _make_state,
)
from test_1211_r1_remaining_layer_readers import _pixel_witness_true, _visible_layer_table_pdf

from socr.core.providers import PROFILE_QWEN_LOCAL
from socr.core.result import PageOutput, PageStatus
from socr.core.state import layer_is_admissible_evidence
from socr.core.tables_trust import TABLE_DISTRUST_KINDS, build_tables_trust
from socr.pipeline.orchestrator import UnifiedPipeline


def test_loaded_source_is_this_checkout() -> None:
    import socr

    assert Path(socr.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1] / "src")


# ---------------------------------------------------------------------------
# P1-1: every arm of verify_scanned_table reads layer_is_evidence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("native_trusted", [False, None, True], ids=["false", "none", "true"])
def test_the_layer_is_excluded_whatever_native_trusted_says(tmp_path: Path, native_trusted) -> None:
    """Argon P1-1, pinned WITHOUT stubbing the function, which is why it was missed.

    Round 1 removed the guard on the deferral, reasoning that the orchestrator derives
    ``native_trusted`` from the same predicate so a suspect page always arrives with
    ``False``. True only of its non-raising path: ``SourceEvidenceTableJudge`` sets
    ``trusted=None`` when its trust source raises or is absent, and ``None`` is "not
    False" -- so the page deferred to the native verifier with its layer intact, and
    round 1's own test passed only because it stubbed ``verify_scanned_table``.

    All three values of ``native_trusted``, both readings, no stub.
    """
    from socr.tables.source_evidence import verify_scanned_table

    doc = fitz.open(_visible_layer_table_pdf(tmp_path))
    try:
        page = doc[0]

        def verdict(text: str, *, layer_is_evidence: bool):
            return verify_scanned_table(
                page,
                text,
                ocr_image_fn=_pixel_witness_true,
                native_trusted=native_trusted,
                layer_is_evidence=layer_is_evidence,
            )

        excluded_layer = verdict(_LAYER_MD, layer_is_evidence=False)
        assert not excluded_layer.passed and not excluded_layer.deferred, (
            "the layer's own reading still passed, or was handed inward undecided: "
            f"passed={excluded_layer.passed} deferred={excluded_layer.deferred} "
            f"reason={excluded_layer.reason!r}"
        )

        excluded_pixels = verdict(_TRUE_MD, layer_is_evidence=False)
        assert excluded_pixels.passed and not excluded_pixels.deferred, (
            f"the pixel-confirmed reading did not pass: {excluded_pixels.reason!r}"
        )

        # Control: with the layer admissible, this page behaves as it always has --
        # otherwise the assertions above would also hold for a function that simply
        # refuses everything.
        included = verdict(_LAYER_MD, layer_is_evidence=True)
        assert included.passed, "control: the layer's reading fails even when admissible"
    finally:
        doc.close()


# ---------------------------------------------------------------------------
# P1-2: the composed judge, and the narrow ruling
# ---------------------------------------------------------------------------


def _compose_and_decide(tmp_path: Path, text: str, *, flags: dict, witness=_pixel_witness_true):
    """Argon's own repro shape: the REAL stack from ``_build_page_judge``."""
    pipeline = _make_pipeline(_make_config())
    state = _make_state(_visible_layer_table_pdf(tmp_path))
    ps = state.pages[1]
    ps.has_tables = True
    ps.is_born_digital = True
    _flag(ps, **flags)
    judge = pipeline._build_page_judge(state)
    # The pixel witness the production judge uses (tesseract on the crop). Set on the
    # INSTANCE, which is where ``__init__`` put it; CI has no tesseract, so a test
    # leaning on the real one would read as "no witness" there and pass vacuously.
    judge._ocr_image_fn = witness
    return judge.assess(
        PageOutput(
            page_num=1, text=text, status=PageStatus.SUCCESS, engine="qwen", audit_passed=True
        ),
        PROFILE_QWEN_LOCAL,
    )


def test_the_layer_does_not_convict_a_pixel_confirmed_reading(tmp_path: Path) -> None:
    """Argon P1-2, end to end through ``_build_page_judge``.

    On a scan-failed page with no covering raster the outer judge PASSES the
    pixel-confirmed ``-350`` against its witness, and the inner native verifier then
    rejected it with ``value_guard_multiset_mismatch`` on the strength of the layer's
    ``350``. The layer convicting what the pixels confirmed.

    The clean arm is the control AND the limit of the ruling: there the layer IS
    evidence, so the verifier convicts exactly as before. Nothing about an admissible
    page changes.
    """
    for label, flags in (
        ("scan-failed", {"positive": False, "scan_failed": True}),
        ("positive finding", {"positive": True}),
    ):
        decision = _compose_and_decide(tmp_path / label.replace(" ", "-"), _TRUE_MD, flags=flags)
        assert not decision.reason.startswith("native_table_verifier:"), (
            f"{label}: the layer convicted a pixel-confirmed reading: {decision.reason!r}"
        )

    clean = _compose_and_decide(tmp_path / "clean", _TRUE_MD, flags={"positive": False})
    assert clean.reason.startswith("native_table_verifier:"), (
        "control: the verifier no longer convicts on an ADMISSIBLE page either, so the "
        f"arms above are not about suspicion: {clean.reason!r}"
    )

    # And the layer's own reading is still refused on a suspect page -- by the pixel
    # lane, which is the half of this that must not weaken.
    layer_reading = _compose_and_decide(
        tmp_path / "layer", _LAYER_MD, flags={"positive": False, "scan_failed": True}
    )
    assert layer_reading.reason.startswith("source_evidence_table:"), (
        f"the layer's own reading was not refused by the pixel lane: {layer_reading.reason!r}"
    )


def test_a_confirmation_needs_the_pixels_not_the_layer_agreeing_with_itself(
    tmp_path: Path,
) -> None:
    """The ruling's own boundary: no witness means no confirmation.

    The bypass is licensed only by a reading of the PIXELS that excluded the layer. With
    no witness the outer judge cannot confirm anything -- it refuses the page -- so the
    inner verifier is never reached and no confirmation is recorded. Without this, a
    page with no tesseract would silently acquire a bypass, which is the fail-open
    direction every round of this ticket has had to close.
    """
    decision = _compose_and_decide(
        tmp_path / "nowitness",
        _TRUE_MD,
        flags={"positive": False, "scan_failed": True},
        witness=lambda _pix: "",
    )
    assert not decision.accept
    assert decision.reason.startswith("source_evidence_table:"), (
        f"a page with no pixel witness was not refused: {decision.reason!r}"
    )


def test_the_swap_guards_stay_green() -> None:
    """The ruling's reason, pinned rather than left in the log.

    #1156's two guards both assert that a SUSPECT page still runs row verification when
    nothing independent read it. The blanket bypass the finding's literal wording asks
    for turns both red (trialled: 5 failures, including these two). This test fails the
    day someone takes the blanket route, and names why.
    """
    from test_gh1156_prior_ocr_not_native_trusted import (
        test_a_detection_error_does_not_turn_a_swap_into_an_acceptance as _swap,
        test_a_transparent_cover_is_not_an_independent_witness as _cover,
    )

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        _cover(Path(tmp) / "cover")
        _swap(Path(tmp) / "swap")


# ---------------------------------------------------------------------------
# P1-3 and P1-4: the resume
# ---------------------------------------------------------------------------

_MD = "| Item | OLS |\n| --- | --- |\n| RowA | 350 |\n"


def _resume(tmp_path: Path, sidecar: dict, *, live_flags: dict):
    pipeline = _make_pipeline(_make_config())
    tmp_path.mkdir(parents=True, exist_ok=True)
    pdf = tmp_path / "doc.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(pdf)
    doc.close()

    state = _make_state(pdf)
    ps = state.pages[1]
    _flag(ps, **live_flags)
    output_dir = tmp_path / "out"
    doc_dir = doc_dir_for(output_dir, relative_key(pdf, pdf.parent))
    (doc_dir / "pages").mkdir(parents=True, exist_ok=True)
    (doc_dir / "pages" / "00001.json").write_text(json.dumps(sidecar), encoding="utf-8")
    page_out = PageOutput(
        page_num=1, text=_MD, status=PageStatus.SUCCESS, engine="qwen", audit_passed=True
    )
    pipeline._restore_terminal_page_state(state, 1, page_out, output_dir)
    return state, ps


def test_the_crop_refusal_survives_a_resume(tmp_path: Path) -> None:
    """Argon P1-3: the kind was in ``TABLE_DISTRUST_KINDS`` but not replayed.

    The event is emitted at the crop reread, which a resumed page skips, so the only
    way it reaches a resumed page is the replay. Without it the page comes back absent
    from ``tables_trust.json`` with its credential restored -- and #1210 made that
    reachable by admitting accepted prior-OCR pages to the ledger, which on the Fed
    scans are the pages this kind fires on.
    """
    assert "table_crop_repair_unverifiable" in TABLE_DISTRUST_KINDS, (
        "the kind is no longer distrust, so replaying it would prove nothing"
    )
    assert "table_crop_repair_unverifiable" in UnifiedPipeline.resume_restore_kinds()

    sidecar = {
        "prior_ocr_layer_suspect": True,
        "audit_events": [
            {
                "page_num": 1,
                "kind": "table_crop_repair_unverifiable",
                "engine": "qwen",
                "detail": "crop-repair fallback not run on p1",
                "data": {},
            }
        ],
    }
    state, _ps = _resume(tmp_path, sidecar, live_flags={"positive": True})
    assert "table_crop_repair_unverifiable" in {e.kind for e in state.events}
    trust = build_tables_trust(state.handle.filename, state.events)
    assert trust.untrusted_pages == [1], (
        f"the resumed page lost its crop refusal: {trust.untrusted_pages}"
    )
    assert "table_crop_repair_unverifiable" in trust.counts_by_kind()


#: A sidecar carrying every layer-derived record a resume can restore.
_LAYER_DERIVED_SIDECAR = {
    "native_cell_binding": {"p1-t0": {"table_id": "p1-t0", "outcome": "contradicted"}},
    "binding_adjudication": {"p1-t0": {"verdict": "contradicted"}},
    "audit_events": [
        {
            "page_num": 1,
            "kind": "native_cell_check_contradicted",
            "engine": "qwen",
            "detail": "table p1-t0: cell binding contradicted",
            "data": {},
        },
        {
            "page_num": 1,
            "kind": "table_binding_boundary_unresolved",
            "engine": "qwen",
            "detail": "table p1-t0: 1 boundary word(s) unresolved",
            "data": {"table_id": "p1-t0", "words": [{"text": "0.51", "bbox": [1, 2, 3, 4]}]},
        },
    ],
}


def test_a_stale_sidecar_cannot_clear_this_runs_suspicion(tmp_path: Path) -> None:
    """Argon P1-4, with his own ``carries_flag=False`` case.

    ``_restore_terminal_page_state`` ASSIGNED ``prior_ocr_layer_suspect`` from the
    sidecar, so a False written by any build that could not see the flag cleared the
    suspicion ``_phase_analyze`` had just established from the PDF -- after which every
    gate in the file read the page as admissible and the layer decided it again. It now
    takes the UNION; removing trust is never destructive, so OR is the safe direction.

    All four layer-derived restores are asserted, not just the flag: the per-cell
    verdict, the binding adjudication, and both replayed events.
    """
    state, ps = _resume(
        tmp_path / "stale", dict(_LAYER_DERIVED_SIDECAR), live_flags={"positive": True}
    )
    assert ps.prior_ocr_layer_suspect is True, "a stale sidecar cleared the live suspicion"
    assert layer_is_admissible_evidence(ps) is False
    assert sorted(getattr(ps, "native_cell_binding", None) or {}) == []
    assert sorted(getattr(ps, "binding_adjudication", None) or {}) == []
    assert {e.kind for e in state.events} == set(), (
        f"a layer-derived event was replayed: {sorted({e.kind for e in state.events})}"
    )
    assert build_tables_trust(state.handle.filename, state.events).untrusted_pages == []


def test_the_binding_records_are_gated_even_when_the_sidecar_carries_the_flag(
    tmp_path: Path,
) -> None:
    """The second half of P1-4: those two restores were ungated on BOTH paths.

    ``_apply_binding_adjudication_meta`` runs from the resume gate and from the restore,
    and ``table_binding_boundary_unresolved`` was replayed regardless. ``bind()`` is a
    geometric check against the page's own layer, so a restored adjudication is the
    layer's verdict carried into a run that would refuse to recompute it -- in either
    direction, since the record both clears contradictions and preserves them.

    The control is the admissible arm, where every one of them still comes back.
    """
    flagged_sidecar = {**_LAYER_DERIVED_SIDECAR, "prior_ocr_layer_suspect": True}
    state, ps = _resume(tmp_path / "flagged", flagged_sidecar, live_flags={"positive": False})
    assert sorted(getattr(ps, "binding_adjudication", None) or {}) == []
    assert "table_binding_boundary_unresolved" not in {e.kind for e in state.events}

    clean_state, clean_ps = _resume(
        tmp_path / "clean", dict(_LAYER_DERIVED_SIDECAR), live_flags={"positive": False}
    )
    assert sorted(getattr(clean_ps, "binding_adjudication", None) or {}) == ["p1-t0"], (
        "control: the binding adjudication is not restored even on an admissible page"
    )
    assert "table_binding_boundary_unresolved" in {e.kind for e in clean_state.events}


def test_the_resume_skip_gate_does_not_apply_the_binding_record_either(tmp_path: Path) -> None:
    """The SECOND call site of ``_apply_binding_adjudication_meta``, which a mutant found
    unpinned.

    It runs from ``_load_terminal_page`` -- the resume SKIP gate -- as well as from the
    restore, and in production both run for a resumed page (``_phase_agentic`` calls the
    gate and then the restore). ``_phase_analyze`` has already run by then, so the live
    suspicion is set and the guard is reachable.

    Driven through the real gate, with a real flushed sidecar, and observed by spying on
    the call: the clean arm must apply it, or this proves nothing about the flagged one.
    """
    from unittest.mock import patch

    pipeline = _make_pipeline(_make_config())

    def resume_through_the_gate(*, flagged: bool):
        src = tmp_path / ("flagged" if flagged else "clean")
        pdf = _visible_layer_table_pdf(src)
        state = _make_state(pdf)
        ps = state.pages[1]
        output = PageOutput(
            page_num=1, text=_MD, status=PageStatus.SUCCESS, engine="qwen", audit_passed=True
        )
        ps.attempts.append(output)
        ps.best_output = output
        ps.binding_adjudication = {"p1-t0": {"verdict": "contradicted"}}
        out_dir = src / "out"
        pipeline._flush_page_fragment(state, 1, output.text, out_dir)
        pipeline._flush_page_sidecar(state, 1, out_dir)

        resumed = _make_state(pdf)
        resumed.pages[1].attempts.clear()
        resumed.pages[1].best_output = None
        _flag(resumed.pages[1], positive=flagged)

        calls: list = []
        real = type(pipeline)._apply_binding_adjudication_meta

        def spy(self, state_, page_num, meta):
            calls.append(page_num)
            return real(self, state_, page_num, meta)

        with patch.object(type(pipeline), "_apply_binding_adjudication_meta", spy):
            restored = pipeline._load_terminal_page(resumed, 1, out_dir)
        return restored, calls

    restored, clean_calls = resume_through_the_gate(flagged=False)
    assert restored is not None, (
        "control: the resume gate did not admit the page at all, so neither arm reaches "
        "the call this pins"
    )
    assert clean_calls == [1], (
        f"control: the skip gate does not apply the binding record on a clean page: {clean_calls}"
    )

    _restored_flagged, flagged_calls = resume_through_the_gate(flagged=True)
    assert flagged_calls == [], (
        "the resume skip gate applied the layer's binding record on a flagged page"
    )


# ---------------------------------------------------------------------------
# The replay audit: the one RESOLVING kind this loop replays
# ---------------------------------------------------------------------------


def _accepted_event(reason: str | None) -> dict:
    data: dict = {"table_id": "p1-t0", "rung_trail": [], "witness_scope": "located"}
    if reason is not None:
        data["reason"] = reason
    return {
        "page_num": 1,
        "kind": "table_ladder_accepted",
        "engine": "qwen",
        "detail": f"table p1-t0 accepted ({reason or 'by the judge ladder'})",
        "data": data,
    }


def test_a_geometry_overruled_acceptance_is_not_restored_on_a_flagged_page(
    tmp_path: Path,
) -> None:
    """Found by auditing all 60 replayed kinds in round 2; UPGRADED in round 5.

    ``table_ladder_accepted`` is the only RESOLVING kind the replay loop carries: it
    CLEARS that table's distrust. ``data["reason"]`` is present only when the readers
    were OVERRULED, and ``verified_by_geometry`` means the overruling evidence was
    ``bind()``'s check against this page's own text layer.

    **Round 2 pinned the DROP of that event at restore; round 5 removed the drop.** Not a
    weakening: the kind is now in ``_LAYER_CAUSED_LADDER_KINDS``, so the page is refused
    at the gate and never restored at all -- a strictly stronger outcome than dropping
    one event from a page that is kept. Round 2's version could not have caught what was
    wrong with dropping, because it drove the restore directly and so never saw the gate
    admit the page; Argon's round-5 P1 did. This test now asserts the stronger thing,
    with round 2's three controls intact.

    The consequence pins live with the round-5 fixture
    (``test_1211_r3_layer_caused_replays.py``), which drives the gate and the backfill.
    """
    from test_1211_r3_layer_caused_replays import (
        _accepted_event,
        _resume_through_the_gate,
        _sidecar_run,
    )

    from socr.judge.table_verdict import (
        REASON_VERIFIED_BY_BLIND_CELL_TRANSCRIPTION,
        REASON_VERIFIED_BY_GEOMETRY,
    )

    def terminal(reason: str | None, *, flagged: bool) -> bool:
        tag = f"{reason or 'plain'}-{flagged}"
        pipeline, pdf, out_dir = _sidecar_run(
            tmp_path / tag, [_accepted_event(reason)], flagged_then=False, disposition=None
        )
        return _resume_through_the_gate(pipeline, pdf, out_dir, flagged_now=flagged)["terminal"]

    assert terminal(REASON_VERIFIED_BY_GEOMETRY, flagged=False), (
        "control: a geometry-overruled acceptance is not a ledger hit even on an "
        "admissible page, so the assertion below is not about suspicion"
    )
    assert not terminal(REASON_VERIFIED_BY_GEOMETRY, flagged=True), (
        "a flagged page whose table the layer's own binding check accepted was skipped "
        "instead of re-read"
    )
    assert terminal(None, flagged=True), (
        "an ORDINARY ladder acceptance carries no overruling reason and is the readers' "
        "own verdict; re-reading it would throw away work the pixels did"
    )
    assert terminal(REASON_VERIFIED_BY_BLIND_CELL_TRANSCRIPTION, flagged=True), (
        "a blind-cell transcription is a third vendor reading the CROP, independent of "
        "the layer, and must stay terminal"
    )
