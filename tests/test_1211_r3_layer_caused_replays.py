"""PR #1211 round 3: the promotion, the candidate-scoped confirmation, and five
layer-derived replayed kinds round 2's audit got wrong.

The rule is unchanged: on the UNION (`state.layer_is_admissible_evidence`, a FAILED
detection scan included), the suspect layer is never evidence for or against a reading.

Round 3's scope is set by one ruling (owner, 2026-10-19). P1-2B is confined to the
PROMOTION path: refusing every unconfirmed table on a suspect page is #1208's rule and
must ship with retention for born-digital flagged pages, so applied here alone it would
refuse without retaining on #1202's ~14,850 pages. Everywhere else the judge keeps what
round 2 settled, and the four tests that pin that stay as they are.
"""

from __future__ import annotations

import json
from pathlib import Path

import fitz
import pytest
from unittest.mock import patch
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

from socr.core.manifest import _credential_admission_refusal
from socr.core.providers import PROFILE_QWEN_LOCAL
from socr.core.result import PageOutput, PageStatus
from socr.core.tables_trust import TABLE_DISTRUST_KINDS, build_tables_trust
from socr.pipeline.agentic import AcceptDecision

_MD = "| Item | OLS |\n| --- | --- |\n| RowA | 350 |\n"


def test_loaded_source_is_this_checkout() -> None:
    import socr

    assert Path(socr.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1] / "src")


def _out(text: str) -> PageOutput:
    return PageOutput(
        page_num=1, text=text, status=PageStatus.SUCCESS, engine="qwen", audit_passed=True
    )


# ---------------------------------------------------------------------------
# P1-2A: the confirmation belongs to the candidate that earned it
# ---------------------------------------------------------------------------


def test_a_confirmation_does_not_carry_to_another_candidate(tmp_path: Path) -> None:
    """Argon P1-2A: ``_pixel_confirmed_pages`` was page-scoped and never cleared.

    A rung-1 confirmation therefore licensed skipping ``verify_native_table`` on an
    unconfirmed rung 2 -- the layer's own ``350`` riding in on the confirmation the true
    ``-350`` earned. It is now keyed by the candidate OBJECT and consumed once.

    The control is the first assertion: rung 1 DOES earn a confirmation, so the second is
    about scope and not about the mechanism being dead.
    """
    pipeline = _make_pipeline(_make_config())
    state = _make_state(_visible_layer_table_pdf(tmp_path))
    ps = state.pages[1]
    ps.has_tables = True
    ps.is_born_digital = True
    _flag(ps, positive=False, scan_failed=True)
    judge = pipeline._build_page_judge(state)
    judge._ocr_image_fn = _pixel_witness_true
    native = judge._inner

    rung1 = _out(_TRUE_MD)
    judge.assess(rung1, PROFILE_QWEN_LOCAL)
    assert native._pixel_confirmed_candidates == {}, (
        "the confirmation was not consumed by the candidate that earned it"
    )

    # Mint one and leave it unconsumed, which is the state round 2 could reach.
    rung1b = _out(_TRUE_MD)
    native._pixel_confirmed_candidates[id(rung1b)] = rung1b
    assert native._pixel_confirmed_candidates, "control: nothing was recorded to carry"

    rung2 = _out(_LAYER_MD)
    native.assess(rung2, PROFILE_QWEN_LOCAL)
    assert id(rung2) not in native._pixel_confirmed_candidates
    assert native._pixel_confirmed_candidates.get(id(rung1b)) is rung1b, (
        "another candidate's assessment consumed this one's confirmation"
    )


# ---------------------------------------------------------------------------
# P1-2B, narrow: the promotion
# ---------------------------------------------------------------------------


class _Accepting:
    def assess(self, output, provider):
        return AcceptDecision(accept=True, reason="leaf accepted", confidence=0.6)


def _promotion_for(tmp_path: Path, markdown: str, *, flags: dict):
    """Drive ``promote_unchecked_table_attempt`` the way the orchestrator does.

    The real heuristic leaf rejects this fixture's prose on BOTH arms (garbage ratio),
    which MASKS what the verifier did, so it is replaced by an accepting stub: the layer
    is then the only thing that can differ between the two readings.
    """
    from socr.pipeline.agentic import (
        PageDecision,
        ProviderAttempt,
        _promotion_inner_judge,
        promote_unchecked_table_attempt,
    )

    pipeline = _make_pipeline(_make_config())
    state = _make_state(_visible_layer_table_pdf(tmp_path))
    ps = state.pages[1]
    ps.has_tables = True
    ps.is_born_digital = True
    _flag(ps, **flags)
    judge = pipeline._build_page_judge(state)
    judge._ocr_image_fn = lambda _pix: ""  # nothing read the pixels
    judge._inner._inner = _Accepting()

    output = _out(markdown)
    judge.assess(output, PROFILE_QWEN_LOCAL)  # marks table_unverifiable
    attempt = ProviderAttempt(
        output=output,
        provider_id="qwen",
        engine="qwen",
        model="m",
        backend="b",
        cost_usd=0.0,
        accepted=False,
    )
    decision = PageDecision(page_num=1, final_output=output, accepted=False, attempts=[attempt])
    return promote_unchecked_table_attempt(decision, inner=_promotion_inner_judge(judge))


def test_the_promotion_runs_the_judge_without_the_layer(tmp_path: Path) -> None:
    """Round 4 (Argon P0) replaces round 3's expectation here, which was the bug.

    Round 3 made this path answer ``accept=False`` on every suspect page and asserted
    ``promoted is None``. That is CONTENT LOSS, not safety: a promotion that refuses
    sends ``settle_scanned_table_decision`` to ``_apply_scanned_table_floor``, which
    demotes ``best_output`` so the GH-90 branch rewrites the page with the D3 marker and
    the model's table is gone. The owner's ruling was "abstain, do not refuse without
    retaining", and round 3 broke the ruling it was implementing.

    The layer's VERDICT is skipped; the normal judge and the string-only structural gate
    still decide. Two pins, both measured in ``scratch/1205r4/``:

    a. Argon's repro -- ``flags={"positive": True}`` -- promotes the ``-350`` attempt
       again, as it did on ``4d14411b``. Nothing is lost.
    b. On a scan-failed page with an empty witness the layer cannot CHOOSE. Each reading
       is run alone and each is promoted keeping its own number. Round 3's asymmetry --
       the true ``-350`` refused, the layer's own ``350`` promoted -- is the control
       below, reproduced by disabling the abstention.
    """
    from socr.pipeline.agentic import _PromotionLayerAbstainJudge

    # (a) Argon's own repro shape.
    promoted = _promotion_for(tmp_path / "positive", _TRUE_MD, flags={"positive": True})
    assert promoted is not None, (
        "Argon P0: the promotion refused on a suspect page, so the scanned floor replaces "
        "the table and nothing is retained"
    )
    assert "-350" in promoted.output.text

    # (b) The layer picks nothing: each reading promoted on its own merits.
    for label, markdown, kept, dropped in (
        ("the true -350", _TRUE_MD, "-350", "| 350 |"),
        ("the layer's 350", _LAYER_MD, "| 350 |", "-350"),
    ):
        out = _promotion_for(
            tmp_path / f"scanfail-{label[-4:]}",
            markdown,
            flags={"positive": False, "scan_failed": True},
        )
        assert out is not None, f"{label}: the promotion refused and the table is lost"
        assert kept in out.output.text, f"{label}: the promoted bytes are not this reading's"
        assert dropped not in out.output.text, f"{label}: the layer substituted the other reading"

    # The control IS round 3's finding: with the abstention disabled the layer decides,
    # and it decides ASYMMETRICALLY -- refusing the true reading, promoting its own. If
    # this arm ever stops showing that, the fixture no longer reaches the layer and
    # everything above proves nothing.
    with patch.object(
        _PromotionLayerAbstainJudge, "_layer_may_not_decide", lambda self, output: False
    ):
        layer_true = _promotion_for(
            tmp_path / "ctl-true", _TRUE_MD, flags={"positive": False, "scan_failed": True}
        )
        layer_own = _promotion_for(
            tmp_path / "ctl-own", _LAYER_MD, flags={"positive": False, "scan_failed": True}
        )
    assert layer_true is None, (
        "control: the layer no longer refuses the true reading, so this fixture does not "
        "exercise the layer and the assertions above prove nothing"
    )
    assert layer_own is not None and "| 350 |" in layer_own.output.text


def test_the_live_chain_still_verifies_what_round_two_settled() -> None:
    """(c) The abstention is PROMOTION-ONLY, and these six are how we know.

    Argon's round-4 P0 names a different fix: delete this wrapper and widen the LIVE
    bypass to the union so ``verify_native_table`` is skipped there too. Measured
    (``scratch/1205r4/probe_blanket_bypass.py``): that turns these six red, #1156's two
    swap guards included -- on a page with no opaque cover the pixels ARE the layer's own
    ink, so dropping row-level verification lets a row swap through. This test fails the
    day someone widens it.
    """
    import tempfile

    from test_1205_table_chain_evidence_readers import (
        test_the_native_verifier_judge_no_longer_decides_on_a_suspect_layer as _decides,
    )
    from test_1211_r1_remaining_layer_readers import (
        test_the_judge_fails_closed_when_the_trust_source_raises as _raises,
        test_the_withheld_exact_pass_exit_still_runs_the_structural_gate as _exact,
    )
    from test_1211_r2_layer_decides_nothing import test_the_swap_guards_stay_green as _guards

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _guards()  # #1156's transparent cover + detection error
        _decides(root / "decides")
        _exact()
        _raises()


# ---------------------------------------------------------------------------
# Round 4 (Argon P1): a layer-caused record makes the page NOT TERMINAL
# ---------------------------------------------------------------------------
#
# Round 3 DROPPED these records at restore, and this section pinned that. Argon's round-4
# P1: dropping a record patches the page's history and leaves three consumers of the same
# verdict reading a story the events no longer tell --
#   1. ``_load_terminal_page`` still read a layer-caused ``table_withheld`` as a content
#      terminal and skipped re-OCR;
#   2. ``ps.table_ladder_disposition`` was restored ungated, so the page's #713 credential
#      was still refused ("adverse table-ladder disposition") -- round 3's own test hid
#      this by setting the disposition to None before asking;
#   3. ``_backfill_missing_table_ladder_terminals`` re-emitted a terminal at assemble for
#      the table whose event had just been dropped, marking the page untrusted again.
#
# So round 4 stops patching: such a page is not a ledger hit and is read again. The
# restore never runs for a page the gate refuses, which is why the single ``return None``
# replaces all of round 3's drop machinery.


def _sidecar_run(root: Path, events: list[dict], *, flagged_then: bool):
    """Write a real sidecar for a page WITHHELD, with ``events`` in its audit trail.

    ``flagged_then`` is the suspicion of the run that WROTE it. Default False, which is
    the only realistic provenance for a layer-caused record: within one socr version the
    gated producers cannot write one on a flagged page, so these sidecars come from a run
    before the page was flagged, or from a build without the gates.
    """
    from socr.core.audit_log import AuditEvent
    from socr.core.result import FailureMode

    pdf = _visible_layer_table_pdf(root)
    pipeline = _make_pipeline(_make_config())
    state = _make_state(pdf)
    ps = state.pages[1]
    _flag(ps, positive=flagged_then)
    ps.has_tables = True
    out = _out(_MD)
    ps.attempts.append(out)
    ps.best_output = out
    ps.table_ladder_disposition = FailureMode.TABLE_WITHHELD
    for ev in events:
        state.events.append(
            AuditEvent(
                page_num=1,
                kind=ev["kind"],
                engine="qwen",
                detail=ev.get("detail", "x"),
                data=dict(ev.get("data") or {}),
            )
        )
    out_dir = root / "out"
    pipeline._flush_page_fragment(state, 1, out.text, out_dir)
    pipeline._flush_page_sidecar(state, 1, out_dir)
    return pipeline, pdf, out_dir


def _resume_through_the_gate(pipeline, pdf: Path, out_dir: Path, *, flagged_now: bool):
    """Drive the REAL gate, then the restore only if the gate admitted the page.

    That ``if`` is production's own (both call sites in ``_phase_agentic`` /
    ``_phase_ocr`` read ``if resumed is not None``), and it is the whole reason one
    ``return None`` is a complete fix.
    """
    from socr.core.manifest import _credential_admission_refusal

    state = _make_state(pdf)
    ps = state.pages[1]
    ps.attempts.clear()
    ps.best_output = None
    ps.has_tables = True
    _flag(ps, positive=flagged_now)
    page = pipeline._load_terminal_page(state, 1, out_dir)
    if page is not None:
        pipeline._restore_terminal_page_state(state, 1, page, out_dir)
    return {
        "terminal": page is not None,
        "disposition": getattr(state.pages[1], "table_ladder_disposition", None),
        # NOT zeroed first: round 3's test set this to None, which is what hid path 2.
        "refusal": _credential_admission_refusal(state, state.pages[1], 1, page or _out(_MD)),
        "untrusted": build_tables_trust(state.handle.filename, state.events).untrusted_pages,
    }


def _event(kind: str, data: dict) -> dict:
    return {"page_num": 1, "kind": kind, "engine": "qwen", "detail": "x", "data": data}


#: ``(kind, the layer-derived cause, a cause of the SAME kind that is layer-free)``.
#: Argon's round-3 five, with the discriminator each one carries.
_FIVE = [
    ("table_ladder_withheld", {"reason": "native_contradiction"}, {"reason": "blind MISMATCH"}),
    (
        "table_ladder_withheld",
        {"reason": "sibling_of_contradicted"},
        {"reason": "blind MISMATCH"},
    ),
    (
        "table_ladder_unverified",
        {"cause": "binding_contradiction"},
        {"cause": "rung_unavailable"},
    ),
    ("table_ladder_unverified", {"cause": "abstained"}, {"cause": "no_rungs_configured"}),
    ("table_escalation_refused", {"gate": "none"}, None),
    ("table_escalation_timeout", {}, None),
    ("table_escalation_withheld", {}, None),
]


@pytest.mark.parametrize(
    ("kind", "layer_cause", "free_cause"),
    _FIVE,
    ids=[f"{k}-{sorted(c.values())[0] if c else 'any'}" for k, c, _f in _FIVE],
)
def test_a_layer_caused_record_makes_the_page_not_terminal(
    tmp_path: Path, kind: str, layer_cause: dict, free_cause: dict | None
) -> None:
    """All three of Argon's paths, asserted together, for each of the five causes.

    Each kind is in ``TABLE_DISTRUST_KINDS`` and in ``CREDENTIAL_BLOCKING_EVENT_KINDS``,
    so a resumed page carrying one is listed untrusted and loses its credential on the
    strength of a layer this run will not read.

    Two controls per arm, and they are what make the assertions mean anything: the same
    sidecar on a page that is NOT suspect now still skips, and a layer-FREE cause of the
    same kind still skips -- because reprocessing a page the pixels decided would throw
    away work for nothing.
    """
    assert kind in TABLE_DISTRUST_KINDS

    tag = f"{kind}-{sorted(layer_cause.values())[0] if layer_cause else 'any'}"
    pipeline, pdf, out_dir = _sidecar_run(
        tmp_path / f"case-{tag}", [_event(kind, layer_cause)], flagged_then=False
    )

    clean = _resume_through_the_gate(pipeline, pdf, out_dir, flagged_now=False)
    assert clean["terminal"] is True, (
        "control: this sidecar is not a ledger hit even on an admissible page, so the "
        "flagged arm below proves nothing"
    )
    assert clean["untrusted"] == [1] and clean["refusal"] != ""
    assert clean["disposition"] is not None

    flagged = _resume_through_the_gate(pipeline, pdf, out_dir, flagged_now=True)
    # 1. the terminal skip itself
    assert flagged["terminal"] is False, (
        "a page withheld by a layer-caused record was still a terminal ledger hit, so it "
        "is never re-read"
    )
    # 2. the disposition, and the credential refusal it drives
    assert flagged["disposition"] is None
    assert flagged["refusal"] == "", (
        "the restored disposition still refuses this page's acceptance credential"
    )
    # 3. the distrust
    assert flagged["untrusted"] == []

    if free_cause is not None:
        pipeline2, pdf2, dir2 = _sidecar_run(
            tmp_path / f"free-{tag}", [_event(kind, free_cause)], flagged_then=False
        )
        free = _resume_through_the_gate(pipeline2, pdf2, dir2, flagged_now=True)
        assert free["terminal"] is True, (
            f"a layer-FREE cause ({free_cause}) made the page reprocess too; that throws "
            "away a verdict the pixels reached"
        )


def test_a_pre_gate_sidecar_is_reprocessed_on_its_own_suspicion(tmp_path: Path) -> None:
    """The UNION, not just this run's analysis.

    A sidecar written by a build without these gates can carry a layer-caused record AND
    its own ``prior_ocr_layer_suspect``. The page must be re-read on the strength of that
    alone, even when this run's detection says the layer is fine -- the record was still
    authored by a layer that was not admissible when it spoke.
    """
    pipeline, pdf, out_dir = _sidecar_run(
        tmp_path / "pre-gate",
        [_event("table_ladder_withheld", {"reason": "native_contradiction"})],
        flagged_then=True,
    )
    result = _resume_through_the_gate(pipeline, pdf, out_dir, flagged_now=False)
    assert result["terminal"] is False
    assert result["refusal"] == "" and result["untrusted"] == []


def test_the_replay_drop_machinery_is_gone() -> None:
    """Round 3's drop is removed, not merely bypassed -- nothing dead is left behind.

    The classification survives as ``_sidecar_layer_caused_records``, which is what the
    gate asks; the two registries it reads are still the single source of that knowledge.
    """
    import inspect

    from socr.pipeline import orchestrator as orch

    src = inspect.getsource(orch.UnifiedPipeline._restore_terminal_page_state)
    assert "_ESCALATION_LAYER_GRADED_KINDS" not in src, (
        "the restore still drops layer-caused records; round 4 replaced that with the gate"
    )
    assert "_LAYER_CAUSED_LADDER_KINDS" not in src

    gate = inspect.getsource(orch.UnifiedPipeline._load_terminal_page)
    assert "_sidecar_layer_caused_records" in gate
    assert (
        orch._sidecar_layer_caused_records(
            {
                "audit_events": [
                    {"kind": "table_ladder_withheld", "data": {"reason": "blind MISMATCH"}}
                ]
            }
        )
        == []
    )
    assert orch._sidecar_layer_caused_records(
        {
            "audit_events": [
                {"kind": "table_ladder_withheld", "data": {"reason": "native_contradiction"}}
            ]
        }
    ) == ["table_ladder_withheld"]


# ---------------------------------------------------------------------------
# P1-1: the resume SKIP gate takes the union with the sidecar
# ---------------------------------------------------------------------------


def test_the_resume_skip_gate_takes_the_union_with_the_sidecar(tmp_path: Path) -> None:
    """Argon P1-1, and the mutant that found this test missing.

    ``_load_terminal_page`` runs BEFORE ``_restore_terminal_page_state`` folds the
    sidecar's suspicion onto the page, so reading the live flag alone left one way in:
    a page whose SIDECAR says suspect -- written by the very run that produced the
    binding record -- restored that record because THIS run's detection happened to say
    otherwise. Argon's repro: sidecar positive, ``live_flags={"positive": False}``.

    Round 2 pinned the other direction only (live flag set, sidecar clean), which is why
    reverting the union to the live flag left the suite green: mutant ``r3m1`` survived
    the round-3 set until this test existed.

    Observed by spying on ``_apply_binding_adjudication_meta`` rather than on the
    returned page, because the gate may well REFUSE a suspect page -- and the finding is
    that the record was applied anyway, on the way to that refusal.
    """
    pipeline = _make_pipeline(_make_config())

    def resume(*, sidecar_flagged: bool, live_flagged: bool):
        src = tmp_path / f"s{int(sidecar_flagged)}-l{int(live_flagged)}"
        pdf = _visible_layer_table_pdf(src)
        state = _make_state(pdf)
        ps = state.pages[1]
        _flag(ps, positive=sidecar_flagged)
        output = PageOutput(
            page_num=1, text=_MD, status=PageStatus.SUCCESS, engine="qwen", audit_passed=True
        )
        ps.attempts.append(output)
        ps.best_output = output
        ps.binding_adjudication = {"p1-t0": {"verdict": "contradicted"}}
        out_dir = src / "out"
        pipeline._flush_page_fragment(state, 1, output.text, out_dir)
        pipeline._flush_page_sidecar(state, 1, out_dir)
        meta = json.loads(
            (
                doc_dir_for(out_dir, relative_key(pdf, pdf.parent)) / "pages" / "00001.json"
            ).read_text(encoding="utf-8")
        )
        assert bool(meta.get("prior_ocr_layer_suspect", False)) is sidecar_flagged, (
            "the fixture did not write the suspicion this arm is about"
        )

        resumed = _make_state(pdf)
        resumed.pages[1].attempts.clear()
        resumed.pages[1].best_output = None
        _flag(resumed.pages[1], positive=live_flagged)

        calls: list[int] = []
        real = type(pipeline)._apply_binding_adjudication_meta

        def spy(self, state_, page_num, meta_):
            calls.append(page_num)
            return real(self, state_, page_num, meta_)

        with patch.object(type(pipeline), "_apply_binding_adjudication_meta", spy):
            pipeline._load_terminal_page(resumed, 1, out_dir)
        return calls

    assert resume(sidecar_flagged=False, live_flagged=False) == [1], (
        "control: the skip gate does not apply the binding record even with neither "
        "source of suspicion, so neither arm below proves anything"
    )
    assert resume(sidecar_flagged=True, live_flagged=False) == [], (
        "the skip gate applied the layer's binding record because only the SIDECAR said "
        "the layer was suspect -- the run that wrote the record"
    )
    assert resume(sidecar_flagged=False, live_flagged=True) == [], (
        "the skip gate applied the record on a page this run's own analysis flagged"
    )


def test_a_trust_source_that_raises_is_not_a_licence_to_let_the_layer_decide() -> None:
    """The promotion wrapper's fail-closed arm, which mutant r3m7 found unpinned.

    ``_native_evidence_suspect`` is a callable off the native judge, and an exception
    from it says nothing about the page. Reading that as "not suspect" would hand the
    promotion back to the layer on exactly the pages whose trust state could not be
    established, so it answers "suspect" and the layer is skipped.

    Round 4 note: since the abstention became a pass-through this direction costs no
    content either -- skipping the layer still lets the normal judge accept -- so the
    conservative answer and the retaining answer are the same one.
    """
    from socr.pipeline.agentic import _PromotionLayerAbstainJudge

    gate_calls: list[int] = []

    class _Native:
        """The collaborator's real interface: an inner chain and a structural gate."""

        def __init__(self, answer) -> None:
            self._answer = answer
            self._inner = _Accepting()

        def _native_evidence_suspect(self, page_num: int) -> bool:
            if isinstance(self._answer, Exception):
                raise self._answer
            return self._answer

        def _apply_structural_gate(self, decision, output, page_num, words, rules):
            gate_calls.append(page_num)
            assert words is None and rules is None, "the gate must be the string-only form"
            return decision

        def assess(self, output, provider):  # the arm the wrapper delegates to
            return AcceptDecision(accept=True, reason="native ran", confidence=0.9)

    output = _out(_MD)

    # Controls: with a working trust source the wrapper tracks it in both directions.
    assert _PromotionLayerAbstainJudge(_Native(False))._layer_may_not_decide(output) is False
    assert _PromotionLayerAbstainJudge(_Native(True))._layer_may_not_decide(output) is True

    raising = _PromotionLayerAbstainJudge(_Native(RuntimeError("no trust state")))
    assert raising._layer_may_not_decide(output) is True, (
        "a trust source that raised handed the promotion back to the layer"
    )

    # ... and the skip still RETAINS: the leaf decides and the structural gate runs.
    decision = raising.assess(output, PROFILE_QWEN_LOCAL)
    assert decision.accept is True, "the fail-closed arm refused, which loses the table"
    assert gate_calls == [1], "the string-only structural gate did not run"

    # The admissible arm delegates to the native judge itself, layer and all.
    admissible = _PromotionLayerAbstainJudge(_Native(False))
    assert admissible.assess(output, PROFILE_QWEN_LOCAL).reason == "native ran"


def test_no_native_judge_in_the_chain_means_no_wrapper() -> None:
    """The insertion condition, named here rather than left to #1125's deadline test.

    The skip works by calling PAST the native judge, so it is only sound where the
    wrapper sits directly on one. With no native judge the chain is left exactly as it
    was -- which is also what keeps ``peeled._inner`` the leaf for every caller that
    inspects the chain's shape.
    """
    from socr.pipeline.agentic import (
        SourceEvidenceTableJudge,
        _PromotionLayerAbstainJudge,
        _promotion_inner_judge,
    )

    leaf = _Accepting()
    peeled = _promotion_inner_judge(SourceEvidenceTableJudge(inner=leaf, get_fitz_page=None))
    assert peeled is leaf
    assert not isinstance(peeled, _PromotionLayerAbstainJudge)
