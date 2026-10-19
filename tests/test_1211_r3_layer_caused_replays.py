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


def test_the_promotion_does_not_pick_a_reading_with_the_layer(tmp_path: Path) -> None:
    """Argon P1-2B, narrowed to this path by the owner's ruling.

    ``promote_unchecked_table_attempt`` PEELS ``SourceEvidenceTableJudge``, so a candidate
    the pixel lane could not check reaches the native verifier unwitnessed -- and the layer
    then chose which reading was promoted. Measured before the fix, with an empty witness
    on a scan-failed page: the true ``-350`` was refused and the layer's own ``350`` was
    promoted.

    After: on a suspect page the layer neither promotes nor refuses. Nothing verified the
    table, so there is no acceptance and no promotion; the page keeps its floor. The clean
    arm is the control -- there the layer IS evidence and it still decides, which is what
    makes this about suspicion.
    """
    for label, markdown in (("the true -350", _TRUE_MD), ("the layer's 350", _LAYER_MD)):
        promoted = _promotion_for(
            tmp_path / f"suspect-{label[-4:]}",
            markdown,
            flags={"positive": False, "scan_failed": True},
        )
        assert promoted is None, f"{label}: the layer decided a promotion on a suspect page"

    # The control cannot be an admissible page: there the pixel lane either DEFERS (a
    # trusted born-digital page) or passes the layer-matching reading outright, so the
    # candidate never becomes a promotion candidate and nothing is promoted either way.
    # So the control disables the new abstention instead, which reproduces Argon's
    # finding inside this test and proves the fixture really does reach the promotion.
    from socr.pipeline.agentic import _PromotionLayerAbstainJudge

    with patch.object(
        _PromotionLayerAbstainJudge, "_layer_may_not_decide", lambda self, output: False
    ):
        unguarded = _promotion_for(
            tmp_path / "unguarded", _LAYER_MD, flags={"positive": False, "scan_failed": True}
        )
    assert unguarded is not None and "350" in unguarded.output.text, (
        "control: with the abstention disabled the layer still does not promote its own "
        "reading, so this fixture does not reach the promotion and the assertions above "
        "prove nothing"
    )
    assert "-350" not in unguarded.output.text


# ---------------------------------------------------------------------------
# P1-3: the five layer-derived replayed kinds
# ---------------------------------------------------------------------------


def _resume(tmp_path: Path, events: list[dict], *, flagged: bool):
    pipeline = _make_pipeline(_make_config())
    tmp_path.mkdir(parents=True, exist_ok=True)
    pdf = tmp_path / "doc.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(pdf)
    doc.close()
    state = _make_state(pdf)
    _flag(state.pages[1], positive=flagged)
    output_dir = tmp_path / "out"
    doc_dir = doc_dir_for(output_dir, relative_key(pdf, pdf.parent))
    (doc_dir / "pages").mkdir(parents=True, exist_ok=True)
    (doc_dir / "pages" / "00001.json").write_text(
        json.dumps({"prior_ocr_layer_suspect": flagged, "audit_events": events}),
        encoding="utf-8",
    )
    page_out = _out(_MD)
    pipeline._restore_terminal_page_state(state, 1, page_out, output_dir)
    return state, page_out


def _event(kind: str, data: dict) -> dict:
    return {"page_num": 1, "kind": kind, "engine": "qwen", "detail": "x", "data": data}


#: ``(kind, the layer-derived cause, a cause of the SAME kind that is layer-free)``.
#: Argon's five, with the discriminator each one carries.
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
def test_a_layer_caused_record_is_not_replayed_on_a_flagged_page(
    tmp_path: Path, kind: str, layer_cause: dict, free_cause: dict | None
) -> None:
    """Argon P1-3, and a correction to round 2's audit table, which called these five
    layer-independent.

    Each is in ``TABLE_DISTRUST_KINDS`` and in ``CREDENTIAL_BLOCKING_EVENT_KINDS``, so one
    replayed event lists the page untrusted and withholds its credential. Measured before
    the fix: all five did exactly that on a resumed flagged page.

    Two controls per arm. The clean page still replays, so the gate is about suspicion;
    and where the kind has a layer-FREE cause (a blind MISMATCH, a rung outage, no rungs
    configured) that cause still replays on the flagged page, because dropping it would
    withhold a page the pixels decided.
    """
    assert kind in TABLE_DISTRUST_KINDS

    tag = f"{kind}-{sorted(layer_cause.values())[0] if layer_cause else 'any'}"
    clean_state, _o = _resume(tmp_path / f"clean-{tag}", [_event(kind, layer_cause)], flagged=False)
    assert kind in {e.kind for e in clean_state.events}, (
        "control: the kind is not replayed even on an admissible page"
    )

    flagged_state, page_out = _resume(
        tmp_path / f"flagged-{tag}", [_event(kind, layer_cause)], flagged=True
    )
    assert kind not in {e.kind for e in flagged_state.events}, (
        "a layer-caused record was replayed onto a flagged page"
    )
    assert (
        build_tables_trust(flagged_state.handle.filename, flagged_state.events).untrusted_pages
        == []
    )
    ps = flagged_state.pages[1]
    ps.table_ladder_disposition = None
    assert _credential_admission_refusal(flagged_state, ps, 1, page_out) == ""

    if free_cause is not None:
        free_state, _o2 = _resume(
            tmp_path / f"free-{tag}", [_event(kind, free_cause)], flagged=True
        )
        assert kind in {e.kind for e in free_state.events}, (
            f"a layer-FREE cause ({free_cause}) was dropped too; that withholds a page "
            "the pixels decided"
        )


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


def test_a_trust_source_that_raises_is_not_a_licence_to_promote() -> None:
    """The promotion wrapper's fail-closed arm, which a mutant found unpinned.

    ``_native_evidence_suspect`` is a callable off the native judge, and an exception
    from it says nothing about the page. Returning "not suspect" there would let the
    layer promote its own reading on exactly the pages whose trust state could not be
    established -- so it answers "suspect" instead. Mutant ``r3m7`` flips it and this
    test is what dies.
    """
    from socr.pipeline.agentic import _PromotionLayerAbstainJudge

    class _Native:
        def __init__(self, answer) -> None:
            self._answer = answer

        def _native_evidence_suspect(self, page_num: int) -> bool:
            if isinstance(self._answer, Exception):
                raise self._answer
            return self._answer

    output = _out(_MD)

    # Controls first: with a working trust source the wrapper tracks it in both
    # directions, so the raising arm below is about the exception and nothing else.
    assert (
        _PromotionLayerAbstainJudge(_Accepting(), _Native(False))._layer_may_not_decide(output)
        is False
    )
    assert (
        _PromotionLayerAbstainJudge(_Accepting(), _Native(True))._layer_may_not_decide(output)
        is True
    )

    raising = _PromotionLayerAbstainJudge(_Accepting(), _Native(RuntimeError("no trust state")))
    assert raising._layer_may_not_decide(output) is True, (
        "a trust source that raised was read as a licence to let the layer promote"
    )
    assert raising.assess(output, PROFILE_QWEN_LOCAL).accept is False

    # No native judge at all: the wrapper is never inserted in that case, and if it
    # somehow were it must not refuse every table on every page.
    assert _PromotionLayerAbstainJudge(_Accepting(), None)._layer_may_not_decide(output) is False
