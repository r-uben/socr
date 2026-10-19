"""#1192: an ACCEPTED reading of an earlier-OCR page is not re-read on every resume.

#1156 Part A flags a page whose text layer is a PREVIOUS OCR run's reading. Such a page is
routed to a model; when the judge accepts that reading it ships, and the page is demoted to
WARNING carrying one clause, ``prior_ocr_layer=UNKNOWN`` -- marked, never checked. The
ledger admitted only a terminal SUCCESS page, so every flagged page was re-OCR'd on every
resume, for ever. ~141,000 such pages on the Fed corpus at a measured 3.39 GPU-s each.

The REJECTED half of the family (the fail-closed floor) is deliberately untouched and must
keep re-reading: a pending owner ruling, because another rung might yet read it.

Hermetic, per ``CLAUDE.md``: the engine registry, the agentic ladder, the page judge and
the judge model are all pinned, so nothing here consults ollama or a provider. Every
end-to-end claim is a DIFFERENCE between two arms run in the same process -- the accepted
arm against the rejected one, and a run against its own resume -- never an absolute outcome
measured on this machine.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import socr
from socr.core.result import FailureMode, PageStatus
from socr.pipeline.orchestrator import (
    UnifiedPipeline,
    _resume_prior_ocr_reading_was_accepted,
)

from test_gh1156_part_a_interim import _prior_ocr_pdf
from test_resume_figure_embed_idempotent import (
    _clean_pdf,
    _crash_after_page_one,
    _page_one,
    _run,
)


def test_loaded_source_is_this_checkout() -> None:
    """The mutation canary, in the house shape: a mutant copy that quietly imports the REAL
    tree proves nothing, and ``"socr" in socr.__file__`` is true of every tree there is."""
    assert Path(socr.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1] / "src")


# ---------------------------------------------------------------------------
# The end-to-end harness: a run, a crash, and a resume
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The proof at process(): what the acceptance, and only the acceptance, changes
# ---------------------------------------------------------------------------


def test_the_resume_re_reads_a_rejected_flagged_page_and_not_an_accepted_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The DIFFERENCE this ticket is: one fixture, one config, two judge answers.

    Both arms run a document, lose page 2 to a crash, and resume. The only thing that
    differs is whether the judge ACCEPTED page 1's reading. Asserted as a difference rather
    than as two absolute counts, because a provider-dependent outcome can legitimately
    differ in CI (``CLAUDE.md``): what cannot differ is that acceptance decides this.
    """
    accepted_resume: list[int] = []
    rejected_resume: list[int] = []

    for accept, resume_reads in ((True, accepted_resume), (False, rejected_resume)):
        tag = "acc" if accept else "rej"
        pdf = _prior_ocr_pdf(tmp_path / f"{tag}.pdf")
        out = tmp_path / f"out-{tag}"
        first_reads: list[int] = []
        _run(out, pdf, monkeypatch, accept=accept, pages_read=first_reads)
        assert 1 in first_reads, (
            f"[{tag}] no model read page 1 on the FIRST run, so this fixture cannot show "
            "anything about re-reading it"
        )
        _crash_after_page_one(out)
        _run(out, pdf, monkeypatch, accept=accept, pages_read=resume_reads)
        assert 2 in resume_reads, (
            f"[{tag}] the resume did not reprocess the page the crash lost, so it never "
            "entered the page loop and the ledger was not the gate under test"
        )

    assert 1 not in accepted_resume, (
        "an ACCEPTED reading of an earlier-OCR page was re-read on resume: "
        f"pages read = {accepted_resume}"
    )
    assert 1 in rejected_resume, (
        "a REJECTED earlier-OCR page was NOT re-read on resume. That half is a pending "
        "owner ruling and must keep reprocessing; without this arm the test cannot tell a "
        f"narrow exception from a blanket 'WARNING pages resume' rule. Read = {rejected_resume}"
    )


def test_the_resumed_accepted_page_is_byte_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Skip-and-KEEP: the page the resume did not read must come back unchanged.

    Both the shipped body and the whole sidecar, because the sidecar is what the acceptance
    meter and Part B read. A resumed page that is cheaper but different is not a resume.
    """
    pdf = _prior_ocr_pdf(tmp_path / "doc.pdf")
    out = tmp_path / "out"
    first: list[int] = []
    _run(out, pdf, monkeypatch, accept=True, pages_read=first)
    before_side, before_body = _page_one(out)
    assert before_side["winning_output"]["audit_passed"] is True, (
        "fixture premise: the reading was not accepted, so this is the wrong arm"
    )

    _crash_after_page_one(out)
    resume: list[int] = []
    _run(out, pdf, monkeypatch, accept=True, pages_read=resume)
    after_side, after_body = _page_one(out)

    assert 1 not in resume, "page 1 was re-read, so byte-identity here proves nothing"
    assert after_body == before_body, "the restored body changed"
    assert after_side == before_side, (
        "the restored sidecar changed; differing keys: "
        f"{sorted(k for k in set(before_side) | set(after_side) if before_side.get(k) != after_side.get(k))}"
    )


def test_a_fingerprint_change_re_reads_the_accepted_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The exception is scoped to the SAME config, and ``render_dpi`` is in the fingerprint.

    The same arm as the skip test, changing only the render DPI. Without this the new
    exception could not be told from one that ignores the run fingerprint, which would let
    a page shipped by one model's reading survive a change of model.
    """
    pdf = _prior_ocr_pdf(tmp_path / "doc.pdf")
    out = tmp_path / "out"
    first: list[int] = []
    _run(out, pdf, monkeypatch, accept=True, pages_read=first)
    _crash_after_page_one(out)

    same: list[int] = []
    _run(out, pdf, monkeypatch, accept=True, pages_read=same)
    assert 1 not in same, "the control arm re-read the page, so the next arm shows nothing"

    _crash_after_page_one(out)
    changed: list[int] = []
    _run(out, pdf, monkeypatch, accept=True, pages_read=changed, render_dpi=199)
    assert 1 in changed, f"a render-DPI change did not re-read the accepted page: {changed}"


def test_the_skipped_page_keeps_its_suspicion_its_layer_and_its_document_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What the skip must NOT buy: a page that comes back cleaner than it left.

    A resumed flagged page has to keep (a) the retained earlier layer, the one artefact
    Part B re-finalizes from, (b) its mark in the shipped bytes, and (c) the document's
    refusal to report SUCCESS over it. #1156 restored the first two for a resume it
    described as unreachable ("the page ledger admits only a terminal SUCCESS page and a
    flagged page is always WARNING"); this ticket makes that path live, so the three are
    pinned on the live path and not only at the restore seam.
    """
    from socr.core.result import DocumentStatus

    pdf = _prior_ocr_pdf(tmp_path / "doc.pdf")
    out = tmp_path / "out"
    first: list[int] = []
    first_result = _run(out, pdf, monkeypatch, accept=True, pages_read=first)
    _crash_after_page_one(out)
    resume: list[int] = []
    result = _run(out, pdf, monkeypatch, accept=True, pages_read=resume)
    side, body = _page_one(out)

    assert 1 not in resume, "page 1 was re-read, so none of this is about a skipped page"
    assert side["prior_ocr_layer_suspect"] is True, "the resumed page lost its suspicion"
    assert "3 Y," in (side.get("prior_ocr_layer_text") or ""), (
        "the resumed page lost the retained earlier layer, which is unrecoverable without "
        "a re-read -- the opposite of what skipping is for"
    )
    assert "prior_ocr_layer=UNKNOWN" in body, "the resumed page shipped without its mark"
    assert result.status is not DocumentStatus.SUCCESS, (
        "the resumed document reports SUCCESS over an unchecked earlier-OCR page"
    )
    assert result.status is first_result.status, (
        f"the resume changed the document status: {first_result.status} -> {result.status}"
    )


# ---------------------------------------------------------------------------
# The clause grid: what each condition of the exception is holding shut
# ---------------------------------------------------------------------------


def _without(key: str):
    return lambda meta: {k: v for k, v in meta.items() if k != key}


def _without_winning_key(key: str):
    def mutate(meta: dict) -> dict:
        return {
            **meta,
            "winning_output": {k: v for k, v in meta["winning_output"].items() if k != key},
        }

    return mutate


def _with_winning(**fields):
    def mutate(meta: dict) -> dict:
        return {**meta, "winning_output": {**meta["winning_output"], **fields}}

    return mutate


_A_SECOND_CLAUSE = {
    "check": "page_judge",
    "result": "UNKNOWN",
    "scope": "page",
    "reason": "the model output was never judged",
}


@pytest.mark.parametrize(
    ("mutate", "should_skip"),
    [
        pytest.param(lambda meta: meta, True, id="control_the_accepted_page_is_skipped"),
        # The page is not this family at all, or its sidecar is incomplete.
        pytest.param(_without("prior_ocr_layer_suspect"), False, id="no_suspicion_recorded"),
        pytest.param(_without("prior_ocr_layer_text"), False, id="the_retained_layer_is_gone"),
        pytest.param(
            lambda meta: {**meta, "prior_ocr_layer_text": ""}, False, id="retained_layer_empty"
        ),
        # The reading was not accepted. This is the clause that excludes the REJECTED half.
        pytest.param(_with_winning(audit_passed=False), False, id="the_reading_was_not_accepted"),
        pytest.param(_with_winning(audit_passed="yes"), False, id="audit_passed_is_not_a_bool"),
        # Not a model's reading of these pixels: both native lanes ship the suspect layer,
        # and a record naming NO reader is not evidence that a model read it (r-uben's
        # review of #1210, Grok P2: ``PageOutput`` defaults to ``engine=""`` beside
        # ``audit_passed=True``).
        pytest.param(_with_winning(engine="native"), False, id="a_native_winner"),
        pytest.param(_with_winning(engine="chart_asset"), False, id="a_chart_asset_winner"),
        pytest.param(_with_winning(engine=""), False, id="an_empty_engine"),
        pytest.param(_without_winning_key("engine"), False, id="no_engine_key_at_all"),
        # Demoted by something more, or something else, than this one mark.
        pytest.param(_with_winning(status="error"), False, id="a_worse_status"),
        pytest.param(
            _with_winning(failure_mode=FailureMode.TABLE_UNVERIFIED.value),
            False,
            id="another_failure_mode",
        ),
        pytest.param(
            lambda meta: {**meta, "table_ladder_incomplete": True},
            False,
            id="a_table_on_the_page_was_never_witnessed",
        ),
        pytest.param(
            lambda meta: {
                **meta,
                "verdict": {
                    **meta["verdict"],
                    "clauses": [*meta["verdict"]["clauses"], _A_SECOND_CLAUSE],
                },
            },
            False,
            id="a_second_clause_rides_along",
        ),
        pytest.param(
            lambda meta: {**meta, "verdict": {"page_num": 1, "clauses": []}},
            False,
            id="no_clause_at_all",
        ),
        # The RESULT is load-bearing, not decoration (r-uben's surviving mutant on #1210):
        # comparing the CHECK alone admits ``prior_ocr_layer=FAIL``, and the two mean
        # opposite things -- UNKNOWN is "no check decided", FAIL is "one did, and found
        # something wrong", which a later run may be able to put right. Same rule as
        # ``PERMANENTLY_UNRESOLVABLE_MODES``, where keying on the check alone was a bug.
        pytest.param(
            lambda meta: {
                **meta,
                "verdict": {
                    **meta["verdict"],
                    "clauses": [{**meta["verdict"]["clauses"][0], "result": "FAIL"}],
                },
            },
            False,
            id="the_clause_says_FAIL_not_UNKNOWN",
        ),
        pytest.param(
            lambda meta: {**meta, "verdict": {"clauses": [{"check": "not_a_check"}]}},
            False,
            id="the_verdict_does_not_parse",
        ),
    ],
)
def test_the_exception_is_held_shut_by_every_one_of_its_clauses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutate, should_skip: bool
) -> None:
    """One production sidecar, one key changed per row, through production's own gate.

    The POSITIVE CONTROL is the first row and it is the point: without it every other row
    passes against a gate that refuses everything, and a gate that never skips is not this
    fix, it is the absence of it.

    The rows are mutated SIDECARS rather than hand-built states, which is how #1145 pins
    the same gate (``test_t5_resume_rule_a``). A mutated sidecar is not a state this build
    produces on any page -- that is deliberate: these clauses are what stops a sidecar
    reaching this gate by some OTHER route (an older build, a hand edit, a class of page
    added later) from being skipped on evidence that does not hold.
    """
    pdf = _prior_ocr_pdf(tmp_path / "doc.pdf")
    out = tmp_path / "out"
    _run(out, pdf, monkeypatch, accept=True, pages_read=[])

    sidecar_path = next(iter(out.rglob("pages/00001.json")))
    baseline = json.loads(sidecar_path.read_text())
    assert baseline["winning_output"]["audit_passed"] is True
    assert baseline.get("prior_ocr_layer_suspect") is True
    assert baseline["winning_output"]["failure_mode"] == (
        FailureMode.PRIOR_OCR_LAYER_UNVERIFIED.value
    ), "fixture premise: the accepted page is not demoted by the earlier-OCR mark"

    _crash_after_page_one(out)
    sidecar_path.write_text(json.dumps(mutate(baseline)))

    resume: list[int] = []
    _run(out, pdf, monkeypatch, accept=True, pages_read=resume)
    assert (1 not in resume) is should_skip, (
        f"expected skip={should_skip}; pages read on the resume = {resume}"
    )


def _admissible() -> tuple[dict, dict]:
    """The smallest (meta, winning) pair the predicate accepts -- the positive control."""
    meta = {
        "prior_ocr_layer_suspect": True,
        "prior_ocr_layer_text": "an earlier OCR run's reading",
        "verdict": {
            "page_num": 1,
            "clauses": [
                {
                    "check": "prior_ocr_layer",
                    "result": "UNKNOWN",
                    "scope": "page",
                    "reason": "earlier OCR, not re-read",
                }
            ],
        },
    }
    winning = {
        "audit_passed": True,
        "engine": "qwen",
        "status": PageStatus.WARNING.value,
        "failure_mode": FailureMode.PRIOR_OCR_LAYER_UNVERIFIED.value,
    }
    return meta, winning


def test_the_predicate_refuses_anything_it_cannot_read() -> None:
    """Doubt is False, including shapes no sidecar has: the gate is fail-closed by itself.

    The last assertion is the positive control: without it every line here passes against a
    predicate hardwired to False, which is not this fix.
    """
    call = _resume_prior_ocr_reading_was_accepted
    assert call(None, {}, flagged_by_this_run=True) is False
    assert call({}, None, flagged_by_this_run=True) is False
    assert call("{}", "{}", flagged_by_this_run=True) is False
    assert call({}, {}, flagged_by_this_run=True) is False

    meta, winning = _admissible()
    assert (
        call({k: v for k, v in meta.items() if k != "verdict"}, winning, flagged_by_this_run=True)
        is False
    ), "a sidecar with NO verdict key was admitted"
    assert call(meta, winning, flagged_by_this_run=False) is False, (
        "the page was admitted although THIS run's analysis does not flag it"
    )
    assert call(meta, winning, flagged_by_this_run=True) is True, (
        "the control pair was refused, so every refusal above proves nothing"
    )


# ---------------------------------------------------------------------------
# The figure phase, which this exception made reachable for a new class of page
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flagged_now", [True, False])
def test_the_skip_needs_this_runs_analysis_to_agree_that_the_page_is_flagged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flagged_now: bool
) -> None:
    """A stale suspicion must not be locked in for ever by the sidecar that records it.

    ``prior_ocr_layer_suspect`` is a UNION that folds the detector's FAILED SCANS in beside
    its positive findings, and a failed scan is transient: a run that could not decide
    demoted the page; a run that CAN decide may find nothing wrong with it. The flag is
    also absent from the run fingerprint, so every other condition can match while the
    current analysis disagrees.

    Pinned as a DIFFERENCE on the analysis alone. Both arms resume the SAME admissible
    sidecar shape -- suspect, retained layer, accepted model reading, WARNING under the
    earlier-OCR mode, one clause -- written onto page 1 by hand. Only the DOCUMENT differs:
    one the detector flags, one it does not. The sibling guards in ``_load_terminal_page``
    (``minus_as_digit_suspect`` and the other two) make exactly this cross-check.
    """
    from socr.core.born_digital import BornDigitalDetector

    pdf = (
        _prior_ocr_pdf(tmp_path / "flagged.pdf")
        if flagged_now
        else _clean_pdf(tmp_path / "clean.pdf")
    )
    assert BornDigitalDetector().detect(pdf).pages[0].prior_ocr_layer is flagged_now, (
        "fixture premise: the detector does not read this document the way the arm needs"
    )

    out = tmp_path / "out"
    _run(out, pdf, monkeypatch, accept=True, pages_read=[])
    sidecar_path = next(iter(out.rglob("pages/00001.json")))
    meta = json.loads(sidecar_path.read_text())

    # One admissible shape, written onto BOTH arms, so nothing but the analysis can differ.
    admissible_meta, admissible_winning = _admissible()
    meta["prior_ocr_layer_suspect"] = True
    meta["prior_ocr_layer_text"] = admissible_meta["prior_ocr_layer_text"]
    meta["verdict"] = admissible_meta["verdict"]
    meta["winning_output"] |= admissible_winning
    meta.pop("table_ladder_incomplete", None)
    sidecar_path.write_text(json.dumps(meta))

    _crash_after_page_one(out)

    # The LEDGER's own decision is the observable here, not the engine count: on the clean
    # document no model reads any page at all (its pages take a native lane), so a page
    # count cannot tell a granted skip from a page nothing was going to read.
    granted: dict[int, bool] = {}
    real = UnifiedPipeline._load_terminal_page

    def spy(self, state, page_num, output_dir):
        got = real(self, state, page_num, output_dir)
        granted[page_num] = got is not None
        return got

    monkeypatch.setattr(UnifiedPipeline, "_load_terminal_page", spy)
    _run(out, pdf, monkeypatch, accept=True, pages_read=[])

    assert 1 in granted, "the ledger was never consulted for page 1, so nothing was tested"
    assert granted[1] is flagged_now, (
        "the resume decision did not follow THIS run's analysis: the detector says "
        f"flagged={flagged_now}, ledger granted the skip = {granted[1]}"
    )
