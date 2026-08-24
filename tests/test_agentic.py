"""Cost-aware agentic per-page router tests.

The router is pure given injected (run_provider, judge), so these use stubs —
no engines, no models. They pin the cost-effective behavior: cheapest-first,
escalate only on rejection, bound by max_attempts, keep best on total failure.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import fitz
import pytest

from socr.core.audit_log import AuditEvent
from socr.core.config import EngineType
from socr.core.providers import provider_ladder
from socr.core.result import PageOutput, PageStatus
from socr.pipeline.agentic import (
    DEFAULT_PROVIDER_TIMEOUTS,
    AcceptDecision,
    HeuristicPageJudge,
    NativeTableVerifierJudge,
    VLMPageJudge,
    route_page,
)

LADDER = provider_ladder(
    {EngineType.GLM, EngineType.GEMINI, EngineType.MISTRAL},
    include_ineligible=True,
)
# -> [GLM(free), GEMINI(0.0002), MISTRAL(0.001)]


def _run(audit=None, conf=None, text=None):
    audit = audit or {}
    conf = conf or {}
    text = text or {}

    def run(engine: EngineType, page_num: int) -> PageOutput:
        return PageOutput(
            page_num=page_num,
            text=text.get(engine, f"ocr from {engine.value}"),
            status=PageStatus.SUCCESS,
            engine=engine.value,
            audit_passed=audit.get(engine, True),
            confidence=conf.get(engine, 0.5),
        )

    return run


class _StubJudge:
    """Accepts only outputs whose engine is in `accept`."""

    def __init__(self, accept):
        self.accept = set(accept)

    def assess(self, output, provider):
        return AcceptDecision(accept=provider.engine in self.accept, reason="stub")


def test_accepts_cheapest_no_escalation():
    d = route_page(1, LADDER, _run(), _StubJudge({EngineType.GLM}))
    assert d.accepted
    assert d.winning_engine == "glm"
    assert len(d.attempts) == 1
    assert d.escalations == 0
    assert d.total_cost_usd == 0.0


def test_escalates_to_next_when_cheap_rejected():
    d = route_page(2, LADDER, _run(), _StubJudge({EngineType.GEMINI}))
    assert d.accepted
    assert d.winning_engine == "gemini"
    assert [a.engine for a in d.attempts] == [EngineType.GLM, EngineType.GEMINI]
    assert d.escalations == 1
    assert d.total_cost_usd == pytest.approx(0.0002)  # free GLM + gemini


def test_climbs_full_ladder_then_accepts_priciest():
    d = route_page(3, LADDER, _run(), _StubJudge({EngineType.MISTRAL}))
    assert d.accepted and d.winning_engine == "mistral"
    assert d.escalations == 2
    assert d.total_cost_usd == pytest.approx(0.0012)


def test_all_rejected_keeps_best_audit_passed():
    # None accepted -> best-effort prefers the audit_passed attempt.
    judge = _StubJudge(set())
    run = _run(audit={EngineType.GLM: False, EngineType.GEMINI: True, EngineType.MISTRAL: False})
    d = route_page(4, LADDER, run, judge)
    assert not d.accepted
    assert len(d.attempts) == 3  # tried everything
    assert d.final_output.engine == "gemini"  # the only audit_passed one


def test_all_rejected_uses_confidence_then_words():
    judge = _StubJudge(set())
    run = _run(
        audit={e: False for e in (EngineType.GLM, EngineType.GEMINI, EngineType.MISTRAL)},
        conf={EngineType.GLM: 0.1, EngineType.GEMINI: 0.9, EngineType.MISTRAL: 0.3},
    )
    d = route_page(5, LADDER, run, judge)
    assert d.final_output.engine == "gemini"  # highest confidence


def test_max_attempts_bounds_cost():
    # Only the cheapest provider is tried even though it's rejected.
    d = route_page(6, LADDER, _run(), _StubJudge({EngineType.MISTRAL}), max_attempts=1)
    assert len(d.attempts) == 1
    assert d.attempts[0].engine == EngineType.GLM
    assert not d.accepted


def test_provider_exception_is_recorded_and_skipped():
    def run(engine, page_num):
        if engine == EngineType.GLM:
            raise RuntimeError("ollama down")
        return PageOutput(
            page_num=page_num,
            text="cloud ok",
            engine=engine.value,
            status=PageStatus.SUCCESS,
            audit_passed=True,
        )

    d = route_page(7, LADDER, run, _StubJudge({EngineType.GEMINI}))
    assert d.accepted and d.winning_engine == "gemini"
    assert d.attempts[0].engine == EngineType.GLM
    assert d.attempts[0].output.status == PageStatus.ERROR  # recorded the failure
    assert d.attempts[1].engine == EngineType.GEMINI


def test_empty_ladder_yields_error():
    d = route_page(8, [], _run(), _StubJudge(set()))
    assert not d.accepted
    assert d.final_output.status == PageStatus.ERROR
    assert d.attempts == []


# --- judge adapters -------------------------------------------------------


def test_heuristic_page_judge_accepts_good_rejects_empty():
    from socr.audit.heuristics import HeuristicsChecker

    judge = HeuristicPageJudge(HeuristicsChecker(min_word_count=10))
    good = PageOutput(page_num=1, text=" ".join(["word"] * 50), status=PageStatus.SUCCESS)
    empty = PageOutput(page_num=1, text="", status=PageStatus.ERROR)
    prof = LADDER[0]
    assert judge.assess(good, prof).accept
    assert not judge.assess(empty, prof).accept


def test_vlm_page_judge_uses_verdict():
    from socr.judge.judge import JudgeVerdict

    class _VJ:
        def __init__(self, faithful):
            self.faithful = faithful

        def judge(self, image_path, ocr_text):
            issues = [] if self.faithful else ["garbled"]
            return JudgeVerdict(faithful=self.faithful, confidence=0.8, issues=issues)

    rendered = []
    judge = VLMPageJudge(_VJ(True), render_image=lambda pn: rendered.append(pn) or f"/img/{pn}.png")
    out = PageOutput(page_num=3, text="some text", status=PageStatus.SUCCESS)
    dec = judge.assess(out, LADDER[0])
    assert dec.accept and rendered == [3]  # it rendered the right page

    judge_bad = VLMPageJudge(_VJ(False), render_image=lambda pn: f"/img/{pn}.png")
    assert not judge_bad.assess(out, LADDER[0]).accept


def test_provider_attempt_records_id_model_backend():
    d = route_page(1, LADDER, _run(), _StubJudge({EngineType.GLM}))
    assert d.accepted
    attempt = d.attempts[0]
    assert attempt.provider_id != ""
    assert attempt.model != ""
    assert attempt.backend != ""


# --- provider timeout (TICKET-C1) -----------------------------------------


def test_slow_provider_timeout_escalates():
    """A provider that exceeds its timeout must escalate, not hang the batch.

    Scenario: GLM is given a 50 ms timeout but sleeps 10 s.
    GEMINI is instant and accepted.  Expected:
      (1) GLM attempt recorded with ERROR status,
      (2) GEMINI was tried and accepted,
      (3) total elapsed < 2 s.
    """

    def run(engine: EngineType, page_num: int) -> PageOutput:
        if engine == EngineType.GLM:
            time.sleep(10)  # simulate a stall
        return PageOutput(
            page_num=page_num,
            text=f"ok from {engine.value}",
            status=PageStatus.SUCCESS,
            engine=engine.value,
            audit_passed=True,
        )

    t0 = time.monotonic()
    d = route_page(
        1,
        LADDER,
        run,
        _StubJudge({EngineType.GEMINI}),
        provider_timeout={EngineType.GLM: 0.05},
    )
    elapsed = time.monotonic() - t0

    # (1) GLM attempt must be recorded as ERROR
    glm_att = next(a for a in d.attempts if a.engine == EngineType.GLM)
    assert glm_att.output.status == PageStatus.ERROR

    # (2) GEMINI must have been tried and accepted
    assert d.accepted
    assert d.winning_engine == "gemini"

    # (3) wall-clock must be well under 2 s (GLM did NOT sleep 10 s)
    assert elapsed < 2.0, f"route_page took {elapsed:.2f}s — GLM stall not intercepted"


def test_no_timeout_runs_normally():
    """Backward compat: omitting provider_timeout must not change routing behaviour."""
    d = route_page(1, LADDER, _run(), _StubJudge({EngineType.GLM}))
    assert d.accepted
    assert d.winning_engine == "glm"
    assert len(d.attempts) == 1


def test_default_provider_timeouts_keys():
    """DEFAULT_PROVIDER_TIMEOUTS must cover the two primary agentic rungs.

    Tests that QWEN and GEMINI keys are present.  Deliberately does NOT assert
    specific float values because the calibration is tunable — the important
    invariant is that the named constant exists and has the right shape.
    """
    assert EngineType.QWEN in DEFAULT_PROVIDER_TIMEOUTS
    assert EngineType.GEMINI in DEFAULT_PROVIDER_TIMEOUTS
    # Both values must be positive finite floats
    for engine, seconds in DEFAULT_PROVIDER_TIMEOUTS.items():
        assert isinstance(seconds, float), f"{engine}: expected float, got {type(seconds)}"
        assert seconds > 0, f"{engine}: timeout must be positive"


# --------------------------------------------------------------------------
# GH-200: the winner-side structural/header escalation gate.
#
# TR-3 (native_verifier) proves the numbers are right; it is blind by
# construction to header loss, detached labels, and star-only row deletion
# (docs/log/2026-08-15_tr3-hand-judgement.md, 4/4 damaged pages). These pin
# NativeTableVerifierJudge._apply_structural_gate, which runs
# structure_check.table_output_defect on whatever text is ABOUT TO SHIP —
# native or (via the same code path) VLM markdown — on every accepting exit
# from assess(), not only the EXACT_PASS short-circuit.
#
# Hermetic: synthetic fitz pages, no ollama/GPU/provider.
# --------------------------------------------------------------------------


_GAP = 60.0  # must exceed the well-separated-lane threshold; see test_native_table_verifier.py


def _md(header: list[str], rows: list[list[str]]) -> str:
    sep = "| " + " | ".join(["---"] * len(header)) + " |"
    lines = ["| " + " | ".join(header) + " |", sep]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _table_page_with_header(header_tokens: list[str] | None, data_values: list[str]) -> fitz.Page:
    """Label + one data row of numeric values; header words (if any) carry a
    '%' marker so header_repair._is_table_header_row recognises the band."""
    doc = fitz.open()
    page = doc.new_page(width=700, height=400)
    xs = [100.0 + i * _GAP for i in range(len(data_values))]
    if header_tokens is not None:
        for x, tok in zip(xs, header_tokens):
            page.insert_text((x, 100.0), tok, fontsize=9)
    page.insert_text((20.0, 140.0), "row1", fontsize=9)
    for x, val in zip(xs, data_values):
        page.insert_text((x, 140.0), val, fontsize=9)
    return page


def _stub_inner(accept: bool = True) -> MagicMock:
    inner = MagicMock()
    inner.assess.return_value = AcceptDecision(accept=accept, reason="inner judge stub")
    return inner


class TestStructuralEscalationGate:
    @pytest.mark.xfail(
        strict=True,
        reason=(
            "GH-200: the header-attribution reject disjunct is parked in "
            "table_output_defect. The REQUIREMENT is unchanged -- a destroyed "
            "header must be rejected -- but every predicate tried so far also "
            "returns HARD on byte-perfect correct tables (significance-star and "
            "n.a. rows), and a false reject destroys good output. This flips to "
            "XPASS the moment a sound predicate is wired back in."
        ),
    )
    def test_exact_pass_with_destroyed_header_is_rejected(self):
        """Direct regression for the agentic.py EXACT_PASS hole: a
        numerically-perfect, header-destroyed table must still reach
        EXACT_PASS in the verifier AND be rejected by the structural gate
        (never shipped at confidence 1.0 by the bare TR-3 accept)."""
        page = _table_page_with_header(["Low%", "Mid%", "High%"], ["0.1", "0.2", "0.3"])
        output_text = _md(["Currency", "", "", ""], [["row1", "0.1", "0.2", "0.3"]])
        output = PageOutput(page_num=1, text=output_text, status=PageStatus.SUCCESS, confidence=0.9)

        events: list[AuditEvent] = []
        inner = _stub_inner(accept=True)
        judge = NativeTableVerifierJudge(
            inner=inner,
            get_fitz_page=lambda pn: page,
            is_table_page=lambda pn: True,
            record_event=events.append,
        )
        decision = judge.assess(output, MagicMock())

        assert any(e.kind == "native_table_verifier_exact_pass" for e in events), (
            "verifier must reach EXACT_PASS on this fixture"
        )
        inner.assess.assert_not_called()  # EXACT_PASS never calls the inner judge
        assert decision.accept is False
        assert decision.reason.startswith("table_structure_failed")
        assert any(e.kind == "table_structure_failed" for e in events)

    @pytest.mark.parametrize(
        "vr_kwargs",
        [
            pytest.param(
                {"state": "EXACT_PASS", "hard_fail": False, "warn": False, "row_count_warn": False},
                id="exact_pass",
            ),
            pytest.param(
                {"state": "AMBIGUOUS", "hard_fail": False, "warn": True, "row_count_warn": False},
                id="warn_delegate",
            ),
            pytest.param(
                {"state": "AMBIGUOUS", "hard_fail": False, "warn": False, "row_count_warn": True},
                id="no_issue_delegate",
            ),
        ],
    )
    def test_structural_gate_covers_all_three_accept_paths(self, monkeypatch, vr_kwargs):
        """Guards against patching only the EXACT_PASS branch: parametrised
        over agentic.py's three accepting exits (:592 warn-delegate,
        :608 EXACT_PASS, :615 no-issue-delegate). Same stub-accepting inner
        judge, same defective (ragged) text; all three must reject."""
        from socr.tables.native_verifier import VerifierResult

        vr = VerifierResult(native_lane_count=3, output_col_count=3, reason="stub", **vr_kwargs)
        monkeypatch.setattr(
            "socr.tables.native_verifier.verify_native_table", lambda page, text: vr
        )

        # Ragged grid: one body row 2 cells, one 3 cells -> B1 shape gate fires.
        defective_text = "| a | b | c |\n| --- | --- | --- |\n| 1 | 2 |\n| 3 | 4 | 5 |"
        output = PageOutput(
            page_num=2, text=defective_text, status=PageStatus.SUCCESS, confidence=0.9
        )
        blank_page = fitz.open().new_page(width=200, height=200)

        inner = _stub_inner(accept=True)
        judge = NativeTableVerifierJudge(
            inner=inner,
            get_fitz_page=lambda pn: blank_page,
            is_table_page=lambda pn: True,
            record_event=lambda evt: None,
        )
        decision = judge.assess(output, MagicMock())
        assert decision.accept is False
        assert decision.reason == "table_structure_failed: grid_shape"

    def test_verifier_exception_rejects_without_consulting_inner(self, monkeypatch):
        """Geometry raises inside the try block -> fail-closed rejection.

        GH-200 originally let the exception path delegate to the inner judge and
        leaned on the words-free grid-shape term to catch a ragged candidate.
        GH-162 supersedes that: a raised verifier ran NO deterministic term, so
        there is nothing to accept on and the inner judge is not consulted at
        all. This ragged candidate is still rejected -- the point of the GH-200
        case -- now for the stronger reason.
        """

        def _raise(*a, **kw):
            raise RuntimeError("boom")

        monkeypatch.setattr("socr.tables.native_verifier.verify_native_table", _raise)

        defective_text = "| a | b | c |\n| --- | --- | --- |\n| 1 | 2 |\n| 3 | 4 | 5 |"
        output = PageOutput(
            page_num=3, text=defective_text, status=PageStatus.SUCCESS, confidence=0.9
        )
        inner = _stub_inner(accept=True)
        judge = NativeTableVerifierJudge(
            inner=inner,
            get_fitz_page=lambda pn: fitz.open().new_page(width=200, height=200),
            is_table_page=lambda pn: True,
            record_event=lambda evt: None,
        )
        decision = judge.assess(output, MagicMock())
        inner.assess.assert_not_called()  # GH-162: no accept path on a raised verifier
        assert decision.accept is False
        assert decision.confidence == 0.0
        assert decision.reason.startswith("table_verifier_error:")

    def test_ragged_candidate_with_clean_multiset_rejects(self):
        """Disjunction control (a): B1's own shape gate must reject even
        when TR-3's numeric multiset is clean -- neither term subsumes the
        other."""
        page = _table_page_with_header(None, ["0.1", "0.2", "0.3", "0.4"])
        # Emitted grid is ragged: row 1 has 3 cells, row 2 has 4.
        output_text = (
            "| label | c1 | c2 |\n| --- | --- | --- |\n"
            "| row1 | 0.1 | 0.2 |\n| row1b | 0.3 | 0.4 | extra |"
        )
        output = PageOutput(page_num=4, text=output_text, status=PageStatus.SUCCESS, confidence=0.9)
        inner = _stub_inner(accept=True)
        judge = NativeTableVerifierJudge(
            inner=inner,
            get_fitz_page=lambda pn: page,
            is_table_page=lambda pn: True,
            record_event=lambda evt: None,
        )
        decision = judge.assess(output, MagicMock())
        assert decision.accept is False
        assert decision.reason == "table_structure_failed: grid_shape"

    def test_rectangular_intact_header_but_multiset_mismatch_rejects_via_tr3(self):
        """Disjunction control (b): a rectangular candidate with an intact
        header but a numeric-multiset mismatch must still be rejected -- by
        TR-3's own hard-fail, independent of the structural gate."""
        page = _table_page_with_header(["Low%", "Mid%", "High%"], ["0.1", "0.2", "0.3"])
        # Output drops 0.3 and invents 0.9 -- multiset mismatch, header intact.
        output_text = _md(["Currency", "Low%", "Mid%", "High%"], [["row1", "0.1", "0.2", "0.9"]])
        output = PageOutput(page_num=5, text=output_text, status=PageStatus.SUCCESS, confidence=0.9)
        inner = _stub_inner(accept=True)
        judge = NativeTableVerifierJudge(
            inner=inner,
            get_fitz_page=lambda pn: page,
            is_table_page=lambda pn: True,
            record_event=lambda evt: None,
        )
        decision = judge.assess(output, MagicMock())
        assert decision.accept is False
        assert decision.reason.startswith("native_table_verifier:")
