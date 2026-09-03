"""TICKET-B1 (GH-353): the table judge ladder gate — the single choke point.

Assembles B0 (witnesses) -> A4 (ladder state machine, A2/A3 rungs injected) ->
A1 (audit-event kinds) -> ``PageState.table_ladder_disposition`` (C3's manifest
guard / C2's document aggregation read this, never ``best_output`` in place).

Hermetic throughout: CI has no ollama and no ``gemini`` binary, so every rung
here is an injected fake (``_build_table_judge_rungs`` is overridden), never
A2's real ``build_ollama_rung`` / A3's real ``make_gemini_rung`` reaching a
socket or a subprocess. ``_available_engines_for_agentic`` is patched so the
OCR provider ladder is non-empty regardless of the host's own ollama/gemini
install, and ``_resolve_judge_model`` is patched to ``""`` so
``_run_fingerprint``'s judge-model resolution never probes a live daemon.

Per the #253/#257 trap, the flag-on/flag-off comparison test never pins an
absolute outcome tuple measured on one machine — it runs both configurations
in the same test and asserts the DIFFERENCE the flag makes.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import fitz
import httpx
import pytest

from socr.core.audit_log import AuditEvent
from socr.core.config import EngineType, PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.providers import PROFILE_QWEN_LOCAL
from socr.core.result import DocumentStatus, FailureMode, PageOutput, PageStatus
from socr.core.state import DocumentState
from socr.judge.table_ladder import TableLadderOutcome
from socr.judge.table_verdict import (
    TABLE_LADDER_ACCEPTED_KIND,
    TABLE_LADDER_REJECTED_KIND,
    TABLE_LADDER_UNVERIFIED_KIND,
    Finding,
    FindingCode,
    RungResult,
    TableJudgeVerdict,
)
from socr.pipeline.orchestrator import UnifiedPipeline

# ---------------------------------------------------------------------------
# Fixtures: minimal PDFs with a known table-locator shape (mirrors B0's own
# fixture builder — see tests/test_table_witness.py).
# ---------------------------------------------------------------------------

_TABLE_MD = (
    "| c0 | c1 | c2 | c3 |\n"
    "| --- | --- | --- | --- |\n"
    "| 10 | 11 | 12 | 13 |\n"
    "| 20 | 21 | 22 | 23 |\n"
    "| 30 | 31 | 32 | 33 |\n"
)
_TWO_TABLE_MD = _TABLE_MD + "\nprose between tables\n\n" + _TABLE_MD


def _ruled_pdf(tmp_path: Path, name: str = "doc.pdf") -> Path:
    """One page, one ruled (fully boxed) table -> exactly 1 located box."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    page = doc.new_page()
    cols = [100, 220, 300, 380]
    rows = [100 + i * 22 for i in range(4)]
    for r, y in enumerate(rows):
        for c, x in enumerate(cols):
            page.insert_text((x + 4, y + 12), f"{r}{c}", fontsize=9)
    for yy in rows:
        page.draw_line((100, yy), (460, yy))
    for xx in cols + [460]:
        page.draw_line((xx, rows[0]), (xx, rows[-1]))
    pdf_path = tmp_path / name
    doc.save(pdf_path)
    doc.close()
    return pdf_path


def _borderless_pdf(tmp_path: Path, name: str = "doc.pdf") -> Path:
    """One page, table-shaped text with NO ruling lines -> 0 located boxes."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    page = doc.new_page()
    cols = [100, 220, 300, 380]
    rows = [100 + i * 22 for i in range(4)]
    for r, y in enumerate(rows):
        for c, x in enumerate(cols):
            page.insert_text((x + 4, y + 12), f"{r}{c}", fontsize=9)
    pdf_path = tmp_path / name
    doc.save(pdf_path)
    doc.close()
    return pdf_path


# ---------------------------------------------------------------------------
# Fake rung helpers
# ---------------------------------------------------------------------------


def _pass_verdict(confidence: str = "high") -> TableJudgeVerdict:
    return TableJudgeVerdict(verdict="PASS", confidence=confidence, findings=[])


def _fail_verdict() -> TableJudgeVerdict:
    return TableJudgeVerdict(
        verdict="FAIL",
        confidence="high",
        findings=[Finding(code=FindingCode.FABRICATED_VALUE, where="r1c1", detail="not in crop")],
    )


class _QueueRung:
    """A ``RungCallable`` that returns pre-baked results in order and records
    every call (crop_path, markdown, prior_findings) for assertion."""

    def __init__(self, results: list[RungResult], rung_id: str = "fake") -> None:
        self._results = list(results)
        self.rung_id = rung_id
        self.calls: list[tuple] = []

    def __call__(self, crop_path, markdown, prior_findings):
        self.calls.append((crop_path, markdown, prior_findings))
        if not self._results:
            raise AssertionError(f"{self.rung_id} called more times than results provided")
        return self._results.pop(0)


def _accept_rung() -> _QueueRung:
    return _QueueRung([RungResult(rung="fake1", ok=True, verdict=_pass_verdict("high"))])


def _reject_rung() -> _QueueRung:
    """Single-rung ladder: FAIL at the last rung exhausts to REJECTED."""
    return _QueueRung([RungResult(rung="fake1", ok=True, verdict=_fail_verdict())])


def _not_s1_rung(error: str = "simulated infra failure") -> _QueueRung:
    """Single-rung ladder: ¬S1 at the last rung exhausts to UNVERIFIED.

    Stands in for missing-binary / HTTP error / parser error alike — per the
    A1 contract every one of those collapses to the same ``RungResult(ok=False)``
    shape, and the gate must treat them uniformly.
    """
    return _QueueRung([RungResult(rung="fake1", ok=False, error=error)])


# ---------------------------------------------------------------------------
# Pipeline / state helpers
# ---------------------------------------------------------------------------


def _make_config(**overrides) -> PipelineConfig:
    kwargs = dict(
        primary_engine=EngineType.QWEN,
        agentic=True,
        judge_backend="heuristic",
        enabled_engines=[EngineType.QWEN],
        quiet=True,
        save_figures=False,
        write_manifest=False,
        table_judge_ladder=True,
    )
    kwargs.update(overrides)
    return PipelineConfig(**kwargs)


def _make_pipeline(config: PipelineConfig | None = None) -> UnifiedPipeline:
    return UnifiedPipeline(config or _make_config())


def _make_state(pdf_path: Path, page_count: int = 1) -> DocumentState:
    with patch.object(DocumentHandle, "__post_init__", lambda self: None):
        handle = DocumentHandle(path=pdf_path, page_count=page_count)
    return DocumentState(handle=handle)


def _bo(text: str, engine: str = "qwen") -> PageOutput:
    return PageOutput(
        page_num=1,
        text=text,
        status=PageStatus.SUCCESS,
        engine=engine,
        audit_passed=True,
    )


def _events_of_kind(state: DocumentState, kind: str) -> list[AuditEvent]:
    return [e for e in state.events if e.kind == kind]


# ---------------------------------------------------------------------------
# _build_table_judge_rungs
# ---------------------------------------------------------------------------


class TestBuildRungs:
    def test_flag_off_returns_empty(self, tmp_path: Path) -> None:
        pipeline = _make_pipeline(_make_config(table_judge_ladder=False))
        assert pipeline._build_table_judge_rungs() == []

    def test_strict_local_returns_empty_even_with_flag_on(self, tmp_path: Path) -> None:
        pipeline = _make_pipeline(_make_config(table_judge_ladder=True, strict_local=True))
        assert pipeline._build_table_judge_rungs() == []

    def test_flag_on_not_strict_local_constructs_two_rungs(self, tmp_path: Path) -> None:
        pipeline = _make_pipeline(
            _make_config(
                table_judge_ladder=True,
                strict_local=False,
                table_judge_rung1_model="glm-5.3-flash:cloud",
                table_judge_rung2_binary="gemini",
            )
        )
        rungs = pipeline._build_table_judge_rungs()
        assert len(rungs) == 2
        # Real construction (build_ollama_rung / make_gemini_rung) is itself
        # hermetic -- host resolution and closure binding only, no I/O.
        assert all(callable(r) for r in rungs)


# ---------------------------------------------------------------------------
# _run_table_judge_gate — skip rules
# ---------------------------------------------------------------------------


class TestGateSkipRules:
    def test_chart_asset_engine_skips_entirely(self, tmp_path: Path) -> None:
        pipeline = _make_pipeline()
        pdf_path = _ruled_pdf(tmp_path)
        state = _make_state(pdf_path)
        ps = state.pages[1]
        bo = _bo(_TABLE_MD, engine="chart_asset")

        with patch(
            "socr.tables.witness.prepare_table_witnesses",
            side_effect=AssertionError("must not be called for chart_asset pages"),
        ):
            pipeline._run_table_judge_gate(state, 1, ps, bo, [_accept_rung()])

        assert ps.table_ladder_disposition is None
        assert not state.events

    def test_empty_text_skips_entirely(self, tmp_path: Path) -> None:
        pipeline = _make_pipeline()
        pdf_path = _ruled_pdf(tmp_path)
        state = _make_state(pdf_path)
        ps = state.pages[1]
        bo = _bo("")

        with patch(
            "socr.tables.witness.prepare_table_witnesses",
            side_effect=AssertionError("must not be called for empty text"),
        ):
            pipeline._run_table_judge_gate(state, 1, ps, bo, [_accept_rung()])

        assert ps.table_ladder_disposition is None
        assert not state.events


# ---------------------------------------------------------------------------
# _run_table_judge_gate — witness fail-open (never a silent pass, never an
# exception)
# ---------------------------------------------------------------------------


class TestWitnessFailOpen:
    def test_missing_witness_is_unverified(self, tmp_path: Path) -> None:
        """Borderless page: 0 located boxes -> WitnessStatus.MISSING -> UNVERIFIED."""
        pipeline = _make_pipeline()
        pdf_path = _borderless_pdf(tmp_path)
        state = _make_state(pdf_path)
        ps = state.pages[1]
        bo = _bo(_TABLE_MD)

        rung = _accept_rung()
        pipeline._run_table_judge_gate(state, 1, ps, bo, [rung])

        assert ps.table_ladder_disposition == FailureMode.TABLE_UNVERIFIED
        assert rung.calls == []  # never reached: witness never LOCATED
        events = _events_of_kind(state, TABLE_LADDER_UNVERIFIED_KIND)
        assert len(events) == 1
        # No rung ever ran for a witness that was never LOCATED -- an empty
        # rung trail, not an absent one (GH-353 rung-trail follow-up).
        #
        # GH-560: and the record says so outright. This terminal used to be
        # labelled "infra problem, retryable on resume" like a rung outage,
        # which promised a retry nothing performs -- the P1 latch correctly
        # does not fire for a no-witness terminal, so the document is skipped
        # on the next run.
        assert events[0].data == {
            "table_id": "p1-t0",
            "rung_trail": [],
            "witness_scope": "none",
            "retryable": False,
        }
        assert "no table witness could be prepared" in events[0].detail
        assert "retryable on resume" not in events[0].detail
        assert ps.table_unverified_unwitnessed is True

    def test_count_mismatch_ambiguous_is_judged_not_abstained(self, tmp_path: Path) -> None:
        """1 box, 2 emitted blocks -> count mismatch -> AMBIGUOUS with a page
        crop -> the judge looks (GH-373). A high PASS accepts; this is no
        longer the UNVERIFIED abstention."""
        pipeline = _make_pipeline()
        pdf_path = _ruled_pdf(tmp_path)
        state = _make_state(pdf_path)
        ps = state.pages[1]
        bo = _bo(_TWO_TABLE_MD)

        rung = _QueueRung(
            [
                RungResult(rung="fake1", ok=True, verdict=_pass_verdict("high")),
                RungResult(rung="fake1", ok=True, verdict=_pass_verdict("high")),
            ]
        )
        pipeline._run_table_judge_gate(state, 1, ps, bo, [rung])

        assert len(rung.calls) == 2
        assert ps.table_ladder_disposition is None
        events = _events_of_kind(state, TABLE_LADDER_ACCEPTED_KIND)
        assert len(events) == 2
        assert {e.data.get("table_id") for e in events} == {"p1-t0", "p1-t1"}
        assert {e.data.get("witness_scope") for e in events} == {"page"}

    def test_witness_preparation_exception_is_unverified_not_raised(self, tmp_path: Path) -> None:
        pipeline = _make_pipeline()
        pdf_path = _ruled_pdf(tmp_path)
        state = _make_state(pdf_path)
        ps = state.pages[1]
        bo = _bo(_TABLE_MD)

        with patch(
            "socr.pipeline.orchestrator.prepare_table_witnesses",
            side_effect=RuntimeError("boom"),
            create=True,
        ):
            # prepare_table_witnesses is imported lazily inside the gate, so
            # patch it at its owning module -- the gate's own import binds
            # the same object.
            with patch(
                "socr.tables.witness.prepare_table_witnesses",
                side_effect=RuntimeError("boom"),
            ):
                pipeline._run_table_judge_gate(state, 1, ps, bo, [_accept_rung()])

        assert ps.table_ladder_disposition == FailureMode.TABLE_UNVERIFIED
        events = _events_of_kind(state, TABLE_LADDER_UNVERIFIED_KIND)
        assert len(events) == 1
        assert "boom" in events[0].detail

    def test_no_rungs_available_is_unverified_no_call(self, tmp_path: Path) -> None:
        """strict_local + ladder (or ladder off): rungs=[] fails open, no call made."""
        pipeline = _make_pipeline()
        pdf_path = _ruled_pdf(tmp_path)
        state = _make_state(pdf_path)
        ps = state.pages[1]
        bo = _bo(_TABLE_MD)

        pipeline._run_table_judge_gate(state, 1, ps, bo, [])

        assert ps.table_ladder_disposition == FailureMode.TABLE_UNVERIFIED
        events = _events_of_kind(state, TABLE_LADDER_UNVERIFIED_KIND)
        assert len(events) == 1
        # rungs=[] -- fail-open before any rung is ever called.
        assert events[0].data == {
            "table_id": "p1-t0",
            "rung_trail": [],
            "witness_scope": "located",
        }


# ---------------------------------------------------------------------------
# _run_table_judge_gate — rung fail-open (missing binary / HTTP error /
# parser error all collapse to the same ok=False contract)
# ---------------------------------------------------------------------------


class TestRungFailOpen:
    @pytest.mark.parametrize(
        "reason",
        [
            "missing binary: [Errno 2] No such file or directory: 'gemini'",
            "httpx.ConnectError: connection refused",
            "no JSON object found in rung output",
        ],
    )
    def test_not_s1_rung_is_unverified(self, tmp_path: Path, reason: str) -> None:
        pipeline = _make_pipeline()
        pdf_path = _ruled_pdf(tmp_path)
        state = _make_state(pdf_path)
        ps = state.pages[1]
        bo = _bo(_TABLE_MD)

        pipeline._run_table_judge_gate(state, 1, ps, bo, [_not_s1_rung(reason)])

        assert ps.table_ladder_disposition == FailureMode.TABLE_UNVERIFIED
        events = _events_of_kind(state, TABLE_LADDER_UNVERIFIED_KIND)
        assert len(events) == 1
        # The rung DID run (¬S1 -- an answer that failed to parse/transport,
        # not "never called"), so it names the configured rung-1 identity.
        assert events[0].data == {
            "table_id": "p1-t0",
            "rung_trail": [
                {"rung": "fake1", "ok": False, "executing": pipeline.config.table_judge_rung1_model}
            ],
            "witness_scope": "located",
        }

    def test_run_table_ladder_exception_is_unverified_not_raised(self, tmp_path: Path) -> None:
        pipeline = _make_pipeline()
        pdf_path = _ruled_pdf(tmp_path)
        state = _make_state(pdf_path)
        ps = state.pages[1]
        bo = _bo(_TABLE_MD)

        with patch(
            "socr.judge.table_ladder.run_table_ladder", side_effect=RuntimeError("ladder blew up")
        ):
            pipeline._run_table_judge_gate(state, 1, ps, bo, [_accept_rung()])

        assert ps.table_ladder_disposition == FailureMode.TABLE_UNVERIFIED
        events = _events_of_kind(state, TABLE_LADDER_UNVERIFIED_KIND)
        assert len(events) == 1


# ---------------------------------------------------------------------------
# _run_table_judge_gate — ladder outcomes reach the right disposition + event
# ---------------------------------------------------------------------------


class TestLadderOutcomes:
    def test_accepted_sets_no_disposition_and_emits_accepted_event(self, tmp_path: Path) -> None:
        pipeline = _make_pipeline()
        pdf_path = _ruled_pdf(tmp_path)
        state = _make_state(pdf_path)
        ps = state.pages[1]
        bo = _bo(_TABLE_MD)

        rung = _accept_rung()
        pipeline._run_table_judge_gate(state, 1, ps, bo, [rung])

        assert rung.calls  # the crop WAS judged
        assert ps.table_ladder_disposition is None
        events = _events_of_kind(state, TABLE_LADDER_ACCEPTED_KIND)
        assert len(events) == 1
        assert events[0].data == {
            "table_id": "p1-t0",
            "rung_trail": [
                {"rung": "fake1", "ok": True, "executing": pipeline.config.table_judge_rung1_model}
            ],
            "witness_scope": "located",
        }

    def test_rejected_sets_disposition_and_emits_rejected_event(self, tmp_path: Path) -> None:
        pipeline = _make_pipeline()
        pdf_path = _ruled_pdf(tmp_path)
        state = _make_state(pdf_path)
        ps = state.pages[1]
        bo = _bo(_TABLE_MD)

        rung = _reject_rung()
        pipeline._run_table_judge_gate(state, 1, ps, bo, [rung])

        assert ps.table_ladder_disposition == FailureMode.TABLE_REJECTED
        events = _events_of_kind(state, TABLE_LADDER_REJECTED_KIND)
        assert len(events) == 1
        assert events[0].data == {
            "table_id": "p1-t0",
            "rung_trail": [
                {"rung": "fake1", "ok": True, "executing": pipeline.config.table_judge_rung1_model}
            ],
            "witness_scope": "located",
        }

    def test_reduce_page_ladder_rejected_wins_over_unverified(self, tmp_path: Path) -> None:
        """Two tables, one REJECTED one UNVERIFIED: page disposition is REJECTED
        (A4's reduce_page_ladder rule: any REJECTED wins)."""
        pipeline = _make_pipeline()
        pdf_path = _ruled_pdf(tmp_path)
        doc = fitz.open(str(pdf_path))
        # Add a second, borderless (unlocatable) table region's markdown block
        # via the page's own text -- reuse the two-table markdown but keep
        # only ONE ruled box, so block 0 pairs LOCATED and block 1 is a
        # separate borderless-shaped block. Simpler: two blocks, one box ->
        # BOTH become AMBIGUOUS (already covered above). To get one REJECTED
        # + one UNVERIFIED on the SAME page, drive the gate directly against
        # two witnesses via a stubbed prepare_table_witnesses.
        doc.close()
        state = _make_state(pdf_path)
        ps = state.pages[1]
        bo = _bo(_TABLE_MD)

        from socr.tables.witness import TableWitness, WitnessScope, WitnessStatus

        crop_path = tmp_path / "crop.png"
        crop_path.write_bytes(b"fake-png")
        witnesses = [
            TableWitness(
                table_id="p1-t0",
                page_num=1,
                block_index=0,
                markdown=_TABLE_MD,
                status=WitnessStatus.LOCATED,
                crop_path=crop_path,
                scope=WitnessScope.LOCATED,
            ),
            TableWitness(
                table_id="p1-t1",
                page_num=1,
                block_index=1,
                markdown=_TABLE_MD,
                status=WitnessStatus.MISSING,
                scope=WitnessScope.NONE,
            ),
        ]

        class _Ctx:
            def __enter__(self):
                return witnesses

            def __exit__(self, *a):
                return False

        with patch("socr.tables.witness.prepare_table_witnesses", return_value=_Ctx()):
            pipeline._run_table_judge_gate(state, 1, ps, bo, [_reject_rung()])

        assert ps.table_ladder_disposition == FailureMode.TABLE_REJECTED
        assert _events_of_kind(state, TABLE_LADDER_REJECTED_KIND)
        assert _events_of_kind(state, TABLE_LADDER_UNVERIFIED_KIND)


# ---------------------------------------------------------------------------
# P1 prep item 1 — the rung-unavailable retry latch, derived at the gate
# BEFORE any mechanical-binding clamp (docs/log/2026-09-02_gh359-ladder-
# terminals-design.md, "Panel and synthesis"; PR #518's
# ``equation_lane_retry_pending`` is the shape to reuse).
#
# The latch is causal, not merely "an unavailable rung occurred somewhere
# in the trail": it fires only when the terminal actually is UNVERIFIED
# *because* the missing answer prevented resolution. A REJECTED terminal (a
# real FAIL is on record) or an ACCEPTED terminal (a real high PASS is on
# record) is a content verdict the retry latch must leave alone, even when
# an earlier rung in the same table's trail was unavailable.
#
# Contract these tests hold the gate to:
#   * ``PageState.table_judge_retry_pending: bool = False``
#   * ``_run_table_judge_gate`` sets it True per the causal table below,
#     using each ``RungResult.unavailable`` bit from the rung split.
#   * a rung callable that raises latches ONLY when the exception is an
#     availability shape (``is_availability_exception``); a programming error
#     ends the table UNVERIFIED without latching (cold review round 2).
# ---------------------------------------------------------------------------


def _unavailable_rung_result(rung: str = "fake") -> RungResult:
    return RungResult(rung=rung, ok=False, error="simulated transport failure", unavailable=True)


def _content_not_s1_rung_result(rung: str = "fake") -> RungResult:
    """¬S1 that is NOT rung-unavailable (e.g. a parse failure) -- content-shaped."""
    return RungResult(rung=rung, ok=False, error="no JSON object found", unavailable=False)


class TestRetryLatchCausalClassification:
    def test_unavailable_then_fail_rejects_and_does_not_latch(self, tmp_path: Path) -> None:
        """C-then-B: rung 1 unavailable, rung 2 FAILs -> REJECTED, no latch.

        A real content verdict is on record; the missing rung-1 answer did
        not prevent resolution, so retrying buys nothing.
        """
        pipeline = _make_pipeline()
        pdf_path = _ruled_pdf(tmp_path)
        state = _make_state(pdf_path)
        ps = state.pages[1]
        bo = _bo(_TABLE_MD)

        rungs = [
            _QueueRung([_unavailable_rung_result("fake1")], rung_id="fake1"),
            _QueueRung(
                [RungResult(rung="fake2", ok=True, verdict=_fail_verdict())], rung_id="fake2"
            ),
        ]
        pipeline._run_table_judge_gate(state, 1, ps, bo, rungs)

        assert ps.table_ladder_disposition == FailureMode.TABLE_REJECTED
        assert ps.table_judge_retry_pending is False

    def test_unavailable_then_high_pass_accepts_and_does_not_latch(self, tmp_path: Path) -> None:
        """C-then-high-PASS: rung 1 unavailable, rung 2 PASSes high -> ACCEPTED, no latch."""
        pipeline = _make_pipeline()
        pdf_path = _ruled_pdf(tmp_path)
        state = _make_state(pdf_path)
        ps = state.pages[1]
        bo = _bo(_TABLE_MD)

        rungs = [
            _QueueRung([_unavailable_rung_result("fake1")], rung_id="fake1"),
            _QueueRung(
                [RungResult(rung="fake2", ok=True, verdict=_pass_verdict("high"))], rung_id="fake2"
            ),
        ]
        pipeline._run_table_judge_gate(state, 1, ps, bo, rungs)

        assert ps.table_ladder_disposition is None
        assert ps.table_judge_retry_pending is False

    def test_unavailable_then_low_pass_is_unverified_and_latches(self, tmp_path: Path) -> None:
        """C-then-low-PASS: no quorum, exhausts UNVERIFIED -- the missing
        rung-1 answer is exactly why resolution failed. Latch."""
        pipeline = _make_pipeline()
        pdf_path = _ruled_pdf(tmp_path)
        state = _make_state(pdf_path)
        ps = state.pages[1]
        bo = _bo(_TABLE_MD)

        rungs = [
            _QueueRung([_unavailable_rung_result("fake1")], rung_id="fake1"),
            _QueueRung(
                [RungResult(rung="fake2", ok=True, verdict=_pass_verdict("low"))], rung_id="fake2"
            ),
        ]
        pipeline._run_table_judge_gate(state, 1, ps, bo, rungs)

        assert ps.table_ladder_disposition == FailureMode.TABLE_UNVERIFIED
        assert ps.table_judge_retry_pending is True

    def test_fail_then_unavailable_is_unverified_and_latches(self, tmp_path: Path) -> None:
        """B-then-C: rung 1 FAILs, rung 2 (the tiebreak) never answers.

        The stronger judge never voted; the missing answer is why the ladder
        could not resolve. Latch."""
        pipeline = _make_pipeline()
        pdf_path = _ruled_pdf(tmp_path)
        state = _make_state(pdf_path)
        ps = state.pages[1]
        bo = _bo(_TABLE_MD)

        rungs = [
            _QueueRung(
                [RungResult(rung="fake1", ok=True, verdict=_fail_verdict())], rung_id="fake1"
            ),
            _QueueRung([_unavailable_rung_result("fake2")], rung_id="fake2"),
        ]
        pipeline._run_table_judge_gate(state, 1, ps, bo, rungs)

        assert ps.table_ladder_disposition == FailureMode.TABLE_UNVERIFIED
        assert ps.table_judge_retry_pending is True

    def test_unavailable_then_unavailable_latches(self, tmp_path: Path) -> None:
        """C-then-C: neither rung ever answered. Latch."""
        pipeline = _make_pipeline()
        pdf_path = _ruled_pdf(tmp_path)
        state = _make_state(pdf_path)
        ps = state.pages[1]
        bo = _bo(_TABLE_MD)

        rungs = [
            _QueueRung([_unavailable_rung_result("fake1")], rung_id="fake1"),
            _QueueRung([_unavailable_rung_result("fake2")], rung_id="fake2"),
        ]
        pipeline._run_table_judge_gate(state, 1, ps, bo, rungs)

        assert ps.table_ladder_disposition == FailureMode.TABLE_UNVERIFIED
        assert ps.table_judge_retry_pending is True

    @pytest.mark.parametrize(
        "exc,latches",
        [
            # Transport shapes: the rung could not be reached. Retry when it returns.
            (ConnectionError("connection refused"), True),
            (TimeoutError("read timed out"), True),
            (httpx.ConnectError("no route to host"), True),
            # Software defects: deterministic, and retrying only reproduces them.
            (TypeError("unsupported operand"), False),
            (AssertionError("invariant broken"), False),
            (KeyError("verdict"), False),
            (ValueError("bad literal"), False),
            (RuntimeError("rung process crashed"), False),
        ],
        ids=lambda v: type(v).__name__ if isinstance(v, BaseException) else str(v),
    )
    def test_only_availability_exceptions_latch(
        self, tmp_path: Path, exc: BaseException, latches: bool
    ) -> None:
        """Cold review round 2, finding 2. A rung is contractually non-raising,
        so ANY exception escaping one is unexpected -- but unavailability is a
        TYPED classification, not "something went wrong". A transport failure
        latches; a programming error must not, or every resume re-runs the
        ladder to reproduce the same crash forever.

        The terminal is UNVERIFIED either way: the classification decides
        whether it is worth retrying, never whether the table is trusted."""
        pipeline = _make_pipeline()
        pdf_path = _ruled_pdf(tmp_path)
        state = _make_state(pdf_path)
        ps = state.pages[1]
        bo = _bo(_TABLE_MD)

        def _raising_rung(crop_path, markdown, prior_findings):
            raise exc

        pipeline._run_table_judge_gate(state, 1, ps, bo, [_raising_rung])

        assert ps.table_ladder_disposition == FailureMode.TABLE_UNVERIFIED
        assert ps.table_judge_retry_pending is latches

    def test_mixed_multi_table_rejected_plus_unavailable_still_latches(
        self, tmp_path: Path
    ) -> None:
        """One table on the page REJECTS (content verdict); a second table on
        the SAME page is unavailable and unresolved. The page-level reducer
        gives REJECTED precedence for ``table_ladder_disposition`` (A4), but
        the latch must still fire, or the second table's unresolved status
        never gets retried once the missing rung comes back (D1b would skip
        the whole page forever via the REJECTED resume exception)."""
        pipeline = _make_pipeline()
        pdf_path = _ruled_pdf(tmp_path)
        state = _make_state(pdf_path)
        ps = state.pages[1]
        bo = _bo(_TABLE_MD)

        from socr.tables.witness import TableWitness, WitnessScope, WitnessStatus

        crop_path = tmp_path / "crop.png"
        crop_path.write_bytes(b"fake-png")
        witnesses = [
            TableWitness(
                table_id="p1-t0",
                page_num=1,
                block_index=0,
                markdown=_TABLE_MD,
                status=WitnessStatus.LOCATED,
                crop_path=crop_path,
                scope=WitnessScope.LOCATED,
            ),
            TableWitness(
                table_id="p1-t1",
                page_num=1,
                block_index=1,
                markdown=_TABLE_MD,
                status=WitnessStatus.LOCATED,
                crop_path=crop_path,
                scope=WitnessScope.LOCATED,
            ),
        ]

        class _Ctx:
            def __enter__(self):
                return witnesses

            def __exit__(self, *a):
                return False

        # Table 0 -> REJECTED (single FAIL rung). Table 1 -> unavailable,
        # unresolved. _QueueRung is consumed in call order across BOTH
        # tables' ladder runs, so queue table 0's then table 1's result.
        rung = _QueueRung(
            [
                RungResult(rung="fake", ok=True, verdict=_fail_verdict()),
                _unavailable_rung_result("fake"),
            ]
        )
        with patch("socr.tables.witness.prepare_table_witnesses", return_value=_Ctx()):
            pipeline._run_table_judge_gate(state, 1, ps, bo, [rung])

        assert ps.table_ladder_disposition == FailureMode.TABLE_REJECTED
        assert ps.table_judge_retry_pending is True, (
            "a mixed page (one REJECTED table, one unavailable/unresolved table) "
            "must still latch -- otherwise the unresolved table is never re-judged"
        )


class TestRetryLatchNonLatchingControls:
    """Every shape that must NOT set the latch -- content refusals, parse
    failures, missing witnesses, and configured-off/strict_local empty rung
    lists, all of which the run fingerprint already describes or which are
    real content verdicts, not rung outages."""

    def test_fail_fail_content_rejection_does_not_latch(self, tmp_path: Path) -> None:
        pipeline = _make_pipeline()
        pdf_path = _ruled_pdf(tmp_path)
        state = _make_state(pdf_path)
        ps = state.pages[1]
        bo = _bo(_TABLE_MD)

        pipeline._run_table_judge_gate(state, 1, ps, bo, [_reject_rung()])

        assert ps.table_ladder_disposition == FailureMode.TABLE_REJECTED
        assert ps.table_judge_retry_pending is False

    def test_low_low_content_uncertainty_without_unavailable_rung_does_not_latch(
        self, tmp_path: Path
    ) -> None:
        pipeline = _make_pipeline()
        pdf_path = _ruled_pdf(tmp_path)
        state = _make_state(pdf_path)
        ps = state.pages[1]
        bo = _bo(_TABLE_MD)

        rungs = [
            _QueueRung(
                [RungResult(rung="fake1", ok=True, verdict=_pass_verdict("low"))], rung_id="fake1"
            ),
            _QueueRung(
                [RungResult(rung="fake2", ok=True, verdict=_pass_verdict("low"))], rung_id="fake2"
            ),
        ]
        pipeline._run_table_judge_gate(state, 1, ps, bo, rungs)

        assert ps.table_ladder_disposition is None  # ruling 1 quorum: accepted
        assert ps.table_judge_retry_pending is False

    def test_content_not_s1_parse_failure_does_not_latch(self, tmp_path: Path) -> None:
        """A ¬S1 that is NOT rung-unavailable (malformed/garbage answer)."""
        pipeline = _make_pipeline()
        pdf_path = _ruled_pdf(tmp_path)
        state = _make_state(pdf_path)
        ps = state.pages[1]
        bo = _bo(_TABLE_MD)

        rung = _QueueRung([_content_not_s1_rung_result("fake1")], rung_id="fake1")
        pipeline._run_table_judge_gate(state, 1, ps, bo, [rung])

        assert ps.table_ladder_disposition == FailureMode.TABLE_UNVERIFIED
        assert ps.table_judge_retry_pending is False, (
            "a parse/schema failure is content-shaped -- retrying immediately "
            "hits the same junk, so it must not latch"
        )

    def test_missing_witness_does_not_latch(self, tmp_path: Path) -> None:
        pipeline = _make_pipeline()
        pdf_path = _borderless_pdf(tmp_path)
        state = _make_state(pdf_path)
        ps = state.pages[1]
        bo = _bo(_TABLE_MD)

        pipeline._run_table_judge_gate(state, 1, ps, bo, [_accept_rung()])

        assert ps.table_ladder_disposition == FailureMode.TABLE_UNVERIFIED
        assert ps.table_judge_retry_pending is False

    def test_empty_rung_list_configured_off_does_not_latch(self, tmp_path: Path) -> None:
        """rungs=[] (flag off / strict_local) is the fingerprint's own signal,
        not transient unavailability -- toggling the flag already reprocesses
        via the fingerprint, so this must not also set the latch."""
        pipeline = _make_pipeline()
        pdf_path = _ruled_pdf(tmp_path)
        state = _make_state(pdf_path)
        ps = state.pages[1]
        bo = _bo(_TABLE_MD)

        pipeline._run_table_judge_gate(state, 1, ps, bo, [])

        assert ps.table_ladder_disposition == FailureMode.TABLE_UNVERIFIED
        assert ps.table_judge_retry_pending is False

    def test_default_page_state_has_no_latch(self, tmp_path: Path) -> None:
        pdf_path = _ruled_pdf(tmp_path)
        state = _make_state(pdf_path)
        assert state.pages[1].table_judge_retry_pending is False


# ---------------------------------------------------------------------------
# Full process() runs — the ticket's required gate test: flag on vs off,
# asserting the DIFFERENCE only (never an absolute outcome pinned).
# ---------------------------------------------------------------------------


def _route_page_returning(text: str, engine: str = "qwen"):
    def _fake_route(page_num, ladder, run_provider, judge, **kwargs):
        from socr.pipeline.agentic import PageDecision, ProviderAttempt

        out = PageOutput(
            page_num=page_num,
            text=text,
            status=PageStatus.SUCCESS,
            engine=engine,
            audit_passed=True,
        )
        prof = ladder[0]
        att = ProviderAttempt(
            engine=prof.engine,
            output=out,
            cost_usd=prof.cost_per_page_usd,
            accepted=True,
            reason="ok",
            provider_id=prof.id,
            model=prof.model,
            backend=prof.backend,
        )
        return PageDecision(page_num=page_num, final_output=out, attempts=[att], accepted=True)

    return _fake_route


class TestProcessFlagDifference:
    def test_flag_off_vs_on_with_rejecting_ladder(self, tmp_path: Path) -> None:
        pdf_path = _ruled_pdf(tmp_path / "off", "doc.pdf")
        pdf_path_on = _ruled_pdf(tmp_path / "on", "doc.pdf")

        # -- flag off: baseline, untouched by the ladder --------------------
        cfg_off = _make_config(table_judge_ladder=False)
        pipeline_off = _make_pipeline(cfg_off)
        with (
            patch(
                "socr.pipeline.orchestrator.route_page",
                side_effect=_route_page_returning(_TABLE_MD),
            ),
            patch.object(
                pipeline_off, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]
            ),
            patch.object(pipeline_off, "_resolve_judge_model", return_value=""),
        ):
            result_off = pipeline_off.process(pdf_path, tmp_path / "off_out")

        assert result_off.status == DocumentStatus.SUCCESS
        assert result_off.pages[0].failure_mode == FailureMode.NONE
        assert "table_rejected" not in (result_off.error or "")

        # -- flag on, injected rejecting ladder ------------------------------
        cfg_on = _make_config(table_judge_ladder=True)
        pipeline_on = _make_pipeline(cfg_on)
        with (
            patch(
                "socr.pipeline.orchestrator.route_page",
                side_effect=_route_page_returning(_TABLE_MD),
            ),
            patch.object(
                pipeline_on, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]
            ),
            patch.object(pipeline_on, "_resolve_judge_model", return_value=""),
            patch.object(pipeline_on, "_build_table_judge_rungs", return_value=[_reject_rung()]),
        ):
            result_on = pipeline_on.process(pdf_path_on, tmp_path / "on_out")

        # The difference the flag makes: document demoted, page disposition
        # carries the terminal, and the ladder's own audit events reach the
        # note surface -- exactly the C2/C3 contract this gate feeds.
        # result.pages[0] is the whole-document aggregate PageOutput
        # (page_num=0) -- per-page terminals surface via result.error, the
        # same C2-established surface tests/test_ladder_status_surfacing.py
        # asserts against, not a per-page PageOutput in this list.
        assert result_on.status == DocumentStatus.AUDIT_FAILED
        assert "table_rejected" in (result_on.error or "")

    def test_native_lane_is_witnessed_too(self, tmp_path: Path) -> None:
        """The former F1 shape: a born-digital native page with a defective
        table is still judged and demoted -- the native lane is not exempt."""
        pdf_path = _ruled_pdf(tmp_path)
        cfg = _make_config(table_judge_ladder=True, native_first=True)
        pipeline = _make_pipeline(cfg)

        from socr.core.born_digital import DocumentAssessment, PageAssessment

        assessment = DocumentAssessment(
            path=pdf_path,
            pages=[
                PageAssessment(
                    page_num=1,
                    is_born_digital=True,
                    native_text=_TABLE_MD,
                    confidence=0.9,
                    has_tables=True,
                )
            ],
        )
        from unittest.mock import MagicMock

        pipeline.bd_detector = MagicMock()
        pipeline.bd_detector.detect.return_value = assessment

        with (
            patch.object(
                pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]
            ),
            patch.object(pipeline, "_resolve_judge_model", return_value=""),
            patch.object(pipeline, "_build_table_judge_rungs", return_value=[_reject_rung()]),
        ):
            result = pipeline.process(pdf_path, tmp_path / "native_out")

        assert result.status == DocumentStatus.AUDIT_FAILED
        assert "table_rejected" in (result.error or "")

    def test_flag_off_makes_zero_witness_calls(self, tmp_path: Path) -> None:
        """Sentinel: with the flag off, the gate is never entered, so witness
        preparation (and therefore crop rendering / any rung transport) never
        fires -- not merely "returns nothing", but genuinely not called."""
        pdf_path = _ruled_pdf(tmp_path)
        cfg = _make_config(table_judge_ladder=False)
        pipeline = _make_pipeline(cfg)

        with (
            patch(
                "socr.pipeline.orchestrator.route_page",
                side_effect=_route_page_returning(_TABLE_MD),
            ),
            patch.object(
                pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]
            ),
            patch.object(pipeline, "_resolve_judge_model", return_value=""),
            patch(
                "socr.tables.witness.prepare_table_witnesses",
                side_effect=AssertionError("must not be called with the flag off"),
            ),
        ):
            result = pipeline.process(pdf_path, tmp_path / "sentinel_out")

        assert result.status == DocumentStatus.SUCCESS
