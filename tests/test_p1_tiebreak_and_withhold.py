"""P1 (tasks t7, t8): the ruled Q1 two-low chain and the Q2 withhold path,
as GH-575 settles them, through ``_run_table_judge_gate``.

Every test here drives the REAL gate entry point with injected reader rungs
and the REAL guard service. Cold review round 1, finding 6: the previous
version patched ``evaluate_cell_guard`` to a preselected answer, so it pinned
the gate's ``if`` statements against a constant and would have stayed green
through a rewrite of the chain it exists to protect. What is mocked now is
only the TRANSPORT -- a callable of the adjudicator's shape whose answer is
controlled -- plus the native binding evidence, which otherwise depends on
the fixture's real geometry.

The terminal table GH-575 rules, and this file pins in full:

===========================  ==================  ==========================
evidence                     Q1 (two low PASS)   Q2 (readers REJECTED)
===========================  ==================  ==========================
binding PASS                 ACCEPTED geometry   ACCEPTED geometry
binding CONTRADICT           UNVERIFIED*         UNVERIFIED*
blind: every cell agrees     ACCEPTED blind      ACCEPTED blind
blind: a cell DISAGREES      UNVERIFIED          WITHHELD
no doubted cell / unresolved UNVERIFIED          UNVERIFIED
no adjudicator configured    UNVERIFIED          UNVERIFIED
adjudicator outage/refusal   UNVERIFIED + latch  UNVERIFIED + latch
adjudicator defect           UNVERIFIED          UNVERIFIED
budget/cap refusal           UNVERIFIED          UNVERIFIED
===========================  ==================  ==========================

``*`` and the adjudicator is never called.

WITHHELD has exactly one cell in that table, and that is the point: bytes are
only hidden when the readers refused them AND a blind third vendor read
something else out of the crop.

Hermetic per CLAUDE.md: no ollama, no CLI, no network, no absolute outcome
measured on this machine -- the difference tests vary one step at a time over
both provider states (``rungs=[]`` and injected rungs).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from _adjudicator_doubles import (
    DOUBLE_RUNG_ID,
    agreeing_adjudicator,
    defective_adjudicator,
    mismatching_adjudicator,
    unavailable_adjudicator,
)

from socr.core.audit_log import AuditEvent
from socr.core.config import EngineType, PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.result import FailureMode, PageOutput, PageStatus
from socr.core.state import DocumentState
from socr.judge.table_verdict import (
    RUNG_KIND_CELL_ADJUDICATOR,
    Finding,
    FindingCode,
    RungResult,
    TableJudgeVerdict,
)
from socr.pipeline.orchestrator import UnifiedPipeline
from socr.tables.binding import BindingEvidence

MARKDOWN = (
    "| c0 | c1 | c2 | c3 |\n"
    "| --- | --- | --- | --- |\n"
    "| 10 | 11 | 12 | 13 |\n"
    "| 20 | 21 | 22 | 23 |\n"
    "| 30 | 31 | 32 | 33 |\n"
)

#: The tokens the EXTRACTION above holds for the cells the readers doubt.
#: An adjudicator that reports these agrees; anything else disagrees.
EXTRACTION = {"R1C2": "11", "R2C3": "22"}


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


def _ruled_pdf(tmp_path: Path, name: str = "doc.pdf") -> Path:
    """One page, one ruled (fully boxed) table -> exactly 1 located witness."""
    import fitz

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


def _make_pipeline(
    *,
    evidence: BindingEvidence = BindingEvidence.ABSTAIN,
    adjudicator=None,
    config: PipelineConfig | None = None,
) -> UnifiedPipeline:
    """A pipeline with the two chain INPUTS controlled and everything else real.

    ``evidence`` replaces what ``bind()`` would say about this fixture (which
    is a property of the PDF, not of the ruling under test); ``adjudicator``
    replaces the network call. The guard service, the gate's terminal
    selection, the latch derivation, the breaker and the event writer all run
    for real.
    """
    pipeline = UnifiedPipeline(config or _make_config())
    pipeline._binding_evidence_for_witness = lambda *a, **kw: (None, evidence)
    pipeline._build_table_cell_adjudicator = lambda: adjudicator
    return pipeline


def _make_state(pdf_path: Path) -> DocumentState:
    with patch.object(DocumentHandle, "__post_init__", lambda self: None):
        handle = DocumentHandle(path=pdf_path, page_count=1)
    return DocumentState(handle=handle)


def _bo(text: str = MARKDOWN) -> PageOutput:
    return PageOutput(
        page_num=1, text=text, status=PageStatus.SUCCESS, engine="qwen", audit_passed=True
    )


def _low_pass_with_doubts(rung: str, doubts: list[str]) -> RungResult:
    return RungResult(
        rung=rung,
        ok=True,
        verdict=TableJudgeVerdict(verdict="PASS", confidence="low", findings=[], doubts=doubts),
    )


def _fail_with_findings(rung: str, wheres: list[str]) -> RungResult:
    findings = [Finding(code=FindingCode.FABRICATED_VALUE, where=w, detail="bad") for w in wheres]
    return RungResult(
        rung=rung,
        ok=True,
        verdict=TableJudgeVerdict(verdict="FAIL", confidence="high", findings=findings),
    )


class _QueueRung:
    def __init__(self, results: list[RungResult], rung_id: str = "fake") -> None:
        self._results = list(results)
        self.rung_id = rung_id
        self.calls: list[tuple] = []

    def __call__(self, crop_path, markdown, prior_findings):
        self.calls.append((crop_path, markdown, prior_findings))
        return self._results.pop(0)


def _events_of_kind(state: DocumentState, kind: str) -> list[AuditEvent]:
    return [e for e in state.events if e.kind == kind]


def _two_low_pass_rungs(doubts_a: list[str] | None = None, doubts_b: list[str] | None = None):
    # ``None`` means "use the default doubt"; an explicitly EMPTY list means a
    # reader that doubted nothing localizable, which is its own ruled case.
    return [
        _QueueRung([_low_pass_with_doubts("r1", ["R1C2"] if doubts_a is None else doubts_a)], "r1"),
        _QueueRung([_low_pass_with_doubts("r2", ["R2C3"] if doubts_b is None else doubts_b)], "r2"),
    ]


def _reject_last_rung(wheres: list[str] | None = None):
    """One rung whose S1 answer FAILs -- REJECTED at the last rung."""
    return [_QueueRung([_fail_with_findings("r1", wheres or ["R1C2"])], "r1")]


def _run(pipeline, tmp_path, rungs, name="doc"):
    state = _make_state(_ruled_pdf(tmp_path / name))
    ps = state.pages[1]
    pipeline._run_table_judge_gate(state, 1, ps, _bo(), rungs)
    return state, ps


#: The two ruled entry paths, so every terminal below is pinned for BOTH.
PATHS = {
    "two_low": _two_low_pass_rungs,
    "rejected": _reject_last_rung,
}


# ---------------------------------------------------------------------------
# Terminals shared by both paths
# ---------------------------------------------------------------------------


class TestTerminalsCommonToBothPaths:
    @pytest.mark.parametrize("path", sorted(PATHS))
    def test_binding_pass_clears_and_the_adjudicator_is_never_asked(self, tmp_path, path):
        adj = mismatching_adjudicator()
        pipeline = _make_pipeline(evidence=BindingEvidence.PASS, adjudicator=adj)
        state, ps = _run(pipeline, tmp_path, PATHS[path](), path)

        assert ps.table_ladder_disposition is None
        events = _events_of_kind(state, "table_ladder_accepted")
        assert len(events) == 1
        assert events[0].data.get("reason") == "verified_by_geometry"
        # Free evidence first: a cleared table costs no call at all.
        assert adj.calls == []

    @pytest.mark.parametrize("path", sorted(PATHS))
    def test_binding_contradict_is_terminal_unverified_on_both_paths(self, tmp_path, path):
        """GH-575 (cold review round 1, finding 1).

        The adjudicator here would CLEAR the table. It is never asked, so a
        contradicted table cannot be published on a lucky blind token -- the
        inversion the previous build allowed on the REJECTED path.
        """
        adj = agreeing_adjudicator(EXTRACTION)
        pipeline = _make_pipeline(evidence=BindingEvidence.CONTRADICT, adjudicator=adj)
        state, ps = _run(pipeline, tmp_path, PATHS[path](), path)

        assert ps.table_ladder_disposition == FailureMode.TABLE_UNVERIFIED
        assert adj.calls == []
        assert _events_of_kind(state, "table_ladder_withheld") == []
        assert _events_of_kind(state, "table_ladder_accepted") == []

    @pytest.mark.parametrize("path", sorted(PATHS))
    def test_every_doubted_cell_agreeing_clears_the_table(self, tmp_path, path):
        adj = agreeing_adjudicator(EXTRACTION)
        pipeline = _make_pipeline(adjudicator=adj)
        state, ps = _run(pipeline, tmp_path, PATHS[path](), path)

        assert ps.table_ladder_disposition is None
        events = _events_of_kind(state, "table_ladder_accepted")
        assert len(events) == 1
        assert events[0].data.get("reason") == "verified_by_blind_cell_transcription"
        assert adj.calls  # it really was asked

    @pytest.mark.parametrize("path", sorted(PATHS))
    def test_a_doubt_set_with_no_canonical_cell_is_unverified_not_withheld(self, tmp_path, path):
        """An empty/unusable doubt set means nobody asked a question, so
        nothing was established: UNVERIFIED on BOTH paths (GH-575)."""
        adj = mismatching_adjudicator()
        pipeline = _make_pipeline(adjudicator=adj)
        if path == "two_low":
            rungs = _two_low_pass_rungs([], [])
        else:
            rungs = [
                _QueueRung(
                    [
                        RungResult(
                            rung="r1",
                            ok=True,
                            verdict=TableJudgeVerdict(
                                verdict="FAIL",
                                confidence="high",
                                findings=[
                                    Finding(
                                        code=FindingCode.NOT_A_TABLE,
                                        where="",
                                        detail="chart, not table",
                                    )
                                ],
                            ),
                        )
                    ],
                    "r1",
                )
            ]
        state, ps = _run(pipeline, tmp_path, rungs, path)

        assert ps.table_ladder_disposition == FailureMode.TABLE_UNVERIFIED
        assert _events_of_kind(state, "table_ladder_withheld") == []
        assert adj.calls == []

    @pytest.mark.parametrize("path", sorted(PATHS))
    def test_a_reference_that_does_not_resolve_is_unverified(self, tmp_path, path):
        """R9C9 is off the emitted grid: the whole set fails closed, and the
        chain does not clear (or condemn) a table on the cells it happened to
        resolve."""
        adj = mismatching_adjudicator()
        pipeline = _make_pipeline(adjudicator=adj)
        rungs = (
            _two_low_pass_rungs(["R9C9"], ["R9C9"])
            if path == "two_low"
            else _reject_last_rung(["R9C9"])
        )
        state, ps = _run(pipeline, tmp_path, rungs, path)

        assert ps.table_ladder_disposition == FailureMode.TABLE_UNVERIFIED
        assert _events_of_kind(state, "table_ladder_withheld") == []
        assert adj.calls == []

    @pytest.mark.parametrize("path", sorted(PATHS))
    def test_no_adjudicator_configured_is_unverified_and_does_not_latch(self, tmp_path, path):
        pipeline = _make_pipeline(adjudicator=None)
        state, ps = _run(pipeline, tmp_path, PATHS[path](), path)

        assert ps.table_ladder_disposition == FailureMode.TABLE_UNVERIFIED
        assert _events_of_kind(state, "table_ladder_withheld") == []
        # No call was attempted, so there is nothing transient to wait for.
        assert getattr(ps, "table_judge_retry_pending", False) is False

    @pytest.mark.parametrize("path", sorted(PATHS))
    def test_an_adjudicator_outage_is_unverified_and_latches_on_its_own_kind(self, tmp_path, path):
        pipeline = _make_pipeline(adjudicator=unavailable_adjudicator())
        state, ps = _run(pipeline, tmp_path, PATHS[path](), path)

        assert ps.table_ladder_disposition == FailureMode.TABLE_UNVERIFIED
        assert _events_of_kind(state, "table_ladder_withheld") == []
        assert ps.table_judge_retry_pending is True
        # The latch names the ADJUDICATOR, not a reader: recovery must be
        # asked about the provider that actually failed.
        assert ps.table_judge_retry_rungs == [RUNG_KIND_CELL_ADJUDICATOR]

    @pytest.mark.parametrize("path", sorted(PATHS))
    def test_a_deterministic_defect_is_unverified_and_never_latches(self, tmp_path, path):
        pipeline = _make_pipeline(adjudicator=defective_adjudicator())
        state, ps = _run(pipeline, tmp_path, PATHS[path](), path)

        assert ps.table_ladder_disposition == FailureMode.TABLE_UNVERIFIED
        assert getattr(ps, "table_judge_retry_pending", False) is False

    @pytest.mark.parametrize("path", sorted(PATHS))
    def test_a_budget_refusal_is_unverified_makes_no_call_and_never_latches(self, tmp_path, path):
        adj = mismatching_adjudicator()
        config = _make_config(
            table_judge_adjudicator_cost_per_call_usd=0.05, max_cost_per_page=0.01
        )
        pipeline = _make_pipeline(adjudicator=adj, config=config)
        state, ps = _run(pipeline, tmp_path, PATHS[path](), path)

        assert ps.table_ladder_disposition == FailureMode.TABLE_UNVERIFIED
        assert _events_of_kind(state, "table_ladder_withheld") == []
        assert adj.calls == []
        # A cap reproduces identically on every rerun; latching it would make
        # the document permanently unskippable and change nothing.
        assert getattr(ps, "table_judge_retry_pending", False) is False

    def test_an_ordinary_high_pass_never_reaches_the_chain_at_all(self, tmp_path: Path):
        adj = mismatching_adjudicator()
        pipeline = _make_pipeline(adjudicator=adj)
        rung = _QueueRung(
            [
                RungResult(
                    rung="r1",
                    ok=True,
                    verdict=TableJudgeVerdict(verdict="PASS", confidence="high", findings=[]),
                )
            ],
            "r1",
        )
        state, ps = _run(pipeline, tmp_path, [rung])

        assert ps.table_ladder_disposition is None
        assert adj.calls == []
        assert _events_of_kind(state, "table_ladder_accepted")[0].data.get("reason") in (None, "")


# ---------------------------------------------------------------------------
# The one terminal where the two paths differ
# ---------------------------------------------------------------------------


class TestBlindMismatchIsTheOnlyRouteToWithheld:
    def test_a_rejected_table_whose_blind_reading_disagrees_is_withheld(self, tmp_path: Path):
        pipeline = _make_pipeline(adjudicator=mismatching_adjudicator())
        state, ps = _run(pipeline, tmp_path, _reject_last_rung())

        assert ps.table_ladder_disposition == FailureMode.TABLE_WITHHELD
        assert len(_events_of_kind(state, "table_ladder_withheld")) == 1
        # A content terminal: nothing transient to retry.
        assert getattr(ps, "table_judge_retry_pending", False) is False

    def test_a_two_low_table_whose_blind_reading_disagrees_is_only_unverified(self, tmp_path: Path):
        """Same evidence, different readers' verdict. Two LOW PASSes are not a
        refusal of the bytes, so a blind disagreement labels the table; it does
        not hide it. Withholding takes readers AND a blind reader against it."""
        pipeline = _make_pipeline(adjudicator=mismatching_adjudicator())
        state, ps = _run(pipeline, tmp_path, _two_low_pass_rungs())

        assert ps.table_ladder_disposition == FailureMode.TABLE_UNVERIFIED
        assert _events_of_kind(state, "table_ladder_withheld") == []

    def test_the_readers_verdict_is_the_only_thing_that_moved(self, tmp_path: Path):
        """The difference test for the pair above, in one place: identical
        fixture, identical geometry, identical blind answer -- only the
        readers' verdict changes, and only the terminal moves with it."""
        outcomes = {}
        for path, build in PATHS.items():
            pipeline = _make_pipeline(adjudicator=mismatching_adjudicator())
            _state, ps = _run(pipeline, tmp_path, build(), f"diff-{path}")
            outcomes[path] = ps.table_ladder_disposition

        assert outcomes["two_low"] == FailureMode.TABLE_UNVERIFIED
        assert outcomes["rejected"] == FailureMode.TABLE_WITHHELD
        assert outcomes["two_low"] != outcomes["rejected"]


# ---------------------------------------------------------------------------
# Provider-state difference: rungs=[] vs injected rungs
# ---------------------------------------------------------------------------


class TestProviderStateDifference:
    """Cold review round 1, finding 6.

    ``rungs`` is the seam ``_run_table_judge_gate`` actually consults -- it is
    the parameter, and ``_build_table_judge_rungs`` returns [] on a machine
    with no reachable provider, which is CI. The previous version injected
    rungs in BOTH arms and patched a reachability helper the gate never calls,
    so its two arms executed identical code.
    """

    @pytest.mark.parametrize("path", sorted(PATHS))
    def test_no_rungs_ends_unverified_without_touching_the_chain(self, tmp_path, path):
        adj = mismatching_adjudicator()
        pipeline = _make_pipeline(adjudicator=adj)
        state, ps = _run(pipeline, tmp_path, [], f"none-{path}")

        assert ps.table_ladder_disposition == FailureMode.TABLE_UNVERIFIED
        assert adj.calls == []
        assert _events_of_kind(state, "table_ladder_withheld") == []

    def test_only_the_rejecting_arm_moves_when_rungs_appear(self, tmp_path: Path):
        """The same fixture and the same adjudicator under both provider
        states. With no rung nothing can be judged, so both paths land on the
        same fail-closed terminal; with rungs, and only then, the readers'
        rejection can reach a withhold."""
        outcomes = {}
        for provider in ("none", "injected"):
            for path, build in PATHS.items():
                pipeline = _make_pipeline(adjudicator=mismatching_adjudicator())
                _state, ps = _run(
                    pipeline,
                    tmp_path,
                    [] if provider == "none" else build(),
                    f"{provider}-{path}",
                )
                outcomes[(provider, path)] = ps.table_ladder_disposition

        assert outcomes[("none", "two_low")] == outcomes[("none", "rejected")]
        assert outcomes[("none", "rejected")] == FailureMode.TABLE_UNVERIFIED
        assert outcomes[("injected", "two_low")] == FailureMode.TABLE_UNVERIFIED
        assert outcomes[("injected", "rejected")] == FailureMode.TABLE_WITHHELD


# ---------------------------------------------------------------------------
# Metering identity and the per-run breaker
# ---------------------------------------------------------------------------


class TestMeteringAndBreaker:
    def test_the_journal_records_the_executing_identity_not_a_default(self, tmp_path: Path):
        """Cold review round 1, finding 5. The double advertises a NON-default
        model, so a journal that named the configured default would be visibly
        wrong here."""
        adj = agreeing_adjudicator(EXTRACTION)
        pipeline = _make_pipeline(adjudicator=adj)
        state, _ps = _run(pipeline, tmp_path, _reject_last_rung())

        versions = [
            r.model_version for r in state.engine_runs if r.engine == "table_blind_cell_adjudicator"
        ]
        assert versions == [DOUBLE_RUNG_ID]
        assert pipeline.config.table_judge_adjudicator_model not in DOUBLE_RUNG_ID

    def test_one_refusal_spares_every_later_table_and_document_in_the_run(self, tmp_path: Path):
        """Cold review round 1, finding 3. A quota/credential refusal is not a
        per-table fact: the next table gets the same answer, so paying for it
        is the refusal amplification the reader breaker already prevents."""
        adj = unavailable_adjudicator(refusal=True, error="quota exceeded")
        pipeline = _make_pipeline(adjudicator=adj)

        # Table 1, document 1: the refusal happens and is recorded.
        _s1, ps1 = _run(pipeline, tmp_path, _reject_last_rung(), "t1")
        assert len(adj.calls) == 1
        assert RUNG_KIND_CELL_ADJUDICATOR in pipeline._table_rung_refused_this_run

        # Tables 2..N in the same run, and a later document in the same batch.
        _s2, ps2 = _run(pipeline, tmp_path, _reject_last_rung(), "t2")
        _s3, ps3 = _run(pipeline, tmp_path, _two_low_pass_rungs(), "t3")
        assert len(adj.calls) == 1

        # Fail-closed, and still latched, so a LATER run retries.
        for ps in (ps1, ps2, ps3):
            assert ps.table_ladder_disposition == FailureMode.TABLE_UNVERIFIED
            assert ps.table_judge_retry_pending is True
            assert ps.table_judge_retry_rungs == [RUNG_KIND_CELL_ADJUDICATOR]

    def test_a_non_refusal_outage_does_not_trip_the_breaker(self, tmp_path: Path):
        """The control. Only an external REFUSAL is a per-run fact; a plain
        outage may end at any moment, so the next table still asks."""
        adj = unavailable_adjudicator(refusal=False)
        pipeline = _make_pipeline(adjudicator=adj)
        _run(pipeline, tmp_path, _reject_last_rung(), "o1")
        _run(pipeline, tmp_path, _reject_last_rung(), "o2")

        assert len(adj.calls) == 2
        assert RUNG_KIND_CELL_ADJUDICATOR not in pipeline._table_rung_refused_this_run


class TestATrippedBreakerLatchesOnlyThePagesItCost:
    """Cold review round 2, N3.

    The breaker suppresses the adjudicator for the rest of the run. The
    suppression is now passed DOWN to the guard instead of being latched at
    the call site, because only the guard knows whether this particular table
    would have reached the adjudicator at all. Latching a table geometry
    already settled, or one whose readers localized nothing, promises a retry
    for a call that was never going to happen -- and one such table is enough
    to reopen the whole page on every later run.
    """

    def _pipeline_with_a_tripped_breaker(self, evidence):
        def _must_not_run(crop_path, cell_refs):
            raise AssertionError("the breaker must spare this call")

        pipeline = _make_pipeline(evidence=evidence, adjudicator=_must_not_run)
        pipeline._table_rung_refused_this_run.add(RUNG_KIND_CELL_ADJUDICATOR)
        return pipeline

    @pytest.mark.parametrize("path", sorted(PATHS))
    @pytest.mark.parametrize("evidence", [BindingEvidence.PASS, BindingEvidence.CONTRADICT])
    def test_geometry_that_settles_the_table_is_not_latched(self, tmp_path, path, evidence):
        pipeline = self._pipeline_with_a_tripped_breaker(evidence)
        _state, ps = _run(pipeline, tmp_path, PATHS[path](), f"{path}-{evidence.value}")

        assert getattr(ps, "table_judge_retry_pending", False) is False
        assert list(getattr(ps, "table_judge_retry_rungs", []) or []) == []

    @pytest.mark.parametrize("path", sorted(PATHS))
    def test_an_empty_doubt_set_is_not_latched(self, tmp_path, path):
        pipeline = self._pipeline_with_a_tripped_breaker(BindingEvidence.ABSTAIN)
        rungs = (
            _two_low_pass_rungs([], [])
            if path == "two_low"
            else _reject_last_rung(["not a cell reference"])
        )
        _state, ps = _run(pipeline, tmp_path, rungs, f"empty-{path}")

        assert getattr(ps, "table_judge_retry_pending", False) is False

    @pytest.mark.parametrize("path", sorted(PATHS))
    def test_a_table_that_really_needed_it_is_still_latched(self, tmp_path, path):
        """The control. Suppression on an ABSTAIN case with resolvable doubts
        IS an outage this page paid for, so it stays retryable."""
        pipeline = self._pipeline_with_a_tripped_breaker(BindingEvidence.ABSTAIN)
        _state, ps = _run(pipeline, tmp_path, PATHS[path](), f"needed-{path}")

        assert ps.table_judge_retry_pending is True
        assert ps.table_judge_retry_rungs == [RUNG_KIND_CELL_ADJUDICATOR]
        assert ps.table_ladder_disposition == FailureMode.TABLE_UNVERIFIED


class TestAnUnreadableBlindAnswerNeverWithholds:
    """Cold review round 2, N2, at the gate.

    The guard's own unit tests pin the disposition; this pins the TERMINAL,
    which is the thing a user sees. A rejected table whose blind reader could
    not read the cells keeps its bytes under UNVERIFIED -- withholding them
    would hide content on the strength of a reading nobody made.
    """

    def _unreadable_adjudicator(self):
        from socr.judge.table_rung_ollama import BlindCellResult

        def _adj(crop_path, cell_refs):
            return BlindCellResult(
                rung="adjudicator:double",
                ok=True,
                tokens={},
                unreadable=tuple(str(r) for r in cell_refs),
            )

        return _adj

    @pytest.mark.parametrize("path", sorted(PATHS))
    def test_it_is_unverified_and_does_not_latch(self, tmp_path, path):
        pipeline = _make_pipeline(adjudicator=self._unreadable_adjudicator())
        state, ps = _run(pipeline, tmp_path, PATHS[path](), f"unreadable-{path}")

        assert ps.table_ladder_disposition == FailureMode.TABLE_UNVERIFIED
        assert _events_of_kind(state, "table_ladder_withheld") == []
        # The adjudicator answered; it is not an outage.
        assert getattr(ps, "table_judge_retry_pending", False) is False
