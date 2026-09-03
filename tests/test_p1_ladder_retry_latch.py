"""P1 prep item 1 (plan tasks t7-t9): the table-judge rung-unavailable retry
latch, proven through the REAL entry paths -- ``UnifiedPipeline.process()``
and ``process_batch()`` -- not just the gate unit (see
``tests/test_table_judge_gate.py`` for the gate-level causal classification
and ``tests/test_table_latch_sidecar.py`` for sparse sidecar persistence).

Design record: docs/log/2026-09-02_gh359-ladder-terminals-design.md, "Panel
and synthesis" -- ``process()`` consults the document-level resume gate
BEFORE any page ledger, so a table page latched UNVERIFIED-because-
unavailable is never re-judged on resume unless the fingerprint changes.
This mirrors PR #518's equation-lane fix
(``tests/test_equation_lane_pipeline_p4r.py``, the ``test_r2_f5_*`` /
``test_r3_f5_*`` tests) -- reuse that shape, do not invent a parallel one.

Contract these tests hold the pipeline to:
  * ``UnifiedPipeline._table_judge_rung_available_now() -> bool`` -- the
    reachability seam. Most tests here patch it directly to isolate the
    resume WIRING; the two ``..._through_the_real_seam`` /
    ``..._does_not_read_as_recovered`` tests deliberately leave it unpatched,
    because cold review round 2 found the wiring correct and the production
    reachability DECISION wrong (a present-but-broken CLI read as recovered).
  * ``_resume_skippable`` grows a table-judge-aware kwarg analogous to
    ``equation_lane_retry_blocks=`` (t7), and ``UnifiedPipeline`` exposes the
    analogous ``_table_judge_retry_blocks_resume()`` predicate the real
    ``process()``/``process_batch()`` call sites use to build it.
  * ``_load_terminal_page`` refuses the D1b REJECTED resume exception when
    the page's table latch is True and the rung is reachable now (t7).
  * Root entries persisted through ``RootIndex.record`` carry
    ``table_judge_retry_pending`` whenever any page's latch is True (t5).

Hermeticity, per CLAUDE.md and the #253/#257 trap: ``_available_engines_for_
agentic`` and ``_resolve_judge_model`` are patched on the pipeline instance;
``route_page`` is stubbed; table rungs are injected via
``_build_table_judge_rungs``; reachability is injected via
``_table_judge_rung_available_now``. No ollama, no ``gemini``/``agy``
binary, no network is reachable from this file. Every assertion is a
DIFFERENCE between two same-fixture runs (or a call-count delta), never an
absolute status tuple pinned from one machine.
"""

from __future__ import annotations

import copy
from pathlib import Path
from unittest.mock import patch

import fitz
import pytest

from socr.core.config import EngineType, PipelineConfig
from socr.core.providers import PROFILE_QWEN_LOCAL
from socr.core.result import PageOutput, PageStatus
from socr.judge.table_verdict import (
    RUNG_KIND_GEMINI,
    Finding,
    FindingCode,
    RungResult,
    TableJudgeVerdict,
)
from socr.pipeline.orchestrator import UnifiedPipeline

_TABLE_MD = (
    "| c0 | c1 | c2 | c3 |\n"
    "| --- | --- | --- | --- |\n"
    "| 10 | 11 | 12 | 13 |\n"
    "| 20 | 21 | 22 | 23 |\n"
    "| 30 | 31 | 32 | 33 |\n"
)


def _ruled_pdf(tmp_path: Path, name: str = "doc.pdf", pages: int = 1) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    for _ in range(pages):
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


def _pass(confidence: str = "high") -> RungResult:
    return RungResult(
        rung="fake",
        ok=True,
        verdict=TableJudgeVerdict(verdict="PASS", confidence=confidence, findings=[]),
    )


def _fail() -> RungResult:
    return RungResult(
        rung="fake",
        ok=True,
        verdict=TableJudgeVerdict(
            verdict="FAIL",
            confidence="high",
            findings=[Finding(code=FindingCode.FABRICATED_VALUE, where="cell", detail="rejects")],
        ),
    )


def _unavailable(rung: str = "fake") -> RungResult:
    """An unreachable rung result.

    ``rung`` matters from cold review round 3 on: the latch records the rung
    KIND that was unavailable, so a test exercising the REAL reachability seam
    must name a real kind ("gemini" / "ollama") rather than the placeholder.
    Tests that patch the seam wholesale can keep the placeholder.
    """
    return RungResult(rung=rung, ok=False, error="simulated transport failure", unavailable=True)


def _health_cli(tmp_path: Path, *, healthy: bool) -> Path:
    """A stub rung-2 CLI at a STABLE path whose health check we control.

    Cold review round 2, finding 1. The path must not change between the two
    runs of a resume test: ``table_judge_rung2_binary`` is bound into the run
    fingerprint, so swapping ``/usr/bin/false`` for ``/usr/bin/true`` would
    reprocess the document for that reason alone and prove nothing about
    reachability. Rewriting the same script in place changes only what the
    health handshake sees.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "stub-agy"
    path.write_text(f"#!/bin/sh\nexit {0 if healthy else 1}\n")
    path.chmod(0o755)
    return path


def _rung1_unreachable():
    """Make the ollama rung's reachability probe answer False, hermetically.

    Only rung 1 is stubbed: rung 2's health check runs for real against the
    coreutils binaries above, which is the decision under test.
    """
    return patch(
        "socr.pipeline.orchestrator.table_judge_ollama_rung_reachable",
        return_value=False,
    )


class _QueueRung:
    """A fake rung callable that returns queued results in order.

    Cold review round 5: it advertises ``rung_kind``, as the production rung
    closures now do. The gate receives opaque callables, so a rung that does
    not say what it is cannot be recognised after it has been rebuilt for the
    next document in a batch. ``rung_kind`` is derived from the queued results
    rather than passed separately, so a fixture cannot claim to be one kind
    while answering as another.
    """

    def __init__(self, results: list[RungResult]) -> None:
        self._results = list(results)
        self.calls: list[tuple] = []
        from socr.judge.table_verdict import rung_kind

        kinds = {rung_kind(r.rung) for r in results if r.rung}
        self.rung_kind = kinds.pop() if len(kinds) == 1 else ""

    def __call__(self, crop_path, markdown, prior_findings):
        self.calls.append((crop_path, markdown, prior_findings))
        if not self._results:
            raise AssertionError("rung called more times than results provided")
        return self._results.pop(0)


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


def _route_page_returning(
    text: str | dict[int, str] = _TABLE_MD,
    engine: str = "qwen",
    routed_pages: list[int] | None = None,
):
    def _fake_route(page_num, ladder, run_provider, judge, **kwargs):
        if routed_pages is not None:
            routed_pages.append(page_num)
        from socr.pipeline.agentic import PageDecision, ProviderAttempt

        page_text = text if isinstance(text, str) else text.get(page_num, _TABLE_MD)
        out = PageOutput(
            page_num=page_num,
            text=page_text,
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


def _process_run(
    pdf_path: Path,
    out_dir: Path,
    *,
    rungs: list | None,
    available: bool | None,
    config_overrides: dict | None = None,
    text: str | dict[int, str] = _TABLE_MD,
    routed_pages: list[int] | None = None,
):
    """One REAL ``process()`` run through the entry path + document resume gate.

    ``available=None`` leaves ``_table_judge_rung_available_now`` UNPATCHED, so
    the production reachability decision itself is under test (cold review
    round 2, finding 1). Callers doing that must make both rung kinds
    hermetic by other means -- see ``_real_seam_patches``.
    """
    pipeline = UnifiedPipeline(_make_config(**(config_overrides or {})))
    patches = [
        patch(
            "socr.pipeline.orchestrator.route_page",
            side_effect=_route_page_returning(text, routed_pages=routed_pages),
        ),
        patch.object(pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]),
        patch.object(pipeline, "_resolve_judge_model", return_value=""),
    ]
    if available is not None:
        patches.append(
            patch.object(pipeline, "_table_judge_rung_available_now", return_value=available)
        )
    if rungs is not None:
        patches.append(patch.object(pipeline, "_build_table_judge_rungs", return_value=rungs))
    from contextlib import ExitStack

    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        return pipeline.process(pdf_path, out_dir)


# ---------------------------------------------------------------------------
# t8 -- single-file process(), both availability directions
# ---------------------------------------------------------------------------


class TestSingleFileRetryLatch:
    def test_unavailable_then_available_re_judges_the_pending_page(self, tmp_path: Path) -> None:
        """C-then-low-PASS (latching) run first with the rung unreachable, then
        rerun once it is reachable: the document must not be skipped whole,
        the recovered rung must actually be called, and completed unaffected
        pages must restore rather than reroute."""
        import json
        from ocr_output_contract import RootIndex

        # 2-page document: page 1 passes cleanly, page 2 encounters unavailable rung
        pdf = _ruled_pdf(tmp_path / "src", pages=2)
        out_dir = tmp_path / "out"

        pending_rung = _QueueRung([_pass("high"), _unavailable(), _pass("low")])
        routed_1: list[int] = []
        result_1 = _process_run(
            pdf, out_dir, rungs=[pending_rung], available=False, routed_pages=routed_1
        )
        assert routed_1 == [1, 2]
        assert pending_rung.calls

        # Page 1 sidecar has no latch; page 2 sidecar has latch; root entry has latch
        sidecar_1 = json.loads(next(out_dir.rglob("pages/00001.json")).read_text())
        assert not sidecar_1.get("table_judge_retry_pending")
        sidecar_2 = json.loads(next(out_dir.rglob("pages/00002.json")).read_text())
        assert sidecar_2.get("table_judge_retry_pending") is True

        root_entry = list(RootIndex(out_dir).files.values())[0]
        assert root_entry.get("table_judge_retry_pending") is True

        recovering_rung = _QueueRung([_pass("high")])
        routed_2: list[int] = []
        result_2 = _process_run(
            pdf, out_dir, rungs=[recovering_rung], available=True, routed_pages=routed_2
        )

        assert result_2.status.value != "skipped", (
            "the document was skipped whole; the pending table page was never re-judged"
        )
        assert recovering_rung.calls, "the recovered rung was never called on the retry run"
        assert routed_2 == [2], (
            "completed unaffected page 1 was rerouted instead of restored from terminal sidecar"
        )

    def test_unavailable_then_still_unavailable_skips_and_does_not_re_run_the_ladder(
        self, tmp_path: Path
    ) -> None:
        """Cold review round 1, finding 1: the document gate refuses the skip
        ONLY when a rung is reachable now. A persistent outage -- latched, then
        still unreachable -- must keep skipping; otherwise every resume re-pays
        timeout x tables x rungs against a rung that is still down."""
        from ocr_output_contract import RootIndex

        pdf = _ruled_pdf(tmp_path / "src", pages=2)
        out_dir = tmp_path / "out"

        pending_rung = _QueueRung([_pass("high"), _unavailable(), _pass("low")])
        routed_1: list[int] = []
        _process_run(pdf, out_dir, rungs=[pending_rung], available=False, routed_pages=routed_1)
        assert routed_1 == [1, 2]

        root_entry = list(RootIndex(out_dir).files.values())[0]
        assert root_entry.get("table_judge_retry_pending") is True

        md_files = [p for p in out_dir.rglob("*.md") if p.parent.name != "pages"]
        assert md_files, "no output markdown was produced"
        before_bytes = md_files[0].read_bytes()

        still_down_rung = _QueueRung([])  # must never be called
        routed_2: list[int] = []
        result_2 = _process_run(
            pdf, out_dir, rungs=[still_down_rung], available=False, routed_pages=routed_2
        )

        assert result_2.status.value == "skipped", (
            "a still-unavailable rung re-opened the document; the ladder would re-run "
            "on every resume of a persistent outage"
        )
        assert still_down_rung.calls == [], "the still-unavailable rung was called again"
        assert routed_2 == [], "pages were re-routed on a still-unavailable resume"
        assert md_files[0].read_bytes() == before_bytes

    def test_latch_survives_a_still_unavailable_skip_and_still_reopens_on_recovery(
        self, tmp_path: Path
    ) -> None:
        """The other half of finding 1: skipping while still down must not
        DISCARD the latch. After a still-unavailable skip, the rung coming back
        must still reopen the document and re-judge the pending page."""
        from ocr_output_contract import RootIndex

        pdf = _ruled_pdf(tmp_path / "src", pages=2)
        out_dir = tmp_path / "out"

        pending_rung = _QueueRung([_pass("high"), _unavailable(), _pass("low")])
        _process_run(pdf, out_dir, rungs=[pending_rung], available=False)

        still_down_rung = _QueueRung([])
        skipped = _process_run(pdf, out_dir, rungs=[still_down_rung], available=False)
        assert skipped.status.value == "skipped"
        assert still_down_rung.calls == []
        assert list(RootIndex(out_dir).files.values())[0].get("table_judge_retry_pending") is True

        recovering_rung = _QueueRung([_pass("high")])
        routed_3: list[int] = []
        result_3 = _process_run(
            pdf, out_dir, rungs=[recovering_rung], available=True, routed_pages=routed_3
        )
        assert result_3.status.value != "skipped"
        assert recovering_rung.calls, "the recovered rung was never called after a skipped resume"
        assert routed_3 == [2]

    def test_a_broken_but_installed_cli_does_not_read_as_recovered(self, tmp_path: Path) -> None:
        """Cold review round 2, finding 1, through the REAL reachability seam.

        The rung-2 binary is present but fails its health check on every run
        (``/usr/bin/false``), and rung 1 is unreachable. That is a PERSISTENT
        outage, not a recovery: the document must stay skipped and the ladder
        must not run again. ``_table_judge_rung_available_now`` is deliberately
        NOT patched here -- patching it is what let the round-1 tests pass over
        this bug."""
        from ocr_output_contract import RootIndex

        pdf = _ruled_pdf(tmp_path / "src", pages=2)
        out_dir = tmp_path / "out"
        cli = _health_cli(tmp_path / "bin", healthy=False)
        broken_cli = {"table_judge_rung2_binary": str(cli)}

        # The latch must name the rung kind the real seam will be asked about.
        pending_rung = _QueueRung([_pass("high"), _unavailable(RUNG_KIND_GEMINI), _pass("low")])
        with _rung1_unreachable():
            _process_run(
                pdf,
                out_dir,
                rungs=[pending_rung],
                available=None,
                config_overrides=broken_cli,
            )
        assert list(RootIndex(out_dir).files.values())[0].get("table_judge_retry_pending") is True

        md_files = [p for p in out_dir.rglob("*.md") if p.parent.name != "pages"]
        assert md_files, "no output markdown was produced"
        before_bytes = md_files[0].read_bytes()

        still_broken_rung = _QueueRung([])  # must never be called
        routed_2: list[int] = []
        with _rung1_unreachable():
            result_2 = _process_run(
                pdf,
                out_dir,
                rungs=[still_broken_rung],
                available=None,
                config_overrides=broken_cli,
                routed_pages=routed_2,
            )

        assert result_2.status.value == "skipped", (
            "an installed-but-broken rung-2 CLI read as recovered; the gate probe and the "
            "rung's own unavailability classification must be the same notion"
        )
        assert still_broken_rung.calls == []
        assert routed_2 == []
        assert md_files[0].read_bytes() == before_bytes

    def test_a_healthy_cli_appearing_does_reopen_through_the_real_seam(
        self, tmp_path: Path
    ) -> None:
        """The control for the test above: with the SAME seam unpatched, a
        rung-2 binary whose health check passes must reopen the document.
        Otherwise the previous test could pass simply by never recovering."""
        pdf = _ruled_pdf(tmp_path / "src", pages=2)
        out_dir = tmp_path / "out"
        cli = _health_cli(tmp_path / "bin", healthy=False)
        same_cli = {"table_judge_rung2_binary": str(cli)}

        # The latch must name the rung kind the real seam will be asked about.
        pending_rung = _QueueRung([_pass("high"), _unavailable(RUNG_KIND_GEMINI), _pass("low")])
        with _rung1_unreachable():
            _process_run(
                pdf,
                out_dir,
                rungs=[pending_rung],
                available=None,
                config_overrides=same_cli,
            )

        # Same path, same fingerprint -- only the health handshake changes.
        _health_cli(tmp_path / "bin", healthy=True)

        recovering_rung = _QueueRung([_pass("high")])
        routed_2: list[int] = []
        with _rung1_unreachable():
            result_2 = _process_run(
                pdf,
                out_dir,
                rungs=[recovering_rung],
                available=None,
                config_overrides=same_cli,
                routed_pages=routed_2,
            )

        assert result_2.status.value != "skipped"
        assert recovering_rung.calls, "a healthy rung-2 CLI did not reopen the latched document"
        assert routed_2 == [2]

    def _rung_kind_seam(self, *, gemini: bool, ollama: bool):
        """Patch BOTH per-kind reachability functions independently.

        Cold review round 3, finding 1: the whole point is that these two
        answers are no longer interchangeable, so a test that wants to say
        "rung 1 is up, rung 2 is down" has to be able to say exactly that.
        """
        from contextlib import ExitStack

        stack = ExitStack()
        stack.enter_context(
            patch(
                "socr.pipeline.orchestrator.table_judge_gemini_rung_reachable",
                return_value=gemini,
            )
        )
        stack.enter_context(
            patch(
                "socr.pipeline.orchestrator.table_judge_ollama_rung_reachable",
                return_value=ollama,
            )
        )
        return stack

    def test_a_healthy_other_rung_does_not_stand_in_for_the_failed_one(
        self, tmp_path: Path
    ) -> None:
        """Cold review round 3, finding 1. Rung 1 answers fine and stays up;
        rung 2 is unavailable and STAYS unavailable. The latch is about rung 2,
        so rung 1 being reachable is not evidence that anything recovered. The
        document must stay skipped."""
        from ocr_output_contract import RootIndex

        pdf = _ruled_pdf(tmp_path / "src")
        out_dir = tmp_path / "out"

        rung1 = _QueueRung([_pass("low")])
        rung2 = _QueueRung([_unavailable(RUNG_KIND_GEMINI)])
        with self._rung_kind_seam(gemini=False, ollama=True):
            _process_run(pdf, out_dir, rungs=[rung1, rung2], available=None)

        entry = list(RootIndex(out_dir).files.values())[0]
        assert entry.get("table_judge_retry_pending") is True
        assert entry.get("table_judge_retry_rungs") == [RUNG_KIND_GEMINI], (
            "the latch must record WHICH rung was unavailable, not just that one was"
        )

        again_1 = _QueueRung([])
        again_2 = _QueueRung([])
        with self._rung_kind_seam(gemini=False, ollama=True):
            result_2 = _process_run(pdf, out_dir, rungs=[again_1, again_2], available=None)

        assert result_2.status.value == "skipped", (
            "a healthy rung 1 was accepted as recovery for a rung 2 that is still down"
        )
        assert again_1.calls == []
        assert again_2.calls == []

    def test_the_failed_rung_coming_back_does_reopen(self, tmp_path: Path) -> None:
        """The control for the test above, changing exactly one thing: the rung
        the latch actually names becomes reachable."""
        pdf = _ruled_pdf(tmp_path / "src")
        out_dir = tmp_path / "out"

        rung1 = _QueueRung([_pass("low")])
        rung2 = _QueueRung([_unavailable(RUNG_KIND_GEMINI)])
        with self._rung_kind_seam(gemini=False, ollama=True):
            _process_run(pdf, out_dir, rungs=[rung1, rung2], available=None)

        again_1 = _QueueRung([_pass("low")])
        again_2 = _QueueRung([_pass("high")])
        with self._rung_kind_seam(gemini=True, ollama=True):
            result_2 = _process_run(pdf, out_dir, rungs=[again_1, again_2], available=None)

        assert result_2.status.value != "skipped"
        assert again_2.calls, "the recovered rung 2 was never called"

    def test_an_old_entry_without_a_rung_list_falls_back_to_any_rung(self, tmp_path: Path) -> None:
        """A latch written before the rung list existed says only "something was
        down". That record cannot answer "which rung", so the gate widens to any
        rung rather than silently never reopening the document."""
        import json

        from ocr_output_contract import RootIndex

        pdf = _ruled_pdf(tmp_path / "src")
        out_dir = tmp_path / "out"

        rung = _QueueRung([_unavailable(RUNG_KIND_GEMINI)])
        with self._rung_kind_seam(gemini=False, ollama=False):
            _process_run(pdf, out_dir, rungs=[rung], available=None)

        # Age the record: drop the rung list, keep the boolean latch.
        index = RootIndex(out_dir)
        rel_key = next(iter(index.files))
        index.files[rel_key].pop("table_judge_retry_rungs", None)
        index.save()
        for sidecar in out_dir.rglob("pages/*.json"):
            meta = json.loads(sidecar.read_text())
            if meta.pop("table_judge_retry_rungs", None) is not None:
                sidecar.write_text(json.dumps(meta, indent=2))

        recovering = _QueueRung([_pass("high")])
        with self._rung_kind_seam(gemini=False, ollama=True):
            result_2 = _process_run(pdf, out_dir, rungs=[recovering], available=None)

        assert result_2.status.value != "skipped", (
            "a pre-rung-list latch must still reopen when some rung is reachable"
        )

    def test_a_quota_refusal_is_not_re_paid_by_the_rest_of_the_run(self, tmp_path: Path) -> None:
        """Cold review round 3, new finding 2. A recognised refusal on a REAL
        call is not a per-table fact: the next table gets the same answer. The
        rung is treated as unreachable for the rest of THIS run, so a second
        page does not pay for it again -- while the latch still persists, so a
        LATER run retries."""
        from ocr_output_contract import RootIndex

        pdf = _ruled_pdf(tmp_path / "src", pages=2)
        out_dir = tmp_path / "out"

        refused = RungResult(
            rung=RUNG_KIND_GEMINI,
            ok=False,
            error="quota exceeded for project",
            unavailable=True,
            refusal=True,
        )
        rung = _QueueRung([refused, refused])
        with self._rung_kind_seam(gemini=True, ollama=False):
            _process_run(pdf, out_dir, rungs=[rung], available=None)

        entry = list(RootIndex(out_dir).files.values())[0]
        assert entry.get("table_judge_retry_pending") is True, (
            "a refusal must still latch: it is transient and a later run should retry"
        )
        assert entry.get("table_judge_retry_rungs") == [RUNG_KIND_GEMINI]

        # A later run, with the rung still reporting healthy: the breaker is
        # per-run state and must NOT have survived it.
        recovering = _QueueRung([_pass("high")])
        with self._rung_kind_seam(gemini=True, ollama=False):
            result_2 = _process_run(pdf, out_dir, rungs=[recovering], available=None)

        assert result_2.status.value != "skipped", (
            "the per-run refusal breaker leaked across the run boundary"
        )
        assert recovering.calls, "the later run never retried the refused rung"

    def test_a_refusal_on_page_one_spares_every_later_page(self, tmp_path: Path) -> None:
        """Cold review round 4, item 6. The breaker used to live only in the
        reachability seam, which is a RESUME decision -- so within one run every
        later page still called the rung that had already refused page 1. Count
        the real calls across a three-page document: exactly one."""
        pdf = _ruled_pdf(tmp_path / "src", pages=3)
        out_dir = tmp_path / "out"

        refusal = RungResult(
            rung=RUNG_KIND_GEMINI,
            ok=False,
            error="quota exceeded",
            unavailable=True,
            refusal=True,
        )
        rung = _QueueRung([refusal, refusal, refusal])
        with self._rung_kind_seam(gemini=True, ollama=False):
            _process_run(pdf, out_dir, rungs=[rung], available=None)

        assert len(rung.calls) == 1, (
            f"pages 2..N re-paid the same refusal ({len(rung.calls)} calls)"
        )

    def test_pages_spared_by_the_breaker_still_latch_the_refused_rung(self, tmp_path: Path) -> None:
        """Sparing a page must not settle it. A refused ladder is transient, so
        the pages that were never called still latch, naming the refused kind,
        and a LATER run retries them."""
        from ocr_output_contract import RootIndex

        pdf = _ruled_pdf(tmp_path / "src", pages=3)
        out_dir = tmp_path / "out"

        refusal = RungResult(
            rung=RUNG_KIND_GEMINI,
            ok=False,
            error="quota exceeded",
            unavailable=True,
            refusal=True,
        )
        rung = _QueueRung([refusal, refusal, refusal])
        with self._rung_kind_seam(gemini=True, ollama=False):
            _process_run(pdf, out_dir, rungs=[rung], available=None)

        entry = list(RootIndex(out_dir).files.values())[0]
        assert entry.get("table_judge_retry_pending") is True
        assert entry.get("table_judge_retry_rungs") == [RUNG_KIND_GEMINI]

        recovering = _QueueRung([_pass("high"), _pass("high"), _pass("high")])
        with self._rung_kind_seam(gemini=True, ollama=False):
            result_2 = _process_run(pdf, out_dir, rungs=[recovering], available=None)

        assert result_2.status.value != "skipped"
        assert recovering.calls, "the pages spared by the breaker were never retried"

    def test_an_equation_forced_reopen_still_honours_the_table_breaker(
        self, tmp_path: Path
    ) -> None:
        """Cold review round 4, the PARTLY-OPEN equation interaction. The
        breaker must hold at the page/rung boundary however the document came
        to be reopened -- including when it is the EQUATION lane's latch, not
        the table lane's, that refused the skip."""
        from ocr_output_contract import RootIndex

        pdf = _ruled_pdf(tmp_path / "src", pages=3)
        out_dir = tmp_path / "out"

        refusal = RungResult(
            rung=RUNG_KIND_GEMINI,
            ok=False,
            error="quota exceeded",
            unavailable=True,
            refusal=True,
        )
        first = _QueueRung([refusal, refusal, refusal])
        with self._rung_kind_seam(gemini=True, ollama=False):
            _process_run(pdf, out_dir, rungs=[first], available=None)
        assert len(first.calls) == 1

        # Age the record into one the TABLE lane alone would skip (its rung is
        # unreachable), and add the equation lane's latch so the equation lane
        # is what reopens it.
        index = RootIndex(out_dir)
        rel_key = next(iter(index.files))
        index.files[rel_key]["equation_lane_retry_pending"] = True
        index.save()

        second = _QueueRung([refusal, refusal, refusal])
        with self._rung_kind_seam(gemini=False, ollama=False):
            result_2 = _process_run(
                pdf,
                out_dir,
                rungs=[second],
                available=None,
                config_overrides={"equation_region_lane": True},
            )

        assert result_2.status.value != "skipped", "the equation latch did not reopen the document"
        assert len(second.calls) == 1, (
            f"an equation-forced reopen re-paid the table refusal on every page "
            f"({len(second.calls)} calls)"
        )

    def test_an_unknown_page_keeps_the_whole_document_unknown(self, tmp_path: Path) -> None:
        """Cold review round 4, new finding 1. UNKNOWN is the top element of the
        document-level union, not the empty set. A latched page restored from a
        record written before the rung list existed means "some rung, we cannot
        say which"; unioning it as nothing narrowed the document to the kinds
        the OTHER pages named, and a recovery of the unnamed rung was then
        missed forever."""
        from ocr_output_contract import RootIndex
        from socr.core.document import DocumentHandle
        from socr.core.result import DocumentStatus, EngineResult
        from socr.core.state import DocumentState

        pdf = _ruled_pdf(tmp_path / "src", pages=2)
        out_dir = tmp_path / "out"
        pipeline = UnifiedPipeline(_make_config())
        state = DocumentState(handle=DocumentHandle.from_path(pdf))
        state.pages[1].table_judge_retry_pending = True
        state.pages[1].table_judge_retry_rungs = []  # pre-kind record
        state.pages[2].table_judge_retry_pending = True
        state.pages[2].table_judge_retry_rungs = [RUNG_KIND_GEMINI]

        pipeline._write_metadata(
            state,
            EngineResult(document_path=pdf, status=DocumentStatus.SUCCESS, engine="qwen"),
            out_dir,
            has_text=True,
        )

        entry = list(RootIndex(out_dir).files.values())[0]
        assert entry.get("table_judge_retry_pending") is True
        assert "table_judge_retry_rungs" not in entry, (
            "one unknown page must keep the document unknown, not narrow it to "
            f"the kinds the other pages named: {entry.get('table_judge_retry_rungs')}"
        )

    def test_an_unknown_document_reopens_when_any_configured_rung_returns(
        self, tmp_path: Path
    ) -> None:
        """The consequence that makes the previous test matter: the widened
        document reopens on a rung no individual page ever named."""
        import json

        from ocr_output_contract import RootIndex

        pdf = _ruled_pdf(tmp_path / "src", pages=2)
        out_dir = tmp_path / "out"

        rung = _QueueRung([_unavailable(RUNG_KIND_GEMINI), _unavailable(RUNG_KIND_GEMINI)])
        with self._rung_kind_seam(gemini=False, ollama=False):
            _process_run(pdf, out_dir, rungs=[rung], available=None)

        # Age page 1's sidecar and the root entry into pre-kind records.
        index = RootIndex(out_dir)
        rel_key = next(iter(index.files))
        index.files[rel_key].pop("table_judge_retry_rungs", None)
        index.save()
        first_sidecar = next(out_dir.rglob("pages/00001.json"))
        meta = json.loads(first_sidecar.read_text())
        meta.pop("table_judge_retry_rungs", None)
        first_sidecar.write_text(json.dumps(meta, indent=2))

        recovering = _QueueRung([_pass("high"), _pass("high")])
        with self._rung_kind_seam(gemini=False, ollama=True):
            result_2 = _process_run(pdf, out_dir, rungs=[recovering], available=None)

        assert result_2.status.value != "skipped", (
            "an unknown-kind document did not reopen when a configured rung returned"
        )

    def test_a_refusal_spares_every_later_file_in_the_same_batch(self, tmp_path: Path) -> None:
        """Cold review round 5, item 2. "Per run" means per public call, and one
        ``process_batch`` is one run. The rungs are rebuilt per document, so
        callable identity alone could not recognise the refused rung in file 2;
        the rung now advertises its kind, and an UNSEEN callable of a kind that
        already refused us this run is the same rung rebuilt."""
        input_dir = tmp_path / "in"
        input_dir.mkdir()
        _ruled_pdf(input_dir, "a.pdf")
        _ruled_pdf(input_dir, "b.pdf")

        refusal = RungResult(
            rung=RUNG_KIND_GEMINI,
            ok=False,
            error="quota exceeded",
            unavailable=True,
            refusal=True,
        )
        built: list[_QueueRung] = []

        def _fresh_rung():
            rung = _QueueRung([refusal])
            built.append(rung)
            return [rung]

        pipeline = UnifiedPipeline(_make_config())
        with (
            self._rung_kind_seam(gemini=True, ollama=False),
            patch("socr.pipeline.orchestrator.route_page", side_effect=_route_page_returning()),
            patch.object(
                pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]
            ),
            patch.object(pipeline, "_resolve_judge_model", return_value=""),
            patch.object(pipeline, "_build_table_judge_rungs", side_effect=_fresh_rung),
        ):
            pipeline.process_batch(input_dir, tmp_path / "out")

        assert len(built) == 2, "the batch did not process both files"
        assert sum(len(rung.calls) for rung in built) == 1, (
            "the same refused kind was called again after its callable was rebuilt "
            "for the next document in the same batch"
        )

    def test_a_second_batch_retries_the_refused_rung(self, tmp_path: Path) -> None:
        """The other half: the breaker is per RUN, so the next
        ``process_batch`` starts a fresh epoch and retries."""
        input_dir = tmp_path / "in"
        input_dir.mkdir()
        _ruled_pdf(input_dir, "a.pdf")
        _ruled_pdf(input_dir, "b.pdf")
        out_dir = tmp_path / "out"

        refusal = RungResult(
            rung=RUNG_KIND_GEMINI,
            ok=False,
            error="quota exceeded",
            unavailable=True,
            refusal=True,
        )
        rungs_1 = {"shared": [_QueueRung([refusal, refusal])]}
        _process_batch_run(input_dir, out_dir, rungs_by_file=rungs_1, available=True)

        recovering = _QueueRung([_pass("high"), _pass("high")])
        results_2, _ = _process_batch_run(
            input_dir, out_dir, rungs_by_file={"shared": [recovering]}, available=True
        )

        assert results_2, "the refused files were not readmitted by the next run"
        assert recovering.calls, "the next batch never retried the refused rung"

    def test_a_healthy_sibling_of_a_refused_rung_is_not_dropped(self, tmp_path: Path) -> None:
        """The rule that makes the batch fix safe: identity beats kind for a
        callable this run has already called without a refusal. Two same-kind
        callables, one of which refused -- the other must survive."""
        from socr.judge.table_ladder import run_table_ladder

        pipeline = UnifiedPipeline(_make_config())
        pipeline._reset_table_judge_rung_probes()

        healthy = _QueueRung([RungResult(rung=RUNG_KIND_GEMINI, ok=False, error="parse defect")])
        refuser = _QueueRung(
            [
                RungResult(
                    rung=RUNG_KIND_GEMINI,
                    ok=False,
                    error="quota exceeded",
                    unavailable=True,
                    refusal=True,
                )
            ]
        )
        crop = tmp_path / "crop.png"
        crop.write_bytes(b"png")
        result = run_table_ladder([healthy, refuser], crop, "md", "t0")
        pipeline._record_table_rung_refusals([healthy, refuser], result)

        # Both results name the same kind, so only identity can tell them
        # apart -- and reordering must not change the answer.
        assert pipeline._live_table_judge_rungs([refuser, healthy]) == [healthy]

    def test_the_audit_trail_names_the_rung_that_actually_ran(self, tmp_path: Path) -> None:
        """Cold review round 5, new finding 1. The trail mapped result index 0
        to rung 1's model and index 1 to rung 2's binary. That held only while
        every run called the configured ladder in order; once the breaker could
        hand it a filtered sublist, a lone surviving rung 2 was recorded as
        having been executed by rung 1's model. False provenance in a citation
        corpus's audit trail is exactly what this trail exists to prevent."""
        from socr.core.audit_log import AuditEvent
        from socr.core.document import DocumentHandle
        from socr.core.result import PageOutput, PageStatus
        from socr.core.state import DocumentState
        from socr.judge.table_ladder import TableLadderOutcome, TableLadderResult
        from socr.judge.table_verdict import TABLE_LADDER_ACCEPTED_KIND

        pdf = _ruled_pdf(tmp_path / "src")
        pipeline = UnifiedPipeline(_make_config())
        pipeline._reset_table_judge_rung_probes()

        rung1 = _QueueRung([])  # refused earlier in this run; never called again
        rung2 = _QueueRung(
            [
                RungResult(
                    rung=RUNG_KIND_GEMINI,
                    ok=True,
                    verdict=TableJudgeVerdict(verdict="PASS", confidence="high", findings=[]),
                )
            ]
        )
        pipeline._record_table_rung_refusals(
            [rung1],
            TableLadderResult(
                table_id="earlier",
                outcome=TableLadderOutcome.UNVERIFIED,
                rung_results=[
                    RungResult(
                        rung="ollama:glm-5.3-flash:cloud",
                        ok=False,
                        error="quota exceeded",
                        unavailable=True,
                        refusal=True,
                    )
                ],
            ),
        )

        state = DocumentState(handle=DocumentHandle.from_path(pdf))
        ps = state.pages[1]
        bo = PageOutput(
            page_num=1,
            text=_TABLE_MD,
            status=PageStatus.SUCCESS,
            engine="qwen",
            audit_passed=True,
        )
        pipeline._run_table_judge_gate(state, 1, ps, bo, [rung1, rung2])

        event = next(
            e
            for e in state.events
            if isinstance(e, AuditEvent) and e.kind == TABLE_LADDER_ACCEPTED_KIND
        )
        trail = event.data["rung_trail"]
        assert len(trail) == 1, trail
        assert trail[0]["rung"] == RUNG_KIND_GEMINI
        assert trail[0]["executing"] == pipeline.config.table_judge_rung2_binary, (
            f"the filtered ladder's sole executor was mislabelled: {trail}"
        )
        assert rung1.calls == [], "the refused rung was called again"

    def test_a_synthesized_refused_result_names_its_own_kind(self, tmp_path: Path) -> None:
        """A result the breaker synthesized had no executor at all, and the
        synthesized list is sorted by kind, so its index was never a ladder
        position. It must still resolve to the configured identity for the kind
        it names."""
        from socr.core.audit_log import AuditEvent
        from socr.core.document import DocumentHandle
        from socr.core.result import PageOutput, PageStatus
        from socr.core.state import DocumentState
        from socr.judge.table_ladder import TableLadderOutcome, TableLadderResult
        from socr.judge.table_verdict import TABLE_LADDER_UNVERIFIED_KIND

        pdf = _ruled_pdf(tmp_path / "src")
        pipeline = UnifiedPipeline(_make_config())
        pipeline._reset_table_judge_rung_probes()

        rung = _QueueRung([])
        pipeline._record_table_rung_refusals(
            [rung],
            TableLadderResult(
                table_id="earlier",
                outcome=TableLadderOutcome.UNVERIFIED,
                rung_results=[
                    RungResult(
                        rung=RUNG_KIND_GEMINI,
                        ok=False,
                        error="quota exceeded",
                        unavailable=True,
                        refusal=True,
                    )
                ],
            ),
        )

        state = DocumentState(handle=DocumentHandle.from_path(pdf))
        ps = state.pages[1]
        bo = PageOutput(
            page_num=1,
            text=_TABLE_MD,
            status=PageStatus.SUCCESS,
            engine="qwen",
            audit_passed=True,
        )
        pipeline._run_table_judge_gate(state, 1, ps, bo, [rung])

        event = next(
            e
            for e in state.events
            if isinstance(e, AuditEvent) and e.kind == TABLE_LADDER_UNVERIFIED_KIND
        )
        trail = event.data["rung_trail"]
        assert [row["rung"] for row in trail] == [RUNG_KIND_GEMINI], trail
        assert trail[0]["executing"] == pipeline.config.table_judge_rung2_binary, trail
        assert rung.calls == []

    def test_the_refusal_breaker_holds_within_one_run(self, tmp_path: Path) -> None:
        """The other half: inside ONE run, once the rung has refused, the gate
        stops reporting it reachable."""
        pdf = _ruled_pdf(tmp_path / "src")
        out_dir = tmp_path / "out"

        pipeline = UnifiedPipeline(_make_config())
        pipeline._reset_table_judge_rung_probes()
        with self._rung_kind_seam(gemini=True, ollama=False):
            assert pipeline._table_judge_rung_available_now([RUNG_KIND_GEMINI]) is True

            from socr.judge.table_ladder import TableLadderOutcome, TableLadderResult

            pipeline._note_table_rung_refusals(
                [
                    TableLadderResult(
                        table_id="t0",
                        outcome=TableLadderOutcome.UNVERIFIED,
                        rung_results=[
                            RungResult(
                                rung=RUNG_KIND_GEMINI,
                                ok=False,
                                error="quota exceeded",
                                unavailable=True,
                                refusal=True,
                            )
                        ],
                    )
                ]
            )
            assert pipeline._table_judge_rung_available_now([RUNG_KIND_GEMINI]) is False, (
                "a rung that refused us this run must not be reported reachable again in it"
            )

    def test_the_reachability_cache_is_per_run_not_per_pipeline(self, tmp_path: Path) -> None:
        """Cold review round 3, new finding 1. The cache must hold WITHIN a run
        and be dropped AT a run boundary, so one reused pipeline object cannot
        carry a stale "unreachable" into its next run and never see recovery."""
        pdf = _ruled_pdf(tmp_path / "src")
        out_dir = tmp_path / "out"

        pipeline = UnifiedPipeline(_make_config())
        probes: list[str] = []
        reachable = {"value": False}

        def _probe(kind):
            probes.append(kind)
            return reachable["value"]

        with (
            patch("socr.pipeline.orchestrator.route_page", side_effect=_route_page_returning()),
            patch.object(
                pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]
            ),
            patch.object(pipeline, "_resolve_judge_model", return_value=""),
            patch.object(pipeline, "_probe_table_judge_rung_kind", side_effect=_probe),
            patch.object(
                pipeline,
                "_build_table_judge_rungs",
                return_value=[_QueueRung([_unavailable(RUNG_KIND_GEMINI)])],
            ),
        ):
            pipeline.process(pdf, out_dir)

            # Within one run the answer is cached: repeated asks probe once.
            pipeline._table_judge_rung_kind_available_now(RUNG_KIND_GEMINI)
            after_first_ask = len(probes)
            pipeline._table_judge_rung_kind_available_now(RUNG_KIND_GEMINI)
            pipeline._table_judge_rung_kind_available_now(RUNG_KIND_GEMINI)
            assert len(probes) == after_first_ask, "the cache did not hold within a run"

            # A NEW run must ask again, and must be able to see recovery.
            reachable["value"] = True
            pipeline._build_table_judge_rungs.return_value = [_QueueRung([_pass("high")])]
            result_2 = pipeline.process(pdf, out_dir)

        assert len(probes) > after_first_ask, (
            "a second run reused the previous run's reachability answer"
        )
        assert result_2.status.value != "skipped", (
            "the reused pipeline never observed the rung coming back"
        )

    def test_available_then_unavailable_restores_without_reprocessing(self, tmp_path: Path) -> None:
        """The other direction: a clean completed run, then the rung going
        away must RESTORE the existing document, not reroute/reprocess it,
        without calling the unavailable rung or changing output bytes."""
        import json
        from ocr_output_contract import RootIndex

        pdf = _ruled_pdf(tmp_path / "src")
        out_dir = tmp_path / "out"

        clean_rung = _QueueRung([_pass("high")])
        result_1 = _process_run(pdf, out_dir, rungs=[clean_rung], available=True)
        assert result_1.status.value != "skipped"
        assert clean_rung.calls

        root_entry = list(RootIndex(out_dir).files.values())[0]
        assert not root_entry.get("table_judge_retry_pending")

        for p in out_dir.rglob("pages/*.json"):
            meta = json.loads(p.read_text())
            assert not meta.get("table_judge_retry_pending")

        md_files = [p for p in out_dir.rglob("*.md") if p.parent.name != "pages"]
        assert md_files, "no output markdown was produced"
        before_bytes = md_files[0].read_bytes()

        gone_rung = _QueueRung([])  # must never be called
        result_2 = _process_run(pdf, out_dir, rungs=[gone_rung], available=False)

        assert result_2.status.value == "skipped", (
            "a finished, non-latched document was reprocessed"
        )
        assert gone_rung.calls == []
        assert md_files[0].read_bytes() == before_bytes, "output bytes changed on skipped restore"

    def test_content_only_rejected_does_not_reopen_when_reachability_changes(
        self, tmp_path: Path
    ) -> None:
        """A REJECTED page (both judges looked and said no, no unavailable
        rung involved) must not carry a latch -- so a later reachability
        change must not reopen the document."""
        import json
        from ocr_output_contract import RootIndex

        pdf = _ruled_pdf(tmp_path / "src")
        out_dir = tmp_path / "out"

        reject_rung = _QueueRung([_fail()])
        _process_run(pdf, out_dir, rungs=[reject_rung], available=False)

        root_entry = list(RootIndex(out_dir).files.values())[0]
        assert not root_entry.get("table_judge_retry_pending")

        for p in out_dir.rglob("pages/*.json"):
            meta = json.loads(p.read_text())
            assert not meta.get("table_judge_retry_pending")

        never_called = _QueueRung([])
        result_2 = _process_run(pdf, out_dir, rungs=[never_called], available=True)

        assert result_2.status.value == "skipped", (
            "a content-only REJECTED page has no rung-unavailable cause and must not latch"
        )
        assert never_called.calls == []

    def test_content_only_unverified_does_not_reopen_when_reachability_changes(
        self, tmp_path: Path
    ) -> None:
        """Two low PASSes with no unavailable rung: ruling 1's quorum accepts,
        so there is nothing pending regardless of reachability -- a control
        against the previous test using the OTHER non-latching shape."""
        import json
        from ocr_output_contract import RootIndex

        pdf = _ruled_pdf(tmp_path / "src")
        out_dir = tmp_path / "out"

        content_not_s1_rung = _QueueRung(
            [RungResult(rung="fake", ok=False, error="no JSON object found", unavailable=False)]
        )
        _process_run(pdf, out_dir, rungs=[content_not_s1_rung], available=False)

        root_entry = list(RootIndex(out_dir).files.values())[0]
        assert not root_entry.get("table_judge_retry_pending")

        for p in out_dir.rglob("pages/*.json"):
            meta = json.loads(p.read_text())
            assert not meta.get("table_judge_retry_pending")

        never_called = _QueueRung([])
        result_2 = _process_run(pdf, out_dir, rungs=[never_called], available=True)

        assert result_2.status.value == "skipped", (
            "a parse-failure ¬S1 (content-shaped, not rung-unavailable) must not latch"
        )
        assert never_called.calls == []

    def test_flag_off_never_touches_the_reachability_seam(self, tmp_path: Path) -> None:
        """The table_judge_ladder flag stays default-off in this task -- a
        flag-off run must never even consult reachability, must never carry a
        latch, and must repeat byte-identically."""
        import json
        from ocr_output_contract import RootIndex

        pdf = _ruled_pdf(tmp_path / "src")
        out_dir = tmp_path / "out"

        with patch.object(
            UnifiedPipeline,
            "_table_judge_rung_available_now",
            side_effect=AssertionError("reachability probed while the ladder flag is off"),
        ):
            result_1 = _process_run(
                pdf,
                out_dir,
                rungs=None,
                available=False,
                config_overrides={"table_judge_ladder": False},
            )
        assert result_1.status.value != "skipped"

        root_entry = list(RootIndex(out_dir).files.values())[0]
        assert not root_entry.get("table_judge_retry_pending")

        for p in out_dir.rglob("pages/*.json"):
            meta = json.loads(p.read_text())
            assert not meta.get("table_judge_retry_pending")

        md_files = [p for p in out_dir.rglob("*.md") if p.parent.name != "pages"]
        assert md_files, "no output markdown was produced"
        before_bytes = md_files[0].read_bytes()

        with patch.object(
            UnifiedPipeline,
            "_table_judge_rung_available_now",
            side_effect=AssertionError("reachability probed while the ladder flag is off"),
        ):
            result_2 = _process_run(
                pdf,
                out_dir,
                rungs=None,
                available=False,
                config_overrides={"table_judge_ladder": False},
            )
        assert result_2.status.value == "skipped"
        assert md_files[0].read_bytes() == before_bytes

    def test_root_index_snapshots_never_persist_a_completed_or_partial_entry_without_the_latch(
        self, tmp_path: Path
    ) -> None:
        """PR #518's finding 5, reused: the latch and the terminal record must
        land in ONE write. Spy on every RootIndex.save and require every
        resumable snapshot taken during the unavailable run already carries
        the latch."""
        from ocr_output_contract import RootIndex

        pdf = _ruled_pdf(tmp_path / "src")
        out_dir = tmp_path / "out"

        snapshots: list[dict] = []
        real_save = RootIndex.save

        def _spy_save(self):
            real_save(self)
            snapshots.append(copy.deepcopy(self.files))

        pending_rung = _QueueRung([_unavailable(), _pass("low")])
        with patch.object(RootIndex, "save", _spy_save):
            _process_run(pdf, out_dir, rungs=[pending_rung], available=False)

        persisted = [
            entry
            for snap in snapshots
            for entry in snap.values()
            if entry.get("status") in ("completed", "partial")
        ]
        assert persisted, "no resumable root entry was ever written"
        for entry in persisted:
            assert entry.get("table_judge_retry_pending") is True, (
                f"a resumable root entry ({entry.get('status')}) was persisted without "
                "the pending-retry latch"
            )

    def test_stale_entry_does_not_become_valid_again_after_a_failed_final_save(
        self, tmp_path: Path
    ) -> None:
        """Mirrors PR #518's stale-entry regression: once a retry is admitted,
        pre-run invalidation must stop an older skippable entry from becoming
        valid again if the FINAL index save on the retry run fails."""
        from ocr_output_contract import RootIndex

        pdf = _ruled_pdf(tmp_path / "src")
        out_dir = tmp_path / "out"

        pending_rung = _QueueRung([_unavailable(), _pass("low")])
        _process_run(pdf, out_dir, rungs=[pending_rung], available=False)

        recovering_rung = _QueueRung([_pass("high")])
        real_save = RootIndex.save
        save_count = 0

        def _fail_final_save(self):
            nonlocal save_count
            save_count += 1
            if save_count > 1:
                raise OSError("simulated final index write failure")
            return real_save(self)

        with patch.object(RootIndex, "save", _fail_final_save):
            try:
                _process_run(pdf, out_dir, rungs=[recovering_rung], available=True)
            except OSError:
                pass

        never_called = _QueueRung([])
        result_after = _process_run(pdf, out_dir, rungs=[never_called], available=False)
        assert result_after.status.value != "skipped", (
            "a failed final save on the retry run left a stale skippable entry behind"
        )

    def test_seeded_stale_entry_cannot_survive_a_failed_terminal_write_mirroring_pr518(
        self, tmp_path: Path
    ) -> None:
        """Explicitly mirror PR #518's r4_f5: seed a matching older entry --
        completed, latch-free, its output since deleted. An unavailable run
        re-emits output but fails its terminal write. The pre-run invalidation
        must ensure that entry does not survive to skip future runs."""
        from ocr_output_contract import (
            DocMetadata,
            RootIndex,
            Status,
            doc_dir_for,
            markdown_path_for,
            relative_key,
            safe_checksum,
            utc_timestamp,
        )

        pdf = _ruled_pdf(tmp_path / "stale_src")
        out_dir = tmp_path / "stale_out"
        out_dir.mkdir(parents=True, exist_ok=True)

        pipeline = UnifiedPipeline(_make_config())
        with patch.object(pipeline, "_resolve_judge_model", return_value=""):
            fingerprint = pipeline._run_fingerprint()
        rel_key = relative_key(pdf, pdf.parent)
        md_path = markdown_path_for(doc_dir_for(out_dir, rel_key), rel_key)

        # Seed a completed, latch-free record whose output is not yet on disk.
        RootIndex(out_dir).record(
            rel_key,
            DocMetadata(
                status=Status.COMPLETED,
                checksum=safe_checksum(pdf),
                model="qwen",
                backend="socr",
                processing_time=1.0,
                timestamp=utc_timestamp(),
                output_path=str(md_path),
                pages=1,
                fingerprint=fingerprint,
            ),
        )
        assert RootIndex(out_dir).files[rel_key].get("table_judge_retry_pending") is None
        assert not md_path.exists()

        # A run encounters unavailable rung, re-emits output, but fails its terminal index save.
        pending_rung = _QueueRung([_unavailable(), _pass("low")])
        with patch.object(RootIndex, "save", side_effect=OSError("simulated index write failure")):
            try:
                _process_run(pdf, out_dir, rungs=[pending_rung], available=False)
            except OSError:
                pass

        # Next run: must NOT skip using the stale latch-free entry.
        never_called = _QueueRung([])
        result_after = _process_run(pdf, out_dir, rungs=[never_called], available=False)
        assert result_after.status.value != "skipped", (
            "a stale latch-free root entry survived the failed write and skipped the document"
        )

    def test_mixed_multi_table_rejected_plus_unavailable_reopens_and_re_judges(
        self, tmp_path: Path
    ) -> None:
        """A page whose reducer verdict is TABLE_REJECTED (one table rejected)
        but which also carries a second, unresolved unavailable table must
        still be reopened on retry -- proving the mixed-page latch case end
        to end, not just at the gate unit."""
        from ocr_output_contract import RootIndex

        pdf = _ruled_pdf(tmp_path / "src")
        out_dir = tmp_path / "out"
        two_table_md = _TABLE_MD + "\nprose between tables\n\n" + _TABLE_MD

        mixed_rung = _QueueRung([_fail(), _unavailable()])
        _process_run(pdf, out_dir, rungs=[mixed_rung], available=False, text=two_table_md)

        root_entry = list(RootIndex(out_dir).files.values())[0]
        assert root_entry.get("table_judge_retry_pending") is True

        recovering_rung = _QueueRung([_pass("high"), _pass("high")])
        result_2 = _process_run(
            pdf, out_dir, rungs=[recovering_rung], available=True, text=two_table_md
        )
        assert result_2.status.value != "skipped", (
            "a mixed REJECTED+unavailable page did not reopen when the rung recovered"
        )
        assert recovering_rung.calls, "the unresolved second table was never re-judged"


# ---------------------------------------------------------------------------
# t9 -- process_batch() pre-gate, both availability directions
# ---------------------------------------------------------------------------


def _process_batch_run(
    input_dir: Path,
    out_dir: Path,
    *,
    rungs_by_file: dict[str, list] | None = None,
    available: bool = True,
    config_overrides: dict | None = None,
):
    pipeline = UnifiedPipeline(_make_config(**(config_overrides or {})))
    probe_calls: list[bool] = []
    current_pdf: list[Path] = []
    orig_process = pipeline.process

    def _available_now(rung_kinds=None):
        probe_calls.append(True)
        return available

    def _spy_process(pdf_path, output_dir=None, **kwargs):
        current_pdf.append(Path(pdf_path))
        return orig_process(pdf_path, output_dir, **kwargs)

    def _make_rungs():
        if not current_pdf:
            return []
        pdf_name = current_pdf[-1].name
        if rungs_by_file is not None:
            if pdf_name in rungs_by_file:
                return rungs_by_file[pdf_name]
            if "shared" in rungs_by_file:
                return rungs_by_file["shared"]
        return []

    with (
        patch("socr.pipeline.orchestrator.route_page", side_effect=_route_page_returning()),
        patch.object(pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]),
        patch.object(pipeline, "_resolve_judge_model", return_value=""),
        patch.object(pipeline, "_table_judge_rung_available_now", side_effect=_available_now),
        patch.object(pipeline, "_build_table_judge_rungs", side_effect=_make_rungs),
        patch.object(pipeline, "process", side_effect=_spy_process),
    ):
        results = pipeline.process_batch(input_dir, out_dir)
    return results, probe_calls


class TestBatchRetryLatch:
    """``process_batch``'s pre-gate calls ``_resume_skippable`` for every
    candidate BEFORE any file reaches ``process()`` -- these tests exercise
    that pre-gate through the real batch entry point, reusing the single-file
    fixture rather than looping ``process()`` manually (t9 requirement)."""

    def test_unavailable_then_available_moves_the_latched_file_into_to_process(
        self, tmp_path: Path
    ) -> None:
        """Unavailable then available moves the latched file into to_process
        and re-judges its pending page, while an unlatched control file remains
        skipped and byte-identical."""
        import json
        from ocr_output_contract import RootIndex

        input_dir = tmp_path / "in"
        input_dir.mkdir()
        _latched_pdf = _ruled_pdf(input_dir, "latched.pdf", pages=2)
        _control_pdf = _ruled_pdf(input_dir, "control.pdf", pages=1)
        out_dir = tmp_path / "out"

        pending_rung = _QueueRung([_pass("high"), _unavailable(), _pass("low")])
        control_clean_rung = _QueueRung([_pass("high")])
        rungs_1 = {
            "latched.pdf": [pending_rung],
            "control.pdf": [control_clean_rung],
        }
        results_1, _ = _process_batch_run(
            input_dir, out_dir, rungs_by_file=rungs_1, available=False
        )
        assert len(results_1) == 2

        # Verify latched vs unlatched root entries and sidecars after run 1
        root_files = RootIndex(out_dir).files
        assert root_files["latched.pdf"].get("table_judge_retry_pending") is True
        assert not root_files["control.pdf"].get("table_judge_retry_pending")

        control_md = out_dir / "control" / "control.md"
        assert control_md.exists(), "control markdown output was not created"
        control_before_bytes = control_md.read_bytes()

        # Run 2: recovered rung for latched; control must never reach judge
        recovering_rung = _QueueRung([_pass("high")])
        control_never_called = _QueueRung([])
        rungs_2 = {
            "latched.pdf": [recovering_rung],
            "control.pdf": [control_never_called],
        }
        results_2, _ = _process_batch_run(input_dir, out_dir, rungs_by_file=rungs_2, available=True)

        assert len(results_2) == 1, (
            f"expected only latched.pdf in to_process, got {[r.document_path.name for r in results_2]}"
        )
        assert results_2[0].document_path.name == "latched.pdf"
        assert results_2[0].status.value != "skipped"
        assert recovering_rung.calls, "the recovered rung was never called on retry"
        assert control_never_called.calls == [], (
            "unlatched control file reached process() instead of being skipped at pre-gate"
        )
        assert control_md.read_bytes() == control_before_bytes, (
            "unlatched control markdown output bytes changed"
        )

    def test_still_unavailable_leaves_the_latched_file_skipped_at_the_pre_gate(
        self, tmp_path: Path
    ) -> None:
        """Cold review round 1, finding 1, through the batch entry path: a
        latched file whose rung is STILL unreachable must stay skipped at the
        pre-gate, never reaching process() and never calling a rung."""
        from ocr_output_contract import RootIndex

        input_dir = tmp_path / "in"
        input_dir.mkdir()
        _ruled_pdf(input_dir, "latched.pdf", pages=2)
        out_dir = tmp_path / "out"

        pending_rung = _QueueRung([_pass("high"), _unavailable(), _pass("low")])
        results_1, _ = _process_batch_run(
            input_dir, out_dir, rungs_by_file={"latched.pdf": [pending_rung]}, available=False
        )
        assert len(results_1) == 1
        assert RootIndex(out_dir).files["latched.pdf"].get("table_judge_retry_pending") is True

        latched_md = out_dir / "latched" / "latched.md"
        assert latched_md.exists(), "latched markdown output was not created"
        before_bytes = latched_md.read_bytes()

        still_down_rung = _QueueRung([])
        results_2, _ = _process_batch_run(
            input_dir, out_dir, rungs_by_file={"latched.pdf": [still_down_rung]}, available=False
        )
        assert results_2 == [], (
            "a still-unavailable latched file was admitted into to_process by the batch pre-gate"
        )
        assert still_down_rung.calls == []
        assert latched_md.read_bytes() == before_bytes

    def test_available_then_unavailable_leaves_the_completed_file_skipped(
        self, tmp_path: Path
    ) -> None:
        """A completed, non-latched document leaves the file skipped at the
        batch pre-gate when reachability is lost, with no rung calls and exact
        byte identity."""
        from ocr_output_contract import RootIndex

        input_dir = tmp_path / "in"
        input_dir.mkdir()
        _pdf = _ruled_pdf(input_dir, "doc.pdf")
        out_dir = tmp_path / "out"

        clean_rung = _QueueRung([_pass("high")])
        results_1, _ = _process_batch_run(
            input_dir, out_dir, rungs_by_file={"doc.pdf": [clean_rung]}, available=True
        )
        assert len(results_1) == 1
        assert results_1[0].status.value != "skipped"
        assert clean_rung.calls

        root_entry = list(RootIndex(out_dir).files.values())[0]
        assert not root_entry.get("table_judge_retry_pending")

        md_files = [p for p in out_dir.rglob("*.md") if p.parent.name != "pages"]
        assert md_files, "no output markdown was produced"
        before_bytes = md_files[0].read_bytes()

        gone_rung = _QueueRung([])
        results_2, _ = _process_batch_run(
            input_dir, out_dir, rungs_by_file={"doc.pdf": [gone_rung]}, available=False
        )

        assert results_2 == [], "a completed, non-latched batch file was reprocessed"
        assert gone_rung.calls == []
        assert md_files[0].read_bytes() == before_bytes, "output bytes changed on skipped restore"

    def test_reachability_is_not_probed_for_a_batch_with_no_latched_candidate(
        self, tmp_path: Path
    ) -> None:
        """Ordinary/flag-off batches must not pay a network-probe cost per
        file scanned -- the probe fires only when a candidate root entry
        actually carries the latch."""
        input_dir = tmp_path / "in"
        input_dir.mkdir()
        _ruled_pdf(input_dir, "a.pdf")
        _ruled_pdf(input_dir, "b.pdf")
        out_dir = tmp_path / "out"

        clean_rung_factory = lambda: _QueueRung([_pass("high")])
        pipeline = UnifiedPipeline(_make_config())

        def _boom():
            raise AssertionError("reachability probed with no latched candidate in the batch")

        with (
            patch("socr.pipeline.orchestrator.route_page", side_effect=_route_page_returning()),
            patch.object(
                pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]
            ),
            patch.object(pipeline, "_resolve_judge_model", return_value=""),
            patch.object(
                pipeline, "_build_table_judge_rungs", side_effect=lambda: [clean_rung_factory()]
            ),
        ):
            pipeline.process_batch(input_dir, out_dir)

        with (
            patch("socr.pipeline.orchestrator.route_page", side_effect=_route_page_returning()),
            patch.object(
                pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]
            ),
            patch.object(pipeline, "_resolve_judge_model", return_value=""),
            patch.object(
                pipeline, "_build_table_judge_rungs", side_effect=lambda: [clean_rung_factory()]
            ),
            patch.object(pipeline, "_table_judge_rung_available_now", side_effect=_boom),
        ):
            # rerun over the now-completed, non-latched documents
            results = pipeline.process_batch(input_dir, out_dir)
        assert results == []

    def test_reachability_is_not_repeatedly_probed_per_file_in_batch_scan(
        self, tmp_path: Path
    ) -> None:
        """A probe spy proves reachability is not checked for unlatched batch
        entries and is memoized across the pre-gate scan rather than probed
        repeatedly per latched file."""
        from ocr_output_contract import RootIndex

        input_dir = tmp_path / "in"
        input_dir.mkdir()
        _latched1 = _ruled_pdf(input_dir, "latched1.pdf")
        _latched2 = _ruled_pdf(input_dir, "latched2.pdf")
        _unlatched = _ruled_pdf(input_dir, "unlatched.pdf")
        out_dir = tmp_path / "out"

        # Run 1: 2 files encounter unavailable rung, 1 file passes cleanly
        pending_1 = _QueueRung([_unavailable(), _pass("low")])
        pending_2 = _QueueRung([_unavailable(), _pass("low")])
        clean = _QueueRung([_pass("high")])
        rungs_1 = {
            "latched1.pdf": [pending_1],
            "latched2.pdf": [pending_2],
            "unlatched.pdf": [clean],
        }
        results_1, _ = _process_batch_run(
            input_dir, out_dir, rungs_by_file=rungs_1, available=False
        )
        assert len(results_1) == 3

        root_files = RootIndex(out_dir).files
        assert root_files["latched1.pdf"].get("table_judge_retry_pending") is True
        assert root_files["latched2.pdf"].get("table_judge_retry_pending") is True
        assert not root_files["unlatched.pdf"].get("table_judge_retry_pending")

        # Run 2 (available=True): 2 latched files move into to_process, unlatched file skipped.
        # A probe spy tracks calls made during the pre-gate scan versus during process().
        pre_gate_probe_calls: list[bool] = []
        process_started = False

        pipeline = UnifiedPipeline(_make_config())
        recovering_1 = _QueueRung([_pass("high")])
        recovering_2 = _QueueRung([_pass("high")])
        unlatched_never_called = _QueueRung([])
        rungs_2 = {
            "latched1.pdf": [recovering_1],
            "latched2.pdf": [recovering_2],
            "unlatched.pdf": [unlatched_never_called],
        }
        current_pdf: list[Path] = []
        orig_process = pipeline.process

        def _spy_probe(kind):
            # Cold review round 3: the invariant is per-RUNG-KIND, not per-call.
            # The pre-gate now asks a per-file question (each entry names the
            # kinds IT is waiting on), so counting calls to the public predicate
            # counts files. What must not repeat is the actual probe.
            if not process_started:
                pre_gate_probe_calls.append(kind)
            return True

        def _spy_process(pdf_path, output_dir=None, **kwargs):
            nonlocal process_started
            process_started = True
            current_pdf.append(Path(pdf_path))
            return orig_process(pdf_path, output_dir, **kwargs)

        def _make_rungs():
            if not current_pdf:
                return []
            pdf_name = current_pdf[-1].name
            return rungs_2.get(pdf_name, [])

        with (
            patch("socr.pipeline.orchestrator.route_page", side_effect=_route_page_returning()),
            patch.object(
                pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]
            ),
            patch.object(pipeline, "_resolve_judge_model", return_value=""),
            patch.object(pipeline, "_probe_table_judge_rung_kind", side_effect=_spy_probe),
            patch.object(pipeline, "_build_table_judge_rungs", side_effect=_make_rungs),
            patch.object(pipeline, "process", side_effect=_spy_process),
        ):
            results = pipeline.process_batch(input_dir, out_dir)

        # Two latched files scanned at the pre-gate: each rung KIND is probed at
        # most once for the whole batch, however many files ask about it.
        assert len(pre_gate_probe_calls) == len(set(pre_gate_probe_calls)), (
            f"a rung kind was probed more than once during the batch scan: {pre_gate_probe_calls}"
        )
        assert {r.document_path.name for r in results} == {"latched1.pdf", "latched2.pdf"}
        assert recovering_1.calls, "latched1 was not re-judged"
        assert recovering_2.calls, "latched2 was not re-judged"
        assert unlatched_never_called.calls == [], "unlatched file was reprocessed"


# t9 also requires running this module alongside
# tests/test_equation_lane_pipeline_p4r.py in the same pytest invocation, to
# prove the shared root-metadata abstraction (t5) left equation-lane
# semantics unchanged -- e.g.:
#   PYTHONPATH=src ~/venvs/socr/bin/pytest tests/test_p1_ladder_retry_latch.py \
#     tests/test_equation_lane_pipeline_p4r.py -q
# That is a verification-run concern (both suites passing together), not a
# single assertion this file can encode.
