"""GH-373: count-mismatch AMBIGUOUS is judged against the full page.

The spec is ``docs/log/2026-09-01_gh373-degraded-witness.md``. Helper-unit
coverage of witness preparation lives in ``test_table_witness.py`` and of
the prompt slot in ``test_table_prompt.py``. This file pins DIFFERENCE at
``UnifiedPipeline.process()``: a count-mismatch page is judged, not
abstained. A helper-unit test going green while the agentic ``if`` is
commented out, or while the gate still treats AMBIGUOUS as ¬S1, is not a
content gate.

Hermetic: CI has no ollama and no ``gemini`` binary. Rungs are injected.
``_available_engines_for_agentic`` is patched; ``_resolve_judge_model`` is
patched to ``""``. Never pin an absolute outcome measured on one machine
(#253/#257): every assertion is a same-process difference.
"""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import fitz

from socr.core.config import EngineType, PipelineConfig
from socr.core.providers import PROFILE_QWEN_LOCAL
from socr.core.result import PageOutput, PageStatus
from socr.judge.table_prompt import build_table_judge_prompt
from socr.judge.table_verdict import Finding, FindingCode, RungResult, TableJudgeVerdict
from _adjudicator_doubles import mismatching_adjudicator
from socr.pipeline.orchestrator import UnifiedPipeline
from socr.tables.binding import BindingEvidence

_TABLE_MD = (
    "| c0 | c1 | c2 | c3 |\n"
    "| --- | --- | --- | --- |\n"
    "| 10 | 11 | 12 | 13 |\n"
    "| 20 | 21 | 22 | 23 |\n"
    "| 30 | 31 | 32 | 33 |\n"
)
_TWO_TABLE_MD = _TABLE_MD + "\nprose between tables\n\n" + _TABLE_MD

_PAGE_SCOPE_PHRASE = "multiple tables may be visible"


def _ruled_pdf(tmp_path: Path, name: str = "doc.pdf") -> Path:
    """One page, one ruled table -> exactly 1 located box."""
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


def _two_ruled_swapped_pdf(tmp_path: Path, name: str = "doc.pdf") -> Path:
    """Two ruled tables with non-overlapping x-spans (so locate finds 2 boxes).

    Cell values are distinct per table so corroboration can see a swap when
    the emitted markdown is in reverse content order of geometry.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    page = doc.new_page()
    top_cols = [100, 220]
    bottom_cols = [340, 460]
    top_rows = [100 + i * 22 for i in range(2)]
    bottom_rows = [320 + i * 22 for i in range(2)]
    top_values = ["111", "222", "333", "444"]
    bottom_values = ["555", "666", "777", "888"]
    for cols, rows, values in (
        (top_cols, top_rows, top_values),
        (bottom_cols, bottom_rows, bottom_values),
    ):
        x0, x1 = cols[0], cols[-1] + 80
        it = iter(values)
        for y in rows:
            for x in cols:
                page.insert_text((x + 4, y + 12), next(it), fontsize=9)
        for yy in rows:
            page.draw_line((x0, yy), (x1, yy))
        for xx in cols + [x1]:
            page.draw_line((xx, rows[0]), (xx, rows[-1]))
    pdf_path = tmp_path / name
    doc.save(pdf_path)
    doc.close()
    return pdf_path


_SWAPPED_MD = (
    "| c | d |\n| --- | --- |\n| 555 | 666 |\n| 777 | 888 |\n"
    "\n"
    "prose between the two tables\n"
    "\n"
    "| a | b |\n| --- | --- |\n| 111 | 222 |\n| 333 | 444 |\n"
)


def _pass(confidence: str = "high") -> RungResult:
    return RungResult(
        rung="fake1",
        ok=True,
        verdict=TableJudgeVerdict(verdict="PASS", confidence=confidence, findings=[]),
    )


def _fail() -> RungResult:
    return RungResult(
        rung="fake1",
        ok=True,
        verdict=TableJudgeVerdict(
            verdict="FAIL",
            confidence="high",
            findings=[
                # GH-575 (cold review round 1, finding 1): a CANONICAL cell
                # reference, because a reader rejection now reaches a withhold
                # only through the blind adjudicator, and the adjudicator is
                # only asked about cells the readers localized. A bare
                # ``"header"`` names no coordinate, so the chain would end
                # UNVERIFIED for want of a question -- the no-doubt-set path,
                # not the count-mismatch rejection this file measures.
                Finding(
                    code=FindingCode.HEADER_MANGLED,
                    where="H1C1",
                    detail="spanning header missing",
                )
            ],
        ),
    )


class _RepeatRung:
    """Returns the same result on every call and records crop/markdown/prompt.

    Prompt is built the way the real rungs do (``build_table_judge_prompt``
    with no scope argument) so a missing ``table_judge_prompt_scope`` wrap
    in the gate makes the page-scope fragment absent and this test red.
    """

    def __init__(self, result: RungResult) -> None:
        self._result = result
        self.calls: list[tuple] = []
        self.prompts: list[str] = []

    def __call__(self, crop_path, markdown, prior_findings):
        self.calls.append((crop_path, markdown, prior_findings))
        self.prompts.append(build_table_judge_prompt(markdown, prior_findings))
        return self._result


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


def _process(
    pipeline: UnifiedPipeline,
    pdf_path: Path,
    out_dir: Path,
    rungs: list,
    text: str,
):
    patches = [
        patch(
            "socr.pipeline.orchestrator.route_page",
            side_effect=_route_page_returning(text),
        ),
        patch.object(pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]),
        patch.object(pipeline, "_resolve_judge_model", return_value=""),
        patch.object(pipeline, "_build_table_judge_rungs", return_value=rungs),
        patch.object(pipeline, "_binding_contradiction_for_witness", return_value=None),
        # GH-575: the binding EVIDENCE is a terminal in its own right now, so
        # isolating the clamp alone no longer isolates bind() from the readers'
        # verdict; and a reader rejection is withheld only when a blind third
        # reader looked and disagreed.
        patch.object(
            pipeline,
            "_binding_evidence_for_witness",
            return_value=(None, BindingEvidence.ABSTAIN),
        ),
        patch.object(
            pipeline, "_build_table_cell_adjudicator", return_value=mismatching_adjudicator()
        ),
    ]
    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        return pipeline.process(pdf_path, out_dir)


def _rejected(error: str | None) -> bool:
    # P1 (owner ruling Q2, 2026-09-03): a reader rejection no guard clears is
    # now reported as ``table_withheld`` -- the same reader verdict with the
    # table's bytes withheld. This file pins WHERE the ladder ends on a
    # degraded (count-mismatch) witness, not which of the two labels it
    # carries, so the predicate accepts both.
    text = error or ""
    return "table_rejected" in text or "table_withheld" in text


def _unverified(error: str | None) -> bool:
    return "table_unverified" in (error or "")


class TestCountMismatchIsJudged:
    def test_rejecting_ladder_on_count_mismatch_is_rejected_not_unverified(
        self, tmp_path: Path
    ) -> None:
        """Same process(), only the flag and the injected rung change.

        Count mismatch used to abstain to UNVERIFIED. After GH-373 a FAIL
        on the page image is REJECTED — the HEADER_MANGLED catch. If the
        gate still treats AMBIGUOUS as ¬S1, or the agentic ``if`` is
        commented out (assemble backfills UNVERIFIED), this goes red.
        """
        pdf_off = _ruled_pdf(tmp_path / "off", "doc.pdf")
        pdf_on = _ruled_pdf(tmp_path / "on", "doc.pdf")

        result_off = _process(
            UnifiedPipeline(_make_config(table_judge_ladder=False)),
            pdf_off,
            tmp_path / "off_out",
            [_RepeatRung(_fail())],
            _TWO_TABLE_MD,
        )
        result_on = _process(
            UnifiedPipeline(_make_config()),
            pdf_on,
            tmp_path / "on_out",
            [_RepeatRung(_fail())],
            _TWO_TABLE_MD,
        )

        assert _rejected(result_off.error) is False
        assert _unverified(result_off.error) is False
        assert _rejected(result_on.error) is True
        assert _unverified(result_on.error) is False

    def test_accepting_count_mismatch_differs_from_missing(self, tmp_path: Path) -> None:
        """Count mismatch (1 box, 2 blocks) is judged; MISSING (0 boxes)
        still abstains. Same accepting rung, same process(), only the
        locator geometry changes."""
        mismatch = _process(
            UnifiedPipeline(_make_config()),
            _ruled_pdf(tmp_path / "mismatch", "doc.pdf"),
            tmp_path / "mismatch_out",
            [_RepeatRung(_pass("high"))],
            _TWO_TABLE_MD,
        )
        missing = _process(
            UnifiedPipeline(_make_config()),
            _borderless_pdf(tmp_path / "missing", "doc.pdf"),
            tmp_path / "missing_out",
            [_RepeatRung(_pass("high"))],
            _TABLE_MD,
        )

        assert _unverified(mismatch.error) is False
        assert _rejected(mismatch.error) is False
        assert _unverified(missing.error) is True
        assert _rejected(missing.error) is False

    def test_accepting_count_mismatch_differs_from_corroboration_contradicted(
        self, tmp_path: Path
    ) -> None:
        """Only count-mismatch AMBIGUOUS gets the degraded look.
        Corroboration-contradicted AMBIGUOUS keeps abstaining."""
        mismatch = _process(
            UnifiedPipeline(_make_config()),
            _ruled_pdf(tmp_path / "mismatch", "doc.pdf"),
            tmp_path / "mismatch_out",
            [_RepeatRung(_pass("high"))],
            _TWO_TABLE_MD,
        )
        contradicted = _process(
            UnifiedPipeline(_make_config()),
            _two_ruled_swapped_pdf(tmp_path / "swap", "doc.pdf"),
            tmp_path / "swap_out",
            [_RepeatRung(_pass("high"))],
            _SWAPPED_MD,
        )

        assert _unverified(mismatch.error) is False
        assert _unverified(contradicted.error) is True
        assert _rejected(contradicted.error) is False

    def test_lone_low_pass_on_page_scope_stays_unverified(self, tmp_path: Path) -> None:
        """GH-359 ruling 1 is not rewritten: a lone low-confidence PASS on
        the page image is UNVERIFIED, same as on a located crop."""
        high = _process(
            UnifiedPipeline(_make_config()),
            _ruled_pdf(tmp_path / "high", "doc.pdf"),
            tmp_path / "high_out",
            [_RepeatRung(_pass("high"))],
            _TWO_TABLE_MD,
        )
        low = _process(
            UnifiedPipeline(_make_config()),
            _ruled_pdf(tmp_path / "low", "doc.pdf"),
            tmp_path / "low_out",
            [_RepeatRung(_pass("low"))],
            _TWO_TABLE_MD,
        )

        assert _unverified(high.error) is False
        assert _unverified(low.error) is True
        assert _rejected(low.error) is False

    def test_page_scope_fragment_reaches_the_prompt_the_rung_would_send(
        self, tmp_path: Path
    ) -> None:
        """Difference: count-mismatch (page scope) vs 1:1 located (no note).

        The spy rung builds the prompt the way CLI1/CLI2 do. If the gate
        stops wrapping ``run_table_ladder`` in ``table_judge_prompt_scope``,
        both prompts lack the fragment and this goes red.
        """
        page_rung = _RepeatRung(_pass("high"))
        located_rung = _RepeatRung(_pass("high"))

        _process(
            UnifiedPipeline(_make_config()),
            _ruled_pdf(tmp_path / "page", "doc.pdf"),
            tmp_path / "page_out",
            [page_rung],
            _TWO_TABLE_MD,
        )
        _process(
            UnifiedPipeline(_make_config()),
            _ruled_pdf(tmp_path / "located", "doc.pdf"),
            tmp_path / "located_out",
            [located_rung],
            _TABLE_MD,
        )

        assert page_rung.prompts, "count-mismatch must reach the judge"
        assert located_rung.prompts, "located table must reach the judge"
        assert any(_PAGE_SCOPE_PHRASE in p.lower() for p in page_rung.prompts)
        assert all(_PAGE_SCOPE_PHRASE not in p.lower() for p in located_rung.prompts)
        assert all("{{SCOPE_NOTE}}" not in p for p in page_rung.prompts)
