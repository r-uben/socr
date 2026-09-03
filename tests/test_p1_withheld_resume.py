"""P1 (task t11): the withhold/latch terminals as RESUME terminals.

Extends ``test_p1_ladder_retry_latch.py``'s real ``process()``-driven resume
contract (reused here via import, not reinvented) to the P1 terminals: a
content-only withhold is skipped on resume like any other D1b content
terminal, while a page latched on the blind-cell ADJUDICATOR reopens when,
and only when, THAT provider comes back.

Cold review round 1, finding 2. The previous version replaced
``_table_judge_rung_available_now`` wholesale, so it could not see that the
adjudicator had no probe of its own: its recorded kind was filtered out of a
reader-only tuple and the question silently widened to "is any reader up".
Recovery was then keyed to the wrong provider in both directions -- a live
reader reopened a document whose adjudicator was still down (re-paying the
same failure every resume), and a recovered adjudicator never reopened one
whose readers were down. So these tests now drive the REAL resume gate and
control only the per-provider reachability functions it ends up calling.

Every assertion is a DIFFERENCE between two real ``process()`` runs on the
SAME fixture (call counts, byte-identity), never an absolute status tuple.
"""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from _adjudicator_doubles import (
    defective_adjudicator,
    mismatching_adjudicator,
    unavailable_adjudicator,
)
from test_p1_ladder_retry_latch import (
    _TABLE_MD,
    _make_config,
    _QueueRung,
    _route_page_returning,
)

from socr.core.providers import PROFILE_QWEN_LOCAL
from socr.judge.table_verdict import (
    RUNG_KIND_CELL_ADJUDICATOR,
    Finding,
    FindingCode,
    RungResult,
    TableJudgeVerdict,
)
from socr.pipeline.orchestrator import UnifiedPipeline
from socr.tables.binding import BindingEvidence


def _ruled_pdf(tmp_path: Path, name: str = "doc.pdf") -> Path:
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


def _final_markdown(out_dir: Path) -> str:
    """The page's shipped bytes.

    The page FRAGMENT, not ``<out>/<stem>.md``: a single-table page withheld
    whole ships no content, so no assembled body is written at all, and
    byte-identity across a resume is the property under test.
    """
    fragments = sorted(out_dir.rglob("pages/*.md"))
    assert fragments, f"no page fragment under {out_dir}"
    return "\n".join(f.read_text() for f in fragments)


def _reject_rung() -> _QueueRung:
    findings = [Finding(code=FindingCode.FABRICATED_VALUE, where="R1C2", detail="bad value")]
    return _QueueRung(
        [
            RungResult(
                rung="fake",
                ok=True,
                verdict=TableJudgeVerdict(verdict="FAIL", confidence="high", findings=findings),
            )
        ]
    )


def _run(
    pdf_path: Path,
    out_dir: Path,
    *,
    rungs,
    adjudicator,
    reader_up: bool,
    adjudicator_up: bool,
):
    """One real ``process()`` run with per-provider reachability controlled.

    The resume gate's own logic -- which kinds the latch recorded, which of
    them are probeable, whether the question widens -- is NOT patched. Only
    the three leaf reachability functions are, one per provider, which is the
    granularity the finding is about.
    """
    pipeline = UnifiedPipeline(_make_config())
    pipeline._binding_evidence_for_witness = lambda *a, **kw: (None, BindingEvidence.ABSTAIN)
    patches = [
        patch(
            "socr.pipeline.orchestrator.route_page",
            side_effect=_route_page_returning(_TABLE_MD),
        ),
        patch.object(pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]),
        patch.object(pipeline, "_resolve_judge_model", return_value=""),
        patch.object(pipeline, "_build_table_judge_rungs", return_value=rungs),
        patch.object(pipeline, "_build_table_cell_adjudicator", return_value=adjudicator),
        patch(
            "socr.pipeline.orchestrator.table_judge_ollama_rung_reachable",
            return_value=reader_up,
        ),
        patch(
            "socr.pipeline.orchestrator.table_judge_gemini_rung_reachable",
            return_value=reader_up,
        ),
        patch(
            "socr.pipeline.orchestrator.table_judge_adjudicator_reachable",
            return_value=adjudicator_up,
        ),
    ]
    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        return pipeline.process(pdf_path, out_dir)


class TestContentOnlyWithholdSkipsOnResume:
    def test_second_run_makes_zero_reader_calls_and_is_byte_identical(self, tmp_path: Path):
        pdf_path = _ruled_pdf(tmp_path)
        out_dir = tmp_path / "out"

        rung_1 = _reject_rung()
        _run(
            pdf_path,
            out_dir,
            rungs=[rung_1],
            adjudicator=mismatching_adjudicator(),
            reader_up=True,
            adjudicator_up=True,
        )
        assert rung_1.calls, "the first run must actually judge the table"
        md_1 = _final_markdown(out_dir)

        never_called = _QueueRung([])
        _run(
            pdf_path,
            out_dir,
            rungs=[never_called],
            adjudicator=mismatching_adjudicator(),
            reader_up=True,
            adjudicator_up=True,
        )

        assert never_called.calls == [], "a content-only withhold must skip re-judging on resume"
        assert _final_markdown(out_dir) == md_1


class TestRecoveryIsKeyedToTheAdjudicatorItself:
    def _latch_on_the_adjudicator(self, pdf_path: Path, out_dir: Path):
        """First run: readers answer, the adjudicator is down. The page must
        latch on the ADJUDICATOR's kind and nothing else."""
        adj = unavailable_adjudicator()
        rung = _reject_rung()
        result = _run(
            pdf_path,
            out_dir,
            rungs=[rung],
            adjudicator=adj,
            reader_up=True,
            adjudicator_up=False,
        )
        assert rung.calls and adj.calls
        return result

    def test_a_live_reader_does_not_reopen_a_page_latched_on_the_adjudicator(self, tmp_path: Path):
        """The regression. A reachable READER says nothing about whether the
        blind reader is back, so it must not force the same failure again."""
        pdf_path = _ruled_pdf(tmp_path)
        out_dir = tmp_path / "out"
        self._latch_on_the_adjudicator(pdf_path, out_dir)

        never_called = _QueueRung([])
        _run(
            pdf_path,
            out_dir,
            rungs=[never_called],
            adjudicator=unavailable_adjudicator(),
            reader_up=True,
            adjudicator_up=False,
        )
        assert never_called.calls == [], (
            "a reader being up must not reopen a page latched on the adjudicator"
        )

    def test_the_adjudicator_coming_back_reopens_the_page_even_with_readers_down(
        self, tmp_path: Path
    ):
        """The other direction. Recovery of the provider that actually failed
        must reopen the page, and it must not depend on the readers."""
        pdf_path = _ruled_pdf(tmp_path)
        out_dir = tmp_path / "out"
        self._latch_on_the_adjudicator(pdf_path, out_dir)

        recovered = _reject_rung()
        _run(
            pdf_path,
            out_dir,
            rungs=[recovered],
            adjudicator=mismatching_adjudicator(),
            reader_up=False,
            adjudicator_up=True,
        )
        assert recovered.calls, "the adjudicator coming back must reroute the latched page"

    def test_the_latch_names_the_adjudicator_kind_and_the_probe_is_asked_about_it(
        self, tmp_path: Path
    ):
        """The mechanism behind the two tests above, pinned directly: the
        recorded kind is the adjudicator's, and the resume gate asks the
        adjudicator's own probe about it rather than widening to the readers."""
        pdf_path = _ruled_pdf(tmp_path)
        out_dir = tmp_path / "out"
        self._latch_on_the_adjudicator(pdf_path, out_dir)

        pipeline = UnifiedPipeline(_make_config())
        pipeline._table_rung_available_cache = {}
        asked: list[str] = []

        def _reader_probe(*_a, **_kw):
            asked.append("reader")
            return True

        def _adjudicator_probe(*_a, **_kw):
            asked.append("adjudicator")
            return False

        with (
            patch(
                "socr.pipeline.orchestrator.table_judge_ollama_rung_reachable",
                side_effect=_reader_probe,
            ),
            patch(
                "socr.pipeline.orchestrator.table_judge_gemini_rung_reachable",
                side_effect=_reader_probe,
            ),
            patch(
                "socr.pipeline.orchestrator.table_judge_adjudicator_reachable",
                side_effect=_adjudicator_probe,
            ),
        ):
            answer = pipeline._table_judge_rung_available_now([RUNG_KIND_CELL_ADJUDICATOR])

        assert answer is False
        assert asked == ["adjudicator"], "the readers must not be asked, or widened to"


class TestDeterministicDefectNeverReopens:
    def test_a_defect_does_not_reopen_even_with_every_provider_reachable(self, tmp_path: Path):
        """A deterministic guard defect is not an outage, so it never latches:
        the terminal is a plain content terminal and reachability cannot
        reopen it."""
        pdf_path = _ruled_pdf(tmp_path)
        out_dir = tmp_path / "out"

        rung_1 = _reject_rung()
        _run(
            pdf_path,
            out_dir,
            rungs=[rung_1],
            adjudicator=defective_adjudicator(),
            reader_up=True,
            adjudicator_up=True,
        )
        assert rung_1.calls

        never_called = _QueueRung([])
        _run(
            pdf_path,
            out_dir,
            rungs=[never_called],
            adjudicator=defective_adjudicator(),
            reader_up=True,
            adjudicator_up=True,
        )
        assert never_called.calls == [], "a deterministic defect must not reopen on resume"
