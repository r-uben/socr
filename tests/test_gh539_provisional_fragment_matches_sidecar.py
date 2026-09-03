"""GH-539: the provisional fragment and its sidecar must agree.

The in-loop crash-recovery flush writes two files side by side: `pages/NNNNN.md`
and `pages/NNNNN.json`. The sidecar finalises its winning output through
`_apply_table_emission_guard`; the fragment wrote `bo.text` raw.

So for a PASSING output whose body holds a width-mismatched GFM table, the
fragment held the invalid table while the sidecar beside it said
`error / table_emission_invalid` with the marker text. `_rewrite_all_fragments`
corrects the fragment at assemble time, so the final `.md` was always right --
but a crash between the two writes left two files on disk contradicting each
other, which is the one thing a crash-recovery copy exists to prevent.

Measured as an EQUALITY between the two provisional artefacts rather than
against a fixed string, so the pin holds whatever the guard decides to emit.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")

from socr.core.config import EngineType, PipelineConfig  # noqa: E402
from socr.core.document import DocumentHandle  # noqa: E402
from socr.core.result import PageOutput, PageStatus  # noqa: E402
from socr.core.state import DocumentState  # noqa: E402
from socr.pipeline.orchestrator import UnifiedPipeline  # noqa: E402

#: Three header cells, two delimiter cells: a width mismatch the emission guard
#: hard-fails. Deliberately wrapped in prose, so a fragment that merely dropped
#: the table would still differ from one that carries the marker.
_INVALID_TABLE_BODY = "Prose above.\n\n| a | b | c |\n|---|---|\n| 1 | 2 | 3 |\n\nProse below.\n"


def _flush(
    tmp_path: Path,
    body: str,
    *,
    native_fallback: str | None = None,
    whole_doc: str | None = None,
):
    """Drive the REAL in-loop provisional flush inside `_phase_agentic`.

    Building the guarded text in the test and writing the two files by hand
    would prove only that the guard is deterministic: removing the production
    call left that version green. The flush is inline in `_phase_agentic`, so
    the loop is what has to run.
    """
    from unittest.mock import patch

    from socr.core.providers import PROFILE_QWEN_LOCAL
    from socr.pipeline.agentic import PageDecision

    tmp_path.mkdir(parents=True, exist_ok=True)
    pdf = tmp_path / "d.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "a text layer long enough to be a real one.")
    doc.save(pdf)
    doc.close()

    pipeline = UnifiedPipeline(
        PipelineConfig(
            primary_engine=EngineType.QWEN,
            enabled_engines=[EngineType.QWEN],
            quiet=True,
            native_first=False,
            # P1 (owner ruling Q3, 2026-09-03): the table-judge ladder is ON by
            # default and fail-closed, so a table page on a machine with no
            # reachable rung ships UNVERIFIED and the document is no longer
            # SUCCESS. This test is about a different lane, so the flag is
            # PINNED off rather than the assertion weakened
            # (docs/log/2026-09-03_p1-ladder-flip.md, "Test audit").
            table_judge_ladder=False,
        )
    )
    state = DocumentState(handle=DocumentHandle.from_path(pdf))

    if whole_doc is not None:
        # A whole-document CLI attempt: `_whole_doc_page_texts` recovers this
        # page's text by splitting it, and the sidecar selects from THAT. If the
        # fragment's selection is not given the same snapshot, the two describe
        # different winners.
        state.whole_doc_attempts.append(
            PageOutput(
                page_num=0,
                text=whole_doc,
                status=PageStatus.SUCCESS,
                engine="qwen",
                audit_passed=True,
            )
        )

    if native_fallback is not None:
        # A REJECTED OCR attempt with a native-text fallback available: the
        # selection the sidecar makes is not `ps.best_output`, which is the
        # divergence cubic P2 on #549 named.
        state.pages[1].is_born_digital = True
        state.pages[1].native_text = native_fallback

    def _routed(page_num, *_a, **_k):
        rejected = native_fallback is not None
        # The whole-document branch is consulted only when the per-page attempt
        # FAILED -- a passing one wins outright -- so that case routes a failure.
        failed = whole_doc is not None
        return PageDecision(
            page_num=page_num,
            final_output=PageOutput(
                page_num=page_num,
                text="" if failed else body,
                status=PageStatus.ERROR if failed else PageStatus.SUCCESS,
                engine="qwen",
                audit_passed=not (rejected or failed),
            ),
            accepted=not (rejected or failed),
        )

    out_dir = tmp_path / "out"
    with (
        patch("socr.pipeline.orchestrator.route_page", side_effect=_routed),
        patch.object(pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]),
        patch.object(pipeline, "_resolve_judge_model", return_value=""),
    ):
        pipeline._phase_agentic(state, out_dir)

    fragments = list(out_dir.rglob("pages/00001.md"))
    sidecars = list(out_dir.rglob("pages/00001.json"))
    assert fragments and sidecars, (
        f"the in-loop flush wrote nothing: {sorted(str(q) for q in out_dir.rglob('*'))}"
    )
    return fragments[0].read_text(), json.loads(sidecars[0].read_text())


def test_the_provisional_fragment_matches_its_sidecar(tmp_path: Path) -> None:
    """The shape the ticket names: a passing page carrying an invalid table."""
    fragment, sidecar = _flush(tmp_path / "invalid", _INVALID_TABLE_BODY)

    winner = sidecar["winning_output"]["text"]
    assert fragment.strip() == winner.strip(), (
        "the crash-recovery fragment and the sidecar written beside it disagree:\n"
        f"fragment={fragment!r}\nsidecar ={winner!r}"
    )
    assert sidecar["status"] == "error", (
        "the fixture no longer trips the emission guard, so this test compares "
        f"two copies of an unguarded body: {sidecar['status']!r}"
    )
    assert "| a | b | c |" not in fragment, (
        "the fragment still carries the raw invalid table the guard rejected"
    )


def test_a_clean_page_is_untouched(tmp_path: Path) -> None:
    """Control: the guard must not rewrite a body that has no defect.

    Without this, a change that replaced every fragment with the marker text
    would satisfy the equality above.
    """
    clean = "Prose above.\n\n| a | b |\n| --- | --- |\n| 1 | 2 |\n\nProse below.\n"
    fragment, sidecar = _flush(tmp_path / "clean", clean)

    assert sidecar["status"] == "success", f"the control page was failed: {sidecar['status']!r}"
    assert "| a | b |" in fragment, "a valid table was stripped from the fragment"
    assert fragment.strip() == sidecar["winning_output"]["text"].strip()


def test_a_page_whose_winner_is_not_best_output_still_agrees(tmp_path: Path) -> None:
    """cubic P2 on #549: the same guard on a DIFFERENT output is two finalisations.

    `ps.best_output` is the raw per-page attempt; the sidecar SELECTS its winner.
    They diverge exactly in the fallback cases `_flush_page_sidecar` documents --
    a rejected OCR attempt overridden by a flagged native-text fallback. Guarding
    `best_output` and selecting separately would leave the two artefacts
    disagreeing again, for a different reason.

    Sourcing both from `finalized_page_record` is what makes the equality hold
    whatever the selection decides.
    """
    fragment, sidecar = _flush(
        tmp_path / "fallback",
        _INVALID_TABLE_BODY,
        native_fallback="Native prose recovered from the text layer.",
    )

    # The scenario has to be REAL (cubic P3 on #549). Both artefacts now come
    # from `finalized_page_record`, so the equality holds whenever the two paths
    # finalise the same record -- INCLUDING if selection regressed and the
    # rejected best_output won both while the native fallback never engaged.
    # The test would stay green while the divergence it is named for went
    # unexercised.
    winner = sidecar["winning_output"]["text"]
    assert "Native prose recovered" in winner, (
        f"the native fallback never shipped, so the winner IS best_output and "
        f"this test measures nothing: {winner!r}"
    )
    assert "| a | b | c |" not in winner, "the rejected attempt's table shipped anyway"

    assert fragment.strip() == winner.strip(), (
        "the fragment and sidecar disagree on a page whose shipped winner is "
        "not best_output:\n"
        f"fragment={fragment!r}\nsidecar ={winner!r}"
    )


def test_a_whole_document_attempt_selects_the_same_winner(tmp_path: Path) -> None:
    """cubic P1 on #549: same function, same page, DIFFERENT inputs.

    `finalized_page_record` takes `whole_doc` as an argument. Omitting it makes
    the fragment's selection reconsider a CLI whole-document attempt differently
    from the sidecar, which passes `_whole_doc_page_texts(state)` -- two
    selections again, one argument down from the previous round.
    """
    fragment, sidecar = _flush(
        tmp_path / "whole_doc",
        _INVALID_TABLE_BODY,
        whole_doc="## Page 1\n\nRecovered from the whole-document attempt.\n",
    )

    winner = sidecar["winning_output"]["text"]
    assert "Recovered from the whole-document attempt" in winner, (
        f"the whole-document recovery never engaged, so this test measures nothing: {winner!r}"
    )
    assert fragment.strip() == winner.strip(), (
        "the fragment and sidecar describe different winners on a page with a "
        f"whole-document attempt:\nfragment={fragment!r}\nsidecar ={winner!r}"
    )


def test_one_provisional_flush_selects_once(tmp_path: Path) -> None:
    """GH-550: agreement by construction, not by coincidence.

    GH-539 made the fragment and the sidecar agree by calling
    `finalized_page_record` in both. Same function, same inputs -- so they
    agreed, but only because someone kept the two call sites aligned by hand.
    #549 spent three rounds on exactly that class: same function one argument
    apart, then the same guard on a different output. A second call is a second
    chance to diverge.

    Counting the calls is the only way to pin "once". An equality assertion
    cannot tell one selection from two that happen to match today, which is
    precisely the state this refactor removes.
    """
    from unittest.mock import patch

    import socr.core.manifest as manifest_mod

    calls: list[int] = []
    real = manifest_mod.finalized_page_record

    def _counting(state, page_num, whole_doc=None, saved_text=None):
        calls.append(page_num)
        return real(state, page_num, whole_doc, saved_text)

    with patch.object(manifest_mod, "finalized_page_record", _counting):
        fragment, sidecar = _flush(tmp_path / "once", _INVALID_TABLE_BODY)

    assert calls == [1], (
        f"the provisional flush selected {len(calls)} time(s) for one page: "
        f"{calls}. Two selections agree only until their inputs drift."
    )
    # The agreement still holds -- the point is that it now holds by
    # construction rather than by two calls matching.
    assert fragment.strip() == sidecar["winning_output"]["text"].strip()
