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


def _flush(tmp_path: Path, body: str, *, native_fallback: str | None = None):
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
        )
    )
    state = DocumentState(handle=DocumentHandle.from_path(pdf))

    if native_fallback is not None:
        # A REJECTED OCR attempt with a native-text fallback available: the
        # selection the sidecar makes is not `ps.best_output`, which is the
        # divergence cubic P2 on #549 named.
        state.pages[1].is_born_digital = True
        state.pages[1].native_text = native_fallback

    def _routed(page_num, *_a, **_k):
        rejected = native_fallback is not None
        return PageDecision(
            page_num=page_num,
            final_output=PageOutput(
                page_num=page_num,
                text=body,
                status=PageStatus.SUCCESS,
                engine="qwen",
                audit_passed=not rejected,
            ),
            accepted=not rejected,
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

    assert fragment.strip() == sidecar["winning_output"]["text"].strip(), (
        "the fragment and sidecar disagree on a page whose shipped winner is "
        "not best_output:\n"
        f"fragment={fragment!r}\nsidecar ={sidecar['winning_output']['text']!r}"
    )
