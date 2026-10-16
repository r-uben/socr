"""#1177: the saved document ``.md`` is the assembly of the FINAL page records.

``_phase_assemble`` saves the document from ``pre_records``. ``_preserve_chart_regions`` runs
after those are built and is the only place ``chart_region_placement_unresolved`` is set, so
the #1145 verdict seam sees the flag only when ``finalized_page_records(state, final_text)``
re-runs it. That second pass puts ``> [page N unchecked: chart_region=FAIL; …]`` on the
record, ``_rewrite_all_fragments`` writes it into ``pages/NNNNN.md``, and before this fix the
document ``.md`` was never re-saved, so it lacked the line (run 23319732:
``document_not_assembly`` = 7,537 pages in 152 documents).

Driven through ``process()`` on the GH-189 synthetic mixed chart+table page, reusing that
file's fixture. The anchorless winner leaves both chart crops at an unestablished position,
so the REAL ``_preserve_chart_regions`` sets the flag late; the anchored winner places them
and is the control arm. Hermetic: ``route_page`` is stubbed with an accepted decision,
``_available_engines_for_agentic`` is patched, ``_resolve_judge_model`` returns "" and
``_crop_descriptions_enabled`` is off (``describe_figure_crops`` defaults on and would call the
local reader on the chart crops), so CI (no Ollama) and a local run take the same path.

No status tuple is pinned (#253/#257). The assertions are the invariant itself -- document
equals fragments assembled -- plus a canary that the flag really was set LATE, which is what
makes the unresolved arm a test of this bug rather than of a pre-set flag.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

fitz = pytest.importorskip("fitz")

import test_gh189_mixed_chart_preservation as g  # noqa: E402
from ocr_output_contract import assemble_pages  # noqa: E402
from verdict_marks_helper import canonical_marks  # noqa: E402

from socr.core.marks import Check  # noqa: E402
from socr.pipeline.orchestrator import UnifiedPipeline  # noqa: E402

#: No unique anchor and no 1:1 table binding: the GH-189 "unplaceable" winner.
ANCHORLESS_WINNER = "| Year | Value | Share |\n| --- | --- | --- |\n" + g.TABLE_ROW


def _process(tmp_path: Path, winner: str, tag: str):
    """``process()`` one synthetic page; return the result, the doc text, the fragments
    and the chart-region flag as it stood before and after ``_preserve_chart_regions``."""
    from socr.core.providers import PROFILE_QWEN_LOCAL

    work = tmp_path / tag
    work.mkdir(parents=True, exist_ok=True)
    pdf = g._make_mixed_chart_table_pdf(work)
    out_dir = work / "out"
    flag: dict[str, bool] = {}
    real_preserve = UnifiedPipeline._preserve_chart_regions

    def _spy(self, state, page_texts, output_dir):
        flag["before"] = bool(state.pages[1].chart_region_placement_unresolved)
        texts = real_preserve(self, state, page_texts, output_dir)
        flag["after"] = bool(state.pages[1].chart_region_placement_unresolved)
        return texts

    pipeline = g._make_pipeline()
    with (
        patch("socr.pipeline.orchestrator.route_page", return_value=g._accepted_decision(winner)),
        patch.object(
            UnifiedPipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]
        ),
        patch.object(UnifiedPipeline, "_resolve_judge_model", return_value=""),
        patch.object(UnifiedPipeline, "_crop_descriptions_enabled", return_value=False),
        patch.object(UnifiedPipeline, "_preserve_chart_regions", _spy),
    ):
        result = pipeline.process(pdf, output_dir=out_dir)

    docs = [p for p in out_dir.rglob("*.md") if p.parent.name != "pages"]
    assert len(docs) == 1, docs
    frags = sorted(out_dir.rglob("pages/*.md"))
    assert frags, "no page fragment was written"
    return result, docs[0].read_text(), frags, flag


def _stitched(frags: list[Path]) -> str:
    return assemble_pages([f.read_text() for f in frags], page_numbers=[int(f.stem) for f in frags])


@pytest.mark.parametrize(
    ("winner", "tag"),
    [
        pytest.param(ANCHORLESS_WINNER, "unresolved", id="unresolved"),
        pytest.param(g.WINNER_TEXT, "placed", id="placed"),
    ],
)
def test_the_saved_document_is_the_fragments_assembled(tmp_path: Path, winner, tag) -> None:
    """CLAUDE.md's invariant, on both arms: the document is the stitched fragments."""
    result, doc, frags, _flag = _process(tmp_path, winner, tag)

    assert doc == _stitched(frags), (
        "the saved document .md is not the page fragments assembled\n"
        f"--- document ---\n{doc}\n--- fragment ---\n{frags[0].read_text()}"
    )
    assert result.pages[0].text == doc, "the returned result disagrees with the saved .md"


def test_a_chart_region_flag_set_after_the_first_save_reaches_the_document(
    tmp_path: Path,
) -> None:
    """The #1177 case: the mark the late flag puts on the fragment is in the document too,
    and is the first block under the page heading (``document_mark_not_first``)."""
    _result, doc, frags, flag = _process(tmp_path, ANCHORLESS_WINNER, "late_flag")

    # Canary: the flag was set by ``_preserve_chart_regions`` itself, i.e. AFTER
    # ``pre_records`` was built and the document first saved. Without it this test could be
    # passing on a page whose flag the first save already saw.
    assert flag == {"before": False, "after": True}, flag

    fragment_marks = canonical_marks(frags[0].read_text(), 1)
    assert any(f"{Check.CHART_REGION.value}=" in m for m in fragment_marks), (
        f"the fragment carries no chart_region mark, so the lever did not fire: {fragment_marks}"
    )
    assert canonical_marks(doc, 1) == fragment_marks, (
        "the document's leading marks under '## Page 1' differ from the fragment's\n"
        f"--- document ---\n{doc}"
    )


def test_the_placed_arm_takes_no_late_flag(tmp_path: Path) -> None:
    """Control for the canary above: on the anchored winner the pass places both crops,
    the flag never goes up, and no chart_region mark appears anywhere."""
    _result, doc, frags, flag = _process(tmp_path, g.WINNER_TEXT, "control")

    assert flag == {"before": False, "after": False}, flag
    for text in (doc, frags[0].read_text()):
        assert not any(f"{Check.CHART_REGION.value}=" in m for m in canonical_marks(text, 1)), text
