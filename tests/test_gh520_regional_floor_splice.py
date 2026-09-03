"""GH-520 part 2: the structure-class floor keeps prose when coverage is provable.

P2 (#490) removed the regional splice and made the exhausted-ladder floor
whole-page, withholding the page's prose. The reason was not that a splice is
wrong -- it was that nothing could prove one covered every table. Round 1's
coverage check read `native_table_region_count`, which `_verify_regions`
derives from `table_regions`, which is built only from SUCCESSFUL
reconstructions. A sibling that failed to parse is simply absent from it, so
the check agreed with the very parser it was auditing and passed while the
collapsed grid shipped inside text labelled "preserved prose".

`docs/log/2026-09-01_p2-structure-class-floor.md` recorded the loss as a known
limitation and named the fix: a detection-level count taken before
reconstruction. #570 added it. This consumes it.

The two fixtures are the ones the ticket names, and they differ in exactly one
thing -- whether the second region parses:

- both regions parse   -> counts agree -> prose outside the tables survives
- one region collapses -> counts differ -> whole page floors, as under P2

Everything else about the page is identical, so the difference cannot come from
anywhere but the guard.

The guard is deliberately narrower than "the counts match". `detected_table_count`
of zero is no evidence rather than a licence (a borderless table seen only by
the lane-cooccupancy pass contributes no bbox and is not counted at all), and a
detected table with no usable bbox -- a table nobody can point at -- fails
closed too. Both are pinned below, because both are the cases where a wrong
guard would leak a collapsed grid.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from socr.core.manifest import _winning_page_output, structure_class_floor_text
from socr.core.result import FailureMode
from test_gh317_structure_class_floor import (
    COLLAPSED_UNIQUE_TOKEN,
    MIXED_VALIDITY_NATIVE_TEXT,
    PROSE_AFTER,
    PROSE_BEFORE,
    UNIQUE_NATIVE_ROW,
    _born_digital_pdf,
    _state,
)

MARKER = "[page 1 failed: unverifiable table — see image]"

SECOND_TABLE_MD = (
    "| horizon | mean | sd |\n|---|---|---|\n| 1y | 0.14 | 0.02 |\n| 5y | 0.21 | 0.05 |\n"
)
SECOND_TABLE_UNIQUE_ROW = "| 5y | 0.21 | 0.05 |"
PROSE_BETWEEN = "Panel B repeats the exercise over longer horizons."

TWO_PARSEABLE_TABLES = (
    f"{PROSE_BEFORE}\n\n"
    "| $n$ | const. | slope | $R^2$ |\n"
    "|---|---|---|---|\n"
    "| 2 | 0.03 | 0.91 | 0.44 |\n"
    "| 5 | 0.07 | 0.85 | 0.51 |\n"
    f"\n{PROSE_BETWEEN}\n\n"
    f"{SECOND_TABLE_MD}\n"
    f"{PROSE_AFTER}\n"
)


def _floored_page(tmp_path: Path, native_text: str, *, detected: int, bboxes: int | None = None):
    """A case-(iii) page carrying the detection-level signal from #570."""
    state = _state(_born_digital_pdf(tmp_path), native_text=native_text)
    ps = state.pages[1]
    ps.d3_floor_png_ref = "![Failed table page 1](figures/failed_table_p1.png)"
    ps.detected_table_count = detected
    boxes = detected if bboxes is None else bboxes
    ps.detected_table_bboxes = [
        (72.0, 100.0 + i * 200.0, 520.0, 250.0 + i * 200.0) for i in range(boxes)
    ]
    return state, ps


def test_both_regions_parse_so_the_prose_survives(tmp_path: Path) -> None:
    """The ticket's second fixture, and what P2 gave up.

    Two detected tables, two parsed blocks: every detected table is accounted
    for in the parser's own block list, so the collapsed-sibling failure cannot
    be happening here. Both grids are replaced by the marker and the prose
    around them ships.
    """
    from socr.tables.reconcile import find_table_blocks

    assert len(find_table_blocks(TWO_PARSEABLE_TABLES)) == 2, (
        "fixture premise: both regions must parse as GFM tables"
    )

    state, _ps = _floored_page(tmp_path, TWO_PARSEABLE_TABLES, detected=2)
    out = _winning_page_output(state, 1)

    assert out.failure_mode is FailureMode.STRUCTURE_CLASS_LADDER_EXHAUSTED
    assert MARKER in out.text, "the failed-table marker did not ship"
    assert PROSE_BEFORE in out.text, "the prose before the tables was withheld"
    assert PROSE_BETWEEN in out.text, "the prose between the tables was withheld"
    assert PROSE_AFTER in out.text, "the prose after the tables was withheld"

    assert UNIQUE_NATIVE_ROW not in out.text, "the first grid's bytes shipped"
    assert SECOND_TABLE_UNIQUE_ROW not in out.text, "the second grid's bytes shipped"


def test_a_collapsed_sibling_floors_the_whole_page(tmp_path: Path) -> None:
    """The ticket's first fixture, and the failure the guard exists for.

    Two tables detected, one parsed. The parser cannot see the collapsed
    region, so a splice keyed off it would replace the good grid and ship the
    collapsed one as prose -- round 1's bug exactly. The counts disagree and
    the whole page floors, which is what P2 does today.
    """
    from socr.tables.reconcile import find_table_blocks

    assert len(find_table_blocks(MIXED_VALIDITY_NATIVE_TEXT)) == 1, (
        "fixture premise: exactly one of the two regions parses"
    )

    state, _ps = _floored_page(tmp_path, MIXED_VALIDITY_NATIVE_TEXT, detected=2)
    out = _winning_page_output(state, 1)

    assert out.failure_mode is FailureMode.STRUCTURE_CLASS_LADDER_EXHAUSTED
    assert MARKER in out.text
    assert COLLAPSED_UNIQUE_TOKEN not in out.text, (
        "the collapsed native region shipped: the parser could not see it, so "
        "no splice can prove it was withheld"
    )
    assert UNIQUE_NATIVE_ROW not in out.text
    assert PROSE_BEFORE not in out.text, (
        "prose survived on a page whose coverage is unprovable; the guard let "
        "the round-1 bug back in"
    )


def test_no_detected_table_is_no_evidence(tmp_path: Path) -> None:
    """Zero is not a licence.

    A borderless table seen only by the lane-cooccupancy pass contributes no
    bbox and is not counted at all, so `detected_table_count == 0` is the
    common case on exactly the pages where the parser's block list is least
    trustworthy. Treating it as "no tables to cover" would splice every one of
    them.
    """
    state, _ps = _floored_page(tmp_path, TWO_PARSEABLE_TABLES, detected=0)
    out = _winning_page_output(state, 1)

    assert PROSE_BEFORE not in out.text, "a page with no detection evidence kept its prose"
    assert UNIQUE_NATIVE_ROW not in out.text
    assert MARKER in out.text


def test_a_table_nobody_can_point_at_fails_closed(tmp_path: Path) -> None:
    """#570's other asymmetry, consumed here.

    A detected table whose bbox is missing, non-finite or degenerate still
    counts but contributes no box, so `count > len(bboxes)`. The parsed-block
    count can then match the count while one detected table has no region at
    all -- and the guard must not read that as coverage.
    """
    state, _ps = _floored_page(tmp_path, TWO_PARSEABLE_TABLES, detected=2, bboxes=1)
    out = _winning_page_output(state, 1)

    assert PROSE_BEFORE not in out.text, "a detected table with no bbox was treated as covered"
    assert MARKER in out.text


def test_the_parser_derived_count_still_cannot_reopen_the_splice(tmp_path: Path) -> None:
    """The retirement guard from P2, restated against the new door.

    `_state` already populates `native_table_region_count` /
    `_identities` from the fixture text, exactly as production does. Those are
    the circular signal, and the new guard must not have quietly started
    reading them: only the detection-level count opens the splice.
    """
    state, ps = _floored_page(tmp_path, MIXED_VALIDITY_NATIVE_TEXT, detected=2)
    assert ps.native_table_region_count == 1, "the circular signal is not populated"

    # Make the circular signal say the page is fully covered -- the round-1 lie.
    ps.native_table_region_count = 2
    out = _winning_page_output(state, 1)

    assert COLLAPSED_UNIQUE_TOKEN not in out.text
    assert PROSE_BEFORE not in out.text, (
        "inflating the parser-derived count reopened the splice; that check is "
        "circular and must stay retired"
    )


def test_the_floor_still_ships_the_png_when_it_splices(tmp_path: Path) -> None:
    """The image is how a human recovers the table, so it must survive the
    splice as well as the whole-page floor."""
    state, _ps = _floored_page(tmp_path, TWO_PARSEABLE_TABLES, detected=2)
    out = _winning_page_output(state, 1)

    assert "figures/failed_table_p1.png" in out.text, (
        "the spliced floor dropped the page image, so the table is recoverable from nothing"
    )


@pytest.mark.parametrize("detected", [1, 3])
def test_a_count_that_disagrees_with_the_blocks_floors(tmp_path: Path, detected: int) -> None:
    """Either direction. Fewer detected than parsed means the parser invented a
    block boundary; more means a detected table is missing from its list. Both
    are unprovable coverage."""
    state, _ps = _floored_page(tmp_path, TWO_PARSEABLE_TABLES, detected=detected)
    out = _winning_page_output(state, 1)

    assert PROSE_BEFORE not in out.text
    assert UNIQUE_NATIVE_ROW not in out.text


def test_an_empty_native_layer_floors(tmp_path: Path) -> None:
    """There is nothing to preserve, and `splice_all_table_regions` returns
    None on empty text -- pinned so the fallback stays the marker rather than
    an empty page."""
    state, _ps = _floored_page(tmp_path, "", detected=2)
    text = structure_class_floor_text(state.pages[1], 1)

    assert text.startswith(MARKER)
    assert "figures/failed_table_p1.png" in text


def test_the_detection_signal_survives_a_resume(tmp_path: Path) -> None:
    """GH-563's lesson, applied to the signal the floor now depends on.

    A skipped page must reach the same floor verdict as the run that measured
    it. If `detected_table_count` lived only in memory, a resumed page would
    restore zero, fail the guard, and floor whole -- silently withholding prose
    that the first run shipped, with nothing in the record saying why the two
    runs disagreed.

    Driven through the real flush and the real restore, not through the field.
    """
    import json

    from socr.core.config import EngineType, PipelineConfig
    from socr.core.result import PageOutput, PageStatus
    from socr.pipeline.orchestrator import UnifiedPipeline

    state, ps = _floored_page(tmp_path, TWO_PARSEABLE_TABLES, detected=2)
    first = _winning_page_output(state, 1)
    assert PROSE_BEFORE in first.text, "run 1 did not splice, so there is nothing to preserve"

    pipeline = UnifiedPipeline(
        PipelineConfig(
            primary_engine=EngineType.QWEN,
            enabled_engines=[EngineType.QWEN],
            quiet=True,
        )
    )
    pipeline._scan_root = state.handle.path.parent
    out_dir = tmp_path / "resume_out"
    pipeline._flush_page_sidecar(state, 1, out_dir)

    meta = json.loads(next(out_dir.rglob("pages/00001.json")).read_text())
    assert meta.get("detected_table_count") == 2, (
        f"the signal never reached the sidecar: {meta.get('detected_table_count')}"
    )
    assert len(meta.get("detected_table_bboxes") or []) == 2

    resumed, resumed_ps = _floored_page(tmp_path, TWO_PARSEABLE_TABLES, detected=0)
    resumed_ps.detected_table_bboxes = []
    page_out = PageOutput(
        page_num=1, text="body", status=PageStatus.SUCCESS, engine="native", audit_passed=True
    )
    pipeline._restore_terminal_page_state(resumed, 1, page_out, out_dir)

    assert resumed.pages[1].detected_table_count == 2, (
        "the detection signal did not survive the resume, so a skipped page "
        "floors whole where the first run kept its prose"
    )
    assert len(resumed.pages[1].detected_table_bboxes) == 2


def test_a_sidecar_without_the_signal_restores_the_safe_zero(tmp_path: Path) -> None:
    """The key is written only when a table was detected, so most sidecars --
    and every sidecar written before this change -- do not carry it. Those must
    restore zero and floor whole, never inherit whatever the object held."""
    from socr.core.config import EngineType, PipelineConfig
    from socr.core.result import PageOutput, PageStatus
    from socr.pipeline.orchestrator import UnifiedPipeline

    state, _ps = _floored_page(tmp_path, TWO_PARSEABLE_TABLES, detected=0)
    pipeline = UnifiedPipeline(
        PipelineConfig(
            primary_engine=EngineType.QWEN, enabled_engines=[EngineType.QWEN], quiet=True
        )
    )
    pipeline._scan_root = state.handle.path.parent
    out_dir = tmp_path / "old_out"
    pipeline._flush_page_sidecar(state, 1, out_dir)

    import json

    meta = json.loads(next(out_dir.rglob("pages/00001.json")).read_text())
    assert "detected_table_count" not in meta, (
        "a page with no detected table wrote the key anyway, changing the byte "
        "shape of nearly every sidecar in the corpus"
    )

    resumed, resumed_ps = _floored_page(tmp_path, TWO_PARSEABLE_TABLES, detected=2)
    page_out = PageOutput(
        page_num=1, text="body", status=PageStatus.SUCCESS, engine="native", audit_passed=True
    )
    pipeline._restore_terminal_page_state(resumed, 1, page_out, out_dir)

    assert resumed.pages[1].detected_table_count == 0, (
        "an absent key left a stale in-memory count in place; a resumed page "
        "would splice on evidence its own sidecar does not carry"
    )
    assert resumed.pages[1].detected_table_bboxes == []


# A pipe-shaped prose block that `find_table_blocks` parses as a table. It is
# not one of the page's detected tables; it is a second block that happens to
# make the counts add up.
DECOY_PIPE_BLOCK = "| note | value |\n|---|---|\n| see appendix | n/a |\n| see footnote | n/a |\n"

COINCIDENT_COUNT_TEXT = (
    f"{PROSE_BEFORE}\n\n"
    "| $n$ | const. | slope | $R^2$ |\n"
    "|---|---|---|---|\n"
    "| 2 | 0.03 | 0.91 | 0.44 |\n"
    "| 5 | 0.07 | 0.85 | 0.51 |\n"
    "\nMaturity 10 30\nconst. 0.11 0.19\nslope 0.78 0.62\n$R^2$ 0.58 0.63\n\n"
    f"{DECOY_PIPE_BLOCK}\n"
    f"{PROSE_AFTER}\n"
)


def test_equal_block_counts_are_not_correspondence(tmp_path: Path) -> None:
    """cubic P1 on #571, and a correction to this PR's own first argument.

    Two tables detected. One collapsed to ragged lines and was never parsed.
    An unrelated pipe-shaped prose block IS parsed. So `find_table_blocks`
    returns two blocks and the detector found two tables -- the counts match by
    coincidence, and a guard built on counting alone splices the two blocks it
    can see and ships the collapsed table as preserved prose. Round 1's bug,
    reached by a different route.

    The first version of this PR claimed equal counts establish that "no
    detected table is missing from the parser's block list". They do not. What
    does is `native_table_region_count == detected_table_count`: reconstruction
    produced a region for every table the detector found. Here it did not, and
    the page floors whole.

    `native_table_region_count` is set to what PRODUCTION would record -- the
    per-region verifier counts successfully reconstructed regions, and the
    collapsed sibling is not one -- rather than left to the test helper, which
    derives it from the same parser and would report 2.
    """
    from socr.tables.reconcile import find_table_blocks

    assert len(find_table_blocks(COINCIDENT_COUNT_TEXT)) == 2, (
        "fixture premise: the parser must see two blocks (the good grid and the "
        "decoy), so the counts coincide"
    )
    assert COLLAPSED_UNIQUE_TOKEN in COINCIDENT_COUNT_TEXT, "fixture premise: the collapsed region"

    state, ps = _floored_page(tmp_path, COINCIDENT_COUNT_TEXT, detected=2)
    ps.native_table_region_count = 1  # what the verifier records: one region reconstructed

    out = _winning_page_output(state, 1)

    assert COLLAPSED_UNIQUE_TOKEN not in out.text, (
        "the collapsed table shipped as preserved prose because an unrelated "
        "pipe block made the counts agree"
    )
    assert PROSE_BEFORE not in out.text, "prose survived on a page with unprovable coverage"
    assert MARKER in out.text


def test_reconstruction_short_of_detection_floors(tmp_path: Path) -> None:
    """The same guard without the decoy: a page where reconstruction produced
    fewer regions than the detector found tables has an unaccounted-for table,
    whatever the parser's block list happens to say."""
    state, ps = _floored_page(tmp_path, TWO_PARSEABLE_TABLES, detected=2)
    ps.native_table_region_count = 1

    out = _winning_page_output(state, 1)

    assert PROSE_BEFORE not in out.text
    assert MARKER in out.text


def test_a_restored_degenerate_bbox_is_not_usable_evidence(tmp_path: Path) -> None:
    """cubic P2 on #571. The detector refuses a non-finite or zero-area box when
    it measures one; the restore accepted it, and the floor guard only reads
    `len(bboxes)` -- so a resumed page could splice on geometry the detector
    would itself have thrown away."""
    import json

    from socr.core.config import EngineType, PipelineConfig
    from socr.core.result import PageOutput, PageStatus
    from socr.pipeline.orchestrator import UnifiedPipeline

    state, _ps = _floored_page(tmp_path, TWO_PARSEABLE_TABLES, detected=2)
    pipeline = UnifiedPipeline(
        PipelineConfig(
            primary_engine=EngineType.QWEN, enabled_engines=[EngineType.QWEN], quiet=True
        )
    )
    pipeline._scan_root = state.handle.path.parent
    out_dir = tmp_path / "bad_geom"
    pipeline._flush_page_sidecar(state, 1, out_dir)

    sidecar = next(out_dir.rglob("pages/00001.json"))
    meta = json.loads(sidecar.read_text())
    assert len(meta["detected_table_bboxes"]) == 2
    meta["detected_table_bboxes"][1] = [5.0, 5.0, 5.0, 9.0]  # zero width
    sidecar.write_text(json.dumps(meta, indent=2))

    resumed, _ = _floored_page(tmp_path, TWO_PARSEABLE_TABLES, detected=0)
    page_out = PageOutput(
        page_num=1, text="body", status=PageStatus.SUCCESS, engine="native", audit_passed=True
    )
    pipeline._restore_terminal_page_state(resumed, 1, page_out, out_dir)

    assert resumed.pages[1].detected_table_bboxes == [], (
        "a degenerate bbox was restored as usable geometry; the whole list must "
        "clear, because a shorter list is a mismatch that fails closed for the "
        "wrong reason"
    )
