"""#1050 step 2: one coherent uncovered block is ONE region, and its words ship.

Step 1 cut Schoenegger p10's framed prompt box into 4 crops and Acosta p21's
pasted clipping into 7. Grouping comes from the page's own geometry (an
enclosing drawn rectangle or a placed raster), so every test here pins a
DIFFERENCE between a page with the container and the same page without it --
never an absolute region count, which depends on the ink filter's corpus cuts.

Hermetic: no provider and no network. The crop reader is stubbed, except where
a test says it drives the real engine with the HTTP layer patched. A few tests
use Tesseract as a stand-in READER (to measure what the model would be shown);
they skip when it is absent. Nothing here verifies a reading — step 2 confirms
nothing (owner ruling 2026-10-14).
"""

from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import fitz
import pytest

from socr.core.config import EngineType, PipelineConfig
from socr.core.providers import PROFILE_QWEN_LOCAL
from socr.core.text_coverage import (
    analyze_page,
    preceding_text_snippet,
    uncovered_text_image_ref,
    uncovered_text_marker,
)
from socr.core.text_recovery import (
    CONFIRMED_STYLE_RE,
    FIGURE_TEXT_PREFIX,
    PAGE_TEXT_PREFIX,
)
from socr.pipeline.orchestrator import UnifiedPipeline

#: conftest's autouse `_figure_description_model_is_absent` replaces this method
#: for EVERY test, so a test of the describer path must reinstall the real one
#: or it measures the stub. Captured at import, before any fixture runs.
_REAL_FIGURE_DESCRIPTION_ASK = UnifiedPipeline._figure_description_ask


def test_loaded_source_is_this_checkout() -> None:
    import socr

    assert Path(socr.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1] / "src")


# Real multi-line paragraphs: the native extractor joins two lone lines into one
# paragraph however far apart they sit, and insertion is paragraph-granular.
# Distinct tails: the anchor is the last words of the block above the region,
# and an anchor that matches twice is refused rather than guessed at.
_ABOVE_LINES = [f"Prose above the box, line {i}, stated before the drawn frame." for i in range(4)]
_BELOW_LINES = [f"Prose below the box, line {i}, stated after the drawn frame." for i in range(4)]
ABOVE = _ABOVE_LINES[0]
BELOW = _BELOW_LINES[0]


def _glyph_row(page, y: float, x0: float = 72.0, n: int = 8) -> None:
    """Path-only 'letters' (same shape as the step-1 file's helper)."""
    x = x0
    widths = (3.0, 4.0, 5.0, 3.5, 4.5, 5.5, 3.2, 4.8)
    for i in range(n):
        w = widths[i % len(widths)]
        page.draw_rect(fitz.Rect(x, y, x + w, y + 6.0), color=(0, 0, 0), width=0.6)
        x += w + 2.0


def _two_blocks_pdf(path: Path, *, container: str, rows=(150.0, 330.0), wide: bool = False) -> Path:
    """Two unread glyph rows far apart, optionally inside one container.

    The rows are separated by well over a glyph height, so step 1's row-linking
    keeps them apart. ``container`` adds the thing that says they belong
    together: ``"rect"`` a drawn frame, ``"page_rect"`` a frame that covers the
    whole page, ``"raster"`` a placed image, ``"none"`` nothing.
    """
    doc = fitz.open()
    page = doc.new_page()
    # Enough real prose that routing keeps the page in the native lane: a page
    # that is mostly drawn paths is a chart page, and the chart lane rewrites
    # the body this test is about.
    y = 60.0
    for line in _ABOVE_LINES:
        page.insert_text((72, y), line)
        y += 13.0
    if container == "raster":
        inner = fitz.open()
        ip = inner.new_page(width=400, height=300)
        ip.draw_rect(fitz.Rect(0, 0, 400, 300), color=(1, 1, 1), fill=(1, 1, 1))
        page.insert_image(
            fitz.Rect(64, 90, 464, 390), pixmap=ip.get_pixmap(alpha=False), overlay=False
        )
        inner.close()
    elif container == "rect":
        page.draw_rect(fitz.Rect(64, 90, 464, 390), color=(0.4, 0.4, 0.4), width=0.8)
    elif container == "page_rect":
        page.draw_rect(page.rect, color=(0.4, 0.4, 0.4), width=0.8)
    elif container == "page_raster":
        inner = fitz.open()
        ip = inner.new_page(width=600, height=780)
        ip.draw_rect(fitz.Rect(0, 0, 600, 780), color=(1, 1, 1), fill=(1, 1, 1))
        page.insert_image(
            fitz.Rect(5, 5, 605, 785), pixmap=ip.get_pixmap(alpha=False), overlay=False
        )
        inner.close()
    for row_y in rows:
        _glyph_row(page, row_y, x0=10.0, n=96) if wide else _glyph_row(page, row_y)
    y = 560.0
    for line in _BELOW_LINES:
        page.insert_text((72, y), line)
        y += 13.0
    doc.save(path)
    doc.close()
    return path


def _regions(pdf: Path):
    with fitz.open(pdf) as doc:
        cov = analyze_page(doc[0])
    return cov.regions


@pytest.mark.parametrize("container", ["rect", "raster"])
def test_a_container_merges_what_is_inside_it(tmp_path: Path, container: str) -> None:
    """A drawn frame or a placed raster turns two separate regions into one."""
    loose = _regions(_two_blocks_pdf(tmp_path / "loose.pdf", container="none"))
    held = _regions(_two_blocks_pdf(tmp_path / f"held_{container}.pdf", container=container))
    assert len(loose) > 1, "fixture must produce several regions without a container"
    assert len(held) < len(loose)
    assert sum(r.merged_from for r in held) == sum(r.merged_from for r in loose)
    assert max(r.merged_from for r in held) > max(r.merged_from for r in loose)


# Ink at both extremes of the page: the regions' own union is page-sized, so a
# container around them would produce a whole-page crop.
_SPANNING_ROWS = (20.0, 770.0)


def test_a_page_sized_merge_is_refused(tmp_path: Path) -> None:
    """The ink guard is on the merged INK, not on the container.

    A page-sized RASTER must still group (Acosta's clipping scan is placed over
    85% of the page); what must never happen is a merged crop that is itself
    the page. Same container, same rows, moved apart until their union is
    page-sized: the merge stops.
    """
    close = _regions(
        _two_blocks_pdf(
            tmp_path / "close.pdf", container="page_raster", rows=(150.0, 330.0), wide=True
        )
    )
    spanning = _regions(
        _two_blocks_pdf(
            tmp_path / "wide.pdf", container="page_raster", rows=_SPANNING_ROWS, wide=True
        )
    )
    assert max(r.merged_from for r in close) == 2, "a page-sized raster still groups"
    assert max(r.merged_from for r in spanning) == 1, "a page-sized merge is refused"


def test_a_page_sized_drawn_frame_is_not_a_container(tmp_path: Path) -> None:
    """A border or background fill is furniture, not a box that groups its contents.

    Unlike a placed raster, a drawn rect spanning the page contains every
    region by construction, so it would merge unrelated ink across the prose
    between them.
    """
    loose = _regions(_two_blocks_pdf(tmp_path / "loose.pdf", container="none"))
    framed = _regions(_two_blocks_pdf(tmp_path / "framed.pdf", container="page_rect"))
    assert [r.merged_from for r in framed] == [r.merged_from for r in loose]


def test_the_merged_box_is_the_ink_not_the_container(tmp_path: Path) -> None:
    """Acosta's clipping raster is 85% of the page; its ink is 5%. Crop the ink."""
    pdf = _two_blocks_pdf(tmp_path / "raster.pdf", container="raster")
    with fitz.open(pdf) as doc:
        page = doc[0]
        raster = [fitz.Rect(i["bbox"]) for i in page.get_image_info() if i.get("bbox")]
        merged = max(analyze_page(page).regions, key=lambda r: r.merged_from)
    assert raster, "fixture must place a raster"
    box = fitz.Rect(merged.pdf)
    assert box in raster[0] or raster[0].contains(box)
    assert box.get_area() < raster[0].get_area()


def test_the_anchor_is_the_block_above_the_region(tmp_path: Path) -> None:
    """Reading-order placement follows the page's own text blocks."""
    pdf = _two_blocks_pdf(tmp_path / "rect.pdf", container="rect")
    with fitz.open(pdf) as doc:
        page = doc[0]
        merged = max(analyze_page(page).regions, key=lambda r: r.merged_from)
        anchor = preceding_text_snippet(page, merged.pdf)
    assert anchor
    assert anchor in " ".join(_ABOVE_LINES)


# --- the pipeline difference -------------------------------------------------


def _pipeline(**overrides) -> UnifiedPipeline:
    defaults = dict(
        primary_engine=EngineType.QWEN,
        local_engine=EngineType.QWEN,
        enabled_engines=[EngineType.QWEN],
        quiet=True,
        recover_corrupt_math=False,
        equation_region_lane=False,
        describe_figure_crops=False,
        judge_backend="heuristic",
        check_text_coverage=True,
    )
    defaults.update(overrides)
    return UnifiedPipeline(PipelineConfig(**defaults))


def _page_text(out: Path) -> str:
    return next(out.rglob("pages/00001.md")).read_text(encoding="utf-8")


def _run(pdf: Path, out: Path, *, reading: str | None, chart_lane: bool = False, **cfg) -> str:
    """Process *pdf* with the crop reader stubbed to return *reading*.

    ``chart_lane=False`` holds the page in the plain native lane. The fixture's
    stroked path-glyphs ARE vector chart marks, so without this the chart-asset
    lane rebuilds the body (it joins the prose lines into one paragraph) and the
    test would be measuring that lane instead of this one.
    """
    pipeline = _pipeline(**cfg)
    with (
        patch.object(pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]),
        patch.object(pipeline, "_resolve_judge_model", return_value=""),
        patch.object(pipeline, "_uncovered_text_reader_available", return_value=True),
        patch.object(pipeline, "_uncovered_text_ask", return_value=lambda _p, _q: reading),
        patch.object(pipeline, "_is_chart_asset_page", return_value=chart_lane),
    ):
        pipeline.process(pdf, out)
    return _page_text(out)


RECOVERED = "Treatment Prompt\nIn this chat you are a superforecaster."


def test_recovery_adds_the_words_and_changes_nothing_else(tmp_path: Path) -> None:
    """The ONLY difference a reading makes is the recovered block and the mark.

    Run twice in one process on one PDF, changing only what the crop reader
    answers. Absolute statuses are not pinned: the provider ladder and the
    witness differ between this host and CI.
    """
    pdf = _two_blocks_pdf(tmp_path / "doc.pdf", container="rect")
    silent = _run(pdf, tmp_path / "o_silent", reading=None)
    read = _run(pdf, tmp_path / "o_read", reading=RECOVERED)

    assert uncovered_text_marker(1) in silent
    assert PAGE_TEXT_PREFIX not in silent
    assert "Treatment Prompt" in read
    assert "In this chat you are a superforecaster." in read
    # Native prose survives both runs, byte for byte.
    for body in (silent, read):
        assert ABOVE in body
        assert BELOW in body


def test_the_flag_off_is_exactly_step_one(tmp_path: Path) -> None:
    """``recover_uncovered_text=False`` reproduces step 1, grouping included.

    Round 2, item E: grouping exists to make ONE crop readable, so with the
    reader off it only changes bytes. Off, the page must carry step 1's
    SEPARATE crops and no recovered block. Byte-identity against main's own
    output is verified on the real review pages, outside the suite.
    """
    pdf = _two_blocks_pdf(tmp_path / "doc.pdf", container="rect")
    off = _run(pdf, tmp_path / "o_off", reading=RECOVERED, recover_uncovered_text=False)
    assert PAGE_TEXT_PREFIX not in off
    assert uncovered_text_marker(1) in off
    assert uncovered_text_image_ref(1, 2) in off, "step 1's separate crops, not one merged crop"


def test_an_unread_region_keeps_its_picture(tmp_path: Path) -> None:
    """No reading -> step 1's picture link and page mark, unchanged."""
    pdf = _two_blocks_pdf(tmp_path / "doc.pdf", container="rect")
    silent = _run(pdf, tmp_path / "o", reading=None)
    assert uncovered_text_image_ref(1, 1) in silent


def test_text_inside_a_figure_goes_beneath_the_figure_image(tmp_path: Path) -> None:
    """2026-10-09 ruling: in a figure the image stays and the words sit under it."""
    pdf = _two_blocks_pdf(tmp_path / "doc.pdf", container="raster")
    read = _run(pdf, tmp_path / "o", reading=RECOVERED)
    ref = uncovered_text_image_ref(1, 1)
    assert ref in read, "a figure region keeps its picture"
    assert FIGURE_TEXT_PREFIX in read
    assert read.index(ref) < read.index(FIGURE_TEXT_PREFIX)


def test_a_region_outside_a_figure_lands_in_reading_order(tmp_path: Path) -> None:
    """Text drawn as shapes is prose: it is substituted in place, not appended."""
    pdf = _two_blocks_pdf(tmp_path / "doc.pdf", container="rect")
    read = _run(pdf, tmp_path / "o", reading=RECOVERED)
    above = read.index(ABOVE)
    below = read.index(BELOW)
    assert above < read.index(PAGE_TEXT_PREFIX) < below


def test_an_anchor_crash_does_not_cost_the_page_its_regions(tmp_path: Path) -> None:
    """A raise while placing must leave step 1's behaviour, not `text_coverage_unknown`.

    `_apply_text_coverage`'s outer try maps ANY escape to "could not decide",
    which drops the regions step 1 would have shipped. Pinned as a difference:
    the same page, same reading, with and without the anchor helper raising.
    """
    pdf = _two_blocks_pdf(tmp_path / "doc.pdf", container="rect")
    ok = _run(pdf, tmp_path / "o_ok", reading=RECOVERED)
    with patch("socr.core.text_coverage.preceding_text_snippet", side_effect=RuntimeError("boom")):
        broken = _run(pdf, tmp_path / "o_broken", reading=RECOVERED)
    # The words still ship, and the page keeps the uncovered-text mark rather
    # than the "image unavailable" mark the unknown branch writes.
    assert PAGE_TEXT_PREFIX in ok
    assert PAGE_TEXT_PREFIX in broken
    assert "image unavailable" not in broken
    # The only difference is where the block landed.
    assert ok != broken


# --- fix round 1 (Astra + Fable, 2026-10-14) ---------------------------------


def test_every_recovered_region_keeps_its_picture_and_says_unverified(
    tmp_path: Path,
) -> None:
    """Owner ruling 2026-10-14: nothing is confirmed, and the picture always stays.

    Token-containment confirmation kept passing wrong readings, so the label
    was dropped rather than patched a fourth time. Pinned on the shipped page:
    the words, the one unverified mark, the picture, and no confirmed-style
    wording anywhere.
    """
    pdf = _two_blocks_pdf(tmp_path / "doc.pdf", container="rect")
    read = _run(pdf, tmp_path / "o", reading=RECOVERED)
    assert "Treatment Prompt" in read
    assert PAGE_TEXT_PREFIX in read
    assert uncovered_text_image_ref(1, 1) in read
    assert not CONFIRMED_STYLE_RE.search(read), read


def _unread_bands_around_a_native_line_pdf(path: Path) -> Path:
    """Two uncovered bands with a native text line between them.

    Merging them produces a crop containing that native line; the model then
    reads it and the page ships it twice (item B).
    """
    doc = fitz.open()
    page = doc.new_page()
    y = 60.0
    for line in _ABOVE_LINES:
        page.insert_text((72, y), line)
        y += 13.0
    page.draw_rect(fitz.Rect(64, 150, 464, 300), color=(0.4, 0.4, 0.4), width=0.8)
    _glyph_row(page, 170.0)
    page.insert_text((72, 230), "A native line trapped between the two uncovered bands.")
    _glyph_row(page, 270.0)
    y = 560.0
    for line in _BELOW_LINES:
        page.insert_text((72, y), line)
        y += 13.0
    doc.save(path)
    doc.close()
    return path


def test_a_merge_that_would_swallow_a_native_line_is_refused(tmp_path: Path) -> None:
    """Item B: a crop must never contain text the page already ships.

    Pinned as a difference against the same page with the trapped line removed:
    without the line the two bands merge, with it they must not.
    """
    import fitz as _fitz

    trapped = _unread_bands_around_a_native_line_pdf(tmp_path / "trapped.pdf")
    free = _two_blocks_pdf(tmp_path / "free.pdf", container="rect", rows=(170.0, 270.0))
    assert max(r.merged_from for r in _regions(free)) == 2, "fixture must merge without the line"
    regions = _regions(trapped)
    assert max(r.merged_from for r in regions) == 1, "the merge must be refused"
    # And no region's box covers the trapped native line.
    with _fitz.open(trapped) as doc:
        page = doc[0]
        line = next(b for b in page.get_text("blocks") if "trapped between" in str(b[4]))
    for reg in regions:
        x0, y0, x1, y1 = reg.pdf
        assert not (line[0] < x1 and line[2] > x0 and line[1] < y1 and line[3] > y0)


def _ruled_table_plus_unread_pdf(path: Path) -> Path:
    """A closed ruled grid plus uncovered ink: selection takes the NATIVE FALLBACK.

    Copied in shape from the step 1 file's fixture of the same name; the grid is
    what pushes selection down the fallback branch this test is about.
    """
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 90), "See the specification for the estimates reported below.")
    rows = (("Year", "Coefficient"), ("2019", "0.31"), ("2020", "0.42"), ("2021", "0.55"))
    top, left, mid, right, pitch = 128.0, 68.0, 190.0, 320.0, 22.0
    for i, (a, b) in enumerate(rows):
        y = top + 16 + i * pitch
        page.insert_text((left + 6, y), a, fontsize=10)
        page.insert_text((mid + 6, y), b, fontsize=10)
    bottom = top + pitch * len(rows)
    for i in range(len(rows) + 1):
        page.draw_line(fitz.Point(left, top + pitch * i), fitz.Point(right, top + pitch * i))
    for x in (left, mid, right):
        page.draw_line(fitz.Point(x, top), fitz.Point(x, bottom))
    _glyph_row(page, 300.0)
    _glyph_row(page, 312.0)
    page.insert_text((60, 520), "A caption sits below the unread marks.")
    doc.save(path)
    doc.close()
    return path


def test_the_native_fallback_carries_the_words_and_the_picture(tmp_path: Path) -> None:
    """Item C: selection's fallback must not drop step 2's block or its picture.

    The fallback rebuilds the page from the frozen native snapshot and used to
    keep only the slice from the page mark onward. A step-2 block placed in
    reading order sits BEFORE that mark, so the page shipped with neither the
    words nor the image.

    Driven at the selection seam rather than through ``process()``: the branch
    needs a native attempt appended AFTER apply decorated the winner (the
    equation sidecar), which a single synthetic page cannot produce on its own.
    The same shape as the step 1 file's sidecar test, with a step-2 block added.
    """
    from socr.core.document import DocumentHandle
    from socr.core.manifest import SelectionProvenance, _select_page_output_tagged
    from socr.core.result import FailureMode, PageOutput, PageStatus
    from socr.core.state import DocumentState, PageState
    from socr.core.text_recovery import recovered_block

    pdf = _two_blocks_pdf(tmp_path / "eq.pdf", container="rect")
    base = "A clean paragraph of native body text on this page."
    equation = "$$P(A \\cup B)=P(A)+P(B)$$"
    mark = uncovered_text_marker(1)
    link = uncovered_text_image_ref(1, 1)
    block = recovered_block("Treatment Prompt", in_figure=False)
    # Apply's shape since step 2: the block is INSIDE the prose, before the mark.
    decorated = base + "\n\n" + block + "\n\n" + link + "\n\n" + mark + "\n"
    bo = PageOutput(
        page_num=1,
        text=decorated,
        status=PageStatus.WARNING,
        engine="native",
        audit_passed=True,
        failure_mode=FailureMode.TEXT_INK_UNREAD,
    )
    eq = PageOutput(
        page_num=1,
        text=base + "\n\n" + equation + "\n",
        status=PageStatus.SUCCESS,
        engine="native+equations",
        audit_passed=True,
        failure_mode=FailureMode.NONE,
    )
    ps = PageState(page_num=1)
    ps.is_born_digital = True
    ps.native_text = base
    ps.has_tables = True
    ps.attempts = [bo, eq]
    ps.best_output = bo
    ps.text_ink_unread = True
    ps.text_coverage_verdict = "flagged"
    state = DocumentState(handle=DocumentHandle(path=pdf))
    state.pages[1] = ps
    winner, provenance = _select_page_output_tagged(state, 1)
    text = winner.text or ""
    assert provenance is SelectionProvenance.NATIVE_FALLBACK
    assert equation in text, "the later native append still survives"
    assert mark in text
    assert link in text, "the picture must survive the fallback"
    assert block in text, "and so must the recovered words"


def test_native_only_never_calls_the_crop_reader(tmp_path: Path) -> None:
    """Item H: --native-only means no model on any page, step 2 included.

    The figure describer already carries this gate
    (`_crop_descriptions_enabled`); step 2 had none and called the reader
    anyway, which is the one promise that flag makes. Pinned as a difference
    over the flag with the SAME stubbed reader.
    """
    pdf = _two_blocks_pdf(tmp_path / "doc.pdf", container="rect")

    def run(out: str, native_only: bool) -> tuple[int, str]:
        calls: list[str] = []

        def ask(path, _prompt):
            calls.append(str(path))
            return RECOVERED

        pipeline = _pipeline(native_only=native_only, table_judge_ladder=False)
        with (
            patch.object(
                pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]
            ),
            patch.object(pipeline, "_resolve_judge_model", return_value=""),
            patch.object(pipeline, "_uncovered_text_reader_available", return_value=True),
            patch.object(pipeline, "_uncovered_text_ask", return_value=ask),
            patch.object(pipeline, "_is_chart_asset_page", return_value=False),
        ):
            pipeline.process(pdf, tmp_path / out)
        return len(calls), _page_text(tmp_path / out)

    free_calls, free_text = run("o_free", False)
    only_calls, only_text = run("o_only", True)
    assert free_calls > 0, "the fixture must reach the reader when the flag is off"
    assert only_calls == 0
    assert PAGE_TEXT_PREFIX in free_text
    assert PAGE_TEXT_PREFIX not in only_text
    assert uncovered_text_marker(1) in only_text, "step 1's mark and picture remain"


def test_a_spent_page_budget_skips_the_remaining_regions(tmp_path: Path) -> None:
    """Item H: a page of many labels cannot hold the run for one deadline each.

    Pinned as a difference over the budget only: same page, same reader, a
    budget that is already gone versus one that is not.
    """
    pdf = _two_blocks_pdf(tmp_path / "doc.pdf", container="none", rows=(150.0, 330.0))

    def run(out: str, spent: bool) -> tuple[int, str]:
        calls: list[str] = []

        def ask(path, _prompt):
            calls.append(str(path))
            return RECOVERED

        pipeline = _pipeline()
        budget = pipeline._uncovered_text_budget
        with (
            patch.object(
                pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]
            ),
            patch.object(pipeline, "_resolve_judge_model", return_value=""),
            patch.object(pipeline, "_uncovered_text_reader_available", return_value=True),
            patch.object(pipeline, "_uncovered_text_ask", return_value=ask),
            patch.object(pipeline, "_is_chart_asset_page", return_value=False),
            patch.object(
                pipeline,
                "_uncovered_text_budget",
                side_effect=lambda st, pn, rg: (
                    _spend(budget(st, pn, rg)) if spent else budget(st, pn, rg)
                ),
            ),
        ):
            pipeline.process(pdf, tmp_path / out)
        return len(calls), _page_text(tmp_path / out)

    def _spend(b):
        b.total_sec = 0.0
        return b

    live_calls, live_text = run("o_live", False)
    spent_calls, spent_text = run("o_spent", True)
    assert live_calls > 0
    assert spent_calls == 0, "no region is read once the page's budget is gone"
    assert PAGE_TEXT_PREFIX in live_text
    assert PAGE_TEXT_PREFIX not in spent_text
    assert uncovered_text_marker(1) in spent_text, "skipped regions ship as step 1 left them"


def test_a_timed_out_read_stops_recovery_not_the_document(tmp_path: Path) -> None:
    """The READ's own outcome stops recovery, with no health probe.

    `_probe_backend_idle` asks `/api/tags`, which answers 200 in milliseconds on
    a jammed GPU, so gating on it made this latch unreachable in the case it
    exists for. Pinned as a difference over the failure KIND: a timeout stops
    recovery for the document, an ordinary failure is just a refusal.

    Renamed in round 7: it never halted the document after round 6, and the old
    name said it did.
    """
    pdf = _two_blocks_pdf(tmp_path / "doc.pdf", container="rect")

    def run(out: str, error: Exception) -> bool:
        def ask(_path, _prompt):
            raise error

        pipeline = _pipeline()
        with (
            patch.object(
                pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]
            ),
            patch.object(pipeline, "_resolve_judge_model", return_value=""),
            patch.object(pipeline, "_uncovered_text_reader_available", return_value=True),
            patch.object(pipeline, "_uncovered_text_ask", return_value=ask),
            patch.object(pipeline, "_is_chart_asset_page", return_value=False),
        ):
            pipeline.process(pdf, tmp_path / out)
        return pipeline._recovery_backend_wedged

    assert run("o_refused", ValueError("no model")) is False
    assert run("o_wedged", TimeoutError("read timed out")) is True


def test_the_real_adapters_swallowed_timeout_is_turned_back_into_one() -> None:
    """Item C: `OllamaFigureEngine.ask` returns None on ANY failure.

    The latch could only ever be armed by an exception injected ABOVE the
    adapter, which is to say never in production. The wrapper turns a `None`
    that consumed the engine's whole deadline back into a TimeoutError, which
    is the one thing a refusal cannot do. Tested on the wrapper itself: through
    `process()` a clock stub is consumed by the budget and the probe too, which
    makes the measurement about the stub rather than the code.
    """
    from socr.engines.gemini_api import FIGURE_ASK_TIMEOUT_SEC

    pipeline = _pipeline()
    with patch.object(pipeline, "_figure_description_ask", return_value=lambda _p, _q: None):
        ask = pipeline._uncovered_text_ask()

    ticks = iter([0.0, 0.0])
    with patch("socr.pipeline.orchestrator.time.monotonic", side_effect=lambda: next(ticks)):
        assert ask("x.png", "p") is None, "a quick None is a refusal, not a wedge"

    ticks = iter([0.0, FIGURE_ASK_TIMEOUT_SEC])
    with patch("socr.pipeline.orchestrator.time.monotonic", side_effect=lambda: next(ticks)):
        with pytest.raises(TimeoutError):
            ask("x.png", "p")


def test_a_wedge_stops_the_rest_of_the_page(tmp_path: Path) -> None:
    """Item C: the regions after a wedge must not be fired into a stuck GPU."""
    pdf = _two_blocks_pdf(tmp_path / "doc.pdf", container="none", rows=(150.0, 330.0))
    calls: list[str] = []

    def ask(path, _prompt):
        calls.append(str(path))
        raise TimeoutError("read timed out")

    pipeline = _pipeline()
    with (
        patch.object(pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]),
        patch.object(pipeline, "_resolve_judge_model", return_value=""),
        patch.object(pipeline, "_uncovered_text_reader_available", return_value=True),
        patch.object(pipeline, "_uncovered_text_ask", return_value=ask),
        patch.object(pipeline, "_is_chart_asset_page", return_value=False),
    ):
        pipeline.process(pdf, tmp_path / "o")
    assert pipeline._recovery_backend_wedged is True
    assert len(calls) == 1, "only the region that wedged was read"


def test_the_page_budget_does_not_grow_with_the_region_count(tmp_path: Path) -> None:
    """Item D: 600s x (regions + 1) gave 23 labels 46 minutes. Not a bound."""
    pipeline = _pipeline()
    one = pipeline._uncovered_text_budget(None, 1, [object()])
    many = pipeline._uncovered_text_budget(None, 1, [object()] * 23)
    assert one.total_sec == many.total_sec
    assert one.total_sec == pipeline._page_ladder_budget_sec([])


@pytest.mark.parametrize("container", ["rect", "raster"])
def test_recovery_off_does_not_group_at_all(tmp_path: Path, container: str) -> None:
    """Item E: with no reader there is nothing to read, so step 2 is step 1.

    Grouping exists to make ONE crop readable. Off, it only changes bytes.
    Pinned as a difference over the flag, on both container shapes.
    """
    pdf = _two_blocks_pdf(tmp_path / f"doc_{container}.pdf", container=container)
    grouped = _regions(pdf)
    with fitz.open(pdf) as doc:
        ungrouped = analyze_page(doc[0], group=False).regions
    assert max(r.merged_from for r in grouped) > 1, "the fixture does group when asked"
    assert all(r.merged_from == 1 for r in ungrouped)
    assert all(r.figure_pdf is None for r in ungrouped)
    assert len(ungrouped) > len(grouped)


def test_the_picture_survives_into_the_final_document(tmp_path: Path) -> None:
    """Item F: the fragment is not the artefact a reader opens.

    Picture assertions elsewhere check `pages/00001.md`; this checks the
    assembled document, which is what ships.
    """
    pdf = _two_blocks_pdf(tmp_path / "doc.pdf", container="rect")
    out = tmp_path / "o"
    _run(pdf, out, reading=RECOVERED)
    final = next(p for p in out.rglob("*.md") if p.parent.name != "pages")
    body = final.read_text(encoding="utf-8")
    assert "Treatment Prompt" in body
    assert uncovered_text_image_ref(1, 1) in body
    assert PAGE_TEXT_PREFIX in body
    assert not CONFIRMED_STYLE_RE.search(body), body


def test_a_resumed_page_keeps_its_words_and_its_picture(tmp_path: Path) -> None:
    """Item F: resume restores the fragment, so the recovery must survive it."""
    pdf = _two_blocks_pdf(tmp_path / "doc.pdf", container="rect")
    out = tmp_path / "o"
    first = _run(pdf, out, reading=RECOVERED)
    calls: list[str] = []

    def ask(path, _prompt):
        calls.append(str(path))
        return RECOVERED

    pipeline = _pipeline()
    with (
        patch.object(pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]),
        patch.object(pipeline, "_resolve_judge_model", return_value=""),
        patch.object(pipeline, "_uncovered_text_reader_available", return_value=True),
        patch.object(pipeline, "_uncovered_text_ask", return_value=ask),
        patch.object(pipeline, "_is_chart_asset_page", return_value=False),
    ):
        pipeline.process(pdf, out)
    second = _page_text(out)
    assert second == first, "a resumed page ships the same bytes"
    assert uncovered_text_image_ref(1, 1) in second


def _two_pages_with_unread_ink_pdf(path: Path) -> Path:
    """Two pages, each with native prose and uncovered ink."""
    doc = fitz.open()
    for _ in range(2):
        page = doc.new_page()
        y = 60.0
        for line in _ABOVE_LINES:
            page.insert_text((72, y), line)
            y += 13.0
        page.draw_rect(fitz.Rect(64, 120, 464, 420), color=(0.4, 0.4, 0.4), width=0.8)
        _glyph_row(page, 150.0)
        _glyph_row(page, 330.0)
        y = 560.0
        for line in _BELOW_LINES:
            page.insert_text((72, y), line)
            y += 13.0
    doc.save(path)
    doc.close()
    return path


def test_a_slow_read_on_page_one_does_not_cut_the_document_short(tmp_path: Path) -> None:
    """Round 6 BLOCKER: a slow crop read must not halt a native document.

    One read taking its full deadline used to arm the ladder's cascade halt, so
    every later page shipped `not_processed_after_halt` — on main a native
    document never halts at all. The causes are mundane (a cold model load, a
    repetition runaway, a stray fail-fast from the shared outstanding-call
    label) and none is a reason to stop reading a page's own native text.

    Pinned against the ALIVE run: page 2's native text must be identical, and
    the document must not carry the halt reason.
    """
    pdf = _two_pages_with_unread_ink_pdf(tmp_path / "two_pages.pdf")

    def run(out: str, wedge: bool) -> tuple[str, bool, int]:
        calls: list[str] = []

        def ask(_path, _prompt):
            calls.append("x")
            if wedge:
                raise TimeoutError("read timed out")
            return RECOVERED

        pipeline = _pipeline()
        with (
            patch.object(
                pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]
            ),
            patch.object(pipeline, "_resolve_judge_model", return_value=""),
            patch.object(pipeline, "_uncovered_text_reader_available", return_value=True),
            patch.object(pipeline, "_uncovered_text_ask", return_value=ask),
            patch.object(pipeline, "_is_chart_asset_page", return_value=False),
        ):
            result = pipeline.process(pdf, tmp_path / out)
        page_two = (tmp_path / out).rglob("pages/00002.md")
        body = next(page_two).read_text(encoding="utf-8")
        return body, "PARTIAL_SAVE_VLM_TIMEOUT" in (result.error or ""), len(calls)

    alive, alive_halted, alive_calls = run("o_alive", wedge=False)
    wedged, wedged_halted, wedged_calls = run("o_wedged", wedge=True)

    assert alive_halted is False
    assert wedged_halted is False, "a slow read must not halt the document"
    assert wedged_calls == 1, "and no further crop is read in it"
    # Page 2 still ships, and its NATIVE text is what the alive run shipped.
    for line in _ABOVE_LINES + _BELOW_LINES:
        assert line in wedged, "page 2 lost its native text"
        assert alive.count(line) == wedged.count(line) == 1
    # The difference is only the recovery: alive has the words, wedged the picture.
    assert PAGE_TEXT_PREFIX in alive
    assert PAGE_TEXT_PREFIX not in wedged
    assert uncovered_text_marker(2) in wedged


def test_a_stopped_reader_says_so_once(tmp_path: Path) -> None:
    """One event per document, not one per skipped region."""
    pdf = _two_pages_with_unread_ink_pdf(tmp_path / "two_pages.pdf")
    pipeline = _pipeline()

    def ask(_path, _prompt):
        raise TimeoutError("read timed out")

    with (
        patch.object(pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]),
        patch.object(pipeline, "_resolve_judge_model", return_value=""),
        patch.object(pipeline, "_uncovered_text_reader_available", return_value=True),
        patch.object(pipeline, "_uncovered_text_ask", return_value=ask),
        patch.object(pipeline, "_is_chart_asset_page", return_value=False),
    ):
        pipeline.process(pdf, tmp_path / "o")
    log = json.loads(next((tmp_path / "o").rglob("audit_log.json")).read_text(encoding="utf-8"))
    events = log if isinstance(log, list) else log.get("events", [])
    kinds = [e.get("kind") for e in events]
    assert kinds.count("uncovered_text_reader_stopped") == 1
    assert "partial_save_vlm_timeout" not in kinds


# --- fix round 3, item 1: the reader never sees native text -------------------


def _raster_under_prose_pdf(path: Path) -> Path:
    """A placed raster whose box sits UNDER a native sentence.

    The reproduced duplication shape: the crop contains a sentence the page's
    text layer already carries, so the model transcribes it and the page ships
    it twice.
    """
    doc = fitz.open()
    page = doc.new_page()
    inner = fitz.open()
    ip = inner.new_page(width=400, height=200)
    ip.draw_rect(fitz.Rect(0, 0, 400, 200), color=(1, 1, 1), fill=(1, 1, 1))
    for i, h in enumerate((80.0, 140.0, 60.0)):
        x = 40.0 + i * 110
        ip.draw_rect(fitz.Rect(x, 180 - h, x + 70, 180), color=(0, 0, 0), fill=(0.2, 0.4, 0.8))
    page.insert_image(
        fitz.Rect(72, 150, 472, 350), pixmap=ip.get_pixmap(alpha=False), overlay=False
    )
    inner.close()
    _glyph_row(page, 170.0, x0=100.0)
    page.insert_text((90, 330), DUPLICATED_SENTENCE)
    y = 60.0
    for line in _ABOVE_LINES:
        page.insert_text((72, y), line)
        y += 13.0
    doc.save(path)
    doc.close()
    return path


DUPLICATED_SENTENCE = "This native sentence must appear exactly once on the page."


def _echoing_reader():
    """A reader that transcribes whatever ink it is shown, honestly.

    It cannot see the mask, so if the crop still carries the native sentence it
    reports it — which is exactly the duplication under test.
    """
    from PIL import Image

    import pytesseract

    def ask(path, _prompt):
        with Image.open(path) as im:
            return pytesseract.image_to_string(im) or None

    return ask


@pytest.mark.parametrize("fixture", ["raster_under_prose", "callout_with_icon", "two_column_chart"])
def test_the_reader_never_sees_native_text(tmp_path: Path, fixture: str) -> None:
    """Round 3, item 1: no recovered word may come from the page's text layer.

    Rather than chase every shape that can enclose prose -- a raster under it,
    a callout, a two-column chart box -- the crop handed to the model has every
    visible native character painted out. The picture that ships stays
    unmasked. Driven with a REAL reader (Tesseract) over the real crop, so the
    test measures what the model would actually be shown.

    The property, not the sentence: a GARBLED transcription of the native text
    is the same duplication as a clean one, so the assertion is that no content
    word of the recovered block appears among the page's native words.
    """
    pytest.importorskip("pytesseract")
    import fitz as _fitz

    from socr.tables.source_evidence import plain_text_tokens

    builder = {
        "raster_under_prose": _raster_under_prose_pdf,
        "callout_with_icon": _framed_callout_pdf,
        "two_column_chart": _right_column_chart_pdf,
    }[fixture]
    pdf = builder(tmp_path / f"{fixture}.pdf")

    pipeline = _pipeline()
    with (
        patch.object(pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]),
        patch.object(pipeline, "_resolve_judge_model", return_value=""),
        patch.object(pipeline, "_uncovered_text_reader_available", return_value=True),
        patch.object(pipeline, "_uncovered_text_ask", return_value=_echoing_reader()),
        patch.object(pipeline, "_is_chart_asset_page", return_value=False),
    ):
        pipeline.process(pdf, tmp_path / "o")
    body = _page_text(tmp_path / "o")

    with _fitz.open(pdf) as doc:
        native = plain_text_tokens(doc[0].get_text())[1]
    # #1145: a MARK is socr's own disclosure, not recovered content, and it is quoted --
    # so a blockquote filter alone now sweeps up the canonical unchecked line, whose own
    # wording ("this page", "native_coverage") is then counted as native text the reader
    # saw. Excluded through PRODUCTION's recogniser, so a mark that stops being recognised
    # fails this test rather than silently widening the exclusion. The assertion keeps its
    # full strength: every genuinely recovered word is still compared.
    from socr.core.marks import line_is_mark

    recovered = "\n".join(
        ln.lstrip("> ")
        for ln in body.splitlines()
        if ln.startswith(">") and "(OCR" not in ln and not line_is_mark(ln.rstrip(), 1)
    )
    leaked = plain_text_tokens(recovered)[1] & native
    assert not leaked, f"the reader saw native text: {sorted(leaked)}"


def test_a_wedge_in_one_document_does_not_halt_the_next(tmp_path: Path) -> None:
    """Round 3, item 2: the latch is per DOCUMENT, not per process.

    `_recovery_backend_wedged` is a class attribute and `process_batch` reuses
    one pipeline instance, so a single wedged document halted every document
    after it. Pinned on two documents in one batch: the second must be read.
    """
    src = tmp_path / "in"
    src.mkdir()
    for name in ("a.pdf", "b.pdf"):
        _two_blocks_pdf(src / name, container="rect")

    seen: list[str] = []
    wedge_once = {"done": False}

    def ask(_path, _prompt):
        seen.append("call")
        if not wedge_once["done"]:
            wedge_once["done"] = True
            raise TimeoutError("read timed out")
        return RECOVERED

    pipeline = _pipeline()
    with (
        patch.object(pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]),
        patch.object(pipeline, "_resolve_judge_model", return_value=""),
        patch.object(pipeline, "_uncovered_text_reader_available", return_value=True),
        patch.object(pipeline, "_uncovered_text_ask", return_value=ask),
        patch.object(pipeline, "_is_chart_asset_page", return_value=False),
    ):
        pipeline.process_batch(src, tmp_path / "o")

    assert len(seen) >= 2, "the second document is still read after the first wedged"
    bodies = [p.read_text(encoding="utf-8") for p in (tmp_path / "o").rglob("pages/00001.md")]
    assert any(PAGE_TEXT_PREFIX in b for b in bodies), "and it recovers its words"


# --- the whole-figure rule (owner ruling 2026-10-14) --------------------------


def _chart_with_labels_pdf(path: Path, *, plot_marks: bool) -> Path:
    """A drawn cluster whose labels are vector-drawn.

    With *plot_marks* it is a chart: bars sit inside the cluster, away from the
    label ink. Without them the cluster is nothing but drawn TEXT in a frame --
    Schoenegger p10's prompt box -- and must stay prose.
    """
    doc = fitz.open()
    page = doc.new_page()
    y = 60.0
    for line in _ABOVE_LINES:
        page.insert_text((72, y), line)
        y += 13.0
    page.draw_rect(fitz.Rect(64, 120, 464, 420), color=(0.4, 0.4, 0.4), width=0.8)
    # A thin filled rect, not draw_line: a zero-height line gives an EMPTY
    # fitz.Rect, which the figure test skips before the rule-thickness check.
    page.draw_rect(fitz.Rect(80, 300, 448, 300.4), color=(0, 0, 0), fill=(0, 0, 0))
    if plot_marks:
        for i, h in enumerate((120.0, 80.0, 160.0, 60.0)):
            x = 110.0 + i * 70
            page.draw_rect(
                fitz.Rect(x, 380 - h, x + 40, 380), color=(0, 0, 0), fill=(0.2, 0.4, 0.8)
            )
    _glyph_row(page, 150.0)
    _glyph_row(page, 400.0)
    y = 560.0
    for line in _BELOW_LINES:
        page.insert_text((72, y), line)
        y += 13.0
    doc.save(path)
    doc.close()
    return path


def test_labels_in_one_figure_become_one_whole_figure_crop(tmp_path: Path) -> None:
    """Owner ruling: one picture per figure, not one per label.

    Pinned as a difference over the plot marks alone: same page, same labels,
    with and without bars inside the cluster.
    """
    from socr.core.text_coverage import crop_box

    chart = _regions(_chart_with_labels_pdf(tmp_path / "chart.pdf", plot_marks=True))
    assert len(chart) == 1
    assert chart[0].merged_from > 1, "every label in the figure is one region"
    assert chart[0].figure_pdf is not None, "and its crop is the whole figure"
    fig = fitz.Rect(chart[0].figure_pdf)
    assert fig.get_area() > fitz.Rect(chart[0].pdf).get_area()
    assert crop_box(chart[0]) == chart[0].figure_pdf


def test_a_box_of_drawn_text_is_not_a_figure(tmp_path: Path) -> None:
    """The owner was explicit: the prompt box "is just text", not an image.

    `chart_region_bboxes` returns one cluster for a chart AND for a box of
    vector-drawn text, and `has_chart_marks` is False on both. The separation
    is that a chart carries marks away from its label ink.
    """
    plain = _regions(_chart_with_labels_pdf(tmp_path / "text.pdf", plot_marks=False))
    assert all(r.figure_pdf is None for r in plain)


def test_the_saved_picture_is_the_whole_figure(tmp_path: Path) -> None:
    """The crop on disk must be the figure, not the sliver of label ink."""
    from PIL import Image

    from socr.core.text_coverage import SCALE

    pdf = _chart_with_labels_pdf(tmp_path / "chart.pdf", plot_marks=True)
    _run(pdf, tmp_path / "o", reading=RECOVERED)
    png = next((tmp_path / "o").rglob("figures/uncovered_text_p1_r1.png"))
    region = _regions(pdf)[0]
    with Image.open(png) as im:
        width, height = im.size
    fig = fitz.Rect(region.figure_pdf)
    assert abs(width - fig.width * SCALE) <= 2
    assert abs(height - fig.height * SCALE) <= 2
    assert height > fitz.Rect(region.pdf).height * SCALE + 2


def test_a_vector_figures_words_go_beneath_its_picture(tmp_path: Path) -> None:
    """Item F: a matplotlib axes frame is a drawn rect, not a raster."""
    chart = _chart_with_labels_pdf(tmp_path / "chart.pdf", plot_marks=True)
    plain = _chart_with_labels_pdf(tmp_path / "text.pdf", plot_marks=False)
    chart_text = _run(chart, tmp_path / "o_chart", reading=RECOVERED)
    plain_text = _run(plain, tmp_path / "o_plain", reading=RECOVERED)
    assert FIGURE_TEXT_PREFIX in chart_text, "a vector chart's words sit under its picture"
    assert PAGE_TEXT_PREFIX in plain_text, "a text box's words are prose"
    ref = uncovered_text_image_ref(1, 1)
    assert chart_text.index(ref) < chart_text.index(FIGURE_TEXT_PREFIX)


# --- fix round 2, items A and B ----------------------------------------------


def _right_column_chart_pdf(path: Path) -> Path:
    """A chart in the RIGHT column with ordinary prose in the left one."""
    doc = fitz.open()
    page = doc.new_page()
    y = 150.0
    for i in range(8):
        page.insert_text((60, y), f"Left line {i}, never duplicated.", fontsize=9)
        y += 13.0
    page.draw_rect(fitz.Rect(330, 140, 560, 340), color=(0.4, 0.4, 0.4), width=0.8)
    for i, h in enumerate((80.0, 140.0, 60.0)):
        x = 350.0 + i * 60
        page.draw_rect(fitz.Rect(x, 320 - h, x + 40, 320), color=(0, 0, 0), fill=(0.2, 0.4, 0.8))
    _glyph_row(page, 160.0, x0=350.0)
    _glyph_row(page, 330.0, x0=350.0)
    doc.save(path)
    doc.close()
    return path


def test_a_right_column_chart_does_not_swallow_the_left_column(tmp_path: Path) -> None:
    """Item A: the figure box is the figure's OWN extent, not the expanded rect."""
    pdf = _right_column_chart_pdf(tmp_path / "two_col.pdf")
    with fitz.open(pdf) as doc:
        page = doc[0]
        figures = [r for r in analyze_page(page).regions if r.figure_pdf]
        prose = [b for b in page.get_text("blocks") if b[6] == 0 and "Left line" in str(b[4])]
    assert figures, "the chart is still a figure"
    assert prose, "the fixture has left-column prose"
    for reg in figures:
        x0, y0, x1, y1 = reg.figure_pdf
        for b in prose:
            assert not (b[0] < x1 and b[2] > x0 and b[1] < y1 and b[3] > y0), (
                "the figure crop reaches into the other column"
            )


def _framed_callout_pdf(path: Path) -> Path:
    """A framed callout: an icon, a drawn heading, and an ordinary native sentence."""
    doc = fitz.open()
    page = doc.new_page()
    y = 60.0
    for line in _ABOVE_LINES:
        page.insert_text((72, y), line)
        y += 13.0
    page.draw_rect(fitz.Rect(64, 150, 464, 300), color=(0.4, 0.4, 0.4), width=0.8)
    page.draw_circle(fitz.Point(90, 180), 10, color=(0, 0, 0), fill=(0.8, 0.2, 0.2))
    _glyph_row(page, 170.0, x0=120.0)
    page.insert_text((80, 240), "This callout sentence is ordinary native prose inside the frame.")
    doc.save(path)
    doc.close()
    return path


def test_a_framed_callout_is_not_a_figure(tmp_path: Path) -> None:
    """Item A: a frame plus an icon around native prose must not become a figure.

    Pinned as a difference against the same frame whose native sentence is
    removed: the sentence is the only thing that changes.
    """
    with_prose = _framed_callout_pdf(tmp_path / "callout.pdf")
    bare = _framed_callout_pdf(tmp_path / "bare.pdf")
    doc = fitz.open(with_prose)
    bdoc = fitz.open(bare)
    bare_page = bdoc[0]
    bare_page.add_redact_annot(fitz.Rect(70, 225, 460, 255))
    bare_page.apply_redactions()
    assert all(r.figure_pdf is None for r in analyze_page(doc[0]).regions), (
        "a callout around native prose is not a figure"
    )
    assert any(r.figure_pdf is not None for r in analyze_page(bare_page).regions), (
        "the same frame and icon WITHOUT the sentence is one"
    )
    doc.close()
    bdoc.close()


def _raster_bar_chart_pdf(path: Path) -> Path:
    """A bar chart placed as a RASTER, with its labels drawn as vector shapes."""
    doc = fitz.open()
    page = doc.new_page()
    chart = fitz.open()
    cp = chart.new_page(width=400, height=300)
    cp.draw_rect(fitz.Rect(0, 0, 400, 300), color=(1, 1, 1), fill=(1, 1, 1))
    for i, h in enumerate((120.0, 200.0, 90.0, 160.0)):
        x = 40.0 + i * 85
        cp.draw_rect(fitz.Rect(x, 260 - h, x + 55, 260), color=(0, 0, 0), fill=(0.2, 0.4, 0.8))
    page.insert_image(
        fitz.Rect(90, 150, 490, 450), pixmap=cp.get_pixmap(alpha=False), overlay=False
    )
    chart.close()
    _glyph_row(page, 420.0, x0=120.0)
    y = 60.0
    for line in _ABOVE_LINES:
        page.insert_text((72, y), line)
        y += 13.0
    doc.save(path)
    doc.close()
    return path


def test_a_raster_chart_ships_one_picture_of_the_whole_image(tmp_path: Path) -> None:
    """Item B: a placed image IS a figure; it has no vector marks to find."""
    pdf = _raster_bar_chart_pdf(tmp_path / "raster.pdf")
    with fitz.open(pdf) as doc:
        page = doc[0]
        regions = analyze_page(page).regions
        placed = [fitz.Rect(i["bbox"]) for i in page.get_image_info() if i.get("bbox")]
    figures = [r for r in regions if r.figure_pdf]
    assert figures, "a raster chart is a figure"
    box = fitz.Rect(figures[0].figure_pdf)
    assert box.get_area() > fitz.Rect(figures[0].pdf).get_area(), "not just the label strip"
    assert any(abs(box.get_area() - p.get_area()) / p.get_area() < 0.05 for p in placed)


# --- item K: the anchor stays in the region's own column ----------------------


def _two_column_pdf(path: Path) -> Path:
    """Unread ink low in the LEFT column, with native text in both columns."""
    doc = fitz.open()
    page = doc.new_page()
    y = 60.0
    for i in range(4):
        page.insert_text((60, y), f"Left column line {i}, ending at the true anchor here.")
        y += 13.0
    y = 130.0
    for i in range(4):
        page.insert_text((330, y), f"Right column line {i}, the wrong anchor entirely.")
        y += 13.0
    _glyph_row(page, 230.0, x0=60.0)
    _glyph_row(page, 242.0, x0=60.0)
    doc.save(path)
    doc.close()
    return path


def test_the_anchor_stays_in_the_regions_own_column(tmp_path: Path) -> None:
    """Item K: a left-column region must not anchor on the right column."""
    from socr.core.text_coverage import preceding_text_snippet

    two_col = _two_column_pdf(tmp_path / "two.pdf")
    with fitz.open(two_col) as doc:
        page = doc[0]
        region = max(analyze_page(page).regions, key=lambda r: r.merged_from)
        anchor = preceding_text_snippet(page, region.pdf)
    assert anchor, "the left column does precede the region"
    assert "true anchor here." in anchor
    assert "wrong anchor entirely." not in anchor


# --- the two real pages the rule came from -----------------------------------
#
# Third-party PDFs in gitignored scratch, so this SKIPS in CI. The synthetic
# pair above is the enforcement; this exists because the rule was derived from
# these two pages.
_SCRATCH = Path("/Users/rubenffuertes/repos/tools/socr/scratch/routing-baseline")
_TEXT_BOX_PAGE = _SCRATCH / "review-pages/in/a_schoenegger_p10.pdf"
_CHART_PAGE = (
    _SCRATCH / "corpus/pdfs/2018__hansen_mcmahon_prat__transparency_and_delibertion"
    "_within_the_fomc_a_computational_linguistics_approach__QJE.pdf"
)


@pytest.mark.skipif(
    not (_TEXT_BOX_PAGE.is_file() and _CHART_PAGE.is_file()),
    reason="third-party review PDFs live in gitignored scratch",
)
def test_the_two_real_pages_the_rule_came_from() -> None:
    """Schoenegger p10 stays prose; Hansen p23 becomes one whole-figure picture."""
    with fitz.open(_TEXT_BOX_PAGE) as doc:
        text_box = analyze_page(doc[0]).regions
    with fitz.open(_CHART_PAGE) as doc:
        chart = analyze_page(doc[22]).regions
    assert len(text_box) == 1 and text_box[0].merged_from == 4
    assert text_box[0].figure_pdf is None, "the prompt box is text, not a figure"
    assert len(chart) == 1 and chart[0].merged_from > 1
    assert chart[0].figure_pdf is not None, "the chart's labels ship under one crop"


# --- fix round 3, item 5 ------------------------------------------------------


def test_an_unreachable_reader_does_not_group(tmp_path: Path) -> None:
    """Round 3 (Fable #5 / Argon #9): no reader ⇒ step 1, grouping included.

    Grouping exists to make ONE crop readable. On a host with no Ollama it only
    changed bytes against main. Pinned as a difference over availability alone,
    with the SAME reader injected in both arms.
    """

    def run(out: str, available: bool) -> str:
        pipeline = _pipeline()
        with (
            patch.object(
                pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]
            ),
            patch.object(pipeline, "_resolve_judge_model", return_value=""),
            patch.object(pipeline, "_uncovered_text_reader_available", return_value=available),
            patch.object(pipeline, "_uncovered_text_ask", return_value=lambda *_: RECOVERED),
            patch.object(pipeline, "_is_chart_asset_page", return_value=False),
        ):
            pipeline.process(tmp_path / "doc.pdf", tmp_path / out)
        return _page_text(tmp_path / out)

    _two_blocks_pdf(tmp_path / "doc.pdf", container="rect")
    reachable = run("o_up", True)
    unreachable = run("o_down", False)
    assert reachable.count("![Uncovered text") == 1, "one merged crop when it can be read"
    assert unreachable.count("![Uncovered text") == 2, "step 1's separate crops when it cannot"
    assert PAGE_TEXT_PREFIX in reachable
    assert PAGE_TEXT_PREFIX not in unreachable
    assert uncovered_text_marker(1) in unreachable


def test_applying_the_coverage_decoration_twice_is_idempotent(tmp_path: Path) -> None:
    """Round 3 (Argon #8): the real piece spans paragraphs, so the guard missed it.

    `_already_present` compared single paragraphs, and the shipped piece is a
    block PLUS its picture — two paragraphs. A second apply inserted it again.
    """
    from socr.core.text_recovery import insert_after_anchor

    piece = (
        f"> {PAGE_TEXT_PREFIX}: hello\n\n"
        "![Uncovered text page 1 region 1](figures/uncovered_text_p1_r1.png)"
    )
    body = "native line\n\nmore prose"
    once, placed = insert_after_anchor(body, "native line", piece)
    assert placed is True
    twice, placed_again = insert_after_anchor(once, "native line", piece)
    assert placed_again is True
    assert twice == once, "the second apply changed the page"
    assert twice.count("hello") == 1


def test_the_metadata_is_unchanged_with_recovery_off(tmp_path: Path) -> None:
    """Round 3 (Argon #9): the sidecar must not advertise step 2 when it is off."""
    import json

    pdf = _two_blocks_pdf(tmp_path / "doc.pdf", container="rect")
    _run(pdf, tmp_path / "o", reading=RECOVERED, recover_uncovered_text=False)
    meta = json.loads(next((tmp_path / "o").rglob("metadata.json")).read_text(encoding="utf-8"))
    blob = json.dumps(meta)
    assert "OCR, not verified" not in blob
    assert not CONFIRMED_STYLE_RE.search(blob), blob


def test_a_call_may_not_start_unless_the_budget_holds_a_whole_one(tmp_path: Path) -> None:
    """Round 3 (Argon #7): "> 0 left" is not a bound.

    `PageLadderBudget` documents that a call starting inside the budget keeps
    its own deadline, so a budget with one second left still admitted a
    120-second read. Pinned at the boundary: just under a full call's worth is
    refused, just over it is admitted.
    """
    from socr.engines.gemini_api import FIGURE_ASK_TIMEOUT_SEC

    pdf = _two_blocks_pdf(tmp_path / "doc.pdf", container="none", rows=(150.0, 330.0))

    def run(out: str, remaining: float) -> int:
        calls: list[str] = []

        def ask(path, _prompt):
            calls.append(str(path))
            return RECOVERED

        pipeline = _pipeline()
        budget = pipeline._uncovered_text_budget

        def sized(state, page_num, regions):
            made = budget(state, page_num, regions)
            made.total_sec = remaining
            return made

        with (
            patch.object(
                pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]
            ),
            patch.object(pipeline, "_resolve_judge_model", return_value=""),
            patch.object(pipeline, "_uncovered_text_reader_available", return_value=True),
            patch.object(pipeline, "_uncovered_text_ask", return_value=ask),
            patch.object(pipeline, "_is_chart_asset_page", return_value=False),
            patch.object(pipeline, "_uncovered_text_budget", side_effect=sized),
        ):
            pipeline.process(pdf, tmp_path / out)
        return len(calls)

    assert run("o_short", FIGURE_ASK_TIMEOUT_SEC - 1) == 0, "cannot hold a whole call"
    assert run("o_ample", FIGURE_ASK_TIMEOUT_SEC * 4) > 0, "can"


# --- fix round 4: everything left sits in the mask ----------------------------


def _invisible_layer_over_drawn_text_pdf(path: Path) -> Path:
    """An alpha-0 OCR layer over vector-drawn words.

    The sentence is not DRAWN by the text layer, but `get_text` returns it and
    the shipped native prose contains it — so if the mask paints only visible
    chars, the model reads it off the pixels and the page ships it twice.
    """
    doc = fitz.open()
    page = doc.new_page()
    y = 60.0
    for line in _ABOVE_LINES:
        page.insert_text((72, y), line)
        y += 13.0
    page.draw_rect(fitz.Rect(64, 150, 464, 300), color=(0.4, 0.4, 0.4), width=0.8)
    _glyph_row(page, 170.0, x0=100.0)
    # The ink is DRAWN (no visible text layer over it); the sentence exists only
    # as an alpha-0 layer on top, exactly as an OCR layer over a raster does.
    # Painting visible chars alone leaves this ink readable in the crop.
    _glyph_row(page, 244.0, x0=90.0)
    page.insert_text((90, 252), INVISIBLE_SENTENCE, fontsize=11, render_mode=3)
    doc.save(path)
    doc.close()
    return path


INVISIBLE_SENTENCE = "An invisible layer carries this sentence already."


def test_an_invisible_native_layer_is_masked_too(tmp_path: Path) -> None:
    """Round 4, item 1: paint every char the shipped prose is built from.

    An alpha-0 OCR layer over a raster or outlined text is not drawn, but
    `get_text` returns it and the native text ships it. Painting only VISIBLE
    chars left that ink readable in the crop, so the sentence shipped twice.

    Measured on the mask itself rather than through a reader, because the
    property is "this ink is gone", and an OCR engine's opinion of a synthetic
    glyph row adds noise, not evidence.
    """
    import numpy as np

    from socr.core import text_coverage as tc

    pdf = _invisible_layer_over_drawn_text_pdf(tmp_path / "inv.pdf")
    with fitz.open(pdf) as doc:
        page = doc[0]
        mat = fitz.Matrix(tc.SCALE, tc.SCALE)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB, alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        visible, invisible = tc.visible_char_boxes(page)
        assert invisible, "the fixture carries an alpha-0 layer"
        masked = tc.native_ink_masked(page, mat, arr)
        with patch.object(tc, "visible_char_boxes", return_value=(visible, [])):
            visible_only = tc.native_ink_masked(page, mat, arr)

    band = (
        slice(int(243 * tc.SCALE), int(253 * tc.SCALE)),
        slice(int(88 * tc.SCALE), int(200 * tc.SCALE)),
    )
    assert int((visible_only[band][..., 0] < 128).sum()) > 0, (
        "painting visible chars alone leaves the covered ink readable"
    )
    assert int((masked[band][..., 0] < 128).sum()) == 0, (
        "the invisible layer's ink reached the reader"
    )


def _drawn_number_below_a_baseline_pdf(path: Path) -> Path:
    """Uncovered ink just under a native line, within its descender band."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 200), "Native line with a jpqy descender tail here.", fontsize=11)
    # Drawn glyph-like marks a fifth of an em below that baseline.
    x = 76.0
    for w in (4.0, 3.2, 4.6, 3.8, 5.0, 4.1):
        page.draw_rect(fitz.Rect(x, 202.2, x + w, 208.2), color=(0, 0, 0), width=0.6)
        x += w + 2.0
    y = 60.0
    for line in _ABOVE_LINES:
        page.insert_text((72, y), line)
        y += 13.0
    doc.save(path)
    doc.close()
    return path


def test_uncovered_ink_under_a_baseline_is_not_painted_out(tmp_path: Path) -> None:
    """Round 4, item 2: paint the covered INK, never the char box.

    A char box spans ascender to descender, so box painting erased uncovered
    ink just below a native line — a drawn "0.058" came back as "U.UD&". What
    is painted is what step 1 classes as covered.
    """
    import numpy as np

    from socr.core.text_coverage import SCALE, native_ink_masked

    pdf = _drawn_number_below_a_baseline_pdf(tmp_path / "below.pdf")
    with fitz.open(pdf) as doc:
        page = doc[0]
        mat = fitz.Matrix(SCALE, SCALE)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB, alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        masked = native_ink_masked(page, mat, arr)
        band = tuple(int(v) for v in fitz.Rect(70, 201.5, 130, 209) * page.rotation_matrix * mat)
    ink_before = int((arr[band[1] : band[3], band[0] : band[2], 0] < 128).sum())
    ink_after = int((masked[band[1] : band[3], band[0] : band[2], 0] < 128).sum())
    assert ink_before > 0, "the fixture draws ink in that band"
    # Round 5, item 4: EXACT. The halo dilation that catches a glyph's
    # anti-aliased edge never enters an uncovered ink component, so a drawn
    # mark within a rule's width of a native glyph survives whole.
    assert ink_after == ink_before, "the mask ate the uncovered marks"


def test_a_mask_failure_means_no_reading(tmp_path: Path) -> None:
    """Round 4, item 3: never read an unmasked crop.

    Falling back to the unmasked crop reopened the duplication the mask exists
    to close, quietly and at debug level. Pinned as a difference over the mask
    alone, with the SAME reader in both arms.
    """
    pdf = _two_blocks_pdf(tmp_path / "doc.pdf", container="rect")
    calls: list[str] = []

    def run(out: str, broken: bool) -> str:
        ctx = (
            patch("socr.core.text_coverage.native_ink_masked", side_effect=RuntimeError("boom"))
            if broken
            else nullcontext()
        )
        pipeline = _pipeline()
        with (
            patch.object(
                pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]
            ),
            patch.object(pipeline, "_resolve_judge_model", return_value=""),
            patch.object(pipeline, "_uncovered_text_reader_available", return_value=True),
            patch.object(
                pipeline,
                "_uncovered_text_ask",
                return_value=lambda *_: (calls.append("x"), RECOVERED)[1],
            ),
            patch.object(pipeline, "_is_chart_asset_page", return_value=False),
            ctx,
        ):
            pipeline.process(pdf, tmp_path / out)
        return _page_text(tmp_path / out)

    ok = run("o_ok", False)
    broken = run("o_broken", True)
    assert PAGE_TEXT_PREFIX in ok
    assert PAGE_TEXT_PREFIX not in broken, "no mask must mean no reading, not an unmasked read"
    assert uncovered_text_marker(1) in broken, "step 1's picture and mark are kept"


def test_a_fail_fast_deadline_counts_as_a_wedge(tmp_path: Path) -> None:
    """Round 4, item 4: a sub-millisecond deadline is still a wedge.

    `TotalDeadlineExceeded` from the outstanding-call guard used to be folded
    into `ask`'s `None`, so a wedge that shows up as a fail-fast was invisible.
    """
    from socr.core.ollama_utils import TotalDeadlineExceeded

    pdf = _two_blocks_pdf(tmp_path / "doc.pdf", container="rect")

    def ask(_path, _prompt):
        raise TotalDeadlineExceeded("outstanding call")

    pipeline = _pipeline()
    with (
        patch.object(pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]),
        patch.object(pipeline, "_resolve_judge_model", return_value=""),
        patch.object(pipeline, "_uncovered_text_reader_available", return_value=True),
        patch.object(pipeline, "_uncovered_text_ask", return_value=ask),
        patch.object(pipeline, "_is_chart_asset_page", return_value=False),
    ):
        pipeline.process(pdf, tmp_path / "o")
    assert pipeline._recovery_backend_wedged is True


def test_coverage_off_never_probes_the_reader(tmp_path: Path) -> None:
    """Round 4, item 6: no flagged pages means nothing to recover."""
    pdf = _two_blocks_pdf(tmp_path / "doc.pdf", container="rect")
    pipeline = _pipeline(check_text_coverage=False)
    with (
        patch.object(pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]),
        patch.object(pipeline, "_resolve_judge_model", return_value=""),
        patch.object(pipeline, "_is_chart_asset_page", return_value=False),
        patch.object(pipeline, "_uncovered_text_reader_available") as probe,
    ):
        pipeline.process(pdf, tmp_path / "o")
    probe.assert_not_called()


def test_applying_the_phase_twice_ships_the_same_bytes(tmp_path: Path) -> None:
    """Round 4, item 8: a second apply must change nothing at all."""
    pdf = _two_blocks_pdf(tmp_path / "doc.pdf", container="rect")
    captured: dict = {}
    reads: list[str] = []
    pipeline = _pipeline()
    apply = pipeline._apply_text_coverage

    def twice(state, page_num, ps, output_dir):
        def recovery_events() -> int:
            return sum(1 for e in state.events if str(e.kind).startswith("uncovered_text"))

        apply(state, page_num, ps, output_dir)
        once = (ps.best_output.text or "") if ps.best_output else ""
        captured["reads_after_first"] = len(reads)
        captured["events_after_first"] = recovery_events()
        apply(state, page_num, ps, output_dir)
        captured["once"] = once
        captured["twice"] = (ps.best_output.text or "") if ps.best_output else ""
        captured["reads"] = len(reads)
        captured["events"] = recovery_events()

    with (
        patch.object(pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]),
        patch.object(pipeline, "_resolve_judge_model", return_value=""),
        patch.object(pipeline, "_uncovered_text_reader_available", return_value=True),
        patch.object(
            pipeline,
            "_uncovered_text_ask",
            return_value=lambda *_: (reads.append("x"), RECOVERED)[1],
        ),
        patch.object(pipeline, "_is_chart_asset_page", return_value=False),
        patch.object(pipeline, "_apply_text_coverage", side_effect=twice),
    ):
        pipeline.process(pdf, tmp_path / "o")
    assert captured["once"], "the fixture decorates the page"
    assert captured["twice"] == captured["once"]
    # Round 5, item 5: not just the bytes. A second apply must not re-read the
    # crop, nor append a second copy of the region's audit event.
    assert captured["reads"] == captured["reads_after_first"], "the crop was read again"
    assert captured["events"] == captured["events_after_first"], "the event was logged twice"


def test_a_budget_refusal_says_so(tmp_path: Path) -> None:
    """Round 4, item 5: OUTPUT.md promises an event; the refusal emitted none.

    `PageLadderBudget.call` only announces when its own clock is spent, so a
    region refused for want of a WHOLE call's room was silent.
    """
    from socr.engines.gemini_api import FIGURE_ASK_TIMEOUT_SEC

    pdf = _two_blocks_pdf(tmp_path / "doc.pdf", container="none", rows=(150.0, 330.0))
    pipeline = _pipeline()
    budget = pipeline._uncovered_text_budget

    def sized(state, page_num, regions):
        made = budget(state, page_num, regions)
        made.total_sec = FIGURE_ASK_TIMEOUT_SEC - 1
        return made

    with (
        patch.object(pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]),
        patch.object(pipeline, "_resolve_judge_model", return_value=""),
        patch.object(pipeline, "_uncovered_text_reader_available", return_value=True),
        patch.object(pipeline, "_uncovered_text_ask", return_value=lambda *_: RECOVERED),
        patch.object(pipeline, "_is_chart_asset_page", return_value=False),
        patch.object(pipeline, "_uncovered_text_budget", side_effect=sized),
    ):
        pipeline.process(pdf, tmp_path / "o")
    events = json.loads(next((tmp_path / "o").rglob("audit_log.json")).read_text(encoding="utf-8"))
    kinds = [
        e.get("kind") for e in (events if isinstance(events, list) else events.get("events", []))
    ]
    assert "uncovered_text_budget_exhausted" in kinds


def test_the_flag_off_page_is_byte_identical_to_main(tmp_path: Path) -> None:
    """Round 4, item 7: the step-1 joiner mutant must die inside the suite.

    With recovery off, step 2 must change NOTHING — including the single
    newline step 1 used between picture links. Compared against the bytes
    main's own code produces for the same page, generated here from the
    un-grouped regions rather than trusted from a stored file.
    """
    from socr.core.text_coverage import (
        analyze_page,
        uncovered_text_image_ref,
        uncovered_text_marker,
    )

    pdf = _two_blocks_pdf(tmp_path / "doc.pdf", container="rect")
    off = _run(pdf, tmp_path / "o_off", reading=RECOVERED, recover_uncovered_text=False)
    with fitz.open(pdf) as doc:
        regions = analyze_page(doc[0], group=False).regions
    expected_tail = (
        uncovered_text_marker(1)
        + "\n\n"
        + "\n".join(uncovered_text_image_ref(1, i) for i in range(1, len(regions) + 1))
    )
    assert off.rstrip().endswith(expected_tail), "step 1 joined picture links with a single newline"


def test_the_metadata_matches_main_with_recovery_off() -> None:
    """Round 5, item 6: the flag must leave a run's identity alone when it is off.

    Hashing `recover_uncovered_text` unconditionally made every OFF run
    invalidate main's fragments and force a reprocess that changes nothing —
    the GH-142 failure shape in the other direction. Load-bearing: it inspects
    the fingerprint PAYLOAD, so restoring the unconditional key fails it. A
    test on the digest alone cannot, because both configs then carry the key
    and still agree with each other.
    """
    captured: dict = {}

    def capture(*args, **kwargs):
        captured.update(kwargs.get("extra") or {})
        return "fp:stub"

    for flag in (True, False):
        pipeline = _pipeline(recover_uncovered_text=flag)
        captured.clear()
        with patch("ocr_output_contract.run_fingerprint", side_effect=capture):
            pipeline._run_fingerprint()
        present = "recover_uncovered_text" in captured
        assert present is flag, (
            "ON must change the run's identity; OFF must leave it exactly as main's"
        )


# --- fix round 5 --------------------------------------------------------------


def _native_label_on_a_fill_pdf(path: Path, fill, color) -> Path:
    """A native label over a coloured fill, as a raster chart's labels sit."""
    doc = fitz.open()
    page = doc.new_page()
    page.draw_rect(fitz.Rect(64, 130, 464, 400), color=fill, fill=fill)
    page.insert_text((90, 200), "HUGE 40 BIG NATIVE words on the fill", fontsize=18, color=color)
    page.insert_text((72, 60), "Ordinary prose above, with a text layer.")
    doc.save(path)
    doc.close()
    return path


@pytest.mark.parametrize(
    ("tag", "fill", "color"),
    [
        ("ggplot_panel", (0.92, 0.92, 0.92), (0, 0, 0)),
        ("shaded", (0.85, 0.85, 0.85), (0, 0, 0)),
        ("mid_grey", (0.6, 0.6, 0.6), (0, 0, 0)),
        ("dark_white_text", (0.2, 0.2, 0.2), (1, 1, 1)),
    ],
)
def test_the_mask_leaves_no_contrast_on_a_coloured_fill(
    tmp_path: Path, tag: str, fill, color
) -> None:
    """Round 5, item 1: painting covered glyphs WHITE was itself a signal.

    On a grey or dark fill — a chart exported as a raster with native labels
    over it — white painting handed the model fat white words, which it read.
    Measured with the reviewer's metric: contrast pixels inside the native char
    boxes, against the fill's own grey. Pinned as a difference against white
    painting on the same page.
    """
    import numpy as np

    from socr.core import text_coverage as tc

    pdf = _native_label_on_a_fill_pdf(tmp_path / f"{tag}.pdf", fill, color)

    def white(arr, paint, _protect):
        out = arr.copy()
        out[paint] = 255
        return out

    with fitz.open(pdf) as doc:
        page = doc[0]
        mat = fitz.Matrix(tc.SCALE, tc.SCALE)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB, alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        visible, _inv = tc.visible_char_boxes(page)
        boxes = [b for b in visible if b[1] > 130 and b[3] < 400]
        masked = tc.native_ink_masked(page, mat, arr)
        with patch.object(tc, "_fill_with_local_background", side_effect=white):
            whitened = tc.native_ink_masked(page, mat, arr)

        grey = int(round(255 * fill[0]))

        def contrast(img) -> int:
            total = 0
            for box in boxes:
                x0, y0, x1, y1 = tc._to_pix(box, page, mat)
                total += int((np.abs(img[y0:y1, x0:x1, 0].astype(int) - grey) > 40).sum())
            return total

        before, after, white_after = contrast(arr), contrast(masked), contrast(whitened)

    assert before > 0, "the fixture puts a native label on the fill"
    assert after == 0, "the label is still legible after masking"
    if fill[0] < 0.8:
        assert white_after > before, "white painting made the label MORE visible"


def test_a_recovered_block_mid_prose_keeps_the_equation_sidecar(tmp_path: Path) -> None:
    """Round 5, item 2: silent content loss on NATIVE_FALLBACK.

    `_native_text_with_appends` accepts an attempt only if it STARTS with the
    native snapshot. A recovered block inserted at its reading-order position
    sits mid-prose, so the decorated attempt failed that test and the GH-36b
    equation-crop sidecar was dropped — from the fragment and the final .md,
    with no event. My round-3 probe anchored at the tail, which keeps the
    prefix, and so missed it.
    """
    from socr.core.document import DocumentHandle
    from socr.core.manifest import SelectionProvenance, _select_page_output_tagged
    from socr.core.result import FailureMode, PageOutput, PageStatus
    from socr.core.state import DocumentState, PageState
    from socr.core.text_recovery import insert_after_anchor, recovered_block

    pdf = _two_blocks_pdf(tmp_path / "doc.pdf", container="rect")
    base = "\n".join(_ABOVE_LINES + _BELOW_LINES)
    equation = "![equation crop](equations/p001_eq1.png)"
    block = recovered_block("Treatment Prompt", in_figure=False)
    link = uncovered_text_image_ref(1, 1)
    mark = uncovered_text_marker(1)

    # The attempt that carries the sidecar is ALSO the one apply decorated: the
    # sidecar landed first, then the coverage block went in MID-prose. A fixture
    # where some other attempt still carries the equation passes without the
    # fix, because the plain prefix test matches that one instead.
    decorated, placed = insert_after_anchor(
        base + "\n\n" + equation,
        " ".join(_ABOVE_LINES[-1].split()[-8:]),
        block + "\n\n" + link,
    )
    assert placed, "the fixture inserts mid-prose"
    assert not decorated.startswith(base), "which is what breaks the prefix test"
    decorated = decorated.rstrip() + "\n\n" + mark + "\n"

    bo = PageOutput(
        page_num=1,
        text=decorated,
        status=PageStatus.WARNING,
        engine="native",
        audit_passed=True,
        failure_mode=FailureMode.TEXT_INK_UNREAD,
    )
    ps = PageState(page_num=1)
    ps.is_born_digital = True
    ps.native_text = base
    ps.has_tables = True
    ps.attempts = [bo]
    ps.best_output = bo
    ps.text_ink_unread = True
    ps.text_coverage_verdict = "flagged"
    state = DocumentState(handle=DocumentHandle(path=pdf))
    state.pages[1] = ps

    winner, provenance = _select_page_output_tagged(state, 1)
    text = winner.text or ""
    assert provenance is SelectionProvenance.NATIVE_FALLBACK
    assert equation in text, "the equation sidecar was dropped"
    assert block in text, "and the recovered words must survive too"
    assert link in text


def test_the_real_engine_surfaces_a_fail_fast_deadline() -> None:
    """Round 5, item 3: tested through the REAL engine, not a stub above it.

    The previous test stubbed the orchestrator's own reader, so deleting the
    `gemini_api` change survived it. This patches the HTTP layer, which is the
    only thing below `ask`.
    """
    from PIL import Image

    from socr.core.ollama_utils import TotalDeadlineExceeded
    from socr.engines.gemini_api import OllamaFigureEngine

    engine = OllamaFigureEngine()
    image = Image.new("RGB", (8, 8), "white")
    with patch(
        "socr.engines.gemini_api.call_with_total_deadline",
        side_effect=TotalDeadlineExceeded("outstanding call"),
    ):
        with pytest.raises(TotalDeadlineExceeded):
            engine.ask(image, "transcribe")
    # Any OTHER failure still folds into None, as the contract says.
    with patch(
        "socr.engines.gemini_api.call_with_total_deadline", side_effect=ValueError("broken")
    ):
        assert engine.ask(image, "transcribe") is None


def test_the_figure_describer_still_never_raises() -> None:
    """Round 5, item 3, collateral: the describer's contract must hold.

    `crop_descriptions.describe_number_free` documents that its `ask` never
    raises, and the orchestrator catches once per DOCUMENT — so a fail-fast
    leaking through would drop every figure description on that document.
    """
    from socr.core.ollama_utils import TotalDeadlineExceeded

    pipeline = _pipeline()
    with (
        patch.object(UnifiedPipeline, "_figure_description_ask", _REAL_FIGURE_DESCRIPTION_ASK),
        patch("socr.engines.gemini_api.OllamaFigureEngine.is_available", return_value=True),
        patch(
            "socr.engines.gemini_api.OllamaFigureEngine.ask",
            side_effect=TotalDeadlineExceeded("outstanding call"),
        ),
    ):
        describe = pipeline._figure_description_ask()
        png = Path(__file__).parent / "_describer_probe.png"
        from PIL import Image

        Image.new("RGB", (8, 8), "white").save(png)
        try:
            assert describe(png, "prompt") is None, "the describer path must not raise"
        finally:
            png.unlink(missing_ok=True)


# --- fix round 6 --------------------------------------------------------------


def _label_over_texture_pdf(path: Path, kind: str) -> Path:
    """A native label printed over a TEXTURED raster, plus an uncovered strip.

    A glyph crossing hatching, diagonals or grain merges into the texture's own
    connected component, so a majority-covered test never fires for it.
    """
    import numpy as np

    doc = fitz.open()
    page = doc.new_page()
    inner = fitz.open()
    ip = inner.new_page(width=400, height=200)
    ip.draw_rect(fitz.Rect(0, 0, 400, 200), color=(1, 1, 1), fill=(1, 1, 1))
    if kind == "hatch":
        for x in range(0, 400, 6):
            ip.draw_line(fitz.Point(x, 0), fitz.Point(x, 200), color=(0, 0, 0), width=1)
    elif kind == "diagonal":
        for x in range(-200, 400, 8):
            ip.draw_line(fitz.Point(x, 0), fitz.Point(x + 200, 200), color=(0, 0, 0), width=1)
    else:
        rng = np.random.default_rng(7)
        for _ in range(4000):
            x, y = int(rng.integers(0, 400)), int(rng.integers(0, 200))
            ip.draw_rect(fitz.Rect(x, y, x + 2, y + 2), color=(0, 0, 0), fill=(0, 0, 0))
    page.insert_image(fitz.Rect(72, 150, 472, 350), pixmap=ip.get_pixmap(alpha=False))
    inner.close()
    page.insert_text((100, 250), "NATIVE LABEL OVER TEXTURE", fontsize=16, color=(0, 0, 0))
    # An UNCOVERED drawn strip inside the same raster: it must survive.
    x = 110.0
    for width in (4.0, 3.2, 4.6, 3.8, 5.0, 4.1):
        page.draw_rect(fitz.Rect(x, 300, x + width, 308), color=(0, 0, 0), width=0.6)
        x += width + 2.0
    page.insert_text((72, 60), "Prose above, with a text layer.")
    doc.save(path)
    doc.close()
    return path


@pytest.mark.parametrize("kind", ["hatch", "diagonal", "noise"])
def test_a_label_over_a_texture_is_masked(tmp_path: Path, kind: str) -> None:
    """Round 6, item 2: a glyph that merges with a texture must still be painted.

    The majority rule asked whether most of a COMPONENT lies under a char box.
    A glyph crossing hatching joins the texture's component, which is almost
    entirely outside, so the glyph stayed legible in the crop. What is painted
    now is the ink INSIDE the char boxes for any component bigger than a glyph
    that touches one — and a GLYPH-SIZED component that merely overlaps a box
    is the opposite case, a drawn mark, which must survive.
    """
    import numpy as np

    from socr.core import text_coverage as tc

    pdf = _label_over_texture_pdf(tmp_path / f"{kind}.pdf", kind)
    with fitz.open(pdf) as doc:
        page = doc[0]
        mat = fitz.Matrix(tc.SCALE, tc.SCALE)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB, alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        masked = tc.native_ink_masked(page, mat, arr)
        visible, _inv = tc.visible_char_boxes(page)
        label_boxes = [b for b in visible if 200 < b[1] < 300]

        def dark(img, boxes) -> int:
            total = 0
            for box in boxes:
                x0, y0, x1, y1 = tc._to_pix(box, page, mat)
                total += int((img[y0:y1, x0:x1, 0] < 100).sum())
            return total

        strip = tc._to_pix((105, 298, 180, 310), page, mat)

        def strip_dark(img) -> int:
            return int((img[strip[1] : strip[3], strip[0] : strip[2], 0] < 100).sum())

        before, after = dark(arr, label_boxes), dark(masked, label_boxes)
        strip_before, strip_after = strip_dark(arr), strip_dark(masked)

    assert before > 0, "the fixture prints a label on the texture"
    assert after <= before * 0.02, "the label is still legible after masking"
    assert strip_before > 0, "and draws an uncovered strip"
    assert strip_after == strip_before, "which the mask must not touch"


def test_prior_recoveries_pair_each_region_with_its_own_block(tmp_path: Path) -> None:
    """Round 6, item 3: look where the block ACTUALLY goes, per region kind.

    In a figure the image leads and the words follow; elsewhere the words lead
    and the image follows. Searching both directions could pair a region with
    its NEIGHBOUR's block, so a second apply would re-attribute the readings.
    Pinned on a read/unread pair: applying twice must change nothing.
    """
    pdf = _chart_with_labels_pdf(tmp_path / "chart.pdf", plot_marks=True)
    captured: dict = {}
    reads: list[str] = []
    pipeline = _pipeline()
    apply = pipeline._apply_text_coverage

    def twice(state, page_num, ps, output_dir):
        apply(state, page_num, ps, output_dir)
        captured["once"] = (ps.best_output.text or "") if ps.best_output else ""
        captured["reads_once"] = len(reads)
        apply(state, page_num, ps, output_dir)
        captured["twice"] = (ps.best_output.text or "") if ps.best_output else ""
        captured["reads_twice"] = len(reads)

    answers = iter([RECOVERED, None, RECOVERED, None])

    def ask(_path, _prompt):
        reads.append("x")
        return next(answers, None)

    with (
        patch.object(pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]),
        patch.object(pipeline, "_resolve_judge_model", return_value=""),
        patch.object(pipeline, "_uncovered_text_reader_available", return_value=True),
        patch.object(pipeline, "_uncovered_text_ask", return_value=ask),
        patch.object(pipeline, "_is_chart_asset_page", return_value=False),
        patch.object(pipeline, "_apply_text_coverage", side_effect=twice),
    ):
        pipeline.process(pdf, tmp_path / "o")

    assert captured["once"], "the fixture decorates the page"
    assert captured["twice"] == captured["once"], "a second apply re-attributed the readings"
    assert captured["reads_twice"] == captured["reads_once"], (
        "an UNREAD prior must not be read again either"
    )


def test_the_collapsed_prefix_test_only_runs_on_a_text_we_edited() -> None:
    """Round 6 (repo soft 2): line breaks carry meaning in a Markdown table.

    The whitespace-collapsed prefix test is safe on a text THIS feature just
    edited, because inserting a block promotes a newline to a blank line. On an
    untouched attempt it would also collapse the row breaks of a table and
    accept a text that is not a true extension of the snapshot.
    """
    from socr.core.manifest import _native_text_with_appends
    from socr.core.result import PageOutput, PageStatus
    from socr.core.state import PageState

    # The snapshot is a two-row table; the attempt has the SAME words with the
    # row break lost. Collapsed they match; as Markdown they are different
    # tables, and this attempt carries no coverage part to justify collapsing.
    base = "| a | b |\n| 1 | 2 |"
    mangled = "| a | b | | 1 | 2 | plus a trailing sentence that makes it longer."
    ps = PageState(page_num=1)
    ps.native_text = base
    ps.attempts = [
        PageOutput(
            page_num=1,
            text=mangled,
            status=PageStatus.SUCCESS,
            engine="native",
            audit_passed=True,
        )
    ]
    assert len(mangled) > len(base)
    assert " ".join(mangled.split()).startswith(" ".join(base.split()))
    assert _native_text_with_appends(ps) == base, (
        "an untouched attempt must face the EXACT prefix test"
    )


def _two_figures_pdf(path: Path) -> Path:
    """Two separate vector charts on one page, each with a drawn label."""
    doc = fitz.open()
    page = doc.new_page()
    y = 60.0
    for line in _ABOVE_LINES:
        page.insert_text((72, y), line)
        y += 13.0
    for top in (140.0, 420.0):
        page.draw_rect(fitz.Rect(64, top, 464, top + 200), color=(0.4, 0.4, 0.4), width=0.8)
        for i, h in enumerate((80.0, 140.0, 60.0)):
            x = 110.0 + i * 90
            page.draw_rect(
                fitz.Rect(x, top + 180 - h, x + 50, top + 180),
                color=(0, 0, 0),
                fill=(0.2, 0.4, 0.8),
            )
        _glyph_row(page, top + 15.0, x0=110.0)
    doc.save(path)
    doc.close()
    return path


def test_each_figure_keeps_its_own_words_on_a_second_apply(tmp_path: Path) -> None:
    """Round 6, item 3: pair a region with ITS block, not its neighbour's.

    In a figure the image leads and the words follow. Searching both directions
    finds the PREVIOUS figure's block first, so a second apply re-attributed
    the readings. Pinned on two figures where only the first was read: applying
    twice must change nothing, and must not re-read the unread one.
    """
    pdf = _two_figures_pdf(tmp_path / "two_figs.pdf")
    regions = _regions(pdf)
    assert len([r for r in regions if r.figure_pdf]) == 2, "the fixture has two figures"

    captured: dict = {}
    reads: list[str] = []
    answers = iter([RECOVERED, None])
    pipeline = _pipeline()
    apply = pipeline._apply_text_coverage

    def ask(_path, _prompt):
        reads.append("x")
        return next(answers, None)

    def twice(state, page_num, ps, output_dir):
        apply(state, page_num, ps, output_dir)
        captured["once"] = (ps.best_output.text or "") if ps.best_output else ""
        captured["reads_once"] = len(reads)
        apply(state, page_num, ps, output_dir)
        captured["twice"] = (ps.best_output.text or "") if ps.best_output else ""
        captured["reads_twice"] = len(reads)

    with (
        patch.object(pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]),
        patch.object(pipeline, "_resolve_judge_model", return_value=""),
        patch.object(pipeline, "_uncovered_text_reader_available", return_value=True),
        patch.object(pipeline, "_uncovered_text_ask", return_value=ask),
        patch.object(pipeline, "_is_chart_asset_page", return_value=False),
        patch.object(pipeline, "_apply_text_coverage", side_effect=twice),
    ):
        pipeline.process(pdf, tmp_path / "o")

    assert captured["once"].count(RECOVERED.splitlines()[0]) == 1, "one figure was read"
    assert captured["twice"] == captured["once"], "a second apply re-attributed the readings"
    assert captured["reads_twice"] == captured["reads_once"], "the unread figure was read again"


# --- fix round 7 --------------------------------------------------------------


def _ticks_under_a_native_line_pdf(path: Path) -> Path:
    """Drawn tick labels on an axis rule, with a native line 2 pt above them."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 200), "Native caption line sits just above the axis.", fontsize=10)
    page.draw_rect(fitz.Rect(72, 203, 400, 204), color=(0, 0, 0), fill=(0, 0, 0))
    x = 80.0
    for width in (4.0, 3.2, 4.6, 3.8, 5.0, 4.1, 3.6, 4.4):
        page.draw_rect(fitz.Rect(x, 205, x + width, 212), color=(0, 0, 0), width=0.6)
        x += width + 2.0
    page.insert_text((72, 60), "Prose above, with a text layer.")
    doc.save(path)
    doc.close()
    return path


def test_a_textures_own_ink_outside_a_char_box_is_protected(tmp_path: Path) -> None:
    """Round 7, item 1 — a regression introduced in round 6.

    A component big enough to be a texture was dropped from `protect`
    altogether, so the fill's halo could shave the part of it OUTSIDE the char
    boxes — which is uncovered ink. Measured on an axis rule whose drawn tick
    labels sit 2 pt under a native line: "2019 0.058 2020" came back as
    "2U19 U.U58 2U2U". Only the ink under a char box is this feature's to
    paint; the rest of the same component is protected like any other.
    """
    import numpy as np

    from socr.core import text_coverage as tc

    pdf = _ticks_under_a_native_line_pdf(tmp_path / "ticks.pdf")
    with fitz.open(pdf) as doc:
        page = doc[0]
        mat = fitz.Matrix(tc.SCALE, tc.SCALE)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB, alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        masked = tc.native_ink_masked(page, mat, arr)
        band = tc._to_pix((70, 204.5, 200, 213), page, mat)

    def dark(img) -> int:
        return int((img[band[1] : band[3], band[0] : band[2], 0] < 128).sum())

    assert dark(arr) > 0, "the fixture draws tick labels under the native line"
    assert dark(masked) == dark(arr), "the halo shaved the tick labels"


def test_an_all_unread_page_is_not_re_read_on_a_second_apply(tmp_path: Path) -> None:
    """Round 7, item 2: I reported this covered in round 6. It was not.

    An all-unread page joins its picture links with a SINGLE newline — that is
    what makes it byte-identical to step 1. `_prior_recoveries` split on blank
    lines only, so it saw one paragraph holding every ref, matched none, and a
    second apply re-read every crop and doubled `uncovered_text_unread`.
    """
    pdf = _two_blocks_pdf(tmp_path / "doc.pdf", container="none", rows=(150.0, 330.0))
    captured: dict = {}
    reads: list[str] = []
    pipeline = _pipeline()
    apply = pipeline._apply_text_coverage

    def ask(_path, _prompt):
        reads.append("x")
        return None  # nothing is ever read: the all-unread page

    def twice(state, page_num, ps, output_dir):
        def unread_events() -> int:
            return sum(1 for e in state.events if e.kind == "uncovered_text_unread")

        apply(state, page_num, ps, output_dir)
        captured["once"] = (ps.best_output.text or "") if ps.best_output else ""
        captured["reads_once"], captured["events_once"] = len(reads), unread_events()
        apply(state, page_num, ps, output_dir)
        captured["twice"] = (ps.best_output.text or "") if ps.best_output else ""
        captured["reads_twice"], captured["events_twice"] = len(reads), unread_events()

    with (
        patch.object(pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]),
        patch.object(pipeline, "_resolve_judge_model", return_value=""),
        patch.object(pipeline, "_uncovered_text_reader_available", return_value=True),
        patch.object(pipeline, "_uncovered_text_ask", return_value=ask),
        patch.object(pipeline, "_is_chart_asset_page", return_value=False),
        patch.object(pipeline, "_apply_text_coverage", side_effect=twice),
    ):
        pipeline.process(pdf, tmp_path / "o")

    assert captured["reads_once"] >= 2, "the fixture offers several crops to the reader"
    assert "\n\n![Uncovered text page 1 region 2]" not in captured["once"], (
        "the all-unread page must keep step 1's single-newline join"
    )
    assert captured["twice"] == captured["once"]
    assert captured["reads_twice"] == captured["reads_once"], "the crops were read again"
    assert captured["events_twice"] == captured["events_once"], "the events were doubled"


def test_a_mid_page_stop_does_not_lose_the_reading_already_shipped(tmp_path: Path) -> None:
    """Round 7, item 4: the prior short-cut must be consulted BEFORE the latch.

    A reader that stops mid-page leaves earlier regions with a shipped
    reading. Checking the stop latch first discarded those on the second
    apply, so the page lost words it had already recovered.
    """
    pdf = _two_blocks_pdf(tmp_path / "doc.pdf", container="none", rows=(150.0, 330.0))
    captured: dict = {}
    answers = iter([RECOVERED])
    pipeline = _pipeline()
    apply = pipeline._apply_text_coverage

    def ask(_path, _prompt):
        try:
            return next(answers)
        except StopIteration:
            raise TimeoutError("read timed out") from None

    def twice(state, page_num, ps, output_dir):
        apply(state, page_num, ps, output_dir)
        captured["once"] = (ps.best_output.text or "") if ps.best_output else ""
        apply(state, page_num, ps, output_dir)
        captured["twice"] = (ps.best_output.text or "") if ps.best_output else ""

    with (
        patch.object(pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]),
        patch.object(pipeline, "_resolve_judge_model", return_value=""),
        patch.object(pipeline, "_uncovered_text_reader_available", return_value=True),
        patch.object(pipeline, "_uncovered_text_ask", return_value=ask),
        patch.object(pipeline, "_is_chart_asset_page", return_value=False),
        patch.object(pipeline, "_apply_text_coverage", side_effect=twice),
    ):
        pipeline.process(pdf, tmp_path / "o")

    assert pipeline._recovery_backend_wedged is True, "the fixture stops the reader mid-page"
    assert "Treatment Prompt" in captured["once"], "region 1 was read before the stop"
    assert captured["twice"] == captured["once"], "the second apply dropped the shipped reading"


def test_the_real_reader_path_stops_on_a_timeout(tmp_path: Path) -> None:
    """Round 7, item 3: I reported this done in round 4. It was not.

    Every stop test stubbed `_uncovered_text_ask`, so wrapping `engine.ask` in
    `except TimeoutError: return None` survived them all. This drives the REAL
    `_uncovered_text_ask` → `engine.ask` path, patching only the HTTP layer.
    """
    from socr.core.ollama_utils import TotalDeadlineExceeded

    pdf = _two_blocks_pdf(tmp_path / "doc.pdf", container="rect")
    pipeline = _pipeline()
    with (
        patch.object(pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]),
        patch.object(pipeline, "_resolve_judge_model", return_value=""),
        patch.object(pipeline, "_uncovered_text_reader_available", return_value=True),
        patch.object(pipeline, "_is_chart_asset_page", return_value=False),
        patch("socr.engines.gemini_api.OllamaFigureEngine.is_available", return_value=True),
        patch(
            "socr.engines.gemini_api.call_with_total_deadline",
            side_effect=TotalDeadlineExceeded("outstanding call"),
        ),
    ):
        pipeline.process(pdf, tmp_path / "o")
    assert pipeline._recovery_backend_wedged is True, (
        "a timeout from the real engine must stop recovery for the document"
    )
    body = _page_text(tmp_path / "o")
    assert PAGE_TEXT_PREFIX not in body
    assert uncovered_text_marker(1) in body, "step 1's picture and mark are kept"


def test_an_unread_figure_does_not_claim_the_next_regions_words(tmp_path: Path) -> None:
    """Round 7, item 5 (repo reviewer, latent): a mixed figure-then-prose tail.

    A figure's block follows its picture; a prose region's block PRECEDES its
    own. So the paragraph after an UNREAD figure's picture is the next prose
    region's block — and pairing by direction alone handed it to the figure.
    """
    from socr.core.text_coverage import uncovered_text_image_ref
    from socr.core.text_recovery import recovered_block
    from socr.pipeline.orchestrator import _prior_recoveries

    class _Region:
        def __init__(self, in_figure: bool) -> None:
            self.figure_pdf = (0.0, 0.0, 1.0, 1.0) if in_figure else None
            self.kind = "raster" if in_figure else "vector_or_bare"
            self.merged_from = 1

    prose_block = recovered_block("PROSE WORDS", in_figure=False)
    body = "\n\n".join(
        [
            "native prose",
            uncovered_text_image_ref(1, 1),  # an UNREAD figure: picture only
            prose_block,  # the PROSE region's words...
            uncovered_text_image_ref(1, 2),  # ...then its picture
        ]
    )
    found = _prior_recoveries(body, 1, [_Region(True), _Region(False)])
    assert found[1].piece is None, "the unread figure claimed the prose region's words"
    assert found[2].piece is not None and "PROSE WORDS" in found[2].piece


def test_a_repeated_reading_is_carried_once_per_region() -> None:
    """Round 7 addendum (cubic): count the parts, do not merely test membership.

    Two regions on a page can produce the SAME reading — a repeated axis
    label, a "Source:" line. Treating one copy in the snapshot as proof that
    both were there dropped the second silently.
    """
    from socr.core.manifest import _coverage_parts_to_carry
    from socr.core.text_recovery import recovered_block

    block = recovered_block("Same words", in_figure=False)
    decorated = "\n\n".join(["prose", block, block, uncovered_text_marker(1)])
    have = "prose\n\n" + block
    carried = _coverage_parts_to_carry(decorated, have)
    assert sum(1 for part in carried if "Same words" in part) == 1, (
        "the second identical reading was dropped"
    )


def _right_column_chart_with_a_left_blob_pdf(path: Path) -> Path:
    """A right-column chart, and an UNRELATED drawing in the left column."""
    doc = fitz.open()
    page = doc.new_page()
    page.draw_rect(fitz.Rect(40, 160, 120, 260), color=(0, 0, 0), fill=(0.9, 0.1, 0.1))
    page.draw_rect(fitz.Rect(330, 140, 560, 340), color=(0.4, 0.4, 0.4), width=0.8)
    for i, h in enumerate((80.0, 140.0, 60.0)):
        x = 350.0 + i * 60
        page.draw_rect(fitz.Rect(x, 320 - h, x + 40, 320), color=(0, 0, 0), fill=(0.2, 0.4, 0.8))
    _glyph_row(page, 160.0, x0=350.0)
    page.insert_text((60, 100), "Prose with a text layer.")
    doc.save(path)
    doc.close()
    return path


def test_a_figures_extent_is_the_marks_inside_its_candidate_rect(tmp_path: Path) -> None:
    """Round 8, item 1: the extent is the round-2 union, and over-inclusion is a LIMIT.

    Round 7 narrowed this, growing the extent only from marks that touch the
    page's uncovered ink, so an unrelated drawing elsewhere in the expanded
    candidate rect would be excluded. Measured over the 133 corpus pages where
    `chart_region_bboxes` fires, that CUT REAL FIGURES: 9 extents differed, 7
    shrank, and Jorda AER p11 lost three right panels mid-plot. A figure cut in
    half is worse than an over-wide crop, so it was reverted.

    What this pins is the behaviour that survived: the extent is every mark
    inside the candidate rect. The known limit — a synthetic two-column page
    whose unrelated left-column drawing is swallowed — is asserted here as a
    LIMIT, so that a future narrowing has to come back and change this test
    deliberately rather than by accident.
    """
    pdf = _right_column_chart_with_a_left_blob_pdf(tmp_path / "blob.pdf")
    with fitz.open(pdf) as doc:
        figures = [r for r in analyze_page(doc[0]).regions if r.figure_pdf]
    assert figures, "the chart is a figure"
    x0 = figures[0].figure_pdf[0]
    assert x0 < 300, (
        "KNOWN LIMIT: the extent includes an unrelated left-column drawing. "
        "Narrowing it cut real figures (round 8); see the decision log."
    )


_JORDA_PAGE = _SCRATCH / "corpus/pdfs/2005__jorda__AER.pdf"


@pytest.mark.skipif(
    not _JORDA_PAGE.is_file(), reason="third-party corpus PDFs live in gitignored scratch"
)
def test_a_real_multi_panel_figure_keeps_all_of_its_panels() -> None:
    """Round 8, item 1: the page the narrowing broke.

    Jorda AER p11 is a 2x3 IRF grid. The round-7 extent put x1 at 367.2 and cut
    the three right panels mid-plot; the round-2 union puts it at 402.7.
    """
    with fitz.open(_JORDA_PAGE) as doc:
        figures = [r for r in analyze_page(doc[10]).regions if r.figure_pdf]
    assert figures, "p11 carries a figure"
    x1 = figures[0].figure_pdf[2]
    assert x1 == pytest.approx(402.7, abs=0.5), f"the right panels were cut (x1={x1})"


def test_an_earlier_prose_region_does_not_take_the_next_ones_words() -> None:
    """Round 8, item 2: the PROSE direction branch was never pinned.

    A prose region's block precedes its own picture, so in
    `[pic1][block2][pic2]` the paragraph after pic1 belongs to region 2.
    Searching both directions hands it to region 1, and `claimed` then starves
    region 2 of its own words — one reading lost, one misattributed.
    """
    from socr.core.text_coverage import uncovered_text_image_ref
    from socr.core.text_recovery import recovered_block
    from socr.pipeline.orchestrator import _prior_recoveries

    class _Region:
        figure_pdf = None
        kind = "vector_or_bare"
        merged_from = 1

    block_two = recovered_block("SECOND REGION WORDS", in_figure=False)
    body = "\n\n".join(
        [
            "native prose",
            uncovered_text_image_ref(1, 1),  # region 1: UNREAD, picture only
            block_two,  # region 2's words...
            uncovered_text_image_ref(1, 2),  # ...then its picture
        ]
    )
    found = _prior_recoveries(body, 1, [_Region(), _Region()])
    assert found[1].piece is None, "region 1 took region 2's words"
    assert found[2].piece is not None and "SECOND REGION WORDS" in found[2].piece, (
        "region 2 was starved of its own words"
    )


# --- fix round 9: a block is claimed by its own MARK ---------------------------


class _FakeRegion:
    """A region of a given kind, for `_prior_recoveries` alone."""

    merged_from = 1

    def __init__(self, in_figure: bool) -> None:
        self.figure_pdf = (0.0, 0.0, 1.0, 1.0) if in_figure else None
        self.kind = "raster" if in_figure else "vector_or_bare"


def _pair_twice(body: str, regions: list) -> list[dict]:
    """`_prior_recoveries` applied twice, as a second apply would."""
    from socr.pipeline.orchestrator import _prior_recoveries

    out = []
    for _ in range(2):
        found = _prior_recoveries(body, 1, regions)
        out.append({i: (rec.piece or "") for i, rec in found.items()})
    return out


def test_a_figures_block_and_a_prose_block_are_told_apart_by_their_marks() -> None:
    """Round 9: ask the BLOCK which kind it is, not the next region.

    Round 7 asked whether the NEXT REGION was prose — the wrong question twice
    over: the paragraph after a picture belongs to whoever wrote it, not to
    whoever comes next in the region list, and the two orders can disagree.
    Each case is applied twice, because a second apply is where a
    misattribution shows up.
    """
    from socr.core.text_coverage import uncovered_text_image_ref
    from socr.core.text_recovery import recovered_block

    fig_block = recovered_block("FIGURE WORDS", in_figure=True)
    prose_block = recovered_block("PROSE WORDS", in_figure=False)
    pic1, pic2 = uncovered_text_image_ref(1, 1), uncovered_text_image_ref(1, 2)

    # (a) a READ figure then an UNREAD prose region.
    body_a = "\n\n".join(["native prose", pic1, fig_block, pic2])
    for found in _pair_twice(body_a, [_FakeRegion(True), _FakeRegion(False)]):
        assert "FIGURE WORDS" in found[1], "the figure lost its own words"
        assert found[2] == "", "the unread prose region was handed the figure's words"

    # (b) an UNREAD figure then a READ prose region.
    body_b = "\n\n".join(["native prose", pic1, prose_block, pic2])
    for found in _pair_twice(body_b, [_FakeRegion(True), _FakeRegion(False)]):
        assert found[1] == "", "the unread figure claimed the prose region's words"
        assert "PROSE WORDS" in found[2], "the prose region was starved of its own words"


def test_three_regions_of_mixed_kinds_keep_their_own_words() -> None:
    """Round 9 (c): an unread figure, a read figure, and a read prose region.

    The prose region's anchor was refused, so its block sits at the tail with
    its picture — the arrangement where a kind-based rule goes wrong.
    """
    from socr.core.text_coverage import uncovered_text_image_ref
    from socr.core.text_recovery import recovered_block

    fig_block = recovered_block("SECOND FIGURE WORDS", in_figure=True)
    prose_block = recovered_block("THIRD REGION WORDS", in_figure=False)
    pics = [uncovered_text_image_ref(1, i) for i in (1, 2, 3)]
    body = "\n\n".join(["native prose", pics[0], pics[1], fig_block, prose_block, pics[2]])

    regions = [_FakeRegion(True), _FakeRegion(True), _FakeRegion(False)]
    for found in _pair_twice(body, regions):
        assert found[1] == "", "the unread figure took someone else's words"
        assert "SECOND FIGURE WORDS" in found[2]
        assert "THIRD REGION WORDS" not in found[2], "figure 2 took the prose region's words"
        assert "THIRD REGION WORDS" in found[3]
        assert "SECOND FIGURE WORDS" not in found[3], "the prose region took figure 2's words"
    # And no block is handed to two regions at once.
    first = _pair_twice(body, regions)[0]
    for words in ("SECOND FIGURE WORDS", "THIRD REGION WORDS"):
        assert sum(1 for piece in first.values() if words in piece) == 1, words
