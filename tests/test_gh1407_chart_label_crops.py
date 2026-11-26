"""#1407: a chart's own labels are not cropped again on top of the chart's picture.

On a chart page whose chart ships as a whole-page picture (``engine: chart_asset``,
``figure_text=UNKNOWN``), the uncovered-text step used to emit one crop and one quoted
line for every label drawn as shapes inside the chart. A region wholly inside a detected
chart region of such a page is now left out, counted in an audit event. Everything else
keeps its crop and its reading.

Every pipeline assertion here is a DIFFERENCE between two runs in one process that
differ only in whether the new rule executes (CLAUDE.md, the no-provider trap): nothing
pins a status or a count measured on one machine.
"""

from __future__ import annotations

import json
import re
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import fitz

from socr.core.config import EngineType, PipelineConfig
from socr.core.providers import PROFILE_QWEN_LOCAL
from socr.core.text_coverage import (
    CoverageRegion,
    analyze_page,
    regions_inside_boxes,
    uncovered_text_image_ref,
)
from socr.pipeline.orchestrator import UnifiedPipeline
from socr.tables.reconstruct import chart_region_bboxes

READING = "Percent"
_CROP_RE = re.compile(r"!\[Uncovered text page 1 region (\d+)\]")
_PROSE = [f"Prose above the chart, line {i}, stated before the drawn frame." for i in range(4)]


def test_loaded_source_is_this_checkout() -> None:
    import socr

    assert Path(socr.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1] / "src")


# --- the containment rule ------------------------------------------------------


def _region(box, figure=None) -> CoverageRegion:
    return CoverageRegion(
        pdf=box, px=(0, 0, 1, 1), glyphs=5, band="body", kind="vector_or_bare", figure_pdf=figure
    )


def test_only_a_region_wholly_inside_one_box_counts() -> None:
    boxes = [(100.0, 100.0, 300.0, 300.0), (320.0, 100.0, 500.0, 300.0)]
    regions = [
        _region((110.0, 110.0, 150.0, 120.0)),  # 1: inside the first box
        _region((290.0, 110.0, 330.0, 120.0)),  # 2: straddles both, inside neither
        _region((100.0, 100.0, 300.0, 300.0)),  # 3: exactly the box
        _region((99.5, 110.0, 150.0, 120.0)),  # 4: half a point outside: kept
        _region((10.0, 400.0, 50.0, 410.0)),  # 5: nowhere near
        _region((330.0, 110.0, 340.0, 120.0)),  # 6: inside the second box
    ]
    assert regions_inside_boxes(regions, boxes) == [1, 3, 6]
    assert regions_inside_boxes(regions, []) == []


def test_the_box_tested_is_the_crop_not_the_ink() -> None:
    """A merged figure region ships its whole FIGURE; that is the box that must fit."""
    box = [(100.0, 100.0, 300.0, 300.0)]
    ink = (110.0, 110.0, 150.0, 120.0)
    assert regions_inside_boxes([_region(ink, figure=(90.0, 90.0, 310.0, 310.0))], box) == []
    assert regions_inside_boxes([_region(ink, figure=(105.0, 105.0, 295.0, 295.0))], box) == [1]


# --- the pipeline difference ---------------------------------------------------


def _glyph_row(page, y: float, x0: float = 72.0, n: int = 8) -> None:
    """Path-only 'letters': text-like ink with no text layer."""
    x = x0
    widths = (3.0, 4.0, 5.0, 3.5, 4.5, 5.5, 3.2, 4.8)
    for i in range(n):
        w = widths[i % len(widths)]
        page.draw_rect(fitz.Rect(x, y, x + w, y + 6.0), color=(0, 0, 0), width=0.6)
        x += w + 2.0


def _chart_page_pdf(path: Path) -> Path:
    """One bar chart with drawn labels inside it, and drawn text far below it."""
    doc = fitz.open()
    page = doc.new_page()
    y = 60.0
    for line in _PROSE:
        page.insert_text((72, y), line)
        y += 13.0
    page.draw_rect(fitz.Rect(64, 120, 464, 420), color=(0.4, 0.4, 0.4), width=0.8)
    page.draw_rect(fitz.Rect(80, 300, 448, 300.4), color=(0, 0, 0), fill=(0, 0, 0))
    for i, h in enumerate((120.0, 80.0, 160.0, 60.0)):
        x = 110.0 + i * 70
        page.draw_rect(fitz.Rect(x, 380 - h, x + 40, 380), color=(0, 0, 0), fill=(0.2, 0.4, 0.8))
    _glyph_row(page, 150.0)  # a label inside the chart
    _glyph_row(page, 400.0)  # another
    _glyph_row(page, 640.0)  # drawn text nowhere near the chart
    doc.save(path)
    doc.close()
    return path


def _geometry(pdf: Path) -> tuple[list[int], int]:
    """``(regions inside a chart region, total regions)`` for the fixture, from the code."""
    with fitz.open(pdf) as doc:
        page = doc[0]
        regions = analyze_page(page).regions
        inside = regions_inside_boxes(regions, [tuple(b) for b in chart_region_bboxes(page)])
    return inside, len(regions)


def _run(
    pdf: Path, out: Path, *, rule: bool, render_fails: bool = False, chart_lane: bool = True
) -> dict:
    pipeline = UnifiedPipeline(
        PipelineConfig(
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
    )
    with ExitStack() as stack:
        enter = stack.enter_context
        enter(
            patch.object(
                pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]
            )
        )
        enter(patch.object(pipeline, "_resolve_judge_model", return_value=""))
        enter(patch.object(pipeline, "_uncovered_text_reader_available", return_value=True))
        enter(patch.object(pipeline, "_uncovered_text_ask", return_value=lambda _p, _q: READING))
        enter(patch.object(pipeline, "_is_chart_asset_page", return_value=chart_lane))
        if not rule:
            enter(
                patch.object(
                    pipeline, "_regions_in_shipped_chart_picture", return_value=(set(), "")
                )
            )
        if render_fails:
            enter(
                patch.object(
                    pipeline, "_render_chart_page_png", side_effect=RuntimeError("no render")
                )
            )
        pipeline.process(pdf, out)
    text = next(out.rglob("pages/00001.md")).read_text(encoding="utf-8")
    sidecar = json.loads(next(out.rglob("pages/00001.json")).read_text(encoding="utf-8"))
    audit = json.loads(next(out.rglob("audit_log.json")).read_text(encoding="utf-8"))
    events = audit if isinstance(audit, list) else audit.get("events", [])
    return {
        "text": text,
        "crops": [int(n) for n in _CROP_RE.findall(text)],
        "pngs": sorted(p.name for p in out.rglob("figures/uncovered_text_*.png")),
        "status": (sidecar["status"], sidecar["failure_mode"], sidecar["engine"]),
        "clauses": [(c["check"], c["result"]) for c in sidecar["verdict"]["clauses"]],
        "events": [e for e in events if e.get("page_num") == 1],
    }


def _kinds(run: dict) -> list[str]:
    return [e["kind"] for e in run["events"]]


def test_a_charts_own_labels_are_not_cropped_on_top_of_its_picture(tmp_path: Path) -> None:
    pdf = _chart_page_pdf(tmp_path / "chart.pdf")
    inside, total = _geometry(pdf)
    assert inside and len(inside) < total, "the fixture needs a region on each side of the rule"
    outside = [i for i in range(1, total + 1) if i not in inside]

    without = _run(pdf, tmp_path / "without", rule=False)
    with_rule = _run(pdf, tmp_path / "with", rule=True)

    assert "![Chart page 1](" in with_rule["text"], "the chart's picture ships"
    # The whole difference: the regions inside the chart are gone, and only those.
    assert sorted(without["crops"]) == list(range(1, total + 1))
    assert sorted(with_rule["crops"]) == outside
    assert with_rule["pngs"] == sorted(f"uncovered_text_p1_r{i}.png" for i in outside)
    assert without["text"].count(READING) - with_rule["text"].count(READING) == len(inside)
    # A surviving region keeps its number, its picture and its reading.
    for i in outside:
        assert uncovered_text_image_ref(1, i) in with_rule["text"]
    # Nothing else about the page moves: status, failure mode, lane, both marks.
    assert with_rule["status"] == without["status"]
    assert with_rule["clauses"] == without["clauses"]

    # Recorded, not just dropped.
    assert "uncovered_text_in_chart_picture" not in _kinds(without)
    (event,) = [e for e in with_rule["events"] if e["kind"] == "uncovered_text_in_chart_picture"]
    assert event["data"]["suppressed"] == len(inside)
    assert event["data"]["regions"] == inside
    assert event["data"]["kept"] == len(outside)
    assert event["data"]["picture"] in with_rule["text"]
    (unread,) = [e for e in with_rule["events"] if e["kind"] == "text_ink_unread"]
    assert unread["data"]["n_regions"] == total
    assert unread["data"]["in_chart_picture"] == len(inside)
    # And none of them cost a model call.
    reads = [k for k in _kinds(with_rule) if k == "uncovered_text_recovered"]
    assert len(reads) == len(outside)


def test_no_picture_means_nothing_is_left_out(tmp_path: Path) -> None:
    """The rule rests on the picture. If it did not ship, every region keeps its crop."""
    pdf = _chart_page_pdf(tmp_path / "chart.pdf")
    without = _run(pdf, tmp_path / "without", rule=False, render_fails=True)
    with_rule = _run(pdf, tmp_path / "with", rule=True, render_fails=True)
    assert "![Chart page 1](" not in with_rule["text"]
    assert with_rule["text"] == without["text"]
    assert with_rule["pngs"] == without["pngs"]
    assert "uncovered_text_in_chart_picture" not in _kinds(with_rule)


def test_a_page_outside_the_chart_lane_is_untouched(tmp_path: Path) -> None:
    """Same PDF held in the native lane: no chart picture, so the rule never fires."""
    pdf = _chart_page_pdf(tmp_path / "chart.pdf")
    without = _run(pdf, tmp_path / "without", rule=False, chart_lane=False)
    with_rule = _run(pdf, tmp_path / "with", rule=True, chart_lane=False)
    assert with_rule["crops"], "the page still has uncovered regions"
    assert with_rule["text"] == without["text"]
    assert "uncovered_text_in_chart_picture" not in _kinds(with_rule)
