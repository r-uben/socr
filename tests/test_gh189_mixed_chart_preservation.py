"""GH-189: a chart on a mixed chart+table page must survive an ACCEPTED rung.

A page carrying both chart marks and a table signal is held out of the
page-level chart lane (GH-150 B1) and keeps its normal route, where the chart is
an inline placeholder in ``ps.native_text``.  That placeholder was only resolved
on the ``not decision.accepted`` branch, and only into ``native_text`` -- so when
the judge ACCEPTED a rung the accepted candidate shipped and the chart left the
document with no trace at page status, document status, metadata or CLI.

Every test here drives the REAL detector, the REAL crop renderer and the REAL
assembly on a synthetic PDF.  ``route_page`` is stubbed to return a real
``PageOutput`` carrying the correct table and the page's own prose and NO image
reference -- which is exactly the shape of an accepted VLM extraction, and the
input on which main drops the chart.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

fitz = pytest.importorskip("fitz")

# The winning candidate: the correct table plus the page's own prose lines, and
# not one image reference. Byte-for-byte what an accepted rung looks like.
WINNER_TEXT = """Figure preamble sentence unique alpha

Table caption unique bravo

| Year | Value | Share |
| --- | --- | --- |
| 2020 | 1.50 | 10.00 |
| 2021 | 2.50 | 20.00 |
| 2022 | 3.50 | 30.00 |

Closing note unique charlie

Final line unique delta"""

TABLE_ROW = "| 2020 | 1.50 | 10.00 |"

_RED, _BLUE, _GREEN = (0.9, 0.1, 0.1), (0.1, 0.1, 0.9), (0.1, 0.8, 0.1)


def _draw_bars(page, y_top: float, y_bottom: float) -> None:
    """A qualifying vector chart cluster: coloured filled bars, no raster."""
    for i, (col, x) in enumerate(
        zip([_RED, _BLUE, _GREEN, _RED, _BLUE], [100, 180, 260, 340, 420])
    ):
        page.draw_rect(fitz.Rect(x, y_top + i * 20, x + 60, y_bottom), color=col, fill=col, width=1)


def _write_page_text(page) -> None:
    """Prose + a numeric grid, laid out so each prose line is a unique anchor."""
    page.insert_text((72, 30), "Figure preamble sentence unique alpha", fontsize=11)
    page.insert_text((72, 250), "Table caption unique bravo", fontsize=11)
    for j, head in enumerate(("Year", "Value", "Share")):
        page.insert_text((90 + j * 120, 285), head, fontsize=10)
    for i, row in enumerate(
        (("2020", "1.50", "10.00"), ("2021", "2.50", "20.00"), ("2022", "3.50", "30.00"))
    ):
        for j, cell in enumerate(row):
            page.insert_text((90 + j * 120, 310 + i * 22), cell, fontsize=10)
    page.insert_text((72, 420), "Closing note unique charlie", fontsize=11)
    page.insert_text((72, 700), "Final line unique delta", fontsize=11)


def _make_mixed_chart_table_pdf(tmp_path: Path) -> Path:
    """Chart above an identifiable table, a second chart below it, unique anchors."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    _write_page_text(page)
    _draw_bars(page, 90, 200)
    _draw_bars(page, 490, 600)
    page.draw_line(fitz.Point(100, 640), fitz.Point(480, 640), color=(0.8, 0.2, 0.0), width=3)
    out = tmp_path / "mixed_chart_table.pdf"
    doc.save(str(out))
    doc.close()
    return out


def _make_chart_free_pdf(tmp_path: Path) -> Path:
    """The SAME page with every vector mark removed -- the byte-identity control."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    _write_page_text(page)
    out = tmp_path / "chart_free.pdf"
    doc.save(str(out))
    doc.close()
    return out


def _make_pipeline(**overrides):
    from socr.core.config import EngineType, PipelineConfig
    from socr.pipeline.orchestrator import UnifiedPipeline

    cfg = PipelineConfig(
        primary_engine=EngineType.DEEPSEEK,
        enabled_engines=list(EngineType),
        agentic=True,
        quiet=True,
        native_first=True,
        save_figures=overrides.pop("save_figures", False),
        describe_figures=False,
        table_judge_ladder=False,
        **overrides,
    )
    return UnifiedPipeline(cfg)


def _make_state(pdf_path: Path, native_text: str):
    from socr.core.born_digital import DocumentAssessment, PageAssessment
    from socr.core.document import DocumentHandle
    from socr.core.state import DocumentState, PageState

    with patch.object(DocumentHandle, "__post_init__", lambda self: None):
        handle = DocumentHandle(path=pdf_path, page_count=1)
    state = DocumentState(handle=handle)
    state.pages[1] = PageState(
        page_num=1,
        is_born_digital=True,
        native_text=native_text,
        needs_ocr_enhancement=False,
        has_tables=True,
    )
    state._last_assessment = DocumentAssessment(
        path=pdf_path,
        pages=[
            PageAssessment(
                page_num=1,
                is_born_digital=True,
                native_text=native_text,
                confidence=0.9,
                needs_ocr_enhancement=False,
                has_tables=True,
            )
        ],
    )
    return state


def _accepted_decision(text: str):
    from socr.core.result import PageOutput, PageStatus

    decision = MagicMock()
    decision.accepted = True
    decision.attempts = []
    decision.final_output = PageOutput(
        page_num=1,
        text=text,
        status=PageStatus.SUCCESS,
        engine="deepseek",
        audit_passed=True,
    )
    decision.winning_engine = "deepseek"
    decision.total_cost_usd = 0.002
    return decision


def _run(pdf: Path, out_dir: Path, *, winner: str = WINNER_TEXT, save_figures: bool = False):
    """Route (accepted) then assemble, with no provider and no judge reachable."""
    from socr.core.providers import PROFILE_QWEN_LOCAL
    from socr.pipeline.orchestrator import UnifiedPipeline

    pipeline = _make_pipeline(save_figures=save_figures)
    state = _make_state(pdf, "Figure preamble sentence unique alpha")
    pipeline._last_assessment = state._last_assessment
    with (
        patch("socr.pipeline.orchestrator.route_page", return_value=_accepted_decision(winner)),
        patch.object(
            UnifiedPipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]
        ),
        patch.object(UnifiedPipeline, "_resolve_judge_model", return_value=""),
    ):
        pipeline._phase_agentic(state, out_dir)
        result = pipeline._phase_assemble(state, out_dir)
    return pipeline, state, result


def _body(result) -> str:
    return result.pages[0].text or ""


def _crop(out_dir: Path, index: int) -> Path:
    hits = list(out_dir.rglob(f"chart_region_p1_{index}.png"))
    return hits[0] if hits else out_dir / "__missing__"


# ---------------------------------------------------------------------------
# Main integration fixture
# ---------------------------------------------------------------------------


def test_accepted_rung_keeps_both_charts_in_source_order(tmp_path: Path) -> None:
    """The falsification test for #189: on main the accepted table ships alone."""
    pdf = _make_mixed_chart_table_pdf(tmp_path)
    out_dir = tmp_path / "out"
    pipeline, state, result = _run(pdf, out_dir)

    # The winner really was the routed model output with no image ref -- so the
    # assertions below inspect content the accepted candidate did NOT carry.
    ps = state.pages[1]
    assert ps.best_output is not None and ps.best_output.engine == "deepseek"
    assert "![" not in (ps.best_output.text or "")

    body = _body(result)
    assert TABLE_ROW in body, f"the routed table did not survive: {body!r}"

    refs = [f"figures/chart_region_p1_{i}.png" for i in (1, 2)]
    for ref in refs:
        assert body.count(ref) == 1, f"expected exactly one reference to {ref}:\n{body}"

    # Each crop is a readable, non-empty PNG on disk.
    for i in (1, 2):
        path = _crop(out_dir, i)
        assert path.exists() and path.stat().st_size > 0, f"crop {i} missing or empty"
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", f"crop {i} is not a PNG"
        fitz.Pixmap(str(path))  # raises if unreadable

    # Source order: alpha, chart 1, bravo, table, charlie, chart 2, delta.
    order = [
        body.index("Figure preamble sentence unique alpha"),
        body.index(refs[0]),
        body.index("Table caption unique bravo"),
        body.index(TABLE_ROW),
        body.index("Closing note unique charlie"),
        body.index(refs[1]),
        body.index("Final line unique delta"),
    ]
    assert order == sorted(order), f"chart refs are not in source order: {order}\n{body}"

    # Nothing was demoted: both regions were preserved AND placed.
    assert not ps.chart_region_render_failed
    assert not ps.chart_region_placement_unresolved
    assert pipeline._chart_region_note(state) is None
    assert "chart region" not in (result.error or "")


def test_the_fragment_and_the_sidecar_agree_with_the_reconciled_body(tmp_path: Path) -> None:
    """The reconciliation lands before the authoritative writers, not after them."""
    import json

    pdf = _make_mixed_chart_table_pdf(tmp_path)
    out_dir = tmp_path / "out"
    _pipeline, _state, result = _run(pdf, out_dir)
    body = _body(result)

    fragments = list(out_dir.rglob("pages/00001.md"))
    assert fragments, "no page fragment was written"
    fragment = fragments[0].read_text()
    assert fragment.strip(), "the fragment is empty"
    assert fragment.strip() in body, "the fragment disagrees with the assembled body"
    assert "figures/chart_region_p1_1.png" in fragment
    assert "figures/chart_region_p1_2.png" in fragment

    sidecars = list(out_dir.rglob("pages/00001.json"))
    assert sidecars, "no page sidecar was written"
    refs = json.loads(sidecars[0].read_text()).get("figure_refs") or []
    paths = {r.get("image_path") for r in refs}
    assert {"figures/chart_region_p1_1.png", "figures/chart_region_p1_2.png"} <= paths, (
        f"the sidecar does not carry the chart crops: {paths}"
    )


def test_reconciliation_is_idempotent(tmp_path: Path) -> None:
    """Running the pass on its own output changes nothing."""
    from socr.figures.chart_regions import (
        chart_region_anchors,
        reconcile_chart_region_refs,
        table_bindings,
    )
    from socr.pipeline.orchestrator import UnifiedPipeline
    from socr.tables.reconstruct import chart_region_bboxes

    pdf = _make_mixed_chart_table_pdf(tmp_path)
    figures_dir = tmp_path / "figures"
    pipeline = _make_pipeline()
    with fitz.open(str(pdf)) as doc:
        bboxes = chart_region_bboxes(doc[0])
        anchors = chart_region_anchors(doc[0], bboxes)
    assert len(bboxes) == 2
    assets = UnifiedPipeline._render_chart_region_crops(pipeline, pdf, 1, bboxes, figures_dir)
    bindings = table_bindings(bboxes, [])

    once, first = reconcile_chart_region_refs(WINNER_TEXT, assets, anchors, bindings)
    twice, second = reconcile_chart_region_refs(once, assets, anchors, bindings)
    assert twice == once, "reconciliation is not idempotent"
    assert all(o.placed for o in first)
    assert all(o.disposition == "already_present" for o in second)


# ---------------------------------------------------------------------------
# Chart-free byte-identity pair
# ---------------------------------------------------------------------------


def test_a_chart_free_document_is_byte_identical_with_and_without_the_pass(
    tmp_path: Path,
) -> None:
    """The invariant this fix must not buy at the cost of every other document.

    Pins a DIFFERENCE, not a value: the same document assembled twice in the same
    process, changing only whether the GH-189 pass executes.
    """
    from socr.pipeline.orchestrator import UnifiedPipeline

    pdf = _make_chart_free_pdf(tmp_path)

    _p1, _s1, with_pass = _run(pdf, tmp_path / "a")
    with patch.object(
        UnifiedPipeline,
        "_preserve_chart_regions",
        side_effect=lambda _state, page_texts, _out: page_texts,
    ):
        _p2, _s2, without_pass = _run(pdf, tmp_path / "b")

    assert _body(with_pass) == _body(without_pass), (
        "the chart-region pass changed a chart-free document's markdown"
    )
    assert with_pass.status == without_pass.status
    assert (with_pass.error or "") == (without_pass.error or "")
    assert not with_pass.figures, f"a chart-free document gained figures: {with_pass.figures}"
    # Positive control: the fixture really does reach the pass (it has a table
    # signal), so the equality above is not vacuous.
    from socr.tables.reconstruct import chart_region_bboxes

    with fitz.open(str(pdf)) as doc:
        assert chart_region_bboxes(doc[0]) == []


def test_save_figures_does_not_duplicate_the_chart_region(tmp_path: Path) -> None:
    """Ordinary figure extraction must not emit a second asset for the same region."""
    pdf = _make_mixed_chart_table_pdf(tmp_path)
    out_dir = tmp_path / "out"
    _pipeline, _state, result = _run(pdf, out_dir, save_figures=True)
    body = _body(result)
    for i in (1, 2):
        assert body.count(f"figures/chart_region_p1_{i}.png") == 1, (
            f"chart region {i} is referenced more than once with --save-figures:\n{body}"
        )
    assert body.count("![") == 2, f"a second asset was emitted for the same region:\n{body}"


# ---------------------------------------------------------------------------
# Render-failure repeat
# ---------------------------------------------------------------------------


def test_one_failed_crop_is_visible_and_the_other_still_ships(tmp_path: Path) -> None:
    """Two crops, one failing at the save boundary: partial loss cannot hide."""
    from socr.core.result import DocumentStatus

    pdf = _make_mixed_chart_table_pdf(tmp_path)
    out_dir = tmp_path / "out"
    real_save = fitz.Pixmap.save

    def _save(self, target, *args, **kwargs):
        if str(target).endswith("chart_region_p1_2.png"):
            raise RuntimeError("pixmap save died")
        return real_save(self, target, *args, **kwargs)

    with patch.object(fitz.Pixmap, "save", _save):
        pipeline, state, result = _run(pdf, out_dir)

    body = _body(result)
    assert TABLE_ROW in body, "the correct table did not survive the render failure"
    assert body.count("figures/chart_region_p1_1.png") == 1, "the surviving crop was lost too"
    assert _crop(out_dir, 1).stat().st_size > 0

    # No broken markdown image link for the crop that was never written.
    assert "](figures/chart_region_p1_2.png)" not in body
    assert "chart_region_p1_2.png" in body, f"the failed region left no visible marker:\n{body}"
    assert "NOT preserved" in body

    ps = state.pages[1]
    assert ps.chart_region_render_failed is True
    assert ps.chart_region_placement_unresolved is False

    # Durable page-level event.
    events = [
        e
        for e in state.events
        if getattr(e, "kind", "") == "chart_region_not_preserved" and e.page_num == 1
    ]
    assert len(events) == 1
    assert events[0].data["region_index"] == 2
    assert events[0].data["region_count"] == 2

    # Document status, metadata note and CLI line all disclose it.
    note = pipeline._chart_region_note(state)
    assert note is not None and "page(s) 1" in note and "preserved nowhere" in note
    assert note in (result.error or ""), f"the note never reached metadata: {result.error!r}"
    assert result.status is not DocumentStatus.SUCCESS
    assert result.audit_passed is False


# ---------------------------------------------------------------------------
# Placement-failure repeat
# ---------------------------------------------------------------------------


def test_unplaceable_crops_are_preserved_but_never_claimed_in_source_order(
    tmp_path: Path,
) -> None:
    """No unique anchor and no 1:1 table binding: preserved, and labelled as such."""
    # Every usable anchor removed; the table alone won the page.
    winner = "| Year | Value | Share |\n| --- | --- | --- |\n" + TABLE_ROW
    pdf = _make_mixed_chart_table_pdf(tmp_path)
    out_dir = tmp_path / "out"
    pipeline, state, result = _run(pdf, out_dir, winner=winner)

    body = _body(result)
    assert TABLE_ROW in body
    for i in (1, 2):
        assert body.count(f"figures/chart_region_p1_{i}.png") == 1, (
            f"chart region {i} was not preserved:\n{body}"
        )
    assert "Unresolved chart placement" in body, f"the placement was claimed silently:\n{body}"
    # Ordered relative to each other, and both after the table -- the block makes
    # no claim about position, and the assertion must not imply one either.
    assert body.index("figures/chart_region_p1_1.png") < body.index("figures/chart_region_p1_2.png")
    assert body.index(TABLE_ROW) < body.index("Unresolved chart placement")

    ps = state.pages[1]
    assert ps.chart_region_placement_unresolved is True
    assert ps.chart_region_render_failed is False
    # audit_passed is the winner-SELECTION flag: flipping it would discard the
    # page's correct table (the #252 defect). It must be untouched.
    assert ps.best_output.audit_passed is True

    note = pipeline._chart_region_note(state)
    assert note is not None and "position in the page could not be established" in note
    assert note in (result.error or ""), f"the note never reached metadata: {result.error!r}"
    # Nothing was LOST, so the document is not demoted for it -- the difference
    # against the placeable run is the note and the labelled block, not a status.
    assert "preserved nowhere" not in note


def test_a_single_source_table_binds_a_chart_with_no_usable_anchor(tmp_path: Path) -> None:
    """The second binding: geometry against the page's only table, never ordinal pairing."""
    from socr.figures.chart_regions import (
        PLACED_TABLE_BOUND,
        ChartRegionAsset,
        reconcile_chart_region_refs,
        table_bindings,
    )

    class _Box:
        def __init__(self, y0, y1):
            self.x0, self.y0, self.x1, self.y1 = 0.0, y0, 500.0, y1

    above, below = _Box(50, 200), _Box(450, 600)
    bindings = table_bindings([above, below], [(70.0, 250.0, 500.0, 400.0)])
    assert bindings == {1: "before", 2: "after"}
    # Two source tables: correspondence is not established, so no binding at all.
    assert table_bindings([above], [(70.0, 250.0, 500.0, 400.0), (70.0, 500.0, 500.0, 600.0)]) == {}

    assets = [
        ChartRegionAsset(1, 1, (0.0, 50.0, 500.0, 200.0), "figures/a.png", True),
        ChartRegionAsset(1, 2, (0.0, 450.0, 500.0, 600.0), "figures/b.png", True),
    ]
    text = "| Year | Value |\n| --- | --- |\n| 2020 | 1.50 |"
    out, outcomes = reconcile_chart_region_refs(text, assets, {}, bindings)
    assert [o.disposition for o in outcomes] == [PLACED_TABLE_BOUND, PLACED_TABLE_BOUND]
    assert out.index("figures/a.png") < out.index("| 2020 | 1.50 |") < out.index("figures/b.png")


def test_an_empty_inventory_returns_the_text_unchanged(tmp_path: Path) -> None:
    from socr.figures.chart_regions import reconcile_chart_region_refs

    out, outcomes = reconcile_chart_region_refs(WINNER_TEXT, [], {}, {})
    assert out == WINNER_TEXT and outcomes == []


def test_a_deleted_crop_is_repaired_on_the_next_run(tmp_path: Path) -> None:
    """The inventory is recomputed from the PDF, so a damaged asset is re-rendered.

    This covers the ASSET half of the resume case and nothing more: the pass
    lives in ``_phase_assemble``, so any run that reaches assembly rebuilds the
    inventory and re-renders. A document-level resume that skips assembly
    entirely is out of scope here (see #170).
    """
    pdf = _make_mixed_chart_table_pdf(tmp_path)
    out_dir = tmp_path / "out"
    _p1, _s1, first = _run(pdf, out_dir)
    crop = _crop(out_dir, 2)
    assert crop.exists()
    crop.unlink()

    _p2, _s2, second = _run(pdf, out_dir)
    assert crop.exists() and crop.stat().st_size > 0, "the deleted crop was not repaired"
    assert _body(second) == _body(first), "the repaired run produced a different body"
