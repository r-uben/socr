"""#1276: a chart a page draws ships as a picture, or the page says it did not.

Two halves, one invariant (``socr.figures.figure_presence``):

* native winners: a chart BOX and a TICK RUN among the words inside that same box ship a crop of
  that region, whatever the page's status (WARNING and ``native+math`` pages included);
* model readings: every ``<!-- Image`` placeholder must be matched one-to-one by a shipped
  figure image, or the page is marked with a figure-specific clause.

Hermetic: no provider, no model. ``process()`` tests pin ``primary_engine`` / ``local_engine`` /
``enabled_engines``, patch ``_available_engines_for_agentic`` and ``_resolve_judge_model``.

CLAUDE.md, #257: nothing here pins an absolute (status, audit_passed) tuple measured on this
machine. Every behavioural test runs the SAME input twice in one process, changing exactly one
thing, and asserts the two outcomes differ exactly as intended. The ``process()`` tests run in
both provider states.
"""

from __future__ import annotations

import json
from pathlib import Path

import fitz
import pytest

from socr.core import manifest
from socr.core.marks import Check, leading_marks
from socr.core.result import FailureMode, PageOutput, PageStatus
from socr.core.state import PageState
from socr.figures import figure_presence as fp
from socr.figures.unshipped_figures import MIN_TICK_RUN
from socr.pipeline.orchestrator import UnifiedPipeline

_PROSE = "The estimated effect of the policy change on output is reported in the table above."
_YEARS = ("2018", "2019", "2020", "2021", "2022")


@pytest.fixture(autouse=True)
def _no_words_check(monkeypatch):
    """The words check (#1294) is its own lane; here it would mark any stub body."""
    from socr.core import verdict

    monkeypatch.setattr(verdict, "page_missing_words", lambda output, p: None)


def test_loaded_source_is_this_checkout() -> None:
    import socr

    assert Path(socr.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1] / "src")


# ---------------------------------------------------------------------------
# Fixtures: a real vector chart, built in-process
# ---------------------------------------------------------------------------


def _bars(page, rect: fitz.Rect, ticks: bool) -> None:
    """Coloured filled bars in an axes frame; with *ticks*, a y-axis tick column and a year
    row under it. ``ticks=False`` keeps the drawing and removes ONLY the numbers."""
    shape = page.new_shape()
    shape.draw_rect(rect)
    shape.finish(color=(0, 0, 0), width=1.5)
    for i in range(6):
        x = rect.x0 + 20 + i * 45
        shape.draw_rect(fitz.Rect(x, rect.y1 - (i + 1) * 25, x + 30, rect.y1 - 5))
        shape.finish(color=(0.8, 0.1, 0.1), fill=(0.8, 0.1, 0.1), width=1.5)
    shape.commit()
    if ticks:
        for k, value in enumerate((0, 10, 20, 30, 40)):
            page.insert_text(
                (rect.x0 - 30, rect.y1 - k * 45), str(value), fontname="helv", fontsize=9
            )
        page.insert_text(
            (rect.x0 + 20, rect.y1 + 14), " ".join(_YEARS), fontname="helv", fontsize=9
        )


def _chart_pdf(path: Path, *, ticks: bool, charts: int = 1) -> Path:
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "Selected indicators", fontname="helv", fontsize=12)
    top = 100
    for _ in range(charts):
        _bars(page, fitz.Rect(110, top, 430, top + 200), ticks)
        top += 300  # beyond the cluster gap: a separate region each
    y = 740
    for _ in range(2):
        page.insert_text((72, y), _PROSE, fontname="helv", fontsize=10)
        y += 14
    doc.save(path)
    doc.close()
    return path


def _regions(pdf: Path, body: str = "", table_bboxes=()):
    from socr.tables.reconstruct import chart_region_bboxes

    with fitz.open(pdf) as doc:
        page = doc[0]
        boxes = chart_region_bboxes(page)
        words = list(page.get_text("words"))
    return (
        words,
        boxes,
        fp.assess_regions(words, boxes, fp.own_region_refs(body, 1), table_bboxes, False),
    )


def test_the_fixture_differs_only_in_its_tick_numbers(tmp_path: Path) -> None:
    _, boxes_on, on = _regions(_chart_pdf(tmp_path / "on.pdf", ticks=True))
    _, boxes_off, off = _regions(_chart_pdf(tmp_path / "off.pdf", ticks=False))
    assert len(boxes_on) == len(boxes_off) == 1, "the chart box is found in both arms"
    assert on[0].ticks >= MIN_TICK_RUN > off[0].ticks
    assert on[0].fires and not off[0].fires


# ---------------------------------------------------------------------------
# The region tie
# ---------------------------------------------------------------------------


def test_evidence_must_lie_in_the_same_region_as_the_box(tmp_path: Path) -> None:
    """A tick run elsewhere on the page does not make a decorative box a chart.

    Same page, same box: only WHERE the numbers sit differs. Inside the box they fire; far
    below it (outside the box and its row margin) they do not.
    """
    pdf = tmp_path / "tie.pdf"
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    _bars(page, fitz.Rect(110, 100, 430, 300), ticks=False)
    for k, value in enumerate((0, 10, 20, 30, 40)):
        page.insert_text((80, 290 - k * 45), str(value), fontname="helv", fontsize=9)
    doc.save(pdf)
    doc.close()
    _, _, inside = _regions(pdf)

    far = tmp_path / "far.pdf"
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    _bars(page, fitz.Rect(110, 100, 430, 300), ticks=False)
    for k, value in enumerate((0, 10, 20, 30, 40)):
        page.insert_text((80, 740 - k * 14), str(value), fontname="helv", fontsize=9)
    doc.save(far)
    doc.close()
    _, _, outside = _regions(far)

    assert inside[0].fires, "fixture premise: the numbers inside the box are evidence"
    assert outside[0].ticks == 0 and not outside[0].fires, (
        "a tick run in ANOTHER region of the page is not evidence for this box"
    )


def test_a_delimiter_free_year_header_table_is_not_a_chart(tmp_path: Path) -> None:
    """``tick_run`` fires on a header row of consecutive years with no delimiters, and a
    decorative band behind it gives a box. The region is a TABLE (recurring numeric columns):
    no crop. The same years as an axis row (no table rows under them) still fire."""

    def table_pdf(path: Path, *, table_rows: bool) -> Path:
        doc = fitz.open()
        page = doc.new_page(width=612, height=792)
        shape = page.new_shape()
        shape.draw_rect(fitz.Rect(60, 90, 560, 330))
        shape.finish(color=(0.2, 0.3, 0.8), fill=(0.85, 0.88, 0.97), width=1.5)
        shape.commit()
        xs = (200, 270, 340, 410, 480)
        for x, year in zip(xs, _YEARS):
            page.insert_text((x, 110), year, fontname="helv", fontsize=9)
        if table_rows:
            for r, label in enumerate(("Output", "Prices", "Wages", "Credit", "Rates")):
                y = 135 + 22 * r
                page.insert_text((70, y), label, fontname="helv", fontsize=9)
                for c, x in enumerate(xs):
                    page.insert_text((x, y), f"{r + 1}.{c + 3}", fontname="helv", fontsize=9)
        doc.save(path)
        doc.close()
        return path

    _, boxes, table = _regions(table_pdf(tmp_path / "table.pdf", table_rows=True))
    _, _, axis_only = _regions(table_pdf(tmp_path / "axis.pdf", table_rows=False))
    assert boxes, "fixture premise: the band gives the production finder a box"
    assert table[0].table_region and not table[0].fires
    assert not axis_only[0].table_region and axis_only[0].fires, (
        "fixture premise: without the table rows the same year row IS evidence"
    )


def test_words_inside_a_detected_table_never_count_as_evidence(tmp_path: Path) -> None:
    words, boxes, with_table = _regions(
        _chart_pdf(tmp_path / "c.pdf", ticks=True), table_bboxes=[(0.0, 0.0, 612.0, 792.0)]
    )
    assert boxes and not with_table[0].fires


# ---------------------------------------------------------------------------
# The shipping boundary (the seam every writer reads one body from)
# ---------------------------------------------------------------------------


def _pipe() -> UnifiedPipeline:
    from socr.core.config import PipelineConfig

    pipe = UnifiedPipeline.__new__(UnifiedPipeline)
    pipe.config = PipelineConfig(quiet=True)
    pipe._assessment_for_page = lambda n: None
    return pipe


def _decide_native(pdf: Path, tmp_path: Path, *, engine: str, status: PageStatus, body: str):
    from socr.core.document import DocumentHandle
    from socr.core.state import DocumentState

    pipe = _pipe()
    state = DocumentState(handle=DocumentHandle(path=pdf, page_count=1))
    ps = PageState(page_num=1)
    ps.is_born_digital = True
    state.pages[1] = ps
    out = PageOutput(page_num=1, text=body, status=status, engine=engine, audit_passed=True)
    ps.best_output = out
    pipe._agentic_figure_presence_page(state, 1, ps, out, tmp_path / "figures")
    return state, ps, out


def _finalize(state, ps, out):
    ps.best_output = out
    return manifest.finalized_page_record(state, 1)


@pytest.mark.parametrize("status", [PageStatus.SUCCESS, PageStatus.WARNING])
@pytest.mark.parametrize("engine", ["native", "native+math"])
def test_a_chart_with_ticks_ships_a_crop_whatever_the_status_or_native_engine(
    tmp_path: Path, engine: str, status: PageStatus
) -> None:
    """The gate is the SHIPPED OUTCOME, not SUCCESS: native+math pages are always WARNING and
    a native page may already be WARNING from another lane."""
    on = _chart_pdf(tmp_path / "on.pdf", ticks=True)
    off = _chart_pdf(tmp_path / "off.pdf", ticks=False)
    body = f"Selected indicators\n\n{_PROSE}"

    st_on, ps_on, out_on = _decide_native(
        on, tmp_path / "a", engine=engine, status=status, body=body
    )
    st_off, ps_off, out_off = _decide_native(
        off, tmp_path / "b", engine=engine, status=status, body=body
    )

    assert ps_on.unshipped_figure_kind == fp.PRESENCE_SHIPPED_KIND
    assert ps_on.unshipped_figure_refs and not ps_off.unshipped_figure_refs
    assert ps_off.unshipped_figure_kind == fp.PRESENCE_PASS_KIND, "no evidence: a PASS record"
    assert (tmp_path / "a" / "figures" / Path(ps_on.unshipped_figure_refs[0]).name).is_file()
    kinds_on = [e.kind for e in st_on.events]
    kinds_off = [e.kind for e in st_off.events]
    assert fp.PRESENCE_SHIPPED_KIND in kinds_on and "visual_values_not_transcribed" in kinds_on
    assert kinds_off == [fp.PRESENCE_PASS_KIND], "the pass record is the ONLY thing it writes"

    rec_on = _finalize(st_on, ps_on, out_on)
    rec_off = _finalize(st_off, ps_off, out_off)
    ref = ps_on.unshipped_figure_refs[0]
    assert f"]({ref})" in rec_on.output.text and "![" not in rec_off.output.text
    assert body in rec_on.output.text, "the page's own text is untouched, the crop is added"
    # The visible #1174 line: a shipped crop says its values are not transcribed.
    assert leading_marks(rec_on.output.text, 1)
    assert any(c.check is Check.FIGURE_TEXT for c in rec_on.verdict.clauses)
    assert not any(c.check is Check.FIGURE_TEXT for c in rec_off.verdict.clauses), (
        "a PASS record never marks"
    )
    if status is PageStatus.SUCCESS:
        assert not leading_marks(rec_off.output.text, 1)
    assert rec_on.output.status is not PageStatus.SUCCESS
    # audit_passed is the winner-selection flag (#252): never touched.
    assert rec_on.output.audit_passed is True and out_on.audit_passed is True


def test_the_pass_record_says_what_it_does_not_claim(tmp_path: Path) -> None:
    st, ps, _ = _decide_native(
        _chart_pdf(tmp_path / "off.pdf", ticks=False),
        tmp_path,
        engine="native",
        status=PageStatus.SUCCESS,
        body=_PROSE,
    )
    (event,) = st.events
    assert "no chart evidence under this check" in event.detail
    assert "not a statement that the page carries no figure" in event.detail
    assert event.data["regions"][0]["fires"] is False


def test_another_regions_image_never_excuses_a_region(tmp_path: Path) -> None:
    """Two charts. The body links region 1's own crop: only region 2 ships a new crop. A
    different image (a logo, a table crop) excuses neither."""
    pdf = _chart_pdf(tmp_path / "two.pdf", ticks=True, charts=2)
    _, boxes, _ = _regions(pdf)
    assert len(boxes) == 2, "fixture premise"

    own = f"{_PROSE}\n\n![chart region 1](figures/chart_region_p1_1.png)"
    other = f"{_PROSE}\n\n![logo](figures/logo.png)\n\n![t](figures/failed_table_p1.png)"
    _, ps_own, _ = _decide_native(
        pdf, tmp_path / "a", engine="native", status=PageStatus.SUCCESS, body=own
    )
    _, ps_other, _ = _decide_native(
        pdf, tmp_path / "b", engine="native", status=PageStatus.SUCCESS, body=other
    )

    assert [Path(r).name for r in ps_own.unshipped_figure_refs] == ["figure_presence_p1_2.png"]
    assert [Path(r).name for r in ps_other.unshipped_figure_refs] == [
        "figure_presence_p1_1.png",
        "figure_presence_p1_2.png",
    ]


def test_the_crop_is_appended_even_when_another_image_is_already_on_the_page(
    tmp_path: Path,
) -> None:
    """The #1187 guard abstains on ANY image. Per-region evidence must not: a page with a logo
    still owes the chart its crop."""
    pdf = _chart_pdf(tmp_path / "c.pdf", ticks=True)
    body = f"{_PROSE}\n\n![logo](figures/logo.png)"
    st, ps, out = _decide_native(
        pdf, tmp_path, engine="native", status=PageStatus.SUCCESS, body=body
    )
    rec = _finalize(st, ps, out)
    assert f"]({ps.unshipped_figure_refs[0]})" in rec.output.text
    # and idempotent: a second finalisation does not double the crop
    again = manifest._apply_unshipped_figure_guard(rec.output, ps)
    assert again.text == rec.output.text


def test_a_crop_that_cannot_be_saved_marks_the_page_figure_specifically(
    tmp_path: Path, monkeypatch
) -> None:
    import socr.figures.chart_regions as cr

    monkeypatch.setattr(cr, "render_chart_region_crops", lambda *a, **k: [])
    pdf = _chart_pdf(tmp_path / "c.pdf", ticks=True)
    st, ps, out = _decide_native(
        pdf, tmp_path, engine="native", status=PageStatus.SUCCESS, body=_PROSE
    )
    assert ps.unshipped_figure_kind == fp.PRESENCE_RENDER_FAILED_KIND
    assert ps.unshipped_figure_unlocated == ["chart region 1"]
    rec = _finalize(st, ps, out)
    assert rec.output.status is not PageStatus.SUCCESS
    assert rec.output.failure_mode is FailureMode.FIGURE_RENDER_FAILED, (
        "transient: retried on resume"
    )
    reasons = [c.reason for c in rec.verdict.clauses if c.scope.startswith("figure:")]
    assert reasons and all("chart region" in r and "no crop" in r for r in reasons)


def test_a_page_the_pdf_cannot_be_read_for_is_recorded_not_passed(tmp_path: Path) -> None:
    st, ps, _ = _decide_native(
        tmp_path / "missing.pdf", tmp_path, engine="native", status=PageStatus.SUCCESS, body=_PROSE
    )
    assert ps.chart_region_inventory_failed
    assert [e.kind for e in st.events] == ["chart_region_inventory_failed"]
    assert not ps.unshipped_figure_kind, "could not look is not 'nothing there'"


def test_a_page_with_a_table_signal_is_left_to_the_chart_region_pass(tmp_path: Path) -> None:
    """GH-189's ``_preserve_chart_regions`` crops EVERY box of exactly that candidate set at
    assembly, so deciding the page here too would crop its chart twice."""
    from socr.core.document import DocumentHandle
    from socr.core.state import DocumentState

    pdf = _chart_pdf(tmp_path / "c.pdf", ticks=True)
    state = DocumentState(handle=DocumentHandle(path=pdf, page_count=1))
    ps = PageState(page_num=1)
    state.pages[1] = ps
    out = PageOutput(page_num=1, text=_PROSE, status=PageStatus.SUCCESS, engine="native")
    arms = {}
    for has_tables in (False, True):
        ps.has_tables = has_tables
        ps.unshipped_figure_kind = ""
        state.events.clear()
        _pipe()._agentic_figure_presence_page(state, 1, ps, out, tmp_path / "figures")
        arms[has_tables] = bool(ps.unshipped_figure_refs)
        ps.unshipped_figure_refs = []
    assert arms == {False: True, True: False}
    # ... but the page is RECORDED, not silent (Fable 5): every checked page has a record.
    ps.has_tables = True
    ps.unshipped_figure_kind = ""
    state.events.clear()
    _pipe()._agentic_figure_presence_page(state, 1, ps, out, tmp_path / "figures")
    (event,) = state.events
    assert event.kind == fp.PRESENCE_PASS_KIND
    assert event.detail == fp.PASS_LEFT_TO_CHART_REGION_PASS
    assert not ps.unshipped_figure_refs


def test_a_page_with_no_chart_box_is_recorded_not_silent(tmp_path: Path) -> None:
    pdf = _chart_pdf(tmp_path / "plain.pdf", ticks=False, charts=0)
    st, ps, out = _decide_native(
        pdf, tmp_path, engine="native", status=PageStatus.SUCCESS, body=_PROSE
    )
    (event,) = st.events
    assert event.kind == fp.PRESENCE_PASS_KIND and event.detail == fp.PASS_NO_BOX
    rec = _finalize(st, ps, out)
    assert rec.output.status is PageStatus.SUCCESS, "a record, never a mark"


def test_the_lane_leaves_other_engines_and_floor_pages_alone(tmp_path: Path) -> None:
    pdf = _chart_pdf(tmp_path / "c.pdf", ticks=True)
    for engine in ("chart_asset", ""):
        st, ps, _ = _decide_native(
            pdf, tmp_path, engine=engine, status=PageStatus.SUCCESS, body=_PROSE
        )
        assert not st.events and not ps.unshipped_figure_kind, engine


# ---------------------------------------------------------------------------
# The model side: placeholders matched one-to-one
# ---------------------------------------------------------------------------

_TWO_FIGURES = (
    "## Charts\n\n<!-- Image (69, 103, 494, 320) -->\n\nFirst panel text.\n\n"
    "<!-- Image (535, 92, 959, 320) -->\n\nSecond panel text."
)


def _write_assets(root: Path, body: str) -> None:
    """Make every image link in *body* resolve to a real file, as a lane that saved it would."""
    import re

    for ref in re.findall(r"!\[[^\]]*\]\(([^)\s]+)\)", body):
        target = root / ref
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"\x89PNG\r\n")


def _decide_model(tmp_path: Path, body: str, **ps_attrs):
    from socr.core.document import DocumentHandle
    from socr.core.state import DocumentState

    tmp_path.mkdir(parents=True, exist_ok=True)
    pdf = _chart_pdf(tmp_path / "plain.pdf", ticks=False, charts=0)
    if not ps_attrs.pop("skip_assets", False):
        _write_assets(tmp_path, body)
    state = DocumentState(handle=DocumentHandle(path=pdf, page_count=1))
    ps = PageState(page_num=1)
    for k, v in ps_attrs.items():
        setattr(ps, k, v)
    state.pages[1] = ps
    out = PageOutput(
        page_num=1, text=body, status=PageStatus.SUCCESS, engine="qwen", audit_passed=True
    )
    ps.best_output = out
    _pipe()._agentic_figure_presence_page(state, 1, ps, out, tmp_path / "figures")
    return state, ps, out


def test_a_two_figure_page_with_one_image_is_marked_for_the_second(tmp_path: Path) -> None:
    one = f"{_TWO_FIGURES}\n\n![chart region 1](figures/chart_region_p1_1.png)"
    two = f"{one}\n\n![chart region 2](figures/chart_region_p1_2.png)"
    st1, ps1, out1 = _decide_model(tmp_path / "a", one)
    st2, ps2, _ = _decide_model(tmp_path / "b", two)

    assert ps1.unshipped_figure_kind == fp.PLACEHOLDER_KIND
    assert ps1.unshipped_figure_unlocated == ["image placeholder 2"]
    assert not ps2.unshipped_figure_kind and not st2.events, "matched one-to-one: nothing owed"
    (event,) = st1.events
    assert event.data == {"placeholders": 2, "images": 1, "unmatched": 1}
    rec = _finalize(st1, ps1, out1)
    assert rec.output.status is not PageStatus.SUCCESS
    reasons = [c.reason for c in rec.verdict.clauses if c.scope.startswith("figure:")]
    assert len(reasons) == 1 and "image placeholder" in reasons[0]
    assert rec.output.failure_mode is FailureMode.FIGURE_NOT_SHOWN


def test_a_placeholder_beside_its_own_image_is_residue_not_a_lost_figure(tmp_path: Path) -> None:
    body = "<!-- Image (1, 2, 3, 4) -->\n\n![chart region 1](figures/chart_region_p1_1.png)"
    st, ps, _ = _decide_model(tmp_path, body)
    assert not st.events and not ps.unshipped_figure_kind


def test_an_image_link_to_a_file_nobody_saved_ships_nothing(tmp_path: Path) -> None:
    """Fable 1. Same body in both arms; only whether the linked file exists differs."""
    body = "<!-- Image (10, 20, 300, 200) -->\n\n![Figure 1](figures/figure_1.png)"
    st_ok, ps_ok, _ = _decide_model(tmp_path / "saved", body)
    st_gone, ps_gone, _ = _decide_model(tmp_path / "gone", body, skip_assets=True)
    assert not ps_ok.unshipped_figure_kind and not st_ok.events
    assert ps_gone.unshipped_figure_kind == fp.PLACEHOLDER_KIND
    (event,) = st_gone.events
    assert event.data == {"placeholders": 1, "images": 0, "unmatched": 1}


def test_a_table_crop_is_not_a_figure_image(tmp_path: Path) -> None:
    body = "<!-- Image (1, 2, 3, 4) -->\n\n![Failed table page 1](figures/failed_table_p1.png)"
    st, ps, _ = _decide_model(tmp_path, body)
    assert ps.unshipped_figure_kind == fp.PLACEHOLDER_KIND


def test_placeholders_inside_fenced_code_are_ignored() -> None:
    body = "```html\n<!-- Image (1, 2, 3, 4) -->\n```\n\ntext\n\n~~~\n<!-- Image (5) -->\n~~~"
    assert fp.placeholder_count(body) == 0
    assert fp.placeholder_count(body + "\n\n<!-- Image (9) -->") == 1


def test_a_page_the_figure_lane_already_marked_is_not_decided_twice(tmp_path: Path) -> None:
    st, ps, _ = _decide_model(
        tmp_path,
        _TWO_FIGURES,
        unshipped_figure_kind="unshipped_figure_unlocated",
        unshipped_figure_unlocated=["Figure 7"],
    )
    assert ps.unshipped_figure_kind == "unshipped_figure_unlocated" and not st.events


# ---------------------------------------------------------------------------
# Through the real process(), in BOTH provider states, plus resume
# ---------------------------------------------------------------------------


def _process_native(tmp_path: Path, monkeypatch, tag: str, pdf: Path, *, providers: bool, out=None):
    from socr.core.config import EngineType, PipelineConfig
    from socr.core.providers import PROFILE_QWEN_LOCAL
    from socr.pipeline import orchestrator as orch

    with monkeypatch.context() as m:
        m.setattr(orch, "get_engine", lambda engine_type: None)
        m.setattr(UnifiedPipeline, "_is_chart_asset_page", lambda self, *a, **k: False)
        pipe = UnifiedPipeline(
            PipelineConfig(
                agentic=True,
                quiet=True,
                primary_engine=EngineType.QWEN,
                local_engine=EngineType.QWEN,
                enabled_engines=[EngineType.QWEN],
                native_first=True,
                write_manifest=False,
                judge_backend="heuristic",
                dual_pass_tables=False,
                detect_equations=False,
                save_figures=False,
            )
        )
        pipe._available_engines_for_agentic = lambda: [PROFILE_QWEN_LOCAL] if providers else []
        pipe._resolve_crop_vlm_model = lambda: None
        pipe._resolve_judge_model = lambda *a, **k: ""
        out = out or (tmp_path / f"out-{tag}")
        pipe.process(pdf, output_dir=out)
    side = json.loads(next(iter(out.rglob("pages/00001.json"))).read_text())
    text = next(iter(out.rglob("pages/00001.md"))).read_text()
    log = json.loads(next(iter(out.rglob("audit_log.json"))).read_text())
    return side, text, log, out


@pytest.mark.parametrize("providers", [True, False], ids=["provider", "no-provider"])
def test_process_ships_the_crop_and_the_page_without_ticks_differs_only_by_it(
    tmp_path: Path, monkeypatch, providers: bool
) -> None:
    on = _process_native(
        tmp_path,
        monkeypatch,
        "on",
        _chart_pdf(tmp_path / "on.pdf", ticks=True),
        providers=providers,
    )
    off = _process_native(
        tmp_path,
        monkeypatch,
        "off",
        _chart_pdf(tmp_path / "off.pdf", ticks=False),
        providers=providers,
    )
    side_on, text_on, log_on, out_on = on
    side_off, text_off, log_off, _ = off

    assert side_on["engine"].startswith("native") and side_off["engine"] == side_on["engine"], (
        "the fixture must drive a NATIVE winner"
    )
    # The difference, exactly: a crop, its record, and the mark that says values are untranscribed.
    assert side_on["unshipped_figure_kind"] == fp.PRESENCE_SHIPPED_KIND
    assert side_off["unshipped_figure_kind"] == fp.PRESENCE_PASS_KIND
    assert side_on["unshipped_figure_refs"] and not side_off["unshipped_figure_refs"]
    ref = side_on["unshipped_figure_refs"][0]
    assert f"]({ref})" in text_on and "![" not in text_off
    assert (next(iter(out_on.rglob("figures"))) / Path(ref).name).is_file()
    kinds_on = {e["kind"] for e in log_on["events"]}
    kinds_off = {e["kind"] for e in log_off["events"]}
    assert {fp.PRESENCE_SHIPPED_KIND, "visual_values_not_transcribed"} <= kinds_on
    assert fp.PRESENCE_PASS_KIND in kinds_off and not (fp.PRESENCE_MARKED_KINDS & kinds_off)
    assert not leading_marks(text_off, 1) or side_off["status"] != "success"
    # #1244 C1: the winner-selection acceptance is the nested value and is not touched; the
    # published top-level label follows the page status (the on-arm page is WARNING).
    assert (
        side_on["winning_output"]["audit_passed"] == side_off["winning_output"]["audit_passed"]
    ), "audit_passed is not touched"
    if side_on["status"] != "success":
        assert side_on["audit_passed"] is False


def test_resume_restores_the_crop_the_record_and_the_mark(tmp_path: Path, monkeypatch) -> None:
    """A terminal page is skipped on resume; its picture, its record and its mark must come
    back from the sidecar and the replayed events, byte for byte."""
    pdf = _chart_pdf(tmp_path / "on.pdf", ticks=True)
    side1, text1, log1, out = _process_native(tmp_path, monkeypatch, "r", pdf, providers=True)
    for md in out.rglob("*.md"):
        if "pages" not in md.parts:
            md.unlink()  # an interrupted document: the per-page ledger must answer
    side2, text2, log2, _ = _process_native(
        tmp_path, monkeypatch, "r", pdf, providers=True, out=out
    )

    assert text2 == text1
    assert side2["unshipped_figure_refs"] == side1["unshipped_figure_refs"]
    assert side2["unshipped_figure_kind"] == fp.PRESENCE_SHIPPED_KIND
    for kind in (fp.PRESENCE_SHIPPED_KIND, "visual_values_not_transcribed"):
        assert [e["kind"] for e in log2["events"]].count(kind) == 1, kind


def test_the_pass_record_survives_resume(tmp_path: Path, monkeypatch) -> None:
    """The PASS record has no other carrier: no mark, no crop. Without replay a resumed
    document would report fewer checked pages than the run that made it."""
    pdf = _chart_pdf(tmp_path / "off.pdf", ticks=False)
    _, _, log1, out = _process_native(tmp_path, monkeypatch, "p", pdf, providers=True)
    for md in out.rglob("*.md"):
        if "pages" not in md.parts:
            md.unlink()
    _, _, log2, _ = _process_native(tmp_path, monkeypatch, "p", pdf, providers=True, out=out)
    assert [e["kind"] for e in log1["events"]].count(fp.PRESENCE_PASS_KIND) == 1
    assert [e["kind"] for e in log2["events"]].count(fp.PRESENCE_PASS_KIND) == 1


def test_every_presence_kind_is_replayed_on_resume() -> None:
    assert fp.PRESENCE_KINDS <= UnifiedPipeline.resume_restore_kinds()


# ---------------------------------------------------------------------------
# The crop extent (#1276 visual check: 121 of 128 crops partial, 37 with neighbour prose)
# ---------------------------------------------------------------------------

_BODY = (
    "The committee judged that the outlook for activity and inflation remained uncertain and "
    "that the risks to the outlook were roughly balanced over the period ahead of it."
)


def _frame_bars(page, rect: fitz.Rect) -> None:
    """Axes frame and five filled bars scaled to fit *rect*, whatever its width."""
    shape = page.new_shape()
    shape.draw_rect(rect)
    shape.finish(color=(0, 0, 0), width=1.5)
    step = rect.width / 6
    for i in range(5):
        x = rect.x0 + step * (i + 0.25)
        shape.draw_rect(fitz.Rect(x, rect.y1 - (i + 1) * 25, x + step * 0.5, rect.y1 - 2))
        shape.finish(color=(0.8, 0.1, 0.1), fill=(0.8, 0.1, 0.1), width=1.5)
    shape.commit()


def _column_page(
    path: Path,
    *,
    panels: int = 1,
    right_ticks: bool = False,
    gap_title: bool = False,
):
    """A two-column page. Left column: prose. Right column: title, chart, axis labels, NOTE and
    SOURCE lines, then body prose that is NOT the figure's. ``panels=2`` puts two panels side by
    side under ONE title. Returns the PDF path and the geometry the assertions read."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_textbox(fitz.Rect(72, 100, 250, 330), _BODY * 3, fontname="helv", fontsize=9)
    left, right, top, bottom = 300.0, 575.0, 150.0, 300.0
    title_y = top - (40 if gap_title else 12)
    page.insert_text(
        (left, title_y),
        "Figure 3. Real GDP growth and inflation, annual rate",
        fontname="helv",
        fontsize=9,
    )
    if panels == 1:
        frames = [fitz.Rect(left, top, right, bottom)]
    else:
        mid = (left + right) / 2
        frames = [fitz.Rect(left, top, mid - 25, bottom), fitz.Rect(mid + 25, top, right, bottom)]
    for frame in frames:
        _frame_bars(page, frame)
        for k, value in enumerate((0, 10, 20, 30, 40)):
            y = frame.y1 - k * 30
            page.insert_text((frame.x0 - 14, y), str(value), fontname="helv", fontsize=8)
            if right_ticks:
                page.insert_text((frame.x1 + 4, y), str(value), fontname="helv", fontsize=8)
        page.insert_text(
            (frame.x0 + 5, frame.y1 + 11), "2018 2019 2020", fontname="helv", fontsize=8
        )
    page.insert_text((left, bottom + 24), "NOTE: Seasonally adjusted.", fontname="helv", fontsize=8)
    page.insert_text(
        (left, bottom + 35), "SOURCE: Bureau of Statistics.", fontname="helv", fontsize=8
    )
    page.insert_textbox(
        fitz.Rect(left, bottom + 80, right, bottom + 200), _BODY * 2, fontname="helv", fontsize=9
    )
    doc.save(path)
    doc.close()
    geometry = {"left": left, "right": right, "top": top, "bottom": bottom, "title_y": title_y}
    return path, geometry


def _groups(pdf: Path):
    from socr.figures import figure_crop
    from socr.tables.reconstruct import chart_region_clusters

    with fitz.open(pdf) as doc:
        page = doc[0]
        clusters = chart_region_clusters(page)
        words = list(page.get_text("words"))
        rect = page.rect
    return figure_crop.crop_groups(words, [tuple(c) for c, _ in clusters], rect), clusters, words


def test_the_crop_stays_in_the_charts_own_column(tmp_path: Path) -> None:
    """(a) The detector's box runs to the page's left edge and takes the neighbouring column's
    prose with it. The crop does not: it starts at the chart's tick labels."""
    pdf, g = _column_page(tmp_path / "c.pdf")
    groups, clusters, words = _groups(pdf)
    (group,) = groups
    assert clusters[0][1].x0 == 0.0, "the box the crop replaces reaches the page edge"
    prose_right = max(w[2] for w in words if w[0] < 260 and w[1] > 90)
    assert group.rect[0] > prose_right, "no word of the left column's prose is inside the crop"
    tick_left = min(w[0] for w in words if w[4] == "40" and w[0] < g["left"])
    assert group.rect[0] <= tick_left, "the y-axis tick labels are inside the crop"


def test_the_crop_takes_its_title_and_notes_and_no_body_prose(tmp_path: Path) -> None:
    """(b) Title above, x-axis labels, NOTE and SOURCE below; the paragraph after them is not
    the figure's and stays out. Same page, title moved beyond the page's line pitch: it goes."""
    pdf, g = _column_page(tmp_path / "c.pdf")
    (group,), _, words = _groups(pdf)
    title_top = min(w[1] for w in words if w[4] == "Figure")
    source_bottom = max(w[3] for w in words if w[4] == "SOURCE:")
    body_top = min(w[1] for w in words if w[1] > g["bottom"] + 70)
    assert group.rect[1] <= title_top and group.rect[3] >= source_bottom
    assert group.rect[3] < body_top

    far, _ = _column_page(tmp_path / "far.pdf", gap_title=True)
    (far_group,), _, far_words = _groups(far)
    far_title_top = min(w[1] for w in far_words if w[4] == "Figure")
    assert far_group.rect[1] > far_title_top, "a title a paragraph away is not this figure's"


def _word(x0, y0, x1, y1, text, block, line=0):
    return (x0, y0, x1, y1, text, block, line, 0)


def test_the_title_just_below_a_figures_notes_is_the_next_figures(tmp_path: Path) -> None:
    """A one-line block under the SOURCE line that opens like a title is the NEXT figure's. The
    same block opening like a note is this figure's. One word differs between the arms."""
    from socr.figures import figure_crop

    core = (100.0, 100.0, 300.0, 200.0)
    reach = {}
    for lead in ("4.", "SOURCE:"):
        words = [
            _word(100, 205, 140, 213, "SOURCE:", 1),
            _word(145, 205, 200, 213, "Bureau", 1),
            _word(100, 216, 120, 224, lead, 2),
            _word(125, 216, 200, 224, "Series", 2),
            # a paragraph long after both, to give the page its line pitch
            _word(100, 300, 150, 308, "word", 3, 0),
            _word(100, 312, 150, 320, "word", 3, 1),
            _word(100, 324, 150, 332, "word", 3, 2),
        ]
        (group,) = figure_crop.crop_groups(words, [core], fitz.Rect(0, 0, 612, 792))
        reach[lead] = group.rect[3]
    assert reach["4."] < 216 < reach["SOURCE:"]


_CORE = (100.0, 100.0, 300.0, 200.0)
_PAGE = fitz.Rect(0, 0, 612, 792)
#: A paragraph far from the figure, to give the page its line pitch (12).
_PARAGRAPH = [_word(100, 500 + 12 * k, 150, 508 + 12 * k, "word", 9, k) for k in range(4)]


def _crop_rect(extra: list, core=_CORE):
    from socr.figures import figure_crop

    (group,) = figure_crop.crop_groups(extra + _PARAGRAPH, [core], _PAGE)
    return group.rect


def _within(seconds: int):
    """Fail instead of hanging: the walk must end on any input (SIGALRM, POSIX)."""
    import contextlib
    import signal

    @contextlib.contextmanager
    def guard():
        def _timeout(*_a):
            raise AssertionError("the crop walk did not terminate")

        previous = signal.signal(signal.SIGALRM, _timeout)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)

    return guard()


def test_a_zero_width_number_beside_the_drawing_ends_the_walk() -> None:
    """Carbon, round 4: a numeric word with ``x0 == x1`` just outside the rectangle pulled the
    edge to itself and was never consumed, so the sideways walk looped forever."""
    for words in (
        [_word(305, 150, 305, 158, "40", 1)],
        [_word(95, 150, 95, 158, "40", 1)],
        [_word(305, 150, 305, 158, "40", 1), _word(310, 150, 310, 158, "50", 2)],
    ):
        with _within(5):
            rect = _crop_rect(words)
        assert rect[0] <= 95 or rect[2] >= 305, "the label is inside the crop"


def test_the_sideways_walk_stops_at_the_nearest_block_that_is_not_a_tick_label() -> None:
    """cubic: prose between the drawing and a number is not crossed. One token differs between
    the arms: the block next to the drawing is a number or a word."""
    reach = {}
    for beside in ("5", "of"):
        reach[beside] = _crop_rect(
            [_word(302, 150, 310, 158, beside, 1), _word(312, 150, 322, 158, "40", 2)]
        )[2]
    assert reach["5"] >= 322, "numbers all the way: both are tick labels"
    assert reach["of"] < 312, "a word in between: the walk stops before the number behind it"


def test_a_short_block_under_the_drawing_needs_a_signal_of_its_own() -> None:
    """cubic: brevity and adjacency alone do not make a sentence part of the figure. Same size and
    the same distance under the drawing. The block that runs past the drawing's width (a column
    of prose) is left out unless its words say units or a note; the same words inside the
    drawing's width are the figure's own label."""
    bottoms = {}
    for tag, x, text in (
        ("plain", 40, ["Then", "we", "left"]),
        ("inside", 100, ["Then", "we", "left"]),
        ("units", 40, ["Billions", "of", "dollars"]),
        ("lead", 40, ["SOURCE:", "Bureau", "reports"]),
    ):
        words = [
            _word(x + 30 * k, 205, x + 25 + 30 * k, 213, token, 1) for k, token in enumerate(text)
        ]
        bottoms[tag] = _crop_rect(words)[3]
    assert bottoms["plain"] < 205 < bottoms["inside"]
    assert bottoms["plain"] < 205 < bottoms["units"]
    assert bottoms["lead"] > 205


def test_a_title_over_two_panels_is_taken_by_both_and_makes_them_one_crop() -> None:
    """A heading wider than either panel and centred on neither: it spans the panels, so it is
    the figure's title and the panels are one figure. One difference: the heading's words sit
    over the panels, or far to the side of them."""
    from socr.figures import figure_crop

    left, right = (100.0, 100.0, 200.0, 200.0), (260.0, 100.0, 360.0, 200.0)
    groups = {}
    for tag, x0 in (("over", 120), ("aside", 420)):
        words = [_word(x0 + 30 * k, 92, x0 + 25 + 30 * k, 100, "Head", 1) for k in range(8)]
        words += _PARAGRAPH
        groups[tag] = [g.indices for g in figure_crop.crop_groups(words, [left, right], _PAGE)]
    assert groups["over"] == [(1, 2)]
    assert groups["aside"] == [(1,), (2,)]


def test_a_block_the_extractor_joined_across_panels_is_cut_where_the_text_is_apart() -> None:
    """The extractor's own block numbering differs by version: PyMuPDF 1.28 puts panel 1's
    x-axis labels and panel 2's first tick label in one block, 1.27 does not. A block here is cut
    at a gap wider than one line pitch and joined only where extents overlap."""
    from socr.figures import figure_crop

    words = [
        _word(105, 203, 130, 211, "2018", 7, 0),
        _word(135, 203, 160, 211, "2019", 7, 0),
        _word(285, 191, 292, 199, "0", 7, 1),
        # one line holding both panels' labels, 170 apart
        _word(105, 230, 130, 238, "2018", 8, 0),
        _word(300, 230, 325, 238, "2018", 8, 0),
        _word(100, 500, 150, 508, "word", 9, 0),
        _word(100, 512, 150, 520, "word", 9, 1),
    ]
    blocks = figure_crop._blocks(words, figure_crop.line_pitch(words))
    spans = sorted((round(b.x0), round(b.x1)) for b in blocks if b.numeric)
    assert spans == [(105, 130), (105, 160), (285, 292), (300, 325)]


def test_tick_labels_outside_the_drawing_on_both_sides_are_inside_the_crop(tmp_path: Path) -> None:
    """(c) Right-hand y-axis labels, which the drawing box clips."""
    pdf, g = _column_page(tmp_path / "c.pdf", right_ticks=True)
    (group,), _, words = _groups(pdf)
    right_label = max(w[2] for w in words if w[4] == "40" and w[0] > g["right"] - 20)
    assert group.rect[2] >= right_label > g["right"]


def test_panels_of_one_figure_under_one_title_are_one_crop(tmp_path: Path) -> None:
    """(d) Two panels, one title: one crop covering both ordinals. The same two charts with the
    title moved out of reach are two crops."""
    pdf, _ = _column_page(tmp_path / "two.pdf", panels=2)
    groups, clusters, _ = _groups(pdf)
    assert len(clusters) == 2
    assert [g.indices for g in groups] == [(1, 2)]
    apart, _ = _column_page(tmp_path / "apart.pdf", panels=2, gap_title=True)
    apart_groups, _, _ = _groups(apart)
    assert [g.indices for g in apart_groups] == [(1,), (2,)], "no shared title, no merge"


def test_the_shipped_crop_is_cut_by_the_group_not_the_box(tmp_path: Path) -> None:
    """End to end through the lane: the PNG on disk is as wide as the group's rectangle and a
    multi-panel figure is ONE file named by its first ordinal."""
    pdf, _ = _column_page(tmp_path / "two.pdf", panels=2)
    st, ps, _ = _decide_native(
        pdf, tmp_path, engine="native", status=PageStatus.SUCCESS, body=_PROSE
    )
    groups, clusters, _ = _groups(pdf)
    assert ps.unshipped_figure_kind == fp.PRESENCE_SHIPPED_KIND
    assert ps.unshipped_figure_refs == [f"figures/{fp.crop_filename(1, 1)}"]
    pix = fitz.Pixmap(str(tmp_path / "figures" / fp.crop_filename(1, 1)))
    box_width = clusters[0][1].x1 - clusters[0][1].x0
    group_width = groups[0].rect[2] - groups[0].rect[0]
    assert group_width < box_width
    assert (
        abs(pix.width / group_width - pix.height / (groups[0].rect[3] - groups[0].rect[1])) < 0.05
    )


# ---------------------------------------------------------------------------
# Fable 3: one row of consecutive years is not a chart scale
# ---------------------------------------------------------------------------


def _box_of_words(lines: list[str]):
    words = []
    y = 10.0
    for n, line in enumerate(lines):
        x = 10.0
        for token in line.split():
            words.append((x, y, x + 20, y + 8, token, 0, n, 0))
            x += 25
        y += 12
    return words, [fitz.Rect(0, 0, 400, y + 5)]


@pytest.mark.parametrize(
    ("lines", "fires"),
    [
        (["2019 2020 2021 2022", "The committee noted that prices rose over the period."], False),
        (["0", "10", "20", "30"], True),
        (["2019 2020 2021 2022", "0 5 10 15"], True),
    ],
)
def test_one_row_of_years_beside_prose_is_not_chart_evidence(lines, fires) -> None:
    words, boxes = _box_of_words(lines)
    (a,) = fp.assess_regions(words, boxes, set(), [], False)
    assert a.ticks >= MIN_TICK_RUN
    assert a.fires is fires
    assert bool(a.axis) is fires


def _grid(rows: list[list[str]], *, one_line_per_word: bool, x0=20.0, step=60.0, y0=20.0):
    """Words laid out on a grid, the text layer's line numbering the only thing that varies:
    one (block, line) per row, or one per word (how many PDFs store a header row)."""
    words, line = [], 0
    for r, row in enumerate(rows):
        for c, token in enumerate(row):
            if not token:
                continue
            x = x0 + c * step
            words.append((x, y0 + r * 12, x + 24, y0 + r * 12 + 8, token, 0, line, 0))
            if one_line_per_word:
                line += 1
        line += 1
    return words, [fitz.Rect(0, 0, 400, y0 + len(rows) * 12 + 5)]


YEARS = ["2019", "2020", "2021", "2022"]


@pytest.mark.parametrize("one_line_per_word", [False, True], ids=["row-per-line", "line-per-year"])
def test_a_year_header_over_table_rows_is_not_a_scale_however_the_layer_lines_it(
    one_line_per_word: bool,
) -> None:
    """Fable 3. The same header row and the same rows beneath it; only the layer's line
    numbering differs, and the answer must not. The delimiter-free table is not chart evidence."""
    rows = [YEARS, ["Output", "Prices", "Hours", "Wages"], ["gains", "rose", "fell", "held"]]
    words, boxes = _grid(rows, one_line_per_word=one_line_per_word)
    (a,) = fp.assess_regions(words, boxes, set(), [], False)
    assert not a.fires and a.axis == ""


@pytest.mark.parametrize("one_line_per_word", [False, True], ids=["row-per-line", "line-per-year"])
def test_a_chart_whose_only_numbers_are_one_x_axis_row_fires(one_line_per_word: bool) -> None:
    """Fable 3, the other half: nothing sits under the row, so it is a chart's last line, not a
    table's header. The previous rule lost this chart."""
    words, boxes = _grid([YEARS], one_line_per_word=one_line_per_word)
    (a,) = fp.assess_regions(words, boxes, set(), [], False)
    assert a.fires and a.axis == "x"


def test_a_note_line_under_an_x_axis_row_does_not_make_it_a_table_header() -> None:
    """A figure's own NOTE line runs the figure's width and sits under its x-axis row. One
    difference: the line under the years opens with ``NOTE:`` or with a plain word."""
    verdict = {}
    for lead in ("NOTE:", "Output"):
        rows = [YEARS, [lead, "prices", "rose", "in", "2012"]]
        words, boxes = _grid(rows, one_line_per_word=False)
        (a,) = fp.assess_regions(words, boxes, set(), [], False)
        verdict[lead] = a.fires
    assert verdict == {"NOTE:": True, "Output": False}


def test_numbers_that_share_no_column_and_no_baseline_are_not_a_scale() -> None:
    """Three numbers in a run, one per line, each in a different column AND row."""
    words = [
        (20.0 + 90 * i, 20.0 + 40 * i, 44.0 + 90 * i, 28.0 + 40 * i, str(10 * i), 0, i, 0)
        for i in range(4)
    ]
    (a,) = fp.assess_regions(words, [fitz.Rect(0, 0, 400, 200)], set(), [], False)
    assert a.ticks >= MIN_TICK_RUN and not a.fires and a.axis == ""


# ---------------------------------------------------------------------------
# Fable 2: on a table-signal model page only boxes with the regional evidence count
# ---------------------------------------------------------------------------


# (the table-page placeholder tests drive _phase_assemble; see section (c) below)


# ---------------------------------------------------------------------------
# Review round on PR #1309: scope ordinal, resume retry, shipped-only credit, records,
# wording, literal context, and the model rule through process()
# ---------------------------------------------------------------------------


def test_the_clause_scope_carries_the_region_ordinal_not_the_list_position() -> None:
    """cubic: failed region 2 alone is ``figure:2`` while the reason says ``chart region 2``."""
    from socr.core.marks import region_scope

    (record,) = fp.clause_records(1, ("chart region 2",), fp.PRESENCE_RENDER_FAILED_KIND)
    assert record["scope"] == region_scope("figure", 2)
    assert record["reason"].startswith("chart region 2:")
    # Two placeholders owed (2 and 3 of 3): each keeps its own ordinal.
    scopes = [
        r["scope"]
        for r in fp.clause_records(
            1, ("image placeholder 2", "image placeholder 3"), fp.PLACEHOLDER_KIND
        )
    ]
    assert scopes == [region_scope("figure", 2), region_scope("figure", 3)]
    # No number in a label: the list position is the fallback, never a crash.
    (fallback,) = fp.clause_records(1, ("a figure",), fp.PLACEHOLDER_KIND)
    assert fallback["scope"] == region_scope("figure", 1)


def test_a_failed_second_region_reaches_the_verdict_with_its_own_scope(
    tmp_path: Path, monkeypatch
) -> None:
    import socr.figures.chart_regions as cr

    real = cr.render_chart_region_crops

    def only_the_first(*args, **kwargs):
        return [a for a in real(*args, **kwargs) if a.region_index == 1]

    monkeypatch.setattr(cr, "render_chart_region_crops", only_the_first)
    pdf = _chart_pdf(tmp_path / "two.pdf", ticks=True, charts=2)
    st, ps, out = _decide_native(
        pdf, tmp_path, engine="native", status=PageStatus.SUCCESS, body=_PROSE
    )
    assert ps.unshipped_figure_unlocated == ["chart region 2"]
    rec = _finalize(st, ps, out)
    scopes = [c.scope for c in rec.verdict.clauses if c.scope.startswith("figure:")]
    assert scopes == ["figure:2"]


def test_a_page_with_two_shipped_crops_records_every_crop(tmp_path: Path) -> None:
    """The #1174 record names each shipped crop, not the first only."""
    pdf = _chart_pdf(tmp_path / "two.pdf", ticks=True, charts=2)
    st, ps, _ = _decide_native(
        pdf, tmp_path, engine="native", status=PageStatus.SUCCESS, body=_PROSE
    )
    assert len(ps.unshipped_figure_refs) == 2
    recorded = [e.data["png_path"] for e in st.events if e.kind == "visual_values_not_transcribed"]
    assert recorded == ps.unshipped_figure_refs


def test_a_render_failure_is_retried_on_resume_and_the_old_record_goes(
    tmp_path: Path, monkeypatch
) -> None:
    """cubic P1: the replay restores ``figure_presence_render_failed`` and the lane returned
    for it. The ONLY difference between the runs is whether the crop can be saved."""
    import socr.figures.chart_regions as cr

    pdf = _chart_pdf(tmp_path / "on.pdf", ticks=True)
    with monkeypatch.context() as m:
        m.setattr(cr, "render_chart_region_crops", lambda *a, **k: [])
        side1, _t1, log1, out = _process_native(tmp_path, monkeypatch, "rf", pdf, providers=True)
    assert side1["unshipped_figure_kind"] == fp.PRESENCE_RENDER_FAILED_KIND
    for md in out.rglob("*.md"):
        if "pages" not in md.parts:
            md.unlink()
    side2, text2, log2, _ = _process_native(
        tmp_path, monkeypatch, "rf", pdf, providers=True, out=out
    )
    kinds = [e["kind"] for e in log2["events"]]
    assert side2["unshipped_figure_kind"] == fp.PRESENCE_SHIPPED_KIND
    assert side2["unshipped_figure_refs"] and f"]({side2['unshipped_figure_refs'][0]})" in text2
    assert not side2["unshipped_figure_unlocated"]
    assert fp.PRESENCE_RENDER_FAILED_KIND not in kinds, "the superseded failure is not kept"
    assert kinds.count(fp.PRESENCE_SHIPPED_KIND) == 1


def test_a_shipped_record_is_not_decided_again_on_the_same_page(tmp_path: Path) -> None:
    """The retry exception is the render failure ONLY."""
    pdf = _chart_pdf(tmp_path / "c.pdf", ticks=True)
    st, ps, out = _decide_native(
        pdf, tmp_path, engine="native", status=PageStatus.SUCCESS, body=_PROSE
    )
    before = [e.kind for e in st.events]
    _pipe()._agentic_figure_presence_page(st, 1, ps, out, tmp_path / "figures")
    assert [e.kind for e in st.events] == before


# -- (c) table-signal pages: judged at assembly from the bytes that ship ---------------------


def _gh189():
    """The GH-189 fixtures: a real mixed chart+table page, a real detector and crop renderer."""
    import test_gh189_mixed_chart_preservation as g

    return g


_PLACEHOLDER_WINNER = "<!-- Image (10, 20, 300, 200) -->\n\n" + _gh189().WINNER_TEXT
_TWO_PLACEHOLDERS_WINNER = "<!-- Image (10, 20, 300, 200) -->\n" + _PLACEHOLDER_WINNER


def _mixed_pdf(path: Path, *, ticks: bool, pages: int = 1) -> Path:
    """The GH-189 page (chart, table, chart) with, when *ticks*, a column of axis numbers beside
    the first chart. The only difference between the arms. Extra pages are chart-free text."""
    g = _gh189()
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    g._write_page_text(page)
    g._draw_bars(page, 90, 200)
    g._draw_bars(page, 490, 600)
    if ticks:
        for k, value in enumerate((0, 10, 20, 30, 40)):
            page.insert_text((60, 200 - k * 25), str(value), fontname="helv", fontsize=9)
    for _ in range(pages - 1):
        extra = doc.new_page(width=612, height=792)
        g._write_page_text(extra)
    doc.save(path)
    doc.close()
    return path


def _table_pipeline_state(pdf: Path, pages: int = 1):
    from socr.core.born_digital import DocumentAssessment, PageAssessment
    from socr.core.document import DocumentHandle
    from socr.core.state import DocumentState

    g = _gh189()
    pipeline = g._make_pipeline()
    state = DocumentState(handle=DocumentHandle(path=pdf, page_count=pages))
    assessed = []
    for n in range(1, pages + 1):
        state.pages[n] = PageState(
            page_num=n,
            is_born_digital=True,
            native_text="Figure preamble sentence unique alpha",
            needs_ocr_enhancement=False,
            has_tables=True,
        )
        assessed.append(
            PageAssessment(
                page_num=n,
                is_born_digital=True,
                native_text="Figure preamble sentence unique alpha",
                confidence=0.9,
                needs_ocr_enhancement=False,
                has_tables=True,
            )
        )
    state._last_assessment = DocumentAssessment(path=pdf, pages=assessed)
    pipeline._last_assessment = state._last_assessment
    return pipeline, state


def _table_page_run(tmp_path: Path, tag: str, *, ticks: bool, providers: bool, crop_fails=False):
    """Page loop then assembly, as the GH-189 tests drive them. One arm per call."""
    from unittest.mock import patch

    from socr.core.providers import PROFILE_QWEN_LOCAL

    g = _gh189()
    pdf = _mixed_pdf(tmp_path / f"{tag}.pdf", ticks=ticks)
    out_dir = tmp_path / f"out-{tag}"
    pipeline, state = _table_pipeline_state(pdf)
    real_save = fitz.Pixmap.save

    def _save(self, target, *args, **kwargs):
        if crop_fails and str(target).endswith("chart_region_p1_1.png"):
            raise RuntimeError("pixmap save died")
        return real_save(self, target, *args, **kwargs)

    with (
        patch(
            "socr.pipeline.orchestrator.route_page",
            return_value=g._accepted_decision(_PLACEHOLDER_WINNER),
        ),
        patch.object(
            UnifiedPipeline,
            "_available_engines_for_agentic",
            return_value=[PROFILE_QWEN_LOCAL] if providers else [],
        ),
        patch.object(UnifiedPipeline, "_resolve_judge_model", return_value=""),
        patch.object(fitz.Pixmap, "save", _save),
    ):
        pipeline._phase_agentic(state, out_dir)
        before = state.pages[1].unshipped_figure_kind
        result = pipeline._phase_assemble(state, out_dir)
    ps = state.pages[1]
    return ps, before, result, g._final_page_status(out_dir), state


@pytest.mark.parametrize("providers", [True, False], ids=["provider", "no-provider"])
def test_a_table_pages_placeholder_is_excused_only_by_a_crop_of_a_box_with_ticks(
    tmp_path: Path, providers: bool
) -> None:
    """Through the page loop and ``_phase_assemble``. Same page, same placeholder, same
    GH-189 crops; the arms differ only in whether the first box carries axis numbers."""
    arms = {}
    for ticks in (True, False):
        ps, before, result, status, state = _table_page_run(
            tmp_path, f"t{ticks}-{providers}", ticks=ticks, providers=providers
        )
        arms[ticks] = (ps.unshipped_figure_kind, before, status, state.pages[1].best_output)
    if providers:
        assert arms[True][3] is not None and arms[True][3].engine == "deepseek", "a MODEL page"
        assert arms[True][1] == arms[False][1] == "", "nothing is decided before the crops exist"
        assert arms[True][0] == ""
        assert arms[False][0] == fp.PLACEHOLDER_KIND
        assert arms[False][2] != "success"
    else:
        assert arms[True][0] == arms[False][0] != fp.PLACEHOLDER_KIND, "no model read the page"


def test_a_table_page_whose_crop_failed_is_marked(tmp_path: Path) -> None:
    ps, _before, _result, status, _state = _table_page_run(
        tmp_path, "failed", ticks=True, providers=True, crop_fails=True
    )
    assert ps.unshipped_figure_kind == fp.PLACEHOLDER_KIND
    assert status != "success"


def test_an_already_linked_crop_does_not_excuse_a_second_placeholder(tmp_path: Path) -> None:
    """Carbon 2: one crop file, linked once by the model and once by GH-189, is one picture."""
    out = tmp_path / "doc"
    (out / "figures").mkdir(parents=True)
    (out / "figures" / "chart_region_p1_1.png").write_bytes(b"png")
    body = (
        "<!-- Image (1, 2, 3, 4) -->\n<!-- Image (5, 6, 7, 8) -->\n"
        "![a](figures/chart_region_p1_1.png)\n\n![b](figures/chart_region_p1_1.png)\n"
    )
    assert fp.placeholder_count(body) == 2
    assert fp.shipped_image_count(body, out) == 1
    assert fp.shipped_image_count(body) == 1, "the same on link text alone"


@pytest.mark.parametrize("providers", [True, False], ids=["provider", "no-provider"])
def test_a_terminal_table_page_never_judged_is_judged_on_resume(
    tmp_path: Path, providers: bool
) -> None:
    """Carbon, round 4. A model page ends a run terminal and unmarked with its placeholders never
    judged (an older run, or an assembly that stopped before the judgement). The resume skips it
    through ``_load_terminal_page`` and must still mark it at assembly, from the bytes. Two
    placeholders and one crop of a box with ticks: one placeholder is unmatched. One difference
    between the arms: whether the first run's judgement was switched off."""
    import shutil
    from unittest.mock import patch

    from socr.core.providers import PROFILE_QWEN_LOCAL
    from socr.pipeline.orchestrator import UnifiedPipeline as U

    g = _gh189()
    outcomes = {}
    for legacy in (False, True):
        tag = f"{legacy}-{providers}"
        pdf = _mixed_pdf(tmp_path / f"{tag}.pdf", ticks=True, pages=2)
        out_dir = tmp_path / f"out-{tag}"
        routed: list[int] = []

        def _route(*args, _routed=routed, **kwargs):
            _routed.append(kwargs.get("page_num") or next(a for a in args if isinstance(a, int)))
            return g._accepted_decision(_TWO_PLACEHOLDERS_WINNER)

        skipped: list[int] = []
        real_load = U._load_terminal_page

        def _spy(self, state, page_num, out, _skipped=skipped, _real=real_load):
            resumed = _real(self, state, page_num, out)
            if resumed is not None:
                _skipped.append(page_num)
            return resumed

        with (
            patch("socr.pipeline.orchestrator.route_page", side_effect=_route),
            patch.object(
                U,
                "_available_engines_for_agentic",
                return_value=[PROFILE_QWEN_LOCAL] if providers else [],
            ),
            patch.object(U, "_resolve_judge_model", return_value=""),
        ):
            if legacy:
                pipeline, state = _table_pipeline_state(pdf, pages=2)
                with patch.object(U, "_reconcile_model_placeholders", lambda *a, **k: None):
                    pipeline._phase_agentic(state, out_dir)
                    pipeline._phase_assemble(state, out_dir)
                first = json.loads(next(iter(out_dir.rglob("pages/00001.json"))).read_text())
                assert (
                    first["terminal"] and first["unshipped_figure_kind"] != fp.PLACEHOLDER_KIND
                ), "page 1 ended terminal and unmarked: its placeholders were never judged"
                for md in out_dir.rglob("*.md"):
                    if "pages" not in md.parts:
                        md.unlink()
                routed.clear()
            pipeline, state = _table_pipeline_state(pdf, pages=2)
            with patch.object(U, "_load_terminal_page", _spy):
                pipeline._phase_agentic(state, out_dir)
            pipeline._phase_assemble(state, out_dir)
        sidecar = json.loads(next(iter(out_dir.rglob("pages/00001.json"))).read_text())
        outcomes[legacy] = (
            state.pages[1].unshipped_figure_kind,
            sidecar["unshipped_figure_kind"],
            list(skipped),
            list(routed),
        )
        shutil.rmtree(out_dir, ignore_errors=True)
    if providers:
        assert outcomes[True][2] == [1, 2], "the resume skipped page 1 through the ledger"
        assert outcomes[True][3] == [], "no page was routed again"
        assert outcomes[False][2] == []
        assert outcomes[True][:2] == outcomes[False][:2] == (fp.PLACEHOLDER_KIND,) * 2
    else:
        assert outcomes[True][0] == outcomes[False][0] != fp.PLACEHOLDER_KIND


# -- (d) a #1187-answered page keeps exactly its #1187 record ---------------------------------


@pytest.mark.parametrize(
    "kind",
    [
        "unshipped_figure_rendered",
        "unshipped_figure_render_failed",
        "unshipped_figure_unlocated",
        "unshipped_figure_chart_skeleton",
    ],
)
def test_a_page_the_1187_lane_answered_keeps_that_record_and_adds_none(
    tmp_path: Path, kind: str
) -> None:
    st, ps, out = _decide_model(
        tmp_path,
        _TWO_FIGURES,
        unshipped_figure_kind=kind,
        unshipped_figure_unlocated=["Figure 7"],
    )
    assert ps.unshipped_figure_kind == kind and not st.events
    from socr.pipeline.orchestrator import UNSHIPPED_FIGURE_MARKED_KINDS

    assert kind in UNSHIPPED_FIGURE_MARKED_KINDS, "that record already marks the page"


# -- (e) the document note and the CLI name the new cases -------------------------------------


def test_the_note_words_the_presence_cases_and_not_as_a_figure_socr_could_not_locate() -> None:
    from socr.figures.unshipped_figures import document_note

    note = document_note([], [3, 5], [], chart_shipped=[2], chart_lost=[3], placeholder=[5])
    assert "page(s) 2 draw a chart that ships as an image" in note
    assert "page(s) 3 draw a chart with axis ticks whose crop could not be saved" in note
    assert "page(s) 5 carry an unmatched figure placeholder" in note
    assert "could not locate" not in note, "pages 3 and 5 are not figures socr could not locate"
    assert ";" not in note, "a note with ';' cannot be replaced as a unit"
    # A plain unshipped-figure page keeps the old sentence beside them.
    assert "page(s) 9 name a figure socr could not locate" in document_note(
        [], [9, 3], [], chart_lost=[3]
    )


def test_the_live_note_and_the_cli_line_read_the_same_events() -> None:
    from socr.core.audit_log import AuditEvent
    from socr.core.document import DocumentHandle
    from socr.core.state import DocumentState

    state = DocumentState(handle=DocumentHandle(path=Path("x.pdf"), page_count=6))
    for page, kind in (
        (2, fp.PRESENCE_SHIPPED_KIND),
        (3, fp.PRESENCE_RENDER_FAILED_KIND),
        (5, fp.PLACEHOLDER_KIND),
    ):
        state.events.append(AuditEvent(page_num=page, kind=kind, engine="", detail="d"))
    assert UnifiedPipeline._presence_pages(state) == ([2], [3], [5])
    note = UnifiedPipeline._unshipped_figure_note(state)
    assert "unmatched figure placeholder" in note and "ships as an image" in note
    assert "could not locate" not in note


def test_refinalize_words_a_partly_lost_chart_page_as_the_live_note_does() -> None:
    """cubic. A page that saved one crop and lost another carries refs AND the lost kind. The
    live note names it by its figure-specific clause only; the tool must say the same."""
    from socr.core.audit_log import AuditEvent
    from socr.core.document import DocumentHandle
    from socr.core.state import DocumentState
    from socr.pipeline.refinalize_unshipped_figures import answered_document_note

    state = DocumentState(handle=DocumentHandle(path=Path("x.pdf"), page_count=4))
    sidecars = []
    for page, kind, refs in (
        (2, fp.PRESENCE_SHIPPED_KIND, ["figures/a.png"]),
        (3, fp.PRESENCE_RENDER_FAILED_KIND, ["figures/b.png"]),
    ):
        state.events.append(AuditEvent(page_num=page, kind=kind, engine="", detail="d"))
        sidecars.append(
            {
                "page_num": page,
                "unshipped_figure_kind": kind,
                "unshipped_figure_refs": refs,
                "unshipped_figure_unlocated": (
                    ["chart region 2"] if kind == fp.PRESENCE_RENDER_FAILED_KIND else []
                ),
            }
        )
    live = UnifiedPipeline._unshipped_figure_note(state)
    assert live and "ships as an image" in live
    assert answered_document_note(sidecars) == live


# -- (g) literal context ----------------------------------------------------------------------


def test_placeholders_in_table_cells_and_inline_code_are_not_counted() -> None:
    assert fp.placeholder_count("| a | b |\n| --- | --- |\n| x | <!-- Image (1) --> |") == 0
    assert fp.placeholder_count("a | b | c <!-- Image (1) -->") == 1, "pipes in prose: no table"
    assert fp.placeholder_count("x | y\n\n<!-- Image (1) -->\n\n| a | b |\n| - | - |") == 1
    assert fp.placeholder_count("a `<!-- Image (x) -->` quoted") == 0
    assert fp.placeholder_count("a ``<!-- Image (x) -->`` quoted") == 0
    assert fp.placeholder_count("<!-- Image (1, 2, 3, 4) -->") == 1
    assert fp.placeholder_count("text `code` then <!-- Image (1) -->") == 1


def test_a_longer_fence_is_not_closed_by_an_inner_shorter_one() -> None:
    body = "````md\n```\n<!-- Image (x) -->\n```\n<!-- Image (y) -->\n````\n\n<!-- Image (z) -->"
    assert fp.placeholder_count(body) == 1


# -- (h) the model rule through the real process(), both provider states -----------------------

_PLACEHOLDER_BODY = (
    "Quarterly results\n\n<!-- Image (10, 20, 300, 200) -->\n\n"
    "The estimated effect of the policy change on output is reported above."
)
_PLAIN_BODY = _PLACEHOLDER_BODY.replace("<!-- Image (10, 20, 300, 200) -->\n\n", "")


class _ModelEngine:
    name = "qwen"
    BODY = _PLACEHOLDER_BODY

    def is_available(self) -> bool:
        return True

    def process_pages(self, pdf_path, page_nums, config, dpi, subprocess_timeout=None, **_kw):
        return [
            PageOutput(page_num=n, text=self.BODY, status=PageStatus.SUCCESS, engine="qwen")
            for n in page_nums
        ]


class _AcceptJudge:
    def assess(self, output, provider):
        from socr.pipeline.agentic import AcceptDecision

        return AcceptDecision(accept=True, reason="")


def _process_model(tmp_path: Path, monkeypatch, tag: str, body: str, *, providers: bool):
    from socr.core.config import EngineType, PipelineConfig
    from socr.core.providers import PROFILE_QWEN_LOCAL
    from socr.pipeline import orchestrator as orch

    engine = _ModelEngine()
    engine.BODY = body
    pdf = _chart_pdf(tmp_path / f"{tag}.pdf", ticks=False, charts=0)
    with monkeypatch.context() as m:
        m.setattr(orch, "get_engine", lambda engine_type: engine)
        m.setattr(UnifiedPipeline, "_is_chart_asset_page", lambda self, *a, **k: False)
        m.setattr(UnifiedPipeline, "_is_agentic_trusted_native", lambda self, *a, **k: False)
        pipe = UnifiedPipeline(
            PipelineConfig(
                agentic=True,
                quiet=True,
                primary_engine=EngineType.QWEN,
                local_engine=EngineType.QWEN,
                enabled_engines=[EngineType.QWEN],
                native_first=True,
                write_manifest=False,
                judge_backend="heuristic",
                dual_pass_tables=False,
                detect_equations=False,
                save_figures=False,
            )
        )
        pipe._available_engines_for_agentic = lambda: [PROFILE_QWEN_LOCAL] if providers else []
        pipe._build_page_judge = lambda state: _AcceptJudge()
        pipe._resolve_crop_vlm_model = lambda: None
        pipe._resolve_judge_model = lambda *a, **k: ""
        pipe.process(pdf, output_dir=tmp_path / f"out-{tag}")
    out = tmp_path / f"out-{tag}"
    side = json.loads(next(iter(out.rglob("pages/00001.json"))).read_text())
    text = next(iter(out.rglob("pages/00001.md"))).read_text()
    # A run with no event writes no audit log at all.
    log_path = next(iter(out.rglob("audit_log.json")), None)
    log = json.loads(log_path.read_text()) if log_path else {"events": []}
    return side, text, log


def test_process_marks_an_unmatched_model_placeholder_and_only_when_a_model_read_the_page(
    tmp_path: Path, monkeypatch
) -> None:
    """One difference per provider state: the placeholder in the model's reading. With a
    provider the model reads the page and the rule fires; with none the engine is never asked,
    the page is the native text in both arms, and the rule has nothing to judge."""
    arms = {}
    for providers in (True, False):
        for tag, body in (("on", _PLACEHOLDER_BODY), ("off", _PLAIN_BODY)):
            side, text, log = _process_model(
                tmp_path, monkeypatch, f"{tag}-{providers}", body, providers=providers
            )
            arms[providers, tag] = (side, text, {e["kind"] for e in log["events"]})

    on, off = arms[True, "on"], arms[True, "off"]
    assert on[0]["engine"] == off[0]["engine"] == "qwen", "the fixture must drive a MODEL page"
    assert on[0]["unshipped_figure_kind"] == fp.PLACEHOLDER_KIND
    assert off[0]["unshipped_figure_kind"] != fp.PLACEHOLDER_KIND
    assert fp.PLACEHOLDER_KIND in on[2] and fp.PLACEHOLDER_KIND not in off[2]
    assert leading_marks(on[1], 1) and on[0]["status"] != "success"
    assert any("image placeholder" in str(v) for v in on[0].get("verdict", {}).get("clauses", []))

    noprov_on, noprov_off = arms[False, "on"], arms[False, "off"]
    assert noprov_on[0]["unshipped_figure_kind"] == noprov_off[0]["unshipped_figure_kind"]
    assert noprov_on[1] == noprov_off[1], "no provider: the model never read the page"
    assert fp.PLACEHOLDER_KIND not in noprov_on[2]


def test_a_restored_render_failure_is_retried_in_place(tmp_path: Path) -> None:
    """The lane reached with the restored kind (a replayed event and a restored state), the
    crop now savable: the retry replaces the old failure instead of being refused by it. One
    difference: only the recorded kind. Through ``process()`` a failed page is reprocessed
    from scratch (its status is not SUCCESS), so this state is reached by the replay path."""
    from socr.core.audit_log import AuditEvent

    pdf = _chart_pdf(tmp_path / "c.pdf", ticks=True)
    outcomes = {}
    for restored in (False, True):
        from socr.core.document import DocumentHandle
        from socr.core.state import DocumentState

        state = DocumentState(handle=DocumentHandle(path=pdf, page_count=1))
        ps = PageState(page_num=1)
        ps.is_born_digital = True
        state.pages[1] = ps
        if restored:
            ps.unshipped_figure_kind = fp.PRESENCE_RENDER_FAILED_KIND
            ps.unshipped_figure_unlocated = ["chart region 1"]
            state.events.append(
                AuditEvent(
                    page_num=1,
                    kind=fp.PRESENCE_RENDER_FAILED_KIND,
                    engine="figure_presence",
                    detail="old",
                )
            )
        out = PageOutput(page_num=1, text=_PROSE, status=PageStatus.SUCCESS, engine="native")
        ps.best_output = out
        _pipe()._agentic_figure_presence_page(state, 1, ps, out, tmp_path / "figures")
        outcomes[restored] = (
            ps.unshipped_figure_kind,
            [e.kind for e in state.events],
            ps.unshipped_figure_unlocated,
        )
    assert outcomes[True] == outcomes[False]
    assert outcomes[True][0] == fp.PRESENCE_SHIPPED_KIND
    assert fp.PRESENCE_RENDER_FAILED_KIND not in outcomes[True][1]
    assert outcomes[True][2] == []
