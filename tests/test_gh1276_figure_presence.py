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
    assert side_on["audit_passed"] == side_off["audit_passed"], "audit_passed is not touched"


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
    assert (a.tick_lines >= fp.MIN_TICK_LINES) is fires


# ---------------------------------------------------------------------------
# Fable 2: on a table-signal model page only boxes with the regional evidence count
# ---------------------------------------------------------------------------


def test_a_bare_box_does_not_excuse_a_placeholder_on_a_table_page(tmp_path: Path) -> None:
    """Same page, same placeholder, same table signal; only the box's tick numbers differ."""
    outcome = {}
    for ticks in (True, False):
        root = tmp_path / str(ticks)
        root.mkdir()
        pdf = _chart_pdf(root / "c.pdf", ticks=ticks)
        from socr.core.document import DocumentHandle
        from socr.core.state import DocumentState

        state = DocumentState(handle=DocumentHandle(path=pdf, page_count=1))
        ps = PageState(page_num=1)
        ps.has_tables = True
        state.pages[1] = ps
        out = PageOutput(
            page_num=1,
            text="<!-- Image (1, 2, 3, 4) -->\n\nA table follows.",
            status=PageStatus.SUCCESS,
            engine="qwen",
        )
        _pipe()._agentic_figure_presence_page(state, 1, ps, out, root / "figures")
        outcome[ticks] = ps.unshipped_figure_kind
    assert outcome == {True: "", False: fp.PLACEHOLDER_KIND}
