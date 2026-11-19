"""#1385: a model-read page whose reading has no figure caption and no axis ticks ships a crop
of the figure region the locator finds, and is marked.

Reopens the "known gap" #1225 recorded. On the 100 hard Fed pages the locator finds a region on
about a dozen such pages, nearly all of them chart pages whose reading kept panel titles and
nothing else (Fed IV-22: three charts, shipped as three titles and a footnote).

Hermetic: no provider, no model. The ``process()`` tests stub the engine, the judge and the
ladder and pin the engines, as #1187's do (CLAUDE.md: CI has no ollama).

Differences, not absolutes (CLAUDE.md, #257): each behavioural test runs one page twice and
changes only whether the PDF carries a figure the locator can find.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_gh1187_unshipped_figure_pages import (
    _Engine,
    _out,
    _pdf_with_a_figure,
    _process,
    _run_dir,
    _tree,
)

from socr.core import manifest
from socr.core.marks import Check, CheckResult, leading_marks
from socr.core.result import FailureMode, PageOutput, PageStatus
from socr.core.state import PageState
from socr.figures.unshipped_figures import (
    CHART_SKELETON_LABEL,
    EVENT_KINDS,
    MARKED_KINDS,
    SHIPPED_KINDS,
    UNCAPTIONED_CROP_KIND,
    UNCAPTIONED_CROP_LOST_KIND,
    clause_records,
    decide,
    document_note,
    figure_captions,
    tick_run,
)
from socr.pipeline import refinalize_unshipped_figures as tool
from socr.pipeline.orchestrator import UnifiedPipeline


@pytest.fixture(autouse=True)
def _no_words_check(monkeypatch):
    """The stub reading holds only part of the page's words; #1294's check is tested elsewhere
    and would otherwise name its own failure mode on both arms."""
    from socr.core import verdict

    monkeypatch.setattr(verdict, "page_missing_words", lambda output, p: None)


#: What a model keeps of a chart page: titles and units, no caption, no axis numbers.
BODY = "## Average Real GDP\n\nQuarterly\n\n## Consumer Prices\n\nMonthly 4.2"
_REGION = [(100.0, 120.0, 500.0, 420.0)]


def test_the_fixture_is_uncaptioned_and_tickless() -> None:
    assert not figure_captions(BODY)
    assert tick_run(BODY) < 3


def test_loaded_source_is_this_checkout() -> None:
    import socr

    assert Path(socr.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1] / "src")


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------


def test_decide_difference_region_found_vs_not() -> None:
    assert decide(BODY, lambda: [], lambda b: ["figures/x.png"]) is None
    crop = decide(BODY, lambda: _REGION, lambda b: ["figures/x.png"])
    assert crop is not None
    assert crop.kind == UNCAPTIONED_CROP_KIND
    assert crop.refs == ("figures/x.png",) and crop.regions == 1
    assert crop.unlocated == (CHART_SKELETON_LABEL,), "a crop never discharges the mark"
    assert crop.marks


def test_a_region_whose_crop_cannot_be_saved_is_marked_as_lost() -> None:
    lost = decide(BODY, lambda: _REGION, lambda b: [])
    assert lost is not None
    assert lost.kind == UNCAPTIONED_CROP_LOST_KIND and lost.refs == () and lost.marks


def test_a_captioned_page_keeps_the_1187_answer() -> None:
    decision = decide(f"Figure 7: Bias.\n\n{BODY}", lambda: _REGION, lambda b: ["f.png"])
    assert decision is not None and decision.kind != UNCAPTIONED_CROP_KIND


def test_ticks_keep_the_1225_answer_and_never_call_the_locator() -> None:
    def boom():
        raise AssertionError("the skeleton branch must not call the locator")

    decision = decide(f"{BODY}\n\n1996 1998 2000 2002 2004", boom, lambda b: [])
    assert decision is not None and decision.kind != UNCAPTIONED_CROP_KIND


def test_the_kinds_are_marked_replayed_and_the_crop_ships() -> None:
    for kind in (UNCAPTIONED_CROP_KIND, UNCAPTIONED_CROP_LOST_KIND):
        assert kind in EVENT_KINDS and kind in MARKED_KINDS
        assert kind in UnifiedPipeline.resume_restore_kinds()
    assert UNCAPTIONED_CROP_KIND in SHIPPED_KINDS
    assert UNCAPTIONED_CROP_LOST_KIND not in SHIPPED_KINDS


def test_the_mark_says_a_crop_is_not_a_reading() -> None:
    crop = clause_records(1, (CHART_SKELETON_LABEL,), UNCAPTIONED_CROP_KIND)[0]
    lost = clause_records(1, (CHART_SKELETON_LABEL,), UNCAPTIONED_CROP_LOST_KIND)[0]
    assert crop["check"] == Check.FIGURE_TEXT.value
    assert crop["result"] == CheckResult.UNKNOWN.value, "never a pass"
    assert "not a reading" in crop["reason"] and "names it" not in crop["reason"]
    assert "preserved nowhere" in lost["reason"] and crop["reason"] != lost["reason"]


def test_the_document_note_words_these_pages_apart() -> None:
    note = document_note([4], [4, 5, 6], uncaptioned_shipped=[5], uncaptioned_lost=[6])
    assert "page(s) 4 ship a crop" in note and "page(s) 4 name a figure" in note
    assert "page(s) 5 name no figure" in note and "page(s) 6 name no figure" in note
    assert "4, 5" not in note and "5, 6" not in note
    assert ";" not in note


def test_the_crop_mark_demotes_a_success_page_and_keeps_its_text() -> None:
    out = _out(BODY)
    marked, verdict = manifest._apply_verdict_marks(
        out,
        PageState(
            page_num=1,
            unshipped_figure_unlocated=[CHART_SKELETON_LABEL],
            unshipped_figure_kind=UNCAPTIONED_CROP_KIND,
        ),
        1,
    )
    assert [c.result for c in verdict.clauses if c.check is Check.FIGURE_TEXT] == [
        CheckResult.UNKNOWN
    ]
    assert marked.status is PageStatus.WARNING
    assert marked.failure_mode is FailureMode.FIGURE_NOT_SHOWN
    assert out.text in marked.text


# ---------------------------------------------------------------------------
# The live run, through the real process()
# ---------------------------------------------------------------------------


def test_process_ships_a_crop_and_marks_and_the_same_page_without_a_figure_is_untouched(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(_Engine, "BODY", BODY)
    with_fig, with_text, with_meta, with_out = _process(
        tmp_path, monkeypatch, "fig", draw_figure=True
    )
    without, no_text, no_meta, _ = _process(tmp_path, monkeypatch, "pro", draw_figure=False)

    assert with_fig["engine"] == without["engine"] == "qwen", "a MODEL-read page"
    assert with_fig["unshipped_figure_kind"] == UNCAPTIONED_CROP_KIND
    ref = with_fig["unshipped_figure_refs"][0]
    assert f"]({ref})" in with_text
    assert (next(iter(with_out.rglob("figures"))) / Path(ref).name).is_file()
    assert BODY in with_text, "the model's own reading is untouched"
    assert leading_marks(with_text, 1), "a crop never turns the page clean"
    assert with_fig["status"] == "warning"
    assert with_fig["failure_mode"] == "figure_not_shown"
    assert "name no figure" in (with_meta.get("error") or "")
    assert "name a figure" not in (with_meta.get("error") or "")
    assert with_meta.get("status") != "completed"

    assert not without.get("unshipped_figure_kind"), "no region, no answer: unchanged"
    assert without["unshipped_figure_refs"] == []
    assert "![" not in no_text
    assert "name no figure" not in (no_meta.get("error") or "")


def test_a_crop_never_removes_another_mark(tmp_path: Path, monkeypatch) -> None:
    """The page's own numbers are unchecked in BOTH arms; the crop adds a mark, removes none."""
    monkeypatch.setattr(_Engine, "BODY", BODY)
    with_fig, with_text, _m, _o = _process(tmp_path, monkeypatch, "a", draw_figure=True)
    without, no_text, _m2, _o2 = _process(tmp_path, monkeypatch, "b", draw_figure=False)
    for text in (with_text, no_text):
        assert "text_layer_numbers" in text or "unchecked" in text or leading_marks(text, 1)
    have = {c["check"] for c in with_fig["verdict"]["clauses"]}
    lack = {c["check"] for c in without["verdict"]["clauses"]}
    # ``page_judge`` is the judge-only clause, which by #1251's design a live run gives only to
    # a page nothing else marks (same for #1187/#1225): it is superseded by any lane's clause.
    assert lack - {"page_judge"} <= have, "every clause the plain page carries survives the crop"
    assert Check.FIGURE_TEXT.value in have - lack


def test_the_resumed_page_is_byte_identical_to_the_fresh_one(tmp_path: Path, monkeypatch) -> None:
    from socr.core.verdict import PageVerdict

    monkeypatch.setattr(_Engine, "BODY", BODY)
    side, text, _meta, _out_dir = _process(tmp_path, monkeypatch, "res", draw_figure=True)
    shipped = PageOutput.from_dict(side["winning_output"])
    resumed = PageState(page_num=1)
    for name in (
        "unshipped_figure_refs",
        "unshipped_figure_captions",
        "unshipped_figure_unlocated",
    ):
        setattr(resumed, name, list(side[name]))
    resumed.unshipped_figure_kind = side["unshipped_figure_kind"]
    again = manifest._apply_unshipped_figure_guard(shipped, resumed)
    again, _ = manifest._apply_verdict_marks(
        again, resumed, 1, published=PageVerdict.from_dict(side["verdict"])
    )
    assert again.text == text


def test_the_cli_reports_the_page(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(_Engine, "BODY", BODY)
    _process(tmp_path, monkeypatch, "cli", draw_figure=True, quiet=False)
    printed = " ".join(capsys.readouterr().out.split())
    assert "name no figure and show none: a crop of the located region ships" in printed
    assert "named a figure" not in printed


# ---------------------------------------------------------------------------
# The re-finalize tool
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("figure", [True, False], ids=["figure", "no-figure"])
def test_the_tool_crops_and_marks_and_a_second_run_changes_nothing(
    tmp_path: Path, figure: bool
) -> None:
    pdf = _pdf_with_a_figure(tmp_path / "paper.pdf", draw_figure=figure)
    out, doc_dir = _run_dir(tmp_path, pdf, BODY)
    before = _tree(out)

    assert len(tool.refinalize_document(doc_dir, dry_run=True, pdf_dir=tmp_path)) == int(figure)
    assert _tree(out) == before, "a dry run writes nothing"

    changes = tool.refinalize_document(doc_dir, pdf_dir=tmp_path)
    assert len(changes) == int(figure)
    if not figure:
        assert _tree(out) == before
        return
    assert changes[0].kind == UNCAPTIONED_CROP_KIND and changes[0].marked
    side = json.loads((doc_dir / "pages" / "00001.json").read_text())
    assert side["unshipped_figure_kind"] == UNCAPTIONED_CROP_KIND
    assert side["unshipped_figure_refs"] and side["status"] == "warning"
    meta = (
        json.loads((out / "metadata.json").read_text()) if (out / "metadata.json").exists() else {}
    )
    note = json.dumps(meta)
    assert "name a figure" not in note
    after = _tree(out)
    assert tool.refinalize_document(doc_dir, pdf_dir=tmp_path) == []
    assert _tree(out) == after, "idempotent"


# ---------------------------------------------------------------------------
# A page the chart-region lane owns is left to it
# ---------------------------------------------------------------------------


def test_decide_abstains_on_a_page_the_chart_region_lane_owns() -> None:
    owned = decide(BODY, lambda: _REGION, lambda b: ["f.png"], chart_region_owns=True)
    free = decide(BODY, lambda: _REGION, lambda b: ["f.png"])
    assert owned is None and free is not None and free.kind == UNCAPTIONED_CROP_KIND


def test_the_live_gate_abstains_on_an_arbitrated_page_and_acts_on_the_same_page_otherwise(
    tmp_path: Path,
) -> None:
    from socr.core.audit_log import AuditEvent
    from socr.core.document import DocumentHandle
    from socr.core.state import DocumentState
    from socr.figures.unshipped_figures import CHART_TABLE_ARBITRATION_KIND

    pdf = _pdf_with_a_figure(tmp_path / "p.pdf", draw_figure=True)
    kinds = []
    for arbitrated in (False, True):
        state = DocumentState(handle=DocumentHandle(path=pdf, page_count=1))
        page = PageState(page_num=1)
        state.pages[1] = page
        if arbitrated:
            state.events.append(
                AuditEvent(page_num=1, kind=CHART_TABLE_ARBITRATION_KIND, engine="", detail="")
            )
        pipe = UnifiedPipeline.__new__(UnifiedPipeline)
        pipe.config = type("C", (), {"figures_max_per_page": 3, "save_figures": False})()
        pipe._agentic_unshipped_figure_page(state, 1, page, _out(BODY), tmp_path / "figs")
        kinds.append(page.unshipped_figure_kind)
    assert kinds == [UNCAPTIONED_CROP_KIND, ""]


def test_the_live_gate_abstains_on_a_table_page_and_acts_on_the_same_page_without_one(
    tmp_path: Path,
) -> None:
    """The chart-region lane's candidate set is the pages with a table signal: a displaced
    inline crop is the regression this pins (Fed p24/p31 swapped theirs for an appended one)."""
    from socr.core.document import DocumentHandle
    from socr.core.state import DocumentState

    pdf = _pdf_with_a_figure(tmp_path / "t.pdf", draw_figure=True)
    kinds = []
    for has_tables in (False, True):
        state = DocumentState(handle=DocumentHandle(path=pdf, page_count=1))
        page = PageState(page_num=1)
        page.has_tables = has_tables
        state.pages[1] = page
        pipe = UnifiedPipeline.__new__(UnifiedPipeline)
        pipe.config = type("C", (), {"figures_max_per_page": 3, "save_figures": False})()
        pipe._agentic_unshipped_figure_page(state, 1, page, _out(BODY), tmp_path / "figs")
        kinds.append(page.unshipped_figure_kind)
    assert kinds == [UNCAPTIONED_CROP_KIND, ""]


def test_the_live_gate_abstains_when_save_figures_runs_the_figure_phase(tmp_path: Path) -> None:
    from socr.core.document import DocumentHandle
    from socr.core.state import DocumentState

    pdf = _pdf_with_a_figure(tmp_path / "s.pdf", draw_figure=True)
    kinds = []
    for save in (False, True):
        state = DocumentState(handle=DocumentHandle(path=pdf, page_count=1))
        page = PageState(page_num=1)
        state.pages[1] = page
        pipe = UnifiedPipeline.__new__(UnifiedPipeline)
        pipe.config = type("C", (), {"figures_max_per_page": 3, "save_figures": save})()
        pipe._agentic_unshipped_figure_page(state, 1, page, _out(BODY), tmp_path / "figs")
        kinds.append(page.unshipped_figure_kind)
    assert kinds == [UNCAPTIONED_CROP_KIND, ""]
