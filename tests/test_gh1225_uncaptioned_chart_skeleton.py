"""#1225: a chart a model-read page shows and NAMES nowhere is marked, not lost.

A VLM that drops a plot keeps its panel titles and tick labels. On the Fed calibration copy
(greenbook-part2 2004-09 p12/p50/p52/p54) the page then counts CHECKED with no figure caption,
so #1187's lane -- which needs a caption -- never sees it. The tick labels are the evidence:
``1996 1998 2000 2002 2004`` on one line, or ``18 / 16 / 14 ...`` down a column.

Hermetic: no provider, no model. The ``process()`` tests stub the engine, the judge and the
ladder and pin the engines, exactly as #1187's do.

Differences, not absolutes (CLAUDE.md, #257): each behavioural test runs one page twice,
changing only whether its body carries the tick lines.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_gh1187_unshipped_figure_pages import (
    _Engine,
    _out,
    _p,
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
    CHART_SKELETON_KIND,
    CHART_SKELETON_LABEL,
    EVENT_KINDS,
    MARKED_KINDS,
    MIN_TICK_RUN,
    UNLOCATED_KIND,
    clause_records,
    decide,
    document_note,
    tick_run,
)
from socr.pipeline import refinalize_unshipped_figures as tool
from socr.pipeline.orchestrator import UnifiedPipeline

#: Shipped text of Fed greenbook-part2 2004-09 p52, first panel, as qwen read it: the title,
#: the unit, the frequency, the year axis and the value axis -- and nothing plotted.
_TITLES = "## Growth of Commercial Mortgage Debt\n\nPercent\n\nQuarterly, s.a.a.r."
_TICKS = "1996 1998 2000 2002 2004\n\nQ2\n\n18\n16\n14\n12\n10"
_SOURCE = "Source. Commercial Mortgage Alert."
SKELETON = f"{_TITLES}\n\n{_TICKS}\n\n{_SOURCE}"
#: The same page without its tick lines: the difference every behavioural test turns on.
NO_TICKS = f"{_TITLES}\n\nQ2\n\n{_SOURCE}"


def test_loaded_source_is_this_checkout() -> None:
    import socr

    assert Path(socr.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1] / "src")


def test_the_fixture_differs_only_in_its_ticks() -> None:
    assert tick_run(SKELETON) >= MIN_TICK_RUN > tick_run(NO_TICKS)


# ---------------------------------------------------------------------------
# The signal
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("body", "run"),
    [
        ("1996 1998 2000 2002 2004", 5),
        ("2000 2001 2002 2003 2004", 5),
        ("18\n16\n14\n12\n10\n8", 6),
        ("4\n\n3\n\n2\n\n1\n\n0", 5),  # a VLM's blank lines do not break an axis
        ("-0.4 -0.2 0.0 0.2 0.4", 5),
        ("1,500 1,600 1,700", 3),
        # Argon, PR #1239: the three spellings a VLM uses on an axis, each pinned.
        ("−0.4 −0.2 0.0 0.2 0.4", 5),  # U+2212 MINUS SIGN
        ("–2 –1 0 1 2", 5),  # U+2013 EN DASH as a minus
        ("10% 20% 30% 40%", 4),
    ],
    ids=[
        "years-2",
        "years-1",
        "column",
        "blank-separated",
        "signed",
        "thousands",
        "unicode-minus",
        "en-dash-minus",
        "percent",
    ],
)
def test_an_axis_is_a_run_of_evenly_spaced_numbers(body: str, run: int) -> None:
    assert tick_run(body) == run


@pytest.mark.parametrize(
    "body",
    [
        "| 2019 | 2020 | 2021 | 2022 |",
        "2019 | 2020 | 2021 | 2022",
        "2019 & 2020 & 2021 & 2022 \\\\",
        "1. first\n2. second\n3. third\n4. fourth",
        "- 1996 1998 2000 2002",
        "GDP 2.1 3.4 1.9 2.8",
        "2.1 3.4 1.9 2.8",
        "In 1996, 1998, 2000 and 2002 the rate rose.",
        "1996 1998",
        "1996\n1998\nFederal funds\n2000\n2002",
        "0 0 0 0",
        "5\n5\n5\n5",
        "10\n20\n```\nx = 1\n```\n30",
        "10\n20\n<pre>\nx = 1\n</pre>\n30",
    ],
    ids=[
        "pipe-table",
        "gfm-table",
        "latex-row",
        "numbered-list",
        "bullet",
        "labelled-values",
        "values",
        "prose",
        "two-terms",
        "column-broken-by-text",
        "repeated-row",
        "repeated-column",
        "column-broken-by-code-block",
        "column-broken-by-pre-block",
    ],
)
def test_content_that_ships_its_numbers_is_not_an_axis(body: str) -> None:
    assert tick_run(body) < MIN_TICK_RUN


@pytest.mark.parametrize("fence", ["```", "~~~", "<pre>"])
def test_a_code_block_is_not_an_axis_and_the_same_lines_outside_one_are(fence: str) -> None:
    lines = "10\n20\n30\n40"
    close = "</pre>" if fence == "<pre>" else fence
    assert tick_run(lines) == 4
    assert tick_run(f"{fence}\n{lines}\n{close}") < MIN_TICK_RUN
    assert tick_run(f"{fence}\n1\n{close}\n\n{lines}") == 4, "the block ends at its close"


@pytest.mark.parametrize(
    "opener", ["<pre>x = 1</pre>", "<prefix>"], ids=["one-line-pre", "not-a-pre-tag"]
)
def test_a_line_that_opens_no_block_does_not_blind_the_rest(opener: str) -> None:
    """A one-line ``<pre>…</pre>`` and a tag that merely starts with ``pre`` open no block, so
    the axis after them still counts (advisor, diff review: both used to hide the page)."""
    assert tick_run(f"{opener}\n10\n20\n30") == 3


@pytest.mark.parametrize(
    "block",
    [
        "```\n<pre>\n</pre>\n10\n20\n30\n```",
        "```\n<pre>\n10\n20\n30\n```",
        "```\n10\n20\n~~~\n30\n```",
        "<pre>\n```\n10\n20\n30\n</pre>",
    ],
    ids=[
        "pre-inside-fence",
        "unclosed-pre-inside-fence",
        "other-fence-inside-fence",
        "fence-inside-pre",
    ],
)
def test_a_block_ends_only_at_its_own_close(block: str) -> None:
    """Argon, PR #1239: one ``in_block`` flag let a ``</pre>`` inside a fence close it, and the
    fence's own close then reopened a block for the rest of the page. Inside the block nothing
    counts, and after the block's OWN close an axis counts again."""
    assert tick_run(block) < MIN_TICK_RUN
    assert tick_run(f"{block}\n\n40\n50\n60") == 3


def test_min_tick_run_is_the_first_term_evenness_constrains() -> None:
    """Any two numbers are evenly spaced. The third is the first that can fail to be."""
    assert tick_run("1996 1998") == 2 < MIN_TICK_RUN
    assert tick_run("1996 1998 2000") == MIN_TICK_RUN
    assert tick_run("1996 1998 2001") == 2


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------


def _never_locate():
    raise AssertionError("the chart-skeleton branch must not call the locator")


def test_decide_difference_ticks_vs_none() -> None:
    assert decide(NO_TICKS, _never_locate, lambda b: []) is None
    decision = decide(SKELETON, _never_locate, lambda b: [])
    assert decision is not None
    assert decision.kind == CHART_SKELETON_KIND
    assert decision.unlocated == (CHART_SKELETON_LABEL,), "the mark names what did not ship"
    assert decision.refs == () and decision.regions == 0, "a mark, never a crop"
    assert decision.marks


def test_a_captioned_page_still_takes_the_1187_branch() -> None:
    """The ticks do not override a caption: #1187 already answers a page that names a figure,
    and on the six captioned firings measured it did (three crops, three marks)."""
    body = f"Figure 7: Bias and MSE.\n\n{SKELETON}"
    decision = decide(body, lambda: [], lambda b: [])
    assert decision is not None
    assert decision.kind == UNLOCATED_KIND
    assert decision.unlocated == ("Figure 7",)


def test_a_cross_reference_label_with_ticks_is_marked_not_dropped() -> None:
    """ "Chart 3 Selected Treasury yields" is LOOSE-only: #1187 owes nothing on it when no region
    is found (NOT_FOUND, unmarked). The ticks are evidence of their own, so with them the page
    is marked -- the difference is the tick lines and nothing else (advisor, diff review)."""
    label = "Chart 3 Selected Treasury yields"
    plain = decide(f"{label}\n\n{NO_TICKS}", lambda: [], lambda b: [])
    ticked = decide(f"{label}\n\n{SKELETON}", lambda: [], lambda b: [])
    assert plain is not None and plain.kind == "unshipped_figure_not_found" and not plain.marks
    assert ticked is not None and ticked.kind == CHART_SKELETON_KIND and ticked.marks
    assert ticked.captions == ("Chart 3",), "the label the page carries is still recorded"
    assert ticked.unlocated == (CHART_SKELETON_LABEL,)


def test_the_kind_is_a_marked_and_replayed_kind() -> None:
    assert CHART_SKELETON_KIND in EVENT_KINDS
    assert CHART_SKELETON_KIND in MARKED_KINDS
    assert CHART_SKELETON_KIND in UnifiedPipeline.resume_restore_kinds()


def test_the_mark_says_ticks_without_values_not_a_named_figure() -> None:
    reason = clause_records(1, (CHART_SKELETON_LABEL,), CHART_SKELETON_KIND)[0]["reason"]
    other = clause_records(1, (CHART_SKELETON_LABEL,), UNLOCATED_KIND)[0]["reason"]
    assert reason != other
    assert reason.startswith(CHART_SKELETON_LABEL)
    assert "axis ticks" in reason and "names it" not in reason


def test_the_document_note_words_a_skeleton_page_apart() -> None:
    plain = document_note([], [3])
    split = document_note([], [3, 7], [7])
    assert "page(s) 3 name a figure" in plain
    assert "page(s) 3 name a figure" in split, "the named-figure page keeps its sentence"
    assert "page(s) 7 carry a chart's axis ticks" in split
    assert "7 name a figure" not in split and "3, 7" not in split
    assert ";" not in split, "a note with ';' cannot be found and replaced as a unit"


def test_a_skeleton_mark_demotes_a_success_page_with_this_lanes_mode() -> None:
    out = _out(SKELETON)
    _clean, clean_verdict = manifest._apply_verdict_marks(out, _p([]), 1)
    marked, verdict = manifest._apply_verdict_marks(
        out,
        _p(
            [],
            unshipped_figure_unlocated=[CHART_SKELETON_LABEL],
            unshipped_figure_kind=CHART_SKELETON_KIND,
        ),
        1,
    )
    figure = [c for c in verdict.clauses if c.check is Check.FIGURE_TEXT]
    assert [c.result for c in figure] == [CheckResult.UNKNOWN]
    assert not [c for c in clean_verdict.clauses if c.check is Check.FIGURE_TEXT], (
        "the same page without this lane's record carries no figure clause"
    )
    assert marked.status is PageStatus.WARNING
    assert marked.failure_mode is FailureMode.FIGURE_NOT_SHOWN, (
        "deterministic: a re-read gives the same body, so this must not be retried on resume"
    )
    assert CHART_SKELETON_LABEL in marked.text.split("\n")[0]
    assert out.text in marked.text, "nothing of the page's own text is removed"


# ---------------------------------------------------------------------------
# The re-finalize tool, on a finished run
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ticks", [True, False], ids=["skeleton", "no-ticks"])
def test_the_tool_marks_a_skeleton_page_and_a_second_run_changes_nothing(
    tmp_path: Path, ticks: bool
) -> None:
    pdf = _pdf_with_a_figure(tmp_path / "paper.pdf", draw_figure=False)
    body = SKELETON if ticks else NO_TICKS
    out, doc_dir = _run_dir(tmp_path, pdf, body)
    before = _tree(out)

    assert len(tool.refinalize_document(doc_dir, dry_run=True, pdf_dir=tmp_path)) == int(ticks)
    assert _tree(out) == before, "a dry run writes nothing"

    changes = tool.refinalize_document(doc_dir, pdf_dir=tmp_path)
    if not ticks:
        assert changes == [] and _tree(out) == before, "no ticks, no change at all"
        return
    assert [c.kind for c in changes] == [CHART_SKELETON_KIND]
    assert changes[0].crops == 0 and "axis ticks" in changes[0].outcome
    after = _tree(out)
    fragment = (doc_dir / "pages" / "00001.md").read_text(encoding="utf-8")
    assert leading_marks(fragment, 1)
    assert body in fragment
    assert fragment in (doc_dir / f"{pdf.stem}.md").read_text(encoding="utf-8")
    assert not list(doc_dir.rglob("*.png")), "a mark, never a crop"
    sidecar = json.loads((doc_dir / "pages" / "00001.json").read_text())
    assert sidecar["unshipped_figure_kind"] == CHART_SKELETON_KIND
    assert sidecar["unshipped_figure_unlocated"] == [CHART_SKELETON_LABEL]
    assert sidecar["status"] == "warning"
    assert sidecar["failure_mode"] == FailureMode.FIGURE_NOT_SHOWN.value
    meta = json.loads((doc_dir / "metadata.json").read_text())
    assert "page(s) 1 carry a chart's axis ticks" in meta["error"]
    audit = json.loads((doc_dir / "audit_log.json").read_text())
    assert [e["kind"] for e in audit["events"]] == [CHART_SKELETON_KIND]

    assert tool.refinalize_document(doc_dir, pdf_dir=tmp_path) == []
    assert _tree(out) == after, "a second run changes 0 bytes"


_XREF = "Chart 3 Selected Treasury yields"


def _record_not_found(doc_dir: Path) -> None:
    """Give the page the record a run finished under #1187 writes for a cross-reference-shaped
    label with no region: ``state_fields`` of a ``NOT_FOUND`` decision, and its event in the
    sidecar and the document audit log (``_agentic_unshipped_figure_page`` +
    ``_flush_page_sidecar``)."""
    event = {
        "kind": "unshipped_figure_not_found",
        "engine": "",
        "detail": "...",
        "data": {"captions": ["Chart 3"], "regions": 0, "rendered": 0, "unlocated": []},
    }
    side_path = doc_dir / "pages" / "00001.json"
    side = json.loads(side_path.read_text())
    side.update(
        unshipped_figure_refs=[],
        unshipped_figure_captions=["Chart 3"],
        unshipped_figure_unlocated=[],
        unshipped_figure_kind="unshipped_figure_not_found",
        audit_events=[event],
    )
    side_path.write_text(json.dumps(side, indent=2), encoding="utf-8")
    audit_path = doc_dir / "audit_log.json"
    audit_path.write_text(
        json.dumps(
            {
                "events": [{"page_num": 1, **event}],
                "event_count": 1,
                "counts": {event["kind"]: 1},
            },
            indent=2,
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize("ticks", [True, False], ids=["skeleton", "no-ticks"])
def test_the_tool_re_decides_a_page_a_finished_run_answered_not_found(
    tmp_path: Path, ticks: bool
) -> None:
    """Bot, PR #1239: a run finished under #1187 records ``NOT_FOUND`` on a
    cross-reference-shaped label, and the tool skipped every answered page -- so the label +
    ticks branch never reached the finished Fed run. ``NOT_FOUND`` shipped and marked nothing,
    so the tool now decides it again: with ticks the page is marked, without them nothing at
    all changes, its event included."""
    pdf = _pdf_with_a_figure(tmp_path / "paper.pdf", draw_figure=False)
    body = f"{_XREF}\n\n{SKELETON if ticks else NO_TICKS}"
    out, doc_dir = _run_dir(tmp_path, pdf, body)
    _record_not_found(doc_dir)
    before = _tree(out)

    changes = tool.refinalize_document(doc_dir, pdf_dir=tmp_path)
    if not ticks:
        assert changes == [] and _tree(out) == before, "the same answer changes 0 bytes"
        return
    assert [c.kind for c in changes] == [CHART_SKELETON_KIND]
    sidecar = json.loads((doc_dir / "pages" / "00001.json").read_text())
    assert sidecar["unshipped_figure_kind"] == CHART_SKELETON_KIND
    assert sidecar["unshipped_figure_captions"] == ["Chart 3"]
    assert sidecar["status"] == "warning"
    assert [e["kind"] for e in sidecar["audit_events"]] == [CHART_SKELETON_KIND]
    audit = json.loads((doc_dir / "audit_log.json").read_text())
    assert [e["kind"] for e in audit["events"]] == [CHART_SKELETON_KIND], (
        "one page, one answer: the superseded NOT_FOUND event is gone"
    )
    assert leading_marks((doc_dir / "pages" / "00001.md").read_text(encoding="utf-8"), 1)
    after = _tree(out)
    assert tool.refinalize_document(doc_dir, pdf_dir=tmp_path) == []
    assert _tree(out) == after, "a second run changes 0 bytes"


def test_the_live_gate_re_decides_not_found_and_acts_only_on_a_new_answer(
    tmp_path: Path,
) -> None:
    """The live lane's half of the same fix. A page state carrying ``NOT_FOUND`` -- what this
    lane itself writes -- is decided again; the same answer appends nothing (#1187's
    answered-once rule), a new one replaces the old event."""
    from socr.core.config import PipelineConfig
    from socr.core.document import DocumentHandle
    from socr.core.state import DocumentState

    pdf = _pdf_with_a_figure(tmp_path / "paper.pdf", draw_figure=False)
    pipe = UnifiedPipeline.__new__(UnifiedPipeline)
    pipe.config = PipelineConfig(quiet=True)

    def run(body: str) -> tuple[PageState, list, list]:
        state = DocumentState(handle=DocumentHandle(path=pdf, page_count=1))
        p = PageState(page_num=1)
        state.pages[1] = p
        pipe._agentic_unshipped_figure_page(state, 1, p, _out(body), tmp_path / "figures")
        assert p.unshipped_figure_kind == "unshipped_figure_not_found", "fixture premise"
        first = list(state.events)
        pipe._agentic_unshipped_figure_page(state, 1, p, _out(body), tmp_path / "figures")
        return p, first, list(state.events)

    _p_plain, first, again = run(f"{_XREF}\n\n{NO_TICKS}")
    assert len(again) == len(first) == 1
    # IDENTITY, not equality: deleting the event and appending an equal one also compares
    # equal, and that was how the mutation check found this assertion vacuous.
    assert again[0] is first[0], "the same answer twice touches nothing"

    # A page answered NOT_FOUND, then passed over again with a different body -- what a
    # ``--retry-failed-pages`` re-read hands this lane in the same process.
    state = DocumentState(handle=DocumentHandle(path=pdf, page_count=1))
    p = PageState(page_num=1)
    state.pages[1] = p
    pipe._agentic_unshipped_figure_page(state, 1, p, _out(f"{_XREF}\n\n{NO_TICKS}"), None)
    assert [e.kind for e in state.events] == ["unshipped_figure_not_found"]
    pipe._agentic_unshipped_figure_page(state, 1, p, _out(f"{_XREF}\n\n{SKELETON}"), None)
    assert p.unshipped_figure_kind == CHART_SKELETON_KIND
    assert p.unshipped_figure_unlocated == [CHART_SKELETON_LABEL]
    assert [e.kind for e in state.events] == [CHART_SKELETON_KIND], "one page, one answer"


def test_refinalize_then_resume_is_byte_identical(tmp_path: Path) -> None:
    """A resumed page is restored from its sidecar and re-finalized; nothing may move."""
    from socr.core.verdict import PageVerdict

    pdf = _pdf_with_a_figure(tmp_path / "paper.pdf", draw_figure=False)
    _out_dir, doc_dir = _run_dir(tmp_path, pdf, SKELETON)
    assert tool.refinalize_document(doc_dir, pdf_dir=tmp_path)

    sidecar = json.loads((doc_dir / "pages" / "00001.json").read_text())
    shipped = PageOutput.from_dict(sidecar["winning_output"])
    resumed = PageState(page_num=1)
    for name in (
        "unshipped_figure_refs",
        "unshipped_figure_captions",
        "unshipped_figure_unlocated",
    ):
        setattr(resumed, name, list(sidecar[name]))
    resumed.unshipped_figure_kind = sidecar["unshipped_figure_kind"]
    again = manifest._apply_unshipped_figure_guard(shipped, resumed)
    again, _ = manifest._apply_verdict_marks(
        again, resumed, 1, published=PageVerdict.from_dict(sidecar["verdict"])
    )
    assert again.text == shipped.text
    assert again.status is PageStatus(sidecar["status"])
    assert again.text == (doc_dir / "pages" / "00001.md").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# The live run, through the real process()
# ---------------------------------------------------------------------------


def _pdf_holding(path: Path, body: str) -> Path:
    """A one-page PDF whose text layer holds every line of *body*.

    So #1185's lane, which marks a model-read number the layer does not hold, stays quiet on
    BOTH arms -- without this the skeleton arm carries a second, independent mark, and the
    document-status difference below could not be attributed to this lane (mutation check:
    taking this kind out of ``MARKED_KINDS`` survived every behavioural test).
    """
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    y = 72
    for line in body.splitlines():
        if line.strip():
            page.insert_text((72, y), line.lstrip("# "), fontname="helv", fontsize=10)
            y += 14
    doc.save(path)
    doc.close()
    return path


def test_process_marks_a_skeleton_page_and_the_same_page_without_ticks_is_clean(
    tmp_path: Path, monkeypatch
) -> None:
    # ONE PDF for both arms, holding the skeleton's lines: the only difference is what the
    # model reads off it.
    pdf = _pdf_holding(tmp_path / "page.pdf", SKELETON)
    monkeypatch.setattr(_Engine, "BODY", SKELETON)
    skel, skel_text, skel_meta, _ = _process(tmp_path, monkeypatch, "skel", pdf=pdf)
    monkeypatch.setattr(_Engine, "BODY", NO_TICKS)
    plain, plain_text, plain_meta, _ = _process(tmp_path, monkeypatch, "plain", pdf=pdf)

    assert skel["engine"] == plain["engine"] == "qwen", "a MODEL-read page, the lane's scope"
    assert skel["unshipped_figure_kind"] == CHART_SKELETON_KIND
    assert skel["unshipped_figure_unlocated"] == [CHART_SKELETON_LABEL]
    assert skel["unshipped_figure_refs"] == [], "a mark, never a crop"
    assert leading_marks(skel_text, 1)
    assert SKELETON in skel_text
    assert skel["status"] == "warning"
    assert "carry a chart's axis ticks" in (skel_meta.get("error") or "")
    # Bot, PR #1239: the live note goes through ``_unshipped_figure_note``, not the tool's
    # ``_lane_pages``; a skeleton page must never be told it names a figure there either.
    assert "name a figure" not in (skel_meta.get("error") or "")
    assert "text layer" not in (skel_meta.get("error") or ""), (
        "the fixture's premise: no #1185 mark, so the status below turns on this lane alone"
    )

    assert not plain.get("unshipped_figure_kind")
    assert CHART_SKELETON_LABEL not in plain_text
    assert skel_meta.get("status") != plain_meta.get("status"), (
        "a document whose only defect is the dropped chart must not be clean"
    )
    assert not plain_meta.get("error"), "the control document is clean"


def test_the_cli_reports_a_skeleton_page(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(_Engine, "BODY", SKELETON)
    _process(
        tmp_path, monkeypatch, "cli", quiet=False, pdf=_pdf_holding(tmp_path / "c.pdf", SKELETON)
    )
    printed = " ".join(capsys.readouterr().out.split())  # the console wraps long lines
    # The CLI line's own wording. The document note (also printed) says "carry a chart's axis
    # ticks" too, so asserting that phrase passed with the CLI line deleted (mutation check).
    assert "with no picture of it; the page is marked: [1]" in printed, printed[-1500:]
    assert "named a figure" not in printed, "a skeleton page names no figure"
