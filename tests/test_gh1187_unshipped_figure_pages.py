"""#1187: a figure a page NAMES never ships as a caption with nothing under it.

Measured on finished run 23319732 (395 PDFs, 17,091 pages): of the 3,082 model-read pages the
meter counts CHECKED, 2,751 ship no picture, and 147 of those name a figure in their own
shipped body -- 132 where ``FigureExtractor`` locates a region (a crop can ship) and 15 where
it does not (only a mark can). 147/3,082 is 4.8%, inside the issue's 5.1% estimate.

Hermetic: nothing here needs a provider. The two end-to-end tests drive the finalisation
guard, the verdict seam and the re-finalize tool, none of which calls a model, and the only
PDFs are built in-process by fitz.

Differences, not absolutes (CLAUDE.md, #257): every behavioural test runs the same input
twice in one process, changing exactly one thing, and asserts the two outcomes differ exactly
as intended -- or, for the tool, that a second run changes nothing at all.
"""

from __future__ import annotations

import json
from pathlib import Path

import fitz
import pytest

from socr.core import manifest
from socr.core.marks import Check, CheckResult, leading_marks, line_is_mark
from socr.core.result import FailureMode, PageOutput, PageStatus
from socr.core.state import PageState
from socr.core.verdict import build_page_verdict
from socr.figures.unshipped_figures import (
    RENDER_FAILED_KIND,
    RENDERED_KIND,
    UNLOCATED_KIND,
    clause_records,
    crop_filename,
    figure_block,
    figure_captions,
)
from socr.pipeline import refinalize_unshipped_figures as tool
from socr.pipeline.orchestrator import UNSHIPPED_FIGURE_EVENT_KINDS, UnifiedPipeline


def _figure_clauses(verdict) -> list:
    """This lane's clauses only.

    After the #1185 merge a stub ``PageState`` has no own text layer, so
    ``text_layer_numbers_clause`` fires on one too. These tests are about THIS lane, and the
    two lanes are independent: a figure a page does not show and numbers its layer disputes
    are different facts and neither undoes the other. Asserting an EMPTY clause list would
    now be asserting that the other lane is switched off.
    """
    return [c for c in verdict.clauses if c.check is Check.FIGURE_TEXT]


def test_loaded_source_is_this_checkout() -> None:
    import socr

    assert Path(socr.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1] / "src")


# ---------------------------------------------------------------------------
# The matcher. Every line below is a line this lane met on run 23319732 or on the
# Fed calibration run; the id says which document it came from.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("line", "label"),
    [
        ("Figure 7: Bias and MSE under a CEE-type VAR Data Generating Process", "Figure 7"),
        ("**Figure 7:** Bias and MSE under a CEE-type VAR", "Figure 7"),
        ("# Fig 3. Real-Time Current-Month Probability of Recession", "Fig 3"),
        ("Fig. 1. Responses over 120 months to one-standard deviation innovation in R3.", "Fig. 1"),
        ("FIGURE 1.--Effects of 1 SD monetary shock.", "FIGURE 1"),
        ("Figure 3.E. The distribution of participants' judgments", "Figure 3.E"),
        ("Exhibit 1", "Exhibit 1"),
        ("Chart 2", "Chart 2"),
    ],
    ids=[
        "herbst-p30",
        "markdown-emphasis",
        "gascon-p16-heading",
        "leeper-p4-100-chars",
        "kekre-lenel-p14",
        "fed-2016-06-dotted-letter",
        "fed-2013-01-p3",
        "fed-1973-11-p18",
    ],
)
def test_a_caption_is_matched_and_names_its_figure(line: str, label: str) -> None:
    assert figure_captions(f"prose above\n{line}\nprose below") == (label,)
    assert figure_captions(f"prose above\n{line}\nprose below", strict=True) == (label,)


@pytest.mark.parametrize(
    "line",
    [
        "Figure 4.5 shows contour maps of the OOS cross-sectional $R^2$",
        "Figure 5.2 illustrates this with a numerical example.",
        "Figure 3.E provides the distribution of participants' judgments regarding the rate",
        "Figures 3.A and 3.B show the distributions of participants' projections",
        "Figure 13 reports the monthly data based estimates of the dynamic response",
    ],
    ids=["nagel-p96", "nagel-p121", "fed-2016-06-15-p17", "fed-plural", "cee-p63"],
)
def test_a_prose_cross_reference_is_not_a_caption(line: str) -> None:
    """The atomic group is what makes these fail.

    Without ``(?>...)`` the regex parses the long label, fails on the tail, BACKTRACKS to the
    first number component and reads the decimal point as the caption's terminator. That is
    how ``Figure 3.E provides ...`` marked a prose page in the first run of the re-finalize
    tool on the Fed calibration copy, and how ``Figure 4.5 shows ...`` put three ``nagel``
    pages in the mark branch.
    """
    assert figure_captions(f"prose above\n{line}\nprose below") == ()
    assert figure_captions(f"prose above\n{line}\nprose below", strict=True) == ()


def test_the_loose_matcher_ships_where_the_strict_one_only_marks() -> None:
    """The branches differ by what a false fire COSTS, and that is why there are two.

    ``Fig. 4.15 2D Gaussians with (a) ...`` (dougherty p74) is a real caption with no
    terminator, and the page has a located region: the loose matcher ships its crop.
    ``Figure 1 (panel a) shows the companies' percentage composition ...`` (damato p13) has
    the same shape and is a cross-reference: the strict matcher refuses to mark on it.
    """
    caption = "Fig. 4.15 2D Gaussians with (a) $\\Sigma$"
    reference = "Figure 1 (panel a) shows the companies' percentage composition by sector"
    for line in (caption, reference):
        assert figure_captions(line) != (), line
        assert figure_captions(line, strict=True) == (), line


def test_an_alt_text_cannot_break_out_of_its_own_link() -> None:
    block = figure_block(1, ["figures/x.png"], figure_captions("Figure 1: a [bracketed] note."))
    assert "[bracketed]" not in block
    assert block.count("](figures/x.png)") == 1
    assert block.startswith("![") and block.endswith(")")


def test_an_unlocated_caption_renders_a_line_the_meter_recognises() -> None:
    (record,) = clause_records(30, ("Figure 7",), UNLOCATED_KIND)
    assert record["check"] == Check.FIGURE_TEXT.value
    assert record["result"] == CheckResult.UNKNOWN.value
    assert record["scope"] == "figure:1"
    assert "Figure 7" in record["reason"]
    from socr.figures.chart_witness import clause_mark

    line = clause_mark(30, record)
    assert line_is_mark(line, 30)
    assert leading_marks(f"{line}\n\nbody text", 30) == [line]


# ---------------------------------------------------------------------------
# The guard: same output, one field differs
# ---------------------------------------------------------------------------

REFS = ["figures/unshipped_figure_p1_1.png", "figures/unshipped_figure_p1_2.png"]


def _out(text: str, **kw) -> PageOutput:
    return PageOutput(
        page_num=1, text=text, status=PageStatus.SUCCESS, engine="qwen", audit_passed=True, **kw
    )


def _p(refs: list[str] | None = None, **kw) -> PageState:
    p = PageState(page_num=1)
    p.unshipped_figure_refs = list(refs or [])
    p.unshipped_figure_captions = ["Figure 7"]
    for k, v in kw.items():
        setattr(p, k, v)
    return p


def test_guard_difference_refs_present_vs_absent() -> None:
    out = _out("Figure 7: Bias and MSE.\n\nNote: the sub-figures show ...")
    off = manifest._apply_unshipped_figure_guard(out, _p([]))
    on = manifest._apply_unshipped_figure_guard(out, _p(REFS))
    assert off is out, "no refs, no change"
    assert on.text.startswith(out.text), "the page's own text is untouched"
    for ref in REFS:
        assert f"]({ref})" in on.text
    assert on.status is out.status and on.audit_passed is out.audit_passed
    assert manifest._apply_unshipped_figure_guard(on, _p(REFS)) is on, "idempotent"


def test_guard_abstains_when_the_body_already_shows_a_picture() -> None:
    """Inverted, this lane's own defect: two pictures of one figure."""
    out = _out("Figure 7: Bias and MSE.\n\n![Chart page 1](figures/chart_page_1.png)")
    assert manifest._apply_unshipped_figure_guard(out, _p(REFS)) is out


@pytest.mark.parametrize("floor", ["scanned_figure_png_ref", "prior_ocr_page_png_ref"])
def test_the_lane_abstains_when_a_floor_already_ships_the_page_image(floor: str) -> None:
    """The same predicate the figure extractor uses, so the two lanes skip the same pages."""
    ref = "![Scanned figure page 1](figures/scanned_figure_page_p1.png)"
    body = f"Figure 7: Bias and MSE.\n\n{ref}"
    p = PageState(page_num=1)
    setattr(p, floor, ref)
    assert manifest.page_ships_its_own_page_image(p, body)


def test_a_floor_image_the_shared_predicate_does_not_name_is_still_caught_by_the_body() -> None:
    """``d3_floor_png_ref`` is NOT in ``floor_page_image_ref``, and must not need to be.

    Two gates, deliberately: the shared predicate answers "does this page ship an image of
    ITSELF", and the body check answers "does this page ship any picture at all". A floor the
    first does not name still reaches the second, so a page cannot get a second picture
    because a list somewhere was not extended.
    """
    d3 = "![Failed table page 1](figures/d3_floor_page_p1.png)"
    p = PageState(
        page_num=1,
    )
    p.d3_floor_png_ref = d3
    body = f"[page 1 failed: unverifiable table]\n\n{d3}"
    assert not manifest.page_ships_its_own_page_image(p, body)
    out = _out(body)
    assert manifest._apply_unshipped_figure_guard(out, _p(REFS)) is out


# ---------------------------------------------------------------------------
# The verdict: an unlocated caption marks the page and demotes it
# ---------------------------------------------------------------------------


def test_verdict_difference_unlocated_caption_vs_none() -> None:
    out = _out("Figure 7: Bias and MSE.\n\nNote: ...")
    clean = build_page_verdict(page_num=1, output=out, p=_p(REFS))
    marked = build_page_verdict(
        page_num=1, output=out, p=_p([], unshipped_figure_unlocated=["Figure 7"])
    )
    assert _figure_clauses(clean) == [], "no figure clause when this lane did not mark"
    figure = _figure_clauses(marked)
    assert [c.check for c in figure] == [Check.FIGURE_TEXT]
    assert figure[0].result is CheckResult.UNKNOWN
    assert figure[0].scope == "figure:1"


def test_an_unlocated_caption_demotes_a_success_page_and_records_why() -> None:
    out = _out("Figure 7: Bias and MSE.\n\nNote: ...")
    clean, _v_clean = manifest._apply_verdict_marks(out, _p(REFS), 1)
    marked, verdict = manifest._apply_verdict_marks(
        out, _p([], unshipped_figure_unlocated=["Figure 7"]), 1
    )
    assert clean.status is PageStatus.SUCCESS or Check.TEXT_LAYER_NUMBERS in {
        c.check for c in _v_clean.clauses
    }, "the control page is demoted by the other lane only, if at all"
    assert marked.status is PageStatus.WARNING, "the status is what other consumers read"
    assert marked.failure_mode is FailureMode.FIGURE_NOT_SHOWN
    assert leading_marks(marked.text, 1), "the page is MARKED: its first block is the mark"
    assert out.text in marked.text, "nothing of the page's own text is removed"
    # Idempotent: the seam runs again over every page at assemble.
    again, _ = manifest._apply_verdict_marks(
        marked, _p([], unshipped_figure_unlocated=["Figure 7"]), 1, published=verdict
    )
    assert again.text == marked.text


def test_a_page_state_that_is_a_mock_produces_no_clause() -> None:
    """``p`` is a stand-in in much of the suite; a Mock attribute must not become a clause."""
    from unittest.mock import MagicMock, Mock

    from socr.core.verdict import _unshipped_figure_clauses

    assert _unshipped_figure_clauses(Mock()) == []
    assert _unshipped_figure_clauses(MagicMock()) == []


def test_every_lane_event_is_replayed_on_resume() -> None:
    assert UNSHIPPED_FIGURE_EVENT_KINDS <= UnifiedPipeline.resume_restore_kinds()


# ---------------------------------------------------------------------------
# End to end: a run on disk, re-finalized
# ---------------------------------------------------------------------------

_CAPTION = "Figure 7: Bias and MSE under a CEE-type VAR"
_PROSE = "The estimated effect of the policy change on output is reported above."


def _pdf_with_a_figure(path: Path, *, draw_figure: bool) -> Path:
    """One page: a caption, prose, and (optionally) a vector chart the locator can find.

    The chart is coloured filled bars plus an axes frame -- enough drawings, enough area,
    and coloured fills, so ``_has_vector_data_marks`` fires and ``_looks_like_table_grid``
    does not. ``draw_figure=False`` keeps the identical text and removes only the marks,
    which is the one thing that differs between the two arms of the end-to-end test.
    """
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 72), _CAPTION, fontname="helv", fontsize=11)
    y = 600
    for _ in range(6):
        page.insert_text((72, y), _PROSE + " ", fontname="helv", fontsize=10)
        y += 14
    if draw_figure:
        shape = page.new_shape()
        shape.draw_rect(fitz.Rect(100, 120, 500, 420))
        shape.finish(color=(0, 0, 0), width=1.5)
        for i in range(8):
            x = 120 + i * 45
            shape.draw_rect(fitz.Rect(x, 400 - (i + 1) * 30, x + 30, 410))
            shape.finish(color=(0.8, 0.1, 0.1), fill=(0.8, 0.1, 0.1), width=1.5)
        shape.commit()
    doc.save(path)
    doc.close()
    return path


def _run_dir(tmp_path: Path, pdf: Path, body: str) -> tuple[Path, Path]:
    """A minimal finished run: one document, one page, written as socr writes it."""
    from ocr_output_contract import safe_checksum

    out = tmp_path / "out"
    doc_dir = out / pdf.stem
    (doc_dir / "pages").mkdir(parents=True)
    (doc_dir / f"{pdf.stem}.md").write_text(f"## Page 1\n\n{body}\n", encoding="utf-8")
    (doc_dir / "pages" / "00001.md").write_text(body, encoding="utf-8")
    winning = _out(body).to_dict()
    (doc_dir / "pages" / "00001.json").write_text(
        json.dumps(
            {
                "page_num": 1,
                "status": "success",
                "audit_passed": True,
                "engine": "qwen",
                "terminal": True,
                "winning_output": winning,
                "audit_events": [],
                "verdict": {"page_num": 1, "clauses": []},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (doc_dir / "manifest.json").write_text(
        json.dumps({"entries": {"1": {"page_num": 1, "blob_ref": ""}}}, indent=2), encoding="utf-8"
    )
    (doc_dir / "metadata.json").write_text(
        json.dumps({"key": pdf.name, "pages": 1, "checksum": safe_checksum(pdf)}, indent=2),
        encoding="utf-8",
    )
    (doc_dir / "audit_log.json").write_text(
        json.dumps({"events": [], "event_count": 0, "counts": {}}, indent=2), encoding="utf-8"
    )
    return out, doc_dir


def _tree(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


def test_the_tool_ships_the_crop_and_a_second_run_changes_nothing(tmp_path: Path) -> None:
    pdf = _pdf_with_a_figure(tmp_path / "paper.pdf", draw_figure=True)
    body = f"{_CAPTION}\n\n{_PROSE}"
    out, doc_dir = _run_dir(tmp_path, pdf, body)

    before = _tree(out)
    dry = tool.refinalize_document(doc_dir, dry_run=True, pdf_dir=tmp_path)
    assert len(dry) == 1 and dry[0].crops >= 1
    assert _tree(out) == before, "a dry run writes nothing at all, not even a PNG"

    changes = tool.refinalize_document(doc_dir, pdf_dir=tmp_path)
    assert len(changes) == 1 and changes[0].kind == RENDERED_KIND
    after_first = _tree(out)
    assert after_first != before

    fragment = (doc_dir / "pages" / "00001.md").read_text(encoding="utf-8")
    ref = f"figures/{crop_filename(1, 1)}"
    assert body in fragment, "the page's own text is untouched"
    assert leading_marks(fragment, 1), (
        "round 2: a page this lane acts on is MARKED even when its crop ships -- a crop is a "
        "picture, not a reading, and the lane cannot say it covers every figure (Argon P0-1)"
    )
    assert f"]({ref})" in fragment
    assert (doc_dir / ref).stat().st_size > 0, "the asset the link points at exists"
    assert fragment in (doc_dir / f"{pdf.stem}.md").read_text(encoding="utf-8")
    sidecar = json.loads((doc_dir / "pages" / "00001.json").read_text())
    assert sidecar["unshipped_figure_refs"] == [ref]
    assert sidecar["unshipped_figure_captions"] == ["Figure 7"]
    assert sidecar["status"] == "warning", (
        "round 2: shipping a crop does not make the page CHECKED (Argon P0-1)"
    )
    assert [e["kind"] for e in sidecar["audit_events"]] == ["unshipped_figure_rendered"]
    audit = json.loads((doc_dir / "audit_log.json").read_text())
    assert [e["kind"] for e in audit["events"]] == ["unshipped_figure_rendered"]
    assert "page(s) 1" in json.loads((doc_dir / "metadata.json").read_text())["error"]

    assert tool.refinalize_document(doc_dir, pdf_dir=tmp_path) == []
    assert _tree(out) == after_first, "a second run changes 0 bytes"


@pytest.mark.parametrize("draw_figure", [True, False], ids=["ship", "mark"])
def test_refinalize_then_resume_is_byte_identical(tmp_path: Path, draw_figure: bool) -> None:
    """The run after the tool must ship what the tool shipped, to the byte.

    A resumed page is restored from its sidecar and re-finalized, so this drives exactly
    that: the fields the sidecar carries go back onto a fresh ``PageState``, the production
    guard and verdict seam run again over the page the tool wrote, and nothing moves.

    BOTH branches, because they re-finalize differently: the ship branch's body already shows
    a picture, which is what makes the guard a no-op on it, while the mark branch's does not
    and its stability rests entirely on the published verdict (Argon, diff review).
    """
    pdf = _pdf_with_a_figure(tmp_path / "paper.pdf", draw_figure=draw_figure)
    body = f"{_CAPTION}\n\n{_PROSE}"
    _out_dir, doc_dir = _run_dir(tmp_path, pdf, body)
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
    from socr.core.verdict import PageVerdict

    refinalized = manifest._apply_unshipped_figure_guard(shipped, resumed)
    refinalized, _ = manifest._apply_verdict_marks(
        refinalized, resumed, 1, published=PageVerdict.from_dict(sidecar["verdict"])
    )
    assert refinalized.text == shipped.text
    assert refinalized.text == (doc_dir / "pages" / "00001.md").read_text(encoding="utf-8")


def test_a_caption_with_no_locatable_region_marks_the_page(tmp_path: Path) -> None:
    """The same page, the same text, with the figure's marks removed from the PDF."""
    pdf = _pdf_with_a_figure(tmp_path / "paper.pdf", draw_figure=False)
    body = f"{_CAPTION}\n\n{_PROSE}"
    _out_dir, doc_dir = _run_dir(tmp_path, pdf, body)
    changes = tool.refinalize_document(doc_dir, pdf_dir=tmp_path)
    assert len(changes) == 1 and changes[0].crops == 0

    fragment = (doc_dir / "pages" / "00001.md").read_text(encoding="utf-8")
    assert leading_marks(fragment, 1), "the page is MARKED"
    assert "Figure 7" in fragment.split("\n")[0], "the mark NAMES the figure that did not ship"
    assert body in fragment, "nothing of the page's own text is removed"
    sidecar = json.loads((doc_dir / "pages" / "00001.json").read_text())
    assert sidecar["unshipped_figure_unlocated"] == ["Figure 7"]
    assert sidecar["status"] == "warning"
    assert sidecar["failure_mode"] == FailureMode.FIGURE_NOT_SHOWN.value


def test_the_tool_leaves_a_page_that_already_shows_a_picture_alone(tmp_path: Path) -> None:
    pdf = _pdf_with_a_figure(tmp_path / "paper.pdf", draw_figure=True)
    body = f"{_CAPTION}\n\n{_PROSE}\n\n![Chart page 1](figures/chart_page_1.png)"
    _out_dir, doc_dir = _run_dir(tmp_path, pdf, body)
    assert tool.refinalize_document(doc_dir, pdf_dir=tmp_path) == []


def test_the_tool_refuses_a_pdf_that_is_not_the_run_input(tmp_path: Path) -> None:
    """A different edition would put a picture of one figure under another's caption."""
    pdf = _pdf_with_a_figure(tmp_path / "paper.pdf", draw_figure=True)
    body = f"{_CAPTION}\n\n{_PROSE}"
    _out_dir, doc_dir = _run_dir(tmp_path, pdf, body)
    _pdf_with_a_figure(pdf, draw_figure=False)  # same name, different bytes
    skipped: dict = __import__("collections").Counter()
    assert tool.refinalize_document(doc_dir, pdf_dir=tmp_path, skipped=skipped) == []
    assert "checksum" in " ".join(skipped)


def test_a_mark_branch_page_is_also_answered_only_once(tmp_path: Path) -> None:
    """The ship branch's own output stops it re-entering; the mark branch's does not.

    A marked page still carries no picture, so the body gate that makes a second run a no-op
    on a crop page does nothing here. ``_already_answered`` is the only thing between this
    page and a second mark on every run, and the mutation check found it unkillable until
    this test existed.
    """
    pdf = _pdf_with_a_figure(tmp_path / "paper.pdf", draw_figure=False)
    out, doc_dir = _run_dir(tmp_path, pdf, f"{_CAPTION}\n\n{_PROSE}")
    assert len(tool.refinalize_document(doc_dir, pdf_dir=tmp_path)) == 1
    after_first = _tree(out)
    assert tool.refinalize_document(doc_dir, pdf_dir=tmp_path) == []
    assert _tree(out) == after_first, "a second run changes 0 bytes on a marked page too"


def test_the_mark_branch_refuses_a_cross_reference_the_ship_branch_would_accept(
    tmp_path: Path,
) -> None:
    """Loose to ship, strict to mark, because a false fire costs differently in each.

    ``Figure 1 (panel a) shows ...`` (damato p13) matches the loose matcher and is a
    cross-reference. With no region located there is no crop to ship, and marking on the
    loose matcher would demote a page whose only offence is a sentence about a figure
    somewhere else.
    """
    pdf = _pdf_with_a_figure(tmp_path / "paper.pdf", draw_figure=False)
    reference = "Figure 1 (panel a) shows the companies' percentage composition by sector"
    out, doc_dir = _run_dir(tmp_path, pdf, f"{reference}\n\n{_PROSE}")
    before = _tree(out)
    assert figure_captions(reference) != (), "the loose matcher does fire on it"
    assert tool.refinalize_document(doc_dir, pdf_dir=tmp_path) == []
    assert _tree(out) == before, "nothing is written and nothing is marked"


def test_the_tool_leaves_a_page_whose_own_image_was_rendered_but_suppressed(
    tmp_path: Path,
) -> None:
    """#1030 renders the page image BEFORE selection; finalisation may then suppress it.

    Such a page ships no ``![`` at all, so the body gate does not catch it -- and the figure
    extractor skips it on exactly this predicate. Adding a crop here would give the page an
    asset its sibling lane deliberately withheld.
    """
    pdf = _pdf_with_a_figure(tmp_path / "paper.pdf", draw_figure=True)
    out, doc_dir = _run_dir(tmp_path, pdf, f"{_CAPTION}\n\n{_PROSE}")
    sidecar_path = doc_dir / "pages" / "00001.json"
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["scanned_figure_png_ref"] = (
        "![Scanned figure page 1](figures/scanned_figure_page_p1.png)"
    )
    sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    before = _tree(out)
    assert tool.refinalize_document(doc_dir, pdf_dir=tmp_path) == []
    assert _tree(out) == before


# ---------------------------------------------------------------------------
# The real shipping boundaries: process() and the finalisation seam
#
# Argon, diff review: with only the helpers exercised, DELETING the call to
# ``_agentic_unshipped_figure_page`` from ``_phase_agentic`` and the call to
# ``_apply_unshipped_figure_guard`` from ``_select_and_finalize_page`` left every test in
# this file green. The two below drive the production seams, so a reverted call site fails.
# ---------------------------------------------------------------------------

_MODEL_BODY = f"{_CAPTION}\n\n{_PROSE}"


class _Engine:
    name = "qwen"
    #: What the stub model reads. A class attribute so a test can change the ONE thing it is
    #: about without re-declaring the call signature, which is how the sentinel test below
    #: silently became vacuous: a wrong-arity replacement made the provider fail, the page fell
    #: back to its clean native text, and the body under test never existed.
    BODY = _MODEL_BODY

    def is_available(self) -> bool:
        return True

    def process_pages(self, pdf_path, page_nums, config, dpi, subprocess_timeout=None, **_kw):
        return [
            PageOutput(page_num=n, text=self.BODY, status=PageStatus.SUCCESS, engine="qwen")
            for n in page_nums
        ]


class _Judge:
    def assess(self, output, provider):
        from socr.pipeline.agentic import AcceptDecision

        return AcceptDecision(accept=True, reason="")


def _process(tmp_path: Path, monkeypatch, tag: str, *, draw_figure: bool, quiet: bool = True):
    """Drive the real ``process()`` on a one-page PDF a model reads.

    Hermetic: the engine, the judge and the engine LADDER are all stubbed, and
    ``primary_engine`` / ``local_engine`` / ``enabled_engines`` are pinned so
    ``resolve_auto_engine`` never probes this machine (CLAUDE.md, #841/#912).

    ``_is_chart_asset_page`` is forced False in BOTH arms, and that is the fixture's point
    rather than a convenience: the page-level chart lane claims a chart-only born-digital
    page and ships its whole-page PNG, so this lane rightly abstains on one. The 147 measured
    pages are the ones that lane DECLINED -- a chart+table page the arbitration held out
    (mechanism 1), or a page that is not native-eligible because its maths is garbled
    (mechanism 2, 97 of the 113 first-pass hits). Forcing the decline reproduces that state
    without needing a PDF that also carries a table and broken maths.
    """
    from socr.core.config import EngineType, PipelineConfig
    from socr.core.providers import PROFILE_QWEN_LOCAL
    from socr.pipeline import orchestrator as orch

    pdf = _pdf_with_a_figure(tmp_path / f"{tag}.pdf", draw_figure=draw_figure)
    with monkeypatch.context() as m:
        m.setattr(orch, "get_engine", lambda engine_type: _Engine())
        m.setattr(UnifiedPipeline, "_is_chart_asset_page", lambda self, *a, **k: False)
        # The page must be READ BY THE MODEL, which is the class #1187 is about. Without this
        # the fixture's clean native text is trusted and ships, the engine never runs, and the
        # sentinel test below is vacuous -- it asserted a strip on a body that never carried
        # the sentinel.
        m.setattr(UnifiedPipeline, "_is_agentic_trusted_native", lambda self, *a, **k: False)
        pipe = UnifiedPipeline(
            PipelineConfig(
                agentic=True,
                quiet=quiet,
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
        pipe._available_engines_for_agentic = lambda: [PROFILE_QWEN_LOCAL]
        pipe._build_page_judge = lambda state: _Judge()
        pipe._resolve_crop_vlm_model = lambda: None
        pipe._resolve_judge_model = lambda *a, **k: ""
        pipe.process(pdf, output_dir=tmp_path / f"out-{tag}")
    out = tmp_path / f"out-{tag}"
    side = json.loads(next(iter(out.rglob("pages/00001.json"))).read_text())
    text = next(iter(out.rglob("pages/00001.md"))).read_text()
    meta = next(
        iter(json.loads(next(iter(out.rglob("metadata.json"))).read_text())["files"].values())
    )
    return side, text, meta, out


def test_process_ships_the_crop_and_the_same_page_without_the_figure_is_untouched(
    tmp_path: Path, monkeypatch
) -> None:
    """One difference, through the real ``process()``: the figure's marks in the PDF.

    ``--save-figures`` is OFF in both arms, which is how run 23319732 and the Fed run were
    made: the crop this lane ships is forced, exactly as the chart lanes force theirs.
    """
    with_fig, with_text, with_meta, with_out = _process(
        tmp_path, monkeypatch, "fig", draw_figure=True
    )
    without, no_text, no_meta, _ = _process(tmp_path, monkeypatch, "pro", draw_figure=False)

    assert with_fig["engine"] == without["engine"] == "qwen", (
        "the fixture must drive a MODEL-read page, which is the class #1187 is about"
    )
    assert with_fig["unshipped_figure_refs"], "no crop shipped through process()"
    assert with_fig["unshipped_figure_kind"] == RENDERED_KIND
    ref = with_fig["unshipped_figure_refs"][0]
    assert f"]({ref})" in with_text
    assert (next(iter(with_out.rglob("figures"))) / Path(ref).name).is_file()
    assert _MODEL_BODY in with_text, "the model's own reading is untouched"
    assert leading_marks(with_text, 1), "a page the lane acts on is MARKED, crop or no crop"
    assert with_fig["status"] == "warning"
    assert "page(s) 1" in (with_meta.get("error") or ""), "the document note names the page"

    # The figure-free arm: same text, no crop -- and marked, because it still NAMES a figure.
    assert without["unshipped_figure_refs"] == []
    assert without["unshipped_figure_unlocated"] == ["Figure 7"]
    assert leading_marks(no_text, 1), "a page that names a figure and shows none is MARKED"
    assert without["status"] == "warning"
    assert "page(s) 1" in (no_meta.get("error") or "")


def test_the_finalisation_seam_applies_the_lane(tmp_path: Path) -> None:
    """``_select_and_finalize_page`` is where every writer reads one body from.

    Driving ``finalized_page_record`` rather than the guard directly is what makes deleting
    the guard's call site a failure rather than a silent revert.
    """
    from socr.core.document import DocumentHandle
    from socr.core.manifest import finalized_page_record
    from socr.core.state import DocumentState

    state = DocumentState(handle=DocumentHandle(path=tmp_path / "x.pdf", page_count=1))
    p = PageState(page_num=1)
    p.is_born_digital = True
    p.best_output = _out(_MODEL_BODY)
    p.unshipped_figure_captions = ["Figure 7"]
    state.pages[1] = p

    p.unshipped_figure_refs = []
    bare = finalized_page_record(state, 1)
    p.unshipped_figure_refs = list(REFS)
    shipped = finalized_page_record(state, 1)

    assert all(f"]({ref})" not in (bare.output.text or "") for ref in REFS)
    for ref in REFS:
        assert f"]({ref})" in (shipped.output.text or "")
    # The other lane marks this stub page too (no own text layer), so the body is no longer
    # the first thing on it. What this test is about is that the crops arrived and the page's
    # own reading survived intact.
    assert _MODEL_BODY in (shipped.output.text or "")

    p.unshipped_figure_refs = []
    p.unshipped_figure_unlocated = ["Figure 7"]
    marked = finalized_page_record(state, 1)
    assert leading_marks(marked.output.text or "", 1), "the seam marks an unshown figure"
    assert marked.output.status is PageStatus.WARNING
    assert any(c.check is Check.FIGURE_TEXT for c in marked.verdict.clauses)
    assert any(c.scope.startswith("figure:") for c in marked.verdict.clauses)


# ---------------------------------------------------------------------------
# The branches a page can take once a region IS located
# ---------------------------------------------------------------------------


def test_a_render_that_fails_marks_the_page_rather_than_claiming_a_picture(
    tmp_path: Path,
) -> None:
    """Geometry is the positive evidence here, so the LOOSE caption may name the figure.

    A region was located, so a figure is on this page whatever the caption's shape; if no crop
    can be saved it is preserved nowhere, and the page must say so. Before this the lane
    emitted the render-failure event while writing no clause, so the CLI and the document note
    called the page marked while it shipped SUCCESS unmarked (Argon, diff review).
    """
    from socr.figures.unshipped_figures import decide

    body = "Fig. 4.15 2D Gaussians with (a) $\\Sigma$\n\nprose"
    assert figure_captions(body, strict=True) == (), "the loose matcher alone fires on it"
    failed = decide(body, lambda: [object()], lambda _b: [])
    assert failed.kind == RENDER_FAILED_KIND
    assert failed.unlocated == ("Fig. 4.15",), "the figure is preserved nowhere; say so"

    # Round 2: a page that DOES ship crops is marked too, naming every figure it declares,
    # because a caption count cannot establish how many figures the page carries.
    shipped = decide(
        "Figure 1: first.\n\nFigure 2: second.\n\nprose",
        lambda: [object()],
        lambda _b: ["figures/unshipped_figure_p1_1.png"],
    )
    assert shipped.kind == RENDERED_KIND
    assert shipped.marks
    assert shipped.unlocated == ("Figure 1", "Figure 2")


def test_every_caption_is_named_in_the_alt_text_not_just_the_first() -> None:
    """A wrapped cross-reference above a real caption would otherwise label the real
    figure's crop with the cross-reference's number (Argon, diff review)."""
    block = figure_block(1, ["figures/a.png"], ("Figure 1", "Figure 2"))
    assert "Figure 1, Figure 2" in block


def test_a_whole_page_box_is_not_a_figure_region(tmp_path: Path) -> None:
    """On a scan the extractor's only region is the page raster itself.

    A whole-page "crop" is the page, not a figure, and the page image is #1030's artifact and
    the floors', under their own names. Shipping one here gave a #1030 control page a second
    picture of itself under a figure caption, which its end-to-end test caught.
    """
    from socr.figures.extractor import SCAN_RASTER_PAGE_COVERAGE_MIN
    from socr.figures.unshipped_figures import locate_figure_regions

    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 400, 500), False)
    pix.set_rect(pix.irect, (235, 235, 235))
    page.insert_image(page.rect, pixmap=pix)
    page.insert_text((72, 700), _CAPTION, fontname="helv", fontsize=10, render_mode=3)
    scan = tmp_path / "scan.pdf"
    doc.save(scan)
    doc.close()

    from socr.figures.extractor import FigureExtractor

    with fitz.open(scan) as d:
        raw = FigureExtractor(save_dir=None, max_total=9, max_per_page=9).extract_page(
            d[0], 1, d, 1
        )
    page_area = 612 * 792
    assert raw.figures, "the extractor does find the page raster"
    assert any(
        (b[2] - b[0]) * (b[3] - b[1]) / page_area >= SCAN_RASTER_PAGE_COVERAGE_MIN
        for b in (f.bbox for f in raw.figures)
        if b is not None
    )
    assert locate_figure_regions(scan, 1, 9) == [], "this lane drops it"


# ---------------------------------------------------------------------------
# Interruption: the sidecar is the commit point, every earlier write is repeatable
# ---------------------------------------------------------------------------


def _interrupt_and_redo(tmp_path: Path, *, draw_figure: bool) -> None:
    """Run the tool, throw away ONLY the sidecars, run it again, and require the same bytes.

    That is exactly a run killed after the fragments, the document ``.md``, the manifest, the
    metadata and the audit log were written and before the sidecars were: the pages come back
    unanswered, so the second run redoes them over text it has already rewritten.

    Three guards stand between that and a corrupted document, and each stayed unkillable until
    this existed (Argon, diff review): ``_merge_fragment`` and ``_swap_once`` must test the NEW
    text first, because this lane only ever ADDS and so ``old`` is a substring of ``new``; and
    the document note must REPLACE its own earlier copy rather than concatenate a second one.
    """
    pdf = _pdf_with_a_figure(tmp_path / "paper.pdf", draw_figure=draw_figure)
    out, doc_dir = _run_dir(tmp_path, pdf, f"{_CAPTION}\n\n{_PROSE}")
    before_sidecars = {p: p.read_bytes() for p in sorted((doc_dir / "pages").glob("*.json"))}
    assert tool.refinalize_document(doc_dir, pdf_dir=tmp_path)
    once = _tree(out)

    for path, raw in before_sidecars.items():
        path.write_bytes(raw)
    assert tool.refinalize_document(doc_dir, pdf_dir=tmp_path), "the page comes back unanswered"
    assert _tree(out) == once, "redoing an interrupted run must give the same bytes"


def test_an_interrupted_ship_branch_run_is_repaired_not_doubled(tmp_path: Path) -> None:
    _interrupt_and_redo(tmp_path, draw_figure=True)


def test_an_interrupted_mark_branch_run_is_repaired_not_doubled(tmp_path: Path) -> None:
    _interrupt_and_redo(tmp_path, draw_figure=False)


def test_a_vlm_sentinel_ref_does_not_make_the_lane_abstain(tmp_path: Path, monkeypatch) -> None:
    """GH-86 strips a phantom ``![...]`` the model invented; this lane must run AFTER it.

    Reading the body first saw the sentinel, called the page "already shows a picture" and
    abstained -- and three lines later the strip left the page with a bare caption and no mark
    (Argon, diff review).
    """
    monkeypatch.setattr(_Engine, "BODY", f"{_CAPTION}\n\n![a figure](image.png)\n\n{_PROSE}")
    side, text, _meta, _out = _process(tmp_path, monkeypatch, "sentinel", draw_figure=True)
    assert side["engine"] == "qwen", "the model reading must be what shipped"
    assert "image.png" not in text, "GH-86 did strip the phantom"
    assert side["unshipped_figure_refs"], "the lane abstained on a picture that was removed"
    assert f"]({side['unshipped_figure_refs'][0]})" in text


def test_the_published_verdict_keeps_a_clause_that_shares_a_check_and_a_scope() -> None:
    """Two clauses can say different things about the same check and region ordinal.

    Keying the merge on ``(check, scope)`` dropped the second, which on a page that already
    carried a ``figure_text`` clause would have silently replaced it (Argon, diff review).
    """
    recorded = {
        "check": Check.FIGURE_TEXT.value,
        "result": CheckResult.UNKNOWN.value,
        "scope": "figure:1",
        "reason": "something else entirely was wrong with this figure",
    }
    new_records = clause_records(1, ("Figure 7",), UNLOCATED_KIND)
    verdict = tool._published_verdict({"verdict": {"clauses": [recorded]}}, 1, new_records)
    reasons = [c.reason for c in verdict.clauses]
    assert recorded["reason"] in reasons
    assert new_records[0]["reason"] in reasons
    # And the merge is still a no-op on a second pass, which is what idempotence needs.
    again = tool._published_verdict(
        {"verdict": {"clauses": [c.to_dict() for c in verdict.clauses]}}, 1, new_records
    )
    assert [c.to_dict() for c in again.clauses] == [c.to_dict() for c in verdict.clauses]


def test_the_abstaining_outcome_is_answered_only_once(tmp_path: Path) -> None:
    """``unshipped_figure_captions`` alone is the abstention's record, and it must count.

    A gate that read only the two fields carrying an ANSWER re-ran the locator and appended a
    duplicate event on every pass over such a page (Argon, diff review).
    """
    from socr.core.document import DocumentHandle
    from socr.core.state import DocumentState

    pdf = _pdf_with_a_figure(tmp_path / "paper.pdf", draw_figure=False)
    state = DocumentState(handle=DocumentHandle(path=pdf, page_count=1))
    p = PageState(page_num=1)
    state.pages[1] = p
    pipe = UnifiedPipeline.__new__(UnifiedPipeline)
    from socr.core.config import PipelineConfig

    pipe.config = PipelineConfig(quiet=True)
    out = _out("Fig. 4.15 2D Gaussians with (a)\n\nprose with no figure on the page")
    pipe._agentic_unshipped_figure_page(state, 1, p, out, tmp_path / "figures")
    first = list(state.events)
    assert first and p.unshipped_figure_captions == ["Fig. 4.15"]
    assert p.unshipped_figure_refs == [] and p.unshipped_figure_unlocated == []
    pipe._agentic_unshipped_figure_page(state, 1, p, out, tmp_path / "figures")
    assert list(state.events) == first, "the page was answered a second time"


def test_the_guard_survives_a_stand_in_page_state() -> None:
    """A bare ``Mock`` attribute is truthy, so ``or []`` kept it and iterating it raised."""
    from unittest.mock import MagicMock, Mock

    out = _out("Figure 7: Bias and MSE.")
    assert manifest._apply_unshipped_figure_guard(out, Mock()) is out
    assert manifest._apply_unshipped_figure_guard(out, MagicMock()) is out


# =========================================================================================
# Round 2: one test per reproduced review finding.
# Reproductions, with command and output, are in
# scratch/routing-baseline/leads/1187/report-r2.md.
# =========================================================================================


def test_a_page_the_lane_acts_on_is_never_checked(tmp_path: Path) -> None:
    """Argon P0-1, reproduced on Fed calibration p3.

    That page heads itself "Exhibit 1" and carries SIX numbered panels; the locator finds two
    regions, both in the left column, and three charts shipped nowhere while the page counted
    CHECKED. ``len(refs) >= len(captions)`` read as "all of them are shown", which a caption
    count can never establish. So no outcome of this lane claims it: every page it acts on is
    marked, and the mark says the crops are not proven complete.
    """
    from socr.figures.unshipped_figures import MARKED_KINDS, decide

    one_caption_many_panels = "Exhibit 1\n\n(1) Equity Prices\n\n(2) Corporate Debt Spreads"
    d = decide(one_caption_many_panels, lambda: [object(), object()], lambda b: ["a.png", "b.png"])
    assert d.kind in MARKED_KINDS and d.marks, "two crops for a six-panel page is not CHECKED"
    assert "cover every figure" in d.detail
    assert clause_records(3, d.unlocated, d.kind)[0]["reason"].endswith("page image")


def test_the_document_is_not_clean_when_a_figure_it_names_is_not_shown(
    tmp_path: Path, monkeypatch
) -> None:
    """Argon P0-2, first half: the document said SUCCESS while carrying an error note."""
    _side, _text, meta, _out = _process(tmp_path, monkeypatch, "docstat", draw_figure=False)
    assert meta.get("error"), "the note is there"
    assert meta.get("status") != "success", (
        "a document whose only defect is a figure it does not show must not be clean"
    )


def test_the_failure_mode_is_this_lanes_own_and_stays_retryable() -> None:
    """Argon P0-3, both halves.

    Sharing ``VISUAL_VALUES_UNTRANSCRIBED`` recorded something FALSE about a page that shipped
    no image, and -- because that mode is permanently unresolvable -- made a TRANSIENT render
    failure unretryable on resume, unlike #1030's ``scanned_figure_render_failed``.
    """
    from socr.core.verdict import PERMANENTLY_UNRESOLVABLE_MODES

    assert FailureMode.FIGURE_NOT_SHOWN not in PERMANENTLY_UNRESOLVABLE_MODES
    out = _out("Figure 7: X.\n\nprose")
    p = _p([], unshipped_figure_unlocated=["Figure 7"], unshipped_figure_kind=UNLOCATED_KIND)
    marked, _ = manifest._apply_verdict_marks(out, p, 1)
    assert marked.failure_mode is FailureMode.FIGURE_NOT_SHOWN


def test_re_deriving_a_marked_page_adds_no_second_contradictory_clause() -> None:
    """Argon P0-3, first half.

    The shared mode row made a second derivation append a page-scoped clause saying the page's
    chart ships as an image -- on a page that shipped none -- and the body stopped being
    idempotent. The seam runs again over every page at assemble, so both matter.
    """
    out = _out("Figure 7: X.\n\nprose")
    p = _p([], unshipped_figure_unlocated=["Figure 7"], unshipped_figure_kind=UNLOCATED_KIND)
    first, v1 = manifest._apply_verdict_marks(out, p, 1)
    second, v2 = manifest._apply_verdict_marks(first, p, 1)
    third, v3 = manifest._apply_verdict_marks(second, p, 1)
    assert [(c.check, c.scope) for c in _figure_clauses(v1)] == [(Check.FIGURE_TEXT, "figure:1")]
    assert v1.to_dict() == v2.to_dict() == v3.to_dict(), "no second clause, ever"
    assert first.text == second.text == third.text, "and the body is idempotent"
    assert "ships as an image" not in second.text


def test_the_guard_leaves_a_whole_page_failure_marker_a_marker() -> None:
    """Fable P1-1, reproduced.

    ``is_page_failed_marker`` admits "a marker plus at most one image" (GH-371), so appending
    two crops made a fail-closed floor stop reading as a floor to the disposition classifier,
    to selection and to the resume gate -- the ruling inverted. The figure is still owed a
    statement and still gets one, from the verdict clause.
    """
    marker = PageOutput(
        page_num=3,
        text="[page 3 failed: table withheld]",
        status=PageStatus.ERROR,
        engine="qwen",
        audit_passed=False,
    )
    assert manifest.is_page_failed_marker(marker.text)
    got = manifest._apply_unshipped_figure_guard(marker, _p(REFS))
    assert got is marker, "the guard must abstain"
    assert manifest.is_page_failed_marker(got.text), "the floor is still a floor"


def test_the_lane_decides_after_both_table_withholding_gates() -> None:
    """Argon P1-1: one gate on the bytes about to ship, not on an earlier copy of them.

    Pinned on the source order rather than on a fixture, because the harm needs a page whose
    table is withheld AFTER the lane has already rendered a crop for it; the ordering is the
    invariant, and #1203 learned the same lesson in the same loop.
    """
    src = Path(__file__).resolve().parents[1] / "src/socr/pipeline/orchestrator.py"
    text = src.read_text()
    lane = text.index("self._agentic_unshipped_figure_page(state, page_num, ps, bo,")
    for gate in (
        "self._withhold_contradicted_unverified_tables(state, page_num, ps, bo)",
        "self._withhold_unconfirmed_table(state, page_num, ps, _chart_figures_dir)",
        "self._sanitize_agentic_page_image_refs(",
    ):
        assert text.index(gate) < lane, f"the lane must decide after {gate}"


@pytest.mark.parametrize(
    "line",
    [
        "Figure C.1 shows the impulse response",
        "Figure I.2 illustrates the mechanism",
        "Figure 1-1 shows the baseline",
        "Exhibit II-3 reports the totals",
    ],
    ids=["C.1", "I.2", "1-1", "II-3"],
)
def test_a_dotted_or_hyphenated_cross_reference_is_not_a_caption(line: str) -> None:
    """Argon P1-2, the false-positive half: all four MARKED a prose page.

    The label stopped at the Roman-numeral letter or the first digit and the terminator ate the
    ``.`` or the ``-``. The alternatives now consume the whole dotted or hyphenated number,
    atomically, so there is nothing left for a terminator to match.
    """
    assert figure_captions(line) == ()
    assert figure_captions(line, strict=True) == ()


@pytest.mark.parametrize(
    ("line", "label"),
    [
        ("Figure A.1: Impulse responses", "Figure A.1"),
        ("Figure E.2: Bias and MSE under a CEE-type VAR DGP (No controls)", "Figure E.2"),
        ("Figure B.2: Robustness", "Figure B.2"),
        ("Figure 1a: Panel one", "Figure 1a"),
        ("Figure 2B: Panel two", "Figure 2B"),
        ("Exhibit A-1: Totals", "Exhibit A-1"),
    ],
    ids=["A.1", "E.2-herbst-p57", "B.2", "1a", "2B", "A-1"],
)
def test_an_appendix_or_subfigure_caption_is_matched(line: str, label: str) -> None:
    """Argon P1-2 / Fable P1-2, the false-negative half.

    ``E.2`` is ``herbst_johannsen`` p57 and p58 -- TWO of the twelve pages the issue
    adjudicates. Round 1 said the locator "found a region on all 12"; true of the locator,
    false of the lane, which never reached it on those two. Fable measured 13 such lines on 11
    pages of the 2,751.
    """
    assert figure_captions(line) == (label,)
    assert figure_captions(line, strict=True) == (label,)


def test_the_tool_records_the_assets_it_attaches(tmp_path: Path) -> None:
    """Argon P1-3a: a page the tool gave a PNG had no asset record in the manifest.

    ``manifest.stale_assets`` therefore never checked the new crops, and the next resume would
    add them -- drift the tool itself introduced. Recorded with the same function
    ``build_manifest`` uses.
    """
    pdf = _pdf_with_a_figure(tmp_path / "paper.pdf", draw_figure=True)
    _out_dir, doc_dir = _run_dir(tmp_path, pdf, f"{_CAPTION}\n\n{_PROSE}")
    assert tool.refinalize_document(doc_dir, pdf_dir=tmp_path)
    entry = json.loads((doc_dir / "manifest.json").read_text())["entries"]["1"]
    paths = [a["logical_path"] for a in entry.get("assets", [])]
    assert paths == [f"figures/{crop_filename(1, 1)}"]
    assert all(a.get("content_hash") for a in entry["assets"])


def test_the_tools_sidecar_events_have_the_live_writers_shape(tmp_path: Path) -> None:
    """Argon P1-3c: the tool added a ``page_num`` key the live writer omits.

    ``_flush_page_sidecar`` builds each per-page event from ``kind/engine/detail/data`` only,
    so the extra key made a resume rewrite the sidecar. The DOCUMENT log does key events by
    page, and still does.
    """
    pdf = _pdf_with_a_figure(tmp_path / "paper.pdf", draw_figure=True)
    _out_dir, doc_dir = _run_dir(tmp_path, pdf, f"{_CAPTION}\n\n{_PROSE}")
    assert tool.refinalize_document(doc_dir, pdf_dir=tmp_path)
    side = json.loads((doc_dir / "pages" / "00001.json").read_text())
    for ev in side["audit_events"]:
        assert set(ev) == {"kind", "engine", "detail", "data"}, ev
    doc_events = json.loads((doc_dir / "audit_log.json").read_text())["events"]
    assert all("page_num" in e for e in doc_events), "the document log still keys by page"


def test_the_mark_says_what_actually_happened_to_this_figure() -> None:
    """Round 1 had one hard-coded reason and used it on every branch, including the one where
    crops DID ship -- so a page with a picture under its caption was told socr could not
    locate it. The clause is now keyed on the outcome."""
    labels = ("Figure 7",)
    unlocated = clause_records(1, labels, UNLOCATED_KIND)[0]["reason"]
    failed = clause_records(1, labels, RENDER_FAILED_KIND)[0]["reason"]
    rendered = clause_records(1, labels, RENDERED_KIND)[0]["reason"]
    assert len({unlocated, failed, rendered}) == 3, "one reason per branch"
    assert "could not locate" in unlocated
    assert "preserved nowhere" in failed
    assert "ships a crop" in rendered and "cover every figure" in rendered


def test_the_document_status_turns_on_this_lane_alone(tmp_path: Path, monkeypatch) -> None:
    """Argon P0-2, pinned as the DIFFERENCE this lane makes.

    Asserting only "the document is not clean" is not enough: the fixture's page is WARNING,
    and something else could be carrying the status. This runs the same document twice,
    changing only whether the PDF holds a figure the page names, and requires the status to
    move with it.
    """
    _s1, _t1, with_fig, _o1 = _process(tmp_path, monkeypatch, "dirty", draw_figure=False)
    clean_body = _PROSE  # names no figure at all, so the lane never fires
    monkeypatch.setattr(_Engine, "BODY", clean_body)
    _s2, _t2, no_fig, _o2 = _process(tmp_path, monkeypatch, "clean", draw_figure=False)
    # The root index uses its own vocabulary for status, so the pin is the DIFFERENCE between
    # two runs of the same document that differ only in whether its page names a figure.
    assert not no_fig.get("error"), "the control document, naming no figure, is clean"
    assert with_fig.get("error"), "the note is there"
    assert with_fig.get("status") != no_fig.get("status"), (
        "a document whose only defect is a figure it does not show must not be clean"
    )


def test_the_cli_reports_a_lost_figure_that_is_the_documents_only_defect(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """Argon P0-2, second half: the lane's two console lines sat inside a block whose
    condition listed 27 page buckets and none of this lane's, so they were unreachable
    whenever a lost figure was all that was wrong with the document."""
    _process(tmp_path, monkeypatch, "cli", draw_figure=False, quiet=False)
    printed = capsys.readouterr().out
    assert "named a figure" in printed, printed[-1500:]


def test_an_interrupted_run_on_an_ALREADY_marked_page_is_repaired_not_doubled(
    tmp_path: Path,
) -> None:
    """The ordering guard in ``_merge_fragment``, on the one shape that still reaches it.

    When this lane adds the page's only mark, the new body does not start with the old one and
    either order works. When the page was ALREADY marked by something else, this lane appends
    crops and nothing else, so ``new`` starts with ``old`` again -- and testing ``old`` first
    finds its copy inside the ``new`` an interrupted run already wrote.
    """
    pdf = _pdf_with_a_figure(tmp_path / "paper.pdf", draw_figure=True)
    prior = "> [page 1 unchecked: page_judge=UNKNOWN; scope=page; reason=another lane said so]"
    out, doc_dir = _run_dir(tmp_path, pdf, f"{prior}\n\n{_CAPTION}\n\n{_PROSE}")
    sidecar_path = doc_dir / "pages" / "00001.json"
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["verdict"] = {
        "page_num": 1,
        "clauses": [
            {
                "check": "page_judge",
                "result": "UNKNOWN",
                "scope": "page",
                "reason": "another lane said so",
            }
        ],
    }
    sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    before_sidecars = {p: p.read_bytes() for p in sorted((doc_dir / "pages").glob("*.json"))}

    assert tool.refinalize_document(doc_dir, pdf_dir=tmp_path)
    once = _tree(out)
    fragment = (doc_dir / "pages" / "00001.md").read_text(encoding="utf-8")
    assert fragment.startswith(prior), "the other lane's mark is still first"
    ref = "figures/" + crop_filename(1, 1)
    assert fragment.count("](" + ref + ")") == 1

    for path, raw in before_sidecars.items():
        path.write_bytes(raw)
    assert tool.refinalize_document(doc_dir, pdf_dir=tmp_path), "the page comes back unanswered"
    assert _tree(out) == once, "redoing an interrupted run must give the same bytes"


# =========================================================================================
# The #1185 merge: two independent lanes, one page.
# =========================================================================================


def test_a_figure_mark_and_a_numbers_mark_co_exist_on_one_page() -> None:
    """Neither lane undoes the other, and the order matches a fresh run.

    #1185 marks a page whose text layer disputes the reading's numbers; #1187 marks a page
    that names a figure it does not show. They are different facts about different things and
    a page can carry both. ``_clauses_ahead_of_text_layer_numbers`` is the single ordering
    authority, and this lane's clauses are inside it, so the figure clause precedes the
    numbers clause on a fresh run.
    """
    from socr.core.verdict import silent_success_clauses

    out = _out("Figure 7: X.\n\nprose")
    p = _p([], unshipped_figure_unlocated=["Figure 7"], unshipped_figure_kind=UNLOCATED_KIND)
    clauses = silent_success_clauses(out, p)
    checks = [c.check for c in clauses]
    assert Check.FIGURE_TEXT in checks, "this lane's clause survives the merge"
    assert Check.TEXT_LAYER_NUMBERS in checks, "the other lane's clause survives too"
    assert checks.index(Check.FIGURE_TEXT) < checks.index(Check.TEXT_LAYER_NUMBERS), (
        "a fresh run puts the figure clause ahead of the numbers clause"
    )

    marked, verdict = manifest._apply_verdict_marks(out, p, 1)
    from socr.core.marks import leading_mark_blocks

    assert leading_marks(marked.text, 1), "the page is MARKED"
    # Two separate blockquote blocks, so ``leading_marks`` (the meter's question -- is the
    # FIRST block all marks) sees one and ``leading_mark_blocks`` walks both.
    assert len(leading_mark_blocks(marked.text, 1)) == 2, "both marks are written, not one"
    assert "Figure 7" in marked.text and "numbers" in marked.text


def test_the_other_lanes_tool_counts_this_lanes_clauses() -> None:
    """``with_text_layer_numbers_clause`` slots its clause by the COUNT ahead of it.

    ``socr-refinalize-own-text`` rebuilds a stand-in ``PageState`` to compute that count. If
    it does not restore this lane's two fields, the count is short on a page that also names
    a figure it does not show, and the numbers mark lands ABOVE the figure mark — unlike a
    fresh run (Fable, round-2 P2).
    """
    from socr.core.state import PageState
    from socr.core.verdict import _clauses_ahead_of_text_layer_numbers

    out = _out("Figure 7: X.\n\nprose")
    blind = PageState(page_num=1)
    seeing = PageState(page_num=1)
    seeing.unshipped_figure_unlocated = ["Figure 7"]
    seeing.unshipped_figure_kind = UNLOCATED_KIND
    assert (
        len(_clauses_ahead_of_text_layer_numbers(out, seeing, frozenset()))
        == len(_clauses_ahead_of_text_layer_numbers(out, blind, frozenset())) + 1
    ), "this lane's clause must be counted among those ahead"

    # Behavioural, not a grep: the stand-in the other lane's tool builds must produce the
    # same count as the page state a live run had.
    from socr.pipeline.refinalize_own_text import slot_reference_state

    stand_in = slot_reference_state(
        {"unshipped_figure_unlocated": ["Figure 7"], "unshipped_figure_kind": UNLOCATED_KIND}, 1
    )
    assert len(_clauses_ahead_of_text_layer_numbers(out, stand_in, frozenset())) == len(
        _clauses_ahead_of_text_layer_numbers(out, seeing, frozenset())
    ), "the other lane's stand-in must count what a live page state counts"


def test_this_lanes_tool_keeps_a_numbers_clause_behind_its_own(tmp_path: Path) -> None:
    """The mirror: this tool must not append its clause below one the other lane wrote.

    Appending would order the two marks one way when this tool runs after
    ``socr-refinalize-own-text`` and the other way when it runs before — and neither would
    match a fresh run on a page both lanes touch.
    """
    numbers = {
        "check": "text_layer_numbers",
        "result": "UNKNOWN",
        "scope": "page",
        "reason": "this page has no text layer of its own",
    }
    verdict = tool._published_verdict(
        {"verdict": {"clauses": [numbers]}},
        1,
        clause_records(1, ("Figure 7",), UNLOCATED_KIND),
    )
    checks = [c.check for c in verdict.clauses]
    assert checks.index(Check.FIGURE_TEXT) < checks.index(Check.TEXT_LAYER_NUMBERS)
    assert len(checks) == 2, "neither clause is dropped"
