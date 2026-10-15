"""#1144: a floored table page must stop destroying the page's own prose.

Coca-Cola CDP 2022 p77 is born-digital and its text layer is EXACT -- 1,053
characters, every number present. On main 25daab9f it shipped ~100 bytes of
whole-page marker: both tables AND all the C8.2b/C8.2c prose, lost to a table
nothing could verify.

Round 1 of this ticket tried to admit the model reading by waiving the
``table_escalation_refused`` event for #713's credential stand-in. Three
reviewers rejected it and I reproduced all three faults: the waiver's evidence
was not replayed on resume (so run 2 destroyed what run 1 shipped), its
detected-count condition failed open on 0/None, and ``table_not_scorable`` --
which it read as "nothing measured wrong" -- means, in its own source comment,
"the score is not evidence either way" (``table_grid.py:446-452``). The waiver
is reverted in full. p77 has no sound positive evidence for its tables: the page
judge never answered, the ladder judged a shared whole-page crop with no
mechanical binding, and the grid could not be graded.

Owner ruling 2026-10-14: ship the page's own PDF text with every table withheld,
via ``native_prose_floor_text`` -- socr's existing ratified mechanism for this
(#649 / #652) -- and withhold every band inside a DETECTED table box, numeric or
not, so no answer can land beside the wrong question. A page with no boxes
behaves exactly as it did before.

Every test pins a DIFFERENCE between two runs in the same process, per this
repo's CI rule.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pytest

from socr.core.result import PageOutput, PageStatus

#: Lines of the fixture's numeral-free question/answer grid. Shipping any of
#: these as loose text is the defect the box rule exists to stop: the answer
#: lands beside the wrong question.
_ANSWER_RE = re.compile(r"(?:^| )(?:Yes|No)(?: |$)", re.M)
#: A sentence outside both table boxes. It must survive.
_PROSE = "State how much fuel your organization consumed, excluding"
#: The model's reading, which is NOT what ships here -- the native layer is.
_MODEL = "Energy totals.\n\n| metric | 2022 |\n| --- | --- |\n| purchased steam | 42,548 |\n"
_NUMBERS = ("42,548", "228", "42,776", "2,993,732")


def test_loaded_source_is_this_checkout() -> None:
    import socr

    assert Path(socr.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1] / "src")


def _p77_like(path: Path) -> Path:
    """A born-digital page with p77's shape: a numeric grid AND a Yes/No grid.

    Verified to reproduce p77's detection exactly -- ``has_tables`` true and
    TWO ``detected_table_bboxes`` -- which is what makes the box rule testable
    at all. The second grid carries no printed numeral anywhere, so the band
    partition's numeral count cannot see it; only its box can.
    """
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 60), "(C8.2a) Report your organization's energy consumption totals.")
    rows = [
        ["purchased steam", "42,548", "228", "42,776"],
        ["self-generated", "4,070", "", "4,070"],
        ["total consumption", "524,332", "2,469,400", "2,993,732"],
    ]
    for r, cells in enumerate(rows):
        for c, cell in enumerate(cells):
            page.insert_text((80 + c * 110, 110 + r * 22), cell, fontsize=9)
    for r in range(4):
        page.draw_line(fitz.Point(76, 96 + r * 22), fitz.Point(516, 96 + r * 22))
    for c in range(5):
        page.draw_line(fitz.Point(76 + c * 110, 96), fitz.Point(76 + c * 110, 96 + 3 * 22))

    page.insert_text((72, 200), "Select the applications of your consumption of fuel.")
    qa = [
        ("Indicate whether your organization undertakes this", ""),
        ("generation of electricity", "Yes"),
        ("generation of heat", "Yes"),
        ("generation of steam", "No"),
        ("generation of cooling", "No"),
    ]
    for r, (q, a) in enumerate(qa):
        page.insert_text((80, 230 + r * 22), q, fontsize=9)
        if a:
            page.insert_text((400, 230 + r * 22), a, fontsize=9)
    for r in range(6):
        page.draw_line(fitz.Point(76, 216 + r * 22), fitz.Point(516, 216 + r * 22))
    for c in (0, 320, 440):
        page.draw_line(fitz.Point(76 + c, 216), fitz.Point(76 + c, 216 + 5 * 22))

    page.insert_text((72, 380), _PROSE)
    page.insert_text((72, 394), "feedstocks, by fuel type. Sustainable biomass heating value.")
    doc.save(str(path))
    doc.close()
    return path


class _GridEngine:
    name = "qwen"

    def is_available(self) -> bool:
        return True

    def process_pages(self, pdf_path, page_nums, config, dpi, subprocess_timeout=None, **_kw):
        return [
            PageOutput(page_num=n, text=_MODEL, status=PageStatus.SUCCESS, engine="qwen")
            for n in page_nums
        ]


class _SleepyLeaf:
    """A page judge that never answers inside the deadline (a real timeout)."""

    def assess(self, output, provider):  # pragma: no cover - the wrapper kills it
        time.sleep(5)
        raise AssertionError("the deadline must fire first")


def _pipeline():
    from socr.core.config import EngineType, PipelineConfig
    from socr.core.providers import PROFILE_QWEN_LOCAL
    from socr.pipeline.orchestrator import UnifiedPipeline

    pipe = UnifiedPipeline(
        PipelineConfig(
            agentic=True,
            quiet=True,
            primary_engine=EngineType.QWEN,
            local_engine=EngineType.QWEN,
            enabled_engines=[EngineType.QWEN],
            native_first=False,
            write_manifest=False,
            judge_backend="heuristic",
            table_judge_ladder=False,
            dual_pass_tables=False,
            detect_equations=False,
            save_figures=False,
        )
    )
    pipe._available_engines_for_agentic = lambda: [PROFILE_QWEN_LOCAL]
    pipe._resolve_judge_model = lambda *a, **k: ""
    pipe._resolve_crop_vlm_model = lambda: None
    pipe._build_page_judge = lambda state: UnifiedPipeline._TimeoutJudge(
        _SleepyLeaf(), 0.05, owner=pipe
    )
    return pipe


def _ship(tmp_path: Path, tag: str, monkeypatch, *, drop_boxes: bool = False):
    """One ``process()`` run over the fixture; returns (body, sidecar, out_dir).

    Hermetic: the ladder, the judge model and the crop model are all pinned, so
    nothing reaches Ollama. ``drop_boxes`` is the control arm -- it clears the
    page's detected boxes right after assessment, which is exactly the state a
    page with no detectable table geometry reaches, and the state p77 would be
    in if the box rule were not there.
    """
    from socr.pipeline import orchestrator as orch

    work = tmp_path / tag
    work.mkdir(parents=True, exist_ok=True)
    pdf = _p77_like(work / "doc.pdf")
    monkeypatch.setattr(orch, "get_engine", lambda engine_type: _GridEngine())
    pipe = _pipeline()

    if drop_boxes:
        # The detector is the seam: a page whose table geometry it cannot see
        # reports no bbox, which is the state ``partition_prose_bands`` falls
        # back to (and the state p77 would be in without the box rule). Patched
        # at the detector rather than on PageState so the control arm travels the
        # same assignment path (state.py:801-803) the real page does.
        from socr.core.born_digital import BornDigitalDetector

        _real_detect = BornDigitalDetector.detect

        def _no_boxes(self, pdf_path, *a, **kw):
            assessment = _real_detect(self, pdf_path, *a, **kw)
            for pa in assessment.pages:
                pa.detected_table_bboxes = []
            return assessment

        monkeypatch.setattr(BornDigitalDetector, "detect", _no_boxes)

    out = work / "out"
    pipe.process(pdf, output_dir=out)
    body = next(iter(out.rglob("pages/00001.md"))).read_text()
    sidecar = json.loads(next(iter(out.rglob("pages/00001.json"))).read_text())
    return body, sidecar, out


# ---------------------------------------------------------------------------
# 1. The ticket: the prose survives, the tables do not ship.
# ---------------------------------------------------------------------------


def test_process_ships_the_prose_and_withholds_both_tables(tmp_path: Path, monkeypatch) -> None:
    """The owner's four acceptance points, through the real ``process()``.

    On main this page shipped the bare whole-page marker and nothing else.
    """
    from socr.core.manifest import (
        BORN_DIGITAL_PROSE_RECOVERED_FLAG,
        SCANNED_PROSE_RECOVERED_FLAG,
    )

    body, sidecar, _ = _ship(tmp_path, "main", monkeypatch)

    # the page's own prose ships
    assert _PROSE in body
    # both tables are withheld, behind a marker and the page image
    assert "failed: unverifiable table" in body
    assert ".png)" in body
    # no table content ships as loose text -- not a number, not an answer
    assert _ANSWER_RE.search(body) is None, "an answer shipped beside a question"
    for number in _NUMBERS:
        assert number not in body
    # and the page says what it is
    # The fixture is BORN-DIGITAL, so the banner must not call it a scan
    # (#1144 round 4: Herbst-Johannsen p43, a born-digital page of three charts,
    # opened "unverified scan").
    assert BORN_DIGITAL_PROSE_RECOVERED_FLAG.format(page_num=1) in body
    assert SCANNED_PROSE_RECOVERED_FLAG.format(page_num=1) not in body
    assert "unverified scan" not in body
    assert sidecar["status"] == "error"
    assert sidecar["failure_mode"] == "page_judge_timeout"
    assert sidecar["disposition"]["ending"] == "fail_closed_marker"


def test_a_boxless_word_cell_row_of_any_vocabulary_is_the_known_residual(
    tmp_path: Path,
) -> None:
    """What is withheld under each box coverage, and what is not.

    DIFFERENCE: which of the fixture's two tables the single detected box
    covers. Both arms pass #1144's box gate, so only coverage varies, and
    NEITHER may ship the word-cell grid's row.

    Deliberately asserts the property, not which rule delivers it. Three rules
    can: the box rule when the box is over that grid, the categorical-row rule
    anywhere, and (round 4) the two-column refusal when the page's own gutter
    shows a band welding two columns. An earlier version of this test pinned the
    uncovered-table case as a visible RESIDUAL; round 4's gutter refusal closed
    it on this fixture, which is why the assertion is now about the outcome --
    a pin on the mechanism would have to be rewritten every time another rule
    reaches the same page.
    """
    from socr.core.manifest import table_floor_text_for_source

    numeric_box = (76.0, 96.0, 516.0, 162.0)
    qa_box = (76.0, 216.0, 516.0, 326.0)

    def _arm(name: str, boxes: list[tuple[float, float, float, float]]) -> str:
        ps = _floor_state(tmp_path / name)
        ps.detected_table_count = len(boxes)
        ps.detected_table_bboxes = boxes
        return table_floor_text_for_source(ps, 1, ps.native_text, prose_recovery=True)

    # Covered by a box, or by both: the grid's rows are withheld.
    for name, boxes in (("qa", [qa_box]), ("both", [numeric_box, qa_box])):
        body = _arm(name, boxes)
        assert _ANSWER_RE.search(body) is None, f"a grid row shipped with boxes={name}"

    # NOT covered by a box: a KNOWN RESIDUAL, pinned.
    #
    # The hole is wider than this fixture's Yes/No vocabulary, which is why the
    # pin is named for the general shape (Fable, PR #1158 round 5): ANY boxless
    # row of short word cells passes every rule -- "Sample Full Full Restricted
    # Full", "Estimator OLS IV OLS IV" as much as "generation of electricity
    # Yes". The box rule needs a box over it; the categorical-row rule only
    # knows the Yes/No vocabulary, and here only ONE cell per row anyway, which
    # by that rule's own definition is a sentence; and the order gate passes
    # because the order is right and only the BINDING is lost.
    #
    # Round 4's gutter check happened to refuse this fixture, masking the
    # residual; round 5 removed that check as subsumed by the order gate, which
    # is true for ORDER and not for this. Measured rather than argued: 0 of the
    # 138 gaining pages ship such a row, with the WIDE vocabulary (Fable
    # measured the same). Pinned, not hidden.
    uncovered = _arm("num", [numeric_box])
    assert _ANSWER_RE.search(uncovered) is not None, (
        "the residual must stay visible; if it stops, something now catches this "
        "shape and this pin should become an assertion that it does"
    )
    # Even so, a MULTI-cell Yes/No row never ships: that rule needs no geometry.
    assert _CATEGORICAL_ROW.search(uncovered) is None
    # And the general shape is what is unguarded, not just this vocabulary: a
    # row of any short word cells would pass the same three rules.
    generic = "Estimator OLS IV OLS IV"
    from socr.tables.row_corroboration import bears_categorical_row

    assert bears_categorical_row(_band(*generic.split())) is False, (
        "if this ever becomes True the residual has narrowed and this pin should "
        "become an assertion that the row is withheld"
    )


# ---------------------------------------------------------------------------
# 2. Resume. Round 1's waiver shipped on run 1 and floored on run 2; this lane
#    reads only PDF-derived state, and that must be demonstrated, not argued.
# ---------------------------------------------------------------------------


def test_a_second_run_over_the_same_output_dir_is_byte_identical(
    tmp_path: Path, monkeypatch
) -> None:
    """The lane is DETERMINISTIC across runs -- which is what this measures.

    #1144 round 4 (Fable P2): the name said "on resume" and that is not what
    happens. An ERROR page fails the ledger gate (``orchestrator.py``'s
    ``_resume_skippable``) and is REPROCESSED, so nothing is restored from the
    sidecar here. What the test does establish is the property the ticket needs
    and round 1's waiver lacked: a second run over the same output directory
    reproduces the bytes and the shipped fields, because this lane reads only
    PDF-derived state. Renamed rather than deleted, and the claim narrowed to
    what it proves.

    The lane reads ``native_words``, ``native_text`` and
    ``detected_table_bboxes``; the first two are re-derived from the PDF on every
    run and the third is restored from the sidecar. A second ``process()`` into
    the SAME output directory must reproduce the fragment and the sidecar's
    shipped fields exactly.
    """
    from socr.pipeline import orchestrator as orch

    body, sidecar, out = _ship(tmp_path, "rerun", monkeypatch)

    monkeypatch.setattr(orch, "get_engine", lambda engine_type: _GridEngine())
    pipe = _pipeline()
    pipe.process(out.parent / "doc.pdf", output_dir=out)
    again = next(iter(out.rglob("pages/00001.md"))).read_text()
    again_sidecar = json.loads(next(iter(out.rglob("pages/00001.json"))).read_text())

    assert again == body, "a second run changed the shipped bytes"
    for field in ("status", "failure_mode", "disposition"):
        assert again_sidecar[field] == sidecar[field]
    assert _ANSWER_RE.search(again) is None
    assert _PROSE in again


# ---------------------------------------------------------------------------
# 3. The recovery is offered to the NATIVE caller only.
# ---------------------------------------------------------------------------


def test_prose_recovery_is_off_by_default_so_a_model_winner_never_reaches_it(
    tmp_path: Path,
) -> None:
    """P1's withhold path passes a MODEL winner's text and must floor bare.

    Shipping a model's prose from a floored page is the design the owner closed
    unmerged in PR #1047 (#1043): corroborating it against a noisy text layer
    recovered 1 of 109 pages. DIFFERENCE: the ``prose_recovery`` flag alone,
    over one page state.
    """
    import fitz

    from socr.core.document import DocumentHandle
    from socr.core.manifest import table_floor_text_for_source
    from socr.core.state import DocumentState

    pdf = _p77_like(tmp_path / "doc.pdf")
    state = DocumentState(handle=DocumentHandle.from_path(pdf))
    ps = state.pages[1]
    doc = fitz.open(pdf)
    ps.native_text = doc[0].get_text()
    ps.native_words = doc[0].get_text("words")
    doc.close()
    ps.is_born_digital = True
    ps.has_tables = True
    ps.detected_table_count = 2
    ps.detected_table_bboxes = [(76.0, 96.0, 516.0, 162.0), (76.0, 216.0, 516.0, 326.0)]
    ps.d3_floor_png_ref = "![Failed table page 1](figures/failed_table_p1.png)"

    off = table_floor_text_for_source(ps, 1, _MODEL)
    on = table_floor_text_for_source(ps, 1, ps.native_text, prose_recovery=True)

    assert _PROSE not in off, "the default must not recover prose"
    assert _PROSE in on
    assert _ANSWER_RE.search(on) is None


def _floor_state(tmp_path: Path):
    """A page state carrying the fixture's native layer and its detected boxes."""
    import fitz

    tmp_path.mkdir(parents=True, exist_ok=True)

    from socr.core.document import DocumentHandle
    from socr.core.state import DocumentState

    pdf = _p77_like(tmp_path / "doc.pdf")
    state = DocumentState(handle=DocumentHandle.from_path(pdf))
    ps = state.pages[1]
    doc = fitz.open(pdf)
    ps.native_text = doc[0].get_text()
    ps.native_words = doc[0].get_text("words")
    doc.close()
    ps.is_born_digital = True
    ps.has_tables = True
    ps.detected_table_count = 2
    ps.detected_table_bboxes = [(76.0, 96.0, 516.0, 162.0), (76.0, 216.0, 516.0, 326.0)]
    ps.d3_floor_png_ref = "![Failed table page 1](figures/failed_table_p1.png)"
    return ps


@pytest.mark.parametrize(
    "flag", ["native_table_lane_refused_rotated", "native_rotated_text_shredded"]
)
def test_a_rotated_page_gets_the_bare_marker_not_confetti(tmp_path: Path, flag: str) -> None:
    """A rotated page's ``get_text()`` order is not its reading order.

    Found by GH-147's own hermetic ``process()`` test, which this change broke
    before the gate existed: its rotated fixture recovered "content despite /
    genuine / grid / sentence a / as rotated" -- the page's words, in the wrong
    order. Shipping that is a WRONG text, which this repo ranks below a missing
    one. ``native_rotated_text_shredded`` does not cover it (False on that
    page), so GH-147's own rotation refusal is carried onto the page state and
    read here.

    DIFFERENCE: the rotation flag alone, over one page state.
    """
    from socr.core.manifest import table_floor_text_for_source

    upright = _floor_state(tmp_path / "upright")
    rotated = _floor_state(tmp_path / "rotated")
    setattr(rotated, flag, True)

    recovered = table_floor_text_for_source(upright, 1, upright.native_text, prose_recovery=True)
    refused = table_floor_text_for_source(rotated, 1, rotated.native_text, prose_recovery=True)

    assert _PROSE in recovered
    assert _PROSE not in refused, "a rotated page must not ship its text layer as prose"
    assert "failed: unverifiable table" in refused
    assert _ANSWER_RE.search(refused) is None


# ---------------------------------------------------------------------------
# 3a. The recovery runs ONLY where a detected box exists (owner, 2026-10-15).
# ---------------------------------------------------------------------------

#: A row of a BORDERLESS categorical table -- the shape the corpus replay caught
#: shipping as loose text on 41 pages ("Call FE Yes Yes Yes Yes Yes").
_CATEGORICAL_ROW = re.compile(r"\b(?:Yes|No)\b(?:\W+\b(?:Yes|No)\b){1,}", re.I)
_BORDERLESS_PROSE = "This paragraph of ordinary prose must survive the floor intact."


def _borderless_pdf(path: Path) -> Path:
    """Prose plus a BORDERLESS word-cell table: no rules, so no detected box.

    Verified against the real detector: `detected_table_count == 0`, boxes empty.
    This is the population the gate exists for -- 430 of the 674 pages the
    ungated recovery gave prose to had no box at all, and 41 of those shipped a
    row like this flattened, with its five columns collapsed into one line.
    """
    fitz = pytest.importorskip("fitz")
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 60), _BORDERLESS_PROSE)
    page.insert_text((72, 74), "It continues on a second line with no printed value anywhere.")
    rows = [
        ("Call FE", ("Yes", "Yes", "Yes", "Yes", "Yes")),
        ("Firm FE", ("Yes", "Yes", "No", "Yes", "No")),
        ("Controls", ("No", "Yes", "No", "Yes", "No")),
    ]
    for r, (label, cells) in enumerate(rows):
        page.insert_text((80, 140 + r * 18), label, fontsize=9)
        for c, cell in enumerate(cells):
            page.insert_text((260 + c * 48, 140 + r * 18), cell, fontsize=9)
    page.insert_text((72, 230), "A closing paragraph, also free of any printed digit at all.")
    doc.save(str(path))
    doc.close()
    return path


def test_a_page_with_no_detected_box_keeps_the_bare_marker(tmp_path: Path) -> None:
    """The gate, and the harm it exists to stop.

    DIFFERENCE: the page's detected boxes, nothing else. The ruled fixture has
    two and recovers its prose; the BORDERLESS fixture has none and keeps
    today's bare marker.

    Dropping the gate makes the borderless arm ship
    "Call FE Yes Yes Yes Yes Yes" -- the table's own row as loose text, five
    columns collapsed, an answer under no heading at all. That is the mutant the
    owner asked for, and this is the test it must fail.
    """
    import fitz

    from socr.core.document import DocumentHandle
    from socr.core.manifest import table_floor_text_for_source
    from socr.core.state import DocumentState

    def _state(pdf: Path):
        state = DocumentState(handle=DocumentHandle.from_path(pdf))
        ps = state.pages[1]
        doc = fitz.open(pdf)
        ps.native_text = doc[0].get_text()
        ps.native_words = doc[0].get_text("words")
        doc.close()
        ps.is_born_digital = True
        # A real page of this shape carries has_tables=True with NO box:
        # Barrot-Sauvagnat p25 floored as structure_class with
        # detected_table_count == 0, because has_tables comes from lane
        # co-occupancy rather than from the bbox pass.
        ps.has_tables = True
        ps.d3_floor_png_ref = "![Failed table page 1](figures/failed_table_p1.png)"
        return ps

    boxed_dir = tmp_path / "boxed"
    boxed_dir.mkdir(parents=True, exist_ok=True)
    boxed = _state(_p77_like(boxed_dir / "doc.pdf"))
    boxed.detected_table_count = 2
    boxed.detected_table_bboxes = [(76.0, 96.0, 516.0, 162.0), (76.0, 216.0, 516.0, 326.0)]

    borderless = _state(_borderless_pdf(tmp_path / "bare" / "doc.pdf"))
    borderless.detected_table_count = 0
    borderless.detected_table_bboxes = []
    # The fixture must really carry the harmful row, or the mutant proves nothing.
    assert _CATEGORICAL_ROW.search(borderless.native_text)

    recovered = table_floor_text_for_source(boxed, 1, boxed.native_text, prose_recovery=True)
    gated = table_floor_text_for_source(borderless, 1, borderless.native_text, prose_recovery=True)

    assert _PROSE in recovered
    assert _ANSWER_RE.search(recovered) is None

    # The gated arm is today's floor: the marker, the image, and nothing else.
    assert _BORDERLESS_PROSE not in gated
    assert _CATEGORICAL_ROW.search(gated) is None
    assert "failed: unverifiable table" in gated
    assert "unverified extraction" not in gated and "unverified scan" not in gated


# ---------------------------------------------------------------------------
# 3a1. Round 4: a rotated page, and a two-column page, are refused.
# ---------------------------------------------------------------------------


def test_a_rotated_page_is_refused_because_its_boxes_are_in_another_space(
    tmp_path: Path,
) -> None:
    """#1144 round 4 P0 (Fable, PR #1158). Reproduced before fixing.

    ``native_words`` are ``get_text("words")`` -- UNROTATED -- while
    ``detected_table_bboxes`` come from ``find_tables()`` in ROTATED space. On
    a /Rotate 90 page the box rule compares two different spaces and matches
    nothing: measured on this fixture, boxes at x=516..626 while the answers sit
    at x=400, and the Yes/No grid shipped as prose. Both earlier rotation flags
    stay False because the TEXT is upright; only the page is turned.

    DIFFERENCE: ``page_rotation`` alone, over one page state. A mutant dropping
    the refusal ships the grid and fails this.
    """
    import fitz

    from socr.core.manifest import table_floor_text_for_source

    # Both arms carry the fixture's own, correct boxes; only ``page_rotation``
    # differs, so the rotation is the single variable. (A contrived arm with the
    # boxes moved into rotated space is not used: it changes the page's apparent
    # layout as well, and the two-column refusal then fires instead, which would
    # make this test pass for the wrong reason.)
    ps = _floor_state(tmp_path / "rot")
    rotated = _floor_state(tmp_path / "rot2")
    rotated.page_rotation = 90

    upright = table_floor_text_for_source(ps, 1, ps.native_text, prose_recovery=True)
    turned = table_floor_text_for_source(rotated, 1, rotated.native_text, prose_recovery=True)

    # The upright arm recovers the page's prose; the rotated arm keeps the bare
    # marker. (The grid's own rows no longer ship in EITHER arm: round 4's
    # categorical-row rule withholds them whatever the geometry says, which is
    # why this pins the rotation difference rather than the original leak.)
    assert _PROSE in upright
    assert _PROSE not in turned
    assert _ANSWER_RE.search(turned) is None
    assert "failed: unverifiable table" in turned
    assert fitz is not None  # the fixture needs pymupdf importable


@pytest.mark.parametrize("rotation", [90, 180, 270])
def test_a_real_rotated_pdf_is_refused_end_to_end(tmp_path: Path, rotation: int) -> None:
    """#1144 round 5 (Fable P1): the rotation WIRING, not a hand-set flag.

    The earlier test set ``page_rotation`` directly, so mutants on the two
    carrying sites -- the stamp in ``born_digital._assess_page`` and the copy in
    ``state.apply_born_digital`` -- and on the ``% 360`` comparison all SURVIVED.
    This drives a real /Rotate 90 PDF through the production path:
    ``detect_page`` -> ``apply_born_digital`` -> the floor. Nothing is set by
    hand, so breaking any link in the chain fails this.

    DIFFERENCE: ``set_rotation(rotation)`` on the same page, nothing else.
    Parametrised over all three rotations (#1144 round 7, Fable P2) so an
    ``== 90`` comparison dies here; the corpus has 6 pages at /Rotate 270.
    """
    import fitz

    from socr.core.born_digital import BornDigitalDetector
    from socr.core.document import DocumentHandle
    from socr.core.manifest import table_floor_text_for_source
    from socr.core.state import DocumentState

    def _ship(rotation: int) -> tuple[str, int]:
        work = tmp_path / f"rot{rotation}"
        work.mkdir(parents=True, exist_ok=True)
        pdf = _p77_like(work / "doc.pdf")
        if rotation:
            doc = fitz.open(pdf)
            doc[0].set_rotation(rotation)
            doc.saveIncr()
            doc.close()
        assessment = BornDigitalDetector().detect(pdf)
        state = DocumentState(handle=DocumentHandle.from_path(pdf))
        state.apply_born_digital(assessment)
        ps = state.pages[1]
        doc = fitz.open(pdf)
        ps.native_words = doc[0].get_text("words")
        doc.close()
        ps.d3_floor_png_ref = "![Failed table page 1](figures/failed_table_p1.png)"
        body = table_floor_text_for_source(ps, 1, ps.native_text, prose_recovery=True)
        return body, int(getattr(ps, "page_rotation", -1))

    upright, upright_rot = _ship(0)
    turned, turned_rot = _ship(rotation)

    # The wiring itself: the value reached PageState from the real detector.
    assert upright_rot == 0
    assert turned_rot == rotation, (
        "page_rotation did not reach PageState through apply_born_digital"
    )
    # And the floor acted on it.
    assert _PROSE in upright
    assert _PROSE not in turned
    assert _ANSWER_RE.search(turned) is None
    assert "failed: unverifiable table" in turned


def test_the_box_gate_fails_closed_when_count_and_boxes_disagree(tmp_path: Path) -> None:
    """#1144 round 7 (repo reviewer, three times; cubic P1 4284621795).

    The gate asked only whether the box list was non-empty, which FAILS OPEN on
    the very disagreement that routes a page into this branch:
    ``len(detected_bboxes) != detected_count`` is one of the refusal paths, so a
    page with count=2 and ONE box arrives with a non-empty list and its unboxed
    table's word cells ship as prose.

    DIFFERENCE: ``detected_table_count`` alone, with the SAME single box.
    """
    from socr.core.manifest import table_floor_text_for_source

    qa_box = (76.0, 216.0, 516.0, 326.0)

    agreeing = _floor_state(tmp_path / "agree")
    agreeing.detected_table_count = 1
    agreeing.detected_table_bboxes = [qa_box]
    shipped = table_floor_text_for_source(agreeing, 1, agreeing.native_text, prose_recovery=True)

    disagreeing = _floor_state(tmp_path / "disagree")
    disagreeing.detected_table_count = 2  # the detector saw two; only one box survived
    disagreeing.detected_table_bboxes = [qa_box]
    refused = table_floor_text_for_source(
        disagreeing, 1, disagreeing.native_text, prose_recovery=True
    )

    assert _PROSE in shipped
    assert _PROSE not in refused, "a page whose box count disagrees must fail closed"
    assert _ANSWER_RE.search(refused) is None
    assert "failed: unverifiable table" in refused


def test_the_order_gate_fails_closed_on_text_it_cannot_tokenise(tmp_path: Path) -> None:
    """#1144 round 7: non-Latin text normalised to NOTHING and shipped unchecked.

    ``_order_tokens`` used an ASCII class, so Cyrillic, Greek and CJK produced an
    empty token list and the gate returned True for anything. Measured before
    fixing: a REVERSED Cyrillic sentence passed its own ground truth, and a CJK
    body passed against unrelated ground truth.

    DIFFERENCE: the same Cyrillic sentence, in order vs reversed.
    """
    from socr.core.manifest import _order_tokens, _prose_follows_native_order

    native = "Инфляция выросла на процентов в текущем году"
    assert len(_order_tokens(native)) == 7, "the tokeniser must see non-Latin words"

    assert _prose_follows_native_order(native, native) is True
    assert _prose_follows_native_order(" ".join(reversed(native.split())), native) is False
    # Greek and CJK too, not only Cyrillic.
    greek = "Η πληθωρισμός αυξήθηκε σημαντικά"
    assert _prose_follows_native_order(" ".join(reversed(greek.split())), greek) is False
    # A body with LINES but no comparable tokens is unverifiable, not in order.
    assert _prose_follows_native_order("··· ---", "anything at all") is False
    # A body with no lines at all has nothing to ship; the caller governs.
    assert _prose_follows_native_order("", "anything at all") is True


# ---------------------------------------------------------------------------
# 3a1b. Round 5: recovered prose must be in the page's own reading order.
# ---------------------------------------------------------------------------


def test_recovered_prose_must_be_an_in_order_subsequence_of_the_page(
    tmp_path: Path,
) -> None:
    """#1144 round 5 (owner ruling, 2026-10-15), measured at 89 of 237 pages.

    The bands are y-only, so a two-column page welds halves of both columns into
    one line. DIFFERENCE: the same page text, shipped in the page's order vs
    interleaved. Ground truth is the page's own ``get_text()`` block order.

    bahdanau p3's real shape is reproduced literally: the left column's "model
    can be used in two ways." and the right column's "activation function. The
    update gate z selects" share a y-band, so the band line welds them.
    """
    from socr.core.manifest import _prose_follows_native_order

    native = (
        "model can be used in two ways. One way is to use\n"
        "the model to generate a target sequence given an\n"
        "activation function. The update gate z selects\n"
        "whether the hidden state is to be updated with\n"
    )
    in_order = "model can be used in two ways. One way is to use\nactivation function. The update gate z selects\n"
    interleaved = (
        "model can be used in two ways. One way is to use activation function. "
        "The update gate z selects\n"
        "the model to generate a target sequence given an whether the hidden "
        "state is to be updated with\n"
    )

    assert _prose_follows_native_order(in_order, native) is True
    assert _prose_follows_native_order(interleaved, native) is False
    # socr's own furniture is never part of the comparison.
    furniture = (
        "[page 3: unverified extraction — the paragraphs below are this page's own "
        "text layer; every numeric row is withheld]\n\n"
        "![Failed table page 3](figures/failed_table_p3.png)\n\n"
        "[page 3 failed: unverifiable table — see image]\n\n"
        "[page image](figures/failed_table_p3.png)\n\n" + in_order
    )
    assert _prose_follows_native_order(furniture, native) is True


def test_the_order_gate_refuses_the_page_and_keeps_the_marker(tmp_path: Path) -> None:
    """The gate at the floor, not only the predicate.

    DIFFERENCE: the page's ``native_text`` ground truth alone. With the
    fixture's real text the recovery ships; with a ground truth whose order the
    recovered bands cannot follow, the page keeps the bare marker. A mutant that
    drops the gate ships the out-of-order body and fails this.
    """
    from socr.core.manifest import table_floor_text_for_source

    ok = _floor_state(tmp_path / "ok")
    shipped = table_floor_text_for_source(ok, 1, ok.native_text, prose_recovery=True)

    scrambled = _floor_state(tmp_path / "bad")
    # Same words, reversed: nothing the bands ship can be an in-order
    # subsequence of this, so the gate must refuse.
    scrambled.native_text = " ".join(reversed((scrambled.native_text or "").split()))
    refused = table_floor_text_for_source(scrambled, 1, scrambled.native_text, prose_recovery=True)

    assert _PROSE in shipped
    assert _PROSE not in refused
    assert "failed: unverifiable table" in refused
    assert "unverified extraction" not in refused


# ---------------------------------------------------------------------------
# 3a2. A row of categorical CELLS is table content, whatever the geometry says.
# ---------------------------------------------------------------------------


def _band(*tokens: str, y: float = 100.0):
    return [
        (72.0 + 40.0 * i, y, 100.0 + 40.0 * i, y + 10.0, t, 0, 0, i) for i, t in enumerate(tokens)
    ]


@pytest.mark.parametrize(
    ("tokens", "is_row"),
    [
        # The measured corpus shapes.
        (("Clustered", "SE", "Yes", "Yes", "Yes", "Yes"), True),
        (("Industry", "FE", "Yes", "Yes", "Yes", "Yes"), True),
        (("Bond", "FE", "NO", "YES", "YES"), True),
        (("Firm", "controls", "no", "yes", "yes"), True),
        # Prose that merely CONTAINS the words. One cell is a sentence.
        (("and", "the", "answer", "is", "no."), False),
        (("Yes,", "it", "does", "apply", "here"), False),
        (("No", "single", "cell", "is", "a", "row"), False),
        # Two of them, but not adjacent: adjacency is what makes it a row.
        (("Controls", "Yes", "and", "Industry", "No"), False),
    ],
)
def test_a_categorical_row_is_two_or_more_adjacent_cells(tokens, is_row: bool) -> None:
    """The shape, both ways. Two ADJACENT cells, because two columns make a row.

    Measured on run 23192118: 361 bands across 147 pages match, and of ~90 read
    by hand every one is a regression table's own specification row. The
    non-matching cases here are the reason the rule needs adjacency and a run of
    two -- at one cell it would withhold any sentence ending "the answer is no".
    """
    from socr.tables.row_corroboration import bears_categorical_row

    assert bears_categorical_row(_band(*tokens)) is is_row


def test_a_categorical_row_is_withheld_even_with_no_box_over_it(tmp_path: Path) -> None:
    """The arce_bengui_bianchi p48 shape, which the box gate does not catch.

    That page has FOUR detected boxes, so it passes #1144's gate, but all four
    sit at y>=438 while the leaking table's row is at y=325. Boxes exist; they
    are not over the table that leaks.

    DIFFERENCE: the categorical-row rule alone, over one band list, with the
    boxes held fixed and deliberately placed away from the row. A mutant that
    disables the rule puts "Clustered SE Yes Yes Yes Yes" back in the prose set
    and fails this.
    """
    from socr.tables.row_corroboration import partition_prose_bands

    words = [
        *_band("A", "sentence", "of", "ordinary", "prose", y=100.0),
        *_band("Clustered", "SE", "Yes", "Yes", "Yes", "Yes", y=325.0),
        *_band("Another", "ordinary", "sentence", "here", y=600.0),
    ]
    # Boxes exist, and none of them covers y=325 -- the real page's geometry.
    boxes = [(59.9, 438.9, 562.6, 639.9), (95.6, 453.7, 302.5, 617.9)]

    prose = {
        " ".join(w[4] for w in band)
        for is_prose, band in partition_prose_bands(words, table_boxes=boxes)
        if is_prose
    }
    assert "A sentence of ordinary prose" in prose
    assert not any("Clustered" in line for line in prose), (
        "a categorical row must be withheld even where no box covers it"
    )
    # The band below the boxes is inside one, so the box rule still does its job.
    assert not any("Another ordinary sentence" in line for line in prose)


# ---------------------------------------------------------------------------
# 3b. The numeral class the withholding decision rests on.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("token", "counts"),
    [
        ("0", True),  # ASCII, the only class before #1144
        ("\U0001d7f6", True),  # MATHEMATICAL BOLD DIGIT ZERO -- the real corpus leak
        ("٥", True),  # ARABIC-INDIC DIGIT FIVE
        ("௧", True),  # TAMIL DIGIT ONE
        ("୧", True),  # ORIYA DIGIT ONE
        ("¹", False),  # SUPERSCRIPT ONE -- a footnote marker, not a value
        ("⁰", False),  # SUPERSCRIPT ZERO
        ("₁", False),  # SUBSCRIPT ONE
        ("½", False),  # VULGAR FRACTION ONE HALF -- the accepted residual
        ("A", False),
        ("", False),
    ],
)
def test_the_withholding_digit_class_is_unicode_decimal_not_ascii_nor_numeric(
    token: str, counts: bool
) -> None:
    """#1144: category Nd exactly -- wider than ASCII, narrower than isnumeric.

    The corpus replay over run 23192118's 1,794 floored pages found three pages
    whose recovered prose carried a printed value the ASCII class could not see
    (mathematical bold digits, and Tamil/Oriya digits in a garbled maths layer).
    Nd closes those. ``isnumeric()`` would also admit superscripts and vulgar
    fractions, which appear in ordinary prose as footnote markers, so adopting
    it would withhold a large share of genuine prose -- the loss this lane
    exists to stop. Both boundaries are pinned, so neither can drift.
    """
    from socr.tables.row_corroboration import bears_printed_numeral

    assert bears_printed_numeral(token) is counts


def test_widening_the_digit_class_can_only_withhold_more() -> None:
    """The direction check that licensed the widening.

    ``partition_prose_bands`` is the only live consumer, and a True there raises
    a band's numeral count, which moves ``is_prose`` toward False. So the prose
    set under the wider class must be a SUBSET of the prose set under the
    narrower one -- never a page shipping more.

    The word list is built directly rather than through a PDF: the base-14
    fonts cannot encode MATHEMATICAL BOLD DIGIT ZERO, so inserting it produced a
    page that did not contain it and the first version of this test could not
    tell the two classes apart. It said so and failed rather than passing
    vacuously; this is the fixed fixture.
    """
    import re as _re

    from socr.tables import row_corroboration as rc

    def _word(y: float, x: float, text: str):
        return (x, y, x + 8.0 * len(text), y + 10.0, text, 0, 0, 0)

    words = [
        _word(50.0, 72.0, "A"),
        _word(50.0, 90.0, "plain"),
        _word(50.0, 130.0, "sentence"),
        # One band whose only printed value is a mathematical bold digit.
        _word(80.0, 72.0, "Share"),
        _word(80.0, 120.0, "\U0001d7f6.\U0001d7fa"),
        _word(80.0, 180.0, "per"),
        _word(80.0, 210.0, "cent"),
        _word(110.0, 72.0, "Another"),
        _word(110.0, 130.0, "sentence"),
    ]

    def _prose(pattern) -> set[str]:
        real = rc._DIGIT_RE
        try:
            rc._DIGIT_RE = pattern
            return {
                " ".join(w[4] for w in band)
                for is_prose, band in rc.partition_prose_bands(words)
                if is_prose
            }
        finally:
            rc._DIGIT_RE = real

    ascii_only = _prose(_re.compile(r"[0-9]"))
    unicode_nd = _prose(_re.compile(r"\d"))

    assert unicode_nd < ascii_only, "the wider class must withhold strictly more, never ship more"
    assert any("Share" in line for line in ascii_only)
    assert not any("Share" in line for line in unicode_nd)
    # The genuine prose bands are untouched by the widening.
    assert {ln for ln in ascii_only if "Share" not in ln} == unicode_nd


# ---------------------------------------------------------------------------
# 3c. The banner must describe the PAGE TYPE, and every form must stay counted.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("born_digital", [True, False])
def test_the_banner_says_scan_only_for_a_scan(tmp_path: Path, born_digital: bool) -> None:
    """DIFFERENCE: ``is_born_digital`` alone, over one page state.

    Before #1144 this lane only ever served scans, so "unverified scan" was
    always true. It now serves structure-class floors on born-digital pages, and
    on those the word is false. Everything after the noun is identical, so the
    two banners make the same claim about the same evidence.
    """
    from socr.core.manifest import table_floor_text_for_source

    ps = _floor_state(tmp_path / f"bd-{born_digital}")
    ps.is_born_digital = born_digital
    body = table_floor_text_for_source(ps, 1, ps.native_text, prose_recovery=True)

    assert _PROSE in body
    if born_digital:
        assert "unverified extraction" in body
        assert "unverified scan" not in body
    else:
        assert "unverified scan" in body
        assert "unverified extraction" not in body
    # The claim itself does not change with the page type.
    assert "every numeric row is withheld" in body


def test_the_shipped_output_records_the_recovery_not_only_the_bytes(
    tmp_path: Path, monkeypatch
) -> None:
    """#1144 round 4 P1 (Fable). The structure-class ending recorded nothing.

    The D3 lane sets ``scanned_prose_recovered`` and the audit note on its
    output; this branch did not, so for every page #1144 adds the sidecar said
    ``scanned_prose_recovered=null`` with ``audit_notes=[]``, the document note
    and CLI line were silent, and ``_is_restored_prose_recovery`` was False on
    the page's own output -- which made the "every banner form accepted" fix
    inert exactly where it was added. Only the BYTES told a recovered-prose
    ERROR from a bare-marker ERROR.

    DIFFERENCE: the sidecar of a page that recovers prose vs one that does not,
    through the real ``process()``.
    """
    from socr.core.manifest import SCANNED_PROSE_RECOVERED_NOTE

    body, sidecar, _ = _ship(tmp_path, "recorded", monkeypatch)
    bare, bare_sidecar, _ = _ship(tmp_path, "bare", monkeypatch, drop_boxes=True)

    assert _PROSE in body
    # Both land on the shipped output, which the sidecar carries as
    # ``winning_output`` and which the document note and CLI line read via
    # ``_scanned_prose_recovered_pages``.
    won = sidecar.get("winning_output") or {}
    assert won.get("scanned_prose_recovered") is True
    assert SCANNED_PROSE_RECOVERED_NOTE in (won.get("audit_notes") or [])

    # The gated page ships the bare marker and must claim no recovery.
    assert _PROSE not in bare
    bare_won = bare_sidecar.get("winning_output") or {}
    assert bare_won.get("scanned_prose_recovered") is not True
    assert SCANNED_PROSE_RECOVERED_NOTE not in (bare_won.get("audit_notes") or [])


def test_every_banner_form_is_still_counted_as_a_prose_recovery() -> None:
    """A second banner form must not become invisible to the table metric.

    ``core.table_counts`` recognises a prose-recovery page by matching this
    banner, and its pattern was built from the single "unverified scan" form's
    literal text. The born-digital form #1144 adds would have gone UNCOUNTED on
    all 674 pages it affects. The forms are enumerated in
    ``manifest.PROSE_RECOVERED_FLAGS`` and this pins that every one of them --
    including any added later -- matches the counter's pattern.
    """
    from socr.core.manifest import (
        NATIVE_TEXT_FLAGS,
        PROSE_RECOVERED_FLAGS,
        native_text_flag,
        prose_recovered_flag,
    )
    from socr.core.table_counts import _PROSE_RECOVERY_RE

    assert len(PROSE_RECOVERED_FLAGS) >= 2
    for flag in PROSE_RECOVERED_FLAGS:
        assert _PROSE_RECOVERY_RE.search(flag.format(page_num=7)), flag
    # Both accessors return a member of their own enumeration, so a caller
    # cannot emit a form the guard above does not cover.
    for bd in (True, False):
        assert any(
            prose_recovered_flag(7, born_digital=bd) == f.format(page_num=7)
            for f in PROSE_RECOVERED_FLAGS
        )
        assert any(
            native_text_flag(7, born_digital=bd) == f.format(page_num=7) for f in NATIVE_TEXT_FLAGS
        )


# ---------------------------------------------------------------------------
# 4. The one piece of round 1 that survives: the resume hardening.
# ---------------------------------------------------------------------------


def test_the_escalation_refusal_survives_resume() -> None:
    """``tables_trust`` reads this kind to keep a page untrusted, and
    ``_escalate_table_page`` is skipped by a terminal resumed page, so the event
    must be replayed or a resumed run reports a page as more trusted than the
    first run did. Round 1 added it; it is kept.
    """
    from socr.core.tables_trust import TABLE_DISTRUST_KINDS
    from socr.pipeline.orchestrator import UnifiedPipeline

    replayed = UnifiedPipeline.resume_restore_kinds()
    assert "table_escalation_refused" in TABLE_DISTRUST_KINDS
    assert "table_escalation_refused" in replayed
    # Its GH-851 siblings, which were already replayed, still are.
    assert "table_escalation_timeout" in replayed
    assert "table_escalation_withheld" in replayed
