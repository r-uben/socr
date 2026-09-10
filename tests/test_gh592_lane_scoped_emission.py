"""GH-592 round 4 (Astra re-review of b8d5063): block membership is not evidence.

Round 3 scoped positional emission to every unconsumed line of every PyMuPDF
block a run contributes to. That is still authority granted by segmentation
rather than by reading order: resegment the identical lines, words and bboxes
so an unrelated LEFT prose column shares the label block and an unrelated
RIGHT prose column shares the name block, and two independent paragraphs get
interleaved line by line even though the four geometric guards in
``_try_aligned_run`` correctly refuse to merge them.

Round 4 repositions a declined line only when its own geometry says it belongs
with the repaired rows: its start falls inside one of the run's two column
lanes, and its baseline band is reachable from the run's band sequence without
crossing a band that contributes no such line.

The first two tests are the reviewer's reproducers, carried over verbatim in
substance. The third pins the band-adjacency half of the criterion, which the
reviewer's pair does not exercise.
"""

from __future__ import annotations

from copy import deepcopy

import fitz
import pytest
from test_born_digital_aligned_runs import _FED_1990_11_13_MINUTES
from test_born_digital_aligned_runs import _build_attendee_list_page
from test_gh592_scoped_positional_emission import _add_prose_columns, _prose_lines

from socr.core import born_digital as bd


def test_entangled_prose_columns_stay_column_major():
    """Prose sharing a real text box with the roster keeps its own order.

    Both columns are authored by ``insert_textbox`` so that the roster rows
    and the paragraph lines below them genuinely land in the same PyMuPDF
    blocks -- no resegmentation involved. The LEFT paragraph even starts at
    the label column's own x, so it is the value lane and the band walk, not
    block identity, that has to get this right.
    """
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (72, 72),
        "Some ordinary running prose establishes the word space measurement here.",
        fontsize=10,
    )
    right = (
        90 + fitz.get_text_length("Mr.", fontsize=10) + 1.2 * fitz.get_text_length(" ", fontsize=10)
    )
    indent = " " * 80
    page.insert_textbox(
        fitz.Rect(90, 104, 300, 500),
        "Mr.\nMr.\nMr.\n\nLEFT first paragraph line\nLEFT second paragraph line\n"
        "LEFT third paragraph line",
        fontsize=10,
    )
    page.insert_textbox(
        fitz.Rect(right, 104, 590, 500),
        "Angell\nGuffey\nCorrigan, Vice Chairman of Committee\n\n"
        + "\n".join(
            indent + s
            for s in (
                "RIGHT first paragraph line",
                "RIGHT second paragraph line",
                "RIGHT third paragraph line",
            )
        ),
        fontsize=10,
    )

    out = bd._assemble_prose_with_aligned_runs(page)

    assert out is not None, "the roster is a genuine run; the assembler must engage"
    baseline = [
        line.strip()
        for line in page.get_text("text").splitlines()
        if line.strip().startswith(("LEFT", "RIGHT"))
    ]
    actual = [
        line.strip() for line in out.splitlines() if line.strip().startswith(("LEFT", "RIGHT"))
    ]
    assert len(baseline) == 6
    assert actual == baseline


def _resegmented(page: fitz.Page):
    """The same page, with the prose columns folded into the roster's blocks.

    Every line, word, text and bbox is unchanged; only the ``(block, line)``
    identities are rewritten, consistently across the ``dict`` and ``words``
    extractions. This is the segmentation a differently-authored PDF of the
    same page could legitimately produce, and it must not change the output.
    """
    data = deepcopy(page.get_text("dict"))
    blocks = [b for b in data["blocks"] if b.get("type", 0) == 0]

    def block_text(block):
        return "\n".join(
            "".join(span.get("text", "") for span in line["spans"]) for line in block["lines"]
        )

    left_run = next(i for i, b in enumerate(blocks) if block_text(b).splitlines().count("Mr.") >= 3)
    right_run = next(i for i, b in enumerate(blocks) if "Greenspan" in block_text(b))
    left_prose = next(i for i, b in enumerate(blocks) if "LEFT paragraph" in block_text(b))
    right_prose = next(i for i, b in enumerate(blocks) if "RIGHT paragraph" in block_text(b))
    merges = {left_prose: left_run, right_prose: right_run}

    new_blocks: list[dict] = []
    remap: dict[int, tuple[int, int]] = {}
    for bi, block in enumerate(blocks):
        if bi in merges:
            continue
        new_index = len(new_blocks)
        remap[bi] = (new_index, 0)
        block = deepcopy(block)
        for source, target in merges.items():
            if target == bi:
                remap[source] = (new_index, len(block["lines"]))
                block["lines"].extend(deepcopy(blocks[source]["lines"]))
        block["bbox"] = tuple(
            fitz.Rect(block["lines"][0]["bbox"]) | fitz.Rect(block["lines"][-1]["bbox"])
        )
        block["number"] = new_index
        new_blocks.append(block)
    data["blocks"] = new_blocks

    new_words = []
    for word in page.get_text("words"):
        new_index, offset = remap[word[5]]
        new_words.append(tuple(word[:5]) + (new_index, word[6] + offset, word[7]))

    class ResegmentedPage:
        def get_text(self, kind):
            if kind == "dict":
                return data
            if kind == "words":
                return new_words
            return page.get_text(kind)

    return ResegmentedPage()


def test_block_entanglement_does_not_authorize_prose_interleave():
    """Sharing a block with a run must not reorder unrelated prose."""
    page = _build_attendee_list_page()
    _add_prose_columns(page)

    out = bd._assemble_prose_with_aligned_runs(_resegmented(page))

    assert out is not None, "the roster is still a genuine run after resegmentation"
    assert _prose_lines(out) == [
        f"{column} paragraph line {i}" for column in ("LEFT", "RIGHT") for i in range(1, 4)
    ], f"resegmentation alone reordered the independent paragraphs: {_prose_lines(out)}"


def test_lane_match_across_an_intervening_row_does_not_travel_with_the_run():
    """Lane membership alone is not enough -- the band walk must stop.

    A line that starts in the value lane but is separated from the run by an
    ordinary full-width paragraph row belongs to whatever follows that row,
    not to the roster. It must stay where block order puts it, i.e. AFTER the
    intervening row, not hoisted up beside the merged rows.
    """
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (72, 72),
        "Some ordinary running prose establishes the word space measurement here.",
        fontsize=10,
    )
    right = (
        90 + fitz.get_text_length("Mr.", fontsize=10) + 1.2 * fitz.get_text_length(" ", fontsize=10)
    )
    page.insert_textbox(fitz.Rect(90, 104, 130, 200), "Mr.\nMr.\nMr.", fontsize=10)
    page.insert_textbox(
        fitz.Rect(right, 104, 560, 200),
        "Angell\nGuffey\nCorrigan, Vice Chairman of Committee",
        fontsize=10,
    )
    page.insert_text(
        (72, 200), "An ordinary full width paragraph row intervenes here.", fontsize=10
    )
    page.insert_text((right, 220), "STRAY LANE LINE", fontsize=10)

    out = bd._assemble_prose_with_aligned_runs(page)

    assert out is not None
    lines = [line.strip() for line in out.splitlines() if line.strip()]
    intervening = next(i for i, line in enumerate(lines) if line.startswith("An ordinary full"))
    stray = lines.index("STRAY LANE LINE")
    merged = next(i for i, line in enumerate(lines) if line.startswith("Mr. Angell"))

    assert merged < intervening < stray, lines


def test_lane_aligned_paragraphs_are_not_interleaved(monkeypatch):
    """GH-592 round 5 (Astra re-review of df45222): the reviewer's reproducer.

    Two independent paragraphs begin far below a roster, at the roster's exact
    two lane x-starts. Under round 4 the outward walk crossed the blank gap --
    which holds no baseline band, so nothing stopped it -- and adopted all six
    prose lines one at a time, interleaving them. Adopting lines individually
    bypasses the very guards that declined their paragraph.

    The monkeypatch only observes the real search's return value and passes it
    through unchanged, to prove the prose is genuinely NOT in an accepted run.
    """
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (72, 72),
        "Some ordinary running prose establishes the word space measurement here.",
        fontsize=10,
    )
    label = "Representative"
    left = 90
    right = (
        left
        + fitz.get_text_length(label, fontsize=10)
        + 1.2 * fitz.get_text_length(" ", fontsize=10)
    )
    page.insert_textbox(fitz.Rect(left, 100, right - 2, 200), "\n".join([label] * 3), fontsize=10)
    page.insert_textbox(
        fitz.Rect(right, 100, 595, 200),
        "Michael Andrew Rutherford\nJonathan Edward Alexander\n"
        "Christopher James Montgomery, Vice Chairman of Committee",
        fontsize=10,
    )
    page.insert_textbox(
        fitz.Rect(left, 250, right - 5, 350), "LEFT one\nLEFT two\nLEFT three", fontsize=10
    )
    page.insert_textbox(
        fitz.Rect(right, 250, 590, 350),
        "RIGHT first independent paragraph line\nRIGHT second independent paragraph line\n"
        "RIGHT third independent paragraph line",
        fontsize=10,
    )

    accepted: list[str] = []
    original = bd._find_aligned_runs

    def record(*args):
        result = original(*args)
        accepted.extend(text for _start, _end, texts in result for text in texts)
        return result

    monkeypatch.setattr(bd, "_find_aligned_runs", record)
    out = bd._assemble_prose_with_aligned_runs(page)

    assert out is not None
    assert accepted and not any("LEFT" in s or "RIGHT" in s for s in accepted), (
        "the prose must be declined by the run search itself, or this proves nothing"
    )
    baseline = [
        line for line in page.get_text("text").splitlines() if line.startswith(("LEFT", "RIGHT"))
    ]
    actual = [line for line in out.splitlines() if line.startswith(("LEFT", "RIGHT"))]
    assert len(baseline) == 6
    assert actual == baseline


@pytest.mark.skipif(not _FED_1990_11_13_MINUTES.exists(), reason="fed-01 corpus not present")
def test_1990_measures_every_alternate_member_band_the_walk_crosses():
    """The measured real-page basis for the round-6 rule and where it stops.

    On 1990-11-13 p1 the ``Alternate Members`` sub-list is three consecutive
    declined bands above the second run. This pins, from the page itself, that
    each of the three stands on its OWN evidence -- its step from the band
    below it is inside the run's own row pitch and it forms an adoptable pair
    -- and that the fourth band up does not, so the walk stops there.
    """
    page = fitz.open(str(_FED_1990_11_13_MINUTES))[0]
    bands, runs, word_space_width = _bands_and_run(page)
    start, end, _merged = runs[-1]
    run_items = [it for band in bands[start : end + 1] for it in band]
    lanes = bd._run_column_lanes(run_items)
    pitch = bd._run_row_pitch(bands, start, end)
    vocabulary = bd._run_label_vocabulary(run_items)
    assert vocabulary == frozenset({"Mr."}), vocabulary

    boundary = bd._band_center(bands[start])
    for offset, surname in ((1, "Gillum"), (2, "Bernard"), (3, "Kohn")):
        band = bands[start - offset]
        center = bd._band_center(band)
        assert any(surname in it["text"] for it in band), [it["text"] for it in band]
        assert abs(center - boundary) <= pitch, (surname, abs(center - boundary), pitch)
        assert (
            bd._adoptable_pair(
                band,
                lanes,
                vocabulary,
                word_space_width,
                bd.ALIGNED_RUN_GAP_MAX_WORD_SPACES,
            )
            is not None
        ), surname
        boundary = center

    stop = bands[start - 4]
    assert abs(bd._band_center(stop) - boundary) > pitch, (
        "the walk must stop at the prose band above the sub-list"
    )
    assert (
        bd._adoptable_pair(
            stop,
            lanes,
            vocabulary,
            word_space_width,
            bd.ALIGNED_RUN_GAP_MAX_WORD_SPACES,
        )
        is None
    )


def _roster_with_leading_pairs(outer_label: str, inner_label: str) -> fitz.Page:
    """A 4-row roster with TWO declined label/value bands directly above it.

    The roster is double-spaced, so its measured row pitch (~29.5pt) leaves
    both leading bands reachable: the inner one ~14.2pt from the run, the
    outer one ~14.8pt further on. Each leading row also carries an out-of-lane
    marker in the left margin, which is what stops ``_find_aligned_runs`` from
    simply absorbing the two rows into the run (the marker column breaks the
    left/right bijection).

    Geometry is therefore identical whatever labels are passed; only the label
    TEXT varies, which is what makes the role condition testable in isolation.
    """
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (72, 72),
        "Some ordinary running prose establishes the word space measurement here.",
        fontsize=10,
    )
    right = (
        90 + fitz.get_text_length("Mr.", fontsize=10) + 1.2 * fitz.get_text_length(" ", fontsize=10)
    )
    page.insert_textbox(fitz.Rect(60, 211, 80, 251), "1\n2", fontsize=10)
    page.insert_textbox(fitz.Rect(90, 211, 130, 251), f"{outer_label}\n{inner_label}", fontsize=10)
    page.insert_textbox(fitz.Rect(right, 211, 560, 251), "Bernard\nGillum", fontsize=10)
    page.insert_textbox(fitz.Rect(90, 240, 130, 400), "Mr.\n\nMr.\n\nMr.\n\nMr.", fontsize=10)
    page.insert_textbox(
        fitz.Rect(right, 240, 560, 400),
        "Angell\n\nGuffey\n\nSeger\n\nCorrigan, Vice Chairman of the Committee",
        fontsize=10,
    )
    return page


def _bands_and_run(page: fitz.Page):
    """The page's baseline bands and its accepted runs."""
    words = page.get_text("words")
    word_space_width = bd._median_word_space_width(words)
    word_width = bd._median_word_width(words) or 0.0
    extents = bd._line_word_extents(words)
    flat = []
    blocks = [b for b in page.get_text("dict")["blocks"] if b.get("type", 0) == 0]
    for bi, block in enumerate(blocks):
        for li, line in enumerate(block.get("lines", []) or []):
            bbox = line.get("bbox")
            extent = extents.get((bi, li))
            if not bbox or extent is None:
                continue
            flat.append(
                {
                    "bi": bi,
                    "li": li,
                    "y0": bbox[1],
                    "y1": bbox[3],
                    "x0": extent[0],
                    "x1": extent[1],
                    "text": "".join(s.get("text", "") for s in line.get("spans", []) or []),
                }
            )
    flat.sort(key=lambda it: it["y0"])
    bands = bd._line_baseline_bands(flat)
    runs = bd._find_aligned_runs(bands, word_space_width, word_width)
    return bands, runs, word_space_width


def _emitted(page: fitz.Page) -> list[str]:
    out = bd._assemble_prose_with_aligned_runs(page)
    assert out is not None
    return [line.strip() for line in out.splitlines() if line.strip()]


def test_the_walk_stops_after_a_band_that_carried_out_of_lane_content():
    """GH-706 review fix, on the fixture that used to pin the opposite.

    Both leading bands are reachable and both hold a pair whose label the run
    observed, so before GH-706's heading fix the walk crossed both. Each band
    here also carries an out-of-lane marker in the left margin -- which is what
    makes the run search decline them in the first place -- and a band with
    content in neither lane may no longer continue the run. The boundary band
    is still adopted, under GH-704's separately reviewed immediate rule; the
    outer one keeps block order.

    This fixture withholds its marker bands from the run search with out-of-lane
    content, so it can no longer show a multi-band walk. A fixture that withholds
    them through fill-share instead still can:
    ``test_a_synthetic_pair_only_continuation_crosses_two_bands`` in
    ``test_gh706_section_heading_boundary.py``. Note also that this test is not a
    unique witness for either stop clause -- it passes with either one deleted --
    so it pins behaviour rather than proving a clause necessary.
    """
    lines = _emitted(_roster_with_leading_pairs("Mr.", "Mr."))
    assert lines[lines.index("Gillum") - 1] == "Mr.", lines
    assert lines.index("Bernard") > lines.index("Mr. Corrigan, Vice Chairman of the Committee"), (
        "the walk must not continue past the band that carried the marker"
    )


def test_an_adjacent_pair_whose_label_the_run_never_observed_is_refused():
    """The witness for the label-role condition, isolated from the geometry.

    Byte-identical fixture apart from the leading rows' label TEXT, which the
    run never observed. That difference alone must refuse the adjacent band,
    leaving both leading rows in block order. This is the evidence geometry
    cannot supply: an independent two-column sentence pair can match the lanes,
    the pitch and the gap exactly.
    """
    page = _roster_with_leading_pairs("Dr.", "Dr.")
    bands, runs, word_space_width = _bands_and_run(page)
    start, end, _merged = runs[0]
    run_items = [it for band in bands[start : end + 1] for it in band]
    assert (
        bd._adoptable_pair(
            bands[start - 1],
            bd._run_column_lanes(run_items),
            bd._run_label_vocabulary(run_items),
            word_space_width,
            bd.ALIGNED_RUN_GAP_MAX_WORD_SPACES,
        )
        is None
    )

    lines = _emitted(page)
    assert lines[lines.index("Gillum") - 1] == "Bernard", (
        "a label the run never observed is not evidence of the same role"
    )


def test_a_far_lane_aligned_pair_the_guards_would_accept_is_still_refused():
    """The witness for the pitch bound, isolated from the guard condition.

    A second label/name pair sits 250pt below the roster, at the roster's exact
    lane starts, and carries an extra out-of-lane marker in its row. The marker
    is what stops ``_find_aligned_runs`` from absorbing the pair into the run
    itself, and the adoption walk's lane filter drops it -- so the pair's lane
    subset DOES satisfy the run's own guards. Only the pitch bound refuses it:
    it is 250pt from the run's edge against a measured row pitch of ~15pt.

    Without this bound the far pair would be hoisted up beside the roster,
    which is the shape Astra's paragraph reproducer generalises.
    """
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (72, 72),
        "Some ordinary running prose establishes the word space measurement here.",
        fontsize=10,
    )
    right = (
        90 + fitz.get_text_length("Mr.", fontsize=10) + 1.2 * fitz.get_text_length(" ", fontsize=10)
    )
    page.insert_textbox(fitz.Rect(90, 104, 130, 260), "Mr.\nMr.\nMr.\nMr.", fontsize=10)
    page.insert_textbox(
        fitz.Rect(right, 104, 560, 260),
        "Angell\nGuffey\nSeger\nCorrigan, Vice Chairman of the Committee",
        fontsize=10,
    )
    page.insert_textbox(fitz.Rect(60, 400, 80, 470), "1\n2", fontsize=10)
    page.insert_textbox(fitz.Rect(90, 400, 130, 470), "Mr.\nMr.", fontsize=10)
    page.insert_textbox(fitz.Rect(right, 400, 560, 470), "Volcker\nPartee", fontsize=10)

    out = bd._assemble_prose_with_aligned_runs(page)

    assert out is not None
    lines = [line.strip() for line in out.splitlines() if line.strip()]
    assert "Mr. Volcker" not in lines, (
        "the far pair must not be merged into the run it is 250pt away from"
    )
    assert lines.index("1") < lines.index("Volcker"), lines
    assert lines.index("Mr. Corrigan, Vice Chairman of the Committee") < lines.index("1"), lines
