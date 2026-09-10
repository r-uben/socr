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
def test_1990_gillum_band_clears_the_pitch_bound_but_the_runs_guards_refuse_it():
    """The measured case that isolates the second condition from the first.

    On 1990-11-13 p1 the band holding ``Mr.`` / ``Gillum, Deputy Assistant
    Secretary`` sits one row-pitch above the second run and starts in both of
    its lanes, so it passes the pitch bound. It is refused because appending it
    takes the value column's fill share from 0.50 to 0.60 against
    ``MEASURE_FILL_SHARE_MAX`` -- the run's own wrapped-body-prose
    discriminator. Pitch alone would have adopted it.

    This is the trade-off recorded on
    ``test_1990_11_13_alternate_secretary_rows_stay_adjacent_to_their_labels``:
    the guard misfires on this genuine sub-list, so refusing the band costs
    that page three correctly-paired rows. Pinned here so the cost is measured
    and visible, not inferred.
    """
    page = fitz.open(str(_FED_1990_11_13_MINUTES))[0]
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

    start, end, _merged = runs[-1]
    run_items = [it for band in bands[start : end + 1] for it in band]
    lanes = bd._run_column_lanes(run_items)
    pitch = bd._run_row_pitch(bands, start, end)

    candidate = bands[start - 1]
    picked = [it for it in candidate if bd._starts_in_a_lane(it["x0"], lanes)]
    step = abs(bd._band_center(candidate) - bd._band_center(bands[start]))

    assert any("Gillum" in it["text"] for it in picked), [it["text"] for it in candidate]
    assert step <= pitch, "the band is one row-pitch away; the pitch bound does NOT refuse it"
    assert (
        bd._try_aligned_run(
            run_items + picked,
            word_space_width,
            bd.ALIGNED_RUN_GAP_MAX_WORD_SPACES,
            word_width,
        )
        is None
    ), "the run's own guards must be what refuses this band"


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
