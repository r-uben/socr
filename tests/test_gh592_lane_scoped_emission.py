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
