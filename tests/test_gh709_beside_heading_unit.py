"""GH-709: a heading printed BESIDE a pair joins it as one emission unit.

GH-704 adopts one declined label/value pair at each aligned-run boundary and
leaves every other line of that band where block order puts it. On 1977-11-15
that is right by accident: ``PRESENT:``, ``Mr.`` and ``Burns, Chairman`` are
all in baseline band 8 (the run starts at band 9), and ``PRESENT:`` happens to
come first in block order, so the heading prints above its own row. Print the
heading beside a LOWER boundary band instead -- ``STAFF:`` next to the first
staff row -- and the same rule prints the member above the heading that
introduces it. Every token survives; the section affiliation does not.

Astra's design note settles it, scoped to the one boundary band: when every
extra line in that band is a standalone heading printed beside the pair, the
band is adopted as ONE unit, rendered left to right, placed where the band sits
relative to the run. When any extra line fails that test the adoption ABSTAINS
and the pair keeps block order. A heading-bearing unit always ends the
continuation walk; it never authorises crossing a section boundary.
"""

import fitz
import pytest
from test_born_digital_aligned_runs import _FED_1977_11_15_MINUTES
from test_born_digital_aligned_runs import _FED_1990_11_13_MINUTES
from test_gh592_lane_scoped_emission import _bands_and_run
from test_gh592_lane_scoped_emission import _roster_with_leading_pairs
from test_gh706_section_heading_boundary import _staff_section_page

from socr.core import born_digital as bd


def _emitted(page: fitz.Page) -> list[str]:
    out = bd._assemble_prose_with_aligned_runs(page)
    assert out is not None
    return [line.strip() for line in out.splitlines() if line.strip()]


def _assert_heading_then_pair(lines: list[str], heading: list[str], value: str) -> None:
    """The round-5 contract: the heading whole and in order, its pair after it.

    Under the round-5 rule the extra never moves, so every line of the heading
    keeps the position the caller gave it, and the label and value are printed
    immediately after its last line. Checking both halves together is what
    distinguishes adoption from the abstention that also leaves a heading
    whole -- rounds 1 to 4 could satisfy the first half by refusing outright.
    """
    start = lines.index(heading[0])
    assert lines[start : start + len(heading) + 2] == [*heading, "Mr.", value], lines


def test_the_staff_heading_precedes_both_staff_members():
    """The GH-709 defect itself, on the fixture the issue was filed with.

    Astra's review of #706 asked for both names checked, not only the
    differential one: ``STAFF:`` must precede ``Burns`` (which #704 adopted and
    hoisted above it) and ``Gillum`` (which #706 already left in block order).
    The preceding roster must still come first -- the unit is placed at its own
    band's position relative to the run, which is after it.
    """
    lines = _emitted(_staff_section_page())

    assert lines.index("Mr. Corrigan, Vice Chairman of Committee") < lines.index("STAFF:")
    assert lines.index("STAFF:") < lines.index("Burns")
    assert lines.index("STAFF:") < lines.index("Gillum, Deputy Assistant Secretary")
    # Pinned as the exact adopted sequence, not as "Mr. precedes Burns". Block
    # order also puts a "Mr." immediately before "Burns" (the two labels print
    # together, then the two values), so the weaker assertion holds just as
    # well when the adoption abstains and nothing is recovered at all.
    _assert_heading_then_pair(lines, ["STAFF:"], "Burns")
    assert lines.count("STAFF:") == 1 and lines.count("Burns") == 1


@pytest.mark.skipif(not _FED_1977_11_15_MINUTES.exists(), reason="fed-01 corpus not present")
def test_1977_present_row_is_emitted_as_one_unit_without_duplicates():
    """The real control, now deliberate rather than accidental.

    Astra remeasured band 8: ``PRESENT:`` (x0 142.0), ``Mr.`` (214.0) and
    ``Burns, Chairman`` (243.0) share it, and the run starts at band 9. So
    ``PRESENT:`` is beside Burns, not above him, and the correct order this page
    has always produced came from block order rather than from a rule. It is now
    produced by the unit's own left-to-right ordering, and every line appears
    exactly once.
    """
    with fitz.open(str(_FED_1977_11_15_MINUTES)) as doc:
        page = doc[0]
        bands, runs, _ = _bands_and_run(page)
        band = bands[runs[0][0] - 1]
        assert [item["text"].strip() for item in sorted(band, key=lambda it: it["x0"])] == [
            "PRESENT:",
            "Mr.",
            "Burns, Chairman",
        ]
        lines = _emitted(page)

    assert lines.index("PRESENT:") < lines.index("Mr.") < lines.index("Burns, Chairman")
    assert lines[lines.index("Burns, Chairman") + 1] == "Mr. Volcker, Vice Chairman", (
        "the rest of the roster must follow the unit unchanged"
    )
    for text in ("PRESENT:", "Mr.", "Burns, Chairman"):
        assert lines.count(text) == 1, (text, lines)


@pytest.mark.skipif(not _FED_1990_11_13_MINUTES.exists(), reason="fed-01 corpus not present")
def test_1990_alternate_members_stay_adjacent_and_the_section_gap_still_stops():
    """The recall control: GH-709 must not cost #706's recovery.

    Kohn, Bernard and Gillum are recovered by the continuation walk, and the
    24.457pt gap above Kohn -- against the staff run's own 12.336pt pitch --
    must still be what stops it, so the alternate-member section above is never
    crossed.
    """
    with fitz.open(str(_FED_1990_11_13_MINUTES)) as doc:
        page = doc[0]
        bands, runs, _ = _bands_and_run(page)
        lines = _emitted(page)

    staff_run = runs[-1]
    pitch = bd._run_row_pitch(bands, staff_run[0], staff_run[1])
    gap = bd._band_center(bands[staff_run[0] - 1]) - bd._band_center(bands[staff_run[0] - 4])
    assert gap > pitch, (gap, pitch)

    for name in (
        "Kohn, Secretary and Economist",
        "Bernard, Assistant Secretary",
        "Gillum, Deputy Assistant Secretary",
    ):
        assert lines[lines.index(name) - 1] == "Mr.", (name, lines)
    assert (
        lines.index("Kohn, Secretary and Economist")
        < lines.index("Bernard, Assistant Secretary")
        < lines.index("Gillum, Deputy Assistant Secretary")
    )
    assert lines.index("and Boston, respectively") < lines.index("Kohn, Secretary and Economist")


def _paragraph_beside_the_boundary_page() -> fitz.Page:
    """The STAFF geometry with a multi-line prose column in place of the heading.

    One line of that column lands on the boundary band, wholly left of the label
    and baseline-aligned with it -- everything a standalone heading satisfies,
    except that its neighbours share its left edge. Adopting the pair would
    print that line inside the roster and leave the rest of its paragraph
    behind.
    """
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (72, 72),
        "Some ordinary running prose establishes the word space measurement here.",
        fontsize=10,
    )
    x = 90
    right = (
        x + fitz.get_text_length("Mr.", fontsize=10) + 1.2 * fitz.get_text_length(" ", fontsize=10)
    )
    page.insert_textbox(fitz.Rect(x, 100, 130, 200), "Mr.\nMr.\nMr.", fontsize=10)
    page.insert_textbox(
        fitz.Rect(right, 100, 560, 200),
        "Angell\nGuffey\nCorrigan, Vice Chairman of Committee",
        fontsize=10,
    )
    rows = sorted({word[1] for word in page.get_text("words") if word[4] == "Mr."})
    first = rows[-1] + (rows[-1] - rows[-2])
    page.insert_textbox(
        fitz.Rect(40, first, 150, first + 80),
        "an unrelated column\nof running prose",
        fontsize=10,
    )
    page.insert_textbox(fitz.Rect(x, first, 130, first + 80), "Mr.\nMr.", fontsize=10)
    page.insert_textbox(
        fitz.Rect(right, first, 560, first + 80),
        "Burns\nGillum, Deputy Assistant Secretary",
        fontsize=10,
    )
    return page


def test_a_line_of_an_unrelated_column_makes_the_adoption_abstain():
    """The ambiguity fallback, isolated.

    Nothing about the offending line's geometry differs from a heading's: it is
    wholly left of the label and shares its baseline. What differs is that other
    lines of its block start at the same x and would be left behind. The
    adoption abstains, so the pair keeps block order and the paragraph stays
    whole and in sequence.
    """
    lines = _emitted(_paragraph_beside_the_boundary_page())

    assert lines.index("an unrelated column") < lines.index("of running prose")
    assert lines.index("Mr. Corrigan, Vice Chairman of Committee") < lines.index(
        "an unrelated column"
    ), ("the run is emitted first, and nothing from the staff section joins it", lines)
    assert lines.index("of running prose") < lines.index("Burns"), (
        "abstention leaves the pair in block order, behind the whole paragraph",
        lines,
    )
    assert lines.count("Burns") == 1


def test_a_heading_genuinely_above_the_pair_is_left_in_its_own_band():
    """The scope limit: the unit never reaches into a neighbouring band.

    Here ``STAFF:`` is printed one band ABOVE the first staff row rather than
    beside it. That band is not the boundary band, so the heading is not fetched
    into the unit; it keeps block order, which already puts it before its
    members.
    """
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (72, 72),
        "Some ordinary running prose establishes the word space measurement here.",
        fontsize=10,
    )
    x = 90
    right = (
        x + fitz.get_text_length("Mr.", fontsize=10) + 1.2 * fitz.get_text_length(" ", fontsize=10)
    )
    page.insert_textbox(fitz.Rect(x, 100, 130, 200), "Mr.\nMr.\nMr.", fontsize=10)
    page.insert_textbox(
        fitz.Rect(right, 100, 560, 200),
        "Angell\nGuffey\nCorrigan, Vice Chairman of Committee",
        fontsize=10,
    )
    rows = sorted({word[1] for word in page.get_text("words") if word[4] == "Mr."})
    pitch = rows[-1] - rows[-2]
    page.insert_text((40, rows[-1] + pitch + 10.75), "STAFF:", fontsize=10)
    page.insert_textbox(fitz.Rect(x, rows[-1] + 2 * pitch, 130, 400), "Mr.\nMr.", fontsize=10)
    page.insert_textbox(
        fitz.Rect(right, rows[-1] + 2 * pitch, 560, 400),
        "Burns\nGillum, Deputy Assistant Secretary",
        fontsize=10,
    )
    lines = _emitted(page)

    assert lines.index("STAFF:") < lines.index("Burns")
    assert lines.count("STAFF:") == 1


def test_a_marker_printed_right_of_the_value_makes_the_adoption_abstain():
    """The witness for "wholly left of the label", isolated.

    Same STAFF geometry, with the extra line printed to the RIGHT of the value
    instead of left of the label. Its reading position within the row is then
    not established by the left-to-right ordering the unit relies on -- it could
    be a trailing note, a page marker, a second column's first line -- so the
    adoption abstains and the pair keeps block order. Without the condition the
    line is swept into the run's group on geometry alone.
    """
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (72, 72),
        "Some ordinary running prose establishes the word space measurement here.",
        fontsize=10,
    )
    x = 90
    right = (
        x + fitz.get_text_length("Mr.", fontsize=10) + 1.2 * fitz.get_text_length(" ", fontsize=10)
    )
    page.insert_textbox(fitz.Rect(x, 100, 130, 200), "Mr.\nMr.\nMr.", fontsize=10)
    page.insert_textbox(
        fitz.Rect(right, 100, 560, 200),
        "Angell\nGuffey\nCorrigan, Vice Chairman of Committee",
        fontsize=10,
    )
    rows = sorted({word[1] for word in page.get_text("words") if word[4] == "Mr."})
    first = rows[-1] + (rows[-1] - rows[-2])
    page.insert_text((480, first + 10.75), "[note]", fontsize=10)
    page.insert_textbox(fitz.Rect(x, first, 130, first + 80), "Mr.\nMr.", fontsize=10)
    page.insert_textbox(
        fitz.Rect(right, first, 560, first + 80),
        "Burns\nGillum, Deputy Assistant Secretary",
        fontsize=10,
    )
    lines = _emitted(page)

    assert lines.index("Mr. Corrigan, Vice Chairman of Committee") < lines.index("Burns"), (
        "the run is still emitted before the staff section",
        lines,
    )
    # Pinned as the whole tail. "[note] before Burns" is true whether the pair
    # abstains or is relocated to sit directly after the note, and relocation
    # is the defect here: the note is printed to the RIGHT of the value, so
    # moving the pair behind it reverses their reading order.
    assert lines[lines.index("[note]") :] == [
        "[note]",
        "Mr.",
        "Mr.",
        "Burns",
        "Gillum, Deputy Assistant Secretary",
    ], lines
    assert lines.count("[note]") == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def _wrapped_heading_page(style: str) -> tuple[fitz.Page, str, str]:
    """Astra's reproducer: a TWO-line heading printed left of a boundary pair.

    Three real layouts, all built with PyMuPDF's own text placement so the
    geometry is measured rather than asserted:

    ``centered``
        ``ALTERNATE`` over ``MEMBERS``, centred in one textbox. Their left
        edges differ by 4.17pt against a 2.78pt word space, and PyMuPDF splits
        them into two separate blocks.
    ``indented``
        ``STAFF:`` over an indented ``advisers`` in one textbox.
    ``positioned``
        the same two lines placed with ``insert_text`` at explicitly different
        x, so the left-edge difference exceeds a word space by measurement.

    In every one of them the second line is the first line's continuation, so
    the first line may not be torn off and moved into the run's unit.
    """
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (72, 72),
        "Some ordinary running prose establishes the word space measurement here.",
        fontsize=10,
    )
    x = 90
    right = (
        x + fitz.get_text_length("Mr.", fontsize=10) + 1.2 * fitz.get_text_length(" ", fontsize=10)
    )
    page.insert_textbox(fitz.Rect(x, 100, 130, 200), "Mr.\nMr.\nMr.", fontsize=10)
    page.insert_textbox(
        fitz.Rect(right, 100, 560, 200),
        "Angell\nGuffey\nCorrigan, Vice Chairman of Committee",
        fontsize=10,
    )
    ys = sorted({w[1] for w in page.get_text("words") if w[4] == "Mr."})
    first = ys[-1] + ys[-1] - ys[-2]
    if style == "centered":
        page.insert_textbox(
            fitz.Rect(10, first, 88, first + 80), "ALTERNATE\nMEMBERS", fontsize=10, align=1
        )
        titles = ("ALTERNATE", "MEMBERS")
    elif style == "positioned":
        page.insert_text((40, first + 10.75), "STAFF:", fontsize=10)
        page.insert_text((50, first + 10.75 + (ys[-1] - ys[-2])), "advisers", fontsize=10)
        titles = ("STAFF:", "advisers")
    else:
        page.insert_textbox(
            fitz.Rect(40, first, 150, first + 80), "STAFF:\n    advisers", fontsize=10
        )
        titles = ("STAFF:", "advisers")
    page.insert_textbox(fitz.Rect(x, first, 130, first + 80), "Mr.\nMr.", fontsize=10)
    page.insert_textbox(
        fitz.Rect(right, first, 560, first + 80),
        "Burns\nGillum, Deputy Assistant Secretary",
        fontsize=10,
    )
    return page, titles[0], titles[1]


@pytest.mark.parametrize("style", ["centered", "indented", "positioned"])
def test_a_wrapped_headings_first_line_is_not_torn_off_its_continuation(style):
    """Astra's P1 against 545de02, reproduced in the repository suite.

    The shared-left-edge test that shipped in 545de02 called the first line of
    each of these headings standalone, adopted it into the pair's unit, and
    left the rest of the heading behind in block order. Widening the word-space
    tolerance would only move the indentation at which that happens; what a
    continuation cannot avoid is being printed over the same horizontal ground,
    so the rule now abstains on horizontal intersection with an adjacent band.
    """
    page, upper, lower = _wrapped_heading_page(style)
    if style == "positioned":
        heading = [w for w in page.get_text("words") if w[4] in ("STAFF:", "advisers")]
        assert abs(heading[0][0] - heading[1][0]) > bd._median_word_space_width(
            page.get_text("words")
        ), "the fixture must actually break the left-edge test"

    lines = _emitted(page)

    assert lines[lines.index(upper) + 1] == lower, lines
    assert lines.index(lower) < lines.index("Burns")


def test_a_centered_headings_first_line_is_refused_on_the_helper_itself():
    """Astra's focused probe, on the helper rather than on emission.

    The two centred lines are ``ALTERNATE`` (x0 19.55) and ``MEMBERS``
    (x0 23.72), and PyMuPDF puts them in two separate blocks, so neither a
    shared left edge nor shared block membership sees them as one heading.
    Their x-extents intersect and ``MEMBERS`` is printed BELOW, which is the
    direction a heading is read in, so the refusal needs no other evidence and
    no measurement of how far apart the two bands are.
    """
    doc = fitz.open()
    page = doc.new_page()
    page.insert_textbox(fitz.Rect(10, 100, 88, 160), "ALTERNATE\nMEMBERS", fontsize=10, align=1)
    records = []
    for bi, block in enumerate(page.get_text("dict")["blocks"]):
        for li, line in enumerate(block["lines"]):
            x0, y0, x1, y1 = line["bbox"]
            records.append(
                dict(
                    bi=bi,
                    li=li,
                    x0=x0,
                    x1=x1,
                    y0=y0,
                    y1=y1,
                    text="".join(s["text"] for s in line["spans"]),
                )
            )
    bands = bd._line_baseline_bands(records)
    assert [it["text"] for band in bands for it in band] == ["ALTERNATE", "MEMBERS"]
    upper = bands[0][0]
    label = dict(bi=9, li=0, x0=90, x1=105, y0=upper["y0"], y1=upper["y1"], text="Mr.")
    assert bd._beside_heading_lines([upper], label, bands, 0) is None


def _heading_with_continuation_page(wide: bool, leading: float) -> fitz.Page:
    """Astra's round-3 reproducer geometry.

    ``STAFF AND OTHER`` is printed beside the boundary pair and continued
    directly below by ``ATTENDEES...``, in the immediately adjacent band. Two
    dimensions vary and nothing else: whether the continuation crosses the
    label lane at x=240, and whether it is set at the roster's own pitch or at
    1.5 times it. Round 2 dismissed the wide continuation at 1.5 pitch --
    neither wholly left of the label nor within the pitch -- and split the
    heading around its first member.
    """
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (72, 72),
        "Some ordinary running prose establishes the word space measurement here.",
        fontsize=10,
    )
    x = 240
    right = (
        x + fitz.get_text_length("Mr.", fontsize=10) + 1.2 * fitz.get_text_length(" ", fontsize=10)
    )
    page.insert_textbox(fitz.Rect(x, 100, 280, 200), "Mr.\nMr.\nMr.", fontsize=10)
    page.insert_textbox(
        fitz.Rect(right, 100, 560, 200),
        "Angell\nGuffey\nCorrigan, Vice Chairman of Committee",
        fontsize=10,
    )
    ys = sorted({w[1] for w in page.get_text("words") if w[4] == "Mr."})
    pitch = ys[-1] - ys[-2]
    first = ys[-1] + pitch
    page.insert_text((40, first + 10.75), "STAFF AND OTHER", fontsize=10)
    continuation = "ATTENDEES AT THE MEETING OF THE COMMITTEE" if wide else "ATTENDEES"
    page.insert_text((40, first + 10.75 + leading * pitch), continuation, fontsize=10)
    page.insert_textbox(fitz.Rect(x, first, 280, first + 100), "Mr.\n\n\nMr.", fontsize=10)
    page.insert_textbox(
        fitz.Rect(right, first, 560, first + 100),
        "Burns\n\n\nGillum, Deputy Assistant Secretary",
        fontsize=10,
    )
    return page


@pytest.mark.parametrize("leading", [1.0, 1.5])
@pytest.mark.parametrize("wide", [False, True])
def test_a_continuation_printed_below_the_heading_always_refuses_the_adoption(wide, leading):
    """Astra's four paired cases: below is below, whatever the shape.

    A heading is read downward, so a line intersecting it from the band below
    is always a possible continuation and there is no evidence that dismisses
    one. Neither of the two facts round 2 dismissed on -- that the line crosses
    the label lane, that it is set at 1.5 times the roster's pitch -- says
    anything about whose line it is.
    """
    page = _heading_with_continuation_page(wide, leading)
    bands, _runs, _ws = _bands_and_run(page)
    heading = next(
        i for i, b in enumerate(bands) if any(it["text"] == "STAFF AND OTHER" for it in b)
    )
    below = next(
        i for i, b in enumerate(bands) if any(it["text"].startswith("ATTENDEES") for it in b)
    )
    assert below == heading + 1, "the fixture must put the continuation in the adjacent band"
    continuation = "ATTENDEES AT THE MEETING OF THE COMMITTEE" if wide else "ATTENDEES"

    lines = _emitted(page)

    assert lines[lines.index("STAFF AND OTHER") + 1] == continuation, lines
    assert lines.index(continuation) < lines.index("Burns")


def test_a_short_paragraph_line_above_the_heading_does_not_affect_the_adoption():
    """What is printed above the heading stopped mattering in round 5.

    Rounds 2 to 4 each ruled on this line and each ruled differently, because
    each of them moved the heading and so had to decide whether the line above
    was part of it. Round 5 moves the pair instead, and the line above is never
    inspected: it stays where it is, the heading stays where it is, and the
    pair is printed after the heading. Its width, its edge and its stack are
    all irrelevant now, which is why the same fixture adopts whatever they are.
    """
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (72, 72),
        "Some ordinary running prose establishes the word space measurement here.",
        fontsize=10,
    )
    page.insert_text((72, 199), "It.", fontsize=10)
    x = 90
    right = (
        x + fitz.get_text_length("Mr.", fontsize=10) + 1.2 * fitz.get_text_length(" ", fontsize=10)
    )
    page.insert_textbox(fitz.Rect(30, 211, 88, 251), "PRESENT:", fontsize=10)
    page.insert_textbox(fitz.Rect(x, 211, 130, 251), "Mr.\nMr.", fontsize=10)
    page.insert_textbox(fitz.Rect(right, 211, 560, 251), "Bernard\nGillum", fontsize=10)
    page.insert_textbox(fitz.Rect(x, 240, 130, 400), "Mr.\n\nMr.\n\nMr.\n\nMr.", fontsize=10)
    page.insert_textbox(
        fitz.Rect(right, 240, 560, 400),
        "Angell\n\nGuffey\n\nSeger\n\nCorrigan, Vice Chairman of the Committee",
        fontsize=10,
    )
    short = next(w for w in page.get_text("words") if w[4] == "It.")
    heading = next(w for w in page.get_text("words") if w[4] == "PRESENT:")
    assert short[2] < 90, "the paragraph's last line must end before the label lane"
    assert short[0] < heading[2] and heading[0] < short[2], (
        "and it must still intersect the heading horizontally"
    )

    lines = _emitted(page)

    assert lines[lines.index("It.") + 1] == "PRESENT:", "the line above keeps its place"
    _assert_heading_then_pair(lines, ["PRESENT:"], "Bernard")


def _heading_above_the_pair_page(lines: list[str], centred: bool) -> fitz.Page:
    """A multi-line heading whose LAST line is printed beside the boundary pair.

    The 1977-11-15 shape -- heading and pair in one band, run below -- with the
    heading's earlier lines stacked directly above it. ``centred`` sets them in
    a box so that each line starts somewhere else, as a display heading does.
    """
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (72, 72),
        "Some ordinary running prose establishes the word space measurement here.",
        fontsize=10,
    )
    x = 90
    right = (
        x + fitz.get_text_length("Mr.", fontsize=10) + 1.2 * fitz.get_text_length(" ", fontsize=10)
    )
    if centred:
        page.insert_textbox(fitz.Rect(20, 175, 88, 255), "\n".join(lines), fontsize=10, align=1)
    else:
        page.insert_text((30, 186), lines[0], fontsize=10)
        page.insert_text((30, 199), lines[1], fontsize=10)
        page.insert_textbox(fitz.Rect(30, 211, 88, 251), lines[2], fontsize=10)
    page.insert_textbox(fitz.Rect(x, 211, 130, 251), "Mr.\nMr.", fontsize=10)
    page.insert_textbox(fitz.Rect(right, 211, 560, 251), "Bernard\nGillum", fontsize=10)
    page.insert_textbox(fitz.Rect(x, 240, 130, 400), "Mr.\n\nMr.\n\nMr.\n\nMr.", fontsize=10)
    page.insert_textbox(
        fitz.Rect(right, 240, 560, 400),
        "Angell\n\nGuffey\n\nSeger\n\nCorrigan, Vice Chairman of the Committee",
        fontsize=10,
    )
    return page


def test_a_display_headings_last_line_is_not_torn_off_the_lines_above_it():
    """A centred display heading keeps all three lines AND gets its member.

    Three centred lines, so no two of them start at the same x: ``ALTERNATE``
    (24.55), ``MEMBERS`` (28.72), ``BOARD`` (36.22) against a 2.78pt word
    space. Rounds 1 to 4 had to decide whether the two lines above ``BOARD``
    were part of the heading before they could move it, and abstained. Round 5
    does not move it, so the heading survives intact and the pair still lands
    beside it -- both halves of what GH-709 asked for, on a shape no earlier
    round could serve.
    """
    page = _heading_above_the_pair_page(["ALTERNATE", "MEMBERS", "BOARD"], centred=True)
    bands, _runs, word_space = _bands_and_run(page)
    starts = [band[0]["x0"] for band in bands[1:4]]
    assert all(abs(a - b) > word_space for a, b in zip(starts, starts[1:])), starts

    lines = _emitted(page)

    _assert_heading_then_pair(lines, ["ALTERNATE", "MEMBERS", "BOARD"], "Bernard")


def test_a_left_aligned_headings_last_line_is_not_torn_off_the_lines_above_it():
    """The same, when the heading is flush left instead of centred.

    Three lines at x0 30, so the last one shares its predecessors' edge. Round
    4 refused precisely because of that shared edge, reading the last line as
    belonging to the block above rather than standing beside the pair. Round 5
    has no such question to answer: nothing above the last line is moved, so
    the heading stays whole either way and the pair follows it.
    """
    page = _heading_above_the_pair_page(["STAFF AND", "OTHER FOLK", "PRESENT:"], centred=False)
    bands, _runs, word_space = _bands_and_run(page)
    starts = [band[0]["x0"] for band in bands[1:4]]
    assert all(abs(start - 30.0) <= word_space for start in starts), starts

    lines = _emitted(page)

    _assert_heading_then_pair(lines, ["STAFF AND", "OTHER FOLK", "PRESENT:"], "Bernard")


def test_a_lone_line_above_the_heading_does_not_affect_the_adoption():
    """One ambiguous line above the heading, and it no longer has to be judged.

    The page opens straight into the roster, so the line intersecting the
    heading from above is the page's first band. Whether it is a paragraph's
    only line or the heading's own first line was unanswerable, and rounds 3
    and 4 abstained on exactly that. Round 5 leaves it alone instead of
    classifying it, and adopts.
    """
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (72, 72),
        "Some ordinary running prose establishes the word space measurement here.",
        fontsize=10,
    )
    x = 90
    right = (
        x + fitz.get_text_length("Mr.", fontsize=10) + 1.2 * fitz.get_text_length(" ", fontsize=10)
    )
    page.insert_textbox(fitz.Rect(30, 211, 88, 251), "PRESENT:", fontsize=10)
    page.insert_textbox(fitz.Rect(x, 211, 130, 251), "Mr.\nMr.", fontsize=10)
    page.insert_textbox(fitz.Rect(right, 211, 560, 251), "Bernard\nGillum", fontsize=10)
    page.insert_textbox(fitz.Rect(x, 240, 130, 400), "Mr.\n\nMr.\n\nMr.\n\nMr.", fontsize=10)
    page.insert_textbox(
        fitz.Rect(right, 240, 560, 400),
        "Angell\n\nGuffey\n\nSeger\n\nCorrigan, Vice Chairman of the Committee",
        fontsize=10,
    )
    bands, _runs, _ws = _bands_and_run(page)
    prose, heading = bands[0][0], bands[1][0]
    assert heading["text"].startswith("PRESENT:")
    assert prose["x0"] < heading["x1"] and heading["x0"] < prose["x1"], (
        "the fixture must have the opening line intersect the heading"
    )

    lines = _emitted(page)

    assert lines[0] == prose["text"].strip(), "the ambiguous line keeps its place"
    _assert_heading_then_pair(lines, ["PRESENT:"], "Bernard")


@pytest.mark.parametrize("indent_middle", [True, False])
def test_a_three_line_heading_whose_last_line_is_indented_stays_whole(indent_middle):
    """Astra's round-3 reproducer, both halves of its pair, now both adopted.

    A display heading need not end on the edge it began with. ``STAFF`` (30),
    ``AND`` (30 or 40), ``OTHERS`` (40 beside the pair) is one heading either
    way. Round 3 tore the 30/30/40 half by reading its first two lines as an
    independent paragraph; round 4 stopped the tear by refusing the adoption
    outright. Round 5 needs neither reading. The heading is never moved, so
    both halves keep all three lines in order, and both get the pair after
    them -- the recovery GH-709 exists for, on the fixture that broke it twice.
    """
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (72, 72),
        "Some ordinary running prose establishes the word space measurement here.",
        fontsize=10,
    )
    x = 90
    right = (
        x + fitz.get_text_length("Mr.", fontsize=10) + 1.2 * fitz.get_text_length(" ", fontsize=10)
    )
    page.insert_textbox(fitz.Rect(x, 211, 130, 251), "Mr.\nMr.", fontsize=10)
    page.insert_textbox(fitz.Rect(right, 211, 560, 251), "Bernard\nGillum", fontsize=10)
    page.insert_textbox(fitz.Rect(x, 240, 130, 400), "Mr.\n\nMr.\n\nMr.\n\nMr.", fontsize=10)
    page.insert_textbox(
        fitz.Rect(right, 240, 560, 400),
        "Angell\n\nGuffey\n\nSeger\n\nCorrigan, Vice Chairman of the Committee",
        fontsize=10,
    )
    page.insert_text((30, 186), "STAFF", fontsize=10)
    page.insert_text((40 if indent_middle else 30, 199), "AND", fontsize=10)
    page.insert_textbox(fitz.Rect(40, 211, 88, 251), "OTHERS", fontsize=10)

    bands, _runs, _ws = _bands_and_run(page)
    indices = [
        next(i for i, band in enumerate(bands) if any(it["text"].strip() == title for it in band))
        for title in ("STAFF", "AND", "OTHERS")
    ]
    assert indices == list(range(indices[0], indices[0] + 3)), indices
    assert all(band[0]["x1"] < x for band in bands[indices[0] : indices[0] + 2]), (
        "the fixture's heading lines must stay left of the label lane"
    )

    lines = _emitted(page)

    _assert_heading_then_pair(lines, ["STAFF", "AND", "OTHERS"], "Bernard")


def _wide_display_heading_page(indent_last: bool = True) -> fitz.Page:
    """A display heading whose first two lines are as wide as prose lines.

    Two aligned lines at x0 30 running past the label lane at x0 200, then a
    last line printed beside ``Mr.``/``Bernard``. ``indent_last`` starts that
    line at x0 40, off the stack's edge, which is the counterexample to the
    third condition; ``False`` starts it at x0 30, on the edge, which is the
    ordinary full-measure paragraph whose last line happens to be short. The
    roster objects are written FIRST so that block order cannot mask a wrong
    adoption -- Astra's round-3 point.
    """
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (72, 72),
        "Some ordinary running prose establishes the word space measurement here.",
        fontsize=10,
    )
    x = 200
    right = (
        x + fitz.get_text_length("Mr.", fontsize=10) + 1.2 * fitz.get_text_length(" ", fontsize=10)
    )
    page.insert_textbox(fitz.Rect(x, 211, x + 40, 251), "Mr.\nMr.", fontsize=10)
    page.insert_textbox(fitz.Rect(right, 211, 560, 251), "Bernard\nGillum", fontsize=10)
    page.insert_textbox(fitz.Rect(x, 240, x + 40, 400), "Mr.\n\nMr.\n\nMr.\n\nMr.", fontsize=10)
    page.insert_textbox(
        fitz.Rect(right, 240, 560, 400),
        "Angell\n\nGuffey\n\nSeger\n\nCorrigan, Vice Chairman of the Committee",
        fontsize=10,
    )
    page.insert_text((30, 186), "STAFF AND OTHER ATTENDEES AT THE", fontsize=10)
    page.insert_text((30, 199), "NOVEMBER MEETING OF THE FEDERAL", fontsize=10)
    page.insert_textbox(
        fitz.Rect(40 if indent_last else 30, 211, x - 2, 251),
        "OPEN MARKET COMMITTEE",
        fontsize=10,
    )
    return page


def test_a_block_whose_last_line_shares_its_edge_keeps_all_three_lines():
    """Three lines flush at x0 30, the last one beside the pair.

    Round 4 refused this on the shared edge, reasoning that a line starting
    where the lines above start is a line OF that block and must not be moved
    into the roster. It is, and it is not moved -- but under round 5 that
    costs nothing, because the pair comes to it. The block stays whole in
    order and the member is printed after it.
    """
    page = _wide_display_heading_page(indent_last=False)
    bands, _runs, word_space = _bands_and_run(page)
    extra = sorted(bands[3], key=lambda it: it["x0"])[0]

    assert extra["text"].strip() == "OPEN MARKET COMMITTEE"
    assert abs(extra["x0"] - bands[2][0]["x0"]) <= word_space, "the last line shares the edge"

    lines = _emitted(page)

    _assert_heading_then_pair(
        lines,
        [
            "STAFF AND OTHER ATTENDEES AT THE",
            "NOVEMBER MEETING OF THE FEDERAL",
            "OPEN MARKET COMMITTEE",
        ],
        "Bernard",
    )


def test_a_wide_display_headings_last_line_is_not_torn_off_the_lines_above_it():
    """Astra's round-4 counterexample, which round 5 turns from xfail to pass.

    Two aligned lines at x0 30 running past the label lane, then an indented
    last line beside the pair. Round 4 shipped this as a strict xfail because
    the heading supplies every piece of evidence the real 1977-11-15 page
    supplies, so no test on the lines above could tell them apart. Astra
    declined that residual, and the round-5 rule removes the question: the
    heading is not moved, so its first two lines cannot end up behind its
    third. The roster prints first, then the whole heading, then the member.
    """
    lines = _emitted(_wide_display_heading_page())

    assert lines.index("Mr. Corrigan, Vice Chairman of the Committee") < lines.index(
        "STAFF AND OTHER ATTENDEES AT THE"
    ), "the run keeps its own position; only the pair travels"
    _assert_heading_then_pair(
        lines,
        [
            "STAFF AND OTHER ATTENDEES AT THE",
            "NOVEMBER MEETING OF THE FEDERAL",
            "OPEN MARKET COMMITTEE",
        ],
        "Bernard",
    )


@pytest.mark.skipif(not _FED_1977_11_15_MINUTES.exists(), reason="Fed corpus not present")
def test_the_1977_present_row_keeps_its_emitted_positions():
    """The real-page acceptance criterion, pinned as indices rather than order.

    Astra measured the cost of the alternative: refusing every intersecting
    line above kept a synthetic heading whole but moved ``Burns, Chairman``
    from index 10 to index 21, past ``Mr. Roos`` and ``Mr. Wallich``, away
    from his own label. Round 5 keeps him at 10, directly after the label the
    page prints beside him, and directly before the run.
    """
    with fitz.open(_FED_1977_11_15_MINUTES) as doc:
        lines = _emitted(doc[0])

    assert lines[8:12] == [
        "PRESENT:",
        "Mr.",
        "Burns, Chairman",
        "Mr. Volcker, Vice Chairman",
    ], lines[:14]


def test_a_marker_column_makes_the_adoption_abstain():
    """The round-5 correction Astra's direction did not anticipate.

    Moving the pair rather than the extra is safe only when the boundary band
    is the ONLY band of its kind. The GH-592 marker fixture has two: declined
    rows carrying markers ``1`` and ``2`` in one left-margin column, of which
    the walk adopts only the boundary one. Relocating its pair to sit after
    ``2`` printed Gillum's row ahead of Bernard's, reversing two roster rows.

    So the band on the far side of the boundary must not look like another
    band of the same series: a candidate in the run's label lane AND a line
    out of both lanes intersecting our extra's column. Here it is both, so the
    page keeps block order entirely.
    """
    page = _roster_with_leading_pairs("Mr.", "Mr.")
    lines = _emitted(page)

    assert lines.index("Bernard") < lines.index("Gillum"), (
        "the two declined rows must not be reversed"
    )
    assert lines[1:7] == ["1", "2", "Mr.", "Mr.", "Bernard", "Gillum"], lines
