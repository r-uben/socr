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
from test_gh706_section_heading_boundary import _staff_section_page

from socr.core import born_digital as bd


def _emitted(page: fitz.Page) -> list[str]:
    out = bd._assemble_prose_with_aligned_runs(page)
    assert out is not None
    return [line.strip() for line in out.splitlines() if line.strip()]


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
    assert lines[lines.index("Burns") - 1] == "Mr.", (
        "the adopted pair must stay adjacent inside the unit"
    )
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
    assert lines.index("[note]") < lines.index("Burns"), (
        "abstention leaves the boundary band in block order",
        lines,
    )
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


def test_a_centered_headings_first_line_is_refused_whatever_the_run_pitch():
    """Astra's focused probe, on the helper itself.

    The two centred lines are ``ALTERNATE`` (x0 19.55) and ``MEMBERS``
    (x0 23.72) and PyMuPDF puts them in two separate blocks, so neither a
    shared left edge nor shared block membership sees them as one heading.
    Their x-extents intersect, and both lie wholly left of the label lane, so
    the refusal holds even with the run pitch set to zero -- it does not depend
    on how far apart the two bands are.
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

    assert bd._beside_heading_lines([upper], label, bands, 0, 0.0) is None


def _wide_continuation_page() -> fitz.Page:
    """A heading whose SECOND line runs past the label lane.

    ``STAFF AND OTHER`` sits wholly left of the label lane; its continuation
    ``ATTENDEES AT THE MEETING`` wraps across that lane, at the roster's own
    leading. The continuation is therefore not itself "left of the pair", and
    only its proximity -- one band away, within the run's row pitch -- marks it
    as the same material.
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
    page.insert_text(
        (40, first + 10.75 + pitch), "ATTENDEES AT THE MEETING OF THE COMMITTEE", fontsize=10
    )
    page.insert_textbox(fitz.Rect(x, first, 280, first + 80), "Mr.\nMr.", fontsize=10)
    page.insert_textbox(
        fitz.Rect(right, first, 560, first + 80),
        "Burns\nGillum, Deputy Assistant Secretary",
        fontsize=10,
    )
    return page


def test_a_continuation_that_crosses_the_label_lane_still_refuses_the_adoption():
    """The other half of the intersection test, and its only witness.

    A neighbour is dismissed only when it is BOTH outside the ground left of
    the pair AND further away than the run's row pitch. Here the continuation
    crosses the label lane, so the first condition alone would dismiss it and
    the heading's first line would be torn off; its proximity is what refuses
    the adoption.
    """
    lines = _emitted(_wide_continuation_page())

    assert (
        lines[lines.index("STAFF AND OTHER") + 1] == "ATTENDEES AT THE MEETING OF THE COMMITTEE"
    ), lines
    assert lines.index("ATTENDEES AT THE MEETING OF THE COMMITTEE") < lines.index("Burns")
