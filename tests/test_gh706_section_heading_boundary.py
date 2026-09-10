"""GH-706: a continuation pair must not cross the heading that introduces it.

The #706 walk adopts declined label/value bands outward from an accepted run,
each on its own evidence measured against the ORIGINAL run. Astra's review of
28b78a1 found that a band can satisfy every one of those conditions and still
carry, out of both lanes, the printed content that gives the pair its section
-- a heading such as ``STAFF:``. The pair is then emitted inside the run's
group while the heading stays behind in block order, so a member is printed
above the heading that introduces it. No token is lost; the section
affiliation is.

The fix is conservative: a pair earns a continuation place only when it is
alone in its band, and the walk does not continue past a band that carried
anything else. The band immediately at the run boundary keeps GH-704's
behaviour, which is separately reviewed and unchanged here.

What the continuation is for, stated plainly: it deliberately recovers pairs
that the run-level fill-share statistic declined. Fill-share measures the
distribution of right-column widths across a whole candidate window, so a
window mixing long role-bearing names with short ones is refused wholesale,
including its pair-only rows. Each recovered pair still has to satisfy, on its
own, the ORIGINAL run's lane starts, row pitch, gap and whole-label
vocabulary, and to bring no other content in its band.
``test_a_synthetic_pair_only_continuation_crosses_two_bands`` isolates that:
the same roster is refused by ``_try_aligned_run`` with fill-share checking and
accepted with only that statistic disabled.
"""

import ast
import subprocess

import fitz
import pytest
from test_born_digital_aligned_runs import _FED_1977_11_15_MINUTES
from test_born_digital_aligned_runs import _FED_1990_11_13_MINUTES
from test_gh592_lane_scoped_emission import _bands_and_run

from socr.core import born_digital as bd

_BASE_COMMIT = "223a171"


def _staff_section_page() -> fitz.Page:
    """A roster, then a staff list at the roster's own pitch, headed ``STAFF:``.

    Astra's reproducer geometry. The heading is printed beside the FIRST staff
    row, out of both lanes, which is what stops ``_find_aligned_runs`` from
    absorbing the staff section into the run and what makes the section
    relationship visible on the page.
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
    # The staff list starts exactly one roster pitch below the roster, so the
    # continuation's distance bound cannot be what refuses it.
    word_rows = sorted({w[1] for w in page.get_text("words") if w[4] == "Mr."})
    pitch = word_rows[-1] - word_rows[-2]
    first = word_rows[-1] + pitch
    # insert_text takes a baseline; an existing word's y0 sits fontsize*1.075 above it.
    page.insert_text((40, first + 10.75), "STAFF:", fontsize=10)
    page.insert_textbox(fitz.Rect(x, first, 130, first + 80), "Mr.\nMr.", fontsize=10)
    page.insert_textbox(
        fitz.Rect(right, first, 560, first + 80),
        "Burns\nGillum, Deputy Assistant Secretary",
        fontsize=10,
    )
    return page


def _base_assembler():
    """The merged base's assembler, loaded from git without touching the tree."""
    try:
        source = subprocess.check_output(
            ["git", "show", f"{_BASE_COMMIT}:src/socr/core/born_digital.py"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    node = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == "_assemble_prose_with_aligned_runs"
    )
    env = dict(vars(bd))
    exec(compile(ast.Module(body=[node], type_ignores=[]), "base_assembler.py", "exec"), env)
    return env["_assemble_prose_with_aligned_runs"]


def test_continuation_does_not_hoist_staff_before_its_heading():
    """Astra's differential reproducer, verbatim in intent.

    The base already misplaces the FIRST staff row, which the immediate-band
    rule adopts; that is a GH-704 residual and is not asserted here. What must
    hold is that #706 does not extend the misplacement to a member the base
    positioned correctly.
    """
    page = _staff_section_page()
    out = bd._assemble_prose_with_aligned_runs(page)
    assert out is not None

    base = _base_assembler()
    if base is not None:
        old = base(page)
        assert old.index("STAFF:") < old.index("Gillum"), (
            "the differential is only meaningful if the base positioned Gillum correctly"
        )
    assert out.index("STAFF:") < out.index("Gillum")


def test_a_continuation_band_carrying_out_of_lane_content_is_not_adopted():
    """The witness for the new condition, isolated from every other one.

    The Gillum band is pair-only and satisfies the distance, lane, gap and
    label-vocabulary conditions; the walk reaches it only because the Burns
    band before it was adopted. It is refused solely because the Burns band
    also carried ``STAFF:``, so the walk stopped there.
    """
    page = _staff_section_page()
    lines = [line.strip() for line in bd._assemble_prose_with_aligned_runs(page).splitlines()]

    assert "Burns" in lines and "Gillum, Deputy Assistant Secretary" in lines, lines
    assert lines.index("STAFF:") < lines.index("Gillum, Deputy Assistant Secretary")
    assert lines.index("Mr. Corrigan, Vice Chairman of Committee") < lines.index("STAFF:"), (
        "the run's own rows must still be emitted before the heading"
    )


def test_a_boundary_band_carrying_a_heading_is_still_adopted():
    """GH-704's immediate-band rule is unchanged by this branch.

    1977-11-15's ``PRESENT:`` / ``Mr.`` / ``Burns, Chairman`` band is exactly a
    boundary band with out-of-lane content, and there the heading precedes the
    pair in block order rather than following it, so adopting the pair does not
    move it across the heading. Adoption there must still happen; only the walk
    past such a band stops. Asserted on synthetic geometry so it holds without
    the Fed corpus, and pinned on the real page in the aligned-run suite.
    """
    doc = fitz.open()
    page = doc.new_page()
    # The opening paragraph runs to the band immediately above the heading, as
    # it does on 1977-11-15 (four lines left-aligned at x0 107-108, ending in
    # "1977, at 9:30 a.m."). It is TWO lines here for the same reason: GH-709
    # dismisses a line intersecting the heading from above only when it
    # continues a left-aligned stack the heading is not part of, and a one-line
    # preamble is not a stack. Modelling the real page's paragraph as a single
    # floating line made this fixture ask for an adoption the real page's
    # geometry never asks for.
    page.insert_text(
        (72, 72),
        "Some ordinary running prose establishes the word space measurement here.",
        fontsize=10,
    )
    page.insert_text(
        (72, 199), "It carries on to the line just above the roster's heading.", fontsize=10
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
    lines = [
        line.strip()
        for line in bd._assemble_prose_with_aligned_runs(page).splitlines()
        if line.strip()
    ]

    assert lines[lines.index("Bernard") - 1] == "Mr.", lines
    assert lines.index("PRESENT:") < lines.index("Mr."), (
        "the heading must keep its place before the pair it introduces"
    )


def _mixed_width_roster_page() -> tuple[fitz.Page, list[str]]:
    """Astra's synthetic positive: long roles first, short names after.

    Every row is a plain two-line ``Mr.`` / name pair at one of two constant
    lane starts. Nothing is out of lane anywhere, so no band carries extra
    content. The only thing that varies is the right column's width, which is
    exactly what the run-level fill-share statistic measures.
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
    names = [
        "Alpha, Deputy Assistant Secretary",
        "Bravo, Deputy Assistant Secretary",
        "Delta, Deputy Assistant Secretary",
        "Eagle, Deputy Assistant Secretary",
        "Angell",
        "Guffey",
        "Seger",
        "Corrigan, Vice Chairman of the Committee",
    ]
    baselines = [100 + 13 * i if i < 2 else 126 + 14 * (i - 2) for i in range(len(names))]
    # The two columns are written as separate passes, as on a real page: each
    # becomes its own block, and the bands are formed by baseline alignment.
    for baseline in baselines:
        page.insert_text((x, baseline), "Mr.", fontsize=10)
    for baseline, name in zip(baselines, names):
        page.insert_text((right, baseline), name, fontsize=10)
    return page, names


def test_a_synthetic_pair_only_continuation_crosses_two_bands():
    """A corpus-free witness that the walk crosses more than one band.

    Astra's reproducer for the re-review of d3e750f. It also isolates WHY the
    two leading rows need the continuation at all: the run search refuses the
    whole roster under normal fill-share checking and accepts it when only that
    statistic is disabled (``word_width=0``), with geometry, text and every
    other guard unchanged.
    """
    page, names = _mixed_width_roster_page()
    bands, runs, word_space_width = _bands_and_run(page)
    roster = [item for band in bands[1:] for item in band]

    refused = bd._try_aligned_run(
        roster,
        word_space_width,
        bd.ALIGNED_RUN_GAP_MAX_WORD_SPACES,
        bd._median_word_width(page.get_text("words")) or 0.0,
    )
    without_fill_share = bd._try_aligned_run(
        roster, word_space_width, bd.ALIGNED_RUN_GAP_MAX_WORD_SPACES, 0.0
    )
    assert refused is None
    assert without_fill_share is not None, (
        "fill-share must be the only reason the whole roster is declined"
    )

    adoptable = []
    for start, end, _ in runs:
        items = [item for band in bands[start : end + 1] for item in band]
        lanes = bd._run_column_lanes(items)
        vocabulary = bd._run_label_vocabulary(items)
        pitch = bd._run_row_pitch(bands, start, end)
        boundary = bd._band_center(bands[start])
        crossed = 0
        for index in range(start - 1, -1, -1):
            band = bands[index]
            if len(band) != 2 or abs(bd._band_center(band) - boundary) > pitch:
                break
            if not bd._adoptable_pair(
                band, lanes, vocabulary, word_space_width, bd.ALIGNED_RUN_GAP_MAX_WORD_SPACES
            ):
                break
            crossed += 1
            boundary = bd._band_center(band)
        adoptable.append(crossed)
    assert max(adoptable) >= 2, adoptable

    lines = bd._assemble_prose_with_aligned_runs(page).splitlines()
    for name in names[:2]:
        assert lines[lines.index(name) - 1] == "Mr.", lines


def test_a_later_band_carrying_a_heading_is_refused_not_merely_last():
    """The witness that the two stop clauses are not one clause.

    ``_assemble_prose_with_aligned_runs`` refuses a non-pair-only band beyond
    the run boundary, and separately stops after adopting one at the boundary.
    Drop the first and the walk would still stop -- but only after adopting the
    offending pair, which is the #706 defect displaced by one band. Here the
    fill-share roster carries ``STAFF:`` beside its OUTER leading row, so that
    row is reached only through an adopted band and must be refused: its pair
    keeps block order after the run instead of being lifted into the group.
    """
    page, _names = _mixed_width_roster_page()
    baseline = min(word[1] for word in page.get_text("words") if word[4] == "Mr.")
    page.insert_text((40, baseline + 10.75), "STAFF:", fontsize=10)
    lines = [line.strip() for line in bd._assemble_prose_with_aligned_runs(page).splitlines()]

    assert lines[lines.index("Bravo, Deputy Assistant Secretary") - 1] == "Mr.", (
        "the pair-only band before the heading must still be adopted"
    )
    assert lines.index("Alpha, Deputy Assistant Secretary") > lines.index(
        "Mr. Corrigan, Vice Chairman of the Committee"
    ), ("the outer band's pair must keep block order, not join the run's group", lines)


@pytest.mark.skipif(
    not (_FED_1977_11_15_MINUTES.exists() and _FED_1990_11_13_MINUTES.exists()),
    reason="fed-01 corpus not present",
)
def test_real_section_boundaries_are_separated_by_more_than_the_run_pitch():
    """The measured answer to "could a real section break slip under the pitch?".

    Astra looked for a heading-free, sub-pitch break between two distinct lists
    on the real pages and found none: on both 1977-11-15 and 1990-11-13 every
    band following an accepted run sits further away than that run's own row
    pitch, so the distance bound stops the walk there without needing to reason
    about section membership. Pinned as a measurement, not as a guarantee about
    documents outside this corpus.
    """
    for path in (_FED_1977_11_15_MINUTES, _FED_1990_11_13_MINUTES):
        with fitz.open(str(path)) as doc:
            bands, runs, _ = _bands_and_run(doc[0])
        assert runs, path.name
        for start, end, _ in runs:
            if end + 1 >= len(bands):
                continue
            pitch = bd._run_row_pitch(bands, start, end)
            gap = bd._band_center(bands[end + 1]) - bd._band_center(bands[end])
            assert gap > pitch, (path.name, start, end, gap, pitch)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
