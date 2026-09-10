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
"""

import ast
import subprocess

import fitz
import pytest

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
    lines = [
        line.strip()
        for line in bd._assemble_prose_with_aligned_runs(page).splitlines()
        if line.strip()
    ]

    assert lines[lines.index("Bernard") - 1] == "Mr.", lines
    assert lines.index("PRESENT:") < lines.index("Mr."), (
        "the heading must keep its place before the pair it introduces"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
