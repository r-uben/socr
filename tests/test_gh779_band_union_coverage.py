"""GH-779: band-split regions leave a duplicate plain-text copy.

GH-152 can split one wide table into a left-band and a right-band region at
a single x gutter (``reconstruct_table_regions``). Both bands cover the same
row range, partitioned only by x. ``born_digital.py``'s overlap-suppress
check (``extract_structured``) measures each region's coverage of a text
BLOCK independently, by design (#145/#718b): a combined index across
unrelated regions on the page would let one region delete a line because
its words happen to appear in a DIFFERENT table.

That per-region-only test has a gap: when a row's own PyMuPDF block/line
grouping spans BOTH bands (observed on real, unruled PDFs -- MuPDF's block
segmentation is geometric, not aware of two separate tabular environments),
neither narrow band clears ``_REGION_COVERAGE_DROP`` (50%) alone, so the
row's lines survive as a duplicate plain-text copy beneath the two
correctly-split markdown tables. Every value stays individually correct and
correctly attributed in both copies -- this is redundancy, not
misattribution -- but it is silent: nothing in status, metadata, or logs
says a suppression was skipped.

Fix: when no single region clears the coverage threshold, test the UNION of
the regions the block actually touches, but only when those regions are
geometrically band SIBLINGS -- disjoint in x, overlapping in y, exactly the
signature GH-152's split produces. That keeps the fix from reopening
#145/#718b: two unrelated regions (different y, or overlapping x) are never
unioned, so a line still cannot be dropped because its words merely appear
in some other table on the page.

Hermetic: PDFs are built in-process with ``fitz`` (``insert_text``), no
corpus content, no provider.
"""

from __future__ import annotations

import fitz
import pytest

from socr.core.born_digital import (
    _REGION_COVERAGE_DROP,
    BornDigitalDetector,
    _rect_coverage,
    _regions_are_band_siblings,
    _union_coverage,
)

CHAR_W = 6.0
FS = 10


def _row_text(cells: list[tuple[float, str]], x0: float) -> str:
    """A single courier-spaced string whose cells land near their target x.

    ONE ``insert_text`` call per row -- not per band -- so PyMuPDF's block
    segmentation groups the whole row (both bands) into a single
    block/line, exactly the shape that triggers GH-779 (confirmed against
    ``page.get_text("dict")`` while building this fixture: each such row
    becomes its own block spanning the full row width).
    """
    parts: list[str] = []
    cursor_chars = 0
    for target_x, text in cells:
        target_chars = round((target_x - x0) / CHAR_W)
        pad = max(1, target_chars - cursor_chars) if parts else max(0, target_chars)
        parts.append(" " * pad + text)
        cursor_chars = target_chars + len(text)
    return "".join(parts)


def _two_band_pdf(tmp_path, *, right_x0: float, page_width: float = 612.0, n_rows: int = 6):
    """Two side-by-side tables, each ROW inserted as ONE call spanning both bands.

    Left band: Label/Mean/SD at x=60/140/190. Right band: Label/Corr/Guide
    starting at ``right_x0``. A narrow gutter (``right_x0`` close to the left
    band) reproduces GH-779: neither band alone covers 50% of the spanning
    row block, but their union does. A wide gutter keeps union coverage
    below the threshold too -- the content-loss guard this ticket must not
    break.
    """
    doc = fitz.open()
    page = doc.new_page(width=page_width, height=792)
    y0, row_h = 100.0, 20.0
    left_x0 = 60.0
    header_cells = [
        (60, "Label"),
        (140, "Mean"),
        (190, "SD"),
        (right_x0, "Label"),
        (right_x0 + 70, "Corr"),
        (right_x0 + 110, "Guide"),
    ]
    page.insert_text((left_x0, y0), _row_text(header_cells, left_x0), fontsize=FS, fontname="cour")
    for i in range(n_rows):
        y = y0 + row_h * (i + 1)
        cells = [
            (60, f"LeftLab{i}"),
            (140, f"{i}.11"),
            (190, f"{i}.22"),
            (right_x0, f"RightLab{i}"),
            (right_x0 + 70, f"{i}.33"),
            (right_x0 + 110, f"{i}.44"),
        ]
        page.insert_text((left_x0, y), _row_text(cells, left_x0), fontsize=FS, fontname="cour")
    path = tmp_path / "bands.pdf"
    doc.save(str(path))
    doc.close()
    return path


def _one_table_pdf(tmp_path, n_rows: int = 6):
    """A single table, no gutter -- the byte-identity control."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    y0, row_h = 100.0, 20.0
    x0 = 60.0
    page.insert_text(
        (x0, y0),
        _row_text([(60, "Label"), (140, "Mean"), (190, "SD")], x0),
        fontsize=FS,
        fontname="cour",
    )
    for i in range(n_rows):
        y = y0 + row_h * (i + 1)
        page.insert_text(
            (x0, y),
            _row_text([(60, f"Lab{i}"), (140, f"{i}.11"), (190, f"{i}.22")], x0),
            fontsize=FS,
            fontname="cour",
        )
    path = tmp_path / "one_table.pdf"
    doc.save(str(path))
    doc.close()
    return path


# ---------------------------------------------------------------------------
# Unit tests: the geometry helpers in isolation
# ---------------------------------------------------------------------------


def test_side_by_side_regions_are_band_siblings():
    left = fitz.Rect(60, 100, 216, 220)
    right = fitz.Rect(348, 100, 492, 220)

    assert _regions_are_band_siblings([left, right]) is True


def test_x_overlapping_regions_are_not_siblings():
    """Two regions that overlap in x are not a band split (a band split is
    disjoint by construction -- it partitions words at a single gutter)."""
    a = fitz.Rect(60, 100, 300, 220)
    b = fitz.Rect(250, 100, 492, 220)

    assert _regions_are_band_siblings([a, b]) is False


def test_y_disjoint_regions_are_not_siblings():
    """Two unrelated tables stacked vertically (different row ranges) must
    never be treated as siblings of a single band split -- unioning them
    would reopen #145/#718b: a line could be dropped because its words
    merely appear in a DIFFERENT, unrelated table elsewhere on the page."""
    top = fitz.Rect(60, 100, 216, 220)
    bottom = fitz.Rect(60, 400, 492, 520)

    assert _regions_are_band_siblings([top, bottom]) is False


def test_single_region_is_trivially_siblings():
    assert _regions_are_band_siblings([fitz.Rect(0, 0, 10, 10)]) is True


def test_union_coverage_of_disjoint_regions_sums():
    block = fitz.Rect(0, 0, 100, 20)
    left = fitz.Rect(0, 0, 40, 20)
    right = fitz.Rect(60, 0, 100, 20)

    cov = _union_coverage(block, [left, right])

    assert cov == pytest.approx(0.8)


def test_union_coverage_does_not_double_count_overlap():
    """Two regions that overlap each other must not sum past their true
    combined area -- summing individually-measured coverage would."""
    block = fitz.Rect(0, 0, 100, 20)
    a = fitz.Rect(0, 0, 60, 20)
    b = fitz.Rect(40, 0, 100, 20)  # overlaps a on [40, 60]

    cov = _union_coverage(block, [a, b])

    assert cov == pytest.approx(1.0)
    naive_sum = _rect_coverage(block, a) + _rect_coverage(block, b)
    assert naive_sum > 1.0, "the naive sum should overshoot -- that's the bug this avoids"


def test_union_coverage_of_no_regions_is_zero():
    assert _union_coverage(fitz.Rect(0, 0, 10, 10), []) == 0.0


def test_union_coverage_zero_area_block_does_not_divide_by_zero():
    assert _union_coverage(fitz.Rect(5, 5, 5, 5), [fitz.Rect(0, 0, 10, 10)]) == 0.0


# ---------------------------------------------------------------------------
# End to end: extract_structured on a real two-band page
# ---------------------------------------------------------------------------


def test_a_row_spanning_two_bands_is_suppressed_not_duplicated(tmp_path):
    """The bug: a row block covering neither band region alone (~37% and
    ~32%) but both together (~69%) used to survive as a duplicate plain-text
    copy beneath the two split tables."""
    path = _two_band_pdf(tmp_path, right_x0=350.0)
    with fitz.open(path) as doc:
        out = BornDigitalDetector().extract_structured(doc[0])

    assert out.count("RightLab3") == 1, "the spanning row must not duplicate"
    assert out.count("LeftLab3") == 1
    # Both tables still ship, correctly split and populated.
    assert "| RightLab0 | 0.33 | 0.44 |" in out
    assert "| LeftLab0 | 0.11 | 0.22 |" in out


def test_below_union_threshold_a_spanning_row_still_ships(tmp_path):
    """Load-bearing negative, the content-loss direction. A wider gutter
    keeps each band's own coverage AND their union below
    ``_REGION_COVERAGE_DROP`` (~23% + ~21% = ~44%). The row must not be
    silently dropped just because it touches two band-sibling regions --
    union coverage still gates suppression, it does not replace the gate."""
    path = _two_band_pdf(tmp_path, right_x0=600.0, page_width=900.0)
    with fitz.open(path) as doc:
        out = BornDigitalDetector().extract_structured(doc[0])

    assert "RightLab3" in out, "content must survive when union coverage is insufficient"
    assert "LeftLab3" in out


def test_content_check_still_vetoes_union_suppression(tmp_path):
    """A word the union's combined markdown does not contain must still
    block suppression -- the union path must not weaken the #145 content
    check, only widen which regions feed it."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    y0, row_h = 100.0, 20.0
    left_x0 = 60.0
    header_cells = [
        (60, "Label"),
        (140, "Mean"),
        (190, "SD"),
        (350, "Label"),
        (420, "Corr"),
        (460, "Guide"),
    ]
    page.insert_text((left_x0, y0), _row_text(header_cells, left_x0), fontsize=FS, fontname="cour")
    for i in range(6):
        y = y0 + row_h * (i + 1)
        cells = [
            (60, f"LeftLab{i}"),
            (140, f"{i}.11"),
            (190, f"{i}.22"),
            (350, f"RightLab{i}"),
            (420, f"{i}.33"),
            (460, f"{i}.44"),
        ]
        if i == 3:
            # A word neither table's markdown will ever contain -- sits in
            # the gutter, outside every lane.
            cells.insert(3, (280, "ZZMARK"))
        page.insert_text((left_x0, y), _row_text(cells, left_x0), fontsize=FS, fontname="cour")
    path = tmp_path / "veto.pdf"
    doc.save(str(path))
    doc.close()

    with fitz.open(path) as d:
        out = BornDigitalDetector().extract_structured(d[0])

    assert "ZZMARK" in out, "a word absent from every region's markdown must survive"
    assert out.count("RightLab3") == 2, (
        "the content check keeps the WHOLE line that carries ZZMARK, "
        "including its own RightLab3 -- duplication, not loss, is the correct trade"
    )
    # Every other row (no unrepresented word) is still correctly suppressed.
    assert out.count("RightLab0") == 1
    assert out.count("RightLab5") == 1


def test_single_region_page_is_unaffected_by_the_union_path(tmp_path, monkeypatch):
    """On a page with only one table region, ``touching`` can never exceed
    one member, so the union path can never engage. Difference-pinned:
    forcing the sibling predicate to always say yes must not change output
    on this fixture, because the length gate in front of it never lets a
    single-region page reach it."""
    import socr.core.born_digital as bd

    path = _one_table_pdf(tmp_path)
    with fitz.open(path) as doc:
        before = BornDigitalDetector().extract_structured(doc[0])

    monkeypatch.setattr(bd, "_regions_are_band_siblings", lambda rects: True)
    with fitz.open(path) as doc:
        after = BornDigitalDetector().extract_structured(doc[0])

    assert before == after
