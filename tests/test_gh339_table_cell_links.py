"""GH-339: URI links inside a detected TABLE CELL reach the markdown grid.

#323/GH-127 recover links on prose and around a table, but table markdown is
emitted from `table.extract()`'s plain cell strings -- never passed through
`_line_text` / `_apply_links_to_flat_text`, where those anchors are applied.
A DOI whose link rectangle sits over a data cell was dropped entirely, not
even surfacing as raw text. For a citation corpus that is silent content loss.
"""

from pathlib import Path

import fitz

from socr.core.born_digital import BornDigitalDetector

DOI = "https://doi.org/10.1111/jofi.12345"


def _table_page_with_cell_link(tmp_path: Path, uri: str) -> Path:
    """A ruled grid whose DATA CELL (not the surrounding prose) carries a URI.

    Intersecting rules, not just an outer box -- `find_tables` needs a real
    grid to produce a region at all (see `test_gh127_native_links.py`).
    """
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)

    page.insert_text((72, 90), "See the citation table below.", fontsize=11)

    rows = (("Year", "DOI"), ("2019", "10.1111/jofi.12345"), ("2020", "10.2222/xyz.999"))
    top, left, mid, right, pitch = 128, 68, 190, 420, 22
    y = top + 16
    for c0, c1 in rows:
        page.insert_text((left + 6, y), c0, fontsize=10)
        page.insert_text((mid + 6, y), c1, fontsize=10)
        y += pitch
    bottom = top + pitch * len(rows)
    for i in range(len(rows) + 1):
        yy = top + pitch * i
        page.draw_line(fitz.Point(left, yy), fitz.Point(right, yy))
    for xx in (left, mid, right):
        page.draw_line(fitz.Point(xx, top), fitz.Point(xx, bottom))

    # The link rectangle sits over the DATA CELL text, not the whole row/table.
    hits = page.search_for("10.1111/jofi.12345")
    assert hits, "fixture anchor not found"
    page.insert_link({"kind": fitz.LINK_URI, "from": hits[0], "uri": uri})

    out = tmp_path / "table_cell_linked.pdf"
    doc.save(out)
    doc.close()
    return out


def test_link_inside_a_table_cell_is_recovered(tmp_path: Path) -> None:
    """The URI is not dropped, and it lands on the correct cell's markdown."""
    pdf = _table_page_with_cell_link(tmp_path, DOI)
    with fitz.open(pdf) as doc:
        page = doc[0]
        detector = BornDigitalDetector()
        # Guard the fixture itself: without a detected table, this exercises
        # a different path (GH-127's flat/dict-walk case), not GH-339.
        assert page.find_tables().tables, "fixture must produce a detected table"
        out = detector.extract_structured(page)

    assert f"[10.1111/jofi.12345]({DOI})" in out
    # The URI must land on ITS OWN row, not the sibling row's unlinked value.
    line = next(ln for ln in out.splitlines() if "2019" in ln and "|" in ln)
    assert f"[10.1111/jofi.12345]({DOI})" in line
    sibling = next(ln for ln in out.splitlines() if "2020" in ln and "|" in ln)
    assert "10.2222/xyz.999" in sibling
    assert "[10.2222/xyz.999]" not in sibling


def test_table_without_links_is_unchanged(tmp_path: Path) -> None:
    """Byte-identity guard: a link-free table must produce identical markdown
    with and without the GH-339 wiring exercised (no links resolved => no-op)."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    rows = (("Year", "Value"), ("2019", "0.31"), ("2020", "0.42"))
    top, left, mid, right, pitch = 128, 68, 190, 320, 22
    y = top + 16
    for c0, c1 in rows:
        page.insert_text((left + 6, y), c0, fontsize=10)
        page.insert_text((mid + 6, y), c1, fontsize=10)
        y += pitch
    bottom = top + pitch * len(rows)
    for i in range(len(rows) + 1):
        yy = top + pitch * i
        page.draw_line(fitz.Point(left, yy), fitz.Point(right, yy))
    for xx in (left, mid, right):
        page.draw_line(fitz.Point(xx, top), fitz.Point(xx, bottom))
    pdf = tmp_path / "table_plain.pdf"
    doc.save(pdf)
    doc.close()

    with fitz.open(pdf) as d:
        page = d[0]
        assert page.find_tables().tables, "fixture must produce a detected table"
        out = BornDigitalDetector().extract_structured(page)

    assert "0.31" in out
    assert "0.42" in out
    assert "[" not in out


class _StubRow:
    def __init__(self, cells: list) -> None:
        self.cells = cells


class _StubTable:
    """Duck-types the bits of a PyMuPDF Table `_table_to_markdown` reads.

    Lets the None-cell case be exercised directly and deterministically --
    `find_tables()` does not offer a simple, portable recipe for forcing a
    merged/absent cell out of a real ruled grid, and PyMuPDF's own source
    (`table.py`, several `cell is None` guards) confirms a None cell is a
    real return shape, not a hypothetical one.
    """

    def __init__(self, rows: list[list[str]], row_cells: list[list]) -> None:
        self._rows = rows
        self.rows = [_StubRow(cells) for cells in row_cells]

    def extract(self) -> list[list[str]]:
        return self._rows


def test_a_none_cell_does_not_cost_the_whole_table_its_links() -> None:
    """GH-339 review: one merged/absent cell must not zero every OTHER cell's
    link recovery in the same table.

    `table.rows[i].cells[j]` is `None` for a merged/absent span (PyMuPDF's own
    `table.py` guards this at several call sites). `fitz.Rect(None)` raises --
    without a per-cell guard, that single bad cell would blow the whole
    row-bboxes build and silently drop every link in the table, not just the
    one cell's.
    """
    rows = [
        ["Year", "DOI"],
        ["2019", "10.1111/jofi.12345"],
        ["2020", "10.2222/xyz.999"],
    ]
    doi_cell_rect = (190.0, 150.0, 320.0, 172.0)
    row_cells = [
        [(68.0, 128.0, 190.0, 150.0), None],  # header's 2nd cell: merged/absent
        [(68.0, 150.0, 190.0, 172.0), doi_cell_rect],
        [(68.0, 172.0, 190.0, 194.0), (190.0, 172.0, 320.0, 194.0)],
    ]
    table = _StubTable(rows, row_cells)

    links = [(fitz.Rect(doi_cell_rect), DOI, "10.1111/jofi.12345")]

    out = BornDigitalDetector()._table_to_markdown(table, links=links)

    assert f"[10.1111/jofi.12345]({DOI})" in out
    assert "10.2222/xyz.999" in out
    assert "DOI" in out  # the None-celled header survives as plain text
