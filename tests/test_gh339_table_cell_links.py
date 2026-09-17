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


def test_link_straddling_two_cells_binds_only_the_majority_cell() -> None:
    """rev-339 review: an oversized link rect that overruns a column rule
    must not fabricate a duplicate value by wrapping BOTH cells it touches.

    Applying every link to every intersecting cell independently let a
    straddling link satisfy `cell_rect.intersects(rect)` on two neighbours;
    if both cells' text contained the anchor substring, the ANCHOR was
    wrapped in both -- `| [2020](url) | [2020](url) |` where only one 2020
    ever carried that citation. The fix assigns each link to the ONE cell
    holding the MAJORITY of its own area (mirrors `_words_in_region`'s
    majority-overlap rule in tables/binding.py), never a bare intersection.
    """
    rows = [["A", "B"], ["2020", "2020"]]
    row_cells = [
        [(68.0, 128.0, 190.0, 150.0), (190.0, 128.0, 320.0, 150.0)],
        [(68.0, 150.0, 190.0, 172.0), (190.0, 150.0, 320.0, 172.0)],
    ]
    table = _StubTable(rows, row_cells)
    uri = "https://example.com/2020"
    # Straddles the x=190 column rule: 20/60 = 1/3 of its area in the LEFT
    # cell, 40/60 = 2/3 in the RIGHT cell -- a clear majority, not a tie.
    straddling = fitz.Rect(170.0, 150.0, 230.0, 172.0)
    links = [(straddling, uri, "2020")]

    out = BornDigitalDetector()._table_to_markdown(table, links=links)

    data_line = out.splitlines()[-1]
    assert data_line == f"| 2020 | [2020]({uri}) |", data_line
    assert data_line.count(uri) == 1, "the link must not be duplicated into both cells"


def test_ambiguous_50_50_straddling_link_is_dropped_not_duplicated() -> None:
    """No cell holds a majority (an exact tie) -- the link is DROPPED, never
    guessed into either cell and never duplicated into both."""
    rows = [["A", "B"], ["2020", "2020"]]
    row_cells = [
        [(68.0, 128.0, 190.0, 150.0), (190.0, 128.0, 320.0, 150.0)],
        [(68.0, 150.0, 190.0, 172.0), (190.0, 150.0, 320.0, 172.0)],
    ]
    table = _StubTable(rows, row_cells)
    uri = "https://example.com/2020"
    tied = fitz.Rect(160.0, 150.0, 220.0, 172.0)  # exactly 30/60 each side
    links = [(tied, uri, "2020")]

    out = BornDigitalDetector()._table_to_markdown(table, links=links)

    assert out.splitlines()[-1] == "| 2020 | 2020 |"
    assert uri not in out


def test_a_well_contained_link_still_binds_under_the_majority_rule() -> None:
    """Guard against over-correcting to strict containment: a normal link
    that sits entirely inside its own cell -- the ordinary case this ticket
    exists to recover -- must still bind. Prove both directions."""
    rows = [["Year", "DOI"], ["2019", "10.1111/jofi.12345"]]
    row_cells = [
        [(68.0, 128.0, 190.0, 150.0), (190.0, 128.0, 320.0, 150.0)],
        [(68.0, 150.0, 190.0, 172.0), (190.0, 150.0, 320.0, 172.0)],
    ]
    table = _StubTable(rows, row_cells)
    links = [(fitz.Rect(200.0, 150.0, 300.0, 172.0), DOI, "10.1111/jofi.12345")]

    out = BornDigitalDetector()._table_to_markdown(table, links=links)

    assert f"[10.1111/jofi.12345]({DOI})" in out


def test_bare_numeric_anchor_still_binds_and_stays_numeric() -> None:
    """rev-339 review: a NUMBER carrying a citation link -- not just a DOI --
    must (a) still be wrapped as a markdown link, and (b) still be seen as a
    numeric token by the value guard downstream (`native_verifier`'s
    NUMBER-COMPLETENESS multiset check), which anchors on `_NUM_TOKEN_RE`
    and does not match `[1204](https://x/note)` unless unwrapped first.
    """
    from socr.tables.native_verifier import _numeric_tokens_from_text

    rows = [["Year", "Value"], ["2019", "1204"]]
    value_cell_rect = (190.0, 150.0, 320.0, 172.0)
    row_cells = [
        [(68.0, 128.0, 190.0, 150.0), (190.0, 128.0, 320.0, 150.0)],
        [(68.0, 150.0, 190.0, 172.0), value_cell_rect],
    ]
    table = _StubTable(rows, row_cells)
    uri = "https://x/note"
    links = [(fitz.Rect(value_cell_rect), uri, "1204")]

    out = BornDigitalDetector()._table_to_markdown(table, links=links)

    data_line = out.splitlines()[-1]
    assert data_line == f"| 2019 | [1204]({uri}) |", data_line
    assert _numeric_tokens_from_text(data_line) == ["2019", "1204"], (
        "the linked number must not drop out of the numeric-completeness check"
    )


def test_multiline_cell_link_recovery() -> None:
    """A cell whose extracted text spans multiple lines (`table.extract()`
    joins wrapped cell text with `\\n`) must still recover a link that sits
    over one of those lines -- untested before rev-339 flagged it fragile
    by construction, even though it could not break it."""
    rows = [
        ["Ref", "Note"],
        ["Smith (2019);\n10.1111/jofi.12345", "see appendix"],
    ]
    note_cell_rect = (190.0, 150.0, 420.0, 194.0)
    ref_cell_rect = (68.0, 150.0, 190.0, 194.0)
    row_cells = [
        [(68.0, 128.0, 190.0, 150.0), (190.0, 128.0, 420.0, 150.0)],
        [ref_cell_rect, note_cell_rect],
    ]
    table = _StubTable(rows, row_cells)
    links = [(fitz.Rect(ref_cell_rect), DOI, "10.1111/jofi.12345")]

    out = BornDigitalDetector()._table_to_markdown(table, links=links)

    assert "Smith (2019);" in out
    assert f"[10.1111/jofi.12345]({DOI})" in out


def test_markdown_linked_number_is_still_a_numeric_token() -> None:
    """Direct-caller pin: `is_numeric_token`/`_numeric_tokens_from_text` must
    unwrap a whole-token markdown link before the anchored `_NUM_TOKEN_RE`
    test, the exact repro rev-339 gave: a linked numeric cell silently
    dropped out of the NUMBER-COMPLETENESS multiset check.
    """
    from socr.tables.native_verifier import _numeric_tokens_from_text, is_numeric_token

    assert is_numeric_token("[1204](https://x/note)") is True
    assert is_numeric_token("[(1,204)](https://x/note)") is True
    assert _numeric_tokens_from_text("| 2019 | 1204 |") == ["2019", "1204"]
    assert _numeric_tokens_from_text("| 2019 | [1204](https://x/note) |") == ["2019", "1204"]


def test_markdown_link_unwrap_does_not_swallow_non_numeric_prose() -> None:
    """The unwrap is scoped to whole-token links; a link embedded in a prose
    sentence (not a bare table-cell token) must not be misread as numeric,
    and a token that merely contains brackets elsewhere is left alone."""
    from socr.tables.native_verifier import is_numeric_token

    assert is_numeric_token("[see note]") is False
    assert is_numeric_token("Panel A.") is False


def test_table_grid_normalize_and_is_numeric_cell_unwrap_markdown_links() -> None:
    """GH-339 consumer sweep: `core/table_grid.py` runs the GH-96 GT-vs-model
    exactness comparison (`score_page`/`score_rows`) and `is_numeric_cell`
    (used by `native_rows.py`) on RAW markdown cells -- it does not go through
    `native_verifier.is_numeric_token`, so it needed its own unwrap step
    rather than inheriting this ticket's `native_verifier` fix for free.

    Without it, a correctly-recovered linked cell (`[1204](url)`) would
    compare unequal to its ground-truth value (`1204`) and would not be
    seen as numeric by `native_rows.py`'s grid detector -- an accuracy
    regression this ticket's own fix would otherwise have introduced.
    """
    from socr.core.table_grid import is_numeric_cell, normalize_cell

    assert normalize_cell("[1204](https://x/note)") == "1204"
    assert normalize_cell("1204") == "1204"
    assert is_numeric_cell("[1204](https://x/note)") is True
    assert is_numeric_cell("[(1,204)](https://x/note)") is True
    # A link is not silently unwrapped when it is not the whole cell/token.
    assert is_numeric_cell("see [1204](https://x/note) above") is False


def test_is_numeric_token_chokepoint_ignores_the_link_wrapper() -> None:
    """The required closing test (rev-339): one assertion at the shared
    chokepoint would have caught the whole GH-339 numeric-decode-gap family
    in one shot, instead of file-by-file. `is_numeric_token` is the shared
    predicate imported (not reimplemented) by `tables/binding.py`,
    `tables/row_corroboration.py`, `tables/adjudication.py`, and
    `tables/header_repair.py` (which `tables/crop_repair.py` calls) --
    fixing it here fixes all six affected call sites named in the review:
    `adjudication.tokens_agree`, `native_verifier._parse_all_data_rows`,
    `native_verifier._output_header_numeric_tokens`,
    `binding._candidate_data_column_indices`,
    `row_corroboration.numeric_body_rows`, and
    `header_repair.detect_header_column_collapse` (+ its
    `crop_repair._max_header_col_gap` caller).
    """
    from socr.tables.native_verifier import is_numeric_token

    assert is_numeric_token("[1204](https://x)") == is_numeric_token("1204")
    assert is_numeric_token("[1204](https://x)") is True
