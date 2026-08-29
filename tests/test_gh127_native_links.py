"""GH-127: hyperlinks in the born-digital text layer reach the markdown.

`page.get_links()` was never called, so every link in the text layer was dropped --
1,103 across half the sampled corpus. For a citation corpus a DOI present in the
source and absent from the output is content loss, not formatting loss.
"""

from pathlib import Path

import fitz
import pytest

from socr.core.born_digital import BornDigitalDetector

DOI = "https://doi.org/10.1234/abc"


def _page_with_link(tmp_path: Path, text: str, anchor: str, uri: str, y: float = 100) -> Path:
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, y), text, fontsize=11)
    hits = page.search_for(anchor)
    assert hits, f"fixture anchor {anchor!r} not found on the page"
    page.insert_link({"kind": fitz.LINK_URI, "from": hits[0], "uri": uri})
    out = tmp_path / "linked.pdf"
    doc.save(out)
    doc.close()
    return out


def _extract(pdf: Path) -> str:
    with fitz.open(pdf) as doc:
        return BornDigitalDetector().extract_structured(doc[0])


def test_anchor_link_becomes_a_markdown_link(tmp_path: Path) -> None:
    """The URI is recovered, and the anchor's own characters are untouched."""
    pdf = _page_with_link(tmp_path, "See Smith 2020 for details.", "Smith 2020", DOI)
    out = _extract(pdf)
    assert f"[Smith 2020]({DOI})" in out
    # Nothing around the anchor was disturbed.
    assert out.startswith("See [")
    assert out.rstrip().endswith("for details.")


def test_page_without_links_is_unchanged(tmp_path: Path) -> None:
    """Byte-identity guard: a link-free page must be exactly what it was before.

    Pins a DIFFERENCE against the raw text layer rather than a literal string, so
    the test does not re-encode extract_structured's own formatting rules.
    """
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 100), "Plain prose, no links at all.", fontsize=11)
    pdf = tmp_path / "plain.pdf"
    doc.save(pdf)
    doc.close()

    with fitz.open(pdf) as d:
        raw = d[0].get_text("text").strip()
        structured = BornDigitalDetector().extract_structured(d[0])
    assert structured == raw
    assert "[" not in structured


def test_self_linking_url_is_not_wrapped(tmp_path: Path) -> None:
    """A visible URL that links to itself needs no anchor -- and must not lose it."""
    url = "https://example.org/paper"
    pdf = _page_with_link(tmp_path, url, url, url)
    out = _extract(pdf)
    assert url in out
    assert f"[{url}]({url})" not in out


def test_internal_page_jump_is_not_emitted_as_a_link(tmp_path: Path) -> None:
    """LINK_GOTO addresses a position in the PDF, not a resource."""
    doc = fitz.open()
    doc.new_page(width=612, height=792)
    doc.new_page(width=612, height=792)
    # Take the page handle AFTER both pages exist: new_page() invalidates an
    # earlier Page object, and the failure surfaces as a confusing AttributeError.
    page = doc[0]
    page.insert_text((72, 100), "See Section 4 below.", fontsize=11)
    hits = page.search_for("Section 4")
    page.insert_link({"kind": fitz.LINK_GOTO, "from": hits[0], "page": 1, "to": fitz.Point(0, 0)})
    pdf = tmp_path / "goto.pdf"
    doc.save(pdf)
    doc.close()

    out = _extract(pdf)
    assert "Section 4" in out
    assert "](" not in out, "an internal page jump must not become a markdown link"


def test_damaged_link_table_does_not_raise(tmp_path: Path, monkeypatch) -> None:
    """Fail-open: a link table that raises degrades to 'no links', never out."""
    pdf = _page_with_link(tmp_path, "See Smith 2020 for details.", "Smith 2020", DOI)
    with fitz.open(pdf) as doc:
        page = doc[0]
        monkeypatch.setattr(
            type(page), "get_links", lambda self, *a, **k: (_ for _ in ()).throw(RuntimeError("x"))
        )
        out = BornDigitalDetector().extract_structured(page)
    assert "Smith 2020" in out
    assert "](" not in out


@pytest.mark.parametrize("y", [100.0, 760.0])
def test_link_recovered_in_body_and_in_footer(tmp_path: Path, y: float) -> None:
    """A journal footer is exactly where a DOI lives, so furniture keeps links too."""
    pdf = _page_with_link(tmp_path, "See Smith 2020 for details.", "Smith 2020", DOI, y=y)
    assert f"[Smith 2020]({DOI})" in _extract(pdf)


# ---------------------------------------------------------------------------
# GH-127 review. The tests above all land on `_apply_links_to_flat_text`: a
# table-free page short-circuits to the flat string and never walks the span
# dict. So `_line_text` -- the other path this change touched, carrying links on
# table pages and on page furniture -- was entirely unguarded, and both of its
# call sites could be deleted with the suite still green.
#
# Note furniture is NOT a y-coordinate: `block_is_page_furniture` selects
# counter-directional blocks, so horizontal text low on the page is just more
# prose on the flat path. A furniture fixture has to be genuinely rotated.
# ---------------------------------------------------------------------------


def _table_page_with_link(tmp_path: Path, uri: str) -> Path:
    """A page whose ruled grid makes `find_tables` produce a real table region."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)

    page.insert_text((72, 90), "See Smith 2020 for the specification.", fontsize=11)

    rows = (("Year", "Coefficient"), ("2019", "0.31"), ("2020", "0.42"), ("2021", "0.55"))
    top, left, mid, right, pitch = 128, 68, 190, 320, 22
    y = top + 16
    for c0, c1 in rows:
        page.insert_text((left + 6, y), c0, fontsize=10)
        page.insert_text((mid + 6, y), c1, fontsize=10)
        y += pitch
    bottom = top + pitch * len(rows)
    # `find_tables` needs INTERSECTING rules, not just an outer box.
    for i in range(len(rows) + 1):
        yy = top + pitch * i
        page.draw_line(fitz.Point(left, yy), fitz.Point(right, yy))
    for xx in (left, mid, right):
        page.draw_line(fitz.Point(xx, top), fitz.Point(xx, bottom))

    hits = page.search_for("Smith 2020")
    assert hits, "fixture anchor not found"
    page.insert_link({"kind": fitz.LINK_URI, "from": hits[0], "uri": uri})

    out = tmp_path / "table_linked.pdf"
    doc.save(out)
    doc.close()
    return out


def test_links_survive_on_a_page_that_has_a_detected_table(tmp_path: Path) -> None:
    """Drives the span-dict walk, not the flat short-circuit.

    Without a real table region the page short-circuits and `_line_text` is never
    the caller -- which is why deleting both of its call sites left the original
    suite passing.
    """
    from socr.core.born_digital import BornDigitalDetector

    pdf = _table_page_with_link(tmp_path, DOI)
    with fitz.open(pdf) as doc:
        page = doc[0]
        detector = BornDigitalDetector()
        # Guard the fixture itself: if the grid stops being detected this test
        # would silently revert to exercising the flat path again.
        assert page.find_tables().tables, "fixture must produce a detected table"
        out = detector.extract_structured(page)

    assert f"[Smith 2020]({DOI})" in out


def test_a_phrase_link_does_not_wrap_the_whole_line(tmp_path: Path) -> None:
    """The removed fallback: coverage-based wrapping marked an entire span.

    A uniform-font line is ONE span, so wrapping on rectangle coverage alone put
    the brackets around every word on the line. Pinned on the table page because
    that is the path where the fallback lived.
    """
    from socr.core.born_digital import BornDigitalDetector

    pdf = _table_page_with_link(tmp_path, DOI)
    with fitz.open(pdf) as doc:
        out = BornDigitalDetector().extract_structured(doc[0])

    line = next(ln for ln in out.splitlines() if "Smith 2020" in ln)
    assert line.strip().startswith("See ["), line
    assert "specification." in line
    # The words outside the link rectangle stay outside the brackets.
    assert "[See Smith 2020 for the specification.]" not in line


def test_a_repeated_anchor_binds_the_uri_to_the_linked_occurrence(tmp_path: Path) -> None:
    """The wrong-mention bug: `text.find()` bound every link to the FIRST match.

    A paper cites "Smith 2020" in the body and again in the references, where the
    DOI actually lives. Content matching put the bibliography's URI on the in-text
    mention. A wrong URI on the wrong citation is worse than a dropped link.
    """
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    # Character-identical occurrences: nothing but POSITION distinguishes them, so
    # a content-matching implementation cannot pass this by accident.
    page.insert_text((72, 100), "As shown in Smith 2020. Effect is large.", fontsize=11)
    page.insert_text((72, 300), "Smith 2020. Journal of Results.", fontsize=11)

    hits = page.search_for("Smith 2020")
    assert len(hits) >= 2, "fixture needs two occurrences"
    # Link ONLY the second (the bibliography entry).
    second = sorted(hits, key=lambda r: r.y0)[1]
    page.insert_link({"kind": fitz.LINK_URI, "from": second, "uri": DOI})

    pdf = tmp_path / "repeated.pdf"
    doc.save(pdf)
    doc.close()

    out = _extract(pdf)

    assert out.count(f"]({DOI})") == 1, out
    body, refs = out.split("\n", 1) if "\n" in out else (out, "")

    # The whole point: the URI is on the references entry, not the in-text mention.
    # Asserted by PLACEMENT, not by an exact anchor spelling -- the covered word
    # legitimately carries its trailing period, so the anchor is "Smith 2020.".
    assert f"]({DOI})" in refs, refs
    assert f"]({DOI})" not in body, f"URI landed on the in-text mention: {body!r}"
    assert "Smith 2020" in body and "[" not in body, body
    # Same anchor string in both halves -- only the offsets differ.
    assert "Smith 2020." in body and "Smith 2020." in refs


def test_two_links_on_one_line_both_survive(tmp_path: Path) -> None:
    """`break` after the first match dropped every later link in the same span."""
    other = "https://doi.org/10.5678/xyz"
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 100), "Compare Alpha 1999 against Beta 2011 here.", fontsize=11)
    for anchor, uri in (("Alpha 1999", DOI), ("Beta 2011", other)):
        hits = page.search_for(anchor)
        assert hits, anchor
        page.insert_link({"kind": fitz.LINK_URI, "from": hits[0], "uri": uri})

    pdf = tmp_path / "two.pdf"
    doc.save(pdf)
    doc.close()

    out = _extract(pdf)

    assert f"[Alpha 1999]({DOI})" in out, out
    assert f"[Beta 2011]({other})" in out, out


def test_a_malformed_link_entry_does_not_cost_the_page_its_other_links(tmp_path: Path) -> None:
    """One damaged entry degrades to fewer links, never to an exception."""
    from socr.core import born_digital as bd

    pdf = _page_with_link(tmp_path, "See Smith 2020 for details.", "Smith 2020", DOI)

    with fitz.open(pdf) as doc:
        page = doc[0]
        real = page.get_links() or []
        assert real, "fixture must have a link"

        # A rectangle-less entry ahead of the good one: the old narrow guard let a
        # malformed word tuple escape from the coverage scan.
        broken = [{"kind": fitz.LINK_URI, "uri": "https://example.invalid/x"}] + real
        page.get_links = lambda *a, **k: broken

        links = bd._uri_links(page)

    assert [ln[1] for ln in links] == [DOI], links


def test_an_unresolvable_anchor_leaves_the_span_alone() -> None:
    """The removed fallback, driven directly.

    It only fired when the anchor could not be resolved from the words -- which no
    end-to-end fixture reaches, so an end-to-end test cannot guard it. With a
    uniform-font line being one span, coverage-based wrapping put the brackets
    around the entire line.
    """
    from socr.core import born_digital as bd

    span = {
        "text": "Estimated coefficient 0.082 significant at 1 percent",
        "bbox": (72, 90, 400, 104),
    }
    # Rect covers the span, but the anchor is empty: unresolvable.
    link = (fitz.Rect(72, 90, 400, 104), DOI, "", ())

    out = bd._line_text([span], [link])

    assert out == span["text"], out
    assert DOI not in out
