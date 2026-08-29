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
