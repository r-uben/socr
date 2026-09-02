"""GH-166: a crop reread that verified nothing must not look clean.

Three failure paths in `TableCropExtractor.extract` -- render error, reader
exception, empty response -- did a bare `continue`, so a page whose crops ALL
failed returned an empty list, indistinguishable from a page with no crops to
read. `_reread_page_tables` then returned `(0, 0)` with no distrust event, and
the incumbent table read as verified because the check that would have
contradicted it left no trace.

Timeouts DID emit `dualpass_crop_timeout`, but that kind was missing from
`TABLE_DISTRUST_KINDS`, so `tables_trust.json` still reported no untrusted
pages. Measured before the fix:

    dualpass_crop_timeout    untrusted_pages=[]
    dualpass_flagged         untrusted_pages=[3]
"""

from __future__ import annotations

import pytest

from socr.core.audit_log import AuditEvent
from socr.core.tables_trust import TABLE_DISTRUST_KINDS, build_tables_trust

fitz = pytest.importorskip("fitz")


@pytest.mark.parametrize("kind", ["dualpass_crop_timeout", "dualpass_crop_failed"])
def test_an_incomplete_reread_makes_the_page_untrusted(kind: str) -> None:
    """Both incomplete-verification kinds must reach `tables_trust.json`."""
    trust = build_tables_trust(
        "doc.pdf",
        [AuditEvent(page_num=3, kind=kind, engine="qwen", detail="x")],
    )
    assert trust.untrusted_pages == [3], (
        f"{kind} left the page trusted, so an incomplete verification reads as "
        f"a completed one: {trust.untrusted_pages}"
    )


def test_both_kinds_are_declared_distrusting() -> None:
    """The set itself, so neither can be quietly dropped again."""
    assert "dualpass_crop_timeout" in TABLE_DISTRUST_KINDS
    assert "dualpass_crop_failed" in TABLE_DISTRUST_KINDS


def test_a_resolving_kind_is_still_not_distrusting() -> None:
    """Control: the fix must not turn the whole set into "everything distrusts".

    If membership were the only assertion, adding every kind would satisfy the
    tests above while destroying the distinction the set exists for.
    """
    assert "table_ladder_accepted" not in TABLE_DISTRUST_KINDS


@pytest.mark.parametrize(
    ("reason", "raises", "reply"),
    [("read_error", True, ""), ("empty_response", False, "")],
    ids=["reader-exception", "empty-response"],
)
def test_a_located_crop_always_yields_a_sentinel(tmp_path, reason, raises, reply) -> None:
    """Acceptance 1: success or a TYPED failure -- never silence."""
    from socr.tables.extract import TableCropExtractor
    from socr.tables.locate import locate_tables

    pdf = tmp_path / "t.pdf"
    doc = fitz.open()
    page = doc.new_page(width=400, height=300)
    page.draw_line(fitz.Point(50, 100), fitz.Point(350, 100))
    page.draw_line(fitz.Point(50, 160), fitz.Point(350, 160))
    for i, y in enumerate((120, 140)):
        page.insert_text((60, y), f"Row{i}", fontsize=9)
        page.insert_text((200, y), f"{i}.5", fontsize=9)
    doc.save(str(pdf))
    doc.close()

    boxes = locate_tables(fitz.open(pdf)[0])
    if not boxes:
        pytest.skip("fixture produced no located table box")

    class _Reader:
        timeout = 5.0

        def read(self, *_a, **_k):
            if raises:
                raise RuntimeError("reader blew up")
            return reply

    crops = TableCropExtractor(_Reader()).extract(pdf, 1, boxes, cascade_probe=False)

    assert crops, "a located crop produced no record at all, which is the defect"
    assert all(getattr(c, "_failed", "") == reason for c in crops), (
        f"expected {reason!r} sentinels, got {[getattr(c, '_failed', None) for c in crops]}"
    )
