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


def test_a_page_skipped_by_the_cascade_guard_still_leaves_a_record(tmp_path) -> None:
    """#489 review (P1): the skip is the same defect one level up.

    When a prior timeout degrades the backend, the extractor skips the page's
    remaining crops. Breaking with no sentinel produced an empty `raw_crops`,
    so the orchestrator had nothing to iterate, emitted no distrust, and the
    SKIPPED page looked verified.
    """
    from socr.tables.extract import TableCropExtractor
    from socr.tables.locate import TableBox

    pdf = tmp_path / "t.pdf"
    doc = fitz.open()
    doc.new_page(width=500, height=600)
    doc.save(str(pdf))
    doc.close()

    boxes = [
        TableBox(bbox=(100.0, 100.0, 460.0, 250.0), source="booktabs"),
        TableBox(bbox=(100.0, 300.0, 460.0, 450.0), source="booktabs"),
    ]

    class _NeverCalled:
        timeout = 5.0

        def read(self, *_a, **_k):
            raise AssertionError("no VLM call may fire once the backend is degraded")

    extractor = TableCropExtractor(_NeverCalled())
    extractor._backend_degraded = True
    crops = extractor.extract(pdf, 1, boxes, cascade_probe=False)

    assert len(crops) == len(boxes), (
        f"a skipped page left {len(crops)} records for {len(boxes)} located "
        "boxes, so the skip is invisible downstream"
    )
    assert all(getattr(c, "_failed", "") == "backend_degraded" for c in crops)


def test_a_failed_crop_forces_flag_only() -> None:
    """#489 review (P2): partial coverage must not auto-patch.

    The existing comment says patching on incomplete evidence risks data loss,
    and gates that on `had_timeout` alone. A crop that FAILED leaves exactly the
    same partial coverage, so it must gate too -- otherwise one failed crop
    beside one successful crop still patches the page.
    """
    import ast
    import pathlib

    src = (
        pathlib.Path(__file__).resolve().parents[1]
        / "src"
        / "socr"
        / "pipeline"
        / "orchestrator.py"
    )
    tree = ast.parse(src.read_text())

    # Two gates decide whether a page may be patched: the initial
    # `effective_auto_patch`, and `needs_crop_fallback`, which can turn it back
    # ON. Gating only the first would let a failed-crop page patch by the second
    # route, so both are pinned.
    def _gated_rhs(name: str) -> str:
        assigns = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
        ]
        gated = [a for a in assigns if "had_timeout" in ast.unparse(a.value)]
        assert gated, f"no {name} assignment carries the coverage gate at all"
        return ast.unparse(gated[0].value)

    for name in ("effective_auto_patch", "needs_crop_fallback"):
        rhs = _gated_rhs(name)
        assert "had_timeout" in rhs, f"{name}: the timeout gate was lost: {rhs}"
        assert "failed_crops" in rhs, (
            f"{name}: a failed crop does not force flag-only, so a page with "
            f"partial crop coverage can still be auto-patched: {rhs}"
        )
