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


def _state_with_table_page(tmp_path):
    """One page whose winner is a table, ready for a crop reread."""
    from socr.core.document import DocumentHandle
    from socr.core.result import PageOutput, PageStatus
    from socr.core.state import DocumentState

    tmp_path.mkdir(parents=True, exist_ok=True)
    pdf = tmp_path / "d.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "born digital text long enough to be a text layer.")
    doc.save(str(pdf))
    doc.close()

    state = DocumentState(handle=DocumentHandle.from_path(pdf))
    out = PageOutput(
        page_num=1,
        text="| Var | Est |\n| --- | --- |\n| a | 1.0 |",
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=True,
    )
    state.pages[1].attempts.append(out)
    state.pages[1].best_output = out
    return state


@pytest.mark.parametrize(
    "reason", ["render_failed", "read_error", "empty_response", "backend_degraded"]
)
def test_a_failed_crop_emits_distrust_at_the_real_caller(tmp_path, reason: str) -> None:
    """#492: the emit itself, driven through `_reread_page_tables`.

    The tests above assert set membership and extractor sentinels; nothing fed a
    `_failed` crop into the orchestrator, so deleting the emit branch would have
    left them all green. `render_failed` was also the one sentinel with no
    coverage at all -- it is parametrised here with the other three.
    """
    from socr.core.config import EngineType, PipelineConfig
    from socr.pipeline.orchestrator import UnifiedPipeline
    from socr.tables.extract import CropTable

    state = _state_with_table_page(tmp_path / reason)
    pipeline = UnifiedPipeline(
        PipelineConfig(
            primary_engine=EngineType.QWEN, enabled_engines=[EngineType.QWEN], quiet=True
        )
    )

    crop = CropTable(markdown="", source="booktabs", bbox=(10.0, 10.0, 200.0, 100.0))
    crop._failed = reason

    pipeline._reread_page_tables(state, 1, [crop], extractor=object())

    kinds = [getattr(e, "kind", "") for e in state.events]
    assert "dualpass_crop_failed" in kinds, (
        f"a {reason} crop produced no distrust event, so the page keeps its "
        f"incumbent table and looks verified: {kinds}"
    )

    event = next(e for e in state.events if e.kind == "dualpass_crop_failed")
    assert event.page_num == 1
    assert event.data.get("reason") == reason, (
        f"the event does not say WHY the crop failed: {event.data}"
    )

    # And it must reach the trust file, which is the surface a consumer reads.
    trust = build_tables_trust("d.pdf", list(state.events))
    assert trust.untrusted_pages == [1], (
        f"the page is still trusted after a failed reread: {trust.untrusted_pages}"
    )


def test_a_render_failure_yields_a_sentinel_from_the_extractor(tmp_path) -> None:
    """#492 item 2: `render_failed` at its SOURCE, not hand-constructed.

    The parametrised test above builds the crop directly, so it pins the
    orchestrator's handling but not the extractor's production of this
    particular sentinel -- deleting the `render_failed` append left the whole
    suite green. `_render_crop` returning None is the real path.
    """
    from unittest.mock import patch

    from socr.tables.extract import TableCropExtractor
    from socr.tables.locate import TableBox

    tmp_path.mkdir(parents=True, exist_ok=True)
    pdf = tmp_path / "t.pdf"
    doc = fitz.open()
    doc.new_page(width=500, height=600)
    doc.save(str(pdf))
    doc.close()

    boxes = [TableBox(bbox=(100.0, 100.0, 460.0, 250.0), source="booktabs")]

    class _NeverCalled:
        timeout = 5.0

        def read(self, *_a, **_k):
            raise AssertionError("no read may happen when the crop did not render")

    with patch.object(TableCropExtractor, "_render_crop", return_value=None):
        crops = TableCropExtractor(_NeverCalled()).extract(pdf, 1, boxes, cascade_probe=False)

    assert len(crops) == len(boxes), f"a crop that failed to RENDER left no record: {crops}"
    assert all(getattr(c, "_failed", "") == "render_failed" for c in crops), (
        f"expected render_failed, got {[getattr(c, '_failed', None) for c in crops]}"
    )


def test_a_timed_out_crop_emits_distrust_at_the_real_caller(tmp_path) -> None:
    """#495 item 1: the TIMEOUT emit, driven through `_reread_page_tables`.

    `dualpass_crop_timeout` has been in `TABLE_DISTRUST_KINDS` and in
    hand-built `AuditEvent` tests since GH-166, but nothing ever fed a
    `_timed_out` crop into the orchestrator -- so deleting the emit branch left
    every one of those green. It is the older sibling of the `_failed` emit
    pinned above and it means the same thing: the reread never completed, so
    the incumbent table is unverified.
    """
    from socr.core.config import EngineType, PipelineConfig
    from socr.pipeline.orchestrator import UnifiedPipeline
    from socr.tables.extract import CropTable

    state = _state_with_table_page(tmp_path / "timeout")
    pipeline = UnifiedPipeline(
        PipelineConfig(
            primary_engine=EngineType.QWEN, enabled_engines=[EngineType.QWEN], quiet=True
        )
    )

    crop = CropTable(markdown="", source="booktabs", bbox=(10.0, 10.0, 200.0, 100.0))
    crop._timed_out = True

    pipeline._reread_page_tables(state, 1, [crop], extractor=object())

    kinds = [getattr(e, "kind", "") for e in state.events]
    assert "dualpass_crop_timeout" in kinds, (
        f"a timed-out crop produced no distrust event, so the page keeps its "
        f"incumbent table and looks verified: {kinds}"
    )
    event = next(e for e in state.events if e.kind == "dualpass_crop_timeout")
    assert event.page_num == 1

    trust = build_tables_trust("d.pdf", list(state.events))
    assert trust.untrusted_pages == [1], (
        f"the page is still trusted after a timed-out reread: {trust.untrusted_pages}"
    )


def _reread_with(tmp_path, *, partial: str | None):
    """Reconcile one page against a crop that DISAGREES with the incumbent.

    ``partial`` adds a second crop that produced nothing -- ``"failed"`` or
    ``"timeout"`` -- which is what makes the page's crop coverage partial.
    Returns ``(patched_delta, flagged_delta, final_text)``.
    """
    from socr.core.config import EngineType, PipelineConfig
    from socr.pipeline.orchestrator import UnifiedPipeline
    from socr.tables.extract import CropTable

    state = _state_with_table_page(tmp_path)
    incumbent = state.pages[1].best_output.text
    pipeline = UnifiedPipeline(
        PipelineConfig(
            primary_engine=EngineType.QWEN,
            enabled_engines=[EngineType.QWEN],
            quiet=True,
            auto_patch_tables=True,
        )
    )

    good = CropTable(
        markdown=incumbent.replace("1.0", "2.0"), source="booktabs", bbox=(10.0, 10.0, 200.0, 100.0)
    )
    crops = [good]
    if partial is not None:
        dud = CropTable(markdown="", source="booktabs", bbox=(10.0, 110.0, 200.0, 200.0))
        if partial == "failed":
            dud._failed = "read_error"
        else:
            dud._timed_out = True
        crops.append(dud)

    patched, flagged = pipeline._reread_page_tables(state, 1, crops, extractor=object())
    return patched, flagged, state.pages[1].best_output.text


def test_full_crop_coverage_still_auto_patches(tmp_path) -> None:
    """The control the flag-only pins below are a difference FROM.

    Without it, a build that never patched anything at all would satisfy them.
    """
    patched, _flagged, text = _reread_with(tmp_path / "clean", partial=None)
    assert patched == 1, "a disagreeing crop with full coverage did not patch"
    assert "2.0" in text, f"the page kept the incumbent value: {text}"


@pytest.mark.parametrize("partial", ["failed", "timeout"])
def test_partial_crop_coverage_flags_instead_of_patching(tmp_path, partial: str) -> None:
    """#495 item 2: the flag-only gate as an OUTCOME, not an AST identifier.

    `test_a_failed_crop_forces_flag_only` reads the source for the names
    `had_timeout` / `failed_crops` in two assignments. That is brittle both
    ways -- it passes on a rename that keeps the identifier and breaks on one
    that changes it without changing behaviour. This drives the real page:
    one crop disagreeing, one crop that produced nothing, auto-patch ON.

    Patching on partial evidence is the data-loss case the gate exists for --
    the missing crop may have covered the rest of the table.
    """
    patched, flagged, text = _reread_with(tmp_path / partial, partial=partial)

    assert patched == 0, (
        f"a page with one {partial} crop was auto-patched on partial coverage: {text}"
    )
    assert flagged >= 1, "the disagreement was neither patched nor flagged -- it vanished"
    assert "1.0" in text, f"the incumbent text was rewritten anyway: {text}"


def _reread_via_fallback(tmp_path, *, partial: str | None):
    """The SECOND route into a patch: the crop-repair fallback.

    `effective_auto_patch` starts False here (auto-patch is off in config); the
    fallback can turn it back ON when the incumbent table is structurally
    broken and the crop reading strictly repairs it. Gating only the first
    assignment would let a partial-coverage page patch by this route, so it
    carries the same `had_timeout` / `failed_crops` gate -- and needs its own
    outcome pin, because the flag-only tests above never reach it.
    """
    from socr.core.config import EngineType, PipelineConfig
    from socr.core.result import PageOutput, PageStatus
    from socr.pipeline.orchestrator import UnifiedPipeline
    from socr.tables.extract import CropTable

    # Header collapsed to one column against three-column data: a defect
    # `page_needs_crop_repair_fallback` recognises and the crop reading fixes.
    broken = "| Var |\n| --- |\n| a | 1.0 | 2.0 |\n| b | 3.0 | 4.0 |"
    repaired = "| Var | Est | SE |\n| --- | --- | --- |\n| a | 1.0 | 2.0 |\n| b | 3.0 | 4.0 |"

    state = _state_with_table_page(tmp_path)
    out = PageOutput(
        page_num=1, text=broken, status=PageStatus.SUCCESS, engine="qwen", audit_passed=True
    )
    state.pages[1].attempts = [out]
    state.pages[1].best_output = out

    pipeline = UnifiedPipeline(
        PipelineConfig(
            primary_engine=EngineType.QWEN,
            enabled_engines=[EngineType.QWEN],
            quiet=True,
            auto_patch_tables=False,
        )
    )
    crops = [CropTable(markdown=repaired, source="booktabs", bbox=(10.0, 10.0, 200.0, 100.0))]
    if partial is not None:
        dud = CropTable(markdown="", source="booktabs", bbox=(10.0, 110.0, 200.0, 200.0))
        if partial == "failed":
            dud._failed = "read_error"
        else:
            dud._timed_out = True
        crops.append(dud)

    patched, flagged = pipeline._reread_page_tables(state, 1, crops, extractor=object())
    return patched, flagged, state.pages[1].best_output.text


def test_the_crop_repair_fallback_patches_a_broken_header(tmp_path) -> None:
    """Control for the pins below: this route really does patch."""
    patched, _flagged, text = _reread_via_fallback(tmp_path / "fb_clean", partial=None)
    assert patched == 1, "the crop-repair fallback did not fire at all"
    assert "Est" in text, f"the collapsed header was not repaired: {text}"


@pytest.mark.parametrize("partial", ["failed", "timeout"])
def test_the_crop_repair_fallback_is_gated_by_partial_coverage(tmp_path, partial: str) -> None:
    """#495 item 2, second route: partial coverage must block the fallback too.

    This is the route the deleted AST check was really there for -- and the
    only one the flag-only tests above do not reach, because auto-patch is off
    in config and the fallback is what turns it back on.

    Parametrised over BOTH kinds (cubic P2 on #499): with auto-patch off,
    `effective_auto_patch` is False whatever `had_timeout` says, so the
    `and not had_timeout` clause inside `needs_crop_fallback` is the only thing
    standing between a timed-out crop and a patch. A failed-only fixture would
    have left reverting that one clause green.
    """
    patched, flagged, text = _reread_via_fallback(tmp_path / f"fb_{partial}", partial=partial)
    assert patched == 0, (
        f"a page with a {partial} crop was patched through the repair fallback: {text}"
    )
    assert flagged >= 1, (
        "the disagreement was neither patched nor flagged -- a route that "
        "returned (0, 0) would satisfy the assertions above while losing it"
    )
    assert "Est" not in text, f"the incumbent text was rewritten anyway: {text}"
