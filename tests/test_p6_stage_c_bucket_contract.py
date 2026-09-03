"""P6 stage C: the three migrated buckets are exact-disposition-pair derived.

Design: `docs/log/2026-09-02_p6-selector-collapse-design.md` (S8 Q3, S9 amendment)
and `docs/log/2026-09-02_p6-stage-ab-disposition-contract.md` (the stage-B
compromise). Stage C's own log is `docs/log/2026-09-03_p6-stage-c-shipped-buckets.md`.

Stage B derived `structure_class_model_pages`, `structure_class_floor_pages` and
`corrupt_math_hybrid_pages` from `SelectionProvenance` (which branch selection
took), not from the public `PageDisposition` (what actually shipped), to stay
byte-preserving. The consequence: a hybrid whose candidate was rewritten by the
post-selection emission guard to `(FAIL_CLOSED_MARKER, INVALID_TABLE_EMISSION)`
was still counted as `corrupt_math_hybrid_shipped` -- bytes that did not ship.

Stage C's contract: each of the three buckets is EXACTLY the set of pages whose
finalized `PageDisposition` equals one pinned pair --

    structure_class_model_pages  -> (MODEL_OUTPUT, STRUCTURE_CLASS)
    structure_class_floor_pages  -> (FAIL_CLOSED_MARKER, STRUCTURE_CLASS)
    corrupt_math_hybrid_pages    -> (MODEL_OUTPUT, CORRUPT_MATH_HYBRID)

`SelectionProvenance` is no longer consulted for membership; it may still be
recorded as provenance in audit-event DATA (nothing here forbids that), but this
module asserts membership never reads it.

Hermetic: constructs `DocumentState` / `FinalizedPageRecord` fixtures directly
(reusing `tests/p6_corpus_fixture.py`'s stage-C corpus), no provider, no pipeline
run.
"""

from __future__ import annotations

import pytest

fitz = pytest.importorskip("fitz")

from p6_corpus_fixture import (  # noqa: E402
    HYBRID_CLEAN_PAGE,
    HYBRID_REWRITTEN_PAGE,
    STRUCT_FLOOR_PAGE,
    STRUCT_MODEL_PASSING_PAGE,
    STRUCT_MODEL_REWRITTEN_PAGE,
    build_stage_c_corpus_state,
    make_stage_c_pdf,
)
from socr.core.document import DocumentHandle  # noqa: E402
from socr.core.manifest import (  # noqa: E402
    FinalizedPageRecord,
    PageDisposition,
    PageEnding,
    PagePrimaryReason,
    SelectionProvenance,
    finalized_page_records,
)
from socr.core.result import PageOutput  # noqa: E402
from socr.core.state import DocumentState  # noqa: E402
from socr.pipeline.orchestrator import _derive_disposition_buckets  # noqa: E402

#: The stage-C contract: bucket name -> its exact PageDisposition pair.
DISPOSITION_BUCKET_PAIRS = {
    "structure_class_model_pages": PageDisposition(
        PageEnding.MODEL_OUTPUT, PagePrimaryReason.STRUCTURE_CLASS
    ),
    "structure_class_floor_pages": PageDisposition(
        PageEnding.FAIL_CLOSED_MARKER, PagePrimaryReason.STRUCTURE_CLASS
    ),
    "corrupt_math_hybrid_pages": PageDisposition(
        PageEnding.MODEL_OUTPUT, PagePrimaryReason.CORRUPT_MATH_HYBRID
    ),
}


def _new_state(tmp_path, page_count: int = 2) -> DocumentState:
    path = tmp_path / "doc.pdf"
    doc = fitz.open()
    for _ in range(page_count):
        doc.new_page().insert_text(
            (54, 72), "born-digital prose long enough to count as a real text layer here."
        )
    doc.save(path)
    doc.close()
    return DocumentState(handle=DocumentHandle.from_path(path))


def _new_buckets(state) -> dict[str, set[int]]:
    return _derive_disposition_buckets(state, finalized_page_records(state))


class TestExactDispositionPairClaimsRegardlessOfProvenance:
    """Direction one: the exact pair claims the page, no matter which branch chose it."""

    @pytest.mark.parametrize(
        "bucket_name, pair",
        sorted(DISPOSITION_BUCKET_PAIRS.items(), key=lambda kv: kv[0]),
    )
    def test_matching_disposition_claims_the_page(self, tmp_path, bucket_name, pair) -> None:
        state = _new_state(tmp_path, page_count=1)
        # A hand-built record with a DIFFERENT selection provenance than the
        # historical stage-B tag set for this bucket, so a provenance-reading
        # implementation would miss it.
        rec = FinalizedPageRecord(
            output=PageOutput(page_num=1, text="text"),
            disposition=pair,
            selection_provenance=SelectionProvenance.PASSING_BEST_OUTPUT,
        )
        buckets = _derive_disposition_buckets(state, [rec])
        assert buckets[bucket_name] == {1}, (bucket_name, pair)
        for other_name, pages in buckets.items():
            if other_name != bucket_name:
                assert 1 not in pages, (bucket_name, other_name)


class TestFormerProvenanceWithADifferentDispositionDoesNotClaim:
    """Direction two: the former provenance tag alone is not enough any more."""

    #: The pre-stage-C provenance tag each migrated bucket used to key on.
    FORMER_PROVENANCE = {
        "structure_class_model_pages": SelectionProvenance.STRUCTURE_CLASS_GRID_PASSING,
        "structure_class_floor_pages": SelectionProvenance.STRUCTURE_CLASS_FLOOR,
        "corrupt_math_hybrid_pages": SelectionProvenance.CORRUPT_MATH_HYBRID,
    }

    @pytest.mark.parametrize(
        "bucket_name, tag", sorted(FORMER_PROVENANCE.items(), key=lambda kv: kv[0])
    )
    def test_former_tag_with_a_rewritten_disposition_does_not_claim(
        self, tmp_path, bucket_name, tag
    ) -> None:
        state = _new_state(tmp_path, page_count=1)
        rec = FinalizedPageRecord(
            output=PageOutput(page_num=1, text="text"),
            disposition=PageDisposition(
                ending=PageEnding.FAIL_CLOSED_MARKER,
                primary_reason=PagePrimaryReason.INVALID_TABLE_EMISSION,
            ),
            selection_provenance=tag,
        )
        buckets = _derive_disposition_buckets(state, [rec])
        for name in DISPOSITION_BUCKET_PAIRS:
            assert 1 not in buckets[name], (bucket_name, tag, name)


class TestGuardRewrittenPagesUsingTheRealCorpus:
    """The two named guard-rewritten cases from the stage-C corpus fixture.

    Inverts stage B's `test_a_guard_rewritten_page_keeps_its_tag_derived_bucket`:
    a page whose finalized bytes are a fail-closed marker is counted ONLY as
    failed, never as "shipped hybrid" / "grid passing" / "grid flagged".
    """

    def test_rewritten_structure_class_page_is_absent_from_all_three_migrated_buckets(
        self, tmp_path
    ) -> None:
        state = build_stage_c_corpus_state(make_stage_c_pdf(tmp_path))
        records = finalized_page_records(state)
        rec = next(r for r in records if r.output.page_num == STRUCT_MODEL_REWRITTEN_PAGE)
        assert rec.disposition == PageDisposition(
            PageEnding.FAIL_CLOSED_MARKER, PagePrimaryReason.INVALID_TABLE_EMISSION
        ), "fixture no longer exercises the guard rewrite -- this test measures nothing"
        assert rec.selection_provenance in (
            SelectionProvenance.STRUCTURE_CLASS_GRID_PASSING,
            SelectionProvenance.STRUCTURE_CLASS_GRID_FLAGGED,
        ), "provenance must still say a grid was chosen, or the migration proves nothing"

        buckets = _derive_disposition_buckets(state, records)
        for name in DISPOSITION_BUCKET_PAIRS:
            assert STRUCT_MODEL_REWRITTEN_PAGE not in buckets[name], name

    def test_rewritten_hybrid_page_is_absent_from_all_three_migrated_buckets(
        self, tmp_path
    ) -> None:
        state = build_stage_c_corpus_state(make_stage_c_pdf(tmp_path))
        records = finalized_page_records(state)
        rec = next(r for r in records if r.output.page_num == HYBRID_REWRITTEN_PAGE)
        assert rec.disposition == PageDisposition(
            PageEnding.FAIL_CLOSED_MARKER, PagePrimaryReason.INVALID_TABLE_EMISSION
        ), "fixture no longer exercises the guard rewrite -- this test measures nothing"
        assert rec.selection_provenance is SelectionProvenance.CORRUPT_MATH_HYBRID

        buckets = _derive_disposition_buckets(state, records)
        for name in DISPOSITION_BUCKET_PAIRS:
            assert HYBRID_REWRITTEN_PAGE not in buckets[name], name

    def test_genuine_structure_floor_stays_in_its_bucket(self, tmp_path) -> None:
        """A REAL floor (no attempt authored a grid) is not the rewritten case.

        The migration must not turn `structure_class_floor_pages` into an empty
        set -- only guard-rewritten pages lose their former bucket. A genuine
        `(FAIL_CLOSED_MARKER, STRUCTURE_CLASS)` page keeps its membership.
        """
        state = build_stage_c_corpus_state(make_stage_c_pdf(tmp_path))
        records = finalized_page_records(state)
        rec = next(r for r in records if r.output.page_num == STRUCT_FLOOR_PAGE)
        assert rec.disposition == PageDisposition(
            PageEnding.FAIL_CLOSED_MARKER, PagePrimaryReason.STRUCTURE_CLASS
        )
        buckets = _derive_disposition_buckets(state, records)
        assert STRUCT_FLOOR_PAGE in buckets["structure_class_floor_pages"]

    def test_clean_controls_keep_their_bucket(self, tmp_path) -> None:
        """The two clean (non-rewritten) named cases are unaffected by the migration."""
        state = build_stage_c_corpus_state(make_stage_c_pdf(tmp_path))
        buckets = _new_buckets(state)
        assert STRUCT_MODEL_PASSING_PAGE in buckets["structure_class_model_pages"]
        assert HYBRID_CLEAN_PAGE in buckets["corrupt_math_hybrid_pages"]


def test_selection_provenance_is_never_read_for_membership() -> None:
    """Structural pin: `_derive_disposition_buckets` must not branch on
    `selection_provenance` for the three migrated buckets any more.

    A bare mention (e.g. still recording it into audit-event DATA elsewhere) is
    fine; what is forbidden is comparing a record's `selection_provenance`
    against a `SelectionProvenance` member to decide membership in one of the
    three migrated buckets. `d3_model_table_pages`, `d3_floor_pages` and
    `flagged_model_pages` are untouched by this pin -- they stay flag-derived and
    never read provenance either way.
    """
    import ast
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "socr"
    tree = ast.parse((src / "pipeline" / "orchestrator.py").read_text())
    fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_derive_disposition_buckets"
    )

    provenance_compares = [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.Compare)
        and any(
            (isinstance(side, ast.Attribute) and side.attr == "selection_provenance")
            for side in (node.left, *node.comparators)
        )
    ]
    assert not provenance_compares, (
        "_derive_disposition_buckets still compares selection_provenance for "
        f"membership -- the stage-C contract reads only .disposition: "
        f"{[ast.unparse(c) for c in provenance_compares]}"
    )
