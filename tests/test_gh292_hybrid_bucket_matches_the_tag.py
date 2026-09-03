"""GH-292: `corrupt_math_hybrid_pages` must name the disposition that ships.

The bucket was `corrupt_math_hybrid is not None`, but the manifest ships the
hybrid only when four further conditions hold. It was therefore strictly
broader than the disposition it names, and diverged in 1,984 of 4,096 synthetic
states -- always claiming pages the manifest ships as something else.

Both consequences are content-visible, and they mirror each other:

- a genuine native-fallback page is dropped from `native_fallback_pages`, whose
  exclusion rests on the false premise "the hybrid ships" -- a real defect
  hidden;
- a page shipping NATIVE_CLEAN (SUCCESS, audit_passed True) drives its document
  to AUDIT_FAILED -- a non-existent defect reported.

The orchestrator's bucket contract is pinned against the finalized record's exact
disposition pair rather than a re-derived predicate. The record's selection tag
and the PageState hybrid flag remain useful provenance, but neither describes the
bytes that ultimately ship.
"""

from __future__ import annotations

import pytest

fitz = pytest.importorskip("fitz")

from socr.core.document import DocumentHandle  # noqa: E402
from socr.core.manifest import (  # noqa: E402
    FinalizedPageRecord,
    PageDisposition,
    PageEnding,
    PagePrimaryReason,
    SelectionProvenance,
    _select_page_output_tagged,
    page_disposition,
)
from socr.core.result import FailureMode, PageOutput, PageStatus  # noqa: E402
from socr.core.state import DocumentState  # noqa: E402
from socr.pipeline.orchestrator import _derive_disposition_buckets  # noqa: E402


def _pdf(tmp_path):
    path = tmp_path / "doc.pdf"
    doc = fitz.open()
    doc.new_page().insert_text(
        (54, 72), "born-digital prose long enough to count as a real text layer here."
    )
    doc.save(path)
    doc.close()
    return path


def _state(tmp_path, shape: str):
    """A born-digital page with `corrupt_math_hybrid` set, in one of four shapes."""
    state = DocumentState(handle=DocumentHandle.from_path(_pdf(tmp_path)))
    p = state.pages[1]
    p.is_born_digital = True
    p.native_text = "native prose with a corrupt equation"

    engine = "qwen" if shape == "wrong_engine" else "native+math"
    hybrid = PageOutput(
        page_num=1,
        text="native prose plus crop-backed region candidate",
        status=PageStatus.WARNING,
        engine=engine,
        audit_passed=False,
        failure_mode=FailureMode.AUDIT_FAILED,
    )
    if shape != "not_in_attempts":
        p.attempts.append(hybrid)
    if shape == "shredded":
        p.native_rotated_text_shredded = True
    p.corrupt_math_hybrid = hybrid
    return state, p


# The predicate the bucket used before this fix.
def _old_bucket_predicate(p) -> bool:
    return getattr(p, "corrupt_math_hybrid", None) is not None


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        ("ships", SelectionProvenance.CORRUPT_MATH_HYBRID),
        ("shredded", SelectionProvenance.ROTATED_TEXT_SHREDDED),
        ("not_in_attempts", SelectionProvenance.NATIVE_CLEAN),
        ("wrong_engine", SelectionProvenance.NATIVE_CLEAN),
    ],
)
def test_the_tag_is_what_the_page_actually_ships(tmp_path, shape, expected) -> None:
    """The three divergent shapes from the ticket, plus the one that does ship."""
    state, p = _state(tmp_path, shape)
    assert _select_page_output_tagged(state, 1)[1] is expected

    # The anchor: the OLD predicate claims all four, which is the bug.
    assert _old_bucket_predicate(p) is True, (
        f"{shape}: fixture no longer exercises the old predicate, so this measures nothing"
    )


@pytest.mark.parametrize("shape", ["not_in_attempts", "wrong_engine"])
def test_a_clean_page_is_not_claimed_as_a_hybrid(tmp_path, shape) -> None:
    """The serious half: these pages are ordinary clean native successes.

    The bucket feeds `pages_ok`, so claiming them drove the document to
    AUDIT_FAILED and emitted a corrupt-math event for a defect the page does
    not have.
    """
    state, _ = _state(tmp_path, shape)
    output, tag = _select_page_output_tagged(state, 1)

    assert tag is not SelectionProvenance.CORRUPT_MATH_HYBRID
    assert output.status is PageStatus.SUCCESS, (
        f"{shape}: fixture must ship a CLEAN page, or the point is lost"
    )
    assert output.audit_passed is True


def test_the_shipping_shape_is_still_claimed(tmp_path) -> None:
    """Control: the fix must not empty the bucket.

    A hybrid that genuinely ships must still be reported, or #292 would have
    traded a false positive for a false negative.
    """
    state, _ = _state(tmp_path, "ships")
    assert _select_page_output_tagged(state, 1)[1] is SelectionProvenance.CORRUPT_MATH_HYBRID
    assert page_disposition(state, 1).ending is PageEnding.MODEL_OUTPUT
    assert page_disposition(state, 1).primary_reason is PagePrimaryReason.CORRUPT_MATH_HYBRID


class TestTheOrchestratorBucketReadsTheShippedDisposition:
    """P6 stage C retarget of the production-line pin.

    Stage B's version of this test required an inline comparison against a
    ``CORRUPT_MATH_HYBRID`` enum member INSIDE the helper body, reading
    ``selection_provenance``. Stage C moves membership to a module-level
    ``{bucket_name: PageDisposition}`` mapping compared against
    ``record.disposition`` (see ``tests/test_p6_stage_c_bucket_contract.py``,
    which pins that ``selection_provenance`` is never compared at all). An
    inline-attribute AST search would therefore conflict with the stage-C shape
    it is meant to allow, so this class proves the SAME dataflow guarantee --
    "ask the manifest which disposition the page shipped, never re-derive it"
    -- behaviourally instead: both ``ending`` AND ``primary_reason`` of the
    finalized record's ``disposition`` must matter, not just one.
    """

    def _record(self, ending: PageEnding, primary_reason: PagePrimaryReason):
        return FinalizedPageRecord(
            output=PageOutput(page_num=1, text="text"),
            disposition=PageDisposition(ending=ending, primary_reason=primary_reason),
            # Deliberately the WRONG provenance tag: membership must not follow it.
            selection_provenance=SelectionProvenance.PASSING_BEST_OUTPUT,
        )

    def test_the_exact_hybrid_pair_claims_the_page(self, tmp_path) -> None:
        state = _empty_state(tmp_path)
        rec = self._record(PageEnding.MODEL_OUTPUT, PagePrimaryReason.CORRUPT_MATH_HYBRID)
        buckets = _derive_disposition_buckets(state, [rec])
        assert buckets["corrupt_math_hybrid_pages"] == {1}

    def test_the_right_reason_with_the_wrong_ending_does_not_claim(self, tmp_path) -> None:
        """``ending`` must flow through too -- a reason-only match is not enough."""
        state = _empty_state(tmp_path)
        rec = self._record(PageEnding.FAIL_CLOSED_MARKER, PagePrimaryReason.CORRUPT_MATH_HYBRID)
        buckets = _derive_disposition_buckets(state, [rec])
        assert 1 not in buckets["corrupt_math_hybrid_pages"]

    def test_the_right_ending_with_the_wrong_reason_does_not_claim(self, tmp_path) -> None:
        """``primary_reason`` must flow through too -- an ending-only match is not enough."""
        state = _empty_state(tmp_path)
        rec = self._record(PageEnding.MODEL_OUTPUT, PagePrimaryReason.ACCEPTED_OUTPUT)
        buckets = _derive_disposition_buckets(state, [rec])
        assert 1 not in buckets["corrupt_math_hybrid_pages"]


def _empty_state(tmp_path):
    path = tmp_path / "doc.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((54, 72), "born-digital prose.")
    doc.save(path)
    doc.close()
    return DocumentState(handle=DocumentHandle.from_path(path))


def test_the_orchestrator_bucket_does_not_re_derive_the_page_state_flag() -> None:
    """The original #292 defect: reading ``PageState.corrupt_math_hybrid`` directly
    instead of asking the finalized record what shipped. Still forbidden under the
    stage-C disposition-pair contract.
    """
    import ast
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "socr"
    tree = ast.parse((src / "pipeline" / "orchestrator.py").read_text())

    derive_fn = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_derive_disposition_buckets"
    ]
    assert len(derive_fn) == 1, "expected _derive_disposition_buckets helper in orchestrator.py"
    fn_body = derive_fn[0]

    flag_reads = [
        node
        for node in ast.walk(fn_body)
        if (isinstance(node, ast.Attribute) and node.attr == "corrupt_math_hybrid")
        or (isinstance(node, ast.Constant) and node.value == "corrupt_math_hybrid")
    ]
    assert not flag_reads, (
        f"the bucket helper is re-deriving the PageState flag again: {ast.unparse(fn_body)}"
    )
