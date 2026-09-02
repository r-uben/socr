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

Pinned against the manifest's own tag rather than a re-derived predicate, since
re-derivation is what produced the bug.
"""

from __future__ import annotations

import pytest

fitz = pytest.importorskip("fitz")

from socr.core.document import DocumentHandle  # noqa: E402
from socr.core.manifest import (  # noqa: E402
    PageEnding,
    PagePrimaryReason,
    SelectionProvenance,
    _select_page_output_tagged,
    page_disposition,
)
from socr.core.result import FailureMode, PageOutput, PageStatus  # noqa: E402
from socr.core.state import DocumentState  # noqa: E402


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


def test_the_orchestrator_bucket_itself_reads_the_tag() -> None:
    """The bucket in `orchestrator.py`, not just the tag it should consult.

    Pinning `page_disposition` alone would stay green if the orchestrator
    went back to re-deriving the predicate -- which is exactly how #292
    happened. This asserts the production line.
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

    # P6 stage A/B: the bucket reads the SELECTION TAG rather than the public
    # disposition, because the two differ by the post-selection guards, which can
    # rewrite the hybrid ending to INVALID_TABLE_EMISSION and would silently empty
    # this bucket. GH-292's demand is unchanged either way -- ask the manifest which
    # ending selection took, never re-derive it -- so the assertion below accepts
    # either vocabulary but keeps the RHS-SPECIFIC dataflow shape the original test
    # proved (cold review round 2, finding 9): the CORRUPT_MATH_HYBRID member must
    # be compared against something that came out of the finalized record, not
    # merely mentioned somewhere in the function.
    tag_vocabularies = {"SelectionProvenance", "PagePrimaryReason", "WinnerKind"}
    record_fields = {"selection_provenance", "disposition", "primary_reason", "ending"}

    def _from_the_record(node: ast.AST, bound: dict[str, bool]) -> bool:
        """True if *node* is a record field access, or a local bound to one."""
        if isinstance(node, ast.Attribute):
            return node.attr in record_fields or _from_the_record(node.value, bound)
        if isinstance(node, ast.Name):
            return bound.get(node.id, False)
        return False

    # Locals assigned from a record field, e.g. ``tag = r.selection_provenance``.
    bound: dict[str, bool] = {}
    for node in ast.walk(fn_body):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                bound[target.id] = _from_the_record(node.value, bound)

    dataflow_compares = [
        node
        for node in ast.walk(fn_body)
        if isinstance(node, ast.Compare)
        and _from_the_record(node.left, bound)
        and any(
            isinstance(cmp_, ast.Attribute)
            and cmp_.attr == "CORRUPT_MATH_HYBRID"
            and isinstance(cmp_.value, ast.Name)
            and cmp_.value.id in tag_vocabularies
            for cmp_ in node.comparators
        )
    ]
    assert dataflow_compares, (
        "the corrupt-math bucket no longer compares a value taken FROM THE FINALIZED "
        "RECORD against a CORRUPT_MATH_HYBRID tag member -- a bare mention of the "
        f"member anywhere in the helper is not the pin: {ast.unparse(fn_body)}"
    )

    # The defect was reading the PageState flag directly.
    flag_reads = [
        node
        for node in ast.walk(fn_body)
        if (isinstance(node, ast.Attribute) and node.attr == "corrupt_math_hybrid")
        or (isinstance(node, ast.Constant) and node.value == "corrupt_math_hybrid")
    ]
    assert not flag_reads, (
        f"the bucket helper is re-deriving the PageState flag again: {ast.unparse(fn_body)}"
    )
