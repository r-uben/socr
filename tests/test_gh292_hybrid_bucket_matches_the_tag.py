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
from socr.core.manifest import WinnerKind, shipped_winner_kind  # noqa: E402
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
        ("ships", WinnerKind.CORRUPT_MATH_HYBRID),
        ("shredded", WinnerKind.ROTATED_TEXT_SHREDDED),
        ("not_in_attempts", WinnerKind.NATIVE_CLEAN),
        ("wrong_engine", WinnerKind.NATIVE_CLEAN),
    ],
)
def test_the_tag_is_what_the_page_actually_ships(tmp_path, shape, expected) -> None:
    """The three divergent shapes from the ticket, plus the one that does ship."""
    state, p = _state(tmp_path, shape)
    assert shipped_winner_kind(state, 1) is expected

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
    from socr.core.manifest import _select_page_output_tagged

    state, _ = _state(tmp_path, shape)
    output, tag = _select_page_output_tagged(state, 1)

    assert tag is not WinnerKind.CORRUPT_MATH_HYBRID
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
    assert shipped_winner_kind(state, 1) is WinnerKind.CORRUPT_MATH_HYBRID


def test_the_orchestrator_bucket_itself_reads_the_tag() -> None:
    """The bucket in `orchestrator.py`, not just the tag it should consult.

    Pinning `shipped_winner_kind` alone would stay green if the orchestrator
    went back to re-deriving the predicate -- which is exactly how #292
    happened. This asserts the production line.
    """
    import ast
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "socr"
    tree = ast.parse((src / "pipeline" / "orchestrator.py").read_text())

    assigns = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "corrupt_math_hybrid_pages" for t in node.targets
        )
    ]
    assert len(assigns) == 1, (
        f"expected exactly one corrupt_math_hybrid_pages assignment, got {len(assigns)}"
    )

    rhs = assigns[0].value
    source = ast.unparse(assigns[0])
    assert "shipped_winner_kind" in source, (
        f"the bucket no longer asks the manifest what ships: {source}"
    )
    assert "CORRUPT_MATH_HYBRID" in source, (
        f"the bucket no longer compares against the disposition: {source}"
    )

    # The defect was reading the PageState flag directly. Look for the attribute
    # access itself, not the substring -- the variable name contains it too.
    flag_reads = [
        node
        for node in ast.walk(rhs)
        if (isinstance(node, ast.Attribute) and node.attr == "corrupt_math_hybrid")
        or (isinstance(node, ast.Constant) and node.value == "corrupt_math_hybrid")
    ]
    assert not flag_reads, f"the bucket is re-deriving the PageState flag again: {source}"
