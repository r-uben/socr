"""R7 part one: the disposition tag on ``_select_page_output_tagged``.

The tag exists so callers stop re-deriving "which branch shipped this page?"
with mirror predicates. That only holds if the tag is TOTAL (every ending
carries one) and EXCLUSIVE (exactly one ending runs). Exclusivity is structural
-- the cascade is loop-free and single-return -- so these tests pin the shape
itself, not a sampled outcome.

Hermetic: pure AST + a wrapper identity check. No provider, no pipeline run.
"""

from __future__ import annotations

import ast
import inspect

from socr.core import manifest
from socr.core.manifest import WinnerKind


def _cascade() -> ast.FunctionDef:
    src = inspect.getsource(manifest)
    return next(
        n
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.FunctionDef) and n.name == "_select_page_output_tagged"
    )


def _returns(fn: ast.FunctionDef) -> list[ast.Return]:
    return [n for n in ast.walk(fn) if isinstance(n, ast.Return)]


def test_cascade_is_loop_free_so_exactly_one_ending_runs() -> None:
    """Exclusivity is a property of the code's SHAPE, not a convention.

    A loop (or a comprehension containing a return, which cannot occur) would
    break the "exactly one ending per page" guarantee the tag depends on.
    """
    fn = _cascade()
    loops = [n for n in ast.walk(fn) if isinstance(n, (ast.For, ast.While))]
    assert loops == [], "a loop in the cascade breaks one-ending-per-page"
    assert len(_returns(fn)) == 15


def test_every_ending_carries_a_tag() -> None:
    """Totality: an untagged return would ship a page no caller can classify."""
    untagged = []
    for r in _returns(_cascade()):
        v = r.value
        ok = (
            isinstance(v, ast.Tuple)
            and len(v.elts) == 2
            and isinstance(v.elts[1], ast.Attribute)
            and isinstance(v.elts[1].value, ast.Name)
            and v.elts[1].value.id == "WinnerKind"
        )
        if not ok:
            untagged.append(r.lineno)
    assert untagged == [], f"untagged returns at lines {untagged}"


def test_tags_and_endings_are_in_bijection() -> None:
    """No dead member, and no two endings sharing a tag.

    A member declared but never returned is a disposition callers will branch on
    and never see. Two endings sharing a tag silently merges two dispositions --
    the exact "counted under a disposition it does not have" bug class R7 exists
    to kill.
    """
    used = [r.value.elts[1].attr for r in _returns(_cascade())]
    assert len(used) == len(set(used)), "two endings share a tag"
    assert set(used) == {k.name for k in WinnerKind}
    assert len({k.value for k in WinnerKind}) == len(list(WinnerKind))


def test_public_wrapper_drops_the_tag_only() -> None:
    """The tag is INTERNAL: the public signature must be unchanged.

    If ``_select_page_output`` ever returned the tuple, every existing caller
    would silently receive a tuple where it expects a PageOutput.
    """
    src = inspect.getsource(manifest._select_page_output)
    assert "_select_page_output_tagged(state, page_num, whole_doc)[0]" in src
    ann = inspect.signature(manifest._select_page_output).return_annotation
    assert ann in ("PageOutput", manifest.PageOutput)
