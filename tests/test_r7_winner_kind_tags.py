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


def _tag_names(ret: ast.Return) -> list[str]:
    """The WinnerKind names an ending can yield, in source order.

    Usually one. An ending whose own body already switches on a flag may yield
    two via a conditional -- ``NATIVE_FALLBACK``/``NATIVE_CLEAN`` do, because
    that single return ships either a demoted fallback or an ordinary native
    success. Returning both names keeps the totality/bijection checks honest
    instead of letting one tag quietly stand for two dispositions.
    """
    v = ret.value
    assert isinstance(v, ast.Tuple) and len(v.elts) == 2, f"untagged return @{ret.lineno}"
    tag = v.elts[1]
    parts = [tag.body, tag.orelse] if isinstance(tag, ast.IfExp) else [tag]
    names = []
    for node in parts:
        assert isinstance(node, ast.Attribute), f"non-WinnerKind tag @{ret.lineno}"
        assert isinstance(node.value, ast.Name) and node.value.id == "WinnerKind"
        names.append(node.attr)
    return names


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
    for r in _returns(_cascade()):
        assert _tag_names(r), f"untagged return at line {r.lineno}"


def test_tags_and_endings_are_in_bijection() -> None:
    """No dead member, and no two endings sharing a tag.

    A member declared but never returned is a disposition callers will branch on
    and never see. Two endings sharing a tag silently merges two dispositions --
    the exact "counted under a disposition it does not have" bug class R7 exists
    to kill.
    """
    used = [n for r in _returns(_cascade()) for n in _tag_names(r)]
    assert len(used) == len(set(used)), "two endings share a tag"
    assert set(used) == {k.name for k in WinnerKind}
    assert len({k.value for k in WinnerKind}) == len(list(WinnerKind))


def test_tag_order_matches_enum_declaration_order() -> None:
    """Precedence lives in the cascade's order, so the enum must mirror it.

    ``WinnerKind``'s docstring makes definition order load-bearing: it is the
    record of which ending outranks which, and part two is told to treat it as
    the authority. A set comparison cannot see order, so on its own it would let
    two tags be transposed -- or the enum reordered by an alphabetising autofix --
    while totality, exclusivity and bijection all still hold. That failure is
    silent and hands every downstream caller a systematically wrong disposition.

    Pinning source order against declaration order also catches a hand-transposed
    pair, which is the one defect the AST checks are structurally blind to.
    """
    fn = _cascade()
    in_source = [n for r in sorted(_returns(fn), key=lambda r: r.lineno) for n in _tag_names(r)]
    assert in_source == [k.name for k in WinnerKind]


def test_public_wrapper_returns_the_output_not_the_tuple() -> None:
    """The tag is INTERNAL: the public function must still yield a bare PageOutput.

    Checked by CALLING the wrapper, not by matching its source text: a substring
    assertion breaks on harmless reformatting and passes if the string only ever
    appears in a comment. Patching the tagged form proves the wrapper really
    delegates and really drops element 1.
    """
    sentinel = object()
    calls: list[tuple] = []

    def _fake(state, page_num, whole_doc=None):
        calls.append((state, page_num, whole_doc))
        return sentinel, WinnerKind.PASSING_BEST_OUTPUT

    original = manifest._select_page_output_tagged
    manifest._select_page_output_tagged = _fake
    try:
        got = manifest._select_page_output("STATE", 7, "WHOLE")
    finally:
        manifest._select_page_output_tagged = original

    assert got is sentinel, "wrapper did not delegate, or did not drop the tag"
    assert calls == [("STATE", 7, "WHOLE")], "wrapper dropped or reordered arguments"

    ann = inspect.signature(manifest._select_page_output).return_annotation
    assert ann in ("PageOutput", manifest.PageOutput)
