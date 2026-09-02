"""GH-293: a page that ships a failure marker is not a native fallback.

`native_fallback_pages` means, in its own words, "OCR was tried and never
passed" -- the page shipped its NATIVE BODY, demoted. A born-digital page with
`native_rotated_text_shredded` takes the `ROTATED_TEXT_SHREDDED` ending and
ships an explicit failure marker instead, with no native body at all. It was in
this list anyway, and also in `failed_pages` (derived from the shipped text) --
two audit events and two CLI lines for one page, with the native_fallback line
asserting something false.

Every sibling bucket that can collide with a fail-closed ending carries an
explicit exclusion. GH-263 added the shredded lane and never got one.

The fix excludes `failed_pages` rather than naming the shredded ending, which
closes the CLASS: the acceptance item "audit the other five include-clause
flags for the same gap" is satisfied by construction, since the exclusion keys
on the SHIPPED TEXT rather than on any particular flag.
"""

from __future__ import annotations

import pytest

fitz = pytest.importorskip("fitz")

from socr.core.document import DocumentHandle  # noqa: E402
from socr.core.manifest import (  # noqa: E402
    PageEnding,
    SelectionProvenance,
    _select_page_output_tagged,
    is_page_failed_marker,
    page_disposition,
)
from socr.core.result import FailureMode, PageOutput, PageStatus  # noqa: E402
from socr.core.state import DocumentState  # noqa: E402

# Every flag on the bucket's include-clause OR. The ticket asks for all six to
# be audited, not just the one the sweep happened to surface.
INCLUDE_FLAGS = [
    "needs_ocr_enhancement",
    "native_table_structure_failed",
    "native_table_unverifiable",
    "native_table_structure_defective",
    "native_table_header_unattributed",
    "chart_asset_render_failed",
]


def _state(tmp_path, include_flag: str, *, shredded: bool):
    path = tmp_path / "doc.pdf"
    doc = fitz.open()
    doc.new_page().insert_text(
        (54, 72), "born-digital prose long enough to count as a real text layer here."
    )
    doc.save(path)
    doc.close()

    state = DocumentState(handle=DocumentHandle.from_path(path))
    p = state.pages[1]
    p.is_born_digital = True
    p.native_text = "native body"
    setattr(p, include_flag, True)
    if shredded:
        p.native_rotated_text_shredded = True
    failing = PageOutput(
        page_num=1,
        text="ocr attempt that never passed",
        status=PageStatus.ERROR,
        engine="qwen",
        audit_passed=False,
        failure_mode=FailureMode.AUDIT_FAILED,
    )
    p.attempts.append(failing)
    p.best_output = failing
    return state, p


def _includes(p) -> bool:
    """The bucket's include-clause, before any exclusion."""
    return bool(
        p.is_born_digital
        and p.native_text
        and any(getattr(p, f, False) for f in INCLUDE_FLAGS)
        and p.attempts
        and not (p.best_output and p.best_output.audit_passed)
    )


@pytest.mark.parametrize("include_flag", INCLUDE_FLAGS)
def test_a_shredded_page_ships_a_marker_under_every_include_flag(tmp_path, include_flag) -> None:
    """The gap is not specific to the flag the sweep found it with.

    Whichever of the six reasons puts the page in the bucket, a shredded page
    ships the marker -- so the exclusion cannot be attached to one flag.
    """
    state, p = _state(tmp_path, include_flag, shredded=True)

    assert _includes(p) is True, (
        f"{include_flag}: fixture does not reach the include-clause, so it measures nothing"
    )
    output, tag = _select_page_output_tagged(state, 1)
    assert tag is SelectionProvenance.ROTATED_TEXT_SHREDDED
    assert output.status is PageStatus.ERROR
    assert is_page_failed_marker(output.text or ""), (
        f"{include_flag}: expected a failure marker, got {output.text!r}"
    )
    assert "native body" not in (output.text or ""), (
        "no native text ships, which is why this is not a native FALLBACK"
    )


@pytest.mark.parametrize("include_flag", INCLUDE_FLAGS)
def test_an_unshredded_page_still_ships_its_native_body(tmp_path, include_flag) -> None:
    """Control: the exclusion must not empty the bucket.

    The same page without shredding is a genuine native fallback -- it ships
    the native body demoted -- and must stay in the list.
    """
    state, p = _state(tmp_path, include_flag, shredded=False)

    assert _includes(p) is True
    output = _select_page_output_tagged(state, 1)[0]
    assert not is_page_failed_marker(output.text or ""), (
        f"{include_flag}: control page ships a marker, so it is not a valid control"
    )
    assert page_disposition(state, 1).ending is not PageEnding.FAIL_CLOSED_MARKER


def test_the_bucket_excludes_failed_pages_not_just_the_shredded_ending() -> None:
    """The production line, resolved from the AST.

    Naming the shredded ending would close the instance; excluding
    `failed_pages` closes the class, because that list is derived from the
    SHIPPED TEXT. Pinned so a later edit cannot narrow it back to one ending.
    """
    import ast
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "socr"
    tree = ast.parse((src / "pipeline" / "orchestrator.py").read_text())

    assigns = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "native_fallback_pages" for t in node.targets)
    ]
    # There are several assignments to this name (a chart-winner prune, a
    # resume prune). The BUCKET is the one built from `state.pages`, not one
    # that filters the list already built.
    buckets = [
        a
        for a in assigns
        if isinstance(a.value, ast.ListComp)
        and any(
            isinstance(node, ast.Attribute) and node.attr == "pages"
            for gen in a.value.generators
            for node in ast.walk(gen.iter)
        )
    ]
    assert len(buckets) == 1, (
        f"expected exactly one native_fallback_pages bucket over state.pages, got {len(buckets)}"
    )
    bucket = buckets[0]

    excludes_failed = any(
        isinstance(node, ast.Compare)
        and any(isinstance(op, ast.NotIn) for op in node.ops)
        and any(isinstance(cmp, ast.Name) and cmp.id == "failed_pages" for cmp in node.comparators)
        for node in ast.walk(bucket.value)
    )
    assert excludes_failed, (
        "the bucket no longer excludes failed_pages, so a marker page can be "
        f"counted twice again: {ast.unparse(bucket)}"
    )


def _run_pipeline(tmp_path, *, shredded: bool):
    """A hermetic end-to-end run over one born-digital page.

    GH-453: this used to patch `route_page` with a failing attempt and claim
    that was how the page reached the native-fallback decision. It was not.
    The page is born-digital with native text, so during `_phase_agentic`
    `_is_agentic_trusted_native` takes `_agentic_native_page` and bypasses the
    ladder entirely -- `route_page` was never called. Measured, then removed
    rather than left as a comment describing a path the test does not take.

    What actually drives the decision is the page shape stamped just before the
    bucket is built, which is the same shape the ticket's own reproduction uses.
    `_available_engines_for_agentic` is patched because CI has no provider.

    Only `native_rotated_text_shredded` differs between the two runs.
    """
    from unittest.mock import patch

    from socr.core.config import EngineType, PipelineConfig
    from socr.core.providers import PROFILE_QWEN_LOCAL
    from socr.pipeline.orchestrator import UnifiedPipeline

    tmp_path.mkdir(parents=True, exist_ok=True)
    pdf = tmp_path / "doc.pdf"
    doc = fitz.open()
    doc.new_page().insert_text(
        (72, 72), "Table 1. Regressions of one-year excess returns on forward rates."
    )
    doc.save(str(pdf))
    doc.close()

    pipeline = UnifiedPipeline(
        PipelineConfig(
            primary_engine=EngineType.QWEN,
            agentic=True,
            judge_backend="heuristic",
            enabled_engines=[EngineType.QWEN],
            quiet=True,
            save_figures=False,
            write_manifest=False,
        )
    )

    # Set the page flags just before the bucket is computed, so the run reaches
    # the real decision, and keep the state so its events can be read after.
    # Reading `state.events` rather than a written audit_log.json keeps this
    # independent of whether the run chose to persist one.
    original_assemble = UnifiedPipeline._phase_assemble
    seen: dict = {}

    def _assemble(self, state, *args, **kwargs):
        # The ticket's exact page shape, set on the real state just before the
        # bucket is built: born-digital, native text present, an OCR attempt
        # that never passed. The heuristic judge otherwise re-stamps the
        # attempt as passing, which excludes the page for an unrelated reason.
        for ps in state.pages.values():
            ps.is_born_digital = True
            ps.native_text = ps.native_text or "native body"
            ps.needs_ocr_enhancement = True
            failed = PageOutput(
                page_num=ps.page_num,
                text="ocr attempt that never passed",
                status=PageStatus.ERROR,
                engine="qwen",
                audit_passed=False,
                failure_mode=FailureMode.AUDIT_FAILED,
            )
            ps.attempts = [failed]
            ps.best_output = failed
            if shredded:
                ps.native_rotated_text_shredded = True
        seen["state"] = state
        return original_assemble(self, state, *args, **kwargs)

    with (
        patch.object(pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]),
        patch("socr.pipeline.orchestrator.probe_ollama_idle", return_value=True),
        patch.object(UnifiedPipeline, "_phase_assemble", _assemble),
    ):
        pipeline.process(pdf, tmp_path / "out")

    assert "state" in seen, "_phase_assemble never ran, so the bucket was never built"
    return [getattr(e, "kind", "") for e in seen["state"].events]


def test_the_real_bucket_drops_a_shredded_page(tmp_path) -> None:
    """The production list, at runtime -- not its AST and not a replica.

    #451 review: the AST pin and the hand-rolled `_includes` replica never run
    the actual bucket, so an edit that kept the token but broke the exclusion
    would pass. This asserts the emitted audit events, which is what the bucket
    produces and what the double-count was visible in.
    """
    shredded = _run_pipeline(tmp_path / "s", shredded=True)
    assert "native_fallback" not in shredded, (
        f"a page shipping a failure marker was counted as a native fallback: {shredded}"
    )


def test_the_real_bucket_keeps_an_unshredded_page(tmp_path) -> None:
    """Control: the exclusion must not empty the bucket at runtime either.

    Without it the two runs would be indistinguishable, and the test above
    would pass on a pipeline that had stopped emitting the event entirely.
    """
    plain = _run_pipeline(tmp_path / "p", shredded=False)
    assert "native_fallback" in plain, f"a genuine native fallback stopped being reported: {plain}"
