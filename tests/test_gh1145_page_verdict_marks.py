"""#1145 slice 1: every page socr did not CHECK carries one visible mark at the top.

Design: ``scratch/routing-baseline/leads/1145/design.md`` §9 (tests T1-T7). T8 and T9 need
slice 3's SUCCESS-demoting clauses and T10 needs slice 2's document stub, so they ship with
those slices; ``silent_success_clauses`` is empty here on purpose.

Everything below pins a DIFFERENCE, never a measured tuple. The repo has reverted a change
for pinning one (#253, see #257): provider-dependent machinery does not run in CI, so a
page's status, ``audit_passed`` and failure mode can legitimately differ there while the
page behaves correctly. So each test runs the pipeline twice in one process, changes exactly
one thing, and asserts the two outcomes are identical or differ exactly as intended -- and
every ``_phase_agentic`` test is parametrised over both provider states.

Hermetic: ``primary_engine`` / ``local_engine`` / ``enabled_engines`` are pinned (so
``resolve_auto_engine`` never probes and the result cannot depend on which models this
machine has pulled, #841/#912), ``_available_engines_for_agentic`` is patched,
``_resolve_judge_model`` returns "" (so ``_phase_judge_hard_pages`` builds no judge and
POSTs nothing) and ``route_page`` is asserted un-called.

The lever for the end-to-end tests is ``chart_region_placement_unresolved``. It is the right
one for three reasons: it is a real signal (260 pages of run 23117187 carry it), its guard
``_apply_chart_region_guard`` is STATUS-ONLY so it changes not one byte of the body on its
own, and it is one of the 322 ``failure_mode=none`` WARNING pages the ticket is about -- the
class where nothing in the mode or the disposition names the failed check.
"""

from __future__ import annotations

import json
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

fitz = pytest.importorskip("fitz")

from socr.core.config import EngineType, PipelineConfig  # noqa: E402
from socr.core.document import DocumentHandle  # noqa: E402
from socr.core.manifest import (  # noqa: E402
    _apply_verdict_marks,
    _with_page_top_banners,
    is_page_failed_marker,
)
from socr.core.marks import (  # noqa: E402
    FLAGGED_TABLE_PREFIX,
    SCOPE_PAGE,
    Check,
    CheckResult,
    leading_mark_blocks,
    leading_marks,
    line_is_mark,
    region_scope,
)
from socr.core.result import PageOutput, PageStatus  # noqa: E402
from socr.core.state import DocumentState  # noqa: E402
from socr.core.verdict import PageVerdict, VerdictClause, marks_to_write  # noqa: E402
from socr.pipeline.orchestrator import UnifiedPipeline  # noqa: E402
from verdict_marks_helper import (  # noqa: E402
    assert_differs_only_by_marks,
    canonical_marks,
)

#: A born-digital prose page. The text layer is clean ASCII and carries no table, no
#: equation and no figure, so the ONLY thing that can put a mark on it is the flag the test
#: sets -- which is what makes "differs exactly by the mark block" a statement about #1145
#: and not about some other guard that happened to fire.
NATIVE_PROSE = (
    "This is a born-digital academic paper with more than enough words to clear\n"
    "the born-digital floor and be classified as a clean native text layer here,\n"
    "presenting a comprehensive analysis with several full sentences of prose."
)


def _prose_pdf(path: Path) -> Path:
    doc = fitz.open()
    page = doc.new_page()
    y = 72
    for line in NATIVE_PROSE.split("\n"):
        page.insert_text((72, y), line, fontsize=11, fontname="helv")
        y += 16
    doc.save(str(path))
    doc.close()
    return path


def _pipeline() -> UnifiedPipeline:
    return UnifiedPipeline(
        PipelineConfig(
            primary_engine=EngineType.QWEN,
            local_engine=EngineType.QWEN,
            enabled_engines=[EngineType.QWEN],
            agentic=True,
            quiet=True,
            native_first=True,
            write_manifest=False,
            judge_backend="heuristic",
            dual_pass_tables=False,
            detect_equations=False,
            recover_clean_equations=False,
            save_figures=False,
        )
    )


def _run(tmp_path: Path, *, unresolved: bool, provider: bool, tag: str):
    """One page through the real agentic loop and the real assemble.

    Returns ``(pipeline, state, result, out_dir)``. ``unresolved`` is the only thing that
    varies between arms.
    """
    from socr.core.providers import PROFILE_QWEN_LOCAL

    work = tmp_path / tag
    work.mkdir(parents=True, exist_ok=True)
    pdf = _prose_pdf(work / "doc.pdf")
    out_dir = work / "out"

    pipeline = _pipeline()
    pipeline._scan_root = pdf.parent
    state = DocumentState(handle=DocumentHandle.from_path(pdf))
    ps = state.pages[1]
    ps.is_born_digital = True
    ps.native_text = NATIVE_PROSE
    if unresolved:
        ps.chart_region_placement_unresolved = True

    ladder = [PROFILE_QWEN_LOCAL] if provider else []
    with ExitStack() as stack:
        stack.enter_context(
            patch.object(pipeline, "_available_engines_for_agentic", return_value=ladder)
        )
        stack.enter_context(patch.object(pipeline, "_resolve_judge_model", return_value=""))
        route = stack.enter_context(patch("socr.pipeline.orchestrator.route_page"))
        pipeline._phase_agentic(state, out_dir)
        result = pipeline._phase_assemble(state, out_dir)
        route.assert_not_called()
    return pipeline, state, result, out_dir


def _fragment(out_dir: Path) -> str:
    return next(out_dir.rglob("pages/00001.md")).read_text()


def _sidecar(out_dir: Path) -> dict:
    return json.loads(next(out_dir.rglob("pages/00001.json")).read_text())


def _final_md(out_dir: Path) -> str:
    return next(p for p in out_dir.rglob("*.md") if p.parent.name != "pages").read_text()


# ---------------------------------------------------------------------------
# Grounding: the lever is real, and it is the only thing that moves.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("provider", [True, False])
def test_the_flag_demotes_the_page_and_nothing_else_touches_it(tmp_path: Path, provider) -> None:
    """The canary for every test below.

    Without this, the whole file could be measuring a page the flag never reached: the
    clean arm must come out CHECKED (no clause at all, so nothing to write) and the flagged
    arm must come out NOT CHECKED with exactly one clause, naming ``chart_region``. Pinning
    a status tuple here is what #253 did and #257 reverted, so the assertions are about the
    verdict -- which is the thing this ticket defines -- and about the clause count.
    """
    _, _, _, clean_dir = _run(tmp_path, unresolved=False, provider=provider, tag="canary_clean")
    _, _, _, flag_dir = _run(tmp_path, unresolved=True, provider=provider, tag="canary_flag")

    assert _sidecar(clean_dir)["verdict"] == {"page_num": 1, "clauses": []}
    flagged = PageVerdict.from_dict(_sidecar(flag_dir)["verdict"])
    assert not flagged.checked
    assert [c.check for c in flagged.clauses] == [Check.CHART_REGION], flagged
    assert flagged.clauses[0].result is CheckResult.FAIL


# ---------------------------------------------------------------------------
# T1 -- difference pinning, on the fragment and on the document
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("provider", [True, False])
def test_t1_the_marked_arm_differs_from_the_checked_arm_exactly_by_the_mark_block(
    tmp_path: Path, provider
) -> None:
    """The core test. Two runs, one flag, and the difference is the disclosure only.

    Three halves, because any one alone is satisfiable by a broken implementation:

    * the CHECKED arm is REPRODUCIBLE -- two runs of it give identical bytes, so the
      comparison below is against a stable baseline and not against run-to-run noise;
    * the CHECKED arm carries NO canonical mark (a mark on a page socr did confirm is the
      failure mode that would make every strip-based conversion in this repo vacuous);
    * the MARKED arm is the CHECKED bytes with the leading mark block added, byte for byte
      -- so the clause disclosed what happened and REWROTE nothing.
    """
    _, _, _, clean_a = _run(tmp_path, unresolved=False, provider=provider, tag="t1_clean_a")
    _, _, _, clean_b = _run(tmp_path, unresolved=False, provider=provider, tag="t1_clean_b")
    _, _, _, marked = _run(tmp_path, unresolved=True, provider=provider, tag="t1_marked")

    assert _fragment(clean_a) == _fragment(clean_b), (
        "the CHECKED arm is not reproducible, so the comparison below is meaningless"
    )
    assert_differs_only_by_marks(_fragment(marked), _fragment(clean_a), 1)
    assert_differs_only_by_marks(_final_md(marked), _final_md(clean_a), 1)


@pytest.mark.parametrize("provider", [True, False])
def test_t1_the_checked_page_is_byte_identical_with_the_verdict_code_live(
    tmp_path: Path, provider
) -> None:
    """A CHECKED page's bytes must not move AT ALL, not merely keep their marks.

    ``_with_page_top_banners`` does ``body.lstrip("\\n")`` and rejoins on ``"\\n\\n"``, so
    running it over a page that needs nothing would renormalise separators it was never
    asked to touch. The guarantee is that it is not reached: a CHECKED page has no clause,
    so there is nothing to write and ``replace()`` never runs.
    """
    _, state, _, out_dir = _run(tmp_path, unresolved=False, provider=provider, tag="t1_identity")
    body = _fragment(out_dir)
    assert body == state.pages[1].best_output.text.strip() or body == (
        state.pages[1].best_output.text
    ), "the flushed fragment is not the finalized body"
    assert canonical_marks(body, 1) == []
    assert leading_marks(body, 1) == [], "a CHECKED page has a mark block at its top"


# ---------------------------------------------------------------------------
# T2 -- one line per uncovered clause, and a region clause is never suppressed
# ---------------------------------------------------------------------------


def test_t2_a_region_scoped_clause_is_never_suppressed_by_a_page_level_mark() -> None:
    """Argon's finding 1, as a property of ``marks_to_write``.

    A page may hold a withheld Table 1, whose ``[page N failed: …]`` marker stands where it
    was, and a RENDERED Table 2 whose evidence check failed. Suppressing per PAGE would let
    Table 1's marker silence Table 2's clause: the meter would score the page MARKED and the
    reader would trust corrupted numbers.

    So the rule is per ``(check, region)``. No legacy family carries a region ordinal, so a
    clause that names one can never be the thing a legacy line is about, and is always
    written. Asserted on the RENDERED line text, never with ``!=`` against a mock: ``!=``
    on a ``MagicMock`` always passes and would make this guard vacuous.
    """
    page_marker = "[page 7 failed: unverifiable table — see image]"
    table_two = region_scope("table", 2)
    verdict = PageVerdict(
        page_num=7,
        clauses=(
            VerdictClause(
                Check.SOURCE_EVIDENCE, CheckResult.FAIL, SCOPE_PAGE, "page-level table evidence"
            ),
            VerdictClause(Check.SOURCE_EVIDENCE, CheckResult.FAIL, table_two, "table 2's evidence"),
        ),
    )

    # The marker stands in for ONE table, so the page ships prose beside it.
    written = marks_to_write(verdict, [page_marker], ships_nothing_else=False)

    assert len(written) == 1, written
    (line,) = written
    assert f"scope={table_two}" in line, line
    assert "scope=page" not in line, line
    assert "table:1" not in line, (
        "the line names the withheld table, impugning a region the marker already covers"
    )
    # And the positive control: when the page ships NOTHING but the marker, no content is
    # at risk and the page-level clause goes too -- but the region clause still does not.
    whole_page = marks_to_write(verdict, [page_marker], ships_nothing_else=True)
    assert [f"scope={table_two}" in ln for ln in whole_page] == [True], whole_page


def test_t2_each_uncovered_clause_gets_exactly_one_line_and_the_order_is_stable() -> None:
    """One line per clause, sorted by ``(check, scope)``.

    The order is load-bearing, not cosmetic: the fragment is compared byte-for-byte against
    the assembled document and against the previous run on resume, so an order that
    depended on dict iteration would make those comparisons flap.
    """
    clauses = (
        VerdictClause(Check.TABLE_STRUCTURE, CheckResult.FAIL, SCOPE_PAGE, "z"),
        VerdictClause(Check.CHART_REGION, CheckResult.FAIL, SCOPE_PAGE, "a"),
        VerdictClause(Check.PAGE_JUDGE, CheckResult.UNKNOWN, SCOPE_PAGE, "m"),
    )
    forward = marks_to_write(PageVerdict(3, clauses), [], ships_nothing_else=False)
    reversed_ = marks_to_write(
        PageVerdict(3, tuple(reversed(clauses))), [], ships_nothing_else=False
    )

    assert len(forward) == 3
    assert forward == reversed_, "the written order depends on the clause order"
    assert [ln.split("unchecked: ")[1].split("=")[0] for ln in forward] == [
        Check.CHART_REGION.value,
        Check.PAGE_JUDGE.value,
        Check.TABLE_STRUCTURE.value,
    ]


# ---------------------------------------------------------------------------
# T3 -- suppression is byte-exact
# ---------------------------------------------------------------------------


def _output(text: str, **kw) -> PageOutput:
    base = PageOutput(
        page_num=1,
        text=text,
        status=PageStatus.WARNING,
        engine="qwen",
        audit_passed=True,
    )
    return replace(base, **kw) if kw else base


class _Flags:
    """A PageState stand-in carrying no flag at all, so only the mode can drive a clause."""

    def __getattr__(self, name: str):  # pragma: no cover - attribute probe only
        raise AttributeError(name)


def test_t3_an_already_marked_page_is_returned_byte_identical() -> None:
    """The test that catches gating on the CLAUSE list instead of the UNCOVERED-mark list.

    A ``[page N failed: …]`` page whose whole body is the marker is WARNING or ERROR, so
    its clause list is NON-empty, and it must still emit nothing and come back unchanged.
    Measured on run 23117187: 1,210 of the 1,402 leading-mark pages are in exactly this
    position. Gating on the clause list would run ``_with_page_top_banners`` over all of
    them and renormalise their separators.
    """
    text = "[page 1 failed: unverifiable table — see image]"
    output = _output(
        text,
        failure_mode=__import__(
            "socr.core.result", fromlist=["FailureMode"]
        ).FailureMode.TABLE_REJECTED,
    )
    out, verdict = _apply_verdict_marks(output, _Flags(), 1)

    assert out.text == text, "an already-marked page's bytes moved"
    assert out is output, "the page was rebuilt when it needed no change"
    assert not verdict.checked, (
        "the verdict was emptied by suppression; the sidecar would say CHECKED while the "
        "page carries a mark, and a resumed re-flush would differ"
    )


def test_t3_the_seam_is_idempotent_over_three_passes() -> None:
    """Every finalization seam re-runs on its own output, and assemble re-enters it.

    A seam that read only the FIRST leading block would re-add a second clause's line on
    every pass. Three passes, because two can hide a cycle of period two.
    """
    from socr.core.result import FailureMode

    output = _output("Prose that ships.", failure_mode=FailureMode.NATIVE_GARBLED_MATH)
    once, v1 = _apply_verdict_marks(output, _Flags(), 1)
    twice, v2 = _apply_verdict_marks(once, _Flags(), 1)
    thrice, v3 = _apply_verdict_marks(twice, _Flags(), 1)

    assert canonical_marks(once.text, 1), "nothing was written on the first pass"
    assert once.text == twice.text == thrice.text
    assert v1 == v2 == v3, "the verdict changed when the page was re-finalized"


# ---------------------------------------------------------------------------
# The two glyph signals, which are not the same signal
# ---------------------------------------------------------------------------


class _PuaFlags:
    """Only ``has_unmapped_math_glyphs`` -- the 2.4% PUA class."""

    has_unmapped_math_glyphs = True

    def __getattr__(self, name: str):
        raise AttributeError(name)


class _MathFontFlags:
    """Only ``has_math_font_typesetting`` -- #140's 36.1% report-only class."""

    has_math_font_typesetting = True

    def __getattr__(self, name: str):
        raise AttributeError(name)


def test_the_pua_class_names_native_glyphs_as_failed_not_unknown() -> None:
    """``FAIL``, because the check RAN and found damage in the bytes that ship.

    ``UNKNOWN`` would tell the reader the glyph check never decided, which is false here:
    the private-use codepoints are in the shipped text. The ``glyphs-unknown`` mutant
    restores ``UNKNOWN`` and this is what kills it.
    """
    from socr.core.result import FailureMode

    out, verdict = _apply_verdict_marks(
        _output("Prose with unrecovered math.", failure_mode=FailureMode.NONE), _PuaFlags(), 1
    )
    assert [c.check for c in verdict.clauses] == [Check.NATIVE_GLYPHS], verdict
    assert verdict.clauses[0].result is CheckResult.FAIL, (
        "the glyph check ran and found damage; UNKNOWN says it never decided"
    )
    assert f"{Check.NATIVE_GLYPHS.value}=FAIL" in canonical_marks(out.text, 1)[0]


def test_the_math_font_class_is_report_only_and_puts_no_line_on_the_page() -> None:
    """#140's ruling, pinned as a NEGATIVE.

    ``has_math_font_typesetting`` fires on 36.1% of the free lane with an ~8% slice the
    region locator can never clear (``docs/log/2026-09-02_p4m-trigger-rates.md``), which is
    why its guard only appends a note. A clause off THAT signal would put a failure line on
    a third of the corpus on the strength of a router signal. This is the half that stops
    the two signals being conflated -- in either direction.
    """
    from socr.core.result import FailureMode

    out, verdict = _apply_verdict_marks(
        _output("Prose with typeset maths.", failure_mode=FailureMode.NONE),
        _MathFontFlags(),
        1,
    )
    assert Check.NATIVE_GLYPHS not in {c.check for c in verdict.clauses}, verdict
    # It is still not CHECKED -- it is a WARNING page -- but the reason is the fallback,
    # not a glyph check that nobody ran.
    assert [c.check for c in verdict.clauses] == [Check.UNKNOWN_CHECK], verdict
    assert all(Check.NATIVE_GLYPHS.value not in ln for ln in canonical_marks(out.text, 1)), (
        canonical_marks(out.text, 1)
    )


def test_two_sources_for_one_check_collapse_to_the_more_specific_line() -> None:
    """One line per (check, region), and the specific wording wins.

    ``NATIVE_GARBLED_MATH`` says the native layer garbled the mathematics AND that no
    reading replaced it; the bare PUA flag says only the first. A page carrying both must
    show the mode's wording once, not two lines about one check.
    """
    from socr.core.result import FailureMode

    out, verdict = _apply_verdict_marks(
        _output("Prose.", failure_mode=FailureMode.NATIVE_GARBLED_MATH), _PuaFlags(), 1
    )
    assert [c.check for c in verdict.clauses] == [Check.NATIVE_GLYPHS], verdict
    assert "no reading replaced it" in verdict.clauses[0].reason, verdict.clauses[0].reason
    assert len(canonical_marks(out.text, 1)) == 1


# ---------------------------------------------------------------------------
# The fail-closed fallback
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", [PageStatus.WARNING, PageStatus.ERROR])
def test_the_fail_closed_fallback_marks_a_page_whose_reason_socr_cannot_name(status) -> None:
    """``warning_unmarked`` must not be able to go non-zero by OMISSION.

    Measured on run 23117187: 34 of the 322 ``failure_mode=none`` WARNING pages carry no
    serialized flag at all, so nothing names the failed check. The fallback says what is
    true -- not confirmed, and socr did not record why -- rather than naming a check that
    did not fail. Deleting it is the ``drop-fallback`` mutant.

    The control is in the same test: a SUCCESS page must NOT get it. Without that, a
    fallback that fired on everything would pass here while demoting the whole corpus.
    """
    from socr.core.result import FailureMode

    unnamed = _output("Prose that ships.", status=status, failure_mode=FailureMode.NONE)
    out, verdict = _apply_verdict_marks(unnamed, _Flags(), 1)

    assert [c.check for c in verdict.clauses] == [Check.UNKNOWN_CHECK], verdict
    assert verdict.clauses[0].result is CheckResult.UNKNOWN
    assert status.value in verdict.clauses[0].reason, verdict.clauses[0].reason
    marks = canonical_marks(out.text, 1)
    assert len(marks) == 1 and f"{Check.UNKNOWN_CHECK.value}=" in marks[0], marks

    checked = _output("Prose that ships.", status=PageStatus.SUCCESS, failure_mode=FailureMode.NONE)
    out_ok, verdict_ok = _apply_verdict_marks(checked, _Flags(), 1)
    assert verdict_ok.checked, "the fallback fired on a page socr DID confirm"
    assert out_ok.text == checked.text


def test_the_fallback_line_is_withheld_when_a_legacy_mark_already_discloses() -> None:
    """A generic line above a specific one is strictly worse for the reader.

    The verdict still RECORDS the fallback clause -- the sidecar must not claim the page was
    CHECKED -- but its LINE is withheld, which is why this lives in emission and not in
    derivation. Pinned so a refactor cannot collapse the two.
    """
    from socr.core.result import FailureMode

    disclosed = "> [table unverified: the grid could not be confirmed]"
    body = disclosed + "\n\nProse that ships."
    out, verdict = _apply_verdict_marks(_output(body, failure_mode=FailureMode.NONE), _Flags(), 1)

    assert [c.check for c in verdict.clauses] == [Check.UNKNOWN_CHECK], verdict
    assert canonical_marks(out.text, 1) == [], "a generic line was put above a specific one"
    assert out.text == body


# ---------------------------------------------------------------------------
# T4 -- banner separation under the parser the meter itself uses
# ---------------------------------------------------------------------------


def test_t4_a_canonical_line_above_an_unquoted_marker_does_not_merge_into_it() -> None:
    """Argon's finding 4, pinned because the risk returns the moment the join changes.

    CommonMark lazy continuation merges a ``>`` line and the plain line under it into ONE
    blockquote when only a single newline separates them -- which would make the canonical
    line and a legacy marker read as one mark and change what the legacy marker is.
    ``_with_page_top_banners`` joins on ``"\\n\\n"``, so they stay two blocks. The
    single-newline arm is the control: without it this test passes on any join at all.
    """
    canonical = VerdictClause(
        Check.CHART_REGION, CheckResult.FAIL, SCOPE_PAGE, "a chart region"
    ).mark(4)
    # A real legacy family, UNQUOTED: #259's kept-table note. Its prefix is the
    # recogniser (core/marks.FLAGGED_TABLE_PREFIX), so this is the emitted shape.
    marker = f"[page 4: {FLAGGED_TABLE_PREFIX}the native grid failed verification]"

    joined = _with_page_top_banners(marker, [canonical])
    assert leading_marks(joined, 4) == [canonical], (
        "the first rendered block is not exactly the canonical line"
    )
    assert leading_mark_blocks(joined, 4) == [canonical, marker], (
        "the legacy marker is not the next all-mark block"
    )

    merged = canonical + "\n" + marker
    assert leading_marks(merged, 4) != [canonical], (
        "the control does not reproduce lazy continuation, so this test proves nothing "
        "about the join"
    )


def test_t4_every_line_of_the_leading_block_is_a_mark_on_a_two_clause_page() -> None:
    """MARKED is the meter's question: is the FIRST rendered block made of mark lines.

    With two clauses the seam writes two single-line blockquotes, so ``leading_marks``
    (the meter's own reader) sees the first and ``leading_mark_blocks`` sees both.
    """
    verdict = PageVerdict(
        page_num=5,
        clauses=(
            VerdictClause(Check.CHART_REGION, CheckResult.FAIL, SCOPE_PAGE, "chart"),
            VerdictClause(Check.NATIVE_GLYPHS, CheckResult.FAIL, SCOPE_PAGE, "glyphs"),
        ),
    )
    lines = marks_to_write(verdict, [], ships_nothing_else=False)
    body = _with_page_top_banners("Prose that ships.", lines)

    assert leading_marks(body, 5) == lines[:1]
    assert leading_mark_blocks(body, 5) == lines
    assert all(line_is_mark(ln, 5) for ln in leading_mark_blocks(body, 5))


# ---------------------------------------------------------------------------
# The two guards the orchestrator asked for by name
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        pytest.param("[page 1 failed: unverifiable table — see image]", id="unverifiable_table"),
        pytest.param("[page 1 failed: no usable OCR output]", id="no_usable_output"),
        pytest.param(
            "[page 1 failed: invalid table emission — table_latex_leak]", id="invalid_emission"
        ),
        pytest.param(
            "[page 1 failed: unverifiable table — see image]\n\n![p](figures/p.png)\n",
            id="marker_plus_image",
        ),
    ],
)
def test_is_page_failed_marker_stays_true_after_the_seam(body: str) -> None:
    """A line above ``[page N failed: …]`` stops the page being a failure marker.

    ``is_page_failed_marker`` tests the FIRST line (GH-371), so prepending anything silently
    un-marks the page for every consumer that asks -- selection's disposition classifier,
    the scanned-figure guard, the resume gate -- and the reader would see nothing wrong.
    One row per failed-marker ending, including the marker-plus-image shape, since that is
    the one ``ships_nothing_else`` still calls whole-page.
    """
    from socr.core.result import FailureMode

    assert is_page_failed_marker(body), "the fixture is not a failure marker to begin with"
    for mode in (
        FailureMode.TABLE_REJECTED,
        FailureMode.NATIVE_GARBLED_MATH,
        FailureMode.NONE,
    ):
        out, _ = _apply_verdict_marks(_output(body, failure_mode=mode), _Flags(), 1)
        assert is_page_failed_marker(out.text), f"{mode.value}: the seam un-marked the page"
        assert out.text == body, f"{mode.value}: the marker body moved"

    # The arm that makes this test LOAD-BEARING. For a page-scoped clause
    # ``page_failed_coverage`` suppresses everything, so nothing is ever written and the
    # guard above is never reached -- a mutant that deletes it survives. A REGION-scoped
    # clause deliberately escapes suppression (no legacy mark names a region), so it is the
    # one input that reaches the guard. It arrives the way production supplies it: as a
    # PUBLISHED verdict, which resume rule (b) feeds straight in.
    region_clause = VerdictClause(
        Check.SOURCE_EVIDENCE, CheckResult.FAIL, region_scope("table", 2), "table 2's evidence"
    )
    published = PageVerdict(page_num=1, clauses=(region_clause,))
    assert marks_to_write(published, [body], ships_nothing_else=True), (
        "a region-scoped clause is being suppressed, so this arm reaches nothing"
    )
    out, verdict = _apply_verdict_marks(_output(body), _Flags(), 1, published=published)
    assert is_page_failed_marker(out.text), (
        "a line was put above the whole-page failure marker; the page now reads as content "
        "to the disposition classifier, the scanned-figure guard and the resume gate"
    )
    assert out.text == body
    assert verdict == published, "the published verdict was not kept"


def test_the_strip_helper_cannot_hide_a_body_change() -> None:
    """The helper 13 converted tests run through must not be able to pass a changed body.

    Its contract is "remove EXACTLY the leading canonical block": a one-byte change
    anywhere under that block must still fail the caller's comparison. Without this pin the
    conversions in those tests would be unfalsifiable.
    """
    from verdict_marks_helper import strip_verdict_marks

    mark = VerdictClause(Check.CHART_REGION, CheckResult.FAIL, SCOPE_PAGE, "chart").mark(1)
    clean = "Prose that ships.\n\n| a | b |\n|---|---|\n| 1 | 2 |\n"
    marked = _with_page_top_banners(clean, [mark])

    assert (
        strip_verdict_marks(marked, 1) == clean.strip("\n")
        or strip_verdict_marks(marked, 1) == clean
    )
    stripped = strip_verdict_marks(marked, 1)

    for changed in (
        clean.replace("| 1 | 2 |", "| 1 | 3 |"),
        clean.replace("Prose", "prose"),
        clean + "extra\n",
    ):
        mutated = _with_page_top_banners(changed, [mark])
        assert strip_verdict_marks(mutated, 1) != stripped, (
            f"the helper hid a body change: {changed!r}"
        )


# ---------------------------------------------------------------------------
# A canonical line must not un-recognise the body under it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "banner_name", ["SCANNED_PROSE_RECOVERED_FLAG", "SCANNED_NATIVE_TEXT_FLAG"]
)
def test_a_canonical_line_above_a_recovery_banner_does_not_un_recognise_it(banner_name) -> None:
    """The second face of the ``is_page_failed_marker`` trap, and it costs CONTENT.

    ``_is_restored_prose_recovery`` (#649 rounds 2-3) vouched for a page by
    ``text.startswith(banner)``. This lane's page ships real prose, and the ``scanned_flag``
    family covers only the four table checks -- so an uncovered clause (a judge refusal)
    writes a canonical line above the banner. Under a ``startswith`` the page then stops
    being a restored recovery and the next run replaces its whole recovered text with the
    bare failed-table marker: the prose this lane exists to save, gone, silently.

    Caught by ``tests/test_gh649_scanned_prose_recovery.py::TestTheRecoverySurvivesRestore``
    on the real finalization path; pinned here as the property, over BOTH banners this lane
    emits, because that test only exercises one of them.
    """
    import socr.core.manifest as manifest

    from socr.core.manifest import _is_restored_prose_recovery

    banner = getattr(manifest, banner_name).format(page_num=1)
    body = banner + "\n\nThe paragraphs this lane recovered, which must survive a resume."
    clause = VerdictClause(Check.PAGE_JUDGE, CheckResult.FAIL, SCOPE_PAGE, "the judge refused")

    class _P:
        best_output = None

    plain = _P()
    plain.best_output = replace(
        _output(body), scanned_prose_recovered=True, status=PageStatus.WARNING
    )
    assert _is_restored_prose_recovery(plain, 1), (
        "the control is not recognised, so the marked arm below proves nothing"
    )

    marked = _P()
    marked.best_output = replace(
        _output(_with_page_top_banners(body, [clause.mark(1)])),
        scanned_prose_recovered=True,
        status=PageStatus.WARNING,
    )
    assert _is_restored_prose_recovery(marked, 1), (
        "a canonical line above the banner un-recognised the recovery; the next run would "
        "replace this page's recovered prose with the bare failed-table marker"
    )

    # The forgeability half, which is why "any leading line" would be the wrong rule: a
    # model echoing the banner below some prose of its own is still not a recovery.
    echoed = _P()
    echoed.best_output = replace(
        _output("Model prose the floor must not ship.\n\n" + banner),
        scanned_prose_recovered=True,
        status=PageStatus.WARNING,
    )
    assert not _is_restored_prose_recovery(echoed, 1), (
        "the banner is being matched anywhere in the body, not at the top of it"
    )


# ---------------------------------------------------------------------------
# Review round 2: two marks that could be FALSE, and one provenance gate
# ---------------------------------------------------------------------------


def test_a_rebuilt_recovery_body_is_never_reported_as_judge_refused() -> None:
    """``FAIL`` means the check ran on THESE bytes. On a rebuilt body it did not.

    The #649/#652 prose-recovery endings build the shipped body from the page's own text
    layer and then copy the refused MODEL attempt's ``judge_accepted`` / ``rejection_class``
    onto it. So the refusal evidence is real and it is about bytes that did not ship. #1142
    skips its banner on exactly this provenance set for exactly this reason; the clause had
    no provenance input and wrote "the page judge did not accept the bytes that ship" above
    both recovery banners.

    UNKNOWN, not silence: no judge ever looked at what shipped, and that is worth saying.
    The CONTROL is the same output with a non-excluded ending, which must still be FAIL --
    without it this test would pass on a clause that never reports a refusal at all.
    """
    from socr.core.manifest import SelectionProvenance
    from socr.core.result import REJECTION_JUDGE_ONLY

    refused = replace(
        _output("The page's own text layer, which no judge saw."),
        judge_accepted=False,
        rejection_class=REJECTION_JUDGE_ONLY,
    )

    _, rebuilt = _apply_verdict_marks(
        refused, _Flags(), 1, provenance=SelectionProvenance.UNVERIFIABLE_TABLE_SCANNED
    )
    judge = [c for c in rebuilt.clauses if c.check is Check.PAGE_JUDGE]
    assert len(judge) == 1, rebuilt
    assert judge[0].result is CheckResult.UNKNOWN, (
        "a rebuilt body is reported as judge-REFUSED; the judge never saw these bytes"
    )
    assert "never judged" in judge[0].reason, judge[0].reason

    _, kept = _apply_verdict_marks(
        refused, _Flags(), 1, provenance=SelectionProvenance.UNVERIFIABLE_TABLE_MODEL_KEPT
    )
    kept_judge = [c for c in kept.clauses if c.check is Check.PAGE_JUDGE]
    assert len(kept_judge) == 1 and kept_judge[0].result is CheckResult.FAIL, (
        "the control lost its refusal, so the arm above proves nothing"
    )


def test_native_glyph_damage_is_not_charged_to_a_model_reading() -> None:
    """The accounting describes the NATIVE text. A model reading replaced it.

    Measured: 7 corpus pages with a qwen winner were told their mathematics is unrecovered
    private-use codepoints while a model reading ships. The gate is #1005's own predicate,
    ``native_math_damage_ships``, which the guard this clause mirrors already uses -- and it
    is not a one-liner: an unknown ending, and any ending that may carry native bytes,
    answer True, and a MODEL body that still carries such codepoints answers True too.

    Three arms, because the gate has three outcomes worth pinning.
    """
    from socr.core.manifest import SelectionProvenance

    model_clean = replace(_output("A model reading with no damaged glyph in it."), engine="qwen")
    _, verdict = _apply_verdict_marks(
        model_clean, _PuaFlags(), 1, provenance=SelectionProvenance.PASSING_BEST_OUTPUT
    )
    assert Check.NATIVE_GLYPHS not in {c.check for c in verdict.clauses}, (
        "a model reading is being charged with the native layer's glyph damage"
    )

    # A model body that STILL carries the codepoints shipped the damage, whoever authored it.
    model_damaged = replace(_output("A model reading carrying \uf766\uf766 anyway."), engine="qwen")
    _, damaged = _apply_verdict_marks(
        model_damaged, _PuaFlags(), 1, provenance=SelectionProvenance.PASSING_BEST_OUTPUT
    )
    assert Check.NATIVE_GLYPHS in {c.check for c in damaged.clauses}, damaged

    # And a native winner is the plain case.
    native = replace(_output("The native layer itself."), engine="native")
    _, kept = _apply_verdict_marks(
        native, _PuaFlags(), 1, provenance=SelectionProvenance.PASSING_BEST_OUTPUT
    )
    assert Check.NATIVE_GLYPHS in {c.check for c in kept.clauses}, kept


def test_a_canonical_line_suppresses_only_its_own_check_and_scope() -> None:
    """One canonical line states one check about one region, and may silence only that.

    It was mapped to ``frozenset(Check)``, so the first line written to a two-clause page
    silenced the second: the page kept line one, lost line two for ever, and the reader
    never learned the second fact. Idempotence is unaffected, and the second half here is
    why -- ``leading_mark_blocks`` returns EVERY leading mark line, so once both are on the
    page each covers its own pair and nothing is rewritten.
    """
    chart = VerdictClause(Check.CHART_REGION, CheckResult.FAIL, SCOPE_PAGE, "chart")
    glyphs = VerdictClause(Check.NATIVE_GLYPHS, CheckResult.FAIL, SCOPE_PAGE, "glyphs")
    verdict = PageVerdict(page_num=2, clauses=(chart, glyphs))

    still_wanted = marks_to_write(verdict, [chart.mark(2)], ships_nothing_else=False)
    assert still_wanted == [glyphs.mark(2)], (
        f"the chart line silenced the glyph clause: {still_wanted!r}"
    )

    # Both present: nothing left to write. This is the idempotence half.
    assert marks_to_write(verdict, [chart.mark(2), glyphs.mark(2)], ships_nothing_else=False) == []

    # A line about ANOTHER page covers nothing here.
    assert marks_to_write(verdict, [chart.mark(9)], ships_nothing_else=False) == [
        chart.mark(2),
        glyphs.mark(2),
    ]


def test_a_published_checked_verdict_on_a_non_success_page_falls_back() -> None:
    """ "Not SUCCESS means at least one clause" must hold HERE, not elsewhere.

    A published verdict saying CHECKED on a page that is not SUCCESS cannot both be true.
    Resume rule (a) refuses such sidecars today, but that rule lives in another module and a
    future skip path could widen it without anyone touching this file. So the contradiction
    is resolved conservatively at the point of use: the page falls through to the fail-closed
    clause and the reader is told it is unconfirmed.
    """
    empty = PageVerdict(page_num=1, clauses=())

    _, warned = _apply_verdict_marks(_output("Prose."), _Flags(), 1, published=empty)
    assert [c.check for c in warned.clauses] == [Check.UNKNOWN_CHECK], warned

    # The control: on a SUCCESS page an empty published verdict is simply the truth.
    checked = _output("Prose.", status=PageStatus.SUCCESS)
    out_ok, ok = _apply_verdict_marks(checked, _Flags(), 1, published=empty)
    assert ok.checked and out_ok.text == checked.text


def test_the_recovery_gate_skips_only_this_pages_own_canonical_lines() -> None:
    """Page-strict, because the stated rule is "this module's own lines for THIS page".

    ``CANONICAL_MARK_RE`` alone matches any page number, so a line about page 9 sitting
    above page 4's recovery banner would be skipped and the banner trusted. ``is_canonical_mark``
    is page-strict and refuses it, which is the fail-closed direction: the page is
    reprocessed rather than vouched for on a line that is not about it.
    """
    import socr.core.manifest as manifest

    from socr.core.manifest import _is_restored_prose_recovery

    banner = manifest.SCANNED_PROSE_RECOVERED_FLAG.format(page_num=4)
    body = banner + "\n\nThe paragraphs this lane recovered."
    own = VerdictClause(Check.PAGE_JUDGE, CheckResult.UNKNOWN, SCOPE_PAGE, "never judged").mark(4)
    other = VerdictClause(Check.PAGE_JUDGE, CheckResult.UNKNOWN, SCOPE_PAGE, "never judged").mark(9)

    class _P:
        best_output = None

    def page_with(text: str):
        holder = _P()
        holder.best_output = replace(
            _output(text), scanned_prose_recovered=True, status=PageStatus.WARNING
        )
        return holder

    assert _is_restored_prose_recovery(page_with(_with_page_top_banners(body, [own])), 4), (
        "this page's own canonical line is not being skipped"
    )
    assert not _is_restored_prose_recovery(page_with(_with_page_top_banners(body, [other])), 4), (
        "a canonical line about ANOTHER page was skipped and the banner trusted"
    )


# ---------------------------------------------------------------------------
# The helper's own three guards (Fable round 1, P2)
# ---------------------------------------------------------------------------


def test_the_strip_helper_does_not_normalise_whitespace() -> None:
    """Exact bytes, because the one real difference here is a whitespace one.

    ``_with_page_top_banners`` does ``body.lstrip("\n")``, so a body that began with a blank
    line comes back from the seam without it. That is a genuine byte difference and the
    helper must REPORT it, not absorb it: a comparison that stripped or collapsed whitespace
    would hide exactly the class of change this seam is capable of making, and every one of
    the converted tests runs through this helper.
    """
    from verdict_marks_helper import strip_verdict_marks

    mark = VerdictClause(Check.CHART_REGION, CheckResult.FAIL, SCOPE_PAGE, "chart").mark(1)

    for body in ("Prose.\n", "Prose.  \n", "  Prose."):
        assert strip_verdict_marks(_with_page_top_banners(body, [mark]), 1) == body, body

    leading_blank = "\nProse.\n"
    assert strip_verdict_marks(_with_page_top_banners(leading_blank, [mark]), 1) != leading_blank, (
        "the helper absorbed the leading newline the seam's lstrip removed"
    )


def test_the_strip_helper_refuses_a_page_with_no_mark() -> None:
    """``required=True`` is not decoration: without it every conversion is a tautology.

    Thirteen tests compare bodies through this helper. If it returned the text unchanged
    when no canonical line was found, the mark could stop being emitted ENTIRELY and all
    thirteen would still pass.
    """
    from verdict_marks_helper import strip_verdict_marks

    with pytest.raises(AssertionError, match="expected a canonical #1145 mark block"):
        strip_verdict_marks("Prose with no mark at all.\n", 1)

    # And the opt-out still works where a test parametrises over both arms itself.
    assert strip_verdict_marks("Prose with no mark at all.\n", 1, required=False) == (
        "Prose with no mark at all.\n"
    )


def test_the_asymmetry_helper_refuses_a_mark_on_the_clean_arm() -> None:
    """The half that catches a mark landing on a page socr DID confirm.

    ``assert_differs_only_by_marks`` asserts three things, and this is the one a reader
    would not think to check: dropping it would let the helper pass while BOTH arms carried
    a canonical line, which is the failure mode where #1145 marks a CHECKED page.
    """
    mark = VerdictClause(Check.CHART_REGION, CheckResult.FAIL, SCOPE_PAGE, "chart").mark(1)
    body = "Prose.\n"
    marked = _with_page_top_banners(body, [mark])

    # Control: the real pair passes.
    assert_differs_only_by_marks(marked, body, 1)

    with pytest.raises(AssertionError, match="carries a canonical mark"):
        assert_differs_only_by_marks(marked, marked, 1)

    with pytest.raises(AssertionError, match="carries no canonical mark"):
        assert_differs_only_by_marks(body, body, 1)


# ---------------------------------------------------------------------------
# T5 -- resume rule (a): no verdict is doubt, and doubt reprocesses
# ---------------------------------------------------------------------------


def _resume_pipeline() -> UnifiedPipeline:
    return UnifiedPipeline(
        PipelineConfig(
            agentic=True,
            judge_backend="heuristic",
            primary_engine=EngineType.QWEN,
            local_engine=EngineType.QWEN,
            enabled_engines=[EngineType.QWEN],
            save_figures=False,
            dual_pass_tables=False,
            detect_equations=False,
            recover_clean_equations=False,
            quiet=True,
            write_manifest=False,
        )
    )


def _skippable_state(pdf_path: Path) -> DocumentState:
    """A page the ledger gate SHOULD skip: SUCCESS, audit_passed, real prose body."""
    state = DocumentState(handle=DocumentHandle.from_path(pdf_path))
    ps = state.pages[1]
    ps.is_born_digital = False
    ps.native_text = ""
    accepted = PageOutput(
        page_num=1,
        text="accepted body text long enough to look like real content here.",
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=True,
    )
    ps.attempts.append(accepted)
    ps.best_output = accepted
    return state


@pytest.mark.parametrize(
    ("mutate", "should_skip"),
    [
        pytest.param(lambda meta: meta, True, id="control_verdict_present_is_skipped"),
        pytest.param(
            lambda meta: {k: v for k, v in meta.items() if k != "verdict"},
            False,
            id="pre_1145_sidecar_has_no_verdict",
        ),
        pytest.param(lambda meta: {**meta, "verdict": None}, False, id="verdict_explicitly_null"),
        pytest.param(
            lambda meta: {**meta, "verdict": {"clauses": [{"check": "not_a_check"}]}},
            False,
            id="verdict_does_not_parse",
        ),
    ],
)
def test_t5_resume_rule_a(tmp_path: Path, mutate, should_skip: bool) -> None:
    """A terminal sidecar with no usable ``verdict`` is reprocessed; with one, it is skipped.

    The POSITIVE CONTROL is the point. Without the first row this test passes on a gate that
    refuses everything -- and a gate that never skips is not the fix, it is the loss of
    resume. A pre-#1145 sidecar is SUCCESS with ``audit_passed=True``, so every other gate
    passes and the page would be skipped for ever, shipping unmarked: that is the defect.
    """
    pdf_path = tmp_path / "doc.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "page 1 text " * 10)
    doc.save(str(pdf_path))
    doc.close()

    out_dir = tmp_path / "out"
    pipeline = _resume_pipeline()
    pipeline._scan_root = pdf_path.parent
    state = _skippable_state(pdf_path)
    pipeline._flush_page_fragment(state, 1, state.pages[1].best_output.text, out_dir)
    pipeline._flush_page_sidecar(state, 1, out_dir, terminal=True)

    sidecar_path = next(out_dir.rglob("pages/00001.json"))
    baseline = json.loads(sidecar_path.read_text())
    assert isinstance(baseline.get("verdict"), dict), (
        "the sidecar carries no verdict, so this test cannot tell the two arms apart"
    )
    sidecar_path.write_text(json.dumps(mutate(baseline)))

    fresh = _resume_pipeline()
    fresh._scan_root = pdf_path.parent
    loaded = fresh._load_terminal_page(_skippable_state(pdf_path), 1, out_dir)
    assert (loaded is not None) is should_skip, (loaded, should_skip)


def test_t5_a_verdict_is_never_synthesised_for_an_old_sidecar(tmp_path: Path) -> None:
    """The refusal must not be paperable-over by inventing a verdict.

    Reprocessing is the ONLY correct answer for a page whose record predates the checks: a
    synthesised empty verdict would vouch for bytes no check on this build ever saw, and it
    would read as CHECKED.
    """
    pdf_path = tmp_path / "doc.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "page 1 text " * 10)
    doc.save(str(pdf_path))
    doc.close()

    out_dir = tmp_path / "out"
    pipeline = _resume_pipeline()
    pipeline._scan_root = pdf_path.parent
    state = _skippable_state(pdf_path)
    pipeline._flush_page_fragment(state, 1, state.pages[1].best_output.text, out_dir)
    pipeline._flush_page_sidecar(state, 1, out_dir, terminal=True)

    sidecar_path = next(out_dir.rglob("pages/00001.json"))
    meta = json.loads(sidecar_path.read_text())
    sidecar_path.write_text(json.dumps({k: v for k, v in meta.items() if k != "verdict"}))

    fresh_state = _skippable_state(pdf_path)
    fresh = _resume_pipeline()
    fresh._scan_root = pdf_path.parent
    assert fresh._load_terminal_page(fresh_state, 1, out_dir) is None
    assert getattr(fresh_state.pages[1], "published_verdict", None) is None, (
        "a verdict was invented for a sidecar that had none"
    )


# ---------------------------------------------------------------------------
# T6 -- resume rule (b): a resumed page takes its PUBLISHED verdict
# ---------------------------------------------------------------------------


def test_t6_the_seam_honours_a_published_verdict_whose_input_is_gone() -> None:
    """Rule (b) as a DIFFERENCE: the same output, with and without a published verdict.

    ``PageOutput.table_label_unverified`` is not serialized, so on a resumed page the input
    to that clause is gone. The two arms are the same bytes and the same flags; only the
    published verdict varies. Recomputing drops the clause and the page comes back from a
    resume cleaner than it left -- which is what the ``recompute-verdict-on-resume`` mutant
    restores and what this assertion kills.
    """
    published = PageVerdict(
        page_num=1,
        clauses=(
            VerdictClause(
                Check.TABLE_LABEL,
                CheckResult.UNKNOWN,
                SCOPE_PAGE,
                "a table label on this page could not be confirmed against the page",
            ),
        ),
    )
    # The resumed output: the field is NOT on it, exactly as the sidecar left it.
    resumed = _output("Prose that ships.\n\n| a | b |\n|---|---|\n| 1 | 2 |")
    assert not getattr(resumed, "table_label_unverified", False)

    with_published, v_pub = _apply_verdict_marks(resumed, _Flags(), 1, published=published)
    recomputed, v_rec = _apply_verdict_marks(resumed, _Flags(), 1)

    assert v_pub == published, "the published verdict was not taken verbatim"
    assert Check.TABLE_LABEL in {c.check for c in v_pub.clauses}
    assert Check.TABLE_LABEL not in {c.check for c in v_rec.clauses}, (
        "the clause survived a recomputation, so this test cannot tell the two apart and "
        "proves nothing about rule (b)"
    )
    assert canonical_marks(with_published.text, 1), "the published clause wrote no line"
    assert f"{Check.TABLE_LABEL.value}=" in canonical_marks(with_published.text, 1)[0]
    assert with_published.text != recomputed.text


def test_t6_the_resume_reader_sets_the_published_verdict_from_the_sidecar(tmp_path: Path) -> None:
    """The wiring half: production, not a hand-set attribute.

    ``_restore_terminal_page_state`` must put the sidecar's verdict on the PageState, since
    that is what the seam reads. No fixture state is edited to make this pass -- the sidecar
    is the one ``_flush_page_sidecar`` wrote.
    """
    pdf_path = tmp_path / "doc.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "page 1 text " * 10)
    doc.save(str(pdf_path))
    doc.close()

    out_dir = tmp_path / "out"
    pipeline = _resume_pipeline()
    pipeline._scan_root = pdf_path.parent
    state = _skippable_state(pdf_path)
    pipeline._flush_page_fragment(state, 1, state.pages[1].best_output.text, out_dir)
    pipeline._flush_page_sidecar(state, 1, out_dir, terminal=True)

    fresh_state = _skippable_state(pdf_path)
    fresh = _resume_pipeline()
    fresh._scan_root = pdf_path.parent
    loaded = fresh._load_terminal_page(fresh_state, 1, out_dir)
    assert loaded is not None, "the control page was not skippable, so nothing was restored"
    assert getattr(fresh_state.pages[1], "published_verdict", None) is None
    fresh._restore_terminal_page_state(fresh_state, 1, loaded, out_dir)

    got = fresh_state.pages[1].published_verdict
    assert isinstance(got, PageVerdict), got
    assert got == PageVerdict.from_dict(_sidecar(out_dir)["verdict"])


# ---------------------------------------------------------------------------
# T7 -- the document surface
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("provider", [True, False])
def test_t7_the_mark_block_is_the_first_rendered_block_under_the_page_heading(
    tmp_path: Path, provider
) -> None:
    """The reader's surface, and the golden byte-identity guarantee with it.

    ``assemble_pages`` writes ``## Page N\\n\\n{body.strip()}`` and nothing else, so the
    fragment's first block IS the first block under the heading. Two things are asserted:
    the mark is there (it would be invisible to a citing reader if it were appended), and
    the final ``.md`` is still exactly the fragments assembled -- the property the repo's
    golden tests guard, which a mark written at assembly rather than at the seam would break.
    """
    from ocr_output_contract import assemble_pages

    _, _, result, out_dir = _run(tmp_path, unresolved=True, provider=provider, tag="t7")

    final = _final_md(out_dir)
    heading = "## Page 1\n\n"
    assert final.startswith(heading), final[:80]
    after_heading = final[len(heading) :]
    marks = leading_marks(after_heading, 1)
    assert marks, f"no mark block under the page heading: {after_heading[:200]!r}"
    assert marks == canonical_marks(_fragment(out_dir), 1)

    rebuilt = assemble_pages([_fragment(out_dir)], page_numbers=[1])
    assert final.strip() == rebuilt.strip(), (
        "the final .md is no longer the fragments assembled; the mark is being written "
        "somewhere other than the seam"
    )
    assert result.pages[0].text.strip().startswith("## Page 1")
