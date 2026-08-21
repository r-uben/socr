"""S1: structure-class winner selection with no binding oracle.

Measured 2026-08-20 (the reachability-measurement build spec): on 8 pages
where socr held two candidates for a structure-class region (tables or
equations), 7 shipped the WORSE (native) one -- with NO distrust flag ever
firing, because the winner-side chain up to that point (``native_verifier``,
``source_evidence``, header anchors) compares numeric MULTISETS. A flattened
table is multiset-identical to a correctly-gridded one; that chain is blind
to the only thing broken (row/column BINDING, not VALUES). NOTE:
``src/socr/benchmark/table_exactness.py`` (#123) already solves the
equivalent problem correctly on the benchmark side, with a global monotone
injective lane -> column map -- this is prior art for S2's eventual binder,
not something this codebase lacked everywhere.

C1 makes the rule unconditional for a structure-class page (C2, tables
only -- see the note below): native may never author the GRID, flag or no
flag. R3 closes the companion routing gap: the rule is meaningless unless a
model attempt actually ran to author one, so a structure-class page must take
at least one model rung before selection (outside ``--native-only``, whose own
narrower table-only bypass is left untouched -- an explicitly deferred
policy question in the S1 spec).

C2 was originally "tables or equations." The #269 PR review (BLOCKING 1)
found this forced every equation-only page through
``_grid_authored_attempt``'s markdown-TABLE-grid check, which an equation
reading never satisfies on its own terms -- causing a real regression vs
``main`` for equation pages in both directions: a correct native equation
transcription got wrongly demoted to WARNING purely because no attempt
authored a *table* grid on a page that was never going to have one (case
(iii) firing on every equation page, unconditionally), or a model attempt
whose text coincidentally resembled a grid (stray pipe characters near a
matrix, an absolute-value bar) could ship over a fine native reading with
nothing verifying it. Fixed by narrowing ``is_structure_class()`` to
``bool(has_tables)`` -- equation-only pages now keep their pre-existing
native-fallback/R3 behaviour entirely, exactly as if S1 did not exist for
them.

S1 ships no binder (R2: "selection ships first WITHOUT the binder"; the
binder is S2, out of scope here). So a grid-authoring model attempt is not
verified for CORRECTNESS -- only that a model actually authored a grid, and
that the still-live multiset-based value guard / structural gate (deleted
only in S2's verifier rewiring, per C4) did not positively hard-reject it.
That gate reuses the SAME allowlist ``flagged_model_page_output`` (#259)
already uses -- ``REJECTION_AMBIGUOUS_DEFERRED``, the one rejection
disposition #259's own code positively writes today (``agentic.py``, the
AMBIGUOUS-verifier-defers-to-judge path) -- for the same fail-safe reason
theirs is an allowlist and not a denylist. #262's SECOND soft disposition,
``REJECTION_JUDGE_ONLY`` ("judge_only"), is a real constant in
``result.py`` but is written by NO path on ``main`` today -- only by #264,
which is not part of this branch (see the PR body's dependency note) -- so
it is deliberately NOT on this allowlist; a rejection_class of "judge_only"
is indistinguishable from any other not-yet-taught disposition and must
default to distrust, same as ``test_an_unrecognised_disposition_never_qualifies``
below.

Hermetic: ``_winning_page_output`` / ``PageState.is_structure_class`` /
``UnifiedPipeline._is_trusted_native_without_ocr`` are all driven directly, no
provider ladder, no ollama. The new symbols
(``FailureMode.STRUCTURE_CLASS_NO_MODEL_ATTEMPT``, ``PageState.is_structure_class``,
``socr.core.manifest.structure_class_grid_winner``) are reached via ``getattr``
or a guarded import, so a run against the pre-S1 baseline fails on BEHAVIOUR
(the winner it picks, or an assertion on a value) rather than on
``AttributeError``/``ImportError`` for a symbol that does not exist yet.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from socr.core.document import DocumentHandle
from socr.core.manifest import _winning_page_output
from socr.core.result import FailureMode, PageOutput, PageStatus
from socr.core.state import DocumentState, PageState

# Written as LITERALS, not imported: importing the constants would make this
# file fail at the pre-S1 baseline with an ImportError instead of on
# behaviour if either ever moved. SOFT_AMBIGUOUS matches the one value #259
# writes on ``main`` today; SOFT_JUDGE_ONLY matches ``REJECTION_JUDGE_ONLY``
# in ``result.py`` (real constant, but written by no path on ``main`` --
# only #264, not part of this branch) and is used to pin that it does NOT
# qualify here.
SOFT_JUDGE_ONLY = "judge_only"
SOFT_AMBIGUOUS = "ambiguous_deferred"
HARD_REFUTED = ""  # what a CERTAIN_FAIL / structural-gate reject leaves behind

# The multiset-blind fixture: same numbers, wrong binding. Native flattens the
# row label out of the grid; the model attempt keeps it paired. A numeric
# multiset comparison of these two strings is IDENTICAL -- that identity is
# the measured defect, not a detail of this fixture.
MODEL_GRID = (
    "Table 2. Yield regressions by maturity\n\n"
    "| $n$ | const. | slope | $R^2$ |\n"
    "|---|---|---|---|\n"
    "| 2 | 0.03 | 0.91 | 0.44 |\n"
    "| 5 | 0.07 | 0.85 | 0.51 |\n"
)
NATIVE_FLATTENED = "0.03 0.91 0.44 0.07 0.85 0.51\nn const. slope R2\n2 5\n"


def _born_digital_pdf(tmp_path: Path) -> Path:
    fitz = pytest.importorskip("fitz")
    path = tmp_path / "structure_class.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "Table 2. Yield regressions by maturity")
    doc.save(str(path))
    doc.close()
    return path


def _state(
    pdf_path: Path,
    *,
    has_tables: bool = True,
    has_equations: bool = False,
    model_text: str = MODEL_GRID,
    model_engine: str = "gemini",
    model_status: PageStatus = PageStatus.SUCCESS,
    model_failure_mode: FailureMode = FailureMode.NONE,
    rejection_class: str = SOFT_AMBIGUOUS,
    native_audit_passed: bool = True,
    native_best_output: bool = True,
) -> DocumentState:
    """The measured shape: a clean-looking native SUCCESS, no flag anywhere,
    and a correct model grid sitting unread in ``attempts``, soft-refused by
    the ladder (#259's ``REJECTION_AMBIGUOUS_DEFERRED`` -- the one soft
    disposition ``main`` writes today; see the module docstring)."""
    state = DocumentState(handle=DocumentHandle.from_path(pdf_path))
    ps = state.pages[1]
    ps.is_born_digital = True
    ps.has_tables = has_tables
    ps.has_equations = has_equations
    ps.native_text = NATIVE_FLATTENED
    # Deliberately: NO native_table_structure_failed, NO
    # native_table_unverifiable, NO native_table_structure_defective, NO
    # native_table_header_unattributed, NO scanned_table_evidence_failed.
    # This is the measured shape -- every legacy flag is unset.

    native_attempt = PageOutput(
        page_num=1,
        text=NATIVE_FLATTENED,
        status=PageStatus.SUCCESS,
        engine="native",
        audit_passed=native_audit_passed,
    )
    model_attempt = PageOutput(
        page_num=1,
        text=model_text,
        status=model_status,
        engine=model_engine,
        audit_passed=False,
        failure_mode=model_failure_mode,
    )
    # setattr, not a constructor kwarg: a kwarg the pre-S1 baseline does not
    # know would raise TypeError there and make every assertion vacuous.
    setattr(model_attempt, "rejection_class", rejection_class)
    ps.attempts.extend([native_attempt, model_attempt])
    if native_best_output:
        ps.best_output = native_attempt
    return state


# ---------------------------------------------------------------------------
# The measured defect: a structure-class page with no distrust flag at all.
# ---------------------------------------------------------------------------


def test_multiset_identical_native_loses_to_the_grid_authoring_model_attempt(
    tmp_path: Path,
) -> None:
    """The 7/8 defect, on its own terms: no flag fires, but native still loses.

    On the pre-S1 baseline ``p.best_output.audit_passed`` is True and no
    distrust flag is set, so ``_winning_page_output`` returns the native
    SUCCESS output immediately -- the flattened table, shipped clean. This is
    the exact behavioural failure this test is written to catch; it is not an
    AttributeError or ImportError on the baseline.
    """
    state = _state(_born_digital_pdf(tmp_path))

    winner = _winning_page_output(state, 1)
    assert winner.engine == "gemini", (
        f"a structure-class page with a grid-authoring model attempt in cache "
        f"shipped {winner.engine!r} instead; text={winner.text!r}"
    )
    assert "| 2 | 0.03 | 0.91 | 0.44 |" in winner.text, winner.text
    assert winner.audit_passed is False, winner.audit_passed


# GH-268: ``find_table_blocks`` requires only 2 consecutive pipe-bearing
# lines -- no GFM separator row -- so plain prose containing two lines that
# each happen to have a "|" registers as an authored 2x2 grid. Filed the same
# day this file's proof requirement was being finished; must not go unfixed
# in the very code this file is proving.
PROSE_WITH_PIPES_NO_SEPARATOR = "revenue | costs were up\nmargins | fell sharply\n"


def test_grid_authored_attempt_rejects_gh268_prose_with_pipes(tmp_path: Path) -> None:
    """A model attempt is not a "grid" just because two of its lines contain
    a pipe. Without a real GFM separator row, ``structure_class_grid_winner``
    must return ``None`` for this attempt -- reached via a guarded import so
    a run against the pre-S1 baseline (where the symbol does not exist yet)
    fails on ``AssertionError``, not ``ImportError``."""
    import socr.core.manifest as _manifest

    structure_class_grid_winner = getattr(_manifest, "structure_class_grid_winner", None)
    assert structure_class_grid_winner is not None, (
        "S1 must add socr.core.manifest.structure_class_grid_winner"
    )

    state = _state(
        _born_digital_pdf(tmp_path),
        model_text=PROSE_WITH_PIPES_NO_SEPARATOR,
    )
    ps = state.pages[1]

    result = structure_class_grid_winner(ps)
    assert result is None, (
        f"GH-268: two pipe-bearing prose lines with no separator row were "
        f"accepted as an authored grid: {result!r}"
    )


def test_equation_only_page_is_not_structure_class(tmp_path: Path) -> None:
    """BLOCKING 1 on #269: C2 is tables only now. An equation-only page (no
    tables) must NOT take the S1 branch at all -- even with a model attempt
    that authors no usable grid, which under the pre-fix ("tables or
    equations") scope wrongly demoted this page to WARNING /
    STRUCTURE_CLASS_NO_MODEL_ATTEMPT purely because it had an equation and no
    grid-shaped attempt, regardless of whether native's own equation reading
    was fine. This fails on the pre-fix S1 commit (which ships WARNING here)
    and passes once ``is_structure_class()`` is tables-only.
    """
    state = _state(
        _born_digital_pdf(tmp_path),
        has_tables=False,
        has_equations=True,
        model_text="Nothing usable here.\n",
    )

    winner = _winning_page_output(state, 1)
    assert winner.engine == "native", winner.engine
    assert winner.status is PageStatus.SUCCESS, (
        f"an equation-only page must ship the ordinary clean-native path "
        f"unaffected by S1, got status={winner.status!r}"
    )
    assert winner.audit_passed is True, winner.audit_passed


def test_equation_only_page_does_not_ship_a_coincidentally_grid_shaped_attempt(
    tmp_path: Path,
) -> None:
    """The other direction of BLOCKING 1's finding: a model attempt on an
    equation-only page that happens to satisfy the table-grid check (this
    fixture's ``MODEL_GRID`` text) must NOT ship just because the page has
    equations -- there is no table on this page for C1's rule to protect.
    Under the pre-fix scope this shipped "gemini"; the fix must keep native.
    """
    state = _state(
        _born_digital_pdf(tmp_path),
        has_tables=False,
        has_equations=True,
        model_text=MODEL_GRID,
    )

    winner = _winning_page_output(state, 1)
    assert winner.engine == "native", winner.engine
    assert winner.status is PageStatus.SUCCESS, winner.status
    assert winner.audit_passed is True, winner.audit_passed


# ---------------------------------------------------------------------------
# Case (iii): no attempt anywhere authored a grid. Native prose still ships
# (C1: dropping prose is the exact content loss this work exists to prevent)
# but never as SUCCESS.
# ---------------------------------------------------------------------------


def test_no_grid_authoring_attempt_ships_native_prose_flagged_not_success(
    tmp_path: Path,
) -> None:
    state = _state(
        _born_digital_pdf(tmp_path),
        model_text="The table on this page could not be read reliably.\n",
    )

    winner = _winning_page_output(state, 1)
    assert winner.engine == "native", winner.engine
    assert winner.text == NATIVE_FLATTENED, winner.text
    assert winner.status is PageStatus.WARNING, winner.status
    assert winner.audit_passed is False, winner.audit_passed
    expected = getattr(FailureMode, "STRUCTURE_CLASS_NO_MODEL_ATTEMPT", None)
    assert expected is not None
    assert winner.failure_mode is expected, winner.failure_mode


def test_no_model_rung_at_all_defers_to_the_pre_existing_native_path(tmp_path: Path) -> None:
    """R3's degenerate case, corrected: zero candidates is not "a worse one".

    ``--native-only``'s table-only OCR bypass is deliberately left untouched
    by S1 (the spec's own "Open, needs the owner" question), and GH-211 /
    GH-195-198 both have pre-existing, deliberately-tested coverage of a
    clean structure-class native page shipping SUCCESS when no OCR ladder
    ever ran for it. There is no second candidate here for C1 to have
    preferred over native, so the new branch must not fire at all -- this
    page ships exactly as the pre-S1 tail logic always shipped it (SUCCESS,
    since no distrust flag is set).
    """
    state = _state(_born_digital_pdf(tmp_path))
    ps = state.pages[1]
    ps.attempts = [a for a in ps.attempts if a.engine == "native"]
    ps.best_output = ps.attempts[0]

    winner = _winning_page_output(state, 1)
    assert winner.engine == "native", winner.engine
    assert winner.text == NATIVE_FLATTENED, winner.text
    assert winner.status is PageStatus.SUCCESS, winner.status
    assert winner.audit_passed is True, winner.audit_passed


def test_prose_and_appended_sidecar_content_survive_case_iii(tmp_path: Path) -> None:
    """C1: dropping prose is the exact content loss S1 exists to prevent.

    Content appended to a native attempt's text after extraction (GH-36b's
    equation sidecar splice) must still reach the shipped page under the new
    branch, exactly as it does under the pre-existing native-fallback branch
    (GH-211).
    """
    state = _state(
        _born_digital_pdf(tmp_path),
        model_text="Nothing usable here.\n",
    )
    ps = state.pages[1]
    sidecar_text = NATIVE_FLATTENED + "\n\n$$E = mc^2$$"
    ps.attempts[0] = PageOutput(
        page_num=1,
        text=sidecar_text,
        status=PageStatus.SUCCESS,
        engine="native",
        audit_passed=True,
    )
    ps.best_output = ps.attempts[0]

    winner = _winning_page_output(state, 1)
    assert "$$E = mc^2$$" in winner.text, winner.text
    assert winner.status is PageStatus.WARNING, winner.status


# ---------------------------------------------------------------------------
# Non-negotiable: audit_passed is the winner-SELECTION flag. Never mutated on
# an already-selected/stored object (the exact #252 regression shape).
# ---------------------------------------------------------------------------


def test_best_output_object_is_never_mutated_case_i(tmp_path: Path) -> None:
    state = _state(_born_digital_pdf(tmp_path))
    ps = state.pages[1]
    stored_native = ps.attempts[0]
    stored_model = ps.attempts[1]
    assert stored_native.audit_passed is True
    assert stored_model.audit_passed is False

    winner = _winning_page_output(state, 1)

    assert stored_native.audit_passed is True, (
        "the native attempt's audit_passed flag was mutated by selection"
    )
    assert stored_model.audit_passed is False, (
        "the model attempt's audit_passed flag was mutated by selection"
    )
    assert winner.engine == "gemini", winner.engine


def test_best_output_object_is_never_mutated_case_iii(tmp_path: Path) -> None:
    state = _state(
        _born_digital_pdf(tmp_path),
        model_text="Nothing usable here.\n",
    )
    ps = state.pages[1]
    stored_native = ps.attempts[0]

    winner = _winning_page_output(state, 1)

    assert stored_native.audit_passed is True, (
        "the stored native attempt's audit_passed flag was mutated by the demotion"
    )
    assert winner is not stored_native, "the demoted page must be a copy, not the stored object"
    assert winner.audit_passed is False, winner.audit_passed


# ---------------------------------------------------------------------------
# Document-level visibility (the gap this segment closed): case (iii) never
# mutates ``p.best_output`` (the non-negotiable audit_passed rule), so
# ``_phase_assemble``'s ``native_fallback_pages`` -- keyed off exactly
# ``p.best_output.audit_passed`` -- cannot see this demotion at all. CLAUDE.md:
# "failures must surface at every level", not just the shipped page text.
# ``structure_class_native_fallback_applies`` is the exported predicate that
# lets ``_phase_assemble`` build its own bucket for this, mirroring
# ``structure_class_grid_winner`` for case (i).
# ---------------------------------------------------------------------------


def _structure_class_native_fallback_applies(ps) -> bool:
    """Guarded import for a symbol that does not exist on the pre-S1 baseline.

    A bare ``from socr.core.manifest import structure_class_native_fallback_applies``
    would fail every call site below with an ``ImportError`` on the pre-S1
    baseline -- a collection-time failure, not a behavioural one. Routing
    through ``getattr`` on the module object instead gives the baseline run a
    clean behavioural ``AssertionError``, matching the convention this file's
    own docstring commits to for every other new symbol.
    """
    import socr.core.manifest as _manifest

    fn = getattr(_manifest, "structure_class_native_fallback_applies", None)
    assert fn is not None, "S1 must add socr.core.manifest.structure_class_native_fallback_applies"
    return fn(ps)


def test_structure_class_native_fallback_applies_true_for_case_iii(tmp_path: Path) -> None:

    state = _state(
        _born_digital_pdf(tmp_path),
        model_text="Nothing usable here.\n",
    )
    ps = state.pages[1]

    # The predicate must see the demotion even though ``p.best_output`` (the
    # object it is handed) still reports the clean native SUCCESS unchanged --
    # exactly the blind spot ``native_fallback_pages`` has today.
    assert ps.best_output.audit_passed is True, "fixture drifted: this must start clean"
    assert _structure_class_native_fallback_applies(ps) is True, (
        "a structure-class page with a real model rung and no grid-authoring "
        "attempt must report as a case (iii) native-fallback page, even though "
        "p.best_output.audit_passed was never touched"
    )


def test_structure_class_native_fallback_applies_false_for_case_i(tmp_path: Path) -> None:
    state = _state(_born_digital_pdf(tmp_path))  # MODEL_GRID: case (i), not (iii)
    ps = state.pages[1]

    assert _structure_class_native_fallback_applies(ps) is False, (
        "a page whose model attempt DID author a grid is case (i), not case "
        "(iii) -- it must not double up in the native-fallback bucket"
    )


def test_structure_class_native_fallback_applies_false_with_no_model_rung(
    tmp_path: Path,
) -> None:
    """Mirrors the R3-scope fix directly: zero real rungs, so no bucket at all."""
    state = _state(_born_digital_pdf(tmp_path))
    ps = state.pages[1]
    ps.attempts = [a for a in ps.attempts if a.engine == "native"]
    ps.best_output = ps.attempts[0]

    assert _structure_class_native_fallback_applies(ps) is False, (
        "zero model-rung attempts must defer to the pre-existing tail logic, "
        "not be folded into the new document-level bucket either"
    )


def test_ordinary_already_passing_non_native_winner_is_not_an_s1_event(
    tmp_path: Path,
) -> None:
    """The false-positive this segment found and fixed.

    A structure-class page whose ``best_output`` is ALREADY a non-native,
    ``audit_passed=True`` reading (the ordinary, everyday happy path -- native
    was never in the running, S1 never needed to intervene) must NOT be
    counted as an S1 rescue/demotion just because that same attempt also
    happens to contain a table grid. This exact page shipped identically, as
    a clean SUCCESS, before S1 existed; flagging it now would be a pure
    regression in document-level status for the common case, not a fix for
    the 2026-08-20 measurement's defect (native silently winning).
    """
    state = _state(_born_digital_pdf(tmp_path))  # MODEL_GRID; case (i)-shaped text
    ps = state.pages[1]
    # The pivotal difference from every other fixture in this module: the
    # model's own grid-authoring attempt is ALREADY best_output, not native.
    model_attempt = ps.attempts[1]
    assert model_attempt.engine == "gemini" and model_attempt.audit_passed is False, model_attempt
    model_attempt.audit_passed = True
    ps.best_output = model_attempt

    winner = _winning_page_output(state, 1)
    assert winner is model_attempt, (
        "the early-return short-circuit must ship this object unchanged -- "
        "S1's branch must never even be entered"
    )
    assert winner.status is PageStatus.SUCCESS, winner.status
    assert winner.audit_passed is True, winner.audit_passed
    assert _structure_class_native_fallback_applies(ps) is False, (
        "an ordinary already-passing non-native winner must not be flagged "
        "as a case (iii) native-fallback page"
    )


def test_structure_class_native_fallback_applies_false_for_non_structure_class(
    tmp_path: Path,
) -> None:
    state = _state(
        _born_digital_pdf(tmp_path),
        has_tables=False,
        has_equations=False,
        model_text="Nothing usable here.\n",
    )
    ps = state.pages[1]

    assert _structure_class_native_fallback_applies(ps) is False, ps.has_tables


def test_orchestrator_assemble_surfaces_case_iii_pages_document_wide(
    tmp_path: Path,
) -> None:
    """End-to-end proof of the fix: ``_phase_assemble`` must flag a case (iii)
    page in its own bucket, not silently pass it through as clean simply
    because ``p.best_output.audit_passed`` was never touched.

    Drives ``_phase_assemble`` directly on a hand-built ``DocumentState`` (no
    provider ladder, no ollama -- hermetic), the same pattern #259's own
    document-level bucket tests use for the analogous "does the new bucket
    actually surface at the assemble phase" proof.
    """
    from socr.core.config import EngineType, PipelineConfig
    from socr.core.result import DocumentStatus
    from socr.pipeline.orchestrator import UnifiedPipeline

    state = _state(
        _born_digital_pdf(tmp_path),
        model_text="Nothing usable here.\n",
    )
    assert state.pages[1].best_output.audit_passed is True, (
        "fixture drifted: p.best_output must start clean, unmutated by the "
        "demotion -- the whole point of the gap this test guards"
    )

    pipeline = UnifiedPipeline(
        PipelineConfig(
            primary_engine=EngineType.QWEN,
            agentic=True,
            enabled_engines=[EngineType.QWEN],
            quiet=True,
            audit_enabled=True,
            save_figures=False,
            write_manifest=False,
        )
    )
    out_dir = tmp_path / "out"
    result = pipeline._phase_assemble(state, out_dir)

    # Page level: native's flagged prose shipped, not a clean grid.
    assert result.status is not DocumentStatus.SUCCESS, result.status

    # Document level: the new bucket carries page 1, and its audit trail is
    # distinct from every other S1/#259/#262 kind.
    kinds = {getattr(e, "kind", "") for e in state.events}
    assert "structure_class_native_fallback" in kinds, (
        f"no document-level trace of the case (iii) demotion for page 1; "
        f"event kinds seen: {sorted(kinds)}"
    )
    assert "structure_class_model_table_kept" not in kinds, kinds


# ---------------------------------------------------------------------------
# Reverse guards: a non-structure-class page, and a structure-class page whose
# model attempt is empty/hallucinated/ERROR/native-engine, must NOT take this
# branch or must not be rescued by a non-grid attempt.
# ---------------------------------------------------------------------------


def test_non_structure_class_page_is_unaffected(tmp_path: Path) -> None:
    """A prose-only page must not be routed through the new branch at all."""
    state = _state(
        _born_digital_pdf(tmp_path),
        has_tables=False,
        has_equations=False,
    )

    winner = _winning_page_output(state, 1)
    assert winner.engine == "native", winner.engine
    assert winner.status is PageStatus.SUCCESS, winner.status
    assert winner.audit_passed is True, winner.audit_passed


def test_empty_model_attempt_does_not_count_as_a_grid(tmp_path: Path) -> None:
    state = _state(_born_digital_pdf(tmp_path), model_text="   \n")

    winner = _winning_page_output(state, 1)
    assert winner.engine == "native", winner.engine
    assert winner.status is PageStatus.WARNING, winner.status


def test_hallucinated_model_grid_does_not_rescue_the_page(tmp_path: Path) -> None:
    state = _state(_born_digital_pdf(tmp_path), model_failure_mode=FailureMode.HALLUCINATION)

    winner = _winning_page_output(state, 1)
    assert winner.engine == "native", winner.engine
    assert winner.status is PageStatus.WARNING, winner.status


def test_error_status_model_grid_does_not_rescue_the_page(tmp_path: Path) -> None:
    state = _state(_born_digital_pdf(tmp_path), model_status=PageStatus.ERROR)

    winner = _winning_page_output(state, 1)
    assert winner.engine == "native", winner.engine
    assert winner.status is PageStatus.WARNING, winner.status


def test_native_labelled_attempts_alone_do_not_trigger_the_new_branch(tmp_path: Path) -> None:
    """No real rung ran here -- both attempts are native-prefixed engine
    names (``native``, ``native+math``, GH-36b's equation-sidecar splice,
    never produced by an external model). ``has_model_rung_attempt`` must
    stay False, so this defers to the pre-existing tail logic exactly as
    GH-211 / GH-195-198 require -- there is no second candidate for C1 to
    have preferred over native.
    """
    state = _state(_born_digital_pdf(tmp_path), model_engine="native+math")

    winner = _winning_page_output(state, 1)
    assert winner.engine == "native", winner.engine
    assert winner.status is PageStatus.SUCCESS, winner.status
    assert winner.audit_passed is True, winner.audit_passed


def test_a_native_labelled_grid_never_qualifies_even_when_a_real_rung_also_ran(
    tmp_path: Path,
) -> None:
    """Native cannot rescue itself -- C1's exclusion is about the LANE, not
    about whether the page has a candidate at all. Once a real rung DOES
    exist (a non-native-prefixed attempt, tripping ``has_model_rung_attempt``),
    a separate native-prefixed attempt sitting in the same ``p.attempts``
    still may not be picked as the grid winner, even though it carries a
    grid and even though its own ``audit_passed`` is True.
    """
    state = _state(_born_digital_pdf(tmp_path), model_text="Nothing usable here.\n")
    ps = state.pages[1]
    ps.attempts.append(
        PageOutput(
            page_num=1,
            text=MODEL_GRID,
            status=PageStatus.SUCCESS,
            engine="native+math",
            audit_passed=True,
        )
    )

    winner = _winning_page_output(state, 1)
    assert winner.engine == "native", winner.engine
    assert winner.status is PageStatus.WARNING, winner.status
    expected = getattr(FailureMode, "STRUCTURE_CLASS_NO_MODEL_ATTEMPT", None)
    assert winner.failure_mode is expected, winner.failure_mode


def test_d3_floor_and_gh259_still_take_precedence_over_the_new_branch(tmp_path: Path) -> None:
    """The new S1 branch sits AFTER the D3 floor and #259's flagged-model
    substitution; it must not shadow either. A page with
    ``native_table_structure_failed`` + ``native_table_unverifiable`` and no
    attempt authoring a grid still ships the D3 marker, not
    ``STRUCTURE_CLASS_NO_MODEL_ATTEMPT``."""
    state = _state(
        _born_digital_pdf(tmp_path),
        model_text="Table could not be read from this page.\n",
    )
    ps = state.pages[1]
    ps.native_table_structure_failed = True
    ps.native_table_unverifiable = True
    ps.best_output = None

    winner = _winning_page_output(state, 1)
    assert "failed: unverifiable table" in winner.text, winner.text
    assert winner.status is PageStatus.ERROR, winner.status


# ---------------------------------------------------------------------------
# S1 ships NO binder (R2). The still-live multiset-based value guard /
# structural gate is not deleted until S2 (C4), so a POSITIVELY hard-rejected
# grid must still fall back to native here too -- same allowlist reasoning as
# #259/#262, applied to the new general branch.
# ---------------------------------------------------------------------------


def test_a_certain_fail_grid_never_qualifies(tmp_path: Path) -> None:
    """``vr.hard_fail``: the value guard proved this grid wrong and mutated
    nothing. Must fall through to native prose, WARNING, case (iii)."""
    state = _state(_born_digital_pdf(tmp_path), rejection_class=HARD_REFUTED)

    winner = _winning_page_output(state, 1)
    assert winner.engine == "native", winner.engine
    assert winner.status is PageStatus.WARNING, winner.status
    expected = getattr(FailureMode, "STRUCTURE_CLASS_NO_MODEL_ATTEMPT", None)
    assert winner.failure_mode is expected, winner.failure_mode


def test_an_unrecognised_disposition_never_qualifies(tmp_path: Path) -> None:
    """Allowlist, not denylist: a disposition nobody taught this code does
    not qualify -- fail-safe direction is distrust, not "author a grid"."""
    state = _state(_born_digital_pdf(tmp_path), rejection_class="some_future_kind")

    winner = _winning_page_output(state, 1)
    assert winner.engine == "native", winner.engine
    assert winner.status is PageStatus.WARNING, winner.status


def test_judge_only_disposition_does_not_qualify_on_this_branch(tmp_path: Path) -> None:
    """``REJECTION_JUDGE_ONLY`` ("judge_only") is a real constant in
    ``result.py``, but is written by no path on ``main`` today -- only #264
    (not part of this branch; see the PR body's dependency note) ever sets
    it. Until that lands, "judge_only" must behave exactly like any other
    disposition this allowlist was never taught: distrust, not "author a
    grid". This pins the rebase decision so a future stacking of #264 on top
    of this branch has an explicit, failing test to widen rather than a
    silent behaviour change."""
    state = _state(_born_digital_pdf(tmp_path), rejection_class=SOFT_JUDGE_ONLY)

    winner = _winning_page_output(state, 1)
    assert winner.engine == "native", winner.engine
    assert winner.status is PageStatus.WARNING, winner.status


def test_the_ambiguous_deferred_disposition_also_qualifies(tmp_path: Path) -> None:
    """#259's own soft disposition -- the one path on ``main`` that
    positively writes a non-empty ``rejection_class`` today -- qualifies."""
    state = _state(_born_digital_pdf(tmp_path), rejection_class=SOFT_AMBIGUOUS)

    winner = _winning_page_output(state, 1)
    assert winner.engine == "gemini", winner.engine


def test_an_accepted_attempt_qualifies_even_with_no_disposition(tmp_path: Path) -> None:
    """``audit_passed=True`` is trust on its own terms -- no rejection ever
    happened, so there is nothing for the allowlist to gate."""
    state = _state(_born_digital_pdf(tmp_path), rejection_class=HARD_REFUTED)
    ps = state.pages[1]
    ps.attempts[1].audit_passed = True

    winner = _winning_page_output(state, 1)
    assert winner.engine == "gemini", winner.engine


# ---------------------------------------------------------------------------
# PageState.is_structure_class -- the C2 predicate itself.
# ---------------------------------------------------------------------------


def test_is_structure_class_predicate() -> None:
    # ``getattr`` with a fallback, not a bare method call: on the pre-S1
    # baseline ``PageState`` has no ``is_structure_class`` at all, and a bare
    # call would fail on ``AttributeError`` rather than on behaviour. The
    # assertion on ``fn is not None`` is itself the behavioural check.
    #
    # BLOCKING 1 on #269: tables only now, not "tables or equations" -- an
    # equation-only page must NOT count, which is the exact regression the
    # review found (see the module docstring).
    fn = getattr(PageState(page_num=1, has_tables=True), "is_structure_class", None)
    assert fn is not None, "S1 must add PageState.is_structure_class"
    assert fn() is True
    assert (
        getattr(PageState(page_num=1, has_equations=True), "is_structure_class", lambda: True)()
        is False
    ), "an equation-only page must not count as structure-class (BLOCKING 1 on #269)"
    assert (
        getattr(
            PageState(page_num=1, has_tables=True, has_equations=True),
            "is_structure_class",
            lambda: False,
        )()
        is True
    ), "a table page stays structure-class regardless of also having equations"
    assert getattr(PageState(page_num=1), "is_structure_class", lambda: True)() is False


# ---------------------------------------------------------------------------
# R3: the model-rung guarantee. A structure-class page must not bypass OCR in
# default agentic mode -- otherwise C1's rule has nothing to select between.
# --native-only keeps its own narrower table-only exception (deliberately, per
# the S1 spec's own "Open, needs the owner" question).
# ---------------------------------------------------------------------------


def _pipeline(*, native_only: bool = False):
    from socr.core.config import EngineType, PipelineConfig
    from socr.pipeline.orchestrator import UnifiedPipeline

    return UnifiedPipeline(
        PipelineConfig(
            primary_engine=EngineType.QWEN,
            agentic=True,
            enabled_engines=[EngineType.QWEN],
            quiet=True,
            native_first=True,
            native_only=native_only,
        )
    )


def test_equation_only_page_still_bypasses_ocr_like_prose(tmp_path: Path) -> None:
    """BLOCKING 1 on #269 reverted R3's equation widening along with C2's:
    since ``is_structure_class()`` is tables-only again, an equation-only
    page has no structure-class protection left to guarantee a model rung
    for, so it must bypass the OCR ladder exactly like an ordinary prose
    page always has (GH-211 / GH-195-198) -- this is the pre-S1 baseline's
    own behaviour, restored. This is a real behavioural flip from the
    pre-fix S1 commit, which returned False (forced a rung) here.
    """
    pipeline = _pipeline()
    ps = PageState(page_num=1, is_born_digital=True, native_text="some text")
    ps.has_tables = False
    ps.has_equations = True

    assert pipeline._is_trusted_native_without_ocr(1, ps) is True, (
        "an equation-only page must bypass the OCR ladder in default agentic "
        "mode now that C2/R3 are scoped to tables only (BLOCKING 1 on #269)"
    )


def test_table_page_still_does_not_bypass_ocr(tmp_path: Path) -> None:
    """Reverse guard: the pre-existing table exclusion must not regress."""
    pipeline = _pipeline()
    ps = PageState(page_num=1, is_born_digital=True, native_text="some text")
    ps.has_tables = True

    assert pipeline._is_trusted_native_without_ocr(1, ps) is False


def test_prose_only_page_still_bypasses_ocr() -> None:
    """Reverse guard: R3 must not widen the bypass exclusion past structure."""
    pipeline = _pipeline()
    ps = PageState(page_num=1, is_born_digital=True, native_text="some text")
    ps.has_tables = False
    ps.has_equations = False

    assert pipeline._is_trusted_native_without_ocr(1, ps) is True


def test_native_only_mode_keeps_its_narrower_table_only_exception() -> None:
    """--native-only's own bypass stays on the pre-existing table-only check.

    Deliberately NOT widened to equations by S1 -- see the "Open, needs the
    owner" question in the S1 build spec. An equation-only page under
    --native-only still bypasses OCR (unaffected by R3), while a table page
    still does not (pre-existing rotated+table exception, unaffected too).
    """
    pipeline = _pipeline(native_only=True)
    ps = PageState(page_num=1, is_born_digital=True, native_text="some text")
    ps.has_tables = False
    ps.has_equations = True

    assert pipeline._is_trusted_native_without_ocr(1, ps) is True, (
        "--native-only's bypass must stay table-only per the S1 spec's deferred "
        "policy question -- do not silently widen it to equations here"
    )


# ---------------------------------------------------------------------------
# has_model_rung_attempt reverse guards. Two REAL pre-existing tests broke
# during S1 development before this gate was added --
# tests/test_native_only_table_status_gh211.py::test_native_only_verified_table_remains_clean
# and tests/test_destruction_check_gh195_197_198.py::test_clean_page_is_not_demoted
# -- both a clean structure-class native page with zero non-native attempts
# in ``p.attempts``. Neither is a "worse of two candidates" (the 2026-08-20
# measurement's shape); there was only ever one candidate. These two mirror
# that discovery directly so a future change cannot silently regress it
# without touching this file too.
# ---------------------------------------------------------------------------


def test_zero_attempts_recorded_defers_to_the_pre_existing_path(tmp_path: Path) -> None:
    """Mirrors test_clean_page_is_not_demoted (GH-195/197/198): a page that
    never entered the OCR ladder at all (``p.attempts`` empty) must not be
    invented a WARNING by the new branch."""
    state = _state(_born_digital_pdf(tmp_path))
    ps = state.pages[1]
    ps.attempts = []
    ps.best_output = None

    winner = _winning_page_output(state, 1)
    assert winner.engine == "native", winner.engine
    assert winner.status is PageStatus.SUCCESS, winner.status
    assert winner.audit_passed is True, winner.audit_passed


def test_native_only_single_attempt_with_a_legacy_flag_still_demotes_via_the_old_path(
    tmp_path: Path,
) -> None:
    """Mirrors the #264/#265-adjacent --native-only shape: exactly one
    attempt (native), no real rung, but a legacy table-distrust flag DID
    fire. The new branch must stay out of the way and let the pre-existing
    ``native_table_defect``/``native_is_fallback`` tail demote it -- same
    outcome (WARNING, ``NATIVE_TABLE_STRUCTURE_FAILED``), old code path.
    """
    state = _state(_born_digital_pdf(tmp_path))
    ps = state.pages[1]
    ps.attempts = [a for a in ps.attempts if a.engine == "native"]
    ps.best_output = ps.attempts[0]
    ps.native_table_structure_defective = True

    winner = _winning_page_output(state, 1)
    assert winner.engine == "native", winner.engine
    assert winner.status is PageStatus.WARNING, winner.status
    assert winner.failure_mode is FailureMode.NATIVE_TABLE_STRUCTURE_FAILED, winner.failure_mode


# ---------------------------------------------------------------------------
# #269 REQUEST-CHANGES follow-up: regression coverage for BLOCKING 2,
# BLOCKING 3, MAJOR 4, MAJOR 6 and MAJOR 7, added after the coordinator's
# review. Each test below is scoped to exactly one finding from that review.
# ---------------------------------------------------------------------------


def _manifest_symbol(name: str):
    """Guarded lookup, matching this file's own convention: a symbol added by
    S1 (or by this review round) is fetched via ``getattr`` so a run against
    the pre-S1/pre-fix baseline fails on an explicit assertion message,
    never on a bare ``ImportError``/``AttributeError``."""
    import socr.core.manifest as _manifest

    fn = getattr(_manifest, name, None)
    assert fn is not None, f"S1 (#269) must define socr.core.manifest.{name}"
    return fn


# ---------------------------------------------------------------------------
# BLOCKING 3: ``_grid_authored_attempt`` must bind the GFM-separator check to
# the SPECIFIC table block, not scan the whole page for a stray separator.
# ---------------------------------------------------------------------------


def test_grid_authored_attempt_rejects_a_real_separator_elsewhere_on_the_page(
    tmp_path: Path,
) -> None:
    """BLOCKING 3 on #269: a real GFM separator row (``|---|---|``) sitting
    isolated elsewhere on the page must not license an UNRELATED prose pair
    as an authored grid. The page-global ``_MD_SEP_RE`` scan the first fix
    used (before BLOCKING 3) found this exact separator and licensed the
    "revenue | costs..." / "margins | fell..." pair below even though
    neither line belongs to the separator's own block."""
    _grid_authored_attempt = _manifest_symbol("_grid_authored_attempt")

    text = (
        "see identification | strategy.\n\n"
        "|---|---|\n\n"
        "revenue | costs were up\n"
        "margins | fell sharply"
    )
    out = PageOutput(
        page_num=1, text=text, status=PageStatus.SUCCESS, engine="gemini", audit_passed=True
    )
    assert _grid_authored_attempt(out) is False, (
        "a separator row belonging to no adjacent block must not license an "
        "unrelated prose pair as an authored grid"
    )


def test_grid_authored_attempt_accepts_a_genuine_one_column_table(tmp_path: Path) -> None:
    """BLOCKING 3 on #269: the page-global ``_MD_SEP_RE`` the first fix used
    requires >= 2 column groups, so it rejected a genuine ONE-column table.
    Binding the check to ``find_table_blocks``'s own per-block parse (which
    accepts any dash-count separator) must accept this real table."""
    _grid_authored_attempt = _manifest_symbol("_grid_authored_attempt")

    text = "| Header |\n|---|\n| value |"
    out = PageOutput(
        page_num=1, text=text, status=PageStatus.SUCCESS, engine="gemini", audit_passed=True
    )
    assert _grid_authored_attempt(out) is True, (
        "a genuine one-column table with a real separator row must be accepted"
    )


# ---------------------------------------------------------------------------
# BLOCKING 2: ``_reaches_structure_class_branch`` must mirror EVERY page
# where ``_winning_page_output`` disagrees with the naive predicate the prior
# version used -- document-level buckets must never claim membership for a
# page whose real winner ships via a different, earlier branch.
# ---------------------------------------------------------------------------


def test_reaches_branch_is_false_for_a_prose_page_with_a_refused_model_attempt(
    tmp_path: Path,
) -> None:
    """The concrete fallout BLOCKING 2 named: a PROSE page (no tables) with a
    grid-shaped model attempt sitting in ``p.attempts`` must NOT be counted
    as reaching the S1 branch -- C2 requires ``has_tables``, and a page that
    never had one is not a structure-class page no matter what its model
    attempt looks like.

    ``native_table_structure_defective`` is set (despite ``has_tables=False``)
    so the gate's own FIRST short-circuit (a passing native ``best_output``
    with no distrust flag at all) does not itself return False before ever
    reaching the ``is_structure_class()`` check this test targets -- the
    prior version's bug was in what happens AFTER that short-circuit is
    bypassed, so the fixture must bypass it too, or this test cannot tell
    the fixed gate apart from the pre-fix one.

    That same flag is ALSO read by the older, pre-S1 GH-151 B1 native-fallback
    tail (``native_table_defect`` / ``native_is_fallback`` further down in
    ``_winning_page_output``, entirely outside the S1 branch this test is
    about) -- so the real winner here is legitimately demoted to WARNING,
    just via that older mechanism, not via S1. The assertion this test
    actually needs is narrower and sharper than "unaffected by S1": the
    winner must still be native's OWN PROSE text, demoted through the
    pre-existing legacy path -- NOT the model's grid-shaped attempt, which is
    exactly what would ship if the S1 branch wrongly claimed this
    non-table page (the BLOCKING 2 bug this test guards against).
    """
    _reaches_structure_class_branch = _manifest_symbol("_reaches_structure_class_branch")

    state = _state(_born_digital_pdf(tmp_path), has_tables=False, model_text=MODEL_GRID)
    ps = state.pages[1]
    ps.native_table_structure_defective = True

    assert _reaches_structure_class_branch(ps) is False, (
        "a prose (non-table) page must never reach the S1 branch regardless "
        "of what its model attempt looks like"
    )
    winner = _winning_page_output(state, 1)
    assert winner.engine == "native", winner.engine
    assert winner.text == NATIVE_FLATTENED, (
        f"a non-table page must ship native's OWN text, demoted via the "
        f"pre-existing legacy path -- never the model's grid attempt; "
        f"got: {winner.text!r}"
    )
    assert winner.status is PageStatus.WARNING, winner.status
    assert winner.audit_passed is False, winner.audit_passed
    assert winner.failure_mode is FailureMode.NATIVE_TABLE_STRUCTURE_FAILED, winner.failure_mode


def test_reaches_branch_is_false_for_a_d3_floor_page(tmp_path: Path) -> None:
    """Mirrors ``test_d3_floor_and_gh259_still_take_precedence_over_the_new_branch``:
    a D3-floor page (``native_table_structure_failed`` + ``native_table_unverifiable``,
    a real failed attempt) ships the fail-closed marker, not the S1 branch --
    the bucket-level gate must agree, not just the winner-level function."""
    _reaches_structure_class_branch = _manifest_symbol("_reaches_structure_class_branch")

    state = _state(
        _born_digital_pdf(tmp_path),
        model_text="Table could not be read from this page.\n",
    )
    ps = state.pages[1]
    ps.native_table_structure_failed = True
    ps.native_table_unverifiable = True
    ps.best_output = None

    assert _reaches_structure_class_branch(ps) is False, (
        "a D3-floor page must not be counted as reaching the S1 branch -- "
        "the fail-closed marker ships instead"
    )
    winner = _winning_page_output(state, 1)
    assert winner.status is PageStatus.ERROR, winner.status


def test_reaches_branch_is_false_for_a_259_flagged_model_page(tmp_path: Path) -> None:
    """A page where ``flagged_model_page_output`` (#259) is not None ships
    the #259 substitution, not the S1 branch -- BLOCKING 2's fallout was a
    bucket that claimed this page for S1 while the real winner was #259's
    ``replace()`` copy. The gate must return False whenever #259 already
    owns the page."""
    _reaches_structure_class_branch = _manifest_symbol("_reaches_structure_class_branch")

    from socr.core.manifest import flagged_model_page_output

    state = _state(_born_digital_pdf(tmp_path), native_best_output=False)
    ps = state.pages[1]
    # The #259 shape: a native-distrust flag fired (NOT the D3-floor pairing),
    # and the model attempt is the current best_output -- fully present, not
    # yet demoted, refused only via the soft ALLOWLIST disposition.
    ps.native_table_structure_defective = True
    ps.best_output = ps.attempts[1]  # the model (gemini) attempt

    assert flagged_model_page_output(ps) is not None, (
        "fixture sanity: this shape must trigger the #259 substitution"
    )
    assert _reaches_structure_class_branch(ps) is False, (
        "a page #259 already owns must not also be counted as reaching the "
        "S1 branch -- #259's substitution ships instead"
    )


def test_reaches_branch_is_false_with_zero_attempts_recorded(tmp_path: Path) -> None:
    """Mirrors ``test_zero_attempts_recorded_defers_to_the_pre_existing_path``:
    a cascade-halt page (``p.attempts`` empty) must not be counted as
    reaching the S1 branch even though ``is_structure_class()`` is True --
    R3's model-rung guarantee found nothing to prove, live or resumed."""
    _reaches_structure_class_branch = _manifest_symbol("_reaches_structure_class_branch")

    state = _state(_born_digital_pdf(tmp_path))
    ps = state.pages[1]
    ps.attempts = []
    ps.best_output = None

    assert _reaches_structure_class_branch(ps) is False, (
        "zero recorded attempts must not be counted as reaching the S1 branch"
    )
    winner = _winning_page_output(state, 1)
    assert winner.status is PageStatus.SUCCESS, winner.status


# ---------------------------------------------------------------------------
# MAJOR 4: ``structure_class_grid_winner`` ranks candidates by
# ``(audit_passed, confidence, word_count)`` rather than taking the LAST
# qualifying attempt in ladder order.
# ---------------------------------------------------------------------------


def test_grid_winner_picks_the_better_attempt_not_the_last_one(tmp_path: Path) -> None:
    """An earlier, BETTER attempt (higher confidence) must win over a later,
    WORSE one. The prior ``reversed(p.attempts)`` + first-match logic took
    the LAST qualifying attempt regardless of quality -- on this fixture that
    is the worse one, purely by ladder position."""
    structure_class_grid_winner = _manifest_symbol("structure_class_grid_winner")

    state = _state(_born_digital_pdf(tmp_path))
    ps = state.pages[1]
    # Replace the single model attempt with two qualifying candidates: an
    # EARLIER, BETTER one and a LATER, WORSE one. Neither is a clean pass
    # (both soft-refused via the allowlist), so the ranking logic -- not the
    # ``best_output`` short-circuit -- is what is under test.
    native_attempt = ps.attempts[0]
    better = PageOutput(
        page_num=1,
        text=MODEL_GRID,
        status=PageStatus.SUCCESS,
        engine="gemini",
        audit_passed=False,
        confidence=0.9,
    )
    setattr(better, "rejection_class", SOFT_AMBIGUOUS)
    worse = PageOutput(
        page_num=1,
        text=MODEL_GRID.replace("0.91", "0.11"),
        status=PageStatus.SUCCESS,
        engine="gemini",
        audit_passed=False,
        confidence=0.2,
    )
    setattr(worse, "rejection_class", SOFT_AMBIGUOUS)
    ps.attempts = [native_attempt, better, worse]
    ps.best_output = native_attempt

    winner = structure_class_grid_winner(ps)
    assert winner is better, (
        f"expected the earlier, higher-confidence attempt to win; got "
        f"engine={winner.engine if winner else None} "
        f"confidence={winner.confidence if winner else None}"
    )


# ---------------------------------------------------------------------------
# MAJOR 6(a): the new S1 audit events must reach a page's ``pages/NNN.json``
# sidecar -- the event-append block runs BEFORE the PP-1 flush loop, not
# after.
# ---------------------------------------------------------------------------


def _run_assemble_with_real_sidecars(state: DocumentState, output_dir: Path):
    """Like ``test_tr3_d3_floor.py``'s ``_run_assemble`` helper, but leaves
    ``_flush_page_fragment`` / ``_flush_page_sidecar`` / ``_stitch_fragments``
    UNPATCHED so real files land on disk -- MAJOR 6(a) is specifically about
    whether the sidecar written by that real flush call captures the S1
    events, which a patched-out flush can never prove."""
    from unittest.mock import patch

    from socr.core.config import EngineType, PipelineConfig
    from socr.pipeline.orchestrator import UnifiedPipeline

    config = PipelineConfig(
        primary_engine=EngineType.QWEN,
        agentic=False,
        enabled_engines=[EngineType.QWEN],
        quiet=True,
        audit_enabled=False,
        save_figures=False,
        write_manifest=False,
    )
    pipeline = UnifiedPipeline(config)
    with (
        patch.object(pipeline, "_save_markdown", return_value=output_dir / "out.md"),
        patch.object(pipeline, "_write_metadata", return_value=None),
        patch.object(pipeline, "_write_manifest", return_value=None),
        patch.object(pipeline, "_rewrite_all_fragments", return_value=None),
    ):
        pipeline._phase_assemble(state, output_dir)
    return pipeline


def test_structure_class_model_kept_event_reaches_the_page_sidecar(tmp_path: Path) -> None:
    """MAJOR 6(a) on #269: before the fix, the PP-1 flush loop ran BEFORE the
    S1 events (``structure_class_model_table_kept`` / ``_native_fallback``)
    were appended to ``state.events``, so ``_flush_page_sidecar`` -- which
    filters ``state.events`` by page_num at the moment it writes -- never
    saw them, and no later pass rewrites a sidecar. A case-(i) page's
    ``pages/NNN.json`` must carry the event in its ``audit_events`` field."""
    import json

    pdf_path = _born_digital_pdf(tmp_path)
    state = _state(pdf_path)
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    _run_assemble_with_real_sidecars(state, output_dir)

    sidecar_path = output_dir / pdf_path.stem / "pages" / "00001.json"
    assert sidecar_path.exists(), f"expected a sidecar at {sidecar_path}"
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    kinds = [e.get("kind") for e in payload.get("audit_events", [])]
    assert "structure_class_model_table_kept" in kinds, (
        f"S1 event missing from the page sidecar's audit_events; got: {kinds}"
    )


def test_structure_class_native_fallback_event_reaches_the_page_sidecar(tmp_path: Path) -> None:
    """Same as above, for case (iii): a model rung ran but authored no
    usable grid, so native's demoted prose ships and
    ``structure_class_native_fallback`` must reach the sidecar."""
    import json

    pdf_path = _born_digital_pdf(tmp_path)
    state = _state(pdf_path, model_text="Nothing usable here.\n")
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    _run_assemble_with_real_sidecars(state, output_dir)

    sidecar_path = output_dir / pdf_path.stem / "pages" / "00001.json"
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    kinds = [e.get("kind") for e in payload.get("audit_events", [])]
    assert "structure_class_native_fallback" in kinds, (
        f"S1 event missing from the page sidecar's audit_events; got: {kinds}"
    )


# ---------------------------------------------------------------------------
# MAJOR 6(b)/(c): the two new S1 event kinds must be wired into
# ``TABLE_DISTRUST_KINDS`` (so ``tables_trust.json`` marks the page
# untrusted) and into ``build_run_audit``'s ``rank`` dict (so they sort by
# disposition rank, not the ``rank.get(kind, 9)`` default).
# ---------------------------------------------------------------------------


def test_new_s1_event_kinds_mark_the_page_as_an_untrusted_table() -> None:
    """MAJOR 6(b): a page whose only distrust signal is one of the two new
    S1 event kinds must still appear in ``TablesTrust.untrusted_pages`` --
    the whole point of the frozenset is that a consumer of
    ``tables_trust.json`` can find every untrusted page from one index."""
    from socr.core.audit_log import AuditEvent
    from socr.core.tables_trust import build_tables_trust

    for kind in ("structure_class_model_table_kept", "structure_class_native_fallback"):
        events = [AuditEvent(page_num=3, kind=kind, engine="gemini", detail="test")]
        trust = build_tables_trust("fake.pdf", events)
        assert 3 in trust.untrusted_pages, (
            f"a page carrying only {kind!r} must be marked untrusted; got "
            f"untrusted_pages={trust.untrusted_pages}"
        )


def test_new_s1_event_kinds_are_ranked_not_defaulted() -> None:
    """MAJOR 6(c): the two new S1 kinds must sort at rank 6 (alongside
    ``flagged_model_table_kept`` / ``native_fallback``), not fall through to
    ``rank.get(kind, 9)``'s default -- which would sort them AFTER
    ``page_failed`` (rank 7) instead of before it."""
    from unittest.mock import patch

    from socr.core.audit_log import AuditEvent, build_run_audit
    from socr.core.document import DocumentHandle
    from socr.core.state import DocumentState

    with patch.object(DocumentHandle, "__post_init__", lambda self: None):
        handle = DocumentHandle(path=Path("/tmp/fake.pdf"), page_count=1)
    state = DocumentState(handle=handle)
    for kind in ("structure_class_model_table_kept", "structure_class_native_fallback"):
        state.events = [
            AuditEvent(page_num=1, kind="page_failed", engine="", detail=""),
            AuditEvent(page_num=1, kind=kind, engine="gemini", detail=""),
        ]
        audit = build_run_audit(state)
        ordered_kinds = [e.kind for e in audit.events]
        assert ordered_kinds == [kind, "page_failed"], (
            f"{kind!r} must sort BEFORE 'page_failed' (rank 6 < rank 7); got order {ordered_kinds}"
        )


# ---------------------------------------------------------------------------
# MAJOR 7(a): a SOFT-rejected grid_winner (accepted only via the
# ``REJECTION_AMBIGUOUS_DEFERRED`` allowlist) must ship as a demoted COPY --
# WARNING status, a real failure_mode, and the ORIGINAL attempt object left
# untouched -- not as SUCCESS/failure_mode NONE while the document flips to
# AUDIT_FAILED (the two-surface contradiction the review named).
# ---------------------------------------------------------------------------


def test_soft_rejected_grid_winner_ships_as_a_demoted_copy_not_the_stored_attempt(
    tmp_path: Path,
) -> None:
    """Strengthens ``test_multiset_identical_native_loses_to_the_grid_authoring_model_attempt``
    (which only checked ``audit_passed is False``) with the three properties
    MAJOR 7(a) actually cares about: WARNING status, a non-NONE failure_mode,
    and that the STORED attempt in ``p.attempts`` is never mutated."""
    state = _state(_born_digital_pdf(tmp_path))  # default: SOFT_AMBIGUOUS, audit_passed=False
    ps = state.pages[1]
    stored_model_attempt = ps.attempts[1]

    winner = _winning_page_output(state, 1)

    assert winner.status is PageStatus.WARNING, winner.status
    assert winner.audit_passed is False, winner.audit_passed
    assert winner.failure_mode is not FailureMode.NONE, (
        "a soft-rejected grid winner must ship with a real failure_mode, not NONE"
    )
    assert winner is not stored_model_attempt, (
        "the winner must be a replace() COPY, not the stored attempt object itself"
    )
    # The stored attempt itself must be untouched -- MAJOR 7(a) demotes on a
    # copy exactly like #259 does, never the #252-round-1 mistake of flipping
    # audit_passed/status/failure_mode on the object still sitting in
    # ``p.attempts``.
    assert stored_model_attempt.status is PageStatus.SUCCESS, stored_model_attempt.status
    assert stored_model_attempt.failure_mode is FailureMode.NONE, stored_model_attempt.failure_mode


# ---------------------------------------------------------------------------
# MAJOR 7(b): resume collapses ``p.attempts`` to the single frozen winner
# (``_restore_terminal_page_state``), so the persisted
# ``structure_class_model_kept_on_resume`` flag is the ONLY way a resumed
# run can re-derive S1 bucket membership for a clean-pass case-(i) page.
# ---------------------------------------------------------------------------


def test_structure_class_bucket_membership_survives_resume(tmp_path: Path) -> None:
    """A clean-pass case-(i) page (grid winner accepted outright, NOT via
    the soft allowlist) is frozen with the MODEL's engine as its winner --
    e.g. "gemini", not "native". On resume, ``p.attempts`` collapses to just
    that one frozen ``PageOutput``. Without the persisted flag,
    ``_reaches_structure_class_branch`` cannot tell this page apart from an
    ordinary already-passing non-native winner and returns False -- silently
    dropping the page out of the S1 bucket on run 2 even though it was
    correctly counted on run 1.
    """
    _reaches_structure_class_branch = _manifest_symbol("_reaches_structure_class_branch")

    from socr.core.config import EngineType, PipelineConfig
    from socr.core.document import DocumentHandle
    from socr.pipeline.orchestrator import UnifiedPipeline

    config = PipelineConfig(
        primary_engine=EngineType.QWEN,
        agentic=False,
        enabled_engines=[EngineType.QWEN],
        quiet=True,
        audit_enabled=False,
        save_figures=False,
        write_manifest=False,
    )
    pipeline = UnifiedPipeline(config)
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    # Run 1 (live): a clean-pass case-(i) page -- native is best_output, but a
    # model attempt authored the grid outright (audit_passed=True, no soft
    # allowlist involved), so ``_winning_page_output`` ships it UNCHANGED,
    # engine="gemini".
    pdf_path = _born_digital_pdf(tmp_path)
    state1 = DocumentState(handle=DocumentHandle.from_path(pdf_path))
    ps1 = state1.pages[1]
    ps1.is_born_digital = True
    ps1.has_tables = True
    ps1.native_text = NATIVE_FLATTENED
    native_attempt = PageOutput(
        page_num=1,
        text=NATIVE_FLATTENED,
        status=PageStatus.SUCCESS,
        engine="native",
        audit_passed=True,
    )
    model_attempt = PageOutput(
        page_num=1,
        text=MODEL_GRID,
        status=PageStatus.SUCCESS,
        engine="gemini",
        audit_passed=True,
    )
    ps1.attempts.extend([native_attempt, model_attempt])
    ps1.best_output = native_attempt

    live_winner = _winning_page_output(state1, 1)
    assert live_winner.engine == "gemini", live_winner.engine
    assert live_winner.audit_passed is True, live_winner.audit_passed

    # Write the REAL sidecar for run 1 (no mocking) so it carries
    # ``structure_class_model_kept: True``.
    pipeline._flush_page_sidecar(state1, 1, output_dir)

    # Run 2 (resumed): a fresh PageState, as the analyze phase would leave it
    # before ``_restore_terminal_page_state`` runs -- is_born_digital/has_tables/
    # native_text already populated, but attempts empty and best_output unset.
    state2 = DocumentState(handle=DocumentHandle.from_path(pdf_path))
    ps2 = state2.pages[1]
    ps2.is_born_digital = True
    ps2.has_tables = True
    ps2.native_text = NATIVE_FLATTENED

    # Before restore: the flag is at its default (False) and the branch must
    # NOT be reachable from the collapsed shape alone -- this is the bug
    # MAJOR 7(b) names, made concrete.
    resumed_output = PageOutput(
        page_num=1,
        text=MODEL_GRID,
        status=PageStatus.SUCCESS,
        engine="gemini",
        audit_passed=True,
    )
    ps2.attempts.append(resumed_output)
    ps2.best_output = resumed_output
    assert ps2.structure_class_model_kept_on_resume is False, (
        "fixture sanity: the flag must default False before restore runs"
    )
    assert _reaches_structure_class_branch(ps2) is False, (
        "without the persisted flag, a resumed page whose attempts collapsed "
        "to just the frozen non-native winner must NOT be counted as "
        "reaching the S1 branch -- this is the exact bug MAJOR 7(b) names"
    )

    # Now actually restore (PP-5's real code path) against the sidecar
    # written by run 1, on a fresh PageState mirroring run 2's shape.
    state3 = DocumentState(handle=DocumentHandle.from_path(pdf_path))
    ps3 = state3.pages[1]
    ps3.is_born_digital = True
    ps3.has_tables = True
    ps3.native_text = NATIVE_FLATTENED
    pipeline._restore_terminal_page_state(state3, 1, resumed_output, output_dir)

    assert ps3.structure_class_model_kept_on_resume is True, (
        "the sidecar's structure_class_model_kept flag must be restored onto the resumed PageState"
    )
    assert _reaches_structure_class_branch(ps3) is True, (
        "WITH the persisted flag restored, the resumed page must be counted "
        "as reaching the S1 branch again, matching what run 1 actually shipped"
    )
