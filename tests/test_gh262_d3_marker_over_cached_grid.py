"""#262: the D3 failed-table marker must not ship while a grid sits in the cache.

Measured by the owner on 2026-08-20, Cochrane-Piazzesi (2002) p15, run
``--agentic --no-native-first``::

    shipped:    "[page 4 failed: unverifiable table — see image]"  (100 chars, 0 decimals)
    in cache:   gemini attempt                                      (2546 chars, 36 decimals)
    page flags: native_table_structure_failed=True, native_table_unverifiable=True
    page status: error

Two independent judges, given the page image, rated the cached extraction
faithful; its only defect is a one-row label shift in one panel. The page's
prose and equations were discarded along with the table.

Every term of the D3 conjunction is a verdict on the NATIVE lane. The floor then
suppressed the whole page. Failing loudly is correct and the marker is the right
instrument; failing loudly while holding the answer is not.

Hermetic: drives ``_winning_page_output`` directly for the selection tests. No
provider ladder, no ollama. The end-to-end surfacing test patches
``_available_engines_for_agentic`` and ``route_page``.

The new failure mode is reached by ``getattr`` and never imported, so a run
against the baseline fails on behaviour rather than on an ``AttributeError``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from socr.core.document import DocumentHandle
from socr.core.manifest import (
    _winning_page_output,
    canonical_page_texts,
    d3_floor_kept_model_output,
    is_page_failed_marker,
)
from socr.core.result import FailureMode, PageOutput, PageStatus
from socr.core.state import DocumentState

# The cached reading: prose, an equation, and a grid. Trimmed from the shape of
# the measured page (two regression panels), not copied value for value -- the
# assertions below are about WHICH text ships, never about what is in it.
# The two provably-soft dispositions, written as LITERALS. Importing the
# constants would make this file fail at the baseline with an ImportError
# instead of on behaviour -- and ``judge_only`` does not exist there at all.
SOFT_AMBIGUOUS = "ambiguous_deferred"
SOFT_JUDGE_ONLY = "judge_only"
# What a CERTAIN_FAIL leaves behind: nothing. ``vr.hard_fail`` returns
# accept=False and mutates no field on the PageOutput, so the reading the value
# guard positively refuted is indistinguishable from one nobody explained --
# which is exactly why the allowlist, not a denylist, is the fail-safe shape.
HARD_REFUTED = ""

MODEL_PAGE = (
    "The Fama-Bliss regressions are reported in Table 4. The one-year forward\n"
    "spread forecasts the one-year excess return.\n\n"
    "$$rx_{t+1}^{(n)} = \\alpha + \\beta (f_t^{(n)} - y_t^{(1)}) + \\varepsilon_{t+1}$$\n\n"
    "| | const. | spread | $R^2$ | $\\chi^2$ |\n"
    "|---|---|---|---|---|\n"
    "| Fama-Bliss | -0.04 | 0.06 | 0.00 | 0.1 |\n"
    "| Unconstrained | 1.96 | -0.06 | 0.23 | 118.5 |\n\n"
    "The unconstrained estimates reject the expectations hypothesis.\n"
)

# What the native lane produced: a collapsed, unverifiable region. Never the
# thing under test -- it is here so the fixture is the real shape.
NATIVE_COLLAPSED = "Fama-Bliss | -0.04 0.06 0.00 0.1 Unconstrained 1.96\n"


def _born_digital_pdf(tmp_path: Path) -> Path:
    fitz = pytest.importorskip("fitz")
    path = tmp_path / "cp_p15.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "Table 4. Fama-Bliss excess return regressions")
    doc.save(str(path))
    doc.close()
    return path


def _state(
    pdf_path: Path,
    *,
    model_text: str = MODEL_PAGE,
    model_engine: str = "gemini",
    model_status: PageStatus = PageStatus.SUCCESS,
    model_failure_mode: FailureMode = FailureMode.NONE,
    unverifiable: bool = True,
    header_unattributed: bool = False,
    scanned_evidence_failed: bool = False,
    is_born_digital: bool = True,
    rejection_class: str = SOFT_JUDGE_ONLY,
) -> DocumentState:
    """The page state the measured run produced, rebuilt field by field."""
    state = DocumentState(handle=DocumentHandle.from_path(pdf_path))
    ps = state.pages[1]
    ps.is_born_digital = is_born_digital
    ps.has_tables = True
    ps.native_text = NATIVE_COLLAPSED
    ps.native_table_structure_failed = True
    ps.native_table_unverifiable = unverifiable
    ps.native_table_header_unattributed = header_unattributed
    ps.scanned_table_evidence_failed = scanned_evidence_failed
    ps.d3_floor_png_ref = "![p1](figures/p001.png)"

    native_attempt = PageOutput(
        page_num=1,
        text=NATIVE_COLLAPSED,
        status=PageStatus.WARNING,
        engine="native",
        audit_passed=False,
    )
    model_attempt = PageOutput(
        page_num=1,
        text=model_text,
        status=model_status,
        engine=model_engine,
        # route_page writes ``att.output.audit_passed = att.accepted`` and the
        # ladder accepted nothing.
        audit_passed=False,
        failure_mode=model_failure_mode,
    )
    # setattr, not a constructor kwarg: a kwarg the baseline does not know
    # would raise TypeError there and make every assertion below vacuous.
    setattr(model_attempt, "rejection_class", rejection_class)
    ps.attempts.extend([native_attempt, model_attempt])
    # The measured run's ladder rejected every rung, so ``_score_per_page``
    # cleared the winner. The grid has to be found in ``attempts``.
    ps.best_output = None
    return state


# ---------------------------------------------------------------------------
# The defect, on the issue's own regression assertion.
# ---------------------------------------------------------------------------


def test_cached_grid_ships_instead_of_the_d3_marker(tmp_path: Path) -> None:
    """The issue's assertion, verbatim."""
    state = _state(_born_digital_pdf(tmp_path))

    winner = _winning_page_output(state, 1)
    assert "failed: unverifiable table" not in winner.text, winner.text
    # The issue writes this as ``winner.text == model_grid_text``. Byte equality
    # is relaxed to containment on purpose and on review instruction: the kept
    # page also carries an in-body flag note and the D3 page image, because a
    # reader holding only the .md would otherwise see a refused table with
    # nothing saying so and no way to check it. The substantive half of the
    # assertion -- the model's reading is what ships, whole and unedited -- is
    # asserted exactly.
    assert MODEL_PAGE.strip() in winner.text, winner.text
    assert winner.text.startswith(MODEL_PAGE.strip()), winner.text
    assert winner.status != PageStatus.SUCCESS, winner.status


def test_the_page_body_that_reaches_the_md_carries_the_grid(tmp_path: Path) -> None:
    """Asserted on ``canonical_page_texts`` too: that is what is saved."""
    state = _state(_born_digital_pdf(tmp_path))

    body = "\n\n".join(canonical_page_texts(state))
    assert "failed: unverifiable table" not in body, body
    assert "| Unconstrained | 1.96 | -0.06 | 0.23 | 118.5 |" in body, body


def test_non_table_content_survives_the_table_verdict(tmp_path: Path) -> None:
    """The prose and the equation are not collateral of a TABLE verdict.

    The measured page lost both. Nothing in the tracker asks for that, and it
    is a first-class requirement rather than a side effect of shipping the
    grid -- so it is asserted on the two kinds of non-table content directly.
    """
    state = _state(_born_digital_pdf(tmp_path))

    winner = _winning_page_output(state, 1)
    assert "The Fama-Bliss regressions are reported in Table 4." in winner.text, winner.text
    assert "The unconstrained estimates reject the expectations hypothesis." in winner.text
    assert "\\varepsilon_{t+1}" in winner.text, winner.text


def test_the_kept_page_is_flagged_never_a_clean_pass(tmp_path: Path) -> None:
    """Kept loudly: demoted status, a failure mode of its own, audit_passed False.

    ``getattr`` on the enum member, not an import: importing a name the fix adds
    would make this file fail at the baseline with an ``AttributeError``
    instead of on behaviour.
    """
    state = _state(_born_digital_pdf(tmp_path))

    winner = _winning_page_output(state, 1)
    assert winner.status is PageStatus.WARNING, winner.status
    assert winner.audit_passed is False, winner.audit_passed
    assert winner.engine == "gemini", winner.engine
    expected = getattr(FailureMode, "MODEL_TABLE_OVER_FAILED_FLOOR", None)
    assert expected is not None, "the fix must give this disposition its own failure mode"
    assert winner.failure_mode is expected, winner.failure_mode


def test_header_only_defect_takes_the_same_path(tmp_path: Path) -> None:
    """GH-200 widened the floor to header-only defects; the gate widens with it."""
    state = _state(_born_digital_pdf(tmp_path), unverifiable=False, header_unattributed=True)

    winner = _winning_page_output(state, 1)
    assert MODEL_PAGE.strip() in winner.text, winner.text


def test_best_output_winner_is_not_mutated(tmp_path: Path) -> None:
    """``audit_passed`` is the winner-SELECTION flag; the demotion is on a copy.

    Flipping it on the stored output sends the page back down to the native
    branch and discards the content this fix exists to keep (#252 round 1).
    """
    state = _state(_born_digital_pdf(tmp_path))
    ps = state.pages[1]
    ps.best_output = ps.attempts[1]
    stored = ps.attempts[1]

    winner = _winning_page_output(state, 1)
    assert MODEL_PAGE.strip() in winner.text, winner.text
    assert winner is not stored
    assert stored.status is PageStatus.SUCCESS, stored.status
    assert stored.failure_mode is FailureMode.NONE, stored.failure_mode


def test_the_kept_page_carries_the_flag_note_and_the_page_image(tmp_path: Path) -> None:
    """The marker's visual backstop is not lost when the marker is superseded.

    What ships in its place is a grid every rung of the ladder refused, so the
    page image is the only way a reader can check it -- the backstop matters
    more here, not less. And a reader holding only the ``.md`` sees neither the
    page status nor the audit log, so the flag has to be in the body too.
    """
    state = _state(_born_digital_pdf(tmp_path))

    winner = _winning_page_output(state, 1)
    assert "verify against the page image before citing" in winner.text, winner.text
    assert "![p1](figures/p001.png)" in winner.text, winner.text
    # The note must never read as a failed page: this page shipped content.
    assert not is_page_failed_marker(winner.text), winner.text
    assert winner.text.index(MODEL_PAGE.strip()) < winner.text.index("verify against"), winner.text


def test_a_page_with_no_rendered_png_still_ships_the_note(tmp_path: Path) -> None:
    """Render failure degrades to note-only, exactly as the floor degraded to
    marker-only. The flag is never conditional on the image."""
    state = _state(_born_digital_pdf(tmp_path))
    state.pages[1].d3_floor_png_ref = ""

    winner = _winning_page_output(state, 1)
    assert "verify against the page image before citing" in winner.text, winner.text
    assert "figures/" not in winner.text, winner.text


# ---------------------------------------------------------------------------
# The allowlist. THE round-2 finding: a grid the value guard positively refuted
# leaves no trace on the PageOutput, so only an allowlist of soft dispositions
# is fail-safe.
# ---------------------------------------------------------------------------


def test_a_certain_fail_grid_never_supersedes_the_floor(tmp_path: Path) -> None:
    """``vr.hard_fail``: the value guard proved this grid wrong and mutated nothing.

    Status stays SUCCESS, ``failure_mode`` stays NONE, ``rejection_class`` stays
    empty. Round 1 of this fix kept exactly this page and shipped a refuted
    table over a fail-closed floor.
    """
    state = _state(_born_digital_pdf(tmp_path), rejection_class=HARD_REFUTED)

    winner = _winning_page_output(state, 1)
    assert "failed: unverifiable table" in winner.text, winner.text
    assert "Fama-Bliss regressions are reported" not in winner.text, winner.text
    assert winner.status is PageStatus.ERROR, winner.status


def test_an_unrecognised_disposition_never_supersedes_the_floor(tmp_path: Path) -> None:
    """Allowlist, not denylist: a disposition nobody taught this code keeps the
    marker rather than defaulting to keep."""
    state = _state(_born_digital_pdf(tmp_path), rejection_class="some_future_kind")

    winner = _winning_page_output(state, 1)
    assert "failed: unverifiable table" in winner.text, winner.text


def test_the_ambiguous_deferred_disposition_also_qualifies(tmp_path: Path) -> None:
    """#259's disposition is soft by the same definition and is on the list."""
    state = _state(_born_digital_pdf(tmp_path), rejection_class=SOFT_AMBIGUOUS)

    winner = _winning_page_output(state, 1)
    assert MODEL_PAGE.strip() in winner.text, winner.text


def test_a_refuted_grid_does_not_displace_an_earlier_soft_one(tmp_path: Path) -> None:
    """ "Most escalated" is not "best".

    Ladder position carries no quality information, so a later CERTAIN_FAIL rung
    must not overwrite an earlier reading whose refusal was provably soft.
    """
    state = _state(_born_digital_pdf(tmp_path), rejection_class=SOFT_JUDGE_ONLY)
    late = PageOutput(
        page_num=1,
        text="| WRONG | WRONG |\n|---|---|\n| 9.99 | 9.99 |\n",
        status=PageStatus.SUCCESS,
        engine="gemini-escalated",
        audit_passed=False,
    )
    setattr(late, "rejection_class", HARD_REFUTED)
    state.pages[1].attempts.append(late)

    winner = _winning_page_output(state, 1)
    assert "WRONG" not in winner.text, winner.text
    assert MODEL_PAGE.strip() in winner.text, winner.text


def test_best_output_is_preferred_over_ladder_position(tmp_path: Path) -> None:
    """When both qualify, the pipeline's own selection wins over the later rung.

    ``_best_effort`` picked ``best_output`` as the most trustworthy attempt.
    That is a judgement; ladder order is not.
    """
    state = _state(_born_digital_pdf(tmp_path), rejection_class=SOFT_JUDGE_ONLY)
    ps = state.pages[1]
    ps.best_output = ps.attempts[1]
    later = PageOutput(
        page_num=1,
        text="| LATER | LATER |\n|---|---|\n| 1.11 | 1.11 |\n",
        status=PageStatus.SUCCESS,
        engine="gemini-escalated",
        audit_passed=False,
    )
    setattr(later, "rejection_class", SOFT_JUDGE_ONLY)
    ps.attempts.append(later)

    winner = _winning_page_output(state, 1)
    assert "LATER" not in winner.text, winner.text
    assert MODEL_PAGE.strip() in winner.text, winner.text


def test_gh259_allowlist_is_not_widened_by_the_new_disposition(tmp_path: Path) -> None:
    """Scope guard. ``REJECTION_JUDGE_ONLY`` is for the D3 floor only.

    #259's fallback is a COMPLETE native reading, not a marker, so it can afford
    the narrower one-value allowlist and deliberately keeps it. If someone later
    folds the new disposition into ``flagged_model_page_output``, this fails.
    """
    from socr.core.manifest import flagged_model_page_output

    state = _state(_born_digital_pdf(tmp_path), unverifiable=False)
    ps = state.pages[1]
    # Not a D3 page: only the structure flag, which is #259's territory.
    ps.best_output = ps.attempts[1]
    assert flagged_model_page_output(ps) is None, "judge_only must not qualify for #259"


# ---------------------------------------------------------------------------
# Reverse guards: the floor must still fire everywhere it did before.
# ---------------------------------------------------------------------------


def test_no_attempt_authored_a_grid_still_ships_the_marker(tmp_path: Path) -> None:
    """The floor exists for a reason. Prose from the model is not a grid."""
    state = _state(
        _born_digital_pdf(tmp_path),
        model_text="The Fama-Bliss regressions are reported in Table 4.\n",
    )

    winner = _winning_page_output(state, 1)
    assert "failed: unverifiable table" in winner.text, winner.text
    assert winner.status is PageStatus.ERROR, winner.status


def test_empty_model_attempt_still_ships_the_marker(tmp_path: Path) -> None:
    """ "Produced nothing" is not "produced something flagged"."""
    state = _state(_born_digital_pdf(tmp_path), model_text="   \n")

    winner = _winning_page_output(state, 1)
    assert "failed: unverifiable table" in winner.text, winner.text


def test_a_grid_the_pipeline_called_a_hallucination_still_ships_the_marker(
    tmp_path: Path,
) -> None:
    """A reading socr positively believes is invented never displaces the floor."""
    state = _state(
        _born_digital_pdf(tmp_path),
        model_failure_mode=FailureMode.HALLUCINATION,
    )

    winner = _winning_page_output(state, 1)
    assert "failed: unverifiable table" in winner.text, winner.text


def test_an_error_status_grid_still_ships_the_marker(tmp_path: Path) -> None:
    """Same, for an attempt the engine itself reported as failed."""
    state = _state(_born_digital_pdf(tmp_path), model_status=PageStatus.ERROR)

    winner = _winning_page_output(state, 1)
    assert "failed: unverifiable table" in winner.text, winner.text


def test_a_native_engine_grid_never_rescues_the_page(tmp_path: Path) -> None:
    """Native is exactly what the floor distrusts; it cannot be the rescue."""
    state = _state(_born_digital_pdf(tmp_path), model_engine="native+math")

    winner = _winning_page_output(state, 1)
    assert "failed: unverifiable table" in winner.text, winner.text


def test_a_chart_asset_native_text_grid_never_rescues_the_page(tmp_path: Path) -> None:
    """#265: chart_asset embeds the same native text D3 distrusts.

    Changing the engine label and appending an image cannot turn that text into
    independent model evidence capable of superseding the floor.
    """
    state = _state(_born_digital_pdf(tmp_path), model_engine="chart_asset")

    assert d3_floor_kept_model_output(state.pages[1]) is None
    winner = _winning_page_output(state, 1)
    assert "failed: unverifiable table" in winner.text, winner.text


def test_a_rotated_shredded_page_is_not_rescued_by_the_d3_predicate(tmp_path: Path) -> None:
    """#263 owns pages whose native text is shredded; #262 must decline."""
    state = _state(_born_digital_pdf(tmp_path), model_engine="qwen")
    state.pages[1].native_rotated_text_shredded = True

    assert d3_floor_kept_model_output(state.pages[1]) is None
    winner = _winning_page_output(state, 1)
    assert "failed: unverifiable table" in winner.text, winner.text


def test_a_plain_qwen_grid_still_supersedes_the_d3_floor(tmp_path: Path) -> None:
    """Control: the two restored guards must not disable #262's intended path."""
    state = _state(_born_digital_pdf(tmp_path), model_engine="qwen")

    assert d3_floor_kept_model_output(state.pages[1]) is not None
    winner = _winning_page_output(state, 1)
    assert winner.engine == "qwen", winner.engine
    assert "failed: unverifiable table" not in winner.text, winner.text


def test_gh90_scanned_source_evidence_floor_still_wins(tmp_path: Path) -> None:
    """GH-90's floor is on the SCANNED lane and is not touched by this change.

    A grid rejected by the source-evidence gate on a scan is a fluent
    hallucination with no source support -- the opposite of the cached reading
    #262 is about, and it must keep failing closed.
    """
    state = _state(
        _born_digital_pdf(tmp_path),
        is_born_digital=False,
        scanned_evidence_failed=True,
    )

    winner = _winning_page_output(state, 1)
    assert "failed: unverifiable table" in winner.text, winner.text
    assert winner.failure_mode is FailureMode.HALLUCINATION, winner.failure_mode


def test_born_digital_page_carrying_the_scanned_flag_is_not_rescued(tmp_path: Path) -> None:
    """Belt and braces: the scanned flag vetoes the rescue on either lane."""
    state = _state(_born_digital_pdf(tmp_path), scanned_evidence_failed=True)

    winner = _winning_page_output(state, 1)
    assert "failed: unverifiable table" in winner.text, winner.text


# ---------------------------------------------------------------------------
# Round 3: "authored a grid" must mean a REAL grid.
#
# ``find_table_blocks`` asks only for two consecutive pipe-bearing lines that
# parse to a non-empty cell grid; its docstring states the safety assumption
# ("prose almost never produces consecutive pipe lines") rather than checking
# it. Measured at origin/main: prose with a pipe on two adjacent lines returns
# ONE block, and so does a header plus separator with no body row. Here the
# predicate's answer suppresses a fail-closed marker, so a phantom grid would
# silence the floor on a page with nothing usable on it -- the inverse of what
# this ticket asks for. All three directions are asserted.
# ---------------------------------------------------------------------------

# Prose that genuinely satisfies the loose parser: two CONSECUTIVE lines, each
# carrying a pipe, no separator row. Built this way on purpose -- a fixture
# without two adjacent pipe lines would pass the tests below for the wrong
# reason and prove nothing.
PROSE_WITH_PIPES = (
    "The Fama-Bliss regressions are reported in Table 4.\n"
    "Estimates for the constrained | unconstrained pair are discussed above.\n"
    "The spread coefficient | and its standard error | are reported in the text.\n"
    "The unconstrained estimates reject the expectations hypothesis.\n"
)

# A grid the model started and never finished: header, separator, no body row.
HALF_EMITTED_GRID = (
    "The Fama-Bliss regressions are reported in Table 4.\n\n"
    "| | const. | spread | $R^2$ | $\\chi^2$ |\n"
    "|---|---|---|---|---|\n"
)


def test_the_loose_parser_really_does_accept_these_fixtures() -> None:
    """Grounding. Without this the three tests below could pass vacuously.

    If ``find_table_blocks`` ever stopped accepting these, the fixtures would
    no longer be exercising the gap between loose and strict, and the guards
    would be asserting nothing.
    """
    from socr.tables.reconcile import find_table_blocks

    assert len(find_table_blocks(PROSE_WITH_PIPES)) >= 1, "fixture has no consecutive pipe lines"
    assert len(find_table_blocks(HALF_EMITTED_GRID)) >= 1, HALF_EMITTED_GRID
    assert len(find_table_blocks(MODEL_PAGE)) >= 1, MODEL_PAGE


def test_prose_containing_pipes_does_not_count_as_an_authored_grid(tmp_path: Path) -> None:
    """Direction 1: the marker must yield to a real extraction, not a phantom one."""
    state = _state(_born_digital_pdf(tmp_path), model_text=PROSE_WITH_PIPES)

    winner = _winning_page_output(state, 1)
    assert "failed: unverifiable table" in winner.text, winner.text
    assert "constrained | unconstrained" not in winner.text, winner.text
    assert winner.status is PageStatus.ERROR, winner.status


def test_a_half_emitted_grid_does_not_count_as_an_authored_grid(tmp_path: Path) -> None:
    """Direction 2: a header and a separator with no body row is not a reading."""
    state = _state(_born_digital_pdf(tmp_path), model_text=HALF_EMITTED_GRID)

    winner = _winning_page_output(state, 1)
    assert "failed: unverifiable table" in winner.text, winner.text
    assert winner.status is PageStatus.ERROR, winner.status


def test_a_real_grid_still_supersedes_the_marker(tmp_path: Path) -> None:
    """Direction 3: the whole point. Tightening must not make the fix inert."""
    state = _state(_born_digital_pdf(tmp_path))

    winner = _winning_page_output(state, 1)
    assert "failed: unverifiable table" not in winner.text, winner.text
    assert MODEL_PAGE.strip() in winner.text, winner.text


# Every case the round-3 review named, plus the ones adversarial testing of the
# predicate itself turned up. Table-driven so the enumeration is the test: a
# future condition is added as a row, not as a new function nobody reads.
#
# ``G`` is a minimal real grid, reused so each row varies ONE thing.
_G = "| a | b |\n|---|---|\n| 1 | 2 |\n"

STRICT_GRID_CASES: list[tuple[str, str, bool]] = [
    # --- the shape, in its legitimate spellings -------------------------
    ("real grid", _G, True),
    ("unbordered spelling throughout", "a | b\n--- | ---\n1 | 2\n", True),
    ("leading pipe only, consistently", "| a | b\n| --- | ---\n| 1 | 2\n", True),
    ("trailing pipe only, consistently", "a | b |\n--- | --- |\n1 | 2 |\n", True),
    ("alignment colons in the separator", "| a | b |\n|:---:|---:|\n| 1 | 2 |\n", True),
    ("spaces inside the separator", "| a | b |\n| --- | --- |\n| 1 | 2 |\n", True),
    ("windows CRLF line endings", _G.replace("\n", "\r\n"), True),
    ("grid after a paragraph", "Some prose here.\n\n" + _G, True),
    ("grid immediately after a fence, no blank line", "```\ncode\n```\n" + _G, True),
    ("two grids separated by prose", _G + "\nprose\n\n" + _G, True),
    ("empty body row, then a real one", "| a | b |\n|---|---|\n|  |  |\n| 1 | 2 |\n", True),
    # A dash-only HEADER cell is odd but the body is real content and there is
    # exactly one separator. Refusing it would reject legitimate tables, so it
    # is accepted -- my round-5 expectation was wrong here, not the predicate.
    ("header cell that is literally --- text", "| --- | b |\n|---|---|\n| 1 | 2 |\n", True),
    # --- round-5 review phantoms (MiniMax) ------------------------------
    (
        "MM: prose header with a doubled separator",
        "noise | a\n| --- | --- |\n| --- | --- |\n| 1 | 2 |\n",
        False,
    ),
    ("MM: prose header, single separator", "noise | a\n| --- | --- |\n| 1 | 2 |\n", False),
    (
        "MM: 4-space indented code block",
        "Example:\n\n    | a | b |\n    |---|---|\n    | 1 | 2 |\n",
        False,
    ),
    ("MM: tab-indented code block", "Example:\n\n\t| a | b |\n\t|---|---|\n\t| 1 | 2 |\n", False),
    ("MM: grid inside an HTML comment", "<!--\n" + _G + "-->\n", False),
    ("MM: single-line HTML comment", "<!-- | a | b |\n|---|---|\n| 1 | 2 | -->\n", False),
    ("MM: unclosed HTML comment", "<!--\n" + _G, False),
    ("MM: escaped pipes in cells", "| a \\| x | b |\n|---|---|\n| 1 | 2 |\n", False),
    ("MM: unicode en-dash separator", "| a | b |\n|–––|–––|\n| 1 | 2 |\n", False),
    ("MM: unicode em-dash separator", "| a | b |\n|———|———|\n| 1 | 2 |\n", False),
    ("MM: blank lines split the grid", "| a | b |\n\n|---|---|\n\n| 1 | 2 |\n", False),
    (
        "MM: fence opened inside a table region",
        "| a | b |\n|---|---|\n```\n| 1 | 2 |\n```\n",
        False,
    ),
    ("MM: header without outer pipes, bordered separator", "a | b\n|---|---|\n| 1 | 2 |\n", False),
    ("MM: separator narrower than the header", "| a | b | c |\n|---|---|\n| 1 | 2 | 3 |\n", False),
    ("MM: blank header row", "|  |  |\n|---|---|\n| 1 | 2 |\n", False),
    # --- round-3/4 phantoms, still refused ------------------------------
    ("table inside a backtick fence", "```\n" + _G + "```\n", False),
    ("fence with an info string", "```markdown\n" + _G + "```\n", False),
    ("table inside a tilde fence", "~~~\n" + _G + "~~~\n", False),
    ("indented fence", "  ```\n" + _G + "  ```\n", False),
    ("tilde fence nested in a backtick fence", "```\n~~~\n" + _G + "~~~\n```\n", False),
    ("backtick fence nested in a tilde fence", "~~~\n```\n" + _G + "```\n~~~\n", False),
    ("HTML comment containing a fence", "<!--\n```\n" + _G + "```\n-->\n", False),
    ("unclosed fence swallows the rest", "```\n" + _G, False),
    ("prose with pipes after a closed fence", "```\n" + _G + "```\nx | y\nz | w\n", False),
    ("separator in the middle", "| a | b |\n| 1 | 2 |\n|---|---|\n", False),
    ("separator-as-body", "| a | b |\n|---|---|\n|---|---|\n", False),
    ("separator wider than the header", "| a | b |\n|---|---|---|\n| 1 | 2 |\n", False),
    ("a later row is ragged", "| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 | 5 |\n", False),
    ("prose with pipes", "x | y is here\nz | w also\n", False),
    ("one-column pipe list", "| a |\n|---|\n| 1 |\n", False),
    ("single-dash separator", "| a | b |\n|-|-|\n| 1 | 2 |\n", False),
    ("two-dash separator", "| a | b |\n|--|--|\n| 1 | 2 |\n", False),
    ("header and separator, no body", "| a | b |\n|---|---|\n", False),
    ("body row of empty cells only", "| a | b |\n|---|---|\n|  |  |\n", False),
    ("rows of empty cells only", "||\n||\n", False),
    ("empty text", "", False),
    # --- TIGHTENED IN ROUND 5, deliberately. Both returned True at f9f16b9.
    # --- The interim is fail-closed, so ambiguity now costs the marker
    # --- (today's behaviour) rather than risking a phantom.
    (
        "r5 TIGHTENED: separator, separator, then a real body row",
        "| a | b |\n|---|---|\n|---|---|\n| 1 | 2 |\n",
        False,
    ),
    ("r5 TIGHTENED: pipe-bearing prose directly above a grid", "x | y\nz | w\n" + _G, False),
    # --- GH-268 requires consistent column counts across all rows
    ("ragged body", "| a | b | c |\n|---|---|---|\n| 1 | 2 |\n", False),
]


@pytest.mark.parametrize(
    ("label", "model_text"),
    [
        ("fenced code block", "Here is the layout:\n\n```\n| a | b |\n|---|---|\n| 1 | 2 |\n```\n"),
        ("separator in the middle", "Table 4.\n\n| a | b |\n| 1 | 2 |\n|---|---|\n"),
        ("separator-as-body", "Table 4.\n\n| a | b |\n|---|---|\n|---|---|\n"),
        # Round 5 (MiniMax). Each of these shipped WARNING with the marker
        # fully suppressed at f9f16b9, driven through this same path.
        (
            "prose line as header",
            "Table 4.\n\nnoise | a\n| --- | --- |\n| --- | --- |\n| 1 | 2 |\n",
        ),
        (
            "4-space indented code block",
            "Example:\n\n    | a | b |\n    |---|---|\n    | 1 | 2 |\n",
        ),
        ("grid inside an HTML comment", "Table 4.\n\n<!--\n| a | b |\n|---|---|\n| 1 | 2 |\n-->\n"),
    ],
)
def test_the_reviewers_phantoms_do_not_suppress_the_marker(
    tmp_path: Path, label: str, model_text: str
) -> None:
    """The three round-3 phantoms, at the level that matters: the shipped page.

    Each of these shipped WARNING with no marker. A predicate test alone would
    not have caught that -- these drive ``_winning_page_output``, so they fail
    if the strict check is ever bypassed at the call site rather than merely
    weakened inside the helper.
    """
    state = _state(_born_digital_pdf(tmp_path), model_text=model_text)

    winner = _winning_page_output(state, 1)
    assert "failed: unverifiable table" in winner.text, f"{label}: {winner.text!r}"
    assert winner.status is PageStatus.ERROR, f"{label}: {winner.status}"
    assert "kept a model reading" not in winner.text, f"{label}: {winner.text!r}"


@pytest.mark.parametrize(
    ("label", "text", "expected"),
    [
        pytest.param(t, x, e, id=t.replace(" ", "_").replace(":", ""))
        for t, x, e in STRICT_GRID_CASES
    ],
)
def test_the_strict_predicate_on_its_own_terms(label: str, text: str, expected: bool) -> None:
    """The shape check, case by case, so a failure names which case broke.

    Reached by ``getattr`` on the module: importing a name the fix adds would
    make this file fail at the baseline with an ImportError rather than on
    behaviour.
    """
    import socr.tables.reconcile as reconcile

    strict = getattr(reconcile, "has_strict_table_grid", None)
    assert strict is not None, "the fix must add a strict grid predicate"
    assert strict(text) is expected, f"{label}: {text!r}"


def test_junk_carrying_pipes_is_refused() -> None:
    """A handful of malformed inputs, refused.

    NAMED HONESTLY, after round 5. This was called
    ``test_the_predicate_fails_closed_on_a_shape_nobody_enumerated`` and was
    cited as evidence that anything unanticipated fails closed. That claim was
    FALSE and a reviewer demonstrated it: the three round-5 phantoms are not in
    this list and did not return False. Seven passing junk cases cannot
    establish a universal, and this test is no longer offered as if they could.
    What it actually does is pin these seven inputs. That is worth having and
    it is all it is.

    The predicate cannot be made sound by text alone -- see the module
    docstring and ``fixes/262-round5-or-design.json``. It is a deliberately
    conservative interim whose refusals cost the marker, which is the
    behaviour that ships today.
    """
    import socr.tables.reconcile as reconcile

    strict = getattr(reconcile, "has_strict_table_grid", None)
    assert strict is not None
    for junk in (
        "|\n|\n",
        "|||\n|||\n",
        "-|-\n-|-\n",
        "| --- |\n| --- |\n",
        "a|b\n",
        "\n\n|\n",
        "| a | b |\n\n|---|---|\n\n| 1 | 2 |\n",  # blank lines break the run
    ):
        assert strict(junk) is False, junk


def test_authored_grids_require_uniform_body_rows() -> None:
    """GH-268 requires consistent column counts across the entire grid."""
    import socr.tables.reconcile as reconcile

    flag = getattr(reconcile, "STRICT_GRID_REQUIRES_UNIFORM_BODY", None)
    assert flag is True, "the uniform-row policy must stay named and discoverable"


def test_find_table_blocks_itself_is_unchanged() -> None:
    """Scope guard. Other callers depend on the loose helper's looseness.

    Tightening it globally would be a silent behaviour change across the
    codebase, so the strict check is additive and this pins the split.
    """
    from socr.tables.reconcile import find_table_blocks

    assert len(find_table_blocks("a | b\nc | d\n")) == 1
    assert len(find_table_blocks("| a | b |\n|---|---|\n")) == 1
    assert len(find_table_blocks("| a |\n|---|\n| 1 |\n")) == 1


# ---------------------------------------------------------------------------
# The allowlist is only worth anything if the disposition is really written.
# These drive the REAL ``NativeTableVerifierJudge``, not a stub, so they prove
# the paths genuinely do and do not set it -- rather than proving that the test
# author remembered to.
# ---------------------------------------------------------------------------


def _fitz_page(rows: list[list[tuple[float, str]]]):
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    page = doc.new_page(width=700, height=900)
    for row_idx, cells in enumerate(rows):
        for x, word in cells:
            page.insert_text((x, 100.0 + row_idx * 30), word, fontsize=9)
    return page


def _assess(fitz_page, text: str, *, inner_accepts: bool):
    from unittest.mock import MagicMock

    from socr.pipeline.agentic import AcceptDecision, NativeTableVerifierJudge

    inner = MagicMock()
    inner.assess.return_value = AcceptDecision(accept=inner_accepts, reason="inner judge")
    output = PageOutput(
        page_num=1, text=text, status=PageStatus.SUCCESS, engine="qwen", confidence=0.9
    )
    judge = NativeTableVerifierJudge(
        inner=inner,
        get_fitz_page=lambda pn: fitz_page,
        is_table_page=lambda pn: True,
        record_event=None,
    )
    return judge.assess(output, MagicMock()), output


def test_the_judge_only_disposition_is_actually_written(tmp_path: Path) -> None:
    """The verifier found nothing to refute and the inner judge alone refused.

    Without this the allowlist would be inert in production: every real refusal
    on this branch would read as "we cannot say why" and keep the marker.
    """
    decision, output = _assess(
        _fitz_page([]),
        "The estimates are reported below.\n\n| a | b |\n|---|---|\n| 1.0 | 2.0 |\n",
        inner_accepts=False,
    )
    assert decision.accept is False, decision
    assert getattr(output, "rejection_class", "") == SOFT_JUDGE_ONLY, output


def test_a_certain_fail_leaves_no_disposition_behind(tmp_path: Path) -> None:
    """The measured fact the allowlist exists for.

    ``vr.hard_fail`` returns accept=False and mutates NOTHING -- so an empty
    ``rejection_class`` genuinely cannot be read as "soft", and a denylist of
    bad dispositions has nothing to match on.
    """
    gap = 60.0
    decision, output = _assess(
        _fitz_page([[(100.0, "0.1"), (100.0 + gap, "0.2"), (100.0 + 2 * gap, "0.3")]]),
        "| label | vals |\n|---|---|\n| row1 | 0.1 |\n",
        inner_accepts=True,
    )
    assert decision.accept is False, decision
    assert "native_table_verifier" in decision.reason, decision.reason
    assert getattr(output, "rejection_class", "") == HARD_REFUTED, output
    assert output.status is PageStatus.SUCCESS
    assert output.failure_mode.value == "none"


def test_an_accepted_reading_carries_no_disposition(tmp_path: Path) -> None:
    """Reverse guard: the marker is only ever set on a refusal."""
    decision, output = _assess(
        _fitz_page([[(100.0, "1.0"), (160.0, "2.0")]]),
        "| a | b |\n|---|---|\n| 1.0 | 2.0 |\n",
        inner_accepts=True,
    )
    assert decision.accept is True, decision
    assert getattr(output, "rejection_class", "") == "", output


# ---------------------------------------------------------------------------
# Surfacing: page, document, metadata and CLI.
# ---------------------------------------------------------------------------


def test_document_audit_metadata_and_trust_sidecar_surface_the_superseded_floor(
    tmp_path: Path,
) -> None:
    """The flag reaches page, document, metadata and CLI — not just one of them.

    Drives ``_phase_assemble`` on the measured page state directly. Hermetic by
    construction: no provider ladder is entered at all, so there is nothing for
    the absence of ollama in CI to change.
    """
    from socr.core.config import EngineType, PipelineConfig
    from socr.core.result import DocumentStatus
    from socr.pipeline.orchestrator import UnifiedPipeline

    state = _state(_born_digital_pdf(tmp_path))
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

    # Page level: the model's reading shipped; the marker did not.
    assert "failed: unverifiable table" not in result.markdown, result.markdown
    assert "| Unconstrained | 1.96 | -0.06 | 0.23 | 118.5 |" in result.markdown, result.markdown
    # Document level: never a clean SUCCESS.
    assert result.status is not DocumentStatus.SUCCESS, result.status

    doc_dir = next(out_dir.rglob("pages")).parent
    # Metadata level: the sidecar records the demoted winner and its own mode.
    sidecar = json.loads((doc_dir / "pages" / "00001.json").read_text())
    assert sidecar["winning_output"]["status"] == "warning", sidecar["winning_output"]
    assert sidecar["winning_output"]["failure_mode"] == "model_table_over_failed_floor", sidecar[
        "winning_output"
    ]

    # Audit level: its own kind, distinct from the floor's disposition and from
    # #259's flag. A consumer must be able to tell "we shipped nothing" from
    # "we shipped a model grid over a hard fail".
    pipeline._write_audit_log(state, doc_dir)
    kinds = [
        e.get("kind", "") for e in json.loads((doc_dir / "audit_log.json").read_text())["events"]
    ]
    assert "d3_floor_model_table_kept" in kinds, kinds
    assert "table_region_unverifiable" not in kinds, kinds
    assert "native_fallback" not in kinds, kinds
    # Trust sidecar: removing the marker must not move the page to "trusted".
    trust = json.loads((doc_dir / "tables_trust.json").read_text())
    assert trust["untrusted_pages"] == [1], trust
    assert trust["pages"]["1"]["reasons"] == ["d3_floor_model_table_kept"], trust


def test_cli_reports_the_superseded_floor(tmp_path: Path, capsys) -> None:
    """CLI level: a distinct, non-quiet line naming the pages."""
    from socr.core.config import EngineType, PipelineConfig
    from socr.pipeline.orchestrator import UnifiedPipeline

    state = _state(_born_digital_pdf(tmp_path))
    pipeline = UnifiedPipeline(
        PipelineConfig(
            primary_engine=EngineType.QWEN,
            agentic=True,
            enabled_engines=[EngineType.QWEN],
            quiet=False,
            audit_enabled=True,
            save_figures=False,
            write_manifest=False,
        )
    )
    pipeline._phase_assemble(state, tmp_path / "out")
    printed = capsys.readouterr().out
    assert "MODEL" in printed and "source image" in printed, printed
    assert "D3 fail-closed" not in printed, printed
