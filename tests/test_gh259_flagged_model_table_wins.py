"""#259: a flagged-but-PRESENT model table must not be replaced by native text.

Measured on the owner's cached run of a 2002 working paper (one dense
regression table with spanning column headers, ``--agentic --no-native-first``):

* every rung of the ladder was rejected by the judge (``manifest.json``
  journal: five attempts, ``accepted: false`` on all five), so
  ``route_page`` returned ``PageDecision(accepted=False)`` and
  ``_best_effort`` kept the qwen attempt as ``final_output``;
* ``orchestrator.py`` then wrote ``att.output.audit_passed = att.accepted``
  (False) and, because the ladder did not accept on a table page, set
  ``ps.native_table_structure_failed = True``;
* ``manifest.py::_winning_page_output`` gates on
  ``p.best_output.audit_passed``, so the qwen output — 26 rows, paired
  headers, all 112 decimals, confirmed correct against the printed page —
  was skipped and the native branch shipped 19 flat rows instead.

Neither lane lost a number. The difference is purely structure, and native's
structure is the wrong one: ``table_not_scorable`` fired on the same page,
so the pipeline already knew the native text did not form a grid at the
moment it chose it as the replacement.

Hermetic: drives ``_winning_page_output`` / ``canonical_page_texts``
directly. No provider ladder, no ``_phase_agentic``, no
``_available_engines_for_agentic`` patch, no ollama.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from socr.core.document import DocumentHandle
from socr.core.manifest import _winning_page_output, canonical_page_texts
from socr.core.result import PageOutput, PageStatus
from socr.core.state import DocumentState

# Two readings of the same table. Both carry every value; only the structure
# differs. The model keeps the spanning header band as a paired row; native
# flattens it into one lane per printed column.
MODEL_TABLE = (
    "Table 1. Regressions of 1-year excess returns on all forward rates\n\n"
    "| $n$ | const. | $y^{(1)}$ | $f^{(1\\to2)}$ | $R^2$ |\n"
    "|---|---|---|---|---|\n"
    "| 2 | -1.96 | -0.94 | 0.74 | 0.34 |\n"
    "| | (0.64) | (0.18) | (0.43) | |\n"
    "| Large T | | | | |\n"
    "| Small T | (0.81) | (0.30) | (0.50) | |\n"
)
NATIVE_TABLE = (
    "Table 1. Regressions of 1-year excess returns on all forward rates\n\n"
    "|  |  | y(1) | (1\u21922) | R2 |\n"
    "| --- | --- | --- | --- | --- |\n"
    "| n | const. |  | f |  |\n"
    "| 2 | \u22121.96 | \u22120.94 | 0.74 | 0.34 |\n"
    "|  | (0.64) | (0.18) | (0.43) |  |\n"
    "|  | (0.81) | (0.30) | (0.50) |  |\n"
    "\nLarge T\nSmall T\nEH\n"
)


def _born_digital_pdf(tmp_path: Path) -> Path:
    fitz = pytest.importorskip("fitz")
    path = tmp_path / "one_table.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "Table 1. Regressions of 1-year excess returns")
    doc.save(str(path))
    doc.close()
    return path


def _state(
    pdf_path: Path,
    *,
    model_text: str,
    model_status: PageStatus = PageStatus.SUCCESS,
    model_engine: str = "qwen",
    structure_failed: bool = True,
    unverifiable: bool = False,
) -> DocumentState:
    """The page state the cached run produced, rebuilt field by field."""
    state = DocumentState(handle=DocumentHandle.from_path(pdf_path))
    ps = state.pages[1]
    ps.is_born_digital = True
    ps.has_tables = True
    ps.native_text = NATIVE_TABLE
    ps.native_table_structure_failed = structure_failed
    ps.native_table_unverifiable = unverifiable

    native_attempt = PageOutput(
        page_num=1,
        text=NATIVE_TABLE,
        status=PageStatus.WARNING,
        engine="native",
        audit_passed=False,
    )
    model_attempt = PageOutput(
        page_num=1,
        text=model_text,
        status=model_status,
        engine=model_engine,
        # route_page: att.output.audit_passed = att.accepted, and the ladder
        # accepted nothing.
        audit_passed=False,
    )
    ps.attempts.extend([native_attempt, model_attempt])
    # _best_effort keeps the most trustworthy attempt as the winner.
    ps.best_output = model_attempt
    return state


def test_flagged_model_table_ships_instead_of_flat_native(tmp_path: Path) -> None:
    """The defect, on the behaviour that exists at the baseline.

    Asserted on ``canonical_page_texts`` as well as the winner, because that
    is what reaches the saved ``.md`` and ``pages/NNN.md``.
    """
    state = _state(_born_digital_pdf(tmp_path), model_text=MODEL_TABLE)

    winner = _winning_page_output(state, 1)
    assert winner.engine == "qwen", (
        f"the model produced a present, flagged table but the winner is "
        f"{winner.engine!r}; text={winner.text!r}"
    )
    assert "$f^{(1\\to2)}$" in winner.text, winner.text

    body = "\n\n".join(canonical_page_texts(state))
    assert "$f^{(1\\to2)}$" in body, body
    # The native reading is a real markdown grid too -- it is just the wrong
    # one: its row labels sit OUTSIDE the table. Its arrival is the defect.
    assert body.strip() != NATIVE_TABLE.strip(), "the damaged native reading shipped"


def test_kept_model_table_is_demoted_and_carries_its_flag(tmp_path: Path) -> None:
    """Kept, but never as a clean pass: the page still says it was flagged."""
    state = _state(_born_digital_pdf(tmp_path), model_text=MODEL_TABLE)

    winner = _winning_page_output(state, 1)
    assert winner.engine == "qwen", winner.engine
    assert winner.status is not PageStatus.SUCCESS, winner.status
    assert winner.audit_passed is False
    assert getattr(winner.failure_mode, "value", "") != "none", winner.failure_mode


def test_model_produced_nothing_still_falls_back_to_native(tmp_path: Path) -> None:
    """Reverse regression: an EMPTY model output must not displace native.

    "Flagged" and "absent" are exactly the two cases the fix separates; a fix
    that always prefers the model would ship empty pages.
    """
    state = _state(_born_digital_pdf(tmp_path), model_text="   \n")

    winner = _winning_page_output(state, 1)
    assert winner.engine == "native", winner.engine
    assert "Large T" in winner.text, winner.text


def test_d3_fail_closed_floor_still_wins_over_a_flagged_model_table(tmp_path: Path) -> None:
    """Reverse regression: the TR-3 D3 floor is a hard-fail, not a flag.

    A page whose per-region geometry verifier hard-failed AND whose ladder
    failed ships the explicit failed-table marker. Preferring the model output
    there would re-open exactly the plausible-but-wrong table D3 exists to
    prevent.
    """
    state = _state(
        _born_digital_pdf(tmp_path),
        model_text=MODEL_TABLE,
        structure_failed=True,
        unverifiable=True,
    )

    winner = _winning_page_output(state, 1)
    assert "failed: unverifiable table" in winner.text, winner.text
    assert winner.status is PageStatus.ERROR, winner.status


def test_model_produced_no_table_on_a_table_page_falls_back_to_native(
    tmp_path: Path,
) -> None:
    """Reverse regression: prose from the model is not a table reading.

    The comparison this fixes is between two readings of a grid. A model
    output with no grid in it did not produce the thing under comparison, so
    native stays the only table reading the page has.
    """
    state = _state(
        _born_digital_pdf(tmp_path),
        model_text="# UNITED STATES\n\nGoldman Sachs revised its forecast upward.\n",
    )

    winner = _winning_page_output(state, 1)
    assert winner.engine == "native", winner.engine
    assert "Large T" in winner.text, winner.text


def test_native_engine_winner_is_untouched(tmp_path: Path) -> None:
    """Reverse regression: this is about MODEL output, not the native lane.

    A native-engine ``best_output`` must keep going through the native branch,
    so the frozen-snapshot handling below it (GH-211) is not bypassed.
    """
    state = _state(_born_digital_pdf(tmp_path), model_text=MODEL_TABLE, model_engine="native+math")

    winner = _winning_page_output(state, 1)
    assert winner.engine == "native", winner.engine


def test_document_status_audit_event_and_cli_surface_the_kept_page(tmp_path: Path) -> None:
    """End to end: the flag reaches page, document, metadata and CLI.

    Hermetic exactly like ``test_agentic_parity_on_ce_like_fixture``:
    ``_available_engines_for_agentic`` and ``route_page`` are patched, so no
    ollama and no provider are needed.
    """
    from unittest.mock import patch

    from socr.core.config import EngineType, PipelineConfig
    from socr.core.providers import PROFILE_QWEN_LOCAL
    from socr.core.result import DocumentStatus
    from socr.pipeline.agentic import PageDecision, ProviderAttempt
    from socr.pipeline.orchestrator import UnifiedPipeline

    fitz = pytest.importorskip("fitz")
    pdf_path = tmp_path / "table_page.pdf"
    doc = fitz.open()
    page = doc.new_page()
    y = 72
    for line in NATIVE_TABLE.splitlines():
        page.insert_text((54, y), line or " ")
        y += 14
    doc.save(str(pdf_path))
    doc.close()

    pipeline = UnifiedPipeline(
        PipelineConfig(
            primary_engine=EngineType.QWEN,
            agentic=True,
            judge_backend="heuristic",
            enabled_engines=[EngineType.QWEN],
            quiet=True,
            audit_enabled=True,
            save_figures=False,
            write_manifest=False,
        )
    )

    def _rejecting_route(page_num, ladder, run_provider, judge, **kwargs):
        out = PageOutput(
            page_num=page_num,
            text=MODEL_TABLE,
            status=PageStatus.SUCCESS,
            engine="qwen",
            audit_passed=False,
        )
        prof = ladder[0]
        att = ProviderAttempt(
            engine=prof.engine,
            output=out,
            cost_usd=0.0,
            accepted=False,
            reason="native_table_verifier: ambiguous_lane_count_mismatch",
            provider_id=prof.id,
            model=prof.model,
            backend=prof.backend,
        )
        return PageDecision(page_num=page_num, final_output=out, attempts=[att])

    with (
        patch.object(pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]),
        patch("socr.pipeline.orchestrator.route_page", side_effect=_rejecting_route),
        patch("socr.pipeline.orchestrator.probe_ollama_idle", return_value=True),
    ):
        result = pipeline.process(pdf_path, tmp_path / "out")

    # Page level: the model's reading is what shipped.
    assert "$f^{(1\\to2)}$" in result.markdown, result.markdown
    # Document level: never a clean SUCCESS.
    assert result.status is not DocumentStatus.SUCCESS, result.status
    # Audit level: its own kind, distinct from native_fallback (nothing fell back).
    audit_path = next((tmp_path / "out").rglob("audit_log.json"))
    kinds = [e.get("kind", "") for e in json.loads(audit_path.read_text())["events"]]
    assert "flagged_model_table_kept" in kinds, kinds
    assert "native_fallback" not in kinds, kinds
