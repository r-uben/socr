"""GH-714 round 2: Astra's review reproducer, transcribed into the suite.

Source: the 2026-09-10 REQUEST_CHANGES review of #714 at 0a10cd5
(``/private/tmp/test_astra_714.py``). It is the evidence that reversed round
1's design, so it lives in the repo rather than in a scratch directory.

Transcribed with the reproduction UNCHANGED -- the real BoE PDF, its saved
sidecar geometry and flags, the real cached qwen candidate, and one text-only
productivity row replaced by a fabricated sentence while every number and all
geometry stay as they were.

Two things are adapted, deliberately, and both are named here:

* **The corpus skip.** The originals ran on a machine with
  ``~/Data/socr/census-boe-2026-09-10``; CI has no such corpus, so the two
  real-page tests skip there. The hermetic controls run everywhere.
* **The round-1 assertion.** Astra's first test asserted that the gated and
  ungated bodies DIFFER, which was true of round 1 (it shipped the candidate
  where the old code shipped the marker). Under round 2 both withhold, so the
  bytes are identical and the REASON is what differs. The assertion is re-pinned
  to that, and the fabrication assertions -- the actual finding -- are
  unchanged and now hold on both sides.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from socr.core.manifest import (
    SelectionProvenance,
    _row_corroborated_grid_winner,
    _select_page_output_tagged,
    finalized_page_record,
)
from socr.core.result import FailureMode, PageOutput
from socr.core.state import DocumentState
from socr.tables import structure_check as sc

from test_gh703_text_table_dominance import (  # noqa: I001  (pytest rootdir import)
    BOE_2018_P1_QWEN,
    BOE_2018_PDF,
)

_CORPUS = pytest.mark.skipif(
    not (BOE_2018_PDF.exists() and BOE_2018_P1_QWEN.exists()),
    reason="BoE census corpus not present on this machine",
)

FABRICATION = (
    "| The Bank guarantees permanent prosperity without any risk. | Unconditional guarantee. |"
)


def make_state(text: str | None = None) -> DocumentState:
    """Astra's fixture builder, verbatim in substance: the real PDF, the run's
    own sidecar geometry and page flags, and the run's own cached candidate.
    """
    import pymupdf

    from socr.core.document import DocumentHandle

    meta = json.loads((BOE_2018_P1_QWEN.parents[2] / "pages/00001.json").read_text())
    state = DocumentState(handle=DocumentHandle.from_path(BOE_2018_PDF))
    ps = state.pages[1]
    for key, value in meta.items():
        if key.startswith(("native_table_", "detected_table_")) or key in (
            "scanned_table_evidence_failed",
            "table_ladder_disposition",
            "table_ladder_incomplete",
        ):
            setattr(ps, key, value)
    ps.is_born_digital = True
    ps.has_tables = True
    with pymupdf.open(BOE_2018_PDF) as doc:
        ps.native_words = doc[0].get_text("words")
        ps.native_text = doc[0].get_text("text")
    out = PageOutput.from_dict(json.loads(BOE_2018_P1_QWEN.read_text()))
    if text is not None:
        out.text = text
    ps.attempts = [out]
    ps.best_output = out
    return state


def _fabricated(raw: str) -> str:
    """One text-only cell replaced. Every numeric value is preserved."""
    lines = raw.splitlines()
    changed = 0
    for i, line in enumerate(lines):
        if line.startswith("|") and "Quarterly hourly labour productivity" in line:
            lines[i] = FABRICATION
            changed += 1
    assert changed == 1
    return "\n".join(lines)


@_CORPUS
def test_real_cache_admission_and_fabricated_prose() -> None:
    """Astra's headline case, re-pinned to round 2.

    Round 1 shipped the real candidate here and, with one prose cell replaced,
    shipped the fabrication -- on corroboration evidence that cannot tell the
    two apart. Round 2 withholds both. The gated and ungated bodies are now the
    same marker, and the FAILURE MODE is the difference: the declined text-table
    route says so in its own words instead of claiming the ladder was exhausted.
    """
    state = make_state()
    raw = state.pages[1].best_output.text

    record = finalized_page_record(state, 1)
    with pytest.MonkeyPatch.context() as m:
        m.setattr(sc, "_native_page_has_column_lanes", lambda words: True)
        refused = finalized_page_record(make_state(), 1)

    assert record.output.text == refused.output.text, "round 2 withholds on both sides"
    assert record.output.failure_mode is FailureMode.ROW_SHAPE_NOT_RECONCILABLE_TEXT_TABLE
    assert refused.output.failure_mode is FailureMode.STRUCTURE_CLASS_LADDER_EXHAUSTED

    fabricated = _fabricated(raw)
    assert fabricated != raw
    kept = finalized_page_record(make_state(fabricated), 1)
    with pytest.MonkeyPatch.context() as m:
        m.setattr(sc, "_native_page_has_column_lanes", lambda words: True)
        blocked = finalized_page_record(make_state(fabricated), 1)

    assert "guarantees permanent prosperity" not in blocked.output.text
    assert "guarantees permanent prosperity" not in kept.output.text


@_CORPUS
def test_the_corroboration_cannot_tell_the_fabrication_apart() -> None:
    """The measurement behind the ruling: A1a scores the honest candidate and
    the fabricated one identically, so it can never be the guard for prose.
    """
    state = make_state()
    raw = state.pages[1].best_output.text
    honest = _row_corroborated_grid_winner(state.pages[1])
    forged = _row_corroborated_grid_winner(make_state(_fabricated(raw)).pages[1])

    # under round 2 the route seats no winner at all -- which is the fix
    assert honest is None and forged is None

    # and the evidence it WOULD have used is byte-identical between the two
    from socr.tables.row_corroboration import corroborate_rows

    words = state.pages[1].native_words
    region = None
    scores = [corroborate_rows(words, text, region) for text in (raw, _fabricated(raw))]
    assert scores[0].bound == scores[1].bound
    assert scores[0].total == scores[1].total
    assert scores[0].extra_numbers == scores[1].extra_numbers


@_CORPUS
def test_zero_numeric_table_is_not_admitted() -> None:
    """Astra's control: an entirely non-numeric table never reaches the
    row-shape check -- A1a returns ``clears=None`` and drops it earlier.

    This is why the declined-route reason is scoped to candidates that DID
    clear A1a: a page like this one floors under the ordinary exhausted reason,
    which is the truthful one for it.
    """
    raw = "| Claim | Promise |\n| --- | --- |\n| The Bank guarantees prosperity | No risk |\n"
    state = make_state(raw)
    assert _row_corroborated_grid_winner(state.pages[1]) is None

    out, tag = _select_page_output_tagged(state, 1)
    assert "guarantees prosperity" not in (out.text or "")
    assert (out.failure_mode, tag) == (
        FailureMode.STRUCTURE_CLASS_LADDER_EXHAUSTED,
        SelectionProvenance.STRUCTURE_CLASS_FLOOR,
    )


@_CORPUS
def test_selection_gates_receive_identical_whole_page_words() -> None:
    """Astra's third control: A1b's eligibility call and A2's term (b) are
    handed the SAME whole-page word list object, not two different populations.
    """
    state = make_state()
    seen: dict[str, list] = {}
    original = sc._native_page_has_column_lanes

    def probe(words):
        caller = inspect.currentframe().f_back.f_code.co_name
        seen.setdefault(caller, []).append(words)
        return original(words)

    with pytest.MonkeyPatch.context() as m:
        m.setattr(sc, "_native_page_has_column_lanes", probe)
        finalized_page_record(state, 1)
        sc._truncated_row_shortfall(state.pages[1].native_words, state.pages[1].best_output.text)

    assert "_row_shape_reconciliation" in seen
    assert "_truncated_row_shortfall" in seen
    assert all(words is state.pages[1].native_words for values in seen.values() for words in values)


def test_fixture_paths_are_the_census_ones() -> None:
    """Runs everywhere: names the corpus this file reproduces against, so a
    machine without it reports "not present" rather than silently testing
    nothing.
    """
    assert BOE_2018_PDF.name == "boe-meetings-2018-scan-p28-30.pdf"
    assert BOE_2018_PDF.parent == Path.home() / "Data/socr/census-boe-2026-09-10/in"
