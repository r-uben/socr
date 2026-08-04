"""GH-96: the table escalation lane.

Re-reads a table page whose output disagrees with its own native text layer, and
keeps the result only when hierarchy-aware exactness measurably improves.

Every test here drives the lane's own methods directly with a stubbed provider. None
touches a network, an engine, or the provider ladder, so nothing depends on ollama or
an API key being present — the CI trap named in CLAUDE.md.

The five cases below are not hypothetical; each is a trap identified during design,
and two of them were observed live.
"""

from __future__ import annotations

import fitz
import pytest

from socr.core.config import EngineType, PipelineConfig
from socr.core.providers import DEFAULT_PROVIDERS, TIER_LOCAL
from socr.core.result import PageOutput, PageStatus
from socr.pipeline.orchestrator import UnifiedPipeline

_ROWS = [
    ("Total effect of decisions", ["42.8", "30.5", "2.6"]),
    ("September energy package", ["43.2", "26.8", "3.7"]),
    ("Energy price guarantee", ["24.8", "26.8", "3.7"]),
    ("Energy bill relief scheme", ["18.4", "9.1", "5.5"]),
]


def _md(rows):
    out = ["| | c1 | c2 | c3 |", "| --- | --- | --- | --- |"]
    for label, values in rows:
        out.append(f"| {label} | " + " | ".join(values) + " |")
    return "\n".join(out)


_PERFECT = _md(_ROWS)
# Imperfect enough to trigger (one wrong value) but far better than _SHIFTED.
_MOSTLY_RIGHT = _md(
    [
        ("Total effect of decisions", ["42.8", "30.5", "2.6"]),
        ("September energy package", ["43.2", "26.8", "3.7"]),
        ("Energy price guarantee", ["24.8", "26.8", "3.7"]),
        ("Energy bill relief scheme", ["18.4", "9.1", "9.9"]),
    ]
)
_SHIFTED = _md(
    [
        ("Total effect of decisions", ["42.8", "30.5", "2.6"]),
        ("September energy package", ["", "", ""]),
        ("Energy price guarantee", ["43.2", "26.8", "3.7"]),
        ("Energy bill relief scheme", ["24.8", "26.8", "3.7"]),
    ]
)


@pytest.fixture(scope="module")
def pdf_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("gh96lane") / "t.pdf"
    doc = fitz.open()
    pg = doc.new_page()
    y = 200.0
    for label, values in _ROWS:
        pg.insert_text((60.0, y), label, fontsize=9)
        for x, v in zip((300.0, 360.0, 420.0), values):
            pg.insert_text((x, y), v, fontsize=9)
        y += 18.0
    pg.draw_line(fitz.Point(50, 190), fitz.Point(470, 190))
    pg.draw_line(fitz.Point(50, y), fitz.Point(470, y))
    doc.save(path)
    doc.close()
    return path


class _State:
    def __init__(self):
        self.events = []
        self.engine_runs = []


class _PageState:
    def __init__(self, has_tables=True):
        self.has_tables = has_tables
        self.attempts = []
        self.native_table_structure_failed = False
        self.native_table_unverifiable = False
        self.scanned_table_evidence_failed = False


def _pipeline(**overrides):
    cfg = PipelineConfig(**overrides)
    pipe = object.__new__(UnifiedPipeline)
    pipe.config = cfg
    return pipe


def _out(text, engine="gemini", status=PageStatus.SUCCESS):
    return PageOutput(page_num=1, text=text, status=status, engine=engine)


_GEMINI = DEFAULT_PROVIDERS[EngineType.GEMINI]


# ----------------------------------------------------------------------
# Provider selection — trap: --strict-local silently bypassed
# ----------------------------------------------------------------------


def test_provider_comes_from_the_tier_filtered_ladder():
    pipe = _pipeline()
    available = [DEFAULT_PROVIDERS[EngineType.QWEN], _GEMINI]

    assert pipe._resolve_table_escalation_provider(available) is _GEMINI


def test_strict_local_leaves_no_escalation_provider():
    """The trap: naming EngineType.GEMINI literally would bypass --strict-local.

    run_provider accepts any engine, so a hardcoded escalation engine would still
    fire on exactly the configuration the reference corpus runs under. Choosing from
    the already-tier-filtered list makes the flag suppress the lane for free.
    """
    pipe = _pipeline()
    local_only = [p for p in DEFAULT_PROVIDERS.values() if p.tier == TIER_LOCAL]

    assert pipe._resolve_table_escalation_provider(local_only) is None


def test_the_lane_can_be_turned_off():
    pipe = _pipeline(escalate_ambiguous_tables=False)

    assert pipe._resolve_table_escalation_provider([_GEMINI]) is None


# ----------------------------------------------------------------------
# Trigger
# ----------------------------------------------------------------------


def test_a_perfect_page_does_not_trigger(pdf_path):
    pipe = _pipeline()
    state = _State()
    with fitz.open(pdf_path) as doc:
        assert not pipe._table_page_needs_escalation(state, 1, doc[0], _PageState(), _out(_PERFECT))


def test_a_disagreeing_page_triggers(pdf_path):
    pipe = _pipeline()
    state = _State()
    with fitz.open(pdf_path) as doc:
        assert pipe._table_page_needs_escalation(state, 1, doc[0], _PageState(), _out(_SHIFTED))


def _one_column_pdf(path, rows):
    """A page whose numeric lines are prose fragments, not a grid."""
    doc = fitz.open()
    pg = doc.new_page()
    y = 200.0
    for label, value in rows:
        pg.insert_text((60.0, y), label, fontsize=9)
        pg.insert_text((300.0, y), value, fontsize=9)
        y += 18.0
    pg.draw_line(fitz.Point(50, 190), fitz.Point(470, 190))
    pg.draw_line(fitz.Point(50, y), fitz.Point(470, y))
    doc.save(path)
    doc.close()
    return fitz.open(path)


def test_prose_fragments_are_not_a_grid(tmp_path):
    """GH-113: the native parser reads numeric prose lines as a table.

    On a full document this escalated a cover date ("November=['2022']") and a
    fragment of "4 per cent" ("cent=['4']"), each time paying for a cloud call to
    compare an empty result against an empty result. PageState.has_tables does NOT
    discriminate — it is True on those pages too.

    A grid needs at least two value columns; these have one.
    """
    opened = _one_column_pdf(tmp_path / "prose.pdf", [("November", "2022"), ("CP", "749")])
    pipe = _pipeline()
    state = _State()

    assert not pipe._table_page_needs_escalation(state, 1, opened[0], _PageState(), _out(_SHIFTED))
    opened.close()
    # #123 TICKET-C1: not-scorable is now surfaced, not silently absent.
    assert [e.kind for e in state.events] == ["table_not_scorable"]


def test_a_single_wide_row_is_not_a_grid(tmp_path):
    """One row of N values is a line; a grid needs two rows sharing that width."""
    path = tmp_path / "onerow.pdf"
    doc = fitz.open()
    pg = doc.new_page()
    pg.insert_text((60.0, 200.0), "1998- 2007 2010- 2019", fontsize=9)
    for x, v in zip((300.0, 360.0, 420.0), ["1.5", "1.0", "2.7"]):
        pg.insert_text((x, 200.0), v, fontsize=9)
    pg.draw_line(fitz.Point(50, 190), fitz.Point(470, 190))
    pg.draw_line(fitz.Point(50, 215), fitz.Point(470, 215))
    doc.save(path)
    doc.close()
    opened = fitz.open(path)
    pipe = _pipeline()
    state = _State()

    assert not pipe._table_page_needs_escalation(state, 1, opened[0], _PageState(), _out(_SHIFTED))
    opened.close()


def test_a_non_grid_page_costs_nothing(tmp_path):
    """End to end: no provider call, no spend.

    #123 TICKET-C1: an ``AuditEvent`` IS now expected — a not-scorable page must
    surface, not disappear. What must still be zero is provider cost.
    """
    opened = _one_column_pdf(tmp_path / "prose2.pdf", [("November", "2022"), ("CP", "749")])
    pdf_path = tmp_path / "prose2.pdf"
    opened.close()
    pipe = _pipeline()
    state, ps = _State(), _PageState()
    bo = _out(_SHIFTED, engine="qwen")
    calls = []

    def run_provider(engine, page_num):
        calls.append(engine)
        return _out(_PERFECT)

    pipe._escalate_table_page(state, 1, ps, bo, _GEMINI, run_provider, pdf_path)

    assert calls == [], "a page with no table must never reach the provider"
    assert [e.kind for e in state.events] == ["table_not_scorable"]
    assert state.engine_runs == []


# ----------------------------------------------------------------------
# The lane
# ----------------------------------------------------------------------


def _run(pipe, pdf_path, incumbent, provider_result, ps=None):
    state, ps = _State(), ps or _PageState()
    bo = _out(incumbent, engine="qwen")

    def run_provider(engine, page_num):
        if isinstance(provider_result, Exception):
            raise provider_result
        return provider_result

    degraded = pipe._escalate_table_page(state, 1, ps, bo, _GEMINI, run_provider, pdf_path)
    return state, ps, bo, degraded


def test_a_measured_improvement_is_kept(pdf_path):
    state, ps, bo, degraded = _run(_pipeline(), pdf_path, _SHIFTED, _out(_PERFECT))

    assert bo.text == _PERFECT
    assert not degraded
    kinds = [e.kind for e in state.events]
    assert "table_escalation_accepted" in kinds
    assert state.engine_runs, "the cloud call must be costed"


def test_a_regression_is_rejected_and_the_incumbent_kept(pdf_path):
    """The case token containment cannot see — every value is native-supported.

    The incumbent must be imperfect enough to trigger, yet still better than the
    candidate. A perfect incumbent never reaches the gate at all: the trigger
    declines to escalate it, which is the behaviour asserted separately.
    """
    state, ps, bo, _ = _run(_pipeline(), pdf_path, _MOSTLY_RIGHT, _out(_SHIFTED))

    assert bo.text == _MOSTLY_RIGHT, "a worse candidate must never replace the incumbent"
    assert "table_escalation_rejected" in [e.kind for e in state.events]


def test_a_native_fallback_result_is_never_accepted(pdf_path):
    """The trap: _run_engine_on_pages turns a failed call into native text.

    Assigning it would swap a structured table for flattened native text — silent
    content loss — so the candidate is only considered when the intended engine
    actually answered.
    """
    state, ps, bo, _ = _run(_pipeline(), pdf_path, _SHIFTED, _out(_PERFECT, engine="native"))

    assert bo.text == _SHIFTED
    assert "table_escalation_refused" in [e.kind for e in state.events]


def test_a_failed_candidate_is_refused(pdf_path):
    state, ps, bo, _ = _run(_pipeline(), pdf_path, _SHIFTED, _out("", status=PageStatus.ERROR))

    assert bo.text == _SHIFTED
    assert "table_escalation_refused" in [e.kind for e in state.events]


def test_a_wedged_provider_disables_the_lane_for_the_document(pdf_path):
    """Observed live: a cloud OCR CLI sat wedged mid-request for 97 minutes.

    Escalation runs inline in the page-major loop, so an unbounded call stalls the
    whole document. The existing cascade-halt cannot see it — that probes ollama,
    not a cloud subprocess.
    """
    import time

    pipe = _pipeline(escalation_timeout_sec=0.2)
    state, ps = _State(), _PageState()
    bo = _out(_SHIFTED, engine="qwen")

    def hang(engine, page_num):
        time.sleep(5)
        return _out(_PERFECT)

    degraded = pipe._escalate_table_page(state, 1, ps, bo, _GEMINI, hang, pdf_path)

    assert degraded is True, "the lane must latch off after a hang"
    assert bo.text == _SHIFTED
    assert "table_escalation_timeout" in [e.kind for e in state.events]


def test_an_exception_never_loses_the_page(pdf_path):
    state, ps, bo, degraded = _run(
        _pipeline(), pdf_path, _SHIFTED, RuntimeError("provider exploded")
    )

    assert bo.text == _SHIFTED
    assert not degraded


# ----------------------------------------------------------------------
# Fail-closed recovery — trap: manifest and fragment diverge
# ----------------------------------------------------------------------


def test_recovering_a_fail_closed_page_clears_its_flags(pdf_path):
    """manifest._winning_page_output re-derives the winner from PageState flags.

    A page still marked unverifiable ships the "[page N failed]" marker even after
    its text is fixed, so the fragment and the manifest would disagree.
    """
    ps = _PageState()
    ps.native_table_unverifiable = True
    ps.native_table_structure_failed = True

    state, ps, bo, _ = _run(_pipeline(), pdf_path, _SHIFTED, _out(_PERFECT), ps=ps)

    assert ps.native_table_unverifiable is False
    assert ps.native_table_structure_failed is False
    assert "table_escalation_recovered_fail_closed" in [e.kind for e in state.events]


def test_a_rejected_escalation_leaves_fail_closed_flags_alone(pdf_path):
    ps = _PageState()
    ps.native_table_unverifiable = True

    state, ps, bo, _ = _run(_pipeline(), pdf_path, _PERFECT, _out(_SHIFTED), ps=ps)

    assert ps.native_table_unverifiable is True


# ----------------------------------------------------------------------
# Resume
# ----------------------------------------------------------------------


def test_the_flag_changes_the_run_fingerprint():
    """Else a resumed run reuses fragments from the other setting and ships a mix."""
    on = _pipeline(escalate_ambiguous_tables=True)
    off = _pipeline(escalate_ambiguous_tables=False)

    assert on._run_fingerprint(EngineType.QWEN) != off._run_fingerprint(EngineType.QWEN)


# ----------------------------------------------------------------------
# Trust surface
# ----------------------------------------------------------------------


def test_an_unresolved_escalation_leaves_the_page_untrusted():
    from socr.core.tables_trust import TABLE_DISTRUST_KINDS

    assert "table_escalation_refused" in TABLE_DISTRUST_KINDS
    assert "table_escalation_timeout" in TABLE_DISTRUST_KINDS
    assert "table_escalation_accepted" not in TABLE_DISTRUST_KINDS
