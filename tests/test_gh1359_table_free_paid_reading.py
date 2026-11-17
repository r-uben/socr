"""#1359: a faithful table-free paid reading may lift the scanned-table floor, under a no-loss rule.

#1124 stops a reader from making a disputed table vanish by omitting it. The one exception: a PAID
reading taken after the witness disputed the table, that the page judge accepts and that holds no
table, lifts the floor only when it still contains every numeric token of the disputed table and
every label the witness did not dispute.

Hermetic per CLAUDE.md: the readers, the ladder and the judges are stubs, and every claim is a
DIFFERENCE between two arms run in one process that differ only in the one thing under test (what
the paid reading keeps), never an absolute outcome measured on this machine.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from test_gh1185_text_layer_numbers_e2e import _doc_dir, _page, _prose

import socr
from socr.core.config import EngineType
from socr.core.marks import Check, CheckResult
from socr.core.result import PageOutput, PageStatus
from socr.core.verdict import PAID_AFTER_DISPUTE_REASON, PAID_OMITS_TABLE_SUFFIX
from socr.pipeline.agentic import (
    AcceptDecision,
    PageDecision,
    ProviderAttempt,
    paid_reading_keeps_disputed_table_tokens,
    settle_scanned_table_decision,
)

WITNESS_DISPUTE = "source_evidence_table: numeric tokens unsupported by page evidence: ['0.0']"
DISPUTED = ("0.0",)

#: The free reader's invented table: axis ticks as rows. ``0.0`` is the token the witness disputes.
FREE_TABLE = (
    "| Date | Percent change |\n| --- | --- |\n| 1/19 | 0.0 |\n| 3/9 | 1.5 |\n| 4/20 | 3.0 |\n"
)
FREE_TEXT = "Evolution of the forecast.\n\n" + FREE_TABLE

#: Paid readings, table-free. KEEPS holds every numeric token of the table; DROPS misses 3.0.
KEEPS = "Evolution of the forecast. Y-axis ticks: 0.0, 1.5, 3.0. X-axis: 1/19, 3/9, 4/20."
DROPS = "Evolution of the forecast. Y-axis ticks: 0.0, 1.5. X-axis: 1/19, 3/9, 4/20."
PAID_WITH_TABLE = "Evolution of the forecast.\n\n| Date | Value |\n| --- | --- |\n| 1/19 | 1.5 |\n"


def test_loaded_source_is_this_checkout() -> None:
    assert Path(socr.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1] / "src")


# --- the condition ---------------------------------------------------------------------------


def _marked(text: str = FREE_TEXT, **kw) -> PageOutput:
    base = dict(
        page_num=1,
        text=text,
        status=PageStatus.SUCCESS,
        engine="qwen",
        table_disputed=WITNESS_DISPUTE,
        table_disputed_values=DISPUTED,
    )
    base.update(kw)
    return PageOutput(**base)


def _paid(text: str, *, after_dispute: bool = True) -> PageOutput:
    return PageOutput(
        page_num=1,
        text=text,
        status=PageStatus.SUCCESS,
        engine="gemini",
        judge_accepted=True,
        paid_after_witness_dispute=after_dispute,
    )


def test_the_condition_separates_a_reading_that_keeps_the_tokens_from_one_that_drops_one() -> None:
    marked = _marked()
    assert paid_reading_keeps_disputed_table_tokens(_paid(KEEPS), marked)
    assert not paid_reading_keeps_disputed_table_tokens(_paid(DROPS), marked)


def test_a_disputed_token_the_paid_reading_omits_blocks_the_lift_too() -> None:
    """The witness's own refutation carries no exemption for numbers (the starting candidate
    exempted them): a table-free reading without ``0.0`` is refused."""
    no_zero = "Evolution of the forecast. Y-axis ticks: 1.5, 3.0. X-axis: 1/19, 3/9, 4/20."
    assert not paid_reading_keeps_disputed_table_tokens(_paid(no_zero), _marked())


@pytest.mark.parametrize(
    "kw",
    [
        {"table_disputed": "", "table_disputed_values": (), "table_unverifiable": "no witness"},
        {"table_disputed_values": ()},
        {"table_unverifiable": "no witness"},
    ],
)
def test_without_the_witness_record_there_is_nothing_to_measure_a_loss_against(kw) -> None:
    assert not paid_reading_keeps_disputed_table_tokens(_paid(KEEPS), _marked(**kw))


def _attempt(engine: EngineType, output: PageOutput, *, accepted: bool, **kw) -> ProviderAttempt:
    return ProviderAttempt(engine=engine, output=output, cost_usd=0.0, accepted=accepted, **kw)


def _decision(paid_text: str, *, paid: bool = True) -> PageDecision:
    free = _attempt(EngineType.QWEN, _marked(), accepted=False, reason=WITNESS_DISPUTE)
    winner = _paid(paid_text, after_dispute=paid)
    return PageDecision(
        1,
        winner,
        [free, _attempt(EngineType.GEMINI, winner, accepted=True, reason="faithful")],
        accepted=True,
    )


def test_settle_lifts_for_the_paid_reading_that_keeps_the_tokens_and_not_for_the_one_that_drops() -> (
    None
):
    lifted, refused = _decision(KEEPS), _decision(DROPS)
    assert settle_scanned_table_decision(lifted) is None
    assert settle_scanned_table_decision(refused) is None
    # the only difference between the arms is what the paid reading kept
    assert lifted.accepted and lifted.final_output.paid_reading_omits_disputed_table
    assert lifted.floor_lift is not None
    assert not refused.accepted
    assert not refused.final_output.paid_reading_omits_disputed_table
    assert refused.floor_lift is None


def test_a_free_reading_never_lifts() -> None:
    """Same bytes, same earlier table: without the paid-after-dispute flag the floor stands."""
    free_win = _decision(KEEPS, paid=False)
    assert settle_scanned_table_decision(free_win) is None
    assert not free_win.accepted and free_win.floor_lift is None
    assert not free_win.final_output.paid_reading_omits_disputed_table


def test_a_paid_reading_with_a_table_behaves_as_before() -> None:
    """A table in the paid reading is #1124's own case: the lift is not consulted."""
    d = _decision(PAID_WITH_TABLE)
    settle_scanned_table_decision(d)
    assert not d.accepted
    assert d.floor_lift is None and not d.final_output.paid_reading_omits_disputed_table


def test_an_unjudged_or_rejected_paid_reading_does_not_lift() -> None:
    d = _decision(KEEPS)
    d.accepted = False
    assert settle_scanned_table_decision(d) is None
    assert d.floor_lift is None and not d.final_output.paid_reading_omits_disputed_table


def test_every_earlier_table_must_be_accounted_for() -> None:
    d = _decision(KEEPS)
    other = _marked("Other.\n\n| A | B |\n| --- | --- |\n| 9.9 | 8.8 |\n", engine="marker")
    d.attempts.insert(1, _attempt(EngineType.MARKER, other, accepted=False))
    settle_scanned_table_decision(d)
    assert not d.accepted and d.floor_lift is None


# --- end to end ------------------------------------------------------------------------------


def _scan_pdf(path: Path, *, pages: int = 1) -> Path:
    """Pages that are a picture and nothing else: no text layer, so the scanned-table floor and
    its ``settle_scanned_table_decision`` apply (a born-digital page skips them)."""
    import fitz

    doc = fitz.open()
    width, height = 612, 792
    # Dark stripes on white: ink, so the blank-page detector does not retire the page.
    row = bytes(
        ([0, 0, 0] if (x // 8) % 2 else [255, 255, 255])[c] for x in range(width) for c in range(3)
    )
    pix = fitz.Pixmap(fitz.csRGB, width, height, row * height, False)
    for _ in range(pages):
        page = doc.new_page(width=612, height=792)
        page.insert_image(page.rect, pixmap=pix)
    doc.save(path)
    doc.close()
    return path


class _Reader:
    def __init__(self, name: str, text: str) -> None:
        self.name = name
        self.text = text

    def is_available(self) -> bool:
        return True

    def process_pages(self, pdf_path, page_nums, config, dpi, subprocess_timeout=None, **_kw):
        return [
            PageOutput(
                page_num=n,
                text=f"{_prose(n)}\n\n{self.text}",
                status=PageStatus.SUCCESS,
                engine=self.name,
            )
            for n in page_nums
        ]


class _DisputingJudge:
    """Rejects the free rung's table as a witness dispute (the way the source-evidence judge
    records it on the output), accepts the paid rung."""

    def assess(self, output, provider):
        if provider.engine.value == "gemini":
            return AcceptDecision(accept=True, reason="faithful")
        output.table_disputed = WITNESS_DISPUTE
        output.table_disputed_values = DISPUTED
        return AcceptDecision(accept=False, reason=WITNESS_DISPUTE)


def _run(out_dir: Path, pdf: Path, monkeypatch, *, free: str, paid: str, free_engine: str = "qwen"):
    from socr.core.config import PipelineConfig
    from socr.core.providers import PROFILE_GEMINI, PROFILE_QWEN_LOCAL
    from socr.pipeline import orchestrator as orch
    from socr.pipeline.orchestrator import UnifiedPipeline

    texts = {"qwen": free, "gemini": paid}
    with monkeypatch.context() as m:
        m.setattr(orch, "get_engine", lambda et: _Reader(et.value, texts.get(et.value, "")))
        pipe = UnifiedPipeline(
            PipelineConfig(
                agentic=True,
                quiet=True,
                primary_engine=EngineType.QWEN,
                local_engine=EngineType.QWEN,
                enabled_engines=[EngineType.QWEN, EngineType.GEMINI],
                native_first=True,
                write_manifest=True,
                judge_backend="heuristic",
                dual_pass_tables=False,
                detect_equations=False,
                save_figures=True,
                describe_figure_crops=False,
            )
        )
        pipe._available_engines_for_agentic = lambda: (
            [PROFILE_QWEN_LOCAL, PROFILE_GEMINI]
            if free_engine == "qwen"
            else [PROFILE_GEMINI, PROFILE_QWEN_LOCAL]
        )
        pipe._build_page_judge = lambda state: _DisputingJudge()
        pipe._resolve_crop_vlm_model = lambda: None
        pipe._resolve_judge_model = lambda *a, **k: ""
        return pipe.process(pdf, output_dir=out_dir)


@pytest.fixture
def arms(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pdf = _scan_pdf(tmp_path / "doc.pdf")
    out = {}
    for name, paid in (("keeps", KEEPS), ("drops", DROPS), ("table", PAID_WITH_TABLE)):
        out[name] = tmp_path / name
        _run(out[name], pdf, monkeypatch, free=FREE_TEXT, paid=paid)
    return pdf, out


def _clauses(sidecar: dict, check: str) -> list[dict]:
    return [c for c in sidecar["verdict"]["clauses"] if c["check"] == check]


def _events(out: Path) -> list[dict]:
    log = json.loads((_doc_dir(out) / "audit_log.json").read_text())
    return [e for e in (log.get("events") if isinstance(log, dict) else log) or []]


def test_the_lifted_page_ships_the_paid_reading_as_a_warning_that_says_what_is_missing(
    arms,
) -> None:
    _pdf, out = arms
    side, body = _page(out["keeps"], 1)
    refused, _ = _page(out["drops"], 1)
    win = side["winning_output"]
    assert win["engine"] == "gemini"
    assert win["paid_reading_omits_disputed_table"] is True
    assert side["status"] == "warning"
    [clause] = _clauses(side, Check.READING_DISPUTE.value)
    assert clause["result"] == CheckResult.UNKNOWN.value
    assert clause["reason"] == PAID_AFTER_DISPUTE_REASON + PAID_OMITS_TABLE_SUFFIX
    assert "earlier reading showed a table that this reading does not contain" in body
    assert KEEPS in body
    # the arm that dropped a token is not lifted: it ends as it does without this change
    assert refused["status"] == "error"
    assert "paid_reading_omits_disputed_table" not in refused["winning_output"]


def test_a_paid_reading_with_a_table_is_unchanged_by_the_lift(arms) -> None:
    _pdf, out = arms
    side, _ = _page(out["table"], 1)
    assert "paid_reading_omits_disputed_table" not in side["winning_output"]
    assert not any(e["kind"] == "paid_reading_lifted_floor" for e in _events(out["table"]))


def test_the_audit_event_is_present_only_for_the_lifted_arm(arms) -> None:
    _pdf, out = arms
    lifted = [e for e in _events(out["keeps"]) if e["kind"] == "paid_reading_lifted_floor"]
    assert len(lifted) == 1
    assert lifted[0]["data"]["earlier_tables"][0]["numeric_tokens"] == 3
    assert not [e for e in _events(out["drops"]) if e["kind"] == "paid_reading_lifted_floor"]


def test_a_paid_reading_with_a_table_that_keeps_every_token_is_not_lifted(
    tmp_path: Path, monkeypatch
) -> None:
    """Load-bearing for the table-free requirement: every token kept, but a table present. The
    #1124 path decides, so no flag, no event and no 'does not contain a table' wording."""
    pdf = _scan_pdf(tmp_path / "doc.pdf")
    out = tmp_path / "tbl"
    kept_with_table = KEEPS + "\n\n| Date | Value |\n| --- | --- |\n| 1/19 | 1.5 |\n"
    _run(out, pdf, monkeypatch, free=FREE_TEXT, paid=kept_with_table)
    side, body = _page(out, 1)
    assert "paid_reading_omits_disputed_table" not in side["winning_output"]
    assert not any(e["kind"] == "paid_reading_lifted_floor" for e in _events(out))
    assert "does not contain" not in body


def test_a_table_present_never_lifts_at_unit_level_even_when_every_token_is_kept() -> None:
    d = _decision(KEEPS + "\n\n| Date | Value |\n| --- | --- |\n| 1/19 | 1.5 |\n")
    assert paid_reading_keeps_disputed_table_tokens(d.final_output, _marked())
    settle_scanned_table_decision(d)
    assert d.floor_lift is None and not d.final_output.paid_reading_omits_disputed_table


# --- labels ----------------------------------------------------------------------------------

LABEL_TABLE = (
    "| Item | Value |\n| --- | --- |\n| Alpha | 0.0 |\n| Bravo | 1.5 |\n| Charlie | 3.0 |\n"
)


def test_the_label_half_of_the_condition_is_load_bearing() -> None:
    marked = _marked("Intro.\n\n" + LABEL_TABLE, table_disputed_values=("charlie",))
    keeps = "Ticks 0.0 1.5 3.0 for alpha bravo charlie."
    drops_undisputed = "Ticks 0.0 1.5 3.0 for alpha charlie."
    drops_disputed = "Ticks 0.0 1.5 3.0 for alpha bravo."
    assert paid_reading_keeps_disputed_table_tokens(_paid(keeps), marked)
    assert not paid_reading_keeps_disputed_table_tokens(_paid(drops_undisputed), marked)
    # a disputed label carries no obligation
    assert paid_reading_keeps_disputed_table_tokens(_paid(drops_disputed), marked)


LABEL_ONLY = "| Month | Note |\n| --- | --- |\n| june | a |\n| july | b |\n"


@pytest.mark.parametrize("disputed", [("june", "july"), ("june",), ("0.0",)])
def test_a_label_only_table_never_lifts_whatever_the_paid_reading_holds(disputed) -> None:
    """Review round: with no numeric token there is nothing to measure a loss against, and a
    weak witness disputes labels wholesale. A paid reading that is only ``Figure.``, and one that
    repeats every label, both leave the floor to today's promotion."""
    marked = _marked("Intro.\n\n" + LABEL_ONLY, table_disputed_values=disputed)
    for paid_text in ("Figure.", "june july a b"):
        assert not paid_reading_keeps_disputed_table_tokens(_paid(paid_text), marked)


def test_numerals_in_unrelated_prose_are_a_documented_residual() -> None:
    """Containment is presence, not placement: prose that happens to carry the table's three
    numbers satisfies it. Pinned so a change is deliberate; the page judge is the guard."""
    prose = "Page 3 of 15. Growth was 1.5 percent; see note 0.0 and section 3.0."
    assert paid_reading_keeps_disputed_table_tokens(_paid(prose), _marked())


def test_the_flag_survives_the_sidecar_round_trip_and_refinalize() -> None:
    from socr.core.manifest import apply_verdict_marks
    from test_gh1346_paid_after_dispute_marked import _output, _p

    out = _output(paid_reading_omits_disputed_table=True)
    assert PageOutput.from_dict(out.to_dict()).paid_reading_omits_disputed_table is True
    assert "paid_reading_omits_disputed_table" not in _output().to_dict()
    first, verdict = apply_verdict_marks(out, _p(), 1)
    assert "earlier reading showed a table that this reading does not contain" in first.text
    restored = PageOutput.from_dict(first.to_dict())
    second, _ = apply_verdict_marks(restored, _p(), 1)
    assert second.text == first.text
    # the flag off: the 1346 wording only
    plain, _ = apply_verdict_marks(replace(out, paid_reading_omits_disputed_table=False), _p(), 1)
    assert "does not contain" not in plain.text
