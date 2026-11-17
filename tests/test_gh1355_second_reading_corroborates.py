"""#1355: a second reading by another engine corroborates a sign or decimal point the witness lost.

The scanned-table witness is Tesseract on a 150 dpi raster and drops minus signs and decimal
points (``-4.1`` read as ``41``). A paid reading run after a witness dispute passes the check for
routing only when the earlier disputed reading of the page by ANOTHER engine carries the same
token (multiset) and the witness holds an unconsumed token with the same digits. The page still
ships marked ``reading_dispute`` (#1346).

Hermetic per CLAUDE.md: no provider, no tesseract, no sample page. Route-level claims are
differences between two arms run in one process (only whether the earlier reading carries the
token differs), never an absolute outcome measured on this machine.
"""

from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import pytest

fitz = pytest.importorskip("fitz")

from socr.core.providers import PROFILE_GEMINI, PROFILE_MARKER, PROFILE_QWEN_LOCAL  # noqa: E402
from socr.core.result import FailureMode, PageOutput, PageStatus  # noqa: E402
from socr.pipeline.agentic import AcceptDecision, SourceEvidenceTableJudge, route_page  # noqa: E402
from socr.tables.source_evidence import (  # noqa: E402
    CORROBORATED_KIND,
    SecondReading,
    SourceEvidenceBundle,
    collect_table_tokens,
    corroborate_unsupported_numeric,
    verify_table_tokens,
)


def C(*tokens: str) -> Counter:
    return Counter(tokens)


# --- the pure function ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("emitted", "witness"),
    [
        ("-4.1", "41"),  # sign and point lost
        ("-4.1", "4.1"),  # sign lost
        ("-4.1", "-41"),  # point lost
        ("1.1", "11"),  # point lost
        ("-5.2", "52"),
        ("-12.0", "120"),  # trailing zero kept: only the point moved
        ("-0.7", "-7"),  # ``.7`` is written ``0.7`` by the tokeniser; the witness read ``7``
        ("0.7", "7"),
    ],
)
def test_sign_or_point_loss_is_corroborated(emitted: str, witness: str) -> None:
    corroborated, still = corroborate_unsupported_numeric(C(emitted), C(witness), C(emitted))
    assert corroborated == C(emitted)
    assert still == C()


def test_one_witness_token_cannot_corroborate_two_emitted_tokens() -> None:
    both, _ = corroborate_unsupported_numeric(C("-4.1"), C("41"), C("-4.1"))
    assert both == C("-4.1")
    # two different emitted tokens with the same digits, one witness token
    corroborated, still = corroborate_unsupported_numeric(
        C("-4.1", "4.1"), C("41"), C("-4.1", "4.1")
    )
    assert sum(corroborated.values()) == 1
    assert sum(still.values()) == 1
    # the same token emitted twice: the second reading must carry it twice, the witness twice
    assert corroborate_unsupported_numeric(C("-4.1", "-4.1"), C("41"), C("-4.1", "-4.1")) == (
        C("-4.1"),
        C("-4.1"),
    )
    assert corroborate_unsupported_numeric(C("-4.1", "-4.1"), C("41", "41"), C("-4.1")) == (
        C(),
        C("-4.1", "-4.1"),
    )
    assert corroborate_unsupported_numeric(C("-4.1", "-4.1"), C("41", "41"), C("-4.1", "-4.1")) == (
        C("-4.1", "-4.1"),
        C(),
    )


def test_a_witness_token_that_directly_supports_another_emitted_token_is_not_spare() -> None:
    # the witness' lone ``4.1`` supports the emitted ``4.1``; nothing is left for ``-4.1``
    corroborated, still = corroborate_unsupported_numeric(
        C("4.1", "-4.1"), C("4.1"), C("4.1", "-4.1")
    )
    assert corroborated == C()
    assert still == C("-4.1")


def test_digits_absent_from_the_witness_stay_unsupported() -> None:
    corroborated, still = corroborate_unsupported_numeric(C("-4.1"), C("42", "3.5"), C("-4.1"))
    assert corroborated == C()
    assert still == C("-4.1")
    # a missing trailing digit is not a lost point
    assert corroborate_unsupported_numeric(C("12.0"), C("12"), C("12.0"))[0] == C()


def test_a_token_the_second_reading_lacks_stays_unsupported() -> None:
    corroborated, still = corroborate_unsupported_numeric(C("-4.1"), C("41"), C("-4.2", "4.1"))
    assert corroborated == C()
    assert still == C("-4.1")


def test_directly_supported_tokens_are_not_reported() -> None:
    corroborated, still = corroborate_unsupported_numeric(
        C("2.0", "-4.1"), C("2.0", "41"), C("2.0")
    )
    assert (corroborated, still) == (C(), C("-4.1"))


# --- verify_table_tokens ---------------------------------------------------------------------


def _bundle(numeric: Counter, content: frozenset[str] = frozenset({"corn", "soy"})):
    return SourceEvidenceBundle(
        numeric=numeric,
        content=content,
        has_content_evidence=True,
        witness_state="reading",
    )


def _tokens(text: str):
    tokens = collect_table_tokens(text)
    assert tokens is not None
    return tokens


TABLE = "| item | a | b |\n|---|---|---|\n| Corn | -4.1 | 2.0 |\n| Soy | 1.5 | 12 |\n"
WITNESS = C("41", "2.0", "1.5", "12")


def _second(engine: str, table: str = TABLE) -> SecondReading:
    return SecondReading(engine=engine, numeric=_tokens(table).numeric)


def test_no_second_reading_keeps_the_dispute() -> None:
    result = verify_table_tokens(_bundle(WITNESS), _tokens(TABLE), engine="gemini")
    assert (result.passed, result.disputed_values) == (False, ("-4.1",))
    assert result.corroborated_values == ()


def test_a_second_reading_by_another_engine_passes_and_is_named() -> None:
    result = verify_table_tokens(
        _bundle(WITNESS), _tokens(TABLE), second_reading=_second("qwen"), engine="gemini"
    )
    assert result.passed is True
    assert result.corroborated_values == ("-4.1",)
    assert result.corroborated_by == "qwen"


@pytest.mark.parametrize("second_engine", ["gemini", ""])
def test_a_same_engine_or_unnamed_second_reading_is_refused(second_engine: str) -> None:
    result = verify_table_tokens(
        _bundle(WITNESS),
        _tokens(TABLE),
        second_reading=_second(second_engine),
        engine="gemini",
    )
    assert result.passed is False
    assert result.corroborated_values == ()


def test_one_uncorroborated_token_keeps_the_whole_dispute() -> None:
    table = "| item | a | b |\n|---|---|---|\n| Corn | -4.1 | 2.0 |\n| Soy | 9.9 | 12 |\n"
    second = SecondReading(engine="qwen", numeric=_tokens(TABLE).numeric)  # no 9.9
    result = verify_table_tokens(
        _bundle(WITNESS), _tokens(table), second_reading=second, engine="gemini"
    )
    assert result.passed is False
    assert result.disputed_values == ("-4.1", "9.9")
    assert result.corroborated_values == ()


def test_a_label_dispute_is_never_corroborated() -> None:
    # every number supported by the witness; the label is not, and the table has no number
    # to anchor the #659 trade-off, so the label dispute stands whatever a second reading says
    alpha = "| item | note |\n|---|---|\n| Wheat | Barley |\n"
    bundle = _bundle(C(), content=frozenset({"corn"}))
    second = SecondReading(engine="qwen", numeric=Counter())
    result = verify_table_tokens(bundle, _tokens(alpha), second_reading=second, engine="gemini")
    assert result.passed is False
    assert "content labels unsupported" in result.reason
    # and a corroborated number does not launder an unsupported label: the label is still
    # reported as unverified exactly as without the second reading
    labelled = "| item | a |\n|---|---|\n| Wheat | -4.1 |\n"
    with_second = verify_table_tokens(
        _bundle(C("41")),
        _tokens(labelled),
        second_reading=SecondReading(engine="qwen", numeric=_tokens(labelled).numeric),
        engine="gemini",
    )
    assert with_second.passed is True
    assert with_second.content_unverified


# --- route_page, with the real witness judge ---------------------------------------------------


class _Accept:
    def assess(self, output, provider):
        return AcceptDecision(accept=True, reason="faithful")


def _p():
    return SimpleNamespace(
        own_layer_text="",
        own_layer_drop_reference="",
        own_layer_word_reference="",
        own_layer_figure_words="",
        is_born_digital=False,
        prior_ocr_layer_suspect=False,
        has_unrecovered_symbol_glyphs=False,
        table_ladder_disposition=None,
        control_bytes_replaced=None,
        row_binding={},
        hidden_footnote_words="",
    )


def _route(earlier_cell: str, ladder: list, texts: dict[str, str] | None = None):
    """Run the ladder against a blank page whose injected witness reads ``41`` for ``-4.1``.

    The earlier (free) rung emits ``earlier_cell`` in that cell; every later rung emits ``-4.1``.
    Only ``earlier_cell`` differs between arms.
    """
    doc = fitz.open()
    page = doc.new_page()
    events: list = []
    judge = SourceEvidenceTableJudge(
        _Accept(),
        lambda n: page,
        events.append,
        ocr_image_fn=lambda pix: "Corn 41 2.0 Soy 1.5 12",
        native_trusted=lambda n: False,
        layer_is_evidence=lambda n: False,
    )

    def run_provider(prof, page_num: int) -> PageOutput:
        cell = earlier_cell if prof is ladder[0] else "-4.1"
        text = (texts or {}).get(prof.engine.value) or TABLE.replace("-4.1", cell)
        return PageOutput(
            page_num=page_num, text=text, status=PageStatus.SUCCESS, engine=prof.engine.value
        )

    decision = route_page(1, ladder, run_provider, judge)
    return decision, events


def test_the_paid_reading_is_accepted_only_when_the_earlier_reading_carries_the_token() -> None:
    ladder = [PROFILE_QWEN_LOCAL, PROFILE_GEMINI]
    carries, carries_events = _route("-4.1", ladder)
    lacks, lacks_events = _route("-4.2", ladder)

    # the free reading is disputed in both arms, the witness is identical in both
    assert [a.accepted for a in carries.attempts] == [False, True]
    assert [a.accepted for a in lacks.attempts] == [False, False]
    assert carries.accepted is True and lacks.accepted is False
    assert all("numeric tokens unsupported" in a.reason for a in lacks.attempts)

    # the accepted arm records the corroboration, twice over
    out = carries.final_output
    assert out.engine == "gemini" and out.paid_after_witness_dispute is True
    assert out.table_witness_corroboration == {
        "tokens": ["-4.1"],
        "corroborated_by": "qwen",
        "engine": "gemini",
    }
    [event] = [e for e in carries_events if e.kind == CORROBORATED_KIND]
    assert event.data == out.table_witness_corroboration and event.engine == "gemini"
    assert not [e for e in lacks_events if e.kind == CORROBORATED_KIND]
    assert lacks.attempts[1].output.table_witness_corroboration is None

    # the second reading is a one-call hand-over, never part of a page's record
    for att in carries.attempts:
        assert att.output.witness_second_reading is None
        assert "witness_second_reading" not in att.output.to_dict()


def test_the_accepted_paid_reading_still_ships_warning_with_the_reading_dispute_clause() -> None:
    from socr.core.manifest import apply_verdict_marks
    from socr.core.marks import Check, CheckResult
    from socr.core.verdict import PAID_AFTER_DISPUTE_REASON

    carries, _ = _route("-4.1", [PROFILE_QWEN_LOCAL, PROFILE_GEMINI])
    shipped, verdict = apply_verdict_marks(carries.final_output, _p(), 1)
    assert shipped.status is PageStatus.WARNING
    assert shipped.failure_mode is not FailureMode.NONE
    [clause] = [c for c in verdict.clauses if c.check is Check.READING_DISPUTE]
    assert (clause.result, clause.reason) == (CheckResult.UNKNOWN, PAID_AFTER_DISPUTE_REASON)
    assert "reading_dispute=UNKNOWN" in shipped.text
    # the corroboration survives the sidecar round trip
    again = PageOutput.from_dict(shipped.to_dict())
    assert again.table_witness_corroboration == carries.final_output.table_witness_corroboration
    # omit-when-unset: a page without corroboration keeps the bytes it always had
    assert "table_witness_corroboration" not in PageOutput(page_num=1, text="x").to_dict()


def test_a_free_reading_is_never_rescued() -> None:
    # two FREE rungs: the first is disputed, the second repeats the same token. Whatever the
    # earlier reading carries, a reading that is not a paid one after a dispute gets no second
    # reading to corroborate with.
    ladder = [PROFILE_QWEN_LOCAL, PROFILE_MARKER]
    decision, events = _route("-4.1", ladder, texts={"marker": TABLE})
    assert decision.accepted is False
    marker = [a for a in decision.attempts if a.engine is PROFILE_MARKER.engine][0]
    assert marker.output.paid_after_witness_dispute is False
    assert marker.output.table_witness_corroboration is None
    assert not [e for e in events if e.kind == CORROBORATED_KIND]


def test_a_paid_reading_after_a_rejection_that_is_not_a_dispute_is_not_rescued() -> None:
    # the earlier rung was refused for a reason that is not a witness dispute, so the paid
    # rung is not "after a dispute" and gets no second reading
    doc = fitz.open()
    page = doc.new_page()
    witness = SourceEvidenceTableJudge(
        _Accept(),
        lambda n: page,
        None,
        ocr_image_fn=lambda pix: "Corn 41 2.0 Soy 1.5 12",
        native_trusted=lambda n: False,
        layer_is_evidence=lambda n: False,
    )

    class _First:
        def assess(self, output, provider):
            if provider is PROFILE_QWEN_LOCAL:
                return AcceptDecision(accept=False, reason="min_words: too short")
            return witness.assess(output, provider)

    def run_provider(prof, page_num: int) -> PageOutput:
        return PageOutput(
            page_num=page_num, text=TABLE, status=PageStatus.SUCCESS, engine=prof.engine.value
        )

    decision = route_page(1, [PROFILE_QWEN_LOCAL, PROFILE_GEMINI], run_provider, _First())
    assert decision.accepted is False
    assert decision.attempts[1].output.paid_after_witness_dispute is False
    assert decision.attempts[1].output.table_witness_corroboration is None
