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


# --- review round: direction, pixel-only witness, engines, which reading, re-assessment ------


@pytest.mark.parametrize(
    ("emitted", "witness"),
    [
        ("-7", "-0.7"),  # ``-.7`` read correctly by the witness, shipped as ``-7`` by the models
        ("4.1", "-4.1"),  # the witness holds a sign the emitted token lacks
        ("41", "4.1"),  # the witness holds a point the emitted token lacks
        ("1234", "1.234"),
        ("-41", "-4.1"),
    ],
)
def test_a_witness_that_is_more_complete_than_the_emitted_token_never_corroborates(
    emitted: str, witness: str
) -> None:
    corroborated, still = corroborate_unsupported_numeric(C(emitted), C(witness), C(emitted))
    assert corroborated == C()
    assert still == C(emitted)
    # the DIFFERENCE is the direction: swap the two tokens and the same digits corroborate
    swapped, _ = corroborate_unsupported_numeric(C(witness), C(emitted), C(witness))
    assert swapped == C(witness)


def test_the_page_that_prints_a_negative_fraction_is_not_waved_through() -> None:
    # printed ``-.7``; the witness read ``-0.7``; the models (both) lost the point: ``-7``
    table = "| item | a |\n|---|---|\n| Corn | -7 |\n| Soy | 2.0 |\n"
    witness = C("-0.7", "2.0")
    second = SecondReading(engine="qwen", numeric=_tokens(table).numeric)
    result = verify_table_tokens(
        _bundle(witness), _tokens(table), second_reading=second, engine="gemini"
    )
    assert result.passed is False
    assert result.disputed_values == ("-7",)
    assert result.corroborated_values == ()


def test_a_witness_token_goes_to_the_emitted_token_it_can_stand_for() -> None:
    # witness ``7`` can stand for either ``-7`` (nothing to add) or ``0.7`` (a point to add);
    # one witness token, two emitted tokens, only one is corroborated and none is double-counted
    corroborated, still = corroborate_unsupported_numeric(
        C("-0.7", "0.7"), C("7"), C("-0.7", "0.7")
    )
    assert sum(corroborated.values()) == 1 and sum(still.values()) == 1
    # with a second witness token both are
    corroborated, still = corroborate_unsupported_numeric(
        C("-0.7", "0.7"), C("7", "-7"), C("-0.7", "0.7")
    )
    assert corroborated == C("-0.7", "0.7") and still == C()


@pytest.mark.parametrize("engine", ["", "gemini"])
def test_both_engines_must_be_known_and_differ(engine: str) -> None:
    result = verify_table_tokens(
        _bundle(WITNESS), _tokens(TABLE), second_reading=_second("qwen"), engine=engine
    )
    assert result.passed is (engine == "gemini")
    assert bool(result.corroborated_values) is (engine == "gemini")


# the witness is a lossy raster only when no text layer is in the bundle


def _layer_page(text: str):
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 100), text, fontsize=11)
    return doc, page


def _verify_scanned(page, *, layer_is_evidence: bool):
    return __import__("socr.tables.source_evidence", fromlist=["x"]).verify_scanned_table(
        page,
        TABLE,
        ocr_image_fn=lambda pix: "Corn 41 2.0 Soy 1.5 12",
        native_trusted=False,
        layer_is_evidence=layer_is_evidence,
        second_reading=_second("qwen"),
        engine="gemini",
    )


def test_the_second_reading_stands_in_for_a_pixel_witness_not_for_a_text_layer() -> None:
    # the layer prints ``4.1`` (exact), the OCR crop reads ``41``; both readers say ``-4.1``
    doc, page = _layer_page("Corn 4.1 2.0 Soy 1.5 12")
    try:
        pixel = _verify_scanned(page, layer_is_evidence=False)  # layer excluded: pixels only
        layered = _verify_scanned(page, layer_is_evidence=True)  # layer in the bundle
    finally:
        doc.close()
    assert pixel.passed is True and pixel.corroborated_values == ("-4.1",)
    assert layered.passed is False and layered.corroborated_values == ()


# --- which earlier reading corroborates -----------------------------------------------------


def _route_cells(cells: dict[str, str], ladder: list):
    """Like ``_route`` but every rung's cell is given by engine: ``-4.1`` is the true value."""
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
        return PageOutput(
            page_num=page_num,
            text=TABLE.replace("-4.1", cells[prof.engine.value]),
            status=PageStatus.SUCCESS,
            engine=prof.engine.value,
        )

    return route_page(1, ladder, run_provider, judge), events


def test_the_most_recent_disputed_reading_by_another_engine_is_the_one_consulted() -> None:
    from socr.core.providers import PROFILE_GLM

    ladder = [PROFILE_QWEN_LOCAL, PROFILE_GLM, PROFILE_GEMINI]
    # arm A: the older reading carries the token, the most recent one does not
    older, _ = _route_cells({"qwen": "-4.1", "glm": "-4.2", "gemini": "-4.1"}, ladder)
    # arm B: the most recent reading carries it, the older one does not
    recent, _ = _route_cells({"qwen": "-4.2", "glm": "-4.1", "gemini": "-4.1"}, ladder)
    assert older.final_output.table_witness_corroboration is None
    assert recent.final_output.table_witness_corroboration is not None
    assert recent.final_output.table_witness_corroboration["corroborated_by"] == "glm"
    assert recent.accepted is True and older.accepted is False


def test_a_refuted_paid_reading_can_corroborate_a_later_paid_rung() -> None:
    from socr.core.providers import PROFILE_MISTRAL

    ladder = [PROFILE_QWEN_LOCAL, PROFILE_MISTRAL, PROFILE_GEMINI]
    carries, _ = _route_cells({"qwen": "-4.2", "mistral": "-4.1", "gemini": "-4.1"}, ladder)
    lacks, _ = _route_cells({"qwen": "-4.2", "mistral": "-4.2", "gemini": "-4.1"}, ladder)
    # the mistral reading is itself refuted (the witness holds ``41``, mistral emits ``-4.1``
    # in arm one: corroborated by nobody, qwen read ``-4.2``), and it then stands for gemini
    assert carries.final_output.table_witness_corroboration is not None
    assert carries.final_output.table_witness_corroboration["corroborated_by"] == "mistral"
    assert lacks.final_output.table_witness_corroboration is None


# --- a later re-assessment has no second reading --------------------------------------------


def test_a_reassessment_without_the_second_reading_refuses_the_corroborated_reading() -> None:
    # The #1013 kept-candidate re-judge and the crop-patch re-judge build a fresh PageOutput,
    # which never carries ``witness_second_reading`` (it is a one-call hand-over). The same
    # bytes that passed through corroboration are therefore refused there: fail closed, the
    # run that resumes diverges only by re-running the ladder.
    carries, _ = _route("-4.1", [PROFILE_QWEN_LOCAL, PROFILE_GEMINI])
    shipped = carries.final_output
    assert shipped.table_witness_corroboration is not None

    doc = fitz.open()
    page = doc.new_page()
    judge = SourceEvidenceTableJudge(
        _Accept(),
        lambda n: page,
        None,
        ocr_image_fn=lambda pix: "Corn 41 2.0 Soy 1.5 12",
        native_trusted=lambda n: False,
        layer_is_evidence=lambda n: False,
    )
    fresh = PageOutput(
        page_num=1,
        text=shipped.text,
        status=PageStatus.SUCCESS,
        engine=shipped.engine,
        paid_after_witness_dispute=True,
    )
    assert judge.assess(fresh, PROFILE_GEMINI).accept is False
    # and the difference is only the hand-over: with it the same bytes pass
    fresh.witness_second_reading = ("qwen", TABLE)
    assert judge.assess(fresh, PROFILE_GEMINI).accept is True
