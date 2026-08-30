"""GH-326: the presence gate — what the native text layer may still assert.

The original #326 wired ``binding.py`` as a SUCCESS gate. Two measurements
disqualified that:

- ``bind()`` fully-checks 0 of 13 real tables, and its "invented value" signal
  fires on pages where no model ran at all (#330).
- Native is the LEAST accurate of three readings — 8/13 rows exact against qwen's
  12/13 and gemini's 11/13, losing row labels where neither model did
  (``docs/log/2026-08-30_model-vs-native-table-rows.md``).

So native's positional assertions are inadmissible. What survives is presence: the
word layer can say a number is on the page or is not, because that claim needs no
rows, columns or labels.

These tests are hermetic — a fake page object supplies the word layer, no PDF is
read and no model runs.
"""

from __future__ import annotations

import pytest

from socr.tables.escalation_canary import (
    PRESENCE_INVENTED,
    PRESENCE_LOST,
    PRESENCE_OK,
    PRESENCE_UNVERIFIABLE,
    presence_verdict,
)


class _Row:
    def __init__(self, values):
        self.values = values


@pytest.fixture
def fake_page(monkeypatch):
    """Install a word layer for the page, without touching a PDF."""

    def _install(values):
        from socr.tables import escalation_canary as ec

        monkeypatch.setattr(ec, "native_rows_from_page", lambda page: [_Row(list(values))])
        return object()

    return _install


def _table(*rows: str) -> str:
    head = "| a | b |\n| --- | --- |\n"
    return head + "".join(f"| {r} |\n" for r in rows)


def test_a_candidate_whose_values_are_all_on_the_page_passes(fake_page):
    page = fake_page(["1.02", "1.30", "0.44"])
    v = presence_verdict(page, _table("1.02 | 1.30", "0.44 | "))
    assert v.status == PRESENCE_OK
    assert not v.blocks_success


def test_a_value_that_is_nowhere_on_the_page_blocks(fake_page):
    """The #270 failure: a number the model wrote that the page does not contain."""
    page = fake_page(["1.02", "1.30"])
    v = presence_verdict(page, _table("1.02 | 1.30", "9.99 | "))
    assert v.status == PRESENCE_INVENTED
    assert v.blocks_success
    assert "9.99" in v.invented


def test_a_substitution_is_caught_because_the_gate_counts(fake_page):
    """The decisive case: counts catch what set containment cannot.

    #270's failure is a substitution — one occurrence of a coefficient becomes
    two, or a value is overwritten by its neighbour. As a SET the candidate
    introduces nothing: {1.02, 1.30} is unchanged however many times 1.02 is
    written. Only multiplicity sees it.
    """
    page = fake_page(["1.02", "1.30"])
    candidate = _table("1.02 | 1.02")  # 1.30 overwritten by its neighbour

    v = presence_verdict(page, candidate)

    assert v.status == PRESENCE_INVENTED, "a set-based gate would report OK here"
    assert v.invented == ("1.02",)
    assert "1.30" in v.lost

    # Pin the difference explicitly: as sets, nothing was introduced.
    from socr.tables.escalation_canary import native_value_counts, table_value_tokens

    assert set(table_value_tokens(candidate)) <= set(native_value_counts(page))


def test_a_missing_value_flags_but_does_not_block(fake_page):
    """Asymmetric on purpose: loss may be a legitimate split, invention cannot."""
    page = fake_page(["1.02", "1.30", "0.44"])
    v = presence_verdict(page, _table("1.02 | 1.30"))
    assert v.status == PRESENCE_LOST
    assert "0.44" in v.lost
    assert not v.blocks_success


def test_a_damaged_text_layer_is_unverifiable_not_a_conviction(fake_page):
    """The corpus contains ⟨0.00⟩ arriving as `h0.00i`.

    A token can be absent from the word layer because the layer misdecoded it.
    Reporting that as invention convicts the model for the text layer's failure —
    which is the exact error this gate replaces.
    """
    page = fake_page(["1.02"])
    candidate = _table("1.02 | 9.99")

    guilty = presence_verdict(page, candidate, encoding_suspect=False)
    spared = presence_verdict(page, candidate, encoding_suspect=True)

    assert guilty.status == PRESENCE_INVENTED and guilty.blocks_success
    assert spared.status == PRESENCE_UNVERIFIABLE and not spared.blocks_success


def test_a_page_with_no_oracle_is_unverifiable(fake_page):
    """No evidence is not evidence of innocence, nor of guilt."""
    page = fake_page([])
    v = presence_verdict(page, _table("1.02 | 1.30"))
    assert v.status == PRESENCE_UNVERIFIABLE
    assert not v.blocks_success


def test_the_verdict_carries_no_positional_field():
    """The contract, enforced structurally.

    Native's row geometry is what the measurement disqualified. A row, column or
    label field on this verdict would reintroduce exactly that assertion, so its
    absence is part of the design rather than an oversight.
    """
    import dataclasses

    from socr.tables.escalation_canary import PresenceVerdict

    names = {f.name for f in dataclasses.fields(PresenceVerdict)}
    for forbidden in ("row", "rows", "column", "columns", "label", "labels", "cell", "cells"):
        assert forbidden not in names


# ---------------------------------------------------------------------------
# The composed acceptance gate: structure -> presence -> image judge, in cost
# order. The ordering is the router's cost premise, so it is pinned, not assumed.
# ---------------------------------------------------------------------------


class _Judge:
    """Records whether it was consulted, so cost order can be asserted."""

    def __init__(self, faithful=True, issues=(), boom=False):
        self.faithful, self.issues, self.boom = faithful, list(issues), boom
        self.calls = 0

    def judge(self, image_path, ocr_text):
        self.calls += 1
        if self.boom:
            raise RuntimeError("judge is down")
        from socr.judge.judge import JudgeVerdict

        return JudgeVerdict(faithful=self.faithful, issues=self.issues)


def test_a_clean_candidate_is_accepted_and_the_judge_confirms(fake_page):
    from socr.tables.escalation_canary import table_acceptance

    page = fake_page(["1.02", "1.30"])
    judge = _Judge(faithful=True)
    v = table_acceptance(page, _table("1.02 | 1.30"), image_judge=judge, page_image="x.png")
    assert v.accepted and v.witness == "image" and v.judged
    assert judge.calls == 1


def test_invention_blocks_before_the_paid_judge_is_ever_called(fake_page):
    """Cost order, pinned: a free check that decides must not spend a model call."""
    from socr.tables.escalation_canary import table_acceptance

    page = fake_page(["1.02"])
    judge = _Judge(faithful=True)
    v = table_acceptance(page, _table("1.02 | 9.99"), image_judge=judge, page_image="x.png")
    assert not v.accepted and v.witness == "presence"
    assert judge.calls == 0, "the image judge was paid for a decision already made for free"


def test_a_malformed_grid_blocks_before_presence_or_the_judge(fake_page):
    """Structure is first because it needs no reference and so cannot be poisoned."""
    from socr.tables.escalation_canary import table_acceptance

    page = fake_page(["1.02", "1.30"])
    judge = _Judge(faithful=True)
    ragged = "| a | b |\n| --- | --- |\n| 1.02 | 1.30 | 9.99 |\n"
    v = table_acceptance(page, ragged, image_judge=judge, page_image="x.png")
    assert not v.accepted and v.witness == "structure"
    assert judge.calls == 0


def test_the_judge_can_reject_what_the_free_checks_passed(fake_page):
    """Its whole purpose: the positional question the free checks cannot answer.

    Every value here IS on the page, so presence passes. Only an image judge can
    see that a value sits in the wrong cell.
    """
    from socr.tables.escalation_canary import table_acceptance

    page = fake_page(["1.02", "1.30"])
    judge = _Judge(faithful=False, issues=["1.30 is in the wrong column"])
    v = table_acceptance(page, _table("1.30 | 1.02"), image_judge=judge, page_image="x.png")
    assert not v.accepted and v.witness == "image" and v.judged
    assert "wrong column" in v.reason


def test_an_absent_judge_does_not_block(fake_page):
    """Otherwise the gate tests availability rather than quality."""
    from socr.tables.escalation_canary import table_acceptance

    page = fake_page(["1.02", "1.30"])
    v = table_acceptance(page, _table("1.02 | 1.30"))
    assert v.accepted and v.witness == "default" and not v.judged


def test_a_judge_that_errors_does_not_convict(fake_page):
    """A broken witness is an absent witness, never a guilty verdict."""
    from socr.tables.escalation_canary import table_acceptance

    page = fake_page(["1.02", "1.30"])
    judge = _Judge(boom=True)
    v = table_acceptance(page, _table("1.02 | 1.30"), image_judge=judge, page_image="x.png")
    assert v.accepted and not v.judged
    assert "unavailable" in v.reason


def test_rejection_demotes_rather_than_discards(fake_page):
    """The #322 disposition: a rejected candidate keeps its content."""
    from socr.tables.escalation_canary import table_acceptance

    page = fake_page(["1.02"])
    v = table_acceptance(page, _table("1.02 | 9.99"))
    assert not v.accepted and v.demote_only
