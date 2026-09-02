"""P4-R t1: arbitrary-text numeric tokenizer and one-way region presence verdict.

Covers `docs/log/2026-09-02_p4r-equation-lane.md` ruling 4 (numeric presence is a
rejection guard, not an acceptance contract) via two new symbols this ticket adds
to ``socr.tables.escalation_canary``:

  - ``text_value_tokens(text) -> Counter``: the whole-text numeric tokenizer,
    factored out of ``native_text_value_counts`` so it also works on arbitrary
    (non-native) candidate text, including LaTeX.
  - ``region_presence_verdict(native_text, candidate_text, *, encoding_suspect,
    corrupt_math) -> PresenceVerdict``: one-way multiset containment
    (candidate ⊆ oracle), never PRESENCE_LOST, ABSTAIN (UNVERIFIABLE) on
    encoding-suspect/corrupt-math/empty-oracle pages.

These tests are written against the acceptance criteria in t1 and will fail
until the symbols exist — that is expected; this file pins the contract before
implementation (t1 is not yet on this branch).
"""

from __future__ import annotations

from collections import Counter

import pytest

from socr.tables.escalation_canary import (
    PRESENCE_INVENTED,
    PRESENCE_LOST,
    PRESENCE_OK,
    PRESENCE_UNVERIFIABLE,
    native_text_value_counts,
)

# Cold review round 1, finding 6: these are LANDED acceptance tests, so they
# import their subject directly. The file used to fall back to a skipif when the
# symbols were missing, which turned "the required implementation was deleted"
# into a green skip.
from socr.tables.escalation_canary import region_presence_verdict, text_value_tokens


# ── text_value_tokens: shared tokenizer, arbitrary text ──────────────────────


class TestTextValueTokens:
    def test_matches_native_text_value_counts_on_plain_prose(self):
        """One regex definition: both entry points agree on plain text."""
        text = "Revenue grew 12.5% to $340.7 million, up from 302.1 in FY23."
        assert text_value_tokens(text) == native_text_value_counts(text)

    def test_leading_decimal_stays_intact(self):
        tokens = text_value_tokens("the discount rate is .75 this quarter")
        assert tokens["0.75"] == 1
        assert "75" not in tokens

    def test_unicode_minus_normalizes(self):
        tokens = text_value_tokens("the residual is −3.2")
        assert any(k.startswith("-") for k in tokens) or "-3.2" in tokens

    def test_multiplicity_preserved(self):
        tokens = text_value_tokens("1.02, then 1.02 again, then 1.02 once more")
        assert tokens["1.02"] == 3

    def test_alphabetic_control_words_do_not_match(self):
        tokens = text_value_tokens(r"\alpha \beta \gamma \sum \int")
        assert sum(tokens.values()) == 0

    def test_latex_tag_brace_content_is_a_candidate_token(self):
        tokens = text_value_tokens(r"E = mc^2 \tag{3}")
        assert tokens["3"] == 1
        # Cold review round 1, finding 2: the exponent is a number the model
        # wrote, so it is a candidate token too. The earlier version of this
        # file asserted the opposite, and that assertion is what licensed the
        # `(?<!\^)` lookbehind an invented exponent walked straight through.
        assert tokens["2"] == 1

    def test_an_exponent_is_a_candidate_token(self):
        assert text_value_tokens("x^9") == Counter({"9": 1})

    def test_a_multi_digit_exponent_is_not_truncated(self):
        """The old lookbehind skipped only the FIRST digit, so `x^999` became 99."""
        assert text_value_tokens("x^999") == Counter({"999": 1})

    def test_an_invented_exponent_is_rejected(self):
        verdict = region_presence_verdict("the page mentions 2 and nothing else", "x^9")
        assert verdict.status == PRESENCE_INVENTED
        assert "9" in verdict.invented

    def test_latex_fraction_brace_contents_are_candidate_tokens(self):
        tokens = text_value_tokens(r"\frac{1}{2}")
        assert tokens["1"] == 1
        assert tokens["2"] == 1

    def test_empty_text_yields_empty_counter(self):
        assert text_value_tokens("") == Counter()


# ── region_presence_verdict: one-way containment, never LOST ─────────────────


class TestRegionPresenceVerdict:
    def test_empty_native_oracle_is_unverifiable(self):
        verdict = region_presence_verdict("", r"x = 3")
        assert verdict.status == PRESENCE_UNVERIFIABLE

    def test_encoding_suspect_abstains_even_with_invented_number(self):
        verdict = region_presence_verdict(
            "the coefficient is 4.2", r"y = 999.9", encoding_suspect=True
        )
        assert verdict.status == PRESENCE_UNVERIFIABLE

    def test_corrupt_math_abstains_even_with_invented_number(self):
        verdict = region_presence_verdict("the coefficient is 4.2", r"y = 999.9", corrupt_math=True)
        assert verdict.status == PRESENCE_UNVERIFIABLE

    def test_encoding_and_corrupt_math_precedence_over_ok_candidate(self):
        """ABSTAIN precedence: a clean-looking candidate still abstains."""
        verdict = region_presence_verdict(
            "the coefficient is 4.2", r"y = 4.2", encoding_suspect=True, corrupt_math=True
        )
        assert verdict.status == PRESENCE_UNVERIFIABLE

    def test_subset_of_oracle_is_ok(self):
        native = "revenue was 340.7 and costs were 302.1 in the reporting period"
        verdict = region_presence_verdict(native, r"x = 340.7")
        assert verdict.status == PRESENCE_OK

    def test_invented_value_rejects(self):
        native = "revenue was 340.7 in the period"
        verdict = region_presence_verdict(native, r"x = 999.9")
        assert verdict.status == PRESENCE_INVENTED
        assert "999.9" in verdict.invented

    def test_multiplicity_surplus_rejects(self):
        """Candidate repeats a value more times than the oracle has it -> INVENTED."""
        native = "the value 1.02 appears once"
        verdict = region_presence_verdict(native, r"1.02 = 1.02")
        assert verdict.status == PRESENCE_INVENTED

    def test_never_returns_lost(self):
        """A region legitimately covers only a subset of page numbers."""
        native = "values are 1, 2, 3, 4, 5 across the table"
        verdict = region_presence_verdict(native, r"x = 1")
        assert verdict.status != PRESENCE_LOST

    def test_leading_decimal_candidate_accepted(self):
        native = "the rate is .75 this quarter"
        verdict = region_presence_verdict(native, r"r = .75")
        assert verdict.status == PRESENCE_OK

    def test_unicode_minus_candidate_accepted(self):
        native = "the residual is −3.2 units"
        verdict = region_presence_verdict(native, "r = −3.2")
        assert verdict.status == PRESENCE_OK

    def test_latex_tag_number_checked_against_oracle(self):
        # Every number the model wrote is checked, the equation tag and the
        # exponent alike. The page must carry both for the reading to pass.
        native = "see equation 3 for the derivation of E = mc 2"
        verdict = region_presence_verdict(native, r"E = mc^2 \tag{3}")
        assert verdict.status == PRESENCE_OK

    def test_an_exponent_absent_from_the_page_rejects(self):
        """Cold review round 1, finding 2: this returned OK before the fix."""
        native = "see equation 3 for the derivation"
        verdict = region_presence_verdict(native, r"E = mc^2 \tag{3}")
        assert verdict.status == PRESENCE_INVENTED
        assert "2" in verdict.invented

    def test_latex_fraction_invented_denominator_rejects(self):
        native = "only the value 1 appears on this page"
        verdict = region_presence_verdict(native, r"\frac{1}{2}")
        assert verdict.status == PRESENCE_INVENTED
        assert "2" in verdict.invented

    def test_docstring_states_rejection_guard_not_acceptance_contract(self):
        doc = (region_presence_verdict.__doc__ or "").lower()
        assert "rejection" in doc
        assert "guard" in doc


# ── Non-regression: existing presence/table-token surface unchanged ──────────


class TestExistingSurfaceUnchanged:
    def test_native_text_value_counts_still_exported_and_working(self):
        assert native_text_value_counts("a value of 12.5 appears")["12.5"] == 1

    def test_table_value_tokens_still_importable(self):
        from socr.tables.escalation_canary import table_value_tokens

        assert callable(table_value_tokens)

    def test_presence_verdict_from_text_unchanged_shape(self):
        from socr.tables.escalation_canary import presence_verdict_from_text

        verdict = presence_verdict_from_text("value 1.0", "| 1.0 |\n|---|\n| 1.0 |")
        assert verdict.status in {
            PRESENCE_OK,
            PRESENCE_INVENTED,
            PRESENCE_LOST,
            PRESENCE_UNVERIFIABLE,
        }
