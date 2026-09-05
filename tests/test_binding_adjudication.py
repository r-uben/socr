"""GH-367: pure disproof rules for a bind() contradiction.

Helper-unit coverage of ``tables.adjudication`` — what counts as disproof,
and what does not. The load-bearing gate/process tests live in
``test_gh367_adjudication_lift.py``; a green helper suite is not a gate.
"""

from __future__ import annotations

from socr.tables.adjudication import (
    ContradictionItem,
    adjudicate,
    items_from_binding,
    markdown_sha256,
    prior_lift_applies,
    token_is_encoding_garbage,
    tokens_agree,
)
from socr.tables.binding import BindingResult, ContradictedCell, RowLabelContradiction


def _cell(**kwargs) -> ContradictionItem:
    return ContradictionItem(kind="cell", col_path=("OLS",), **kwargs)


def _label(**kwargs) -> ContradictionItem:
    # Rule tests supply an independently established address; caller tests
    # exercise the geometry that establishes it.
    kwargs.setdefault("cell_bbox", kwargs.get("native_bbox"))
    return ContradictionItem(kind="row_label", col_path=(), **kwargs)


class TestParseTranscribeOutput:
    def test_token_empty_or_missing_is_none(self) -> None:
        from socr.judge.cell_transcribe import parse_transcribe_output

        assert parse_transcribe_output('{"token":"RowA"}') == "RowA"
        assert parse_transcribe_output('{"token":"  RowA  "}') == "RowA"
        assert parse_transcribe_output('{"token":""}') is None
        assert parse_transcribe_output("not json") is None
        assert parse_transcribe_output('{"verdict":"PASS"}') is None


class TestEncodingGarbage:
    def test_pua_replacement_control_are_garbage_row_a_is_not(self) -> None:
        assert token_is_encoding_garbage("R\ufffd")
        assert token_is_encoding_garbage("\uf8ff")
        assert token_is_encoding_garbage("A\x00B")
        assert not token_is_encoding_garbage("RowA")
        assert not token_is_encoding_garbage("100")
        assert not token_is_encoding_garbage("LoIic6")


class TestTokensAgree:
    def test_numeric_and_label_use_bind_normalizers(self) -> None:
        assert tokens_agree("100", "100", kind="cell")
        assert tokens_agree("RowA", "rowa", kind="row_label")
        assert not tokens_agree("RowA", "RowB", kind="row_label")
        assert not tokens_agree("100", "200", kind="cell")

    def test_inline_math_wrapped_cell_agrees_with_plain_value(self) -> None:
        """GH-582: a VLM cell typeset as inline math must be disprovable by
        the raster transcriber, not held forever because the wrap defeats
        ``is_numeric_token`` on the model side of the comparison."""
        assert tokens_agree("$-0.06$", "−0.06", kind="cell")
        # A genuinely different value stays a disagreement even wrapped.
        assert not tokens_agree("$-0.06$", "−0.60", kind="cell")
        # A malformed doubled delimiter is not a balanced whole-token wrap;
        # it must not unwrap to "$43" and fall into the currency-prefix
        # strip as a false numeric match (GH-582 round-2 review).
        assert not tokens_agree("$$43$", "43", kind="cell")

    def test_inline_math_wrapped_label_agrees_with_plain_label(self) -> None:
        """GH-582: ``\\text{}``/``^`` math notation around a row label must
        fold to the same key as its plain-text rendering."""
        assert tokens_agree("Adjusted $\\text{R}^2$", "Adjusted R2", kind="row_label")
        assert not tokens_agree("Adjusted $\\text{R}^2$", "Constant", kind="row_label")

    def test_sibling_latex_presentation_labels_agree(self) -> None:
        """GH-585: Greek-letter commands and alphabetic word commands are
        sibling presentation classes to the GH-582 wrap, and must agree the
        same way once mapped."""
        assert tokens_agree("∆Slope (3m)", r"$\Delta \text{ Slope (3m)}$", kind="row_label")
        assert tokens_agree(
            "∆log Comm. price (3m)",
            r"$\Delta \log \text{ Comm. price (3m)}$",
            kind="row_label",
        )
        # A genuine text difference (missing "500 (3m)") still disagrees.
        assert not tokens_agree(
            "∆log S&P",
            r"$\Delta \log \text{ S\&P 500 (3m)}$",
            kind="row_label",
        )
        # A bare symbolic label still folds to an empty key on both sides —
        # tokens_agree's empty-key refusal (not a false match) still applies.
        assert not tokens_agree("β", r"$\beta$", kind="row_label")

    def test_distinct_greek_letters_do_not_falsely_agree(self) -> None:
        """GH-585 review round 2: before the Unicode transliteration step,
        every Greek letter erased identically under ``normalize_label``'s
        ASCII-only filter, so two DIFFERENT Greek-letter labels with the
        same trailing text falsely agreed."""
        assert not tokens_agree("α Coefficient", r"$\beta$ Coefficient", kind="row_label")
        assert tokens_agree("α Coefficient", r"$\alpha$ Coefficient", kind="row_label")

    def test_greek_variant_command_does_not_agree_with_its_base_letter(self) -> None:
        """GH-585 review round 2: ``\\varepsilon`` is not established to be
        the same symbol as ``\\epsilon`` in this corpus — they must stay
        distinct, unlike the U+0394/U+2206 Delta pair which really is the
        same printed glyph."""
        assert not tokens_agree(r"$\epsilon$ x", r"$\varepsilon$ x", kind="row_label")

    def test_unsupported_word_command_does_not_earn_an_agreement(self) -> None:
        """GH-585 review round 3: retaining the backslash at the helper is
        not yet evidence that it survives the actual compare — ``\\logx``
        is not a real LaTeX macro and must not agree with either ``log`` or
        ``logx`` through ``tokens_agree``."""
        assert not tokens_agree(r"\logx y", "log y", kind="row_label")
        assert not tokens_agree(r"\logx y", "logx y", kind="row_label")
        assert tokens_agree(r"\log y", "log y", kind="row_label")

    def test_sentinel_strings_from_earlier_rounds_do_not_collide_with_literal_prose(
        self,
    ) -> None:
        """GH-585 review round 4: rounds 2/3 represented a Greek letter or an
        unsupported command as an ordinary STRING folded into
        ``normalize_label``'s flat key, so a label that happened to spell
        that sentinel text out in prose falsely agreed with the real Greek
        letter/unsupported command. ``label_key``'s structured tuple can
        never collide with prose this way — a ``("greek", ...)``/``("cmd",
        ...)`` tag is a different tuple shape from ``("lit", ...)``
        regardless of what the literal text says."""
        assert not tokens_agree("α Coefficient", "greekalpha Coefficient", kind="row_label")
        assert not tokens_agree(r"\logx y", "unmappedlogx y", kind="row_label")

    def test_variant_unicode_glyph_agrees_with_its_variant_command(self) -> None:
        """GH-585 review round 4: a native PDF's text layer can emit the
        variant-glyph codepoint directly (``ϑ`` U+03D1) instead of the base
        letter; it must agree with its ``\\vartheta`` command counterpart,
        not the base ``\\theta``/``θ`` letter."""
        assert tokens_agree("ϑ Coefficient", r"$\vartheta$ Coefficient", kind="row_label")
        assert not tokens_agree("ϑ Coefficient", r"$\theta$ Coefficient", kind="row_label")

    def test_internal_chunk_before_a_greek_token_keeps_its_trailing_digit(self) -> None:
        """GH-585 review round 5: ``normalize_label``'s trailing-footnote-
        suffix rule is a whole-label rule. Applied to an INTERNAL literal
        chunk of ``label_key`` -- one that ends where a Greek token follows,
        not where the label itself ends -- it discarded a real digit and
        collapsed ``"Model1 α"``/``"Model2 α"`` into the same key."""
        assert not tokens_agree(r"Model1 $\alpha$", r"Model2 $\alpha$", kind="row_label")

    def test_trailing_footnote_digit_on_the_labels_own_end_still_folds(self) -> None:
        """GH-585 review round 5: the suffix rule must still apply to the
        FINAL chunk, the one reaching the label's true end, so a genuine
        trailing footnote digit keeps agreeing exactly as before the fix."""
        assert tokens_agree("Coefficient1", "Coefficient", kind="row_label")

    def test_internal_chunk_before_a_greek_token_keeps_its_footnote_marker_digit(
        self,
    ) -> None:
        """GH-585 review round 6: round 5 fixed the bare-digit suffix rule
        but the end-anchored footnote-MARKER regex (``<sup>1</sup>``, ``$^1$``,
        unicode superscripts) also ran on internal chunks, so
        ``Model<sup>1</sup> α``/``Model<sup>2</sup> α`` still collapsed."""
        assert not tokens_agree(
            r"Model<sup>1</sup> $\alpha$", r"Model<sup>2</sup> $\alpha$", kind="row_label"
        )

    def test_trailing_footnote_marker_on_the_labels_own_end_still_folds(self) -> None:
        """GH-585 review round 6: a genuine footnote marker on the label's
        own end (the final chunk) still folds away exactly as before."""
        assert tokens_agree("Coefficient<sup>1</sup>", "Coefficient", kind="row_label")

    def test_bare_greek_symbol_with_a_trailing_digit_is_not_treated_as_a_footnote(
        self,
    ) -> None:
        """GH-585 review round 6: the seat explicitly wants ``α1`` and ``α``
        to stay distinct rather than have the trailing digit read as a
        footnote on a bare symbol -- ``α1`` is not a lone Greek token, so it
        is not bare-symbolic either; it simply must not agree with ``α``."""
        assert not tokens_agree("α1", "α", kind="row_label")


class TestAdjudicate:
    def test_all_abstained_is_held_and_never_transcribes_native_bbox(self) -> None:
        from unittest.mock import Mock

        transcribe = Mock(return_value="RowB")
        item = _label(
            row_path=("RowA",),
            native_token="RowA",
            model_token="RowB",
            native_bbox=(1, 2, 3, 4),
            cell_bbox=None,
            abstain_reason="no column edge",
        )
        record = adjudicate((item,), markdown="md", transcribe=transcribe)
        assert record.status == "held"
        assert record.items[0].outcome == "abstained"
        assert record.items[0].disproof is None
        transcribe.assert_not_called()
        saved = record.to_dict()["items"][0]
        assert saved["cell_bbox"] is None
        assert saved["address_source"] is None
        assert saved["abstain_reason"] == "no column edge"
        assert tuple(saved["native_bbox"]) == (1, 2, 3, 4)

    def test_geometry_address_is_used_instead_of_native_bbox(self) -> None:
        from unittest.mock import Mock

        cell_bbox = (0, 0, 10, 10)
        transcribe = Mock(return_value="RowB")
        item = _label(
            row_path=("RowA",),
            native_token="RowA",
            model_token="RowB",
            native_bbox=(0, 0, 5, 10),
            cell_bbox=cell_bbox,
            address_source="geometry",
        )
        record = adjudicate((item,), markdown="md", transcribe=transcribe)
        assert record.status == "lifted"
        transcribe.assert_called_once_with(cell_bbox)
        assert record.to_dict()["items"][0]["address_source"] == "geometry"

    def test_prior_lift_cannot_overrule_current_abstention(self) -> None:
        from dataclasses import replace
        from unittest.mock import Mock

        item = _label(
            row_path=("RowA",),
            native_token="RowA",
            model_token="RowB",
            native_bbox=(0, 0, 1, 1),
        )
        first = adjudicate((item,), markdown="md", transcribe=lambda _: "RowB")
        assert first.status == "lifted"
        abstaining = replace(item, cell_bbox=None, abstain_reason="no column edge")
        transcribe = Mock(return_value="RowB")
        second = adjudicate(
            (abstaining,), markdown="md", prior=first.to_dict(), transcribe=transcribe
        )
        assert second.status == "held"
        assert second.items[0].outcome == "abstained"
        transcribe.assert_not_called()

    def test_prior_requires_matching_geometry_and_method(self) -> None:
        import copy
        from unittest.mock import Mock

        item = _label(
            row_path=("RowA",),
            native_token="RowA",
            model_token="RowB",
            cell_bbox=(0, 0, 10, 10),
            address_source="ordinal geometry",
        )
        matching = adjudicate((item,), markdown="md", transcribe=lambda _: "RowB").to_dict()
        legacy = copy.deepcopy(matching)
        legacy["signatures"] = [list(item.signature())]
        moved = copy.deepcopy(matching)
        moved["signatures"][0][-1] = moved["signatures"][0][-1].replace("10, 10", "20, 20")
        old_method = copy.deepcopy(matching)
        old_method["signatures"][0][-1] = old_method["signatures"][0][-1].replace(
            "geometry_cell_transcription", "native_crop_transcription"
        )
        changed_source = copy.deepcopy(matching)
        changed_source["signatures"][0][-1] = changed_source["signatures"][0][-1].replace(
            "ordinal geometry", "native pointer"
        )
        for prior, calls in (
            (legacy, 1),
            (moved, 1),
            (old_method, 1),
            (changed_source, 1),
            (matching, 0),
        ):
            transcribe = Mock(return_value="RowB")
            result = adjudicate((item,), markdown="md", prior=prior, transcribe=transcribe)
            assert result.status == "lifted"
            assert transcribe.call_count == calls
            if calls:
                transcribe.assert_called_once_with(item.cell_bbox)
            else:
                # Reusing evidence must not erase its original method.
                assert prior_lift_applies(result.to_dict(), "md", (item,))

    def test_encoding_garbage_disproves_without_transcriber(self) -> None:
        items = (
            _label(
                row_path=("R\ufffd",),
                native_token="R\ufffd",
                model_token="RowA",
                native_bbox=(1, 1, 2, 2),
            ),
        )
        record = adjudicate(items, markdown="md", transcribe=None)
        assert record.status == "lifted"
        assert record.items[0].disproof == "encoding_garbage"

    def test_well_formed_without_transcriber_is_held(self) -> None:
        items = (
            _label(
                row_path=("RowA",),
                native_token="RowA",
                model_token="RowB",
                native_bbox=(1, 1, 2, 2),
            ),
        )
        record = adjudicate(items, markdown="md", transcribe=None)
        assert record.status == "held"
        assert record.items[0].disproof is None

    def test_transcription_matching_markdown_not_native_lifts(self) -> None:
        bbox = (0.0, 0.0, 10.0, 10.0)
        items = (
            _label(
                row_path=("RowA",),
                native_token="RowA",
                model_token="RowB",
                native_bbox=bbox,
            ),
        )
        record = adjudicate(items, markdown="md", transcribe=lambda _b: "RowB")
        assert record.status == "lifted"
        assert record.items[0].disproof == "raster_transcription"

    def test_transcription_matching_native_does_not_lift(self) -> None:
        bbox = (0.0, 0.0, 10.0, 10.0)
        items = (
            _label(
                row_path=("RowA",),
                native_token="RowA",
                model_token="RowB",
                native_bbox=bbox,
            ),
        )
        record = adjudicate(items, markdown="md", transcribe=lambda _b: "RowA")
        assert record.status == "held"

    def test_partial_disproof_lifts_nothing(self) -> None:
        items = (
            _label(
                row_path=("RowA",),
                native_token="RowA",
                model_token="RowB",
                native_bbox=(0, 0, 1, 1),
            ),
            _label(
                row_path=("RowB",),
                native_token="RowB",
                model_token="RowA",
                native_bbox=(0, 1, 1, 2),
            ),
        )
        tokens = iter(["RowB", "RowB"])
        record = adjudicate(items, markdown="md", transcribe=lambda _b: next(tokens))
        assert record.status == "held"
        assert record.items[0].disproof == "raster_transcription"
        assert record.items[1].disproof is None

    def test_ordinary_pass_shaped_transcriber_returning_unrelated_token_is_held(
        self,
    ) -> None:
        items = (
            _label(
                row_path=("RowA",),
                native_token="RowA",
                model_token="RowB",
                native_bbox=(0, 0, 1, 1),
            ),
        )
        record = adjudicate(items, markdown="md", transcribe=lambda _b: "PASS")
        assert record.status == "held"

    def test_prior_lift_matches_signatures_and_markdown(self) -> None:
        items = (
            _label(
                row_path=("RowA",),
                native_token="RowA",
                model_token="RowB",
                native_bbox=(0, 0, 1, 1),
            ),
        )
        first = adjudicate(items, markdown="md", transcribe=lambda _b: "RowB")
        assert first.status == "lifted"
        second = adjudicate(
            items,
            markdown="md",
            prior=first.to_dict(),
            transcribe=lambda _b: "RowA",
        )
        assert second.status == "lifted"
        assert second.items[0].disproof == "prior_lift"

    def test_prior_lift_does_not_apply_when_markdown_changes(self) -> None:
        items = (
            _label(
                row_path=("RowA",),
                native_token="RowA",
                model_token="RowB",
                native_bbox=(0, 0, 1, 1),
            ),
        )
        first = adjudicate(items, markdown="md", transcribe=lambda _b: "RowB")
        assert prior_lift_applies(first.to_dict(), "other", items) is False
        second = adjudicate(
            items,
            markdown="other",
            prior=first.to_dict(),
            transcribe=lambda _b: "RowA",
        )
        assert second.status == "held"

    def test_items_from_binding_cover_both_conviction_lists(self) -> None:
        result = BindingResult(
            contradicted_cells=[
                ContradictedCell(
                    row_path=("RowA",),
                    col_path=("OLS",),
                    native_token="100",
                    model_token="200",
                    native_bbox=(1, 2, 3, 4),
                )
            ],
            row_label_contradictions=[
                RowLabelContradiction(
                    row_path=("RowA",),
                    candidate_label="RowB",
                    native_bbox=(5, 6, 7, 8),
                )
            ],
        )
        items = items_from_binding(result)
        assert len(items) == 2
        assert items[0].kind == "cell"
        assert items[0].native_bbox == (1, 2, 3, 4)
        assert items[1].kind == "row_label"
        assert items[1].native_token == "RowA"
        assert markdown_sha256("a") != markdown_sha256("b")


class TestPriorLiftKeepsMultiplicity:
    """GH-390. Two contradictions can share a signature -- the same kind, paths
    and normalized tokens at two different loci. ``prior_lift_applies``
    originally compared SETS, which collapsed them, so a prior lift recording
    ONE signature matched a current set of TWO and cleared a contradiction
    nothing ever disproved. On resume that is silent: the second contradiction
    never gets looked at again.

    #388 shipped the sorted-sequence compare; every existing prior-lift helper
    uses a single item, so reverting that line stayed green. This is the pin.
    """

    def _twin_items(self) -> tuple:
        """Two items with identical signatures at different bboxes.

        The bbox is deliberately NOT part of the signature -- the signature is
        (kind, row_path, col_path, native, model) -- which is exactly why two
        real contradictions can collide.
        """
        return (
            _label(
                row_path=("RowA",),
                native_token="RowA",
                model_token="RowB",
                native_bbox=(0, 0, 1, 1),
            ),
            _label(
                row_path=("RowA",),
                native_token="RowA",
                model_token="RowB",
                native_bbox=(10, 10, 11, 11),
            ),
        )

    def test_the_two_items_really_do_share_a_signature(self) -> None:
        """Precondition. If signatures ever stop colliding, the test below
        would pass for the wrong reason and guard nothing."""
        first, second = self._twin_items()
        assert first.signature() == second.signature()
        assert first.native_bbox != second.native_bbox

    def test_a_one_signature_prior_does_not_clear_two_contradictions(self) -> None:
        items = self._twin_items()

        # A prior lift that only ever disproved ONE of them.
        one_only = adjudicate(items[:1], markdown="md", transcribe=lambda _b: "RowB")
        assert one_only.status == "lifted"
        assert len(one_only.to_dict()["signatures"]) == 1

        assert not prior_lift_applies(one_only.to_dict(), "md", items), (
            "a prior lift of one contradiction must not satisfy two -- a set "
            "compare collapses the duplicate and clears an undisproved item"
        )

    def test_adjudicate_re_examines_rather_than_reusing_the_short_prior(self) -> None:
        """Difference pin at the caller: with the one-signature prior, the two
        items must be judged afresh, so a transcriber that agrees with NATIVE
        leaves the table held rather than riding the stale lift."""
        items = self._twin_items()
        one_only = adjudicate(items[:1], markdown="md", transcribe=lambda _b: "RowB")

        reused = adjudicate(
            items, markdown="md", prior=one_only.to_dict(), transcribe=lambda _b: "RowA"
        )
        assert reused.status == "held"
        assert all(o.disproof != "prior_lift" for o in reused.items)

    def test_a_matching_two_signature_prior_still_lifts(self) -> None:
        """Control: multiplicity is the only thing that changed. A prior that
        genuinely covers both items must still be reused, or the fix would have
        broken resume instead of tightening it."""
        items = self._twin_items()
        both = adjudicate(items, markdown="md", transcribe=lambda _b: "RowB")
        assert both.status == "lifted"
        assert len(both.to_dict()["signatures"]) == 2

        assert prior_lift_applies(both.to_dict(), "md", items)
        again = adjudicate(items, markdown="md", prior=both.to_dict(), transcribe=lambda _b: "RowA")
        assert again.status == "lifted"
        assert all(o.disproof == "prior_lift" for o in again.items)


def test_frozen_prediction_gate_rejects_verdict_drift_and_unchecked_clears() -> None:
    import ast
    import json
    from dataclasses import replace
    from pathlib import Path

    import pytest

    from socr.benchmark.replay_binding import ReplayRow, assert_prediction

    prediction = json.loads(
        (Path(__file__).parent / "fixtures/replay_binding/controls/c2b_prediction.json").read_text()
    )
    grouped = {}
    for doc, page, table, kind, tokens, verdict, reason in prediction["verdicts"]:
        native, model = tokens.split("|", 1)
        item = ContradictionItem(
            kind=kind,
            row_path=(),
            col_path=(),
            native_token=native,
            model_token=model,
            cell_bbox=ast.literal_eval(reason.split("cell=")[1])
            if verdict == "addressed"
            else None,
            address_source=reason if verdict == "addressed" else None,
            abstain_reason=reason if verdict == "abstained" else None,
        )
        grouped.setdefault((doc, page, table), []).append(item)
    for key in prediction["cleared_tables"]:
        grouped[tuple(key)] = []
    rows = [
        ReplayRow(
            doc_slug=doc,
            page_num=page,
            table_id=table,
            recorded_status="held",
            recorded_item_count=len(items),
            fresh_item_count=len(items),
            multiset_match=True,
            added=(),
            removed=(),
            final_disposition="held",
            label_accuracy="unavailable",
            crop_coverage="unavailable",
            address_items=tuple(items),
        )
        for (doc, page, table), items in grouped.items()
    ]
    assert_prediction(rows, prediction)
    first = rows[0]
    wrong_reason = replace(first.address_items[0], abstain_reason="different failure")
    with pytest.raises(AssertionError, match="prediction mismatch"):
        assert_prediction(
            [replace(first, address_items=(wrong_reason, *first.address_items[1:])), *rows[1:]],
            prediction,
        )
    with pytest.raises(AssertionError, match="prediction mismatch"):
        assert_prediction(
            [replace(first, address_items=first.address_items[1:]), *rows[1:]], prediction
        )
    for changes in ({"unchecked": True}, {"unreplayable": True}, {"fresh_item_count": 1}):
        with pytest.raises(AssertionError, match="cleared table regressed"):
            assert_prediction([*rows[:-1], replace(rows[-1], **changes)], prediction)
