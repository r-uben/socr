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


class TestAdjudicate:
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
