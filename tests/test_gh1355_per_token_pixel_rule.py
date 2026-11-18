"""#1355: the witness stand-in is judged PER TOKEN on a born-digital (mixed) bundle.

A page whose text layer holds a few words is not page-covering-raster, so the evidence bundle
merges the exact text layer with pixel OCR (``build_born_digital_evidence``). The earlier rule
switched corroboration off for that whole page. Now a disputed token may be corroborated when
the witness token standing behind it was READ FROM PIXELS, and never when the text layer holds
the same digits in any form (an exact source has already spoken). Every arm pins a DIFFERENCE
against the arm that passes, so a hardcoded outcome cannot satisfy it.

Hermetic: the OCR reader is injected, so no tesseract is needed.
"""

from __future__ import annotations

from collections import Counter

import pytest

fitz = pytest.importorskip("fitz")

from socr.tables.source_evidence import (  # noqa: E402
    SecondReading,
    SourceEvidenceBundle,
    _merge_source_bundles,
    collect_table_tokens,
    corroborate_unsupported_numeric,
    verify_scanned_table,
)

TABLE = "| item | a | b |\n|---|---|---|\n| Corn | -4.1 | 2.0 |\n| Soy | 1.5 | 12 |\n"
TABLE_POS = "| item | a | b |\n|---|---|---|\n| Corn | 4.1 | 2.0 |\n| Soy | 1.5 | 12 |\n"
TABLE_TWICE = "| item | a | b |\n|---|---|---|\n| Corn | -4.1 | -4.1 |\n| Soy | 1.5 | 12 |\n"
OCR_LOSSY = "Corn 41 2.0 Soy 1.5 12"


def _second(engine: str, table: str = TABLE) -> SecondReading:
    tokens = collect_table_tokens(table)
    assert tokens is not None
    return SecondReading(engine=engine, numeric=tokens.numeric)


def _run(
    *,
    layer: str = "Corn Soy item",
    ocr: str = OCR_LOSSY,
    table: str = TABLE,
    second: SecondReading | None = None,
    engine: str = "gemini",
    layer_is_evidence: bool = True,
):
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 100), layer, fontsize=11)
    try:
        return verify_scanned_table(
            page,
            table,
            ocr_image_fn=lambda pix: ocr,
            native_trusted=False,
            layer_is_evidence=layer_is_evidence,
            second_reading=second if second is not None else _second("qwen"),
            engine=engine,
        )
    finally:
        doc.close()


def test_mixed_bundle_pixel_witness_corroborates_a_lost_sign_and_point() -> None:
    ok = _run()
    assert ok.passed is True
    assert ok.corroborated_values == ("-4.1",)
    assert ok.corroborated_by == "qwen"
    # the only thing that differs is the second reading's presence
    no_second = _run(second=SecondReading(engine="", numeric=Counter()))
    assert no_second.passed is False and no_second.disputed_values == ("-4.1",)


@pytest.mark.parametrize("layer_token", ["4.1", "41", "-41"])
# (a layer holding the exact "-4.1" supports it directly: a pass, not a corroboration)
def test_a_text_layer_holding_the_same_digits_vetoes_the_token(layer_token: str) -> None:
    baseline = _run()
    vetoed = _run(layer=f"Corn Soy item {layer_token}")
    assert baseline.passed is True
    assert vetoed.passed is False
    assert vetoed.corroborated_values == ()
    assert "-4.1" in vetoed.disputed_values


def test_a_text_layer_holding_other_digits_does_not_veto() -> None:
    other = _run(layer="Corn Soy item 7.7 2.0")
    assert other.passed is True and other.corroborated_values == ("-4.1",)


def test_the_same_engine_as_the_second_reading_is_refused() -> None:
    assert _run(engine="qwen").passed is False
    assert _run(engine="gemini").passed is True


def test_a_second_reading_that_does_not_carry_the_token_is_refused() -> None:
    other = _second("qwen", TABLE.replace("-4.1", "-4.2"))
    assert _run(second=other).passed is False


def test_a_witness_more_complete_than_the_emitted_token_is_refused() -> None:
    # witness reads -4.1 (sign present), the reading under check emits 4.1
    refused = _run(
        ocr="Corn -4.1 2.0 Soy 1.5 12", table=TABLE_POS, second=_second("qwen", TABLE_POS)
    )
    assert refused.passed is False
    # control: the same witness supports an emitted -4.1 directly
    assert _run(ocr="Corn -4.1 2.0 Soy 1.5 12").passed is True


def test_one_pixel_token_backs_one_emitted_token() -> None:
    twice = _run(table=TABLE_TWICE, second=_second("qwen", TABLE_TWICE))
    assert twice.passed is False
    # control: two pixel tokens back two
    both = _run(table=TABLE_TWICE, second=_second("qwen", TABLE_TWICE), ocr="Corn 41 41 Soy 1.5 12")
    assert both.passed is True and both.corroborated_values == ("-4.1", "-4.1")


def test_pure_pixel_arm_is_unchanged() -> None:
    pixel = _run(layer_is_evidence=False)
    assert pixel.passed is True and pixel.corroborated_values == ("-4.1",)
    # the layer is not consulted on this arm, so a layer holding the digits does not veto
    assert _run(layer="Corn 4.1", layer_is_evidence=False).passed is True


# --- the merge: provenance survives the per-token MAX ----------------------------------------


def _bundle(numeric: Counter) -> SourceEvidenceBundle:
    return SourceEvidenceBundle(
        numeric=numeric,
        content=frozenset({"corn"}),
        has_content_evidence=True,
        witness_state="reading",
    )


def test_merge_keeps_the_pixel_and_layer_tokens_apart_under_max() -> None:
    merged = _merge_source_bundles(
        _bundle(Counter({"4.1": 1, "7": 2})), _bundle(Counter({"41": 1, "7": 1}))
    )
    assert merged.numeric == Counter({"4.1": 1, "41": 1, "7": 2})  # max, not sum
    assert merged.layer_numeric == Counter({"4.1": 1, "7": 2})
    assert merged.pixel_numeric == Counter({"41": 1, "7": 1})


# --- the function: only designated stand-ins count --------------------------------------------


def test_a_token_outside_the_pixel_stand_ins_cannot_stand_for_the_emitted_one() -> None:
    emitted, second = Counter({"-4.1": 1}), Counter({"-4.1": 1})
    # evidence token 41 is not among the pixel tokens
    got, still = corroborate_unsupported_numeric(
        emitted, Counter({"41": 1}), second, standin=Counter(), exact_keys=frozenset()
    )
    assert (sum(got.values()), still) == (0, Counter({"-4.1": 1}))
    # control: designate it as pixel-read
    got, still = corroborate_unsupported_numeric(
        emitted, Counter({"41": 1}), second, standin=Counter({"41": 1}), exact_keys=frozenset()
    )
    assert (got, sum(still.values())) == (Counter({"-4.1": 1}), 0)


def test_the_exact_layer_digit_key_vetoes_whatever_the_marks() -> None:
    emitted, second = Counter({"-4.1": 1}), Counter({"-4.1": 1})
    got, still = corroborate_unsupported_numeric(
        emitted,
        Counter({"41": 1}),
        second,
        standin=Counter({"41": 1}),
        exact_keys=frozenset({"41"}),
    )
    assert sum(got.values()) == 0 and still == Counter({"-4.1": 1})
