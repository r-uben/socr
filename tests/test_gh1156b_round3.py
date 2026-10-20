"""#1156 Part B, review round 3 (PR #1217 at 29bf2421): the masker's last forms, the
per-position invariant, and the gate's view of a reading the sidecar does not keep.

The re-finalize tool's round-3 paths (the cache, the PDF's identity) are pinned in
``test_gh1156b_masked_layer.py`` beside its end-to-end harness. Hermetic.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from socr.tables.prior_ocr_mask import (
    NUMBER_WITHHELD,
    carries_a_numeral,
    mask_numbers,
    masked_layer_text,
    numeric_tokens_in_order,
)

W = NUMBER_WITHHELD


def test_loaded_source_is_this_checkout() -> None:
    import socr

    assert Path(socr.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1] / "src")


@pytest.mark.parametrize("layer", ["&#xBD", "a &#xBD b", "&#189 and more"])
def test_a_reference_without_its_semicolon_is_still_a_numeral(layer: str) -> None:
    """An HTML parser decodes "&#xBD" as a half; its hex digits are letters, so neither the
    grammar nor the invariant saw the numeral it writes (Argon, round 2 re-review)."""
    masked, withheld = mask_numbers(layer)
    assert withheld == 1 and "&#" not in masked


def test_a_numeral_that_is_a_letter_by_category_keeps_the_floor() -> None:
    """CJK numerals are ``Lo`` by category and ``isnumeric`` by value: the grammar does not
    mask them, so the invariant must refuse the page (Argon, round 2 re-review)."""
    layer = "The Committee met. Rates rose by 三 per cent."
    assert carries_a_numeral("三")
    assert masked_layer_text(layer)[0] == ""
    assert masked_layer_text(layer.replace("三", "3"))[0], "control: a digit masks"


def test_a_confirmed_page_still_proves_nothing_else_survives(monkeypatch) -> None:
    """Fable P2-3: the invariant switched off for the whole page as soon as anything was
    confirmed. It is now asked of the layer with every number withheld."""
    from socr.tables import prior_ocr_mask

    layer = "The Committee met on July 12-13, 1983."
    every = frozenset(range(len(numeric_tokens_in_order(layer))))
    text, withheld, _ = masked_layer_text(layer, every)
    assert text and withheld == 0, "control: a whole-page confirmation ships its numbers"

    real = prior_ocr_mask.mask_numbers

    def misses_the_year(t, confirmed=frozenset(), token=W):
        masked, n = real(t, confirmed, token=token)
        return masked.replace(token, "1983", 1) if not confirmed else masked, n

    monkeypatch.setattr(prior_ocr_mask, "mask_numbers", misses_the_year)
    assert masked_layer_text(layer, every)[0] == "", (
        "a grammar miss shipped because the page carried a confirmation"
    )


def _suspect():
    from socr.core.document import DocumentHandle
    from socr.core.state import DocumentState, PageState

    state = DocumentState(handle=DocumentHandle(path=Path("x.pdf"), page_count=1))
    p = PageState(page_num=1)
    p.is_born_digital = True
    p.native_text = "The Committee met and reviewed the outlook.\nRates rose from 3 to 4 per cent."
    p.prior_ocr_layer = True
    p.prior_ocr_layer_suspect = True
    p.needs_ocr_enhancement = True
    state.pages[1] = p
    return state


@pytest.mark.parametrize("held", [True, False])
def test_the_attempt_record_tells_the_gate_a_reading_held_a_table(held: bool) -> None:
    """Fable P1-2: a page rebuilt from its sidecar has no readings; the record of each
    attempt says whether it held a table."""
    from socr.core.manifest import SelectionProvenance, _select_and_finalize_page

    state = _suspect()
    state.pages[1].attempts_summary_restored = [{"engine": "qwen", "held_a_table": held}]
    provenance = _select_and_finalize_page(state, 1).selection_provenance
    assert (provenance is SelectionProvenance.PRIOR_OCR_LAYER_MASKED) is (not held)


def test_a_suspect_pages_attempt_record_says_whether_each_reading_held_a_table() -> None:
    from socr.core.result import PageOutput, PageStatus
    from socr.pipeline.orchestrator import _attempts_summary

    state = _suspect()
    p = state.pages[1]
    table = "| a | b |\n| --- | --- |\n| Loans | 1 |\n| Gold | 2 |"
    p.attempts = [
        PageOutput(page_num=1, text=table, status=PageStatus.WARNING, engine="qwen"),
        PageOutput(page_num=1, text="prose", status=PageStatus.WARNING, engine="gemini"),
    ]
    assert [row["held_a_table"] for row in _attempts_summary(p)] == [True, False]
    p.prior_ocr_layer_suspect = False
    assert all("held_a_table" not in row for row in _attempts_summary(p)), (
        "every other page's sidecar must keep its bytes"
    )


def test_a_masked_page_without_its_image_is_not_mined_for_figures() -> None:
    """Argon P1-4: with no page image, the extractor would take the scan as a "figure" and
    caption it after the mask."""
    from socr.core.manifest import page_ships_its_own_page_image
    from socr.tables.prior_ocr_mask import NUMBER_WITHHELD_NO_IMAGE

    p = _suspect().pages[1]
    assert page_ships_its_own_page_image(p, f"Rates rose to {NUMBER_WITHHELD_NO_IMAGE}.")
    assert not page_ships_its_own_page_image(p, "A model reading with no token.")
    p.prior_ocr_layer = False
    assert not page_ships_its_own_page_image(p, f"Rates rose to {NUMBER_WITHHELD_NO_IMAGE}.")


# ---------------------------------------------------------------------------
# Review round 4
# ---------------------------------------------------------------------------


def test_a_first_attempt_refused_as_an_invalid_table_is_a_table() -> None:
    """Argon round 3 P1-1: the emission guard refuses a malformed table without a table
    event; on a retried page only the first attempt's failure mode says so."""
    from socr.core.manifest import SelectionProvenance, _select_and_finalize_page

    for mode, masks in (("table_emission_invalid", False), ("audit_failed", True)):
        state = _suspect()
        state.pages[1].page_retry = {"first_attempt": {"audit_events": [], "failure_mode": mode}}
        provenance = _select_and_finalize_page(state, 1).selection_provenance
        assert (provenance is SelectionProvenance.PRIOR_OCR_LAYER_MASKED) is masks, mode


def test_a_masked_page_with_nothing_to_withhold_is_not_mined_either() -> None:
    """Fable round 3 P2-2: keyed on the token, a masked layer with no number was mined."""
    from socr.core.manifest import page_ships_its_own_page_image

    p = _suspect().pages[1]
    banner = (
        "> [page 1 unchecked: prior_ocr_layer=UNKNOWN; scope=page; reason=earlier OCR, not re-read]"
    )
    assert page_ships_its_own_page_image(p, f"{banner}\n\nPlain prose with no number.")
    assert not page_ships_its_own_page_image(p, "Plain prose with no number.")


def test_a_resumed_suspect_pages_record_gains_its_table_answer() -> None:
    """Argon round 3 P2-2: a record restored from before ``held_a_table`` existed."""
    from socr.core.result import PageOutput, PageStatus
    from socr.pipeline.orchestrator import _attempts_summary

    p = _suspect().pages[1]
    table = "| a | b |\n| --- | --- |\n| Loans | 1 |\n| Gold | 2 |"
    p.attempts = [PageOutput(page_num=1, text=table, status=PageStatus.SUCCESS, engine="qwen")]
    p.attempts_summary_restored = [{"engine": "qwen", "accepted": True}]
    assert _attempts_summary(p) == [{"engine": "qwen", "accepted": True, "held_a_table": True}]
    p.prior_ocr_layer_suspect = False
    assert _attempts_summary(p) == [{"engine": "qwen", "accepted": True}]


# ---------------------------------------------------------------------------
# Review round 5
# ---------------------------------------------------------------------------


def test_a_judge_reason_longer_than_the_record_still_finds_its_reading(
    tmp_path: Path, monkeypatch
) -> None:
    """The socr bot, round 4 re-review: nothing pinned the cut. The run records a reason cut
    to the summary limit, so the tool must cut the cached reading's reason the same way."""
    from collections import Counter

    from socr.pipeline.agentic import AcceptDecision
    from socr.pipeline.orchestrator import ATTEMPT_SUMMARY_REASON_MAX_CHARS
    from socr.pipeline.refinalize_prior_ocr import refinalize_document

    import test_gh1156b_masked_layer as e2e

    long_reason = "  the reading  drops\na row; " * ATTEMPT_SUMMARY_REASON_MAX_CHARS
    monkeypatch.setattr(
        e2e._RejectingJudge,
        "assess",
        lambda self, output, provider: AcceptDecision(accept=False, reason=long_reason),
    )
    doc = e2e._pre_part_b(e2e._floor_run(tmp_path, monkeypatch))
    row = json.loads((doc / "pages" / "00001.json").read_text())["attempts_summary"][0]
    assert len(row["rejection_reason"]) == ATTEMPT_SUMMARY_REASON_MAX_CHARS, "control: cut"
    skipped: Counter = Counter()
    assert refinalize_document(doc, pdf_dir=tmp_path, skipped=skipped)
    assert skipped == Counter()
