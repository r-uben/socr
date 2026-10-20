"""#1156 Part B, review round 2 (PR #1217): the masker's leaks, the table gate, and the
re-finalize tool.

Each test reproduces one review claim that was REAL when measured. Pins are differences
or properties, never an outcome measured on one machine. Hermetic: no provider, no Ollama.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from socr.tables.prior_ocr_mask import (
    NUMBER_WITHHELD,
    carries_a_numeral,
    mask_numbers,
    masked_layer_text,
)

W = NUMBER_WITHHELD


def test_loaded_source_is_this_checkout() -> None:
    import socr

    assert Path(socr.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1] / "src")


# ---------------------------------------------------------------------------
# P0-3: the masker
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("layer", "shipped"),
    [
        # A leading-dot decimal followed by a fraction glyph or a superscript: the glyph
        # shipped, because only the integer form could take a tail.
        (".5½", W),
        ("-.8¹", W),
        # A character reference to punctuation is that punctuation, not a number: its code
        # digits shipped.
        ("&#8212;", "—"),
        ("&#160;", " "),
        # A named entity that is not a numeral is not part of a number.
        ("3&ndash;5", f"{W}&ndash;{W}"),
        # Digits of every script group into one number.
        ("１２３", W),
        ("-.５", W),
    ],
)
def test_argon_round2_forms_mask_completely(layer: str, shipped: str) -> None:
    masked, _withheld = mask_numbers(layer)
    assert masked == shipped
    assert not carries_a_numeral(masked)


def test_a_range_after_a_fraction_glyph_withholds_both_sides() -> None:
    """Cosmetic only: the dash goes with the second number. No numeral survives."""
    masked, withheld = mask_numbers("3½-4")
    assert withheld == 2 and not carries_a_numeral(masked)


def test_a_reference_to_markup_stays_an_entity() -> None:
    """Decoding "&#60;" to "<" would put raw HTML on the page."""
    masked, withheld = mask_numbers("&#60;script&#62; and &#38; &#x22;")
    assert withheld == 0
    assert masked == "&lt;script&gt; and &amp; &quot;"


def test_a_layer_that_still_carries_a_numeral_keeps_its_floor(monkeypatch) -> None:
    """The positive invariant, checked against a masker that misses a number: whatever the
    grammar leaves, the text that would ship is asked whether it carries a numeral."""
    from socr.tables import prior_ocr_mask

    layer = "The Committee met and rates rose to 3 per cent."
    text, _withheld, _shreds = masked_layer_text(layer)
    assert text and not carries_a_numeral(text), "control: the real masker ships the prose"

    monkeypatch.setattr(prior_ocr_mask, "mask_numbers", lambda t, c=frozenset(), token=W: (t, 0))
    leaked, _withheld, _shreds = masked_layer_text(layer)
    assert leaked == "", "a body carrying a numeral was allowed to ship"


# ---------------------------------------------------------------------------
# Fable P2-1: the document cannot read SUCCESS over a masked page
# ---------------------------------------------------------------------------


def _png(w: int, h: int) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("L", (w, h), 245).save(buf, format="PNG")
    return buf.getvalue()


def _scan_pdf(path: Path) -> Path:
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    for row, line in enumerate(
        [
            "The Committee met and reviewed conditions in the economy.",
            "Discount rates were raised from 3 to 3 Y, per cent at the Banks.",
            "Members discussed the outlook for prices and employment.",
        ]
    ):
        page.insert_text((72, 120 + row * 24), line, fontsize=10)
    page.insert_image(page.rect, stream=_png(300, 400))
    doc.save(path)
    doc.close()
    return path


@pytest.mark.parametrize("native_only", [True, False], ids=["native-only", "no-provider"])
@pytest.mark.parametrize("needs_ocr", [True, False], ids=["routed", "not-routed"])
def test_the_document_is_never_success_over_a_masked_page(
    tmp_path: Path, monkeypatch, needs_ocr: bool, native_only: bool
) -> None:
    """No model runs (``--native-only``), so no reading exists and ``best_output`` may be
    empty; the masked page must still keep the document out of SUCCESS. The not-routed arm
    is the corner Fable named: an assessment that did not set ``needs_ocr_enhancement``."""
    from socr.core.born_digital import BornDigitalDetector
    from socr.core.config import EngineType, PipelineConfig
    from socr.core.result import DocumentStatus
    from socr.pipeline.orchestrator import UnifiedPipeline

    original = BornDigitalDetector._assess_page

    def assess(self, page, page_num):
        pa = original(self, page, page_num)
        if pa.prior_ocr_layer:
            pa.needs_ocr_enhancement = needs_ocr
        return pa

    monkeypatch.setattr(BornDigitalDetector, "_assess_page", assess)
    pipe = UnifiedPipeline(
        PipelineConfig(
            agentic=True,
            quiet=True,
            primary_engine=EngineType.QWEN,
            local_engine=EngineType.QWEN,
            enabled_engines=[EngineType.QWEN],
            native_first=True,
            native_only=native_only,
            write_manifest=False,
            judge_backend="heuristic",
            dual_pass_tables=False,
            detect_equations=False,
            save_figures=False,
        )
    )
    pipe._available_engines_for_agentic = lambda: []
    pipe._resolve_judge_model = lambda *a, **k: ""
    pipe._resolve_crop_vlm_model = lambda: None
    result = pipe.process(_scan_pdf(tmp_path / "scan.pdf"), output_dir=tmp_path / "out")
    side = json.loads(next((tmp_path / "out").rglob("pages/00001.json")).read_text())

    assert side["disposition"]["primary_reason"] == "prior_ocr_layer_masked"
    assert result.status is not DocumentStatus.SUCCESS


# ---------------------------------------------------------------------------
# P0-2: the table gate asks every signal, and each one holds the page on its floor
# ---------------------------------------------------------------------------

_LAYER = "The Committee met and reviewed the outlook.\nRates rose from 3 to 3 Y, per cent."


def _suspect(*, attempt_text: str = "A MODEL READING OF PROSE"):
    from socr.core.document import DocumentHandle
    from socr.core.result import FailureMode, PageOutput, PageStatus
    from socr.core.state import DocumentState, PageState

    state = DocumentState(handle=DocumentHandle(path=Path("x.pdf"), page_count=1))
    p = PageState(page_num=1)
    p.is_born_digital = True
    p.native_text = _LAYER
    p.prior_ocr_layer = True
    p.prior_ocr_layer_suspect = True
    p.needs_ocr_enhancement = True
    rejected = PageOutput(
        page_num=1,
        text=attempt_text,
        status=PageStatus.WARNING,
        engine="qwen",
        audit_passed=False,
        failure_mode=FailureMode.AUDIT_FAILED,
    )
    p.attempts = [rejected]
    p.best_output = rejected
    state.pages[1] = p
    return state


def _provenance(state):
    from socr.core.manifest import _select_and_finalize_page

    return _select_and_finalize_page(state, 1).selection_provenance


_MD_TABLE = "| Item | 1957 |\n| --- | --- |\n| Loans | 12.4 |\n| Deposits | 40.2 |"
_HTML_TABLE = "<table><tr><td>Loans</td><td>12.4</td></tr><tr><td>Gold</td><td>2</td></tr></table>"


@pytest.mark.parametrize(
    "signal",
    [
        "markdown-reading",
        "html-reading",
        "possible-table-flag",
        "table-not-reconstructed-flag",
        "detected-table-count",
        "table-ladder-ran",
        "first-attempt-event",
    ],
)
def test_every_table_signal_holds_the_page_on_its_floor(signal: str) -> None:
    from socr.core.manifest import SelectionProvenance
    from socr.core.result import FailureMode

    assert _provenance(_suspect()) is SelectionProvenance.PRIOR_OCR_LAYER_MASKED, "control"
    if signal == "markdown-reading":
        state = _suspect(attempt_text=_MD_TABLE)
    elif signal == "html-reading":
        state = _suspect(attempt_text=_HTML_TABLE)
    else:
        state = _suspect()
        p = state.pages[1]
        if signal == "possible-table-flag":
            p.possible_table_structure_not_reconstructed = True
        elif signal == "table-not-reconstructed-flag":
            p.table_not_reconstructed = True
        elif signal == "detected-table-count":
            p.detected_table_count = 1
        elif signal == "table-ladder-ran":
            p.table_ladder_disposition = FailureMode.TABLE_WITHHELD
        elif signal == "first-attempt-event":
            p.page_retry = {"first_attempt": {"audit_events": [{"kind": "table_structure_failed"}]}}
    assert _provenance(state) is not SelectionProvenance.PRIOR_OCR_LAYER_MASKED


def test_a_stacked_column_in_the_layer_holds_the_page_on_its_floor() -> None:
    """The reproduced leak: a table under the raster with no signal anywhere else."""
    from socr.core.manifest import SelectionProvenance

    columns = "The staff reported these figures.\nLoans\n12.4\n13.1\nDeposits\n40.2\n41.0"
    folio = "IV -\n14\nThe staff reported these figures.\nRates rose sharply.\n-55-"
    state = _suspect()
    state.pages[1].native_text = columns
    assert _provenance(state) is not SelectionProvenance.PRIOR_OCR_LAYER_MASKED
    furniture = _suspect()
    furniture.pages[1].native_text = folio
    assert _provenance(furniture) is SelectionProvenance.PRIOR_OCR_LAYER_MASKED, (
        "a folio split over two lines is furniture, not a column"
    )


def test_a_column_needs_two_stacked_values_between_lines_of_words() -> None:
    from socr.tables.prior_ocr_mask import holds_a_stacked_column

    assert holds_a_stacked_column(f"Loans\n{W}\n{W}\nGold")
    assert not holds_a_stacked_column(f"Loans\n{W}\nGold\n{W}")  # a label and one value
    assert not holds_a_stacked_column(f"{W}\n{W}\nThe text starts here.\nIt ends.")  # edge
    assert not holds_a_stacked_column(f"Text.\nMore text.\n{W}\n{W}")  # bottom edge


def test_a_column_below_the_last_words_is_a_column_and_a_footer_is_not() -> None:
    """Argon round 2: labels first, their values stacked after the last worded line. The
    footer of a Fed page -- a date over a folio -- is the furniture that must stay."""
    from socr.tables.prior_ocr_mask import holds_a_stacked_column

    labels_then_values = f"Loans\nDeposits\nBorrowings\n{W}\n{W}\n{W}\n{W}"
    footer = f"The Committee adjourned.\nIt will meet again.\n{W}/{W}/{W}\n-{W}-"
    header = f"{W}/{W}/{W}\n-{W}-\nThe Committee met.\nIt adjourned."
    values_then_labels = f"{W}\n{W}\n{W}\n{W}\nLoans\nDeposits\nGold"
    assert holds_a_stacked_column(labels_then_values)
    assert holds_a_stacked_column(values_then_labels)
    assert not holds_a_stacked_column(footer)
    assert not holds_a_stacked_column(header)


def test_a_retried_page_remembers_that_its_first_reading_held_a_table() -> None:
    """The first attempt's readings leave with it; whether one held a table must not."""
    from socr.core.manifest import SelectionProvenance

    state = _suspect()
    state.pages[1].page_retry = {
        "first_attempt": {"audit_events": [], "reading_held_a_table": True}
    }
    assert _provenance(state) is not SelectionProvenance.PRIOR_OCR_LAYER_MASKED
    state = _suspect()
    state.pages[1].page_retry = {
        "first_attempt": {"audit_events": [], "reading_held_a_table": False}
    }
    assert _provenance(state) is SelectionProvenance.PRIOR_OCR_LAYER_MASKED


def test_the_retry_reset_records_a_table_in_the_first_reading() -> None:
    from socr.core.manifest import _select_and_finalize_page
    from socr.pipeline.orchestrator import UnifiedPipeline

    for text, held in ((_HTML_TABLE, True), ("A MODEL READING OF PROSE", False)):
        state = _suspect(attempt_text=text)
        snapshot = state.pages[1].__class__(page_num=1)
        first = _select_and_finalize_page(state, 1)
        UnifiedPipeline._reset_page_for_retry(state, 1, snapshot, [], first, [])
        assert state.pages[1].page_retry["first_attempt"]["reading_held_a_table"] is held
