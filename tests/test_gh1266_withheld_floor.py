"""#1266 merge gate (PR #1273): a withheld table never ships through a text splice.

The TABLE_WITHHELD ladder guard spliced the SELECTED output's text (often a model reading)
behind GH-520's coverage conditions; the splice replaces only parsed GFM blocks, so a
parsed grid beside an un-parsed ``Loose 7.7 8.8`` row shipped the row (Astra, P0). The
guard and the nothing-produced ending now call ``withheld_table_floor_text``: the
born-digital geometry floor, else the whole-page marker. Real PyMuPDF geometry; every test
pins a DIFFERENCE.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_gh520_regional_floor_splice import _floored_page  # noqa: E402
from test_gh1266_floor_geometry import TAIL_SENTENCE  # noqa: E402

from socr.core.manifest import (  # noqa: E402
    SCANNED_PROSE_RECOVERED_NOTE,
    _apply_ladder_disposition_guard,
    _is_prose_recovery_body,
    table_floor_text_for_source,
    withheld_table_floor_text,
)
from socr.core.result import FailureMode, PageOutput, PageStatus  # noqa: E402

MARKER = "[page 1 failed: unverifiable table — see image]"
PNG = "![Failed table page 1](figures/failed_table_p1.png)"
#: One parseable GFM table, prose above it, and an un-parsed numeric row below it.
SPLICE_LEAK_TEXT = (
    f"{TAIL_SENTENCE}\n\n| a | b |\n|---|---|\n| 1.5 | 2.5 |\n| 3.5 | 4.5 |\n\nLoose 7.7 8.8\n"
)
LEAK_TOKENS = ("Loose", "7.7", "8.8", "1.5", "2.5", "3.5", "4.5")


def _withheld_page(tmp_path: Path, *, born_digital: bool = True):
    """A page whose ladder disposition is TABLE_WITHHELD, with a MODEL winner carrying the
    parsed grid plus the un-parsed row, and real word geometry for the same text."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    _state, ps = _floored_page(tmp_path, SPLICE_LEAK_TEXT, detected=1, geometry=True)
    ps.is_born_digital = born_digital
    ps.table_ladder_disposition = FailureMode.TABLE_WITHHELD
    winner = PageOutput(
        page_num=1,
        text=SPLICE_LEAK_TEXT,
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=True,
    )
    return ps, winner


def _whole_page(text: str) -> bool:
    return text.strip() == f"{MARKER}\n\n{PNG}"


def test_the_withheld_guard_ships_the_geometry_floor_not_the_splice(tmp_path: Path) -> None:
    ps, winner = _withheld_page(tmp_path)
    old = table_floor_text_for_source(ps, 1, winner.text)
    assert "Loose 7.7 8.8" in old, "premise: the old guard splice shipped the un-parsed row"

    out = _apply_ladder_disposition_guard(winner, 1, ps)
    for tok in LEAK_TOKENS:
        assert tok not in out.text, f"table content {tok!r} shipped"
    assert TAIL_SENTENCE in out.text, "the page's own prose was withheld"
    assert _is_prose_recovery_body(out.text, 1)
    assert out.status is PageStatus.ERROR
    assert out.audit_passed is False
    assert out.failure_mode is FailureMode.TABLE_WITHHELD
    assert out.scanned_prose_recovered is True
    assert SCANNED_PROSE_RECOVERED_NOTE in out.audit_notes


def test_a_withheld_scan_is_whole_page(tmp_path: Path) -> None:
    """DIFFERENCE: the same page as a scan ships the whole-page marker plus image."""
    born, winner = _withheld_page(tmp_path / "b")
    scan, scan_winner = _withheld_page(tmp_path / "s", born_digital=False)
    born_out = _apply_ladder_disposition_guard(winner, 1, born)
    scan_out = _apply_ladder_disposition_guard(scan_winner, 1, scan)
    assert TAIL_SENTENCE in born_out.text
    assert _whole_page(scan_out.text), scan_out.text
    assert scan_out.scanned_prose_recovered is False
    assert SCANNED_PROSE_RECOVERED_NOTE not in scan_out.audit_notes


def test_a_positive_prior_ocr_finding_is_whole_page(tmp_path: Path) -> None:
    clean, _ = _withheld_page(tmp_path / "c")
    flagged, _ = _withheld_page(tmp_path / "f")
    flagged.prior_ocr_layer = True
    assert TAIL_SENTENCE in withheld_table_floor_text(clean, 1)
    assert _whole_page(withheld_table_floor_text(flagged, 1))


def test_recovery_off_and_a_fallback_marker_give_the_callers_own_marker(tmp_path: Path) -> None:
    ps, _ = _withheld_page(tmp_path)
    assert TAIL_SENTENCE in withheld_table_floor_text(ps, 1)
    assert withheld_table_floor_text(ps, 1, recovery=False, fallback_marker="[x]") == "[x]"


def test_re_finalizing_a_withheld_floor_body_is_byte_identical(tmp_path: Path) -> None:
    """The ``already_withheld`` shortcut: the guard over its own output changes no byte, and
    over a words-less page (as after a restore) still keeps the frozen body."""
    ps, winner = _withheld_page(tmp_path)
    once = _apply_ladder_disposition_guard(winner, 1, ps)
    twice = _apply_ladder_disposition_guard(once, 1, ps)
    assert twice.text == once.text
    assert twice.scanned_prose_recovered is True
    ps.native_words = []
    thrice = _apply_ladder_disposition_guard(once, 1, ps)
    assert thrice.text == once.text
