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

    # A scan whose GH-520 guard FAILS: the scan floor would ship its own any-digit recovery
    # (premise), and the withheld floor still ships the whole page -- no native floor on a
    # scan, whichever way the scan lane would have gone.
    from socr.core.manifest import structure_class_floor_text

    refused, refused_winner = _withheld_page(tmp_path / "r", born_digital=False)
    refused.native_table_region_count = 2
    assert _is_prose_recovery_body(structure_class_floor_text(refused, 1), 1), "premise"
    refused_out = _apply_ladder_disposition_guard(refused_winner, 1, refused)
    assert _whole_page(refused_out.text), refused_out.text


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


# ---------------------------------------------------------------------------
# round 2 (Astra P0-2): the keep-as-is shortcut needs the typed floor credential
# ---------------------------------------------------------------------------

#: A model reading that echoes the floor's marker line beside an un-parsed numeric row.
FORGED = f"{MARKER}\n\nLoose 7.7 8.8\n"


def test_a_reading_that_echoes_the_marker_is_withheld(tmp_path: Path) -> None:
    """The shortcut used to trust the marker SUBSTRING, so this body skipped all withholding
    (fails on 53ba3b78). Without the typed credential it is withheld like any reading."""
    ps, _winner = _withheld_page(tmp_path)
    forged = PageOutput(
        page_num=1, text=FORGED, status=PageStatus.SUCCESS, engine="qwen", audit_passed=True
    )
    out = _apply_ladder_disposition_guard(forged, 1, ps)
    assert "Loose" not in out.text and "7.7" not in out.text, out.text
    assert TAIL_SENTENCE in out.text
    assert out.table_floor_body is True


def test_a_credentialed_floor_body_is_kept_and_round_trips_the_sidecar(tmp_path: Path) -> None:
    """DIFFERENCE: the same bytes with and without the credential. The guard's own output
    (credentialed) is kept byte-identical, also after a ``to_dict``/``from_dict`` round trip
    (the sidecar's winning output); an old record without the key carries no credential."""
    ps, winner = _withheld_page(tmp_path)
    once = _apply_ladder_disposition_guard(winner, 1, ps)
    restored = PageOutput.from_dict(once.to_dict())
    assert restored.table_floor_body is True
    assert _apply_ladder_disposition_guard(restored, 1, ps).text == once.text
    legacy = once.to_dict()
    legacy.pop("table_floor_body")
    assert PageOutput.from_dict(legacy).table_floor_body is False
    forged_dict = {**legacy, "table_floor_body": "true"}  # not a JSON boolean
    assert PageOutput.from_dict(forged_dict).table_floor_body is False


def test_a_saved_body_keeps_the_credential_only_when_it_is_the_floors_own(
    tmp_path: Path,
) -> None:
    """Re-finalize with a saved body (``_select_and_finalize_page``): the floor's own
    finalized bytes keep the credential and come back byte-identical; a different saved body
    (a reading that echoes the marker) loses it and is withheld."""
    from test_gh1266_d3_regional_floor import _d3
    from test_gh1266_floor_geometry import GRID_BOX, _build, _draw_p10

    from socr.core.manifest import _select_and_finalize_page

    state, ps = _build(tmp_path, _draw_p10, [GRID_BOX], name="sv.pdf")
    _d3(ps)
    ps.table_ladder_disposition = FailureMode.TABLE_WITHHELD
    first = _select_and_finalize_page(state, 1).output
    assert first.table_floor_body is True
    again = _select_and_finalize_page(state, 1, saved_text=first.text).output
    assert again.text == first.text and again.table_floor_body is True
    forged = _select_and_finalize_page(state, 1, saved_text=FORGED).output
    assert "Loose" not in forged.text, forged.text


# ---------------------------------------------------------------------------
# round 3 (Astra P0): the credential is bound to a digest of the body it certifies
# ---------------------------------------------------------------------------


def test_a_forged_record_credential_is_dropped() -> None:
    """``from_dict`` accepts the credential only as a JSON ``true`` with a digest of the
    record's own text. DIFFERENCE: the same record with a matching digest keeps it."""
    from socr.core.manifest import table_floor_digest_of

    base = PageOutput(page_num=1, text=FORGED, status=PageStatus.ERROR, engine="native").to_dict()
    assert PageOutput.from_dict({**base, "table_floor_body": True}).table_floor_body is False
    wrong = {**base, "table_floor_body": True, "table_floor_digest": table_floor_digest_of("x")}
    assert PageOutput.from_dict(wrong).table_floor_body is False
    right = {**base, "table_floor_body": True, "table_floor_digest": table_floor_digest_of(FORGED)}
    assert PageOutput.from_dict(right).table_floor_body is True


def test_a_certified_body_with_one_changed_token_loses_the_credential(tmp_path: Path) -> None:
    """DIFFERENCE: the guard's own credentialed output is kept as is; the same output with one
    token changed no longer matches its digest and is withheld again."""
    from dataclasses import replace

    ps, winner = _withheld_page(tmp_path)
    once = _apply_ladder_disposition_guard(winner, 1, ps)
    assert _apply_ladder_disposition_guard(once, 1, ps).text == once.text
    tampered = replace(once, text=once.text + "\n\nLoose 7.7 8.8\n")
    out = _apply_ladder_disposition_guard(tampered, 1, ps)
    assert "Loose" not in out.text, out.text


def test_socr_s_own_hyphen_reflow_keeps_the_credential_and_nothing_else_does() -> None:
    from socr.core.manifest import _same_body_modulo_marks

    selected = "The committee judged that mone-\ntary conditions stayed loose."
    assert _same_body_modulo_marks(
        "The committee judged that monetary conditions stayed loose.", selected, 1
    )
    assert _same_body_modulo_marks(
        "The committee judged that mone-tary conditions stayed loose.", selected, 1
    )
    assert not _same_body_modulo_marks(
        "The committee judged that money conditions stayed loose.", selected, 1
    )
    signed = "Net exports -0.4"
    assert not _same_body_modulo_marks("Net exports 0.4", signed, 1), "a sign change survived"


def _hyphen_pdf(path: Path) -> Path:
    """Ending A's process() page plus a prose line split at a line-end hyphen (Probe D)."""
    import fitz
    from test_gh1266_floor_geometry import _process_pdf

    src = _process_pdf(path.with_name("base.pdf"))
    doc = fitz.open(str(src))
    page = doc[0]
    page.insert_text((72, 320), "The committee judged that mone-")
    page.insert_text((72, 334), "tary conditions remained accommodative overall.")
    doc.save(str(path))
    doc.close()
    return path


def test_the_credential_survives_the_reflow_into_the_sidecar(tmp_path: Path, monkeypatch) -> None:
    """Fable round 3 P2: on a page whose floor body carries a line-end hyphen, the #1074 re-flow
    joins it; the sidecar's winning output still carries the credential, with a digest of the
    shipped text."""
    import json

    from test_gh1266_floor_geometry import _run

    from socr.core.manifest import table_floor_digest_of

    pdf = _hyphen_pdf(tmp_path / "doc.pdf")
    _frag, meta, _out = _run(tmp_path, pdf, monkeypatch, ladder=False, tag="hy")
    wo = meta["winning_output"]
    assert wo.get("scanned_prose_recovered") is True, "premise: a recovered floor body"
    assert "mone-\ntary" not in wo["text"], "premise: the re-flow joined the hyphen"
    assert wo.get("table_floor_body") is True
    assert wo.get("table_floor_digest") == table_floor_digest_of(wo["text"])
    del json


def test_a_restored_page_does_not_inherit_the_credential_for_a_changed_fragment(
    tmp_path: Path,
) -> None:
    """``_load_terminal_page`` replaces the record's text with the fragment on disk. A D3 floor
    page made content-terminal (TABLE_REJECTED) restores with its credential when the fragment
    is the certified body, and WITHOUT it when the fragment differs."""
    from test_gh371_d3_region_splice import _make_pipeline
    from test_gh1266_d3_regional_floor import _d3
    from test_gh1266_floor_geometry import GRID_BOX, _build, _draw_p10

    from socr.core.manifest import _winning_page_output

    state, ps = _build(tmp_path, _draw_p10, [GRID_BOX], name="rs.pdf")
    _d3(ps)
    ps.table_ladder_disposition = FailureMode.TABLE_REJECTED
    winner = _winning_page_output(state, 1, None)
    assert winner.table_floor_body is True, "premise: a credentialed floor body"
    pipeline = _make_pipeline()
    pipeline._resolve_judge_model = lambda *a, **k: ""  # hermetic: no judge probe
    pipeline._scan_root = state.handle.path.parent
    out_dir = tmp_path / "out"
    pipeline._flush_page_fragment(state, 1, winner.text, out_dir)
    sidecar = pipeline._flush_page_sidecar(state, 1, out_dir, terminal=True)
    same = pipeline._load_terminal_page(state, 1, out_dir)
    assert same is not None, "premise: the page restores"
    assert same.table_floor_body is True
    frag = sidecar.with_suffix(".md")
    frag.write_text(frag.read_text().replace("Growth", "Grwth"))
    changed = pipeline._load_terminal_page(state, 1, out_dir)
    assert changed is not None, "premise: the changed fragment still restores"
    assert changed.table_floor_body is False, "a changed fragment inherited the credential"
    # round 4: the recovery facts go with the credential they relied on
    assert same.scanned_prose_recovered is True, "premise: the floor body is a recovery"
    assert changed.scanned_prose_recovered is False, "a changed fragment kept the recovery flag"


def test_an_altered_recovered_body_is_not_replayed(tmp_path: Path) -> None:
    """Astra round 4 P0: a words-less D3 page whose saved recovered body was ALTERED (a row
    appended) keeps the recovery flag and the banner, but its credential no longer matches.
    DIFFERENCE: the intact body replays byte for byte; the altered one is not reused, and with
    no words the page fails closed -- the appended row never ships."""
    from dataclasses import replace

    from test_gh1266_d3_regional_floor import _d3
    from test_gh1266_floor_geometry import GRID_BOX, _build, _draw_p10

    from socr.core.manifest import _winning_page_output

    live, live_ps = _build(tmp_path, _draw_p10, [GRID_BOX], name="lv.pdf")
    _d3(live_ps)
    certified = _winning_page_output(live, 1, None)
    assert certified.table_floor_body and certified.scanned_prose_recovered, "premise"

    def _restored(output, name: str):
        state, ps = _build(tmp_path, _draw_p10, [GRID_BOX], name=name)
        _d3(ps)
        ps.native_words = []
        ps.attempts = [output]
        ps.best_output = output
        return state

    intact = _winning_page_output(_restored(certified, "in.pdf"), 1, None)
    assert intact.text == certified.text
    altered = replace(certified, text=certified.text + "\n\nLoose 7.7 8.8\n")
    out = _winning_page_output(_restored(altered, "al.pdf"), 1, None)
    assert "Loose" not in out.text and "7.7" not in out.text, out.text
