"""#1266 B1: the TR-3 D3 regional ending takes the born-digital geometry floor too.

The D3 ending (``SelectionProvenance.UNVERIFIABLE_TABLE_NATIVE``) spliced ``native_text``
around its markers: ``splice_all_table_regions`` on a header-unattributed page,
``splice_failed_table_regions`` over the TR-3 ordinals otherwise. Both replace only the
parsed GFM blocks, so every un-parsed table line shipped verbatim (503 of 539 scored Fed
pages leaked). Owner ruling 2026-11-05: the ending takes the structure-class geometry floor
(``structure_class_floor_text``), which withholds EVERY detected box -- a TR-3-verified
sibling included, the accepted price -- and records the recovery on the page output.

Pages are built with PyMuPDF so the words carry real geometry. Every behavioural test pins a
DIFFERENCE: the same page run twice, changing only the thing under test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_gh317_structure_class_floor import PROSE_BEFORE  # noqa: E402
from test_gh520_regional_floor_splice import (  # noqa: E402
    PROSE_BETWEEN,
    TWO_PARSEABLE_TABLES,
    _floored_page,
)
from test_gh1266_floor_geometry import (  # noqa: E402
    GRID_BOX,
    GRID_TOKENS,
    MARKER,
    P94_BOX,
    PNG,
    SENTENCE_WITH_YEAR,
    TAIL_SENTENCE,
    _build,
    _draw_p10,
    _draw_p94,
    _draw_p94_header_only,
    _gated_process_pdf,
    _process_pdf,
    _restore_and_assemble,
    _run,
    _whole_page,
)

from socr.core.manifest import (  # noqa: E402
    BORN_DIGITAL_PROSE_RECOVERED_FLAG,
    SCANNED_PROSE_RECOVERED_FLAG,
    SCANNED_PROSE_RECOVERED_NOTE,
    SelectionProvenance,
    _select_and_finalize_page,
    _select_page_output_tagged,
    finalized_page_records,
    native_prose_floor_text,
    splice_all_table_regions,
    splice_failed_table_regions,
)
from socr.core.result import FailureMode, PageOutput, PageStatus  # noqa: E402

D3 = SelectionProvenance.UNVERIFIABLE_TABLE_NATIVE
BANNER = BORN_DIGITAL_PROSE_RECOVERED_FLAG.format(page_num=1)

#: One parseable GFM table, prose above it, and an un-parsed numeric line below it: the
#: splice replaced the block and shipped ``Loose 7.7 8.8`` verbatim.
SPLICE_LEAK_TEXT = (
    f"{TAIL_SENTENCE}\n\n| a | b |\n|---|---|\n| 1.5 | 2.5 |\n| 3.5 | 4.5 |\n\nLoose 7.7 8.8\n"
)
SPLICE_LEAK_TOKENS = ("Loose", "7.7", "8.8", "1.5", "2.5", "3.5", "4.5")
#: Tokens of the FIRST table of ``TWO_PARSEABLE_TABLES`` (the TR-3-verified sibling when
#: ordinal 1 failed) and of the second (the failed one).
VERIFIED_SIBLING_TOKENS = ("0.03", "0.91", "0.44", "0.07", "0.85", "0.51", "slope")
FAILED_TABLE_TOKENS = ("0.14", "0.21", "0.02", "0.05", "5y")


def _d3(ps, *, header_unattributed: bool = False, ordinals=(0,)):
    """Make *ps* a TR-3 D3 page: the per-region verifier hard-failed (or the header could
    not be attributed), the ladder ran and left one rejected attempt, no model grid kept."""
    ps.native_table_structure_failed = True
    ps.native_table_unverifiable = not header_unattributed
    ps.native_table_header_unattributed = header_unattributed
    ps.native_table_unverifiable_ordinals = list(ordinals)
    failed = PageOutput(
        page_num=1,
        text="failed ocr attempt",
        status=PageStatus.WARNING,
        engine="qwen",
        audit_passed=False,
        failure_mode=FailureMode.NATIVE_TABLE_STRUCTURE_FAILED,
    )
    ps.attempts = [failed]
    ps.best_output = None
    return ps


def _assert_d3_ending(out, prov) -> None:
    """The ending itself is unchanged: same tag, status, engine, verdict and failure mode."""
    assert prov is D3, prov
    assert out.status is PageStatus.ERROR
    assert out.engine == "native"
    assert out.audit_passed is False
    assert out.failure_mode is FailureMode.NATIVE_TABLE_STRUCTURE_FAILED


def _assert_recorded(out, recovered: bool) -> None:
    assert out.scanned_prose_recovered is recovered
    assert out.audit_notes == ([SCANNED_PROSE_RECOVERED_NOTE] if recovered else [])


# ---------------------------------------------------------------------------
# a. header-unattributed route
# ---------------------------------------------------------------------------


def test_a_header_unattributed_d3_page_ships_no_table_line_and_records_the_recovery(
    tmp_path: Path,
) -> None:
    """DIFFERENCE: the same page through the old helper and through the D3 ending."""
    state, ps = _floored_page(tmp_path, SPLICE_LEAK_TEXT, detected=1, geometry=True)
    _d3(ps, header_unattributed=True)
    old = splice_all_table_regions(ps.native_text, marker_line=MARKER, png_ref=PNG)
    assert old and "Loose 7.7 8.8" in old, "premise: the old splice ships the un-parsed line"

    out, prov = _select_page_output_tagged(state, 1)
    _assert_d3_ending(out, prov)
    assert BANNER in out.text
    assert TAIL_SENTENCE in out.text, "the page's prose was withheld"
    for tok in SPLICE_LEAK_TOKENS:
        assert tok not in out.text, f"table content {tok!r} shipped as loose text"
    assert MARKER in out.text
    _assert_recorded(out, True)


def test_both_d3_routes_ship_the_same_floor(tmp_path: Path) -> None:
    """DIFFERENCE: the same page, only the route flag differs (header-unattributed versus
    failed ordinals). Both routes now take the one floor, so the bytes are identical."""
    for sub in ("h", "f"):
        (tmp_path / sub).mkdir()
    hs, hps = _floored_page(tmp_path / "h", SPLICE_LEAK_TEXT, detected=1, geometry=True)
    fs, fps = _floored_page(tmp_path / "f", SPLICE_LEAK_TEXT, detected=1, geometry=True)
    _d3(hps, header_unattributed=True)
    _d3(fps, header_unattributed=False, ordinals=[0])
    old_failed = splice_failed_table_regions(
        fps.native_text,
        failed_ordinals=[0],
        expected_count=fps.native_table_region_count,
        marker_line=MARKER,
        png_ref=PNG,
        region_identities=list(fps.native_table_region_identities),
    )
    assert old_failed and "Loose 7.7 8.8" in old_failed, "premise: the ordinal splice leaked"
    h_out, h_prov = _select_page_output_tagged(hs, 1)
    f_out, f_prov = _select_page_output_tagged(fs, 1)
    _assert_d3_ending(h_out, h_prov)
    _assert_d3_ending(f_out, f_prov)
    assert h_out.text == f_out.text
    for tok in SPLICE_LEAK_TOKENS:
        assert tok not in f_out.text


# ---------------------------------------------------------------------------
# b. failed-ordinal route with a VERIFIED sibling (owner-accepted price)
# ---------------------------------------------------------------------------


def test_b_a_verified_sibling_no_longer_ships_on_a_failed_ordinal_d3_page(
    tmp_path: Path,
) -> None:
    """The price the owner accepted (2026-11-05): the floor withholds every detected box,
    so the sibling TR-3 VERIFIED no longer ships its GFM. Pinned so the per-region
    follow-up changes it deliberately. DIFFERENCE: the old ordinal splice on the same page
    shipped that sibling."""
    state, ps = _floored_page(tmp_path, TWO_PARSEABLE_TABLES, detected=2, geometry=True)
    _d3(ps, ordinals=[1])
    old = splice_failed_table_regions(
        ps.native_text,
        failed_ordinals=[1],
        expected_count=ps.native_table_region_count,
        marker_line=MARKER,
        png_ref=PNG,
        region_identities=list(ps.native_table_region_identities),
    )
    assert old and "| 2 | 0.03 | 0.91 | 0.44 |" in old, "premise: the sibling's GFM shipped"

    out, prov = _select_page_output_tagged(state, 1)
    _assert_d3_ending(out, prov)
    for tok in VERIFIED_SIBLING_TOKENS:
        assert tok not in out.text, f"verified sibling content {tok!r} shipped"
    for tok in FAILED_TABLE_TOKENS:
        assert tok not in out.text, f"failed table content {tok!r} shipped"
    assert PROSE_BETWEEN in out.text and PROSE_BEFORE in out.text, "the prose was withheld"
    assert BANNER in out.text
    _assert_recorded(out, True)


# ---------------------------------------------------------------------------
# c. header-only box with rows below (p94 shape)
# ---------------------------------------------------------------------------


def test_c_a_header_only_box_with_rows_below_is_whole_page(tmp_path: Path) -> None:
    """DIFFERENCE: the same header-only box with and without the unboxed data rows."""
    state, ps = _build(tmp_path, _draw_p94, [P94_BOX], name="p94.pdf")
    _d3(ps)
    out, prov = _select_page_output_tagged(state, 1)
    _assert_d3_ending(out, prov)
    assert _whole_page(out.text), out.text
    assert "Indicator" not in out.text
    _assert_recorded(out, False)

    ctl, ctl_ps = _build(tmp_path, _draw_p94_header_only, [P94_BOX], name="ctl.pdf")
    _d3(ctl_ps)
    ctl_out, ctl_prov = _select_page_output_tagged(ctl, 1)
    _assert_d3_ending(ctl_out, ctl_prov)
    assert SENTENCE_WITH_YEAR in ctl_out.text
    _assert_recorded(ctl_out, True)


# ---------------------------------------------------------------------------
# d. prior-OCR layer (#1156 decision (1))
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flag", ["prior_ocr_layer", "invisible_text_over_raster"])
def test_d_a_positive_prior_ocr_finding_is_whole_page_and_unknown_takes_the_floor(
    tmp_path: Path, flag: str
) -> None:
    """DIFFERENCE over one page: clean, suspect (unknown: a failed scan) and a POSITIVE
    finding. Positive -> whole-page marker plus image (as before); suspect -> the same
    geometry floor as clean (#1156: "unknown is not a finding")."""
    clean, clean_ps = _build(tmp_path, _draw_p10, [GRID_BOX], name="clean.pdf")
    sus, sus_ps = _build(tmp_path, _draw_p10, [GRID_BOX], name="sus.pdf")
    pos, pos_ps = _build(tmp_path, _draw_p10, [GRID_BOX], name="pos.pdf")
    for ps in (clean_ps, sus_ps, pos_ps):
        _d3(ps)
    sus_ps.prior_ocr_layer_suspect = True
    setattr(pos_ps, flag, True)

    clean_out, clean_prov = _select_page_output_tagged(clean, 1)
    sus_out, sus_prov = _select_page_output_tagged(sus, 1)
    pos_out, pos_prov = _select_page_output_tagged(pos, 1)
    for out, prov in ((clean_out, clean_prov), (sus_out, sus_prov), (pos_out, pos_prov)):
        _assert_d3_ending(out, prov)

    assert SENTENCE_WITH_YEAR in clean_out.text
    for tok in GRID_TOKENS:
        assert tok not in clean_out.text
    assert sus_out.text == clean_out.text, "a suspect layer took a different lane"
    assert _whole_page(pos_out.text), pos_out.text
    assert SENTENCE_WITH_YEAR not in pos_out.text
    _assert_recorded(clean_out, True)
    _assert_recorded(pos_out, False)


# ---------------------------------------------------------------------------
# f. the scanned D3 ending is untouched
# ---------------------------------------------------------------------------


def test_f_the_scanned_d3_ending_is_unchanged(tmp_path: Path) -> None:
    """DIFFERENCE: the same printed page as a scan (``UNVERIFIABLE_TABLE_SCANNED``) and as
    a born-digital D3 page. The scan keeps its own body -- exactly ``native_prose_floor_text``
    under the any-digit rule, the call that branch has always made -- and its own banner."""
    scan, scan_ps = _build(tmp_path, _draw_p10, [GRID_BOX], born_digital=False, name="s.pdf")
    born, born_ps = _build(tmp_path, _draw_p10, [GRID_BOX], name="b.pdf")
    scan_ps.scanned_table_evidence_failed = True
    _d3(born_ps)

    scan_out, scan_prov = _select_page_output_tagged(scan, 1)
    born_out, born_prov = _select_page_output_tagged(born, 1)
    assert scan_prov is SelectionProvenance.UNVERIFIABLE_TABLE_SCANNED
    _assert_d3_ending(born_out, born_prov)
    expected = native_prose_floor_text(
        scan_ps, 1, marker_line=MARKER, png_ref=scan_ps.d3_floor_png_ref
    )
    assert expected and scan_out.text == expected
    assert SCANNED_PROSE_RECOVERED_FLAG.format(page_num=1) in scan_out.text
    assert SENTENCE_WITH_YEAR not in scan_out.text, "a scan used the born-digital rule"
    assert SENTENCE_WITH_YEAR in born_out.text


# ---------------------------------------------------------------------------
# e. resume / re-finalize byte-identity
# ---------------------------------------------------------------------------


def test_e_finalize_twice_and_with_the_saved_body_is_byte_identical(tmp_path: Path) -> None:
    for draw, recovered, name in ((_draw_p10, True, "r.pdf"), (_draw_p94, False, "w.pdf")):
        boxes = [GRID_BOX] if recovered else [P94_BOX]
        state, ps = _build(tmp_path, draw, boxes, name=name)
        _d3(ps)
        first = _select_and_finalize_page(state, 1)
        second = _select_and_finalize_page(state, 1)
        assert first.output.text == second.output.text
        again = _select_and_finalize_page(state, 1, saved_text=first.output.text)
        assert again.output.text == first.output.text
        assert again.output.scanned_prose_recovered is recovered


def _restored_d3(tmp_path: Path, body: str, *, typed: bool, name: str):
    """A D3 page whose state was rebuilt WITH ``is_born_digital`` and ``native_text`` but no
    ``native_words`` (the live-run cache the sidecar does not carry)."""
    state, ps = _build(tmp_path, _draw_p10, [GRID_BOX], name=name)
    _d3(ps)
    ps.native_words = []
    out = PageOutput(
        page_num=1,
        text=body,
        status=PageStatus.ERROR,
        engine="native",
        audit_passed=False,
        failure_mode=FailureMode.NATIVE_TABLE_STRUCTURE_FAILED,
        scanned_prose_recovered=typed,
    )
    ps.attempts = [out]
    ps.best_output = out
    return state


def test_e_a_words_less_d3_page_replays_its_typed_recovery_and_otherwise_fails_closed(
    tmp_path: Path,
) -> None:
    """The D3 ending's replay is the same typed one as the structure-class ending's. A page
    in the D3 branch without words replays a frozen born-digital recovery byte for byte;
    without the typed credential, or with a scan banner, it fails CLOSED (whole page)."""
    live, live_ps = _build(tmp_path, _draw_p10, [GRID_BOX], name="live.pdf")
    _d3(live_ps)
    body, prov = _select_page_output_tagged(live, 1)
    assert prov is D3 and SENTENCE_WITH_YEAR in body.text
    body = body.text

    rep_out, rep_prov = _select_page_output_tagged(
        _restored_d3(tmp_path, body, typed=True, name="a.pdf"), 1
    )
    _assert_d3_ending(rep_out, rep_prov)
    assert rep_out.text == body
    _assert_recorded(rep_out, True)

    untyped, untyped_prov = _select_page_output_tagged(
        _restored_d3(tmp_path, body, typed=False, name="b.pdf"), 1
    )
    _assert_d3_ending(untyped, untyped_prov)
    assert _whole_page(untyped.text)

    scan_body = body.replace(BANNER, SCANNED_PROSE_RECOVERED_FLAG.format(page_num=1))
    scan_out, scan_prov = _select_page_output_tagged(
        _restored_d3(tmp_path, scan_body, typed=True, name="c.pdf"), 1
    )
    _assert_d3_ending(scan_out, scan_prov)
    assert _whole_page(scan_out.text)

    # A POSITIVE prior-OCR finding outranks the replay too (#1156): the layer's text is not
    # shipped even when a typed recovery of it is on the page. BOTH positive routes, each on
    # its own (Astra review 3, Q7: a bypass for one flag only must not survive).
    for flag in ("prior_ocr_layer", "invisible_text_over_raster"):
        flagged = _restored_d3(tmp_path, body, typed=True, name=f"d-{flag}.pdf")
        setattr(flagged.pages[1], flag, True)
        flagged_out, flagged_prov = _select_page_output_tagged(flagged, 1)
        _assert_d3_ending(flagged_out, flagged_prov)
        assert _whole_page(flagged_out.text), flag
        _assert_recorded(flagged_out, False)


@pytest.fixture
def forced_tr3_failure(monkeypatch):
    """Every table region the per-region geometry verifier examines hard-fails (TR-3), so a
    real PDF reaches the D3 ending through ``process()`` on its own."""
    from socr.tables import native_verifier

    real = native_verifier.verify_native_table_region

    def _hard_fail(page, text, rect):
        result = real(page, text, rect)
        result.hard_fail = True
        result.reason = "forced by the #1266 B1 test"
        return result

    monkeypatch.setattr(native_verifier, "verify_native_table_region", _hard_fail)


def _own_lines(body: str) -> list[str]:
    return [
        ln
        for ln in body.splitlines()
        if ln.strip() and not ln.lstrip().startswith((">", "![", "[page"))
    ]


def _run_route(tmp_path: Path, pdf: Path, monkeypatch, *, ladder: bool, native_first: bool, tag):
    """``_run`` with the route knob exposed: ``native_first`` (the CLI default is True) sends a
    failed native table through the native-table-first refusal; False sends the page through
    the OCR ladder."""
    import dataclasses
    import json

    from test_gh1144_prose_survives_the_table_floor import _GridEngine, _pipeline

    from socr.core.providers import PROFILE_QWEN_LOCAL
    from socr.pipeline import orchestrator as orch

    monkeypatch.setattr(orch, "get_engine", lambda engine_type: _GridEngine())
    pipe = _pipeline()
    pipe.config = dataclasses.replace(pipe.config, native_first=native_first)
    pipe._available_engines_for_agentic = lambda: [PROFILE_QWEN_LOCAL] if ladder else []
    out = tmp_path / tag
    pipe.process(pdf, output_dir=out)
    frag = next(iter(out.rglob("pages/00001.md"))).read_text()
    meta = json.loads(next(iter(out.rglob("pages/00001.json"))).read_text())
    return frag, meta, out


#: (ladder, native_first) routes measured to end at D3. With the OCR route
#: (``native_first=False``) and an empty ladder the providerless loop never sets
#: ``native_table_structure_failed`` (orchestrator: only ``native_fallback_pages``, which needs
#: ``native_first``), so that cell ends at the structure-class floor and is checked separately.
_D3_ROUTES = [(True, False), (True, True), (False, True)]


def test_process_d3_differential_and_both_provider_states(
    tmp_path: Path, monkeypatch, forced_tr3_failure
) -> None:
    """process() over a real PDF whose table hard-fails TR-3, in BOTH provider states.

    Every route in ``_D3_ROUTES`` -- populated ladder via the OCR ladder, populated ladder via
    the native-table-first refusal, and an EMPTY ladder via that refusal -- ends at D3 (sidecar
    ``failure_mode``), and in each one toggling ONLY the geometry floor
    (``geometry_floor=False`` inside ``table_floor_text_for_source``) changes the bytes. The
    page's own lines are identical across the routes. The fourth cell (empty ladder, OCR
    route) does not carry the D3 conjunction and ends at the structure-class floor -- the
    same floor, so the same own lines.
    """
    from socr.core import manifest as manifest_mod

    pdf = _process_pdf(tmp_path / "doc.pdf")
    own = {}
    for ladder, native_first in _D3_ROUTES:
        tag = f"{'ladder' if ladder else 'empty'}-{'nf' if native_first else 'ocr'}"
        frag, meta, out = _run_route(
            tmp_path, pdf, monkeypatch, ladder=ladder, native_first=native_first, tag=f"on-{tag}"
        )
        assert meta["failure_mode"] == FailureMode.NATIVE_TABLE_STRUCTURE_FAILED.value, tag
        assert BANNER in frag, tag
        for tok in ("Real GDP", "Final sales", "2.6", "4.1"):
            assert tok not in frag, f"{tag}: table content {tok!r} shipped"
        assert meta["winning_output"]["scanned_prose_recovered"] is True, tag
        assert SCANNED_PROSE_RECOVERED_NOTE in meta["winning_output"]["audit_notes"], tag
        meta_doc = next(iter(out.rglob("doc/metadata.json"))).read_text()
        assert "born-digital table could not be verified" in meta_doc, f"{tag}: surfacing silent"
        with monkeypatch.context() as mp:
            real = manifest_mod.table_floor_text_for_source
            mp.setattr(
                manifest_mod,
                "table_floor_text_for_source",
                lambda *a, **k: real(*a, **{**k, "geometry_floor": False}),
            )
            off_frag, off_meta, _ = _run_route(
                tmp_path,
                pdf,
                monkeypatch,
                ladder=ladder,
                native_first=native_first,
                tag=f"off-{tag}",
            )
        assert off_meta["failure_mode"] == meta["failure_mode"], f"{tag}: route differs"
        assert off_frag != frag, f"{tag}: the geometry floor changed nothing on the D3 page"
        own[tag] = _own_lines(frag)

    s1_frag, s1_meta, _ = _run_route(
        tmp_path, pdf, monkeypatch, ladder=False, native_first=False, tag="on-empty-ocr"
    )
    assert "structure_class" in s1_meta["failure_mode"], s1_meta["failure_mode"]
    own["empty-ocr"] = _own_lines(s1_frag)
    assert len({tuple(v) for v in own.values()}) == 1, own


@pytest.mark.parametrize("shape", ["recovered", "whole-page"])
def test_e_a_real_restore_reproduces_the_live_d3_page_byte_for_byte(
    tmp_path: Path, monkeypatch, forced_tr3_failure, shape: str
) -> None:
    """Live process() (D3 ending) -> sidecar -> a FRESH pipeline restores the page with no
    ``native_words`` -> finalize -> assemble into a CLEAN destination (the compared fragment
    and final .md are deleted first). Fragment and final .md are byte-identical to live.

    Which path the replay takes, measured: the sidecar restores every D3 flag but not
    ``is_born_digital`` or ``native_text``, so the restored page does NOT re-enter the D3
    branch -- it ships its frozen winner through ``BEST_OUTPUT_UNVERIFIED``. The second leg
    gives the restored state ``is_born_digital`` and ``native_text`` back (still no words):
    then it DOES re-enter D3, and the replay (recovered) or the fail-closed marker
    (whole-page) reproduces the live bytes.
    """
    import json

    from test_gh1144_prose_survives_the_table_floor import _pipeline

    from socr.core.document import DocumentHandle
    from socr.core.state import DocumentState

    make = _process_pdf if shape == "recovered" else _gated_process_pdf
    pdf = make(tmp_path / "doc.pdf")
    frag, meta, out = _run(tmp_path, pdf, monkeypatch, ladder=True, tag="live")
    assert meta["failure_mode"] == FailureMode.NATIVE_TABLE_STRUCTURE_FAILED.value, "premise"
    recovered = shape == "recovered"
    assert (BANNER in frag) is recovered, frag
    assert bool(meta["winning_output"].get("scanned_prose_recovered")) is recovered

    def _final_path(root: Path):
        found = [p for p in root.rglob("*.md") if p.parent.name != "pages"]
        return found[0] if found else None

    live_final = _final_path(out)
    final_md = live_final.read_text() if live_final else None
    assert (final_md is not None) is recovered, "live final .md presence"
    frag_path = next(iter(out.rglob("pages/00001.md")))
    frag_path.unlink()
    if live_final is not None:
        live_final.unlink()
    assert not frag_path.exists() and _final_path(out) is None

    state = _restore_and_assemble(pdf, meta, out, monkeypatch, expected_body=frag)
    if recovered:
        assert frag_path.exists(), "the restore's assemble did not rewrite the page fragment"
        assert frag_path.read_text() == frag, "the page fragment changed across a real restore"
        final2 = _final_path(out)
        assert final2 is not None and final2.read_text() == final_md
    else:
        assert _final_path(out) is None, "a whole-page document gained a final .md"
    meta2 = json.loads(next(iter(out.rglob("pages/00001.json"))).read_text())
    for key in ("status", "failure_mode"):
        assert meta2[key] == meta[key], key
    for key in ("scanned_prose_recovered", "audit_notes"):
        assert meta2["winning_output"].get(key) == meta["winning_output"].get(key), key
    _, restored_prov = _select_page_output_tagged(state, 1)
    assert restored_prov is SelectionProvenance.BEST_OUTPUT_UNVERIFIED, restored_prov

    # Second leg: the D3 branch itself, re-entered words-less.
    fresh = _pipeline()
    fresh._scan_root = pdf.parent
    again = DocumentState(handle=DocumentHandle.from_path(pdf))
    fresh._restore_terminal_page_state(again, 1, PageOutput.from_dict(meta["winning_output"]), out)
    ps = again.pages[1]
    import fitz

    with fitz.open(str(pdf)) as doc:
        ps.native_text = doc[0].get_text()
    ps.is_born_digital = True
    assert not (ps.native_words or []), "premise: no words"
    _, prov = _select_page_output_tagged(again, 1)
    assert prov is D3, prov
    assert finalized_page_records(again)[0].output.text == frag


# ---------------------------------------------------------------------------
# commit 3: the D3 floor no longer needs an OCR attempt (Astra review 3, P1 #1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("header_unattributed", [False, True])
def test_a_d3_page_with_no_attempt_takes_the_floor(tmp_path: Path, header_unattributed) -> None:
    """DIFFERENCE: the same page with one rejected attempt and with NONE ship the same floor.

    Before, the zero-attempt page declined D3 and S1 and shipped ``_native_text_with_appends``
    -- the TR-3-refuted grid's native bytes. With no attempt there is no model to keep."""
    from socr.core.manifest import (
        _reaches_structure_class_branch,
        d3_floor_kept_model_output,
    )

    one, one_ps = _build(tmp_path, _draw_p10, [GRID_BOX], name="one.pdf")
    none, none_ps = _build(tmp_path, _draw_p10, [GRID_BOX], name="none.pdf")
    _d3(one_ps, header_unattributed=header_unattributed)
    _d3(none_ps, header_unattributed=header_unattributed)
    none_ps.attempts = []
    none_ps.best_output = None
    assert any(tok in none_ps.native_text for tok in GRID_TOKENS), "premise: grid in native"
    assert d3_floor_kept_model_output(none_ps) is None
    assert not _reaches_structure_class_branch(none_ps)

    one_out, one_prov = _select_page_output_tagged(one, 1)
    none_out, none_prov = _select_page_output_tagged(none, 1)
    _assert_d3_ending(one_out, one_prov)
    _assert_d3_ending(none_out, none_prov)
    assert none_out.text == one_out.text
    for tok in GRID_TOKENS:
        assert tok not in none_out.text, f"table content {tok!r} shipped"
    assert SENTENCE_WITH_YEAR in none_out.text
    _assert_recorded(none_out, True)


def test_a_zero_attempt_d3_page_surfaces_at_document_level(tmp_path: Path) -> None:
    """The bucket behind the D3 audit event, the document status and the CLI line follows the
    floor: a zero-attempt D3 page is in ``d3_floor_pages`` and emits
    ``table_region_unverifiable`` at assemble."""
    from unittest.mock import patch

    from test_gh371_d3_region_splice import _make_pipeline

    from socr.pipeline.orchestrator import _derive_disposition_buckets

    state, ps = _build(tmp_path, _draw_p10, [GRID_BOX], name="z.pdf")
    _d3(ps)
    ps.attempts = []
    ps.best_output = None
    buckets = _derive_disposition_buckets(state, finalized_page_records(state))
    assert 1 in buckets["d3_floor_pages"]

    pipeline = _make_pipeline()
    out_dir = tmp_path / "out"
    with (
        patch.object(pipeline, "_save_markdown", return_value=out_dir / "out.md"),
        patch.object(pipeline, "_write_metadata", return_value=None),
        patch.object(pipeline, "_write_manifest", return_value=None),
        patch.object(pipeline, "_rewrite_all_fragments", return_value=None),
        patch.object(pipeline, "_flush_page_fragment", return_value=None),
        patch.object(pipeline, "_flush_page_sidecar", return_value=None),
        patch.object(pipeline, "_stitch_fragments", return_value=""),
    ):
        pipeline._phase_assemble(state, out_dir)
    assert "table_region_unverifiable" in [e.kind for e in state.events]


def _two_page_pdf(path: Path, *, gated: bool) -> Path:
    """The D3 page (recovered or whole-page shape) plus an ordinary prose page, so assemble
    has document text and writes every page's fragment and sidecar afresh."""
    import fitz

    first = (_gated_process_pdf if gated else _process_pdf)(path.with_name("p1.pdf"))
    doc = fitz.open(str(first))
    doc.new_page().insert_text((72, 72), "An ordinary page of prose follows the table page.")
    doc.save(str(path))
    doc.close()
    return path


@pytest.mark.parametrize("shape", ["recovered", "whole-page"])
def test_e_a_restore_regenerates_fragment_and_sidecar_byte_for_byte(
    tmp_path: Path, monkeypatch, forced_tr3_failure, shape: str
) -> None:
    """Astra review 3, P2: FRESH artifacts in both arms. Live process() on a two-page
    document (the D3 page plus a prose page) -> a fresh pipeline restores both pages from
    their sidecars -> AFTER the restore has consumed them, every fragment, every sidecar and
    the final .md are deleted -> assemble. The D3 page's fragment and sidecar must be
    regenerated and equal to live (status, failure mode, recovery flag, audit notes, text),
    and the final .md too."""
    import json

    from test_gh1144_prose_survives_the_table_floor import _pipeline

    from socr.core.document import DocumentHandle
    from socr.core.state import DocumentState

    pdf = _two_page_pdf(tmp_path / "doc.pdf", gated=shape == "whole-page")
    frag, meta, out = _run(tmp_path, pdf, monkeypatch, ladder=True, tag="live")
    assert meta["failure_mode"] == FailureMode.NATIVE_TABLE_STRUCTURE_FAILED.value, "premise"
    recovered = shape == "recovered"
    assert (BANNER in frag) is recovered, frag
    pages_dir = next(iter(out.rglob("pages/00001.md"))).parent
    live = {
        n: (
            (pages_dir / f"{n:05d}.md").read_text(),
            json.loads((pages_dir / f"{n:05d}.json").read_text()),
        )
        for n in (1, 2)
    }
    finals = [p for p in out.rglob("*.md") if p.parent.name != "pages"]
    assert len(finals) == 1, finals
    final_path, final_md = finals[0], finals[0].read_text()

    fresh = _pipeline()
    fresh._scan_root = pdf.parent
    state = DocumentState(handle=DocumentHandle.from_path(pdf))
    for n in (1, 2):
        fresh._restore_terminal_page_state(
            state, n, PageOutput.from_dict(live[n][1]["winning_output"]), out
        )
    assert not (state.pages[1].native_words or []), "premise: a restored page has no words"
    for n in (1, 2):
        (pages_dir / f"{n:05d}.md").unlink()
        (pages_dir / f"{n:05d}.json").unlink()
    final_path.unlink()

    assert finalized_page_records(state)[0].output.text == frag
    fresh._phase_assemble(state, out)

    for n in (1, 2):
        assert (pages_dir / f"{n:05d}.md").exists(), f"fragment {n} not regenerated"
        assert (pages_dir / f"{n:05d}.json").exists(), f"sidecar {n} not regenerated"
    assert (pages_dir / "00001.md").read_text() == frag
    meta2 = json.loads((pages_dir / "00001.json").read_text())
    for key in ("status", "failure_mode"):
        assert meta2[key] == meta[key], key
    for key in ("scanned_prose_recovered", "audit_notes", "text", "status", "failure_mode"):
        assert meta2["winning_output"].get(key) == meta["winning_output"].get(key), key
    assert final_path.exists() and final_path.read_text() == final_md
