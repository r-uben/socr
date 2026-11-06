"""#1269 Part A: record why the chart floor declined a page.

Hermetic: every page is drawn with PyMuPDF; nothing touches a provider, the network or
Ollama. Outcomes are pinned as DIFFERENCES (recording on vs off, table evidence vs none,
key set vs absent), never as a value measured on one machine.
"""

from __future__ import annotations

import json

import fitz
import pytest

import socr.core.manifest as m
from socr.figures.extractor import has_chart_marks
from socr.tables.reconstruct import chart_region_bboxes

from test_gh1155_chart_witness import (
    _chart_floor_document_state,
    _preservation_pipeline,
)


def _save(doc, path):
    doc.save(path)
    doc.close()
    return path


def _framed_thin_chart(tmp_path):
    """Framed plot, thin black strokes only: census True, production boxer [].

    No fills, no colour, no thick stroke, no numeric tick labels, no table.
    """
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 760), "A sentence of ordinary prose carrying no figures", fontsize=10)
    page.draw_rect(fitz.Rect(100, 200, 500, 500), color=(0, 0, 0), width=0.5)
    for i in range(60):
        x = 110 + i * 6
        page.draw_line((x, 490), (x, 300 + (i * 37) % 150), color=(0, 0, 0), width=0.3)
    return _save(doc, tmp_path / "framed.pdf")


def _filled_bars_no_ticks(tmp_path):
    """Coloured bars, no axis labels: a region exists, the reader reads no panel."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 40), "Some prose above the shapes", fontsize=11)
    red = (0.9, 0.1, 0.1)
    for i, x in enumerate((100, 180, 260, 340, 420)):
        page.draw_rect(fitz.Rect(x, 500 - i * 40, x + 60, 680), color=red, fill=red)
    return _save(doc, tmp_path / "bars.pdf")


def _prose_only(tmp_path):
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "Plain prose and nothing drawn. " * 4, fontsize=10)
    return _save(doc, tmp_path / "prose.pdf")


class _P:
    """A page stand-in that accepts the recorder's attributes."""


def _probe(pdf, tmp_path):
    class _State:
        class handle:
            path = pdf

        figures_dir = tmp_path

    p = _P()
    builder = m.chart_floor_replacement(_State(), 1, p, marker_line="[m]", png_ref="")
    return builder, p


def test_framed_thin_chart_records_no_chart_bboxes_and_the_census_disagreement(tmp_path):
    pdf = _framed_thin_chart(tmp_path)
    with fitz.open(pdf) as doc:
        assert has_chart_marks(doc[0]) is True, "premise: the census fires"
        assert chart_region_bboxes(doc[0]) == [], "premise: production boxes nothing"
    builder, p = _probe(pdf, tmp_path)
    assert builder is None
    assert p.chart_floor_probe == {"outcome": m.CHART_FLOOR_NO_BBOXES, "census_chart_marks": True}


def test_filled_bars_without_ticks_record_no_panels_with_refusals(tmp_path):
    pdf = _filled_bars_no_ticks(tmp_path)
    builder, p = _probe(pdf, tmp_path)
    assert builder is None
    probe = p.chart_floor_probe
    assert probe["outcome"] == m.CHART_FLOOR_NO_PANELS
    assert probe["regions"] >= 1
    assert all(k.isdigit() for k in probe["refusals"])
    assert probe["refusals"], "the reader's reasons are kept, not just the count"
    assert all(isinstance(v, str) and v for v in probe["refusals"].values())


def test_prose_only_page_records_no_chart_bboxes_and_census_false(tmp_path):
    builder, p = _probe(_prose_only(tmp_path), tmp_path)
    assert builder is None
    assert p.chart_floor_probe == {"outcome": m.CHART_FLOOR_NO_BBOXES, "census_chart_marks": False}


def test_missing_pdf_path_records_probe_unavailable():
    class _State:
        handle = None

    p = _P()
    assert m.chart_floor_replacement(_State(), 1, p, marker_line="[m]", png_ref="") is None
    assert p.chart_floor_probe == {"outcome": m.CHART_FLOOR_PROBE_UNAVAILABLE}


# --------------------------------------------------------------------------------------
# The seam: the probe runs eagerly, only the floor's real decline persists.
# --------------------------------------------------------------------------------------


def _finalize(pdf, *, table_evidence=False, **kwargs):
    state = _chart_floor_document_state(pdf)
    if table_evidence:
        state.pages[1].detected_table_count = 1
    out = m._select_and_finalize_page(state, 1, **kwargs).output.text
    return state.pages[1], out


def test_floor_decline_persists_the_probe_record(tmp_path):
    ps, _ = _finalize(_framed_thin_chart(tmp_path))
    assert ps.chart_floor_abstention == {
        "outcome": m.CHART_FLOOR_NO_BBOXES,
        "census_chart_marks": True,
    }


def test_table_evidence_gets_the_probe_but_not_an_abstention(tmp_path):
    """Kimi 4: the eager probe also runs for pages the floor never declined."""
    pdf = _framed_thin_chart(tmp_path)
    control, _ = _finalize(pdf)
    with_tables, _ = _finalize(pdf, table_evidence=True)
    assert with_tables.chart_floor_probe is not None, "the probe ran"
    assert with_tables.chart_floor_probe == control.chart_floor_probe
    assert control.chart_floor_abstention is not None, "control: no table evidence records it"
    assert with_tables.chart_floor_abstention is None


_REPLACEMENT = "A replacement body with no chart floor marker."


def test_saved_text_that_is_not_the_bare_floor_body_clears_the_abstention(tmp_path):
    """Astra 3 (P1): the abstention must describe the bytes that ship.

    #1266 round 5: on a floor page a saved text that no floor certified no longer ships --
    the recomputed floor does -- so the bytes that ship ARE the bare floor body and the
    abstention, which describes them, stays. (The name is kept for history.)"""
    state = _chart_floor_document_state(_framed_thin_chart(tmp_path))
    ps = state.pages[1]
    body = m._select_and_finalize_page(state, 1).output.text
    assert ps.chart_floor_abstention is not None, "premise: the floor declined"
    shipped = m._select_and_finalize_page(state, 1, saved_text=_REPLACEMENT).output.text
    assert _REPLACEMENT not in shipped and shipped == body
    assert ps.chart_floor_abstention is not None


def test_saved_text_on_a_fresh_state_invents_no_abstention(tmp_path):
    state = _chart_floor_document_state(_framed_thin_chart(tmp_path))
    m._select_and_finalize_page(state, 1, saved_text=_REPLACEMENT)
    assert state.pages[1].chart_floor_abstention is None
    # Control, the one thing changed: no saved text records the decline.
    control = _chart_floor_document_state(_framed_thin_chart(tmp_path))
    m._select_and_finalize_page(control, 1)
    assert control.pages[1].chart_floor_abstention is not None


def test_saved_text_that_is_the_real_bare_floor_body_keeps_the_abstention(tmp_path):
    state = _chart_floor_document_state(_framed_thin_chart(tmp_path))
    ps = state.pages[1]
    body = m._select_and_finalize_page(state, 1).output.text
    recorded = dict(ps.chart_floor_abstention)
    m._select_and_finalize_page(state, 1, saved_text=body)
    assert ps.chart_floor_abstention == recorded


def test_a_stale_chart_floor_body_flag_does_not_erase_a_valid_abstention(tmp_path, monkeypatch):
    """Astra review 2 (P1): ``chart_floor_body`` is set once and never cleared.

    Success (the floor ships a chart body: the #1155 flag goes True), then a decline on
    the real page (bare floor, abstention recorded), then a finalize under that same
    bare body as saved text. The bytes are identical to the declined selection's, so the
    abstention must survive; trusting the stale flag deleted it.
    """
    state = _chart_floor_document_state(_framed_thin_chart(tmp_path))
    ps = state.pages[1]
    with monkeypatch.context() as patched:
        patched.setattr(m, "chart_floor_replacement", lambda *a, **kw: lambda: "a chart body")
        m._select_and_finalize_page(state, 1)
    assert ps.chart_floor_body is True, "premise: the flag went True and stays"
    body = m._select_and_finalize_page(state, 1).output.text
    assert ps.chart_floor_abstention is not None, "premise: the real page declines"
    assert ps.chart_floor_body is True, "premise: #1155's flag is still stale"
    recorded = dict(ps.chart_floor_abstention)
    m._select_and_finalize_page(state, 1, saved_text=body)
    assert ps.chart_floor_abstention == recorded


def test_a_body_the_last_selection_replaced_with_a_chart_clears_the_abstention(
    tmp_path, monkeypatch
):
    state = _chart_floor_document_state(_framed_thin_chart(tmp_path))
    ps = state.pages[1]
    body = m._select_and_finalize_page(state, 1).output.text
    assert ps.chart_floor_abstention is not None, "premise"
    monkeypatch.setattr(m, "chart_floor_replacement", lambda *a, **kw: lambda: "a chart body")
    m._select_and_finalize_page(state, 1)
    ps.chart_floor_abstention = {"outcome": "stale"}
    m._select_and_finalize_page(state, 1, saved_text=body)
    assert ps.chart_floor_abstention is None


def test_bare_floor_body_plus_appended_prose_clears_the_abstention(tmp_path):
    state = _chart_floor_document_state(_framed_thin_chart(tmp_path))
    ps = state.pages[1]
    body = m._select_and_finalize_page(state, 1).output.text
    m._select_and_finalize_page(state, 1, saved_text=body)
    assert ps.chart_floor_abstention is not None, "control: the bare body keeps it"
    # #1266 round 5: the appended prose is no body a floor certified, so it does not ship;
    # the bare floor body does, and the abstention describing it stays.
    shipped = m._select_and_finalize_page(
        state, 1, saved_text=body + "\n\nProse nobody checked."
    ).output.text
    assert "Prose nobody checked" not in shipped and shipped == body
    assert ps.chart_floor_abstention is not None


def test_bare_floor_shape_is_the_marker_and_the_page_image_only():
    marker = "[page 3 failed: unverifiable table \u2014 see image]"
    image = "![Failed table page 3](figures/failed_table_p3.png)"
    assert m._is_bare_floor_body(f"{marker}\n\n{image}", 3, image)
    assert m._is_bare_floor_body(marker, 3, "")
    assert not m._is_bare_floor_body(f"{marker}\n\n{image}\nextra prose", 3, image)
    assert not m._is_bare_floor_body(f"{marker}\n\n{image}", 4, image)


def test_a_rotated_page_is_recorded_as_declined_for_rotation(tmp_path):
    pdf = _framed_thin_chart(tmp_path)
    upright = _chart_floor_document_state(pdf)
    m._select_and_finalize_page(upright, 1)
    rotated = _chart_floor_document_state(pdf)
    rotated.pages[1].page_rotation = 90
    m._select_and_finalize_page(rotated, 1)
    assert upright.pages[1].chart_floor_abstention["outcome"] == m.CHART_FLOOR_NO_BBOXES
    record = rotated.pages[1].chart_floor_abstention
    assert record["outcome"] == m.CHART_FLOOR_PAGE_ROTATED
    assert record["probe_outcome"] == m.CHART_FLOOR_NO_BBOXES
    assert record["census_chart_marks"] is True


def test_a_later_finalize_that_takes_another_branch_clears_the_abstention(
    tmp_path, _hermetic_judge
):
    """An earlier decline must not outlive the bytes it described (same process).

    The first finalize declines (abstention set). The one thing then changed is table
    evidence, which sends the page down another branch; the second finalize, with no
    saved body, must leave no abstention, and the sidecar must carry no key.
    """
    pdf = _framed_thin_chart(tmp_path)
    state = _chart_floor_document_state(pdf)
    ps = state.pages[1]
    m._select_and_finalize_page(state, 1)
    assert ps.chart_floor_abstention is not None, "premise: the floor declined"

    ps.detected_table_count = 1
    m._select_and_finalize_page(state, 1)
    assert ps.chart_floor_abstention is None
    assert ps.chart_floor_probe is not None, "the probe ran again; only the record was reset"

    pipeline = _preservation_pipeline()
    out_dir = tmp_path / "later"
    pipeline._flush_page_fragment(state, 1, "x", out_dir)
    sidecar = pipeline._flush_page_sidecar(state, 1, out_dir, terminal=True)
    assert "chart_floor_abstention" not in json.loads(sidecar.read_text())


def test_empty_body_records_empty_body_with_probe_counts(tmp_path, monkeypatch):
    pdf = _filled_bars_no_ticks(tmp_path)
    monkeypatch.setattr(
        m,
        "chart_floor_replacement",
        lambda state, n, p, **kw: (
            m._record_chart_floor_probe(p, {"outcome": "panels_read", "regions": 2, "panels": 1})
            or (lambda: "")
        ),
    )
    ps, _ = _finalize(pdf)
    assert ps.chart_floor_abstention == {
        "outcome": m.CHART_FLOOR_EMPTY_BODY,
        "regions": 2,
        "panels": 1,
    }


def test_no_probe_record_means_probe_not_run(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "chart_floor_replacement", lambda *a, **kw: None)
    ps, _ = _finalize(_framed_thin_chart(tmp_path))
    assert ps.chart_floor_abstention == {"outcome": m.CHART_FLOOR_PROBE_NOT_RUN}


@pytest.mark.parametrize("builder", [_framed_thin_chart, _filled_bars_no_ticks])
def test_recording_changes_neither_the_return_contract_nor_the_page_bytes(
    tmp_path, monkeypatch, builder
):
    pdf = builder(tmp_path)
    recorded_ps, recorded_text = _finalize(pdf)
    recorded_return, _ = _probe(pdf, tmp_path)

    monkeypatch.setattr(m, "_record_chart_floor_probe", lambda *a, **k: None)
    monkeypatch.setattr(m, "_record_chart_floor_abstention", lambda *a, **k: None)
    off_ps, off_text = _finalize(pdf)
    off_return, off_probe_page = _probe(pdf, tmp_path)

    assert recorded_text == off_text
    assert (recorded_return is None) == (off_return is None)
    assert recorded_ps.chart_floor_abstention is not None
    assert off_ps.chart_floor_abstention is None
    assert getattr(off_probe_page, "chart_floor_probe", None) is None


# --------------------------------------------------------------------------------------
# Persistence: sparse key, restore, live value wins, byte identity.
# --------------------------------------------------------------------------------------


@pytest.fixture
def _hermetic_judge(monkeypatch):
    from socr.pipeline.orchestrator import UnifiedPipeline

    monkeypatch.setattr(UnifiedPipeline, "_resolve_judge_model", lambda self, *a, **k: "")


def _flushed(tmp_path, pdf, out_name="out"):
    out_dir = tmp_path / out_name
    pipeline = _preservation_pipeline()
    state = _chart_floor_document_state(pdf)
    body = m._select_and_finalize_page(state, 1).output.text
    state.pages[1].best_output = m.replace(state.pages[1].best_output, text=body)
    pipeline._flush_page_fragment(state, 1, body, out_dir)
    sidecar = pipeline._flush_page_sidecar(state, 1, out_dir, terminal=True)
    return pipeline, state, body, out_dir, sidecar


def test_sidecar_carries_the_abstention(tmp_path, _hermetic_judge):
    _, state, _, _, with_key = _flushed(tmp_path, _framed_thin_chart(tmp_path), "a")
    assert json.loads(with_key.read_text())["chart_floor_abstention"] == (
        state.pages[1].chart_floor_abstention
    )


def test_restore_puts_it_back_and_a_live_value_wins(tmp_path, _hermetic_judge):
    from socr.core.state import DocumentHandle, DocumentState

    pdf = _framed_thin_chart(tmp_path)
    pipeline, state, body, out_dir, _ = _flushed(tmp_path, pdf)
    written = state.pages[1].chart_floor_abstention

    resumed = DocumentState(handle=DocumentHandle.from_path(pdf))
    pipeline._restore_terminal_page_state(resumed, 1, state.pages[1].best_output, out_dir)
    assert resumed.pages[1].chart_floor_abstention == written

    live = DocumentState(handle=DocumentHandle.from_path(pdf))
    live.pages[1].chart_floor_abstention = {"outcome": "live"}
    pipeline._restore_terminal_page_state(live, 1, state.pages[1].best_output, out_dir)
    assert live.pages[1].chart_floor_abstention == {"outcome": "live"}


def test_write_restore_then_saved_text_finalize_keeps_the_abstention(tmp_path, _hermetic_judge):
    """Floor pages are ERROR, so the ledger skip does not apply (Kimi 6): drive the
    restorer directly, then re-finalize under the saved body, as assemble does.

    HONEST SCOPE: ``_flush_page_sidecar`` finalizes with no saved body, so it resets the
    record and recomputes it from source (the floor sets it again only if it declines
    again). A resumed state rebuilt from the sidecar alone has no native fields, so a
    sidecar re-flush of it carries no abstention. That is the intended arm: the record
    describes the bytes of the selection that wrote it. What survives a resume is the
    saved-text finalization, pinned here, and the restorer itself.
    """
    from socr.core.state import DocumentHandle, DocumentState

    pdf = _framed_thin_chart(tmp_path)
    pipeline, state, body, out_dir, sidecar = _flushed(tmp_path, pdf)
    written = json.loads(sidecar.read_text())["chart_floor_abstention"]
    assert written

    resumed = DocumentState(handle=DocumentHandle.from_path(pdf))
    best = state.pages[1].best_output
    pipeline._restore_terminal_page_state(resumed, 1, best, out_dir)
    rps = resumed.pages[1]
    assert rps.chart_floor_abstention == written
    rps.best_output = best
    rps.attempts = [best]
    m._select_and_finalize_page(resumed, 1, saved_text=body)
    assert rps.chart_floor_abstention == written
    # The one thing changed: no saved body. The record is recomputed, and this state has
    # no floor inputs, so nothing declines and nothing is recorded.
    m._select_and_finalize_page(resumed, 1)
    assert rps.chart_floor_abstention is None
