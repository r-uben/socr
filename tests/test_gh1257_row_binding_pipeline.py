"""#1257 at ``process()``: the new row-binding evidence as DIFFERENCES between two arms.

Both arms run the same pipeline on the same synthetic PDF with the same fake model output; the
control arm patches the new evidence out (``evidence_off``), so the only thing that changes between
arms is the evidence.  Both provider states are driven (CLAUDE.md, the no-provider trap): no absolute
page status is pinned, only what the evidence changes and what it must not.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

pytest.importorskip("fitz")

from test_gh1257_row_binding_evidence import (  # noqa: E402
    EVIDENCE_WRAPPED_LABEL,
    WRAP_ABOVE,
    WRAP_ABOVE_MD,
    V,
    evidence_off,
    layout,
    page_of,
    table_md,
)
from test_gh1267_row_binding_pipeline import (  # noqa: E402
    CONTRADICTED_KIND,
    UNVERIFIABLE_KIND,
    UNVERIFIABLE_MARK,
    _assembled,
    _events,
    _fragment,
    _run,
    _sidecar,
)

from socr.core.document import DocumentHandle  # noqa: E402
from socr.core.result import PageOutput, PageStatus  # noqa: E402
from socr.core.state import DocumentState  # noqa: E402
from socr.pipeline.orchestrator import UnifiedPipeline  # noqa: E402

PROVIDERS = [True, False]


def _pdf(tmp_path: Path, items, rotation: int, name: str) -> Path:
    path = tmp_path / f"{name}_{rotation}.pdf"
    page_of(items, rotation).parent.save(path)
    return path


def _arms(tmp_path, markdown: str, items, rotation: int, providers: bool):
    pdf = _pdf(tmp_path, items, rotation, "wrapped")
    kw = dict(markdown=markdown, pdf=pdf, providers=providers)
    _, res_on, state_on, on, _ = _run(
        tmp_path, None, rotation, check_runs=True, out_name="on", **kw
    )
    with evidence_off():
        _, res_off, _, off, _ = _run(
            tmp_path, None, rotation, check_runs=True, out_name="off", **kw
        )
    return res_on, state_on, on, res_off, off


@pytest.mark.parametrize("providers", PROVIDERS)
@pytest.mark.parametrize("rotation", [0, 90])
def test_a_wrapped_label_table_loses_only_its_mark(tmp_path, rotation, providers):
    markdown = table_md(WRAP_ABOVE_MD)
    res_on, _, on, res_off, off = _arms(tmp_path, markdown, layout(WRAP_ABOVE), rotation, providers)
    if not providers:
        return _inert_without_providers(on, off, res_on, res_off)
    side_on, side_off = _sidecar(on), _sidecar(off)
    rec_on, rec_off = side_on["row_binding"]["p1-t0"], side_off["row_binding"]["p1-t0"]

    # the verdict, with its evidence, in the record: terminal in the sidecar
    assert rec_on["outcome"] == "bound" and rec_off["outcome"] == "unverifiable"
    assert rec_on["rows_bound"] == 4 and rec_on["rows_total"] == 4
    assert rec_on["evidence"] == {EVIDENCE_WRAPPED_LABEL: 1} and "evidence" not in rec_off
    assert rec_on["rows"] == []
    assert [r["md_line"] for r in rec_off["rows"]] == [3]  # exactly the wrapped row

    # the marks: only the control arm carries the mark and its audit event
    assert _fragment(on).count(UNVERIFIABLE_MARK) == 0
    assert _fragment(off).count(UNVERIFIABLE_MARK) == 1
    assert _events(on, UNVERIFIABLE_KIND) == [] and len(_events(off, UNVERIFIABLE_KIND)) == 1
    assert _events(on, CONTRADICTED_KIND) == []
    assert "row_binding_unverifiable_tables" not in side_on
    assert side_off["row_binding_unverifiable_tables"] == [0]
    assert side_on["winning_output"]["status"] != PageStatus.WARNING.value or (
        side_off["winning_output"]["status"] == PageStatus.WARNING.value
    )
    assert side_off["winning_output"]["failure_mode"] == "table_unverified"
    assert side_on["winning_output"]["failure_mode"] != "table_unverified"

    # the bytes: the table ships identically in both arms (the evidence changes the mark, not text)
    table = table_md(WRAP_ABOVE_MD).rstrip("\n")
    assert table in _fragment(on) and table in _fragment(off)
    assert table in _assembled(on) and table in _assembled(off)
    assert (res_on.status == res_off.status) or (res_off.status is not PageStatus.SUCCESS)


@pytest.mark.parametrize("providers", PROVIDERS)
def test_a_slid_wrapped_row_is_marked_in_both_arms(tmp_path, providers):
    """The values of the wrapped row moved one row down: the evidence must not bind it, so the two
    arms record the same outcome, the same rows and the same mark."""
    slid = [WRAP_ABOVE_MD[0], (WRAP_ABOVE_MD[1][0], V[2]), ("Receipts", V[1]), WRAP_ABOVE_MD[3]]
    res_on, _, on, res_off, off = _arms(tmp_path, table_md(slid), layout(WRAP_ABOVE), 0, providers)
    if not providers:
        return _inert_without_providers(on, off, res_on, res_off)
    rec_on = _sidecar(on)["row_binding"]["p1-t0"]
    rec_off = _sidecar(off)["row_binding"]["p1-t0"]
    assert rec_on["outcome"] == rec_off["outcome"] != "bound"
    assert rec_on["rows"] == rec_off["rows"]
    assert "evidence" not in rec_on
    assert _fragment(on) == _fragment(off)
    assert _assembled(on) == _assembled(off)
    assert res_on.status == res_off.status


def _inert_without_providers(on, off, res_on, res_off) -> None:
    """With no provider the loop bails before routing (no reading, so no check): the evidence has
    nothing to change.  Pinned as a difference of zero, not as an absolute status."""
    side_on, side_off = _sidecar(on), _sidecar(off)
    assert "row_binding" not in side_on and "row_binding" not in side_off

    def shipped(out: Path) -> list[str]:  # every markdown the run wrote, page fragments included
        return sorted(p.read_text(encoding="utf-8") for p in out.rglob("*.md"))

    assert shipped(on) == shipped(off)
    assert res_on.status == res_off.status
    assert _events(on, UNVERIFIABLE_KIND) == _events(off, UNVERIFIABLE_KIND) == []


def _restore(pipeline, state, out_dir, pdf):
    """Restore page 1's record from its sidecar into a state that carries none of it."""
    assert pipeline._flush_page_sidecar(state, 1, out_dir, terminal=True) is not None
    resumed = DocumentState(handle=DocumentHandle(path=pdf, page_count=1))
    resumed.pages[1] = copy.deepcopy(state.pages[1])
    UnifiedPipeline.rollback_row_binding_page_state(resumed, 1, resumed.pages[1])
    restored = PageOutput(
        page_num=1, text="x", status=PageStatus.SUCCESS, engine="qwen", audit_passed=True
    )
    pipeline._restore_terminal_page_state(resumed, 1, restored, out_dir)
    return resumed


@pytest.mark.parametrize("providers", PROVIDERS)
def test_the_bound_record_and_its_evidence_survive_flush_and_restore(tmp_path, providers):
    pdf = _pdf(tmp_path, layout(WRAP_ABOVE), 0, "wrapped")
    pipeline, _, state, out, _ = _run(
        tmp_path,
        None,
        0,
        check_runs=True,
        out_name="on",
        markdown=table_md(WRAP_ABOVE_MD),
        pdf=pdf,
        providers=providers,
    )
    if not providers:
        assert not state.pages[1].row_binding  # no reading, no check, no record
        return
    before = copy.deepcopy(state.pages[1].row_binding)
    assert before["p1-t0"]["evidence"] == {EVIDENCE_WRAPPED_LABEL: 1}
    resumed = _restore(pipeline, state, out, pdf)
    ps = resumed.pages[1]
    assert ps.row_binding == before  # outcome, counts and evidence, as recorded
    assert ps.row_binding_unverifiable_tables == [] and ps.row_binding_contradicted_tables == []
    assert ps.row_binding_mark_only is False
    assert [e.kind for e in resumed.events if e.page_num == 1].count(UNVERIFIABLE_KIND) == 0
    assert ps.unchecked_table_kind == ""


@pytest.mark.parametrize("providers", PROVIDERS)
def test_a_resumed_run_ships_the_same_bytes_and_the_same_record(tmp_path, providers):
    from ocr_output_contract import RootIndex

    pdf = _pdf(tmp_path, layout(WRAP_ABOVE), 0, "wrapped")
    kw = dict(markdown=table_md(WRAP_ABOVE_MD), pdf=pdf, providers=providers)
    _, _, _, out, _ = _run(tmp_path, None, 0, check_runs=True, out_name="out", **kw)
    frag, doc, side = _fragment(out), _assembled(out), _sidecar(out)
    Path(RootIndex(out).path).unlink()  # force the page-level path: the sidecar is the source
    _run(tmp_path, None, 0, check_runs=True, out_name="out", **kw)
    assert _fragment(out) == frag and _assembled(out) == doc
    again = _sidecar(out)
    assert again.get("row_binding") == side.get("row_binding")
    assert bool(side.get("row_binding")) is providers
    assert again["winning_output"]["status"] == side["winning_output"]["status"]
    assert again["winning_output"]["failure_mode"] == side["winning_output"]["failure_mode"]


@pytest.mark.parametrize("providers", PROVIDERS)
def test_a_contradicted_table_is_withheld_in_both_arms(tmp_path, providers):
    """Convictions are untouched: the table a shifted wrapped row convicts is withheld whether or not
    the new evidence runs, with the same rows."""
    shifted = [
        WRAP_ABOVE_MD[0],
        (WRAP_ABOVE_MD[1][0], []),
        ("Receipts", V[1]),
        ("", V[2]),
        ("Outlays", V[3]),
    ]
    res_on, _, on, res_off, off = _arms(
        tmp_path, table_md(shifted), layout(WRAP_ABOVE), 0, providers
    )
    if not providers:
        return _inert_without_providers(on, off, res_on, res_off)
    rec_on = _sidecar(on)["row_binding"]["p1-t0"]
    rec_off = _sidecar(off)["row_binding"]["p1-t0"]
    assert rec_on["outcome"] == rec_off["outcome"] == "contradicted"
    assert (
        rec_on["rows"] == rec_off["rows"]
        and rec_on["withheld_markdown"] == rec_off["withheld_markdown"]
    )
    assert len(_events(on, CONTRADICTED_KIND)) == len(_events(off, CONTRADICTED_KIND)) == 1
    assert _fragment(on) == _fragment(off)
    assert res_on.status == res_off.status
