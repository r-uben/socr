"""#1282: ``socr-refinalize-point-drops`` on a synthetic finished run.

The reference is a live run of the SAME code with the step enabled: re-finalizing the run made
with the step disabled must give that run's page, byte for byte where bytes are the point.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch

import pytest

pytest.importorskip("fitz")

from test_gh1267_row_binding_check import md  # noqa: E402
from test_gh1267_row_binding_pipeline import (  # noqa: E402
    _assembled,
    _events,
    _fragment,
    _run,
    _sidecar,
)
from test_gh1282_point_repair_pipeline import (  # noqa: E402
    DISPLACED,
    DROPPED,
    KIND,
    NATIVE,
    SLIPPED,
    _noop,
)

from socr.pipeline import refinalize_point_drops as tool  # noqa: E402
from socr.pipeline.orchestrator import UnifiedPipeline  # noqa: E402


def _tree(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _doc_meta(out: Path) -> dict:
    """The document's own ``metadata.json`` (not the run's root index)."""
    docs = [
        m
        for m in (json.loads(p.read_text()) for p in out.rglob("metadata.json"))
        if "files" not in m
    ]
    assert len(docs) == 1
    return docs[0]


def _runs(tmp_path, rows, rotation, native=NATIVE, markdown=None):
    """``(on, off, pdf_dir)``: the live run with the step, and the run without it."""
    _, _, _, on, pdf = _run(
        tmp_path, rows, rotation, check_runs=True, out_name="on", native=native, markdown=markdown
    )
    with patch.object(UnifiedPipeline, "_repair_dropped_points", _noop):
        _, _, _, off, _ = _run(
            tmp_path,
            rows,
            rotation,
            check_runs=True,
            out_name="off",
            pdf=pdf,
            native=native,
            markdown=markdown,
        )
    return on, off, pdf.parent


def _tool(out, pdf_dir, *args, capsys=None):
    code = tool.main([str(out), "--pdf-dir", str(pdf_dir), *args])
    assert code == 0
    return capsys.readouterr().out if capsys else ""


@pytest.mark.parametrize("rotation", [0, 90])
def test_dry_run_is_the_default_and_writes_nothing(tmp_path, rotation, capsys):
    _, off, pdf_dir = _runs(tmp_path, SLIPPED, rotation)
    before = _tree(off)
    out = _tool(off, pdf_dir, capsys=capsys)
    assert _tree(off) == before
    assert "Net exports" in out and "-7 -> -.7" in out
    assert "1  pages would repair, 1 cell(s)" in out and "dry run" in out


@pytest.mark.parametrize("rotation", [0, 90])
def test_apply_gives_the_page_a_live_run_with_the_step_gives(tmp_path, rotation, capsys):
    on, off, pdf_dir = _runs(tmp_path, SLIPPED, rotation)
    dry = _tool(off, pdf_dir, capsys=capsys)
    assert _fragment(off) != _fragment(on)
    stale = _doc_meta(off).get("error") or ""
    assert "text layer does not hold" in stale  # the note the tool must remove
    applied = _tool(off, pdf_dir, "--apply", capsys=capsys)
    # the dry run reported exactly the cells the run changed
    assert [l for l in dry.splitlines() if " -> " in l and " p1 " in l] == [
        l for l in applied.splitlines() if " -> " in l and " p1 " in l
    ]
    assert _fragment(off) == _fragment(on)
    assert _assembled(off) == _assembled(on)
    side_on, side_off = _sidecar(on), _sidecar(off)
    for key in ("status", "failure_mode", "audit_passed", "verdict"):
        assert side_off[key] == side_on[key], key
    assert side_off["winning_output"]["text"] == side_on["winning_output"]["text"]
    assert side_off["row_binding"] == side_on["row_binding"]
    # events: the repair event is there, the text-layer ones are gone, as in the live run
    assert _events(off, KIND) == _events(on, KIND) and len(_events(off, KIND)) == 1
    for kind in ("numbers_disagree_with_text_layer", "numbers_dropped_from_reading"):
        assert _events(off, kind) == _events(on, kind) == []
    kinds = lambda out: [
        e["kind"] for e in json.loads(next(out.rglob("audit_log.json")).read_text())["events"]
    ]  # noqa: E731
    assert kinds(off) == kinds(on)
    # the document: notes and counts as in the live run (status is never moved back, see the tool)
    meta_on, meta_off = _doc_meta(on), _doc_meta(off)
    assert "text layer does not hold" not in (meta_on.get("error") or "")
    for key in ("error", "tables"):
        assert meta_off.get(key) == meta_on.get(key), key
    # the log names every old and new cell
    log = json.loads(next(off.rglob(tool.LOG_NAME)).read_text())
    assert log["pages"]["1"]["cells"] == [
        {"table_id": "p1-t0", "label": "Net exports", "old": "-7", "new": "-.7"}
    ]


@pytest.mark.parametrize("rotation", [0, 90])
def test_apply_twice_changes_nothing_the_second_time(tmp_path, rotation, capsys):
    _, off, pdf_dir = _runs(tmp_path, SLIPPED, rotation)
    _tool(off, pdf_dir, "--apply", capsys=capsys)
    once = _tree(off)
    out = _tool(off, pdf_dir, "--apply", capsys=capsys)
    assert _tree(off) == once
    assert "0  pages repaired" in out


# Fable's two-table case: t0 has the dropped point, t1 a label slip (`Imxorts`)
TWO_NATIVE = [
    *NATIVE[:3],
    ("Wages", ["5.1", "5.2", "5.3"]),
    ("Prices", ["6.1", "6.2", "6.3"]),
]
TWO_MARKDOWN = (
    md([NATIVE[0], ("Net exports", ["-7", "1.5", "2.5"]), NATIVE[2]])
    + "\nProse between the tables.\n\n"
    + md([("Wages", ["5.1", "5.2", "5.3"]), ("Pricex", ["6.1", "6.2", "6.3"])])
)


def _live_bytes(out: Path) -> dict:
    """What a reader and a resume see: fragment, document, sidecar (timings aside), events."""
    sidecar = {k: v for k, v in _sidecar(out).items() if k != "timings_s"}
    audit = json.loads(next(out.rglob("audit_log.json")).read_text())
    return {
        "fragment": _fragment(out),
        "document": _assembled(out),
        "sidecar": sidecar,
        "events": audit["events"],
    }


@pytest.mark.parametrize("rotation", [0, 90])
@pytest.mark.parametrize("case", ["dropped", "slipped", "two_tables"])
def test_offline_bytes_equal_live_bytes(tmp_path, rotation, case, capsys):
    """Ruling 3: the step enabled (live) and the step-off run plus the tool give the same
    fragment, document, sidecar and events -- including a table that binds completely after the
    repair (DROPPED: it stays recorded UNVERIFIABLE, no mark lifts) and Fable's two-table page.

    The two-table page has no ruled boxes; with both blocks scanning the whole page the
    shared-region refusal (review fix B) repairs nothing, live or offline.  Both sides are handed
    one box per table, the regions they get when ``locate_tables`` finds them."""
    from contextlib import nullcontext

    from test_gh1267_row_binding_check import LINE_PITCH, V0
    from test_gh1282_point_repair_review_fixes import one_box_per_table

    kw = {
        "dropped": dict(rows=DROPPED),
        "slipped": dict(rows=SLIPPED),
        "two_tables": dict(rows=None, native=TWO_NATIVE, markdown=TWO_MARKDOWN),
    }[case]
    bands = [(V0 - 10, V0 + 2.5 * LINE_PITCH), (V0 + 2.5 * LINE_PITCH, V0 + 5 * LINE_PITCH)]
    boxed = one_box_per_table(rotation, bands) if case == "two_tables" else nullcontext()
    with boxed:
        on, off, pdf_dir = _runs(tmp_path, kw.pop("rows"), rotation, **kw)
        assert _fragment(off) != _fragment(on)
        out = _tool(off, pdf_dir, "--apply", capsys=capsys)
    assert "1  pages repaired" in out
    live, offline = _live_bytes(on), _live_bytes(off)
    for key in live:
        assert offline[key] == live[key], key


@pytest.mark.parametrize("rotation", [0, 90])
def test_a_page_the_tool_leaves_alone_is_named_in_the_output_and_the_log(
    tmp_path, rotation, capsys
):
    """The one branch that still declines a proposed repair: it would leave the page with no
    clause at all (a lifted mark).  The page is NAMED, with the document, in the output and in the
    log, and nothing else is written.  The branch is reached with a published verdict that carries
    only the text-layer clauses, which the repair recomputes away."""
    from socr.core.marks import Check
    from socr.core.verdict import PageVerdict

    _, off, pdf_dir = _runs(tmp_path, SLIPPED, rotation)
    real = PageVerdict.from_dict.__func__

    def only_text_layer(cls, data):
        v = real(cls, data)
        text = (Check.TEXT_LAYER_NUMBERS, Check.TEXT_LAYER_DROPS)
        return replace(v, clauses=tuple(c for c in v.clauses if c.check in text))

    before = _tree(off)
    with patch.object(PageVerdict, "from_dict", classmethod(only_text_layer)):
        out = _tool(off, pdf_dir, "--apply", capsys=capsys)
    doc = next(off.rglob("pages")).parent.name
    assert f"{doc} p1 left alone" in out and "lifted mark" in out
    assert "0  pages repaired" in out
    log = json.loads(next(off.rglob(tool.LOG_NAME)).read_text())
    assert list(log["left_alone"]) == ["1"] and "lifted mark" in log["left_alone"]["1"]
    after = _tree(off)
    assert {k for k in after if after[k] != before.get(k)} == {
        str(next(off.rglob(tool.LOG_NAME)).relative_to(off))
    }


@pytest.mark.parametrize("rotation", [0, 90])
def test_a_contradicted_table_is_not_touched(tmp_path, rotation, capsys):
    _, off, pdf_dir = _runs(tmp_path, DISPLACED, rotation)
    before = _tree(off)
    out = _tool(off, pdf_dir, "--apply", capsys=capsys)
    assert _tree(off) == before
    assert "0  pages repaired" in out


def test_a_pdf_that_is_not_the_runs_input_is_refused(tmp_path, capsys):
    _, off, pdf_dir = _runs(tmp_path, SLIPPED, 0)
    other = tmp_path / "other"
    other.mkdir()
    # same name, different bytes
    src = next(pdf_dir.glob("*.pdf"))
    (other / src.name).write_bytes(src.read_bytes() + b"\n%x")
    before = _tree(off)
    out = _tool(off, other, "--apply", capsys=capsys)
    assert _tree(off) == before
    assert "not the run's input" in out
