"""Round 1b: #1206's per-cell native table check abstains on a suspect layer.

Agreed with the owner of #1206 (the socr bot). The per-cell check
(``tables.native_cell_check``, wired at ``_run_native_cell_binding_check``) compares each
emitted cell against the page's OWN text layer. On a page whose layer may not serve as
evidence -- ``state.layer_is_admissible_evidence``, the UNION, so a FAILED detection scan
is included -- it must abstain COMPLETELY: no ``native_cell_binding`` verdicts, no
``native_cell_check_*`` events, no distrust in ``tables_trust.json``, and no demotion
under ``--native-cell-gating``.

Two paths, because a verdict has two ways to reach a page:

* the LIVE run, which #1206 already gates at the top of the check (this file is the test
  its owner asked for, and it is what makes that gate killable);
* the RESUME, which it did not: a sidecar from an earlier run restored its per-cell
  verdicts and replayed their distrust events, so the page came back listed untrusted on
  the strength of a check the live run had refused to run.

Hermetic: no provider, no model, no OCR. ``process()`` is never entered.
"""

from __future__ import annotations

import json
from pathlib import Path

import fitz
import pytest
from ocr_output_contract import doc_dir_for, relative_key
from test_1205_table_chain_evidence_readers import _flag, _make_config, _make_pipeline, _make_state

from socr.core.result import PageOutput, PageStatus
from socr.core.tables_trust import TABLE_DISTRUST_KINDS, build_tables_trust
from socr.pipeline.orchestrator import UnifiedPipeline

_MD = "| Item | OLS | IV |\n| --- | --- | --- |\n| RowA | 350 | 200 |\n| RowB | 300 | 400 |\n"

#: The three flag states every pin here runs over: admissible, a FAILED scan ("unknown"),
#: and a POSITIVE finding. The union covers the last two.
_ARMS = (
    ("clean", {"positive": False}),
    ("failed scan", {"positive": False, "scan_failed": True}),
    ("positive finding", {"positive": True}),
)


def test_loaded_source_is_this_checkout() -> None:
    import socr

    assert Path(socr.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1] / "src")


def test_the_cell_check_kinds_are_distrust_and_are_replayed_on_resume() -> None:
    """Why both paths matter, read from the registries rather than assumed.

    Three of the four kinds are in ``TABLE_DISTRUST_KINDS``, so one replayed event lists
    the page in ``tables_trust.json``; the fourth is a PASS, which a suspect layer may
    not give either. All four are in ``resume_restore_kinds``, which is what makes the
    resume path a second emit site rather than a copy of the first.
    """
    kinds = UnifiedPipeline.native_cell_check_event_kinds()
    assert kinds, "the kind set is empty, so every assertion below is vacuous"
    assert kinds & TABLE_DISTRUST_KINDS, "no cell-check kind reaches tables_trust"
    assert kinds <= UnifiedPipeline.resume_restore_kinds(), (
        "a cell-check kind is not replayed on resume, so the resume pin covers less than it claims"
    )


def _born_digital_table_pdf(tmp_path: Path, name: str = "bd.pdf") -> Path:
    """A born-digital ruled table whose text layer carries the same values the model
    emits, so the per-cell check has something to verify and the clean arm PASSES."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    page = doc.new_page()
    cols = [100, 180, 250]
    rows = [120, 142, 164]
    kw = {"fontsize": 9}
    for i, head in enumerate(("Item", "OLS", "IV")):
        page.insert_text((cols[i] + 4, rows[0] + 12), head, **kw)
    for r, (label, a, b) in enumerate((("RowA", "350", "200"), ("RowB", "300", "400"))):
        page.insert_text((cols[0] + 4, rows[r + 1] + 12), label, **kw)
        page.insert_text((cols[1] + 4, rows[r + 1] + 12), a, **kw)
        page.insert_text((cols[2] + 4, rows[r + 1] + 12), b, **kw)
    for yy in [*rows, rows[-1] + 22]:
        page.draw_line((100, yy), (330, yy))
    for xx in [*cols, 330]:
        page.draw_line((xx, rows[0]), (xx, rows[-1] + 22))
    pdf_path = tmp_path / name
    doc.save(pdf_path)
    doc.close()
    return pdf_path


def _live_arm(tmp_path: Path, *, gating: bool, flags: dict):
    pipeline = _make_pipeline(_make_config(native_cell_gating=gating))
    state = _make_state(_born_digital_table_pdf(tmp_path))
    ps = state.pages[1]
    ps.is_born_digital = True
    ps.has_tables = True
    ps.detected_table_count = 1
    _flag(ps, **flags)
    bo = PageOutput(
        page_num=1, text=_MD, status=PageStatus.SUCCESS, engine="qwen", audit_passed=True
    )
    ps.best_output = bo
    pipeline._run_native_cell_binding_check(state, 1, ps, bo)
    kinds = UnifiedPipeline.native_cell_check_event_kinds()
    return {
        "events": sorted({e.kind for e in state.events if e.kind in kinds}),
        "bindings": sorted(getattr(ps, "native_cell_binding", None) or {}),
        "disposition": getattr(ps, "table_ladder_disposition", None),
        "untrusted": build_tables_trust(state.handle.filename, state.events).untrusted_pages,
    }


@pytest.mark.parametrize("gating", [False, True], ids=["gating-off", "gating-on"])
def test_a_flagged_page_records_no_cell_check_at_all(tmp_path: Path, gating: bool) -> None:
    """The test #1206's owner asked for: both gating states, with the clean control.

    ``--native-cell-gating`` only decides whether the page is DEMOTED; the verdicts, the
    events and the distrust are recorded either way. So a gate that only suppressed the
    demotion would leave the layer's reading in the sidecar and in
    ``tables_trust.json``, which is why both states are run and all four outputs are
    asserted, not just the demotion.
    """
    clean = _live_arm(tmp_path / f"clean-{gating}", gating=gating, flags={"positive": False})
    assert clean["events"], (
        "control: the per-cell check records nothing even on an admissible page, so the "
        "flagged arms below prove nothing"
    )
    assert clean["bindings"] == ["p1-t0"]

    for label, flags in _ARMS[1:]:
        arm = _live_arm(
            tmp_path / f"{label}-{gating}".replace(" ", "-"), gating=gating, flags=flags
        )
        assert arm["events"] == [], f"{label}: cell-check events recorded on a flagged page"
        assert arm["bindings"] == [], f"{label}: a per-cell verdict was recorded"
        assert arm["disposition"] is None, f"{label}: the page was demoted"
        assert arm["untrusted"] == [], f"{label}: the page is listed untrusted"


def _resume_arm(tmp_path: Path, *, flags: dict, carries_flag: bool):
    """Resume one page from a sidecar that DOES carry per-cell verdicts and events."""
    pipeline = _make_pipeline(_make_config())
    pdf = _born_digital_table_pdf(tmp_path)
    state = _make_state(pdf)
    ps = state.pages[1]
    _flag(ps, **flags)

    output_dir = tmp_path / "out"
    doc_dir = doc_dir_for(output_dir, relative_key(pdf, pdf.parent))
    (doc_dir / "pages").mkdir(parents=True, exist_ok=True)
    sidecar: dict = {
        "native_cell_binding": {"p1-t0": {"table_id": "p1-t0", "outcome": "contradicted"}},
        "audit_events": [
            {
                "page_num": 1,
                "kind": "native_cell_check_contradicted",
                "engine": "qwen",
                "detail": "table p1-t0: cell binding contradicted",
                "data": {},
            }
        ],
    }
    if carries_flag:
        # What #1156 Part A actually persists for a flagged page, and what the restore
        # reads back BEFORE the two sites gated here.
        sidecar["prior_ocr_layer_suspect"] = True
    (doc_dir / "pages" / "00001.json").write_text(json.dumps(sidecar), encoding="utf-8")

    page_out = PageOutput(
        page_num=1, text=_MD, status=PageStatus.SUCCESS, engine="qwen", audit_passed=True
    )
    pipeline._restore_terminal_page_state(state, 1, page_out, output_dir)
    kinds = UnifiedPipeline.native_cell_check_event_kinds()
    trust = build_tables_trust(state.handle.filename, state.events)
    return {
        "events": sorted({e.kind for e in state.events if e.kind in kinds}),
        "bindings": sorted(getattr(ps, "native_cell_binding", None) or {}),
        "untrusted": trust.untrusted_pages,
        "advisory_native_cell": sorted(trust.advisory_native_cell_pages),
    }


def test_a_resume_does_not_bring_the_cell_verdicts_back(tmp_path: Path) -> None:
    """The second emit site, which #1206's live gate does not reach.

    A verdict has two ways onto a page, and the live run is only one of them. Measured
    before the fix: the flagged arm came back with the restored verdict, the replayed
    ``native_cell_check_contradicted`` event, and the page in ``untrusted_pages`` -- a
    per-cell check against the suspect layer deciding the page after all, one run later.

    Nothing is deleted: the sidecar keeps what its run wrote. This only refuses to act
    on it.
    """
    clean = _resume_arm(tmp_path / "clean", flags={"positive": False}, carries_flag=False)
    assert clean["events"] == ["native_cell_check_contradicted"], (
        "control: nothing is replayed even on an admissible page, so the flagged arm "
        "below proves nothing"
    )
    assert clean["bindings"] == ["p1-t0"]
    assert clean["untrusted"] == []
    assert clean["advisory_native_cell"] == [1]

    for label, flags in _ARMS[1:]:
        arm = _resume_arm(tmp_path / label.replace(" ", "-"), flags=flags, carries_flag=True)
        assert arm["events"] == [], f"{label}: a cell-check event was replayed"
        assert arm["bindings"] == [], f"{label}: a per-cell verdict was restored"
        assert arm["untrusted"] == [], f"{label}: the page came back listed untrusted"
        assert arm["advisory_native_cell"] == [], f"{label}: advisory native-cell distrust replayed"


def test_the_resume_reads_suspicion_from_both_the_sidecar_and_this_run() -> None:
    """Replaced in round 2 (Argon P1-4). What this pinned has been fixed underneath it.

    It used to demonstrate the restore ORDER by showing that a sidecar carrying the
    verdicts but NOT the suspicion flag restored them, while the same sidecar with the
    flag did not. That first arm was the bug: the restore ASSIGNED
    ``prior_ocr_layer_suspect`` from the sidecar, so a stale False -- anything written by
    a build that could not see the flag -- cleared the suspicion ``_phase_analyze`` had
    just established from the PDF, and every gate in the file then read the page as
    admissible.

    It now takes the UNION of the sidecar and this run's live analysis, so either source
    alone is enough. Both arms are pinned, with the neither-source arm as the control
    that the restore still works at all.
    """
    import pytest as _pytest

    tmp = Path("scratch/1205r2/tmp_order")

    both_absent = _resume_arm(tmp / "neither", flags={"positive": False}, carries_flag=False)
    assert both_absent["bindings"] == ["p1-t0"], (
        "control: nothing is restored even on an admissible page, so the arms below prove nothing"
    )

    live_only = _resume_arm(tmp / "live-only", flags={"positive": True}, carries_flag=False)
    assert live_only["bindings"] == [], (
        "a stale sidecar False cleared the suspicion this run's own analysis found"
    )

    sidecar_only = _resume_arm(tmp / "sidecar-only", flags={"positive": False}, carries_flag=True)
    assert sidecar_only["bindings"] == [], (
        "the sidecar's own flag is not reaching the gates, so the restore order is wrong"
    )
    del _pytest
