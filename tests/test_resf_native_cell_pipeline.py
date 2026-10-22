"""End-to-end: RESF born-digital tables + default-on native cell check."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("fitz")

from socr.core.config import EngineType, PipelineConfig  # noqa: E402
from socr.core.manifest import UNVERIFIED_TABLE_MARK  # noqa: E402
from socr.core.providers import PROFILE_QWEN_LOCAL  # noqa: E402
from socr.core.result import DocumentStatus, PageStatus  # noqa: E402
from socr.pipeline.orchestrator import UnifiedPipeline  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures" / "resf_native_cell"

_NATIVE_CELL_AUDIT_PREFIX = "native_cell_check_"


def _page_sidecar(out_dir: Path, page_num: int) -> dict:
    sidecar = next(out_dir.rglob(f"pages/{page_num:05d}.json"))
    return json.loads(sidecar.read_text(encoding="utf-8"))


def _tables_trust(out_dir: Path) -> dict:
    path = next(out_dir.rglob("tables_trust.json"))
    return json.loads(path.read_text(encoding="utf-8"))


def _audit_events(out_dir: Path) -> list[dict]:
    path = next(out_dir.rglob("audit_log.json"))
    return json.loads(path.read_text(encoding="utf-8")).get("events", [])


def _run_page(
    tmp_path: Path,
    pdf_name: str,
    md: str,
    *,
    page_has_tables: bool | None = None,
    locate_tables_empty: bool = False,
    native_cell_gating: bool = False,
    skip_native_cell_check: bool = False,
):
    from socr.core.result import PageOutput
    from socr.pipeline.agentic import PageDecision, ProviderAttempt

    pdf = FIXTURES / pdf_name if not Path(pdf_name).is_absolute() else Path(pdf_name)
    out_dir = tmp_path / pdf.name.replace(".pdf", "_out")
    cfg = PipelineConfig(
        primary_engine=EngineType.QWEN,
        agentic=True,
        strict_local=True,
        judge_backend="heuristic",
        enabled_engines=[EngineType.QWEN],
        table_judge_ladder=False,
        quiet=True,
        save_figures=False,
        write_manifest=True,
        native_cell_gating=native_cell_gating,
    )
    pipeline = UnifiedPipeline(cfg)

    def _fake_route(page_num, ladder, run_provider, judge, **kwargs):
        out = PageOutput(
            page_num=page_num,
            text=md,
            status=PageStatus.SUCCESS,
            engine="qwen",
            audit_passed=True,
        )
        prof = ladder[0]
        att = ProviderAttempt(
            engine=prof.engine,
            output=out,
            cost_usd=prof.cost_per_page_usd,
            accepted=True,
            reason="ok",
            provider_id=prof.id,
            model=prof.model,
            backend=prof.backend,
        )
        return PageDecision(page_num=page_num, final_output=out, attempts=[att], accepted=True)

    captured: dict = {}
    original_assemble = pipeline._phase_assemble

    def _assemble_wrapper(state, od):
        captured["state"] = state
        return original_assemble(state, od)

    with contextlib.ExitStack() as stack:
        stack.enter_context(patch("socr.pipeline.orchestrator.route_page", side_effect=_fake_route))
        stack.enter_context(
            patch.object(
                pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]
            )
        )
        stack.enter_context(patch.object(pipeline, "_resolve_judge_model", return_value=""))
        stack.enter_context(patch.object(pipeline, "_plan_native_table_first", return_value=None))
        stack.enter_context(patch.object(pipeline, "_build_table_judge_rungs", return_value=[]))
        stack.enter_context(patch.object(pipeline, "_is_chart_asset_page", return_value=False))
        stack.enter_context(
            patch.object(pipeline, "_reconcile_chart_table_grids", lambda *a, **k: 0)
        )
        stack.enter_context(
            patch.object(pipeline, "_suppress_chart_table_skeletons", lambda *a, **k: 0)
        )
        stack.enter_context(patch("socr.tables.reconstruct.chart_region_bboxes", return_value=[]))
        if page_has_tables is False:
            stack.enter_context(patch.object(pipeline, "_page_has_tables", return_value=False))
        if locate_tables_empty:
            stack.enter_context(patch("socr.tables.locate.locate_tables", return_value=[]))
        if skip_native_cell_check:
            stack.enter_context(
                patch.object(pipeline, "_run_native_cell_binding_check", lambda *a, **k: None)
            )
        stack.enter_context(
            patch.object(pipeline, "_phase_assemble", side_effect=_assemble_wrapper)
        )
        result = pipeline.process(pdf, out_dir)

    return pipeline, result, captured["state"], out_dir


def _winning(sidecar: dict) -> dict:
    return sidecar["winning_output"]


def _assert_advisory_no_demotion(sidecar: dict, ps) -> None:
    win = _winning(sidecar)
    assert win["status"] == PageStatus.SUCCESS.value
    assert win.get("failure_mode") in (None, "", "none")
    assert UNVERIFIED_TABLE_MARK not in win["text"]
    assert ps.table_ladder_disposition is None
    assert not getattr(ps, "unchecked_table_kind", "")


def test_correct_table_verified_end_to_end_p59(tmp_path: Path) -> None:
    md = (FIXTURES / "p59_tableau1.md").read_text(encoding="utf-8")
    _pipe, result, state, out_dir = _run_page(tmp_path, "resf_p59.pdf", md)
    ps = state.pages[1]
    sidecar = _page_sidecar(out_dir, 1)
    win = sidecar["winning_output"]

    assert result.status in (DocumentStatus.SUCCESS, DocumentStatus.AUDIT_FAILED)
    assert ps.native_cell_binding
    assert all(entry.get("outcome") == "verified" for entry in ps.native_cell_binding.values()), (
        ps.native_cell_binding
    )
    assert win["status"] == PageStatus.SUCCESS.value
    assert UNVERIFIED_TABLE_MARK not in win["text"]
    kinds = [e["kind"] for e in _audit_events(out_dir) if e.get("page_num") == 1]
    assert "native_cell_check_verified" in kinds
    assert "native_cell_check_skipped" not in kinds
    try:
        trust = _tables_trust(out_dir)
    except StopIteration:
        trust = {"pages": {}}
    assert "1" not in trust.get("pages", {})


def test_p67_strict_labels_ambiguous_advisory_no_demotion(tmp_path: Path) -> None:
    md = (FIXTURES / "p67_tableau8.md").read_text(encoding="utf-8")
    _pipe, _result, state, out_dir = _run_page(tmp_path, "resf_p67.pdf", md)
    ps = state.pages[1]
    sidecar = _page_sidecar(out_dir, 1)
    assert ps.native_cell_binding
    assert all(
        entry.get("outcome") in ("ambiguous", "contradicted")
        for entry in ps.native_cell_binding.values()
    )
    kinds = [e["kind"] for e in _audit_events(out_dir) if e.get("page_num") == 1]
    assert "native_cell_check_verified" not in kinds
    assert any(k.startswith(_NATIVE_CELL_AUDIT_PREFIX) for k in kinds)
    _assert_advisory_no_demotion(sidecar, ps)
    trust = _tables_trust(out_dir)
    assert "1" not in trust.get("pages", {})
    adv = trust.get("native_cell_advisory") or {}
    assert 1 in adv.get("pages", [])


def test_mutated_table_contradicted_advisory_no_demotion(tmp_path: Path) -> None:
    md = (FIXTURES / "p59_tableau1.md").read_text(encoding="utf-8")
    bad = md.replace("Solde public | -5,5 | -6,1", "Solde public | -6,1 | -5,5")
    _pipe, _result, state, out_dir = _run_page(tmp_path, "resf_p59.pdf", bad)
    sidecar = _page_sidecar(out_dir, 1)
    ps = state.pages[1]

    assert ps.native_cell_binding
    assert any(v.get("outcome") == "contradicted" for v in ps.native_cell_binding.values())
    kinds = [e["kind"] for e in _audit_events(out_dir) if e.get("page_num") == 1]
    assert "native_cell_check_contradicted" in kinds
    trust = _tables_trust(out_dir)
    assert "1" not in trust.get("pages", {})
    adv = trust["native_cell_advisory"]
    assert "native_cell_check_contradicted" in adv["counts_by_kind"]
    _assert_advisory_no_demotion(sidecar, ps)


def test_no_geometry_skips_advisory_no_demotion(tmp_path: Path) -> None:
    md = (FIXTURES / "p59_tableau1.md").read_text(encoding="utf-8")
    _pipe, _result, state, out_dir = _run_page(
        tmp_path, "resf_p59.pdf", md, locate_tables_empty=True
    )
    _assert_advisory_no_demotion(_page_sidecar(out_dir, 1), state.pages[1])


def test_mutated_table_contradicted_with_gating_demotes(tmp_path: Path) -> None:
    md = (FIXTURES / "p59_tableau1.md").read_text(encoding="utf-8")
    bad = md.replace("Solde public | -5,5 | -6,1", "Solde public | -6,1 | -5,5")
    _pipe, _result, state, out_dir = _run_page(
        tmp_path, "resf_p59.pdf", bad, native_cell_gating=True
    )
    sidecar = _page_sidecar(out_dir, 1)
    win = sidecar["winning_output"]

    assert any(
        v.get("outcome") == "contradicted" for v in state.pages[1].native_cell_binding.values()
    )
    assert win["status"] == PageStatus.WARNING.value
    assert win["failure_mode"] == "table_unverified"
    assert UNVERIFIED_TABLE_MARK in win["text"]


def test_gating_on_native_cell_distrust_in_main_trust_index(tmp_path: Path) -> None:
    """With gating, native-cell distrust must not be filed under native_cell_advisory."""
    md = (FIXTURES / "p59_tableau1.md").read_text(encoding="utf-8")
    bad = md.replace("Solde public | -5,5 | -6,1", "Solde public | -6,1 | -5,5")
    _run_page(tmp_path, "resf_p59.pdf", bad, native_cell_gating=True)
    trust = _tables_trust(tmp_path / "resf_p59_out")
    assert 1 in trust["untrusted_pages"]
    assert trust["counts_by_kind"]["native_cell_check_contradicted"] == 1
    assert "native_cell_check_contradicted" in trust["pages"]["1"]["reasons"]
    assert not trust.get("native_cell_advisory")


def test_gating_off_native_cell_distrust_stays_advisory_only(tmp_path: Path) -> None:
    """Default advisory: sidecar records native-cell under native_cell_advisory only."""
    md = (FIXTURES / "p59_tableau1.md").read_text(encoding="utf-8")
    bad = md.replace("Solde public | -5,5 | -6,1", "Solde public | -6,1 | -5,5")
    _run_page(tmp_path, "resf_p59.pdf", bad, native_cell_gating=False)
    trust = _tables_trust(tmp_path / "resf_p59_out")
    assert trust.get("untrusted_pages") == []
    assert "1" not in trust.get("pages", {})
    adv = trust["native_cell_advisory"]
    assert 1 in adv["pages"]
    assert adv["counts_by_kind"]["native_cell_check_contradicted"] == 1


def test_no_geometry_skips_with_gating_demotes(tmp_path: Path) -> None:
    md = (FIXTURES / "p59_tableau1.md").read_text(encoding="utf-8")
    _run_page(tmp_path, "resf_p59.pdf", md, locate_tables_empty=True, native_cell_gating=True)
    sidecar = _page_sidecar(out_dir=tmp_path / "resf_p59_out", page_num=1)
    win = sidecar["winning_output"]
    assert win["status"] == PageStatus.WARNING.value
    assert UNVERIFIED_TABLE_MARK in win["text"]


def test_advisory_output_matches_check_disabled_except_metadata(tmp_path: Path) -> None:
    """Default advisory mode matches disabling the check on page outcome fields."""
    md = (FIXTURES / "p59_tableau1.md").read_text(encoding="utf-8")
    bad = md.replace("Solde public | -5,5 | -6,1", "Solde public | -6,1 | -5,5")

    _a, _r, _state_adv, out_adv = _run_page(tmp_path / "adv", "resf_p59.pdf", bad)
    _b, _r, _state_off, out_off = _run_page(
        tmp_path / "off", "resf_p59.pdf", bad, skip_native_cell_check=True
    )

    side_adv = _page_sidecar(out_adv, 1)
    side_off = _page_sidecar(out_off, 1)
    win_adv = dict(_winning(side_adv))
    win_off = dict(_winning(side_off))

    assert win_adv["text"] == win_off["text"]
    assert win_adv["status"] == win_off["status"]
    assert win_adv.get("failure_mode") == win_off.get("failure_mode")

    assert side_adv.get("native_cell_binding")
    assert not side_off.get("native_cell_binding")

    kinds_adv = {
        e["kind"]
        for e in _audit_events(out_adv)
        if e.get("page_num") == 1 and e["kind"].startswith(_NATIVE_CELL_AUDIT_PREFIX)
    }
    audit_off = list(out_off.rglob("audit_log.json"))
    kinds_off: set[str] = set()
    if audit_off:
        kinds_off = {
            e["kind"]
            for e in json.loads(audit_off[0].read_text()).get("events", [])
            if e.get("page_num") == 1 and e["kind"].startswith(_NATIVE_CELL_AUDIT_PREFIX)
        }
    assert kinds_adv
    assert not kinds_off
