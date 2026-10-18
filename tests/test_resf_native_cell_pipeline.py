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
):
    from socr.core.result import PageOutput
    from socr.pipeline.agentic import PageDecision, ProviderAttempt

    pdf = FIXTURES / pdf_name
    out_dir = tmp_path / pdf_name.replace(".pdf", "_out")
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
        elif pdf_name == "resf_p59.pdf":
            import socr.tables.locate as locate_mod

            _orig_locate = locate_mod.locate_tables

            def _p59_locate(page):
                boxes = _orig_locate(page)
                if len(boxes) >= 2:
                    return [boxes[1], boxes[0]]
                return boxes

            stack.enter_context(patch.object(locate_mod, "locate_tables", side_effect=_p59_locate))
        stack.enter_context(
            patch.object(pipeline, "_phase_assemble", side_effect=_assemble_wrapper)
        )
        result = pipeline.process(pdf, out_dir)

    return pipeline, result, captured["state"], out_dir


@pytest.mark.parametrize(
    "pdf,md_file",
    [
        ("resf_p59.pdf", "p59_tableau1.md"),
        ("resf_p67.pdf", "p67_tableau8.md"),
    ],
)
def test_correct_table_verified_end_to_end(tmp_path: Path, pdf: str, md_file: str) -> None:
    md = (FIXTURES / md_file).read_text(encoding="utf-8")
    _pipe, result, state, out_dir = _run_page(tmp_path, pdf, md)
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


def test_mutated_table_contradicted_end_to_end(tmp_path: Path) -> None:
    md = (FIXTURES / "p59_tableau1.md").read_text(encoding="utf-8")
    bad = md.replace("Solde public | -5,5 | -6,1", "Solde public | -6,1 | -5,5")
    _pipe, result, state, out_dir = _run_page(tmp_path, "resf_p59.pdf", bad)
    sidecar = _page_sidecar(out_dir, 1)
    win = sidecar["winning_output"]

    assert state.pages[1].native_cell_binding
    assert any(
        v.get("outcome") == "contradicted" for v in state.pages[1].native_cell_binding.values()
    )
    assert win["status"] == PageStatus.WARNING.value
    assert win["failure_mode"] == "table_unverified"
    assert UNVERIFIED_TABLE_MARK in win["text"]
    kinds = [e["kind"] for e in _audit_events(out_dir) if e.get("page_num") == 1]
    assert "native_cell_check_contradicted" in kinds
    trust = _tables_trust(out_dir)
    assert "1" in trust.get("pages", {})
    assert "native_cell_check_contradicted" in trust["pages"]["1"].get("reasons", [])


def test_no_geometry_skips_with_mark(tmp_path: Path) -> None:
    """Born-digital model table with no ``locate_tables`` geometry must not ship plain success."""
    md = (FIXTURES / "p59_tableau1.md").read_text(encoding="utf-8")
    out_dir = tmp_path / "resf_p59_out"
    _run_page(tmp_path, "resf_p59.pdf", md, locate_tables_empty=True)
    sidecar = _page_sidecar(out_dir, 1)
    win = sidecar["winning_output"]
    assert win["status"] == PageStatus.WARNING.value
    assert UNVERIFIED_TABLE_MARK in win["text"]
    assert sidecar.get("native_cell_binding")
    kinds = [e["kind"] for e in _audit_events(out_dir) if e.get("page_num") == 1]
    assert "native_cell_check_skipped" in kinds
