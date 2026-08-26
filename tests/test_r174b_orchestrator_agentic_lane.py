"""R174b acceptance tests: Orchestrator agentic lane & fingerprint contract (Tasks t2, t8, t9, t14).

Verifies:
- UnifiedPipeline.process() unconditionally routes through _phase_agentic.
- _run_fingerprint excludes the 7 dead fingerprint keys:
  multi_engine, multi_engine_determinants, truncation_retries, max_retries,
  consensus, consensus_use_llm, consensus_ollama_model.
- _run_fingerprint retains all live determinants.
- All tests driving process() patch _available_engines_for_agentic.
- Negative call assertions use concrete sentinels/counters.
- Historical manifest compatibility: consensus(<engine>) strings resolve to the underlying engine.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from socr.core.config import EngineType, PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.providers import (
    PROFILE_GEMINI,
    PROFILE_GLM,
    ProviderProfile,
)
from socr.core.result import DocumentStatus, EngineResult, PageOutput, PageStatus
from socr.core.state import DocumentState
from socr.pipeline.orchestrator import UnifiedPipeline


def _real_pdf(path: Path, n_pages: int = 2) -> Path:
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    for i in range(n_pages):
        doc.new_page().insert_text((72, 72), f"page {i + 1} content text for testing")
    doc.save(str(path))
    doc.close()
    return path


def _make_config(**overrides) -> PipelineConfig:
    defaults = dict(
        primary_engine=EngineType.AUTO,
        enabled_engines=[EngineType.GLM, EngineType.GEMINI],
        quiet=True,
        tiered=False,
        audit_enabled=False,
    )
    defaults.update(overrides)
    return PipelineConfig(**defaults)


class TestOrchestratorAgenticLane:
    """Acceptance tests for unconditional agentic orchestration."""

    @pytest.mark.parametrize("providers_available", [True, False])
    def test_process_unconditionally_executes_agentic_phase(
        self, tmp_path: Path, providers_available: bool
    ):
        """UnifiedPipeline.process() must execute _phase_agentic for True and False."""
        pdf_path = _real_pdf(tmp_path / "doc.pdf", n_pages=2)
        sample_providers: list[ProviderProfile] = (
            [PROFILE_GLM, PROFILE_GEMINI] if providers_available else []
        )

        for agentic_setting in (True, False):
            out_dir = tmp_path / f"out_agentic_{agentic_setting}_{providers_available}"
            config = _make_config(agentic=agentic_setting)
            pipeline = UnifiedPipeline(config)

            call_count = 0

            def _stub_phase_agentic(state: DocumentState, output_dir: Path) -> None:
                nonlocal call_count
                call_count += 1
                for i in range(1, state.handle.page_count + 1):
                    state.apply_result(
                        EngineResult(
                            document_path=pdf_path,
                            engine="gemini" if providers_available else "native",
                            status=DocumentStatus.SUCCESS,
                            pages=[
                                PageOutput(
                                    page_num=i,
                                    text=f"Processed text for page {i}",
                                    status=PageStatus.SUCCESS,
                                    engine="gemini" if providers_available else "native",
                                    audit_passed=True,
                                )
                            ],
                        )
                    )

            with (
                patch.object(
                    pipeline,
                    "_available_engines_for_agentic",
                    return_value=sample_providers,
                ),
                patch.object(pipeline, "_phase_agentic", side_effect=_stub_phase_agentic),
            ):
                result = pipeline.process(pdf_path, output_dir=out_dir)

            assert call_count == 1, (
                f"_phase_agentic call count mismatch for agentic={agentic_setting}: {call_count}"
            )
            assert result.status == DocumentStatus.SUCCESS
            assert result.pages_processed == 2

    def test_fingerprint_excludes_dead_legacy_keys(self):
        """UnifiedPipeline._run_fingerprint AST must exclude dead keys and retain live keys."""
        import ast

        repo_root = Path(__file__).resolve().parent.parent
        orch_path = repo_root / "src/socr/pipeline/orchestrator.py"
        tree = ast.parse(orch_path.read_text(encoding="utf-8"), filename=str(orch_path))

        unified_class = next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "UnifiedPipeline"
        )
        fp_func = next(
            n
            for n in unified_class.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and n.name == "_run_fingerprint"
        )

        dict_keys = set()
        for node in ast.walk(fp_func):
            if isinstance(node, ast.Dict):
                for k in node.keys:
                    if isinstance(k, ast.Constant) and isinstance(k.value, str):
                        dict_keys.add(k.value)

        dead_keys = {
            "multi_engine",
            "multi_engine_determinants",
            "truncation_retries",
            "max_retries",
            "consensus",
            "consensus_use_llm",
            "consensus_ollama_model",
        }
        present_dead = dead_keys & dict_keys
        assert not present_dead, (
            f"Dead keys found as dictionary keys in _run_fingerprint: {present_dead}"
        )

        live_keys = {
            "socr_source_digest",
            "enabled_engines",
            "enabled_engine_determinants",
        }
        missing_live = live_keys - dict_keys
        assert not missing_live, (
            f"Live keys missing from dictionary keys in _run_fingerprint: {missing_live}"
        )

    def test_historical_consensus_label_compatibility(self, tmp_path: Path):
        """Historical manifests with consensus(<engine>) strings must resolve."""
        from socr.core.cache import BlobStore
        from socr.core.manifest import build_manifest

        pdf = _real_pdf(tmp_path / "legacy_doc.pdf", n_pages=1)
        handle = DocumentHandle.from_path(pdf)
        state = DocumentState(handle=handle)
        state.apply_result(
            EngineResult(
                document_path=pdf,
                engine="consensus(gemini)",
                status=DocumentStatus.SUCCESS,
                pages=[
                    PageOutput(
                        page_num=1,
                        text="historical consensus output",
                        status=PageStatus.SUCCESS,
                        engine="consensus(gemini)",
                        audit_passed=True,
                    )
                ],
            )
        )

        fp_inputs = {"gemini": ("gemini-3-flash-preview", "socr", "convert", None)}
        store = BlobStore(tmp_path / "cache")
        manifest = build_manifest(state, store, dpi=120, fingerprint_inputs=fp_inputs)
        fp = manifest.entries[1].fingerprint

        assert fp.model_version == "gemini-3-flash-preview"
        assert fp.prompt_hash.startswith("fp:")
