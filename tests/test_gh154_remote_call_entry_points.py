"""GH-154 round 3 (Astra/Codex review): an EXPLICIT ``--max-cost-per-page 0``
must forbid a cloud/remote call at EVERY entry point, not just the main
routing ladder and the clean-equation lane.

Round 1/2 wired ``max_cost_per_page_pinned`` into ``provider_ladder`` and the
two ladder-building call sites. The review found four more direct entry
points that skip the ladder entirely and read the config flags themselves --
each is covered here, all pinned to the shared
``socr.core.providers.zero_cap_pinned_forbids_cloud`` predicate. The blind-cell
adjudicator's own cap gate (``judge/table_cell_guard.py``) is covered in
``tests/test_table_cell_guard.py::test_gh154_pinned_zero_forbids_the_adjudicator_call``.

Hermetic: no provider, no network, no live model.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from socr.core.config import PipelineConfig
from socr.core.providers import zero_cap_pinned_forbids_cloud
from socr.pipeline.orchestrator import UnifiedPipeline


def _pipeline(**overrides) -> UnifiedPipeline:
    cfg = PipelineConfig(quiet=True, **overrides)
    pipe = object.__new__(UnifiedPipeline)
    pipe.config = cfg
    return pipe


# ---------------------------------------------------------------------------
# The shared predicate itself
# ---------------------------------------------------------------------------


def test_predicate_false_for_unset_default():
    cfg = PipelineConfig()
    assert cfg.max_cost_per_page == 0.0
    assert cfg.max_cost_per_page_pinned is False
    assert zero_cap_pinned_forbids_cloud(cfg) is False


def test_predicate_true_only_when_pinned_and_zero():
    assert zero_cap_pinned_forbids_cloud(
        PipelineConfig(max_cost_per_page=0.0, max_cost_per_page_pinned=True)
    )
    # Pinned but a real positive cap: the predicate is about the ZERO case only.
    assert not zero_cap_pinned_forbids_cloud(
        PipelineConfig(max_cost_per_page=0.05, max_cost_per_page_pinned=True)
    )
    # Zero but not pinned (the ordinary "no cap" default): untouched.
    assert not zero_cap_pinned_forbids_cloud(
        PipelineConfig(max_cost_per_page=0.0, max_cost_per_page_pinned=False)
    )


# ---------------------------------------------------------------------------
# 1 — direct corrupt-math model call (Astra's original probe)
# ---------------------------------------------------------------------------


def test_pinned_zero_blocks_direct_cloud_math():
    pipe = _pipeline(max_cost_per_page=0, max_cost_per_page_pinned=True, math_model="qwen3.5:cloud")
    assert pipe._corrupt_math_model_disabled_reason()


def test_unpinned_zero_still_allows_direct_cloud_math_by_default():
    # Control: the ordinary omitted-flag default must be untouched.
    pipe = _pipeline(max_cost_per_page=0.0, math_model="qwen3.5:cloud")
    assert pipe._corrupt_math_model_disabled_reason() == ""


# ---------------------------------------------------------------------------
# 2 — clean-equation-region lane's early cloud guard
# ---------------------------------------------------------------------------


def test_pinned_zero_blocks_equation_lane_cloud_model():
    pipe = _pipeline(
        max_cost_per_page=0,
        max_cost_per_page_pinned=True,
        clean_equation_model="qwen3.5:cloud",
    )
    profile, reason = pipe._equation_lane_provider(available_profiles=[])
    assert profile is None
    assert "max-cost-per-page 0" in reason


# ---------------------------------------------------------------------------
# 3 — cloud table-judge ladder construction + reachability
# ---------------------------------------------------------------------------


def test_pinned_zero_empties_the_table_judge_ladder():
    pipe = _pipeline(
        table_judge_ladder=True,
        max_cost_per_page=0,
        max_cost_per_page_pinned=True,
    )
    assert pipe._build_table_judge_rungs() == []


def test_unpinned_zero_still_builds_the_table_judge_ladder_by_default():
    pipe = _pipeline(table_judge_ladder=True, max_cost_per_page=0.0)
    assert pipe._build_table_judge_rungs() != []


def test_pinned_zero_makes_no_table_judge_rung_available():
    pipe = _pipeline(
        table_judge_ladder=True,
        max_cost_per_page=0,
        max_cost_per_page_pinned=True,
    )
    assert pipe._table_judge_rung_available_now() is False


# ---------------------------------------------------------------------------
# 4 — blind-cell adjudicator construction
# ---------------------------------------------------------------------------


def test_pinned_zero_refuses_to_build_the_adjudicator():
    pipe = _pipeline(
        table_judge_ladder=True,
        max_cost_per_page=0,
        max_cost_per_page_pinned=True,
    )
    assert pipe._build_table_cell_adjudicator() is None


def test_unpinned_zero_still_builds_the_adjudicator_by_default():
    pipe = _pipeline(table_judge_ladder=True, max_cost_per_page=0.0)
    assert pipe._build_table_cell_adjudicator() is not None


# ---------------------------------------------------------------------------
# Round 4 (Astra/Codex re-review of 80dbdb6): five more dispatch boundaries
# that read the config flags directly, bypassing every ladder above.
# ---------------------------------------------------------------------------


# 5 — table crop reread's vision-model resolver -----------------------------


def test_pinned_zero_blocks_crop_reader_cloud_model():
    pipe = _pipeline(
        max_cost_per_page=0, max_cost_per_page_pinned=True, judge_model="qwen3.5:cloud"
    )
    assert "cloud" not in (pipe._resolve_crop_vlm_model() or "")


def test_unpinned_zero_still_allows_crop_reader_cloud_model_by_default():
    pipe = _pipeline(max_cost_per_page=0.0, judge_model="qwen3.5:cloud")
    assert pipe._resolve_crop_vlm_model() == "qwen3.5:cloud"


def test_pinned_zero_leaves_hpc_server_backend_alone():
    # Control (review: "do not equate every server URL with paid cloud"): a
    # configured vLLM/HPC inference server is not cloud egress, so the policy
    # must not touch it even when pinned.
    pipe = _pipeline(
        max_cost_per_page=0,
        max_cost_per_page_pinned=True,
        qwen_backend="vllm",
        qwen_vllm_model="Qwen/Qwen3-VL-30B-A3B-Instruct",
    )
    assert pipe._resolve_crop_vlm_model() == "Qwen/Qwen3-VL-30B-A3B-Instruct"


# 6 — cell disproof transcriber ----------------------------------------------


def test_pinned_zero_blocks_disproof_transcriber():
    from pathlib import Path

    pipe = _pipeline(max_cost_per_page=0, max_cost_per_page_pinned=True)
    with patch("socr.judge.cell_transcribe.transcribe_cell") as mock_call:
        result = pipe._transcribe_cell_token(Path("/tmp/not-read-by-stub.png"))
    mock_call.assert_not_called()
    assert result is None


def test_unpinned_zero_still_calls_disproof_transcriber_by_default():
    from pathlib import Path

    pipe = _pipeline(max_cost_per_page=0.0)
    with patch("socr.judge.cell_transcribe.transcribe_cell", return_value="10") as mock_call:
        result = pipe._transcribe_cell_token(Path("/tmp/not-read-by-stub.png"))
    mock_call.assert_called_once()
    assert result == "10"


# 7 — figure-description Gemini fallback ------------------------------------


def test_pinned_zero_blocks_figure_description_gemini_fallback(monkeypatch):
    pipe = _pipeline(max_cost_per_page=0, max_cost_per_page_pinned=True)
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-test")
    with (
        patch("socr.engines.gemini_api.OllamaFigureEngine.is_available", return_value=False),
        patch("socr.engines.gemini_api.GeminiAPIEngine.initialize") as mock_init,
    ):
        engine = pipe._get_vision_engine()
    mock_init.assert_not_called()
    assert engine is None


def test_unpinned_zero_still_tries_figure_description_gemini_fallback_by_default(monkeypatch):
    pipe = _pipeline(max_cost_per_page=0.0)
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-test")
    with (
        patch("socr.engines.gemini_api.OllamaFigureEngine.is_available", return_value=False),
        patch("socr.engines.gemini_api.GeminiAPIEngine.initialize", return_value=True) as mock_init,
    ):
        engine = pipe._get_vision_engine()
    mock_init.assert_called_once()
    assert engine is not None


# 8 — legacy clean-equation recovery (direct process_equation_region call) --


def _state_with_one_equation_region(tmp_path):
    import fitz

    from socr.core.audit_log import AuditEvent
    from socr.core.document import DocumentHandle
    from socr.core.state import DocumentState

    pdf = tmp_path / "doc.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(str(pdf))
    doc.close()
    state = DocumentState(DocumentHandle(pdf))
    state.events.append(
        AuditEvent(
            page_num=1,
            kind="equation_region_detected",
            engine="equation_region",
            data={"crop_path": None},
        )
    )
    return state


def test_pinned_zero_blocks_legacy_clean_equation_cloud_model(tmp_path):
    from socr.core.result import PageOutput, PageStatus

    state = _state_with_one_equation_region(tmp_path)
    po = PageOutput(page_num=1, text="native", status=PageStatus.SUCCESS, engine="qwen")

    pipe = _pipeline(
        max_cost_per_page=0,
        max_cost_per_page_pinned=True,
        clean_equation_model="qwen3.5:cloud",
    )

    with patch("socr.math.equation_latex.process_equation_region") as mock_proc:
        pipe._attach_equation_latex_sidecars(state, [po])
    mock_proc.assert_not_called()
    assert po.text == "native"  # unchanged: no sidecar attached


def test_unpinned_zero_still_calls_legacy_clean_equation_cloud_model_by_default(tmp_path):
    from socr.core.result import PageOutput, PageStatus
    from socr.math.equation_latex import EquationLatexResult

    state = _state_with_one_equation_region(tmp_path)
    po = PageOutput(page_num=1, text="native", status=PageStatus.SUCCESS, engine="qwen")

    pipe = _pipeline(max_cost_per_page=0.0, clean_equation_model="qwen3.5:cloud")

    fake_result = EquationLatexResult(
        region_index=0,
        page_num=1,
        crop_path=None,
        raw_latex="",
        validation_ok=False,
        validation_reason="no crop",
        latex_attached=False,
        model_id="qwen3.5:cloud",
    )
    with patch("socr.math.equation_latex.process_equation_region") as mock_proc:
        mock_proc.return_value = fake_result
        pipe._attach_equation_latex_sidecars(state, [po])
    mock_proc.assert_called_once()


# 9 — HPC pipeline's Gemini fallback -----------------------------------------


def _hpc_pipeline(**overrides):
    from socr.pipeline.hpc_pipeline import HPCPipeline

    cfg = PipelineConfig(quiet=True, **overrides)
    pipe = object.__new__(HPCPipeline)
    pipe.config = cfg
    return pipe


def test_pinned_zero_blocks_hpc_gemini_fallback():
    pipe = _hpc_pipeline(max_cost_per_page=0, max_cost_per_page_pinned=True)
    with patch("socr.engines.gemini.GeminiEngine.is_available") as mock_avail:
        result = pipe._fallback_to_gemini([1, 2])
    mock_avail.assert_not_called()
    assert result == {}


def test_unpinned_zero_still_tries_hpc_gemini_fallback_by_default():
    pipe = _hpc_pipeline(max_cost_per_page=0.0)
    with patch("socr.engines.gemini.GeminiEngine.is_available", return_value=False) as mock_avail:
        result = pipe._fallback_to_gemini([1, 2])
    mock_avail.assert_called_once()
    assert result == {}


# ---------------------------------------------------------------------------
# Round 5 (Astra/Codex re-review of 5b9ba7f): the ordinary page-level VLM
# judge -- the default judge_backend=auto path every agentic page takes --
# still shipped page images to the cloud under pinned zero.
# ---------------------------------------------------------------------------

# 10 — _resolve_judge_model: explicit / cached / candidate-ladder cloud -----


@pytest.mark.parametrize("explicit_model", ["", "qwen3.5:cloud"])
def test_pinned_zero_blocks_page_judge_cloud_model(explicit_model):
    pipe = _pipeline(max_cost_per_page=0, max_cost_per_page_pinned=True, judge_model=explicit_model)
    pipe._judge_model_cache = False
    with patch("socr.judge.ollama_judge.OllamaVisionJudge.is_available", return_value=True):
        chosen = pipe._resolve_judge_model()
    assert "cloud" not in (chosen or "").lower()


def test_unpinned_zero_still_resolves_cloud_page_judge_by_default():
    pipe = _pipeline(max_cost_per_page=0.0)
    pipe._judge_model_cache = False
    with patch("socr.judge.ollama_judge.OllamaVisionJudge.is_available", return_value=True):
        chosen = pipe._resolve_judge_model()
    assert chosen == "qwen3.5:cloud"


def test_pinned_zero_ignores_a_stale_cached_cloud_identity():
    # A cloud identity memoized before this pipeline instance's cap was
    # pinned (or from a prior call under a different policy) must not be
    # returned verbatim -- it is re-resolved under the current policy.
    pipe = _pipeline(max_cost_per_page=0, max_cost_per_page_pinned=True)
    pipe._judge_model_cache = "qwen3.5:cloud"
    with patch("socr.judge.ollama_judge.OllamaVisionJudge.is_available", return_value=True):
        chosen = pipe._resolve_judge_model()
    assert "cloud" not in (chosen or "").lower()


def test_pinned_zero_with_no_local_judge_available_resolves_none():
    pipe = _pipeline(max_cost_per_page=0, max_cost_per_page_pinned=True)
    pipe._judge_model_cache = False
    with patch("socr.judge.ollama_judge.OllamaVisionJudge.is_available", return_value=False):
        chosen = pipe._resolve_judge_model()
    assert chosen is None


# 11 — the composed judge builder: zero cloud provider calls under the policy


def test_pinned_zero_degrades_composed_judge_to_heuristic_with_no_cloud_call(tmp_path):
    from socr.core.config import PipelineConfig
    from socr.core.document import DocumentHandle
    from socr.core.state import DocumentState
    from socr.pipeline.orchestrator import JUDGE_IDENTITY_HEURISTIC, UnifiedPipeline

    from test_p35_cold_review_round2 import _build_fixture_pdf

    pdf = _build_fixture_pdf(tmp_path)
    state = DocumentState(DocumentHandle(pdf))

    cfg = PipelineConfig(
        quiet=True,
        judge_backend="auto",
        max_cost_per_page=0,
        max_cost_per_page_pinned=True,
    )
    pipe = UnifiedPipeline(cfg)

    # A spy in place of the real OllamaVisionJudge: records every model it
    # was CONSTRUCTED with (never available, forcing the heuristic
    # degradation this test also checks for) so the assertion is "no
    # instance was ever built for the cloud model" -- a stronger claim than
    # merely mocking `.is_available()`, which cannot see which model asked.
    seen_models: list[str] = []

    class _SpyOllamaVisionJudge:
        def __init__(self, model: str) -> None:
            seen_models.append(model)
            self.model = model

        def is_available(self) -> bool:
            return False

    with patch("socr.judge.ollama_judge.OllamaVisionJudge", _SpyOllamaVisionJudge):
        pipe._build_page_judge(state)

    assert "qwen3.5:cloud" not in seen_models, "no provider call is permitted under pinned zero"
    assert state.agentic_judge_model == JUDGE_IDENTITY_HEURISTIC
    assert any(e.kind == "judge_degraded_to_heuristic" for e in state.events)
