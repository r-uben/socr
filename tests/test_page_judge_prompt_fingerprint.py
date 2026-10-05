"""A wording edit to the page-judge prompt must invalidate resume.

``prompts/judge_page.md`` is data, outside ``_socr_source_digest``'s ``.py``
hash, and ``judge_model`` names only the model. The 2026-10-04 tolerance edit
changed verdicts on real pages without moving either, so terminal pages judged
under the old wording would have resumed. Difference pins, hermetic: the judge
resolver and the prompt digest are patched, nothing is probed.
"""

from __future__ import annotations

import pytest

from socr.core.config import EngineType, PipelineConfig
from socr.pipeline import orchestrator as orch


def _pipeline(monkeypatch, judge_model):
    p = orch.UnifiedPipeline(
        PipelineConfig(
            primary_engine=EngineType.QWEN,
            local_engine=EngineType.QWEN,
            enabled_engines=[EngineType.QWEN],
        )
    )
    monkeypatch.setattr(p, "_resolve_judge_model", lambda: judge_model)
    return p


def test_a_prompt_edit_moves_the_fingerprint_under_a_vlm_judge(monkeypatch):
    p = _pipeline(monkeypatch, "qwen3.8:27b")
    monkeypatch.setattr(orch, "_page_judge_prompt_digest", lambda: "wording-before")
    before = p._run_fingerprint()
    monkeypatch.setattr(orch, "_page_judge_prompt_digest", lambda: "wording-after")
    after = p._run_fingerprint()
    assert before != after, "a page-judge prompt edit must invalidate resume"


@pytest.mark.parametrize("wording", ["wording-before", "wording-after"])
def test_the_heuristic_judge_ignores_the_prompt(monkeypatch, wording):
    """No VLM judge reads the prompt, so its wording must not churn the ledger."""
    p = _pipeline(monkeypatch, None)
    monkeypatch.setattr(orch, "_page_judge_prompt_digest", lambda: "wording-before")
    baseline = p._run_fingerprint()
    monkeypatch.setattr(orch, "_page_judge_prompt_digest", lambda: wording)
    assert p._run_fingerprint() == baseline


def test_the_digest_reads_the_shipped_prompt():
    import hashlib

    from socr.judge.judge import load_judge_prompt

    expected = hashlib.sha256(load_judge_prompt().encode("utf-8")).hexdigest()
    assert orch._page_judge_prompt_digest() == expected
