"""#1346: the paid-after-dispute flag survives the kept-candidate re-judge on resume.

Run 1: a paid reading taken after a witness dispute is judged and the page judge times out, so
the sidecar keeps the candidate (#1013). Run 2: the re-judge accepts those exact bytes with no
second OCR call, and the page ships. The flag must travel with the bytes through the sidecar,
or the reading ships unmarked. Every claim is a difference between two runs that differ only in
whether the first run's reading carried the flag (CLAUDE.md, #257).
"""

from __future__ import annotations

import pytest

pytest.importorskip("fitz")

from test_gh1013_rejudge_timed_out_candidate import (  # noqa: E402
    _CONTROL_LAYER,
    _MODEL_TEXT,
    _Harness,
    _Judge,
    _kinds,
)

from socr.core.marks import Check  # noqa: E402
from socr.core.providers import PROFILE_QWEN_LOCAL  # noqa: E402
from socr.core.result import FailureMode  # noqa: E402
from socr.pipeline.orchestrator import UnifiedPipeline  # noqa: E402


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        UnifiedPipeline, "_available_engines_for_agentic", lambda self: [PROFILE_QWEN_LOCAL]
    )
    monkeypatch.setattr(UnifiedPipeline, "_resolve_judge_model", lambda self, *a, **kw: "")


class _FlaggingJudge(_Judge):
    """Times out in run 1 on a reading flagged as taken after a witness dispute (or not)."""

    def __init__(self, flagged: bool) -> None:
        super().__init__()
        self.flagged = flagged

    def assess(self, output, provider):
        if self.mode == "timeout":
            output.paid_after_witness_dispute = self.flagged
        return super().assess(output, provider)


def _two_runs(tmp_path, tag: str, *, flagged: bool, layer_words: str) -> _Harness:
    h = _Harness(tmp_path / tag)
    h.judge = _FlaggingJudge(flagged)
    h.own_layer_text = _CONTROL_LAYER
    h.own_layer_words = layer_words
    h.run()
    assert h.ocr_calls == [1]
    assert "judge_timeout_candidate" in h.sidecar()
    h.ocr_calls.clear()
    h.judge.mode = "accept"
    h.run()
    assert h.ocr_calls == [], "the re-judge must ship the kept bytes with no OCR"
    assert "rejudge_accepted" in _kinds(h)
    assert _MODEL_TEXT in h.shipped()
    return h


def _clauses(h: _Harness) -> set[str]:
    return {c["check"] for c in h.sidecar()["verdict"]["clauses"]}


def test_the_sidecar_carries_the_flag_with_the_kept_bytes(tmp_path) -> None:
    h = _Harness(tmp_path / "side")
    h.judge = _FlaggingJudge(True)
    h.run()
    kept = h.sidecar()["judge_timeout_candidate"]["candidate"]
    assert kept["paid_after_witness_dispute"] is True
    plain = _Harness(tmp_path / "plain")
    plain.judge = _FlaggingJudge(False)
    plain.run()
    assert (
        "paid_after_witness_dispute" not in plain.sidecar()["judge_timeout_candidate"]["candidate"]
    )


def test_a_rejudged_paid_reading_after_a_dispute_ships_marked(tmp_path) -> None:
    flagged = _two_runs(tmp_path, "flagged", flagged=True, layer_words=_CONTROL_LAYER)
    control = _two_runs(tmp_path, "control", flagged=False, layer_words=_CONTROL_LAYER)
    won = flagged.sidecar()["winning_output"]
    assert won["paid_after_witness_dispute"] is True
    assert "paid_after_witness_dispute" not in control.sidecar()["winning_output"]
    assert Check.READING_DISPUTE.value in _clauses(flagged)
    assert Check.READING_DISPUTE.value not in _clauses(control)
    assert flagged.sidecar()["status"] != control.sidecar()["status"]
    assert "reading_dispute" in flagged.shipped() and "reading_dispute" not in control.shipped()


def test_the_dispute_clause_keeps_the_judge_only_clause_beside_it(tmp_path) -> None:
    """No independent words evidence on the page: #1251's page_judge clause and its mode are
    what the page carries without the flag, and the flag adds a clause without taking either."""
    flagged = _two_runs(tmp_path, "flagged", flagged=True, layer_words="")
    control = _two_runs(tmp_path, "control", flagged=False, layer_words="")
    assert Check.PAGE_JUDGE.value in _clauses(control)  # the control does carry #1251's clause
    assert _clauses(flagged) == _clauses(control) | {Check.READING_DISPUTE.value}
    assert (
        flagged.sidecar()["winning_output"]["failure_mode"]
        == control.sidecar()["winning_output"]["failure_mode"]
    )
    assert flagged.sidecar()["winning_output"]["failure_mode"] != FailureMode.NONE.value
