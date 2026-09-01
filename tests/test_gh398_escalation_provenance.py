"""GH-398: pin the ESCALATION PageOutput writer, not just the agentic B3 one.

#385 pinned the agentic B3 writer. The other writer -- in `_escalate_table_page`
-- stayed unpinned, and #396 corrected its docstring to say so rather than claim
coverage it did not have. Reverting

    out.provider_backend, out.provider_model = resolved_provenance(profile, self.config)

to `profile.backend` / `profile.model` kept the whole suite green while an
escalated page recorded the registry LABEL instead of the backend that ran.

That is the value the manifest exists for: a genuinely misrouted run (`qwen-ocr`
missing from PATH, the documented HPC gotcha) and a correct one are
indistinguishable in the record if this is wrong.

Pinned as a DIFFERENCE between two configs identical but for `qwen_backend`,
driven through the production method's accepted branch. `PROFILE_QWEN_LOCAL`
declares `backend="ollama"` in the registry, so under `--qwen-backend vllm` the
registry label and the executed backend disagree -- which is exactly the case
the reverted line gets wrong and a helper-only test would not notice.
"""

from __future__ import annotations

from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")

from socr.core.config import EngineType, PipelineConfig  # noqa: E402
from socr.core.document import DocumentHandle  # noqa: E402
from socr.core.providers import PROFILE_QWEN_LOCAL, resolved_provenance  # noqa: E402
from socr.core.result import PageOutput, PageStatus  # noqa: E402
from socr.core.state import DocumentState  # noqa: E402
from socr.pipeline.orchestrator import UnifiedPipeline  # noqa: E402

INCUMBENT = "| Var | Est |\n| --- | --- |\n| a | 1 |"
CANDIDATE = "| Var | Est | SE |\n| --- | --- | --- |\n| a | 1 | (0.02) |\n| b | 2 | (0.03) |"


def _pdf(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "doc.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "Table 1. Regressions of excess returns.")
    doc.save(str(path))
    doc.close()
    return path


def _escalate(tmp_path: Path, qwen_backend: str) -> PageOutput:
    """Run the real `_escalate_table_page` accepted branch and return the winner."""
    from unittest.mock import patch

    pdf = _pdf(tmp_path)
    pipeline = UnifiedPipeline(
        PipelineConfig(
            primary_engine=EngineType.QWEN,
            enabled_engines=[EngineType.QWEN],
            qwen_backend=qwen_backend,
            quiet=True,
        )
    )
    state = DocumentState(handle=DocumentHandle.from_path(pdf))
    ps = state.pages[1]
    incumbent = PageOutput(
        page_num=1,
        text=INCUMBENT,
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=True,
    )
    ps.attempts.append(incumbent)
    ps.best_output = incumbent

    candidate = PageOutput(
        page_num=1,
        text=CANDIDATE,
        status=PageStatus.SUCCESS,
        engine=PROFILE_QWEN_LOCAL.engine.value,
        audit_passed=True,
    )

    # The lane's two gates are about WHETHER to escalate; this test is about what
    # the accepted branch RECORDS, so both are forced open and the write itself is
    # left entirely to production code. `decide_escalation` is imported inside the
    # method, so it is patched on its defining module.
    class _Accepted:
        accepted = True
        reason = "candidate measures better"
        gate = "exactness"
        delta = 1.0

    with (
        patch.object(UnifiedPipeline, "_table_page_needs_escalation", return_value=True),
        patch(
            "socr.tables.escalation_decision.decide_escalation",
            return_value=_Accepted(),
        ),
    ):
        _, winner = pipeline._escalate_table_page(
            state,
            1,
            ps,
            incumbent,
            PROFILE_QWEN_LOCAL,
            lambda _profile, _page: candidate,
            pdf,
        )
    return winner


def test_the_escalated_page_records_the_backend_that_ran(tmp_path: Path) -> None:
    """Under vllm the registry label and the executed backend disagree."""
    winner = _escalate(tmp_path / "v", "vllm")

    assert winner.text == CANDIDATE, (
        "the candidate was not accepted, so the writer never ran and this test "
        f"measures nothing: {winner.text!r}"
    )
    expected_backend, expected_model = resolved_provenance(
        PROFILE_QWEN_LOCAL,
        PipelineConfig(quiet=True, qwen_backend="vllm"),
    )
    assert expected_backend != PROFILE_QWEN_LOCAL.backend, (
        "fixture no longer diverges from the registry label, so a reverted writer would pass"
    )
    assert winner.provider_backend == expected_backend, (
        f"escalated page recorded {winner.provider_backend!r}, the registry label, "
        f"not the backend that ran ({expected_backend!r})"
    )
    assert winner.provider_model == expected_model


def test_the_two_backends_are_recorded_differently(tmp_path: Path) -> None:
    """The difference itself: only `qwen_backend` changes between the runs."""
    vllm = _escalate(tmp_path / "d1", "vllm")
    ollama = _escalate(tmp_path / "d2", "ollama")

    assert vllm.provider_backend != ollama.provider_backend, (
        "both configs recorded the same backend, so the writer is not reading "
        f"the config at all: {vllm.provider_backend!r}"
    )
    assert ollama.provider_backend == PROFILE_QWEN_LOCAL.backend
