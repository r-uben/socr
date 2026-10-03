"""#1004: a page judge TIMEOUT on a page whose native layer is known bad.

The model read has no verdict (a timeout is a missing verdict, not evidence), so it does
not ship; native ships, but under its OWN failure mode at WARNING, the document is
AUDIT_FAILED, and the page is counted once. A completed rejection is never reinterpreted
as a timeout.

Hermetic: no provider, no judge. Every pin is a DIFFERENCE between two states that differ
in exactly one thing (CLAUDE.md, #257), never an absolute outcome measured locally.
"""

from __future__ import annotations

import pytest

fitz = pytest.importorskip("fitz")

from socr.core.config import EngineType, PipelineConfig  # noqa: E402
from socr.core.document import DocumentHandle  # noqa: E402
from socr.core.manifest import (  # noqa: E402
    PageEnding,
    SelectionProvenance,
    _select_page_output_tagged,
    native_untrusted_judge_timeout,
    page_disposition,
)
from socr.core.result import (  # noqa: E402
    JUDGE_OUTCOME_COMPLETED,
    JUDGE_OUTCOME_TIMEOUT,
    DocumentStatus,
    FailureMode,
    PageOutput,
    PageStatus,
)
from socr.core.state import DocumentState  # noqa: E402
from socr.pipeline.orchestrator import UnifiedPipeline  # noqa: E402

_MODEL_TEXT = "model read of the page: the estimate is 0.42"
_NATIVE_TEXT = "native layer of the page with garbled symbols"


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch: pytest.MonkeyPatch) -> None:
    from socr.core.providers import PROFILE_QWEN_LOCAL

    monkeypatch.setattr(
        UnifiedPipeline, "_available_engines_for_agentic", lambda self: [PROFILE_QWEN_LOCAL]
    )
    monkeypatch.setattr(UnifiedPipeline, "_resolve_judge_model", lambda self, *a, **kw: "")


def _state(tmp_path, tag, *, outcome, enhancement=True, structure=False):
    path = tmp_path / f"{tag}.pdf"
    doc = fitz.open()
    doc.new_page().insert_text(
        (54, 72), "born-digital prose long enough to count as a real text layer here."
    )
    doc.save(path)
    doc.close()
    state = DocumentState(handle=DocumentHandle.from_path(path))
    p = state.pages[1]
    p.is_born_digital = True
    p.native_text = _NATIVE_TEXT
    p.needs_ocr_enhancement = enhancement
    model = PageOutput(
        page_num=1,
        text=_MODEL_TEXT,
        status=PageStatus.WARNING,
        engine="qwen",
        audit_passed=False,
        failure_mode=FailureMode.AUDIT_FAILED,
        judge_outcome=outcome,
    )
    p.attempts.append(model)
    p.best_output = model
    if structure:
        p.has_tables = True
        p.native_text = "0.03 0.91 0.44\nn slope R2\n"
    return state


def _ship(state):
    return _select_page_output_tagged(state, 1)


def test_loaded_source_is_this_checkout() -> None:
    from pathlib import Path

    import socr

    assert (
        Path(socr.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1] / "src")
    ), socr.__file__


def test_timeout_differs_from_completed_rejection_on_the_same_page(tmp_path) -> None:
    timed_out, t_tag = _ship(_state(tmp_path, "a", outcome=JUDGE_OUTCOME_TIMEOUT))
    rejected, r_tag = _ship(_state(tmp_path, "b", outcome=JUDGE_OUTCOME_COMPLETED))

    # Same bytes, same WARNING, same audit flag; only the cause differs.
    assert timed_out.text == rejected.text == _NATIVE_TEXT
    assert timed_out.status is rejected.status is PageStatus.WARNING
    assert timed_out.audit_passed is rejected.audit_passed
    assert timed_out.failure_mode is FailureMode.NATIVE_UNTRUSTED_JUDGE_TIMEOUT
    assert rejected.failure_mode is not FailureMode.NATIVE_UNTRUSTED_JUDGE_TIMEOUT
    assert t_tag is SelectionProvenance.NATIVE_UNTRUSTED_JUDGE_TIMEOUT
    assert r_tag is not SelectionProvenance.NATIVE_UNTRUSTED_JUDGE_TIMEOUT
    # Never plain success, never the model bytes.
    assert timed_out.failure_mode is not FailureMode.NONE
    assert _MODEL_TEXT not in timed_out.text


def test_timeout_needs_the_native_layer_to_be_known_bad(tmp_path) -> None:
    flagged = _state(tmp_path, "c", outcome=JUDGE_OUTCOME_TIMEOUT, enhancement=True)
    clean = _state(tmp_path, "d", outcome=JUDGE_OUTCOME_TIMEOUT, enhancement=False)
    assert native_untrusted_judge_timeout(flagged.pages[1]) is True
    assert native_untrusted_judge_timeout(clean.pages[1]) is False
    assert _ship(flagged)[0].failure_mode is FailureMode.NATIVE_UNTRUSTED_JUDGE_TIMEOUT
    assert _ship(clean)[0].failure_mode is not FailureMode.NATIVE_UNTRUSTED_JUDGE_TIMEOUT


def test_later_completed_rejection_of_the_same_bytes_supersedes_the_timeout(tmp_path) -> None:
    state = _state(tmp_path, "e", outcome=JUDGE_OUTCOME_TIMEOUT)
    p = state.pages[1]
    p.attempts.append(
        PageOutput(
            page_num=1,
            text=_MODEL_TEXT,
            status=PageStatus.WARNING,
            engine="gemini",
            audit_passed=False,
            judge_outcome=JUDGE_OUTCOME_COMPLETED,
        )
    )
    assert native_untrusted_judge_timeout(p) is False


def test_structure_class_page_keeps_the_713_path(tmp_path) -> None:
    """A structure-class page never reaches the new ending, timeout or not."""
    a = _state(tmp_path, "f", outcome=JUDGE_OUTCOME_TIMEOUT, structure=True)
    b = _state(tmp_path, "g", outcome=JUDGE_OUTCOME_COMPLETED, structure=True)
    assert a.pages[1].is_structure_class()
    assert native_untrusted_judge_timeout(a.pages[1]) is False
    assert _ship(a)[1] is not SelectionProvenance.NATIVE_UNTRUSTED_JUDGE_TIMEOUT
    assert _ship(b)[1] is not SelectionProvenance.NATIVE_UNTRUSTED_JUDGE_TIMEOUT


def _assemble(tmp_path, tag, **kw):
    state = _state(tmp_path, tag, **kw)
    pipeline = UnifiedPipeline(
        PipelineConfig(
            quiet=True,
            enabled_engines=[EngineType.GEMINI],
            primary_engine=EngineType.GEMINI,
            table_judge_ladder=False,
        )
    )
    pipeline._scan_root = tmp_path
    result = pipeline._phase_assemble(state, tmp_path / f"out_{tag}")
    return result, state


def test_document_audit_failed_and_page_counted_once(tmp_path) -> None:
    result, state = _assemble(tmp_path, "h", outcome=JUDGE_OUTCOME_TIMEOUT)
    _, cstate = _assemble(tmp_path, "i", outcome=JUDGE_OUTCOME_COMPLETED)

    assert result.status is DocumentStatus.AUDIT_FAILED
    kinds = [e.kind for e in state.events if e.page_num == 1]
    ckinds = [e.kind for e in cstate.events if e.page_num == 1]
    # Own event, and NOT the generic one: one page, one count.
    assert kinds.count("native_untrusted_judge_timeout") == 1
    assert "native_fallback" not in kinds
    assert "native_untrusted_judge_timeout" not in ckinds
    assert ckinds.count("native_fallback") == 1


def test_disposition_pair_is_the_demoted_native_ending(tmp_path) -> None:
    d = page_disposition(_state(tmp_path, "j", outcome=JUDGE_OUTCOME_TIMEOUT), 1)
    assert d.ending is PageEnding.DEMOTED_NATIVE
