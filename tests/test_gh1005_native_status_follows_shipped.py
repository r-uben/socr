"""#1005: a native-text status follows the SHIPPED output.

The native-damage accounting (unmapped math glyphs, minus-as-digit, invisible scan
layer, garbled math) describes the NATIVE text layer. When a model reading wins, the
page takes that reading's own status.

Pinned as a DIFFERENCE, never an absolute value: the same flagged page is finalized
twice in this process, changing only whether native or a model reading ships.

Hermetic: `_available_engines_for_agentic` is patched, `_resolve_judge_model` is "",
and no path that reaches a provider is entered.
"""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import pytest

fitz = pytest.importorskip("fitz")

from socr.core.config import EngineType, PipelineConfig  # noqa: E402
from socr.core.document import DocumentHandle  # noqa: E402
from socr.core.manifest import finalized_page_record  # noqa: E402
from socr.core.result import FailureMode, PageOutput, PageStatus  # noqa: E402
from socr.core.state import DocumentState  # noqa: E402
from socr.math.accounting import UNRESOLVED_MATH_KIND  # noqa: E402
from socr.pipeline.orchestrator import UnifiedPipeline  # noqa: E402

PUA = chr(0xF766)
NATIVE_TEXT = f"Prose of the page with the equation {PUA}{PUA} = {PUA}o(T) in it, long enough."
MODEL_TEXT = "Prose of the page with the equation $T = f(T)$ in it, long enough."

#: damage name -> (PageState attribute carrying it, value)
DAMAGE = {
    "unmapped_math": ("has_unmapped_math_glyphs", True),
    "minus_as_digit": ("minus_as_digit_hits", 2),
    "invisible_scan": ("invisible_text_over_raster", True),
    "garbled_math": ("garbled_math_signals", ["sig"]),
}


def _pdf(path: Path) -> Path:
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "born digital page", fontsize=11)
    doc.save(str(path))
    doc.close()
    return path


def _state(tmp_path: Path, tag: str, damage: str, *, model_ships: bool) -> DocumentState:
    pdf = _pdf(tmp_path / f"{tag}.pdf")
    state = DocumentState(handle=DocumentHandle.from_path(pdf))
    ps = state.pages[1]
    ps.is_born_digital = True
    ps.native_text = NATIVE_TEXT
    attr, value = DAMAGE[damage]
    setattr(ps, attr, value)
    if model_ships:
        out = PageOutput(
            page_num=1,
            text=MODEL_TEXT,
            status=PageStatus.SUCCESS,
            engine="qwen",
            audit_passed=True,
        )
        ps.attempts.append(out)
        ps.best_output = out
    else:
        # OCR was tried and never passed: native ships as the flagged fallback.
        ps.needs_ocr_enhancement = True
        ps.attempts.append(
            PageOutput(
                page_num=1,
                text=MODEL_TEXT,
                status=PageStatus.WARNING,
                engine="qwen",
                audit_passed=False,
            )
        )
        ps.best_output = None
    return state


def _assemble(tmp_path: Path, state: DocumentState, tag: str):
    from socr.core.providers import PROFILE_QWEN_LOCAL

    pipeline = UnifiedPipeline(
        PipelineConfig(
            primary_engine=EngineType.DEEPSEEK,
            enabled_engines=list(EngineType),
            agentic=True,
            quiet=True,
            native_first=True,
        )
    )
    pipeline._scan_root = tmp_path
    with ExitStack() as stack:
        stack.enter_context(
            patch.object(
                pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]
            )
        )
        stack.enter_context(patch.object(pipeline, "_resolve_judge_model", return_value=""))
        return pipeline._phase_assemble(state, tmp_path / f"{tag}_out")


@pytest.mark.parametrize("damage", sorted(DAMAGE))
def test_status_follows_the_shipped_output(tmp_path: Path, damage: str) -> None:
    native = finalized_page_record(_state(tmp_path, "n", damage, model_ships=False), 1).output
    model = finalized_page_record(_state(tmp_path, "m", damage, model_ships=True), 1).output

    assert model.text == MODEL_TEXT
    assert model.status is PageStatus.SUCCESS
    assert model.failure_mode is FailureMode.NONE
    assert model.audit_passed is True
    assert not [n for n in model.audit_notes if "native" in n]
    # The difference: the same flagged page, native shipping, is demoted (or noted).
    assert (native.status, native.failure_mode, tuple(native.audit_notes)) != (
        model.status,
        model.failure_mode,
        tuple(model.audit_notes),
    )
    if damage == "unmapped_math":
        assert native.status is PageStatus.WARNING
        assert any("unmapped math glyphs" in n for n in native.audit_notes)


def test_unmapped_math_document_level_follows_the_shipped_output(tmp_path: Path) -> None:
    native_state = _state(tmp_path, "n", "unmapped_math", model_ships=False)
    model_state = _state(tmp_path, "m", "unmapped_math", model_ships=True)
    native_result = _assemble(tmp_path, native_state, "n")
    model_result = _assemble(tmp_path, model_state, "m")

    def kinds(state):
        return [e for e in state.events if getattr(e, "kind", "") == UNRESOLVED_MATH_KIND]

    assert len(kinds(native_state)) == 1
    assert kinds(model_state) == []
    assert "unrecovered math glyphs" in (native_result.error or "")
    assert "unrecovered math glyphs" not in (model_result.error or "")
    assert model_state.status != native_state.status


def test_a_model_body_that_still_carries_the_damage_keeps_the_warning(tmp_path: Path) -> None:
    state = _state(tmp_path, "m", "unmapped_math", model_ships=True)
    leaked = PageOutput(
        page_num=1,
        text=MODEL_TEXT + f" {PUA}",
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=True,
    )
    state.pages[1].attempts[:] = [leaked]
    state.pages[1].best_output = leaked
    assert finalized_page_record(state, 1).output.status is PageStatus.WARNING
