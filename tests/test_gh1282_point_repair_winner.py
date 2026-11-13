"""#1282: a repair is recorded only when the repaired text ships (older attempt wins).

``_repair_dropped_points`` repairs ``best_output`` before ``finalized_page_record`` picks what
ships.  When an OLDER attempt wins, the repair must leave no event and no evidence.

Both arms run ``process()`` on the same synthetic PDF and fake model output.  ``best_output`` is
made audit-failed in BOTH arms (an audit-passed ``best_output`` ships before the structure-class
branch is consulted, which is how the first version of this test masked itself), and the older
attempt's text differs from it by a marker line, so the fragment says which attempt shipped.  The
arms differ in one thing: whether the structure-class winner is the older attempt or
``best_output``.  The assertion is the DIFFERENCE: the repair event appears iff the repaired
text ships.  Run in both provider states, with ``primary_engine`` / ``local_engine`` /
``enabled_engines`` pinned and ``_available_engines_for_agentic`` patched by ``_run``.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

pytest.importorskip("fitz")

from test_gh1267_row_binding_check import md  # noqa: E402
from test_gh1267_row_binding_pipeline import _events, _fragment, _run, _sidecar  # noqa: E402
from test_gh1282_point_repair_pipeline import DROPPED, NATIVE  # noqa: E402

from socr.core.result import PageOutput, PageStatus  # noqa: E402
from socr.pipeline.orchestrator import UnifiedPipeline  # noqa: E402


KIND = "table_point_repaired"
OLDER_MARKER = "Older attempt marker."


def _older(p):
    """An older attempt: the model's table with the point still dropped, plus a marker line."""
    return PageOutput(
        page_num=p.best_output.page_num,
        text=md(DROPPED) + "\n\n" + OLDER_MARKER,
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=True,
    )


def audit_failed_best_output(pipeline_cls):
    """Patch: ``best_output`` fails its audit just before the page is finalized, so the selector
    reaches the structure-class branch instead of shipping it directly."""
    original = pipeline_cls._apply_text_coverage

    def _wrapped(self, state, page_num, ps, *a, **k):
        if ps.best_output is not None:
            ps.best_output.audit_passed = False
        return original(self, state, page_num, ps, *a, **k)

    return patch.object(pipeline_cls, "_apply_text_coverage", _wrapped)


def _run_with_winner(tmp_path, name, older_wins, providers):
    older_holder = {}

    def _winner(p):
        if not older_wins or p.best_output is None:
            return p.best_output
        return older_holder.setdefault("o", _older(p))

    with (
        audit_failed_best_output(UnifiedPipeline),
        patch("socr.core.manifest._reaches_structure_class_branch", return_value=True),
        patch("socr.core.manifest.structure_class_grid_winner", side_effect=_winner),
    ):
        return _run(
            tmp_path, DROPPED, 0, check_runs=True, out_name=name, native=NATIVE, providers=providers
        )


@pytest.mark.parametrize("providers", [True, False], ids=["ladder", "no_ladder"])
def test_repair_event_agrees_with_the_shipped_reading(tmp_path, providers):
    _, _, st_best, best, pdf = _run_with_winner(tmp_path, "best", False, providers)
    _, _, st_old, old, _ = _run_with_winner(tmp_path, "old", True, providers)
    ev_best, ev_old = _events(best, KIND), _events(old, KIND)
    if not providers:
        # no model reading to repair: the arms cannot differ
        assert ev_best == ev_old == []
        return
    # control: best_output ships, so the repair ships and its event fires
    assert "| -.7 |" in _fragment(best) and OLDER_MARKER not in _fragment(best)
    assert len(ev_best) == 1
    # the older attempt ships (its marker is in the fragment), unrepaired, and nothing describes
    # bytes that did not ship
    assert OLDER_MARKER in _fragment(old)
    assert "| -.7 |" not in _fragment(old)
    assert ev_old == []
    assert all(not ps.point_repairs for ps in st_old.pages.values())
    assert any(ps.point_repairs for ps in st_best.pages.values())
    assert all("repairs" not in r for r in _sidecar(old).get("row_binding", {}).values())
    assert _sidecar(best)["row_binding"]["p1-t0"]["repairs"] == ev_best[0]["data"]["rows"]


def _native_fallback_run(tmp_path, name, audit_passed, providers, pdf=None):
    """OCR enhancement is wanted in both arms; only whether ``best_output`` passed its audit
    differs (an audit-passed reading ships; an audit-failed one yields to the native text)."""
    original = UnifiedPipeline._apply_text_coverage

    def _wrapped(self, state, page_num, ps, *a, **k):
        if ps.best_output is not None:
            ps.best_output.audit_passed = audit_passed
            ps.needs_ocr_enhancement = True
        return original(self, state, page_num, ps, *a, **k)

    with patch.object(UnifiedPipeline, "_apply_text_coverage", _wrapped):
        return _run(
            tmp_path,
            DROPPED,
            0,
            check_runs=True,
            out_name=name,
            native=NATIVE,
            providers=providers,
            pdf=pdf,
        )


@pytest.mark.parametrize("providers", [True, False], ids=["ladder", "no_ladder"])
def test_a_page_whose_model_reading_yields_to_the_native_fallback_records_no_repair(
    tmp_path, providers
):
    """Shipping boundary: the finalizer substitutes its native-fallback branch for an audit-failed
    model reading when OCR enhancement is wanted (on this table page the branch ships the
    fail-closed floor, not the native bytes), so the repaired model table never ships.  The arms differ in
    ``best_output.audit_passed`` alone."""
    _, _, st_model, model, pdf = _native_fallback_run(tmp_path, "model", True, providers)
    _, _, st_native, native, _ = _native_fallback_run(tmp_path, "native", False, providers, pdf)
    ev_model, ev_native = _events(model, KIND), _events(native, KIND)
    if not providers:
        assert ev_model == ev_native == []
        return
    # control: the audit-passed model reading is the winner, so its repair ships and is recorded
    assert "| -.7 |" in _fragment(model)
    assert len(ev_model) == 1
    # native text ships instead: its bytes are not the model's table, and nothing claims a repair
    assert _fragment(native) != _fragment(model)
    assert "| -7 |" not in _fragment(native) and "| -.7 |" not in _fragment(native)
    assert ev_native == []
    assert all(not ps.point_repairs for ps in st_native.pages.values())
    assert any(ps.point_repairs for ps in st_model.pages.values())
    assert all("repairs" not in r for r in _sidecar(native).get("row_binding", {}).values())
