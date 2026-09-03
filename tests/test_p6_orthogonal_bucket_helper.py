"""P6: the orthogonal assemble buckets remain mechanically unchanged.

The reference below is deliberately kept in test code. It is the predicate block
that existed in ``_phase_assemble`` before extraction; the production helper is
then compared with it over the reusable P6 corpus and focused shapes for every
orthogonal key.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

fitz = pytest.importorskip("fitz")

from p6_corpus_fixture import build_corpus_state, make_pdf  # noqa: E402

from socr.core.audit_log import AuditEvent  # noqa: E402
from socr.core.document import DocumentHandle  # noqa: E402
from socr.core.result import FailureMode, PageOutput, PageStatus  # noqa: E402
from socr.core.state import DocumentState  # noqa: E402
from socr.pipeline.orchestrator import (  # noqa: E402
    _ORTHOGONAL_ASSEMBLE_BUCKET_NAMES,
    _derive_orthogonal_assemble_buckets,
)

EXPECTED_KEYS = (
    "native_only_distrust_pages",
    "value_drift_pages",
    "fabricated_ref_pages",
    "text_grid_rejected_pages",
    "chart_detection_failed_pages",
    "table_rejected_pages",
    "table_unverified_pages",
)


def _old_orthogonal_assemble_buckets(state):
    """The pre-refactor predicates, copied without semantic simplification."""
    from socr.pipeline.orchestrator import _table_ladder_terminal

    config = getattr(state, "_assemble_config", None)
    native_only = bool(getattr(config, "native_only", False))
    native_only_distrust_pages = [
        n
        for n, p in sorted(state.pages.items())
        if p.is_born_digital
        and p.native_text
        and native_only
        and getattr(p, "native_table_unverifiable", False)
        and not p.native_table_structure_failed
        and p.attempts
        and all((a.engine or "").startswith("native") for a in p.attempts)
        and not (p.best_output and p.best_output.audit_passed)
    ]
    value_drift_pages = sorted(
        {
            getattr(e, "page_num", 0)
            for e in state.events
            if getattr(e, "kind", "") == "table_value_drift_unadjudicated"
            and getattr(e, "page_num", 0)
        }
    )
    fabricated_ref_pages = sorted(
        n for n, p in state.pages.items() if getattr(p, "fabricated_image_refs", 0)
    )
    text_grid_rejected_pages = sorted(
        n for n, p in state.pages.items() if getattr(p, "text_grid_rejected", False)
    )
    chart_detection_failed_pages = sorted(
        n for n, p in state.pages.items() if getattr(p, "chart_asset_detection_failed", False)
    )
    table_rejected_pages = sorted(
        n for n, p in state.pages.items() if _table_ladder_terminal(p) == FailureMode.TABLE_REJECTED
    )
    table_unverified_pages = sorted(
        n
        for n, p in state.pages.items()
        if _table_ladder_terminal(p) == FailureMode.TABLE_UNVERIFIED
    )
    return {
        "native_only_distrust_pages": native_only_distrust_pages,
        "value_drift_pages": value_drift_pages,
        "fabricated_ref_pages": fabricated_ref_pages,
        "text_grid_rejected_pages": text_grid_rejected_pages,
        "chart_detection_failed_pages": chart_detection_failed_pages,
        "table_rejected_pages": table_rejected_pages,
        "table_unverified_pages": table_unverified_pages,
    }


def _focused_state(tmp_path):
    path = tmp_path / "orthogonal.pdf"
    doc = fitz.open()
    for page_num in range(1, 8):
        doc.new_page().insert_text((54, 72), f"orthogonal page {page_num}")
    doc.save(path)
    doc.close()

    state = DocumentState(handle=DocumentHandle.from_path(path))
    state._assemble_config = SimpleNamespace(native_only=True)

    native = PageOutput(
        page_num=1,
        text="native text",
        status=PageStatus.WARNING,
        engine="native",
        audit_passed=False,
    )
    native_page = state.pages[1]
    native_page.is_born_digital = True
    native_page.native_text = native.text
    native_page.native_table_unverifiable = True
    native_page.attempts.append(native)
    native_page.best_output = native

    state.events.append(
        AuditEvent(
            page_num=2,
            kind="table_value_drift_unadjudicated",
            engine="native",
            detail="focused test",
        )
    )
    state.pages[3].fabricated_image_refs = 1
    state.pages[4].text_grid_rejected = True
    state.pages[5].chart_asset_detection_failed = True

    rejected = PageOutput(
        page_num=6,
        text="rejected",
        status=PageStatus.WARNING,
        engine="qwen",
        audit_passed=False,
    )
    state.pages[6].best_output = rejected
    state.pages[6].table_ladder_disposition = FailureMode.TABLE_REJECTED

    unverified = PageOutput(
        page_num=7,
        text="unverified",
        status=PageStatus.WARNING,
        engine="qwen",
        audit_passed=False,
        failure_mode=FailureMode.TABLE_UNVERIFIED,
    )
    state.pages[7].best_output = unverified
    return state


@pytest.mark.parametrize("builder", ["corpus", "focused"])
def test_orthogonal_helper_matches_pre_refactor_predicates(tmp_path, builder) -> None:
    if builder == "corpus":
        state = build_corpus_state(make_pdf(tmp_path))
        state._assemble_config = SimpleNamespace(native_only=False)
    else:
        state = _focused_state(tmp_path)

    assert _derive_orthogonal_assemble_buckets(state) == _old_orthogonal_assemble_buckets(state)


def test_orthogonal_helper_has_exact_key_inventory(tmp_path) -> None:
    state = _focused_state(tmp_path)
    buckets = _derive_orthogonal_assemble_buckets(state)

    assert _ORTHOGONAL_ASSEMBLE_BUCKET_NAMES == EXPECTED_KEYS
    assert tuple(buckets) == EXPECTED_KEYS


def test_orthogonal_helper_does_not_mutate_state(tmp_path) -> None:
    state = _focused_state(tmp_path)
    pages = dict(state.pages)
    events = list(state.events)
    _derive_orthogonal_assemble_buckets(state)

    assert state.pages == pages
    assert state.events == events
