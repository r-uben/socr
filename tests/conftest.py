"""Test-only seam holding the PRE-CHANGE assemble bucket predicates (P6 stage A/B).

Stage A/B is behaviour-preserving: the six selection-shaped assemble buckets must
have EXACTLY the membership the deleted `_phase_assemble` predicates produced. The
predicates below are reconstructed verbatim from `git show HEAD:src/socr/pipeline/
orchestrator.py` (the `_phase_assemble` bucket block) so the difference can be
asserted mechanically rather than argued.

Two things live here:

* :func:`old_disposition_buckets` -- the pre-change predicates, reconstructed.
* an autouse guard that wraps `orchestrator._derive_disposition_buckets` for the
  WHOLE suite, so every fixture that drives `_phase_assemble` (34 modules, plus
  the PP-2 golden corpus fixture) asserts old-membership == new-membership on
  every real assemble, with no per-module opt-in to forget.

The seam is test-only. Production code carries no pre-change path.
"""

from __future__ import annotations

import pytest

#: The six selection-shaped buckets stage B re-derives. `native_fallback_pages`,
#: `failed_pages` and the six orthogonal buckets are deliberately NOT here: they
#: stay flag/event/text-derived and their production code is untouched.
P6_BUCKET_NAMES = (
    "d3_model_table_pages",
    "d3_floor_pages",
    "flagged_model_pages",
    "structure_class_model_pages",
    "structure_class_floor_pages",
    "corrupt_math_hybrid_pages",
)


def old_disposition_buckets(state) -> dict[str, set[int]]:
    """The six buckets as `_phase_assemble` computed them BEFORE P6 stage B.

    Reconstructed verbatim from HEAD. The one substitution is `shipped_winner_kind`
    / `WinnerKind.CORRUPT_MATH_HYBRID`, which stage A renamed to
    `_select_page_output_tagged` / `SelectionProvenance.CORRUPT_MATH_HYBRID` with the
    16 rows and their order preserved; the call is made with no `whole_doc`, exactly
    as the old bucket did.
    """
    from socr.core.manifest import (
        SelectionProvenance,
        _select_page_output_tagged,
        d3_floor_kept_model_output,
        flagged_model_page_output,
        structure_class_floor_applies,
        structure_class_grid_winner,
    )

    d3_model_table_pages = {
        n for n, p in sorted(state.pages.items()) if d3_floor_kept_model_output(p) is not None
    }
    d3_floor_pages = {
        n
        for n, p in sorted(state.pages.items())
        if p.is_born_digital
        and p.native_table_structure_failed
        and (
            getattr(p, "native_table_unverifiable", False)
            or getattr(p, "native_table_header_unattributed", False)
        )
        and bool(p.attempts)
        and n not in d3_model_table_pages
    }
    flagged_model_pages = {
        n for n, p in sorted(state.pages.items()) if flagged_model_page_output(p) is not None
    }
    structure_class_model_pages = {
        n for n, p in sorted(state.pages.items()) if structure_class_grid_winner(p) is not None
    }
    structure_class_floor_pages = {
        n for n, p in sorted(state.pages.items()) if structure_class_floor_applies(p)
    }
    corrupt_math_hybrid_pages = {
        n
        for n in sorted(state.pages)
        if _select_page_output_tagged(state, n)[1] is SelectionProvenance.CORRUPT_MATH_HYBRID
    }
    return {
        "d3_model_table_pages": d3_model_table_pages,
        "d3_floor_pages": d3_floor_pages,
        "flagged_model_pages": flagged_model_pages,
        "structure_class_model_pages": structure_class_model_pages,
        "structure_class_floor_pages": structure_class_floor_pages,
        "corrupt_math_hybrid_pages": corrupt_math_hybrid_pages,
    }


@pytest.fixture
def p6_old_buckets():
    """Expose the pre-change predicates to a test that wants them explicitly."""
    return old_disposition_buckets


def assert_buckets_unchanged(state, new: dict[str, set[int]]) -> None:
    """Raise unless *new* has exactly the pre-change membership for *state*."""
    old = old_disposition_buckets(state)
    if new != old:
        drift = {
            name: {"old": sorted(old[name]), "new": sorted(new[name])}
            for name in P6_BUCKET_NAMES
            if old[name] != new[name]
        }
        raise AssertionError(f"P6 stage B changed assemble bucket membership: {drift}")


#: One entry per comparison the guard actually PERFORMED, appended by the wrapper and
#: cleared at the start of every test. Cold review round 2, finding 8: without this a
#: guard that is installed but never reached looks exactly like a guard that passes.
#: ``tests/test_p6_stage_ab_difference.py`` asserts a real ``_phase_assemble`` fills it,
#: and that removing the monkeypatch leaves it empty.
GUARD_CALL_LOG: list[dict[str, set[int]]] = []


@pytest.fixture(autouse=True)
def _p6_bucket_difference_guard(monkeypatch):
    """Assert old-membership == new-membership on EVERY real `_phase_assemble`.

    This is the difference pin the stage A/B acceptance bar asks for, applied to
    every fixture in the suite that drives assemble rather than to a hand-picked
    list. It pins a DIFFERENCE (old vs new derivation of the same run), never an
    absolute tuple, so it is provider-state independent and hermetic in CI.
    """
    from socr.pipeline import orchestrator as _orch

    real = _orch._derive_disposition_buckets
    GUARD_CALL_LOG.clear()

    def _checked(state, records):
        new = real(state, records)
        assert_buckets_unchanged(state, new)
        GUARD_CALL_LOG.append({name: set(pages) for name, pages in new.items()})
        return new

    _checked.__wrapped__ = real
    monkeypatch.setattr(_orch, "_derive_disposition_buckets", _checked)
