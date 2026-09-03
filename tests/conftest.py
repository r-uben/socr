"""Test-only seam holding the stage-C assemble bucket contract (P6 stage C).

Stage C implements a two-rule assemble bucket contract:

1. **Flag-derived assemble buckets** (`d3_model_table_pages`, `d3_floor_pages`,
   `flagged_model_pages`) remain based on native-lane verdicts and PageState flags,
   matching the pre-change predicates in :func:`old_disposition_buckets` exactly.
2. **Migrated disposition buckets** (`structure_class_model_pages`,
   `structure_class_floor_pages`, `corrupt_math_hybrid_pages`) are derived solely
   from exact `PageDisposition` pair equality on finalized page records:
   - ``structure_class_model_pages`` -> ``(MODEL_OUTPUT, STRUCTURE_CLASS)``
   - ``structure_class_floor_pages`` -> ``(FAIL_CLOSED_MARKER, STRUCTURE_CLASS)``
   - ``corrupt_math_hybrid_pages``   -> ``(MODEL_OUTPUT, CORRUPT_MATH_HYBRID)``
   ``SelectionProvenance`` is never read for membership in these three buckets.
3. **Orthogonal assemble buckets** (`native_only_distrust_pages`, `value_drift_pages`,
   `fabricated_ref_pages`, `text_grid_rejected_pages`, `chart_detection_failed_pages`,
   `table_rejected_pages`, `table_unverified_pages`) remain based on configuration,
   page flags, events, and table-ladder terminals, matching
   :func:`old_orthogonal_assemble_buckets` exactly.

Two things live here:

* :func:`old_disposition_buckets` -- the pre-change predicates, kept verbatim as
  the stage-A/B reference.
* :func:`old_orthogonal_assemble_buckets` -- the pre-extraction orthogonal assemble
  predicates.
* :func:`assert_stage_c_disposition_buckets` -- stage-C two-rule assertion for
  disposition buckets.
* :func:`assert_orthogonal_buckets_unchanged` -- exact equality assertion for
  orthogonal assemble buckets.
* an autouse guard that wraps `orchestrator._derive_disposition_buckets` and
  `orchestrator._derive_orthogonal_assemble_buckets` for the WHOLE suite, so every
  fixture that drives `_phase_assemble` asserts conformance on every real assemble,
  with no per-module opt-in to forget.

The seam is test-only. Production code carries no pre-change path.
"""

from __future__ import annotations

import importlib

import pytest

from socr.core.manifest import (
    PageDisposition,
    PageEnding,
    PagePrimaryReason,
    finalized_page_records,
)
from socr.pipeline.orchestrator import _ORTHOGONAL_ASSEMBLE_BUCKET_NAMES

#: The six selection-shaped buckets.
P6_BUCKET_NAMES = (
    "d3_model_table_pages",
    "d3_floor_pages",
    "flagged_model_pages",
    "structure_class_model_pages",
    "structure_class_floor_pages",
    "corrupt_math_hybrid_pages",
)

#: The three flag-derived bucket names.
FLAG_DERIVED_BUCKET_NAMES = (
    "d3_model_table_pages",
    "d3_floor_pages",
    "flagged_model_pages",
)

#: The three migrated disposition bucket names and their exact PageDisposition pairs.
STAGE_C_MIGRATED_DISPOSITION_BUCKETS: dict[str, PageDisposition] = {
    "structure_class_model_pages": PageDisposition(
        PageEnding.MODEL_OUTPUT, PagePrimaryReason.STRUCTURE_CLASS
    ),
    "structure_class_floor_pages": PageDisposition(
        PageEnding.FAIL_CLOSED_MARKER, PagePrimaryReason.STRUCTURE_CLASS
    ),
    "corrupt_math_hybrid_pages": PageDisposition(
        PageEnding.MODEL_OUTPUT, PagePrimaryReason.CORRUPT_MATH_HYBRID
    ),
}

#: The seven orthogonal bucket names.
ORTHOGONAL_BUCKET_NAMES = _ORTHOGONAL_ASSEMBLE_BUCKET_NAMES


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


def old_orthogonal_assemble_buckets(state) -> dict[str, list[int]]:
    """The pre-refactor orthogonal assemble predicates, copied without simplification."""
    from socr.core.result import FailureMode
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
    # P1 (owner ruling Q2, 2026-09-03): a FOURTH orthogonal table bucket. Added
    # to this pre-refactor oracle deliberately, not to make a failing guard go
    # quiet: the guard's job is to prove the P6 extraction did not change
    # membership, and a new terminal that did not exist when the oracle was
    # written is a deliberate extension of the vocabulary, not drift. The three
    # table buckets stay mutually exclusive because ``_table_ladder_terminal``
    # returns exactly one mode per page.
    table_withheld_pages = sorted(
        n for n, p in state.pages.items() if _table_ladder_terminal(p) == FailureMode.TABLE_WITHHELD
    )

    return {
        "native_only_distrust_pages": native_only_distrust_pages,
        "value_drift_pages": value_drift_pages,
        "fabricated_ref_pages": fabricated_ref_pages,
        "text_grid_rejected_pages": text_grid_rejected_pages,
        "chart_detection_failed_pages": chart_detection_failed_pages,
        "table_rejected_pages": table_rejected_pages,
        "table_unverified_pages": table_unverified_pages,
        "table_withheld_pages": table_withheld_pages,
    }


@pytest.fixture
def p6_old_buckets():
    """Expose the pre-change predicates to a test that wants them explicitly."""
    return old_disposition_buckets


@pytest.fixture
def p6_old_orthogonal_buckets():
    """Expose the pre-extraction orthogonal predicates to a test that wants them explicitly."""
    return old_orthogonal_assemble_buckets


def assert_stage_c_disposition_buckets(state, records, new: dict[str, set[int]]) -> None:
    """Raise unless *new* satisfies the stage-C two-rule contract for *state* and *records*."""
    if records is None:
        records = finalized_page_records(state)

    old = old_disposition_buckets(state)

    # Rule 1: The three flag-derived buckets must match old_disposition_buckets(state) exactly.
    for name in FLAG_DERIVED_BUCKET_NAMES:
        expected = old[name]
        actual = new.get(name, set())
        if actual != expected:
            raise AssertionError(
                f"Stage-C contract violation on flag-derived bucket '{name}': "
                f"expected={sorted(expected)}, actual={sorted(actual)}. "
                "Violated rule: flag-derived buckets must match "
                "old_disposition_buckets(state) exactly."
            )

    # Rule 2: For each migrated bucket, membership must equal page numbers of records whose
    # disposition equals the bucket's exact pair.
    for name, target_pair in STAGE_C_MIGRATED_DISPOSITION_BUCKETS.items():
        expected = {r.output.page_num for r in records if r.disposition == target_pair}
        actual = new.get(name, set())
        if actual != expected:
            raise AssertionError(
                f"Stage-C contract violation on disposition-derived bucket '{name}': "
                f"expected={sorted(expected)}, actual={sorted(actual)}. "
                f"Violated rule: migrated bucket '{name}' must equal page numbers of records "
                f"whose disposition equals {target_pair}."
            )


def assert_orthogonal_buckets_unchanged(state, new: dict[str, list[int]]) -> None:
    """Raise unless *new* has exactly the pre-extraction orthogonal membership for *state*."""
    old = old_orthogonal_assemble_buckets(state)
    if new != old:
        drift = {
            name: {"old": old.get(name, []), "new": new.get(name, [])}
            for name in _ORTHOGONAL_ASSEMBLE_BUCKET_NAMES
            if old.get(name, []) != new.get(name, [])
        }
        raise AssertionError(
            f"P6 orthogonal assemble bucket membership changed: {drift}. "
            "Violated rule: orthogonal assemble buckets must match "
            "old_orthogonal_assemble_buckets(state) exactly."
        )


#: Separate call logs for disposition and orthogonal assemble bucket derivations.
DISPOSITION_GUARD_CALL_LOG: list[dict[str, set[int]]] = []
ORTHOGONAL_GUARD_CALL_LOG: list[dict[str, list[int]]] = []

#: Backwards-compatibility alias for tests referencing GUARD_CALL_LOG.
GUARD_CALL_LOG = DISPOSITION_GUARD_CALL_LOG


@pytest.fixture(autouse=True)
def _p6_bucket_difference_guard(monkeypatch):
    """Assert stage-C disposition contract and orthogonal equality on EVERY real `_phase_assemble`.

    This pins the stage-C two-rule contract across every fixture in the suite that drives
    assemble.
    """
    from socr.pipeline import orchestrator as _orch

    real_disp = _orch._derive_disposition_buckets
    real_orth = _orch._derive_orthogonal_assemble_buckets

    DISPOSITION_GUARD_CALL_LOG.clear()
    ORTHOGONAL_GUARD_CALL_LOG.clear()

    def _checked_disp(state, records):
        new = real_disp(state, records)
        assert_stage_c_disposition_buckets(state, records, new)
        DISPOSITION_GUARD_CALL_LOG.append({name: set(pages) for name, pages in new.items()})
        return new

    _checked_disp.__wrapped__ = getattr(real_disp, "__wrapped__", real_disp)
    monkeypatch.setattr(_orch, "_derive_disposition_buckets", _checked_disp)

    def _checked_orth(state):
        new = real_orth(state)
        assert_orthogonal_buckets_unchanged(state, new)
        ORTHOGONAL_GUARD_CALL_LOG.append({name: list(pages) for name, pages in new.items()})
        return new

    _checked_orth.__wrapped__ = getattr(real_orth, "__wrapped__", real_orth)
    monkeypatch.setattr(_orch, "_derive_orthogonal_assemble_buckets", _checked_orth)


# ---------------------------------------------------------------------------
# P1 (owner ruling Q3): the ladder is ON by default from 2026-09-03, so any
# test that builds a pipeline without overriding ``_build_table_judge_rungs``
# now constructs REAL rungs -- an ollama HTTP client (used by reader rung 1 AND,
# on its own model, by the blind-cell adjudicator) and a CLI subprocess.
#
# On a developer machine those are present (ollama up, ``agy`` on PATH), so the
# suite would make live model calls: slow,
# quota-spending, and above all MACHINE-DEPENDENT in exactly the way
# CLAUDE.md's #253/#257 note warns about -- the same test would take one path
# here and a different one in CI, where none of the three exists.
#
# This fixture pins the suite to CI's environment: no daemon, no binaries. It
# does not weaken any assertion and it does not touch the ladder flag -- a
# test that wants a rung still injects one, exactly as before.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _table_judge_rungs_are_absent(monkeypatch):
    import httpx

    def _no_daemon(*_args, **_kwargs):
        raise httpx.ConnectError("no ollama daemon (hermetic test environment)")

    def _no_binary(*_args, **_kwargs):
        raise FileNotFoundError("judge CLI not installed (hermetic test environment)")

    for module_path, seams in (
        ("socr.judge.table_rung_ollama", ("_post_chat",)),
        ("socr.judge.table_rung_gemini", ("_run_gemini_cli", "_run_health_check")),
    ):
        module = importlib.import_module(module_path)
        for seam in seams:
            monkeypatch.setattr(
                module,
                seam,
                _no_daemon if module_path.endswith("ollama") else _no_binary,
                raising=True,
            )
    monkeypatch.setattr("socr.judge.table_rung_ollama.httpx.get", _no_daemon, raising=True)
    monkeypatch.setattr("socr.judge.table_rung_gemini.shutil.which", lambda _b: None)
