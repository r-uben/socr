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

import contextlib
import errno
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


# ---------------------------------------------------------------------------
# GH-984: no test may reach the AMBIENT Ollama daemon.
#
# CI has no Ollama, so a test that quietly reaches one passes there and
# behaves differently on a developer machine -- and with a live-but-busy daemon
# it blocks in a real generation call, hanging the whole local suite. This
# guard turns that leak class from a hang into a loud, attributed failure.
#
# It is deliberately NOT a blanket "no sockets" rule: only the host:port the
# deployment is configured to use (``OLLAMA_HOST``, default 127.0.0.1:11434),
# captured BEFORE the test body runs, is refused. Tests that stand up their own
# loopback server -- and point ``OLLAMA_HOST`` at it from inside the test -- keep
# working, as does any other loopback port.
#
# LIMITS (what this guard does NOT see):
# * Child processes. Exec'd subprocess engines (qwen-ocr, deepseek, ...) do not
#   inherit these monkeypatches, so a real subprocess launch can reach Ollama
#   unobserved. Unit tests must stub the subprocess launch boundary.
# * Proxies. A connection routed through an HTTP(S) proxy connects to the proxy
#   address, not the Ollama endpoint, so endpoint matching is bypassed.
# ---------------------------------------------------------------------------


def _ambient_ollama_endpoints() -> set[tuple[str, int]]:
    import socket
    from urllib.parse import urlsplit

    from socr.tables.extract import resolve_ollama_host

    endpoints = {("127.0.0.1", 11434), ("::1", 11434)}
    try:
        parts = urlsplit(resolve_ollama_host())
        host, port = parts.hostname, parts.port or 11434
        if host:
            endpoints.add((host, port))
            for info in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM):
                endpoints.add((info[4][0], info[4][1]))
    except (ValueError, OSError):
        pass  # unparseable/unresolvable host: the default endpoints still guard
    return endpoints


@contextlib.contextmanager
def ollama_connection_guard():
    """Refuse and record connections to the ambient Ollama endpoint; yield the record.

    Refusal alone is not enough: a probe's own ``except`` clause swallows it, so
    the caller must inspect the yielded list afterwards.
    """
    import socket
    import traceback

    forbidden = _ambient_ollama_endpoints()
    violations: list[str] = []
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def _hit(address) -> bool:
        if not (isinstance(address, tuple) and len(address) >= 2):
            return False
        if (address[0], address[1]) not in forbidden:
            return False
        sites = [
            f"{f.filename.rsplit('/', 1)[-1]}:{f.name}"
            for f in traceback.extract_stack()
            if "/src/socr/" in f.filename
        ]
        violations.append(f"{address[0]}:{address[1]} via " + " > ".join(sites))
        return True

    def _connect(self, address):
        if _hit(address):
            raise ConnectionRefusedError(f"GH-984: test reached live Ollama at {address[:2]}")
        return real_connect(self, address)

    def _connect_ex(self, address):
        if _hit(address):
            return errno.ECONNREFUSED
        return real_connect_ex(self, address)

    socket.socket.connect = _connect
    socket.socket.connect_ex = _connect_ex
    try:
        yield violations
    finally:
        socket.socket.connect = real_connect
        socket.socket.connect_ex = real_connect_ex


@pytest.fixture(autouse=True)
def _no_live_ollama():
    with ollama_connection_guard() as violations:
        yield
    if violations:
        pytest.fail(
            "GH-984: this test connected to the configured Ollama host:\n  "
            + "\n  ".join(violations)
            + "\nPatch the call (e.g. _resolve_judge_model -> '' and pin engines, "
            "see CLAUDE.md) instead of reaching a live daemon.",
            pytrace=False,
        )


# Modules whose pipelines run ``_run_fingerprint`` -> ``_resolve_judge_model``,
# which probes the ambient Ollama judge model with a real generation call. Their
# subject is never the judge probe, so they get it pinned to "no judge" (the
# outcome CI sees). This is an explicit opt-in list, NOT an autouse fixture: a
# NEW test file that reaches the probe fails the guard above instead of being
# silently hidden, and the modules that exercise ``_resolve_judge_model`` itself
# (e.g. test_gh873, test_gh903) are not on it.
#
# RISK: the pin is MODULE-WIDE, so it also silently covers every FUTURE test added
# to a listed module. A new test there that is meant to exercise the judge probe
# will see "" and never reach it, and the guard cannot flag what no longer
# connects. Put such a test in its own module, off these lists.
_JUDGE_PROBE_PINNED_MODULES = frozenset(
    (
        "test_a1c_header_binding_unverified_surfacing.py",
        "test_agentic_figures.py",
        "test_b2_routing.py",
        "test_canon_remediation.py",
        "test_canon_round2.py",
        "test_chart_lane.py",
        "test_dual_pass_tables.py",
        "test_equation_lane_pipeline_p4r.py",
        "test_gh165_unresolved_math_outcome.py",
        "test_gh171_sidecar_carries_figures.py",
        "test_gh177_exit_code_policy.py",
        "test_gh238_caption_engine_identity.py",
        "test_gh262_d3_marker_over_cached_grid.py",
        "test_gh317_structure_class_floor.py",
        "test_gh346_content_defect_clear_and_resume.py",
        "test_gh371_d3_region_splice.py",
        "test_gh488_figure_sidecar_end_to_end.py",
        "test_gh493_resume_figure_sidecar.py",
        "test_gh498_figure_repair_through_process.py",
        "test_gh519_visual_values_debt.py",
        "test_gh520_regional_floor_splice.py",
        "test_gh560_unwitnessed_wording.py",
        "test_gh625_ditto_unresolved.py",
        "test_gh635_chart_reader.py",
        "test_gh635_chart_table_skeletons.py",
        "test_gh649_scanned_prose_recovery.py",
        "test_gh652_prose_witness_trust.py",
        "test_gh658_no_witness_backend_reason.py",
        "test_gh659_label_unverified_finalization.py",
        "test_gh697_prose_recovery_surfacing.py",
        "test_gh713_judge_timeout_credential.py",
        "test_gh713_round2_credential_lifecycle.py",
        "test_gh713_round3_supersession_identity.py",
        "test_gh714_a1b_text_table_gate.py",
        "test_gh734b_wired_grid_reconciliation.py",
        "test_gh819_native_audit_resume.py",
        "test_gh916_native_ship_gate.py",
        "test_gh96_escalation_lane.py",
        "test_ladder_status_surfacing.py",
        "test_native_only_table_status_gh211.py",
        "test_orchestrator.py",
        "test_p6_cold_review_round2.py",
        "test_p6_disposition_finalization.py",
        "test_p6_stage_ab_difference.py",
        "test_p6_stage_c_difference.py",
        "test_pp1_fragment_flush.py",
        "test_qwen_fingerprint_determinants.py",
        "test_resume_source_version_gh214.py",
        "test_rotated_native_table_first.py",
        "test_s1_structure_class_winner_gh_reachability.py",
        "test_silent_content_destruction.py",
        "test_structural_gate_b1_gh151.py",
        "test_tr3_d3_floor.py",
    )
)


# Modules that drive ``_phase_agentic`` without pinning the provider ladder:
# ``_available_engines_for_agentic`` would otherwise probe every Ollama-backed
# engine's availability against the ambient daemon. Pinned to the local profile,
# as CLAUDE.md prescribes; a test that pins its own ladder overrides this.
_ENGINES_PINNED_MODULES = frozenset(
    (
        "test_chart_lane.py",
        "test_gh498_figure_repair_through_process.py",
        "test_gh519_visual_values_debt.py",
    )
)


@pytest.fixture
def judge_probe_pinned(monkeypatch):
    from socr.pipeline.orchestrator import UnifiedPipeline

    monkeypatch.setattr(UnifiedPipeline, "_resolve_judge_model", lambda self: "")


@pytest.fixture
def engines_pinned(monkeypatch):
    from socr.core.providers import PROFILE_QWEN_LOCAL
    from socr.pipeline.orchestrator import UnifiedPipeline

    monkeypatch.setattr(
        UnifiedPipeline, "_available_engines_for_agentic", lambda self: [PROFILE_QWEN_LOCAL]
    )


def pytest_collection_modifyitems(items):
    for item in items:
        if item.path.name in _JUDGE_PROBE_PINNED_MODULES:
            item.fixturenames.insert(0, "judge_probe_pinned")
        if item.path.name in _ENGINES_PINNED_MODULES:
            item.fixturenames.insert(0, "engines_pinned")
