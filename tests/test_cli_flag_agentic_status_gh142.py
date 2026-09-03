"""GH-142: every CLI-backed config field must have a known status on the agentic path.

Two flags were found promising behaviour the default agentic path does not
deliver (`--max-cost-per-page` not constraining `$0.00` cloud rungs, `--no-audit`
inert — #139). A sweep found eleven more that are hashed into the run fingerprint
but never acted on, and three that are neither hashed nor read.

A flag that lies is worse than a missing one: the user believes a constraint is
in force and scripts around it. So this test pins the status of every field and
fails when one changes category — or when a new field is added without being
classified at all.

METHOD. A recording proxy logs every config field read AND its caller during a
hermetic agentic run. Caller attribution is load-bearing: `_run_fingerprint`
reads most of the config on every page flush purely to hash it, so "was read" is
not evidence a flag does anything. A field whose only readers are bookkeeping is
hashed but never acted on.

LIMIT, stated because it is easy to mistake for a result: instrumentation cannot
distinguish "dead" from "read only on a path this fixture never takes". Fields in
`_UNEXERCISED` are NOT claims of deadness — they are fields whose consumers are
stubbed or unreached here, verified by reading the source instead.
"""

from __future__ import annotations

import dataclasses
from unittest.mock import patch

import fitz
import pytest

from socr.core.config import PipelineConfig
from socr.core.result import PageOutput, PageStatus
from socr.pipeline.agentic import AcceptDecision
from socr.pipeline.orchestrator import UnifiedPipeline

# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------

#: Read by real logic during an agentic run — the flag does something.
#: VERIFIED by `test_live_flags_are_actually_live`; do not add by inspection.
_LIVE = {
    "agentic",
    "audit_min_words",
    "cost_budget",
    "detect_equations",
    "dual_pass_tables",
    # P4-R: read for every page by `_is_equation_region_lane_page`, which
    # `_is_agentic_trusted_native` consults before the free native bypass.
    "equation_region_lane",
    "escalate_ambiguous_tables",
    "figures_max_per_page",
    "figures_max_total",
    "max_cost_per_page",
    "math_model",
    "native_first",
    "native_only",
    "primary_engine",
    "quiet",
    "qwen_backend",
    "qwen_model",
    "recover_clean_equations",
    "recover_corrupt_math",
    "render_dpi",
    "reprocess",
    "save_figures",
    "strict_local",
    "write_manifest",
    # GH-353 TICKET-B1: read unconditionally by `_build_table_judge_rungs`
    # (doc-scoped, called once per page-major run before the per-page loop)
    # to decide whether to construct the ladder's rung sequence at all --
    # acted on even when False, since that is the branch that returns `[]`.
    "table_judge_ladder",
}

#: Hashed into the run fingerprint but never acted on. R174b DELETED every consumer
#: named here (`_phase_repair`, `_backbone_native_first`, `_phase_consensus`,
#: `_phase_judge_hard_pages`, and the single-/multi-engine branches of `process`),
#: so these are no longer merely inert — they are DEAD. Five former members
#: (consensus_enabled, consensus_ollama_model, consensus_use_llm, max_retries,
#: truncation_retries) were deleted outright as config fields. The survivors below
#: still over-invalidate the fingerprint; #142 should be revisited now that the
#: lane that justified them is gone.
#:
#: These are not harmless: the fingerprint OVER-invalidates, so toggling a flag
#: that changes nothing reprocesses an entire corpus.
_INERT_BUT_FINGERPRINTED = {
    "chunk_size",
    "chunk_threshold",
    "local_engine",
    "tiered",
    "figures_engine",
    # GH-530 moved this here from `_INERT_AND_UNFINGERPRINTED`, which claimed it
    # was "absent from the fingerprint". It is not: `clean_equation_model` is in
    # the `extra` dict `_run_fingerprint` builds. The misclassification was
    # invisible because the guard for that set checks `_acted_on`, which filters
    # out `<bookkeeping>` reads -- so a field read ONLY by the fingerprint
    # satisfied both sets. Found by the assertion added below, which looks at
    # the dict instead of the readers.
    #
    # Note this is the BETTER position to be in: the note on the old set warned
    # that absence hides the field from the resume ledger in the modes where it
    # DOES work (#133's class). Being fingerprinted means that gap does not
    # exist for this field.
    "clean_equation_model",
}

#: Not read on the agentic path, and genuinely absent from the run fingerprint.
#:
#: GH-525 moved these two here from `_INERT_BUT_FINGERPRINTED`: they gate
#: nothing (GH-142 rejected their CLI flags for it) and their keys are now
#: dropped from the fingerprint, so a YAML-only toggle can no longer invalidate
#: terminal pages for a run that behaves identically. That absence costs nothing
#: here -- neither field has a mode in which it DOES work, so there is no
#: resume-ledger blind spot traded for it -- and `_warn_inert_config` names them
#: at run start when a config sets them, so ignored is not the same as silent.
_INERT_AND_UNFINGERPRINTED = {
    "fallback_chain",
    "judge_hard_pages",
}

#: NOT a deadness claim. Consumers are stubbed or unreached by this fixture:
#: engine-internal settings, the escalation lane (needs a second ladder rung),
#: and HPC routing. Verified by reading the source, not by this run.
_UNEXERCISED = {
    "enabled_engines",  # consumed by _available_engines_for_agentic, stubbed here
    "escalation_timeout_sec",  # escalation lane needs a 2nd ladder rung
    "hpc",
    "marker_device",
    "deepseek_vllm_url",
    "qwen_vllm_model",
    "qwen_model_pinned",
    "judge_backend",  # read by _build_page_judge, stubbed here
    # Read only inside the table-extractor block, which is gated on
    # _resolve_crop_vlm_model() finding a live vision model. Pinned to None in
    # the fixture so this classification does not depend on whether the machine
    # running the tests happens to have one pulled.
    "auto_patch_tables",
    "qwen_vllm_url",
    # GH-222: read by `_probe_backend_idle`, which the cascade-halt guard calls
    # only after an attempt times out. This fixture produces no timeout, so the
    # field is genuinely unexercised here rather than dead — the probe is
    # covered directly by tests/test_gh222_probe_host.py.
    "ollama_host",
    "describe_figures",  # read by the figure-description lane, not reached here
    # Per-engine model/task settings. Read by the engine when it actually runs;
    # `_run_engine_on_pages` is stubbed here, so no engine subprocess is ever
    # constructed. Their appearance under `_engine_determinants` is fingerprint
    # bookkeeping only, which is why correct caller attribution matters.
    "gemini_model",
    "gemini_task",
    "glm_backend",
    "glm_task",
    "deepseek_backend",
    "deepseek_task",
    "mistral_model",
    "nougat_model",
    # Both real consumers (_build_page_judge, _resolve_crop_vlm_model) are
    # stubbed; the remaining reader is _resolve_judge_model under the fingerprint.
    "judge_model",
    # Consumed outside the orchestrator entirely (cli.py, engine subprocesses):
    "dry_run",
    "verbose",
    "workers",
    "timeout",
    "output_dir",
    # GH-353 TICKET-B1: read and conditionally fingerprinted only when
    # `table_judge_ladder` is True (inside `_build_table_judge_rungs` /
    # `_run_fingerprint`'s ternary, which short-circuits the untaken branch).
    # This fixture's config leaves the flag at its False default, so these
    # never get touched here -- covered directly by
    # tests/test_table_judge_gate.py, which turns the flag on.
    "table_judge_rung1_model",
    "table_judge_rung1_host",
    "table_judge_rung2_binary",
    "table_judge_timeout_sec",
    # P1: the blind-cell adjudicator's identity and per-call cost. Same
    # classification and the same reason as the rung fields above -- read and
    # fingerprinted only when the ladder flag is on, and only on the two ruled
    # guard-chain paths (a two-low PASS pair, or a reader rejection).
    "table_judge_adjudicator_model",
    "table_judge_adjudicator_host",
    "table_judge_adjudicator_cost_per_call_usd",
}

#: `_warn_inert_config` reads the inert fields in order to REPORT that they are
#: being ignored (GH-525). Reading a value to say it does not matter is not
#: acting on it -- crediting it as real behaviour would reclassify the two
#: fields this guard exists to keep classified as inert, and the run's own
#: warning would be the evidence that they are live.
_BOOKKEEPING = {
    "_run_fingerprint",
    "_engine_determinants",
    "_write_metadata",
    "_warn_inert_config",
}

#: Frames to scan above a config read when attributing it. Must be deep enough
#: to see a bookkeeping caller through its helpers: `_run_fingerprint` ->
#: `_engine_determinants` -> `BaseEngine.resolved_model_version` -> the read is
#: already three frames, and the fingerprint itself is reached from
#: `_flush_page_sidecar` and `_load_terminal_page`. Twelve leaves margin without
#: walking the whole stack on every attribute access.
_STACK_SCAN_DEPTH = 12

#: Recorded instead of a function name when ANY frame in the scanned window is
#: bookkeeping. Attributing to the innermost frame alone is not enough: engine
#: determinants are read inside `resolved_model_version` /
#: `fingerprint_determinants`, whose names are not in `_BOOKKEEPING`, so a
#: fingerprint-only read would be credited to the helper and counted as real
#: behaviour — silently classifying an inert flag as live, which is exactly what
#: this guard exists to prevent. Caught in review of this file's first version.
_BOOKKEEPING_TOKEN = "<bookkeeping>"


# --------------------------------------------------------------------------
# Hermetic agentic run with a recording config
# --------------------------------------------------------------------------


class _Recorder:
    """Transparent config proxy recording each field read and its caller."""

    def __init__(self, inner, fields, calls):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_fields", fields)
        object.__setattr__(self, "_calls", calls)

    def __getattr__(self, name):
        import sys

        fields = object.__getattribute__(self, "_fields")
        if name in fields:
            # Scan the whole window, not just the innermost frame: a read done
            # inside an engine helper on behalf of the fingerprint must be
            # credited to the fingerprint, not to the helper.
            frames: list[str] = []
            frame = sys._getframe(1)
            depth = 0
            while frame is not None and depth < _STACK_SCAN_DEPTH:
                fn = frame.f_code.co_name
                if fn not in ("__getattr__", "<genexpr>", "<listcomp>", "<dictcomp>"):
                    frames.append(fn)
                frame = frame.f_back
                depth += 1
            attributed = (
                _BOOKKEEPING_TOKEN
                if any(fn in _BOOKKEEPING for fn in frames)
                else (frames[0] if frames else "<unknown>")
            )
            object.__getattribute__(self, "_calls").setdefault(name, set()).add(attributed)
        return getattr(object.__getattribute__(self, "_inner"), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_inner"), name, value)


class _YesJudge:
    def assess(self, output, provider):
        return AcceptDecision(accept=True, reason="stub")


def _fixture_pdf(path):
    doc = fitz.open()
    # prose
    page = doc.new_page()
    y = 80
    for _ in range(14):
        page.insert_text(
            (60, y), "Estimated coefficient 0.082 significant at 1 percent", fontsize=9
        )
        y += 16
    # table
    page = doc.new_page()
    y = 80
    for row in range(10):
        page.insert_text((60, y), f"Row{row}   {row}.12   {row}.34", fontsize=9)
        y += 16
    page.draw_line(fitz.Point(55, 70), fitz.Point(360, 70))
    page.draw_line(fitz.Point(55, y), fitz.Point(360, y))
    # chart
    page = doc.new_page()
    page.insert_text((60, 80), "Figure 1", fontsize=10)
    for i in range(10):
        page.draw_line(fitz.Point(60 + i * 20, 300), fitz.Point(80 + i * 20, 260 - i * 3))
    # scanned (no text layer)
    doc.new_page().draw_rect(fitz.Rect(50, 50, 400, 300))
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture(scope="module")
def readers(tmp_path_factory):
    """field -> set of functions that read it during a hermetic agentic run."""
    from socr.core import providers

    fields = {f.name for f in dataclasses.fields(PipelineConfig)}
    calls: dict[str, set[str]] = {}
    tmp = tmp_path_factory.mktemp("flagsweep")

    cfg = PipelineConfig(
        agentic=True,
        quiet=True,
        detect_equations=True,
        recover_clean_equations=True,
        recover_corrupt_math=True,
        save_figures=True,
    )
    pipe = UnifiedPipeline(_Recorder(cfg, fields, calls))

    detect = pipe.bd_detector.detect

    def detect_with_corrupt_math(path):
        assessment = detect(path)
        assessment.pages[0].has_corrupt_math = True
        assessment.pages[0].has_equations = True
        assessment.pages[0].needs_ocr_enhancement = True
        return assessment

    pipe.bd_detector.detect = detect_with_corrupt_math
    pipe._available_engines_for_agentic = lambda: [providers.PROFILE_QWEN_LOCAL]
    pipe._build_page_judge = lambda state: _YesJudge()
    # Cold review round 1, finding 8: `_run_fingerprint` resolves the judge
    # identity, and the resolver PROBES Ollama over HTTP. Left live, this
    # fixture made up to three real network calls and its result depended on
    # local Ollama state -- the exact non-hermeticity this repo's CI trap
    # documents. Pinned so the classification is the same everywhere.
    pipe._resolve_judge_model = lambda: ""
    # Determinism, not convenience: _resolve_crop_vlm_model probes Ollama over
    # HTTP. Left live, the classification depends on whether the developer
    # happens to have a vision model pulled — the table-extractor block (and
    # everything it reads) runs on a workstation and is skipped in CI. That is
    # the exact local-passes/CI-fails trap this repo documents, and it failed
    # this test's first CI run on `qwen_vllm_url`. Pinned to None so the fixture
    # takes the same branch everywhere.
    pipe._resolve_crop_vlm_model = lambda: None
    pipe._run_engine_on_pages = lambda state, nums, nat, eng, phase, profile=None: [
        PageOutput(
            page_num=p,
            text=f"| a | b |\n| --- | --- |\n| 1 | 2 |\n\ntext {p}",
            status=PageStatus.SUCCESS,
            engine=str(getattr(eng, "value", eng)),
        )
        for p in nums
    ]

    with patch("socr.math.recover.recover_math_regions", return_value=[]):
        pipe.process(_fixture_pdf(tmp / "f.pdf"), output_dir=tmp / "out")
    return calls


def _acted_on(readers) -> set[str]:
    return {f for f, who in readers.items() if who - {_BOOKKEEPING_TOKEN}}


# --------------------------------------------------------------------------
# The guard
# --------------------------------------------------------------------------


def test_every_config_field_is_classified():
    """A new flag must be classified, not silently assumed to work.

    This is the assertion that stops recurrence: adding a field without deciding
    whether it does anything on the default path now fails CI.
    """
    known = _LIVE | _INERT_BUT_FINGERPRINTED | _INERT_AND_UNFINGERPRINTED | _UNEXERCISED
    actual = {f.name for f in dataclasses.fields(PipelineConfig)}

    assert actual - known == set(), (
        f"unclassified config field(s): {sorted(actual - known)}. Decide whether each "
        "works on the agentic path, is non-agentic by design, or should be rejected "
        "as an incompatible combination (see #142)."
    )
    assert known - actual == set(), (
        f"classification lists a field that no longer exists: {sorted(known - actual)}"
    )


@pytest.mark.parametrize("field", sorted(_INERT_BUT_FINGERPRINTED))
def test_inert_fields_are_still_inert(field, readers):
    """If one of these starts doing something, the classification is stale.

    Failing here is good news — it means a flag was wired up — but the lists and
    the help text must be updated to match.
    """
    assert field not in _acted_on(readers), (
        f"{field} is now acted on during an agentic run; move it to _LIVE and "
        "update the CLI help text (#142)"
    )


@pytest.mark.parametrize("field", sorted(_INERT_AND_UNFINGERPRINTED))
def test_unfingerprinted_inert_fields_stay_out_of_the_fingerprint(field, readers):
    assert field not in _acted_on(readers)


def test_known_live_flags_really_are_live(readers):
    """The positive control: without this the guard could pass on a broken run.

    A subset that must be exercised by this fixture specifically — if these stop
    being read, the harness has silently stopped driving the pipeline (which has
    happened once already, via an unwritten fixture directory).
    """
    must_be_live = {
        "agentic",
        "native_first",
        "dual_pass_tables",
        "detect_equations",
        "max_cost_per_page",
        "strict_local",
    }
    missing = must_be_live - _acted_on(readers)
    assert missing == set(), f"harness is not driving the pipeline; unread: {sorted(missing)}"


@pytest.mark.parametrize("field", sorted(_LIVE))
def test_live_flags_are_actually_live(field, readers):
    """Every _LIVE entry must be observed, not assumed.

    Without this the classification could quietly rot in the dangerous
    direction: a flag that stopped working would sit in _LIVE and no test would
    notice. Assembling this list by inspection got 10 of 42 entries wrong on the
    first attempt, which is exactly why it must be asserted rather than trusted.
    """
    assert field in _acted_on(readers), (
        f"{field} is classified _LIVE but nothing outside bookkeeping read it during "
        "an agentic run — either it stopped working, or it belongs in _UNEXERCISED "
        "with a note on which consumer this fixture does not reach (#142)"
    )


# --------------------------------------------------------------------------
# The recorder's own correctness
# --------------------------------------------------------------------------


def test_reads_through_a_bookkeeping_helper_are_attributed_to_bookkeeping():
    """Regression: attribution must look past the innermost frame.

    The first version of this file recorded only the innermost non-recorder
    frame. Engine determinants are read inside `resolved_model_version` /
    `fingerprint_determinants`, which are called BY `_engine_determinants` for
    the fingerprint — so those reads were credited to the helper and counted as
    real behaviour. Nine fields were classified _LIVE on that basis, and the
    suite passed, "verifying" them.

    That is the exact failure this guard exists to prevent, so it gets its own
    test rather than resting on the classification lists being right.
    """
    calls: dict[str, set[str]] = {}
    cfg = PipelineConfig()
    rec = _Recorder(cfg, {"gemini_model"}, calls)

    def resolved_model_version():
        return rec.gemini_model

    def _engine_determinants():  # a name in _BOOKKEEPING
        return resolved_model_version()

    _engine_determinants()

    assert calls["gemini_model"] == {_BOOKKEEPING_TOKEN}
    assert "resolved_model_version" not in calls["gemini_model"]


def test_reads_outside_bookkeeping_keep_their_real_caller():
    """The other direction: genuine behaviour must not be swallowed as bookkeeping."""
    calls: dict[str, set[str]] = {}
    rec = _Recorder(PipelineConfig(), {"strict_local"}, calls)

    def _phase_agentic():
        return rec.strict_local

    _phase_agentic()

    assert calls["strict_local"] == {"_phase_agentic"}


@pytest.mark.parametrize("field", sorted(_INERT_AND_UNFINGERPRINTED))
def test_unfingerprinted_fields_really_are_absent_from_the_fingerprint(field):
    """The set's NAME, asserted (GH-530).

    `test_unfingerprinted_inert_fields_stay_out_of_the_fingerprint` above checks
    `field not in _acted_on(readers)` -- but a fingerprint read is attributed to
    `<bookkeeping>` and `_acted_on` filters that token out, so a field read ONLY
    by `_run_fingerprint` passes it either way. The check proves the field is
    inert; it does not prove the second half of the name.

    Found by probing the GH-530 reclassification: putting `judge_hard_pages`
    back into the fingerprint dict left this file green. This closes that, by
    looking at the dict the fingerprint is actually built from.
    """
    import ocr_output_contract as contract

    from socr.core.config import PipelineConfig
    from socr.pipeline.orchestrator import UnifiedPipeline

    captured = {}
    real = contract.run_fingerprint

    def spy(model, backend, task, prompt, extra=None):
        captured["extra"] = extra or {}
        return real(model, backend, task, prompt, extra=extra)

    # `run_fingerprint` is imported inside `_run_fingerprint`, so patching the
    # contract module is what the call resolves through.
    #
    # `_resolve_judge_model` is patched for hermeticity (cubic P2 on #532): it
    # probes Ollama, so without this each parametrised case waits on a local
    # daemon that CI does not have and a workstation may or may not -- the exact
    # local-passes/CI-fails trap this repo documents, and the reason the fixture
    # above does the same.
    pipe = UnifiedPipeline(PipelineConfig(quiet=True))
    pipe._resolve_judge_model = lambda: ""
    with patch.object(contract, "run_fingerprint", spy):
        pipe._run_fingerprint()

    assert captured, "the fingerprint was not built, so this asserts nothing"
    assert field not in captured["extra"], (
        f"{field} is classified unfingerprinted but appears in the run "
        f"fingerprint. Either move it to _INERT_BUT_FINGERPRINTED, or drop the "
        f"key -- a config-only toggle otherwise invalidates terminal pages for a "
        f"run that behaves identically (GH-525)."
    )
