"""Preflight (#1080): check that every tool the run's ENABLED features need is there.

A run used to discover a missing tool page by page and report it after the
fact. Corpus run 22922695 on Rorqual lost 181 scanned-table pages in 74 papers
to an absent Tesseract witness, and 2,748 formula regions on 1,159 pages to
maths readers that could not address a vLLM-only cluster; the default page
judge had earlier returned 410 for days while still listed by ``/api/tags``.
Each is a yes/no fact the machine answers exactly, so this is a plain check, not
a model.

Every probe is one small function returning a :class:`Check`
``(name, needed, ok, detail)`` and calls an existing probe rather than
re-implementing it. ``needed`` is derived from the config the run will use, so a
disabled feature never makes its tool required.

What this module does not do: it does not read a page, never sends a request to
a cloud model that ``--strict-local`` / ``--max-cost-per-page 0`` forbids (such a
check reports MISSING, "forbidden", without probing), and cannot say a tool will
keep working for the length of the run. It does not retry: a transient probe
failure stops the run, loudly and fail-closed. Under ``socr batch`` / workers
each process probes independently, so one failure stops only its own shard.

Every CLI path that processes pages through the pipeline is gated (``process``,
``batch``, ``library``, ``process --hpc-sequential``). Two are deliberately not:
the Python API (``UnifiedPipeline.process``; a caller who wants the check calls
``run_preflight`` itself) and ``socr benchmark run``, which runs bare engines
through ``BenchmarkRunner`` (own ``is_available`` skip, no judge, Tesseract or
maths lane), so the pipeline's checks do not describe it.

Preflight cannot know whether an input has scanned pages, so every run is treated
as possibly using a model, ``--native-only`` included (genuine scans still route
to OCR under it). A convert-only run on a machine without a model opts out per
tool with ``--allow-missing``, and the run records what it went without.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from socr.core.ollama_utils import DEFAULT_PROBE_TIMEOUT_SEC, probe_model_generation
from socr.core.providers import is_cloud_model, zero_cap_pinned_forbids_cloud
from socr.engines.registry import get_engine, has_cli_engine
from socr.tables.extract import probe_openai_server_idle, resolve_ollama_host
from socr.tables.source_evidence import WITNESS_READING, classical_ocr_backend_state

#: The check names, which are also the values ``--allow-missing`` accepts.
CHECK_MODEL_SERVER = "model-server"
CHECK_ENGINE_CLI = "engine-cli"
CHECK_PAGE_JUDGE = "page-judge"
CHECK_TESSERACT = "tesseract"
CHECK_MATHS_READERS = "maths-readers"
CHECK_TABLE_JUDGE = "table-judge"
CHECK_NAMES: tuple[str, ...] = (
    CHECK_MODEL_SERVER,
    CHECK_ENGINE_CLI,
    CHECK_PAGE_JUDGE,
    CHECK_TESSERACT,
    CHECK_MATHS_READERS,
    CHECK_TABLE_JUDGE,
)

#: Report status words, one per row.
STATUS_PRESENT = "present"
STATUS_MISSING = "MISSING"
STATUS_ALLOWED = "MISSING (allowed)"
STATUS_NOT_NEEDED = "not needed"


@dataclass(frozen=True)
class Check:
    """One probe result. ``ok`` is meaningful whether or not the tool is ``needed``."""

    name: str
    needed: bool
    ok: bool
    detail: str = ""

    @property
    def missing(self) -> bool:
        """A needed tool that is not there: the only state that stops a run."""
        return self.needed and not self.ok


def _cloud_forbidden(config, model: str) -> str:
    """Why a request to *model* must not be sent under this run's policy, else "".

    Same helpers the pipeline's own policy checks use (``is_cloud_model``,
    ``zero_cap_pinned_forbids_cloud``): preflight must never be the one request
    that crosses a boundary the run itself would refuse.
    """
    if not is_cloud_model(model):
        return ""
    if config.strict_local:
        return f"{model}: forbidden by --strict-local (cloud model, not probed)"
    if zero_cap_pinned_forbids_cloud(config):
        return f"{model}: forbidden by --max-cost-per-page 0 (cloud model, not probed)"
    return ""


def _ollama_host(config) -> str:
    return resolve_ollama_host(getattr(config, "ollama_host", None)).rstrip("/")


def _probe_ollama_model(config, model: str) -> tuple[bool, str]:
    """``(ok, detail)`` for a real generation on *model*, policy-checked first."""
    forbidden = _cloud_forbidden(config, model)
    if forbidden:
        return False, forbidden
    host = _ollama_host(config)
    ok, reason = probe_model_generation(host, model, DEFAULT_PROBE_TIMEOUT_SEC)
    label = f"ollama {host} model {model}"
    return ok, label if ok else f"{label}: {reason}"


def check_model_server(config) -> Check:
    """The configured model server answers a real generation on its resolved backend.

    ``probe_model_generation`` and not ``/api/tags``: a listing answers 200 in
    0.03 s on a jammed GPU. A vLLM/API backend is checked with
    ``probe_openai_server_idle`` (listing, then a 1-token generation canary,
    GH-221). Under the HPC lane the server is the one the pipeline starts itself
    (``manage_server``), so only an operator-run server is checked.
    """
    from socr.core.config import EngineType
    from socr.engines.qwen import LOCAL_BACKENDS, resolve_qwen_intent

    if config.hpc.enabled:
        if config.hpc.manage_server:
            return Check(CHECK_MODEL_SERVER, False, True, "HPC lane starts its own vLLM server")
        url, model = config.hpc.vllm_url, config.hpc.ocr_model
        forbidden = _cloud_forbidden(config, model)
        if forbidden:
            return Check(CHECK_MODEL_SERVER, True, False, forbidden)
        ok = probe_openai_server_idle(url, model=model)
        label = f"vllm {url} model {model}"
        return Check(
            CHECK_MODEL_SERVER, True, ok, label if ok else f"{label}: no generation answered"
        )
    if EngineType.QWEN not in config.enabled_engines:
        return Check(CHECK_MODEL_SERVER, False, True, "qwen is not an enabled engine")
    backend, model = resolve_qwen_intent(config)
    if backend in LOCAL_BACKENDS:
        ok, detail = _probe_ollama_model(config, model)
        return Check(CHECK_MODEL_SERVER, True, ok, detail)
    forbidden = _cloud_forbidden(config, model)
    if forbidden:
        return Check(CHECK_MODEL_SERVER, True, False, forbidden)
    url = config.qwen_vllm_url
    label = f"{backend} {url} model {model}"
    try:
        ok = probe_openai_server_idle(url, model=model)
    except Exception as exc:  # a broken probe is "not reachable"
        return Check(CHECK_MODEL_SERVER, True, False, f"{label}: {exc}")
    return Check(CHECK_MODEL_SERVER, True, ok, label if ok else f"{label}: no generation answered")


def check_engine_cli(config) -> Check:
    """The OCR executable each engine shells out to is installed and answers.

    A model server can be up while ``qwen-ocr`` is absent, and every page then
    fails. Qwen is checked by its executable alone (``BaseEngine.is_available``;
    ``QwenEngine.is_available`` also lists Ollama models, which the generation
    probe above answers better and which wrongly rejects a pinned cloud tag). A
    concrete non-qwen primary engine is checked by its own ``is_available``,
    which is what the pipeline calls. Under the HPC lane only Nougat is a CLI.
    """
    from socr.core.config import EngineType
    from socr.engines.base import BaseEngine

    missing: list[str] = []
    checked: list[str] = []
    if config.hpc.enabled:
        if config.hpc.use_nougat:
            engine = get_engine(EngineType.NOUGAT)
            checked.append(engine.cli_command)
            if not engine.is_available():
                missing.append(engine.cli_command)
    else:
        if EngineType.QWEN in config.enabled_engines:
            engine = get_engine(EngineType.QWEN)
            checked.append(engine.cli_command)
            if not BaseEngine.is_available(engine):
                missing.append(engine.cli_command)
        primary = config.primary_engine
        if primary not in (EngineType.AUTO, EngineType.QWEN) and has_cli_engine(primary):
            engine = get_engine(primary)
            checked.append(engine.name)
            if not engine.is_available():
                missing.append(engine.name)
    if not checked:
        return Check(CHECK_ENGINE_CLI, False, True, "no CLI engine in this run")
    if missing:
        return Check(
            CHECK_ENGINE_CLI, True, False, f"not installed or not answering: {', '.join(missing)}"
        )
    return Check(CHECK_ENGINE_CLI, True, True, ", ".join(checked))


def check_page_judge(config, pipeline) -> Check:
    """The judge model the run will use is reachable (``_resolve_judge_model``).

    ``_resolve_judge_model`` already skips a cloud candidate that strict-local /
    zero-cap policy forbids without probing it.
    """
    if config.hpc.enabled or config.judge_backend == "heuristic":
        return Check(CHECK_PAGE_JUDGE, False, True, "heuristic judge or HPC lane")
    model = pipeline._resolve_judge_model()
    if model:
        return Check(CHECK_PAGE_JUDGE, True, True, model)
    reason = getattr(pipeline, "_judge_unavailable_reason", "") or "no judge model answered"
    return Check(CHECK_PAGE_JUDGE, True, False, reason)


def check_tesseract(config) -> Check:
    """Classical-OCR witness: ``pytesseract`` imports and the binary answers.

    Needed whenever scanned-table verification can run, which is whenever a
    model output can carry a table. State names are those of
    ``classical_ocr_with_state``.
    """
    if config.hpc.enabled:  # not needed: do not probe a tool the lane never uses
        return Check(CHECK_TESSERACT, False, True, "HPC lane does not use it")
    state, detail = classical_ocr_backend_state()
    ok = state == WITNESS_READING
    return Check(
        CHECK_TESSERACT,
        True,
        ok,
        "pytesseract and tesseract answer" if ok else f"{state}: {detail}",
    )


def check_maths_readers(config, pipeline) -> Check:
    """The maths lanes can address the backend AND their models answer.

    Two questions, because "present" must mean the reader can run.
    ``_equation_lane_backend_addressable``: the lanes post to the Ollama generate
    API, or (#1063) to the run's OpenAI-compatible qwen server when its URL and
    model id are configured. Then the model a lane will call answers a real
    generation: each of ``math_model`` / ``clean_equation_model`` on the Ollama
    host, or on an OpenAI-compatible run the served model through the 1-token
    ``probe_openai_server_idle`` canary (a lane model naming anything the server
    does not serve is reported, not replaced).
    """
    from socr.core.providers import PROFILE_QWEN_LOCAL

    # "Enabled" is the pipeline's own run-level predicate, not a re-derivation:
    # --native-only switches the region lane and the corrupt-math recovery off.
    lane_models: dict[str, str] = {}
    if not config.hpc.enabled:
        if pipeline._equation_region_lane_enabled():
            lane_models["equation_region_lane"] = config.clean_equation_model
        if pipeline._corrupt_math_recovery_enabled():
            lane_models["recover_corrupt_math"] = config.math_model
        if pipeline._clean_equation_recovery_enabled():
            lane_models["recover_clean_equations"] = config.clean_equation_model
    if not lane_models:
        return Check(CHECK_MATHS_READERS, False, True, "no maths lane enabled")
    lanes = ", ".join(lane_models)
    if not pipeline._equation_lane_backend_addressable(PROFILE_QWEN_LOCAL):
        return Check(
            CHECK_MATHS_READERS,
            True,
            False,
            f"{lanes}: the configured backend is neither the Ollama API nor an "
            "OpenAI-compatible server with a URL and model id, so formula regions will "
            "be detected and never read (#1063)",
        )
    served = pipeline._maths_openai_served_model()
    if served is not None:
        # #1063: an OpenAI-compatible run reads through the qwen server, whose one
        # served model is the only model a lane can call. A lane model naming
        # anything else is refused (never silently replaced), so it is reported.
        problems = [
            refusal
            for model in dict.fromkeys(lane_models.values())
            if (refusal := pipeline._maths_model_refusal(model))
        ]
        # Policy on the model that will actually be called, independent of the lane tags.
        forbidden = _cloud_forbidden(config, served)
        if forbidden:
            problems.append(forbidden)
        url = config.qwen_vllm_url
        if not problems and not probe_openai_server_idle(url, model=served, require_reply=True):
            problems.append(f"vllm {url} model {served}: no generation answered")
        if problems:
            return Check(CHECK_MATHS_READERS, True, False, f"{lanes}: " + "; ".join(problems))
        return Check(CHECK_MATHS_READERS, True, True, f"{lanes}: vllm {url} model {served}")
    problems: list[str] = []
    for model in dict.fromkeys(lane_models.values()):
        ok, detail = _probe_ollama_model(config, model)
        if not ok:
            problems.append(detail)
    if problems:
        return Check(CHECK_MATHS_READERS, True, False, f"{lanes}: " + "; ".join(problems))
    return Check(CHECK_MATHS_READERS, True, True, lanes)


def check_table_judge(config) -> Check:
    """A table-judge reader rung is reachable (folds the P1 startup notices).

    Not needed under ``--strict-local``: both rungs are cloud services there, so
    the ladder is refused by policy and the operator already chose it. The old
    notice text is the detail, so that choice stays visible.
    """
    if config.hpc.enabled or not config.table_judge_ladder:
        return Check(CHECK_TABLE_JUDGE, False, True, "table-judge ladder off or HPC lane")
    # The same two policies ``_build_table_judge_rungs`` refuses the rungs under.
    if config.strict_local or zero_cap_pinned_forbids_cloud(config):
        policy = "--strict-local" if config.strict_local else "--max-cost-per-page 0"
        return Check(
            CHECK_TABLE_JUDGE,
            False,
            False,
            f"--table-judge-ladder and {policy} are both enabled; both rungs are "
            "cloud services, so every table page will be TABLE_UNVERIFIED "
            f"(drop {policy} or pass --no-table-judge-ladder)",
        )
    from socr.cli import any_table_judge_reader_reachable

    if any_table_judge_reader_reachable(config):
        return Check(CHECK_TABLE_JUDGE, True, True, "a judge rung answers")
    return Check(
        CHECK_TABLE_JUDGE,
        True,
        False,
        f"neither the ollama-cloud rung ({config.table_judge_rung1_model}) nor the CLI rung "
        f"({config.table_judge_rung2_binary}) answered; every table page will ship UNVERIFIED "
        "(pass --no-table-judge-ladder to ship tables unjudged)",
    )


def run_preflight(config, pipeline) -> list[Check]:
    """All checks for this run, in report order."""
    return [
        check_model_server(config),
        check_engine_cli(config),
        check_page_judge(config, pipeline),
        check_tesseract(config),
        check_maths_readers(config, pipeline),
        check_table_judge(config),
    ]


def missing_checks(checks: Iterable[Check], allowed: Iterable[str] = ()) -> list[Check]:
    """Needed-and-missing checks that the operator did NOT opt out of."""
    allow = set(allowed)
    return [c for c in checks if c.missing and c.name not in allow]


def ran_without(checks: Iterable[Check], allowed: Iterable[str] = ()) -> list[str]:
    """Names of needed-and-missing checks the operator allowed: what the run goes without."""
    allow = set(allowed)
    return [c.name for c in checks if c.missing and c.name in allow]


def status_of(check: Check, allowed: Iterable[str] = ()) -> str:
    if not check.needed:
        return STATUS_NOT_NEEDED
    if check.ok:
        return STATUS_PRESENT
    return STATUS_ALLOWED if check.name in set(allowed) else STATUS_MISSING


def format_report(checks: Iterable[Check], allowed: Iterable[str] = ()) -> list[str]:
    """One line per check: ``<name> <status> -- <detail>``."""
    allow = tuple(allowed)
    return [f"{c.name:<14} {status_of(c, allow):<18} {c.detail}".rstrip() for c in checks]
