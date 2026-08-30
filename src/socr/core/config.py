"""Configuration for socr v1.0."""

import dataclasses
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class EngineType(str, Enum):
    """Available OCR engines.

    Each value (except AUTO and HPC-only types) maps 1:1 to a sibling
    CLI tool at ../ocr/{name}-ocr-cli.
    """

    AUTO = "auto"  # Auto-detect best available engine
    NOUGAT = "nougat"
    DEEPSEEK = "deepseek"
    MISTRAL = "mistral"
    GEMINI = "gemini"
    MARKER = "marker"
    GLM = "glm"  # GLM-OCR via Ollama or transformers (local)
    QWEN = "qwen"  # Qwen-VL via ollama/vllm/cloud API (local-or-cloud, best open OCR)
    # HPC-only types (not backed by sibling CLIs — use vLLM HTTP API directly)
    DEEPSEEK_VLLM = "deepseek-vllm"
    VLLM = "vllm"


# Default engine priority: local free -> cheap cloud -> expensive cloud.
# Qwen-VL leads the local tier: on socOCRbench it scores ~0.47-0.58 vs GLM 0.37 and
# DeepSeek 0.09, so it is the local model worth trying first.
ENGINE_PRIORITY: dict[EngineType, int] = {
    EngineType.QWEN: 0,
    EngineType.GLM: 1,
    EngineType.NOUGAT: 2,
    EngineType.DEEPSEEK: 3,
    EngineType.MARKER: 4,
    EngineType.GEMINI: 5,
    EngineType.MISTRAL: 6,
    EngineType.DEEPSEEK_VLLM: 7,
    EngineType.VLLM: 8,
}

# Auto-selection order: try CLI engines until one is available.
# Local-first -> Ollama Cloud -> paid cloud edge case. Qwen leads because its
# default backend is qwen3.5:cloud (Ollama Cloud, no extra key): ~0.57 quality
# at ~49s/page on the owner's Mac and the only engine that cleared all three
# hard page types (math/table/equation). Gemini is the quality escalation when
# Qwen is unavailable. DeepSeek-OCR (~0.085 socOCRbench) and Mistral (worse AND
# ~5x pricier than Gemini) are deliberately OUT of the auto path; reach them
# only via an explicit --primary. Empirics in [[reference-sococrbench]].
AUTO_ENGINE_ORDER: list[EngineType] = [
    EngineType.QWEN,  # qwen3.5:cloud — practical cheap winner; native PDF-free per-page
    EngineType.GEMINI,  # Best quality, paid — escalation when Qwen is unavailable
    EngineType.MARKER,  # Local, layout-aware
    EngineType.GLM,  # Local, small model, fast
    EngineType.NOUGAT,  # Local, academic papers only
]

# GH-353 table judge ladder — CLI₁ (ollama-cloud glm-5.3-flash) per-call wall-clock
# budget. NOT a made-up round number: the GH-356 bake-off
# (docs/log/2026-08-30_gh356-bakeoff.md) measured glm-5.3-flash exceed a 300 s cap on
# a dense grid (NS p42) and still land the *correct* verdict on retry at a 590 s cap.
# The bake-off's explicit follow-up is "timeout >= 600 s or streaming with progress
# detection" — this is that floor. A rung timeout is escalate-as-substitute (¬S1),
# never a reject, per the GH-353 design (docs/log/2026-08-30_table-judge-ladder.md).
TABLE_JUDGE_TIMEOUT_SEC_DEFAULT: float = 600.0


@dataclass
class HPCConfig:
    """HPC-specific configuration (vLLM direct API, not CLI-based).

    HPC sequential mode uses vLLM HTTP API directly for per-page OCR,
    with Nougat for LaTeX and optional Gemini cloud fallback.
    """

    enabled: bool = False
    sequential: bool = False
    vllm_url: str = ""
    vllm_port: int = 8000
    ocr_model: str = "deepseek-ai/DeepSeek-OCR"
    vision_model: str = "Qwen/Qwen2-VL-7B-Instruct"
    use_nougat: bool = True
    manage_server: bool = True
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 8192
    server_startup_timeout: int = 180
    audit_enabled: bool = True
    cloud_fallback: bool = True
    use_llm_reconciler: bool = False
    reconciler_model: str = ""
    render_dpi: int = 200
    parallel_pages: int = 1

    def __post_init__(self) -> None:
        if not self.vllm_url:
            self.vllm_url = os.environ.get("VLLM_BASE_URL", f"http://localhost:{self.vllm_port}/v1")


# Fields of PipelineConfig that ``from_file`` restores with bespoke logic
# (enums, lists of enums, Path). Everything else is restored generically from
# ``dataclasses.fields``, so a newly added field persists by default instead of
# having to be opted in to a hand-maintained name list (#240). Keep this set in
# sync with the explicit blocks in ``PipelineConfig.from_file``.
_FROM_FILE_EXPLICIT_FIELDS: frozenset[str] = frozenset(
    {
        "primary_engine",
        "local_engine",
        "fallback_chain",
        "figures_engine",
        "enabled_engines",
        "output_dir",
        "hpc",
    }
)

# Keys accepted in YAML that are not PipelineConfig fields (legacy aliases).
_FROM_FILE_LEGACY_KEYS: frozenset[str] = frozenset({"fallback_engine"})

# Fields DELETED from PipelineConfig, mapped to why. A stale config file naming one
# would otherwise hit the generic "unrecognised key" error, which says a typo was
# made rather than that the setting is gone -- so each gets its own explanation.
_FROM_FILE_REMOVED_FIELDS: dict[str, str] = {
    "audit_enabled": (
        "the quality audit is not a removable stage and this field had no consumer "
        "(GH-139). In agentic mode -- the default -- the judge IS the routing "
        "algorithm: escalation happens because a judge rejected a rung, so with no "
        "gate there is no accept/escalate signal for the ladder to act on. The "
        "legacy branches that once honoured it were removed in #298. Remove the "
        "key; to reduce model spend use strict_local, max_cost_per_page or "
        "cost_budget"
    ),
}


@dataclass
class PipelineConfig:
    """Single configuration for the socr pipeline.

    Replaces the previous 8+ nested dataclasses (AgentConfig, EngineConfig,
    NougatConfig, DeepSeekConfig, MistralConfig, GeminiConfig, VLLMConfig,
    DeepSeekVLLMConfig, AuditConfig).
    """

    # --- Engine routing ---
    primary_engine: EngineType = EngineType.AUTO
    local_engine: EngineType = EngineType.AUTO  # Cheap local engine for easy pages
    fallback_chain: list[EngineType] = field(default_factory=lambda: [EngineType.GEMINI])
    figures_engine: EngineType = EngineType.GEMINI
    enabled_engines: list[EngineType] = field(default_factory=lambda: list(EngineType))

    # --- Native-first + tiered routing ---
    native_first: bool = True  # Use native text for born-digital prose
    tiered: bool = True  # Route easy pages to local engine, hard pages to primary
    # ``native_only`` is the positive counterpart of ``--no-native-first``.
    # When True, clean born-digital pages (including those with
    # ``needs_ocr_enhancement``) are NEVER sent to OCR — the native text layer is
    # trusted as-is. Genuine scans (``is_born_digital=False``) still route to OCR
    # as normal, and figure extraction can still run without forcing whole-page OCR.
    #
    # Policy interaction:
    #   --native-only   native_first=True, native_only=True   → trust all BD pages
    #   (default)       native_first=True, native_only=False  → trust clean BD; OCR enhance
    #   --no-native-first native_first=False, native_only=*   → OCR all pages
    #
    # Setting both ``--native-only`` and ``--no-native-first`` is incoherent; the
    # CLI warns and ``--no-native-first`` wins (everything goes to OCR).
    native_only: bool = False

    # --- Math recovery (font-corrupted equations) ---
    # When a born-digital page's prose is clean but its math is font-map corrupted
    # ('=' -> '¼', '(' -> 'ð'), keep the native prose and image-OCR only the
    # equation regions to LaTeX. Opt-in: needs a local vision model (Ollama).
    recover_corrupt_math: bool = False
    # qwen3.5:cloud (Ollama Cloud) is the practical winner: reliable on dense
    # equation regions where local qwen3-vl:30b-a3b-instruct times out, no extra
    # key.  Override with --math-model qwen3-vl:30b-a3b-instruct for offline runs.
    math_model: str = "qwen3.5:cloud"  # Ollama Cloud VLM used for equation -> LaTeX

    # --- GH-36a: Display-equation region detection (model-free) ---
    # Detect display-equation regions on born-digital pages using PyMuPDF
    # geometry (math-font spans + centring).  Saves crop PNGs and records
    # provenance; does NOT call a model or splice LaTeX (that is GH-36b).
    # Default-off: the throughput cost must be measured (see GH-36a AC5)
    # before making this default-on, and the GH-36b engine/validation gate
    # must land before any replacement/splicing occurs.
    detect_equations: bool = False

    # --- GH-36b: Clean-equation → LaTeX via local VLM + 1A structural gate ---
    # Reads each detected equation crop (from detect_equations above) with the
    # local qwen3-vl:30b-a3b-instruct VLM, validates the output structurally
    # with pylatexenc (1A gate, offline, deterministic), and attaches
    # 1A-validated LaTeX ADJACENTLY to the inlined crop PNG (1C non-destructive
    # sidecar policy).  Bad/hallucinated LaTeX never replaces native text or the
    # crop — the crop is always retained as the visual ground truth.
    #
    # Default-OFF: the default-on decision awaits real-corpus throughput
    # measurement (per consilium 20260615T210537Z-6621 and GH-36b AC).
    # Requires detect_equations=True to have any effect (no detected regions,
    # no engine calls).
    recover_clean_equations: bool = False
    # Model for the clean-equation crop→LaTeX engine call (GH-36b only).
    # MUST default to the validated local instruct VLM — NEVER to a cloud
    # model — per consilium 20260615T210537Z-6621 (local-first mandate).
    # Kept separate from ``math_model`` (corrupt-font path) which defaults to
    # qwen3.5:cloud for historical reasons and operator convenience there.
    # Override with --clean-equation-model for a different local model or an
    # explicit cloud opt-in.  Never use ":8b" (wrong model tier) or ":30b"
    # (thinking/non-instruct, runs away on dense regions).
    clean_equation_model: str = "qwen3-vl:30b-a3b-instruct"

    # --- Processing ---
    # ``Path("output")`` is the LEGACY SENTINEL meaning "unset". The canon
    # default output root is ``<input-parent>/ocr/`` (resolved per-input via the
    # contract's ``resolve_output_root`` in UnifiedPipeline._resolve_output_root);
    # this default never becomes a literal ``output/`` directory unless a user
    # explicitly sets it. A user-set value (incl. -o) is honored verbatim.
    output_dir: Path = field(default_factory=lambda: Path("output"))
    timeout: int = 1800  # Single timeout for all engine subprocesses
    chunk_threshold: int = 30  # Chunk PDFs longer than this many pages
    chunk_size: int = 20  # Pages per chunk
    render_dpi: int = 300  # DPI for page rendering; 300 resolves small table digits/parens
    #   (200 misreads e.g. "(0.001)" as "(0.007)"); override per-run with --dpi.
    workers: int = 1  # Concurrent workers (passed to CLI --workers flag)
    save_figures: bool = False
    # ``describe_figures`` is a separate opt-in from ``save_figures``.
    # ``--save-figures`` extracts PNGs and appends image-ref markdown only;
    # ``--describe-figures`` additionally calls the VLM caption engine.
    # Keeping them apart lets the operator archive clean deterministic PNGs
    # without coupling the run to non-authoritative model prose.
    describe_figures: bool = False
    figures_max_total: int = 25
    figures_max_per_page: int = 3

    # --- Audit ---
    # GH-139: `audit_enabled` was DELETED here. #298 removed every gate that read
    # it, leaving a field that changed nothing but still hashed into the run
    # fingerprint. `--no-audit` (its CLI setter) is now rejected in cli.py, and a
    # config file naming it gets an explanation via ``_FROM_FILE_REMOVED_FIELDS``.
    # ``HPCConfig.audit_enabled`` is a DIFFERENT field and is still live.
    audit_min_words: int = 50
    # VLM judge on HARD pages (tables/equations): a vision model checks the OCR
    # against the page image to catch SEMANTIC corruption the heuristics can't —
    # flipped signs, wrong digits (0.001->0.007), swapped columns. Rejected pages
    # re-route through repair. No-ops if no vision judge model is available.
    judge_hard_pages: bool = True
    # Dual-pass table extraction: on table pages, crop each precisely-located
    # table (ruled or booktabs), re-read the crop with the judge VLM, and
    # reconcile against the whole-page OCR. Crop-vs-page disagreement is a
    # corruption flag; the crop reading is authoritative and patched in. No-ops
    # if no vision model is available. Reuses the judge model ladder.
    # GH-96: re-read a table page with a cloud engine when its emitted table
    # disagrees with the page's own native text layer, and keep the result only if
    # hierarchy-aware exactness measurably improves. Automatic wherever a cloud rung
    # is already in the ladder — `--strict-local` and `--max-cost-per-page` continue
    # to suppress it by tier and cost, because the provider is chosen from the
    # already-filtered ladder rather than named.
    escalate_ambiguous_tables: bool = True

    # Wall-clock budget for one escalation call. A cloud OCR CLI was observed
    # wedged mid-request for 97 minutes with no timeout of its own; escalation runs
    # inline in the page-major loop, so an unbounded call stalls the whole document.
    escalation_timeout_sec: float = 120.0

    dual_pass_tables: bool = True
    # Auto-patch the crop reading into the page on disagreement. Default OFF
    # (flag-only): the crop reader's numeric fidelity is unproven, and a silent
    # wrong patch to a research number is worse than a missed correction. Opt in
    # with --auto-patch-tables once the crop reader is trusted on held-out data.
    auto_patch_tables: bool = False

    # --- Agentic cost-aware routing ---
    # Agentic is the ONLY mode: per-page cheapest-first routing with judge-gated
    # escalation. R174b deleted the deterministic backbone -> audit -> repair
    # pipeline and the --legacy-routing flag that reached it; this field is kept
    # only because the run fingerprint and a small number of call sites read it.
    agentic: bool = True  # per-page: try cheapest provider, judge escalates
    strict_local: bool = False  # if True, agentic ladder uses only local/free rungs
    judge_backend: str = "auto"  # "auto" | "vlm" | "heuristic"
    judge_model: str = ""  # VLM model for the judge (e.g. qwen2-vl:7b); "" = default
    max_cost_per_page: float = 0.0  # 0 = no per-page price cap
    cost_budget: float = 0.0  # 0 = unlimited total budget per document
    write_manifest: bool = False  # write reproducibility manifest + blob cache

    # --- GH-353: table judge ladder ---
    # Two-rung table-page acceptance gate: CLI1 (ollama-cloud vision judge) -> CLI2
    # (gemini CLI) -> terminal disposition (REJECTED/UNVERIFIED). Design:
    # docs/log/2026-08-30_table-judge-ladder.md; CLI1 seat decided by the GH-356
    # bake-off (docs/log/2026-08-30_gh356-bakeoff.md). Default OFF: golden/
    # byte-identity tests must stay byte-identical with the flag off, and the gate
    # itself (TICKET-B1) has not landed yet.
    table_judge_ladder: bool = False
    # CLI1 rung: ollama-cloud vision judge model. glm-5.3-flash:cloud won the
    # GH-356 bake-off outright (every verdict + code correct across all three
    # rounds, including the GH-273 binding-shift case two other candidates missed).
    table_judge_rung1_model: str = "glm-5.3-flash:cloud"
    # CLI1 rung: ollama host/endpoint the judge POSTs `/api/chat` to. None resolves
    # like ``ollama_host`` above (OLLAMA_HOST env var, then the localhost default)
    # rather than hardcoding localhost, which would misfire on any non-default
    # ollama deployment.
    table_judge_rung1_host: str | None = None
    # CLI2 rung: subprocess binary name/path for the gemini-family CLI invoker
    # (A3). Default is ``agy`` (Antigravity CLI), not ``gemini`` — the pre-merge
    # B1 live smoke (2026-08-30) found the bare `gemini` binary can no longer
    # authenticate headlessly on this machine ("Gemini Code Assist for
    # individuals" free tier retired; Google's own IneligibleTierError message
    # says to migrate to Antigravity). `agy` reaches the same model family
    # through a live, working headless surface (smoke: schema-perfect,
    # unfenced JSON, all six decoy defects caught — transcript referenced in
    # docs/log/2026-08-30_gh353-ticket-a3.md). A missing binary is still a
    # normal ¬S1 substitution, not a config error, whichever binary is
    # configured.
    table_judge_rung2_binary: str = "agy"
    # Per-call wall-clock budget for EITHER rung (see TABLE_JUDGE_TIMEOUT_SEC_DEFAULT
    # for the bake-off measurement behind the default). A timeout is ¬S1
    # (escalate-as-substitute to the next rung / terminal), never treated as a
    # judge FAIL.
    #
    # Interaction with --strict-local: strict_local forbids cloud egress, and BOTH
    # ladder rungs are cloud (ollama-cloud, gemini CLI). So
    # ``strict_local and table_judge_ladder`` makes every rung unavailable before
    # the first call — the gate (TICKET-B1) must fail-open each table page to
    # UNVERIFIED (never a silent PASS, never an exception) rather than call out.
    # This field only documents the interaction; TICKET-B1 implements the gate.
    table_judge_timeout_sec: float = TABLE_JUDGE_TIMEOUT_SEC_DEFAULT

    # --- Batch flags ---
    reprocess: bool = False
    dry_run: bool = False
    quiet: bool = False
    verbose: bool = False

    # --- HPC ---
    hpc: HPCConfig = field(default_factory=HPCConfig)

    # --- Engine-specific overrides (flat) ---
    # These map 1:1 to CLI flags on the sibling *-ocr-cli tools.
    deepseek_backend: str = "ollama"  # "ollama" or "vllm"
    deepseek_task: str = "convert"  # "convert", "ocr", "layout", "extract", "parse"
    deepseek_vllm_url: str = "http://localhost:8000/v1"
    glm_backend: str = "ollama"  # "ollama", "transformers", or "vllm"
    glm_task: str = "text"  # "text", "formula", "table", "figure"
    qwen_backend: str = "auto"  # "auto", "ollama", "vllm", or "api"
    # HF model id served by vLLM/SGLang for the agentic VLM crop path when
    # qwen_backend == "vllm" (HPC: Ollama is forbidden on server GPUs). The HF
    # equivalent of the local ollama tag qwen3-vl:30b-a3b-instruct.
    qwen_vllm_model: str = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    # OpenAI-compatible base URL of the vLLM server for the agentic VLM path.
    qwen_vllm_url: str = "http://localhost:8000/v1"
    # GH-222: base URL of the Ollama daemon the cascade-halt liveness probe should
    # ask about. ``None`` means "resolve it" — the OLLAMA_HOST environment
    # variable, then the localhost default — rather than "assume localhost", which
    # made the probe indict a healthy machine on every non-Ollama deployment.
    ollama_host: str | None = None
    # Default sentinel: empty string means "not user-pinned; let the engine resolver pick
    # the right model for the resolved backend." When qwen_model_pinned is True the value
    # is an explicit user override and must reach qwen-ocr unchanged.
    # Local/auto-local resolution always uses the validated instruct MoE
    # (qwen3-vl:30b-a3b-instruct); cloud or vllm/api paths keep their own defaults.
    qwen_model: str = ""
    # True when the user passed --qwen-model explicitly. The resolver honours this flag
    # to avoid rewriting a deliberate model pin (e.g. qwen3.5:cloud for cloud-only runs).
    qwen_model_pinned: bool = False
    nougat_model: str = "0.1.0-base"
    marker_device: str = "auto"
    gemini_model: str = "gemini-3-flash-preview"
    gemini_task: str = "convert"  # "convert", "extract", "table", "describe_figure"
    mistral_model: str = "mistral-ocr-latest"

    def __post_init__(self) -> None:
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        # One env var drives both vLLM paths: when VLLM_BASE_URL is set (the gate
        # the qwen-ocr whole-page CLI uses) and the crop URL is still the default,
        # adopt it so the crop reader and whole-page OCR hit the same server.
        _vllm_env = os.environ.get("VLLM_BASE_URL")
        if _vllm_env and self.qwen_vllm_url == "http://localhost:8000/v1":
            self.qwen_vllm_url = _vllm_env

    def get_engines_by_priority(self) -> list[EngineType]:
        """Get enabled engines sorted by priority."""
        return sorted(self.enabled_engines, key=lambda e: ENGINE_PRIORITY.get(e, 99))

    @classmethod
    def from_file(cls, path: Path | str) -> "PipelineConfig":
        """Load configuration from a YAML file."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        config = cls()

        # Engine routing
        if "primary_engine" in data:
            config.primary_engine = EngineType(data["primary_engine"])
        if "local_engine" in data:
            config.local_engine = EngineType(data["local_engine"])
        if "fallback_chain" in data:
            config.fallback_chain = [EngineType(e) for e in data["fallback_chain"]]
        elif "fallback_engine" in data:
            # Legacy: single engine -> wrap in a list
            config.fallback_chain = [EngineType(data["fallback_engine"])]
        if "figures_engine" in data:
            config.figures_engine = EngineType(data["figures_engine"])
        if "enabled_engines" in data:
            config.enabled_engines = [EngineType(e) for e in data["enabled_engines"]]

        # Everything else: restored generically from the dataclass definition, so
        # a field added to PipelineConfig persists through a config file without
        # anyone remembering to add its name here (#240).
        for f in dataclasses.fields(cls):
            if f.name not in _FROM_FILE_EXPLICIT_FIELDS and f.name in data:
                setattr(config, f.name, data[f.name])

        if "output_dir" in data:
            config.output_dir = Path(data["output_dir"])

        # HPC config -- only allow known fields to prevent injection
        unknown_hpc: list[str] = []
        if "hpc" in data and isinstance(data["hpc"], dict):
            allowed = {f.name for f in dataclasses.fields(HPCConfig)}
            hpc_data = {k: v for k, v in data["hpc"].items() if k in allowed}
            config.hpc = HPCConfig(**hpc_data)
            unknown_hpc = sorted(f"hpc.{k}" for k in data["hpc"] if k not in allowed)
        elif "hpc" in data:
            # A non-mapping ``hpc:`` silently yielded the default HPCConfig, which is
            # the same silent-drop failure this rule exists to prevent.
            unknown_hpc = ["hpc (must be a mapping)"]

        # An unrecognised key is a typo or a stale setting. Dropping it silently is
        # exactly how a spend cap or a mode switch fails to take effect with no
        # signal at any level, so it fails the load instead (#240).
        #
        # Keys are stringified before sorting: YAML permits non-string keys, and a
        # bare ``1:`` would otherwise crash on the sort/join while building the very
        # message meant to explain the problem.
        # A field socr has DELETED is reported by name, not as a typo (see
        # ``_FROM_FILE_REMOVED_FIELDS``). Same doctrine as the unknown-key error
        # below: never let a setting a user believes is in force vanish silently.
        removed = sorted(k for k in data if k in _FROM_FILE_REMOVED_FIELDS)
        if removed:
            detail = "; ".join(f"'{k}': {_FROM_FILE_REMOVED_FIELDS[k]}" for k in removed)
            raise ValueError(
                f"Removed setting(s) in config file {path}: {', '.join(removed)}. {detail}."
            )

        known = {f.name for f in dataclasses.fields(cls)} | _FROM_FILE_LEGACY_KEYS
        unknown = sorted(str(k) for k in data if k not in known) + unknown_hpc
        if unknown:
            raise ValueError(
                f"Unrecognised key(s) in config file {path}: {', '.join(unknown)}. "
                "Valid top-level keys are the field names of PipelineConfig "
                "(plus the legacy alias 'fallback_engine'); keys under 'hpc' are the "
                "field names of HPCConfig — both defined in socr/core/config.py. "
                "A key socr does not recognise is rejected rather than ignored, so a "
                "typo cannot silently drop a setting."
            )

        return config

    @classmethod
    def load(
        cls, profile: str | None = None, config_path: Path | str | None = None
    ) -> "PipelineConfig":
        """Load configuration from profile or custom path.

        Search order:
            1. config_path if provided
            2. ~/.config/socr/{profile}.yaml
            3. ~/.config/socr/config.yaml
            4. Default PipelineConfig()
        """
        config_dir = Path.home() / ".config" / "socr"

        if config_path:
            path = Path(config_path)
            if path.exists():
                return cls.from_file(path)
            raise FileNotFoundError(f"Config file not found: {path}")

        if profile:
            profile_path = (config_dir / f"{profile}.yaml").resolve()
            if not profile_path.is_relative_to(config_dir.resolve()):
                raise ValueError(f"Invalid profile name: {profile!r}")
            if profile_path.exists():
                return cls.from_file(profile_path)
            raise FileNotFoundError(f"Profile not found: {profile_path}")

        default_path = config_dir / "config.yaml"
        if default_path.exists():
            return cls.from_file(default_path)

        return cls()


# Backward-compat property: ``config.fallback_engine`` reads/writes the first
# element of ``fallback_chain``.  Defined outside the class body so that
# @dataclass doesn't treat it as a field.


def _fallback_engine_get(self: PipelineConfig) -> EngineType | None:
    return self.fallback_chain[0] if self.fallback_chain else None


def _fallback_engine_set(self: PipelineConfig, value: EngineType) -> None:
    self.fallback_chain = [value]


PipelineConfig.fallback_engine = property(_fallback_engine_get, _fallback_engine_set)  # type: ignore[attr-defined]
