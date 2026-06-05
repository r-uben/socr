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
# Prefers cloud (best quality) > local.
AUTO_ENGINE_ORDER: list[EngineType] = [
    EngineType.GEMINI,      # Best quality/price, native PDF via Files API
    EngineType.MISTRAL,     # Cloud fallback, structured output
    EngineType.DEEPSEEK,    # Local, needs Ollama + model pulled
    EngineType.GLM,         # Local, small model, fast
    EngineType.NOUGAT,      # Local, academic papers only
    EngineType.MARKER,      # Local, layout-aware
]


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

    # --- Processing ---
    output_dir: Path = field(default_factory=lambda: Path("output"))
    timeout: int = 1800  # Single timeout for all engine subprocesses
    max_retries: int = 2
    truncation_retries: int = 1  # Retry same engine on truncation before fallback
    chunk_threshold: int = 30  # Chunk PDFs longer than this many pages
    chunk_size: int = 20  # Pages per chunk
    render_dpi: int = 300  # DPI for page rendering; 300 resolves small table digits/parens
    #   (200 misreads e.g. "(0.001)" as "(0.007)"); override per-run with --dpi.
    workers: int = 1  # Concurrent workers (passed to CLI --workers flag)
    save_figures: bool = False
    figures_max_total: int = 25
    figures_max_per_page: int = 3

    # --- Audit ---
    audit_enabled: bool = True
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
    dual_pass_tables: bool = True
    # Auto-patch the crop reading into the page on disagreement. Default OFF
    # (flag-only): the crop reader's numeric fidelity is unproven, and a silent
    # wrong patch to a research number is worse than a missed correction. Opt in
    # with --auto-patch-tables once the crop reader is trusted on held-out data.
    auto_patch_tables: bool = False

    # --- Agentic cost-aware routing (all default-off = unchanged behavior) ---
    agentic: bool = False  # per-page: try cheapest provider, judge escalates
    judge_backend: str = "auto"  # "auto" | "vlm" | "heuristic"
    judge_model: str = ""  # VLM model for the judge (e.g. qwen2-vl:7b); "" = default
    max_cost_per_page: float = 0.0  # 0 = no per-page price cap
    cost_budget: float = 0.0  # 0 = unlimited total budget per document
    write_manifest: bool = False  # write reproducibility manifest + blob cache

    # --- Multi-engine ---
    multi_engine: list[EngineType] = field(default_factory=list)  # empty = single engine mode

    # --- Consensus ---
    consensus_enabled: bool = False
    consensus_use_llm: bool = False
    consensus_ollama_model: str = ""

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
    # Default to qwen3.5:cloud (Ollama Cloud, vision): ~0.57 quality at ~49s/page on the
    # owner's Mac — faster AND better than local qwen3-vl:8b, no extra key. Trade-off: it
    # is ONLINE and uses the Ollama plan. For offline/private or true-free --agentic, pass
    # --qwen-model qwen3-vl:8b. Empirics in [[reference-sococrbench]].
    qwen_model: str = "qwen3.5:cloud"
    nougat_model: str = "0.1.0-base"
    marker_device: str = "auto"
    gemini_model: str = "gemini-3-flash-preview"
    gemini_task: str = "convert"  # "convert", "extract", "table", "describe_figure"
    mistral_model: str = "mistral-ocr-latest"

    def __post_init__(self) -> None:
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

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
        if "multi_engine" in data:
            config.multi_engine = [EngineType(e) for e in data["multi_engine"]]

        # Scalar fields
        scalar_fields = [
            "native_first", "tiered",
            "timeout", "max_retries", "truncation_retries",
            "chunk_threshold", "chunk_size", "render_dpi", "workers",
            "save_figures", "figures_max_total",
            "figures_max_per_page", "audit_enabled", "audit_min_words",
            "consensus_enabled", "consensus_use_llm", "consensus_ollama_model",
            "reprocess", "dry_run", "quiet", "verbose",
            "deepseek_backend", "deepseek_task", "deepseek_vllm_url",
            "glm_backend", "glm_task", "qwen_backend", "qwen_model",
            "nougat_model", "marker_device",
            "gemini_model", "gemini_task", "mistral_model",
        ]
        for key in scalar_fields:
            if key in data:
                setattr(config, key, data[key])

        if "output_dir" in data:
            config.output_dir = Path(data["output_dir"])

        # HPC config -- only allow known fields to prevent injection
        if "hpc" in data and isinstance(data["hpc"], dict):
            allowed = {f.name for f in dataclasses.fields(HPCConfig)}
            hpc_data = {k: v for k, v in data["hpc"].items() if k in allowed}
            config.hpc = HPCConfig(**hpc_data)

        return config

    @classmethod
    def load(cls, profile: str | None = None, config_path: Path | str | None = None) -> "PipelineConfig":
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
