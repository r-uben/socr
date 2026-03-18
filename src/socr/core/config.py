"""Configuration for socr v1.0."""

import dataclasses
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class EngineType(str, Enum):
    """Available OCR engines."""

    NOUGAT = "nougat"
    DEEPSEEK = "deepseek"
    MISTRAL = "mistral"
    GEMINI = "gemini"
    MARKER = "marker"
    GLM = "glm"  # GLM-OCR via Ollama or transformers (local)
    DEEPSEEK_VLLM = "deepseek-vllm"  # DeepSeek via vLLM HTTP API (HPC mode)
    VLLM = "vllm"  # Generic vLLM vision model (figures only, HPC mode)


# Default engine priority: local free -> cheap cloud -> expensive cloud
ENGINE_PRIORITY: dict[EngineType, int] = {
    EngineType.GLM: 0,
    EngineType.NOUGAT: 1,
    EngineType.DEEPSEEK: 2,
    EngineType.MARKER: 3,
    EngineType.GEMINI: 4,
    EngineType.MISTRAL: 5,
    EngineType.DEEPSEEK_VLLM: 6,
    EngineType.VLLM: 7,
}


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
    primary_engine: EngineType = EngineType.DEEPSEEK
    fallback_chain: list[EngineType] = field(default_factory=lambda: [EngineType.GEMINI])
    figures_engine: EngineType = EngineType.GEMINI
    enabled_engines: list[EngineType] = field(default_factory=lambda: list(EngineType))

    # --- Processing ---
    output_dir: Path = field(default_factory=lambda: Path("output"))
    timeout: int = 1800  # Single timeout for all engine subprocesses
    max_retries: int = 2
    save_figures: bool = False
    figures_max_total: int = 25
    figures_max_per_page: int = 3

    # --- Audit ---
    audit_enabled: bool = True
    audit_min_words: int = 50

    # --- Batch flags ---
    reprocess: bool = False
    dry_run: bool = False
    quiet: bool = False
    verbose: bool = False

    # --- HPC ---
    hpc: HPCConfig = field(default_factory=HPCConfig)

    # --- Engine-specific overrides (flat) ---
    deepseek_backend: str = "ollama"  # "ollama" or "vllm"
    deepseek_vllm_url: str = "http://localhost:8000/v1"
    glm_backend: str = "ollama"  # "ollama", "transformers", or "vllm"
    nougat_model: str = "0.1.0-small"
    marker_device: str = "auto"
    gemini_model: str = "gemini-3-flash-preview"
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
        if "fallback_chain" in data:
            config.fallback_chain = [EngineType(e) for e in data["fallback_chain"]]
        elif "fallback_engine" in data:
            # Legacy: single engine -> wrap in a list
            config.fallback_chain = [EngineType(data["fallback_engine"])]
        if "figures_engine" in data:
            config.figures_engine = EngineType(data["figures_engine"])
        if "enabled_engines" in data:
            config.enabled_engines = [EngineType(e) for e in data["enabled_engines"]]

        # Scalar fields
        scalar_fields = [
            "timeout", "max_retries", "save_figures", "figures_max_total",
            "figures_max_per_page", "audit_enabled", "audit_min_words",
            "reprocess", "dry_run", "quiet", "verbose",
            "deepseek_backend", "deepseek_vllm_url", "glm_backend", "nougat_model",
            "marker_device", "gemini_model", "mistral_model",
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
