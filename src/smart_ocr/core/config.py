"""Configuration for smart-ocr."""

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
    VLLM = "vllm"
    DEEPSEEK_VLLM = "deepseek-vllm"  # DeepSeek-OCR via vLLM (HPC mode)


class AuditModel(str, Enum):
    """Available audit models via Ollama."""

    LLAMA = "llama3.2"
    QWEN = "qwen2.5"
    DEEPSEEK = "deepseek-r1:32b"  # Reasoning model, best for quality analysis


@dataclass
class EngineConfig:
    """Configuration for a single engine."""

    enabled: bool = True
    priority: int = 0  # Lower is higher priority
    max_retries: int = 2
    timeout: int = 300  # seconds (5 min default - increase for large docs)
    figure_timeout: int = 180  # seconds per figure description (3 min default)


@dataclass
class NougatConfig(EngineConfig):
    """Nougat-specific configuration."""

    model: str = "0.1.0-small"
    batch_size: int = 1
    no_skipping: bool = False
    recompute: bool = False


@dataclass
class DeepSeekConfig(EngineConfig):
    """DeepSeek-specific configuration."""

    model: str = "deepseek-ocr:latest"  # Via Ollama
    ollama_host: str = "http://localhost:11434"


@dataclass
class MistralConfig(EngineConfig):
    """Mistral-specific configuration."""

    api_key: str = ""
    model: str = "pixtral-large-latest"  # Vision-capable model

    def __post_init__(self) -> None:
        if not self.api_key:
            self.api_key = os.environ.get("MISTRAL_API_KEY", "")


@dataclass
class GeminiConfig(EngineConfig):
    """Gemini-specific configuration."""

    api_key: str = ""
    model: str = "gemini-3-flash-preview"

    def __post_init__(self) -> None:
        if not self.api_key:
            self.api_key = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")


@dataclass
class VLLMConfig(EngineConfig):
    """vLLM-specific configuration for vision models via OpenAI-compatible API."""

    base_url: str = ""
    api_key: str = ""  # Optional, some vLLM deployments don't require auth
    model: str = "Qwen/Qwen2-VL-7B-Instruct"  # Common vision models: Qwen2-VL, InternVL2
    max_tokens: int = 1024
    temperature: float = 0.1

    def __post_init__(self) -> None:
        if not self.base_url:
            self.base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
        if not self.api_key:
            self.api_key = os.environ.get("VLLM_API_KEY", "EMPTY")  # vLLM default


@dataclass
class DeepSeekVLLMConfig(EngineConfig):
    """DeepSeek-OCR via vLLM configuration (HPC mode).

    Uses vLLM's OpenAI-compatible API to run DeepSeek-OCR model for text extraction.
    Unlike the standard VLLMConfig (figures-only), this engine performs OCR.
    """

    base_url: str = ""
    api_key: str = ""
    model: str = "deepseek-ai/DeepSeek-OCR"  # DeepSeek-OCR model
    max_tokens: int = 4096  # OCR needs more tokens than figure description
    temperature: float = 0.0  # Deterministic for OCR

    def __post_init__(self) -> None:
        if not self.base_url:
            self.base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
        if not self.api_key:
            self.api_key = os.environ.get("VLLM_API_KEY", "EMPTY")


@dataclass
class HPCConfig:
    """Configuration for HPC multi-agent mode.

    HPC mode runs multiple OCR engines in parallel locally via vLLM,
    then reconciles outputs intelligently. No cloud fallback in HPC mode.
    """

    enabled: bool = False
    vllm_url: str = ""
    ocr_model: str = "deepseek-ai/DeepSeek-OCR"
    vision_model: str = "OpenGVLab/InternVL2-26B"
    reconciler_model: str = ""  # Same as ocr_model if empty
    use_nougat: bool = True  # Include Nougat for LaTeX equations
    use_llm_reconciler: bool = False  # Use LLM for conflict resolution

    def __post_init__(self) -> None:
        if not self.vllm_url:
            self.vllm_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
        if not self.reconciler_model:
            self.reconciler_model = self.ocr_model


@dataclass
class AuditConfig:
    """Configuration for quality audit."""

    enabled: bool = True
    model: AuditModel = AuditModel.DEEPSEEK  # deepseek-r1:32b - reasoning model
    ollama_host: str = "http://localhost:11434"
    min_word_count: int = 50
    min_confidence: float = 0.7
    check_encoding: bool = True
    check_structure: bool = True
    cross_check_enabled: bool = False  # Use alternate local engine on flagged pages
    cross_check_pages: int = 2  # How many flagged pages to cross-check


@dataclass
class AgentConfig:
    """Main configuration for smart-ocr."""

    # Engine configurations
    nougat: NougatConfig = field(default_factory=NougatConfig)
    deepseek: DeepSeekConfig = field(default_factory=DeepSeekConfig)
    mistral: MistralConfig = field(default_factory=MistralConfig)
    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    vllm: VLLMConfig = field(default_factory=VLLMConfig)
    deepseek_vllm: DeepSeekVLLMConfig = field(default_factory=DeepSeekVLLMConfig)

    # Audit configuration
    audit: AuditConfig = field(default_factory=AuditConfig)

    # HPC mode configuration
    hpc: HPCConfig = field(default_factory=HPCConfig)

    # Processing options
    output_dir: Path = field(default_factory=lambda: Path("output"))
    output_format: str = "markdown"  # markdown, json, txt
    render_dpi: int | str = "auto"  # "auto", or explicit: 150, 200, 300
    include_figures: bool = True
    save_figures: bool = False  # Save extracted figure images to disk
    figures_max_total: int = 25  # Hard cap across document
    figures_max_per_page: int = 3  # Hard cap per page
    figures_context_max_chars: int = 1200  # Truncate page text context for vision calls
    figure_timeout: int = 180  # seconds per figure description (3 min default)
    parallel_pages: int = 4  # Number of pages to process in parallel (set to 1 for sequential)
    parallel_figures: int = 2  # Number of figures to describe in parallel
    verbose: bool = False

    # Routing overrides (set via CLI/config to force choices)
    use_primary_override: bool = False
    use_fallback_override: bool = False
    use_figures_engine_override: bool = False

    # Routing preferences
    primary_engine: EngineType = EngineType.NOUGAT  # For academic
    fallback_engine: EngineType = EngineType.MISTRAL  # Cloud fallback
    figures_engine: EngineType = EngineType.GEMINI  # For figure description

    def __post_init__(self) -> None:
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

        # Set priorities (lower = higher priority)
        # Local free engines first, then cheap cloud, then expensive cloud
        self.nougat.priority = 0  # Free, local, academic-focused
        self.deepseek.priority = 1  # Free, local, general
        self.deepseek_vllm.priority = 1  # Free, local, HPC mode OCR
        self.vllm.priority = 2  # Free, local, vision-capable (for figures)
        self.gemini.priority = 3  # Cheap cloud ($0.0002/page)
        self.mistral.priority = 4  # Expensive cloud ($0.001/page)

    def get_enabled_engines(self) -> list[EngineType]:
        """Get list of enabled engines sorted by priority."""
        engines = []
        for engine_type, config in [
            (EngineType.NOUGAT, self.nougat),
            (EngineType.DEEPSEEK, self.deepseek),
            (EngineType.DEEPSEEK_VLLM, self.deepseek_vllm),
            (EngineType.VLLM, self.vllm),
            (EngineType.MISTRAL, self.mistral),
            (EngineType.GEMINI, self.gemini),
        ]:
            if config.enabled:
                engines.append((engine_type, config.priority))

        return [e[0] for e in sorted(engines, key=lambda x: x[1])]

    def get_engine_config(self, engine: EngineType) -> EngineConfig:
        """Get configuration for a specific engine."""
        return {
            EngineType.NOUGAT: self.nougat,
            EngineType.DEEPSEEK: self.deepseek,
            EngineType.DEEPSEEK_VLLM: self.deepseek_vllm,
            EngineType.VLLM: self.vllm,
            EngineType.MISTRAL: self.mistral,
            EngineType.GEMINI: self.gemini,
        }[engine]

    @classmethod
    def from_file(cls, path: Path | str) -> "AgentConfig":
        """Load configuration from a YAML file."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)

        # Build config from dict
        config = cls()

        if "nougat" in data:
            config.nougat = NougatConfig(**data["nougat"])
        if "deepseek" in data:
            config.deepseek = DeepSeekConfig(**data["deepseek"])
        if "deepseek_vllm" in data:
            config.deepseek_vllm = DeepSeekVLLMConfig(**data["deepseek_vllm"])
        if "mistral" in data:
            config.mistral = MistralConfig(**data["mistral"])
        if "gemini" in data:
            config.gemini = GeminiConfig(**data["gemini"])
        if "vllm" in data:
            config.vllm = VLLMConfig(**data["vllm"])
        if "audit" in data:
            config.audit = AuditConfig(**data["audit"])
        if "hpc" in data:
            config.hpc = HPCConfig(**data["hpc"])

        for key in [
            "output_dir",
            "output_format",
            "include_figures",
            "save_figures",
            "figures_max_total",
            "figures_max_per_page",
            "figures_context_max_chars",
            "figure_timeout",
            "parallel_pages",
            "parallel_figures",
            "verbose",
            "use_primary_override",
            "use_fallback_override",
            "use_figures_engine_override",
            "render_dpi",
        ]:
            if key in data:
                setattr(config, key, data[key])

        # Engine routing overrides (allow string names in config files)
        if "primary_engine" in data:
            config.primary_engine = EngineType(data["primary_engine"])
            # Only force override if user didn't explicitly set a flag.
            if "use_primary_override" not in data:
                config.use_primary_override = True
        if "fallback_engine" in data:
            config.fallback_engine = EngineType(data["fallback_engine"])
            if "use_fallback_override" not in data:
                config.use_fallback_override = True
        if "figures_engine" in data:
            config.figures_engine = EngineType(data["figures_engine"])
            if "use_figures_engine_override" not in data:
                config.use_figures_engine_override = True

        return config

    @classmethod
    def load_config(cls, profile: str | None = None, config_path: Path | str | None = None) -> "AgentConfig":
        """Load configuration from profile or custom path.

        Args:
            profile: Profile name to load from ~/.config/smart-ocr/{profile}.yaml
            config_path: Direct path to a config file (takes precedence over profile)

        Returns:
            AgentConfig instance

        Config search order:
            1. config_path if provided
            2. ~/.config/smart-ocr/{profile}.yaml if profile provided
            3. ~/.config/smart-ocr/config.yaml (default config)
            4. Empty AgentConfig() if no config found
        """
        config_dir = Path.home() / ".config" / "smart-ocr"

        if config_path:
            path = Path(config_path)
            if path.exists():
                return cls.from_file(path)
            raise FileNotFoundError(f"Config file not found: {path}")

        if profile:
            profile_path = config_dir / f"{profile}.yaml"
            if profile_path.exists():
                return cls.from_file(profile_path)
            raise FileNotFoundError(f"Profile not found: {profile_path}")

        # Try default config
        default_path = config_dir / "config.yaml"
        if default_path.exists():
            return cls.from_file(default_path)

        return cls()
