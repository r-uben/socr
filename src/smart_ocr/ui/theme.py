"""Minimal theme for smart-ocr terminal UI."""

from rich.style import Style
from rich.theme import Theme

# Muted, zen-like colors
PRIMARY_COLOR = "#7C9CB5"      # Muted blue
ACCENT_COLOR = "#A8B5A0"       # Sage green  
WARN_COLOR = "#C9A86C"         # Muted gold
ERROR_COLOR = "#B07878"        # Muted red
DIM_COLOR = "#6B7280"          # Gray

# Engine colors (muted)
NOUGAT_COLOR = "#B08968"       # Warm brown
DEEPSEEK_COLOR = "#7C9CB5"     # Muted blue
DEEPSEEK_VLLM_COLOR = "#6B8FAB"  # Deeper blue (HPC mode)
MISTRAL_COLOR = "#9B8AA6"      # Muted purple
GEMINI_COLOR = "#7BA695"       # Sage
OLLAMA_COLOR = "#8BA888"       # Soft green
VLLM_COLOR = "#C9A86C"         # Muted gold (local vision)

STAGE_COLORS = {
    "classify": "#9B8AA6",
    "primary": "#7C9CB5",
    "audit": "#8BA888",
    "fallback": "#B08968",
    "figures": "#7BA695",
    "reconcile": "#6B8FAB",  # HPC reconciliation
}

ENGINE_STYLES = {
    "nougat": Style(color=NOUGAT_COLOR),
    "deepseek": Style(color=DEEPSEEK_COLOR),
    "deepseek-vllm": Style(color=DEEPSEEK_VLLM_COLOR),
    "mistral": Style(color=MISTRAL_COLOR),
    "gemini": Style(color=GEMINI_COLOR),
    "ollama": Style(color=OLLAMA_COLOR),
    "vllm": Style(color=VLLM_COLOR),
}

# No emojis - simple text markers
ENGINE_ICONS = {
    "nougat": "",
    "deepseek": "",
    "deepseek-vllm": "",
    "mistral": "",
    "gemini": "",
    "ollama": "",
    "vllm": "",
}

ENGINE_LABELS = {
    "nougat": "nougat",
    "deepseek": "deepseek",
    "deepseek-vllm": "deepseek-vllm",
    "mistral": "mistral",
    "gemini": "gemini",
    "ollama": "ollama",
    "vllm": "vllm",
}

# Minimal status markers
STATUS_ICONS = {
    "success": "+",
    "warning": "!",
    "error": "x",
    "pending": ".",
    "running": "*",
    "skipped": "-",
}

AGENT_THEME = Theme({
    "nougat": ENGINE_STYLES["nougat"],
    "deepseek": ENGINE_STYLES["deepseek"],
    "deepseek-vllm": ENGINE_STYLES["deepseek-vllm"],
    "mistral": ENGINE_STYLES["mistral"],
    "gemini": ENGINE_STYLES["gemini"],
    "ollama": ENGINE_STYLES["ollama"],
    "vllm": ENGINE_STYLES["vllm"],
    "success": Style(color=ACCENT_COLOR),
    "warning": Style(color=WARN_COLOR),
    "error": Style(color=ERROR_COLOR),
    "info": Style(color=PRIMARY_COLOR),
    "dim": Style(color=DIM_COLOR),
    "stage.title": Style(color="#D1D5DB"),
    "stage.border": Style(color="#4B5563"),
    "header": Style(color="#D1D5DB"),
    "highlight": Style(color=WARN_COLOR),
})
