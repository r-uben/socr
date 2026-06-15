"""Shared figure-description prompt and non-authoritative caption marker.

All figure-description engines (GeminiAPIEngine, OllamaFigureEngine, VLLMEngine)
import from this module so the anti-fabrication instruction and the self-identifying
marker are defined exactly once and cannot drift between paths.
"""

# Marker prepended to every model-generated caption in the output markdown so
# readers know the text was produced by a vision model and may not be accurate.
CAPTION_MARKER = "[model-generated, non-authoritative gist]"


def build_figure_prompt(figure_type: str, context: str) -> str:
    """Return the hardened figure-description prompt.

    Instructs the model to describe only what is visibly legible and to omit
    (not guess) numeric values, axis ranges, tick-mark numbers, thresholds, and
    arrow labels that cannot be read clearly from the image.  The goal is a
    searchable gist, not a verbatim transcription — "omit what you cannot read"
    rather than "describe nothing".
    """
    type_prefix = (
        f"This appears to be a {figure_type}. " if figure_type and figure_type != "unknown" else ""
    )
    base = (
        f"{type_prefix}"
        "Describe what this figure visibly shows. "
        "Focus on the overall structure, general trend or pattern, and any "
        "clearly legible labels or categories that you can read directly from "
        "the image. "
        "IMPORTANT — omit anything you cannot clearly read: do NOT guess or "
        "infer specific numeric values, axis ranges, tick-mark numbers, "
        "thresholds, or arrow labels that are not unambiguously legible. "
        "If a value is blurry, small, or uncertain, skip it entirely rather "
        "than estimating. "
        "The goal is a concise, honest gist that avoids fabricating details."
    )
    if context:
        base += f"\n\nContext from surrounding text: {context[:500]}"
    return base


def wrap_caption(description: str) -> str:
    """Prepend the non-authoritative marker to a model-generated description.

    Returns the description unchanged if it is empty or already starts with
    the marker (idempotent).  Error/unavailable strings must NOT be wrapped —
    the caller is responsible for skipping wrap_caption on those paths.
    """
    if not description:
        return description
    if description.startswith(CAPTION_MARKER):
        return description
    return f"{CAPTION_MARKER} {description}"
