"""Per-model Gemini token pricing and USD cost computation.

Source (retrieved 2026-10-30): https://ai.google.dev/gemini-api/docs/pricing
Google bills thinking tokens at the same rate as output (candidates) tokens.
Cached context reads use the published cached-input rate; non-cached prompt
tokens use the standard input rate.
"""

from __future__ import annotations

from dataclasses import dataclass

from socr.core.config import DEFAULT_GEMINI_MODEL, RETIRED_GEMINI_MODELS

PRICING_SOURCE_URL = "https://ai.google.dev/gemini-api/docs/pricing"
PRICING_AS_OF = "2026-10-30"


@dataclass(frozen=True)
class ModelTokenRates:
    """USD per 1M tokens."""

    input_per_million: float
    output_per_million: float
    cached_input_per_million: float


# One table — update here when Google changes list prices.
_GEMINI_RATES: dict[str, ModelTokenRates] = {
    DEFAULT_GEMINI_MODEL: ModelTokenRates(0.50, 3.00, 0.05),
    "gemini-3-flash-preview": ModelTokenRates(0.50, 3.00, 0.05),
    "gemini-2.5-flash": ModelTokenRates(0.30, 2.50, 0.075),
    "gemini-2.5-pro": ModelTokenRates(1.25, 10.00, 0.31),
}

for _retired in RETIRED_GEMINI_MODELS:
    _GEMINI_RATES.setdefault(_retired, _GEMINI_RATES["gemini-3-flash-preview"])


@dataclass(frozen=True)
class GeminiUsage:
    prompt_token_count: int | None = None
    candidates_token_count: int | None = None
    thoughts_token_count: int | None = None
    cached_content_token_count: int | None = None
    total_token_count: int | None = None
    usage_metadata_present: bool = False

    @property
    def billable_output_tokens(self) -> int:
        return max(0, (self.candidates_token_count or 0) + (self.thoughts_token_count or 0))


def rates_for_model(model: str) -> ModelTokenRates | None:
    key = (model or "").strip()
    if not key:
        return None
    if key in _GEMINI_RATES:
        return _GEMINI_RATES[key]
    # Best-effort prefix fallback for preview aliases.
    for name, rates in sorted(_GEMINI_RATES.items(), key=lambda x: -len(x[0])):
        if key.startswith(name.split("-preview")[0]):
            return rates
    return _GEMINI_RATES.get(DEFAULT_GEMINI_MODEL)


def _token_int(value: int | None) -> int:
    if value is None:
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def compute_gemini_usd(model: str, usage: GeminiUsage) -> float | None:
    """Return USD for one call, or ``None`` when usage metadata was absent."""
    if not usage.usage_metadata_present:
        return None
    rates = rates_for_model(model)
    if rates is None:
        return None
    prompt = _token_int(usage.prompt_token_count)
    cached = min(_token_int(usage.cached_content_token_count), prompt)
    non_cached = prompt - cached
    out_tok = usage.billable_output_tokens
    if not isinstance(out_tok, int):
        out_tok = _token_int(getattr(usage, "candidates_token_count", 0)) + _token_int(
            getattr(usage, "thoughts_token_count", 0)
        )
    usd = (
        non_cached * rates.input_per_million / 1_000_000
        + cached * rates.cached_input_per_million / 1_000_000
        + out_tok * rates.output_per_million / 1_000_000
    )
    return usd


# Reference OCR page shape (cluster audit Oct 2026): ~2k prompt, ~1.5k answer,
# ~5k thinking — used ONLY to derive ladder ordering estimates, not as a flat
# per-page constant in routing math.
_REFERENCE_OCR_USAGE = GeminiUsage(
    prompt_token_count=2000,
    candidates_token_count=1500,
    thoughts_token_count=5000,
    cached_content_token_count=0,
    total_token_count=8500,
    usage_metadata_present=True,
)


def reference_page_cost_usd(model: str | None = None) -> float:
    """Token-derived $/page estimate for ``PROFILE_GEMINI`` ladder ordering."""
    m = model or DEFAULT_GEMINI_MODEL
    return compute_gemini_usd(m, _REFERENCE_OCR_USAGE) or 0.0
