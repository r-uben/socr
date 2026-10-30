"""Per-call Gemini spend ledger (tokens + USD + outcomes)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from socr.core.gemini_pricing import (
    PRICING_AS_OF,
    PRICING_SOURCE_URL,
    GeminiUsage,
    compute_gemini_usd,
)

GeminiCallOutcome = Literal[
    "success",
    "empty",
    "error",
    "rejected",
    "skipped_budget",
    "skipped_dedup",
    "skipped_no_metadata",
]


@dataclass
class GeminiCallRecord:
    document_key: str
    page_num: int
    engine: str
    model: str
    outcome: GeminiCallOutcome
    usd: float | None
    usage: GeminiUsage
    transient_retry: bool = False
    detail: str = ""
    call_id: int = 0


@dataclass
class GeminiSpendTracker:
    """Process-scoped spend totals (one Python process / public run).

    Cross-process budget sharing is out of scope: concurrent workers in the
    same process share this tracker via the pipeline instance.
    """

    calls: list[GeminiCallRecord] = field(default_factory=list)
    _next_id: int = 1

    @property
    def run_usd_total(self) -> float:
        total = 0.0
        for call in self.calls:
            if call.usd is not None:
                total += call.usd
        return total

    def document_usd_total(self, document_key: str) -> float:
        total = 0.0
        for call in self.calls:
            if call.document_key == document_key and call.usd is not None:
                total += call.usd
        return total

    def record_call(
        self,
        *,
        document_key: str,
        page_num: int,
        engine: str,
        model: str,
        usage: GeminiUsage,
        outcome: GeminiCallOutcome,
        transient_retry: bool = False,
        detail: str = "",
    ) -> int:
        usd = compute_gemini_usd(model, usage)
        if outcome == "skipped_no_metadata":
            usd = None
        call_id = self._next_id
        self._next_id += 1
        self.calls.append(
            GeminiCallRecord(
                call_id=call_id,
                document_key=document_key,
                page_num=page_num,
                engine=engine,
                model=model,
                outcome=outcome,
                usd=usd,
                usage=usage,
                transient_retry=transient_retry,
                detail=detail,
            )
        )
        return call_id

    def set_outcome(self, call_id: int, outcome: GeminiCallOutcome) -> None:
        for call in self.calls:
            if call.call_id == call_id:
                call.outcome = outcome
                return

    def summarize(self, *, document_key: str | None = None) -> dict[str, Any]:
        calls = self.calls
        if document_key is not None:
            calls = [c for c in calls if c.document_key == document_key]

        tokens = {
            "prompt": 0,
            "candidates": 0,
            "thoughts": 0,
            "cached_content": 0,
            "total_reported": 0,
        }
        by_outcome: dict[str, int] = {}
        usd_known = 0.0
        usd_unknown_calls = 0
        for call in calls:
            by_outcome[call.outcome] = by_outcome.get(call.outcome, 0) + 1
            if call.usd is None:
                usd_unknown_calls += 1
            else:
                usd_known += call.usd
            u = call.usage
            if u.usage_metadata_present:
                tokens["prompt"] += u.prompt_token_count or 0
                tokens["candidates"] += u.candidates_token_count or 0
                tokens["thoughts"] += u.thoughts_token_count or 0
                tokens["cached_content"] += u.cached_content_token_count or 0
                tokens["total_reported"] += u.total_token_count or 0

        return {
            "version": 1,
            "pricing_source": PRICING_SOURCE_URL,
            "pricing_as_of": PRICING_AS_OF,
            "scope": "per_process",
            "run_usd_total": self.run_usd_total,
            "document_usd_total": self.document_usd_total(document_key) if document_key else None,
            "usd_known_in_scope": usd_known,
            "calls_missing_usd": usd_unknown_calls,
            "tokens": tokens,
            "calls_by_outcome": by_outcome,
            "calls": [
                {
                    "call_id": c.call_id,
                    "document_key": c.document_key,
                    "page_num": c.page_num,
                    "engine": c.engine,
                    "model": c.model,
                    "outcome": c.outcome,
                    "usd": c.usd,
                    "transient_retry": c.transient_retry,
                    "detail": c.detail,
                    "usage": asdict(c.usage),
                }
                for c in calls
            ],
        }


def _optional_int(value) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def usage_from_llm_fields(
    *,
    prompt: int | None,
    candidates: int | None,
    thoughts: int | None,
    cached: int | None,
    total: int | None,
    metadata_present: bool,
) -> GeminiUsage:
    prompt_i = _optional_int(prompt)
    cand_i = _optional_int(candidates)
    thoughts_i = _optional_int(thoughts)
    cached_i = _optional_int(cached)
    total_i = _optional_int(total)
    present = metadata_present and any(
        v is not None for v in (prompt_i, cand_i, thoughts_i, cached_i, total_i)
    )
    return GeminiUsage(
        prompt_token_count=prompt_i,
        candidates_token_count=cand_i,
        thoughts_token_count=thoughts_i,
        cached_content_token_count=cached_i,
        total_token_count=total_i,
        usage_metadata_present=present,
    )


def outcome_from_page_status(*, success: bool, empty: bool, error: bool) -> GeminiCallOutcome:
    if success:
        return "success"
    if empty:
        return "empty"
    return "error"


def apply_usage_to_page_output(output, usage: GeminiUsage, *, model: str) -> None:
    """Attach parsed usage + token-derived USD to a ``PageOutput``."""
    from socr.core.gemini_pricing import compute_gemini_usd

    output.llm_input_tokens = usage.prompt_token_count
    output.llm_output_tokens = usage.billable_output_tokens or usage.candidates_token_count
    output.llm_thoughts_tokens = usage.thoughts_token_count
    output.llm_cached_tokens = usage.cached_content_token_count
    output.gemini_usage_metadata_present = usage.usage_metadata_present
    usd = compute_gemini_usd(model, usage)
    if usd is not None:
        output.cost_usd = usd
    elif not usage.usage_metadata_present:
        output.cost_usd = None


def attempt_cost_usd_for_profile(profile, output) -> float | None:
    """USD for one provider attempt: token truth for Gemini, profile estimate otherwise."""
    from socr.core.config import EngineType

    if profile.engine == EngineType.GEMINI and output is not None:
        if getattr(output, "gemini_usage_metadata_present", False):
            if getattr(output, "cost_usd", None) is not None:
                return float(output.cost_usd)
            return 0.0
        return profile.cost_per_page_usd
    return profile.cost_per_page_usd
