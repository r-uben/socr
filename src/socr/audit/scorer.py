"""Failure-mode scorer: maps audit heuristic results to FailureMode enum values.

Wraps HeuristicsChecker output — the checker does detection, this module
does classification so the repair router knows what fallback strategy to use.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from socr.audit.heuristics import HeuristicsChecker, HeuristicsResult
from socr.core.result import FailureMode


# Priority order: higher index = higher priority when selecting primary failure.
# Rationale: hallucination and refusal are the most actionable (they tell
# the router to switch models), while low word count is often a symptom
# rather than a root cause.
_PRIORITY: dict[FailureMode, int] = {
    FailureMode.LOW_WORD_COUNT: 1,
    FailureMode.GARBAGE: 2,
    FailureMode.TRUNCATED: 3,
    FailureMode.EMPTY_OUTPUT: 4,
    FailureMode.REFUSAL: 5,
    FailureMode.HALLUCINATION: 6,
}


@dataclass
class ScoringResult:
    """Outcome of failure-mode classification."""

    failure_modes: list[FailureMode] = field(default_factory=list)
    primary_failure: FailureMode = FailureMode.NONE
    confidence: float = 1.0
    details: dict[FailureMode, str] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return len(self.failure_modes) == 0


# Mapping from HeuristicsChecker metric names to (FailureMode, detail template).
_METRIC_MAP: dict[str, tuple[FailureMode, str]] = {
    "Empty output": (FailureMode.EMPTY_OUTPUT, "No text extracted from document"),
    "LLM refusal": (FailureMode.REFUSAL, "Model refused to process the input"),
    "CID artifacts": (FailureMode.GARBAGE, "PDF font mapping failures (CID references)"),
    "Hallucination loops": (
        FailureMode.HALLUCINATION,
        "Repeated sentence patterns indicate model hallucination loop",
    ),
    "Formatting hallucination": (
        FailureMode.HALLUCINATION,
        "Model hallucinated formatting instructions instead of OCR output",
    ),
    "Word count": (FailureMode.LOW_WORD_COUNT, "Extracted text has too few words"),
    "Garbage ratio": (FailureMode.GARBAGE, "High ratio of non-text characters"),
    "Truncation check": (FailureMode.TRUNCATED, "Output appears truncated relative to document page count"),
}


class FailureModeScorer:
    """Classify audit failures into actionable FailureMode values.

    Two entry points:
      - ``score(text, engine=...)`` — runs HeuristicsChecker then classifies.
      - ``score_from_audit(audit_result)`` — classifies a pre-existing result.
    """

    def __init__(self, checker: HeuristicsChecker | None = None) -> None:
        self.checker = checker or HeuristicsChecker()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, text: str, engine: str = "", expected_pages: int = 0) -> ScoringResult:
        """Run heuristic checks on *text* and classify any failures."""
        audit = self.checker.check(text, expected_pages=expected_pages)
        return self.score_from_audit(audit)

    def score_from_audit(self, audit: HeuristicsResult) -> ScoringResult:
        """Classify an existing ``HeuristicsResult`` into failure modes."""
        if audit.passed:
            return ScoringResult()

        modes: list[FailureMode] = []
        details: dict[FailureMode, str] = {}

        for metric in audit.metrics:
            if metric.passed:
                continue
            if metric.severity not in ("error", "warning"):
                continue

            mapping = _METRIC_MAP.get(metric.name)
            if mapping is None:
                continue

            mode, detail_template = mapping
            # Enrich detail with actual metric value when useful.
            detail = f"{detail_template} ({metric.value})"

            if mode not in modes:
                modes.append(mode)
            # Later metrics of the same mode overwrite — fine, both are valid.
            details[mode] = detail

        if not modes:
            return ScoringResult()

        primary = self._select_primary(modes)
        confidence = self._estimate_confidence(modes, audit)

        return ScoringResult(
            failure_modes=modes,
            primary_failure=primary,
            confidence=confidence,
            details=details,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _select_primary(modes: list[FailureMode]) -> FailureMode:
        """Pick the most actionable failure mode by priority."""
        return max(modes, key=lambda m: _PRIORITY.get(m, 0))

    @staticmethod
    def _estimate_confidence(
        modes: list[FailureMode], audit: HeuristicsResult
    ) -> float:
        """Rough confidence in the diagnosis.

        Multiple corroborating failures increase confidence. A single
        warning-level failure gets lower confidence than a clear error.
        """
        error_count = len(audit.errors)
        warning_count = len(audit.warnings)

        if error_count >= 2:
            return 0.95
        if error_count == 1 and warning_count >= 1:
            return 0.85
        if error_count == 1:
            return 0.75
        # Only warnings, no errors — shouldn't normally reach here since
        # audit.passed would be True, but handle gracefully.
        return 0.5
