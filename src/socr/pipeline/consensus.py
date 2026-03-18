"""Multi-engine consensus: select the best output from multiple engine attempts.

When multiple engines process the same page, this module compares their outputs
and selects (or merges) the best version.  Two strategies are available:

  1. **Heuristic** (default): scores each attempt by word count, structural
     richness (headers, tables, lists), audit status, and engine confidence,
     then picks the highest-scoring output.

  2. **LLM arbiter** (optional): sends the top candidates to a local Ollama
     model that identifies discrepancies and returns the most accurate version.

The entry point is ``ConsensusEngine.reconcile_document(state)``, which
iterates over all pages in a ``DocumentState`` and returns a
``ConsensusResult`` per page.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

import httpx

from socr.core.result import PageOutput, PageStatus
from socr.core.state import DocumentState

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------


@dataclass
class ConsensusResult:
    """Result of multi-engine consensus for a page."""

    page_num: int
    selected_engine: str  # which engine's output was chosen
    merged_text: str  # the final text (may be merged from multiple)
    agreement_score: float  # 0-1, how much engines agreed
    discrepancies: list[str] = field(default_factory=list)  # notable differences


# ------------------------------------------------------------------
# Scoring helpers (pure functions, no state)
# ------------------------------------------------------------------


def _word_set(text: str) -> set[str]:
    """Lowercase word set for Jaccard similarity."""
    return set(text.lower().split())


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two sets."""
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def _count_structure(text: str) -> int:
    """Count structural markdown elements (headers, tables, list items)."""
    count = 0
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            count += 1
        elif stripped.startswith("|") and stripped.endswith("|"):
            count += 1
        elif re.match(r"^[-*+]\s", stripped) or re.match(r"^\d+\.\s", stripped):
            count += 1
    return count


def _score_attempt(attempt: PageOutput) -> float:
    """Score a single attempt on multiple criteria.

    Returns a composite score (higher is better).  The components are
    weighted so that no single signal dominates:
      - word count     : log-scaled, avoids rewarding padding
      - structure count: bounded contribution
      - audit bonus    : flat bonus for passing audit
      - confidence     : engine-reported confidence
    """
    import math

    wc = attempt.word_count
    # Log-scale word count to reward having content without linearly
    # rewarding padding.  +1 to avoid log(0).
    wc_score = math.log1p(wc)

    struct_count = _count_structure(attempt.text)
    struct_score = min(struct_count, 20)  # cap contribution

    audit_bonus = 10.0 if attempt.audit_passed else 0.0

    conf_score = attempt.confidence * 5.0  # scale 0-5

    return wc_score + struct_score + audit_bonus + conf_score


def _pairwise_agreement(attempts: list[PageOutput]) -> float:
    """Average pairwise Jaccard similarity across all attempt pairs."""
    if len(attempts) < 2:
        return 1.0

    word_sets = [_word_set(a.text) for a in attempts]
    total = 0.0
    count = 0
    for i in range(len(word_sets)):
        for j in range(i + 1, len(word_sets)):
            total += _jaccard(word_sets[i], word_sets[j])
            count += 1
    return total / count if count else 1.0


def _find_discrepancies(attempts: list[PageOutput]) -> list[str]:
    """Identify notable differences between attempts.

    Looks at word-count spread and per-engine audit status divergence.
    """
    discs: list[str] = []
    if len(attempts) < 2:
        return discs

    word_counts = [(a.engine, a.word_count) for a in attempts]
    wc_values = [wc for _, wc in word_counts]
    if wc_values:
        spread = max(wc_values) - min(wc_values)
        avg = sum(wc_values) / len(wc_values) if wc_values else 1
        if avg > 0 and spread / avg > 0.3:
            sorted_wcs = sorted(word_counts, key=lambda x: x[1], reverse=True)
            desc = ", ".join(f"{eng}={wc}" for eng, wc in sorted_wcs)
            discs.append(f"Word count spread: {desc}")

    audit_statuses = {a.engine: a.audit_passed for a in attempts}
    passed = [e for e, v in audit_statuses.items() if v]
    failed = [e for e, v in audit_statuses.items() if not v]
    if passed and failed:
        discs.append(
            f"Audit divergence: passed=[{', '.join(passed)}], "
            f"failed=[{', '.join(failed)}]"
        )

    return discs


# ------------------------------------------------------------------
# Ollama LLM helper
# ------------------------------------------------------------------

_OLLAMA_COMPARE_PROMPT = """\
You are an expert OCR quality judge. Below are {n} different OCR outputs of \
the same document page. Compare them carefully.

{outputs_block}

Instructions:
1. Identify which output is the most accurate, complete, and well-formatted.
2. If parts of different outputs are better, merge the best parts into a \
single coherent version.
3. Return your answer as JSON with two keys:
   - "selected": the 1-based index of the best output (or 0 if you merged)
   - "text": the final best text

Return ONLY the JSON object, no other text.
"""


def _call_ollama(
    prompt: str,
    model: str,
    base_url: str = "http://localhost:11434",
    timeout: float = 120.0,
) -> str | None:
    """Call Ollama generate API and return the response text, or None on failure."""
    url = f"{base_url}/api/generate"
    try:
        resp = httpx.post(
            url,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "")
    except (httpx.HTTPError, httpx.TimeoutException, Exception) as exc:
        logger.warning("Ollama call failed: %s", exc)
        return None


def _parse_llm_response(
    raw: str, attempts: list[PageOutput]
) -> tuple[str, str] | None:
    """Parse the LLM JSON response.

    Returns (selected_engine, merged_text) or None on parse failure.
    """
    # Try to extract JSON from the response (the model may wrap it in markdown)
    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not json_match:
        return None
    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError:
        return None

    text = data.get("text", "")
    selected_idx = data.get("selected", 0)

    if not text:
        return None

    if isinstance(selected_idx, int) and 1 <= selected_idx <= len(attempts):
        engine = attempts[selected_idx - 1].engine
    else:
        engine = "llm-merged"

    return engine, text


# ------------------------------------------------------------------
# ConsensusEngine
# ------------------------------------------------------------------


class ConsensusEngine:
    """Selects the best output from multiple engine attempts per page."""

    def __init__(
        self,
        use_llm: bool = False,
        ollama_model: str = "",
        ollama_url: str = "http://localhost:11434",
    ) -> None:
        self.use_llm = use_llm
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url

    # ------------------------------------------------------------------
    # Heuristic selection
    # ------------------------------------------------------------------

    def select_best(self, attempts: list[PageOutput]) -> ConsensusResult:
        """Pick the best output from multiple attempts using heuristics.

        Strategy:
        1. Filter out failed/empty attempts.
        2. Score each by word count, structure, audit status, confidence.
        3. Pick the highest scoring one.
        4. Calculate agreement by comparing word overlap between attempts.
        """
        if not attempts:
            return ConsensusResult(
                page_num=0,
                selected_engine="none",
                merged_text="",
                agreement_score=0.0,
            )

        page_num = attempts[0].page_num

        # Filter out empty / error attempts
        viable = [
            a
            for a in attempts
            if a.text.strip() and a.status != PageStatus.ERROR
        ]

        if not viable:
            # All failed — return the first attempt's text as a last resort
            return ConsensusResult(
                page_num=page_num,
                selected_engine=attempts[0].engine,
                merged_text=attempts[0].text,
                agreement_score=0.0,
                discrepancies=["All attempts failed or produced empty output"],
            )

        if len(viable) == 1:
            a = viable[0]
            return ConsensusResult(
                page_num=page_num,
                selected_engine=a.engine,
                merged_text=a.text,
                agreement_score=1.0,
            )

        # Score each viable attempt
        scored = [(a, _score_attempt(a)) for a in viable]
        scored.sort(key=lambda x: x[1], reverse=True)
        best_attempt = scored[0][0]

        agreement = _pairwise_agreement(viable)
        discrepancies = _find_discrepancies(viable)

        return ConsensusResult(
            page_num=page_num,
            selected_engine=best_attempt.engine,
            merged_text=best_attempt.text,
            agreement_score=agreement,
            discrepancies=discrepancies,
        )

    # ------------------------------------------------------------------
    # LLM-based selection
    # ------------------------------------------------------------------

    def select_best_with_llm(
        self, attempts: list[PageOutput], ollama_model: str = ""
    ) -> ConsensusResult:
        """Use an LLM to compare outputs and select/merge the best.

        Sends the top 2-3 viable attempts to an Ollama model.  Falls back
        to heuristic selection if Ollama is unavailable or returns garbage.
        """
        model = ollama_model or self.ollama_model
        if not model:
            logger.info("No Ollama model specified, falling back to heuristic")
            return self.select_best(attempts)

        # Pre-filter viable
        viable = [
            a
            for a in attempts
            if a.text.strip() and a.status != PageStatus.ERROR
        ]
        if len(viable) < 2:
            return self.select_best(attempts)

        # Take top 3 by heuristic score to limit prompt size
        scored = sorted(viable, key=_score_attempt, reverse=True)[:3]

        outputs_block = "\n\n".join(
            f"--- Output {i + 1} (engine: {a.engine}) ---\n{a.text}"
            for i, a in enumerate(scored)
        )
        prompt = _OLLAMA_COMPARE_PROMPT.format(
            n=len(scored), outputs_block=outputs_block
        )

        raw = _call_ollama(prompt, model, base_url=self.ollama_url)
        if raw is None:
            logger.info("Ollama unavailable, falling back to heuristic")
            return self.select_best(attempts)

        parsed = _parse_llm_response(raw, scored)
        if parsed is None:
            logger.warning("Could not parse LLM response, falling back to heuristic")
            return self.select_best(attempts)

        engine_name, merged_text = parsed
        page_num = attempts[0].page_num if attempts else 0

        agreement = _pairwise_agreement(viable)
        discrepancies = _find_discrepancies(viable)

        return ConsensusResult(
            page_num=page_num,
            selected_engine=engine_name,
            merged_text=merged_text,
            agreement_score=agreement,
            discrepancies=discrepancies,
        )

    # ------------------------------------------------------------------
    # Document-level reconciliation
    # ------------------------------------------------------------------

    def reconcile_document(
        self, state: DocumentState
    ) -> list[ConsensusResult]:
        """Run consensus across all pages that have multiple attempts."""
        results: list[ConsensusResult] = []

        for page_num in sorted(state.pages):
            page_state = state.pages[page_num]

            # Skip born-digital pages — they use native text
            if page_state.is_born_digital and page_state.native_text:
                continue

            if len(page_state.attempts) < 2:
                continue

            if self.use_llm and self.ollama_model:
                cr = self.select_best_with_llm(page_state.attempts)
            else:
                cr = self.select_best(page_state.attempts)

            # Ensure correct page_num (defensive)
            cr.page_num = page_num
            results.append(cr)

            # Update the page state's best_output with the consensus winner
            page_state.best_output = PageOutput(
                page_num=page_num,
                text=cr.merged_text,
                status=PageStatus.SUCCESS if cr.merged_text.strip() else PageStatus.ERROR,
                engine=f"consensus({cr.selected_engine})",
                audit_passed=True,  # consensus output is trusted
                confidence=cr.agreement_score,
            )

        return results
