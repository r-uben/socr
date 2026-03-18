"""Multi-engine consensus: select the best output from multiple engine attempts.

When multiple engines process the same page, this module compares their outputs
and selects (or merges) the best version.  Two strategies are available:

  1. **Heuristic** (default): scores each attempt using either grounded scoring
     (against native text from born-digital detection) or ungrounded scoring
     (word count, structural richness, audit status, confidence).  Grounded
     scoring uses WER to pick the output closest to the native text layer and
     penalises hallucination (excess word count).

  2. **LLM arbiter** (optional): sends the top candidates to a local Ollama
     model that identifies discrepancies and returns the most accurate version.

The entry point is ``ConsensusEngine.reconcile_document(state)``, which
iterates over all pages in a ``DocumentState`` and returns a
``ConsensusResult`` per page.
"""

from __future__ import annotations

import json
import logging
import math
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
# Edit-distance helpers (self-contained, no imports from benchmark)
# ------------------------------------------------------------------


def _levenshtein(seq_a: list[str], seq_b: list[str]) -> int:
    """Levenshtein edit distance between two sequences.

    Uses O(min(m, n)) space via a single-row DP approach.
    """
    m, n = len(seq_a), len(seq_b)

    # Optimise by making the shorter sequence the column dimension
    if m < n:
        seq_a, seq_b = seq_b, seq_a
        m, n = n, m

    prev = list(range(n + 1))
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            curr[j] = min(
                curr[j - 1] + 1,      # insertion
                prev[j] + 1,          # deletion
                prev[j - 1] + cost,   # substitution
            )
        prev, curr = curr, prev

    return prev[n]


def _normalize_words(text: str) -> list[str]:
    """Lowercase and split text into words for WER computation."""
    return text.lower().split()


def _compute_wer(hypothesis: str, reference: str) -> float:
    """Word Error Rate: edit_distance(ref, hyp) / len(ref).

    Returns 0.0 when both are empty.  Can exceed 1.0 when the hypothesis
    has many insertions relative to the reference.
    """
    ref_words = _normalize_words(reference)
    hyp_words = _normalize_words(hypothesis)
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return _levenshtein(ref_words, hyp_words) / len(ref_words)


# ------------------------------------------------------------------
# Scoring helpers (pure functions, no state)
# ------------------------------------------------------------------


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


def _score_attempt(attempt: PageOutput, reference_text: str = "") -> float:
    """Score a single attempt, optionally grounded against native text.

    When *reference_text* is available the score is dominated by WER
    against the reference (lower WER = higher score) with a penalty for
    hallucination (word count far exceeding the reference).

    When no reference is available, falls back to an ungrounded heuristic
    based on word count, structural richness, audit status, and confidence.
    """
    if reference_text.strip():
        return _score_attempt_grounded(attempt, reference_text)
    return _score_attempt_ungrounded(attempt)


def _score_attempt_grounded(attempt: PageOutput, reference_text: str) -> float:
    """Score grounded against native text.

    Components (all on a 0-100 scale so they dominate over any residual
    ungrounded signal):
      - WER fidelity : (1 - WER) * 70, capped at 0
      - Audit bonus  : +15 if audit passed
      - Hallucination penalty : -20 if word count exceeds reference by >50%
      - Structure bonus: +5 * min(struct_ratio, 1) where struct_ratio is
        attempt structure count / max(reference structure count, 1)
    """
    wer = _compute_wer(attempt.text, reference_text)
    # Clamp WER to [0, 2] so extremely bad outputs don't produce absurd
    # negative scores, but still get penalised heavily.
    wer_clamped = min(wer, 2.0)
    fidelity = (1.0 - wer_clamped) * 70.0

    audit_bonus = 15.0 if attempt.audit_passed else 0.0

    ref_wc = len(reference_text.split())
    hyp_wc = attempt.word_count
    hallucination_penalty = 0.0
    if ref_wc > 0 and hyp_wc > ref_wc * 1.5:
        hallucination_penalty = -20.0

    # Structure: reward matching or exceeding reference structure, but
    # don't penalise for less (the reference text layer may not have
    # markdown structure at all).
    ref_struct = max(_count_structure(reference_text), 1)
    hyp_struct = _count_structure(attempt.text)
    struct_bonus = 5.0 * min(hyp_struct / ref_struct, 1.0)

    return fidelity + audit_bonus + hallucination_penalty + struct_bonus


def _score_attempt_ungrounded(attempt: PageOutput) -> float:
    """Original ungrounded heuristic (no reference text).

    Components:
      - word count     : log-scaled, avoids rewarding padding
      - structure count: log-scaled (no hard cap)
      - audit bonus    : flat bonus for passing audit
      - confidence     : engine-reported confidence
    """
    wc = attempt.word_count
    wc_score = math.log1p(wc)

    struct_count = _count_structure(attempt.text)
    # Log-scale structure count instead of hard cap at 20
    struct_score = math.log1p(struct_count) * 5.0

    audit_bonus = 10.0 if attempt.audit_passed else 0.0
    conf_score = attempt.confidence * 5.0

    return wc_score + struct_score + audit_bonus + conf_score


# ------------------------------------------------------------------
# Agreement helpers
# ------------------------------------------------------------------


def _agreement_score(text_a: str, text_b: str) -> float:
    """Sequence-aware agreement between two texts using 1 - WER.

    Unlike Jaccard on word sets, this preserves word order.  A score of
    1.0 means identical word sequences; 0.0 means completely different.
    Clamped to [0, 1].
    """
    wer = _compute_wer(text_a, text_b)
    return max(0.0, 1.0 - wer)


def _pairwise_agreement(attempts: list[PageOutput]) -> float:
    """Average pairwise sequence-aware agreement across all attempt pairs."""
    if len(attempts) < 2:
        return 1.0

    total = 0.0
    count = 0
    for i in range(len(attempts)):
        for j in range(i + 1, len(attempts)):
            total += _agreement_score(attempts[i].text, attempts[j].text)
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
        quiet: bool = False,
    ) -> None:
        self.use_llm = use_llm
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url
        self.quiet = quiet

    # ------------------------------------------------------------------
    # Heuristic selection
    # ------------------------------------------------------------------

    def select_best(
        self, attempts: list[PageOutput], reference_text: str = ""
    ) -> ConsensusResult:
        """Pick the best output from multiple attempts using heuristics.

        When *reference_text* is provided (e.g. native text from a
        born-digital PDF), scoring is grounded against it: the output
        closest to the reference wins, and outputs with excessive word
        count are penalised as likely hallucination.

        When no reference is available, falls back to ungrounded scoring
        (word count + structure + audit + confidence).
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
            # All failed -- return the first attempt's text as a last resort
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
        scored = [(a, _score_attempt(a, reference_text=reference_text)) for a in viable]
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
        """Run consensus across all pages and whole-doc attempts.

        Handles two cases:
          1. Per-page attempts (HTTP engines) -- compare per page.
          2. Whole-doc attempts (CLI engines, page_num=0) -- compare at
             document level and promote the winner.

        When native text is available (born-digital detection), it is
        passed as reference for grounded scoring.
        """
        results: list[ConsensusResult] = []

        # --- Whole-doc consensus ---
        # CLI engines produce page_num=0 whole-doc outputs.  When we have
        # 2+ whole-doc attempts, pick the best one and promote it.
        if len(state.whole_doc_attempts) >= 2:
            # Assemble full native text from all pages for grounding
            native_full = "\n\n".join(
                p.native_text
                for p in state.pages.values()
                if p.native_text
            )
            cr = self._select_best_impl(
                state.whole_doc_attempts, reference_text=native_full
            )
            cr.page_num = 0
            results.append(cr)

            if not self.quiet:
                logger.info(
                    f"Whole-doc consensus: selected {cr.selected_engine} "
                    f"(agreement={cr.agreement_score:.2f})"
                )

            # Replace the whole-doc attempts list so state.text picks
            # the consensus winner (move it to the end).
            winner = PageOutput(
                page_num=0,
                text=cr.merged_text,
                status=PageStatus.SUCCESS if cr.merged_text.strip() else PageStatus.ERROR,
                engine=f"consensus({cr.selected_engine})",
                audit_passed=True,
                confidence=cr.agreement_score,
            )
            state.whole_doc_attempts.append(winner)

        # --- Per-page consensus ---
        for page_num in sorted(state.pages):
            page_state = state.pages[page_num]

            # Skip born-digital pages -- they use native text
            if page_state.is_born_digital and page_state.native_text:
                continue

            if len(page_state.attempts) < 2:
                continue

            cr = self._select_best_impl(
                page_state.attempts,
                reference_text=page_state.native_text or "",
            )
            cr.page_num = page_num
            results.append(cr)

            page_state.best_output = PageOutput(
                page_num=page_num,
                text=cr.merged_text,
                status=PageStatus.SUCCESS if cr.merged_text.strip() else PageStatus.ERROR,
                engine=f"consensus({cr.selected_engine})",
                audit_passed=True,
                confidence=cr.agreement_score,
            )

        return results

    def _select_best_impl(
        self, attempts: list[PageOutput], reference_text: str = ""
    ) -> ConsensusResult:
        """Route to heuristic or LLM consensus."""
        if self.use_llm and self.ollama_model:
            return self.select_best_with_llm(attempts)
        return self.select_best(attempts, reference_text=reference_text)
