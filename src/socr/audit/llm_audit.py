"""LLM-based quality audit using Ollama."""

import json
import re
from dataclasses import dataclass, field

import httpx


@dataclass
class LLMAuditResult:
    """Result of LLM-based quality audit."""

    verdict: str = "unknown"  # acceptable, needs_review, poor
    confidence: float = 0.0
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    reasoning: str = ""

    @property
    def passed(self) -> bool:
        """Check if audit passed."""
        return self.verdict == "acceptable"


class LLMAuditor:
    """LLM-based quality auditor using Ollama."""

    AUDIT_PROMPT = """You are an OCR quality auditor. Analyze this extracted text and determine if it's acceptable quality.

<extracted_text>
{text}
</extracted_text>

Evaluate based on:
1. Readability: Can humans understand the text?
2. Completeness: Does it seem like a complete extraction (no obvious missing parts)?
3. Accuracy: Are there obvious OCR errors (garbled text, wrong characters)?
4. Structure: Is the structure preserved (headers, paragraphs, lists)?

Respond in JSON format:
{{
    "verdict": "acceptable" | "needs_review" | "poor",
    "confidence": 0.0-1.0,
    "issues": ["list of specific issues found"],
    "suggestions": ["suggestions for improvement"],
    "reasoning": "brief explanation of your verdict"
}}

Only respond with valid JSON, no other text."""

    def __init__(
        self,
        model: str = "llama3.2",
        ollama_host: str = "http://localhost:11434",
        timeout: int = 60,
    ) -> None:
        self.model = model
        self.ollama_host = ollama_host
        self.timeout = timeout
        self._client: httpx.Client | None = None

    def _get_client(self) -> httpx.Client:
        """Get or create HTTP client."""
        if not self._client:
            self._client = httpx.Client(
                base_url=self.ollama_host,
                timeout=self.timeout,
            )
        return self._client

    def is_available(self) -> bool:
        """Check if Ollama and model are available."""
        try:
            client = self._get_client()
            response = client.get("/api/tags")
            if response.status_code != 200:
                return False

            models = response.json().get("models", [])
            model_names = [m.get("name", "").split(":")[0] for m in models]
            return self.model.split(":")[0] in model_names

        except Exception:
            return False

    def audit(self, text: str, max_chars: int = 4000) -> LLMAuditResult:
        """Audit OCR output using LLM."""
        if not text or not text.strip():
            return LLMAuditResult(
                verdict="poor",
                confidence=1.0,
                issues=["Empty text output"],
                reasoning="No text was extracted",
            )

        # Truncate if too long
        if len(text) > max_chars:
            text = text[:max_chars] + "\n\n[... truncated for audit ...]"

        try:
            client = self._get_client()
            response = client.post(
                "/api/generate",
                json={
                    "model": self.model,
                    "prompt": self.AUDIT_PROMPT.format(text=text),
                    "stream": False,
                    "format": "json",
                },
            )

            if response.status_code != 200:
                return LLMAuditResult(
                    verdict="unknown",
                    issues=[f"Ollama error: {response.status_code}"],
                )

            llm_response = response.json().get("response", "")
            return self._parse_response(llm_response)

        except Exception as e:
            return LLMAuditResult(
                verdict="unknown",
                issues=[f"Audit error: {e}"],
            )

    def _parse_response(self, response: str) -> LLMAuditResult:
        """Parse LLM response into structured result."""
        def to_result(data: dict) -> LLMAuditResult:
            return LLMAuditResult(
                verdict=data.get("verdict", "unknown"),
                confidence=float(data.get("confidence", 0.0)),
                issues=data.get("issues", []),
                suggestions=data.get("suggestions", []),
                reasoning=data.get("reasoning", ""),
            )

        # 1) If the model obeyed instructions, this should work.
        try:
            data = json.loads(response)
            if isinstance(data, dict):
                return to_result(data)
        except Exception:
            pass

        # 2) Otherwise, try to extract the first valid JSON object from the response,
        # including nested structures.
        decoder = json.JSONDecoder()
        for start in range(len(response)):
            if response[start] != "{":
                continue
            try:
                data, _end = decoder.raw_decode(response[start:])
                if isinstance(data, dict):
                    return to_result(data)
            except Exception:
                continue

        # 3) Fall back to heuristic parsing.
        try:
            raise json.JSONDecodeError("No JSON object found", response, 0)
        except json.JSONDecodeError:
            # Try to extract verdict from plain text
            verdict = "unknown"
            if "acceptable" in response.lower():
                verdict = "acceptable"
            elif "poor" in response.lower():
                verdict = "poor"
            elif "needs_review" in response.lower() or "review" in response.lower():
                verdict = "needs_review"

            return LLMAuditResult(
                verdict=verdict,
                reasoning=response[:500],
            )

    def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            self._client.close()
            self._client = None
