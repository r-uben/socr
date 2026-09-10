"""Stateless OCR-faithfulness judge.

A ``Judge`` looks at one rendered page image plus a candidate OCR transcription
and returns a ``JudgeVerdict``. It holds no state across pages — Python calls it
once per flagged page inside a bounded loop it owns itself.

The judge prompt is loaded from ``prompts/judge_page.md`` (policy as data). The
model is asked to return a small JSON verdict; ``parse_verdict`` tolerates the
usual model noise (markdown fences, leading prose) and coerces to the dataclass.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import httpx

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "judge_page.md"

VALID_ACTIONS = {"accept", "retry_same", "escalate_engine"}


#: #713: exception TYPES that mean the PAGE judge never returned a verdict
#: because the call did not complete in time. ``httpx.TimeoutException`` covers
#: connect/read/write/pool timeouts from the Ollama and HTTP judge backends;
#: builtin ``TimeoutError`` covers ``socket.timeout`` and
#: ``concurrent.futures.TimeoutError`` (both aliases of it since 3.11), and
#: ``subprocess.TimeoutExpired`` covers a CLI-backed judge.
#:
#: Everything else -- a transport error that is not a timeout, an HTTP status
#: error, a decode failure, a defect in our own code -- is deliberately NOT a
#: timeout. The distinction is load-bearing: only a timeout can license #713's
#: credentialed stand-in, so widening this tuple widens what may ship.
_PAGE_JUDGE_TIMEOUT_EXCEPTIONS: tuple[type[BaseException], ...] = (
    TimeoutError,
    subprocess.TimeoutExpired,
    httpx.TimeoutException,
)


class PageJudgeTimeoutError(TimeoutError):
    """#713 round 2 (Astra P1-3): the page judge missed its wall-clock deadline.

    Raised by the orchestrator's ``_TimeoutJudge`` adapter when the inner judge
    does not answer in time. Before this the adapter turned its own deadline
    into ``AcceptDecision(accept=False, reason="judge timeout")``, so the real
    ``_phase_agentic`` loop never entered ``route_page``'s exception branch and
    the typed ``judge_outcome`` was never written on a production timeout -- the
    one path #713 exists for. A rejection is also the wrong shape for it: a
    deadline is a MISSING verdict, not a negative one.

    Subclasses ``TimeoutError`` so ``is_page_judge_timeout`` classifies it
    without a special case, and so any caller that already handled a timeout
    from an inner judge keeps handling this one.

    The message deliberately contains the word "timeout": ``_phase_agentic``'s
    cascade-halt probe scans attempt reasons for that substring to decide
    whether a wedged backend should stop the document, and that check reads the
    interpolated ``judge raised: {exc}`` text.
    """


def is_page_judge_timeout(exc: BaseException) -> bool:
    """Whether ``exc`` means the page judge TIMED OUT rather than misbehaved.

    Classified by exception TYPE, never by the text of the message. The judge
    guard builds its ``judge_reason`` by interpolating ``str(exc)``, and that
    string can carry any words a remote service or a model chose to emit -- a
    gate keyed on the substring "timed out" is a gate any upstream can open.
    """
    return isinstance(exc, _PAGE_JUDGE_TIMEOUT_EXCEPTIONS)


def load_judge_prompt() -> str:
    """Read the judge prompt template (policy lives in the .md, not in code)."""
    return _PROMPT_PATH.read_text(encoding="utf-8")


@dataclass
class JudgeVerdict:
    """A single-page faithfulness verdict."""

    faithful: bool
    issues: list[str] = field(default_factory=list)
    confidence: float = 0.0
    suggested_action: str = "accept"
    raw: str = ""  # raw model output, kept for the manifest journal / debugging

    @property
    def is_good(self) -> bool:
        return self.faithful


class Judge(Protocol):
    """Anything that can turn (page image, ocr text) into a verdict."""

    def judge(self, image_path: Path, ocr_text: str) -> JudgeVerdict: ...


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of model output.

    Handles ```json fences and leading/trailing prose by locating the first
    balanced-looking ``{...}`` span. Raises ValueError if none parses.
    """
    stripped = text.strip()
    # Fast path: whole thing is JSON.
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    # Strip a ```json ... ``` fence if present.
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    if fence:
        return json.loads(fence.group(1))
    # Last resort: first '{' to last '}'.
    start, end = stripped.find("{"), stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(stripped[start : end + 1])
    raise ValueError(f"no JSON object found in judge output: {text[:200]!r}")


def parse_verdict(text: str) -> JudgeVerdict:
    """Parse a model response into a JudgeVerdict, coercing/validating fields."""
    data = _extract_json(text)
    faithful = bool(data.get("faithful", False))
    issues = [str(i) for i in data.get("issues", []) or []]
    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    action = data.get("suggested_action") or ("accept" if faithful else "escalate_engine")
    if action not in VALID_ACTIONS:
        action = "accept" if faithful else "escalate_engine"
    return JudgeVerdict(
        faithful=faithful,
        issues=issues,
        confidence=confidence,
        suggested_action=action,
        raw=text,
    )
