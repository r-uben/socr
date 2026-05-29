"""The judge: a stateless per-page verdict on OCR faithfulness.

Python owns the loop; the judge is the single-turn decision function it calls.
The judge PROMPT is policy and lives in ``src/socr/prompts/judge_page.md`` — not
baked into code — so it can be iterated without touching control flow.
"""

from socr.judge.judge import (
    Judge,
    JudgeVerdict,
    load_judge_prompt,
    parse_verdict,
)

__all__ = ["Judge", "JudgeVerdict", "load_judge_prompt", "parse_verdict"]
