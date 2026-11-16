"""#1347: the page banner counts layer-silent numbers and names contradictions.

A layer-silent dispute ("reading says B, not in the text layer") is counted in one phrase in
the shipped banner; a contradiction ("text layer says A, reading says B") is named. The
sidecar's clause reason keeps every value. The rule is by kind, not by length.
"""

from __future__ import annotations

import re

from socr.benchmark.scoreboard import _is_marked
from socr.core.manifest import _apply_verdict_marks
from socr.core.marks import Check
from socr.core.result import PageOutput, PageStatus
from socr.core.state import PageState
from socr.core.verdict import DAMAGED_LAYER_PREFIX, PageVerdict, text_layer_numbers_clause
from socr.tables.reading_agreement import (
    LAYER_SILENT_COUNT_TEMPLATE,
    summarise_disputes_for_banner,
)

LAYER = "Korea\n-.8\n-.9\n4.0"
#: layer number -> the minus-dropped writing the reading ships for it (one contradiction each)
DROPPED = [("-.8", "-8"), ("-.9", "-9")]


def _reading(silent: int, contradictions: int) -> str:
    """One layer number kept, ``contradictions`` minus-dropped writings of layer numbers,
    ``silent`` numbers the layer lacks."""
    cells = (
        [read if i < contradictions else ref for i, (ref, read) in enumerate(DROPPED)]
        + ["4.0"]
        + [str(1000 + 7 * i) for i in range(silent)]
    )
    return "| Korea | " + " | ".join(cells) + " |"


def _page(damaged: bool = True) -> PageState:
    p = PageState(page_num=1)
    p.own_layer_text = LAYER
    p.is_born_digital = not damaged
    return p


def _out(text: str) -> PageOutput:
    return PageOutput(page_num=1, text=text, status=PageStatus.SUCCESS, engine="qwen")


def _banner(out: PageOutput) -> str:
    return out.text.splitlines()[0]


def _shipped(silent: int, contradictions: int, damaged: bool = True):
    return _apply_verdict_marks(_out(_reading(silent, contradictions)), _page(damaged), 1)


def test_banner_counts_silent_and_names_contradictions_sidecar_keeps_all() -> None:
    n = 5
    out, verdict = _shipped(n, 1)
    banner = _banner(out)
    assert LAYER_SILENT_COUNT_TEMPLATE.format(count=n) in banner
    assert "text layer says -.8, reading says -8" in banner
    assert "not in the text layer" not in banner
    assert "1000" not in banner
    clause = next(c for c in verdict.clauses if c.check is Check.TEXT_LAYER_NUMBERS)
    for i in range(n):
        assert f"reading says {1000 + 7 * i}, not in the text layer" in clause.reason
    assert "text layer says -.8, reading says -8" in clause.reason
    assert clause.reason.count(" / ") + 1 == n + 1


def test_only_silent_entries() -> None:
    out, verdict = _shipped(4, 0)
    banner = _banner(out)
    assert banner.endswith(DAMAGED_LAYER_PREFIX + LAYER_SILENT_COUNT_TEMPLATE.format(count=4) + "]")
    assert "text layer says" not in banner
    assert _is_marked(out.text)


def test_only_contradictions_is_unchanged() -> None:
    clause = text_layer_numbers_clause(_out(_reading(0, 2)), _page())
    assert clause is not None
    assert summarise_disputes_for_banner(clause.reason) == clause.reason
    assert "text layer says -.8, reading says -8" in clause.mark(1)
    assert "text layer says -.9, reading says -9" in clause.mark(1)
    assert "number(s) in the reading" not in clause.mark(1)


def test_failing_layer_case_counts_too() -> None:
    """The FAIL case (a clean born-digital layer) takes the same treatment."""
    out, _ = _shipped(3, 1, damaged=False)
    banner = _banner(out)
    assert "FAIL" in banner
    assert LAYER_SILENT_COUNT_TEMPLATE.format(count=3) in banner
    assert "not in the text layer" not in banner


def test_banner_is_independent_of_the_number_of_silent_entries() -> None:
    """No length threshold: the banner differs only in the count."""
    small = _banner(_shipped(2, 1)[0])
    large = _banner(_shipped(40, 1)[0])
    assert re.sub(r"\d+ number", "N number", small) == re.sub(r"\d+ number", "N number", large)


def test_refinalize_is_idempotent_and_matches_a_stored_old_format_reason() -> None:
    out, verdict = _shipped(6, 1)
    again, _ = _apply_verdict_marks(out, _page(), 1)
    assert again.text == out.text
    # a sidecar written before #1347 holds the full reason; rebuilt from it, the banner is the same
    stored = PageVerdict.from_dict(verdict.to_dict())
    assert stored.marks() == verdict.marks()
    assert _banner(out) == stored.marks()[0]
    assert _is_marked(out.text)


def test_record_form_is_lossless() -> None:
    clause = text_layer_numbers_clause(_out(_reading(3, 1)), _page())
    assert clause is not None
    assert clause.reason.startswith(DAMAGED_LAYER_PREFIX)
    assert clause.reason.count("not in the text layer") == 3


def test_unparseable_reason_is_returned_unchanged() -> None:
    for reason in ("", "no number to check", "reading says 1, not in the text layer / free text"):
        assert summarise_disputes_for_banner(reason) == reason
