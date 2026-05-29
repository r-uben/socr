"""TICKET-16: judge + benchmark harness tests (no live model required).

Proves the scoring math and the parsing/IO plumbing with a deterministic stub
judge. The real lite-model judge gets measured against a labeled corpus later;
this guarantees the harness reports FP/FN correctly when it does.
"""

from __future__ import annotations

import json
from pathlib import Path

from socr.judge.benchmark import (
    BenchmarkReport,
    LabeledPage,
    load_dataset,
    run_benchmark,
)
from socr.judge.judge import JudgeVerdict, load_judge_prompt, parse_verdict

# --------------------------------------------------------------------------
# prompt (policy as data)
# --------------------------------------------------------------------------


def test_judge_prompt_loads_and_is_policy():
    prompt = load_judge_prompt()
    assert "faithful" in prompt
    assert "suggested_action" in prompt
    # No baked-in numeric thresholds — the model reasons about the page.
    assert "do not apply fixed numeric cutoffs" in prompt.lower()


# --------------------------------------------------------------------------
# verdict parsing — tolerate the usual model noise
# --------------------------------------------------------------------------


def test_parse_plain_json():
    v = parse_verdict('{"faithful": true, "issues": [], "confidence": 0.9, '
                      '"suggested_action": "accept"}')
    assert v.faithful and v.is_good
    assert v.confidence == 0.9
    assert v.suggested_action == "accept"


def test_parse_fenced_json_with_prose():
    raw = 'Here is my verdict:\n```json\n{"faithful": false, ' \
          '"issues": ["table garbled"], "confidence": 0.8, ' \
          '"suggested_action": "escalate_engine"}\n```\nDone.'
    v = parse_verdict(raw)
    assert not v.faithful
    assert v.issues == ["table garbled"]
    assert v.suggested_action == "escalate_engine"


def test_parse_coerces_bad_action_and_clamps_confidence():
    v = parse_verdict('{"faithful": false, "confidence": 5.0, '
                      '"suggested_action": "nonsense"}')
    assert v.confidence == 1.0  # clamped to [0,1]
    assert v.suggested_action == "escalate_engine"  # not-faithful default


def test_parse_missing_fields_defaults_safely():
    v = parse_verdict('{"faithful": true}')
    assert v.faithful
    assert v.issues == []
    assert v.suggested_action == "accept"


# --------------------------------------------------------------------------
# dataset loading
# --------------------------------------------------------------------------


def _write_dataset(root: Path, entries):
    (root / "ocr").mkdir(parents=True, exist_ok=True)
    (root / "images").mkdir(parents=True, exist_ok=True)
    labels = []
    for e in entries:
        ocr_rel = f"ocr/{e['page_id']}.md"
        (root / ocr_rel).write_text(e["ocr_text"], encoding="utf-8")
        (root / "images" / f"{e['page_id']}.png").write_bytes(b"\x89PNG fake")
        labels.append(
            {
                "page_id": e["page_id"],
                "image": f"images/{e['page_id']}.png",
                "ocr": ocr_rel,
                "label": e["label"],
            }
        )
    (root / "labels.json").write_text(json.dumps(labels), encoding="utf-8")


def test_load_dataset(tmp_path):
    _write_dataset(
        tmp_path,
        [
            {"page_id": "p1", "ocr_text": "good text", "label": "good"},
            {"page_id": "p2", "ocr_text": "garbage", "label": "mangled"},
        ],
    )
    pages = load_dataset(tmp_path)
    assert len(pages) == 2
    by_id = {p.page_id: p for p in pages}
    assert by_id["p1"].ocr_text == "good text"
    assert by_id["p1"].needs_repair is False
    assert by_id["p2"].needs_repair is True


# --------------------------------------------------------------------------
# scoring math — the headline FP/FN numbers
# --------------------------------------------------------------------------


class _StubJudge:
    """Judge that calls a page faithful iff 'good' is in the OCR text.

    Lets us construct a known confusion matrix without a model.
    """

    def judge(self, image_path: Path, ocr_text: str) -> JudgeVerdict:
        faithful = "good" in ocr_text
        return JudgeVerdict(faithful=faithful, confidence=1.0,
                            suggested_action="accept" if faithful else "escalate_engine")


def test_benchmark_confusion_matrix():
    # Construct each cell deliberately:
    #   TP: mangled + judge-says-bad  -> ocr without 'good', label mangled
    #   FN: mangled + judge-says-good -> ocr with 'good',    label mangled
    #   FP: good    + judge-says-bad  -> ocr without 'good', label good
    #   TN: good    + judge-says-good -> ocr with 'good',    label good
    dataset = [
        LabeledPage("tp", Path("x"), "garbled junk", "mangled"),
        LabeledPage("fn", Path("x"), "good but actually mangled", "mangled"),
        LabeledPage("fp", Path("x"), "junk", "good"),
        LabeledPage("tn", Path("x"), "good clean text", "good"),
    ]
    report = run_benchmark(_StubJudge(), dataset)
    assert (report.tp, report.fn, report.fp, report.tn) == (1, 1, 1, 1)
    assert report.accuracy == 0.5
    assert report.precision == 0.5
    assert report.recall == 0.5
    assert report.false_negative_rate == 0.5  # corpus poisoning
    assert report.false_positive_rate == 0.5  # budget burning
    assert "corpus poisoning" in report.summary()


def test_perfect_judge_zero_error():
    dataset = [
        LabeledPage("a", Path("x"), "good text", "good"),
        LabeledPage("b", Path("x"), "garbage", "mangled"),
    ]
    report = run_benchmark(_StubJudge(), dataset)
    assert report.false_negative_rate == 0.0
    assert report.false_positive_rate == 0.0
    assert report.accuracy == 1.0


def test_empty_report_no_divide_by_zero():
    report = BenchmarkReport()
    assert report.precision == 0.0
    assert report.recall == 0.0
    assert report.accuracy == 0.0
    assert report.false_negative_rate == 0.0
