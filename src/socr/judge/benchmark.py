"""Judge benchmark: measure a judge against labeled OCR pages.

The judge is the highest-risk component (both panel reviewers said so). Before
trusting it inside a repair loop, we measure it on a ground-truth set:

  - **False negative** (label=mangled, judge says faithful): garbage enters the
    corpus. This is the dangerous error.
  - **False positive** (label=good, judge says not faithful): we burn budget
    re-OCRing a perfect page. This is the wasteful error.

Positive class = "needs repair" (i.e. NOT faithful). The dataset is loaded from
disk (study design lives in code; the labeled data does not) so the same harness
runs on any labeled set.

Dataset layout::

    <dataset_dir>/labels.json   # [{"page_id","image","ocr","label"}, ...]
    <dataset_dir>/images/...    # rendered page PNGs referenced by "image"
    <dataset_dir>/ocr/...       # candidate transcriptions referenced by "ocr"

``label`` is "good" or "mangled".
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from socr.judge.judge import Judge, JudgeVerdict

GOOD = "good"
MANGLED = "mangled"
VALID_LABELS = {GOOD, MANGLED}


@dataclass
class LabeledPage:
    page_id: str
    image_path: Path
    ocr_text: str
    label: str  # "good" | "mangled"

    @property
    def needs_repair(self) -> bool:
        """Ground truth: a mangled page is the positive (needs-repair) class."""
        return self.label == MANGLED


def load_dataset(dataset_dir: Path | str) -> list[LabeledPage]:
    """Load labeled pages from ``<dataset_dir>/labels.json``."""
    dataset_dir = Path(dataset_dir)
    labels_file = dataset_dir / "labels.json"
    entries = json.loads(labels_file.read_text(encoding="utf-8"))
    pages: list[LabeledPage] = []
    for e in entries:
        label = e["label"]
        if label not in VALID_LABELS:
            raise ValueError(f"page {e.get('page_id')!r}: bad label {label!r}")
        ocr_path = dataset_dir / e["ocr"]
        pages.append(
            LabeledPage(
                page_id=str(e["page_id"]),
                image_path=dataset_dir / e["image"],
                ocr_text=ocr_path.read_text(encoding="utf-8"),
                label=label,
            )
        )
    return pages


@dataclass
class PageOutcome:
    page_id: str
    label: str
    predicted_needs_repair: bool
    verdict: JudgeVerdict


@dataclass
class BenchmarkReport:
    """Confusion matrix + the two error rates that actually matter."""

    tp: int = 0  # mangled, correctly flagged (not faithful)
    fp: int = 0  # good, wrongly flagged   -> burns budget
    tn: int = 0  # good, correctly passed
    fn: int = 0  # mangled, wrongly passed -> poisons corpus
    outcomes: list[PageOutcome] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def false_negative_rate(self) -> float:
        """Mangled pages let through — the corpus-poisoning rate."""
        denom = self.fn + self.tp
        return self.fn / denom if denom else 0.0

    @property
    def false_positive_rate(self) -> float:
        """Good pages needlessly re-OCR'd — the budget-burning rate."""
        denom = self.fp + self.tn
        return self.fp / denom if denom else 0.0

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.total if self.total else 0.0

    def summary(self) -> str:
        return (
            f"n={self.total}  acc={self.accuracy:.2f}  "
            f"precision={self.precision:.2f}  recall={self.recall:.2f}\n"
            f"  FN rate (corpus poisoning) = {self.false_negative_rate:.2f} "
            f"({self.fn} mangled pages passed as good)\n"
            f"  FP rate (budget burning)   = {self.false_positive_rate:.2f} "
            f"({self.fp} good pages flagged for re-OCR)"
        )


def run_benchmark(judge: Judge, dataset: list[LabeledPage]) -> BenchmarkReport:
    """Run the judge over a labeled set and score it."""
    report = BenchmarkReport()
    for page in dataset:
        verdict = judge.judge(page.image_path, page.ocr_text)
        predicted_repair = not verdict.faithful
        if page.needs_repair and predicted_repair:
            report.tp += 1
        elif page.needs_repair and not predicted_repair:
            report.fn += 1
        elif not page.needs_repair and predicted_repair:
            report.fp += 1
        else:
            report.tn += 1
        report.outcomes.append(
            PageOutcome(
                page_id=page.page_id,
                label=page.label,
                predicted_needs_repair=predicted_repair,
                verdict=verdict,
            )
        )
    return report
