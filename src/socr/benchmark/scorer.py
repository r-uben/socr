"""Benchmark scoring: WER, CER, and NES for OCR quality evaluation.

Compares OCR output against ground truth using:
  - WER: Word Error Rate (edit_distance / ref_words)
  - CER: Character Error Rate (edit_distance / ref_chars)
  - NES: Normalized Edit Similarity (1 - edit_distance / max(len_pred, len_gt))
    More robust than WER for OCR evaluation (per socOCRbench).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from socr.core.result import EngineResult


@dataclass
class PageScore:
    """Score for a single page."""

    page_num: int
    word_error_rate: float  # WER
    character_error_rate: float  # CER
    normalized_edit_similarity: float  # NES (0-1, higher is better)
    word_count_ratio: float  # predicted/actual word count


@dataclass
class DocumentScore:
    """Aggregate score for a full document."""

    paper_name: str
    engine: str
    pages: list[PageScore] = field(default_factory=list)
    overall_wer: float = 0.0
    overall_cer: float = 0.0
    overall_nes: float = 0.0  # Normalized Edit Similarity
    processing_time: float = 0.0


def _levenshtein(seq_a: list[str], seq_b: list[str]) -> int:
    """Compute Levenshtein edit distance between two sequences.

    Uses the standard dynamic programming approach with O(min(m,n)) space.

    Args:
        seq_a: Reference sequence.
        seq_b: Hypothesis sequence.

    Returns:
        Edit distance (insertions + deletions + substitutions).
    """
    m, n = len(seq_a), len(seq_b)

    # Optimize by making the shorter sequence the column dimension
    if m < n:
        seq_a, seq_b = seq_b, seq_a
        m, n = n, m

    # Single-row DP: previous row
    prev = list(range(n + 1))
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            curr[j] = min(
                curr[j - 1] + 1,  # insertion
                prev[j] + 1,  # deletion
                prev[j - 1] + cost,  # substitution
            )
        prev, curr = curr, prev

    return prev[n]


def _levenshtein_chars(a: str, b: str) -> int:
    """Levenshtein distance on character sequences."""
    return _levenshtein(list(a), list(b))


class BenchmarkScorer:
    """Score OCR output against ground truth text."""

    def score(self, predicted: str, ground_truth: str) -> float:
        """Compute Word Error Rate between predicted and ground truth text.

        WER = edit_distance(ref_words, hyp_words) / len(ref_words)

        Args:
            predicted: OCR output text.
            ground_truth: Reference text.

        Returns:
            WER as a float (0.0 = perfect, 1.0 = all words wrong,
            >1.0 possible if insertions exceed reference length).
        """
        ref_words = ground_truth.split()
        hyp_words = predicted.split()

        if not ref_words:
            return 0.0 if not hyp_words else 1.0

        distance = _levenshtein(ref_words, hyp_words)
        return distance / len(ref_words)

    def score_cer(self, predicted: str, ground_truth: str) -> float:
        """Compute Character Error Rate.

        CER = edit_distance(ref_chars, hyp_chars) / len(ref_chars)

        Args:
            predicted: OCR output text.
            ground_truth: Reference text.

        Returns:
            CER as a float.
        """
        if not ground_truth:
            return 0.0 if not predicted else 1.0

        distance = _levenshtein_chars(ground_truth, predicted)
        return distance / len(ground_truth)

    def score_nes(self, predicted: str, ground_truth: str) -> float:
        """Compute Normalized Edit Similarity.

        NES = 1 - edit_distance(pred, gt) / max(len(pred), len(gt))

        Range: 0-1 (1.0 = perfect match). More robust than WER for OCR
        because it normalizes by the longer string (not just reference),
        avoiding >1.0 scores and handling insertions fairly.

        Per socOCRbench (Dasanaike 2026).
        """
        if not ground_truth and not predicted:
            return 1.0
        if not ground_truth or not predicted:
            return 0.0

        distance = _levenshtein_chars(predicted, ground_truth)
        max_len = max(len(predicted), len(ground_truth))
        return 1.0 - (distance / max_len)

    def score_page(
        self,
        predicted: str,
        ground_truth: str,
        page_num: int,
    ) -> PageScore:
        """Score a single page.

        Args:
            predicted: OCR output for this page.
            ground_truth: Reference text for this page.
            page_num: Page number (1-indexed).

        Returns:
            PageScore with WER, CER, NES, and word count ratio.
        """
        wer = self.score(predicted, ground_truth)
        cer = self.score_cer(predicted, ground_truth)
        nes = self.score_nes(predicted, ground_truth)

        ref_wc = len(ground_truth.split()) if ground_truth else 0
        hyp_wc = len(predicted.split()) if predicted else 0
        wc_ratio = hyp_wc / ref_wc if ref_wc > 0 else (0.0 if not hyp_wc else float("inf"))

        return PageScore(
            page_num=page_num,
            word_error_rate=wer,
            character_error_rate=cer,
            normalized_edit_similarity=nes,
            word_count_ratio=wc_ratio,
        )

    def score_document(
        self,
        result: EngineResult,
        ground_truth_dir: Path,
    ) -> DocumentScore:
        """Score a full document against per-page ground truth files.

        Expects ground truth files at:
            ground_truth_dir/page_1.txt
            ground_truth_dir/page_2.txt
            ...

        Args:
            result: EngineResult from an OCR engine.
            ground_truth_dir: Directory with per-page ground truth text files.

        Returns:
            DocumentScore with per-page and overall metrics.
        """
        page_scores: list[PageScore] = []

        # Collect all ground truth text and predicted text for overall scoring
        all_gt_words: list[str] = []
        all_pred_words: list[str] = []
        all_gt_chars: list[str] = []
        all_pred_chars: list[str] = []

        for page_output in result.pages:
            page_num = page_output.page_num
            gt_file = ground_truth_dir / f"page_{page_num}.txt"

            if not gt_file.exists():
                continue

            gt_text = gt_file.read_text(encoding="utf-8").strip()
            pred_text = page_output.text.strip() if page_output.text else ""

            page_score = self.score_page(pred_text, gt_text, page_num)
            page_scores.append(page_score)

            all_gt_words.extend(gt_text.split())
            all_pred_words.extend(pred_text.split())
            all_gt_chars.extend(list(gt_text))
            all_pred_chars.extend(list(pred_text))

        # Overall WER, CER, and NES across all pages
        overall_wer = 0.0
        overall_cer = 0.0
        overall_nes = 0.0
        if all_gt_words:
            overall_wer = _levenshtein(all_gt_words, all_pred_words) / len(all_gt_words)
        if all_gt_chars:
            dist = _levenshtein(all_gt_chars, all_pred_chars)
            overall_cer = dist / len(all_gt_chars)
            max_len = max(len(all_gt_chars), len(all_pred_chars))
            overall_nes = 1.0 - (dist / max_len) if max_len > 0 else 1.0

        return DocumentScore(
            paper_name=result.document_path.stem,
            engine=result.engine,
            pages=page_scores,
            overall_wer=overall_wer,
            overall_cer=overall_cer,
            overall_nes=overall_nes,
            processing_time=result.processing_time,
        )
