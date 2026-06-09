"""Benchmark scoring: WER, CER, NES, and table-cell fidelity.

Compares OCR output against ground truth using:
  - WER: Word Error Rate (edit_distance / ref_words)
  - CER: Character Error Rate (edit_distance / ref_chars)
  - NES: Normalized Edit Similarity (1 - edit_distance / max(len_pred, len_gt))
    More robust than WER for OCR evaluation (per socOCRbench).
  - Table cells: structure + numeric-cell EXACTNESS against hand-verified
    grid GT (``page_N.table.md``) — a pretty table with wrong digits is
    unacceptable in a research corpus.

Coverage is a HARD GATE: every expected page must be scored; a page the
engine did not cover scores as a total failure, never as a silent skip.
(The historical scorer skipped unmatched pages, so a page_num=0 whole-doc
output matched nothing and every engine scored a perfect 0.0 WER.)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from socr.core.result import EngineResult

logger = logging.getLogger(__name__)


@dataclass
class PageScore:
    """Score for a single page."""

    page_num: int
    word_error_rate: float  # WER
    character_error_rate: float  # CER
    normalized_edit_similarity: float  # NES (0-1, higher is better)
    word_count_ratio: float  # predicted/actual word count
    page_type: str = ""  # benchmark page-type taxonomy (page_types.py)
    covered: bool = True  # engine produced output for this page
    table_cell_accuracy: float | None = None  # numeric-cell exactness vs grid GT
    table_structure_ok: bool | None = None  # row/col shape matches grid GT


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
    expected_pages: int = 0
    pages_missing: list[int] = field(default_factory=list)  # expected but not covered

    @property
    def coverage(self) -> float:
        """Fraction of SCORED pages the engine actually covered.

        Denominated in scored pages (the GT set), not the PDF's page count:
        when hand-verified GT covers a subset of pages, a run that produced
        nothing must score coverage 0.0 — not 1 - few/many (issue #39
        review). With no scored pages at all, coverage is 0 when pages are
        known missing (e.g. empty GT dir against a real page count).
        """
        if self.pages:
            return sum(1 for p in self.pages if p.covered) / len(self.pages)
        return 0.0 if self.pages_missing else 1.0

    def by_page_type(self) -> dict[str, dict[str, float]]:
        """Macro-averaged metrics per page type.

        The corpus is prose-dominant, so a single micro-average drowns the
        table/equation pages where engine quality actually differs; macro
        aggregation by type is what calibration consumes.
        """
        groups: dict[str, list[PageScore]] = {}
        for p in self.pages:
            groups.setdefault(p.page_type or "untyped", []).append(p)
        out: dict[str, dict[str, float]] = {}
        for ptype, scores in groups.items():
            n = len(scores)
            out[ptype] = {
                "pages": float(n),
                "wer": sum(s.word_error_rate for s in scores) / n,
                "cer": sum(s.character_error_rate for s in scores) / n,
                "nes": sum(s.normalized_edit_similarity for s in scores) / n,
                "coverage": sum(1.0 for s in scores if s.covered) / n,
            }
            cell_scores = [
                s.table_cell_accuracy for s in scores if s.table_cell_accuracy is not None
            ]
            if cell_scores:
                out[ptype]["table_cell_accuracy"] = sum(cell_scores) / len(cell_scores)
        return out


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
        expected_pages: int = 0,
        page_types: dict[int, str] | None = None,
    ) -> DocumentScore:
        """Score a full document against per-page ground truth files.

        Expects ground truth files at:
            ground_truth_dir/page_1.txt          (text GT, one per page)
            ground_truth_dir/page_N.table.md     (optional hand-verified grid
                                                  GT for table pages)

        COVERAGE IS A HARD GATE. Every page with ground truth must be scored:
        a page the engine produced no output for scores WER=1, CER=1, NES=0
        and is listed in ``pages_missing``. A single whole-document
        ``page_num=0`` output is split into pages via the contract's
        ``## Page N`` markers first; a markerless blob covers page 1 only and
        every other page counts as missing (loudly logged).

        Args:
            result: EngineResult from an OCR engine.
            ground_truth_dir: Directory with per-page ground truth text files.
            expected_pages: Real page count of the source PDF. 0 = derive
                from the ground-truth files present.
            page_types: Optional page_num -> page-type map (page_types.py),
                recorded on each PageScore for macro aggregation.

        Returns:
            DocumentScore with per-page and overall metrics.
        """
        page_types = page_types or {}
        predictions = self._per_page_predictions(result, expected_pages)

        gt_pages = sorted(
            int(m.group(1))
            for f in ground_truth_dir.glob("page_*.txt")
            if (m := re.fullmatch(r"page_(\d+)\.txt", f.name))
        )
        if not gt_pages:
            # No text GT at all: this must be a zero score, never a vacuous
            # perfect one (the 0.0-WER catastrophe's Stage-2 corner: a GT dir
            # that exists but has no page_N.txt yet).
            logger.error(
                "benchmark: %s contains no page_N.txt ground truth — scoring zero",
                ground_truth_dir,
            )
            return DocumentScore(
                paper_name=result.document_path.stem,
                engine=result.engine,
                overall_wer=1.0,
                overall_cer=1.0,
                overall_nes=0.0,
                processing_time=result.processing_time,
                expected_pages=expected_pages,
                pages_missing=list(range(1, expected_pages + 1)),
            )
        if expected_pages and len(gt_pages) != expected_pages:
            logger.warning(
                "benchmark: %s has GT for %d page(s) but the PDF has %d",
                ground_truth_dir,
                len(gt_pages),
                expected_pages,
            )

        page_scores: list[PageScore] = []
        pages_missing: list[int] = []

        # Collect all ground truth text and predicted text for overall scoring
        all_gt_words: list[str] = []
        all_pred_words: list[str] = []
        all_gt_chars: list[str] = []
        all_pred_chars: list[str] = []

        for page_num in gt_pages:
            gt_text = (ground_truth_dir / f"page_{page_num}.txt").read_text(
                encoding="utf-8"
            ).strip()
            pred_text = (predictions.get(page_num) or "").strip()
            # A legitimately blank GT page is covered by a (correctly) empty
            # prediction — an honest engine must not be punished for it.
            covered = bool(pred_text) or not gt_text

            page_score = self.score_page(pred_text, gt_text, page_num)
            page_score.page_type = page_types.get(page_num, "")
            page_score.covered = covered
            if not covered:
                # The hard gate: an uncovered page is a total failure, never
                # a silent skip (score_page already yields WER>=1/NES=0 for
                # empty predictions against non-empty GT).
                pages_missing.append(page_num)

            # Hand-verified grid GT, when present, scores table fidelity.
            table_gt = ground_truth_dir / f"page_{page_num}.table.md"
            if table_gt.exists():
                accuracy, structure_ok = self.score_table_cells(
                    pred_text, table_gt.read_text(encoding="utf-8")
                )
                page_score.table_cell_accuracy = accuracy
                page_score.table_structure_ok = structure_ok

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
            expected_pages=expected_pages or len(gt_pages),
            pages_missing=pages_missing,
        )

    @staticmethod
    def _per_page_predictions(result: EngineResult, expected_pages: int) -> dict[int, str]:
        """page_num -> predicted text, resolving whole-document outputs.

        A single ``page_num=0`` output (document-level CLI engines) is split
        on the contract's ``## Page N`` markers, keyed by the marker's OWN
        page number — positional renumbering would misattribute every page
        after a skipped one, scoring page 3's correct text against page 2's
        GT (issue #39 review, critical). A markerless blob maps to page 1
        only; the coverage gate then counts pages 2..N as missing.
        """
        pages = [p for p in result.pages if p.page_num > 0]
        if pages:
            return {p.page_num: p.text or "" for p in pages}

        whole = next((p for p in result.pages if p.page_num == 0), None)
        if whole is None or not (whole.text or "").strip():
            return {}

        from ocr_output_contract import PAGE_MARKER_RE

        text = whole.text
        matches = list(PAGE_MARKER_RE.finditer(text))
        if not matches:
            return {1: text}

        predictions: dict[int, str] = {}
        for idx, m in enumerate(matches):
            n = int(m.group(1))
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            body = text[m.end() : end].strip()
            if idx == 0:
                # Preamble before the first marker belongs to that page.
                preamble = text[: m.start()].strip()
                if preamble:
                    body = f"{preamble}\n\n{body}".strip()
            if n in predictions:
                logger.warning(
                    "benchmark: duplicate '## Page %d' marker in %s output; keeping first",
                    n,
                    result.document_path.name,
                )
                continue
            predictions[n] = body

        if expected_pages and len(predictions) != expected_pages:
            logger.warning(
                "benchmark: whole-doc output for %s covers %d page(s), "
                "expected %d — uncovered pages score as failures",
                result.document_path.name,
                len(predictions),
                expected_pages,
            )
        return predictions

    # ------------------------------------------------------------------
    # Table fidelity (vs hand-verified grid GT)
    # ------------------------------------------------------------------

    # Sign may be ASCII hyphen, Unicode minus (U+2212), or en-dash (U+2013) —
    # GT typed by hand and engine output legitimately differ here, so cells
    # are normalized before detection AND comparison. Stars/percent/dagger
    # significance markers and currency prefixes stay part of the cell.
    _NUMERIC_CELL_RE = re.compile(r"[-+(]?\s*[$€£]?\s*\d[\d,]*(?:\.\d+)?\s*[)%*†]*")
    # An escaped pipe (``a\|b``, the form core/html_tables emits inside
    # cells) is content, not a column separator.
    _CELL_SPLIT_RE = re.compile(r"(?<!\\)\|")

    @classmethod
    def _norm_cell(cls, cell: str) -> str:
        return (
            cell.replace("−", "-")
            .replace("–", "-")
            .replace("\\|", "|")
            .strip()
        )

    @classmethod
    def _markdown_table_cells(cls, text: str) -> list[list[str]]:
        """Cell grid from the markdown pipe-table rows in *text*."""
        rows: list[list[str]] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped.startswith("|"):
                continue
            inner = stripped.strip("|")
            # Skip separator rows (|---|---|)
            if set(inner.replace("|", "").strip()) <= set("-: "):
                continue
            rows.append([cls._norm_cell(c) for c in cls._CELL_SPLIT_RE.split(inner)])
        return rows

    @classmethod
    def _is_numeric_cell(cls, cell: str) -> bool:
        return bool(cls._NUMERIC_CELL_RE.fullmatch(cell.strip()))

    @classmethod
    def score_table_cells(cls, predicted: str, gt_table_md: str) -> tuple[float, bool]:
        """(numeric-cell exactness, structure_ok) against a grid GT.

        When the predicted grid has the same shape as the GT, numeric cells
        are compared position-by-position (exact string match after
        whitespace normalization). When shapes differ, structure is failed
        and exactness degrades to multiset recall of the GT's numeric values
        in the prediction — content-presence credit without structure credit.
        """
        gt_rows = cls._markdown_table_cells(gt_table_md)
        pred_rows = cls._markdown_table_cells(predicted)
        gt_numeric = [
            (r, c, cell.strip())
            for r, row in enumerate(gt_rows)
            for c, cell in enumerate(row)
            if cls._is_numeric_cell(cell)
        ]
        if not gt_numeric:
            return 1.0, bool(gt_rows) == bool(pred_rows)

        structure_ok = len(pred_rows) == len(gt_rows) and all(
            len(p) == len(g) for p, g in zip(pred_rows, gt_rows)
        )
        if structure_ok:
            hits = sum(
                1
                for r, c, value in gt_numeric
                if pred_rows[r][c].strip() == value
            )
            return hits / len(gt_numeric), True

        pred_values: list[str] = [
            cell.strip()
            for row in pred_rows
            for cell in row
            if cls._is_numeric_cell(cell)
        ]
        pool = list(pred_values)
        hits = 0
        for _, _, value in gt_numeric:
            if value in pool:
                pool.remove(value)
                hits += 1
        return hits / len(gt_numeric), False
