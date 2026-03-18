"""Benchmark runner: execute OCR engines on benchmark papers and collect scores.

Runs each engine on each paper, scores against ground truth, and serializes
results for downstream calibration and analysis.
"""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from socr.benchmark.dataset import BenchmarkPaper, BenchmarkSet
from socr.benchmark.scorer import BenchmarkScorer, DocumentScore
from socr.core.config import EngineType, PipelineConfig
from socr.core.result import EngineResult
from socr.engines.registry import get_engine

logger = logging.getLogger(__name__)


@dataclass
class EngineRun:
    """Result of running a single engine on a single paper."""

    paper_name: str
    engine: str
    result: EngineResult
    score: DocumentScore | None = None


@dataclass
class BenchmarkResults:
    """Collection of engine runs with serialization."""

    runs: list[EngineRun] = field(default_factory=list)
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def by_engine(self) -> dict[str, list[EngineRun]]:
        """Group runs by engine name."""
        groups: dict[str, list[EngineRun]] = {}
        for run in self.runs:
            groups.setdefault(run.engine, []).append(run)
        return groups

    def by_paper(self) -> dict[str, list[EngineRun]]:
        """Group runs by paper name."""
        groups: dict[str, list[EngineRun]] = {}
        for run in self.runs:
            groups.setdefault(run.paper_name, []).append(run)
        return groups

    def save(self, path: Path) -> None:
        """Save results as JSON (scores + metadata, without full EngineResult text)."""
        data = {
            "timestamp": self.timestamp,
            "runs": [
                {
                    "paper_name": run.paper_name,
                    "engine": run.engine,
                    "success": run.result.success,
                    "failure_mode": run.result.failure_mode.value,
                    "processing_time": run.result.processing_time,
                    "word_count": run.result.word_count,
                    "score": _score_to_dict(run.score) if run.score else None,
                }
                for run in self.runs
            ],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> BenchmarkResults:
        """Load results from JSON.

        Reconstructs EngineRun with minimal EngineResult stubs (no page text).
        """
        from socr.core.result import DocumentStatus, FailureMode

        data = json.loads(path.read_text())
        runs: list[EngineRun] = []

        for entry in data["runs"]:
            status = DocumentStatus.SUCCESS if entry["success"] else DocumentStatus.ERROR
            failure_mode = FailureMode(entry.get("failure_mode", "none"))

            result = EngineResult(
                document_path=Path(entry["paper_name"]),
                engine=entry["engine"],
                status=status,
                failure_mode=failure_mode,
                processing_time=entry.get("processing_time", 0.0),
            )

            score = _dict_to_score(entry["score"]) if entry.get("score") else None

            runs.append(
                EngineRun(
                    paper_name=entry["paper_name"],
                    engine=entry["engine"],
                    result=result,
                    score=score,
                )
            )

        return cls(runs=runs, timestamp=data["timestamp"])


class BenchmarkRunner:
    """Run OCR engines on benchmark papers and score results."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.scorer = BenchmarkScorer()

    def run(
        self,
        benchmark: BenchmarkSet,
        output_dir: Path,
        engines: list[EngineType] | None = None,
    ) -> BenchmarkResults:
        """Run each engine on each paper, score against ground truth.

        Args:
            benchmark: The benchmark paper set.
            output_dir: Directory for intermediate engine outputs.
            engines: Engines to run (default: all CLI-available engines from registry).

        Returns:
            BenchmarkResults with all runs and scores.
        """
        engine_types = engines or _available_engines()
        results = BenchmarkResults()

        for paper in benchmark.papers:
            for engine_type in engine_types:
                engine_run = self.run_single(paper, engine_type, output_dir)
                results.runs.append(engine_run)

        return results

    def run_single(
        self,
        paper: BenchmarkPaper,
        engine_type: EngineType,
        output_dir: Path,
    ) -> EngineRun:
        """Run a single engine on a single paper.

        Args:
            paper: Benchmark paper to process.
            engine_type: Engine to use.
            output_dir: Directory for engine output.

        Returns:
            EngineRun with result and optional score.
        """
        engine = get_engine(engine_type)
        engine_name = engine_type.value

        if not engine.is_available():
            logger.warning("Engine %s is not available, skipping %s", engine_name, paper.name)
            result = EngineResult(
                document_path=paper.pdf_path,
                engine=engine_name,
                status=_unavailable_status(),
                failure_mode=_unavailable_failure(),
                error=f"Engine {engine_name} not available",
            )
            return EngineRun(paper_name=paper.name, engine=engine_name, result=result)

        logger.info("Running %s on %s (%d pages)", engine_name, paper.name, paper.page_count)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_out = Path(tmpdir)
            result = engine.process_document(paper.pdf_path, tmp_out, self.config)

        score: DocumentScore | None = None
        if result.success and paper.ground_truth_path and paper.ground_truth_path.exists():
            score = self.scorer.score_document(result, paper.ground_truth_path)
            score.paper_name = paper.name
            score.engine = engine_name

        return EngineRun(
            paper_name=paper.name,
            engine=engine_name,
            result=result,
            score=score,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _available_engines() -> list[EngineType]:
    """Return CLI engine types that are available on this machine."""
    from socr.engines.registry import _ENGINES

    available: list[EngineType] = []
    for engine_type in _ENGINES:
        try:
            engine = get_engine(engine_type)
            if engine.is_available():
                available.append(engine_type)
        except Exception:
            continue
    return available


def _unavailable_status():
    from socr.core.result import DocumentStatus
    return DocumentStatus.ERROR


def _unavailable_failure():
    from socr.core.result import FailureMode
    return FailureMode.MODEL_UNAVAILABLE


def _score_to_dict(score: DocumentScore) -> dict:
    """Serialize a DocumentScore to a JSON-safe dict."""
    return {
        "paper_name": score.paper_name,
        "engine": score.engine,
        "overall_wer": score.overall_wer,
        "overall_cer": score.overall_cer,
        "processing_time": score.processing_time,
        "pages": [
            {
                "page_num": p.page_num,
                "word_error_rate": p.word_error_rate,
                "character_error_rate": p.character_error_rate,
                "word_count_ratio": p.word_count_ratio,
            }
            for p in score.pages
        ],
    }


def _dict_to_score(d: dict) -> DocumentScore:
    """Deserialize a DocumentScore from a dict."""
    from socr.benchmark.scorer import PageScore

    pages = [
        PageScore(
            page_num=p["page_num"],
            word_error_rate=p["word_error_rate"],
            character_error_rate=p["character_error_rate"],
            word_count_ratio=p["word_count_ratio"],
        )
        for p in d.get("pages", [])
    ]
    return DocumentScore(
        paper_name=d["paper_name"],
        engine=d["engine"],
        pages=pages,
        overall_wer=d["overall_wer"],
        overall_cer=d["overall_cer"],
        processing_time=d.get("processing_time", 0.0),
    )
