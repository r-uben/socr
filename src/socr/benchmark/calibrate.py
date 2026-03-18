"""Calibrate repair routing from benchmark data.

Analyzes benchmark results to determine optimal engine chains per category
and failure-mode recovery rates. Produces a CalibrationReport that can be
applied to a PipelineConfig.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from socr.benchmark.runner import BenchmarkResults, EngineRun
from socr.core.config import EngineType, PipelineConfig


@dataclass
class EngineProfile:
    """Performance profile for an engine across benchmark categories."""

    engine: str
    category_wer: dict[str, float] = field(default_factory=dict)  # category -> average WER
    failure_mode_recovery: dict[str, float] = field(
        default_factory=dict
    )  # failure_mode -> recovery rate (0-1)
    avg_processing_time: float = 0.0


@dataclass
class CalibrationReport:
    """Output of the calibration: profiles and recommended chains."""

    profiles: list[EngineProfile] = field(default_factory=list)
    recommended_chain: dict[str, list[str]] = field(
        default_factory=dict
    )  # category -> ordered engine list

    def save(self, path: Path) -> None:
        """Save calibration report as JSON."""
        data = {
            "profiles": [
                {
                    "engine": p.engine,
                    "category_wer": p.category_wer,
                    "failure_mode_recovery": p.failure_mode_recovery,
                    "avg_processing_time": p.avg_processing_time,
                }
                for p in self.profiles
            ],
            "recommended_chain": self.recommended_chain,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> CalibrationReport:
        """Load calibration report from JSON."""
        data = json.loads(path.read_text())
        profiles = [
            EngineProfile(
                engine=p["engine"],
                category_wer=p.get("category_wer", {}),
                failure_mode_recovery=p.get("failure_mode_recovery", {}),
                avg_processing_time=p.get("avg_processing_time", 0.0),
            )
            for p in data.get("profiles", [])
        ]
        return cls(
            profiles=profiles,
            recommended_chain=data.get("recommended_chain", {}),
        )


class RepairCalibrator:
    """Analyze benchmark results to determine optimal repair routing."""

    def calibrate(self, results: BenchmarkResults) -> CalibrationReport:
        """Analyze benchmark results to determine optimal repair routing.

        Steps:
            1. For each engine, compute average WER per category.
            2. For each failure mode, determine which engine best recovers
               (lowest WER on papers where other engines failed).
            3. Generate recommended fallback chains per category.
            4. Output a CalibrationReport.

        Args:
            results: Benchmark results from BenchmarkRunner.

        Returns:
            CalibrationReport with profiles and recommended chains.
        """
        by_engine = results.by_engine()

        # Build per-engine profiles
        profiles: list[EngineProfile] = []
        for engine_name, runs in sorted(by_engine.items()):
            profile = self._build_profile(engine_name, runs)
            profiles.append(profile)

        # Compute per-category recommended chains from category WER
        categories = self._collect_categories(results)
        recommended: dict[str, list[str]] = {}
        for category in sorted(categories):
            chain = self._rank_engines_for_category(category, profiles)
            recommended[category] = chain

        return CalibrationReport(profiles=profiles, recommended_chain=recommended)

    def apply_to_config(
        self,
        report: CalibrationReport,
        config: PipelineConfig,
    ) -> PipelineConfig:
        """Apply calibration results to a pipeline config.

        Sets primary_engine to the best overall engine and fallback_chain
        to the remaining engines ordered by average WER.

        Args:
            report: Calibration report from calibrate().
            config: Pipeline config to update.

        Returns:
            Updated PipelineConfig (mutated in place and returned).
        """
        if not report.profiles:
            return config

        # Rank engines by their average WER across all categories (lower is better)
        ranked = sorted(
            report.profiles,
            key=lambda p: _avg_wer(p),
        )

        # Filter to engines that exist in EngineType
        valid_engines: list[EngineType] = []
        for profile in ranked:
            try:
                valid_engines.append(EngineType(profile.engine))
            except ValueError:
                continue

        if valid_engines:
            config.primary_engine = valid_engines[0]
            config.fallback_chain = valid_engines[1:]

        return config

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_profile(engine_name: str, runs: list[EngineRun]) -> EngineProfile:
        """Build an EngineProfile from a list of runs for one engine."""
        # Group by category — we need the paper category from somewhere.
        # Runs carry paper_name but not category. We infer category from
        # score.paper_name if score exists, but we don't have category info
        # in EngineRun. We'll use a placeholder approach: group by paper and
        # track WER, then aggregate. Since the caller provides full results
        # that include all engines x papers, we compute category_wer from
        # the score data.
        #
        # Category information: we need an external mapping. For now, we'll
        # use the failure_mode from EngineResult for failure_mode_recovery,
        # and compute category_wer only when paper metadata is available.

        # Category WER: collect from scores
        category_wers: dict[str, list[float]] = {}
        processing_times: list[float] = []

        # Failure mode recovery: papers where this engine succeeded vs. total
        failure_counts: dict[str, int] = {}
        failure_successes: dict[str, int] = {}

        for run in runs:
            processing_times.append(run.result.processing_time)

            # Track failure mode stats
            fm = run.result.failure_mode.value
            if fm != "none":
                failure_counts[fm] = failure_counts.get(fm, 0) + 1
                # "Recovered" if engine still produced usable output (has a score)
                if run.score and run.score.overall_wer < 0.5:
                    failure_successes[fm] = failure_successes.get(fm, 0) + 1

            if run.score is not None:
                # We don't have category in EngineRun, so we use "_all" as a
                # fallback. The caller can enrich this with paper metadata.
                category_wers.setdefault("_all", []).append(run.score.overall_wer)

        # Compute averages
        avg_category_wer: dict[str, float] = {}
        for cat, wers in category_wers.items():
            avg_category_wer[cat] = sum(wers) / len(wers) if wers else 1.0

        # Failure mode recovery rates
        recovery: dict[str, float] = {}
        for fm, count in failure_counts.items():
            recovery[fm] = failure_successes.get(fm, 0) / count

        avg_time = sum(processing_times) / len(processing_times) if processing_times else 0.0

        return EngineProfile(
            engine=engine_name,
            category_wer=avg_category_wer,
            failure_mode_recovery=recovery,
            avg_processing_time=avg_time,
        )

    def calibrate_with_categories(
        self,
        results: BenchmarkResults,
        paper_categories: dict[str, str],
    ) -> CalibrationReport:
        """Calibrate with explicit paper -> category mapping.

        This is the enriched version that produces per-category WER data
        in the engine profiles.

        Args:
            results: Benchmark results.
            paper_categories: Mapping of paper_name -> category.

        Returns:
            CalibrationReport with per-category data.
        """
        by_engine = results.by_engine()

        profiles: list[EngineProfile] = []
        for engine_name, runs in sorted(by_engine.items()):
            profile = self._build_profile_with_categories(
                engine_name, runs, paper_categories
            )
            profiles.append(profile)

        categories = set(paper_categories.values())
        recommended: dict[str, list[str]] = {}
        for category in sorted(categories):
            chain = self._rank_engines_for_category(category, profiles)
            recommended[category] = chain

        return CalibrationReport(profiles=profiles, recommended_chain=recommended)

    @staticmethod
    def _build_profile_with_categories(
        engine_name: str,
        runs: list[EngineRun],
        paper_categories: dict[str, str],
    ) -> EngineProfile:
        """Build profile with per-category WER from external category mapping."""
        category_wers: dict[str, list[float]] = {}
        processing_times: list[float] = []
        failure_counts: dict[str, int] = {}
        failure_successes: dict[str, int] = {}

        for run in runs:
            processing_times.append(run.result.processing_time)
            category = paper_categories.get(run.paper_name, "_unknown")

            fm = run.result.failure_mode.value
            if fm != "none":
                failure_counts[fm] = failure_counts.get(fm, 0) + 1
                if run.score and run.score.overall_wer < 0.5:
                    failure_successes[fm] = failure_successes.get(fm, 0) + 1

            if run.score is not None:
                category_wers.setdefault(category, []).append(run.score.overall_wer)

        avg_category_wer: dict[str, float] = {}
        for cat, wers in category_wers.items():
            avg_category_wer[cat] = sum(wers) / len(wers) if wers else 1.0

        recovery: dict[str, float] = {}
        for fm, count in failure_counts.items():
            recovery[fm] = failure_successes.get(fm, 0) / count

        avg_time = sum(processing_times) / len(processing_times) if processing_times else 0.0

        return EngineProfile(
            engine=engine_name,
            category_wer=avg_category_wer,
            failure_mode_recovery=recovery,
            avg_processing_time=avg_time,
        )

    @staticmethod
    def _collect_categories(results: BenchmarkResults) -> set[str]:
        """Collect all categories from results (via score paper names or defaults)."""
        # Without explicit category data, we return a single "_all" bucket
        return {"_all"}

    @staticmethod
    def _rank_engines_for_category(
        category: str,
        profiles: list[EngineProfile],
    ) -> list[str]:
        """Rank engines by WER for a specific category.

        Engines without data for this category are placed last, sorted by
        their overall average WER.
        """
        with_data: list[tuple[float, str]] = []
        without_data: list[tuple[float, str]] = []

        for profile in profiles:
            wer = profile.category_wer.get(category)
            if wer is not None:
                with_data.append((wer, profile.engine))
            else:
                # Use average across all categories as tiebreaker
                avg = _avg_wer(profile)
                without_data.append((avg, profile.engine))

        with_data.sort()
        without_data.sort()

        return [name for _, name in with_data] + [name for _, name in without_data]


def _avg_wer(profile: EngineProfile) -> float:
    """Average WER across all categories for a profile."""
    if not profile.category_wer:
        return 1.0
    return sum(profile.category_wer.values()) / len(profile.category_wer)
