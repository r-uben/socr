"""Benchmark dataset definitions for OCR quality evaluation.

Defines the paper selection, categories, and serialization for the benchmark set.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class BenchmarkPaper:
    """A single paper in the benchmark set."""

    name: str  # e.g. "guerrieri_lorenzoni_werning"
    pdf_path: Path
    category: str  # "math_heavy", "table_heavy", "text_only", "figure_heavy", "mixed", "edge_case", "scanned"
    page_count: int
    ground_truth_path: Path | None = None  # path to ground truth directory
    notes: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.pdf_path, str):
            self.pdf_path = Path(self.pdf_path)
        if isinstance(self.ground_truth_path, str):
            self.ground_truth_path = Path(self.ground_truth_path)


@dataclass
class BenchmarkSet:
    """Collection of benchmark papers with serialization."""

    papers: list[BenchmarkPaper] = field(default_factory=list)
    created: str = ""  # ISO timestamp

    def __post_init__(self) -> None:
        if not self.created:
            self.created = datetime.now(timezone.utc).isoformat()

    def by_category(self) -> dict[str, list[BenchmarkPaper]]:
        """Group papers by category."""
        groups: dict[str, list[BenchmarkPaper]] = {}
        for paper in self.papers:
            groups.setdefault(paper.category, []).append(paper)
        return groups

    def save(self, path: Path) -> None:
        """Save benchmark set as JSON."""
        data = {
            "created": self.created,
            "papers": [
                {
                    "name": p.name,
                    "pdf_path": str(p.pdf_path),
                    "category": p.category,
                    "page_count": p.page_count,
                    "ground_truth_path": str(p.ground_truth_path) if p.ground_truth_path else None,
                    "notes": p.notes,
                }
                for p in self.papers
            ],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> BenchmarkSet:
        """Load benchmark set from JSON."""
        data = json.loads(path.read_text())
        papers = [
            BenchmarkPaper(
                name=p["name"],
                pdf_path=Path(p["pdf_path"]),
                category=p["category"],
                page_count=p["page_count"],
                ground_truth_path=Path(p["ground_truth_path"]) if p.get("ground_truth_path") else None,
                notes=p.get("notes", ""),
            )
            for p in data["papers"]
        ]
        return cls(papers=papers, created=data["created"])


# --- Benchmark paper definitions ---

PAPERS_DIR_DEFAULT = Path(
    "~/Library/Mobile Documents/com~apple~CloudDocs/library/Papers/papers"
).expanduser()

BENCHMARK_PAPERS: list[dict] = [
    # Stress tests
    {
        "name": "fernandez_fuertes__monetary_policy_shocks",
        "filename": "2025__fernandez_fuertes__monetary_policy_shocks_a_new_hope__JMP.pdf",
        "category": "mixed",
        "page_count": 79,
        "notes": "Math+tables+figures, 79 pages",
    },
    {
        "name": "guerrieri_lorenzoni_werning__global_price_shocks",
        "filename": "2025__guerrieri_lorenzoni_werning__global_price_shocks_and_international_monetary_coordination__WP.pdf",
        "category": "math_heavy",
        "page_count": 44,
        "notes": "301 math markers, 44 pages",
    },
    {
        "name": "eisenbach_lucca_townsend__economics_bank_supervision",
        "filename": "2016__eisenbach_lucca_townsend__economics_of_bank_supervision__NBER.pdf",
        "category": "mixed",
        "page_count": 58,
        "notes": "Math+tables, 58 pages",
    },
    # Specialists
    {
        "name": "goldsmith_pinkham_hirtle_lucca__parsing_bank_supervision",
        "filename": "2016__goldsmith_pinkham_hirtle_lucca__parsing_bank_supervision__NYFED.pdf",
        "category": "table_heavy",
        "page_count": 59,
        "notes": "128 tables, 59 pages",
    },
    {
        "name": "bugel_hidalgo_luetticke__unconventional_unified",
        "filename": "2026__bugel_hidalgo_luetticke__unconventional_unified_narrative_mp_shocks__WP.pdf",
        "category": "figure_heavy",
        "page_count": 45,
        "notes": "16 figures, 45 pages",
    },
    {
        "name": "raso__strategic_or_sincere_guidance",
        "filename": "2010__raso__strategic_or_sincere_guidance__YLJ.pdf",
        "category": "text_only",
        "page_count": 43,
        "notes": "Law journal, dense footnotes, 43 pages",
    },
    # Baselines & edge cases
    {
        "name": "elliott_golub_leduc__supply_network_formation",
        "filename": "2022__elliott_golub_leduc__supply_network_formation_and_fragility__AER.pdf",
        "category": "mixed",
        "page_count": 47,
        "notes": "Math+figures, 47 pages",
    },
    {
        "name": "correia_luck_verner__supervising_failing_banks",
        "filename": "2025__correia_luck_verner__supervising_failing_banks__NBER.pdf",
        "category": "table_heavy",
        "page_count": 55,
        "notes": "105 tables, 55 pages",
    },
    {
        "name": "shrimali_ahmad__when_central_banks_all_ears",
        "filename": "2025__shrimali_ahmad__when_central_banks_are_all_ears__FRL.pdf",
        "category": "edge_case",
        "page_count": 6,
        "notes": "6 pages, quick baseline",
    },
    {
        "name": "haim__how_binding_administrative_guidance",
        "filename": "2025__haim__how_binding_is_administrative_guidance__JELS.pdf",
        "category": "edge_case",
        "page_count": 35,
        "notes": "Failed OCR, 35 pages",
    },
]


def build_benchmark_set(papers_dir: Path | None = None) -> BenchmarkSet:
    """Build the benchmark set from the paper definitions.

    Args:
        papers_dir: Directory containing the PDF files.
                    Defaults to the standard Papers library location.

    Returns:
        BenchmarkSet with resolved paths.

    Raises:
        FileNotFoundError: If a required PDF is missing.
    """
    papers_dir = papers_dir or PAPERS_DIR_DEFAULT
    papers: list[BenchmarkPaper] = []
    missing: list[str] = []

    for spec in BENCHMARK_PAPERS:
        pdf_path = papers_dir / spec["filename"]
        if not pdf_path.exists():
            missing.append(spec["filename"])
            continue
        papers.append(
            BenchmarkPaper(
                name=spec["name"],
                pdf_path=pdf_path,
                category=spec["category"],
                page_count=spec["page_count"],
                notes=spec["notes"],
            )
        )

    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} benchmark PDFs in {papers_dir}:\n"
            + "\n".join(f"  - {f}" for f in missing)
        )

    return BenchmarkSet(papers=papers)
