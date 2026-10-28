"""Offline projection of cascade routing from recorded ``manifest.json`` files."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from socr.routing.cascade_policy import (
    SOURCE_EVIDENCE_MARKER,
    SparsePageSignals,
    is_word_count_rejection,
    sparse_page_ok,
)


@dataclass
class ReplayRules:
    """Policy toggles for NEW routing relative to recorded manifests."""

    block_paid_on_source_evidence: bool = True
    sparse_word_count_exempt: bool = True
    audit_min_words: int = 40


@dataclass
class PageReplayRow:
    page_num: int
    recorded_gemini: bool
    would_gemini_old: bool
    would_gemini_new: bool
    block_reason: str = ""
    last_qwen_reason: str = ""


@dataclass
class ManifestReplayReport:
    manifest_path: Path
    page_count: int
    recorded_gemini_pages: int = 0
    old_gemini_pages: int = 0
    new_gemini_pages: int = 0
    engine_final: Counter = field(default_factory=Counter)
    block_reasons: Counter = field(default_factory=Counter)
    rows: list[PageReplayRow] = field(default_factory=list)


def _last_reason_before_engine(journal: list[dict], engine: str) -> str:
    idx = next((i for i, e in enumerate(journal) if e.get("engine") == engine), None)
    if idx is None or idx == 0:
        return ""
    for entry in reversed(journal[:idx]):
        if not entry.get("accepted"):
            return str(entry.get("reason") or "")
    return ""


def _infer_sparse(entry: dict, journal: list[dict]) -> bool:
    fp = entry.get("fingerprint") or {}
    if str(fp.get("engine") or "") == "chart_asset":
        return True
    return any(is_word_count_rejection(str(j.get("reason") or "")) for j in journal)


def replay_manifest(path: Path, rules: ReplayRules | None = None) -> ManifestReplayReport:
    rules = rules or ReplayRules()
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.get("entries") or {}
    report = ManifestReplayReport(manifest_path=path, page_count=len(entries))

    for key, entry in entries.items():
        page_num = int(entry.get("page_num") or key)
        journal = entry.get("journal") or []
        fp = entry.get("fingerprint") or {}
        final_engine = str(fp.get("engine") or "unknown")
        report.engine_final[final_engine] += 1

        gemini_attempted = any(j.get("engine") == "gemini" for j in journal)
        if gemini_attempted:
            report.recorded_gemini_pages += 1
            report.old_gemini_pages += 1

        qwen_reason = _last_reason_before_engine(journal, "gemini")
        would_new = gemini_attempted
        block_reason = ""
        if gemini_attempted:
            if rules.block_paid_on_source_evidence and SOURCE_EVIDENCE_MARKER in qwen_reason:
                would_new = False
                block_reason = "source_evidence_table"
            elif rules.sparse_word_count_exempt and is_word_count_rejection(qwen_reason):
                # Manifests rarely carry raster metadata; a Word-count reject is
                # the recorded signal that the page was cover/chart/sparse-shaped.
                would_new = False
                block_reason = "sparse_word_count"
            elif rules.sparse_word_count_exempt and _infer_sparse(entry, journal):
                signals = SparsePageSignals(has_chart_asset=True)
                if sparse_page_ok(signals, audit_min_words=rules.audit_min_words):
                    would_new = False
                    block_reason = "sparse_chart_asset"
        if would_new:
            report.new_gemini_pages += 1
        if block_reason:
            report.block_reasons[block_reason] += 1

        report.rows.append(
            PageReplayRow(
                page_num=page_num,
                recorded_gemini=gemini_attempted,
                would_gemini_old=gemini_attempted,
                would_gemini_new=would_new,
                block_reason=block_reason,
                last_qwen_reason=qwen_reason,
            )
        )
    return report


def replay_paths(paths: list[Path], rules: ReplayRules | None = None) -> dict:
    totals = Counter()
    engines = Counter()
    blocks = Counter()
    for path in paths:
        rep = replay_manifest(path, rules=rules)
        totals["manifests"] += 1
        totals["pages"] += rep.page_count
        totals["recorded_gemini_pages"] += rep.recorded_gemini_pages
        totals["old_gemini_pages"] += rep.old_gemini_pages
        totals["new_gemini_pages"] += rep.new_gemini_pages
        engines.update(rep.engine_final)
        blocks.update(rep.block_reasons)
    return {
        "totals": dict(totals),
        "final_engines": dict(engines),
        "gemini_blocks": dict(blocks),
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Project cascade routing from manifest.json files")
    parser.add_argument(
        "paths", nargs="+", help="manifest.json files or directories containing them"
    )
    parser.add_argument("--audit-min-words", type=int, default=40)
    args = parser.parse_args(argv)

    manifest_paths: list[Path] = []
    for raw in args.paths:
        p = Path(raw)
        if p.is_file() and p.name == "manifest.json":
            manifest_paths.append(p)
        elif p.is_dir():
            manifest_paths.extend(sorted(p.rglob("manifest.json")))
    rules = ReplayRules(audit_min_words=args.audit_min_words)
    summary = replay_paths(manifest_paths, rules=rules)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
