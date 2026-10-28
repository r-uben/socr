"""Offline projection of cascade routing from recorded ``manifest.json`` files."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from socr.routing.cascade_policy import (
    SOURCE_EVIDENCE_MARKER,
    SparsePageSignals,
    blocks_paid_escalation,
    classify_escalation_reason,
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
    escalation_stratum: str = ""


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


def resolve_manifest_pdf_path(
    manifest_path: Path, manifest_data: dict | None = None
) -> Path | None:
    """Best-effort PDF path for a manifest (for comparison re-run lists)."""
    data = manifest_data
    if data is None:
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except OSError:
            return None
    name = str(data.get("pdf_filename") or "").strip()
    if not name:
        return None
    for candidate in (
        manifest_path.parent / name,
        manifest_path.parent.parent / name,
    ):
        if candidate.is_file():
            return candidate
    return manifest_path.parent / name


def sum_journal_tokens(manifest_paths: list[Path]) -> dict:
    """Sum ``llm_input_tokens`` / ``llm_output_tokens`` from recorded manifest journals."""
    inp = out = 0
    inp_known = out_known = 0
    for path in manifest_paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        for entry in (data.get("entries") or {}).values():
            for row in entry.get("journal") or []:
                if row.get("llm_input_tokens") is not None:
                    inp += int(row["llm_input_tokens"])
                    inp_known += 1
                if row.get("llm_output_tokens") is not None:
                    out += int(row["llm_output_tokens"])
                    out_known += 1
    return {
        "llm_input_tokens": inp,
        "llm_output_tokens": out,
        "journal_rows_with_input_tokens": inp_known,
        "journal_rows_with_output_tokens": out_known,
    }


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
            if rules.block_paid_on_source_evidence and blocks_paid_escalation(qwen_reason):
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

        stratum = classify_escalation_reason(qwen_reason) if gemini_attempted else ""

        report.rows.append(
            PageReplayRow(
                page_num=page_num,
                recorded_gemini=gemini_attempted,
                would_gemini_old=gemini_attempted,
                would_gemini_new=would_new,
                block_reason=block_reason,
                last_qwen_reason=qwen_reason,
                escalation_stratum=stratum,
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
        totals["blocked_gemini_under_new_rules"] += rep.recorded_gemini_pages - rep.new_gemini_pages
        engines.update(rep.engine_final)
        blocks.update(rep.block_reasons)
    token_totals = sum_journal_tokens(paths)
    return {
        "totals": dict(totals),
        "final_engines": dict(engines),
        "gemini_blocks": dict(blocks),
        "recorded_tokens": token_totals,
        "replay_honesty": {
            "recorded_gemini_pages_old_rules": totals["old_gemini_pages"],
            "blocked_from_gemini_new_rules": totals["blocked_gemini_under_new_rules"],
            "projected_gemini_pool_new_rules": totals["new_gemini_pages"],
            "marker_gemma_absorption": (
                "not modeled offline — replay only applies NEW Gemini-suppression "
                "rules; it cannot predict how many former Gemini pages marker or "
                "Gemma would accept without re-running OCR"
            ),
            "blocked_pages_shipping": (
                "not modeled offline — blocked pages are assumed not to call Gemini; "
                "live disposition (FLAGGED / fail-closed floor / best-effort) depends "
                "on table flags and ladder outcomes, not replay alone"
            ),
        },
    }


def select_stratified_comparison_pages(
    manifest_paths: list[Path],
    *,
    rules: ReplayRules | None = None,
    limit: int = 200,
) -> dict:
    """Pick up to ``limit`` pages that still project to Gemini under NEW rules.

    Stratifies by ``classify_escalation_reason`` on the last free-rung reject
    before Gemini (``table_structure``, ``garbage_chars``, ``min_words``, etc.).
    """
    rules = rules or ReplayRules()
    pool: list[tuple[Path, PageReplayRow, dict]] = []
    stratum_counts: Counter = Counter()
    blocked_strata: Counter = Counter()

    for path in manifest_paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        pdf_path = resolve_manifest_pdf_path(path, data)
        rep = replay_manifest(path, rules=rules)
        for row in rep.rows:
            if not row.recorded_gemini:
                continue
            if row.would_gemini_new:
                stratum_counts[row.escalation_stratum or "other"] += 1
                pool.append((path, row, data))
            elif row.block_reason:
                blocked_strata[row.block_reason] += 1

    by_stratum: dict[str, list[tuple[Path, PageReplayRow, dict]]] = defaultdict(list)
    for item in pool:
        by_stratum[item[1].escalation_stratum or "other"].append(item)

    selected: list[dict] = []
    queues = {k: list(v) for k, v in sorted(by_stratum.items())}
    while len(selected) < limit and any(queues.values()):
        for stratum in list(queues.keys()):
            if len(selected) >= limit:
                break
            queue = queues[stratum]
            if not queue:
                continue
            manifest_path, row, data = queue.pop(0)
            pdf = resolve_manifest_pdf_path(manifest_path, data)
            selected.append(
                {
                    "manifest": str(manifest_path.resolve()),
                    "pdf_path": str(pdf.resolve()) if pdf else "",
                    "pdf_filename": str(data.get("pdf_filename") or ""),
                    "page_num": row.page_num,
                    "stratum": row.escalation_stratum or "other",
                    "last_qwen_reason": row.last_qwen_reason,
                }
            )

    return {
        "comparison_limit": limit,
        "pool_size": len(pool),
        "selected_count": len(selected),
        "stratum_counts_in_pool": dict(stratum_counts),
        "gemini_block_reasons": dict(blocked_strata),
        "selected_pages": selected,
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Project cascade routing from manifest.json files")
    parser.add_argument(
        "paths", nargs="+", help="manifest.json files or directories containing them"
    )
    parser.add_argument("--audit-min-words", type=int, default=40)
    parser.add_argument(
        "--stratified-out",
        type=Path,
        default=None,
        help="Write stratified Gemini comparison page list (JSON) for capped paid re-runs",
    )
    parser.add_argument(
        "--comparison-limit",
        type=int,
        default=200,
        help="Max pages in --stratified-out (default 200)",
    )
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
    if args.stratified_out is not None:
        stratified = select_stratified_comparison_pages(
            manifest_paths, rules=rules, limit=args.comparison_limit
        )
        summary["stratified_comparison"] = {
            "comparison_limit": stratified["comparison_limit"],
            "pool_size": stratified["pool_size"],
            "selected_count": stratified["selected_count"],
            "stratum_counts_in_pool": stratified["stratum_counts_in_pool"],
            "gemini_block_reasons": stratified["gemini_block_reasons"],
        }
        args.stratified_out.parent.mkdir(parents=True, exist_ok=True)
        args.stratified_out.write_text(json.dumps(stratified, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
