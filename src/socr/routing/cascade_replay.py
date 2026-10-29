"""Offline projection of cascade routing from recorded ``manifest.json`` files."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

from socr.routing.cascade_policy import (
    classify_escalation_reason,
    is_word_count_rejection,
    paid_gemini_block_after_reject,
    sparse_signals_from_manifest_entry,
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


@dataclass(frozen=True)
class ResolvedManifestPdf:
    """PDF path resolution for replay / comparison lists."""

    path: Path | None
    resolved: bool
    lookup: str = "unresolved"


@dataclass
class PdfRootIndex:
    """Stem and basename lookup under a corpus PDF root."""

    by_name: dict[str, Path] = field(default_factory=dict)
    by_stem: dict[str, Path] = field(default_factory=dict)
    by_hash_name: dict[str, Path] = field(default_factory=dict)

    @classmethod
    def build(cls, pdf_root: Path) -> PdfRootIndex:
        idx = cls()
        if not pdf_root.is_dir():
            return idx
        for path in pdf_root.rglob("*.pdf"):
            if not path.is_file():
                continue
            idx.by_name.setdefault(path.name, path)
            idx.by_stem.setdefault(path.stem, path)
            if len(path.stem) == 64 and all(c in "0123456789abcdef" for c in path.stem.lower()):
                idx.by_hash_name[path.stem.lower()] = path
        return idx


def resolve_manifest_pdf_path(
    manifest_path: Path,
    manifest_data: dict | None = None,
    *,
    pdf_root: Path | None = None,
    pdf_index: PdfRootIndex | None = None,
) -> ResolvedManifestPdf:
    """Resolve a manifest's PDF for comparison re-runs."""
    data = manifest_data
    if data is None:
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except OSError:
            return ResolvedManifestPdf(path=None, resolved=False, lookup="unresolved")
    name = str(data.get("pdf_filename") or "").strip()
    pdf_hash = str(data.get("pdf_file_hash") or "").strip().lower()
    if not name:
        return ResolvedManifestPdf(path=None, resolved=False, lookup="unresolved")

    candidates: list[tuple[str, Path]] = [
        ("manifest_sibling", manifest_path.parent / name),
        ("manifest_parent", manifest_path.parent.parent / name),
    ]
    index = pdf_index
    if pdf_root is not None and index is None:
        index = PdfRootIndex.build(pdf_root)
    if index is not None:
        if name in index.by_name:
            candidates.append(("pdf_root_name", index.by_name[name]))
        stem = Path(name).stem
        if stem in index.by_stem:
            candidates.append(("pdf_root_stem", index.by_stem[stem]))
        if pdf_hash and pdf_hash in index.by_hash_name:
            candidates.append(("pdf_root_hash", index.by_hash_name[pdf_hash]))
    elif pdf_root is not None:
        candidates.append(("pdf_root_name", pdf_root / name))

    for lookup, candidate in candidates:
        if candidate.is_file():
            return ResolvedManifestPdf(path=candidate, resolved=True, lookup=lookup)

    fallback = candidates[0][1] if candidates else None
    return ResolvedManifestPdf(path=fallback, resolved=False, lookup="unresolved")


_GEMINI_NON_READ_FAILURE_MODES = frozenset({"cli_error", "model_unavailable"})


def sum_gemini_journal_stats(manifest_paths: list[Path]) -> dict[str, int]:
    """Split recorded Gemini journal rows from real reads vs infra failures."""
    stats: Counter = Counter()
    for path in manifest_paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except OSError:
            continue
        for entry in (data.get("entries") or {}).values():
            for row in entry.get("journal") or []:
                if row.get("engine") != "gemini":
                    continue
                stats["gemini_journal_rows"] += 1
                failure_mode = str(row.get("failure_mode") or "").lower()
                if failure_mode == "cli_error":
                    stats["gemini_cli_error_rows"] += 1
                elif failure_mode == "model_unavailable":
                    stats["gemini_model_unavailable_rows"] += 1
                elif row.get("accepted"):
                    stats["gemini_accepted_rows"] += 1
                else:
                    stats["gemini_other_failure_rows"] += 1
    non_read = stats["gemini_cli_error_rows"] + stats["gemini_model_unavailable_rows"]
    stats["gemini_real_read_rows"] = stats["gemini_journal_rows"] - non_read
    return dict(stats)


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
            signals = sparse_signals_from_manifest_entry(entry, journal)
            block_reason = paid_gemini_block_after_reject(
                qwen_reason,
                signals,
                audit_min_words=rules.audit_min_words,
                block_paid_on_source_evidence=rules.block_paid_on_source_evidence,
                sparse_word_count_exempt=rules.sparse_word_count_exempt,
            )
            if block_reason:
                would_new = False
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
    gemini_journal = sum_gemini_journal_stats(paths)
    return {
        "totals": dict(totals),
        "final_engines": dict(engines),
        "gemini_blocks": dict(blocks),
        "recorded_tokens": token_totals,
        "recorded_gemini_journal": gemini_journal,
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


def _round_robin_stratified_pick(
    by_stratum: dict[str, list[tuple[Path, PageReplayRow, dict]]],
    *,
    limit: int,
    seed: int,
    max_pages_per_doc: int,
) -> list[tuple[Path, PageReplayRow, dict]]:
    """Spread picks across manifests (round-robin), with a per-doc cap."""
    rng = random.Random(seed)
    stratum_doc_queues: dict[str, dict[Path, deque]] = {}
    doc_order: dict[str, list[Path]] = {}
    for stratum, items in by_stratum.items():
        by_doc: dict[Path, list] = defaultdict(list)
        for item in items:
            by_doc[item[0]].append(item)
        for doc_items in by_doc.values():
            rng.shuffle(doc_items)
        stratum_doc_queues[stratum] = {doc: deque(items) for doc, items in by_doc.items()}
        doc_order[stratum] = sorted(by_doc.keys(), key=str)

    per_doc_selected: Counter = Counter()
    selected: list[tuple[Path, PageReplayRow, dict]] = []
    strata = sorted(stratum_doc_queues.keys())
    while len(selected) < limit and strata:
        progressed = False
        for stratum in list(strata):
            queues = stratum_doc_queues[stratum]
            if not queues:
                continue
            for doc in list(doc_order[stratum]):
                if len(selected) >= limit:
                    break
                if per_doc_selected[doc] >= max_pages_per_doc:
                    continue
                queue = queues.get(doc)
                if not queue:
                    continue
                selected.append(queue.popleft())
                per_doc_selected[doc] += 1
                progressed = True
                if not queue:
                    del queues[doc]
            if not queues:
                strata.remove(stratum)
        if not progressed:
            break
    return selected


def select_stratified_comparison_pages(
    manifest_paths: list[Path],
    *,
    rules: ReplayRules | None = None,
    limit: int = 200,
    pdf_root: Path | None = None,
    seed: int = 0,
    max_pages_per_doc: int = 4,
) -> dict:
    """Pick up to ``limit`` pages that still project to Gemini under NEW rules.

    Stratifies by ``classify_escalation_reason`` on the last free-rung reject
    before Gemini (``table_structure``, ``garbage_chars``, ``min_words``, etc.).
    """
    rules = rules or ReplayRules()
    pool: list[tuple[Path, PageReplayRow, dict]] = []
    stratum_counts: Counter = Counter()
    blocked_strata: Counter = Counter()

    pdf_index = PdfRootIndex.build(pdf_root) if pdf_root is not None else None

    for path in manifest_paths:
        data = json.loads(path.read_text(encoding="utf-8"))
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

    picked = _round_robin_stratified_pick(
        by_stratum,
        limit=limit,
        seed=seed,
        max_pages_per_doc=max_pages_per_doc,
    )
    selected: list[dict] = []
    for manifest_path, row, data in picked:
        resolved = resolve_manifest_pdf_path(
            manifest_path, data, pdf_root=pdf_root, pdf_index=pdf_index
        )
        selected.append(
            {
                "manifest": str(manifest_path.resolve()),
                "pdf_path": str(resolved.path.resolve()) if resolved.path else "",
                "pdf_path_resolved": resolved.resolved,
                "pdf_path_lookup": resolved.lookup,
                "pdf_filename": str(data.get("pdf_filename") or ""),
                "page_num": row.page_num,
                "stratum": row.escalation_stratum or "other",
                "last_qwen_reason": row.last_qwen_reason,
            }
        )

    docs_in_selection = len({item["manifest"] for item in selected})
    return {
        "comparison_limit": limit,
        "max_pages_per_doc": max_pages_per_doc,
        "selection_seed": seed,
        "pool_size": len(pool),
        "selected_count": len(selected),
        "selected_manifest_count": docs_in_selection,
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
    parser.add_argument(
        "--pdf-root",
        type=Path,
        default=None,
        help="Directory of source PDFs (stem/filename/hash lookup for stratified lists)",
    )
    parser.add_argument(
        "--stratified-seed",
        type=int,
        default=0,
        help="RNG seed for stratified page selection (default 0)",
    )
    parser.add_argument(
        "--max-pages-per-doc",
        type=int,
        default=4,
        help="Cap stratified picks per manifest (default 4)",
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
            manifest_paths,
            rules=rules,
            limit=args.comparison_limit,
            pdf_root=args.pdf_root,
            seed=args.stratified_seed,
            max_pages_per_doc=args.max_pages_per_doc,
        )
        summary["stratified_comparison"] = {
            "comparison_limit": stratified["comparison_limit"],
            "max_pages_per_doc": stratified["max_pages_per_doc"],
            "selection_seed": stratified["selection_seed"],
            "pool_size": stratified["pool_size"],
            "selected_count": stratified["selected_count"],
            "selected_manifest_count": stratified["selected_manifest_count"],
            "stratum_counts_in_pool": stratified["stratum_counts_in_pool"],
            "gemini_block_reasons": stratified["gemini_block_reasons"],
        }
        args.stratified_out.parent.mkdir(parents=True, exist_ok=True)
        args.stratified_out.write_text(json.dumps(stratified, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
