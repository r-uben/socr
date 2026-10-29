"""Hermetic tests for offline cascade replay projection."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from socr.routing.cascade_policy import classify_escalation_reason
from socr.routing.cascade_replay import (
    ReplayRules,
    resolve_manifest_pdf_path,
    replay_manifest,
    select_stratified_comparison_pages,
    sum_gemini_journal_stats,
)


def _write_manifest(tmp_path: Path, page_journal: list[dict]) -> Path:
    manifest = {
        "schema_version": "1",
        "page_count": 1,
        "entries": {
            "1": {
                "page_num": 1,
                "fingerprint": {"engine": "gemini"},
                "journal": page_journal,
            }
        },
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_replay_blocks_gemini_after_source_evidence(tmp_path: Path) -> None:
    path = _write_manifest(
        tmp_path,
        [
            {
                "engine": "qwen",
                "accepted": False,
                "reason": "source_evidence_table: numeric tokens unsupported",
            },
            {"engine": "gemini", "accepted": False, "reason": "heuristics failed"},
        ],
    )
    rep = replay_manifest(path, ReplayRules())
    assert rep.recorded_gemini_pages == 1
    assert rep.new_gemini_pages == 0
    assert rep.block_reasons["source_evidence_table"] == 1


def test_replay_keeps_gemini_for_structure_failures(tmp_path: Path) -> None:
    path = _write_manifest(
        tmp_path,
        [
            {
                "engine": "qwen",
                "accepted": False,
                "reason": "table_structure_failed: grid_shape",
            },
            {"engine": "gemini", "accepted": True, "reason": "heuristics passed"},
        ],
    )
    rep = replay_manifest(path, ReplayRules())
    assert rep.new_gemini_pages == 1


def test_classify_escalation_reason_buckets() -> None:
    assert classify_escalation_reason("source_evidence_table: foo") == "source_evidence_table"
    assert classify_escalation_reason("Word count 12 < 40") == "min_words"
    assert classify_escalation_reason("table_structure_failed: grid") == "table_structure"
    assert classify_escalation_reason("native garbage layer") == "garbage_chars"


def test_stratified_comparison_respects_limit(tmp_path: Path) -> None:
    journals = [
        (
            "table_structure_failed: x",
            [
                {"engine": "qwen", "accepted": False, "reason": "table_structure_failed: x"},
                {"engine": "gemini", "accepted": True, "reason": ""},
            ],
            None,
        ),
        (
            "Word count 5 < 40",
            [
                {"engine": "qwen", "accepted": False, "reason": "Word count 5 < 40"},
                {"engine": "gemini", "accepted": True, "reason": ""},
            ],
            {"is_born_digital": True, "word_count": 5},
        ),
    ]
    paths = []
    for i, (_reason, journal, routing_meta) in enumerate(journals, start=1):
        entry = {
            "page_num": 1,
            "fingerprint": {"engine": "gemini"},
            "journal": journal,
        }
        if routing_meta:
            entry["routing_meta"] = routing_meta
        manifest = {
            "schema_version": "1",
            "pdf_filename": f"p{i}.pdf",
            "page_count": 1,
            "entries": {"1": entry},
        }
        path = tmp_path / f"m{i}" / "manifest.json"
        path.parent.mkdir()
        path.write_text(json.dumps(manifest), encoding="utf-8")
        paths.append(path)

    # Second manifest would be blocked under NEW rules (sparse word count)
    out = select_stratified_comparison_pages(paths, limit=200)
    assert out["pool_size"] == 1
    assert out["selected_count"] == 1
    assert out["selected_pages"][0]["stratum"] == "table_structure"


def test_resolve_pdf_path_marks_unresolved_without_file(tmp_path: Path) -> None:
    manifest = {
        "schema_version": "1",
        "pdf_filename": "missing.pdf",
        "entries": {},
    }
    path = tmp_path / "out" / "doc" / "manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    resolved = resolve_manifest_pdf_path(path, manifest)
    assert resolved.resolved is False
    assert resolved.lookup == "unresolved"


def test_resolve_pdf_path_via_pdf_root_stem(tmp_path: Path) -> None:
    pdf_root = tmp_path / "pdfs"
    pdf_root.mkdir()
    real_pdf = pdf_root / "nested" / "report.pdf"
    real_pdf.parent.mkdir()
    real_pdf.write_bytes(b"%PDF-1.4\n")
    manifest = {
        "schema_version": "1",
        "pdf_filename": "report.pdf",
        "pdf_file_hash": "abc",
        "entries": {},
    }
    man_path = tmp_path / "cluster" / "out" / "report" / "manifest.json"
    man_path.parent.mkdir(parents=True)
    man_path.write_text(json.dumps(manifest), encoding="utf-8")
    resolved = resolve_manifest_pdf_path(man_path, manifest, pdf_root=pdf_root)
    assert resolved.resolved is True
    assert resolved.path == real_pdf
    assert resolved.lookup in ("pdf_root_stem", "pdf_root_name")


def test_stratified_selection_spreads_across_manifests(tmp_path: Path) -> None:
    paths: list[Path] = []
    for doc in range(3):
        entries = {}
        for page in range(1, 11):
            entries[str(page)] = {
                "page_num": page,
                "fingerprint": {"engine": "gemini"},
                "journal": [
                    {
                        "engine": "qwen",
                        "accepted": False,
                        "reason": "table_structure_failed: grid",
                    },
                    {"engine": "gemini", "accepted": True, "reason": ""},
                ],
            }
        manifest = {
            "schema_version": "1",
            "pdf_filename": f"doc{doc}.pdf",
            "page_count": 10,
            "entries": entries,
        }
        path = tmp_path / f"m{doc}" / "manifest.json"
        path.parent.mkdir()
        path.write_text(json.dumps(manifest), encoding="utf-8")
        paths.append(path)

    out = select_stratified_comparison_pages(paths, limit=12, max_pages_per_doc=4, seed=42)
    assert out["selected_count"] == 12
    assert out["selected_manifest_count"] == 3
    counts = Counter(p["manifest"] for p in out["selected_pages"])
    assert all(c <= 4 for c in counts.values())
    assert len(counts) == 3


def test_sum_gemini_journal_stats_splits_cli_errors(tmp_path: Path) -> None:
    manifest = {
        "schema_version": "1",
        "entries": {
            "1": {
                "page_num": 1,
                "journal": [
                    {"engine": "gemini", "accepted": False, "failure_mode": "cli_error"},
                ],
            },
            "2": {
                "page_num": 2,
                "journal": [
                    {"engine": "gemini", "accepted": True, "failure_mode": "none"},
                ],
            },
            "3": {
                "page_num": 3,
                "journal": [
                    {
                        "engine": "gemini",
                        "accepted": False,
                        "failure_mode": "model_unavailable",
                    },
                ],
            },
        },
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    stats = sum_gemini_journal_stats([path])
    assert stats["gemini_journal_rows"] == 3
    assert stats["gemini_cli_error_rows"] == 1
    assert stats["gemini_model_unavailable_rows"] == 1
    assert stats["gemini_accepted_rows"] == 1
    assert stats["gemini_real_read_rows"] == 1


def test_latex_control_words_not_labels() -> None:
    from socr.tables.source_evidence import collect_table_tokens

    md = "| a | b |\n|---|---|\n| $x_3, \\ldots, x_n$ | $\\frac{1}{2}$ |"
    tokens = collect_table_tokens(md)
    assert tokens is not None
    assert "ldots" not in tokens.content
    assert "frac" not in tokens.content
