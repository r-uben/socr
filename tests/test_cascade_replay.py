"""Hermetic tests for offline cascade replay projection."""

from __future__ import annotations

import json
from pathlib import Path

from socr.routing.cascade_policy import classify_escalation_reason
from socr.routing.cascade_replay import (
    ReplayRules,
    replay_manifest,
    select_stratified_comparison_pages,
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
        ),
        (
            "Word count 5 < 40",
            [
                {"engine": "qwen", "accepted": False, "reason": "Word count 5 < 40"},
                {"engine": "gemini", "accepted": True, "reason": ""},
            ],
        ),
    ]
    paths = []
    for i, (_reason, journal) in enumerate(journals, start=1):
        manifest = {
            "schema_version": "1",
            "pdf_filename": f"p{i}.pdf",
            "page_count": 1,
            "entries": {
                "1": {
                    "page_num": 1,
                    "fingerprint": {"engine": "gemini"},
                    "journal": journal,
                }
            },
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


def test_latex_control_words_not_labels() -> None:
    from socr.tables.source_evidence import collect_table_tokens

    md = "| a | b |\n|---|---|\n| $x_3, \\ldots, x_n$ | $\\frac{1}{2}$ |"
    tokens = collect_table_tokens(md)
    assert tokens is not None
    assert "ldots" not in tokens.content
    assert "frac" not in tokens.content
