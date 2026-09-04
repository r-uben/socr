"""TICKET-A1: fresh bind() vs frozen ``binding_adjudication`` replay.

Hermetic. Fixtures under ``tests/fixtures/replay_binding/corpus`` (generated
by ``generate_fixture.py``, checked in) — no live corpus, no PDF/OCR
provider, no network. ``ollama`` / ``qwen-ocr`` are removed from ``PATH``
for the whole module to prove nothing in this path shells out to either.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from socr.benchmark.replay_binding import (
    ReplayRow,
    discover_pages,
    replay_corpus,
    replay_page,
)

FIXTURE_CORPUS = Path(__file__).parent / "fixtures" / "replay_binding" / "corpus"


@pytest.fixture(autouse=True)
def _no_provider_on_path(monkeypatch, tmp_path):
    """Prove this module never shells out: strip any dir containing
    ``ollama`` or ``qwen-ocr`` from PATH for every test in this file."""
    empty_bin = tmp_path / "empty_bin"
    empty_bin.mkdir()
    kept = [
        p
        for p in os.environ.get("PATH", "").split(os.pathsep)
        if not (Path(p) / "ollama").exists() and not (Path(p) / "qwen-ocr").exists()
    ]
    monkeypatch.setenv("PATH", os.pathsep.join([str(empty_bin), *kept]))
    assert shutil.which("ollama") is None
    assert shutil.which("qwen-ocr") is None


def test_discover_pages_finds_both_fixture_pages():
    records = discover_pages(FIXTURE_CORPUS)
    assert {(r.doc_slug, r.page_num) for r in records} == {("doc00", 1), ("doc00", 2)}


def test_replay_corpus_seven_row_shape_on_two_page_fixture():
    """Same contract the real corpus proves at 7 rows (one row per recorded
    table): here, 2 recorded tables -> 2 rows, exactly."""
    rows = replay_corpus(FIXTURE_CORPUS)
    assert len(rows) == 2
    assert all(isinstance(r, ReplayRow) for r in rows)


def test_fresh_bind_matches_frozen_record_on_unchanged_tree():
    """The corpus-level Done-when, at fixture scale: on the UNCHANGED tree,
    fresh bind() reproduces the frozen record as an EXACT multiset (kind,
    native_token, model_token; duplicate counts preserved) -- not merely
    matching recorded/lifted-held status."""
    rows = replay_corpus(FIXTURE_CORPUS)
    by_table = {r.table_id: r for r in rows}
    p1 = by_table["p1-t0"]
    assert p1.recorded_item_count == 1
    assert p1.fresh_item_count == 1
    assert p1.multiset_match is True
    assert p1.added == ()
    assert p1.removed == ()


def test_fail_closed_marker_falls_back_to_cache_candidate():
    """Page 2's winning_output.text is the D3 marker; the real candidate
    text only exists in cache/. Replay must recover it and still match."""
    rows = replay_corpus(FIXTURE_CORPUS)
    p2 = next(r for r in rows if r.table_id == "p2-t0")
    assert p2.multiset_match is True
    assert "fail-closed marker" in p2.note
    assert "cache/aa/" in p2.note


def test_perturbed_recorded_items_report_exact_delta():
    """A hermetic fixture perturbs the RECORDED sidecar (simulating what a
    binder change on this tree would produce relative to an old recording)
    and asserts the exact expected added/removed delta -- not a bare
    mismatch flag."""
    records = discover_pages(FIXTURE_CORPUS)
    record = next(r for r in records if r.page_num == 1)

    # The real recorded item, for reference:
    real_item = ("row_label", "Treasury yield", "2Y Treasury yield")

    # Perturb: drop the real item, add a bogus one bind() would never
    # produce on this fixture.
    bogus_item = {
        "kind": "row_label",
        "native_token": "Term premium",
        "model_token": "10Y Term premium",
    }
    perturbed_adjudication = {
        "p1-t0": {
            "status": "held",
            "items": [bogus_item],
        }
    }
    perturbed_record = record.__class__(
        doc_slug=record.doc_slug,
        page_num=record.page_num,
        sidecar_path=record.sidecar_path,
        pdf_path=record.pdf_path,
        cache_dir=record.cache_dir,
        model_markdown=record.model_markdown,
        is_fail_closed_marker=record.is_fail_closed_marker,
        binding_adjudication=perturbed_adjudication,
    )

    rows = replay_page(perturbed_record, labels=None)
    assert len(rows) == 1
    row = rows[0]
    assert row.multiset_match is False
    assert row.added == (real_item,)
    assert row.removed == (("row_label", "Term premium", "10Y Term premium"),)


def test_sidecar_bytes_unchanged_by_replay():
    sidecar_path = FIXTURE_CORPUS / "out" / "doc00" / "doc00" / "pages" / "00001.json"
    before = sidecar_path.read_bytes()
    replay_corpus(FIXTURE_CORPUS)
    after = sidecar_path.read_bytes()
    assert before == after


def test_report_formats_without_raising():
    from socr.benchmark.replay_binding import format_report

    rows = replay_corpus(FIXTURE_CORPUS)
    report = format_report(rows)
    assert "p1-t0" in report
    assert "p2-t0" in report


def test_main_cli_exits_zero_and_prints_rows(capsys):
    from socr.benchmark.replay_binding import main

    exit_code = main([str(FIXTURE_CORPUS)])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "p1-t0" in out
    assert "p2-t0" in out


def test_labels_file_reported_when_absent_from_table(tmp_path):
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps({"doc00:some-other-table": {}}))
    labels = json.loads(labels_path.read_text())

    records = discover_pages(FIXTURE_CORPUS)
    record = next(r for r in records if r.page_num == 1)
    rows = replay_page(record, labels=labels)
    assert rows[0].label_accuracy == "n/a (no hand-read label for this table)"
    assert rows[0].crop_coverage == "n/a (no hand-read label for this table)"


def test_labels_absent_entirely_reported_as_unavailable():
    records = discover_pages(FIXTURE_CORPUS)
    record = next(r for r in records if r.page_num == 1)
    rows = replay_page(record, labels=None)
    assert "no --labels file" in rows[0].label_accuracy
    assert "no --labels file" in rows[0].crop_coverage
