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


def test_discover_pages_finds_all_fixture_pages():
    records = discover_pages(FIXTURE_CORPUS)
    assert {(r.doc_slug, r.page_num) for r in records} == {
        ("doc00", 1),
        ("doc00", 2),
        ("doc00", 3),
        ("doc00", 4),
        ("doc00", 5),
        ("doc00", 6),
        ("doc00", 7),
    }


def test_replay_corpus_row_shape_matches_recorded_table_count():
    """Same contract the real corpus proves at 7 rows (one row per recorded
    table): here, 7 recorded tables -> 7 rows, exactly."""
    rows = replay_corpus(FIXTURE_CORPUS)
    assert len(rows) == 7
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


def test_fail_closed_marker_falls_back_to_cache_candidate_by_provenance():
    """Page 2's winning_output.text is the D3 marker; the real candidate
    text only exists in cache/, identified by the ``table_binding_adjudicated``
    audit event's engine, never by scoring candidates against bind()."""
    rows = replay_corpus(FIXTURE_CORPUS)
    p2 = next(r for r in rows if r.table_id == "p2-t0")
    assert p2.unreplayable is False
    assert p2.multiset_match is True
    assert "fail-closed marker" in p2.note
    assert "by provenance" in p2.note
    assert "cache/aa/" in p2.note


def test_ambiguous_provenance_is_unreplayable_and_never_calls_bind(monkeypatch):
    """Page 3: two distinct cache candidates share the recorded provenance
    engine. The row must come back unreplayable, and ``bind()`` -- the
    function under test -- must NEVER be called to break the tie (that was
    the rejected design: letting bind() choose its own input)."""
    import socr.benchmark.replay_binding as replay_binding_module

    def _bind_must_not_be_called(*args, **kwargs):
        raise AssertionError("bind() was called for an unreplayable row")

    monkeypatch.setattr(replay_binding_module, "bind", _bind_must_not_be_called)

    sidecar_path = FIXTURE_CORPUS / "out" / "doc00" / "doc00" / "pages" / "00003.json"
    before = sidecar_path.read_bytes()

    record3 = next(r for r in discover_pages(FIXTURE_CORPUS) if r.page_num == 3)
    rows = replay_page(record3, labels=None)
    p3 = next(r for r in rows if r.table_id == "p3-t0")

    assert p3.unreplayable is True
    assert p3.multiset_match is False
    assert p3.added == ()
    assert p3.removed == ()
    assert "ambiguous" in p3.note
    assert "2 distinct cache candidates" in p3.note

    after = sidecar_path.read_bytes()
    assert before == after


def test_conflicting_provenance_engines_is_unreplayable_and_never_calls_bind(monkeypatch):
    """Page 4: the table has TWO table_binding_adjudicated events naming
    DIFFERENT engines, each with exactly one matching cache candidate (so
    the CACHE side alone would look unambiguous). Provenance itself is
    what is ambiguous here -- one level up from the cache-collision case
    -- and must also come back unreplayable without ever calling bind()."""
    import socr.benchmark.replay_binding as replay_binding_module

    def _bind_must_not_be_called(*args, **kwargs):
        raise AssertionError("bind() was called for an unreplayable row")

    monkeypatch.setattr(replay_binding_module, "bind", _bind_must_not_be_called)

    sidecar_path = FIXTURE_CORPUS / "out" / "doc00" / "doc00" / "pages" / "00004.json"
    before = sidecar_path.read_bytes()

    record4 = next(r for r in discover_pages(FIXTURE_CORPUS) if r.page_num == 4)
    assert record4.provenance_engines_by_table["p4-t0"] == frozenset({"qwen", "gemini"})

    rows = replay_page(record4, labels=None)
    p4 = next(r for r in rows if r.table_id == "p4-t0")

    assert p4.unreplayable is True
    assert p4.multiset_match is False
    assert p4.added == ()
    assert p4.removed == ()
    assert "conflicting engines" in p4.note
    assert "provenance itself is" in p4.note

    after = sidecar_path.read_bytes()
    assert before == after


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
        provenance_engines_by_table=record.provenance_engines_by_table,
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
    assert "p3-t0" in report
    assert "p4-t0" in report
    assert "p5-t99" in report
    assert "p6-t0" in report
    assert "p7-t0" in report
    assert "UNREPLAYABLE" in report


def test_main_cli_exits_zero_and_prints_rows(capsys):
    from socr.benchmark.replay_binding import main

    exit_code = main([str(FIXTURE_CORPUS)])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "p1-t0" in out
    assert "p2-t0" in out
    assert "p3-t0" in out
    assert "p4-t0" in out
    assert "p5-t99" in out
    assert "p6-t0" in out
    assert "p7-t0" in out


@pytest.mark.parametrize(
    ("page_num", "table_id", "note_fragment"),
    [
        (5, "p5-t99", "not found among this tree's witnesses"),
        (6, "p6-t0", "no located box this tree"),
        (7, "p7-t0", "no native words on this page"),
    ],
)
def test_replay_table_failure_is_unreplayable_not_a_binder_delta(
    monkeypatch, page_num, table_id, note_fragment
):
    """TICKET-A1c / GH-595: a non-empty replay_table note is UNREPLAYABLE,
    never a comparison row with an empty fresh side."""
    import socr.benchmark.replay_binding as replay_binding_module

    def _bind_must_not_be_called(*args, **kwargs):
        raise AssertionError("bind() was called for an unreplayable row")

    monkeypatch.setattr(replay_binding_module, "bind", _bind_must_not_be_called)

    sidecar_path = FIXTURE_CORPUS / "out" / "doc00" / "doc00" / "pages" / f"{page_num:05d}.json"
    before = sidecar_path.read_bytes()

    record = next(r for r in discover_pages(FIXTURE_CORPUS) if r.page_num == page_num)
    rows = replay_page(record, labels=None)
    row = next(r for r in rows if r.table_id == table_id)

    assert row.unreplayable is True
    assert row.added == ()
    assert row.removed == ()
    assert note_fragment in row.note

    after = sidecar_path.read_bytes()
    assert before == after


def test_cli_prints_unreplayable_for_a1c_failures(capsys):
    from socr.benchmark.replay_binding import main

    exit_code = main([str(FIXTURE_CORPUS)])
    assert exit_code == 0
    out = capsys.readouterr().out
    for table_id in ("p5-t99", "p6-t0", "p7-t0"):
        line = next(ln for ln in out.splitlines() if table_id in ln)
        assert "UNREPLAYABLE" in line, line
    assert "not found among this tree's witnesses" in out
    assert "no located box this tree" in out
    assert "no native words on this page" in out


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
