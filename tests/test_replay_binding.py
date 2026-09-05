"""TICKET-A1: fresh bind() vs frozen ``binding_adjudication`` replay.

Hermetic. Fixtures under ``tests/fixtures/replay_binding/corpus`` (generated
by ``generate_fixture.py``, checked in) — no live corpus, no PDF/OCR
provider, no network. ``ollama`` / ``qwen-ocr`` are removed from ``PATH``
for the whole module to prove nothing in this path shells out to either.
"""

from __future__ import annotations

import hashlib
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
        ("doc00", 8),
        ("doc00", 9),
        ("doc00", 10),
        ("doc00", 11),
    }


def test_replay_corpus_row_shape_matches_recorded_table_count():
    """Same contract the real corpus proves at 7 rows (one row per recorded
    table): here, 11 recorded tables -> 11 rows, exactly."""
    rows = replay_corpus(FIXTURE_CORPUS)
    assert len(rows) == 11
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
    bogus_key = ("row_label", "Term premium", "10Y Term premium")
    assert bogus_key not in row.removed
    assert bogus_key in row.unchecked_removed
    assert "no candidate row" in row.note


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
    assert "p8-t0" in report
    assert "p9-t0" in report
    assert "p10-t0" in report
    assert "p11-t0" in report
    assert "UNREPLAYABLE" in report
    assert "UNCHECKED" in report


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
    assert "p8-t0" in out
    assert "p9-t0" in out
    assert "p10-t0" in out
    assert "p11-t0" in out


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


def test_parse_grid_failure_is_unchecked_not_a_binder_clear():
    """Located witness whose candidate grid fails parse_grid: empty fresh
    items are UNCHECKED, never NO with the recorded items as frozen-only."""
    record = next(r for r in discover_pages(FIXTURE_CORPUS) if r.page_num == 8)
    sidecar_path = record.sidecar_path
    before = sidecar_path.read_bytes()
    rows = replay_page(record, labels=None)
    assert len(rows) == 1
    row = rows[0]
    assert row.table_id == "p8-t0"
    assert row.unreplayable is False
    assert row.unchecked is True
    assert row.added == ()
    assert row.removed == ()
    assert row.fresh_item_count == 0
    assert row.recorded_item_count >= 1
    assert "no checks" in row.note
    assert row.row_labels_checked == 0
    assert row.fully_checked is False
    assert sidecar_path.read_bytes() == before


def test_cli_prints_unchecked_for_parse_grid_failure(capsys):
    from socr.benchmark.replay_binding import main

    exit_code = main([str(FIXTURE_CORPUS)])
    assert exit_code == 0
    out = capsys.readouterr().out
    line = next(ln for ln in out.splitlines() if "p8-t0" in ln)
    assert "UNCHECKED" in line, line
    assert "UNREPLAYABLE" not in line
    words = line.split()
    assert "NO" not in words


def test_unrelated_row_checked_disputed_unbound_is_unchecked():
    """A sibling row is bound and compared; the disputed label is omitted
    from the candidate, so it is UNCHECKED, not a frozen-only clear."""
    record = next(r for r in discover_pages(FIXTURE_CORPUS) if r.page_num == 9)
    rows = replay_page(record, labels=None)
    assert len(rows) == 1
    row = rows[0]
    assert row.unreplayable is False
    assert row.unchecked is True
    assert row.removed == ()
    assert row.fresh_item_count == 0
    assert row.recorded_item_count >= 1
    assert "no candidate row" in row.note
    assert row.row_labels_checked is not None and row.row_labels_checked >= 1
    assert row.unchecked_removed


def test_mixed_unchecked_removed_is_no_not_a_silent_clear():
    """One recorded item remains as a fresh contradiction; another vanished
    without per-row evidence. The vanished item is UNCHECKED; the row is NO."""
    record = next(r for r in discover_pages(FIXTURE_CORPUS) if r.page_num == 10)
    rows = replay_page(record, labels=None)
    assert len(rows) == 1
    row = rows[0]
    assert row.unreplayable is False
    assert row.unchecked is False
    assert row.multiset_match is False
    assert row.fresh_item_count >= 1
    assert row.unchecked_removed
    for key in row.unchecked_removed:
        assert key not in row.removed
    assert "Gone yield" in str(row.unchecked_removed)


def test_cli_mixed_row_is_no_with_unchecked_item(capsys):
    from socr.benchmark.replay_binding import format_report

    rows = replay_corpus(FIXTURE_CORPUS)
    report = format_report(rows)
    p9_line = next(ln for ln in report.splitlines() if "p9-t0" in ln)
    assert "UNCHECKED" in p9_line, p9_line
    p10_line = next(ln for ln in report.splitlines() if "p10-t0" in ln)
    assert "NO" in p10_line.split(), p10_line
    assert "UNCHECKED" not in p10_line.split()
    assert "UNCHECKED:" in report


def test_duplicate_candidate_labels_are_unchecked():
    """Two candidate rows share a stub; the disputed frozen model_token
    matches both, so the row is UNCHECKED even if a sibling is bound."""
    record = next(r for r in discover_pages(FIXTURE_CORPUS) if r.page_num == 11)
    rows = replay_page(record, labels=None)
    assert len(rows) == 1
    row = rows[0]
    assert row.unreplayable is False
    assert row.unchecked_removed
    assert "matches" in row.note and "rows" in row.note
    for key in row.unchecked_removed:
        assert key not in row.removed


def test_successful_bind_carries_coverage_and_is_not_unchecked():
    rows = replay_corpus(FIXTURE_CORPUS)
    p1 = next(r for r in rows if r.table_id == "p1-t0")
    assert p1.unchecked is False
    assert p1.unreplayable is False
    assert p1.row_labels_checked is not None and p1.row_labels_checked >= 1
    assert p1.fully_checked is not None
    assert p1.column_binding_unverifiable is not None
    assert p1.native_unbound_count is not None


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


@pytest.mark.parametrize("field", ["token", "verdict", "reason", "kind", "order", "missing"])
def test_frozen_gate_rejects_prediction_copy_changed_without_artifact(monkeypatch, tmp_path, field):
    """An authenticated markdown hash must not bless a different JSON oracle."""
    import socr.benchmark.replay_binding as replay_binding_module

    root = Path(__file__).resolve().parents[1]
    fixture_relative = Path("tests/fixtures/replay_binding/controls/c2b_prediction.json")
    prediction = json.loads((root / fixture_relative).read_text())
    artifact = tmp_path / prediction["artifact"]
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes((root / prediction["artifact"]).read_bytes())
    corpus = tmp_path / prediction["corpus_name"]
    corpus.mkdir()
    manifest = corpus / "SHA256SUMS"
    manifest.write_bytes(b"")
    prediction["manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    fixture = tmp_path / fixture_relative
    fixture.parent.mkdir(parents=True)
    fixture.write_text(json.dumps(prediction))
    monkeypatch.setattr(
        replay_binding_module, "__file__", str(tmp_path / "src/socr/benchmark/replay_binding.py")
    )

    assert replay_binding_module._frozen_prediction(corpus) == prediction
    if field == "order":
        prediction["verdicts"].reverse()
    elif field == "missing":
        prediction["verdicts"].pop()
    else:
        index = {"kind": 3, "token": 4, "verdict": 5, "reason": 6}[field]
        prediction["verdicts"][0][index] = "tampered"
    fixture.write_text(json.dumps(prediction))

    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == prediction["artifact_sha256"]
    with pytest.raises(AssertionError, match="JSON verdicts differ from the markdown artifact"):
        replay_binding_module._frozen_prediction(corpus)
