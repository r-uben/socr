"""GH-1291: the audit scoreboard counts what the shipped files say, page by page."""

from __future__ import annotations

import json
from pathlib import Path

import pymupdf
import pytest

from socr.benchmark import scoreboard


def _page(doc_dir: Path, num: int, text: str, **sidecar) -> None:
    pages = doc_dir / "pages"
    pages.mkdir(parents=True, exist_ok=True)
    (pages / f"{num:05d}.md").write_text(text)
    body = {"page_num": num, "status": "success", "engine": "native", "audit_passed": True}
    body.update(sidecar)
    (pages / f"{num:05d}.json").write_text(json.dumps(body))


def _finish(doc_dir: Path, status: str = "completed", **meta) -> None:
    (doc_dir / "metadata.json").write_text(json.dumps({"status": status, **meta}))
    (doc_dir / "manifest.json").write_text("{}")
    (doc_dir / "audit_log.json").write_text("{}")


@pytest.fixture
def run(tmp_path: Path) -> Path:
    root = tmp_path / "run" / "t0" / "out" / "w0"
    a = root / "doc-a"
    _page(a, 1, "Plain prose.\n")
    _page(
        a,
        2,
        "> [page 2 unchecked: x=FAIL]\n\ntable\n",
        status="warning",
        engine="qwen",
        verdict={"clauses": [{"check": "x", "result": "FAIL"}]},
        audit_events=[{"kind": "table_row_binding_unverifiable"}],
    )
    _page(
        a,
        3,
        "[page 3 failed: unverifiable table — see image]\n",
        status="error",
        engine="qwen",
        audit_passed=False,
        failure_mode="hallucination",
    )
    _page(
        a, 4, "Chart <!-- Image (1) --> and \\begin{tabular} and \x00 and ![f](figures/gone.png)\n"
    )
    _page(
        a,
        5,
        "2^{3/4} percent\n",
        status="warning",
        engine="native+math",
        audit_passed=False,
        failure_mode="equation",
    )
    _finish(a, "completed")
    (a / "tables_trust.json").write_text(
        json.dumps({"pages": {"1": {"flagged": True}, "2": {"flagged": True}}})
    )

    b = root / "doc-b"
    _page(
        b,
        1,
        "Unmarked failure.\n",
        status="error",
        engine="native",
        audit_passed=False,
        failure_mode="x",
        verdict={"clauses": [{"check": "unknown", "result": "UNKNOWN"}]},
    )
    return tmp_path / "run"


def test_file_counts(run: Path) -> None:
    c = scoreboard.score_run(run, None, None, workers=2)
    assert c["coverage.documents"] == 2
    assert c["coverage.pages"] == 6
    assert c["coverage.documents_unfinished"] == 1  # doc-b has no metadata.json
    assert c["coverage.documents_torn"] == 1
    assert c["coverage.document_status.completed"] == 1
    assert c["labels.status.success"] == 2
    assert c["labels.status_engine.warning|qwen"] == 1
    assert c["labels.completed_documents_with_non_success_pages"] == 1
    assert c["labels.completed_documents_with_error_pages"] == 1
    assert c["labels.warning_audit_passed_true"] == 1
    assert c["labels.warning_audit_passed_true_with_fail_clause"] == 1
    assert c["labels.non_success_unmarked"] == 2  # doc-a p5 and doc-b p1
    assert c["labels.error_hallucination"] == 1
    assert c["labels.tables_trust_flagged"] == 2
    assert c["labels.tables_trust_flagged_on_success"] == 1
    assert c["tells.success_image_comment"] == 1
    assert c["tells.success_raw_latex_tabular"] == 1
    assert c["tells.success_control_bytes"] == 1
    assert c["tells.pages_with_image_link_to_missing_file"] == 1
    assert c["tells.see_image_without_image"] == 1
    assert c["tells.native_math_pages"] == 1
    assert c["tells.native_math_power_fraction"] == 1
    assert c["events.table_row_binding_unverifiable"] == 1
    assert c["records.clean_native_success_no_trace"] == 2
    assert c["records.pages_clause_check_unknown"] == 1
    assert c["records.documents_processing_time_zero"] == 1


def test_two_runs_count_only_shared_documents(run: Path, tmp_path: Path) -> None:
    other = tmp_path / "other" / "doc-a"
    _page(other, 1, "Plain prose.\n")
    out = tmp_path / "out.json"
    scoreboard.main([str(run), str(tmp_path / "other"), "--json", str(out), "--workers", "2"])
    results = json.loads(out.read_text())
    assert [r["coverage.documents"] for r in results.values()] == [1, 1]


def test_duplicate_document_keeps_the_fuller_folder(run: Path) -> None:
    # "a0" sorts before "t0", so the sparser copy is found first and must lose.
    dup = run / "a0" / "out" / "w0" / "doc-a"
    _page(dup, 1, "x\n")
    assert scoreboard.find_documents(run)["doc-a"].parts[-4] == "t0"
    assert scoreboard.score_run(run, None, None, 2)["coverage.duplicate_document_folders"] == 1


def test_text_layer_counts_missing_words_numbers_and_dropped_decimals(tmp_path: Path) -> None:
    pdfs = tmp_path / "pdfs"
    pdfs.mkdir()
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Core inflation fell -.9 percent in 2003, rates -0.27")
    pdf.save(pdfs / "doc-t.pdf")
    doc = tmp_path / "run" / "doc-t"
    _page(doc, 1, "inflation fell -9 percent in 2003, rates 0.27\n")
    _finish(doc)
    c = scoreboard.score_run(tmp_path / "run", None, pdfs, workers=1)
    assert c["textlayer.words_missing"] == 1  # "core"
    assert c["textlayer.numbers_missing"] == 1  # ".9"; the audit's count ignores signs
    assert c["textlayer.dropped_decimal_point_values"] == 1
    assert c["textlayer.pages_dropped_decimal_point"] == 1
    assert c["textlayer.sign_dropped_values"] == 1  # -0.27 shipped as 0.27
    assert c["textlayer.pages_sign_dropped"] == 1


def test_text_layer_is_read_without_glyph_recovery(tmp_path: Path, monkeypatch) -> None:
    import socr.core.pdf as pdf_mod

    calls = []
    monkeypatch.setattr(pdf_mod, "apply_glyph_recovery", lambda *a, **k: calls.append(a))
    pdf = pymupdf.open()
    pdf.new_page().insert_text((72, 72), "Words")
    pdf.save(tmp_path / "doc-r.pdf")
    doc = tmp_path / "doc-r"
    _page(doc, 1, "Words\n")
    c = scoreboard.count_text_layer((doc, tmp_path / "doc-r.pdf"))
    assert c["textlayer.words"] == 1
    assert calls == []


def test_unreadable_files_are_counted_and_the_count_goes_on(run: Path, tmp_path: Path) -> None:
    a = run / "t0" / "out" / "w0" / "doc-a"
    (a / "pages" / "00006.md").write_text("orphan fragment\n")
    (a / "pages" / "00007.json").write_text('{"page_num": 7, "sta')
    (a / "tables_trust.json").write_text('{"pages": ')
    c_dir = run / "t0" / "out" / "w0" / "doc-c"
    _page(c_dir, 1, "fine\n")
    _finish(c_dir)
    (c_dir / "metadata.json").write_text("{")
    gold_path = tmp_path / "gold.jsonl"
    gold_path.write_text(
        json.dumps({"doc": "doc-a", "page": 6, "md_sha256": "0" * 64, "content": "clean"}) + "\n"
    )
    queue = tmp_path / "queue.jsonl"
    c = scoreboard.score_run(run, None, None, 2, scoreboard.load_gold(gold_path), queue)
    assert c["coverage.pages_without_sidecar"] == 1
    assert c["coverage.unreadable_sidecars"] == 1
    assert c["coverage.unreadable_tables_trust"] == 1
    assert c["coverage.unreadable_metadata"] == 1
    assert c["coverage.documents"] == 3
    assert c["gold.sidecar_unreadable_needs_review"] == 1
    assert [json.loads(x)["why"] for x in queue.read_text().splitlines()] == ["sidecar"]


def test_render_shows_delta_only_for_two_runs() -> None:
    from collections import Counter

    out = scoreboard.render(
        [Counter({"labels.status.success": 3}), Counter({"labels.status.success": 1})],
        ["run1", "run2"],
    )
    assert "-2" in out


def test_gold_carries_verdicts_only_to_identical_bytes(run: Path, tmp_path: Path) -> None:
    import hashlib

    def sha(doc: str, page: int) -> str:
        path = run / "t0" / "out" / "w0" / doc / "pages" / f"{page:05d}.md"
        return hashlib.sha256(path.read_bytes()).hexdigest()

    gold_path = tmp_path / "gold.jsonl"
    entries = [
        # Judged wrong, still byte-identical, still unmarked SUCCESS: silent.
        {"doc": "doc-a", "page": 1, "md_sha256": sha("doc-a", 1), "content": "defective"},
        # Judged wrong, identical bytes, page carries a mark.
        {"doc": "doc-a", "page": 2, "md_sha256": sha("doc-a", 2), "content": "defective"},
        # Judged on other bytes: needs a fresh read.
        {"doc": "doc-a", "page": 4, "md_sha256": "0" * 64, "content": "clean"},
        # Two seats disagree on the same bytes.
        {"doc": "doc-b", "page": 1, "md_sha256": sha("doc-b", 1), "content": "clean"},
        {"doc": "doc-b", "page": 1, "md_sha256": sha("doc-b", 1), "content": "defective"},
        {"doc": "doc-z", "page": 1, "md_sha256": "1" * 64, "content": "clean"},
    ]
    gold_path.write_text("".join(json.dumps(e) + "\n" for e in entries))
    queue = tmp_path / "queue.jsonl"
    c = scoreboard.score_run(run, None, None, 2, scoreboard.load_gold(gold_path), queue)
    assert c["gold.pages"] == 4
    assert c["gold.pages_not_in_run"] == 1
    assert c["gold.carried.defective_silent"] == 1
    assert c["gold.carried.defective_marked"] == 1  # doc-a p2 carries a visible mark
    assert c["gold.carried.defective_unmarked_non_success"] == 1  # doc-b p1: ERROR, no mark
    assert c["gold.carried.seats_disagree"] == 1
    assert c["gold.changed_needs_review"] == 1
    assert c["gold.changed_needs_review.unmarked_success"] == 1
    assert [json.loads(x)["page"] for x in queue.read_text().splitlines()] == [4]

    outside = scoreboard.score_run(run, {"doc-a"}, None, 2, scoreboard.load_gold(gold_path))
    assert outside["gold.pages_outside_docs"] == 2  # doc-b p1 (two seats) and doc-z p1
    assert outside["gold.pages_not_in_run"] == 0


def test_render_keeps_per_digest_rows_out_of_the_table() -> None:
    from collections import Counter

    out = scoreboard.render(
        [Counter({"records.source_digest.abc": 2, "records.distinct_source_digests": 1})],
        ["run1"],
    )
    assert "records.distinct_source_digests" in out
    assert "source_digest.abc" not in out
