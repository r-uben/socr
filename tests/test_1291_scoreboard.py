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
    shared = set(scoreboard.find_documents(run)) & set(
        scoreboard.find_documents(tmp_path / "other")
    )
    c = scoreboard.score_run(run, shared, None, workers=2)
    assert shared == {"doc-a"}
    assert c["coverage.documents"] == 1


def test_duplicate_document_keeps_the_fuller_folder(run: Path) -> None:
    # "a0" sorts before "t0", so the sparser copy is found first and must lose.
    dup = run / "a0" / "out" / "w0" / "doc-a"
    _page(dup, 1, "x\n")
    assert scoreboard.find_documents(run)["doc-a"].parts[-4] == "t0"


def test_text_layer_counts_missing_words_numbers_and_dropped_decimals(tmp_path: Path) -> None:
    pdfs = tmp_path / "pdfs"
    pdfs.mkdir()
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Core inflation fell -.9 percent in 2003")
    pdf.save(pdfs / "doc-t.pdf")
    doc = tmp_path / "run" / "doc-t"
    _page(doc, 1, "inflation fell -9 percent in 2003\n")
    _finish(doc)
    c = scoreboard.score_run(tmp_path / "run", None, pdfs, workers=1)
    assert c["textlayer.words_missing"] == 1  # "core"
    assert c["textlayer.numbers_missing"] == 1  # ".9"
    assert c["textlayer.dropped_decimal_point_values"] == 1
    assert c["textlayer.pages_dropped_decimal_point"] == 1


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
    assert (
        c["gold.carried.defective_marked"] == 2
    )  # doc-a p2, and doc-b p1 (an unmarked ERROR is not SUCCESS)
    assert c["gold.carried.seats_disagree"] == 1
    assert c["gold.changed_needs_review"] == 1
    assert c["gold.changed_needs_review.unmarked_success"] == 1
    assert [json.loads(x)["page"] for x in queue.read_text().splitlines()] == [4]
