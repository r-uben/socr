"""GH-993: readable tables per paper, derived from what socr already records.

The end-to-end cases drive ``process()`` hermetically on the GH-359 harness (ruled-grid
PDF, scripted table-judge rungs, provider and judge-model patched) and reach each class
of table through the real ladder: PASS (verified text), a reader FAIL with a mismatching
blind adjudicator (WITHHELD: marker only), an infra failure (UNVERIFIED text kept).
No case pins an outcome measured on one machine: each asserts either the metadata
against the artifacts the same run wrote, or a DIFFERENCE between two runs that change
exactly one thing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_gh359_ladder_terminals import (
    _fail,
    _make_config,
    _not_s1,
    _pass,
    _process,
    _QueueRung,
    _ruled_pdf,
)
from test_gh359_ladder_terminals import _TABLE_MD

from socr.core.table_counts import TableCounts, count_from_sidecars, count_page_tables
from socr.pipeline.orchestrator import UnifiedPipeline


def _run(tmp_path: Path, result, *, quiet: bool = True):
    pdf = _ruled_pdf(tmp_path, "doc.pdf")
    pipeline = UnifiedPipeline(_make_config(quiet=quiet))
    # The native scoring diagnostic (``table_not_scorable``) is a property of the ruled
    # fixture's geometry, not of the ladder verdict under test; left on, it would flag
    # every page and nothing could be "verified". It is a trust event like any other,
    # so a flagged-page case is covered separately on ``count_page_tables``.
    pipeline._surface_table_scoring = lambda *args, **kwargs: None
    out = tmp_path / "out"
    _process(pipeline, pdf, out, [_QueueRung([result])])
    return out / "doc"


def _tables(doc_dir: Path) -> dict:
    return json.loads((doc_dir / "metadata.json").read_text(encoding="utf-8"))["tables"]


def test_verified_withheld_and_unverified_tables_are_counted(tmp_path: Path) -> None:
    verified = _tables(_run(tmp_path / "v", _pass("high")))
    withheld = _tables(_run(tmp_path / "w", _fail()))
    unverified = _tables(_run(tmp_path / "u", _not_s1()))

    assert verified == {"shipped_text": 1, "verified_text": 1, "unverified_text": 0, "withheld": 0}
    assert withheld == {"shipped_text": 0, "verified_text": 0, "unverified_text": 0, "withheld": 1}
    assert unverified == {
        "shipped_text": 1,
        "verified_text": 0,
        "unverified_text": 1,
        "withheld": 0,
    }


def test_withholding_one_more_table_moves_the_counts_by_exactly_one(tmp_path: Path) -> None:
    """Difference pin: same document, only the reader verdict changes."""
    kept = _tables(_run(tmp_path / "kept", _pass("high")))
    cut = _tables(_run(tmp_path / "cut", _fail()))
    assert cut["withheld"] - kept["withheld"] == 1, (kept, cut)
    assert kept["shipped_text"] - cut["shipped_text"] == 1, (kept, cut)
    assert kept["verified_text"] - cut["verified_text"] == 1, (kept, cut)


@pytest.mark.parametrize(
    "result", [_pass("high"), _fail(), _not_s1()], ids=["verified", "withheld", "unverified"]
)
def test_metadata_block_matches_what_an_independent_reader_derives(tmp_path: Path, result) -> None:
    """The block equals the counts derived from the sidecars and the trust file."""
    doc_dir = _run(tmp_path, result)
    derived = count_from_sidecars(doc_dir)
    assert derived is not None and _tables(doc_dir) == derived.to_dict()


def test_root_index_keeps_the_contract_shape(tmp_path: Path) -> None:
    doc_dir = _run(tmp_path, _pass("high"))
    root = json.loads((doc_dir.parent / "metadata.json").read_text(encoding="utf-8"))
    assert "tables" not in json.dumps(root)


def test_cli_line_is_printed_once_per_document(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    capsys.readouterr()
    _run(tmp_path, _pass("high"), quiet=False)
    lines = [ln for ln in capsys.readouterr().out.splitlines() if "tables:" in ln]
    assert [ln.strip() for ln in lines] == ["tables: 1 as text (1 verified), 0 withheld"], lines


def test_final_body_guard_withholding_is_counted(tmp_path: Path) -> None:
    """A table the post-figure body guard withholds must be in the metadata counts.

    The counts are first derived from the pre-figure records, which hold a clean page;
    only the re-derivation from the final records sees the invalid-emission marker.
    """
    from unittest.mock import patch

    import fitz
    from ocr_output_contract import assemble_pages

    from socr.core.config import EngineType, PipelineConfig
    from socr.core.document import DocumentHandle
    from socr.core.result import PageOutput, PageStatus
    from socr.core.state import DocumentState

    pdf_path = tmp_path / "paper.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((60, 60), "Clean native page text for final-body testing.")
    doc.save(pdf_path)
    doc.close()
    state = DocumentState(handle=DocumentHandle.from_path(pdf_path))
    output = PageOutput(
        page_num=1,
        text="Clean selected page text.",
        status=PageStatus.SUCCESS,
        engine="native",
        audit_passed=True,
    )
    state.pages[1].attempts.append(output)
    state.pages[1].best_output = output
    invalid_final = assemble_pages(
        [r"| A | \multicolumn{2}{c}{B} |" + "\n| --- | --- |\n| 1 | 2 |"]
    )
    pipeline = UnifiedPipeline(
        PipelineConfig(
            save_figures=True,
            write_manifest=True,
            quiet=True,
            primary_engine=EngineType.QWEN,
            local_engine=EngineType.QWEN,
            enabled_engines=[EngineType.QWEN],
            judge_backend="heuristic",
        )
    )
    out_dir = tmp_path / "out"
    with patch.object(pipeline, "_describe_and_embed_figures", return_value=invalid_final):
        pipeline._phase_assemble(state, out_dir)

    block = json.loads((out_dir / "paper" / "metadata.json").read_text())["tables"]
    assert block["withheld"] == 1 and block["shipped_text"] == 0, block


# ---------------------------------------------------------------------------
# Per-page derivation: each class of table, with exact expected counts.
# ---------------------------------------------------------------------------

_MARKER = "[page 2 failed: unverifiable table — see image]\n\n![Failed table page 2](f.png)"


def test_page_classes_are_counted_exactly() -> None:
    t = _TABLE_MD
    assert count_page_tables(t, "success", "none") == TableCounts(1, 1, 0, 0)
    # WARNING with text kept: shipped, not verified.
    assert count_page_tables(t, "warning", "table_rejected") == TableCounts(1, 0, 0, 0)
    assert count_page_tables(t, "warning", "table_unverified") == TableCounts(1, 0, 1, 0)
    # A SUCCESS page carrying a distrust flag is not verified.
    flagged = count_page_tables(t, "success", "none", trust_reasons=["native_table_verifier_warn"])
    assert flagged == TableCounts(1, 0, 0, 0)
    assert count_page_tables(_MARKER, "error", "table_withheld") == TableCounts(0, 0, 0, 1)
    # Regional splice: prose, one table kept as text, one withheld.
    assert count_page_tables(f"prose\n\n{t}\n\n{_MARKER}", "error", "x") == TableCounts(1, 0, 0, 1)
    # Prose is not a table.
    assert count_page_tables("just words | with a pipe", "success", "none") == TableCounts()


def test_prose_recovery_page_withholds_at_least_one_not_one_per_run() -> None:
    from socr.core.manifest import SCANNED_PROSE_RECOVERED_FLAG

    marker = "[page 4 failed: unverifiable table — see image]"
    text = "\n\n".join(
        [SCANNED_PROSE_RECOVERED_FLAG.format(page_num=4), "words", marker, "more words", marker]
    )
    assert count_page_tables(text, "error", "x").withheld == 1


def test_sidecar_reader_returns_none_without_sidecars(tmp_path: Path) -> None:
    assert count_from_sidecars(tmp_path) is None


# ---------------------------------------------------------------------------
# socr library: per-paper entry and corpus total
# ---------------------------------------------------------------------------


def _library_with_three_documents(tmp_path: Path):
    from test_gh964_library import _fake_doc, _pdf, _write_cfg

    from socr import library as lib

    cfg = lib.load_library_config(_write_cfg(tmp_path))
    for stem in ("recorded", "legacy", "bare"):
        _pdf(cfg.pdf_dir / f"{stem}.pdf")
    # 1. written by this pipeline version: the block is in metadata.json
    rec = _fake_doc(cfg.text_dir, "recorded")
    meta = json.loads((rec / "metadata.json").read_text())
    meta["tables"] = {"shipped_text": 4, "verified_text": 3, "unverified_text": 1, "withheld": 2}
    (rec / "metadata.json").write_text(json.dumps(meta))
    # 2. written before the block existed: derived from the sidecar
    old = _fake_doc(cfg.text_dir, "legacy")
    (old / "pages" / "00000.json").write_text(
        json.dumps(
            {
                "page_num": 1,
                "status": "success",
                "failure_mode": "none",
                "winning_output": {"page_num": 1, "text": _TABLE_MD},
            }
        )
    )
    # 3. nothing to derive from: not recorded, which is not zero
    _fake_doc(cfg.text_dir, "bare")
    return lib, cfg


def test_manifest_has_a_per_paper_tables_entry_and_summary_a_corpus_total(tmp_path: Path) -> None:
    lib, cfg = _library_with_three_documents(tmp_path)
    summary = lib.refresh_index(cfg)
    docs = json.loads(cfg.manifest.read_text())["documents"]

    assert docs["recorded"]["tables"] == {
        "shipped_text": 4,
        "verified_text": 3,
        "unverified_text": 1,
        "withheld": 2,
    }
    assert docs["legacy"]["tables"] == {
        "shipped_text": 1,
        "verified_text": 1,
        "unverified_text": 0,
        "withheld": 0,
    }
    assert docs["bare"]["tables"] is None
    # The total is over the documents that have counts; the denominator is reported.
    assert summary["tables"] == {
        "shipped_text": 5,
        "verified_text": 4,
        "unverified_text": 1,
        "withheld": 2,
    }
    assert summary["tables_recorded_documents"] == 2
