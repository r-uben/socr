"""#1185 end to end: a model reading is CHECKED only if its numbers are the page's own.

Measured on the Fed calibration run: 2018 Tealbook p125, a rotated born-digital table, is
read by a model; Korea 2017 Q4 prints ``-.8`` and shipped as ``-8``, judged "faithful".
The fixture here is that page's shape -- rotated text, so the pipeline refuses the native
table lane and routes the page to the model -- with a stub reader whose ``-8`` or ``-.8``
is the ONLY thing that differs between arms.

Hermetic, per ``CLAUDE.md``: the engine registry, the agentic ladder, the page judge and
the judge model are pinned. Every claim is a DIFFERENCE between two arms run in the same
process, or a run against its own re-finalization and resume, never an absolute outcome.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

import socr
from socr.core import verdict as verdict_module
from socr.pipeline.orchestrator import UnifiedPipeline, _page_blob_key

from test_gh1156_part_a_interim import _FixedJudge

#: The page's rows, as the PDF prints them; ``-.8`` is the cell the motivating page lost.
_ROWS = ("Korea 4.0 2.6 5.7 -.8 4.0", "China 6.9 7.0 6.6 6.4 7.1", "Mexico 2.3 1.1 2.5 2.4")


def test_loaded_source_is_this_checkout() -> None:
    assert Path(socr.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1] / "src")


def _rotated_pdf(path: Path, *, pages: int = 2) -> Path:
    """Born-digital pages of rotated table rows: the text layer is the page's own."""
    import fitz

    doc = fitz.open()
    for _ in range(pages):
        page = doc.new_page(width=612, height=792)
        for i, row in enumerate(_ROWS * 6):
            page.insert_text((100 + i * 20, 700), row, fontsize=9, rotate=90)
    doc.save(path)
    doc.close()
    return path


class _Reader:
    """The stub model reading: the page's rows, with Korea's fourth cell as given."""

    name = "qwen"

    def __init__(self, korea_q4: str, pages_read: list[int]) -> None:
        self._korea_q4 = korea_q4
        self._pages_read = pages_read

    def is_available(self) -> bool:
        return True

    def process_pages(self, pdf_path, page_nums, config, dpi, subprocess_timeout=None, **_kw):
        from socr.core.result import PageOutput, PageStatus

        self._pages_read.extend(page_nums)
        text = (
            f"Korea grew 4.0 2.6 5.7 {self._korea_q4} 4.0. China grew 6.9 7.0 6.6 6.4 7.1. "
            "Mexico grew 2.3 1.1 2.5 2.4."
        )
        return [
            PageOutput(page_num=n, text=text, status=PageStatus.SUCCESS, engine="qwen")
            for n in page_nums
        ]


def _run(out_dir: Path, pdf: Path, monkeypatch, *, korea_q4: str, pages_read: list[int]):
    from socr.core.config import EngineType, PipelineConfig
    from socr.core.providers import PROFILE_QWEN_LOCAL
    from socr.pipeline import orchestrator as orch

    with monkeypatch.context() as m:
        m.setattr(orch, "get_engine", lambda engine_type: _Reader(korea_q4, pages_read))
        pipe = UnifiedPipeline(
            PipelineConfig(
                agentic=True,
                quiet=True,
                primary_engine=EngineType.QWEN,
                local_engine=EngineType.QWEN,
                enabled_engines=[EngineType.QWEN],
                native_first=True,
                write_manifest=True,
                judge_backend="heuristic",
                dual_pass_tables=False,
                detect_equations=False,
                save_figures=True,
                describe_figure_crops=False,
            )
        )
        pipe._available_engines_for_agentic = lambda: [PROFILE_QWEN_LOCAL]
        pipe._build_page_judge = lambda state: _FixedJudge(True)
        pipe._resolve_crop_vlm_model = lambda: None
        pipe._resolve_judge_model = lambda *a, **k: ""
        return pipe.process(pdf, output_dir=out_dir)


def _doc_dir(out: Path) -> Path:
    return next(out.rglob("pages")).parent


def _page(out: Path, n: int = 1) -> tuple[dict, str]:
    pages = _doc_dir(out) / "pages"
    return (
        json.loads((pages / f"{n:05d}.json").read_text()),
        (pages / f"{n:05d}.md").read_text(),
    )


def _clauses(sidecar: dict, check: str = "text_layer_numbers") -> list[dict]:
    return [c for c in (sidecar.get("verdict") or {}).get("clauses") or [] if c["check"] == check]


def test_the_reading_that_drops_the_point_is_marked_and_the_faithful_one_is_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The DIFFERENCE this ticket is: one page, one judge that accepts both readings, and
    the one character of difference decides whether the page is marked."""
    arms = {}
    for q4 in ("-.8", "-8"):
        read: list[int] = []
        out = tmp_path / f"out{q4}"
        _run(out, _rotated_pdf(tmp_path / f"p{q4}.pdf"), monkeypatch, korea_q4=q4, pages_read=read)
        assert 1 in read, "fixture premise: the model did not read the page"
        arms[q4] = _page(out)

    faithful, dropped = arms["-.8"], arms["-8"]
    assert _clauses(faithful[0]) == [], faithful[0]["verdict"]
    [clause] = _clauses(dropped[0])
    assert clause["result"] == "FAIL"
    assert "text layer says -.8, reading says -8" in clause["reason"]
    assert dropped[1].startswith(
        "> [page 1 unchecked: text_layer_numbers=FAIL; scope=page; "
        "reason=text layer says -.8, reading says -8]"
    )
    assert "text_layer_numbers" not in faithful[1]
    assert dropped[0]["failure_mode"] == "numbers_disagree_with_text_layer"
    # Every other check saw the same page: the clause is the whole difference.
    others = lambda s: [c for c in s["verdict"]["clauses"] if c["check"] != "text_layer_numbers"]  # noqa: E731
    assert others(faithful[0]) == others(dropped[0])


def _crash_after_page_one(out_dir: Path) -> None:
    """What a crash between page 1's flush and page 2's leaves: see
    ``test_resume_figure_embed_idempotent._crash_after_page_one``."""
    for page_two in out_dir.rglob("pages/00002.*"):
        page_two.unlink()
    for md in out_dir.rglob("*.md"):
        if "pages" not in md.parts:
            md.unlink()


def test_a_disputed_page_is_kept_on_resume_byte_for_byte(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same config re-reads the same numbers, so re-OCR'ing the page on every resume
    would never converge: it is skip-and-kept, and comes back unchanged."""
    pdf = _rotated_pdf(tmp_path / "doc.pdf")
    out = tmp_path / "out"
    _run(out, pdf, monkeypatch, korea_q4="-8", pages_read=[])
    before = _page(out)
    assert _clauses(before[0])[0]["result"] == "FAIL", "fixture premise: page 1 not disputed"
    _crash_after_page_one(out)
    resume: list[int] = []
    _run(out, pdf, monkeypatch, korea_q4="-8", pages_read=resume)
    assert 2 in resume, "the resume did not reach the page loop, so the ledger was not tested"
    assert 1 not in resume, f"the disputed page was re-read on resume: {resume}"
    assert _page(out) == before


#: Fields a run writes from the clock, which no two runs share.
_TIMING_KEYS = ("timings_s", "processing_time", "page_cost_usd")


def _comparable(sidecar: dict) -> dict:
    out = {k: v for k, v in sidecar.items() if k not in _TIMING_KEYS}
    out["winning_output"] = {
        k: v for k, v in (sidecar.get("winning_output") or {}).items() if k not in _TIMING_KEYS
    }
    for key in ("page_fingerprint", "refinalized_from", "socr_version", "socr_source_digest"):
        out.pop(key, None)
    return out


def test_refinalize_then_resume_equals_a_run_of_this_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A finished run made without the check, re-finalized with no OCR, must be what a run
    of this code writes; and a resume of it must keep it byte for byte."""
    from socr.pipeline.refinalize_own_text import refinalize_document

    pdf = _rotated_pdf(tmp_path / "doc.pdf")
    fresh = tmp_path / "fresh"
    _run(fresh, pdf, monkeypatch, korea_q4="-8", pages_read=[])

    old = tmp_path / "old"
    with monkeypatch.context() as m:
        m.setattr(verdict_module, "text_layer_numbers_clause", lambda output, p: None)
        _run(old, pdf, monkeypatch, korea_q4="-8", pages_read=[])
    assert _clauses(_page(old)[0]) == [], "fixture premise: the old run already had the check"

    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    shutil.copy(pdf, pdf_dir / pdf.name)
    changes = refinalize_document(_doc_dir(old), pdf_dir)
    assert {c.page_num for c in changes} == {1, 2}

    for n in (1, 2):
        (re_side, re_body), (fr_side, fr_body) = _page(old, n), _page(fresh, n)
        assert re_body == fr_body
        assert _comparable(re_side) == _comparable(fr_side)
        assert re_side["page_fingerprint"] == _page_blob_key(re_side["winning_output"])
    doc_md = lambda out: next(p for p in _doc_dir(out).glob("*.md")).read_text()  # noqa: E731
    assert doc_md(old) == doc_md(fresh)
    manifest = json.loads((_doc_dir(old) / "manifest.json").read_text())
    for n in (1, 2):
        ref = manifest["entries"][str(n)]["blob_ref"]
        assert _page(old, n)[0]["page_fingerprint"].endswith(ref)
        assert (_doc_dir(old) / "cache" / ref[:2] / f"{ref}.json").is_file()
    meta = lambda out: json.loads((_doc_dir(out) / "metadata.json").read_text())  # noqa: E731
    assert meta(old).get("tables") == meta(fresh).get("tables")

    # Idempotent: a second pass changes nothing.
    assert refinalize_document(_doc_dir(old), pdf_dir) == []

    # Re-finalize, then resume: the re-finalized page survives the ledger unchanged.
    before = _page(old)
    _crash_after_page_one(old)
    resume: list[int] = []
    _run(old, pdf, monkeypatch, korea_q4="-8", pages_read=resume)
    assert 2 in resume and 1 not in resume, resume
    after = _page(old)
    differ = sorted(
        k for k in set(before[0]) | set(after[0]) if before[0].get(k) != after[0].get(k)
    )
    assert after[1] == before[1], "the restored body changed"
    assert after[0] == before[0], f"the restored sidecar changed; differing keys: {differ}"
    assert doc_md(old) == doc_md(fresh)


def _levels(out: Path, result) -> dict:
    """The disagreement at every level a reader of the run looks at."""
    from socr.core.result import contract_status_for

    sidecar = _page(out)[0]
    doc = json.loads((_doc_dir(out) / "metadata.json").read_text())
    root = json.loads((out / "metadata.json").read_text())
    [entry] = root["files"].values()
    return {
        "page": (sidecar["status"], sidecar["failure_mode"]),
        "document": result.status.value,
        "metadata": doc["status"],
        "root index": entry["status"],
        "cli": contract_status_for(result).value,
        "error note": "page(s) 1, 2" in (result.error or ""),
        "audit log": json.loads((_doc_dir(out) / "audit_log.json").read_text())["counts"].get(
            "numbers_disagree_with_text_layer", 0
        ),
    }


def test_a_disagreement_reaches_the_document_the_root_index_and_the_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Failures surface at every level (Fable, PR #1219 review P1-A): the one character of
    difference must change the page, the document, its metadata, the run's root index and
    the CLI's verdict -- and nothing else between the arms."""
    levels = {}
    for q4 in ("-.8", "-8"):
        out = tmp_path / f"out{q4}"
        pdf = _rotated_pdf(tmp_path / f"p{q4}.pdf")
        levels[q4] = _levels(out, _run(out, pdf, monkeypatch, korea_q4=q4, pages_read=[]))
    faithful, dropped = levels["-.8"], levels["-8"]
    assert {k for k in faithful if faithful[k] != dropped[k]} == set(faithful), (faithful, dropped)
    assert dropped["page"] == ("warning", "numbers_disagree_with_text_layer")
    assert (faithful["document"], dropped["document"]) == ("success", "audit_failed")
    for level in ("metadata", "root index", "cli"):
        assert (faithful[level], dropped[level]) == ("completed", "partial"), level
    assert (faithful["error note"], dropped["error note"]) == (False, True)
    assert (faithful["audit log"], dropped["audit log"]) == (0, 2)


def test_refinalize_carries_a_disagreement_to_the_document_and_the_root_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The tool's half of P1-A: a re-finalized FAIL turns the document's recorded status and
    the root index's into what a run of this code records, with no OCR."""
    from socr.pipeline.refinalize_own_text import refinalize_document

    pdf = _rotated_pdf(tmp_path / "doc.pdf")
    fresh, old = tmp_path / "fresh", tmp_path / "old"
    _run(fresh, pdf, monkeypatch, korea_q4="-8", pages_read=[])
    with monkeypatch.context() as m:
        m.setattr(verdict_module, "text_layer_numbers_clause", lambda output, p: None)
        _run(old, pdf, monkeypatch, korea_q4="-8", pages_read=[])

    def recorded(out: Path) -> tuple[dict, dict]:
        doc = json.loads((_doc_dir(out) / "metadata.json").read_text())
        [entry] = json.loads((out / "metadata.json").read_text())["files"].values()
        return doc, entry

    assert recorded(old)[0]["status"] == "completed", "fixture premise: old run was clean"
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    shutil.copy(pdf, pdf_dir / pdf.name)
    refinalize_document(_doc_dir(old), pdf_dir)
    keys = ("status", "error")
    for got, want in zip(recorded(old), recorded(fresh)):
        assert {k: got.get(k) for k in keys} == {k: want.get(k) for k in keys}
    audit = lambda out: json.loads((_doc_dir(out) / "audit_log.json").read_text())  # noqa: E731
    assert audit(old)["counts"] == audit(fresh)["counts"]
    assert audit(old)["counts"].get("numbers_disagree_with_text_layer") == 2
    # A second pass adds nothing anywhere.
    snapshot = {p: p.read_bytes() for p in old.rglob("*") if p.is_file()}
    refinalize_document(_doc_dir(old), pdf_dir)
    assert {p: p.read_bytes() for p in old.rglob("*") if p.is_file()} == snapshot
