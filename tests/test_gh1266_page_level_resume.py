"""#1266: a PAGE-level resume reproduces both floor endings byte for byte.

A second ``process()`` over an untouched output directory is the DOCUMENT-level resume skip
(``_resume_skip`` -> ``DocumentStatus.SKIPPED``): it touches no file and so cannot fail
(#1267 merge gate, Fable P1). This test forces the page-level path the way Fable's probe did:
the root index (``RootIndex(out).path``, the out-root ``metadata.json``) and ONE sibling
page's fragment and sidecar (the ordinary prose page) are deleted, so the document gate
reopens the document and every page goes through ``_load_terminal_page``.

One five-page PDF carries every floor shape: (1) an ending-A structure-class floor page that
RECOVERS prose, (2) an ending-A WHOLE-PAGE floor page, (3) a D3 page that RECOVERS, (4) a D3
WHOLE-PAGE page (TR-3 forced to hard-fail on pages 3 and 4 only), (5) an ordinary prose page.

Restored or reprocessed, measured on 4feb7781 and asserted below: EVERY floor page is
REPROCESSED live on the page-level resume; none is restored. ``_load_terminal_page`` refuses
all four: the ending-A floor modes are named in ``floor_shipped`` (orchestrator, the
content-terminal clause), which FORFEITS the content-terminal exception -- a floored page is
never skippable -- and D3 ships ERROR / ``NATIVE_TABLE_STRUCTURE_FAILED``, which fails the
SUCCESS gate. So the identity proven here is live recomputation against the run-1 bytes on
disk, through the same seam (``_rewrite_all_fragments``) that writes every fragment again.

Provider states, each where its ending is reachable (measured):

* ``ladder-ocr`` (populated ladder, OCR route): pages 1-2 end-A (page-judge-timeout floor),
  pages 3-4 D3.
* ``empty-ocr`` (empty ladder, OCR route): pages 1-4 all end-A (structure-class floor; with
  no ``native_first`` the providerless loop never sets ``native_table_structure_failed``).
* ``empty-nf`` (empty ladder, ``native_first``): pages 3-4 D3 (native-table-first refusal);
  pages 1-2 are not floors there and are not asserted.
"""

from __future__ import annotations

import dataclasses
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_gh1266_floor_geometry import _gated_process_pdf, _process_pdf  # noqa: E402

from socr.core.result import FailureMode  # noqa: E402

BANNER_PREFIX = "unverified extraction"
MARKER_PREFIX = "failed: unverifiable table"
PROSE_PAGE = 5
#: 0-based page numbers whose table regions the TR-3 verifier is forced to hard-fail.
TR3_FAILED_PAGES = frozenset({2, 3})
RECOVERS = {1: True, 2: False, 3: True, 4: False}
ENDING_A_MODES = {
    FailureMode.STRUCTURE_CLASS_LADDER_EXHAUSTED.value,
    FailureMode.PAGE_JUDGE_TIMEOUT.value,
    FailureMode.ROW_SHAPE_NOT_RECONCILABLE_TEXT_TABLE.value,
}
D3_MODE = FailureMode.NATIVE_TABLE_STRUCTURE_FAILED.value
ROUTES = {
    "ladder-ocr": (True, False, {1: "A", 2: "A", 3: "D3", 4: "D3"}),
    "empty-ocr": (False, False, {1: "A", 2: "A", 3: "A", 4: "A"}),
    "empty-nf": (False, True, {3: "D3", 4: "D3"}),
}
#: Round 3 (Astra P2): the floor credential and its digest are compared too.
WINNER_KEYS = (
    "scanned_prose_recovered",
    "audit_notes",
    "text",
    "table_floor_body",
    "table_floor_digest",
)


@pytest.fixture
def tr3_fails_on_pages_3_and_4(monkeypatch):
    from socr.tables import native_verifier

    real = native_verifier.verify_native_table_region

    def _verify(page, text, rect):
        result = real(page, text, rect)
        if page.number in TR3_FAILED_PAGES:
            result.hard_fail = True
            result.reason = "forced by the #1266 page-level resume test"
        return result

    monkeypatch.setattr(native_verifier, "verify_native_table_region", _verify)


def _five_page_pdf(tmp_path: Path) -> Path:
    import fitz

    parts = [
        _process_pdf(tmp_path / "a.pdf"),
        _gated_process_pdf(tmp_path / "b.pdf"),
        _process_pdf(tmp_path / "c.pdf"),
        _gated_process_pdf(tmp_path / "d.pdf"),
    ]
    doc = fitz.open()
    for part in parts:
        with fitz.open(str(part)) as src:
            doc.insert_pdf(src)
    doc.new_page().insert_text((72, 72), "An ordinary page of prose closes the document.")
    path = tmp_path / "doc.pdf"
    doc.save(str(path))
    doc.close()
    return path


def _process(pdf: Path, out: Path, monkeypatch, *, ladder: bool, native_first: bool):
    """One hermetic ``process()``: engines, judge model and provider ladder pinned."""
    from test_gh1144_prose_survives_the_table_floor import _GridEngine, _pipeline

    from socr.core.providers import PROFILE_QWEN_LOCAL
    from socr.pipeline import orchestrator as orch

    monkeypatch.setattr(orch, "get_engine", lambda engine_type: _GridEngine())
    pipe = _pipeline()
    pipe.config = dataclasses.replace(pipe.config, native_first=native_first)
    pipe._available_engines_for_agentic = lambda: [PROFILE_QWEN_LOCAL] if ladder else []
    return pipe.process(pdf, output_dir=out)


def _sections(doc_md: str) -> dict[int, str]:
    """The document .md split into its ``## Page N`` sections."""
    parts = re.split(r"(?m)^## Page (\d+)\s*$", doc_md)
    return {int(parts[i]): parts[i + 1] for i in range(1, len(parts) - 1, 2)}


def _snapshot(out: Path) -> tuple[Path, dict[int, tuple[str, dict]], Path, str]:
    pages = next(iter(out.rglob("pages/00001.md"))).parent
    snap = {
        n: (
            (pages / f"{n:05d}.md").read_text(),
            json.loads((pages / f"{n:05d}.json").read_text()),
        )
        for n in range(1, PROSE_PAGE + 1)
    }
    finals = [p for p in out.rglob("*.md") if p.parent.name != "pages"]
    assert len(finals) == 1, finals
    return pages, snap, finals[0], finals[0].read_text()


@pytest.mark.parametrize("route", sorted(ROUTES))
def test_a_page_level_resume_reproduces_every_floor_page_byte_for_byte(
    tmp_path: Path, monkeypatch, tr3_fails_on_pages_3_and_4, route: str
) -> None:
    from ocr_output_contract import RootIndex

    from socr.core.result import DocumentStatus
    from socr.pipeline.orchestrator import UnifiedPipeline

    ladder, native_first, endings = ROUTES[route]
    pdf = _five_page_pdf(tmp_path)
    out = tmp_path / "out"

    # run 1
    _process(pdf, out, monkeypatch, ladder=ladder, native_first=native_first)
    pages, live, final_path, final_md = _snapshot(out)
    live_sections = _sections(final_md)
    for n, ending in endings.items():
        frag, meta = live[n]
        expected = ENDING_A_MODES if ending == "A" else {D3_MODE}
        assert meta["failure_mode"] in expected, f"premise p{n}: {meta['failure_mode']}"
        assert meta["status"] == "error", f"premise p{n}"
        assert (BANNER_PREFIX in frag) is RECOVERS[n], f"premise p{n}: recovery shape"
        assert bool(meta["winning_output"].get("scanned_prose_recovered")) is RECOVERS[n]
        assert MARKER_PREFIX in frag
        assert n in live_sections, f"premise p{n}: no section in the document .md"

    # force the PAGE-level path: the root index plus one sibling page's artifacts
    root_index = RootIndex(out).path
    assert root_index.exists(), "premise: the root index exists after run 1"
    root_index.unlink()
    (pages / f"{PROSE_PAGE:05d}.md").unlink()
    (pages / f"{PROSE_PAGE:05d}.json").unlink()

    calls: Counter = Counter()
    restored: list[int] = []
    real_load = UnifiedPipeline._load_terminal_page
    real_restore = UnifiedPipeline._restore_terminal_page_state

    def _load(self, state, page_num, output_dir):
        calls[page_num] += 1
        return real_load(self, state, page_num, output_dir)

    def _restore(self, state, page_num, page_out, output_dir):
        restored.append(page_num)
        return real_restore(self, state, page_num, page_out, output_dir)

    monkeypatch.setattr(UnifiedPipeline, "_load_terminal_page", _load)
    monkeypatch.setattr(UnifiedPipeline, "_restore_terminal_page_state", _restore)
    result = _process(pdf, out, monkeypatch, ladder=ladder, native_first=native_first)
    monkeypatch.setattr(UnifiedPipeline, "_load_terminal_page", real_load)
    monkeypatch.setattr(UnifiedPipeline, "_restore_terminal_page_state", real_restore)

    # premises: the page-level path ran, and how each floor page came back
    assert result.status is not DocumentStatus.SKIPPED, "run 2 was the document-level skip"
    for n in endings:
        assert calls[n] >= 1, f"_load_terminal_page never consulted for p{n}: {dict(calls)}"
        assert n not in restored, f"p{n} was restored; the docstring says reprocessed"

    # byte identity for every floor page, and the deleted sibling regenerated
    pages2, again, final2_path, final2_md = _snapshot(out)
    assert pages2 == pages and final2_path == final_path
    sections2 = _sections(final2_md)
    for n in endings:
        frag, meta = live[n]
        frag2, meta2 = again[n]
        assert frag2 == frag, f"p{n}: fragment changed across a page-level resume"
        for key in ("status", "failure_mode"):
            assert meta2[key] == meta[key], f"p{n}: {key}"
        for key in WINNER_KEYS:
            assert meta2["winning_output"].get(key) == meta["winning_output"].get(key), (
                f"p{n}: winning_output.{key}"
            )
        assert sections2[n] == live_sections[n], f"p{n}: document .md section changed"
    frag5, meta5 = live[PROSE_PAGE]
    assert again[PROSE_PAGE][0] == frag5, "the deleted sibling fragment was not regenerated"
    assert again[PROSE_PAGE][1]["status"] == meta5["status"]
    assert final2_md == final_md
