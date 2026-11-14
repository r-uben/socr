"""#1183 C2 hardening (carried into C3): what the C2 re-checks left open.

* a ``row_binding`` PASS outside ``table:N`` is NOT_RUN;
* a digest key is written for a PASS only;
* the scoreboard counts a PASS that is not tied to the sidecar's own winning text;
* after assemble the page fragment's body is the text every page-scope PASS was computed on.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_gh1294_layer_words import _rotated_pdf, _run

import socr
from socr.core.pass_trace import TraceEntry, TraceResult, trace_to_dicts
from socr.core.verdict import (
    ROW_BINDING_CHECK,
    ROW_BINDING_OFF_TABLE_REASON,
    page_text_digest,
    verify_pass_digests,
)


def test_loaded_source_is_this_checkout() -> None:
    assert Path(socr.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1] / "src")


TEXT = "| a | b |\n| - | - |\n| Korea | 4.0 |\n"


@pytest.mark.parametrize("scope", ["page", "figure:1", "table", "table:x"])
def test_a_row_binding_pass_outside_a_table_scope_is_not_run(scope: str) -> None:
    digest = page_text_digest(TEXT)
    bad = TraceEntry(ROW_BINDING_CHECK, TraceResult.PASS, scope, "", text_sha=digest)
    [got] = verify_pass_digests([bad], TEXT)
    assert got.result is TraceResult.NOT_RUN and got.reason == ROW_BINDING_OFF_TABLE_REASON
    # The same digest on another check at page scope is a PASS: only row binding is table-bound.
    ok = TraceEntry("text_layer_words", TraceResult.PASS, "page", "", text_sha=digest)
    assert verify_pass_digests([ok], TEXT)[0].result is TraceResult.PASS


@pytest.mark.parametrize("result", [r for r in TraceResult if r is not TraceResult.PASS])
def test_only_a_pass_writes_a_digest_key(result: TraceResult) -> None:
    e = TraceEntry("x", result, "page", "r", text_sha="sha256:a", table_sha="sha256:b")
    assert set(e.to_dict()) == {"check", "result", "scope", "reason"}
    p = TraceEntry("x", TraceResult.PASS, "page", "", text_sha="sha256:a", table_sha="sha256:b")
    assert {"text_sha", "table_sha"} <= set(p.to_dict())


def _sidecar(checks: list[dict], text: str | None) -> dict:
    return {
        "page_num": 1,
        "status": "success",
        "engine": "qwen",
        "checks": checks,
        "winning_output": {} if text is None else {"text": text},
    }


def _count(tmp_path: Path, sidecars: list[dict]):
    from socr.benchmark import scoreboard

    pages = tmp_path / "doc" / "pages"
    pages.mkdir(parents=True)
    for n, sc in enumerate(sidecars, start=1):
        sc = {**sc, "page_num": n}
        (pages / f"{n:05d}.json").write_text(json.dumps(sc))
        (pages / f"{n:05d}.md").write_text("x")
    return scoreboard.count_document(tmp_path / "doc")


def test_the_scoreboard_counts_a_pass_not_tied_to_the_sidecars_own_text(tmp_path: Path) -> None:
    good = trace_to_dicts(
        [
            TraceEntry(
                "text_layer_words", TraceResult.PASS, "page", "", text_sha=page_text_digest(TEXT)
            )
        ]
    )
    stale = trace_to_dicts(
        [
            TraceEntry(
                "text_layer_words", TraceResult.PASS, "page", "", text_sha=page_text_digest("old")
            )
        ]
    )
    bare = trace_to_dicts([TraceEntry("text_layer_words", TraceResult.PASS, "page", "")])
    not_pass = trace_to_dicts([TraceEntry("text_layer_words", TraceResult.NOT_RUN, "page", "r")])
    counts = _count(
        tmp_path,
        [
            _sidecar(good, TEXT),
            _sidecar(stale, TEXT),
            _sidecar(bare, TEXT),
            _sidecar(good, None),
            _sidecar(not_pass, TEXT),
            _sidecar([], TEXT),
        ],
    )
    assert counts["records.pages_pass_digest_mismatch"] == 3


@pytest.mark.parametrize("ladder", [True, False], ids=["provider", "no provider"])
def test_after_assemble_every_page_scope_pass_names_the_fragment_body_and_the_sidecar_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ladder: bool
) -> None:
    """The fragment is the shipped text (marks included): ``_rewrite_all_fragments`` writes
    ``record.output.text`` and the sidecar's ``winning_output.text`` is the same string, so a PASS
    about the sidecar text is a PASS about the page the reader gets."""
    out = tmp_path / "out"
    _run(
        out,
        _rotated_pdf(tmp_path / "doc.pdf"),
        monkeypatch,
        lose_label=False,
        pages_read=[],
        ladder=ladder,
    )
    pages = next(out.rglob("pages"))
    seen = 0
    for sidecar_path in sorted(pages.glob("*.json")):
        sidecar = json.loads(sidecar_path.read_text())
        fragment = sidecar_path.with_suffix(".md").read_text()
        assert sidecar["winning_output"]["text"] == fragment
        for e in sidecar["checks"]:
            if e["result"] == "PASS" and e["scope"] == "page":
                seen += 1
                assert e["text_sha"] == page_text_digest(fragment)
    if ladder:
        assert seen, "fixture premise: some page-scope PASS exists to compare"
