"""#1257 release rule: B1's exact evidence frees a TABLE's record, a PAGE's mark needs independent evidence.

A table bound only by the new evidence keeps its BOUND record.  The page's row-binding mark goes only
when the page text that ships has independent page-scope evidence (``verdict.row_binding_release_evidence``):
for a model reading the words and numbers checks against the layer; for a native text (the layer
itself, so a words check is circular) a digest-verified ``native_coverage`` PASS and no FAIL or UNKNOWN.
Otherwise the table keeps a mark that says what is true.  Every pipeline test is a DIFFERENCE between
two layers of the same page (CLAUDE.md, the no-provider trap) and runs in both provider states.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fitz")

from test_gh1257_row_binding_evidence import (  # noqa: E402
    LABEL_U,
    PITCH,
    WRAP_ABOVE,
    WRAP_ABOVE_MD,
    Y0,
    layout,
    page_of,
    table_md,
)
from test_gh1267_row_binding_pipeline import (  # noqa: E402
    UNVERIFIABLE_KIND,
    UNVERIFIABLE_MARK,
    _fragment,
    _run,
    _sidecar,
)

from socr.core.marks import Check, row_binding_held_table_mark  # noqa: E402
from socr.core.pass_trace import TraceEntry, TraceResult  # noqa: E402
from socr.core.verdict import (  # noqa: E402
    page_text_digest,
    row_binding_release_evidence,
)

PROVIDERS = [True, False]
HELD_PHRASE = "page text not independently confirmed"
#: a word the layer prints that the model did not transcribe: the words check FAILs on it.
OMITTED = "Appendix"


def _pdf(tmp_path: Path, items, name: str) -> Path:
    path = tmp_path / f"{name}.pdf"
    page_of(items, 0).parent.save(path)
    return path


def _items(*, omitted_word: bool):
    items = layout(WRAP_ABOVE)
    if omitted_word:
        items = [*items, (LABEL_U, Y0 + 8 * PITCH, OMITTED)]
    return items


def _arm(tmp_path, providers: bool, *, omitted_word: bool, name: str):
    pdf = _pdf(tmp_path, _items(omitted_word=omitted_word), name)
    _, res, state, out, _ = _run(
        tmp_path,
        None,
        0,
        check_runs=True,
        out_name=name,
        markdown=table_md(WRAP_ABOVE_MD),
        pdf=pdf,
        providers=providers,
    )
    return res, state, out


# ---------------------------------------------------------------------------
# the pipeline: freed with evidence, kept without, both provider states
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("providers", PROVIDERS)
def test_the_mark_goes_only_with_independent_page_evidence(tmp_path, providers):
    res_ok, _, ok = _arm(tmp_path, providers, omitted_word=False, name="ok")
    res_held, state_held, held = _arm(tmp_path, providers, omitted_word=True, name="held")
    if not providers:
        # no provider: the loop bails before routing, no reading, no record; both layers alike
        assert "row_binding" not in _sidecar(ok) and "row_binding" not in _sidecar(held)
        assert res_ok.status == res_held.status
        return
    side_ok, side_held = _sidecar(ok), _sidecar(held)
    # the record is BOUND, with its evidence, in both: B1's evidence is unchanged
    assert side_ok["row_binding"]["p1-t0"]["outcome"] == "bound"
    assert side_held["row_binding"]["p1-t0"]["outcome"] == "bound"
    assert "evidence" in side_ok["row_binding"]["p1-t0"]
    assert "evidence" in side_held["row_binding"]["p1-t0"]
    # freed: no mark, no list, the words check passed at page scope on the shipped text
    assert "row_binding_unverifiable_tables" not in side_ok
    assert HELD_PHRASE not in _fragment(ok)
    assert any(
        e["check"] == "text_layer_words" and e["result"] == "PASS" for e in side_ok["checks"]
    )
    # kept: the table is listed, carries the reworded mark, the page is marked and not SUCCESS
    assert side_held["row_binding_unverifiable_tables"] == [0]
    frag = _fragment(held)
    assert frag.count(row_binding_held_table_mark(1)) == 1
    assert UNVERIFIABLE_MARK not in frag  # it does not claim the binding is unsettled
    assert side_held["winning_output"]["status"] != "success"
    # no unverifiable audit event: the table IS bound; the page surfaces through status and mark
    assert [e for e in side_held["audit_events"] if e["kind"] == UNVERIFIABLE_KIND] == []
    assert state_held.pages[1].row_binding_unverifiable_tables == [0]
    # the table's bytes are identical either way (the rule changes marks, not text)
    table = table_md(WRAP_ABOVE_MD).rstrip("\n")
    assert table in _fragment(ok) and table in frag


@pytest.mark.parametrize("providers", PROVIDERS)
def test_a_held_table_survives_resume(tmp_path, providers):
    """The held mark is re-derived from the restored record and list: a resumed page ships it again."""
    from ocr_output_contract import RootIndex

    pdf = _pdf(tmp_path, _items(omitted_word=True), "res")
    kw = dict(markdown=table_md(WRAP_ABOVE_MD), pdf=pdf, providers=providers)
    _, _, _, out, _ = _run(tmp_path, None, 0, check_runs=True, out_name="res", **kw)
    before = _fragment(out)
    side = _sidecar(out)
    Path(RootIndex(out).path).unlink()  # force the page-level path: the sidecar is the source
    _run(tmp_path, None, 0, check_runs=True, out_name="res", **kw)
    assert _fragment(out) == before
    again = _sidecar(out)
    assert again.get("row_binding_unverifiable_tables") == side.get(
        "row_binding_unverifiable_tables"
    )
    assert again["winning_output"]["status"] == side["winning_output"]["status"]
    if providers:
        assert row_binding_held_table_mark(1) in before


# ---------------------------------------------------------------------------
# the predicate: both page kinds
# ---------------------------------------------------------------------------

TEXT = "A page.\n"
D = page_text_digest(TEXT)


def _pass(check: Check, scope: str = "page") -> TraceEntry:
    return TraceEntry(check.value, TraceResult.PASS, scope, "ok", text_sha=D)


WORDS = _pass(Check.TEXT_LAYER_WORDS)
NUMBERS = _pass(Check.TEXT_LAYER_NUMBERS)
COVERAGE = _pass(Check.NATIVE_COVERAGE)


def _entry(check: Check, result: TraceResult, scope: str = "page") -> TraceEntry:
    return TraceEntry(check.value, result, scope, "x")


@pytest.mark.parametrize(
    ("entries", "released"),
    [
        ([WORDS, NUMBERS], True),
        ([WORDS, _entry(Check.TEXT_LAYER_NUMBERS, TraceResult.NOT_APPLICABLE)], True),
        ([NUMBERS], False),  # no words check
        ([WORDS], False),  # numbers neither passed nor inapplicable
        ([WORDS, _entry(Check.TEXT_LAYER_NUMBERS, TraceResult.NOT_RUN)], False),
        ([WORDS, NUMBERS, _entry(Check.PAGE_JUDGE, TraceResult.UNKNOWN)], False),
        ([WORDS, NUMBERS, _entry(Check.CONTROL_BYTES, TraceResult.FAIL)], False),
        ([_pass(Check.TEXT_LAYER_WORDS, "table_rows"), NUMBERS], False),  # not page scope
        ([TraceEntry(Check.TEXT_LAYER_WORDS.value, TraceResult.PASS, "page", "x"), NUMBERS], False),
        ([COVERAGE], False),  # a model page is not released on ink coverage alone
    ],
)
def test_a_model_page_is_released_on_the_words_and_numbers_checks(entries, released):
    assert row_binding_release_evidence(entries, TEXT, model_read=True) is released


@pytest.mark.parametrize(
    ("entries", "released"),
    [
        ([COVERAGE], True),
        ([COVERAGE, _entry(Check.CONTROL_BYTES, TraceResult.NOT_APPLICABLE)], True),
        ([WORDS, NUMBERS], False),  # circular: the text IS the layer
        ([], False),
        ([COVERAGE, _entry(Check.NATIVE_COVERAGE, TraceResult.FAIL, "region:1")], False),
        ([COVERAGE, _entry(Check.TEXT_LAYER_WORDS, TraceResult.UNKNOWN)], False),
        ([_entry(Check.NATIVE_COVERAGE, TraceResult.NOT_RUN)], False),
        ([_pass(Check.NATIVE_COVERAGE, "region:1")], False),  # not page scope
        ([TraceEntry(Check.NATIVE_COVERAGE.value, TraceResult.PASS, "page", "x")], False),
        (
            [
                TraceEntry(
                    Check.NATIVE_COVERAGE.value, TraceResult.PASS, "page", "x", text_sha="other"
                )
            ],
            False,
        ),
    ],
)
def test_a_native_page_is_released_only_on_ink_coverage(entries, released):
    assert row_binding_release_evidence(entries, TEXT, model_read=False) is released
