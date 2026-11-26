"""#1409: ``socr-refinalize-second-reader`` on a synthetic finished run. No model, no network.

Every pipeline test here runs in BOTH provider states and pins a DIFFERENCE, never the tuple
one machine produced: the tool is run on copies of ONE finished run, changing only the second
reading (one that agrees, one that disputes a number), and the copies are compared. With a
provider the run's page ships unmarked and is selected; without one it ships marked and must
not be touched, whatever the reading says. Which of the two a run is, is read off the run.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import pytest

pytest.importorskip("fitz")

import fitz  # noqa: E402
from test_gh1267_row_binding_check import GROUP  # noqa: E402
from test_gh1267_row_binding_pipeline import _run  # noqa: E402

from socr.core.marks import (  # noqa: E402
    SCOPE_PAGE,
    Check,
    CheckResult,
    page_unchecked_clause_mark,
    parse_canonical_mark,
)
from socr.core.result import FailureMode  # noqa: E402
from socr.pipeline import refinalize_second_reader as tool  # noqa: E402

MODEL = "second-reader-under-test"


@pytest.fixture(autouse=True)
def _no_words_check(monkeypatch):
    """As in the other re-finalize suites: the stub reading carries only part of the page."""
    from socr.core import verdict

    monkeypatch.setattr(verdict, "page_missing_words", lambda output, p: None)


def _tree(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _doc_dir(out: Path) -> Path:
    (pages,) = [p for p in out.rglob("pages") if p.is_dir()]
    return pages.parent


def _sidecar(out: Path) -> dict:
    return json.loads((_doc_dir(out) / "pages" / "00001.json").read_text(encoding="utf-8"))


def _fragment(out: Path) -> str:
    return (_doc_dir(out) / "pages" / "00001.md").read_text(encoding="utf-8")


def _meta(out: Path) -> dict:
    return json.loads((_doc_dir(out) / "metadata.json").read_text(encoding="utf-8"))


class Reader:
    """A fake second reader: returns one text, counts its calls."""

    def __init__(self, text: str, finish_reason: str | None = "stop"):
        self.text, self.finish_reason, self.calls = text, finish_reason, 0

    def __call__(self, png: bytes) -> tuple[str, str | None]:
        assert png.startswith(b"\x89PNG")
        self.calls += 1
        return self.text, self.finish_reason


def _forbidden(png: bytes):
    raise AssertionError("the reader must not be called")


def _apply(out: Path, pdf_dir: Path, reader, *, apply: bool = True, readings_dir=None) -> list:
    skipped: Counter = Counter()
    return tool.refinalize_document(
        _doc_dir(out),
        pdf_dir,
        apply=apply,
        readings_dir=readings_dir,
        read_page=reader,
        reader_model=MODEL,
        prompt_sha256="sha256:prompt",
        skipped=skipped,
    )


@pytest.fixture(params=[True, False], ids=["provider", "no-provider"])
def run(request, tmp_path):
    """``(out_dir, pdf_dir, copy)``: one finished run and a function that copies it."""
    _, _, _, out, pdf = _run(
        tmp_path, GROUP, 0, check_runs=True, out_name="run", providers=request.param
    )

    def copy(name: str) -> Path:
        return Path(shutil.copytree(out, tmp_path / name))

    return out, pdf.parent, copy


def _agreeing(out: Path) -> str:
    return _fragment(out)


def _disputing(out: Path) -> str:
    """The page's own text with the minus the page does not print on one number (8.8)."""
    text = _fragment(out)
    assert text.count(" 8.8") == 1
    return text.replace(" 8.8", " -8.8")


# --- the difference a reading makes ------------------------------------------------------


def test_dry_run_writes_nothing_and_calls_no_reader(run):
    out, pdf_dir, _copy = run
    before = _tree(out)
    _apply(out, pdf_dir, _forbidden, apply=False)
    assert _tree(out) == before


def test_an_agreeing_reading_leaves_every_file_of_the_run_as_it_was(run):
    out, pdf_dir, _copy = run
    before = _tree(out)
    selected = not tool.selects(_sidecar(out))
    reader = Reader(_agreeing(out))
    results = _apply(out, pdf_dir, reader)
    after = _tree(out)
    assert {name: after[name] for name in before} == before
    doc = _doc_dir(out).name
    new = set(after) - set(before)
    if selected:
        assert [r.outcome for r in results] == ["agree"]
        assert reader.calls == 1
        assert new == {f"{doc}/{tool.LOG_NAME}", f"{doc}/{tool.READINGS_DIR}/00001.json"}
    else:
        assert (results, reader.calls, new) == ([], 0, set())


def test_a_disputing_reading_marks_the_page_and_changes_nothing_else_about_it(run):
    out, pdf_dir, copy = run
    agree, dispute = copy("agree"), copy("dispute")
    selected = not tool.selects(_sidecar(out))
    _apply(agree, pdf_dir, Reader(_agreeing(out)))
    results = _apply(dispute, pdf_dir, Reader(_disputing(out)))
    if not selected:
        # A page the run already marked: the reading cannot matter.
        assert _tree(dispute) == _tree(agree) == _tree(out)
        assert results == []
        return
    (result,) = results
    assert result.outcome == "disagree"
    old, new = _fragment(agree), _fragment(dispute)
    assert _fragment(agree) == _fragment(out)
    # The shipped text is the run's text under the new mark, and nothing else.
    assert new == f"{result.mark}\n\n{old}"
    line = result.mark
    assert parse_canonical_mark(line, 1)[0] is Check.SECOND_READER_NUMBERS
    assert "second_reader_numbers=FAIL" in line and "-8.8" in line
    side, was = _sidecar(dispute), _sidecar(agree)
    assert was == _sidecar(out)
    assert (side["status"], side["failure_mode"], side["audit_passed"]) == (
        "warning",
        FailureMode.NUMBERS_DISAGREE_WITH_SECOND_READER.value,
        False,
    )
    assert [(c["check"], c["result"]) for c in side["verdict"]["clauses"]] == [
        ("second_reader_numbers", "FAIL")
    ]
    assert side["winning_output"]["text"] == new
    assert [(c["check"], c["result"]) for c in side["checks"] if c["result"] == "FAIL"] == [
        ("second_reader_numbers", "FAIL")
    ]
    doc_md = next(p for p in _doc_dir(dispute).glob("*.md")).read_text(encoding="utf-8")
    assert doc_md.count(line) == 1 and f"## Page 1\n\n{new}" in doc_md
    # Surfaced at the document too: status, note, audit event.
    meta = _meta(dispute)
    assert meta["status"] == "partial"
    assert tool.disagree_note([1]) in meta["error"]
    assert _meta(agree) == _meta(out)
    root = json.loads((dispute / "metadata.json").read_text(encoding="utf-8"))
    assert root["files"][meta["key"]]["status"] == "partial"
    log = json.loads((_doc_dir(dispute) / "audit_log.json").read_text(encoding="utf-8"))
    events = [e for e in log["events"] if e["kind"] == tool.SECOND_READER_DISAGREE_KIND]
    assert [e["page_num"] for e in events] == [1]
    assert log["counts"][tool.SECOND_READER_DISAGREE_KIND] == 1
    # The manifest replays the marked page.
    manifest = json.loads((_doc_dir(dispute) / "manifest.json").read_text(encoding="utf-8"))
    blob = manifest["entries"]["1"]["blob_ref"]
    assert (
        blob
        != json.loads((_doc_dir(out) / "manifest.json").read_text())["entries"]["1"]["blob_ref"]
    )
    record = json.loads((_doc_dir(dispute) / tool.LOG_NAME).read_text(encoding="utf-8"))
    assert record["pages"]["1"]["outcome"] == "disagree"
    assert record["pages"]["1"]["before"] == {"status": "success"}


def test_a_second_apply_changes_nothing_and_calls_no_reader(run):
    out, pdf_dir, copy = run
    for name, text in (("agree", _agreeing(out)), ("dispute", _disputing(out))):
        target = copy(name)
        _apply(target, pdf_dir, Reader(text))
        first = _tree(target)
        _apply(target, pdf_dir, _forbidden)
        assert _tree(target) == first, name


def test_a_marked_page_is_never_touched_whatever_a_later_reading_says(run):
    """Never lifts a mark, never adds to a marked page: a page that carries a mark is not
    compared, with a reading that agrees and with one that disputes other numbers."""
    out, pdf_dir, copy = run
    marked = copy("marked")
    _apply(marked, pdf_dir, Reader(_disputing(out)))
    assert tool.selects(_sidecar(marked))
    state = _tree(marked)
    other = _fragment(out).replace("5.5", "55")
    for n, later in enumerate((_agreeing(out), other)):
        target = Path(shutil.copytree(marked, marked.parent / f"later-{n}"))
        # forget the stored reading, so only the later one is at hand
        shutil.rmtree(_doc_dir(target) / tool.READINGS_DIR, ignore_errors=True)
        (_doc_dir(target) / tool.LOG_NAME).unlink(missing_ok=True)
        before = _tree(target)
        reader = Reader(later)
        assert _apply(target, pdf_dir, reader) == []
        assert reader.calls == 0
        assert _tree(target) == before
        assert _fragment(target) == _fragment(marked)
    assert _tree(marked) == state


def test_a_success_page_that_carries_a_mark_is_not_compared(run):
    """A mark does not always demote: a SUCCESS page whose text carries one is left alone."""
    out, pdf_dir, _copy = run
    if tool.selects(_sidecar(out)):
        pytest.skip("the page is not selected in this provider state")
    mark = (
        "> [table unverified: socr could not check this table against the page; verify it "
        "against the page image before citing]"
    )
    doc = _doc_dir(out)
    path = doc / "pages" / "00001.json"
    side = json.loads(path.read_text(encoding="utf-8"))
    old = side["winning_output"]["text"]
    side["winning_output"]["text"] = f"{mark}\n\n{old}"
    path.write_text(json.dumps(side, indent=2, ensure_ascii=False), encoding="utf-8")
    path.with_suffix(".md").write_text(f"{mark}\n\n{old}", encoding="utf-8")
    before = _tree(out)
    reader = Reader(_disputing(out))
    assert _apply(out, pdf_dir, reader) == []
    assert reader.calls == 0
    assert _tree(out) == before


def test_a_cut_off_reading_is_unknown_and_marks_the_page(run):
    out, pdf_dir, copy = run
    if tool.selects(_sidecar(out)):
        pytest.skip("the run already marked this page: covered by the untouched tests")
    cut, whole = copy("cut"), copy("whole")
    text = _agreeing(out)
    _apply(whole, pdf_dir, Reader(text, "stop"))
    (result,) = _apply(cut, pdf_dir, Reader(text, tool.FINISH_REASON_CUT_OFF))
    assert result.outcome == "unknown"
    assert _fragment(whole) == _fragment(out)
    assert _fragment(cut) == f"{result.mark}\n\n{_fragment(out)}"
    assert "second_reader_numbers=UNKNOWN" in result.mark
    side = _sidecar(cut)
    assert side["failure_mode"] == FailureMode.SECOND_READER_UNUSABLE.value
    # UNKNOWN is not a disagreement: no event, no note of one.
    assert tool.NOTE_PREFIX not in (_meta(cut).get("error") or "")
    stored = json.loads((_doc_dir(cut) / tool.READINGS_DIR / "00001.json").read_text())
    assert stored["finish_reason"] == tool.FINISH_REASON_CUT_OFF


def test_the_seam_publishing_anything_but_the_added_clause_stops_the_document(run, monkeypatch):
    out, pdf_dir, _copy = run
    if tool.selects(_sidecar(out)):
        pytest.skip("the page is not selected in this provider state")
    from dataclasses import replace

    from socr.core import manifest
    from socr.core.verdict import PageVerdict

    def lifting(output, p, page_num, published=None, **_kw):
        return replace(output, text="> [mark]\n\n" + output.text), PageVerdict(page_num=page_num)

    monkeypatch.setattr(manifest, "apply_verdict_marks", lifting)
    before = _tree(out)
    readings = out.parent / "readings"
    _write_reading(readings, out, _disputing(out))
    with pytest.raises(RuntimeError, match="did not publish exactly the added clause"):
        _apply(out, pdf_dir, None, readings_dir=readings)
    assert _tree(out) == before


def test_marking_that_would_change_the_shipped_text_stops_the_document(run, monkeypatch):
    out, pdf_dir, _copy = run
    if tool.selects(_sidecar(out)):
        pytest.skip("the page is not selected in this provider state")
    from dataclasses import replace

    from socr.core import manifest

    real = manifest.apply_verdict_marks

    def rewriting(output, p, page_num, published=None, **kw):
        final, verdict = real(output, p, page_num, published=published, **kw)
        return replace(final, text=final.text.replace(" 8.8", " 88")), verdict

    monkeypatch.setattr(manifest, "apply_verdict_marks", rewriting)
    before = _tree(out)
    readings = out.parent / "readings"
    _write_reading(readings, out, _disputing(out))
    with pytest.raises(RuntimeError, match="would change its shipped text"):
        _apply(out, pdf_dir, None, readings_dir=readings)
    assert _tree(out) == before


# --- readings ---------------------------------------------------------------------------


def _write_reading(root: Path, out: Path, reading_text: str, **changes) -> Path:
    doc = _doc_dir(out)
    record = tool.reading_record(
        page_num=1,
        input_checksum=_meta(out)["checksum"],
        text=reading_text,
        finish_reason=None,
        reader_model="made-elsewhere",
        prompt_sha256=None,
        render_dpi=None,
    )
    record.update(changes)
    path = tool.reading_path(root, doc.name, doc / "pages" / "00001.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def test_an_imported_reading_is_used_without_a_reader_and_kept_with_the_document(run):
    out, pdf_dir, copy = run
    if tool.selects(_sidecar(out)):
        pytest.skip("the page is not selected in this provider state")
    target = copy("imported")
    readings = out.parent / "readings"
    _write_reading(readings, target, _disputing(out))
    dry = _apply(target, pdf_dir, _forbidden, apply=False, readings_dir=readings)
    assert [r.outcome for r in dry] == ["disagree"]
    assert _tree(target) == _tree(out)
    done = _apply(target, pdf_dir, _forbidden, readings_dir=readings)
    assert [(r.outcome, r.mark) for r in done] == [(r.outcome, r.mark) for r in dry]
    kept = json.loads((_doc_dir(target) / tool.READINGS_DIR / "00001.json").read_text())
    assert kept["reader_model"] == "made-elsewhere"
    # the same bytes as a reading the tool made itself with that text
    own = copy("own")
    _apply(own, pdf_dir, Reader(_disputing(out), None))
    assert _fragment(own) == _fragment(target)


@pytest.mark.parametrize(
    "changes, why",
    [
        ({"input_checksum": "sha256:another-file"}, "another input file"),
        ({"page_num": 2}, "another page's"),
        ({"format": 0}, "not in this tool's format"),
        ({"text": None}, "holds no text"),
    ],
)
def test_a_reading_for_something_else_is_not_used(run, changes, why):
    out, pdf_dir, _copy = run
    if tool.selects(_sidecar(out)):
        pytest.skip("the page is not selected in this provider state")
    readings = out.parent / "readings"
    _write_reading(readings, out, _disputing(out), **changes)
    before = _tree(out)
    skipped: Counter = Counter()
    results = tool.refinalize_document(
        _doc_dir(out), pdf_dir, apply=True, readings_dir=readings, skipped=skipped
    )
    assert results == [] and _tree(out) == before
    assert any(why in reason for reason in skipped)


def test_the_documents_own_reading_wins_over_an_imported_one(run):
    out, pdf_dir, _copy = run
    if tool.selects(_sidecar(out)):
        pytest.skip("the page is not selected in this provider state")
    _apply(out, pdf_dir, Reader(_agreeing(out)))
    readings = out.parent / "readings"
    _write_reading(readings, out, _disputing(out))
    before = _tree(out)
    assert [r.outcome for r in _apply(out, pdf_dir, _forbidden, readings_dir=readings)] == ["agree"]
    assert _tree(out) == before


# --- selection --------------------------------------------------------------------------


def _clean_sidecar(text: str = "Growth was 2.5 percent.") -> dict:
    return {
        "page_num": 4,
        "terminal": True,
        "status": "success",
        "verdict": {"page_num": 4, "clauses": []},
        "winning_output": {"page_num": 4, "status": "success", "text": text},
    }


def test_an_unmarked_success_page_with_a_number_is_selected():
    assert tool.selects(_clean_sidecar()) == ""


@pytest.mark.parametrize(
    "change, why",
    [
        ({"terminal": False}, "not finished"),
        ({"status": "warning"}, "not SUCCESS"),
        (
            {
                "verdict": {
                    "clauses": [{"check": "page_judge", "result": "UNKNOWN", "scope": "page"}]
                }
            },
            "verdict clause",
        ),
    ],
)
def test_a_page_that_is_not_a_clean_success_is_not_selected(change, why):
    assert why in tool.selects({**_clean_sidecar(), **change})


@pytest.mark.parametrize(
    "mark",
    [
        page_unchecked_clause_mark(4, Check.PAGE_JUDGE, CheckResult.UNKNOWN, SCOPE_PAGE, "why"),
        "> [table unverified: socr could not check this table against the page; verify it "
        "against the page image before citing]",
    ],
)
def test_a_page_whose_text_carries_a_mark_is_not_selected(mark):
    assert "mark" in tool.selects(_clean_sidecar(f"{mark}\n\nGrowth was 2.5 percent."))


def test_a_page_without_a_number_is_not_selected():
    assert "no number" in tool.selects(_clean_sidecar("Only words on this page."))


# --- charts -----------------------------------------------------------------------------


def _text_page(rotation: int = 0) -> fitz.Page:
    doc = fitz.open()
    page = doc.new_page(width=600, height=800)
    page.insert_text((100, 100), "axis 2017 2018", fontsize=10)
    page.insert_text((100, 700), "body 99", fontsize=10)
    page.set_rotation(rotation)
    return page


#: Where the centre ``(x, y)`` of a word on a 600 x 800 page is DRAWN once the page is
#: rotated, written out by hand (checked against the rendered ink when this test was written)
#: so the expectation does not come from the matrix the tool itself uses.
_DISPLAYED = {
    0: lambda x, y: (x, y),
    90: lambda x, y: (800 - y, x),
    180: lambda x, y: (600 - x, 800 - y),
    270: lambda x, y: (y, 600 - x),
}


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_chart_labels_are_the_layer_words_touching_a_box_in_the_displayed_frame(rotation):
    page = _text_page(rotation)
    (word,) = [w for w in page.get_text("words") if w[4] == "2017"]
    x, y = _DISPLAYED[rotation]((word[0] + word[2]) / 2, (word[1] + word[3]) / 2)
    assert tool.chart_label_text(page, [[x - 2, y - 2, x + 2, y + 2]]) == "2017"
    assert tool.chart_label_text(page, [[0, 0, 5, 5]]) == ""


class _NoPriorOcr:
    invisible_text_over_raster = invisible_text_scan_failed = False
    prior_ocr_layer = prior_ocr_layer_scan_failed = False


class _PriorOcr(_NoPriorOcr):
    prior_ocr_layer = True


def test_a_page_has_a_layer_of_its_own_only_with_words_that_are_not_an_earlier_ocr():
    page = _text_page()
    blank = fitz.open().new_page()
    assert tool._has_own_layer({}, _NoPriorOcr(), page)
    assert not tool._has_own_layer({}, _NoPriorOcr(), blank)
    assert not tool._has_own_layer({}, _PriorOcr(), page)
    assert not tool._has_own_layer({"prior_ocr_layer_suspect": True}, _NoPriorOcr(), page)


def test_a_picture_page_without_its_own_layer_is_not_compared_and_not_a_pass(run, monkeypatch):
    out, pdf_dir, copy = run
    if tool.selects(_sidecar(out)):
        pytest.skip("the page is not selected in this provider state")
    scan, layer = copy("scan"), copy("layer")
    for target in (scan, layer):
        path = _doc_dir(target) / "pages" / "00001.json"
        side = json.loads(path.read_text(encoding="utf-8"))
        side["figure_refs"] = [{"figure_num": 1, "bbox": [0.0, 0.0, 10.0, 10.0]}]
        path.write_text(json.dumps(side, indent=2, ensure_ascii=False), encoding="utf-8")
    (with_layer,) = _apply(layer, pdf_dir, Reader(_disputing(out)))
    assert with_layer.outcome == "disagree"
    monkeypatch.setattr(tool, "_has_own_layer", lambda sidecar, pa, page: False)
    before = _tree(scan)
    (result,) = _apply(scan, pdf_dir, Reader(_disputing(out)))
    assert result.outcome == tool.OUTCOME_NOT_APPLICABLE and result.mark == ""
    after = _tree(scan)
    assert {name: after[name] for name in before} == before
    record = json.loads((_doc_dir(scan) / tool.LOG_NAME).read_text(encoding="utf-8"))
    assert record["pages"]["1"]["outcome"] == tool.OUTCOME_NOT_APPLICABLE
    assert "comparison" not in record["pages"]["1"]


def test_figure_boxes_are_the_sidecars_recorded_boxes():
    side = {"figure_refs": [{"bbox": [1, 2, 3, 4]}, {"bbox": None}, {"image_path": "x.png"}]}
    assert tool.figure_boxes(side) == [[1.0, 2.0, 3.0, 4.0]]
    assert tool.figure_boxes({}) == []


# --- the reader client ------------------------------------------------------------------


class _Response:
    def __init__(self, payload, status=200):
        self._payload, self.status_code = payload, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class _Client:
    def __init__(self, response):
        self.response, self.posts = response, []

    def post(self, url, json):
        self.posts.append((url, json))
        return self.response

    def close(self):
        pass


def _engine(response):
    from socr.engines.vllm import VLLMConfig, VLLMEngine

    engine = VLLMEngine(VLLMConfig(base_url="http://127.0.0.1:9/v1", model=MODEL))
    engine._client = _Client(response)
    return engine


def test_read_page_sends_the_png_and_the_prompt_and_returns_the_raw_text():
    payload = {"choices": [{"message": {"content": "  raw <table>\n"}, "finish_reason": "length"}]}
    engine = _engine(_Response(payload))
    out = engine.read_page(b"\x89PNG-bytes", "the prompt", max_tokens=77, temperature=0.0)
    assert out == ("  raw <table>\n", "length")
    ((url, body),) = engine._client.posts
    assert url == "/chat/completions"
    assert (body["model"], body["max_tokens"], body["temperature"]) == (MODEL, 77, 0.0)
    system, user = body["messages"]
    assert system == {"role": "system", "content": "the prompt"}
    (image,) = user["content"]
    assert image["image_url"]["url"].startswith("data:image/png;base64,")


@pytest.mark.parametrize(
    "response", [_Response({"choices": []}), _Response({"error": "x"}, status=500)]
)
def test_read_page_raises_when_no_reading_was_made(response):
    with pytest.raises(RuntimeError):
        _engine(response).read_page(b"\x89PNG", "p", max_tokens=1, temperature=0.0)


def test_the_reader_is_given_the_evaluations_settings(monkeypatch):
    from socr.engines.vllm import VLLMEngine

    seen = {}

    def read_page(self, png, prompt, *, max_tokens, temperature):
        seen.update(prompt=prompt, max_tokens=max_tokens, temperature=temperature)
        seen.update(timeout=self.config.timeout, model=self.config.model)
        return "text", "stop"

    monkeypatch.setattr(VLLMEngine, "read_page", read_page)
    assert tool.vllm_reader("http://127.0.0.1:9/v1", MODEL, "the prompt")(b"png") == (
        "text",
        "stop",
    )
    assert seen == {
        "prompt": "the prompt",
        "max_tokens": tool.READER_MAX_TOKENS,
        "temperature": tool.READER_TEMPERATURE,
        "timeout": tool.READER_TIMEOUT_S,
        "model": MODEL,
    }


def test_the_page_image_is_rendered_at_the_named_dpi():
    page = fitz.open().new_page(width=72, height=144)
    pix = fitz.Pixmap(tool.render_page_png(page))
    assert (pix.width, pix.height) == (tool.READER_RENDER_DPI, 2 * tool.READER_RENDER_DPI)


# --- the command ------------------------------------------------------------------------


def test_the_command_dry_runs_by_default_and_applies_on_request(run, capsys):
    out, pdf_dir, _copy = run
    selected = not tool.selects(_sidecar(out))
    readings = out.parent / "readings"
    _write_reading(readings, out, _disputing(out))
    before = _tree(out)
    args = [str(out), "--pdf-dir", str(pdf_dir), "--readings-dir", str(readings)]
    assert tool.main(args) == 0
    dry = capsys.readouterr().out
    assert _tree(out) == before
    assert "dry run" in dry
    assert tool.main([*args, "--apply"]) == 0
    applied = capsys.readouterr().out
    assert ("second_reader_numbers=FAIL" in applied) is selected
    assert ("second_reader_numbers=FAIL" in dry) is selected
    assert (_tree(out) != before) is selected


def test_the_reader_arguments_go_together(tmp_path):
    with pytest.raises(SystemExit):
        tool.main([str(tmp_path), "--pdf-dir", str(tmp_path), "--reader-url", "http://x/v1"])


def test_the_command_is_installed_under_its_name():
    import tomllib

    scripts = tomllib.loads(Path("pyproject.toml").read_text())["project"]["scripts"]
    assert scripts[tool.TOOL] == "socr.pipeline.refinalize_second_reader:main"


# --- the grammar ------------------------------------------------------------------------


@pytest.mark.parametrize("check", list(Check))
@pytest.mark.parametrize("result", [CheckResult.FAIL, CheckResult.UNKNOWN])
def test_every_check_renders(check, result):
    """What ``marks.Check``'s docstring promises: every member renders into a mark line that
    parses back to itself."""
    line = page_unchecked_clause_mark(7, check, result, SCOPE_PAGE, "a reason / with 1.5")
    assert parse_canonical_mark(line, 7) == (check, SCOPE_PAGE)


@pytest.mark.parametrize(
    "result, mode",
    [
        (CheckResult.FAIL, FailureMode.NUMBERS_DISAGREE_WITH_SECOND_READER),
        (CheckResult.UNKNOWN, FailureMode.SECOND_READER_UNUSABLE),
    ],
)
def test_each_result_of_the_clause_demotes_with_its_own_permanent_mode(result, mode):
    from socr.core import manifest, verdict

    clause = verdict.VerdictClause(Check.SECOND_READER_NUMBERS, result, SCOPE_PAGE, "why")
    assert manifest._verdict_demotion_mode((clause,)) is mode
    assert mode in verdict.PERMANENTLY_UNRESOLVABLE_MODES
    assert mode in manifest._VERDICT_DERIVED_MODES
    assert verdict.LAYER_CHECK_MODES[mode] is Check.SECOND_READER_NUMBERS


def test_no_run_derives_or_traces_the_clause():
    """Tool-only: the check has no slot in the always-run pass trace and no silent-success
    writer, so a fresh run's sidecars are what they were before #1409."""
    from socr.core import verdict

    name = Check.SECOND_READER_NUMBERS.value
    assert name not in {check for check, _scope in verdict.PASS_TRACE_SLOTS}
    assert name not in verdict.TEXT_DEPENDENT_CHECKS
    assert Check.SECOND_READER_NUMBERS not in {row[0] for row in verdict._MODE_CLAUSES.values()}
