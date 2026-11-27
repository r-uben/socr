"""#1409 step 3: ``socr-refinalize-two-readers`` on synthetic finished runs. No model, no network.

Every pipeline test runs in BOTH provider states and pins a DIFFERENCE: the tool is run on
copies of ONE finished run, changing only the second reading (one that agrees with the first,
one that differs in exactly one thing), and the copies are compared. Whether a run's page can
be lifted at all is read off the run, never assumed from the provider state: without a
provider the scanned document has no ``.md`` (every page floored) and the tool must leave it
alone whatever the readings say.
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
from test_gh1267_row_binding_check import GROUP, HEIGHT, WIDTH  # noqa: E402
from test_gh1267_row_binding_pipeline import _run  # noqa: E402

from socr.benchmark import acceptance  # noqa: E402
from socr.pipeline import refinalize_two_readers as tool  # noqa: E402
from socr.pipeline.refinalize_second_reader import reading_path, reading_record  # noqa: E402
from socr.tables import two_readers as rule  # noqa: E402

MODEL_A, MODEL_B = "reader-a-under-test", "reader-b-under-test"

TABLE_A = """<table>
  <thead>
    <tr>
      <th></th>
      <th colspan="2">1997</th>
      <th>1998</th>
    </tr>
    <tr>
      <th>Item</th>
      <th>Q3</th>
      <th>Q4</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Exports</td>
      <td>0.8</td>
      <td>-1.3</td>
      <td>-0.7</td>
    </tr>
  </tbody>
</table>"""
TABLE_B = (
    '<table><tr><th></th><th colspan="2"><strong>1997</strong></th><th>1998</th></tr>'
    "<tr><th>Item</th><th>Q3</th><th>Q4</th><th></th></tr>"
    "<tr><td>Exports</td><td>0.8</td><td>−1.3</td><td>-0.7</td></tr></table>"
)
READING_A = f"""11/13/90

-2-

MR. KOHN. Inflation reaches 2.9 percent by the end of the year.

{TABLE_A}

NOTE. Prices are on a chain-weighted basis.
"""
READING_B = f"""```markdown
<page_header>11/13/90 -2-</page_header>

MR. KOHN. Inflation reaches 2.9 percent by the end of the year.
{TABLE_B}

NOTE. Prices are on a chain-weighted basis.
```"""

#: ``(what is changed in reading B, the reason it must give)``: each condition failing alone.
ONE_THING_WRONG = {
    "number": (lambda b: b.replace("2.9 percent", "2.8 percent"), rule.NUMBERS_DIFFER),
    "sign": (lambda b: b.replace("−1.3", "1.3"), rule.NUMBERS_DIFFER),
    "word": (lambda b: b.replace("Inflation", "Deflation"), rule.WORDS_DIFFER),
    "order": (
        lambda b: b.replace("chain-weighted basis", "basis chain-weighted"),
        rule.WORD_ORDER_DIFFERS,
    ),
    "tilde": (lambda b: b.replace("−1.3", "~1.3"), rule.TILDE_BEFORE_DIGIT),
    "picture": (
        lambda b: b.replace("NOTE.", "![image](image_1.png)\n\nNOTE."),
        rule.IMAGE_PLACEHOLDER,
    ),
    "reader-only table": (
        lambda b: b.replace("NOTE.", "<table><tr><td>1997</td><td>1.37</td></tr></table>\n\nNOTE."),
        rule.TABLE_WITHOUT_COUNTERPART,
    ),
    "grid": (
        lambda b: b.replace('<th colspan="2"><strong>1997</strong></th>', "<th>1997</th>"),
        rule.TABLE_GRIDS_DIFFER,
    ),
    "furniture": (lambda b: b.replace("11/13/90 -2-", "11/13/90 -3-"), rule.FURNITURE_UNMATCHED),
    "unclosed table": (lambda b: b.replace("</table>", ""), rule.UNUSABLE_UNCLOSED),
}


@pytest.fixture(autouse=True)
def _no_words_check(monkeypatch):
    from socr.core import verdict

    monkeypatch.setattr(verdict, "page_missing_words", lambda output, p: None)


def _scan(path: Path) -> Path:
    """A true scan: one page that is a picture, with no text layer at all."""
    doc = fitz.open()
    page = doc.new_page(width=WIDTH, height=HEIGHT)
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 200, 260), False)
    pix.clear_with(255)
    for x in range(20, 180):
        for y in range(40, 44):
            pix.set_pixel(x, y, (0, 0, 0))
    page.insert_image(page.rect, pixmap=pix)
    doc.save(path)
    doc.close()
    return path


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


def _liftable(out: Path) -> bool:
    """The run's page is one the tool may lift: not clean, in a document that has a ``.md``."""
    return not tool.selects(_sidecar(out)) and len(list(_doc_dir(out).glob("*.md"))) == 1


def _readings(root: Path, out: Path, model: str, text: str, finish_reason="stop") -> Path:
    doc = _doc_dir(out)
    record = reading_record(
        page_num=1,
        input_checksum=_meta(out)["checksum"],
        text=text,
        finish_reason=finish_reason,
        reader_model=model,
        prompt_sha256=None,
        render_dpi=300,
    )
    path = reading_path(root, doc.name, doc / "pages" / "00001.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record), encoding="utf-8")
    return root


def _apply(
    out: Path, pdf_dir: Path, b: str, *, apply=True, a=READING_A, finish_b="stop", model_b=MODEL_B
):
    root = out.parent / f"readings-{out.name}"
    shutil.rmtree(root, ignore_errors=True)
    dir_a = _readings(root / "a", out, MODEL_A, a)
    dir_b = _readings(root / "b", out, model_b, b, finish_b)
    skipped: Counter = Counter()
    results = tool.refinalize_document(
        _doc_dir(out), pdf_dir, dir_a, dir_b, apply=apply, skipped=skipped
    )
    return results, skipped


@pytest.fixture(params=[True, False], ids=["provider", "no-provider"])
def scan_run(request, tmp_path):
    """``(out_dir, pdf_dir, copy)``: a finished run of a one-page scan."""
    pdf = _scan(tmp_path / "scan.pdf")
    _, _, _, out, _ = _run(
        tmp_path, GROUP, 0, check_runs=True, out_name="run", providers=request.param, pdf=pdf
    )

    def copy(name: str) -> Path:
        return Path(shutil.copytree(out, tmp_path / name))

    return out, tmp_path, copy


@pytest.fixture(params=[True, False], ids=["provider", "no-provider"])
def born_digital_run(request, tmp_path):
    _, _, _, out, pdf = _run(
        tmp_path, GROUP, 0, check_runs=True, out_name="run", providers=request.param
    )
    return out, pdf.parent


# --- what ships -------------------------------------------------------------------------


def test_the_shipped_text_is_the_reading_as_socr_ships_a_model_reading():
    text, refused = tool.shipped_text(READING_A)
    assert refused == ""
    assert text.startswith("11/13/90\n\n-2-\n\nMR. KOHN.")
    assert "| | 1997 | | 1998 |\n|---|---|---|---|\n| Item | Q3 | Q4 | |" in text
    assert "| Exports | 0.8 | -1.3 | -0.7 |" in text
    assert "<" not in text
    assert tool.shipped_text(READING_A) == (text, "")  # a function of the reading alone


def test_a_reading_with_reader_furniture_tags_ships_cleanly():
    """Readers that tag furniture (``<page_header>``, ``<page_footer>``, ``<page_number>``)
    unwrap cleanly to plain lines instead of failing ``conversion_refused``."""
    reading_with_footer = (
        READING_B.removesuffix("```")
        + "\n<page_footer>Page <page_number>10</page_number> of 96</page_footer>\n```"
    )
    text, refused = tool.shipped_text(reading_with_footer)
    assert refused == ""
    assert "11/13/90 -2-" in text
    assert "Page 10 of 96" in text
    assert "| Exports | 0.8 | −1.3 | -0.7 |" in text
    assert (
        "<page_header>" not in text and "<page_footer>" not in text and "<page_number>" not in text
    )


def test_a_reading_with_markup_socr_cannot_convert_is_not_shipped():
    text, refused = tool.shipped_text(READING_A + "\n<custom_widget>opaque</custom_widget>\n")
    assert text == "" and "markup" in refused


@pytest.mark.parametrize(
    "old, new, what",
    [
        ("Inflation", "Deflation", "words"),  # the numbers are untouched
        ("-1.3", "1.3", "numbers"),  # the words are untouched: a dash is not part of a word
    ],
)
def test_a_conversion_that_would_change_the_reading_is_refused(monkeypatch, old, new, what):
    from socr.core import html_tables

    real = html_tables.clean_residual_html
    monkeypatch.setattr(
        html_tables, "clean_residual_html", lambda text: real(text).replace(old, new)
    )
    text, refused = tool.shipped_text(READING_A)
    assert text == "" and refused.endswith(f"would change the reading's {what}")


def test_the_seam_publishing_anything_but_the_clean_agreed_text_stops_the_document(
    scan_run, monkeypatch
):
    out, pdf_dir, _copy = scan_run
    if not _liftable(out):
        pytest.skip("the run's document is not one the tool rewrites in this provider state")
    from dataclasses import replace

    from socr.core import manifest

    real = manifest.apply_verdict_marks

    def marking(output, p, page_num, published=None, **kw):
        final, verdict = real(output, p, page_num, published=published, **kw)
        return replace(final, text="> [mark]\n\n" + final.text), verdict

    monkeypatch.setattr(manifest, "apply_verdict_marks", marking)
    before = _tree(out)
    with pytest.raises(RuntimeError, match="did not publish the agreed text clean"):
        _apply(out, pdf_dir, READING_B)
    assert _tree(out) == before


# --- the difference a second reading makes ----------------------------------------------


def test_dry_run_writes_nothing(scan_run):
    out, pdf_dir, _copy = scan_run
    before = _tree(out)
    results, _ = _apply(out, pdf_dir, READING_B, apply=False)
    assert _tree(out) == before
    assert [r.outcome for r in results] == ([tool.OUTCOME_LIFTED] if _liftable(out) else [])


def test_agreement_lifts_the_mark_and_ships_the_first_reading(scan_run):
    out, pdf_dir, copy = scan_run
    agree, differ = copy("agree"), copy("differ")
    results, _ = _apply(agree, pdf_dir, READING_B)
    changed, _ = _apply(differ, pdf_dir, ONE_THING_WRONG["number"][0](READING_B))
    if not _liftable(out):
        # No reading can matter: the run's document is not one the tool may rewrite.
        assert _tree(agree) == _tree(differ) == _tree(out)
        assert results == [] and changed == []
        return
    assert [r.outcome for r in results] == [tool.OUTCOME_LIFTED]
    assert [r.outcome for r in changed] == [tool.OUTCOME_DISAGREE]
    expected, _refused = tool.shipped_text(READING_A)
    assert _fragment(agree) == expected
    assert _fragment(differ) == _fragment(out)
    side, was = _sidecar(agree), _sidecar(out)
    assert _sidecar(differ) == was
    assert (side["status"], side["failure_mode"], side["audit_passed"], side["engine"]) == (
        "success",
        "none",
        True,
        rule.ENGINE,
    )
    assert side["verdict"]["clauses"] == [] and was["verdict"]["clauses"]
    assert side["winning_output"]["text"] == expected
    # The durable record: the rule, both readers, both reading digests, the shipped text.
    record = side[rule.RULE]
    assert record["rule"] == rule.RULE and record["shipped_reading"] == "a"
    assert [record["readers"][s]["reader_model"] for s in "ab"] == [MODEL_A, MODEL_B]
    assert record["readers"]["a"]["text_sha256"] != record["readers"]["b"]["text_sha256"]
    assert (
        record["shipped_text_sha256"] == "sha256:" + hashlib.sha256(expected.encode()).hexdigest()
    )
    assert side["refinalized_from"]["status"] == was["status"]
    assert side["refinalized_from"]["verdict"] == was["verdict"]
    passes = [c for c in side["checks"] if c["result"] == "PASS"]
    assert [c["check"] for c in passes] == [rule.RULE] and passes[0]["text_sha"]
    assert not [c for c in side["checks"] if c["result"] in ("FAIL", "UNKNOWN")]
    assert [e["kind"] for e in side["audit_events"]].count(rule.RULE) == 1
    # No mark and no provenance line on the page; the page and the meter agree it is checked.
    assert not acceptance.has_any_mark_line(expected, 1)
    page = acceptance.Page("doc", 1, side, _fragment(agree), None)
    assert acceptance.judge_page(page).checked
    doc_md = next(_doc_dir(agree).glob("*.md")).read_text(encoding="utf-8")
    assert f"## Page 1\n\n{expected}" in doc_md
    # The two readings are kept with the document.
    for side_name, text in (("a", READING_A), ("b", READING_B)):
        kept = json.loads(
            (_doc_dir(agree) / tool.READINGS_DIR / side_name / "00001.json").read_text()
        )
        assert kept["text"] == text
    # The document: status never moved; the note naming page 1 as unfinished is gone.
    assert _meta(agree)["status"] == _meta(out)["status"]
    assert "did not finish" in _meta(out)["error"]
    assert "did not finish" not in (_meta(agree).get("error") or "")
    assert _meta(differ) == _meta(out)
    log = json.loads((_doc_dir(agree) / tool.LOG_NAME).read_text(encoding="utf-8"))["pages"]["1"]
    assert (log["outcome"], log["before"]["status"], log["after"]["status"]) == (
        tool.OUTCOME_LIFTED,
        was["status"],
        "success",
    )
    events = json.loads((_doc_dir(agree) / "audit_log.json").read_text())["events"]
    assert [e["page_num"] for e in events if e["kind"] == rule.RULE] == [1]


@pytest.mark.parametrize("what", sorted(ONE_THING_WRONG))
def test_each_condition_failing_alone_leaves_the_page_as_the_run_left_it(scan_run, what):
    out, pdf_dir, _copy = scan_run
    change, reason = ONE_THING_WRONG[what]
    before = _tree(out)
    results, _ = _apply(out, pdf_dir, change(READING_B))
    after = _tree(out)
    assert {name: after[name] for name in before} == before
    if not _liftable(out):
        assert results == [] and after == before
        return
    (result,) = results
    assert result.outcome == tool.OUTCOME_DISAGREE and reason in result.reasons
    doc = _doc_dir(out).name
    assert set(after) - set(before) == {f"{doc}/{tool.LOG_NAME}"}
    log = json.loads((_doc_dir(out) / tool.LOG_NAME).read_text(encoding="utf-8"))["pages"]["1"]
    assert log["outcome"] == tool.OUTCOME_DISAGREE
    assert reason in [f["reason"] for f in log["comparison"]["failures"]]


def test_a_cut_off_reading_leaves_the_page_as_the_run_left_it(scan_run):
    out, pdf_dir, _copy = scan_run
    before = _tree(out)
    results, _ = _apply(out, pdf_dir, READING_B, finish_b="length")
    after = _tree(out)
    assert {name: after[name] for name in before} == before
    if _liftable(out):
        assert results[0].reasons == (rule.UNUSABLE_CUT_OFF,)


def test_two_readings_from_one_reader_model_are_not_two_readers(scan_run):
    out, pdf_dir, _copy = scan_run
    before = _tree(out)
    results, skipped = _apply(out, pdf_dir, READING_B, model_b=MODEL_A)
    assert results == [] and _tree(out) == before
    if _liftable(out):
        assert any("two different reader models" in reason for reason in skipped)


def test_a_reading_for_another_input_is_not_used(scan_run):
    out, pdf_dir, _copy = scan_run
    root = out.parent / "readings-other"
    _readings(root / "a", out, MODEL_A, READING_A)
    path = _readings(root / "b", out, MODEL_B, READING_B)
    file = next(path.rglob("00001.json"))
    record = json.loads(file.read_text())
    record["input_checksum"] = "sha256:another-file"
    file.write_text(json.dumps(record))
    before = _tree(out)
    skipped: Counter = Counter()
    assert (
        tool.refinalize_document(
            _doc_dir(out), pdf_dir, root / "a", root / "b", apply=True, skipped=skipped
        )
        == []
    )
    assert _tree(out) == before


def test_no_note_is_added_to_a_document_that_carried_none(scan_run):
    out, pdf_dir, _copy = scan_run
    if not _liftable(out):
        pytest.skip("the run's document is not one the tool rewrites in this provider state")
    meta_path = _doc_dir(out) / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["error"] = "untrusted tables on 1 page(s), 1 flag(s) (see tables_trust.json)"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    results, _ = _apply(out, pdf_dir, READING_B)
    assert [r.outcome for r in results] == [tool.OUTCOME_LIFTED]
    assert _meta(out)["error"] == meta["error"]


def test_a_second_apply_changes_nothing(scan_run):
    out, pdf_dir, copy = scan_run
    for name, b in (("agree", READING_B), ("differ", ONE_THING_WRONG["word"][0](READING_B))):
        target = copy(name)
        _apply(target, pdf_dir, b)
        first = _tree(target)
        again, _ = _apply(target, pdf_dir, b)
        assert _tree(target) == first, name
        if name == "agree":
            assert again == []  # a lifted page is a clean page: no longer selected


# --- pages the tool never touches -------------------------------------------------------


def test_a_born_digital_page_is_never_touched(born_digital_run):
    """With a provider the page is clean; without one it is a marked floor. Two agreeing
    readings change nothing in either state."""
    out, pdf_dir = born_digital_run
    before = _tree(out)
    results, _skipped = _apply(out, pdf_dir, READING_B)
    assert results == [] and _tree(out) == before


def test_a_page_with_a_text_layer_of_its_own_is_never_touched(born_digital_run, monkeypatch):
    """The layer is what rules a born-digital page out, whatever else the sidecar says: with
    the sidecar's own exclusions switched off, the page is still left alone."""
    out, pdf_dir = born_digital_run
    monkeypatch.setattr(tool, "selects", lambda sidecar: "")
    before = _tree(out)
    results, skipped = _apply(out, pdf_dir, READING_B)
    assert results == [] and _tree(out) == before
    assert list(skipped) == ["not selected: the page has a text layer of its own"]


def test_a_page_with_a_recorded_chart_is_never_touched(scan_run):
    out, pdf_dir, _copy = scan_run
    path = _doc_dir(out) / "pages" / "00001.json"
    side = json.loads(path.read_text(encoding="utf-8"))
    side["figure_refs"] = [{"figure_num": 1, "bbox": [0.0, 0.0, 10.0, 10.0]}]
    path.write_text(json.dumps(side, indent=2, ensure_ascii=False), encoding="utf-8")
    before = _tree(out)
    results, skipped = _apply(out, pdf_dir, READING_B)
    assert results == [] and _tree(out) == before
    assert any("chart or figure" in reason for reason in skipped)


def _marked_sidecar(text: str = "[page 4 failed: unverifiable table — see image]", **keys) -> dict:
    return {
        "page_num": 4,
        "terminal": True,
        "status": "error",
        "verdict": {"page_num": 4, "clauses": [{"check": "unknown", "result": "UNKNOWN"}]},
        "winning_output": {"page_num": 4, "status": "error", "text": text},
        **keys,
    }


def test_a_marked_page_without_a_figure_is_selected():
    assert tool.selects(_marked_sidecar()) == ""
    floor = "[page 4 failed: x]\n\n![Failed table page 4](figures/failed_table_p4.png)"
    assert tool.selects(_marked_sidecar(floor)) == ""


def test_a_clean_success_page_is_never_selected():
    clean = {
        "page_num": 4,
        "terminal": True,
        "status": "success",
        "verdict": {"page_num": 4, "clauses": []},
        "winning_output": {"page_num": 4, "status": "success", "text": "Growth was 2.5 percent."},
    }
    assert tool.is_clean(clean)
    assert "SUCCESS without a mark" in tool.selects(clean)
    marked = {
        **clean,
        "winning_output": {
            **clean["winning_output"],
            "text": "> [page 4 unchecked: page_judge=UNKNOWN; scope=page; reason=x]\n\nGrowth.",
        },
    }
    assert not tool.is_clean(marked)


@pytest.mark.parametrize("key", tool.FIGURE_KEYS)
def test_every_record_of_a_figure_rules_the_page_out(key):
    assert "chart or figure" in tool.selects(_marked_sidecar(**{key: ["x"]}))


def test_a_picture_that_is_not_a_floors_own_rules_the_page_out():
    chart = "text\n\n![chart region 1](figures/chart_region_p4_1.png)"
    assert "chart or figure" in tool.selects(_marked_sidecar(chart))


def test_an_unfinished_page_is_not_selected():
    assert "not finished" in tool.selects(_marked_sidecar(terminal=False))


# --- a floor page -----------------------------------------------------------------------


def test_a_picture_only_floor_becomes_the_agreed_text(scan_run):
    """Whatever the run's page was, the page two readings agree on is the same page."""
    out, _pdf_dir, _copy = scan_run
    side = _sidecar(out)
    path = _doc_dir(out) / "pages" / "00001.json"
    readings = [
        reading_record(
            page_num=1,
            input_checksum="sha256:x",
            text=text,
            finish_reason="stop",
            reader_model=model,
            prompt_sha256=None,
            render_dpi=300,
        )
        for model, text in ((MODEL_A, READING_A), (MODEL_B, READING_B))
    ]
    done = tool.compare_page("doc", path, side, *readings)
    expected, _ = tool.shipped_text(READING_A)
    assert done.result.outcome == tool.OUTCOME_LIFTED
    assert done.result.before == (side["disposition"]["primary_reason"], side["status"])
    new = done.new_sidecar
    assert (new["status"], new["winning_output"]["text"]) == ("success", expected)
    assert new["disposition"] == {"ending": "model_output", "primary_reason": "accepted_output"}
    assert not {e["kind"] for e in new["audit_events"]} & {"page_failed", "invisible_scan_unread"}
    assert acceptance.judge_page(acceptance.Page("doc", 1, new, expected, None)).checked


# --- the meter --------------------------------------------------------------------------


def _lifted_sidecar(scan_run) -> tuple[dict, str]:
    out, _pdf_dir, _copy = scan_run
    side = _sidecar(out)
    readings = [
        reading_record(
            page_num=1,
            input_checksum="sha256:x",
            text=text,
            finish_reason="stop",
            reader_model=model,
            prompt_sha256=None,
            render_dpi=300,
        )
        for model, text in ((MODEL_A, READING_A), (MODEL_B, READING_B))
    ]
    done = tool.compare_page("doc", _doc_dir(out) / "pages" / "00001.json", side, *readings)
    return done.new_sidecar, done.new_text


def _checked(side: dict, text: str) -> bool:
    return acceptance.judge_page(acceptance.Page("doc", 1, side, text, None)).checked


def test_the_meter_counts_a_page_checked_only_on_the_rules_own_record(scan_run):
    side, text = _lifted_sidecar(scan_run)
    assert _checked(side, text)
    without = {k: v for k, v in side.items() if k != rule.RULE}
    assert not _checked(without, text)
    one_reader = json.loads(json.dumps(side))
    one_reader[rule.RULE]["readers"]["b"]["reader_model"] = MODEL_A
    assert not _checked(one_reader, text)
    no_pass = {**side, "checks": [c for c in side["checks"] if c["check"] != rule.RULE]}
    assert not _checked(no_pass, text)
    other_text = json.loads(json.dumps(side))
    other_text["winning_output"]["text"] = text.replace("2.9", "2.8")
    assert not _checked(other_text, text.replace("2.9", "2.8"))


# --- the command ------------------------------------------------------------------------


def test_the_command_dry_runs_by_default_and_applies_on_request(scan_run, capsys):
    out, pdf_dir, _copy = scan_run
    root = out.parent / "readings-cli"
    _readings(root / "a", out, MODEL_A, READING_A)
    _readings(root / "b", out, MODEL_B, READING_B)
    args = [
        str(out),
        "--pdf-dir",
        str(pdf_dir),
        "--readings-dir",
        str(root / "a"),
        "--readings-dir-b",
        str(root / "b"),
    ]
    before = _tree(out)
    liftable = _liftable(out)
    assert tool.main(args) == 0
    dry = capsys.readouterr().out
    assert _tree(out) == before and "dry run" in dry
    assert ("-> lifted" in dry) is liftable
    assert tool.main([*args, "--apply"]) == 0
    assert (_tree(out) != before) is liftable


def test_lifting_clears_stale_sidecar_keys_updates_tables_trust_and_survives_resume(scan_run):
    """Lifting a failed/withheld scanned page must:
    1. clear stale failed-run sidecar keys so resume doesn't reject/re-demote it;
    2. strip old distrust events and unlink/update ``tables_trust.json`` + metadata note;
    3. survive ``_restore_terminal_page_state`` -> ``_flush_page_sidecar`` so
       ``acceptance.rule_two_readers`` stays green after resuming another page.
    """
    out, pdf_dir, _copy = scan_run
    if not _liftable(out):
        pytest.skip("the run's document is not one the tool rewrites in this provider state")

    doc_dir = _doc_dir(out)
    side_path = doc_dir / "pages" / "00001.json"
    side = json.loads(side_path.read_text(encoding="utf-8"))
    side.update(
        {
            "judge_rejected": True,
            "table_ladder_disposition": "table_withheld",
            "table_ladder_incomplete": True,
            "table_judge_retry_pending": True,
            "table_judge_retry_rungs": ["ollama"],
            "d3_floor_png_ref": "figures/failed_table_p1.png",
            "row_binding": {"p1-t0": {"status": "contradicted"}},
            "row_binding_contradicted_tables": [0],
            "unchecked_table_kind": "disputed",
            "withheld_table_text": "| bad |",
            "audit_events": [
                *side.get("audit_events", []),
                {
                    "kind": "table_ladder_withheld",
                    "engine": "qwen",
                    "detail": "old table withheld",
                    "data": {"table_id": "p1-t0"},
                },
            ],
        }
    )
    side_path.write_text(json.dumps(side, indent=2), encoding="utf-8")

    audit_path = doc_dir / "audit_log.json"
    audit = (
        json.loads(audit_path.read_text(encoding="utf-8"))
        if audit_path.is_file()
        else {"pdf_filename": "scan.pdf", "event_count": 0, "counts": {}, "events": []}
    )
    audit["events"].append(
        {
            "page_num": 1,
            "kind": "table_ladder_withheld",
            "engine": "qwen",
            "detail": "old table withheld",
            "data": {"table_id": "p1-t0"},
        }
    )
    audit["event_count"] = len(audit["events"])
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    trust_path = doc_dir / "tables_trust.json"
    trust_path.write_text(
        json.dumps({"pdf_filename": "scan.pdf", "untrusted_pages": [1], "pages": {"1": {}}}),
        encoding="utf-8",
    )
    meta_path = doc_dir / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["error"] = "untrusted tables on 1 page(s), 1 flag(s) (see tables_trust.json)"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Lift using READING_B (tagged furniture) as reader A and READING_A as reader B.
    results, _ = _apply(out, pdf_dir, READING_A, a=READING_B)
    assert [r.outcome for r in results] == [tool.OUTCOME_LIFTED]

    lifted = _sidecar(out)
    assert lifted["judge_rejected"] is False
    assert lifted["table_ladder_disposition"] is None
    assert lifted["table_ladder_incomplete"] is False
    assert lifted["d3_floor_png_ref"] == ""
    for popped in (
        "table_judge_retry_pending",
        "table_judge_retry_rungs",
        "row_binding",
        "row_binding_contradicted_tables",
        "unchecked_table_kind",
        "withheld_table_text",
    ):
        assert popped not in lifted, popped
    assert "table_ladder_withheld" not in [e["kind"] for e in lifted["audit_events"]]
    assert not trust_path.exists(), "stale tables_trust.json must be removed once clean"
    assert "untrusted tables" not in (_meta(out).get("error") or "")

    # Simulate resuming this lifted page through orchestrator restore + re-flush.
    from socr.core.config import EngineType, PipelineConfig
    from socr.core.document import DocumentHandle
    from socr.core.manifest import finalized_page_record
    from socr.core.result import PageOutput
    from socr.core.state import DocumentState
    from socr.pipeline.orchestrator import UnifiedPipeline

    pdf = pdf_dir / "scan.pdf"
    pipe = UnifiedPipeline(
        PipelineConfig(
            primary_engine=EngineType.QWEN,
            local_engine=EngineType.QWEN,
            enabled_engines=[EngineType.QWEN],
            quiet=True,
        )
    )
    pipe._available_engines_for_agentic = lambda: []
    pipe._resolve_crop_vlm_model = lambda: None
    pipe._resolve_judge_model = lambda *a, **k: ""
    pipe._scan_root = pdf.parent
    state = DocumentState(handle=DocumentHandle.from_path(pdf))
    shipped_out = PageOutput.from_dict(lifted["winning_output"])
    pipe._restore_terminal_page_state(state, 1, shipped_out, out)
    assert state.pages[1].two_reader_agreement == lifted[rule.RULE]
    assert any(e.kind == rule.RULE and e.page_num == 1 for e in state.events)

    rec = finalized_page_record(state, 1)
    pipe._flush_page_sidecar(state, 1, out, terminal=True, record=rec)
    after_resume = _sidecar(out)
    assert after_resume.get(rule.RULE) == lifted[rule.RULE]
    assert acceptance.judge_page(
        acceptance.Page("doc", 1, after_resume, _fragment(out), None)
    ).checked


def test_crash_between_doc_md_and_sidecar_write_recovers_cleanly(scan_run):
    """If the tool crashed after writing the fragment and document ``.md`` but before writing
    the sidecar, rerunning must finish cleanly instead of raising ``RuntimeError``."""
    out, pdf_dir, _copy = scan_run
    if not _liftable(out):
        pytest.skip("the run's document is not one the tool rewrites in this provider state")

    side_path = _doc_dir(out) / "pages" / "00001.json"
    orig_sidecar = side_path.read_bytes()

    results1, _ = _apply(out, pdf_dir, READING_B)
    assert [r.outcome for r in results1] == [tool.OUTCOME_LIFTED]

    # Revert ONLY the sidecar to simulate a crash after fragment + doc_md were written.
    side_path.write_bytes(orig_sidecar)
    results2, _ = _apply(out, pdf_dir, READING_B)
    assert [r.outcome for r in results2] == [tool.OUTCOME_LIFTED]
    assert _sidecar(out)["engine"] == rule.ENGINE


def test_the_command_is_installed_under_its_name():
    import tomllib

    scripts = tomllib.loads(Path("pyproject.toml").read_text())["project"]["scripts"]
    assert scripts[tool.TOOL] == "socr.pipeline.refinalize_two_readers:main"
