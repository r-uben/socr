"""#1130: the acceptance meter, on synthetic run folders (hermetic, no provider, no model)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from ocr_output_contract import assemble_pages

from socr.benchmark import acceptance as acc
from socr.core.blank_page import blank_page_marker
from socr.core.manifest import DISPUTED_TABLE_MARK_RE, page_failed_marker
from socr.core.result import JUDGE_OUTCOME_COMPLETED
from socr.core.text_coverage import VERDICT_FLAGGED, VERDICT_PASSED, VERDICT_UNKNOWN

DISPOSITION = {"ending": "model_output", "primary_reason": "accepted_output"}
PASS_EVENT = {"data": {"result": acc.EVIDENCE_PASS}}


def test_loaded_source_is_this_checkout() -> None:
    import socr

    assert (
        Path(socr.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1] / "src")
    ), socr.__file__


def page(
    text: str = "Body text.",
    *,
    engine: str = "qwen",
    status: str = "success",
    failure_mode: str = "none",
    audit_passed: bool = True,
    events: list[dict] | None = None,
    wo_extra: dict | None = None,
    doc_text: str | None = None,
    text_coverage: str | None = None,
) -> dict:
    """One synthetic page: the sidecar fields the meter reads plus its two Markdown surfaces."""
    wo = {
        "status": status,
        "failure_mode": failure_mode,
        "audit_passed": audit_passed,
        "engine": engine,
        "text": text,
        "rejection_class": "",
        "judge_outcome": JUDGE_OUTCOME_COMPLETED if engine == "qwen" else "",
        "judge_reason": "",
        "table_disputed": "",
        "table_unverifiable": "",
        "table_label_unverified": "",
        **(wo_extra or {}),
    }
    return {
        "text": text,
        "doc_text": text if doc_text is None else doc_text,
        "sidecar": {
            "status": status,
            "failure_mode": failure_mode,
            "audit_passed": audit_passed,
            "terminal": True,
            "engine": engine,
            "disposition": DISPOSITION,
            "winning_output": wo,
            "audit_events": events or [],
            **({"text_coverage": text_coverage} if text_coverage is not None else {}),
        },
    }


def checked_native(text: str = "Native text.") -> dict:
    return page(text, engine="native", text_coverage=VERDICT_PASSED)


def write_doc(root: Path, stem: str, pages: list[dict], *, metadata: dict | None = None) -> Path:
    d = root / "t0" / "out" / "w0" / stem
    (d / "pages").mkdir(parents=True)
    entries = {}
    for n, p in enumerate(pages, start=1):
        side = {**p["sidecar"], "page_num": n}
        (d / "pages" / f"{n:05d}.json").write_text(json.dumps(side))
        (d / "pages" / f"{n:05d}.md").write_text(p["text"])
        entries[str(n)] = {
            "disposition": side["disposition"],
            "journal": [{"failure_mode": side["failure_mode"]}],
        }
    (d / "manifest.json").write_text(json.dumps({"entries": entries}))
    (d / f"{stem}.md").write_text(assemble_pages([p["doc_text"] for p in pages]))
    meta = {"key": f"{stem}.pdf", "status": "completed", "pages": len(pages)}
    (d / "metadata.json").write_text(json.dumps({**meta, **(metadata or {})}))
    return d


def write_failed_doc(root: Path, stem: str, doc_md: str | None) -> None:
    d = root / "t0" / "out" / "w0" / stem
    d.mkdir(parents=True)
    meta = {"key": f"{stem}.pdf", "status": "failed", "error": "unreadable_input: 0 of 2 loaded"}
    (d / "metadata.json").write_text(json.dumps(meta))
    if doc_md is not None:
        (d / f"{stem}.md").write_text(doc_md)


def inputs_csv(path: Path, docs: dict[str, int]) -> Path:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["out_file", "n_pages"])
        for stem, n in docs.items():
            w.writerow([f"{stem}.pdf", n])
    return path


def run_meter(tmp_path: Path, docs: dict[str, list[dict]]) -> acc.Report:
    root = tmp_path / "run"
    for stem, pages in docs.items():
        write_doc(root, stem, pages)
    return measure(root, {s: len(p) for s, p in docs.items()})


def measure(root: Path, docs: dict[str, int]) -> acc.Report:
    return acc.measure(root, [(f"{stem}.pdf", n) for stem, n in docs.items()])


def ids(rep: acc.Report, cls: str) -> set[str]:
    return set(rep.violations.get(cls, []))


MARK = page_failed_marker(1)


def test_clean_checked_and_marked_pages_score_zero(tmp_path: Path) -> None:
    rep = run_meter(
        tmp_path,
        {
            "d": [
                page("A fully judged page."),
                page(blank_page_marker(), engine="blank", events=[{"kind": acc.EVENT_BLANK_PAGE}]),
                checked_native(),
                page(page_failed_marker(4), status="error", failure_mode="empty_output"),
            ]
        },
    )
    assert rep.violating_ids() == set()
    assert rep.surface_count() == 0
    assert set(rep.checked) == {"qwen", "blank", "native"}
    assert len(rep.marked["error"]) == 1


def test_native_success_without_witness_differs_from_with_witness(tmp_path: Path) -> None:
    rep = run_meter(tmp_path, {"d": [page(engine="native"), checked_native()]})
    assert ids(rep, acc.NATIVE_UNCHECKED) == {"d:1"}


@pytest.mark.parametrize("verdict", [VERDICT_FLAGGED, VERDICT_UNKNOWN, None])
def test_only_a_passed_coverage_verdict_checks_a_native_page(tmp_path: Path, verdict) -> None:
    """#1050 step 1 writes ``text_coverage`` sparsely: passed / flagged / unknown / absent."""
    rep = run_meter(
        tmp_path, {"d": [page(engine="native", text_coverage=verdict), checked_native()]}
    )
    assert ids(rep, acc.NATIVE_UNCHECKED) == {"d:1"}


def test_native_equations_overlap_counts_once_in_meter(tmp_path: Path) -> None:
    unverified = {
        "kind": acc.EVENT_EQUATION_ATTACHED,
        "data": {"semantic_fidelity_verified": False},
    }
    rep = run_meter(tmp_path, {"d": [page(engine="native+equations", events=[unverified])]})
    assert ids(rep, acc.NATIVE_UNCHECKED) == {"d:1"}
    assert ids(rep, acc.EQUATION_UNVERIFIED_SUCCESS) == {"d:1"}
    assert len(rep.violating_ids()) == 1


def test_native_equations_verified_with_witness_is_checked(tmp_path: Path) -> None:
    events = [
        {"kind": acc.EVENT_EQUATION_ATTACHED, "data": {"semantic_fidelity_verified": True}},
    ]
    eq = page(engine="native+equations", events=events, text_coverage=VERDICT_PASSED)
    rep = run_meter(tmp_path, {"d": [eq]})
    assert rep.violating_ids() == set()


def test_native_math_success_is_never_checked(tmp_path: Path) -> None:
    rep = run_meter(tmp_path, {"d": [page(engine="native+math")]})
    assert ids(rep, acc.SUCCESS_UNCHECKED) == {"d:1"}


def test_chart_success_needs_figure_text_and_values_evidence(tmp_path: Path) -> None:
    text_only = [{"kind": acc.EVIDENCE_FIGURE_TEXT, **PASS_EVENT}]
    both = text_only + [{"kind": acc.EVIDENCE_CHART_VALUES, **PASS_EVENT}]
    not_transcribed = text_only + [{"kind": acc.EVENT_VALUES_NOT_TRANSCRIBED}]
    rep = run_meter(
        tmp_path,
        {
            "d": [
                page(engine="chart_asset"),
                page(engine="chart_asset", events=text_only),
                page(engine="chart_asset", events=both),
                page(engine="chart_asset", events=not_transcribed),
            ]
        },
    )
    assert ids(rep, acc.CHART_UNCHECKED) == {"d:1", "d:2"}
    assert rep.violating_ids() == {"d:1", "d:2"}


def test_success_with_failed_audit_is_a_violation(tmp_path: Path) -> None:
    rep = run_meter(tmp_path, {"d": [page(audit_passed=False)]})
    assert ids(rep, acc.SUCCESS_AUDIT_FALSE) == {"d:1"}


def test_unmarked_warning_splits_by_failure_mode_and_error_is_its_own_class(
    tmp_path: Path,
) -> None:
    rep = run_meter(
        tmp_path,
        {
            "d": [
                page(status="warning", failure_mode="none"),
                page(status="warning", failure_mode="audit_failed"),
                page(status="error", failure_mode="empty_output"),
            ]
        },
    )
    assert ids(rep, acc.WARNING_UNMARKED) == {"d:1", "d:2"}
    assert set(rep.warning_unmarked_by_failure_mode["none"]) == {"d:1"}
    assert set(rep.warning_unmarked_by_failure_mode["audit_failed"]) == {"d:2"}
    assert ids(rep, acc.ERROR_UNMARKED) == {"d:3"}


def test_success_page_carrying_a_mark_is_a_contradiction(tmp_path: Path) -> None:
    rep = run_meter(tmp_path, {"d": [page(f"Text.\n\n{MARK}")]})
    assert ids(rep, acc.SUCCESS_WITH_MARK) == {"d:1"}


def test_table_only_acceptance_needs_audit_passed_and_the_verifier_reason(tmp_path: Path) -> None:
    bypass = {"judge_reason": "native_table_verifier: EXACT_PASS"}
    rep = run_meter(
        tmp_path,
        {"d": [page(wo_extra=bypass), page(wo_extra={"judge_reason": "ok"})]},
    )
    assert ids(rep, acc.TABLE_ONLY_ACCEPTANCE) == {"d:1"}


def test_qwen_success_without_a_completed_judge_is_unchecked(tmp_path: Path) -> None:
    rep = run_meter(tmp_path, {"d": [page(wo_extra={"judge_outcome": "page_judge_timeout"})]})
    assert ids(rep, acc.SUCCESS_UNCHECKED) == {"d:1"}


@pytest.mark.parametrize(
    "hidden",
    ["<!-- {m} -->", "<!--\n{m}\n-->", "```\n{m}\n```", "~~~\n{m}\n~~~", "> ```\n> {m}\n> ```"],
)
def test_a_wrapped_mark_at_the_very_top_does_not_count(tmp_path: Path, hidden: str) -> None:
    # the wrapper IS the first block: a parser that rendered its content would mark the page
    text = hidden.format(m=MARK) + "\n\nProse."
    rep = run_meter(tmp_path, {"d": [page(text, status="warning", failure_mode="audit_failed")]})
    assert ids(rep, acc.WARNING_UNMARKED) == {"d:1"}
    # control: the bare line at the top is a mark, so the wrapper is what decided it
    top = MARK + "\n\nProse."
    rep2 = run_meter(
        tmp_path / "ctl", {"d": [page(top, status="warning", failure_mode="audit_failed")]}
    )
    assert rep2.violating_ids() == set()


def test_the_blank_receipt_is_not_a_mark(tmp_path: Path) -> None:
    blank = page(blank_page_marker(), engine="blank", events=[{"kind": acc.EVENT_BLANK_PAGE}])
    rep = run_meter(tmp_path, {"d": [blank]})
    assert ids(rep, acc.SUCCESS_WITH_MARK) == set()
    assert acc.leading_marks(blank_page_marker(), 1) == []


def test_blank_without_its_event_is_unchecked(tmp_path: Path) -> None:
    rep = run_meter(tmp_path, {"d": [page(blank_page_marker(), engine="blank")]})
    assert ids(rep, acc.SUCCESS_UNCHECKED) == {"d:1"}


def test_every_prose_recovery_banner_is_recognised_as_a_mark() -> None:
    """#1144 round 5 (Fable, PR #1158): a banner the lane stamps but the meter
    does not know is a MARKED page counted as a violation.

    `_scanned_flag_mark` listed the two SCANNED_* templates by name, which was
    complete while that lane only served scans. #1144 serves structure-class
    floors on BORN-DIGITAL pages, whose banner says "unverified extraction", and
    all 138 pages it recovers are born-digital -- so every one of them counted
    `error_unmarked`. Pinned over the EMITTERS' own enumerations, so a banner
    added later fails here instead of silently becoming a violation.
    """
    from socr.benchmark import acceptance
    from socr.core.manifest import NATIVE_TEXT_FLAGS, PROSE_RECOVERED_FLAGS

    flags = (*PROSE_RECOVERED_FLAGS, *NATIVE_TEXT_FLAGS)
    assert len(flags) == 4, "two banners x two page kinds; update this pin deliberately"
    for flag in flags:
        line = flag.format(page_num=7)
        assert acceptance._scanned_flag_mark(7, line), line
    # The born-digital wording specifically, since that is the one that was missing.
    assert any("unverified extraction" in f for f in flags)
    assert any("unverified scan" in f for f in flags)


def test_every_legacy_mark_family_is_recognised() -> None:
    disputed = "> [table disputed: a second reading (Tesseract) disagrees; check]"
    assert DISPUTED_TABLE_MARK_RE.match(disputed)
    samples = [
        page_failed_marker(3),
        "[page 3 failed: unverifiable table — see image]",
        disputed,
        "> [table unverified: socr could not check this table against the page; x]",
        "> [page unchecked: socr could not confirm this page's table(s)",
        "[page 3: flagged table kept — value drift — verify against the source before citing]",
        acc.SCANNED_PROSE_RECOVERED_FLAG.format(page_num=3),
        acc.SCANNED_NATIVE_TEXT_FLAG.format(page_num=3),
        acc.judge_timeout_note(3),
        acc.d3_superseded_note(3),
        "[page 3: uncovered text — see image]",
        '> [page 3 unchecked: native_coverage=UNKNOWN; scope=page; reason="no witness"]',
    ]
    for line in samples:
        assert acc.leading_marks(line, 3) == [line], line
    assert acc.leading_marks(page_failed_marker(4), 3) == []  # a mark naming another page


def test_marked_page_whose_document_is_not_its_assembly_is_a_violation(tmp_path: Path) -> None:
    ok = page_failed_marker(1)
    good = run_meter(tmp_path / "g", {"d": [page(ok, status="error", failure_mode="empty_output")]})
    assert good.violating_ids() == set()
    bad = page(ok, status="error", failure_mode="empty_output", doc_text="Different.")
    rep = run_meter(tmp_path / "b", {"d": [bad]})
    assert ids(rep, acc.DOCUMENT_NOT_ASSEMBLY) == {"d:1"}


def test_page_md_disagreeing_with_the_document_is_not_assembly(tmp_path: Path) -> None:
    agree = run_meter(tmp_path / "a", {"d": [page("Same text.")]})
    drift = run_meter(tmp_path / "b", {"d": [page("Same text.", doc_text="Other text.")]})
    assert agree.violating_ids() == set()
    assert ids(drift, acc.DOCUMENT_NOT_ASSEMBLY) == {"d:1"}


def test_trailing_whitespace_per_line_is_not_a_disagreement(tmp_path: Path) -> None:
    rep = run_meter(
        tmp_path, {"d": [page("Line one.  \nLine two.", doc_text="Line one.\nLine two.")]}
    )
    assert rep.surface_count() == 0


def test_disposition_and_failure_mode_mismatch_against_the_manifest(tmp_path: Path) -> None:
    root = tmp_path / "run"
    d = write_doc(root, "d", [page("Text.")])
    manifest = json.loads((d / "manifest.json").read_text())
    manifest["entries"]["1"]["disposition"] = {"ending": "x", "primary_reason": "y"}
    manifest["entries"]["1"]["journal"][-1]["failure_mode"] = "timeout"
    (d / "manifest.json").write_text(json.dumps(manifest))
    rep = measure(root, {"d": 1})
    assert rep.surface[acc.SURFACE_DISPOSITION] == ["d:1"]
    assert rep.diagnostic[acc.SURFACE_FAILURE_MODE] == ["d:1"]
    assert acc.SURFACE_FAILURE_MODE not in rep.surface  # diagnostic only: not in the gate


def test_missing_manifest_entry_is_unknown_not_agreement(tmp_path: Path) -> None:
    root = tmp_path / "run"
    d = write_doc(root, "d", [page("Text.")])
    (d / "manifest.json").write_text(json.dumps({"entries": {}}))
    rep = measure(root, {"d": 1})
    assert rep.surface[acc.SURFACE_DISPOSITION] == ["d:1"]


def test_document_in_the_input_manifest_with_no_output_is_unaccounted(tmp_path: Path) -> None:
    root = tmp_path / "run"
    write_doc(root, "present", [page("Text.")])
    rep = measure(root, {"present": 1, "absent": 3})
    assert ids(rep, acc.UNACCOUNTED) == {"absent:1", "absent:2", "absent:3"}


def test_missing_extra_and_nonterminal_pages_are_unaccounted_and_nonterminal_is_named(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    d = write_doc(root, "d", [page("One."), page("Two."), page("Three.")])
    side = json.loads((d / "pages" / "00002.json").read_text())
    side["terminal"] = False
    (d / "pages" / "00002.json").write_text(json.dumps(side))
    rep = measure(root, {"d": 2})
    assert ids(rep, acc.UNACCOUNTED) == {"d:2", "d:3"}
    assert ids(rep, acc.NONTERMINAL_PAGE) == {"d:2"}


def test_failed_document_is_marked_only_by_a_document_wide_failure_mark(tmp_path: Path) -> None:
    unmarked = tmp_path / "u"
    write_failed_doc(unmarked, "f", None)
    assert ids(measure(unmarked, {"f": 2}), acc.DOCUMENT_FAILED_UNMARKED) == {"f:1", "f:2"}
    # a page mark or a table mark names one page or one table, not the document
    for scoped in (page_failed_marker(1), "> [table disputed: a second reading disagrees; x]"):
        d = tmp_path / ("s" + str(abs(hash(scoped))))
        write_failed_doc(d, "f", scoped + "\n")
        assert ids(measure(d, {"f": 2}), acc.DOCUMENT_FAILED_UNMARKED) == {"f:1", "f:2"}
    marked = tmp_path / "m"
    write_failed_doc(marked, "f", "[document failed: unreadable input]\n")
    rep2 = measure(marked, {"f": 2})
    assert rep2.violating_ids() == set()
    assert len(rep2.marked["document_failed"]) == 2


def test_metadata_counts_missing_is_reported_outside_the_meter(tmp_path: Path) -> None:
    rep = run_meter(tmp_path, {"d": [page("Text.")]})
    assert rep.metadata_counts_missing == ["d"]
    assert rep.violating_ids() == set()
    assert rep.surface_count() == 0


def test_metadata_counts_that_disagree_with_sidecars_are_a_surface_violation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    write_doc(root, "d", [page("Text.")], metadata={acc.METADATA_COUNTS_FIELD: {"success": 2}})
    assert measure(root, {"d": 1}).surface[acc.SURFACE_METADATA] == ["d"]


def test_main_prints_meter_writes_report_and_sets_exit_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "run"
    write_doc(root, "bad", [page(engine="native")])
    write_doc(root, "ok", [page("Judged.")])
    csv_path = inputs_csv(tmp_path / "manifest.csv", {"bad": 1, "ok": 1, "gone": 2})
    report = tmp_path / "report.json"
    code = acc.main([str(root), "--inputs", str(csv_path), "--report", str(report)])
    out = capsys.readouterr().out
    assert code == 1
    assert out.startswith("METER = 3 of 4"), out
    data = json.loads(report.read_text())
    assert data["meter"] == 3
    assert data["violations"][acc.NATIVE_UNCHECKED] == ["bad:1"]
    assert data["violations"][acc.UNACCOUNTED] == ["gone:1", "gone:2"]


def test_main_exits_zero_when_everything_is_checked_or_marked(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "run"
    write_doc(root, "ok", [page("Judged.")])
    csv_path = inputs_csv(tmp_path / "manifest.csv", {"ok": 1})
    assert (
        acc.main([str(root), "--inputs", str(csv_path), "--report", str(tmp_path / "r.json")]) == 0
    )
    assert capsys.readouterr().out.startswith("METER = 0 of 1")


def test_surface_disagreement_alone_fails_the_exit_code(tmp_path: Path) -> None:
    root = tmp_path / "run"
    write_doc(root, "d", [page("Judged.", doc_text="Other.")])
    csv_path = inputs_csv(tmp_path / "manifest.csv", {"d": 1})
    assert (
        acc.main([str(root), "--inputs", str(csv_path), "--report", str(tmp_path / "r.json")]) == 1
    )


# ---- round 2 (Astra's review) ---------------------------------------------------------


def test_checked_page_must_equal_the_judged_text(tmp_path: Path) -> None:
    same = page("Judged text.")
    swapped = page("Judged text.")
    swapped["text"] = swapped["doc_text"] = "Replacement with every number removed."
    rep = run_meter(tmp_path, {"d": [same, swapped]})
    assert ids(rep, acc.SHIPPED_BYTES_DIFFER) == {"d:2"}
    assert rep.checked["qwen"] == ["d:1"]


def test_missing_judged_text_is_a_violation(tmp_path: Path) -> None:
    p = page("Judged text.")
    del p["sidecar"]["winning_output"]["text"]
    assert ids(run_meter(tmp_path, {"d": [p]}), acc.SHIPPED_BYTES_DIFFER) == {"d:1"}


def test_generated_marks_are_removed_before_the_bytes_compare(tmp_path: Path) -> None:
    mark = page_failed_marker(1)
    p = page(mark + "\n\nText.", status="warning", failure_mode="audit_failed")
    p["sidecar"]["winning_output"]["text"] = "Text."
    rep = run_meter(tmp_path, {"d": [p]})
    assert rep.violating_ids() == set()


HIDDEN = {
    "unclosed_comment": "<!--\n{m}",
    "comment_opened_earlier": "<!-- start\n\n{m}\n\nstill hidden",
    "invalid_fence_closer": "```\ncode\n``` not-a-closer\n{m}",
    "shorter_fence_closer": "````\ncode\n```\n{m}",
    "multiline_image_alt": "![alt text\n{m}\n](pic.png)",
    "multiline_link_text": "[link\n{m}\ntext](http://x)",
}


@pytest.mark.parametrize("hidden", list(HIDDEN))
def test_hidden_marks_do_not_make_a_page_marked(tmp_path: Path, hidden: str) -> None:
    text = HIDDEN[hidden].format(m=MARK)  # the wrapper opens the page: nothing precedes it
    rep = run_meter(tmp_path, {"d": [page(text, status="warning", failure_mode="audit_failed")]})
    assert ids(rep, acc.WARNING_UNMARKED) == {"d:1"}


def test_document_with_a_comment_before_page_one_is_not_the_assembly(tmp_path: Path) -> None:
    root = tmp_path / "run"
    d = write_doc(root, "d", [page("Judged and checked.")])
    doc = d / "d.md"
    doc.write_text("<!--\n" + doc.read_text())  # hides the whole page in a rendered document
    rep = measure(root, {"d": 1})
    assert ids(rep, acc.DOCUMENT_NOT_ASSEMBLY) == {"d:1"}
    assert rep.violating_ids() == {"d:1"}


def test_comment_closed_and_reopened_on_one_line_hides_nothing(tmp_path: Path) -> None:
    text = "<!--\nhidden --> shown <!--\n" + MARK + "\n-->"
    rep = run_meter(tmp_path, {"d": [page(text, status="warning", failure_mode="audit_failed")]})
    assert ids(rep, acc.WARNING_UNMARKED) == {"d:1"}


def test_blockquoted_fence_hides_nothing(tmp_path: Path) -> None:
    text = "> ```\n" + MARK + "\n> ```"
    rep = run_meter(tmp_path, {"d": [page(text, status="warning", failure_mode="audit_failed")]})
    assert ids(rep, acc.WARNING_UNMARKED) == {"d:1"}


def test_script_block_hides_nothing(tmp_path: Path) -> None:
    text = "<script>\n" + MARK + "\n</script>"
    rep = run_meter(tmp_path, {"d": [page(text, status="warning", failure_mode="audit_failed")]})
    assert ids(rep, acc.WARNING_UNMARKED) == {"d:1"}


def test_unlisted_output_without_metadata_is_found_by_structure(tmp_path: Path) -> None:
    root = tmp_path / "run"
    write_doc(root, "listed", [page("Text.")])
    stray = write_doc(root, "stray", [page("Bad.", status="warning", failure_mode="audit_failed")])
    (stray / "metadata.json").unlink()
    rep = measure(root, {"listed": 1})
    assert ids(rep, acc.UNLISTED_DOCUMENT) == {"stray:*"}
    assert acc.main_exit_code(rep) == 1


def test_listed_output_without_metadata_is_still_measured(tmp_path: Path) -> None:
    root = tmp_path / "run"
    d = write_doc(root, "d", [page("Bad.", status="warning", failure_mode="audit_failed")])
    (d / "metadata.json").unlink()
    assert ids(measure(root, {"d": 1}), acc.WARNING_UNMARKED) == {"d:1"}


def test_consecutive_top_marks_and_leading_blank_lines_count(tmp_path: Path) -> None:
    second = page_failed_marker(1).replace("no usable OCR output", "x y")
    text = "\n\n" + MARK + "\n" + second + "\n\nBody"
    rep = run_meter(tmp_path, {"d": [page(text, status="warning", failure_mode="audit_failed")]})
    assert rep.violating_ids() == set()
    assert len(acc.leading_marks(text, 1)) == 2


def test_mark_not_at_top_is_a_diagnostic_naming_the_page(tmp_path: Path) -> None:
    low = page("Body\n\n" + MARK, status="warning", failure_mode="audit_failed")
    none = page("Body", status="warning", failure_mode="audit_failed")
    rep = run_meter(tmp_path, {"d": [low, none]})
    assert rep.diagnostic[acc.MARK_NOT_AT_TOP] == ["d:1"]
    assert ids(rep, acc.WARNING_UNMARKED) == {"d:1", "d:2"}


def test_same_stem_under_two_paths_does_not_collapse(tmp_path: Path) -> None:
    root = tmp_path / "run"
    write_doc(root, "d", [page("One."), page("Two."), page("Three.")])
    rep = acc.measure(root, [("a/d.pdf", 3), ("b/d.pdf", 1)])
    assert rep.input_pages == 4
    assert ids(rep, acc.DUPLICATE_INPUT_STEM) == {
        "a/d.pdf:1",
        "a/d.pdf:2",
        "a/d.pdf:3",
        "b/d.pdf:1",
    }
    assert len(rep.violating_ids()) == 4


def test_noncanonical_page_file_is_reported_and_cannot_overwrite(tmp_path: Path) -> None:
    root = tmp_path / "run"
    d = write_doc(root, "d", [page(engine="native")])
    good = (d / "pages" / "00001.json").read_text()
    clean = json.loads(good)
    clean["engine"] = "qwen"
    (d / "pages" / "1.json").write_text(json.dumps(clean))
    (d / "pages" / "1.md").write_text("Body text.")
    rep = measure(root, {"d": 1})
    assert ids(rep, acc.NONCANONICAL_PAGE_FILE) == {"d:1.json", "d:1.md"}
    assert ids(rep, acc.NATIVE_UNCHECKED) == {"d:1"}


def test_unlisted_document_is_a_violation_even_with_no_pages(tmp_path: Path) -> None:
    root = tmp_path / "run"
    write_doc(root, "listed", [page("Text.")])
    write_failed_doc(root, "stray", None)
    rep = measure(root, {"listed": 1})
    assert ids(rep, acc.UNLISTED_DOCUMENT) == {"stray:*"}
    assert acc.main_exit_code(rep) == 1


def test_nonterminal_page_is_named(tmp_path: Path) -> None:
    root = tmp_path / "run"
    d = write_doc(root, "d", [page("One.")])
    side = json.loads((d / "pages" / "00001.json").read_text())
    side["terminal"] = False
    (d / "pages" / "00001.json").write_text(json.dumps(side))
    assert ids(measure(root, {"d": 1}), acc.NONTERMINAL_PAGE) == {"d:1"}


def test_disposition_missing_on_both_sides_is_unknown_not_agreement(tmp_path: Path) -> None:
    root = tmp_path / "run"
    d = write_doc(root, "d", [page("Text.")])
    side = json.loads((d / "pages" / "00001.json").read_text())
    del side["disposition"]
    (d / "pages" / "00001.json").write_text(json.dumps(side))
    man = json.loads((d / "manifest.json").read_text())
    del man["entries"]["1"]["disposition"]
    (d / "manifest.json").write_text(json.dumps(man))
    assert measure(root, {"d": 1}).surface[acc.SURFACE_DISPOSITION] == ["d:1"]


def test_failure_mode_diagnostic_does_not_fail_the_exit_code(tmp_path: Path) -> None:
    root = tmp_path / "run"
    d = write_doc(root, "d", [page("Text.")])
    man = json.loads((d / "manifest.json").read_text())
    man["entries"]["1"]["journal"][-1]["failure_mode"] = "timeout"
    (d / "manifest.json").write_text(json.dumps(man))
    rep = measure(root, {"d": 1})
    assert rep.diagnostic[acc.SURFACE_FAILURE_MODE] == ["d:1"]
    assert acc.main_exit_code(rep) == 0


# ---- round 4 (the CommonMark parser decides what is visible) ----------------------------

#: Every mark form, with a page number of 1.
MARK_FORMS = {
    "page_failed": page_failed_marker(1),
    "table_unverified": "> [table unverified: socr could not check this table]",
    "table_disputed": "> [table disputed: a second reading disagrees]",
    "page_unchecked": "> [page unchecked: socr could not confirm this page's table(s)]",
    "flagged_table": "[page 1: flagged table kept — value drift — verify before citing]",
    "canonical": "> [page 1 unchecked: native_coverage=UNKNOWN; scope=page; reason=none]",
}


@pytest.mark.parametrize("form", list(MARK_FORMS))
def test_a_mark_that_is_a_reference_definition_is_not_visible(tmp_path: Path, form: str) -> None:
    mark = MARK_FORMS[form]
    visible = run_meter(
        tmp_path / "v",
        {"d": [page(mark + "\n\nBody.", status="warning", failure_mode="audit_failed")]},
    )
    assert visible.violating_ids() == set(), form  # the form is a recognised mark
    hidden = mark + ": /hidden\n\nBody."
    rep = run_meter(
        tmp_path / "h", {"d": [page(hidden, status="warning", failure_mode="audit_failed")]}
    )
    assert ids(rep, acc.WARNING_UNMARKED) == {"d:1"}, form


def test_page_ending_with_an_open_comment_hides_the_next_page(tmp_path: Path) -> None:
    mark2 = page_failed_marker(2)
    one = page("Body one.\n\n<!--", status="success")
    two = page(mark2 + "\n\nmore\n\n-->", status="error", failure_mode="empty_output")
    rep = run_meter(tmp_path, {"d": [one, two]})
    assert ids(rep, acc.PAGE_LEAKS_RAW_HTML) == {"d:1"}
    assert ids(rep, acc.DOCUMENT_PAGE_HIDDEN) == {"d:2"}
    assert "d:2" in rep.violating_ids()
    assert rep.marked.get("error", []) == []  # a hidden mark is not counted MARKED


@pytest.mark.parametrize(
    "tail", ["<script>", "<pre>", "<style>", "<?php", "<![CDATA[", "<!DOCTYPE x"]
)
def test_unclosed_raw_html_blocks_leak(tmp_path: Path, tail: str) -> None:
    rep = run_meter(tmp_path, {"d": [page("Body.\n\n" + tail), page("Next.")]})
    assert ids(rep, acc.PAGE_LEAKS_RAW_HTML) == {"d:1"}


def test_closed_comment_and_ordinary_html_do_not_leak(tmp_path: Path) -> None:
    pages = [
        page("A <!-- note --> b.\n\n<!-- x -->"),
        page("<div>\nhtml\n</div>"),
        page("<b>x</b>"),
    ]
    rep = run_meter(tmp_path, {"d": pages})
    assert ids(rep, acc.PAGE_LEAKS_RAW_HTML) == set()
    assert ids(rep, acc.DOCUMENT_PAGE_HIDDEN) == set()


def test_first_block_html_comment_before_a_mark_is_not_marked(tmp_path: Path) -> None:
    text = "<!-- x -->\n\n" + MARK
    rep = run_meter(tmp_path, {"d": [page(text, status="warning", failure_mode="audit_failed")]})
    assert ids(rep, acc.WARNING_UNMARKED) == {"d:1"}


def test_mark_inside_a_paragraph_with_prose_is_not_marked(tmp_path: Path) -> None:
    text = MARK + "\nprose continues in the same paragraph"
    rep = run_meter(tmp_path, {"d": [page(text, status="warning", failure_mode="audit_failed")]})
    assert ids(rep, acc.WARNING_UNMARKED) == {"d:1"}


def test_document_mark_must_be_the_first_visible_block_under_its_heading(tmp_path: Path) -> None:
    root = tmp_path / "run"
    p = page(MARK, status="error", failure_mode="empty_output")
    d = write_doc(root, "d", [p])
    ok = measure(root, {"d": 1})
    assert ids(ok, acc.DOCUMENT_MARK_NOT_FIRST) == set()
    # same page fragment, but the document has a comment between heading and mark: the page
    # fragment itself is untouched, so only the parsed-document check can see it
    doc = d / "d.md"
    doc.write_text(doc.read_text().replace("## Page 1\n\n", "## Page 1\n\n<!-- hide -->\n\n"))
    rep = measure(root, {"d": 1})
    assert ids(rep, acc.DOCUMENT_MARK_NOT_FIRST) == {"d:1"}
    assert ids(rep, acc.DOCUMENT_NOT_ASSEMBLY) == {"d:1"}


def test_hidden_pages_are_those_until_the_next_visible_heading(tmp_path: Path) -> None:
    pages = [
        page("One."),
        page("Two.\n\n<!--"),
        page("Three."),
        page("Four.\n\n-->\n\nend"),
        page("Five."),
    ]
    rep = run_meter(tmp_path, {"d": pages})
    assert ids(rep, acc.DOCUMENT_PAGE_HIDDEN) == {"d:3", "d:4"}


# ---- round 5 (cubic review) -----------------------------------------------------------


def test_indented_fragment_is_not_the_judged_text(tmp_path: Path) -> None:
    indented = page("Judged text.")
    indented["text"] = indented["doc_text"] = "    Judged text."  # renders as a code block
    rep = run_meter(tmp_path, {"d": [page("Judged text."), indented]})
    assert ids(rep, acc.SHIPPED_BYTES_DIFFER) == {"d:2"}


def test_surrounding_blank_lines_and_trailing_spaces_are_still_ignored(tmp_path: Path) -> None:
    p = page("Judged text.")
    p["text"] = p["doc_text"] = "\n\nJudged text.   \n\n"
    p["sidecar"]["winning_output"]["text"] = "Judged text."
    assert run_meter(tmp_path, {"d": [p]}).violating_ids() == set()


def test_unlisted_document_is_reported_and_gates_but_is_not_a_page(tmp_path: Path) -> None:
    root = tmp_path / "run"
    write_doc(root, "listed", [page("Text.")])
    write_doc(root, "stray", [page("One."), page("Two.")])
    rep = measure(root, {"listed": 1})
    assert rep.violations[acc.UNLISTED_DOCUMENT] == ["stray:*"]
    assert rep.violating_ids() == {"stray:1", "stray:2"}  # real pages only, no "stray:*"
    assert rep.document_level_count() == 1
    assert acc.main_exit_code(rep) == 1


def test_unlisted_document_with_no_pages_still_fails_the_gate_with_meter_zero(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    write_doc(root, "listed", [page("Text.")])
    write_failed_doc(root, "stray", None)
    rep = measure(root, {"listed": 1})
    assert rep.violating_ids() == set()
    assert acc.main_exit_code(rep) == 1


def test_noncanonical_file_names_are_document_level(tmp_path: Path) -> None:
    root = tmp_path / "run"
    d = write_doc(root, "d", [page("Text.")])
    (d / "pages" / "extra.txt").write_text("x")
    rep = measure(root, {"d": 1})
    assert rep.violating_ids() == set()
    assert rep.document_level_count() == 1
    assert acc.main_exit_code(rep) == 1
