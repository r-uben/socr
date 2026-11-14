"""#1306: control bytes of a born-digital layer and hidden duplicate footnotes.

Two repairs at the shipping boundary (``manifest._apply_damaged_layer_guard``):

* a C0 control byte (not tab, LF, CR) becomes U+FFFD and the page is marked, never deleted:
  deleting ``\\x010.47`` would ship a clean, positive ``0.47`` for -0.47;
* the invisible tagged copy of a footnote (``[Footnote N. ... End footnote N.]``) is removed
  only when the printed note, same number and same full text, is on the page.

Hermetic: synthetic fitz PDFs, no provider, no judge. Pipeline pins are DIFFERENCES between
two runs in one process (the guard live vs switched off), in both provider states, never an
absolute outcome measured on one machine (CLAUDE.md, #257).
"""

from __future__ import annotations

import json
from pathlib import Path

import fitz
import pytest

from socr.core import manifest
from socr.core.config import EngineType, PipelineConfig
from socr.core.damaged_layer import (
    KEPT_NO_PRINTED_COPY,
    KEPT_NUMBER_MISMATCH,
    KEPT_PRINTED_NOTE_DIFFERS,
    REPLACEMENT_CHAR,
    replace_control_bytes,
    strip_hidden_footnotes,
)
from socr.core.result import DocumentStatus, FailureMode
from socr.pipeline.orchestrator import UnifiedPipeline
from socr.tables.layer_words import WordReference, missing_words

_PROSE = (
    "The estimated effect of the policy change on output is reported below for each "
    "specification, with robust standard errors clustered by region and year. "
)
NOTE = "The advance estimate for second-quarter GDP will be released tomorrow."


def test_loaded_source_is_this_checkout() -> None:
    import socr

    src = Path(__file__).resolve().parents[1] / "src"
    assert Path(socr.__file__).resolve().is_relative_to(src), socr.__file__


# ----------------------------------------------------------------------------- control bytes


def test_every_c0_code_but_tab_lf_cr_becomes_u_fffd() -> None:
    changed = {c for c in range(0x20) if replace_control_bytes(f"a{chr(c)}b")[0] != f"a{chr(c)}b"}
    assert changed == set(range(0x20)) - {9, 10, 13}
    assert replace_control_bytes("a\tb\nc\rd") == ("a\tb\nc\rd", {})


def test_a_control_byte_is_replaced_never_deleted() -> None:
    text, found = replace_control_bytes("coefficient \x010.47 and \x00% and CLASS\x03II")
    assert (
        text
        == f"coefficient {REPLACEMENT_CHAR}0.47 and {REPLACEMENT_CHAR}% and CLASS{REPLACEMENT_CHAR}II"
    )
    assert dict(found) == {"U+0001": 1, "U+0000": 1, "U+0003": 1}
    assert "0.47" in text and " 0.47" not in text, "the number must not read as a clean one"


# ----------------------------------------------------------------------------- footnote rule


def _page(note: str = NOTE, printed: str | None = NOTE, wrapper_note: str | None = None) -> str:
    wrapped = wrapper_note if wrapper_note is not None else note
    body = f"policy rates.1 [Footnote 1. {wrapped} End footnote 1.]   On the other hand."
    lines = ["Body text line.", body, "More body text."]
    if printed is not None:
        lines.append(f"1 {printed}")
    return "\n".join(lines)


def test_exact_printed_copy_strips_the_wrapper_and_records_it() -> None:
    result = strip_hidden_footnotes(_page())
    assert "[Footnote" not in result.text and "End footnote" not in result.text
    assert "policy rates.1 On the other hand." in result.text
    assert f"1 {NOTE}" in result.text, "the printed copy stays"
    assert [(s.note, s.printed_copies, s.text) for s in result.stripped] == [(1, 1, NOTE)]
    assert result.kept == ()


def test_printed_copy_may_wrap_and_use_other_whitespace() -> None:
    printed = NOTE.replace(" will", "\n   will")
    assert strip_hidden_footnotes(_page(printed=printed)).stripped


@pytest.mark.parametrize("end", ["End footnote 1.", "End of footnote 1.", "End of Footnote 1."])
def test_both_closing_forms_and_the_unbracketed_wrapper(end: str) -> None:
    text = f"MR. KOS.1 Footnote 1. {NOTE} {end}   Thank you.\n1 {NOTE}"
    assert strip_hidden_footnotes(text).text == f"MR. KOS.1 Thank you.\n1 {NOTE}"
    bracketed = f"MR. KOS.1 [Footnote 1. {NOTE} {end}]   Thank you.\n1 {NOTE}"
    assert strip_hidden_footnotes(bracketed).text == f"MR. KOS.1 Thank you.\n1 {NOTE}"


def test_a_differing_printed_note_keeps_the_wrapper() -> None:
    page = _page(printed="The advance estimate for second-quarter GDP will be released today.")
    result = strip_hidden_footnotes(page)
    assert result.text == page
    assert [(k.note, k.reason) for k in result.kept] == [(1, KEPT_PRINTED_NOTE_DIFFERS)]


def test_a_continued_note_keeps_the_wrapper() -> None:
    """The printed copy on this page holds only the first part; the rest is on the next page."""
    page = _page(note=NOTE + " The release includes revisions.", printed=NOTE)
    assert strip_hidden_footnotes(page).text == page


def test_a_lone_wrapper_is_kept() -> None:
    page = _page(printed=None)
    result = strip_hidden_footnotes(page)
    assert result.text == page
    assert [(k.note, k.reason) for k in result.kept] == [(1, KEPT_NO_PRINTED_COPY)]


def test_the_wrapper_never_vouches_for_itself() -> None:
    page = f"text.1 [Footnote 1. {NOTE} End footnote 1.] more"
    assert strip_hidden_footnotes(page).text == page


def test_a_printed_copy_with_another_number_does_not_count() -> None:
    page = _page(printed=None) + f"\n2 {NOTE}"
    assert strip_hidden_footnotes(page).text == page


def test_a_multi_line_wrapper_is_not_recognised_and_stays() -> None:
    page = f"text [Footnote 1. {NOTE}\n End footnote 1.] more\n1 {NOTE}"
    assert strip_hidden_footnotes(page).text == page


def test_opening_and_closing_numbers_must_agree() -> None:
    page = f"text [Footnote 1. {NOTE} End footnote 2.] more\n1 {NOTE}"
    result = strip_hidden_footnotes(page)
    assert result.text == page
    assert [(k.note, k.reason) for k in result.kept] == [(1, KEPT_NUMBER_MISMATCH)]


def test_two_notes_one_matching_one_not_strip_one_record() -> None:
    page = (
        f"a.1 [Footnote 1. {NOTE} End footnote 1.] b.2 [Footnote 2. Other words here. "
        f"End footnote 2.] c\n1 {NOTE}\n2 Different words here."
    )
    result = strip_hidden_footnotes(page)
    assert [s.note for s in result.stripped] == [1]
    assert "[Footnote 2." in result.text and "[Footnote 1." not in result.text


def test_the_old_lowercase_form_without_brackets_is_left_alone() -> None:
    page = (
        "Messrs. A,1 footnote 1 Attended the meeting.end of footnote 1 footnote 2 end of footnote 2"
    )
    assert strip_hidden_footnotes(page).text == page


# ----------------------------------------------------------------------------- words check


def test_stripping_causes_no_new_missing_word_and_hides_no_other() -> None:
    from socr.tables.layer_words import word_tokens_in_order

    page = _page()
    stripped = strip_hidden_footnotes(page)
    # The layer holds the note twice (hidden and printed) plus the wrapper's own words.
    reference = WordReference(" ".join(word_tokens_in_order(page)))
    assert (
        missing_words(
            reference, stripped.text, figure_shipped=False, excused=stripped.stripped_words
        )
        == []
    )
    # Without the excuse the stripped page would read as having lost the duplicate.
    assert missing_words(reference, stripped.text, figure_shipped=False)
    # A word the page really lost still counts: the excuse is the wrapper's words only.
    lost = stripped.text.replace("second-quarter", "")
    assert "second" in missing_words(
        reference, lost, figure_shipped=False, excused=stripped.stripped_words
    )


# ----------------------------------------------------------------------------- process()


class _StubEngine:
    name = "qwen"

    def __init__(self) -> None:
        self.calls = 0

    def is_available(self) -> bool:
        return True

    def process_pages(self, pdf_path, page_nums, config, dpi, subprocess_timeout=None, **_kw):
        from socr.core.result import PageOutput, PageStatus

        self.calls += 1
        return [
            PageOutput(page_num=n, text="PROVIDER_READ", status=PageStatus.SUCCESS, engine="qwen")
            for n in page_nums
        ]


class _AcceptingJudge:
    def assess(self, output, provider):
        from socr.pipeline.agentic import AcceptDecision

        return AcceptDecision(accept=True, reason="stub accepts all")


def _pdf(path: Path, lines: list[str]) -> Path:
    doc = fitz.open()
    page = doc.new_page()
    y = 72
    for _ in range(8):
        page.insert_text((72, y), _PROSE, fontname="helv", fontsize=10)
        y += 14
    for line in lines:
        page.insert_text((72, y + 10), line, fontname="helv", fontsize=10)
        y += 14
    doc.save(path)
    doc.close()
    return path


def _run(
    tmp_path: Path,
    tag: str,
    monkeypatch,
    lines: list[str],
    *,
    provider: bool,
    guard: bool = True,
    native_only: bool = False,
    reprocess: bool = False,
):
    from socr.core.providers import PROFILE_QWEN_LOCAL

    engine = _StubEngine()
    pdf = tmp_path / f"{tag}.pdf"
    if not pdf.exists():
        _pdf(pdf, lines)
    with monkeypatch.context() as m:
        m.setattr("socr.pipeline.orchestrator.get_engine", lambda engine_type: engine)
        if not guard:
            m.setattr(manifest, "_apply_damaged_layer_guard", lambda output, p: output)
        pipe = UnifiedPipeline(
            PipelineConfig(
                agentic=True,
                quiet=True,
                primary_engine=EngineType.QWEN,
                local_engine=EngineType.QWEN,
                enabled_engines=[EngineType.QWEN],
                native_first=True,
                native_only=native_only,
                reprocess=reprocess,
                write_manifest=False,
                judge_backend="heuristic",
                dual_pass_tables=False,
                detect_equations=False,
                save_figures=False,
            )
        )
        pipe._available_engines_for_agentic = lambda: [PROFILE_QWEN_LOCAL] if provider else []
        pipe._build_page_judge = lambda state: _AcceptingJudge()
        pipe._resolve_crop_vlm_model = lambda: None
        pipe._resolve_judge_model = lambda *a, **k: ""
        result = pipe.process(pdf, output_dir=tmp_path / f"out-{tag}")
    return result, engine


def _sidecar(tmp_path: Path, tag: str) -> dict:
    found = sorted((tmp_path / f"out-{tag}").rglob("pages/00001.json"))
    assert len(found) == 1, found
    return json.loads(found[0].read_text())


def _page_text(tmp_path: Path, tag: str) -> str:
    found = sorted((tmp_path / f"out-{tag}").rglob("pages/00001.md"))
    assert len(found) == 1, found
    return found[0].read_text()


def _final_md(tmp_path: Path, tag: str) -> str:
    found = [
        f
        for f in sorted((tmp_path / f"out-{tag}").rglob("*.md"))
        if "pages" not in f.parts and "equations" not in f.parts
    ]
    assert len(found) == 1, found
    return found[0].read_text()


def _kinds(side: dict) -> list[str]:
    return [e["kind"] for e in side["audit_events"]]


def _without_marks(text: str) -> str:
    return "\n".join(
        line for line in text.split("\n") if not line.startswith("> [page 1 unchecked")
    )


CONTROL_LINES = ["the coefficient is \x010.47 here"]
PROVIDER_STATES = [
    pytest.param(True, False, id="provider"),
    pytest.param(True, True, id="provider-native-only"),
    pytest.param(False, False, id="no-provider-native-fallback"),
]


def test_setup_canary_fitz_keeps_the_control_byte(tmp_path: Path) -> None:
    pdf = _pdf(tmp_path / "c.pdf", CONTROL_LINES)
    with fitz.open(pdf) as doc:
        assert "\x01" in doc[0].get_text("text")


@pytest.mark.parametrize(("provider", "native_only"), PROVIDER_STATES)
def test_control_byte_ships_as_u_fffd_with_a_mark_never_a_clean_number(
    tmp_path: Path, monkeypatch, provider: bool, native_only: bool
) -> None:
    kw = {"provider": provider, "native_only": native_only}
    _run(tmp_path, "off", monkeypatch, CONTROL_LINES, guard=False, **kw)
    result, _ = _run(tmp_path, "on", monkeypatch, CONTROL_LINES, **kw)
    off, on = _page_text(tmp_path, "off"), _page_text(tmp_path, "on")
    side = _sidecar(tmp_path, "on")

    if not provider or native_only:
        # The shipped text is native: the arms differ only by the repair and its mark.
        assert "\x01" in off, "setup: without the guard the byte ships"
        assert _without_marks(on).strip() == _without_marks(off).strip().replace(
            "\x01", REPLACEMENT_CHAR
        )
        assert f"is {REPLACEMENT_CHAR}0.47 here" in on
        assert "\x01" not in on and "is 0.47 here" not in on
        assert "control_bytes=FAIL" in on
        assert "control_bytes_replaced" in _kinds(side)
        event = next(e for e in side["audit_events"] if e["kind"] == "control_bytes_replaced")
        assert event["data"] == {"page": 1, "count": 1, "codes": {"U+0001": 1}}
        assert side["status"] != "success"
        assert result.status is not DocumentStatus.SUCCESS
        assert "control characters" in (result.error or "")
        # The shipped file and the fragment agree byte for byte.
        assert on in _final_md(tmp_path, "on")
    # No arm ships a control byte, whatever read the page.
    assert "\x01" not in on


def test_a_control_byte_outside_the_digit_case_no_longer_ships_a_clean_success(
    tmp_path: Path, monkeypatch
) -> None:
    """#990 reroutes only a byte before a digit. A NUL before ``%`` shipped as trusted native
    SUCCESS (2008-06-25 projections p20); it now ships U+FFFD, marked, demoted."""
    lines = ["rates rose 1 1/4 \x00% over the period"]
    _run(tmp_path, "off", monkeypatch, lines, provider=True, guard=False)
    result, engine = _run(tmp_path, "on", monkeypatch, lines, provider=True)
    off, on = _sidecar(tmp_path, "off"), _sidecar(tmp_path, "on")
    assert engine.calls == 0, "setup: the page stays on the native lane"
    assert off["status"] == "success"
    assert on["status"] == "warning"
    assert on["failure_mode"] == FailureMode.CONTROL_BYTES_REPLACED.value
    assert f"{REPLACEMENT_CHAR}%" in _page_text(tmp_path, "on")
    assert result.status is not DocumentStatus.SUCCESS


@pytest.mark.parametrize("provider", [True, False])
def test_a_clean_page_is_untouched_by_the_guard(tmp_path: Path, monkeypatch, provider) -> None:
    lines = ["the coefficient is 0.47 here"]
    _run(tmp_path, "off", monkeypatch, lines, provider=provider, guard=False)
    _run(tmp_path, "on", monkeypatch, lines, provider=provider)
    assert _page_text(tmp_path, "on") == _page_text(tmp_path, "off")
    assert _sidecar(tmp_path, "on")["status"] == _sidecar(tmp_path, "off")["status"]
    assert "control_bytes_replaced" not in _kinds(_sidecar(tmp_path, "on"))


FOOT_LINES = [
    f"policy rates.1 [Footnote 1. {NOTE} End footnote 1.]   On the other hand.",
    "more body text on the page.",
    f"1 {NOTE}",
]
DIFFERING_LINES = [
    FOOT_LINES[0],
    "more body text on the page.",
    "1 The advance estimate for second-quarter GDP will be released today.",
]


@pytest.mark.parametrize("provider", [True, False])
def test_hidden_footnote_stripped_once_with_a_record_and_no_new_warning(
    tmp_path: Path, monkeypatch, provider: bool
) -> None:
    _run(tmp_path, "off", monkeypatch, FOOT_LINES, provider=provider, guard=False)
    result, _ = _run(tmp_path, "on", monkeypatch, FOOT_LINES, provider=provider)
    off, on = _page_text(tmp_path, "off"), _page_text(tmp_path, "on")
    side_off, side_on = _sidecar(tmp_path, "off"), _sidecar(tmp_path, "on")

    assert "[Footnote 1." in off, "setup: the layer carries the hidden copy"
    assert "[Footnote" not in on and "End footnote" not in on
    assert f"1 {NOTE}" in on
    assert on.count(NOTE) == 1 and off.count(NOTE) == 2
    stripped = [e for e in side_on["audit_events"] if e["kind"] == "hidden_footnote_stripped"]
    assert [e["data"]["note"] for e in stripped] == [1]
    assert stripped[0]["data"]["printed_copies"] == 1
    # Stripping is a repair, never a demotion, and the words check sees no new loss.
    assert side_on["status"] == side_off["status"]
    assert side_on["failure_mode"] == side_off["failure_mode"]
    assert "words_missing_from_page" not in _kinds(side_on)
    assert "words_missing_from_page" not in _kinds(side_off)
    assert result.status is not DocumentStatus.ERROR
    assert on in _final_md(tmp_path, "on")
    if side_off.get("text_layer_words") is not None:
        assert side_on["text_layer_words"]["missing"] == 0


@pytest.mark.parametrize("provider", [True, False])
def test_a_differing_printed_note_leaves_both_arms_identical(
    tmp_path: Path, monkeypatch, provider: bool
) -> None:
    _run(tmp_path, "off", monkeypatch, DIFFERING_LINES, provider=provider, guard=False)
    _run(tmp_path, "on", monkeypatch, DIFFERING_LINES, provider=provider)
    assert _page_text(tmp_path, "on") == _page_text(tmp_path, "off")
    assert "[Footnote 1." in _page_text(tmp_path, "on")
    assert "hidden_footnote_stripped" not in _kinds(_sidecar(tmp_path, "on"))


def test_a_lone_wrapper_is_kept_through_the_pipeline(tmp_path: Path, monkeypatch) -> None:
    lines = FOOT_LINES[:2]
    _run(tmp_path, "off", monkeypatch, lines, provider=True, guard=False)
    _run(tmp_path, "on", monkeypatch, lines, provider=True)
    assert _page_text(tmp_path, "on") == _page_text(tmp_path, "off")


def _spy_restores(monkeypatch) -> list:
    restored: list = []
    orig = UnifiedPipeline._load_terminal_page

    def spy(self, *a, **kw):
        out = orig(self, *a, **kw)
        if out is not None:
            restored.append(out)
        return out

    monkeypatch.setattr(UnifiedPipeline, "_load_terminal_page", spy)
    return restored


def test_resume_restores_the_stripped_page_and_keeps_its_event(tmp_path: Path, monkeypatch) -> None:
    _run(tmp_path, "r", monkeypatch, FOOT_LINES, provider=True)
    first_text = _page_text(tmp_path, "r")
    first_events = [
        e
        for e in _sidecar(tmp_path, "r")["audit_events"]
        if e["kind"] == "hidden_footnote_stripped"
    ]
    assert len(first_events) == 1
    restored = _spy_restores(monkeypatch)
    _run(tmp_path, "r", monkeypatch, FOOT_LINES, provider=True, reprocess=True)
    assert len(restored) == 1, "setup: the ledger restores the finished page"
    assert _page_text(tmp_path, "r") == first_text
    again = [
        e
        for e in _sidecar(tmp_path, "r")["audit_events"]
        if e["kind"] == "hidden_footnote_stripped"
    ]
    assert again == first_events, "one record, restored, not lost and not doubled"


def test_resume_restores_the_control_byte_page_with_its_mark_and_event(
    tmp_path: Path, monkeypatch
) -> None:
    lines = ["rates rose 1 1/4 \x00% over the period"]
    _run(tmp_path, "q", monkeypatch, lines, provider=True)
    first = _sidecar(tmp_path, "q")
    first_text = _page_text(tmp_path, "q")
    restored = _spy_restores(monkeypatch)
    result, _ = _run(tmp_path, "q", monkeypatch, lines, provider=True, reprocess=True)
    assert len(restored) == 1, "a control-byte page is permanent: the same layer, the same bytes"
    again = _sidecar(tmp_path, "q")
    assert _page_text(tmp_path, "q") == first_text
    assert again["failure_mode"] == first["failure_mode"] == "control_bytes_replaced"
    assert [e for e in again["audit_events"] if e["kind"] == "control_bytes_replaced"] == [
        e for e in first["audit_events"] if e["kind"] == "control_bytes_replaced"
    ]
    assert result.status is not DocumentStatus.SUCCESS
