"""#1306: control bytes of a born-digital layer and hidden duplicate footnotes.

Two repairs at the shipping boundary (``manifest._apply_damaged_layer_guard``):

* a C0 control byte (not tab, LF, CR) becomes U+FFFD and the page is marked, never deleted:
  deleting ``\\x010.47`` would ship a clean, positive ``0.47`` for -0.47;
* the invisible tagged copy of a footnote (``[Footnote N. ... End footnote N.]``) is removed
  only from native text, only when the PDF's painted glyphs do not draw it (hidden text), and
  only when the printed note is a complete note line, same number and same full text.

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
    KEPT_NOT_EVIDENCED_HIDDEN,
    REPLACEMENT_CHAR,
    painted_form,
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


#: What a page that paints only its visible words (not the wrapper's frame) reads as.
CLEAN = painted_form(f"policy rates.1 On the other hand. Body text line. More body text. 1 {NOTE}")


def _strip(text: str, painted: str | None = CLEAN):
    return strip_hidden_footnotes(text, painted)


def _page(note: str = NOTE, printed: str | None = NOTE, wrapper_note: str | None = None) -> str:
    wrapped = wrapper_note if wrapper_note is not None else note
    body = f"policy rates.1 [Footnote 1. {wrapped} End footnote 1.]   On the other hand."
    lines = ["Body text line.", body, "More body text."]
    if printed is not None:
        lines.append(f"1 {printed}")
    return "\n".join(lines)


def test_exact_printed_copy_strips_the_wrapper_and_records_it() -> None:
    result = _strip(_page())
    assert "[Footnote" not in result.text and "End footnote" not in result.text
    assert "policy rates.1 On the other hand." in result.text
    assert f"1 {NOTE}" in result.text, "the printed copy stays"
    assert [(s.note, s.printed_copies, s.text) for s in result.stripped] == [(1, 1, NOTE)]
    assert result.kept == ()


def test_printed_copy_may_wrap_and_use_other_whitespace() -> None:
    printed = NOTE.replace(" will", "\n   will")
    assert _strip(_page(printed=printed)).stripped


@pytest.mark.parametrize("end", ["End footnote 1.", "End of footnote 1.", "End of Footnote 1."])
def test_both_closing_forms_and_the_unbracketed_wrapper(end: str) -> None:
    text = f"MR. KOS.1 Footnote 1. {NOTE} {end}   Thank you.\n1 {NOTE}"
    assert _strip(text).text == f"MR. KOS.1 Thank you.\n1 {NOTE}"
    bracketed = f"MR. KOS.1 [Footnote 1. {NOTE} {end}]   Thank you.\n1 {NOTE}"
    assert _strip(bracketed).text == f"MR. KOS.1 Thank you.\n1 {NOTE}"


def test_a_differing_printed_note_keeps_the_wrapper() -> None:
    page = _page(printed="The advance estimate for second-quarter GDP will be released today.")
    result = _strip(page)
    assert result.text == page
    assert [(k.note, k.reason) for k in result.kept] == [(1, KEPT_PRINTED_NOTE_DIFFERS)]


def test_a_continued_note_keeps_the_wrapper() -> None:
    """The printed copy on this page holds only the first part; the rest is on the next page."""
    page = _page(note=NOTE + " The release includes revisions.", printed=NOTE)
    assert _strip(page).text == page


def test_a_lone_wrapper_is_kept() -> None:
    page = _page(printed=None)
    result = _strip(page)
    assert result.text == page
    assert [(k.note, k.reason) for k in result.kept] == [(1, KEPT_NO_PRINTED_COPY)]


def test_the_wrapper_never_vouches_for_itself() -> None:
    page = f"text.1 [Footnote 1. {NOTE} End footnote 1.] more"
    assert _strip(page).text == page


def test_a_printed_copy_with_another_number_does_not_count() -> None:
    page = _page(printed=None) + f"\n2 {NOTE}"
    assert _strip(page).text == page


def test_a_multi_line_wrapper_is_not_recognised_and_stays() -> None:
    page = f"text [Footnote 1. {NOTE}\n End footnote 1.] more\n1 {NOTE}"
    assert _strip(page).text == page


def test_opening_and_closing_numbers_must_agree() -> None:
    page = f"text [Footnote 1. {NOTE} End footnote 2.] more\n1 {NOTE}"
    result = _strip(page)
    assert result.text == page
    assert [(k.note, k.reason) for k in result.kept] == [(1, KEPT_NUMBER_MISMATCH)]


def test_two_notes_one_matching_one_not_strip_one_record() -> None:
    page = (
        f"a.1 [Footnote 1. {NOTE} End footnote 1.] b.2 [Footnote 2. Other words here. "
        f"End footnote 2.] c\n1 {NOTE}\n2 Different words here."
    )
    result = _strip(page)
    assert [s.note for s in result.stripped] == [1]
    assert "[Footnote 2." in result.text and "[Footnote 1." not in result.text


def test_the_old_lowercase_form_without_brackets_is_left_alone() -> None:
    page = (
        "Messrs. A,1 footnote 1 Attended the meeting.end of footnote 1 footnote 2 end of footnote 2"
    )
    assert _strip(page).text == page


# ----------------------------------------------------------------------------- words check


def test_stripping_causes_no_new_missing_word_and_hides_no_other() -> None:
    from socr.tables.layer_words import word_tokens_in_order

    page = _page()
    stripped = _strip(page)
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


# --- hidden-text evidence (Astra blocker 1): the text alone is never proof


def test_a_wrapper_the_pdf_paints_is_kept_even_with_a_matching_printed_note() -> None:
    """A visible quoted example, printed in full, and the real note elsewhere."""
    page = _page()
    painted = painted_form(f"[Footnote 1. {NOTE} End footnote 1.] 1 {NOTE}")
    result = _strip(page, painted)
    assert result.text == page
    assert [(k.note, k.reason) for k in result.kept] == [(1, KEPT_NOT_EVIDENCED_HIDDEN)]
    assert result.stripped == () and result.stripped_words == ""


def test_painted_glyphs_not_read_keep_every_wrapper() -> None:
    page = _page()
    result = _strip(page, None)
    assert result.text == page
    assert [(k.note, k.reason) for k in result.kept] == [(1, KEPT_NOT_EVIDENCED_HIDDEN)]


@pytest.mark.parametrize(
    "frame", ["Footnote 1.", "End footnote 1.", "End of footnote 1."], ids=["open", "end", "end-of"]
)
def test_either_painted_frame_half_is_enough_to_keep(frame: str) -> None:
    painted = painted_form(f"x {frame} y 1 {NOTE}")
    assert _strip(_page(), painted).text == _page()


def test_a_printed_note_that_only_begins_with_the_wrapper_words_does_not_count() -> None:
    """cubic: a longer, differing printed note shares the wrapper's words as a prefix."""
    page = _page(printed=NOTE + " It is also revised later.")
    result = _strip(page)
    assert result.text == page
    assert [(k.note, k.reason) for k in result.kept] == [(1, KEPT_PRINTED_NOTE_DIFFERS)]


def test_a_printed_copy_must_start_its_line() -> None:
    page = _page(printed=None) + f"\nsee also 1 {NOTE}"
    result = _strip(page)
    assert result.text == page
    assert [(k.note, k.reason) for k in result.kept] == [(1, KEPT_NO_PRINTED_COPY)]


# --- token boundary (Astra blocker 2)


def test_an_attached_word_stays_a_separate_token() -> None:
    page = f"Growth[Footnote 1. {NOTE} End footnote 1.] slowed.\n1 {NOTE}"
    assert _strip(page).text == f"Growth slowed.\n1 {NOTE}"
    attached = f"Growth[Footnote 1. {NOTE} End footnote 1.]slowed.\n1 {NOTE}"
    assert _strip(attached).text == f"Growth slowed.\n1 {NOTE}"


def test_an_attached_number_stays_a_separate_token() -> None:
    page = f"rose 1[Footnote 1. {NOTE} End footnote 1.] 2 percent.\n1 {NOTE}"
    assert _strip(page).text == f"rose 1 2 percent.\n1 {NOTE}"
    attached = f"rose 1[Footnote 1. {NOTE} End footnote 1.]2 percent.\n1 {NOTE}"
    assert _strip(attached).text == f"rose 1 2 percent.\n1 {NOTE}"


def test_a_wrapper_before_punctuation_leaves_no_space() -> None:
    w = f"[Footnote 1. {NOTE} End footnote 1.]"
    assert _strip(f"the rate{w}.\n1 {NOTE}").text == f"the rate.\n1 {NOTE}"
    assert _strip(f"the rate {w}.\n1 {NOTE}").text == f"the rate.\n1 {NOTE}"
    assert _strip(f"Growth{w} slowed\n1 {NOTE}").text == f"Growth slowed\n1 {NOTE}"
    assert _strip(f"1{w} 2\n1 {NOTE}").text == f"1 2\n1 {NOTE}"
    assert _strip(f"x {w} (y)\n1 {NOTE}").text == f"x (y)\n1 {NOTE}"


def test_edge_and_adjacent_wrappers_keep_one_space_or_none() -> None:
    w = f"[Footnote 1. {NOTE} End footnote 1.]"
    assert _strip(f"{w} Starts the line.\n1 {NOTE}").text == f"Starts the line.\n1 {NOTE}"
    assert _strip(f"Ends the line. {w}\n1 {NOTE}").text == f"Ends the line.\n1 {NOTE}"
    two = f"a {w}{w} b\n1 {NOTE}"
    assert _strip(two).text == f"a b\n1 {NOTE}"


@pytest.mark.parametrize("painted_n", ["01", "1"])
def test_a_leading_zero_wrapper_painted_in_either_form_is_kept(painted_n: str) -> None:
    """Astra round 3: the layer says ``Footnote 01.``, the printed line ``01 Note.``; the
    frame is painted as ``01`` or ``1``. Either way it is visible, so it stays."""
    w = f"[Footnote 01. {NOTE} End footnote 01.]"
    page = f"policy rates.1 {w} More.\n01 {NOTE}"
    painted = painted_form(
        f"policy rates.1 [Footnote {painted_n}. {NOTE} End footnote {painted_n}.]"
    )
    assert _strip(page, painted).text == page
    assert _strip(page, CLEAN).stripped, "setup: with no painted frame it is stripped"


# ----------------------------------------------------------------------------- process()


class _StubEngine:
    name = "qwen"

    def __init__(self, text: str = "PROVIDER_READ") -> None:
        self.calls = 0
        self.text = text

    def is_available(self) -> bool:
        return True

    def process_pages(self, pdf_path, page_nums, config, dpi, subprocess_timeout=None, **_kw):
        from socr.core.result import PageOutput, PageStatus

        self.calls += 1
        return [
            PageOutput(page_num=n, text=self.text, status=PageStatus.SUCCESS, engine="qwen")
            for n in page_nums
        ]


class _AcceptingJudge:
    def assess(self, output, provider):
        from socr.pipeline.agentic import AcceptDecision

        return AcceptDecision(accept=True, reason="stub accepts all")


def _hidden(text: str) -> tuple[str, bool]:
    """A segment drawn invisibly (render mode 3): in the layer, not painted."""
    return (text, True)


def _pdf(path: Path, lines: list) -> Path:
    doc = fitz.open()
    page = doc.new_page()
    y = 72
    for _ in range(8):
        page.insert_text((72, y), _PROSE, fontname="helv", fontsize=10)
        y += 14
    for line in lines:
        # A line is a string, or segments drawn left to right on one baseline.
        x = 72
        for seg in [(line, False)] if isinstance(line, str) else line:
            text, hidden = seg if isinstance(seg, tuple) else (seg, False)
            # Hidden text is small so the line stays inside the page (a clipped word is lost).
            size = 5 if hidden else 10
            page.insert_text(
                (x, y + 10), text, fontname="helv", fontsize=size, render_mode=3 if hidden else 0
            )
            x += fitz.get_text_length(text + " ", "helv", size)
        y += 14
    doc.save(path)
    doc.close()
    return path


def _run(
    tmp_path: Path,
    tag: str,
    monkeypatch,
    lines: list,
    *,
    provider: bool,
    model_text: str = "PROVIDER_READ",
    guard: bool = True,
    native_only: bool = False,
    reprocess: bool = False,
):
    from socr.core.providers import PROFILE_QWEN_LOCAL

    engine = _StubEngine(model_text)
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


def test_a_control_byte_in_a_model_reading_is_replaced_and_marked(
    tmp_path: Path, monkeypatch
) -> None:
    """The provider arm with the byte reaching shipped text: a clean native page cannot carry
    one, so the model's own reading does. Guard off ships the byte; guard on ships U+FFFD and
    the mark; the engine path (the stub was called) is the same in both arms."""
    reading = "the model read a coefficient of \x010.47 here"
    lines = CONTROL_LINES
    kw = {"provider": True, "model_text": reading}
    _, eng_off = _run(tmp_path, "off", monkeypatch, lines, guard=False, **kw)
    result, eng_on = _run(tmp_path, "on", monkeypatch, lines, **kw)
    off, on = _page_text(tmp_path, "off"), _page_text(tmp_path, "on")
    assert eng_on.calls == eng_off.calls
    assert eng_on.calls > 0, "setup: the page went to the provider"
    assert "\x01" in off, "setup: without the guard the model's byte ships"
    assert "\x01" not in on and f"{REPLACEMENT_CHAR}0.47" in on
    assert "control_bytes=FAIL" in on
    assert "control_bytes_replaced" in _kinds(_sidecar(tmp_path, "on"))
    assert result.status is not DocumentStatus.SUCCESS


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


WRAPPER = f"[Footnote 1. {NOTE} End footnote 1.]"
FOOT_LINES = [
    ["policy rates.1", _hidden(WRAPPER), "On the other hand."],
    "more body text on the page.",
    "",
    "",
    f"1 {NOTE}",  # a footnote sits apart from the body: its own paragraph in the shipped text
]
VISIBLE_WRAPPER_LINES = [  # the same bytes in the layer, but the PDF paints them
    f"policy rates.1 {WRAPPER} On the other hand.",
    "more body text on the page.",
    "",
    "",
    f"1 {NOTE}",  # a footnote sits apart from the body: its own paragraph in the shipped text
]
DIFFERING_LINES = [
    FOOT_LINES[0],
    "more body text on the page.",
    "",
    "",
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


@pytest.mark.parametrize("provider", [True, False])
def test_a_wrapper_the_pdf_paints_is_kept_through_the_pipeline(
    tmp_path: Path, monkeypatch, provider: bool
) -> None:
    """Same bytes in the layer as FOOT_LINES, but drawn: the guard must leave the page alone."""
    _run(tmp_path, "off", monkeypatch, VISIBLE_WRAPPER_LINES, provider=provider, guard=False)
    _run(tmp_path, "on", monkeypatch, VISIBLE_WRAPPER_LINES, provider=provider)
    on = _page_text(tmp_path, "on")
    assert "[Footnote 1." in on, "setup: the wrapper is in the shipped layer"
    assert on == _page_text(tmp_path, "off")
    assert "hidden_footnote_stripped" not in _kinds(_sidecar(tmp_path, "on"))


def test_the_inline_strip_keeps_the_token_boundary_through_the_pipeline(
    tmp_path: Path, monkeypatch
) -> None:
    lines = [["Growth", _hidden(WRAPPER), "slowed."], "", "", f"1 {NOTE}"]
    _run(tmp_path, "on", monkeypatch, lines, provider=False)
    on = _page_text(tmp_path, "on")
    assert "[Footnote" not in on
    assert "Growth slowed." in on and "Growthslowed" not in on


@pytest.mark.parametrize("provider", [True, False])
def test_a_leading_zero_wrapper_the_pdf_paints_is_kept_through_the_pipeline(
    tmp_path: Path, monkeypatch, provider: bool
) -> None:
    w = f"[Footnote 01. {NOTE} End footnote 01.]"
    lines = [f"policy rates.1 {w} On the other hand.", "more body text.", "", "", f"01 {NOTE}"]
    _run(tmp_path, "off", monkeypatch, lines, provider=provider, guard=False)
    _run(tmp_path, "on", monkeypatch, lines, provider=provider)
    on = _page_text(tmp_path, "on")
    assert "[Footnote 01." in on, "setup: the painted wrapper is in the shipped layer"
    assert on == _page_text(tmp_path, "off")
    assert "hidden_footnote_stripped" not in _kinds(_sidecar(tmp_path, "on"))


def test_the_leading_zero_wrapper_drawn_invisibly_is_stripped(tmp_path: Path, monkeypatch) -> None:
    w = f"[Footnote 01. {NOTE} End footnote 01.]"
    lines = [["policy rates.", _hidden(w), "On the other hand."], "", "", f"01 {NOTE}"]
    _run(tmp_path, "on", monkeypatch, lines, provider=False)
    assert "[Footnote" not in _page_text(tmp_path, "on")


def _guard(engine: str, painted: str | None, text: str):
    from socr.core.result import PageOutput, PageStatus
    from socr.core.state import PageState

    p = PageState(page_num=1)
    p.painted_layer_text = painted
    out = PageOutput(page_num=1, text=text, status=PageStatus.SUCCESS, engine=engine)
    return manifest._apply_damaged_layer_guard(out, p).text, p


def test_the_guard_never_rewrites_a_model_reading() -> None:
    """Astra/cubic: a model reading that happens to carry a wrapper and a printed note is the
    model's text. It is not stripped, and nothing is recorded for the event writer."""
    page = _page()
    text, p = _guard("qwen", CLEAN, page)
    assert text == page
    assert p.hidden_footnotes_stripped == []


def test_the_guard_strips_native_text_only_with_painted_evidence() -> None:
    page = _page()
    text, p = _guard("native", CLEAN, page)
    assert "[Footnote" not in text
    assert [s["note"] for s in p.hidden_footnotes_stripped] == [1]
    assert _guard("native", None, page)[0] == page, "no evidence, no strip"


def _actualtext_pdf(path: Path, replacement: str) -> Path:
    """A page whose visible ``1`` carries ``/ActualText`` *replacement*, as the Fed's tagged
    PDFs carry the footnote wrapper: extraction returns the replacement, the page paints ``1``."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 50), "placeholder", fontname="helv")  # creates the content stream
    hexed = "FEFF" + replacement.encode("utf-16-be").hex().upper()
    stream = "BT /helv 10 Tf 72 700 Td (policy rates.) Tj ET\n"
    stream += f"BT /helv 10 Tf 150 700 Td /Span <</ActualText <{hexed}>>> BDC (1) Tj EMC ET\n"
    doc.update_stream(page.get_contents()[0], stream.encode())
    doc.save(path)
    doc.close()
    return path


def test_painted_glyphs_of_an_actualtext_wrapper_page_lack_the_frame(tmp_path: Path) -> None:
    """The layer reads the wrapper; the page paints only the ``1``. This is the evidence the
    strip rests on, taken from a real ``/ActualText`` marked-content PDF."""
    from socr.core.born_digital import BornDigitalDetector

    pdf = _actualtext_pdf(tmp_path / "a.pdf", f"1 {WRAPPER}")
    with fitz.open(pdf) as doc:
        page = doc[0]
        assert "[Footnote 1." in page.get_text("text"), "setup: the layer carries the wrapper"
        painted = BornDigitalDetector()._painted_layer_text(page)
    assert painted is not None and "footnote" not in painted
    assert "policyrates.1" in painted
    # And a wrapper actually drawn is in the painted glyphs.
    drawn = _pdf(tmp_path / "d.pdf", [WRAPPER])
    with fitz.open(drawn) as doc:
        assert "footnote1." in BornDigitalDetector()._painted_layer_text(doc[0])


def _tr3_pdf(path: Path, mode: int) -> Path:
    """A page whose wrapper is drawn in render mode *mode* at full opacity."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 50), "placeholder", fontname="helv")
    stream = "BT /helv 10 Tf 72 700 Td (policy rates.) Tj ET\n"
    stream += f"BT /helv 10 Tf {mode} Tr 72 680 Td ({WRAPPER}) Tj ET\n"
    doc.update_stream(page.get_contents()[0], stream.encode())
    doc.save(path)
    doc.close()
    return path


@pytest.mark.parametrize("mode,visible", [(3, False), (0, True)])
def test_invisible_render_mode_text_is_not_painted_at_full_opacity(
    tmp_path: Path, mode: int, visible: bool
) -> None:
    from socr.core.born_digital import BornDigitalDetector

    pdf = _tr3_pdf(tmp_path / f"t{mode}.pdf", mode)
    with fitz.open(pdf) as doc:
        painted = BornDigitalDetector()._painted_layer_text(doc[0])
    assert painted is not None
    assert ("footnote1." in painted) is visible
    assert "policyrates." in painted


def test_a_span_without_ink_flags_is_not_painted_whatever_its_alpha() -> None:
    """The filled / stroked bits are read in addition to alpha: a span with alpha 255 and
    neither bit (what a render-mode-3 span can report) is not painted."""
    from socr.core.born_digital import BornDigitalDetector

    class _Page:
        def get_text(self, kind, flags=0):
            def span(text, flags):
                return {"text": text, "alpha": 255, "char_flags": flags}

            clear = {"text": "transparent", "alpha": 0, "char_flags": 16}
            line = {
                "spans": [span("shown", 16), span("stroked", 32), span("hiddenframe", 0), clear]
            }
            return {"blocks": [{"lines": [line]}]}

    painted = BornDigitalDetector()._painted_layer_text(_Page())
    assert painted == "shownstroked"
