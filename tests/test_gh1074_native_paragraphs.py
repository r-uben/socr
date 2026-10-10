"""#1074: native prose ships as paragraphs, with line-end split words rejoined on a witness.

``get_text("text")`` returns one printed line per output line. The reflow runs at the one
finalize seam (``manifest._apply_native_paragraphs``), from paragraphs read off the page
geometry at analyze time and matched to the shipped lines by exact content.

The end-to-end guard is a DIFFERENCE between two runs in one process that change only
whether the seam acts (CLAUDE.md, #257: a value measured on one machine is not a CI
contract). Hermetic: ``_available_engines_for_agentic`` patched, ``_resolve_judge_model``
-> "", both provider states.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import fitz
import pytest

from socr.core import manifest
from socr.core.cache import BlobStore
from socr.core.config import EngineType, PipelineConfig
from socr.core.manifest import Manifest
from socr.core.manifest import replay as manifest_replay
from socr.core.native_paragraphs import (
    NativeVocabulary,
    native_vocabulary,
    page_paragraphs,
    reflow_native_prose,
)
from socr.core.providers import PROFILE_QWEN_LOCAL
from socr.pipeline import orchestrator as orch
from socr.pipeline.orchestrator import UnifiedPipeline


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        UnifiedPipeline, "_available_engines_for_agentic", lambda self: [PROFILE_QWEN_LOCAL]
    )
    monkeypatch.setattr(UnifiedPipeline, "_resolve_judge_model", lambda self, *a, **kw: "")


def test_loaded_source_is_this_checkout() -> None:
    import socr

    assert Path(socr.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1] / "src")


# ---------------------------------------------------------------------------
# The synthetic page. Courier 10pt: every glyph advances CH points, so a line of N
# characters is exactly N * CH wide and "full width" is an exact right edge.
# ---------------------------------------------------------------------------

CH = 6.0
LEFT = 72.0
WIDTH = 78  # characters in a full line
INDENT = 18.0  # three glyphs
POOL = ["market", "rates", "prices", "output", "policy", "survey", "wages", "credit", "growth"]


def line_of(width: int, head: str = "", tail: str = "") -> str:
    """A line of exactly ``width`` characters starting with ``head`` and ending with ``tail``."""
    for offset in range(len(POOL)):
        toks = [head] if head else []
        i = offset
        while True:
            cur = " ".join(toks + ([tail] if tail else []))
            rem = width - len(cur)
            if rem == 0:
                return cur
            if rem >= 14:
                toks.append(POOL[i % len(POOL)])
                i += 1
                continue
            if rem >= 2:
                toks.append("z" * (rem - 1))
                if len(" ".join(toks + ([tail] if tail else []))) == width:
                    return " ".join(toks + ([tail] if tail else []))
            break
    raise AssertionError("cannot build a line of that width")


# Split words in the page: the first rejoins on a witness ("inflation" appears whole),
# the second keeps its hyphen on a witness ("high-frequency" appears hyphenated), the
# third has no witness and stays as printed.
A_P1 = [
    line_of(WIDTH - 3, head="Alpha inflation"),
    line_of(WIDTH, tail="infla-"),
    line_of(WIDTH, head="tion"),
    line_of(31, tail="short"),
]
A_P2_LEFT = line_of(30, head="Bravo high-frequency")
A_P2_RIGHT = line_of(24, tail="ments")
# (b) is set at 13pt so its pitch (22, above 1.5x the size) is its own size class: one body
# pitch per size class is the realistic case, and mixing two pitches at one size would
# leave the modal pitch to whichever shape has more lines.
SIZE_B = 13.0
CH_B = 0.6 * SIZE_B
WIDTH_B = 60
INDENT_B = 3 * CH_B
B_P1 = [
    line_of(WIDTH_B - 3, head="Charlie"),
    line_of(WIDTH_B, tail="high-"),
    line_of(30, head="frequency"),
]
B_P2 = [
    line_of(WIDTH_B - 3, head="Delta"),
    line_of(WIDTH_B, tail="zork-"),
    line_of(25, head="wump"),
]
C_ENV = [line_of(40) for _ in range(3)]


def build_pdf(path: Path) -> Path:
    doc = fitz.open()
    page = doc.new_page()
    font = fitz.Font("cour")

    def draw(x: float, y: float, text: str, size: float = 10.0) -> None:
        page.insert_text((x, y), text, fontname="cour", fontsize=size)

    # (a) two indent-only paragraphs in ONE block (a single text object), pitch 12.
    # (d) the last line of the second is written as two same-baseline fragments.
    y = 72.0
    tw = fitz.TextWriter(page.rect)
    rows = [(LEFT + INDENT, A_P1[0]), (LEFT, A_P1[1]), (LEFT, A_P1[2]), (LEFT, A_P1[3])]
    rows += [(LEFT + INDENT, line_of(WIDTH - 3, head="Bravo")), (LEFT, line_of(WIDTH))]
    for x, text in rows:
        tw.append((x, y), text, font=font, fontsize=10)
        y += 12
    tw.append((LEFT, y), A_P2_LEFT, font=font, fontsize=10)
    tw.write_text(page)
    draw(LEFT + 36 * CH, y, A_P2_RIGHT)
    y += 12 + 36
    # (b) the same shape at pitch 22 (> 1.5x the size): one block per line.
    for para in (B_P1, B_P2):
        for k, text in enumerate(para):
            draw(LEFT + INDENT_B if k == 0 else LEFT, y, text, SIZE_B)
            y += 22
    y += 30
    # (c) a three-line environment, narrower than the column, full inside itself.
    tw2 = fitz.TextWriter(page.rect)
    for text in C_ENV:
        tw2.append((LEFT + 30, y), text, font=font, fontsize=10)
        y += 12
    tw2.write_text(page)
    doc.save(path)
    doc.close()
    return path


@pytest.fixture(scope="module")
def synth_pdf(tmp_path_factory) -> Path:
    return build_pdf(tmp_path_factory.mktemp("synth") / "synth.pdf")


def _shape(paragraphs) -> list[list[int]]:
    """Per paragraph, the MuPDF-line count of each printed line."""
    return [[len(printed) for printed in para] for para in paragraphs]


def test_geometry_reads_five_paragraphs_with_the_fragment_line_inside_its_own(synth_pdf) -> None:
    with fitz.open(synth_pdf) as doc:
        paragraphs = page_paragraphs(doc[0])
    # (a) 4 + 3 printed lines (the last is two fragments), (b) 3 + 3, (c) 3.
    assert _shape(paragraphs) == [[1] * 4, [1, 1, 2], [1] * 3, [1] * 3, [1] * 3]
    assert [frag for frag in paragraphs[1][-1]] == [A_P2_LEFT, A_P2_RIGHT]


def test_each_shape_is_pinned_by_its_own_assertion(synth_pdf) -> None:
    """The design shapes: (a) indent split, (b) pitch merge, (c) R_env, (d) fragments."""
    with fitz.open(synth_pdf) as doc:
        paragraphs = page_paragraphs(doc[0])
    flat = [[f for printed in para for f in printed] for para in paragraphs]
    assert flat[0] == A_P1  # (a) paragraph 1 ends where the indent begins
    assert flat[1][0].startswith("Bravo") and flat[1][-2:] == [A_P2_LEFT, A_P2_RIGHT]  # (a), (d)
    assert flat[2] == B_P1 and flat[3] == B_P2  # (b) one block per line, still two paragraphs
    assert flat[4] == C_ENV  # (c) one paragraph, not one per line


def _font_page(path: Path) -> Path:
    """One block: a paragraph whose second line is set in italics, a bold heading, a paragraph."""
    doc = fitz.open()
    page = doc.new_page()
    roman, italic, bold = fitz.Font("cour"), fitz.Font("coit"), fitz.Font("cobo")
    tw = fitz.TextWriter(page.rect)
    rows = [
        (roman, line_of(WIDTH, head="Sources including")),
        (italic, line_of(WIDTH, head="Citizen")),  # italic runs across the line break
        (roman, line_of(31, tail="short")),
        (bold, "Bold heading"),
        (roman, line_of(WIDTH, head="After")),
        (roman, line_of(WIDTH)),
        (roman, line_of(31, tail="end")),
    ]
    y = 72.0
    for font, text in rows:
        tw.append((LEFT, y), text, font=font, fontsize=10)
        y += 12
    tw.write_text(page)
    doc.save(path)
    doc.close()
    return path


def test_a_font_change_alone_is_not_a_boundary_but_a_short_line_before_one_is(tmp_path) -> None:
    pdf = _font_page(tmp_path / "fonts.pdf")
    with fitz.open(pdf) as doc:
        paragraphs = page_paragraphs(doc[0])
    # Italic emphasis on a full-width line: ONE paragraph. The bold heading that follows a
    # short line stays its own paragraph, and the body after it another.
    assert _shape(paragraphs) == [[1] * 3, [1], [1] * 3]


# ---------------------------------------------------------------------------
# Pure reflow: hyphen rule, protected content, adjacency, idempotence
# ---------------------------------------------------------------------------


def _vocab(*texts: str) -> NativeVocabulary:
    return native_vocabulary(list(texts))


def _para(*printed: str):
    return tuple((p,) for p in printed)


def test_hyphen_rejoins_only_on_a_witness_for_the_joined_word() -> None:
    vocab = _vocab("The inflation rate and the market.")
    text = "the rate of infla-\ntion is high"
    out = reflow_native_prose(text, (_para("the rate of infla-", "tion is high"),), vocab)
    assert out == "the rate of inflation is high"


def test_hyphen_is_kept_when_the_hyphenated_form_is_witnessed() -> None:
    vocab = _vocab("A high-frequency series. The highfrequency typo.")
    out = reflow_native_prose(
        "of high-\nfrequency data", (_para("of high-", "frequency data"),), vocab
    )
    assert out == "of high-frequency data"  # witnessed both ways: tie keeps the hyphen


def test_hyphen_kept_when_only_the_hyphenated_form_is_witnessed() -> None:
    vocab = _vocab("zero-coupon bonds")
    out = reflow_native_prose("zero-\ncoupon", (_para("zero-", "coupon"),), vocab)
    assert out == "zero-coupon"


def test_unwitnessed_split_is_kept_as_printed_hyphen_included() -> None:
    vocab = _vocab("nothing relevant here")
    out = reflow_native_prose("a de-\nspite b", (_para("a de-", "spite b"),), vocab)
    assert out == "a de-spite b"


def test_the_split_occurrence_is_not_its_own_witness() -> None:
    # "infla-\ntion" twice and nowhere else: the joined form has no witness.
    text = "infla-\ntion and infla-\ntion"
    assert "inflation" not in _vocab(text).words


def test_dash_glued_to_its_word_joins_without_a_space_and_removes_nothing() -> None:
    vocab = _vocab("x")
    for a, b, want in [
        ("Long-", "Run effects", "Long-Run effects"),
        ("in 1994–", "2000", "in 1994–2000"),
        ("the article—", "is clear", "the article—is clear"),
    ]:
        assert reflow_native_prose(f"{a}\n{b}", (_para(a, b),), vocab) == want


def test_fragments_join_with_a_space_and_never_apply_the_hyphen_rule() -> None:
    vocab = _vocab("inflation")
    para = (("left part-", "tion right"),)  # ONE printed line, two fragments
    assert reflow_native_prose("left part-\ntion right", (para,), vocab) == "left part- tion right"


def test_protected_content_is_untouched() -> None:
    vocab = _vocab("inflation")
    fence = "```\nfoo bar-\nbaz qux\n```"
    table = "| a | b |\n| - | - |\n| foo | bar |"
    latex = "$$\nx = y\n$$"
    comment = "<!-- note\nfoo bar -->"
    figure = "![Figure 1](figures/f1.png)"
    for block in (fence, table, latex, comment, figure):
        text = f"{block}\nafter this"
        paragraph = _para(*block.split("\n"), "after this")
        assert reflow_native_prose(text, (paragraph,), vocab) == text, block


def test_math_and_comment_interiors_are_never_joined() -> None:
    """The lines BETWEEN delimiters are protected too: a joined `a-`/`b` would delete a minus."""
    vocab = _vocab("ab")
    for text in (
        "$$\na-\nb\n$$",
        "See <!-- note\na-\nb\n--> end",
        "\\[\na-\nb\n\\]",
        "\\begin{align}\na-\nb\n\\end{align}",
    ):
        assert reflow_native_prose(text, (_para("a-", "b"),), vocab) == text, text
    # A range that opens and closes on one line protects only that line.
    text = "p <!-- c --> q\na-\nb"
    assert reflow_native_prose(text, (_para("a-", "b"),), vocab) == "p <!-- c --> q\nab"


@pytest.mark.parametrize(
    "text",
    [
        "See $x+\na-\nb\n+c$",  # inline math spanning lines
        "\\[\n\\[\n\\]\na-\nb\n\\]",  # nested: the first close must not release
        "\\begin{equation}\na-\nb\n\\end{other}",  # mismatched environment
        "\\begin{equation}\n\\end{other}\na-\nb\n\\end{equation}",  # ...closed late
        "```\ncode\n```\nignored\na-\nb\nlast $y$",  # a later marker re-protects the middle
    ],
    ids=["inline-multiline", "nested", "mismatched", "closed-late", "fence-then-math"],
)
def test_every_line_between_the_first_and_last_marker_is_protected(text: str) -> None:
    assert reflow_native_prose(text, (_para("a-", "b"),), _vocab("ab")) == text


@pytest.mark.parametrize(
    "text",
    [
        "\\begin{equation}\n\\end{other}\na-\nb",  # same count, different names
        "$$ x\na-\nb",  # odd $$
        "\\[ x\na-\nb",  # \\[ without \\]
        "\\( x\na-\nb",  # \\( without \\)
        "<!-- x\na-\nb",  # comment never closed
        "```\ncode\na-\nb",  # odd fence lines
        "costs $5 here\na-\nb",  # an odd single $ (currency) over-protects, on purpose
    ],
    ids=["env-names", "dollars", "bracket", "paren", "comment", "fence", "currency"],
)
def test_unbalanced_markers_protect_to_the_end_of_the_text(text: str) -> None:
    assert reflow_native_prose(text, (_para("a-", "b"),), _vocab("ab")) == text


def test_balanced_markers_leave_the_tail_to_reflow() -> None:
    text = "$x$ and $y$\na-\nb"
    assert reflow_native_prose(text, (_para("a-", "b"),), _vocab("ab")) == "$x$ and $y$\nab"


def test_prose_above_and_below_a_math_block_still_reflows() -> None:
    text = "top c-\nd\n\n$$\na-\nb\n$$\n\nbottom e-\nf"
    out = reflow_native_prose(
        text,
        (_para("top c-", "d"), _para("a-", "b"), _para("bottom e-", "f")),
        _vocab("cd ef ab"),
    )
    assert out == "top cd\n\n$$\na-\nb\n$$\n\nbottom ef"


def test_a_link_across_a_line_break_is_left_intact() -> None:
    # The shipped lines differ from the geometry lines (the link was spliced in), so the
    # paragraph is not found and nothing is joined or split.
    text = "see [the long\nlink text](https://example.org/a) here"
    out = reflow_native_prose(text, (_para("see the long", "link text here"),), _vocab("x"))
    assert out == text


def test_blank_line_only_between_adjacent_native_paragraphs() -> None:
    vocab = _vocab("x")
    text = "p1 a\np1 b\np2 a\np2 b\nforeign\np3 a"
    paragraphs = (_para("p1 a", "p1 b"), _para("p2 a", "p2 b"), _para("p3 a"))
    out = reflow_native_prose(text, paragraphs, vocab)
    # p1|p2 adjacent -> blank; p2|foreign|p3 keeps the foreign line's adjacency on both sides.
    assert out == "p1 a p1 b\n\np2 a p2 b\nforeign\np3 a"


def test_a_joined_line_that_collides_with_another_geometry_line_still_reaches_a_fixed_point() -> (
    None
):
    """Pass 1 would give `alpha beta\\ngamma`, which pass 2 would join again."""
    vocab = _vocab("x")
    paragraphs = (_para("alpha", "beta"), _para("alpha beta", "gamma"))
    text = "alpha\nbeta\ngamma"
    once = reflow_native_prose(text, paragraphs, vocab)
    assert reflow_native_prose(once, paragraphs, vocab) == once
    # Resume (geometry) and replay (none) agree on what the page ships.
    assert reflow_native_prose(once, (), vocab) == once


def test_reflow_is_idempotent_and_never_changes_without_geometry() -> None:
    vocab = _vocab("inflation")
    paragraphs = (_para("a infla-", "tion b"), _para("c"))
    once = reflow_native_prose("a infla-\ntion b\nc", paragraphs, vocab)
    assert once == "a inflation b\n\nc"
    assert reflow_native_prose(once, paragraphs, vocab) == once
    assert reflow_native_prose("a infla-\ntion b\nc", (), vocab) == "a infla-\ntion b\nc"
    assert reflow_native_prose("a infla-\ntion b\nc", paragraphs, None) == "a infla-\ntion b\nc"


# ---------------------------------------------------------------------------
# End to end through process(): the same page with the seam acting and not acting
# ---------------------------------------------------------------------------


class _NoEngine:
    name = "qwen"

    def is_available(self) -> bool:
        return True

    def process_pages(self, *a, **kw):  # pragma: no cover - the native lane never calls it
        raise AssertionError("native page must not reach an engine")


class _Judge:
    def assess(self, output, provider):
        from socr.pipeline.agentic import AcceptDecision

        return AcceptDecision(accept=True, reason="")


def _pipe(providers: bool) -> UnifiedPipeline:
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
        )
    )
    pipe._available_engines_for_agentic = lambda: [PROFILE_QWEN_LOCAL] if providers else []
    pipe._build_page_judge = lambda state: _Judge()
    pipe._resolve_crop_vlm_model = lambda: None
    pipe._resolve_judge_model = lambda *a, **k: ""
    return pipe


def _run(monkeypatch, pdf: Path, out: Path, *, providers: bool, reflow: bool, resumed=None) -> dict:
    with monkeypatch.context() as m:
        if resumed is not None:
            real = UnifiedPipeline._load_terminal_page

            def spy(self, *a, **kw):
                got = real(self, *a, **kw)
                if got is not None:
                    resumed.append(got.page_num)
                return got

            m.setattr(UnifiedPipeline, "_load_terminal_page", spy)
        m.setattr(orch, "get_engine", lambda engine_type: _NoEngine())
        if not reflow:
            m.setattr(manifest, "_apply_native_paragraphs", lambda output, p, vocab: output)
        _pipe(providers).process(pdf, output_dir=out)
    return {
        "page": next(iter(out.rglob("pages/00001.md"))).read_text(),
        "side": json.loads(next(iter(out.rglob("pages/00001.json"))).read_text()),
        "final": next(iter(out.rglob("synth.md"))).read_text(),
        "manifest": next(iter(out.rglob("manifest.json"))),
    }


def _paragraphs(text: str) -> list[str]:
    return [p for p in text.split("\n\n") if p.strip()]


def _nonspace(text: str) -> Counter:
    return Counter(text.replace("\n", "").replace(" ", ""))


@pytest.mark.parametrize("providers", [True, False], ids=["provider", "no-provider"])
def test_e2e_reflow_on_vs_off_is_a_paragraph_difference_and_a_listed_hyphen(
    tmp_path, monkeypatch, synth_pdf, providers
) -> None:
    on = _run(monkeypatch, synth_pdf, tmp_path / "on", providers=providers, reflow=True)
    off = _run(monkeypatch, synth_pdf, tmp_path / "off", providers=providers, reflow=False)

    # Off: the printed lines as before, no paragraph break anywhere on the page.
    assert "\n\n" not in off["page"].strip()
    # On: 2 + 2 + 1 paragraphs.
    body = on["page"].strip()
    paras = _paragraphs(body)
    assert len(paras) == 5, body
    assert all("\n" not in p for p in paras), "a paragraph is one line of text"

    # (d): both fragments of the printed line sit in the paragraph that owns them.
    assert paras[1].endswith(f"{A_P2_LEFT} {A_P2_RIGHT}")
    # Hyphens: witnessed rejoin, hyphenated witness keeps, no witness keeps as printed.
    assert "inflation" in paras[0] and "infla-" not in body
    assert "high-frequency" in paras[2]
    assert "zork-wump" in paras[3]
    # (c): the environment is one paragraph.
    assert paras[4] == " ".join(C_ENV)

    # Content: the non-space multiset differs by exactly the one removed hyphen.
    assert _nonspace(off["page"]) - _nonspace(on["page"]) == Counter({"-": 1})
    assert not (_nonspace(on["page"]) - _nonspace(off["page"]))

    # Fragments, final .md and the page sidecar agree on the bytes.
    expected = manifest.assemble_pages([body])
    assert on["final"] == expected, "final .md carries the page body exactly once"
    assert on["side"]["winning_output"]["text"] == on["page"], "sidecar body, byte for byte"

    # Replay needs no geometry and returns the same page text.
    mpath = on["manifest"]
    replayed = manifest_replay(Manifest.load(mpath), BlobStore(mpath.parent / "cache"))
    assert replayed == expected, "replay returns the same body, once"

    # Re-finalizing the shipped text changes nothing.
    with fitz.open(synth_pdf) as doc:
        geometry = page_paragraphs(doc[0])
    vocab = native_vocabulary([off["page"]])
    assert reflow_native_prose(on["page"], geometry, vocab) == on["page"]
    assert reflow_native_prose(off["page"], geometry, vocab) == on["page"]


@pytest.mark.parametrize("providers", [True, False], ids=["provider", "no-provider"])
def test_e2e_resume_is_byte_identical(tmp_path, monkeypatch, synth_pdf, providers) -> None:
    out = tmp_path / "r"
    first = _run(monkeypatch, synth_pdf, out, providers=providers, reflow=True)
    for meta in out.rglob("metadata.json"):
        meta.unlink()
    for audit in out.rglob("audit_log.json"):
        audit.unlink()
    resumed: list[int] = []
    second = _run(monkeypatch, synth_pdf, out, providers=providers, reflow=True, resumed=resumed)
    assert resumed == [1], "the terminal ledger actually served page 1"
    assert second["page"] == first["page"]
    assert second["final"] == first["final"]
