"""#652 rounds 13-16: the review viewer must render a recovered scan's escaped
native text as the characters that were printed.

``review.html`` does not use markdown-it-py. It embeds its own regex renderer
(``renderMd``/``inline`` in ``socr.review.html._TEMPLATE``), and that renderer
did not implement CommonMark backslash escapes: Astra fed it
``_escaped_native_line('*emphasis*')`` and got ``<p>\\<i>emphasis\\</i></p>``
-- the backslashes leaked into the page and the asterisks activated anyway.
Native recovery now emits that encoding for every line of a scanned page's own
text layer, so a literal asterisk on the page became italics in socr's own
review instrument.

Round 14 then fixed how those escapes are held while the regexes run. Round
13 parked each one at a fixed private-use codepoint and decoded that whole
range off the output, which rewrote a page's OWN private-use glyphs -- this
corpus carries them from math and symbol fonts -- into the punctuation they
happened to encode. The placeholder namespace is now generated per render and
verified absent from the source, and only the tokens a render created are
restored.

Round 15 replaced the codec again. A namespace checked absent from the source
can still be formed across the boundary between the source and an inserted
token: with a source letter abutting the token, the scan opens a token one
character early and swallows the real one. The token is now bracketed by a
delimiter chosen deterministically as the first control character the source
does not contain, which makes every occurrence of that delimiter one this
render wrote, and the pairing unambiguous whatever text abuts the token.

Round 16 removed vertical tab and form feed from the candidate list. Both are
whitespace to JavaScript, and the renderer trims table cells and lets the
heading and list regexes consume the run of whitespace after their marker, so
either one is deleted at a cell edge or straight after a ``#`` and the token it
opened is never closed.

These tests run the ACTUAL JavaScript socr ships, extracted from the template
and executed under Node, which is the only way to test the renderer that
reaches a reader. They skip cleanly where ``node`` is not installed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from html.parser import HTMLParser

import pytest

from socr.core.manifest import _escaped_native_line
from socr.review.html import _TEMPLATE

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")

#: The renderer's own source, verbatim: everything from ``esc`` up to the first
#: function that touches the DOM. Extracted rather than reimplemented -- a
#: second copy of these regexes would pass while the shipped ones failed.
_RENDERER_JS = _TEMPLATE[_TEMPLATE.index("function esc(") : _TEMPLATE.index("function head(")]

_DRIVER = (
    '\nconst fs = require("fs");'
    '\nprocess.stdout.write(renderMd(JSON.parse(fs.readFileSync(0, "utf8"))));'
)


def _render(markdown: str, *, setup: str = "") -> str:
    """*markdown* through the review viewer's own renderMd, under Node.

    *setup* is JavaScript run after the renderer's own source and before it is
    called, which is how a test forces one particular delimiter choice."""
    return subprocess.run(
        ["node", "-e", _RENDERER_JS + "\n" + setup + _DRIVER],
        input=json.dumps(markdown),
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    ).stdout


class _Visible(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def _visible(html: str) -> str:
    """What a reader actually sees, with the markup stripped back off."""
    parser = _Visible()
    parser.feed(html)
    return "".join(parser.parts)


#: One line per construct the escaper protects, as a scan might print them.
_NATIVE_LINES = [
    "<!--",
    "The committee retained the original mandate.",
    "# literal heading marker",
    "> quoted",
    "- bulleted",
    "*emphasis* and `code` and [link](x) and A & B",
    "~~struck~~",
    "| Bank | Member |",
    "***",
    "---",
]


def test_escaped_native_text_renders_as_its_own_characters() -> None:
    """Astra's prose12 reproduction, widened to every protected construct.

    Nothing on such a page was verified and the page authored none of this
    syntax, so none of it may activate -- and the escaping that stops it must
    not be visible either."""
    rendered = _render("\n".join(_escaped_native_line(line) for line in _NATIVE_LINES))

    for tag in ("<i>", "<b>", "<h1>", "<h2>", "<li>", "<ul>", "<code>", "<a ", "<table>", "<pre>"):
        assert tag not in rendered, tag
    assert "<!--" not in rendered
    assert "\\" not in rendered

    visible = _visible(rendered)
    assert "The committee retained the original mandate." in visible
    for line in _NATIVE_LINES:
        assert line in visible, line


def test_the_escaped_left_angle_bracket_is_still_not_raw_html() -> None:
    """The safety boundary is unchanged. Honouring the escape puts back a
    literal ``<``, and that literal goes back THROUGH ``esc`` -- a page whose
    text layer prints a script tag is text, not markup."""
    rendered = _render(_escaped_native_line("<script>alert(1)</script>"))

    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered
    assert "alert(1)" in _visible(rendered)


def test_ordinary_model_prose_still_renders_as_markdown() -> None:
    """The regression guard for everything else in the viewer. A document with
    no escapes must render exactly as it did: the placeholder pass is a no-op
    on it, so headings, tables, lists, emphasis and code spans all still
    work."""
    rendered = _render(
        "# Minutes\n\n"
        "The Committee reviewed **economic** conditions and *employment*.\n\n"
        "| Category | Jan |\n|---|---|\n| Decrease | 17 |\n\n"
        "- first item\n- second item\n\n"
        "Inline `code` here.\n"
    )

    for tag in ("<h1>", "<b>", "<i>", "<table>", "<li>", "<code>", "<mark>"):
        assert tag in rendered, tag


def test_a_code_fence_keeps_its_own_backslashes() -> None:
    """CommonMark does not process escapes inside a fence, and a fence here is
    a model's code sample. ``protect`` skips fenced content for that reason,
    so a backslash printed in a code block survives as itself."""
    rendered = _render('```\nprintf("a\\*b");\n```\n')

    assert "a\\*b" in _visible(rendered)


#: Astra's prose13 reproductions. Round 13 encoded each protected escape as a
#: fixed private-use codepoint (``0xE000 + the character``) and decoded the
#: whole of U+E021-U+E07E off the rendered output. That range is representable
#: input: this corpus carries private-use glyphs from math and symbol fonts,
#: so a page's own U+E031 became the digit ``1`` and U+E02A became ``*`` -- in
#: the instrument used to judge digit fidelity, and inside code fences too,
#: because the decode pass ran over the whole output rather than over the
#: tokens the render had created.
@pytest.mark.parametrize(
    ("source", "glyph"),
    [
        ("Native symbol ", ""),
        ("Native symbol ", ""),
        ("```\nNative symbol \n```", ""),
    ],
    ids=["digit-range", "punctuation-range", "inside-a-fence"],
)
def test_a_private_use_glyph_in_the_source_is_not_decoded(source: str, glyph: str) -> None:
    """The page's own characters survive. No fixed range can be assumed
    unused, so the placeholder namespace is generated per render and checked
    absent from the source; only tokens this render created are restored."""
    visible = _visible(_render(source))

    assert glyph in visible
    assert "1" not in visible
    assert "*" not in visible


def test_a_private_use_glyph_beside_a_real_escape_survives_the_restoration() -> None:
    """Both mechanisms on one line: the escape is honoured and the glyph that
    would once have been mistaken for one is left exactly as printed."""
    visible = _visible(_render("Rate  and " + _escaped_native_line("*starred*")))

    assert "" in visible
    assert "*starred*" in visible
    assert "1" not in visible


def test_a_forged_private_use_sentinel_cannot_inject_html() -> None:
    """Astra's companion probe. Under round 13 these decoded into real angle
    brackets; they now pass through as themselves, and either way no element
    is created."""
    rendered = _render("scriptalert(1)/script")

    assert "<script" not in rendered
    assert "script" in rendered


def test_a_long_run_of_letters_in_the_source_is_not_mistaken_for_a_token() -> None:
    """The placeholder is lowercase letters, so a document made of lowercase
    letters is the adversarial case for the namespace check. It is verified
    absent from this document's own source before any token is emitted."""
    source = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa bbbbbbbbbbbbbbbb " + _escaped_native_line("*x*")

    assert "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa bbbbbbbbbbbbbbbb *x*" in _visible(_render(source))


#: Every delimiter the codec will try, in the order it tries them, read out of
#: the renderer itself rather than restated here.
_DELIMS = [chr(n) for n in list(range(1, 9)) + list(range(14, 32)) + [127]]


def _force(delims: list[str]) -> str:
    """JavaScript that replaces the candidate list with *delims*."""
    return "ESC_DELIMS = " + json.dumps(delims) + ";"


def test_the_renderer_tries_the_delimiters_this_file_names() -> None:
    """The list above is a copy, so it is checked against the shipped one. If
    the renderer's candidates change, the pins below must be re-derived."""
    reported = json.loads(
        subprocess.run(
            ["node", "-e", _RENDERER_JS + "\nprocess.stdout.write(JSON.stringify(ESC_DELIMS));"],
            text=True,
            capture_output=True,
            check=True,
            timeout=30,
        ).stdout
    )

    assert reported == _DELIMS


#: Astra's prose14 counterexample, in the shape the delimiter codec takes it:
#: a source character abutting the token on the left, and a token whose index
#: letters could extend into the text on the right.
_ABUTTING = "a" + _escaped_native_line("*") + "b"


@pytest.mark.parametrize("delim", _DELIMS, ids=[f"U+{ord(d):04X}" for d in _DELIMS])
def test_every_delimiter_choice_renders_the_same_page(delim: str) -> None:
    """The invariant round 14 violated. Which delimiter gets picked is an
    implementation detail of the source, so it must not be observable in the
    output -- and the abutting case is the one that made namespace choice
    observable before."""
    source = _ABUTTING + "\n\n" + "\n".join(_escaped_native_line(line) for line in _NATIVE_LINES)

    assert _render(source, setup=_force([delim])) == _render(source, setup=_force(_DELIMS))


def test_the_source_character_beside_a_token_is_not_eaten() -> None:
    """The counterexample read directly: both neighbours survive and the
    escaped asterisk comes back as itself."""
    assert _visible(_render(_ABUTTING, setup=_force([_DELIMS[0]]))) == "a*b"


def test_adjacent_tokens_each_restore_themselves() -> None:
    """Astra's companion control. Four escapes with nothing between them: the
    delimiters pair off left to right and no match spans two tokens."""
    assert _visible(_render(_escaped_native_line("*&<["))) == "*&<["


def test_a_source_holding_every_delimiter_but_the_last_uses_the_last() -> None:
    """The fallback walk. A page carrying twenty-eight of the twenty-nine
    candidates still gets a clean codec from the one it does not carry."""
    source = "".join(_DELIMS[:-1]) + "\n\n" + _ABUTTING

    assert "a*b" in _visible(_render(source))


def test_digits_beside_an_escape_keep_their_own_marking() -> None:
    """The index is letters because a decimal index would be swallowed by the
    number marker below. A real number next to a token is still marked, and
    the token still restores."""
    rendered = _render("Rate 17.5% " + _escaped_native_line("*starred*") + " on 2019")

    assert "<mark>17.5%</mark>" in rendered
    assert "<mark>2019</mark>" in rendered
    assert "*starred*" in _visible(rendered)


def test_a_source_holding_every_delimiter_falls_back_to_no_protection() -> None:
    """The stated giving-up point. With no delimiter left, escapes are not
    honoured and the page renders as it did before round 13 -- a visible
    defect, not a silent rewrite of the page's own characters. The characters
    that exhausted the list are themselves untouched."""
    source = "".join(_DELIMS) + _escaped_native_line("*x*")
    rendered = _render(source)

    assert "<i>" in rendered
    assert "\\" in rendered
    for delim in _DELIMS:
        assert delim in rendered, hex(ord(delim))


def test_no_candidate_delimiter_is_whitespace_to_the_engine() -> None:
    """The premise the proof rests on, checked by the machine.

    Round 15's list held U+000B and U+000C. JavaScript calls both whitespace,
    so ``c.trim()`` on a table cell and the ``\\s+`` in the heading and list
    regexes deleted them, and a token that lost a delimiter could never be
    closed. The argument above ``protect`` now says the delimiter is not
    whitespace, and this asks the real engine rather than assuming it."""
    verdict = json.loads(
        subprocess.run(
            [
                "node",
                "-e",
                _RENDERER_JS
                + "\nprocess.stdout.write(JSON.stringify(ESC_DELIMS.map("
                + "d => [/\\s/.test(d), d.trim() === ''])));",
            ],
            text=True,
            capture_output=True,
            check=True,
            timeout=30,
        ).stdout
    )

    assert verdict == [[False, False]] * len(_DELIMS)


#: Every place the renderer has structure around the text, as Astra listed
#: them: the two edges of a table cell, straight after a heading marker, after
#: a list bullet and after a blockquote marker. Each is a context where a
#: whitespace delimiter used to be trimmed or eaten.
_STRUCTURAL_SOURCES = {
    "table-cell-edges": "| Heading | Other |\n| --- | --- |\n| "
    + _escaped_native_line("*literal*")
    + " | text |",
    "heading": "# " + _escaped_native_line("*literal*"),
    "list-item": "- " + _escaped_native_line("*literal*"),
    "blockquote": "> " + _escaped_native_line("*literal*"),
}


@pytest.mark.parametrize("delim", _DELIMS, ids=[f"U+{ord(d):04X}" for d in _DELIMS])
@pytest.mark.parametrize("shape", sorted(_STRUCTURAL_SOURCES), ids=sorted(_STRUCTURAL_SOURCES))
def test_a_literal_survives_every_structure_under_every_delimiter(shape: str, delim: str) -> None:
    """Round 15 pinned equivalence only on paragraph-shaped text, which is why
    the trimming contexts went unmeasured. Every candidate must carry a literal
    through every structure the renderer builds."""
    rendered = _render(_STRUCTURAL_SOURCES[shape], setup=_force([delim]))

    assert "*literal*" in _visible(rendered)
    assert "<i>" not in rendered


def test_the_delimiter_the_source_forces_still_carries_a_heading() -> None:
    """Astra's natural-selection case, which needs no forcing at all. A fenced
    code sample holding U+0001 to U+0008 walks the candidate list past all
    eight; round 15 landed on the vertical tab and lost the next heading."""
    source = (
        "```\n"
        + "".join(chr(n) for n in range(1, 9))
        + "\n```\n# "
        + _escaped_native_line("*literal*")
    )

    assert "*literal*" in _visible(_render(source))


def test_a_whitespace_delimiter_is_refused_rather_than_used_badly() -> None:
    """The premise is enforced where the choice is made. Handed nothing but a
    whitespace candidate, ``protect`` takes none and the page falls back to no
    escape protection -- visibly wrong rather than a token the renderer cuts in
    half and shows as stray index letters."""
    rendered = _render("# " + _escaped_native_line("*literal*"), setup=_force(["\v", "\f"]))

    assert "\\" in rendered
    assert "a" not in _visible(rendered).replace("literal", "")
