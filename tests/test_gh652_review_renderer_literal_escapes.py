"""#652 rounds 13-14: the review viewer must render a recovered scan's escaped
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


def _render(markdown: str) -> str:
    """*markdown* through the review viewer's own renderMd, under Node."""
    return subprocess.run(
        ["node", "-e", _RENDERER_JS + _DRIVER],
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
