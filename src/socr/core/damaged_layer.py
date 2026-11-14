"""#1306: two defects of a born-digital text layer that the native lane used to ship as is.

Both are repairs of shipped text, never a drop of content:

* **Control bytes.** A C0 control character other than tab, line feed and carriage return
  stands for a glyph the layer could not map (a minus, a percent sign, a symbol; the same
  code means different glyphs in different documents, ``docs/log/2026-10-03_990``). Deleting
  it would turn ``\\x010.47`` (-0.47) into a clean ``0.47``, the worst outcome. It becomes
  U+FFFD, which is visible, and the page is marked (:func:`replace_control_bytes`).
* **Hidden footnote wrapper.** The Fed's accessible PDFs carry an invisible tagged copy of
  each footnote in the running text: ``[Footnote 3. text End footnote 3.]`` (or ``End of
  footnote``, or the same without the brackets). The printed footnote is a second copy of the
  same words. The wrapper is removed only when BOTH hold (:func:`strip_hidden_footnotes`):
  the PDF's own painted glyphs do not draw the wrapper (its ``Footnote N.`` / ``End footnote
  N.`` frame is replacement text, not ink; evidence from page analysis, never from the text
  alone), and the printed note is a complete note line on the page (starts a line with the
  same note number, same full text, ends its line). A lone, continued or differing note, or
  a wrapper the layer visibly paints, keeps its wrapper because it may be content.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

#: What a control byte becomes. A visible character the page's reader cannot mistake for a
#: sign or a digit.
REPLACEMENT_CHAR = "�"

#: Every C0 control character except tab (0x09), line feed (0x0a) and carriage return
#: (0x0d), which are whitespace the layer legitimately carries. Vertical tab and form feed
#: (0x0b, 0x0c) are control bytes: ``glyph_recovery._CONTROL_BYTE_BEFORE_DIGIT`` (#990)
#: counts the same set.
CONTROL_BYTE_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def replace_control_bytes(text: str) -> tuple[str, Counter]:
    """*text* with every control byte replaced by U+FFFD, and the code points found.

    The counter's keys are ``U+0003``-style names. The replacement is one character per
    byte, so the shipped text keeps the byte's position and a reader sees where it stood.
    """
    found: Counter = Counter()

    def _one(match: re.Match) -> str:
        found[f"U+{ord(match.group()):04X}"] += 1
        return REPLACEMENT_CHAR

    return CONTROL_BYTE_RE.sub(_one, text or ""), found


#: A footnote wrapper on ONE line: ``[Footnote N. <text> End footnote N.]``, ``End of
#: footnote`` as the closing words, brackets either both present or both absent. A wrapper
#: broken across lines is not recognised and so never stripped (kept, conservative). The
#: opening and closing numbers must be equal (checked by the caller, which keeps a wrapper
#: whose numbers differ).
_WRAPPER_RE = re.compile(
    r"(?P<open>\[)?Footnote (?P<n>\d+)\. (?P<body>[^\n]+?) End (?:of )?footnote (?P<m>\d+)\.(?(open)\])",
    re.IGNORECASE,
)

#: Why a wrapper was kept. Named so a record and a test cannot drift.
KEPT_NUMBER_MISMATCH = "number_mismatch"
KEPT_NO_PRINTED_COPY = "printed_copy_absent"
KEPT_PRINTED_NOTE_DIFFERS = "printed_note_differs"
#: A printed copy is on the page but nothing shows the wrapper is hidden text: the painted
#: glyphs draw its frame, or the page's painted glyphs were not read. Kept (fail closed).
KEPT_NOT_EVIDENCED_HIDDEN = "hidden_not_evidenced"


@dataclass(frozen=True)
class StrippedFootnote:
    """One hidden wrapper removed: the note, the words it held and how many printed copies
    of the same note the page carries."""

    note: int
    #: the wrapper's whole text as it stood in the layer (brackets and tags included)
    wrapper: str
    #: the note's own text, whitespace-normalised
    text: str
    printed_copies: int

    def to_dict(self) -> dict:
        return {
            "note": self.note,
            "chars": len(self.wrapper),
            "printed_copies": self.printed_copies,
            "text": self.text,
        }


@dataclass(frozen=True)
class KeptFootnote:
    """One hidden wrapper left in place, and why."""

    note: int
    reason: str


@dataclass(frozen=True)
class FootnoteResult:
    text: str
    stripped: tuple[StrippedFootnote, ...] = ()
    kept: tuple[KeptFootnote, ...] = ()
    #: the words the stripped wrappers held, for the words check's reference
    stripped_words: str = field(default="")


def _normalised(text: str) -> str:
    return " ".join(text.split())


def painted_form(painted: str) -> str:
    """The painted glyphs of a page reduced to what a wrapper's frame is looked for in:
    lower case, no whitespace (a line or a span may break inside the frame)."""
    return "".join((painted or "").split()).lower()


def _frame_is_painted(painted: str, note: int) -> bool:
    """True iff the page's painted glyphs draw the wrapper's ``Footnote N.`` opener or its
    ``End [of] footnote N.`` closer. Hidden accessibility text is replacement text (PDF
    ``/ActualText``) or invisible text (render mode 3); neither is painted ink, so neither
    frame is in the painted glyphs. A visibly printed example has both."""
    n = str(note)
    return (
        f"footnote{n}." in painted
        or f"endfootnote{n}." in painted
        or f"endoffootnote{n}." in painted
    )


def _printed_copy_re(note: int, body: str) -> re.Pattern:
    """The printed form of a note, a COMPLETE note line: it starts a line with its number
    (``1``, ``1.`` or ``1)``), carries exactly the same words (whitespace between words free to
    differ: a printed line wraps) and ends where its line ends. A note that only begins with
    the wrapper's words, or sits inside a longer line, does not match."""
    words = r"\s+".join(re.escape(w) for w in body.split())
    return re.compile(rf"(?m)^[ \t]*0*{note}[.)]?[ \t]+{words}[ \t]*$")


def _printed_number_re(note: int) -> re.Pattern:
    return re.compile(rf"(?m)^[ \t]*0*{note}[.)]?[ \t]+\S")


def strip_hidden_footnotes(text: str, painted: str | None = None) -> FootnoteResult:
    """*text* without the hidden wrappers that are shown hidden and whose printed copy is there.

    *painted* is the page's painted glyphs from the PDF (see :func:`painted_form`), ``None``
    when they were not read. A wrapper is stripped iff all hold: it is on one line; its
    opening and closing numbers agree; the page carries, OUTSIDE every wrapper, a complete
    printed note line with that number and exactly its text; and *painted* is known and does
    not draw the wrapper's frame. Removal keeps the token boundary: the neighbours of an
    inline wrapper are joined by one space, never concatenated. One record per wrapper
    stripped. Everything else is kept and reported with its reason.
    """
    text = text or ""
    matches = list(_WRAPPER_RE.finditer(text))
    if not matches:
        return FootnoteResult(text)
    # The printed copies are looked for in the text with the wrappers blanked out: a wrapper
    # is a copy of its own words and must never vouch for itself.
    outside = text
    for m in reversed(matches):
        outside = outside[: m.start()] + "\n" + outside[m.end() :]
    stripped: list[StrippedFootnote] = []
    kept: list[KeptFootnote] = []
    cut: list[tuple[int, int]] = []
    for m in matches:
        n_open, n_close = int(m.group("n")), int(m.group("m"))
        if n_open != n_close:
            kept.append(KeptFootnote(n_open, KEPT_NUMBER_MISMATCH))
            continue
        body = _normalised(m.group("body"))
        copies = len(_printed_copy_re(n_open, body).findall(outside))
        if not copies:
            if _printed_number_re(n_open).search(outside):
                kept.append(KeptFootnote(n_open, KEPT_PRINTED_NOTE_DIFFERS))
            else:
                kept.append(KeptFootnote(n_open, KEPT_NO_PRINTED_COPY))
        elif painted is None or _frame_is_painted(painted, n_open):
            kept.append(KeptFootnote(n_open, KEPT_NOT_EVIDENCED_HIDDEN))
        else:
            stripped.append(StrippedFootnote(n_open, m.group(), body, copies))
            cut.append((m.start(), m.end()))
    if not cut:
        return FootnoteResult(text, kept=tuple(kept))
    res = ""
    pos = 0
    for start, end in cut:
        res += text[pos:start]
        res = res.rstrip(" \t")
        while end < len(text) and text[end] in " \t":
            end += 1
        # Both neighbours on this line: one space keeps them two tokens (``Growth[..] slowed``,
        # ``1[..] 2``). A wrapper at either edge of its line leaves no separator to keep.
        if res and not res.endswith("\n") and end < len(text) and text[end] != "\n":
            res += " "
        pos = end
    res += text[pos:]
    words = " ".join(s.wrapper for s in stripped)
    return FootnoteResult(res, tuple(stripped), tuple(kept), words)
