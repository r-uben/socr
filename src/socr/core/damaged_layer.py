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
  same words. The wrapper is removed only when the printed copy, with the SAME note number
  and the FULL text, is on the same page (:func:`strip_hidden_footnotes`); a lone, continued
  or differing note keeps its wrapper because it may be the only copy of those words.
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


def _printed_copy_re(note: int, body: str) -> re.Pattern:
    """The printed form of a note: its number (``1``, ``1.`` or ``1)``), then the SAME words,
    whitespace between words free to differ (a printed line wraps). Bounded by whitespace or
    the text's ends on both sides, so a note is never found inside a longer word or number."""
    words = r"\s+".join(re.escape(w) for w in body.split())
    return re.compile(rf"(?<![^\s])0*{note}[.)]?[ \t]+{words}(?![^\s])")


def _printed_number_re(note: int) -> re.Pattern:
    return re.compile(rf"(?m)^[ \t]*0*{note}[.)]?[ \t]+\S")


def strip_hidden_footnotes(text: str) -> FootnoteResult:
    """*text* without the hidden wrappers whose printed copy is on the page.

    A wrapper is stripped iff all hold: it is on one line; its opening and closing numbers
    agree; and the page carries, OUTSIDE every wrapper, a printed copy with that number
    and exactly its text (whitespace-normalised, nothing else). One record per wrapper
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
        if copies:
            stripped.append(StrippedFootnote(n_open, m.group(), body, copies))
            cut.append((m.start(), m.end()))
        elif _printed_number_re(n_open).search(outside):
            kept.append(KeptFootnote(n_open, KEPT_PRINTED_NOTE_DIFFERS))
        else:
            kept.append(KeptFootnote(n_open, KEPT_NO_PRINTED_COPY))
    if not cut:
        return FootnoteResult(text, kept=tuple(kept))
    res = ""
    pos = 0
    for start, end in cut:
        res += text[pos:start]
        # The layer's separator after the wrapper goes with it; the space before stays,
        # unless the wrapper ended its line (then nothing follows to need it).
        while end < len(text) and text[end] in " \t":
            end += 1
        if end >= len(text) or text[end] == "\n":
            res = res.rstrip(" \t")
        pos = end
    res += text[pos:]
    words = " ".join(s.wrapper for s in stripped)
    return FootnoteResult(res, tuple(stripped), tuple(kept), words)
