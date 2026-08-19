"""Recover text from embedded symbol fonts that ship no ToUnicode map (#217).

## The failure

Some publisher PDFs (AER-era typesetting is the observed case) embed Type1 symbol
fonts with **no ``/ToUnicode`` CMap and an empty ``/Differences`` array**. A PDF
text extractor has nothing to decode with, so it hands back the raw byte as
Latin-1. The character the reader sees on the page and the character the text
layer yields are then completely different:

===========  ==================  ==================================================
byte shipped  the page prints     consequence
===========  ==================  ==================================================
``2``         ``−`` (minus)       ``-0.12`` extracts as ``20.12`` -- **sign flip**
``1``         ``+``               ``+0.4`` extracts as ``10.4``
``5``         ``=``
``~`` ``!``   ``(`` ``)``         standard-error brackets destroyed
``p``         ``π``               inflation reads as the letter p
``m``         ``μ``
===========  ==================  ==================================================

The sign flip is the dangerous one: a negative coefficient ships as a large
positive number, faithfully, under a SUCCESS status, with nothing flagged. That
is the exact "a wrong number is worse than a missing one" failure this repo
treats as unacceptable.

## The recovery

The map is not lost -- it is in the font program. Each embedded Type1 font carries
its own encoding vector (``dup <code> /<glyphname> put``) mapping character code
to *glyph name*. So:

    code --(embedded encoding vector)--> glyph name --(GLYPH_UNICODE)--> character

``repair_symbol_font_text`` reads that vector out of every embedded font that has
no ToUnicode, builds the CMap the publisher omitted, and writes it into the
in-memory document. Every downstream reader -- ``get_text("text")``,
``"words"``, ``"dict"``, ``"rawdict"``, table extraction, the benchmark scorer --
then returns correct text with no call-site changes.

## Why the table is a table

The glyph names follow Monotype's ``H<number>`` convention, which *looks*
arithmetic and is not. Within a single font ``H9253`` is gamma (number minus
8306) while ``H9004`` is Delta (minus 8088), and the tail of the Greek block
decodes to accented vowels that are not what the glyph draws. A formula would
silently invent characters, so every entry below was established by rendering the
glyph from the page and identifying it visually.

Entries were produced independently by three agents on different vendors and then
reconciled: 24 of 29 were unanimous; the rest are noted at their entry. Anything
not in this table is left untouched and reported, never guessed -- see
``GlyphRepairReport.unmapped_glyphs``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import fitz

logger = logging.getLogger(__name__)

# Verified glyph-name -> character. Keyed by glyph NAME, not by (font, code), so
# the table applies to any document embedding these fonts rather than only the
# one it was derived from.
#
# Provenance: rendered from
# 2000__romer_romer__federal_reserve_information_and_behaviour_of_interest_rates
# and identified visually by three independent agents (Claude / Gemini / Grok).
GLYPH_UNICODE: dict[str, str] = {
    # --- operators and delimiters (the sign-flip class) ---
    "H11001": "+",
    "H11002": "−",  # MINUS SIGN -- the #217 headline defect
    "H11005": "=",
    "H20849": "(",
    "H20850": ")",
    # --- uppercase Greek ---
    "H9004": "Δ",  # Delta -- note the offset differs from the H92xx block
    # --- lowercase Greek ---
    "H9251": "α",  # alpha
    "H9252": "β",  # beta
    "H9253": "γ",  # gamma
    "H9254": "δ",  # delta
    "H9257": "η",  # eta
    "H9258": "θ",  # theta
    "H9260": "κ",  # kappa
    "H9261": "λ",  # lambda
    "H9262": "μ",  # mu
    "H9263": "ν",  # nu
    "H9266": "π",  # pi
    "H9267": "ρ",  # rho
    "H9271": "υ",  # upsilon
    "H9274": "ψ",  # psi
    "H9275": "ω",  # omega
    # Reconciled 2/3, and both readings are the same letter in a different
    # variant form -- phi vs script phi, epsilon vs lunate epsilon. Arithmetic on
    # the glyph number decodes these two to accented vowels, which is not what
    # the glyph draws; the rendered form decided them.
    "H9278": "ϕ",  # phi symbol
    "H9280": "ϵ",  # lunate epsilon
    # --- combining marks ---
    "H6126": "̄",  # COMBINING MACRON (an overbar, e.g. a sample mean)
}

# ``dup <code> /<glyphname> put`` -- the Type1 encoding-vector idiom.
_ENCODING_ENTRY = re.compile(r"dup\s+(\d+)\s*/([A-Za-z0-9._]+)\s+put")

# Glyph names that carry no text and need no mapping, so they are not reported as
# unmapped. ``.notdef`` is the Type1 placeholder for "no glyph".
_NON_TEXT_GLYPHS = frozenset({"space", ".notdef"})

# Monotype's symbol-font naming scheme (``H11002``, ``H9266``): the namespace of
# the broken class, whether or not GLYPH_UNICODE covers the particular glyph.
_MONOTYPE_SYMBOL_NAME = re.compile(r"H\d+")


@dataclass
class GlyphRepairReport:
    """What ``repair_symbol_font_text`` did, so it can be surfaced rather than magic.

    A repair silently changing extracted text would be its own failure mode: the
    numbers would improve and nobody could tell which pages were touched or
    whether anything was left broken.
    """

    repaired_fonts: list[str] = field(default_factory=list)
    mapped_glyph_count: int = 0
    #: Glyph names found in a repaired font that the table does not cover. These
    #: characters are left EXACTLY as the extractor produced them -- guessing a
    #: substitution would be fabrication -- and are reported so the page can be
    #: treated as suspect.
    unmapped_glyphs: list[str] = field(default_factory=list)
    #: font xref -> the CMap bytes attached to it. Kept so the same plan can be
    #: replayed onto a later, freshly-opened copy of the same file without
    #: re-deriving it: the derivation walks every page's text and is the
    #: expensive half, while replaying is a handful of object writes (#246).
    #: Object xrefs are a property of the file's bytes, so a plan is only valid
    #: for a file whose identity is unchanged -- see ``socr.core.pdf``.
    cmaps: dict[int, bytes] = field(default_factory=dict)

    @property
    def repaired(self) -> bool:
        return bool(self.repaired_fonts)

    @property
    def complete(self) -> bool:
        """True when every glyph in every repaired font was covered by the table."""
        return self.repaired and not self.unmapped_glyphs

    @property
    def needs_attention(self) -> bool:
        """True when some drawn glyph is still the wrong character.

        Deliberately independent of :attr:`repaired`. If *nothing* drawn is in
        the table, no CMap is attached and ``repaired`` is False -- yet that is
        the most corrupted document there is. Gating the alarm on "did we attach
        something" would keep the worst case silent.
        """
        return bool(self.unmapped_glyphs)


def _parse_type1_encoding(font_buffer: bytes) -> dict[int, str]:
    """Character code -> glyph name, read from an embedded Type1 font program."""
    try:
        text = font_buffer.decode("latin-1", "replace")
    except Exception:  # pragma: no cover - decode with 'replace' does not raise
        return {}
    return {int(code): name for code, name in _ENCODING_ENTRY.findall(text)}


def _is_affected_font(encoding: dict[int, str]) -> bool:
    """Whether this font is the broken class at all, from its glyph names alone.

    A font with no ToUnicode is NOT automatically broken. Plenty of ordinary text
    fonts ship without one and encode perfectly standard glyph names — ``A``,
    ``ampersand``, ``bracketleft``, ``Adieresis`` — which the extractor already
    decodes correctly through the standard encoding. Nothing there needs
    recovering and nothing there is at risk.

    Treating "absent from GLYPH_UNICODE" as "unrecoverable" conflated the two and
    was wrong twice over: it reported ~7% of an ordinary paper library as
    containing unrecoverable characters when those papers were fine, and because
    a document needing attention is never cached, it made every one of them
    rescan on every open (#246).

    The defect this module exists for is specific and recognisable: Monotype's
    ``H<number>`` symbol fonts, whose names mean nothing to a standard decoder. A
    font is affected iff at least one glyph it encodes carries such a name.
    Membership in GLYPH_UNICODE is deliberately NOT the test: a subsetted symbol
    font can draw only glyphs the table does not cover yet be exactly as
    corrupted, and gating on "can we recover something" would make the most
    corrupted document the quietest one. Such a font is repaired where the table
    allows and its drawn unknowns are reported, never silently skipped.
    """
    return any(_MONOTYPE_SYMBOL_NAME.fullmatch(name) for name in encoding.values())


def _build_tounicode_cmap(mapping: dict[int, str]) -> bytes:
    """Build the ``/ToUnicode`` CMap the publisher omitted.

    ``mapping`` is character code -> replacement text. Values are emitted as
    UTF-16BE, which is what the CMap ``bfchar`` format requires.
    """
    entries = "".join(
        f"<{code:02X}> <{''.join(f'{ord(c):04X}' for c in text)}>\n"
        for code, text in sorted(mapping.items())
    )
    return (
        "/CIDInit /ProcSet findresource begin\n"
        "12 dict begin\nbegincmap\n"
        "/CMapName /SOCR-Recovered def\n/CMapType 2 def\n"
        "1 begincodespacerange\n<00> <FF>\nendcodespacerange\n"
        f"{len(mapping)} beginbfchar\n{entries}endbfchar\n"
        "endcmap\nCMapName currentdict /CMap defineresource pop\n"
        "end\nend"
    ).encode("latin-1")


def _used_codes_by_font(doc: fitz.Document) -> dict[str, set[int]]:
    """Base font name -> the character codes actually drawn on some page.

    A symbol font encodes far more glyphs than any one paper draws. Reporting an
    encoded-but-never-drawn glyph as unmapped would mark every document suspect
    forever, which is an alarm nobody can act on. Only glyphs that reach the page
    matter, so usage is measured before anything is rewritten.

    Called only when the document actually has a font missing its ToUnicode map,
    so a clean document pays nothing for this pass.
    """
    used: dict[str, set[int]] = {}
    for page_num in range(doc.page_count):
        try:
            blocks = doc[page_num].get_text("rawdict").get("blocks", [])
        except Exception as exc:
            logger.debug("[glyph] page %d rawdict failed: %s", page_num, exc)
            continue
        for block in blocks:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    name = str(span.get("font", ""))
                    if not name:
                        continue
                    bucket = used.setdefault(name, set())
                    for char in span.get("chars", []):
                        text = char.get("c") or ""
                        if text:
                            bucket.add(ord(text[0]))
    return used


def _matches_font(subset_name: str, span_name: str) -> bool:
    """Whether a rawdict span font name refers to this font.

    ``get_page_fonts`` reports the subset-tagged name (``SRQIVV+Universal-Greek``)
    while rawdict spans report an untagged, sometimes truncated form
    (``Universal-GreekwithMathP``), so the two cannot be compared directly.
    """
    base = subset_name.split("+", 1)[-1]
    return base.startswith(span_name) or span_name.startswith(base)


def _font_has_tounicode(doc: fitz.Document, xref: int) -> bool:
    try:
        kind, _ = doc.xref_get_key(xref, "ToUnicode")
    except Exception:
        return True  # cannot tell -> do not touch it
    return kind != "null"


def _attach_cmap(doc: fitz.Document, font_xref: int, cmap: bytes) -> None:
    """Point *font_xref* at a new stream holding *cmap*."""
    stream_xref = doc.get_new_xref()
    doc.update_object(stream_xref, "<<>>")
    doc.update_stream(stream_xref, cmap)
    doc.xref_set_key(font_xref, "ToUnicode", f"{stream_xref} 0 R")


def replay_cmaps(doc: fitz.Document, cmaps: dict[int, bytes]) -> None:
    """Re-apply an already-derived plan to a freshly-opened copy of the file.

    Skips any font that already carries a ToUnicode, so replaying onto a
    document that was somehow repaired already cannot double-attach.
    """
    for font_xref, cmap in cmaps.items():
        try:
            if _font_has_tounicode(doc, font_xref):
                continue
            _attach_cmap(doc, font_xref, cmap)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("[glyph] replay onto xref %d failed (%s)", font_xref, exc)


def repair_symbol_font_text(doc: fitz.Document) -> GlyphRepairReport:
    """Give every embedded font that lacks a ToUnicode map the one it should have.

    Mutates *doc* in memory only; the file on disk is never written. Fonts that
    already carry a ToUnicode map are left alone -- the publisher's own map wins,
    since this recovery exists only for the case where there is none.

    Returns a :class:`GlyphRepairReport`. Callers must surface it: a page whose
    report is ``repaired`` but not ``complete`` still contains characters this
    module could not identify.
    """
    report = GlyphRepairReport()
    seen: set[int] = set()
    unmapped: set[str] = set()
    used_by_font: dict[str, set[int]] | None = None

    for page_num in range(doc.page_count):
        try:
            page_fonts = doc.get_page_fonts(page_num, full=True)
        except Exception as exc:
            logger.debug("[glyph] page %d font list failed: %s", page_num, exc)
            continue

        for entry in page_fonts:
            xref, basefont = entry[0], str(entry[3])
            if xref in seen:
                continue
            seen.add(xref)

            if _font_has_tounicode(doc, xref):
                continue
            try:
                _name, _ext, _type, buffer = doc.extract_font(xref)
            except Exception as exc:
                logger.debug("[glyph] %s: font not extractable (%s)", basefont, exc)
                continue
            if not buffer:
                continue

            encoding = _parse_type1_encoding(buffer)
            if not encoding:
                continue

            # Cheap gate, from glyph names only — no page text is read. An
            # ordinary font that merely lacks a ToUnicode is not this defect and
            # is neither repaired nor reported (#246).
            if not _is_affected_font(encoding):
                continue

            # Measured lazily and once: only a document that actually has a
            # broken font pays for the usage pass.
            if used_by_font is None:
                used_by_font = _used_codes_by_font(doc)
            drawn = {
                code
                for span_name, codes in used_by_font.items()
                if _matches_font(basefont, span_name)
                for code in codes
            }

            mapping: dict[int, str] = {}
            for code, glyph in encoding.items():
                if glyph in _NON_TEXT_GLYPHS:
                    continue
                replacement = GLYPH_UNICODE.get(glyph)
                if replacement is None:
                    # Only a glyph that reaches the page is a real gap. Encoded
                    # but never drawn is noise, and it is left unmapped either
                    # way -- guessing a substitution would be fabrication.
                    if code in drawn:
                        unmapped.add(glyph)
                    continue
                mapping[code] = replacement
            if not mapping:
                # Nothing in this font is recoverable. Not a repair -- but the
                # unmapped glyphs collected above are still reported, so the
                # most corrupted case is not the quietest one.
                continue

            try:
                cmap = _build_tounicode_cmap(mapping)
                _attach_cmap(doc, xref, cmap)
            except Exception as exc:
                logger.warning("[glyph] %s: could not attach ToUnicode (%s)", basefont, exc)
                continue

            report.repaired_fonts.append(basefont)
            report.mapped_glyph_count += len(mapping)
            report.cmaps[xref] = cmap
            logger.info(
                "[glyph] %s: recovered %d glyph(s) from the embedded encoding vector",
                basefont,
                len(mapping),
            )

    report.unmapped_glyphs = sorted(unmapped)
    return report
