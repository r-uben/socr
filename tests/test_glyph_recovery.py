"""Symbol fonts with no ToUnicode map must not ship sign-flipped numbers (#217).

The corpus this was found in is copyrighted and this repo is public, so no fixture
PDF can be committed. These tests therefore drive the module with synthetic Type1
encoding vectors and a stub document that mimics the PyMuPDF surface actually
used. That keeps them hermetic -- no corpus, no network, no provider.
"""

from __future__ import annotations

import re

import pytest

from socr.core.glyph_recovery import (
    GLYPH_UNICODE,
    GlyphRepairReport,
    _build_tounicode_cmap,
    _matches_font,
    _parse_type1_encoding,
    repair_symbol_font_text,
)

# A Type1 font program carries its encoding as ``dup <code> /<glyph> put``.
SYNTHETIC_TYPE1 = b"""
/Encoding 256 array
0 1 255 {1 index exch /.notdef put} for
dup 32 /space put
dup 50 /H11002 put
dup 112 /H9266 put
dup 126 /H20849 put
dup 200 /H99999 put
readonly def
"""


class StubDoc:
    """The narrow slice of the fitz.Document surface the module touches."""

    def __init__(self, fonts, chars_by_font, tounicode=None, buffers=None):
        self._fonts = fonts  # [(xref, ext, type, basefont, name, enc)]
        self._chars = chars_by_font  # {span_font_name: "chars drawn"}
        self._tounicode = tounicode or {}  # {xref: bool has_map}
        self._buffers = buffers or {}
        self.page_count = 1
        self.attached: dict[int, bytes] = {}
        #: font xref -> the CMap stream xref it was actually pointed at. A map
        #: that is written but never linked has no effect, so the two are
        #: tracked separately, exactly as the PDF does it.
        self.activated: dict[int, int] = {}
        self._next_xref = 900

    # -- font discovery -------------------------------------------------
    def get_page_fonts(self, page_num, full=True):
        return self._fonts

    def xref_get_key(self, xref, key):
        if key == "ToUnicode":
            return ("xref", "1 0 R") if self._tounicode.get(xref) else ("null", "null")
        return ("null", "null")

    def extract_font(self, xref):
        return ("Font", "pfa", "Type1", self._buffers.get(xref, SYNTHETIC_TYPE1))

    # -- text -----------------------------------------------------------
    def __getitem__(self, page_num):
        return self

    def _decode(self, code: int) -> str:
        """Apply an attached ToUnicode map the way a real extractor would.

        Without this the stub would return the raw byte no matter what the
        module did, so a regression that skipped ``xref_set_key`` -- the step
        that actually makes the map take effect -- would pass every test while
        the sign flip stayed broken.

        A code absent from the map falls back to the raw byte, which is the real
        PyMuPDF behaviour (verified against the source PDF: omitting the minus
        from the table leaves ``20.12`` extracting as ``20.12``, unchanged).
        """
        for stream_xref in self.activated.values():
            data = self.attached.get(stream_xref)
            if data is None:
                continue
            mapped = self._cmap_entries(data)
            if code in mapped:
                return mapped[code]
        return chr(code)

    @staticmethod
    def _cmap_entries(cmap: bytes) -> dict[int, str]:
        """Parse ``<code> <utf16be>`` pairs back out of an attached CMap."""
        body = cmap.decode("latin-1")
        pairs = re.findall(r"<([0-9A-F]{2})>\s*<([0-9A-F]{4,})>", body)
        return {
            int(code, 16): "".join(chr(int(value[i : i + 4], 16)) for i in range(0, len(value), 4))
            for code, value in pairs
        }

    def get_text(self, kind):
        spans = [
            {"font": name, "chars": [{"c": self._decode(ord(c))} for c in drawn]}
            for name, drawn in self._chars.items()
        ]
        return {"blocks": [{"lines": [{"spans": spans}]}]}

    def extracted_text(self) -> str:
        """What a reader would now get for the drawn characters."""
        return "".join(self._decode(ord(c)) for drawn in self._chars.values() for c in drawn)

    # -- mutation -------------------------------------------------------
    def get_new_xref(self):
        self._next_xref += 1
        return self._next_xref

    def update_object(self, xref, text):
        return None

    def update_stream(self, xref, data):
        self.attached[xref] = data

    def xref_set_key(self, xref, key, value):
        if key != "ToUnicode":
            return
        self._tounicode[xref] = True
        self.activated[xref] = int(str(value).split()[0])


def _doc(drawn="2p", has_map=False):
    return StubDoc(
        fonts=[(212, "pfa", "Type1", "ABCDEF+Universal-GreekwithMathPi", "F1", "")],
        chars_by_font={"Universal-GreekwithMathP": drawn},
        tounicode={212: has_map},
    )


class TestEncodingParsing:
    def test_reads_code_to_glyph_name(self):
        enc = _parse_type1_encoding(SYNTHETIC_TYPE1)
        assert enc[50] == "H11002"
        assert enc[112] == "H9266"
        assert enc[32] == "space"

    def test_non_type1_buffer_yields_nothing(self):
        assert _parse_type1_encoding(b"not a font program") == {}


class TestCMap:
    def test_emits_utf16be_bfchar_entries(self):
        cmap = _build_tounicode_cmap({0x32: "−"}).decode("latin-1")
        assert "beginbfchar" in cmap and "endbfchar" in cmap
        assert "<32> <2212>" in cmap

    def test_entry_count_matches_the_declared_count(self):
        cmap = _build_tounicode_cmap({0x31: "+", 0x32: "−", 0x35: "="}).decode("latin-1")
        assert "3 beginbfchar" in cmap


class TestTable:
    @pytest.mark.parametrize(
        ("glyph", "char"),
        [
            ("H11002", "−"),  # the sign-flip defect
            ("H11001", "+"),
            ("H11005", "="),
            ("H20849", "("),  # standard-error brackets
            ("H20850", ")"),
            ("H9266", "π"),
            ("H9262", "μ"),
            ("H9004", "Δ"),
        ],
    )
    def test_verified_entries(self, glyph, char):
        assert GLYPH_UNICODE[glyph] == char

    def test_minus_is_the_unicode_minus_not_a_hyphen(self):
        """A hyphen would be a different character than the page prints."""
        assert GLYPH_UNICODE["H11002"] == "−"

    def test_no_entry_maps_to_a_latin_digit_or_letter(self):
        """The whole defect is symbol glyphs decoding as Latin text.

        An entry mapping back into ASCII alphanumerics would reintroduce it, so
        it is almost certainly a transcription slip rather than a real glyph.
        """
        offenders = {g: c for g, c in GLYPH_UNICODE.items() if c.isascii() and c.isalnum()}
        assert not offenders


class TestFontNameMatching:
    def test_subset_tag_is_ignored(self):
        assert _matches_font("ABCDEF+Universal-GreekwithMathPi", "Universal-GreekwithMathP")

    def test_unrelated_font_does_not_match(self):
        assert not _matches_font("ABCDEF+Universal-GreekwithMathPi", "Times-Roman")


class TestRepair:
    def test_font_without_tounicode_is_repaired(self):
        doc = _doc()
        report = repair_symbol_font_text(doc)
        assert report.repaired
        assert report.mapped_glyph_count > 0
        assert doc.attached, "no CMap stream was written"

    def test_repair_maps_the_minus_sign(self):
        doc = _doc()
        repair_symbol_font_text(doc)
        cmap = next(iter(doc.attached.values())).decode("latin-1")
        assert "<32> <2212>" in cmap

    def test_extracted_text_is_actually_corrected(self):
        """The outcome, not the mechanism: '2' must read as a minus afterwards.

        This is the defect in one line -- '-0.12' shipping as '20.12'. Asserting
        only that a CMap stream was attached would still pass if the map were
        never made to take effect.
        """
        doc = _doc(drawn="2p")
        assert doc.extracted_text() == "2p"
        repair_symbol_font_text(doc)
        assert doc.extracted_text() == "−π"

    def test_forgetting_to_activate_the_map_is_caught(self):
        """Attaching the stream but never pointing the font at it must fail.

        Guards the one step that makes PyMuPDF honour the map.
        """
        doc = _doc(drawn="2")
        doc.xref_set_key = lambda *a, **k: None  # attach, but never activate
        repair_symbol_font_text(doc)
        assert doc.extracted_text() == "2", "map took effect without being activated"

    def test_font_with_its_own_tounicode_is_left_alone(self):
        """The publisher's own map wins; this recovery is only for its absence."""
        doc = _doc(has_map=True)
        report = repair_symbol_font_text(doc)
        assert not report.repaired
        assert not doc.attached

    def test_unknown_glyph_that_is_drawn_is_reported(self):
        """Code 200 maps to H99999, which the table does not cover."""
        doc = _doc(drawn="2" + chr(200))
        report = repair_symbol_font_text(doc)
        assert report.unmapped_glyphs == ["H99999"]
        assert not report.complete

    def test_unknown_glyph_that_is_never_drawn_is_not_reported(self):
        """Encoded-but-unused glyphs must not mark every document suspect."""
        doc = _doc(drawn="2")
        report = repair_symbol_font_text(doc)
        assert report.unmapped_glyphs == []
        assert report.complete

    def test_unknown_glyph_is_not_guessed(self):
        """An unmapped glyph must be absent from the CMap, not approximated."""
        doc = _doc(drawn="2" + chr(200))
        repair_symbol_font_text(doc)
        cmap = next(iter(doc.attached.values())).decode("latin-1")
        assert "<C8>" not in cmap

    def test_unknown_glyph_survives_a_partial_repair_unchanged(self):
        """A partial CMap must not disturb the codes it omits.

        Verified against the source PDF as well: deleting the minus entry from
        the table leaves ``20.12`` extracting as ``20.12``, byte for byte.
        """
        doc = _doc(drawn="2" + chr(200))
        repair_symbol_font_text(doc)
        text = doc.extracted_text()
        assert text[0] == "−", "the mapped glyph was not corrected"
        assert text[1] == chr(200), "the unmapped glyph was altered"

    def test_all_drawn_glyphs_unmapped_still_reports(self):
        """The worst case must not be the silent one.

        When nothing drawn is in the table no CMap is attached, so an early
        return keyed on 'was anything repaired' would skip the warning on the
        most corrupted documents there are.
        """
        doc = _doc(drawn=chr(200))
        report = repair_symbol_font_text(doc)
        assert report.unmapped_glyphs == ["H99999"]
        assert report.needs_attention

    def test_space_is_not_reported_as_unmapped(self):
        doc = _doc(drawn="2 ")
        report = repair_symbol_font_text(doc)
        assert report.unmapped_glyphs == []

    def test_unextractable_font_does_not_raise(self):
        doc = _doc()
        doc.extract_font = lambda xref: (_ for _ in ()).throw(RuntimeError("no font"))
        report = repair_symbol_font_text(doc)
        assert not report.repaired


class TestReport:
    def test_complete_is_false_when_nothing_was_repaired(self):
        """A document with no broken font is not 'completely repaired'."""
        assert not GlyphRepairReport().complete


class TestDetectorSurfacing:
    """The report must reach something that ships, not only the log (#217).

    A warning in a log file is not a signal a caller can act on -- the same gap
    #136 was filed for.
    """

    def _detector_with(self, report):
        from socr.core.born_digital import BornDigitalDetector, PageAssessment

        detector = BornDigitalDetector()
        detector.last_glyph_repair = report
        page = dict(is_born_digital=True, native_text="x", confidence=1.0)
        return detector, [PageAssessment(page_num=1, **page), PageAssessment(page_num=2, **page)]

    def test_unrecovered_glyphs_mark_every_page(self):
        """The missing map is a font property, so no page of it is clean."""
        report = GlyphRepairReport(repaired_fonts=["F"], unmapped_glyphs=["H99999"])
        detector, pages = self._detector_with(report)
        detector._mark_unrecovered_glyphs(pages)
        assert all(p.has_unrecovered_symbol_glyphs for p in pages)

    def test_a_fully_recovered_document_marks_nothing(self):
        report = GlyphRepairReport(repaired_fonts=["F"], mapped_glyph_count=3)
        detector, pages = self._detector_with(report)
        detector._mark_unrecovered_glyphs(pages)
        assert not any(p.has_unrecovered_symbol_glyphs for p in pages)

    def test_missing_report_marks_nothing(self):
        detector, pages = self._detector_with(None)
        detector._mark_unrecovered_glyphs(pages)
        assert not any(p.has_unrecovered_symbol_glyphs for p in pages)

    def test_failed_recovery_does_not_leave_a_stale_report(self, tmp_path, monkeypatch):
        """A reused detector must not attribute the previous PDF's report here."""
        import socr.core.born_digital as bd

        detector, _ = self._detector_with(GlyphRepairReport(repaired_fonts=["previous"]))
        monkeypatch.setattr(
            bd, "repair_symbol_font_text", lambda doc: (_ for _ in ()).throw(RuntimeError())
        )
        detector._recover_symbol_fonts(object(), tmp_path / "doc.pdf")
        assert detector.last_glyph_repair is None
