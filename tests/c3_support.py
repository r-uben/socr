"""#1251 C3 helper: independent evidence for a stub model reading.

A model reading the page judge accepted is CHECKED only when something independent of any model
confirmed it: the words check PASSED at page scope (the page's own text layer owes words, and the
reading carries them all) and the numbers check PASSED or does not apply. A test whose control
page is a stub model reading (``engine="qwen"``, SUCCESS) is about the loop, the resume gate or a
counter, not about that label, so its fixture has to give the page the evidence a real
born-digital page has. This module does that in the two shapes the tests come in.

* **State level** (a hand-built ``PageState``): :func:`give_word_evidence` sets the page's
  ``own_layer_word_reference`` to the words of the reading, exactly the field
  ``BornDigitalDetector._own_layer_word_reference`` fills from a real layer.
* **PDF level** (a synthetic PDF the pipeline assesses): :func:`print_reading_layer` prints the
  reading's words and numbers onto the PDF page as its text layer, so the real assessment builds
  the reference and the real checks pass. The words are placed in the page body, and the reading
  is given per page, so that no page repeats another's text: text identical on every page is
  running furniture, which the words check does not owe.

Nothing here patches a check. A reading that does not carry the layer's words still fails the
real words check, which is the point of using the real one.
"""

from __future__ import annotations

from typing import Any

from socr.tables.layer_words import word_tokens_in_order

#: Where the reading's words are printed on a PDF page, in points from the top-left: inside the
#: body, away from the top and bottom edges where a repeated slot is excused (``slot_predicate``).
BODY_ORIGIN = (72.0, 300.0)

#: One line of the layer is at most this many words, so a long reading wraps inside the page
#: width (612 pt) at ``LAYER_FONT_SIZE``; a word the page cut off would not be in the layer.
WORDS_PER_LINE = 8
LAYER_FONT_SIZE = 9
LAYER_LINE_HEIGHT = 12.0


def reading_words(text: str) -> str:
    """The words the words check compares in *text* (four or more letters, case-folded)."""
    return " ".join(word_tokens_in_order(text))


def give_word_evidence(page_state: Any, text: str | None = None) -> Any:
    """Give a hand-built ``PageState`` the words reference a born-digital layer would owe.

    *text* defaults to the page's best output. The numbers check reads ``own_layer_text``, which
    the caller's fixture already sets where it needs a number agreed; this helper does not touch
    it. Returns the page state for chaining."""
    if text is None:
        text = (page_state.best_output.text if page_state.best_output else "") or ""
    page_state.own_layer_word_reference = reading_words(text)
    return page_state


def print_reading_layer(page: Any, text: str, *, top: float = BODY_ORIGIN[1]) -> None:
    """Print *text* (a reading's words and numbers) onto a PyMuPDF *page* as its text layer."""
    tokens = text.split()
    x = BODY_ORIGIN[0]
    for i in range(0, len(tokens), WORDS_PER_LINE):
        line = " ".join(tokens[i : i + WORDS_PER_LINE])
        page.insert_text(
            (x, top + (i // WORDS_PER_LINE) * LAYER_LINE_HEIGHT), line, fontsize=LAYER_FONT_SIZE
        )


#: One line of prose a synthetic multi-page PDF prints on each page, and a stub reading repeats.
PAGE_LINE = "Quarterly growth by country as reported by the national statistics offices"
_PAGE_NAMES = ("alpha", "beta", "gamma", "delta", "epsilon", "zeta", "theta", "kappa")


def page_line(n: int) -> str:
    """Page *n*'s line. It differs per page: text identical on every page is running furniture,
    and the words check does not owe furniture."""
    if n <= len(_PAGE_NAMES):
        return f"{PAGE_LINE} {_PAGE_NAMES[n - 1]}"
    # Beyond the named pages: an all-letter token (digits spelled a..j) unique to the page.
    return f"{PAGE_LINE} page{''.join('abcdefghij'[int(d)] for d in str(n))}"


def add_page_lines(pdf_path: Any, *, top: float = 100.0) -> None:
    """Print :func:`page_line` on every page of the PDF at *pdf_path*, in place.

    The page's own text layer then owes the words of that line, which a stub reading carrying
    :func:`page_line` returns, so the real words check PASSES. ``top`` is above the body the
    fixtures print (a stub table or grid), so the line does not overlap it."""
    import fitz

    doc = fitz.open(str(pdf_path))
    try:
        for i, page in enumerate(doc, start=1):
            print_reading_layer(page, page_line(i), top=top)
        doc.saveIncr()
    finally:
        doc.close()


def reading_with_layer_words(pdf_path: Any, text: str, *, page_num: int = 1) -> str:
    """*text* followed by the words the page's own layer owes a reading.

    For a stub reading of a REAL committed PDF that transcribes only a table: the page's layer
    also prints a title, notes and prose, which the words check owes the reading. A real model
    reading would carry them; the stub carries them as one trailing paragraph of words, no
    digits, so the numbers check compares exactly what it compared before. The words are the
    page assessment's own ``own_layer_word_reference`` (what the real run compares against:
    line-end hyphens joined, furniture and figures left out), so the real words check PASSES
    and is not bypassed."""
    from socr.core.born_digital import BornDigitalDetector

    assessment = BornDigitalDetector().detect(pdf_path)
    owed = (assessment.pages[page_num - 1].own_layer_word_reference or "").strip()
    return f"{text.rstrip()}\n\n{owed}\n" if owed else text
