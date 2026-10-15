"""#1050 step 2: pure unit tests for ``socr.core.text_recovery``.

Hermetic: no PDF, no pipeline, no model, no Tesseract, no network. Pins the
reading cleaner, the recovered-block marks, the anchor insertion and the
witness check that decide what a model reading may ship as.
"""

from __future__ import annotations


def test_loaded_source_is_this_checkout() -> None:
    """The imported socr package is this worktree's src, not another checkout's."""
    import socr
    from pathlib import Path

    assert Path(socr.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1] / "src")


def test_clean_reading_rejects_empty_and_no_text_replies() -> None:
    """Empty input, whitespace, and the model's NO TEXT reply all mean no reading."""
    from socr.core import text_recovery

    assert text_recovery.clean_reading(None) is None
    assert text_recovery.clean_reading("") is None
    assert text_recovery.clean_reading("   ") is None
    assert text_recovery.clean_reading(text_recovery.EMPTY_REPLY) is None
    assert text_recovery.clean_reading("no text") is None


def test_clean_reading_rejects_commentary_openings() -> None:
    """A refusal is dropped; narration that carries page text is NOT.

    Superseded in round 3 (Fable #6): this used to drop any answer opening
    "I ", "It appears" or "Here is", which threw away genuine transcriptions.
    Only an actual refusal is dropped now; a lead-in is stripped and the words
    after it kept. See ``test_a_lead_in_is_stripped_not_used_to_drop_the_reading``.
    """
    from socr.core import text_recovery

    assert text_recovery.clean_reading("I'm sorry, I cannot read this.") is None
    assert text_recovery.clean_reading("As an AI, I cannot do that.") is None
    # Narration-shaped, but it carries the page's own words: kept.
    assert text_recovery.clean_reading("The image shows a bar chart.") == (
        "The image shows a bar chart."
    )


def test_clean_reading_unwraps_fenced_block() -> None:
    """A fenced block is replaced by its contents, without the fence lines."""
    from socr.core import text_recovery

    assert text_recovery.clean_reading("```\ntext\n```") == "text"


def test_clean_reading_strips_outer_blank_lines_and_keeps_interior() -> None:
    """Leading and trailing blank lines go; interior blank lines stay."""
    from socr.core import text_recovery

    assert text_recovery.clean_reading("\n\nalpha\n\nbeta\n\n") == "alpha\n\nbeta"


def test_recovered_block_single_line_in_figure() -> None:
    """A one-line reading inside a figure gets the figure mark exactly."""
    from socr.core import text_recovery

    block = text_recovery.recovered_block("hello", in_figure=True)
    assert block == "> Text in figure (OCR, not verified): hello"


def test_recovered_block_single_line_not_in_figure() -> None:
    """A one-line reading outside a figure gets the shapes mark exactly."""
    from socr.core import text_recovery

    block = text_recovery.recovered_block("hello", in_figure=False)
    assert block == "> Text drawn as shapes (OCR, not verified): hello"


def test_recovered_block_multiline_keeps_lines_and_blank_quote_lines() -> None:
    """Every reading line is quoted, and a blank reading line becomes a bare '>'."""
    from socr.core import text_recovery

    block = text_recovery.recovered_block("alpha\n\nbeta", in_figure=False)
    assert block == "> Text drawn as shapes (OCR, not verified):\n>\n> alpha\n>\n> beta"


def test_recovered_block_every_form_matches_the_mark_regex() -> None:
    """Every block shape the module produces is matched by RECOVERED_MARK_RE."""
    from socr.core import text_recovery

    blocks = [
        text_recovery.recovered_block("hello", in_figure=True),
        text_recovery.recovered_block("hello", in_figure=False),
        text_recovery.recovered_block("alpha\n\nbeta", in_figure=False),
        text_recovery.recovered_block("alpha\nbeta", in_figure=True),
    ]
    for block in blocks:
        assert text_recovery.RECOVERED_MARK_RE.search(block), block


def test_no_block_can_ever_claim_it_was_checked() -> None:
    """Owner ruling 2026-10-14 (item G): nothing ships as confirmed.

    Token-containment confirmation kept passing wrong readings through three
    review rounds -- a label/value swap, dash-led numbers, glued tokens -- so
    the label was dropped rather than patched again. There is one mark and it
    says the reading was not verified.
    """
    from socr.core import text_recovery

    for block in (
        text_recovery.recovered_block("hello", in_figure=True),
        text_recovery.recovered_block("hello", in_figure=False),
        text_recovery.recovered_block("alpha\nbeta", in_figure=True),
    ):
        assert text_recovery.UNVERIFIED in block
        assert not text_recovery.CONFIRMED_STYLE_RE.search(block), block
    # The seam the old check used is gone, not merely unused.
    assert not hasattr(text_recovery, "check_against_witness")
    assert not hasattr(text_recovery, "RecoveryCheck")


def test_insert_after_anchor_places_block_after_its_paragraph() -> None:
    """The block becomes its own paragraph right after the anchor's paragraph."""
    from socr.core import text_recovery

    body = "First paragraph.\n\nSecond paragraph has the anchor here.\n\nThird paragraph."
    new_body, placed = text_recovery.insert_after_anchor(body, "anchor here", "> BLOCK")
    assert placed is True
    assert new_body == (
        "First paragraph.\n\nSecond paragraph has the anchor here.\n\n> BLOCK\n\nThird paragraph."
    )


def test_insert_after_empty_anchor_places_block_first() -> None:
    """An empty anchor means nothing precedes the region, so the block goes first."""
    from socr.core import text_recovery

    new_body, placed = text_recovery.insert_after_anchor("Alpha.\n\nBeta.", "", "> BLOCK")
    assert placed is True
    assert new_body == "> BLOCK\n\nAlpha.\n\nBeta."


def test_insert_after_missing_anchor_returns_the_same_body() -> None:
    """A missing anchor returns (body, False) with the identical string, never a rebuild."""
    from socr.core import text_recovery

    body = "Alpha.\n\nBeta."
    new_body, placed = text_recovery.insert_after_anchor(body, "not in the page", "> BLOCK")
    assert placed is False
    assert new_body is body


def test_insert_after_anchor_ignores_whitespace_differences() -> None:
    """Runs of spaces and newlines in the anchor still match the paragraph text."""
    from socr.core import text_recovery

    body = "Intro.\n\nthe quick brown fox jumps.\n\nOutro."
    new_body, placed = text_recovery.insert_after_anchor(body, "the  quick\nbrown", "> BLOCK")
    assert placed is True
    assert new_body == "Intro.\n\nthe quick brown fox jumps.\n\n> BLOCK\n\nOutro."


def test_insert_after_anchor_does_not_duplicate_a_present_block() -> None:
    """A block already in the body is reported placed and inserted no second time."""
    from socr.core import text_recovery

    body = "Alpha.\n\n> BLOCK\n\nBeta."
    new_body, placed = text_recovery.insert_after_anchor(body, "Alpha.", "> BLOCK")
    assert placed is True
    assert new_body == body
    assert new_body.count("> BLOCK") == 1


def test_insert_after_an_ambiguous_anchor_refuses_to_guess() -> None:
    """A snippet that matches twice does not determine a position, so it falls back."""
    from socr.core import text_recovery

    body = "running header\n\nmiddle text\n\nrunning header\n\ntail"
    new_body, placed = text_recovery.insert_after_anchor(body, "running header", "> BLOCK")
    assert placed is False
    assert new_body is body


def test_an_earlier_recovered_block_cannot_anchor_the_next_one() -> None:
    """A match inside a blockquote is skipped: it is a previous region's own words."""
    from socr.core import text_recovery

    body = "intro line\n\n> Text in figure (OCR): the shared phrase\n\nthe shared phrase\n\ntail"
    new_body, placed = text_recovery.insert_after_anchor(body, "the shared phrase", "> BLOCK")
    assert placed is True
    assert new_body == (
        "intro line\n\n> Text in figure (OCR): the shared phrase\n\n"
        "the shared phrase\n\n> BLOCK\n\ntail"
    )


def test_a_prefix_block_is_not_mistaken_for_one_already_placed() -> None:
    """A second region whose reading is a prefix of the first's must still be placed.

    `block in body` reported it placed and the caller then dropped the region's
    picture as well — the whole region vanished (Argon, 2026-10-14).
    """
    from socr.core import text_recovery

    first = "> Text drawn as shapes (OCR): Total 100 USD"
    second = "> Text drawn as shapes (OCR): Total 100"
    body, placed = text_recovery.insert_after_anchor("alpha line\nbeta line", "alpha line", first)
    assert placed is True
    body2, placed2 = text_recovery.insert_after_anchor(body, "alpha line", second)
    assert placed2 is True
    assert second in body2.replace(first, "")


def test_two_regions_under_one_anchor_keep_their_order() -> None:
    """The second block goes after the first, not before it."""
    from socr.core import text_recovery

    one = text_recovery.recovered_block("ONE", in_figure=False)
    two = text_recovery.recovered_block("TWO", in_figure=False)
    body, _ = text_recovery.insert_after_anchor("alpha line\nbeta line", "alpha line", one)
    body, _ = text_recovery.insert_after_anchor(body, "alpha line", two)
    assert body.index("ONE") < body.index("TWO") < body.index("beta line")


def test_an_anchor_ending_in_punctuation_still_matches() -> None:
    """Native sentence tails end on a full stop; a \\b there would never match."""
    from socr.core import text_recovery

    body = "A sentence that ends here.\nnext line"
    new_body, placed = text_recovery.insert_after_anchor(body, "that ends here.", "> BLOCK")
    assert placed is True
    assert new_body == "A sentence that ends here.\n\n> BLOCK\n\nnext line"


def test_a_short_anchor_does_not_match_inside_a_word() -> None:
    """ "is a" must not anchor on "this apple"."""
    from socr.core import text_recovery

    body = "this apple\nis a pear\ntail"
    new_body, placed = text_recovery.insert_after_anchor(body, "is a", "> BLOCK")
    assert placed is True
    assert new_body == "this apple\nis a pear\n\n> BLOCK\n\ntail"


def test_two_regions_above_all_native_text_keep_page_order() -> None:
    """Item D: an empty anchor must append after what is already there.

    Both regions sit above every native block, so both get an empty anchor.
    Inserting each at position 0 shipped them second-before-first.
    """
    from socr.core import text_recovery

    first = text_recovery.recovered_block("FIRST", in_figure=False)
    second = text_recovery.recovered_block("SECOND", in_figure=False)
    body, _ = text_recovery.insert_after_anchor("native prose", "", first)
    body, _ = text_recovery.insert_after_anchor(body, "", second)
    assert body.index("FIRST") < body.index("SECOND") < body.index("native prose")


def test_a_block_is_never_separated_from_its_own_picture() -> None:
    """Item E: skipping only blockquotes left B1, B2, pic2, pic1.

    Every region now ships its words AND its picture, so the thing that must
    be stepped over is the whole placed part, not just the quoted lines.
    """
    from socr.core import text_recovery

    first = (
        text_recovery.recovered_block("ONE", in_figure=False)
        + "\n\n![Uncovered text page 1 region 1](figures/uncovered_text_p1_r1.png)"
    )
    second = (
        text_recovery.recovered_block("TWO", in_figure=False)
        + "\n\n![Uncovered text page 1 region 2](figures/uncovered_text_p1_r2.png)"
    )
    body, _ = text_recovery.insert_after_anchor("alpha line\nbeta line", "alpha line", first)
    body, placed = text_recovery.insert_after_anchor(body, "alpha line", second)
    assert placed is True
    assert body.index("ONE") < body.index("region 1") < body.index("TWO")
    assert body.index("TWO") < body.index("region 2") < body.index("beta line")


def test_a_lead_in_is_stripped_not_used_to_drop_the_reading() -> None:
    """Round 3 (Fable #6): a regex must not throw away page content.

    The old rule dropped any answer opening "I ", "It appears" or "Here is" —
    and a label starting with "I", or a figure note reading "It appears in
    Table 3", is ordinary page text. Dropping it is silent content loss.
    """
    from socr.core import text_recovery

    # Genuine transcriptions that the old rule discarded.
    assert text_recovery.clean_reading("I am the axis label") == "I am the axis label"
    assert text_recovery.clean_reading("It appears in Table 3") == "It appears in Table 3"
    # A lead-in is stripped; the transcription after it survives.
    assert text_recovery.clean_reading("Here is the transcription: Quarters") == "Quarters"
    assert text_recovery.clean_reading("The text is: Top 1% share") == "Top 1% share"
    assert text_recovery.clean_reading("Transcription:\nQuarters") == "Quarters"


def test_only_a_refusal_drops_the_reading() -> None:
    """A refusal has nothing in it to ship, so it alone returns None."""
    from socr.core import text_recovery

    for refusal in (
        "I am sorry, I cannot read this image.",
        "I'm unable to read it",
        "As an AI, I cannot help.",
        "Sorry, the image is unreadable.",
    ):
        assert text_recovery.clean_reading(refusal) is None, refusal


def test_a_native_blockquote_is_not_mistaken_for_a_placed_part() -> None:
    """Round 6 (cubic): a leading ">" does not make a paragraph ours.

    A native page may quote, and socr's own unverified-table mark is a
    blockquote too. Treating every quoted paragraph as a placed part moved the
    insertion point PAST text that is not ours to step over, so the recovered
    words landed below someone else's quote.
    """
    from socr.core import text_recovery

    native_quote = "> the author is quoting someone here"
    body = f"anchor line\n\n{native_quote}\n\nfollowing prose"
    block = text_recovery.recovered_block("Recovered words", in_figure=False)
    new_body, placed = text_recovery.insert_after_anchor(body, "anchor line", block)
    assert placed is True
    # Ours goes immediately after the anchor, BEFORE the native quote.
    assert new_body.index(block) < new_body.index(native_quote)


def test_our_own_placed_block_is_still_stepped_over() -> None:
    """The other half: a real recovery block IS ours, so the next one follows it."""
    from socr.core import text_recovery

    first = text_recovery.recovered_block("FIRST", in_figure=False)
    second = text_recovery.recovered_block("SECOND", in_figure=False)
    body, _ = text_recovery.insert_after_anchor("anchor line\n\ntail", "anchor line", first)
    body, placed = text_recovery.insert_after_anchor(body, "anchor line", second)
    assert placed is True
    assert body.index("FIRST") < body.index("SECOND") < body.index("tail")
