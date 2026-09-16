"""Tests for GH-778: ``_native_label_lane`` decodes an entity-encoded label.

``_native_label_lane`` (``header_repair.py``) tokenises a candidate grid's
label and compares it word-for-word against native PyMuPDF words. Native
words never carry HTML entities -- the label can, because it comes from the
model's candidate grid. Pre-fix, ``re.findall(r"[\\w&]+", label.casefold())``
ran on the RAW label, so an entity adjacent to a word corrupted that word's
own token (``'Far&nbsp;outcome'`` -> ``['far&nbsp', 'outcome']``, the first
token no longer equal to the native ``'far'``) and the lane match silently
abstained.

Hermetic: synthetic fitz pages + markdown grids, no ollama/GPU. Reuses the
``TestRepairTooNarrowSpanningHeader`` fixture geometry from
``test_header_repair.py`` -- the label this function receives is exactly the
last cell of a widened header row in that repair.
"""

from __future__ import annotations

from socr.tables.header_repair import (
    repair_table_headers_in_text,
    repair_table_headers_on_page,
)
from socr.tables.reconcile import find_table_blocks
from test_header_repair import _make_narrow_spanning_header_page, _md_table


def _malformed(label: str) -> str:
    return _md_table(
        ["", "Dependent variable:", "", ""],
        [
            ["", "Near outcome", "", label],
            ["", "(1)", "(2)", "(3)", "(4)"],
            ["Signal", "-4.8", "", "-4.1", "-0.2"],
            ["Control", "0.1", "0.2", "0.3", "0.4"],
        ],
    )


def test_entity_encoded_label_binds_where_plain_form_does() -> None:
    """An entity glued to the label word must still bind (GH-778)."""
    page = _make_narrow_spanning_header_page()

    plain_md, plain_count = repair_table_headers_on_page(page, _malformed("Far outcome"))
    entity_md, entity_count = repair_table_headers_on_page(page, _malformed("Far&nbsp;outcome"))

    assert plain_count == 1, "baseline: plain label must bind to its native geometry"
    assert entity_count == 1, (
        "an entity-encoded label must bind exactly like its plain form; "
        "pre-fix the corrupted token ('far&nbsp' != 'far') aborts the repair"
    )

    plain_grid = find_table_blocks(plain_md)[0].grid
    entity_grid = find_table_blocks(entity_md)[0].grid
    # Same column the plain label lands in -- the entity only changed the
    # label's spelling, not the geometry it should bind to.
    assert plain_grid[1].index("Far outcome") == entity_grid[1].index("Far&nbsp;outcome")


def test_entity_inside_the_label_is_the_corrupted_token_case() -> None:
    """The entity sits between two words with no surrounding space.

    This is the case the ticket calls out as most important: the entity does
    not just add a bogus extra token, it glues onto and corrupts the token
    that would otherwise have matched.
    """
    page = _make_narrow_spanning_header_page()
    md, count = repair_table_headers_on_page(page, _malformed("Far&nbsp;outcome"))
    assert count == 1
    grid = find_table_blocks(md)[0].grid
    assert "Far&nbsp;outcome" in grid[1]


def test_genuinely_different_label_still_does_not_bind() -> None:
    """Decoding must not make the matcher promiscuous.

    A label with no relation to the native geometry -- entity-encoded or not
    -- must still abstain. This is the load-bearing negative test: it must
    fail (bind) on a matcher that decodes too aggressively or fuzzily.
    """
    page = _make_narrow_spanning_header_page()
    md, count = repair_table_headers_on_page(page, _malformed("Something&nbsp;else&nbsp;entirely"))
    assert count == 0, "an unrelated label must not bind to any native lane"
    assert md == _malformed("Something&nbsp;else&nbsp;entirely")


def test_abstains_without_page_geometry_even_for_entity_label() -> None:
    """No native words at all -- must abstain regardless of the label form."""
    md, count = repair_table_headers_in_text([], _malformed("Far&nbsp;outcome"))
    assert count == 0
    assert md == _malformed("Far&nbsp;outcome")


def test_literal_ampersand_in_label_still_binds() -> None:
    """A real ampersand in the label ('Profit & Loss') must survive decode.

    ``&`` stays in the label-token character class on purpose: after
    ``html.unescape`` runs, any ``&`` still present is a literal ampersand in
    the text itself, and native words tokenise it as its own "&" word.
    """
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=600, height=300)
    page.insert_text((150.0, 80.0), "Near outcome", fontsize=9)
    page.insert_text((315.0, 80.0), "Profit & Loss", fontsize=9)
    for x, ordinal in zip([150.0, 215.0, 280.0, 345.0], ["(1)", "(2)", "(3)", "(4)"]):
        page.insert_text((x, 110.0), ordinal, fontsize=9)
    for x, value in zip([150.0, 215.0, 280.0, 345.0], ["-4.8", None, "-4.1", "-0.2"]):
        if value is not None:
            page.insert_text((x, 140.0), value, fontsize=9)
    for x, value in zip([150.0, 215.0, 280.0, 345.0], ["0.1", "0.2", "0.3", "0.4"]):
        page.insert_text((x, 170.0), value, fontsize=9)

    md, count = repair_table_headers_on_page(page, _malformed("Profit & Loss"))
    assert count == 1, "a label with a literal ampersand must still bind"
    grid = find_table_blocks(md)[0].grid
    assert "Profit & Loss" in grid[1]
