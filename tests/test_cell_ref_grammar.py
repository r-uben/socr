"""P1 (docs/log/2026-09-03_p1-ladder-flip.md, task t1): the canonical,
value-free cell-reference grammar the tiebreak/withhold chain uses to name
doubted or FAIL-flagged cells without ever carrying their contents.

Two things live here:

- ``parse_cell_ref`` / ``CellRef`` -- the grammar itself: ``R2C3`` (body row 2,
  col 3) and ``H1C2`` (header row 1, col 2), 1-indexed to match how a judge
  would describe "the third column of the second row" in prose.
- ``resolve_cell_refs`` -- a conservative resolver from a set of ``CellRef``
  against ONE emitted markdown table (via the existing strict grid parser),
  returning the extraction token at each ref. A missing, malformed,
  out-of-range, or non-unique reference makes the WHOLE requested set
  unresolved -- never a partial, guessed answer -- because a partial resolve
  would let the guard service silently drop the very cells that were flagged.

No new comparison rule: token equality delegates to
``socr.tables.adjudication.tokens_agree`` wherever this module needs to know
whether two tokens "match" -- this file does not duplicate or approximate that
logic; it only pins the wiring.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# NOTE: table_verdict.py is A1's owned module; this is the natural home for
# the grammar it adds. Import from there per the plan (t1: "Extend
# TableJudgeVerdict ... canonical cell grammar").
from socr.judge.table_verdict import CellRef, parse_cell_ref, resolve_cell_refs

BODY_TABLE = (
    "| Region | OLS | IV  |\n"
    "| ------ | --- | --- |\n"
    "| RowA   | 100 | 200 |\n"
    "| RowB   | 300 | 400 |\n"
)


# --------------------------------------------------------------------------
# Grammar: parsing
# --------------------------------------------------------------------------


class TestParseCellRef:
    @pytest.mark.parametrize(
        "text,row,col,header",
        [
            ("R2C3", 2, 3, False),
            ("R1C1", 1, 1, False),
            ("H1C2", 1, 2, True),
            ("H2C1", 2, 1, True),
        ],
    )
    def test_valid_refs_parse(self, text: str, row: int, col: int, header: bool) -> None:
        ref = parse_cell_ref(text)
        assert isinstance(ref, CellRef)
        assert ref.row == row
        assert ref.col == col
        assert ref.is_header is header

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "R2",
            "C3",
            "row2col3",
            "R2 C3",
            "R2C3 ",
            "r2c3",  # lowercase not accepted -- one canonical spelling
            "R0C1",  # 1-indexed: zero is out of the grammar, not just out of range
            "R2C0",
            "R-1C1",
            "RXC1",
            "R2CX",
            "H0C1",
            "X1C1",
            "R2C3extra",
            "R2C3H1",
        ],
    )
    def test_malformed_refs_reject(self, text: str) -> None:
        with pytest.raises(ValueError):
            parse_cell_ref(text)

    def test_ref_equality_and_hash_are_value_based(self) -> None:
        assert parse_cell_ref("R2C3") == parse_cell_ref("R2C3")
        assert len({parse_cell_ref("R2C3"), parse_cell_ref("R2C3")}) == 1

    def test_ref_str_round_trips_to_canonical_form(self) -> None:
        # A ref built by hand and printed back must reproduce the SAME
        # canonical spelling the grammar accepts -- this is what a prompt
        # asking the model to "answer keyed by exactly these references" relies on.
        ref = parse_cell_ref("R2C3")
        assert str(ref) == "R2C3"
        header = parse_cell_ref("H1C2")
        assert str(header) == "H1C2"


# --------------------------------------------------------------------------
# Resolver: happy path
# --------------------------------------------------------------------------


class TestResolveCellRefs:
    def test_body_cell_resolves_to_its_token(self) -> None:
        resolved = resolve_cell_refs(BODY_TABLE, [parse_cell_ref("R1C2")])
        assert resolved is not None
        assert resolved[parse_cell_ref("R1C2")] == "100"

    def test_header_cell_resolves_to_its_token(self) -> None:
        # Round 3, NEW A: header columns are PHYSICAL, like body columns, so
        # H1C1 is the leftmost header cell. It used to skip one column on the
        # assumption that every table's first column is a row label.
        resolved = resolve_cell_refs(BODY_TABLE, [parse_cell_ref("H1C1")])
        assert resolved is not None
        assert resolved[parse_cell_ref("H1C1")] == "Region"

    def test_multiple_refs_resolve_together(self) -> None:
        refs = [parse_cell_ref("R1C2"), parse_cell_ref("R2C3"), parse_cell_ref("H1C2")]
        resolved = resolve_cell_refs(BODY_TABLE, refs)
        assert resolved is not None
        assert resolved[parse_cell_ref("R1C2")] == "100"
        assert resolved[parse_cell_ref("R2C3")] == "400"
        assert resolved[parse_cell_ref("H1C2")] == "OLS"

    def test_the_leftmost_column_is_c1_in_both_row_families(self) -> None:
        resolved = resolve_cell_refs(BODY_TABLE, [parse_cell_ref("R1C1"), parse_cell_ref("H1C1")])
        assert resolved is not None
        assert resolved[parse_cell_ref("R1C1")] == "RowA"
        assert resolved[parse_cell_ref("H1C1")] == "Region"

    def test_the_last_header_column_is_addressable(self) -> None:
        """The off-by-one used to make the RIGHTMOST header cell unreachable:
        with three columns, H1C3 fell outside the shifted bound."""
        resolved = resolve_cell_refs(BODY_TABLE, [parse_cell_ref("H1C3")])
        assert resolved is not None
        assert resolved[parse_cell_ref("H1C3")] == "IV"


# --------------------------------------------------------------------------
# Resolver: whole-set failure on any bad ref
# --------------------------------------------------------------------------


class TestResolveCellRefsFailsClosed:
    def test_out_of_range_row_unresolves_the_whole_set(self) -> None:
        refs = [parse_cell_ref("R1C2"), parse_cell_ref("R99C1")]
        assert resolve_cell_refs(BODY_TABLE, refs) is None

    def test_out_of_range_column_unresolves_the_whole_set(self) -> None:
        refs = [parse_cell_ref("R1C2"), parse_cell_ref("R1C99")]
        assert resolve_cell_refs(BODY_TABLE, refs) is None

    def test_header_row_out_of_range_unresolves_the_whole_set(self) -> None:
        refs = [parse_cell_ref("H1C1"), parse_cell_ref("H2C1")]
        assert resolve_cell_refs(BODY_TABLE, refs) is None

    def test_table_that_does_not_parse_as_a_grid_unresolves(self) -> None:
        not_a_table = "This is prose, not a markdown table at all."
        assert resolve_cell_refs(not_a_table, [parse_cell_ref("R1C1")]) is None

    def test_empty_ref_list_resolves_to_empty_mapping_not_none(self) -> None:
        # Distinguishes "nothing was asked" from "something was asked and
        # could not be resolved" -- callers (the guard service, t6) treat an
        # empty doubts/findings set as NOT_CLEARED for a different reason,
        # but the resolver itself must not conflate the two.
        resolved = resolve_cell_refs(BODY_TABLE, [])
        assert resolved == {}

    def test_duplicate_refs_do_not_unresolve_the_set(self) -> None:
        refs = [parse_cell_ref("R1C2"), parse_cell_ref("R1C2")]
        resolved = resolve_cell_refs(BODY_TABLE, refs)
        assert resolved is not None
        assert resolved[parse_cell_ref("R1C2")] == "100"

    def test_one_bad_ref_among_many_good_ones_unresolves_all(self) -> None:
        """The whole-set-unresolved rule: a resolver that returned the good
        refs and dropped only the bad one would silently clear cells the
        gate never actually verified."""
        refs = [
            parse_cell_ref("R1C1"),
            parse_cell_ref("R1C2"),
            parse_cell_ref("R2C3"),
            parse_cell_ref("R7C7"),
        ]
        assert resolve_cell_refs(BODY_TABLE, refs) is None


# --------------------------------------------------------------------------
# Comparison delegates to tokens_agree: numeric and label normalization
# --------------------------------------------------------------------------


class TestNumericNormalization:
    def test_resolved_token_agrees_with_numeric_variants(self) -> None:
        from socr.tables.adjudication import tokens_agree

        resolved = resolve_cell_refs(BODY_TABLE, [parse_cell_ref("R1C2")])
        assert resolved is not None
        tok = resolved[parse_cell_ref("R1C2")]
        assert tok == "100"
        # Numeric normalization via tokens_agree
        assert tokens_agree(tok, "100", kind="cell")
        assert tokens_agree(tok, " 100 ", kind="cell")
        assert tokens_agree(tok, "(100)", kind="cell")
        assert tokens_agree(tok, "100%", kind="cell")
        assert not tokens_agree(tok, "200", kind="cell")

    def test_resolved_row_label_agrees_with_normalized_label(self) -> None:
        from socr.tables.adjudication import tokens_agree

        resolved = resolve_cell_refs(BODY_TABLE, [parse_cell_ref("R1C1")])
        assert resolved is not None
        tok = resolved[parse_cell_ref("R1C1")]
        assert tok == "RowA"
        assert tokens_agree(tok, "rowa", kind="row_label")
        assert tokens_agree(tok, "Row A", kind="row_label")


# --------------------------------------------------------------------------
# Structural-only findings
# --------------------------------------------------------------------------


class TestStructuralOnlyFindings:
    def test_structural_where_is_not_cell_localizable(self) -> None:
        from socr.judge.table_verdict import parse_table_verdict

        payload = {
            "verdict": "FAIL",
            "confidence": "high",
            "findings": [
                {
                    "code": "STRUCTURE_MERGED",
                    "where": "table-wide",
                    "detail": "columns merged across header",
                }
            ],
        }
        import json

        verdict = parse_table_verdict(json.dumps(payload))
        assert verdict.findings[0].where == "table-wide"
        with pytest.raises(ValueError):
            parse_cell_ref(verdict.findings[0].where)


# --------------------------------------------------------------------------
# The coordinate CONTRACT: one grammar, shared by every consumer.
# --------------------------------------------------------------------------


class TestOneCoordinateContract:
    """Rounds 2 (N1), 3 (NEW A) and 4 (NEW 1): ONE physical coordinate rule.

    Round 2 found the rule written out twice, in prose, with the two copies
    disagreeing about which physical column ``C1`` is. Round 3 found that
    fixing the wording was not enough, because the rule ITSELF assumed every
    table has a leading name column: header coordinates skipped it, body
    coordinates did not, and nothing in this repo detects one. Round 4 found
    that the shared fragment's WORKED EXAMPLES carried literal cell values,
    which the blind-transcription prompt then spliced in -- an answer key
    handed to a reader whose entire value is that it has seen nothing but the
    image.

    So the contract is now: one PHYSICAL, stub-free rule, in one file, shown to
    every prompt; and worked examples in a SECOND file that only the reader
    prompt sees, whose every cell holds its own reference rather than a
    plausible value.
    """

    #: Each shape: the table, and what the ONE rule says each reference names.
    #: Spelled out here rather than computed, so this file is an independent
    #: statement of the contract rather than an echo of the implementation.
    SHAPES = {
        "stubbed": (
            "| Region | Alpha | Beta |\n| --- | --- | --- |\n| North | 11 | 12 |\n",
            {
                "H1C1": "Region",
                "H1C2": "Alpha",
                "H1C3": "Beta",
                "R1C1": "North",
                "R1C2": "11",
                "R1C3": "12",
            },
        ),
        "stubless": (
            "| Alpha | Beta |\n| --- | --- |\n| 11 | 12 |\n",
            {"H1C1": "Alpha", "H1C2": "Beta", "R1C1": "11", "R1C2": "12"},
        ),
        "spanning_header": (
            "| Region | Panel A | Panel A | Panel B | Panel B |\n"
            "| Region | Early | Late | Early | Late |\n"
            "| --- | --- | --- | --- | --- |\n"
            "| North | 11 | 12 | 41 | 42 |\n",
            {
                "H1C1": "Region",
                "H1C2": "Panel A",
                "H1C4": "Panel B",
                "H2C1": "Region",
                "H2C2": "Early",
                "H2C5": "Late",
                "R1C1": "North",
                "R1C2": "11",
                "R1C5": "42",
            },
        ),
    }

    BANNED_VOCABULARY = ("stub", "row label", "row-label")

    def test_both_prompts_splice_the_same_rule(self):
        from socr.judge.table_prompt import build_table_judge_prompt
        from socr.judge.table_rung_ollama import build_blind_cell_prompt
        from socr.judge.table_verdict import load_cell_ref_grammar

        grammar = load_cell_ref_grammar()
        assert grammar.strip()
        assert grammar in build_table_judge_prompt(BODY_TABLE)
        assert grammar in build_blind_cell_prompt(["R1C1"])

    def test_neither_prompt_leaves_a_placeholder_unfilled(self):
        from socr.judge.table_prompt import build_table_judge_prompt
        from socr.judge.table_rung_ollama import build_blind_cell_prompt

        for prompt in (build_table_judge_prompt(BODY_TABLE), build_blind_cell_prompt(["R1C1"])):
            assert "{{CELL_REF_GRAMMAR}}" not in prompt
            assert "{{CELL_REF_EXAMPLES}}" not in prompt

    def test_the_rule_never_mentions_a_leading_name_column(self):
        """Round 3, NEW A. A rule that talks about one is a rule that is wrong
        on every table without one, and nothing here detects one.

        Round 4, NEW 2: checked in the rule file, the examples file, the
        RESOLVER's own source, and both prompts AS SENT -- the claim was
        previously made about the fragment alone while the resolver's comments
        still used the vocabulary.
        """
        import socr.judge.table_verdict as verdict_module
        from socr.judge.table_prompt import build_table_judge_prompt
        from socr.judge.table_rung_ollama import build_blind_cell_prompt
        from socr.judge.table_verdict import load_cell_ref_examples, load_cell_ref_grammar

        surfaces = {
            "the rule": load_cell_ref_grammar(),
            "the examples": load_cell_ref_examples(),
            "the resolver's source": Path(verdict_module.__file__).read_text(encoding="utf-8"),
            "the reader prompt as sent": build_table_judge_prompt(BODY_TABLE),
            "the blind prompt as sent": build_blind_cell_prompt(["R1C1"]),
        }
        for name, text in surfaces.items():
            lowered = text.lower()
            for banned in self.BANNED_VOCABULARY:
                assert banned not in lowered, f"{name} still says {banned!r}"

    @pytest.mark.parametrize("shape", sorted(SHAPES))
    def test_the_resolver_agrees_with_the_rule_on_every_shape(self, shape):
        """Over a stubbed table, a table with no leading name column, and a
        multi-row spanning header. A model reading the rule and socr resolving
        the same reference must land on the SAME physical cell."""
        table, cells = self.SHAPES[shape]
        resolved = resolve_cell_refs(table, list(cells))
        assert resolved is not None, f"{shape}: every reference must resolve"
        assert {str(k): v for k, v in resolved.items()} == cells

    def test_every_worked_example_is_self_describing_and_true(self):
        """The loop closer, round 4 edition.

        Each example cell holds its own reference, so the check is exact and
        needs no second copy of the expected values: resolving every reference
        in every example table must return the reference itself. An example
        that named the wrong cell -- the round-3 off-by-one, say -- fails here.
        """
        from socr.judge.table_verdict import load_cell_ref_examples

        tables = _markdown_tables(load_cell_ref_examples())
        assert len(tables) >= 3, "the examples must cover more than one shape"
        for table in tables:
            refs = _refs_in(table)
            assert refs, "each example table must be filled with references"
            resolved = resolve_cell_refs(table, refs)
            assert resolved is not None
            assert {str(k): v for k, v in resolved.items()} == {r: r for r in refs}

    def test_the_leftmost_column_is_c1_for_headers_and_bodies_alike(self):
        """The round-3 disagreement, stated on its own so a regression names
        itself. On a table with no leading name column the two families must
        not diverge."""
        table, _cells = self.SHAPES["stubless"]
        resolved = resolve_cell_refs(table, ["H1C1", "R1C1"])
        assert {str(k): v for k, v in resolved.items()} == {"H1C1": "Alpha", "R1C1": "11"}


def _markdown_tables(text: str) -> list[str]:
    """Every contiguous markdown table in ``text``, as its own block."""
    tables, current = [], []
    for line in text.splitlines():
        if line.strip().startswith("|"):
            current.append(line.strip())
        elif current:
            tables.append("\n".join(current) + "\n")
            current = []
    if current:
        tables.append("\n".join(current) + "\n")
    return tables


def _refs_in(table: str) -> list[str]:
    """The canonical references a self-describing example table is filled with."""
    return sorted(set(re.findall(r"\b[RH]\d+C\d+\b", table)))


#: The canonical reference grammar, as a matcher. Used by the blind-prompt
#: guards below to find coordinates in a prompt without assuming anything about
#: how a leak might be spelled.
_REF_PATTERN = re.compile(r"[RH]\d+C\d+")


class TestTheBlindPromptCarriesNoAnswers:
    """Cold review rounds 4, 5 and 6 — the same blocking defect, three times.

    A blind reader's entire value is that it has seen nothing but the image. A
    prompt that tells it what any coordinate contains destroys that: a
    text-only model with an answer key in front of it returns a schema-valid
    match without looking at a pixel, and the gate publishes the table as
    ``verified_by_blind_cell_transcription``.

    Round 4 found the leak in the shared rule's PROSE example
    (``R1C2 is 11``). Round 5 found it again in the blind template's JSON
    output-format example (``{"R1C2": "1.24"}``). Round 6 found that the
    guards written for those two were still lexical, and that a NON-numeric
    binding in a fourth spelling (``for R1C2 write N/A``) passed all four of
    them and published a wrong table.

    So the invariant is STRUCTURAL now. The prompt is policy plus a generated
    request list, and:

    1. the POLICY contains no concrete coordinate at all -- placeholders like
       ``<ref>`` are fine, a real reference is not;
    2. concrete coordinates appear ONLY in the request list, which carries
       coordinates and nothing else.

    Any binding of a coordinate to a value has to name a concrete coordinate,
    so it cannot exist without breaking (1) -- whatever syntax it is written
    in, including one nobody has thought of. The lexical checks are kept below
    as named regressions, not as the defence.
    """

    def _prompt(self, refs=("R1C2", "R2C3")):
        from socr.judge.table_rung_ollama import build_blind_cell_prompt

        return list(refs), build_blind_cell_prompt(list(refs))

    # -- the structural invariant ------------------------------------------

    def test_the_policy_half_contains_no_concrete_coordinate(self):
        """(1). The half that is the same for every table names no cell, so it
        can bind none."""
        from socr.judge.table_rung_ollama import split_blind_cell_prompt

        _requested, prompt = self._prompt()
        policy, _request_list = split_blind_cell_prompt(prompt)
        found = _REF_PATTERN.findall(policy)
        assert found == [], f"the blind prompt's policy text names cells: {sorted(set(found))}"

    def test_the_request_list_carries_coordinates_and_nothing_else(self):
        """(2). The generated half names the cells, and says nothing about
        them."""
        from socr.judge.table_rung_ollama import REQUEST_LIST_HEADING, split_blind_cell_prompt

        requested, prompt = self._prompt()
        _policy, request_list = split_blind_cell_prompt(prompt)
        assert set(_REF_PATTERN.findall(request_list)) == set(requested)
        remainder = _REF_PATTERN.sub("", request_list.replace(REQUEST_LIST_HEADING, ""))
        assert remainder.strip(" ,\n") == "", (
            f"the request list carries more than coordinates: {remainder!r}"
        )

    def test_the_split_is_a_property_of_the_prompt_not_of_the_builder(self):
        """The boundary must be recoverable from the text alone, so a guard can
        assert the invariant on what actually went out rather than on what a
        builder claims it produced."""
        from socr.judge.table_rung_ollama import (
            build_blind_cell_prompt_parts,
            split_blind_cell_prompt,
        )

        requested, prompt = self._prompt()
        split_policy, split_request = split_blind_cell_prompt(prompt)
        built_policy, built_request = build_blind_cell_prompt_parts(requested)
        # Modulo the join whitespace, the two halves recovered from the text
        # are the two halves the builder made.
        assert split_policy.strip() == built_policy.strip()
        assert split_request.strip() == built_request.strip()
        with pytest.raises(ValueError):
            split_blind_cell_prompt("a prompt with no request list at all")

    def test_a_binding_in_any_spelling_breaks_the_invariant(self):
        """The round-6 exploit and its relatives, as a table.

        Each of these passed every lexical guard. Each names a concrete
        coordinate, so each breaks property (1) -- which is the point of
        stating the invariant structurally instead of enumerating syntaxes.
        """
        from socr.judge.table_rung_ollama import split_blind_cell_prompt

        spellings = [
            "for R1C2 write N/A",
            "R1C2: N/A",
            "| R1C2 | N/A |",
            "`R1C2` is `11`",
            '{"R1C2": "1.24"}',
            "R1C2 usually contains a dash",
        ]
        _requested, clean = self._prompt()
        clean_policy, _ = split_blind_cell_prompt(clean)
        assert _REF_PATTERN.findall(clean_policy) == []
        for spelling in spellings:
            policy, _request = split_blind_cell_prompt(
                clean.replace("Rules:", spelling + "\n\nRules:")
            )
            assert _REF_PATTERN.findall(policy), f"{spelling!r} must break the invariant"

    # -- named regressions, kept as belt and braces -------------------------

    def test_stripping_the_requested_coordinates_leaves_no_digit(self):
        """Round 5. Every plausible numeric cell value is a digit string, so a
        prompt whose only digits are the coordinates it was handed cannot state
        a numeric value by any means."""
        _requested, prompt = self._prompt()
        stripped = _REF_PATTERN.sub("", prompt)
        digits = sorted({character for character in stripped if character.isdigit()})
        assert digits == [], f"the blind prompt carries non-coordinate digits: {digits}"

    def test_no_json_pair_is_keyed_by_a_coordinate(self):
        """Round 5's exact residual: ``{"R1C2": "1.24", ...}``."""
        _requested, prompt = self._prompt()
        pairs = re.findall(r'"([^"]*)"\s*:\s*(?:"([^"]*)"|null|[^,}\s]+)', prompt)
        bound = [(key, value) for key, value in pairs if _REF_PATTERN.fullmatch(key)]
        assert bound == [], f"the blind prompt binds coordinates to values: {bound}"

    def test_no_prose_sentence_states_what_a_coordinate_holds(self):
        """Round 4's exact residual: ``R1C2 is 11``."""
        _requested, prompt = self._prompt()
        stated = re.findall(r"`([RH]\d+C\d+)` is `([^`]*)`", prompt)
        assert stated == [], f"the blind prompt hands over answers: {stated}"

    def test_the_worked_examples_reach_the_reader_prompt_only(self):
        from socr.judge.table_prompt import build_table_judge_prompt
        from socr.judge.table_rung_ollama import build_blind_cell_prompt
        from socr.judge.table_verdict import load_cell_ref_examples

        examples = load_cell_ref_examples()
        assert examples in build_table_judge_prompt(BODY_TABLE)
        blind = build_blind_cell_prompt(["R1C1"])
        assert examples not in blind
        for line in examples.splitlines():
            if line.strip().startswith("|"):
                assert line.strip() not in blind

    def test_the_rule_itself_contains_no_digits(self):
        """The rule goes to both prompts, so it carries the same obligation."""
        from socr.judge.table_verdict import load_cell_ref_grammar

        digits = [c for c in load_cell_ref_grammar() if c.isdigit()]
        assert digits == [], f"the rule carries digits: {digits}"

    def test_the_blank_cell_protocol_survives_all_of_this(self):
        """The control. ``""`` and ``null`` are the two answers the guard
        depends on being distinguishable, so the rules that establish them must
        still be in the prompt -- the fixes must not have removed the protocol
        along with the answer key."""
        _requested, prompt = self._prompt()
        assert '`""`' in prompt
        assert "`null`" in prompt
