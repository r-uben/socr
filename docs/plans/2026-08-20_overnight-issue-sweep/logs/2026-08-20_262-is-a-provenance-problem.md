# #262 is a provenance problem wearing a syntax costume — and main already has the hole

2026-08-20, after four rejected rounds on PR #264.

## The live defect, verified against `origin/main`

The merged #259 keep-predicate (`manifest.flagged_model_page_output`, merged at `653728a`)
decides "did an attempt author a grid?" via `reconcile.find_table_blocks`. Run against
`origin/main` today:

    fenced code sample     find_table_blocks -> 1 block(s)  => "authored a grid" = True
    pipe-bearing prose     find_table_blocks -> 1 block(s)  => "authored a grid" = True

So a model attempt containing a fenced code sample, or two lines of prose with pipes in
them, already counts as an authored table **in merged code**, and can supersede the native
reading. Nobody reviewed #259 for this, because nobody was looking for it yet.

**This is not a defect introduced by #264. #264 is where it was found.**

## Why four rounds failed

Rounds 1-4 each tightened a markdown predicate and each was defeated by a reviewer within
the hour. Round 4 was explicitly a positive shape check that "fails closed on a shape
nobody anticipated"; MiniMax falsified that with three inputs.

The code owner's diagnosis, which is the useful part:

> The failure class is not syntax, it is provenance. The three new phantoms are REAL grids
> that are not readings of the page, and no amount of markdown shape-checking can see that,
> because the shape is genuinely there.

A table inside a code fence, inside an indented code block, inside an HTML comment, or
headed by a prose fragment is *syntactically perfect markdown*. A parser cannot reject it
without ceasing to be a parser. The question being asked — "is this text a reading of THIS
page?" — is not answerable from the text.

The author retracted its own round-4 framing rather than defending it, and accepted
MiniMax's correction that its "two of three" claim had collapsed two orderings.

## Why the obvious remedy is not obvious

"Record it upstream instead of re-deriving it" is the #260 answer (`rejection_class`), and
it is directionally right. But the extraction side does not currently hold a phantom-proof
notion of "authored a grid" either — it holds **three** text-only parsers with the same
blind spot:

- `reconcile.find_table_blocks` (loose by design, documented as such)
- `native_verifier._parse_output_col_count` (`native_verifier.py:335`)
- `structure_check.check_markdown`, which delegates to `find_table_blocks`

So persisting a flag computed by any of them persists the same hole with better plumbing.
What separates a reading from a phantom is **geometry** — the native words on the page —
and that is in scope in the verifier and discarded before assemble sees it.

## Cost, as estimated by the code owner

Same shape as #260: a schema field, one write site
(`NativeTableVerifierJudge.assess`, where `vr` and `words` are both in scope — where
`rejection_class` is already written), **two** read sites
(`d3_floor_kept_model_output` for #262 **and** `flagged_model_page_output` for #259 —
both, or the fix is half-applied), serialization through `PageOutput.to_dict/from_dict`
and the page sidecar, absent field defaulting to "not grounded" → marker, which is
fail-closed.

## What the owner has to decide

1. What counts as a GROUNDED grid? Candidate: `output_col_count >= 2` AND the verifier ran
   with native words in scope. Needs a rule for pages where the verifier legitimately
   bypasses (scans, no words): marker or grid?
2. **Is #259 in scope?** Its merged predicate has the same hole. Fixing only #262 leaves a
   known live defect on main; fixing both changes behaviour on a ruling already made.
3. Resume: sidecars written before the field exists carry no grounding and default to the
   marker, so those pages re-OCR. Fail-closed is right, but it changes resume cost on an
   existing corpus — and the run fingerprint has no source-version component, a known gap.

## Recommendation on PR #264

**Do not merge as the answer to #262. Do not close it either.** Everything below the grid
predicate has been independently verified by four models and holds: the `rejection_class`
allowlist, the call site reached exactly once, `audit_passed` never flipped, source
attempts unmutated, non-table content preserved, four-surface reporting, the reverse
guards, `find_table_blocks` byte-identical, no multiset oracle.

Park it behind the design pass and land it once the grounded fact exists, swapping
`has_strict_table_grid` for a read of the recorded verdict.

## The transferable lesson

Asking a syntax question of a provenance problem produces a bound that cannot be reached
by enumeration. Four rounds is the cost of not noticing which kind of question you are
asking. The tell was available from round 1: every rejection was a *new category* of input,
not a *tighter case* of the previous one.
