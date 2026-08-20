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

---

## Settled by measurement: there is no grounding signal on a refused attempt

Round 5 was rejected on a **fifth** container class — raw HTML blocks (`<pre>`, `<div>`,
`<code>`, `<script>`, `<style>`, `<blockquote>`). Five rounds, five container types. I
forbade a sixth strip and asked for one measurement instead.

**Hypothesis.** Restrict the D3 keep path to `REJECTION_AMBIGUOUS_DEFERRED`, refusing
`judge_only` and empty. That disposition is written when the verifier ran against native
words and deferred, so it should be positive evidence of provenance — already on the
attempt, already serialized, no new plumbing. A `<pre>`-wrapped phantom cannot have been
geometrically compared and deferred on, because nothing on the page matches it.

**FALSIFIED, with a real judge and a single case.** Page: 3 rows × 2 numeric lanes of real
values. Output: a 5-column grid whose numbers (`9.91`, `9.92`, …) appear **nowhere on the
page**, wrapped in `<pre>`. Observed `rejection_class`: **`ambiguous_deferred`**. A phantom
maximally unrelated to the page earns the disposition the rule would have treated as proof
of provenance.

**Mechanism, and it is the same bug a fourth time.** `ambiguous_deferred` is set on the
`vr.warn` branch, and `vr.warn` fires iff `col_gap = abs(output_col_count -
native_lane_count) >= 2`. `output_col_count` comes from `_parse_output_col_count` — a
**third** text-only markdown scanner, with exactly the same HTML/fence/comment blindness as
`find_table_blocks` and `check_markdown`. The disposition never certifies that the text
reads the page; it is a text-derived number wearing a geometric costume.

Since this was the strongest grounding signal available on an attempt today — every other
field (`audit_passed`, `status`, `failure_mode`, `engine`) carries strictly less — nothing
currently recorded distinguishes *"a grid that reads this page"* from *"a grid that appears
in this text."* The design note's claim is now measured rather than argued.

## The finding that resizes #268

The code owner corrected its own earlier estimate, which had #268 as "persist a fact the
verifier already computes."

The verifier's only POSITIVE corroboration state is `VerifierState.EXACT_PASS` — row counts
match, no label-binding problem, multisets clean. But `EXACT_PASS` returns
`AcceptDecision(accept=True)` (`agentic.py:655-671`), so an EXACT_PASS attempt is
**accepted** and never reaches the D3 keep path — which by construction only ever sees
attempts that **every rung refused**.

**Among refused attempts there is no corroboration signal at all** — only degrees of "not
refuted". `CERTAIN_FAIL` says refuted; `AMBIGUOUS` says undecided; neither says
corroborated.

So #268 cannot persist an existing fact. It must **compute a new one**: a positive,
quantitative corroboration measure that survives refusal — for instance, how many of the
emitted grid's numeric rows matched native word-geometry rows, carried on the attempt even
when the judge refuses. That is the signal a `<pre>`-wrapped phantom cannot fake, because
its numbers are not on the page. `_value_guard` already performs the per-row pairing; what
is missing is retaining and surfacing the result on the refusal path.

## Outcome

**PR #264: PARK.** Not merged, not closed, CI green at `887f17e`. The policy layer beneath
the predicate has been verified by five independent models and should be rescued via the
split, not thrown away. Round 5's strips and its 51-case corpus move to #268 — they are
correct as far as they go and load-bearing today.

Five rounds, five reviewers, five models, zero phantoms shipped to `main` from this PR. The
gate cost five rounds and prevented five distinct silent-corruption paths from landing.
