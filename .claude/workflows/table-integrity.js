export const meta = {
  name: 'table-integrity',
  description:
    'Settle and ship the table-structure integrity work: GPT/Grok/Opus design the escalation predicate and header-attribution check in parallel, Opus adjudicates, implementation runs strictly serially in the one checkout, then all three models review the combined diff adversarially.',
  whenToUse:
    'Run after the 2026-08-15 TR-3 hand judgement (docs/log/2026-08-15_tr3-hand-judgement.md). Settles PR #200 and ships #211, the two unfiled bugs, and wave 3b. Pass {skipDesign:true} in args to jump straight to implementation using an already-ratified spec.',
  phases: [
    { title: 'Design', detail: 'gpt-sol, grok and opus each specify the predicate independently — read-only' },
    { title: 'Adjudicate', detail: 'opus picks/synthesizes one spec and writes it to docs/log/' },
    { title: 'Implement', detail: 'STRICTLY SERIAL — one branch at a time in the single checkout' },
    { title: 'Review', detail: 'gpt-sol, grok and opus adversarially review each branch diff' },
  ],
}

// ---------------------------------------------------------------------------
// WHY THIS SHAPE
//
// socr's editable install resolves `import socr` to the MAIN tree. A git
// worktree would therefore test main's source, not its own — false greens.
// So: isolation:'worktree' is FORBIDDEN here, there is exactly ONE checkout,
// and every file-writing agent MUST run serially. Parallelism lives in the
// read-only design and review panels, where it is safe and actually useful.
//
// The doer never grades its own work: implementers do not review, reviewers
// do not edit.
// ---------------------------------------------------------------------------

const HOUSE_RULES = `
NON-NEGOTIABLE REPO RULES (violating any of these fails the task):
- NEVER work on main. Branch first. NEVER push. NEVER merge. NEVER open a PR.
- Stage files BY NAME. Never \`git add -A\` or \`git add .\`.
- NEVER add a Co-Authored-By trailer.
- Python runs via \`~/venvs/socr/bin/{python,pytest}\` or \`uv run\`. Never \`python file.py\`.
- Tests: \`~/venvs/socr/bin/pytest <paths> -q\`.
- BLOCKING lint gate, run it EXACTLY like CI: \`uvx ruff@0.16.0 format --check .\`
  Do NOT use ~/venvs/socr/bin/ruff for the format gate — it is an older ruff that
  reports clean on files CI rejects. This exact gap turned main red and blocked four PRs.
- CI has NO ollama and NO provider. Any test driving _phase_agentic / process() in
  agentic mode MUST patch _available_engines_for_agentic (e.g. return [PROFILE_QWEN_LOCAL]
  from socr.core.providers) or it passes locally and fails in CI.
- NO magic numbers or hardcoded thresholds, in code or prompts. Derive from data, or use a
  named documented constant.
- NO silent content loss. A failure must surface at page status, document status, metadata
  AND the CLI — not just one of them.
- --native-only means the ladder is OFF: record and surface a structural signal, never let it
  reroute. This ruling is settled; do not relitigate it.
- orphan_rows stays out of every gate (fires on 69.9% of blocks and is a legitimate
  standard-error row). No majority-vote predicates.
- This is a PUBLIC repo. NEVER commit corpus PDFs or page extracts — they are copyrighted.
  Use generated look-alike fixtures.
- Do NOT use a git worktree. The editable install resolves to the main tree; a worktree
  produces false green tests.
`

const EVIDENCE_FLOOR = `
GROUNDING REQUIREMENT: every claim you make must cite a file:line you actually read, or a
doc path, or a command you actually ran with its real output. If you could not read
something, say so explicitly. Do NOT invent file paths, line numbers, test names or
measurements. An answer with fabricated anchors is worse than no answer.
`

const CONTEXT = `
## The measurement that triggers this work

docs/log/2026-08-15_tr3-hand-judgement.md — READ IT FIRST.

The owner hand-judged TR-3's firings (rendered page vs socr markdown, side by side).
Stopped at 7 of 34 because the result was unambiguous:
  - 4 of 4 REAL TABLES were damaged, each with several independent defects.
  - 3 of 7 firings were NOT TABLES AT ALL (two book back-matter indexes, one figure).

Defects seen, and whether TR-3 (a numeric-token multiset check) can detect them:
  header band destroyed / detached from columns  4/4  NO  (no digits change)
  row labels detached or off-by-one              3/4  NO
  whole row deleted, cells only significance stars 1/4 NO  (row has no numerals)
  coefficient row torn across two output rows    2/4  yes
  content re-emitted as duplicate loose text     1/4  no

On all four damaged pages EVERY NUMERAL WAS CORRECT. Not one was reliably attributable
to both its row and its column. That is the failure mode that matters in a citation corpus.

## The panel ruling (2026-08-15, codex gpt-5.6-sol + gemini antigravity)

Archived at ~/.local/share/consilium/runs/2026/08/20260815T003453Z-3777/

1. The prior plan — swap B1's predicate for TR-3 — is DEAD. TR-3 has ~57% precision as a
   table signal and is blind to 3 of the 5 observed defect classes.
2. SHIP #200's plumbing as an escalation gate, KEEPING B1's own predicate
   (ragged OR detached_label_rows). Gemini: "B1 isn't over-firing; it is accurately
   identifying the exact complexity envelope that the native parser is incapable of
   safely reconstructing."
3. ADD a header-attribution term. Gemini's cheapest formulation: for every numeric data
   column in the body, its bounding x-interval must intersect the x-interval of at least one
   non-empty cell in the header band. Named falsifiers: implied/spacer columns with empty
   header cells, header-less two-column indexes, severe spatial skew.
4. The two signals are NOT NESTED: TR-3 misses 31 of the 66 pages B1 flags and catches 27
   B1 misses. The final predicate may legitimately be a DISJUNCTION. Do not assume one
   subsumes the other.
5. VERIFIED CODE FACT (checked against the tree, not an opinion):
   native_verifier.py:1055-1058 sets EXACT_PASS on (no hard_fail AND no warn AND no
   row_count_warn) — all numeric-multiset conditions. agentic.py:595-612 then returns
   AcceptDecision(accept=True, confidence=1.0) WITHOUT EVER CALLING the inner visual judge.
   grep finds zero header checks in native_verifier.py and no structural check of any kind
   on agentic.py's accept path. So a MODEL-produced table with correct numbers and a
   destroyed header ships at confidence 1.0 today. The winner-side check is therefore a hole
   in the acceptance path, not an optional follow-up.

## What is NOT established (do not overclaim)

The 62 TR-3 pages are a SELECTED sample. "When a native table page fires TR-3 the table is
broken" is supported. "Native table reconstruction is always broken" is NOT measured. The
base rate over all 245 native-table pages is unknown.
`

const SPEC_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['predicate', 'headerCheck', 'winnerSide', 'nativeOnly', 'risks', 'acceptanceTests', 'evidence', 'confidence'],
  properties: {
    predicate: {
      type: 'string',
      description:
        'The exact escalation predicate to ship, as pseudocode or a Python expression. Say explicitly whether it is a disjunction with TR-3 and why.',
    },
    headerCheck: {
      type: 'string',
      description:
        'The cheapest SOUND header-attribution check, precise enough to implement. State its inputs (geometry? markdown? both), where it runs, and what it does on an unverifiable case.',
    },
    headerCheckFalsifier: {
      type: 'string',
      description: 'The concrete case that would prove this header check wrong, and how to test for it.',
    },
    winnerSide: {
      type: 'string',
      description:
        'Where exactly the structural check must run on an ACCEPTED candidate (native or model) to close the EXACT_PASS hole. Give file:line.',
    },
    nativeOnly: {
      type: 'string',
      description:
        'How the gate behaves under --native-only, honouring "record and surface, never reroute".',
    },
    failClosed: {
      type: 'string',
      description:
        'What ships when the gate fires at the TOP rung and there is nowhere left to escalate. The D3 floor (manifest.py:303-333, marker + PNG ref, audit_passed=False) is the existing precedent.',
    },
    risks: { type: 'array', items: { type: 'string' } },
    acceptanceTests: {
      type: 'array',
      description: 'Concrete test cases with expected outcomes. Must be hermetic — no ollama, no provider.',
      items: { type: 'string' },
    },
    evidence: {
      type: 'array',
      description: 'file:line anchors or doc paths you ACTUALLY read. MUST be non-empty.',
      items: { type: 'string' },
    },
    confidence: { type: 'string', enum: ['low', 'medium', 'high'] },
  },
}

const REVIEW_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['verdict', 'findings', 'evidence'],
  properties: {
    verdict: { type: 'string', enum: ['ACCEPT', 'ACCEPT_WITH_NITS', 'REJECT'] },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['severity', 'file', 'summary', 'failureScenario'],
        properties: {
          severity: { type: 'string', enum: ['blocker', 'major', 'minor'] },
          file: { type: 'string' },
          line: { type: 'integer' },
          summary: { type: 'string' },
          failureScenario: {
            type: 'string',
            description: 'Concrete inputs/state -> wrong output. Not a style opinion.',
          },
        },
      },
    },
    evidence: { type: 'array', items: { type: 'string' } },
  },
}

const PANEL = [
  { key: 'gpt', agentType: 'gpt-sol' },
  { key: 'grok', agentType: 'grok' },
  { key: 'opus', agentType: undefined }, // inherits the session model (Opus)
]

// ===========================================================================
// PHASE 1 — Design (parallel, READ-ONLY)
// ===========================================================================

const skipDesign = Boolean(args && args.skipDesign)
let spec = null

if (!skipDesign) {
  phase('Design')
  log('Three models specify the escalation predicate independently. Read-only — no file writes.')

  const proposals = (
    await parallel(
      PANEL.map((m) => () =>
        agent(
          `You are specifying a change to socr, a multi-engine document OCR pipeline for an academic
citation corpus. In this corpus a WRONG or UNATTRIBUTABLE number is far worse than a missing one.

${CONTEXT}

${EVIDENCE_FLOOR}

READ-ONLY TASK. Do not edit, create or delete any file. Do not run git. Read the repo and answer.

Read at minimum:
  docs/log/2026-08-15_tr3-hand-judgement.md
  docs/log/2026-08-14_gh151-b1-escalation-decision.md
  docs/log/2026-08-14_gh151-b1-predicate-design.md
  src/socr/tables/native_verifier.py  (especially the EXACT_PASS decision ~line 1040-1060)
  src/socr/pipeline/agentic.py        (especially the accept path ~line 530-615)
  src/socr/core/manifest.py           (the D3 fail-closed floor ~line 300-335)
  the branch feat/151-b1-structural-gate, diffed against main

${HOUSE_RULES}

Specify the change. Be concrete enough that an implementer needs no further decisions.
Where the panel ruling above is wrong, say so and give the file:line that proves it —
disagreeing with it is allowed and valuable, but only with evidence.`,
          { label: `design:${m.key}`, phase: 'Design', agentType: m.agentType, schema: SPEC_SCHEMA },
        ),
      ),
    )
  ).filter(Boolean)

  if (!proposals.length) throw new Error('Design panel returned nothing — aborting before any write.')
  log(`${proposals.length}/3 design proposals returned.`)

  // =========================================================================
  // PHASE 2 — Adjudicate (single writer)
  // =========================================================================

  phase('Adjudicate')

  spec = await agent(
    `Three models independently specified the same change to socr. Adjudicate and produce ONE spec.

${CONTEXT}

THE THREE PROPOSALS (JSON):
${JSON.stringify(proposals, null, 2)}

${EVIDENCE_FLOOR}

Your job:
1. Verify each proposal's load-bearing claims against the actual code. A proposal citing a
   file:line that does not say what it claims is disqualified on that point — check, do not trust.
2. Where they agree, say so once. Where they conflict, resolve it by evidence and severity,
   NOT by vote. If the conflict cannot be resolved from the code, name it as an OPEN DECISION
   for the owner rather than inventing closure.
3. Produce one implementable spec covering: the escalation predicate, the header-attribution
   check, where the winner-side check runs, --native-only behaviour, and fail-closed semantics
   at the top rung.

${HOUSE_RULES}

WRITE your adjudication to docs/log/2026-08-15_200-shipping-spec.md on the EXISTING branch
docs/200-tr3-hand-judgement (it is already checked out; do NOT create a new branch and do NOT
touch main). Run \`uvx ruff@0.16.0 format --check .\` and fix anything it flags. Commit with
\`docs(200): ratify the shipping spec for the structural escalation gate\`, staging that one
file BY NAME. Do not push.

Return the spec as structured output as well.`,
    { label: 'adjudicate', phase: 'Adjudicate', schema: SPEC_SCHEMA },
  )

  log('Spec ratified and committed to docs/200-tr3-hand-judgement.')
}

// ===========================================================================
// PHASE 3 — Implement (STRICTLY SERIAL — one checkout, one branch at a time)
// ===========================================================================

phase('Implement')
log('Serial by necessity: one checkout, editable install, no worktrees.')

const SPEC_BLOCK = spec
  ? `\n\nTHE RATIFIED SPEC (also committed at docs/log/2026-08-15_200-shipping-spec.md):\n${JSON.stringify(spec, null, 2)}\n`
  : '\n\nRead the ratified spec at docs/log/2026-08-15_200-shipping-spec.md.\n'

// `reviewers` is per-ticket on purpose: the two #200 branches carry the correctness
// boundary and get two independent models each; the bounded tickets get one. This keeps the
// whole run at 14 agents (3 design + 1 adjudicate + 4 implement + 6 review).
const TICKETS = [
  {
    key: 'T1-docs-and-issues',
    agentType: undefined, // Opus
    reviewers: ['gpt'],
    branch: 'docs/200-tr3-hand-judgement (EXISTING — check it out, do not create it)',
    task: `File the two bugs the hand judgement surfaced. Both are UNFILED.

BUG A — an accepted model table is never structurally checked.
  native_verifier.py:1055-1058 sets EXACT_PASS on purely numeric-multiset conditions;
  agentic.py:595-612 then returns AcceptDecision(accept=True, confidence=1.0) without ever
  calling the inner visual judge. There are zero header checks in native_verifier.py. So a
  model-produced table with correct numbers and a destroyed header ships at full confidence.
  Title it: \`bug(agentic): EXACT_PASS accepts a model table with no structural check\`

BUG B — non-tables reach table reconstruction.
  3 of 7 TR-3 firings were not tables: two book back-matter indexes and one figure. A
  two-column index is short dense lines with page numbers trailing each line — that right-hand
  run of page numbers is a numeric lane, so a numeric-lane geometry check fires by construction.
  Title it: \`bug(tables): book indexes and figures are routed to table reconstruction\`

For each: search existing issues first (\`gh issue list\`, \`gh search issues\`) and if a
near-duplicate exists, cite it in the body so GitHub backlinks rather than filing a duplicate.
Give concrete inputs/state, actual result, expected result, and the smallest safe reproduction.
Cite docs/log/2026-08-15_tr3-hand-judgement.md as the evidence.
NEVER paste corpus page content into a public issue — describe the shape, do not reproduce it.

Then append a short "Filed" section to docs/log/2026-08-15_tr3-hand-judgement.md listing both
issue numbers, and commit that one file by name.`,
  },
  {
    key: 'T2-issue-211',
    agentType: 'gpt-sol',
    reviewers: ['opus'],
    branch: 'fix/211-native-only-table-status',
    task: `Fix issue #211. Read it first with \`gh issue view 211\`.

Under --native-only, table pages take the free native path and are appended to page_outputs
with a HARDCODED PageStatus.SUCCESS and audit_passed=True (orchestrator.py:911-919), and
nothing is appended to ps.attempts. Because _score_per_page opens with
\`if not page_state.attempts: continue\` (orchestrator.py:3679), the native table structure
gate never runs on them — so a table this pipeline is known to damage ships labelled clean.

The settled ruling: under --native-only the ladder is OFF — RECORD AND SURFACE, NEVER REROUTE.
So the fix is about status and surfacing, not routing. Do not make it escalate.

The failure must surface at page status, document status, metadata AND the CLI.
Add hermetic regression tests. Do NOT assert against a bare MagicMock — \`!=\` and \`not in\`
against one always pass and would make the guard vacuous.`,
  },
  {
    key: 'T3-escalation-gate',
    agentType: undefined, // Opus — most design-sensitive
    reviewers: ['gpt', 'grok'],
    branch: 'feat/200-structural-escalation-gate',
    task: `Ship #200 as an escalation gate per the ratified spec.

Start from the existing branch feat/151-b1-structural-gate — its wiring (PageState field,
audit event, manifest locks, TABLE_DISTRUST_KINDS membership, the shared structural_gate_fires
entry point) is already reviewed and sound. Keep the plumbing; change what it DOES.

Keep B1's predicate (ragged OR detached_label_rows). Do NOT swap in TR-3 — that plan is dead
(TR-3 is blind to header loss, detached labels, and star-only row deletion). If the spec calls
for a disjunction WITH TR-3, implement it as a disjunction, not a replacement.

Also close the winner-side hole: the structural check must run on whatever is ABOUT TO SHIP,
native or model, not only on the native attempt. See the EXACT_PASS path above.

--native-only: record and surface, never reroute. At the top rung with nowhere left to
escalate: fail closed following the existing D3 precedent (manifest.py:303-333 — marker +
PNG ref, audit_passed=False). NOT needs_repair (barred by the --native-only ruling), NOT
deletion.`,
  },
  {
    key: 'T4-header-attribution',
    agentType: 'grok',
    reviewers: ['opus', 'gpt'],
    branch: 'feat/200-header-attribution',
    task: `Add the header-attribution check per the ratified spec.

This is the term that catches the 4-of-4 defect: every numeral correct, none attributable to
a column. Branch from feat/200-structural-escalation-gate (T3) so the gate exists to hang it on.

Gemini's cheapest formulation, which the spec may have revised — follow the SPEC where they
differ: for every numeric data column in the body, its bounding x-interval must intersect the
x-interval of at least one non-empty cell in the header band.

Handle the named falsifiers explicitly, each with a test:
  - implied/spacer columns with a legitimately empty header cell
  - header-less two-column indexes (failing these is desirable, but be deliberate about it)
  - severe spatial skew, where a column's centre of mass shifts but it is logically aligned
  - multi-level and spanning headers, which are COMMON in this corpus and must not be failed

An unverifiable case must fail CLOSED and be surfaced, never silently passed.
No magic numbers — if you need a band size or tolerance, derive it from the page's own
geometry or make it a named, documented constant with its justification in the docstring.`,
  },
]

const implemented = []
for (const t of TICKETS) {
  const r = await agent(
    `Implement ONE ticket in the socr repo, end to end: code, tests, lint, commit.

${CONTEXT}
${SPEC_BLOCK}
${HOUSE_RULES}

${EVIDENCE_FLOOR}

BRANCH: ${t.branch}
Create it from main unless the ticket says otherwise. Never work on main.

YOUR TICKET:
${t.task}

DONE MEANS ALL OF:
  1. \`~/venvs/socr/bin/pytest <the paths you touched> -q\` passes, and you paste the real count.
  2. \`uvx ruff@0.16.0 format --check .\` reports clean. Use EXACTLY this command.
  3. Every new test is hermetic — no ollama, no provider. If it drives agentic mode, it patches
     _available_engines_for_agentic.
  4. One commit, Conventional Commits style, files staged BY NAME.
  5. Not pushed, not merged, no PR opened.

If you cannot finish, commit what genuinely works, and say plainly what is incomplete and why.
Do NOT report success on partial work — a false green here is worse than an honest stop.`,
    { label: t.key, phase: 'Implement', agentType: t.agentType ?? 'socr-implementer' },
  )
  implemented.push({ ticket: t.key, branch: t.branch, reviewers: t.reviewers, report: r })
  log(`${t.key} done (${implemented.length}/${TICKETS.length}).`)
}

// ===========================================================================
// PHASE 4 — Review (parallel, READ-ONLY — the doer never grades its own work)
// ===========================================================================

phase('Review')

const reviews = await pipeline(
  implemented.filter((x) => x.report),
  (item) =>
    parallel(
      PANEL.filter((m) => (item.reviewers || ['opus']).includes(m.key)).map((m) => () =>
        agent(
          `Adversarially review ONE completed socr ticket. Your job is to FIND FAILURES, not to
approve. Default to skepticism; a finding you cannot ground in a file:line is not a finding.

${CONTEXT}
${HOUSE_RULES}
${EVIDENCE_FLOOR}

TICKET: ${item.ticket}
BRANCH: ${item.branch}
THE IMPLEMENTER REPORTED:
${String(item.report).slice(0, 6000)}

READ-ONLY. Do not edit anything. Diff the branch against main and read the actual code.

Check specifically, because these have bitten this repo before:
  - Does any new test assert against a bare MagicMock? \`!=\` and \`not in\` against one ALWAYS
    pass, which silently neuters exactly the regression guard that matters.
  - Would any agentic-mode test pass locally (ollama installed) but fail in CI (no provider)
    because it does not patch _available_engines_for_agentic?
  - Does \`uvx ruff@0.16.0 format --check .\` actually pass? Run it yourself; do not take the
    implementer's word.
  - Any magic number or hardcoded threshold introduced in code or a prompt?
  - Does a failure surface at page status, document status, metadata AND the CLI — or only one?
  - Does anything reroute under --native-only? That is forbidden.
  - Does the final assembled .md stay byte-identical to whole-doc assembly? There are golden
    tests guarding this.
  - Did the implementer's claimed test counts and commands actually run, with that result?

Report only findings you verified. An empty findings list with verdict ACCEPT is a valid and
useful answer if the diff is genuinely sound.`,
          { label: `review:${m.key}:${item.ticket}`, phase: 'Review', agentType: m.agentType, schema: REVIEW_SCHEMA },
        ),
      ),
    ).then((vs) => ({ ticket: item.ticket, branch: item.branch, reviews: vs.filter(Boolean) })),
)

const summary = reviews.filter(Boolean).map((r) => {
  const all = r.reviews.flatMap((v) => v.findings || [])
  const blockers = all.filter((f) => f.severity === 'blocker')
  const rejects = r.reviews.filter((v) => v.verdict === 'REJECT').length
  return {
    ticket: r.ticket,
    branch: r.branch,
    verdicts: r.reviews.map((v) => v.verdict),
    rejectCount: rejects,
    blockers,
    majors: all.filter((f) => f.severity === 'major'),
  }
})

const needsWork = summary.filter((s) => s.rejectCount > 0 || s.blockers.length > 0)

log(
  needsWork.length
    ? `${needsWork.length}/${summary.length} branches have blockers or a REJECT — owner decision needed.`
    : `All ${summary.length} branches reviewed clean. Nothing merged — the owner merges after CI green.`,
)

return {
  spec,
  implemented: implemented.map((x) => ({ ticket: x.ticket, branch: x.branch })),
  review: summary,
  needsWork: needsWork.map((s) => s.ticket),
  nextHumanAction:
    'Hand-judge ~20 pages where the B1 shape predicate fires and TR-3 does not. That converts B1 26.9% from a firing rate into a defect rate and is the only step no agent can do. Then merge the clean branches after CI green.',
  notMerged: 'Nothing was pushed, merged or opened as a PR. Every branch is local.',
}
