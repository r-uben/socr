export const meta = {
  name: 'design-panel',
  description: 'GPT and Fable independently propose a design for each NEEDS-DESIGN ticket; disagreements get one rebuttal round; Opus synthesizes for the owner to ratify.',
  whenToUse:
    'Run when socr has NEEDS-DESIGN tickets whose architectural fork must be settled before any implementer can be dispatched. Pass ticket ids via args (array of strings); omit for the 2026-08-09 backlog default.',
  phases: [
    { title: 'Propose', detail: 'gpt-sol and fable each design the ticket independently' },
    { title: 'Rebut', detail: 'only where the two disagree on the core fork' },
    { title: 'Chain', detail: 'GH-114-DESIGN, which depends on the GH-49B decision' },
  ],
}

// ---------------------------------------------------------------------------
// Agents are READ-ONLY and return structured proposals. Nothing is written to
// docs/log/ by the workflow — Opus synthesizes one note per ticket afterwards,
// and the owner ratifies. This keeps the doer from grading its own work and
// stops two models racing on the same file path.
// ---------------------------------------------------------------------------

const PROPOSAL_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['ticket', 'recommendation', 'rationale', 'evidence', 'risks', 'acceptanceTest', 'ownerDecision', 'confidence'],
  properties: {
    ticket: { type: 'string' },
    recommendation: {
      type: 'string',
      description: 'The fork you pick, in one sentence. Must be a concrete choice, not a summary of options.',
    },
    rationale: { type: 'string', description: 'At most 5 sentences. Why this fork over the other(s).' },
    evidence: {
      type: 'array',
      description: 'file:line anchors, commit shas, or repo doc paths you actually read. MUST be non-empty.',
      items: { type: 'string' },
    },
    priorArt: {
      type: 'string',
      description:
        'What the repo has ALREADY tried on this problem (docs/plans/, docs/log/, git history), and whether your recommendation repeats or diverges from it. Say "none found" only after looking.',
    },
    risks: { type: 'array', items: { type: 'string' } },
    acceptanceTest: {
      type: 'string',
      description: 'A proposed outsider-checkable Done when: a named command, path, or metric. No vibes.',
    },
    ownerDecision: {
      type: 'string',
      description:
        'What the human owner must decide that you CANNOT decide from the code — taste, cost tolerance, product scope. If you believe nothing is left to them, say so explicitly and justify it.',
    },
    confidence: { type: 'string', enum: ['low', 'medium', 'high'] },
  },
}

const REBUTTAL_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['ticket', 'position', 'reasoning'],
  properties: {
    ticket: { type: 'string' },
    position: { type: 'string', enum: ['hold', 'concede', 'revise'] },
    reasoning: { type: 'string', description: 'At most 4 sentences.' },
    revisedRecommendation: { type: 'string', description: 'Required if position is revise. Empty string otherwise.' },
    strongestPointAgainstMe: {
      type: 'string',
      description: 'The single best argument in the other proposal, stated fairly, even if you hold.',
    },
  },
}

const VOTERS = [
  { key: 'gpt', agentType: 'gpt-sol' },
  { key: 'fable', agentType: 'general-purpose', model: 'fable' },
]

const DEFAULT_TICKETS = ['GH-127-D-DESIGN', 'GH-56-DESIGN', 'GH-49B-DESIGN']
const CHAINED_TICKET = 'GH-114-DESIGN'

function proposalPrompt(ticket, upstream) {
  return `You are doing a READ-ONLY design pass on ONE ticket in the socr repository (multi-engine document OCR, Python, src/socr).

Ticket: ${ticket}

MANDATORY procedure:
1. Read the ticket verbatim in /Users/rubenffuertes/repos/tools/socr/docs/plans/TICKETS.md — find the section headed with this ticket id under "Open-issue backlog — 2026-08-09 reconciliation". It states the Problem and the Sharp question(s). Answer THOSE.
2. Read the code the ticket names. Actually Read the functions, do not reason from the ticket text alone.
3. Read the PRIOR ART before recommending anything: docs/plans/ (especially table-repair/ and progressive-pages/), docs/log/, and \`git log\` on the files involved. This repo has already tried and rejected approaches in this area. A recommendation that silently repeats a recorded failure is worse than no recommendation. Fill the priorArt field honestly — if your pick repeats something already tried, say so and justify why it should work this time.
4. Honour the repo's standing rules from CLAUDE.md: no magic thresholds (derive from data or use a named documented constant), no silent content loss, local model is qwen3-vl:30b-a3b-instruct.
${upstream ? `\n5. UPSTREAM DECISION — this ticket depends on ${CHAINED_TICKET === ticket ? 'GH-49B-DESIGN' : 'an upstream ticket'}, whose panel concluded:\n${upstream}\nDesign consistently with that, or explain precisely why it must be revisited.\n` : ''}
HARD RULES:
- You are a LEAF NODE. Do NOT call the Agent/Task tool. Do NOT spawn subagents or helpers. Do NOT call SendMessage or message any other agent. Your structured return value is your ONLY output channel — nothing you send anywhere else will reach anyone. Do your own reading.
- Do NOT go looking for support for a conclusion you have already picked. Form the recommendation from the code, then spend your remaining effort trying to REFUTE it. If you cannot refute it, say so in rationale.
- Do NOT modify any file. Do NOT write to docs/log/. Do NOT run git commit, gh issue edit, or any write command. Your output is the deliverable.
- Pick ONE fork. "It depends" is not a recommendation. If the choice genuinely belongs to the human, still state your best pick AND put the human's part in ownerDecision.
- evidence must be non-empty and must be things you actually opened.
- Work independently. Do not speculate about what another model would say.`
}

function rebuttalPrompt(ticket, mine, theirs) {
  return `Design panel, rebuttal round, ticket ${ticket} (socr repository).

You previously proposed:
${JSON.stringify(mine, null, 2)}

An independent reviewer, working from the same code, proposed something different:
${JSON.stringify(theirs, null, 2)}

Re-examine the code and the repo's prior art with their argument in hand. Then either:
- hold    — your recommendation stands; say why theirs fails on evidence, not on preference.
- concede — theirs is better; say what you missed.
- revise  — a third option is better than both; state it in revisedRecommendation.

You MUST fill strongestPointAgainstMe with the best argument on the other side, stated fairly, even when you hold. Changing your mind under evidence is the point of this round, not a failure.

READ-ONLY: do not modify any file or run any write command.
LEAF NODE: do not call the Agent/Task tool, do not spawn helpers, do not call SendMessage. Re-read the code yourself. Judge their argument on the evidence already in front of you — do not go recruiting support for the position you already hold.`
}

async function runTicket(ticket, upstream) {
  const proposals = (
    await parallel(
      VOTERS.map((v) => () =>
        agent(proposalPrompt(ticket, upstream), {
          agentType: v.agentType,
          ...(v.model ? { model: v.model } : {}),
          label: `${ticket}:${v.key}`,
          phase: upstream ? 'Chain' : 'Propose',
          schema: PROPOSAL_SCHEMA,
        }).then((p) => (p ? { voter: v.key, ...p } : null)),
      ),
    )
  ).filter(Boolean)

  if (proposals.length < 2) {
    log(`${ticket}: only ${proposals.length} proposal(s) returned — no rebuttal round possible.`)
    return { ticket, proposals, rebuttals: [], agreed: null }
  }

  // Disagreement is judged on the recommendation text, not on incidental wording.
  // Cheap normalization only; Opus makes the real call at synthesis time.
  const norm = (s) => s.toLowerCase().replace(/[^a-z0-9 ]/g, ' ').replace(/\s+/g, ' ').trim()
  const agreed = norm(proposals[0].recommendation) === norm(proposals[1].recommendation)

  if (agreed) {
    log(`${ticket}: proposals converged on first pass — skipping rebuttal.`)
    return { ticket, proposals, rebuttals: [], agreed: true }
  }

  log(`${ticket}: proposals diverged — running one rebuttal round.`)
  const rebuttals = (
    await parallel(
      proposals.map((mine, i) => () => {
        const theirs = proposals[1 - i]
        const v = VOTERS.find((x) => x.key === mine.voter)
        return agent(rebuttalPrompt(ticket, mine, theirs), {
          agentType: v.agentType,
          ...(v.model ? { model: v.model } : {}),
          label: `${ticket}:${mine.voter}:rebut`,
          phase: 'Rebut',
          schema: REBUTTAL_SCHEMA,
        }).then((r) => (r ? { voter: mine.voter, ...r } : null))
      }),
    )
  ).filter(Boolean)

  return { ticket, proposals, rebuttals, agreed: false }
}

// ---------------------------------------------------------------------------

const tickets = Array.isArray(args) && args.length ? args : DEFAULT_TICKETS

log(`Design panel over ${tickets.length} independent ticket(s), then ${CHAINED_TICKET} chained on the GH-49B outcome.`)

phase('Propose')
const results = await pipeline(tickets, (t) => runTicket(t, null))

// GH-114-DESIGN declares depends-on GH-49B-DESIGN, so it runs after, seeded with
// whatever that panel concluded. Running it in parallel would design against an
// unsettled upstream.
phase('Chain')
const upstream = results.find((r) => r && r.ticket === 'GH-49B-DESIGN')
const upstreamText = upstream
  ? upstream.proposals.map((p) => `- ${p.voter}: ${p.recommendation}`).join('\n') +
    (upstream.agreed ? '\n(both agreed)' : '\n(UNSETTLED — the two disagreed; treat as an open fork)')
  : '(GH-49B-DESIGN was not run in this batch — treat the binding semantics as unsettled)'

const chained = await runTicket(CHAINED_TICKET, upstreamText)

const all = [...results.filter(Boolean), chained]
const unresolved = all.filter((r) => r.agreed === false)
log(`${all.length - unresolved.length} converged, ${unresolved.length} still contested (${unresolved.map((u) => u.ticket).join(', ') || 'none'}).`)

// Nothing is ratified here. Opus synthesizes; the owner decides.
return { results: all, contested: unresolved.map((u) => u.ticket) }
