export const meta = {
  name: 'free-tier-audit',
  description:
    "Falsify (not confirm) the thesis that socr's free native-text lane ships unverified; if it survives, propose the cheapest deterministic audit.",
  whenToUse:
    'Run when a claim about socr routing/cost architecture needs independent adversarial checking before it becomes an issue. Pass a thesis override via args (string); omit for the free-tier-audit thesis.',
  phases: [
    { title: 'Falsify', detail: 'gpt-sol and fable each try to REFUTE the thesis, then propose an audit only if it survives' },
    { title: 'Rebut', detail: 'only where the two disagree on the verdict or the audit design' },
  ],
}

// ---------------------------------------------------------------------------
// This panel exists to KILL a claim the orchestrator (Opus) made, not to
// decorate it. The claim was produced by a fast trace and may be wrong: the
// most valuable single output here is a pointer to verification of the free
// lane that the orchestrator missed. Agents are read-only leaf nodes.
// ---------------------------------------------------------------------------

const THESIS = `socr's cost routing is sound in shape — free-first, page-major, cheapest-first, judge-gated —
but it has one structural asymmetry: EVERYTHING PAID IS VERIFIED, EVERYTHING FREE IS TRUSTED.

Specifically, the orchestrator claims:
  (a) A page taking the free native-text lane (_is_trusted_native_without_ocr,
      src/socr/pipeline/orchestrator.py:1107-1126) is selected by a PRE-decision from three cheap
      signals (native_first + born_digital + native_text; not needs_ocr_enhancement; no tables),
      and its OUTPUT is then never checked by anything.
  (b) Pages passing that gate never enter ocr_pages (orchestrator.py:1916) so they never reach the
      judge/exactness/escalation machinery that guards the paid path.
  (c) Therefore GH-64 (borderless 2-column tables evade detection, ship as flattened prose,
      status SUCCESS, no flag, $0.00) is not merely a detection bug — it is the first observed
      instance of a general class: THE CHEAP LANE SHIPS WITHOUT PROOF.
  (d) The right fix is a cheap DETERMINISTIC post-hoc audit of free-lane output (no model call —
      a model call would defeat the purpose), which does not currently exist and is not ticketed.`

const VERDICT_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: [
    'thesisVerdict',
    'missedVerification',
    'reasoning',
    'leakInventory',
    'auditProposal',
    'auditCost',
    'falsePositiveRisk',
    'ticketShape',
    'evidence',
    'confidence',
  ],
  properties: {
    thesisVerdict: {
      type: 'string',
      enum: ['confirmed', 'partially-confirmed', 'refuted'],
      description:
        'refuted = the free lane IS meaningfully verified somewhere the orchestrator missed. partially-confirmed = some verification exists but does not cover the GH-64 class.',
    },
    missedVerification: {
      type: 'array',
      description:
        'THE MOST VALUABLE FIELD. Any place in the repo where free-lane (cost_usd=0.0, engine="native"/"chart_asset") output IS checked, audited, scored, flagged or gated AFTER it is produced. Look hard before returning an empty array: grep the assemble phase, manifest freeze, tables_trust, audit_log, AuditEvent kinds, the born-digital detector post-checks, and any 1A/1B gate. Empty array is a strong claim — only return it if you searched and found nothing.',
      items: { type: 'string' },
    },
    reasoning: { type: 'string', description: 'At most 6 sentences. Why the verdict.' },
    leakInventory: {
      type: 'array',
      description:
        'Concrete OTHER ways a page can take the free lane and be wrong, beyond GH-64 borderless tables. Each item: the page shape, and which of the three gate signals fails to catch it. Empty if you believe GH-64 is the only one.',
      items: { type: 'string' },
    },
    auditProposal: {
      type: 'string',
      description:
        'The cheapest DETERMINISTIC post-hoc check on free-lane output that would have caught GH-64. NO model calls, NO rasterization if avoidable. Name the signal and where it hooks in. If you refuted the thesis, write "n/a — see missedVerification".',
    },
    auditCost: {
      type: 'string',
      description: 'What your audit reads per page and roughly what that costs (ms, or "reuses X already computed"). Be concrete — an audit that costs a VLM call is a failed answer.',
    },
    falsePositiveRisk: {
      type: 'string',
      description:
        'What legitimate prose your audit would wrongly flag, and why that is acceptable or how it is bounded. An audit that flags everything is useless; say honestly where yours sits.',
    },
    ticketShape: {
      type: 'string',
      description:
        'One of: widen GH-64 in place / new issue superseding GH-64 / new issue alongside GH-64 / no ticket needed. Justify in one sentence.',
    },
    evidence: {
      type: 'array',
      description: 'file:line anchors you actually opened. MUST be non-empty. Include the ones you checked that DISCONFIRMED something you initially believed.',
      items: { type: 'string' },
    },
    confidence: { type: 'string', enum: ['low', 'medium', 'high'] },
  },
}

const REBUTTAL_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['position', 'reasoning', 'strongestPointAgainstMe'],
  properties: {
    position: { type: 'string', enum: ['hold', 'concede', 'revise'] },
    reasoning: { type: 'string', description: 'At most 4 sentences.' },
    revisedAuditProposal: { type: 'string', description: 'Required if position is revise. Empty string otherwise.' },
    strongestPointAgainstMe: {
      type: 'string',
      description: 'The single best argument on the other side, stated fairly, even if you hold.',
    },
  },
}

const VOTERS = [
  { key: 'gpt', agentType: 'gpt-sol' },
  { key: 'fable', agentType: 'general-purpose', model: 'fable' },
]

function falsifyPrompt(thesis) {
  return `You are adversarially testing a claim about the socr repository (multi-engine document OCR, Python, src/socr) at /Users/rubenffuertes/repos/tools/socr, branch chore/triage-open-issues @ 0479005.

The claim was made by another model after a FAST trace. It may be wrong. Your job is to try to REFUTE it first, and only propose a fix if it survives.

=== THE CLAIM UNDER TEST ===
${thesis}
=== END CLAIM ===

MANDATORY procedure, in this order:

1. TRY TO REFUTE IT. Before anything else, go looking for verification of the free lane that the
   claimant missed. Concretely: grep for where PageOutput with engine="native" or "chart_asset" or
   cost_usd=0.0 is inspected downstream. Read _phase_assemble, the manifest freeze path
   (_winning_page_output), tables_trust, core/audit_log.py and the AuditEvent kinds, the
   born-digital detector, and any gate referred to as 1A/1B. If free-lane output IS checked
   somewhere, the claim is refuted or partially refuted and THAT is your headline finding.

2. Only if it survives step 1: verify the claim's own four anchors rather than trusting them —
   orchestrator.py:1107-1126, :1916, :2598, :4509. Report any that are wrong or misread.

3. Only then: build the leak inventory and design the cheapest deterministic audit.

CONSTRAINTS ON THE AUDIT (if you propose one):
- NO model/VLM call. The entire point of the free lane is that it costs nothing; an audit that
  costs an inference call defeats it. Prefer signals already computed during born-digital
  detection or table location.
- Honour CLAUDE.md: NO magic thresholds — derive from the page's own data or name and document a
  constant. NO silent content loss: a flag must surface at page, document and CLI level.
- Read the prior art before proposing: docs/plans/table-repair/, docs/plans/progressive-pages/,
  docs/log/. Approaches have already been tried and rejected here. Repeating a recorded failure
  without saying so is a failed answer.

HARD RULES:
- You are a LEAF NODE. Do NOT call the Agent/Task tool. Do NOT spawn subagents or helpers. Do NOT
  call SendMessage or message any other agent. Your structured return value is your ONLY output
  channel. Do your own reading.
- Do NOT go looking for support for a conclusion you have already reached. You are being paid to
  falsify, not to agree. Confirming the claim is an acceptable outcome ONLY after a real attempt
  to break it — and your evidence array must show where you looked.
- Do NOT modify any file. Do NOT run git commit, gh, or any write command.
- Work independently. Do not speculate about what another model would say.`
}

function rebuttalPrompt(mine, theirs) {
  return `Falsification panel, rebuttal round (socr repository).

You previously returned:
${JSON.stringify(mine, null, 2)}

An independent reviewer, working from the same code, returned something different:
${JSON.stringify(theirs, null, 2)}

Re-examine the code with their argument in hand. Pay particular attention to their
missedVerification entries — if they found post-hoc checking of the free lane that you did not,
that outranks any disagreement about audit design.

Then: hold (say why theirs fails on evidence, not preference) / concede (say what you missed) /
revise (a third answer beats both).

You MUST fill strongestPointAgainstMe fairly even when you hold. Changing your mind under
evidence is the point of this round.

READ-ONLY: do not modify any file or run any write command.
LEAF NODE: do not call the Agent/Task tool, do not spawn helpers, do not call SendMessage.
Re-read the code yourself; do not go recruiting support for the position you already hold.`
}

// ---------------------------------------------------------------------------

const thesis = typeof args === 'string' && args.trim() ? args : THESIS

phase('Falsify')
log('Two models independently attempt to REFUTE the free-tier-audit thesis before proposing anything.')

const verdicts = (
  await parallel(
    VOTERS.map((v) => () =>
      agent(falsifyPrompt(thesis), {
        agentType: v.agentType,
        ...(v.model ? { model: v.model } : {}),
        label: `falsify:${v.key}`,
        phase: 'Falsify',
        schema: VERDICT_SCHEMA,
      }).then((r) => (r ? { voter: v.key, ...r } : null)),
    ),
  )
).filter(Boolean)

if (verdicts.length < 2) {
  log(`Only ${verdicts.length} verdict(s) returned — no rebuttal round possible.`)
  return { verdicts, rebuttals: [], contested: null }
}

const refuters = verdicts.filter((v) => v.thesisVerdict === 'refuted')
if (refuters.length) {
  log(`REFUTED by ${refuters.map((r) => r.voter).join(', ')} — check missedVerification first.`)
}

// Contested if the verdicts differ, OR if both confirm but propose different audits.
const norm = (s) => (s || '').toLowerCase().replace(/[^a-z0-9 ]/g, ' ').replace(/\s+/g, ' ').trim()
const verdictDiffers = verdicts[0].thesisVerdict !== verdicts[1].thesisVerdict
const auditDiffers = norm(verdicts[0].auditProposal) !== norm(verdicts[1].auditProposal)
const contested = verdictDiffers || auditDiffers

if (!contested) {
  log('Both converged on verdict and audit — skipping rebuttal.')
  return { verdicts, rebuttals: [], contested: false }
}

log(`Contested (verdict differs: ${verdictDiffers}, audit differs: ${auditDiffers}) — one rebuttal round.`)
phase('Rebut')

const rebuttals = (
  await parallel(
    verdicts.map((mine, i) => () => {
      const v = VOTERS.find((x) => x.key === mine.voter)
      return agent(rebuttalPrompt(mine, verdicts[1 - i]), {
        agentType: v.agentType,
        ...(v.model ? { model: v.model } : {}),
        label: `rebut:${mine.voter}`,
        phase: 'Rebut',
        schema: REBUTTAL_SCHEMA,
      }).then((r) => (r ? { voter: mine.voter, ...r } : null))
    }),
  )
).filter(Boolean)

return { verdicts, rebuttals, contested: true }
