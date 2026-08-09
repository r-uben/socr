export const meta = {
  name: 'triage-issues',
  description: 'GPT and Fable independently triage each open issue as still-valid / already-fixed / superseded; Opus reconciles.',
  whenToUse:
    'Run when the open-issue backlog has drifted and you need evidence-grounded verdicts on which issues are stale. Pass issue numbers via args (array of numbers); omit to triage all open issues.',
  phases: [
    { title: 'Vote', detail: 'gpt-terra and fable each read the issue and the code, independently' },
    { title: 'Reconcile', detail: 'collect both ballots per issue for Opus to judge in-session' },
  ],
}

// ---------------------------------------------------------------------------
// Verdict taxonomy. An agent may only claim an issue is stale if it cites
// concrete evidence (file:line or commit sha). No evidence -> "unknown".
// ---------------------------------------------------------------------------
const BALLOT_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['issue', 'verdict', 'confidence', 'evidence', 'rationale'],
  properties: {
    issue: { type: 'number' },
    verdict: {
      type: 'string',
      enum: ['still-valid', 'partially-addressed', 'already-fixed', 'superseded', 'wontfix-scope', 'unknown'],
    },
    confidence: { type: 'string', enum: ['low', 'medium', 'high'] },
    evidence: {
      type: 'array',
      description:
        'Concrete anchors: "src/socr/pipeline/orchestrator.py:412" or a commit sha. MUST be non-empty for every verdict except "unknown" and "still-valid".',
      items: { type: 'string' },
    },
    supersededBy: {
      type: 'string',
      description: 'Issue number or initiative that replaced this one. Empty string if not applicable.',
    },
    rationale: { type: 'string', description: 'At most 3 sentences. What changed, or why nothing did.' },
  },
}

const VOTERS = [
  { key: 'gpt', agentType: 'gpt-terra' },
  { key: 'fable', agentType: 'general-purpose', model: 'fable' },
]

function ballotPrompt(issue) {
  return `You are triaging ONE open GitHub issue in the socr repository (multi-engine document OCR, Python, src/socr).

Issue to triage: #${issue}

Your job: decide whether this issue is STILL REAL against the code as it exists on \`main\` today, or whether the repo has moved past it.

MANDATORY procedure — do all of it before voting:
1. \`gh issue view ${issue} --comments\` — read the body AND every comment. Later comments often narrow or invalidate the original claim.
2. Identify the specific code the issue is about. Actually read it: Grep/Glob into src/socr, then Read the relevant functions. Do not reason from the issue text alone.
3. Check whether the repo already moved: \`git log --oneline -15 -- <the files you identified>\`, and skim docs/plans/ and docs/log/ for an initiative that covers this ground (the progressive-pages page-major rewrite in docs/plans/progressive-pages/ landed after several of these issues were filed).

Verdicts:
- still-valid          — the defect or gap is still present in the code today.
- partially-addressed  — some of the issue is fixed, a named part is not.
- already-fixed        — the code now does what the issue asked. Requires evidence.
- superseded           — a different issue or landed initiative replaced this framing. Requires evidence and supersededBy.
- wontfix-scope        — the request is coherent but outside what socr is for.
- unknown              — you could not ground a verdict. This is an acceptable, honest answer.

HARD RULES:
- Any verdict of already-fixed, partially-addressed, superseded, or wontfix-scope MUST carry at least one concrete evidence anchor (file:line you actually read, or a commit sha you actually saw in git log). An unsupported staleness claim is worse than "unknown" — if you cannot cite, vote unknown.
- Do NOT modify any file. Do NOT run \`gh issue close\`, \`gh issue edit\`, or any other write command. This is read-only triage.
- Vote independently. Do not speculate about what another model would say.
- Age alone is not evidence. An issue filed in June that nothing has touched is still-valid, not old.`
}

// ---------------------------------------------------------------------------

const issues = Array.isArray(args) && args.length ? args : null
if (!issues) {
  throw new Error(
    'Pass the issue numbers explicitly, e.g. args: [127, 114, 64, 56, 51, 50, 49, 46, 39]. ' +
      'Workflow scripts cannot shell out to `gh issue list`, so the caller must resolve them first.',
  )
}

log(`Triaging ${issues.length} issues x ${VOTERS.length} independent voters = ${issues.length * VOTERS.length} agents.`)

phase('Vote')

const ballots = await pipeline(issues, (issue) =>
  parallel(
    VOTERS.map((v) => () =>
      agent(ballotPrompt(issue), {
        agentType: v.agentType,
        ...(v.model ? { model: v.model } : {}),
        label: `#${issue}:${v.key}`,
        phase: 'Vote',
        schema: BALLOT_SCHEMA,
      }).then((b) => (b ? { voter: v.key, ...b } : { voter: v.key, issue, verdict: 'unknown', confidence: 'low', evidence: [], rationale: 'agent returned nothing' })),
    ),
  ),
)

phase('Reconcile')

const byIssue = issues.map((issue, i) => {
  const cast = (ballots[i] || []).filter(Boolean)
  const verdicts = [...new Set(cast.map((b) => b.verdict))]
  return {
    issue,
    agreed: verdicts.length === 1,
    verdicts,
    ballots: cast,
  }
})

const disputed = byIssue.filter((r) => !r.agreed)
log(`${byIssue.length - disputed.length} unanimous, ${disputed.length} disputed (#${disputed.map((d) => d.issue).join(', #') || 'none'}).`)

// Deliberately NOT auto-resolving. Opus reconciles in-session; the human decides
// what to close. Nothing here writes to GitHub.
return { results: byIssue, disputed: disputed.map((d) => d.issue) }
