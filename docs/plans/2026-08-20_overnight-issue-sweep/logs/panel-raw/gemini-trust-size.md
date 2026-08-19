1. **TICKET-E (Lane O: `cascade-halt` #221+#222+#227 & #249+#248+#189) is too big and structurally unsafe for one-shot PR generation.**
   - *Problem:* `orchestrator.py` spans over 5,000 lines. Bundling #221 (GPU wedge detection/health ping semantics), #222 (backend host abstraction beyond localhost), and #227 (recovering state when subsequent rungs succeed after timeout) into a single agent prompt guarantees state-machine regression. Similarly, grouping #249 (chart vs phantom table grading), #248 (text layer noise causing table inflation), and #189 into one ticket invites the exact regression pattern seen in PR #250.
   - *Minimum change:* Split by layer, not file:
     - Ticket 1: Extract backend probe endpoint & health interface to `core/` (solves #222 cleanly without touching loop logic).
     - Ticket 2: Cascade latch logic in `orchestrator.py` (#221 + #227).
     - Defer #249/#248 entirely from wave 4 fixes to `NEEDS-MEASUREMENT` analysis (corpus measurement required; cannot one-shot safely overnight).

2. **TICKET-E (Lane T: GH-144 trio #195/#197/#198) is over-fragmented ceremony.**
   - *Problem:* Issues #195, #197, and #198 all touch table markdown parsing/cleaning heuristics (`src/socr/tables/`). Dispatching 3 separate agents to create 3 independent branches across the same module in parallel creates guaranteed merge conflicts and 3x PR overhead for minor regex/format fixes.
   - *Minimum change:* Merge #195, #197, #198 into a single ticket: `TICKET-E-TABLES-CLEANUP`.

3. **TICKET-A1 and TICKET-A2 are ceremony that should fold into A3/runner harness.**
   - *Problem:* A standalone ticket for checking `gh auth status` and running a dummy Python import canary creates ticketing overhead before any real work starts. If auth fails or PYTHONPATH fails, the runner script itself should abort instantly at startup.
   - *Minimum change:* Merge A1, A2, and A3 into a single `PREFLIGHT` shell step executed synchronously by the orchestrator before dispatching any subagent waves.

4. **Adjudication (Stream C) allows fabricated citations to pass unchecked.**
   - *Problem:* The prompt states that LLM triagers must provide `path:line` evidence. However, TICKET-C adjudicators are LLMs asked to check LLM outputs. An LLM adjudicator evaluating plausible but hallucinated `file:line` quotes will hallucinate agreement or mark `evidence_verified: true` without executing a tool.
   - *Minimum change:* Implement a mechanical (non-LLM) verification script in Stream C:
     ```bash
     python3 -c "
     import json, subprocess, sys
     verdicts = json.load(open(sys.argv[1]))
     for v in verdicts:
         for cit in v.get('citations', []):
             path, line = cit['path'], int(cit['line'])
             res = subprocess.run(['git', 'show', f'origin/main:{path}'], capture_output=True, text=True)
             assert res.returncode == 0, f'File missing: {path}'
             lines = res.stdout.splitlines()
             assert 1 <= line <= len(lines), f'Line out of range: {path}:{line}'
             assert cit['snippet'].strip() in lines[line-1].strip(), f'Snippet mismatch at {path}:{line}'
     "
     ```
     If mechanical verification fails, the verdict is rejected automatically before reaching Stream D.

5. **TICKET-D1 and D2 will vandalize the live issue tracker without human gate.**
   - *Problem:* Granting autonomous closing (`gh issue close`) and commenting on open issues overnight based on consensus between 3 models is high risk (the prompt itself notes #249 required 3 human revisions and PR #250 reintroduced bugs). Closing valid user bug reports with confident wrong LLM claims destroys trust immediately.
   - *Minimum change:* Restrict Stream D overnight: generate a staged payload `tracker_actions.json` containing proposed comments and closures. Do NOT execute `gh issue close` or `gh issue comment` unattended. Let the morning report present a 1-click batch review/apply script (`./apply_tracker_actions.sh`).

6. **TICKET-E<n> PR CI check definition is vacuous for unattended agents.**
   - *Problem:* The graph states "PR open; `gh pr checks` all pass", but Hard Fact #3 notes CI has no Ollama/VLM backend and agentic tests mock engine availability. An agent can delete/skip tests or mock around edge cases to get CI green while failing silent content loss invariants.
   - *Minimum change:* Add a mandatory negative-proof verification step in the harness: run the added regression test on `origin/main` (before the fix) and assert failure exit code `> 0`, then run on the branch and assert `0`, recording the raw stdout diff in the PR body.
