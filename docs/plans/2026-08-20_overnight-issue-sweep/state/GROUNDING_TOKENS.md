# Grounding tokens at main_sha (53b0637)

Each dispatched agent is asked for one of these. The value is unguessable, so a
matching token proves the agent actually ran a command against the pinned tree —
the counter-measure to the silent-fabrication failure mode (a headless agent
denied a tool permission produces confident invented output and exits 0).

    git -C /Users/rubenffuertes/repos/.worktrees/socr-night-base \
        show "53b0637b928c486e9ff3023fa9527905fec017b2:<path>" | shasum -a 256 | cut -c1-16

| path | token |
|---|---|
| `src/socr/tables/reconstruct.py`   | `96e86cc412fa0fcc` |
| `src/socr/pipeline/orchestrator.py`| `023fce53432dc5aa` |
| `src/socr/core/providers.py`       | `daa97398162448dd` |
| `src/socr/math/detect_equations.py`| `c849ec91c38784ed` |

## zsh trap — quote the revision argument

`git show $SHA:src/...` **unquoted in zsh** silently applies the `:s` history
modifier (the `s` after the colon is read as "substitute", with the next character
as its delimiter). It does not error; it produces a different, wrong blob and
therefore a plausible wrong hash. My first table had
`detect_equations.py = 229716060395f99c` from exactly this; the correct value is
`c849ec91c38784ed`, which `triage-b4-grok` reported independently and which a
properly quoted command confirms.

Always write `git show "${SHA}:<path>"`. The canary caught the orchestrator's own
error before it could be used to reject an honest agent.

## Sandbox finding — some vendor subagents cannot reach the venv

`adjudicate-b3` (gemini-pro) reported, honestly rather than faking it:

    CANARY FAIL: interpreter not found: /Users/rubenffuertes/venvs/socr/bin/python

The interpreter exists and is executable from the orchestrator's shell, so this is
a per-agent sandbox restriction, not a broken path. The same agent's `git show`
calls worked — it read code fine.

Consequence, applied 2026-08-20: `isolation_canary.sh` grew a `--read-only` mode
that proves the worktree is the pinned one and is not the main checkout, without
touching Python. Read-only agents (triage, adjudication, review board) should use
it. **Anyone who runs tests still needs full mode** — import isolation is the whole
point there.

Practical rule for the review board: the access proof that matters is a `git show`
of a cited line, not the canary. A reviewer that cannot produce that must reject.
