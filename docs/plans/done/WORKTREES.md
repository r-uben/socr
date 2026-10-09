# Worktree cleanup list

Done 2026-10-08 against the rule in `GOAL.md`. **Nothing has been removed yet.**

**Rule.** REMOVE a worktree whose branch is merged into origin/main, or whose PR is merged or
closed, and which has no uncommitted changes and no unpushed commits. Anything else is KEEP.
Never removed: the main checkout, this worktree, a worktree another Herdr space has open
(`audit-exp-table-checks`), and the prose-splice worktree `agent-aefdb458bcd16d3ae`.

**Counts.** 76 worktrees: 58 REMOVE, 18 KEEP (4 protected, 12 with unsaved work, 2 unmerged with no PR).

**How.** "Unpushed" means GitHub does not have the commit, checked per commit through the API;
that covers squash-merged branches whose remote branch was deleted. Every squash-merged or
closed branch's local head was compared with its PR's final head: all are identical except
`cursor/native-table-first-629d`, which is 3 commits behind it. So no branch holds work beyond
what its PR carried. Three branches with no PR of their own have their head inside a merged PR
(#269, #907, #959). REMOVE candidates were also checked for ignored files.

## KEEP — unsaved work

| Path | Branch | State |
|---|---|---|
| `~/.herdr/worktrees/socr/chore-agents-md` | `chore/agents-md` | 1 unpushed commit (2026-10-01, rename CLAUDE.md to AGENTS.md). No Herdr space open. |
| `~/.local/state/socr-housekeeping/r901/wt` | `(detached)` | Detached at `bda96605`; 11 staged or modified files (#901 rotated-upright-frame work, incl. new `docs/log/2026-09-25_901-rotated-upright-frame.md`). |
| `~/repos/.worktrees/socr-binding` | `feat/binding-oracle` | PR #266 merged; 4 untracked files `tests/test_adversarial_binding*.py`. |
| `~/repos/.worktrees/socr-progressive-pages` | `feat/progressive-pages` | 7 unpushed commits (last 2026-06-16); 1 untracked `docs/log/2026-06-16_PP-2-design.md`. |
| `~/repos/tools/curia367-codex` | `curia367/codex` | Curia #367 attempt; 4 modified, 3 untracked (`binding_adjudication.py`, its test, a log). |
| `~/repos/tools/curia367-grok` | `curia367/grok` | Curia #367 attempt; 7 modified, 6 untracked. |
| `~/repos/tools/curia373-codex` | `curia373/codex` | Curia #373 attempt; 16 modified, 2 untracked. |
| `~/repos/tools/curia373-grok` | `curia373/grok` | Curia #373 attempt; 8 modified, 3 untracked. |
| `~/repos/tools/socr-fix-988` | `fix/988-shortfall-missing-content` | #988 work; 2 modified (`row_corroboration.py`, `structure_check.py`), 3 untracked (log, test, `gh988_coke_2021_p74` fixture). |
| `~/repos/tools/socr/.claude/worktrees/agent-a49c5fcc074776c29` | `fix/901-rotated-upright-frame` | 1 unpushed commit (2026-09-26, fix(901): verify rotated table pages in an upright frame). |
| `~/repos/tools/socr/.claude/worktrees/agent-a83929fab99e74615` | `fix/925-header-band-stub` | 1 unpushed commit (2026-10-01, wip(925): stub-first header absorption). |
| `~/repos/tools/socr/.claude/worktrees/agent-a995e88e5bffc276f` | `fix/968-total-deadline` | PR #975 merged; 1 modified `src/socr/core/ollama_utils.py`. |

## KEEP — protected

| Path | Branch | State |
|---|---|---|
| `~/repos/tools/socr` | `docs/chart-data-stage2-status` | Main checkout (on `docs/chart-data-stage2-status`, PR #754 merged); clean. |
| `~/.herdr/worktrees/socr/audit-exp-table-checks` | `fix/table-checks-nav-and-shortfall` | Open Herdr space `[audit] exp:table-checks`; clean; holds ignored #988 audit data in `scratch/`. |
| `~/.herdr/worktrees/socr/impl-exp-done-goal` | `impl/exp-done-goal` | This worktree; 1 unpushed commit (`GOAL.md`) plus the commit that adds this file. |
| `~/repos/tools/socr/.claude/worktrees/agent-aefdb458bcd16d3ae` | `fix/table-only-floor-keeps-corroborated-prose` | Prose-splice work (#1043); committed as `a28e3d6f` during this scan, which is on GitHub; not merged. |

## KEEP — unmerged, no PR

| Path | Branch | State |
|---|---|---|
| `~/repos/.worktrees/socr-gemini-model` | `feat/gemini-model-setting` | Clean and pushed, but unmerged and no PR (last commit 2026-10-07, `--gemini-model`). |
| `~/repos/tools/socr/.claude/worktrees/agent-acba7808bf7c8a4bd` | `fix/924-split-date-phrase` | Clean and pushed, but unmerged and no PR (2026-10-01, ship-gate `split_phrase`); #924 is closed. |

## REMOVE — clean, pushed, merged or closed

| Path | Branch | Why |
|---|---|---|
| `/private/tmp/claude-501/-Users-rubenffuertes-repos-tools-socr/e06a984d-b8e5-4cf4-866c-d6e5431d082d/scratchpad/wt` | `fix/judge-bounded-reply-and-tolerance` | PR #1039 merged; head is on origin/main. Lives in another Claude session's scratchpad under `/private/tmp`. |
| `~/repos/.worktrees/s1-winner-selection` | `feat/s1-winner-selection-v2` | Head is in merged PR #269. |
| `~/repos/.worktrees/socr-643` | `fix/643-footnote-bands` | PR #663 closed unmerged; local head is the PR head. |
| `~/repos/.worktrees/socr-749` | `detached c45afeb2` | Head is on origin/main; no commits of its own. Named after protected #749 but holds no commits, changes or ignored files. |
| `~/repos/.worktrees/socr-before` | `detached ba46be20` | Head is on origin/main; no commits of its own. |
| `~/repos/.worktrees/socr-trial` | `detached ed2550e2` | Head is on origin/main; no commits of its own. |
| `~/repos/tools/socr/.claude/worktrees/agent-a028104ff233fdaef` | `chore/cleanup-ship-gate` | PR #929 merged (squash); local head is the PR head. |
| `~/repos/tools/socr/.claude/worktrees/agent-a039596049719c5e5` | `worktree-agent-a039596049719c5e5` | Head is in merged PR #907. |
| `~/repos/tools/socr/.claude/worktrees/agent-a06ed7395274bf3e3` | `fix/1034-flaky-budget-test` | PR #1037 merged; head is on origin/main. |
| `~/repos/tools/socr/.claude/worktrees/agent-a0d82c9243b06bc84` | `chore/cleanup-ollama-judge` | PR #927 merged (squash); local head is the PR head. |
| `~/repos/tools/socr/.claude/worktrees/agent-a0f924eb7fc59315e` | `fix/invisible-scan-floor-not-native` | PR #1028 merged; head is on origin/main. |
| `~/repos/tools/socr/.claude/worktrees/agent-a10ae3e71cd673dfd` | `fix/942-header-band-runs` | PR #943 merged (squash); local head is the PR head. |
| `~/repos/tools/socr/.claude/worktrees/agent-a12931476f20daa79` | `worktree-agent-a12931476f20daa79` | Head is on origin/main; no commits of its own. |
| `~/repos/tools/socr/.claude/worktrees/agent-a166125627e55e4cb` | `fix/withhold-contradicted-unverified-tables` | PR #1023 merged; head is on origin/main. |
| `~/repos/tools/socr/.claude/worktrees/agent-a2088c46946e25168` | `docs/960-detector-docs-current` | PR #1019 merged; head is on origin/main. |
| `~/repos/tools/socr/.claude/worktrees/agent-a2170ce52b280772a` | `fix/1005-native-math-status-follows-shipped` | PR #1016 merged; head is on origin/main. |
| `~/repos/tools/socr/.claude/worktrees/agent-a21a0d2ceba79ee3c` | `feat/993-readable-tables-metric` | PR #997 merged (squash); local head is the PR head. |
| `~/repos/tools/socr/.claude/worktrees/agent-a28f71a8e945b339d` | `fix/1004-judge-timeout-untrusted-native` | PR #1012 merged; head is on origin/main. |
| `~/repos/tools/socr/.claude/worktrees/agent-a2b83f54f844a5435` | `fix/949-geometryless-block` | PR #950 merged (squash); local head is the PR head. |
| `~/repos/tools/socr/.claude/worktrees/agent-a334bf7414b595a8a` | `fix/936-prose-in-header` | PR #944 merged (squash); local head is the PR head. |
| `~/repos/tools/socr/.claude/worktrees/agent-a347caca15bf07d55` | `fix/917-text-in-numeric-column` | PR #931 merged (squash); local head is the PR head. |
| `~/repos/tools/socr/.claude/worktrees/agent-a35fe37437f9d74e6` | `fix/905-retired-cloud-model` | PR #911 merged (squash); local head is the PR head. |
| `~/repos/tools/socr/.claude/worktrees/agent-a3d4deabff005cb68` | `fix/928-shared-md-parsing` | PR #983 merged (squash); local head is the PR head. |
| `~/repos/tools/socr/.claude/worktrees/agent-a45e50f3c5c6517a7` | `fix/851-escalation-latch-evidence` | PR #937 merged (squash); local head is the PR head. |
| `~/repos/tools/socr/.claude/worktrees/agent-a4aaff88421de9aaa` | `feat/964-library-config` | PR #966 merged (squash); local head is the PR head. |
| `~/repos/tools/socr/.claude/worktrees/agent-a4abdf6b2c4807bf1` | `fix/974-page-budget-daemon` | PR #978 merged (squash); local head is the PR head. |
| `~/repos/tools/socr/.claude/worktrees/agent-a4d3e5335c0d3da22` | `fix/913-minus2-reroute` | PR #946 merged (squash); local head is the PR head. |
| `~/repos/tools/socr/.claude/worktrees/agent-a4e556216fb6048e5` | `fix/994-flattened-table-demote` | PR #998 merged; head is on origin/main. |
| `~/repos/tools/socr/.claude/worktrees/agent-a50a8adda8358c41a` | `fix/934-prose-line-predicate` | PR #938 closed unmerged; local head is the PR head. |
| `~/repos/tools/socr/.claude/worktrees/agent-a5202618a55929c30` | `fix/976-host-userinfo` | PR #977 merged (squash); local head is the PR head. |
| `~/repos/tools/socr/.claude/worktrees/agent-a6310290969ce3cd5` | `fix/969-973-library-hardening` | PR #1007 merged; head is on origin/main. |
| `~/repos/tools/socr/.claude/worktrees/agent-a676aa2a8bfd55940` | `fix/truncated-shortfall-symmetric` | PR #996 closed unmerged; local head is the PR head. Only an ignored `uv.lock`. |
| `~/repos/tools/socr/.claude/worktrees/agent-a74235571cd795eb9` | `fix/1001-halted-doc-not-skippable` | PR #1002 merged; head is on origin/main. |
| `~/repos/tools/socr/.claude/worktrees/agent-a74298e3852b8b1a6` | `fix/903-judge-model-retired` | PR #906 merged (squash); local head is the PR head. |
| `~/repos/tools/socr/.claude/worktrees/agent-a744e0e6ae2b89f69` | `fix/1020-cli-failure-placeholder-not-success` | PR #1021 merged; head is on origin/main. |
| `~/repos/tools/socr/.claude/worktrees/agent-a82d97b09d2b0869f` | `fix/917-gate-direction-header` | PR #926 merged (squash); local head is the PR head. |
| `~/repos/tools/socr/.claude/worktrees/agent-a84e7bd5cdefa205e` | `fix/932-tnc-numeric-classifier` | PR #935 merged (squash); local head is the PR head. |
| `~/repos/tools/socr/.claude/worktrees/agent-a8efbf697bf25a1a1` | `fix/902-rotation-sign` | PR #915 merged (squash); local head is the PR head. |
| `~/repos/tools/socr/.claude/worktrees/agent-a90480b738155a099` | `fix/958-followup` | Head is in merged PR #959. |
| `~/repos/tools/socr/.claude/worktrees/agent-a96d98be396575f2a` | `fix/881-per-page-load-failure` | PR #947 merged (squash); local head is the PR head. |
| `~/repos/tools/socr/.claude/worktrees/agent-a9dedb818d23ec939` | `fix/984-test-hermeticity` | PR #986 merged (squash); local head is the PR head. |
| `~/repos/tools/socr/.claude/worktrees/agent-a9f7105465ef91cee` | `fix/951-word-split-across-cells` | PR #955 merged (squash); local head is the PR head. |
| `~/repos/tools/socr/.claude/worktrees/agent-ab6c98c8b2d3bdad5` | `fix/917-quarantine-rotated-ship` | PR #918 merged (squash); local head is the PR head. |
| `~/repos/tools/socr/.claude/worktrees/agent-ab9596136e0f7dea7` | `fix/judge-timeout-no-halt` | PR #989 merged (squash); local head is the PR head. |
| `~/repos/tools/socr/.claude/worktrees/agent-abe21b056211063a3` | `fix/916-native-ship-gate` | PR #920 merged (squash); local head is the PR head. |
| `~/repos/tools/socr/.claude/worktrees/agent-abfa16116936f0455` | `cursor/native-table-first-629d` | PR #907 merged (squash); local head is 3 commits behind the PR head. |
| `~/repos/tools/socr/.claude/worktrees/agent-ac09897f62afb623b` | `feat/1013-rejudge-timed-out-candidate` | PR #1018 merged; head is on origin/main. |
| `~/repos/tools/socr/.claude/worktrees/agent-ac69c8687dd95576e` | `fix/942b-run-clause-spacing` | PR #948 merged (squash); local head is the PR head. |
| `~/repos/tools/socr/.claude/worktrees/agent-ac6c48012ef3794dc` | `fix/940-resume-probe` | PR #985 merged (squash); local head is the PR head. |
| `~/repos/tools/socr/.claude/worktrees/agent-ac9be219bc3cf3c55` | `fix/961-invisible-text-scan` | PR #963 merged (squash); local head is the PR head. |
| `~/repos/tools/socr/.claude/worktrees/agent-acb2afbfe038393c3` | `fix/990-control-byte-reroute` | PR #992 merged (squash); local head is the PR head. |
| `~/repos/tools/socr/.claude/worktrees/agent-acc761518334c94e8` | `fix/910-ollama-http-check` | PR #912 merged (squash); local head is the PR head. |
| `~/repos/tools/socr/.claude/worktrees/agent-ad2d4db844de77bb9` | `fix/945-run-floor` | PR #954 merged (squash); local head is the PR head. |
| `~/repos/tools/socr/.claude/worktrees/agent-ae09ab36678053101` | `fix/scanned-figures-as-images` | PR #1031 merged; head is on origin/main. |
| `~/repos/tools/socr/.claude/worktrees/agent-ae407084205ea5bbc` | `docs/output-reference` | PR #965 merged (squash); local head is the PR head. |
| `~/repos/tools/socr/.claude/worktrees/agent-af8d59442f1a928b1` | `fix/991-flaky-timing-tests` | PR #1011 merged; head is on origin/main. |
| `~/repos/tools/socr/.claude/worktrees/figdesc` | `feat/figure-descriptions-number-free` | PR #1033 merged; head is on origin/main. |
| `~/repos/tools/socr/.claude/worktrees/gh960` | `fix/960-garbled-math-reroute` | PR #1006 merged; head is on origin/main. |

## Done (2026-10-09)

- Removed 57 of the 58 REMOVE worktrees with `git worktree remove`, without `--force`. Each
  was re-checked just before removal: same head as listed, no changes, no process with its
  working directory inside, nothing modified in the last two hours. No branches were deleted.
- Skipped `~/repos/tools/socr/.claude/worktrees/agent-a12931476f20daa79`. On the first pass a
  housekeeping scan (`native-tables/scan.py`) was running inside it. The owner then killed the
  scan, and the worktree passed the re-check: clean, head `ed2550e2` already on origin/main,
  no ignored files. Git still refused to remove it, because it is locked by a live Claude
  session (pid 2957, running since 2026-10-04). It holds nothing of its own, so it can go
  once that session ends.
- Added after this list was written, so not in the tables above:
  - `/private/tmp/fable1047/wt`, made by another session on the #1043 commit.
  - `~/repos/.worktrees/socr-1048`, on `fix/1048-unverified-tables-ship-marked`, made by this
    session for #1048.
