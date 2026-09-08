# 2026-09-08 night audit — tables/figures/formulas

Method: owner asked for an issue audit + fixes on tables, figures, formulas, autonomous
until 03:00. Codex (gpt-6-astra) in a Herdr pane did the design brainstorm and adversarial
review; Sonnet/Opus subagents implemented in per-ticket worktrees; Sonnet reviewer then
Astra on every diff; merge only on CI green for the exact head. GPT/Gemini/Grok/Cursor
subagent types were unreachable from this session.

Triage: 55 open issues, five buckets; verdicts in the triage tables (summarise: 2 closed as
fixed/superseded — #591 (fixed by #651, remainder #649/#652) and #330 (superseded by #359);
policy questions #625/#624/#601 need an owner ruling; measurement debt on #146/#144/#64).

Merged tonight (main 9367c83 -> 935669c+): #656 E1 fall-through (#660); #219 PazoMath (#661,
Refs — mixed-font line unverified); #642 lane gate + floor event pin (#662); test pins
#646/#648/#613 (#665); #658 no-witness reason (#666, diagnostic slice; pytesseract packaging
+ flagged-witness path still open); docs #174/#615/#620 (#667); #650 fabrication fixture
(#669); #164/#157 equation sidecar scope (#664); #165 outcome-based unresolved-math
accounting (#675, Refs — routing criterion open); #154/#160/#637 cost caps + nougat (#670);
#609 majority-overlap membership + boundary evidence (#673, Refs — sub-case 1 surfaced not
bound; #608 alignment still separate).

#189 mixed-page chart preservation merged as #672 (main 0d45085) after 6 Astra rounds: vector-chart regions on mixed pages are now inventoried and reconciled at assembly; raster localisation and document-level asset resume stay open (#170).
#659 label tokens flag-not-reject merged as #668 after 5 Astra rounds and two rebases.


Parked: #643 (PR #663 draft) — four rounds showed no geometric/lexical footnote exclusion
that does not delete a real table row; needs font-size / bottom-rule evidence. Filed
comments on #643 and #219.

Figures procedure (Astra design, docs in scratchpad summarised here): classify
provisionally, never discard on uncertainty; ship image asset + caption + alt text per
class; data reading (#635) only behind provenance + series/unit binding; audits reconcile
detected-region inventory vs shipped assets. #189 implements the inventory invariant for
vector charts; raster-chart localisation and #170/#496 remain.

Formulas: keep flags default-off; #165 makes the unresolved-glyph warning outcome-based;
#219 adds the font; #164/#157 fix the legacy sidecar scope; PUA-only routing (#165 AC1) and
#140 clean-flattened math remain.

Pattern of record: Sonnet reviewers accepted 10/10 diffs; Astra found real holes in 9/10 on
round 1 and needed 2-5 rounds on the four larger diffs; every hole was one hop past the spec
(resume allowlist, final selector overwriting a cause, historical-union diagnosis, dead
reporting fields).

Process traps hit: shared git stash across worktrees (three accidental pops); a conflicting
PR runs no CI; new event kinds need `resume_restore_kinds`; MagicMock negatives.
