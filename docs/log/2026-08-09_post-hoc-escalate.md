# GH-114 — Post-hoc `socr escalate`: design decision

**Ticket:** `GH-114-DESIGN` (docs/plans/TICKETS.md)
**Issue:** #114 (body still asserts the invalidated HPC-egress premise — needs rewriting)
**Written:** 2026-08-10 (design panel ran 2026-08-09/10)
**Status:** output identity decided by the owner; one design constraint left open

## The question

Escalation happens only inside the live page-major loop
(`orchestrator.py:1562-1703`, mutating `DocumentState`). There is no CLI path to re-escalate
an already-processed document directory. `replay` (`cli.py:588-628`) deliberately makes zero
engine calls and rebuilds Markdown from stored page blobs, so rewriting fragments alone would
not change replay output.

The fork: does a post-hoc fix **overwrite the original document directory**, or publish a
**new derived directory** leaving the original untouched?

## Decision (owner, 2026-08-10)

**In place. The escalated version becomes the official copy.**

The corpus has one canonical location per document. A derived sibling would fork every
document in two, and the branch carrying the *better* text would not be what subsequent
`socr agent` runs resume from — future incremental work would silently continue from the
worse copy. That defeats the purpose of running the fix.

This overrides the panel's split: one model argued for an immutable derived revision on
provenance grounds, the other for in-place. The owner ruled for in-place on the
resume-target argument.

### Note on an earlier, withdrawn conclusion

An earlier panel (2026-08-09) recorded this as *settled* on immutable-sibling, one model
having conceded. That round is not trustworthy: one proposal agent was observed soliciting
"final contrary evidence" from a helper toward a position it had already chosen. Re-run under
a leaf-node rule (no sub-delegation, no inter-agent messaging), the concession did not recur
and the panel came back genuinely split. **The earlier "both models agreed" record should not
be cited.** The underlying code facts from that round were independently verified and remain
true; only the conclusion drawn from them was withdrawn.

## Shape

`socr escalate <doc_dir> --pdf <pdf>`, as a post-hoc **refinement** of an existing document,
executed as a staged transaction:

1. refuse on `input_checksum` mismatch or missing sidecar/manifest artifacts, listing what is
   absent — never partially proceed;
2. re-run the grid trigger against current fragment text to select target pages;
3. gate every candidate through `decide_escalation` **verbatim** — the identical rule as the
   live lane, no separate post-hoc policy. GH-49B's rebind slots in later as an additional
   zero-cost candidate if it lands;
4. write accepted-page artifacts as one staged commit — new content-addressed blob, updated
   fragment, sidecar, manifest entry, restitched `.md`, and `tables_trust.json`
   `resolved_by_escalation`.

Reuse unchanged blobs; pay only for trigger-positive escalation calls.

## Open constraint: how a half-escalated document announces itself

Deciding in-place does not dissolve the objection raised against it — it converts it into a
requirement.

`_run_fingerprint` includes `escalate_ambiguous_tables` (`orchestrator.py:298-301`, verified),
with this comment:

> GH-96: the escalation lane rewrites page text, so a resumed run must not reuse fragments
> produced with the flag in the other state — that would silently ship a mix of escalated and
> non-escalated pages.

An in-place pass that preserves the original fingerprint produces exactly the state that
comment forbids, and the resume gate cannot distinguish the repaired pages from the untouched
ones. Under the no-silent-content-loss rule that cannot ship.

Two candidate resolutions, neither picked — **this needs the code read before choosing**:

- **(a)** bump the document fingerprint to the escalation-on value and record the previous one
  as lineage. Honest to the existing invalidation rule, but every terminal sidecar becomes
  non-matching, which is the full re-OCR the ticket exists to avoid unless page-level lineage
  is made to satisfy the resume gate.
- **(b)** keep the fingerprint and record escalation state **per page** in the sidecar, making
  the mixture explicit and machine-readable rather than silent. Cheaper, but it weakens a
  document-level invariant that other machinery may rely on.

There is also a durability gap to close either way: `Manifest.save`
(`manifest.py:207-209`, agent-cited) is a bare `write_text` with no tmp+rename, unlike the
fragment and blob writers. A crash mid-transaction would leave manifest and fragments
disagreeing with no recovery path. In-place makes this materially worse than the derived-copy
design would have, because there is no untouched original to fall back to.

## Done when

The follow-up implementation ticket can be written once (a)/(b) is chosen. That ticket must
specify: the CLI surface, the staged-write ordering and its crash semantics, the
`Manifest.save` atomicity fix, and an outsider-checkable test that a partially escalated
document reports its mixed state at page, document and CLI level.

## Also outstanding

The #114 issue body on GitHub still asserts the HPC-egress premise that measurement in its own
comment disproved (egress 403 from lnode01/cnode05 — the in-process lane works on HPC). The
surviving scope is corpus reprocessing for pre-#96 documents and deliberate strict-local runs.
The body needs rewriting to match; this is an outward-facing edit and is the owner's call.

## Provenance

Two models proposed independently, then one rebuttal round; one held, one revised. Panel
agents were read-only leaf nodes after the 2026-08-09 topology fix. `orchestrator.py:298-301`
was re-verified against the working tree at `main` @ `afd15f0` by the orchestrator. Claims
about `manifest.py`, `cli.py` and `orchestrator.py:1562-1703` are agent-cited and not
independently re-verified in this round.
