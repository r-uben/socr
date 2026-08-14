# STATUS — fake-native pages

Last updated: 2026-08-14

## Stage

**Scaffolded, not dispatched.** The defect is diagnosed and evidenced on one page; the corpus
population is unmeasured, which is TICKET-A1.

Found while hand-judging TR-3 firings for `#205`: the first flagged page turned out not to be a
table-reconstruction defect at all. It is a 1994 NBER **scan** whose baked-in OCR text layer is
corrupt (`Eftes of Fedal Fixbds`, `Sigrificace`, `0271` for `0.271`), which socr classifies as
born-digital and reads natively.

## Base state (clean before tickets)

- Repo `~/repos/tools/socr`, `main` at `ce2d84d`. Full suite 1591 passed / 1 xfailed.
- Evidence for the diagnosis is in this plan's `TICKETS.md` context section — measured, not
  asserted.
- The reproducing page: `1994__christiano_eichenbaum_evans__..._NBER.pdf`, fitz index 34
  (printed page 35). Raster coverage **0.998**, 2077 chars, 0 non-printable, 0 CID artifacts,
  334 words, mean word length 5.22 — every existing quality check passes.

## Ticket board

| Ticket | Stream | Status | depends-on | Wave |
|--------|--------|--------|------------|------|
| A1 | measure | TODO | — | 1 |
| B1 | detect | TODO | A1 | 2 |
| B2 | detect | BLOCKED (conditional) | A1 | 2 |
| C1 | consequence | TODO | B1 | 3 |
| C2 | consequence | TODO | C1 | 4 |

## Dispatch waves

- **Wave 1:** A1 — read-only, `logs/` only, no collisions.
- **Wave 2:** B1 on `born_digital.py`. B2 only if A1 shows a population B1 misses.
- **Wave 3:** C1 — `born_digital.py`, `orchestrator.py`, `state.py`. **Contended, see below.**
- **Wave 4:** C2 — `logs/` and an issue comment.

## File contention — the reason C1 cannot dispatch on demand

`src/socr/core/born_digital.py`, `orchestrator.py` and `state.py` are claimed by:

- **PR #200** (GH-151 B1) — open and held while its direction is decided.
- **GH-150 C1** (chart placeholder merge) — queued, blocked on #200.
- **GH-152 A2** — may need `born_digital.py` if reading order stays in its `Done when`.

C1 here joins that queue. B1 touches only `born_digital.py` and can go as soon as #200's fate is
settled. The cross-plan coordinator at `docs/plans/extraction-defects/STATUS.md` owns the global
order; this plan does not override it.

## Relationship to the work this interrupted

This plan does not supersede GH-151 B1 or `#205`; it may **invalidate their inputs**.

- `#205` measured TR-3 firing on 68 of 491 native table pages (13.8%). An unknown share of those
  may be fake-native pages, where the mismatch is corrupt source text rather than a
  reconstruction defect. A1 quantifies this; C2 re-measures it.
- The GH-151 B1 shape gate counts garbled labels as genuine textual damage. Same contamination.
- **Nothing on #200 should be decided on the current figures until A1 lands.** That is the
  practical cost of this discovery and the reason it is worth doing first.

## Next action

**Dispatch TICKET-A1.** Read-only, no file contention, and its second figure — what share of
TR-3's firings are fake-native pages — is the number that unblocks the `#205` and PR #200
decisions.
