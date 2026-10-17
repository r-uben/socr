# #1189: PyMuPDF aborts the process; one abort ends the rest of a cluster shard

Date: 2026-10-17. Branch `fix/issue-1189`, base `7b3b3c09`.

## Diagnosis

One defect, not two. `page.get_texttrace()` decrements `None`'s reference count once per span
it returns (with `gc.collect()` forced around each call: 309 spans → −320, 515 → −520). The
count reaches zero and the interpreter aborts with `Fatal Python error: none_dealloc`. That is
an abort, not an exception: no `except`, `finally` or context manager runs.

It is cumulative within a process. The abort point moves with the starting page and tracks
cumulative spans, so no single-page extract reproduces it. `find_tables` is a bystander: it
completes the strongin paper alone, and the fatal decref simply lands wherever `None` is
touched next.

Affected versions: PyMuPDF 1.27.2.2, 1.27.2.3, 1.28.0. Clean: 1.26.7, 1.27.1, 1.27.2, 1.28.2
(each reproduced 3/3). Upstream #5042, fixed in 1.28.2. The cluster ran
`1.27.2.2+computecanada`. Corpus rate: **45 of a seeded 200-document Fed sample abort**.

## Decision

Stop calling `get_texttrace`. Both callers read the text layer's own `alpha` through one
shared helper, `text_layer_invisible_char_counts` (`tables/source_evidence.py`). No version
bound: a bound the Alliance wheelhouse cannot satisfy would brick the Fed run. The
misattributed comment at `source_evidence.py:436` is corrected.

**Behaviour change.** The two routes are not equivalent by construction; three synthetic
shapes diverge:
- clip-only text (mode 7) and a transparent fill (mode 0, `ca=0`): `alpha` is more correct;
- outlined text (mode 2: transparent fill, opaque stroke): `alpha` is wrong, but the page is
  routed to OCR, which costs a re-read and loses nothing.
No page of the 506 measured is any of the three. All three are pinned as tests.

The helper fails closed when `get_text("dict")` returns `None`. A source guard (an AST walk,
since a regex is defeated by `getattr`) keeps `get_texttrace` out of `src`.

## Evidence

27 tests, including difference tests on the production path at both call sites. 7 external
mutants, all killed. Full suite 8,026 passed. On the UNFIXED PyMuPDF 1.27.2.3, all 45
previously aborting documents now complete: 0 aborts, 0 non-zero exits. Lead artefacts:
`scratch/routing-baseline/leads/1189/`.

## Not done here

The write-ahead tombstone (an aborted document is recorded and not retried forever) is its
own ticket, because a different abort could still end a shard silently. A shard driver that
captures only stdout never sees `Fatal Python error`, which goes to fd 2.
