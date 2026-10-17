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

## Review round 1 (PR #1193, r-uben)

**Blocker, confirmed and fixed: the alpha route failed open below PyMuPDF 1.25.3.**
Span `alpha` first appears in 1.25.3. Verified directly, one throwaway venv per version:
1.24.14, 1.25.0 and 1.25.2 emit spans with no `alpha` key; 1.25.3, 1.26.7 and 1.28.2 do.
`pyproject` still said `pymupdf>=1.24.0`, so on a resolvable install the 255 default in
`text_layer_span_is_visible` read every span as visible, `_has_invisible_text_over_raster`
returned False on a real OCR-layer scan, and the scan shipped as trusted native `success` —
silently. The pre-#1189 `get_texttrace` route worked on those versions, so this was a
regression introduced by the fix, and CI (locked wheel only) could not see it.

Both remedies applied, as asked:

1. Floor raised to `pymupdf>=1.25.3`. The Alliance wheel `1.27.2.2+computecanada` still
   satisfies it, and there is still no upper bound.
2. `text_layer_invisible_char_counts` now **raises** when a text span carries no `alpha`
   key. A missing key is not evidence of visible ink, and both callers already turn a raise
   into "unknown, go to OCR".

Measured with 1.25.2 shadowed in at the real caller, `test_gh961_invisible_text_scan.py`:

| | result on 1.25.2 |
|---|---|
| main `7b3b3c09` | 29/29 pass |
| PR before this round | 12 fail, including `test_e2e_provider_absent_is_never_trusted_success` (`assert 'success' != 'success'`) |
| PR after this round | 8 fail, **none of them a silent trust**; `test_e2e_provider_absent_is_never_trusted_success` passes |

The 8 remaining failures are all the new `ValueError`, on a version now below the declared
floor: they pin exact verdicts that socr deliberately no longer produces there. Refusing is
the intended behaviour for an unsupported wheel; the alternative is the silent wrong
`success` above. Those tests were not edited.

## Soft, also applied: image payloads out of the text dict

`get_text("dict")` kept `TEXT_PRESERVE_IMAGES`, so on exactly the pages #961 targets — a
full-page raster — MuPDF carried the image into the result, which the helper never reads.
Now `flags=fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_IMAGES`.

Measured here, three Fed scans, 12 pages each, median per page:

| document | default | images off |
|---|---:|---:|
| 1998-02-04 transcript | 61.8 ms | 3.9 ms |
| 1967-05-02 historical minutes | 50.7 ms | 3.6 ms |
| 1988-09-20 beige book | 57.6 ms | 3.7 ms |

About 15x, or roughly 50 ms per raster page. At the Fed re-OCR's scale (~350,000 pages)
that is hours of CPU for bytes nothing reads. The reviewer measured a larger gap on a flate
scan of theirs (105 ms -> 0.7 ms, ~7 MB per page); same direction, and the image payload
here was ~0.1 MB per page, so the magnitude depends on how the scan is encoded. **Pinned as
a flag difference, not a timing test** — a wall-clock threshold in CI would be a magic
number measuring the runner's load.

## Evidence after this round

31 tests in the #1189 file. **10 external mutants, all killed**, each by a named test —
including accepting an `alpha`-less span (M8), dropping the version floor (M9), and
reaching `get_texttrace` via `getattr` (M7), which the regex guard this replaced would have
missed. The harness aborted rather than reporting a false kill three times during this
ticket: twice on a stale anchor after the source moved, once on a substring collision that
would have mutated a different function.
