# Goal: the archive is done

Agreed with the owner on 2026-10-08. This file replaces every open plan as the thing
socr works towards. If work is not on this page, it waits.

## What "done" means

1. **The 45 archive papers run on the Bocconi cluster** with the current recipe: Qwen3-VL-30B
   reads, Qwen3.6-27B checks, both served by vLLM. The job script is
   `/scratch/3179349/jobs/socr-archive.sbatch`.
2. **The run passes its health check.** Zero "No engines available", the checker served
   requests, and zero model-404s.
3. **The owner looks through the papers in the side-by-side view** (PDF page beside the text
   socr wrote): `~/.local/state/socr-housekeeping/pdf-vs-md/`, rebuilt with
   `uv run build-pdf-vs-md <out-dir>`.
4. **No wrong number ships unlabelled.** A table or page socr cannot confirm is withheld or
   marked "unverified".
5. **Accepted papers are promoted** into `~/papers/text` with `socr library`, never by hand.

## The one rule from now on

**Fix only what the owner can see in the side-by-side view.** A problem found by a review
round, a synthetic test or a reviewer's hypothetical does not get built unless it shows up
on a real page of a real paper. When it does, the issue names the paper and page.

## What stops

- New review-round tickets that no real page shows.
- Detector and threshold tuning on old samples.
- New engines or models, unless a page in the view needs one.

## Cleanup (before the archive run)

- **Issues.** Close or label `parked` every open issue that does not name a real paper and
  page seen to fail. Keep the rest, sorted by what the owner sees.
- **Worktrees.** Remove every worktree whose branch is merged or abandoned. Keep only work in
  progress.

## In progress when this was written

- Keep a page's correct prose as text when only its table fails, if the prose matches the
  page's own text layer. Seen on Forsythe-Lundholm 1990, p25.

## Open questions for the owner

- **Tables on scanned pages that cannot be checked:** withhold them, or ship them marked
  "unverified"?
- **Running on the cluster:** equation recovery and figure descriptions call the Mac's Ollama,
  so they do not run there. Is that acceptable for the archive run?
