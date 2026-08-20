# Rebasing the stack after #251 was squash-merged

2026-08-20 ~09:35. Owner merged #251 to `main` (squash → `dc0f13f`) and sent the
other four back for round 2.

## Why a plain rebase would have been wrong

`#251` was **squash**-merged, so its branch commit `b562acc` is **not** an ancestor of
the new `main` — the change is in `main` under a different SHA. A plain
`git rebase main fix/225-…` would therefore have tried to replay `b562acc` on top of a
`main` that already contains its content: at best a duplicate commit, at worst a
conflict, and in either case a diff that no longer says what the PR is about.

The correct form drops the already-merged commit explicitly:

    git rebase --onto origin/main b562acc fix/225-phantom-image-urls

and then, up the stack, each branch is replayed off the commit it was actually cut at
(its merge-base with its predecessor), onto the predecessor's **new** head.

## What was done

| branch | cut at | rebased onto | old head | new head |
|---|---|---|---|---|
| `fix/225-phantom-image-urls` | `b562acc` | `origin/main` (`dc0f13f`) | `4243212` | **`7696719`** |
| `fix/205-tr3-auditevent` | `4200171` | new 225 | `83f6ac8` | **`83924e9`** |
| `fix/195-197-198-destruction-check` | `83f6ac8` | new 205 | `163ffcb` | **`df4b24a`** |
| `fix/222-probe-interface` | `163ffcb` | new 195 | `2fe7355` | **`a157f76`** |

All four rebased with **zero conflicts**. Force-pushed with `--force-with-lease`
(authorised by the owner for these overnight branches only).

## Two problems this fixed for free

1. **`#253`/`#254`/`#255` were branched off `4200171`** — before `#252`'s round-2
   content-loss fix — so none of them contained it. Replaying them onto the new `225`
   head carried the fix up the stack automatically. Verified: all four branches now
   contain both merged `#161` and the `gh225` guard test.
2. **`#252` now targets `main`**, so for the first time in this stack a PR above the
   bottom gets **real CI**. `ci.yml` only triggers on `pull_request: branches: [main]`.

Suite on the rebased top of stack (`fix/222`, which contains everything):
**1854 passed, 3 xfailed, 0 failed**, canary OK.

## Consequence for the non-vacuity proofs

The pinned baseline `53b0637` is no longer the right reference for these PRs. `#252`'s
baseline is now `main` = `dc0f13f`; each stacked PR's baseline is its own base branch.
The proof obligation is unchanged in substance — the new test must fail without the
change and pass with it — but the commit it is run against has moved, and each PR body
must say which one it used.

The `ci.yml` trigger widening recommended in
`2026-08-20_stacked-prs-get-no-ci.md` is still worth doing: this rebase fixed CI for
`#252` only, and the same problem returns for `#253`–`#255` until each one reaches the
bottom of the stack.
