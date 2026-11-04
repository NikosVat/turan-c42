# README.md changes for teorth/optimizationproblems (against `main`, commit 87b4ebb)

These two changes are written against the current `main` of teorth/optimizationproblems
(commit `87b4ebbd426ff8db597a2004cc45da17903bdcf2`). They do not include the changes of the open PRs
[#302](https://github.com/teorth/optimizationproblems/pull/302) (Bakeas) and [#305](https://github.com/teorth/optimizationproblems/pull/305) (Pinho). If those are merged first, the row-42 cell
below still applies as is (it then replaces `0.6863607950*`), and the Recent-progress line goes above theirs.

## 1. Row 42 of the table

Replace the line

```
| [42](https://teorth.github.io/optimizationproblems/constants/42a.html) | Turan's pure power sum constant | >0.5 | 0.69368 (0.6873358063*) |
```

by

```
| [42](https://teorth.github.io/optimizationproblems/constants/42a.html) | Turan's pure power sum constant | >0.5 | 0.69368 (0.686360751*) |
```

## 2. Recent progress

Insert as the first item under `## Recent progress`:

```
- [42](https://teorth.github.io/optimizationproblems/constants/42a.html) **improved upper bound (unrefereed):** $C\_{42} \leq 0.686360751$* by Nikolaos Vatopoulos with Claude (Anthropic), 4 Nov 2026, using the limit theorem of V. Bakeas ([PR #302](https://github.com/teorth/optimizationproblems/pull/302)) in the arbitrary-knot form of R. Pinho ([PR #305](https://github.com/teorth/optimizationproblems/pull/305)): a 33-piece phase profile and an a posteriori certifier that computes the residual of the Volterra equation exactly on every cell; two independently written Arb checkers ([note, certificate and checkers](<REPO_URL>/tree/<COMMIT>)).
```
