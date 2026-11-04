# README.md changes for teorth/optimizationproblems (on top of PR #305, commit bac1d85)

## 1. Row 42 of the table

Replace the line

```
| [42](https://teorth.github.io/optimizationproblems/constants/42a.html) | Turan's pure power sum constant | >0.5 | 0.69368 (0.6863607950*) |
```

by

```
| [42](https://teorth.github.io/optimizationproblems/constants/42a.html) | Turan's pure power sum constant | >0.5 | 0.69368 (0.686360751*) |
```

(On current `main`, commit 87b4ebb, the line to replace still shows `0.69368 (0.6873358063*)`; PR #305 changes it
to `0.6863607950*`. The replacement is the same in either case.)

## 2. Recent progress

Insert as the first item under `## Recent progress` (above the PR #305 line):

```
- [42](https://teorth.github.io/optimizationproblems/constants/42a.html) **improved upper bound (unrefereed):** $C\_{42} \leq 0.686360751$* by N. Vatopoulos with Claude (Anthropic), 4 Nov 2026, using Bakeas's Theorem A: a 33-piece phase profile and an a posteriori certifier that computes the residual of the Volterra equation exactly on every cell; two independently written Arb checkers ([note, certificate and checkers](<REPO_URL>/tree/<COMMIT>)).
```
