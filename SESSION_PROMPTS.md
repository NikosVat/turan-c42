# Session prompts — paste one per cloud session at claude.ai/code

Model: Claude Opus 5.5. Repo: your `turan-c42` repo (CLAUDE.md is loaded automatically).
Rough budget out of $100: S1 ≈ $12 · S2–S4 ≈ $50 total · S5 ≈ $15 · S6 ≈ $10 · S7 ≈ $5 · buffer ≈ $8.
Watch the credit meter; if it's burning faster, cut S4.

---

## S1 — Reproduce and build the evaluator (run first, alone)

Read CLAUDE.md. Then:
1. Check https://teorth.github.io/optimizationproblems/constants/42a.html and the open PRs at github.com/teorth/optimizationproblems/pulls for anything better than 0.68637. Update CLAUDE.md's target if so.
2. Install deps, clone Bakeas's repo into third_party/, and reproduce PASS for c42_profile_0.68637.json with both his checkers at M=2560. Record runtimes in LOG.md.
3. Read his NOTES.md fully. Write `MATH.md`: a precise summary of the linearization, the integral equation for F, and every hypothesis Theorem A uses (mark which ones depend on |s_k| = T exactly, on the 1/2 split, and on piecewise linearity).
4. Build `src/evaluator.py`: a fast float (and differentiable, JAX preferred) evaluator of F for an arbitrary profile (phase φ on a grid / spline, optional modulus ρ). Validate: reproduces 0.693676 for constant phase (Harcos) and ≈0.68636 for Bakeas's profile.
5. Write `PLAN.md` ranking directions A–D from CLAUDE.md with your own assessment of which is most likely to beat 0.68637 and why.
Commit and push to main. Final message: what reproduced, evaluator accuracy, and your ranking.

---

## S2, S3, S4 — Search (start all three in parallel after S1 is on main)

Use the same text, changing only the DIRECTION line.

Read CLAUDE.md, MATH.md, PLAN.md, LOG.md. Work on branch `search-<letter>`.
DIRECTION: <A: richer phase profile, same proof | B: varying modulus ρ ≤ 1 — first check rigorously in MATH.md that Theorem A still holds | C or D: whichever PLAN.md ranked higher>
Optimize to minimize the certified quantity (the T where T = F). Start small (few parameters, low M), scale up what works. Checkpoint results/best_profile.json and commit+push every 30–45 min and on every improvement. Run a quick rigorous Arb check (moderate M) on any candidate below 0.68637 to see whether the margin survives, and test finite-n agreement.
Stop when you have ~2.5 hours of work done, or no improvement in the last hour. Final message: best float value, best rigorously-checked value, what was tried, what failed and why.

---

## S5 — Certify the winner

Read CLAUDE.md, LOG.md, and the three search branches. Pick the best candidate (or say none beats 0.68637 and stop).
Merge it into branch `certificate`. Then: choose an exact rational claim just above the value; run the rigorous checker at increasing M until stable; check every hypothesis of Theorem A rigorously; add corruption tests that must FAIL; if the proof changed, write the full modified proof into NOTES.md with changes marked. Commit and push. Final message: the exact claimed bound, the margin, and anything you are unsure of.

---

## S6 — Independent checker (fresh session — must not see search/checker code)

Read ONLY `NOTES.md`, `certificate/`, and the certificate-format section of README.md on branch `certificate`. Do NOT open src/, checkers/rigorous_check.py, LOG.md or the search branches.
Write `checkers/independent_check.py` from scratch in Arb ball arithmetic (python-flint), verifying every inequality NOTES.md says is needed. Run it on the certificate and on the corruptions. Also review NOTES.md as a skeptical referee and write `REVIEW.md` listing any gap or error. Commit and push to `certificate`.

---

## S7 — Package

On branch `certificate`: fix anything REVIEW.md found (rerun both checkers if anything changed), write README.md (claim, verify commands, runtimes, credits, AI-use statement) and PR_DRAFT.md per CLAUDE.md, record SHA-256 of the certificate, then open a pull request from `certificate` into main of this repo. Final message: claim, verification status, open concerns.
