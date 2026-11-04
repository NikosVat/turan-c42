# Session prompts, round 2 (after S1)

Model: Opus 5.5. Every session starts from `main`. Credits expire 4 Nov, so aim to finish S7 by tomorrow early afternoon.
Before starting, note your remaining credit. Rough split of what's left: S2 30% · S3 20% · S4 10% · S5 15% · S6 15% · S7 5% · buffer 5%.

Target: certified upper bound strictly below 0.6863607950 (Pinho, [PR #305](https://github.com/teorth/optimizationproblems/pull/305)).

---

## S2 — Tighter certifier (START NOW, most important)

Read CLAUDE.md, MATH.md, PLAN.md, LOG.md. First run `nproc; free -g; df -h` and log the machine's resources. Everything you build must fit this machine.

Goal: a rigorous checker that certifies F for a given profile to within ≤ 5×10⁻⁹ of its true value, using the RAM available here. Pinho needed N = 14.2M and 95 GB for 3×10⁻⁸. We need ≥ 10× better accuracy per unit of compute. Work on branch `certifier`.

Candidate techniques. Pick by expected payoff, and log your reasoning:
1. A posteriori / validated numerics. Solve the integral equation in floats at moderate N. Then bound the residual of the continuous equation rigorously in Arb. Then convert the residual into an error bound on F through the contraction constant: ‖error‖ ≤ ‖residual‖ / (1 − q), with q computed rigorously. This avoids solving in Arb at huge N.
2. Per-cell derivative bounds (third or higher derivative on each cell) instead of a single global bound.
3. Higher-order quadrature with rigorous remainder terms, using the profile's piecewise smoothness (knots aligned with cell edges).
4. Exploit the convolution / Toeplitz structure, e.g. FFT with rigorous rounding-error bounds.

Every hypothesis of Theorem A must still be checked rigorously: contraction, denominator positivity, and everything Bakeas's and Pinho's checkers check. Never weaken a check to gain speed.

Validation gates, in order (log each one):
(a) certifies Bakeas's 0.68637 profile;
(b) certifies Pinho's [PR #305](https://github.com/teorth/optimizationproblems/pull/305) profile at 0.6863607950 using far less memory than 95 GB;
(c) shows how tight it is: the certified bound minus the float value, as a function of N;
(d) corruptions FAIL: a perturbed profile, and a claim 1×10⁻⁹ below the true F.

Commit and push after each gate. Final message: tightness reached, memory and runtime, and whether a record of about 2–4×10⁻⁸ below Pinho is certifiable.

---

## S3 — Direction A search (start NOW, in parallel with S2)

Read CLAUDE.md, MATH.md, PLAN.md, LOG.md. Branch `search-A`. Optimize only on the extrapolated objective (`--extrap`).

1. Finish the stopped 96-piece run (halved Pinho knots). Then try 192 pieces and free knot positions. Report the fixed point T = F for each, with a Richardson error estimate.
2. Measure saturation: gain versus number of pieces. Report the smallest profile that captures most of the gain, because certifier cost may grow with pieces.
3. Save the 2–3 best candidates of different sizes to `results/` as exact-decimal JSON, with float F and estimated error. Commit and push on every improvement and at least every 40 min.

Stop after about 2.5 h, or once there has been no improvement for 45 min. Final message: best fixed point and its margin below 0.6863607950, plus a candidate table.

---

## S4 — Check the "exactness" claim and draft the findings note (start NOW, cheap)

Read MATH.md and PLAN.md. Branch `findings`.

S1 claimed that Harcos's reduction is exact at finite n: the first ⌊n/2⌋ power sums determine the best achievable max over the rest. Also, 120 random non-smooth starts at n = 400 all reach the smooth optimum, with an edge that shrinks like n^−0.8.

1. Prove or refute the exactness claim carefully in `FINDINGS.md`. If it is exact, state precisely what the n = 400 experiments imply about R_n itself. Is the computed optimum R_n, or only an upper bound for R_n? Is the optimization global? Be rigorous about what is proved versus numerical evidence.
2. Re-run the finite-n experiments at 2–3 values of n (e.g. 200, 400, 800), with more starts if cheap, and fit the trend.
3. Draft `FINDINGS.md` as a short note. It is useful whether or not a record lands: the reduction to choosing the first half, the evidence that smooth profiles are near-optimal, and why directions B, C and D cannot help.

Commit and push. Final message: is the claim proved, and what does the evidence say about the actual value of C42?

---

## DECISION POINT (you, after S2 finishes)

- **S2 passed gate (b) and its tightness is below S3's margin** → run S5, then S6, then S7 for the record.
- **Otherwise** → skip S5 and S6. Run S7 in "findings" mode.

---

## S5 — Certify the record

Merge `certifier` and the best `search-A` candidate into branch `certificate`. Choose the smallest candidate whose margin survives certification. Set an exact rational claim just above the certified value, which must be < 0.6863607950. Run at increasing N until the result is stable. Add corruption tests. Write `NOTES.md`: the full proof, which is Bakeas's Theorem A plus the new error analysis, with every new step marked. Commit and push.

---

## S6 — Independent checker (blind)

Read ONLY `NOTES.md`, `certificate/`, and the certificate-format section of the README on branch `certificate`. Do NOT open src/, checkers/, LOG.md or other branches. Write `checkers/independent_check.py` from scratch in Arb, implementing the validation method described in NOTES.md. Run it on the certificate and on the corruptions. Review NOTES.md as a skeptical referee in `REVIEW.md`. Commit and push.

---

## S7 — Package

- **Record mode:** on `certificate`, fix anything REVIEW.md raised and rerun both checkers. Write README.md and PR_DRAFT.md as described in CLAUDE.md, credit Pinho's [PR #305](https://github.com/teorth/optimizationproblems/pull/305) as well, record the SHA-256, and open a PR into main.
- **Findings mode:** on `findings`, polish FINDINGS.md and write PR_DRAFT.md for a "comment only, no bound change" PR to constants/42a.md. Many such PRs are already open in that repo. Summarize the structural result and the numerical evidence, credit Bakeas and Pinho, and add an AI-use statement.

