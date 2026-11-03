# Mission: new upper bound for Turán's pure power sum constant C42

You are working autonomously in a Claude Code cloud session. The owner is not watching.
Work carefully, commit often, and never overstate a result.

## The problem

C42 = limsup_{n→∞} R_n, where
R_n = min over z_1..z_n ∈ ℂ with max_i |z_i| = 1 of max_{1≤k≤n} |Σ_i z_i^k|.
(Equivalent normalization: z_1 = 1.) Also Erdős problem #519.
Table entry: https://teorth.github.io/optimizationproblems/constants/42a.html

## State of the art (as of 3 Nov 2026) — the number to beat is 0.68637

| Bound | Who | Method | Source |
|---|---|---|---|
| 0.69368 | Harcos (in Biró 2000) | linearization, constant phase | refereed |
| 0.6906538* | Griego | two-block coefficient cancellation | github.com/sebastian-griego/turan-c42-certificate (v1.0.0) |
| 0.688983* | Röhrig | refined step profile | teorth/optimizationproblems PR #184 |
| 0.6888863712* | Pinho | 256 middle blocks | github.com/rpinho/c42-certificate |
| 0.6873358063* | Koshta | first-block cut-off τ = 41/200, T3/T4 correction terms | github.com/DennisKoshta/c42-turan-power-sum-certificate (current table record) |
| **0.68637*** | Bakeas | Harcos linearization with **phase varying from the start** (τ = 0), s_k = T·e^{iφ(k/n)}, φ piecewise linear with 24 pieces | **github.com/Bakeas/turan-c42-bound v1.1.0** (PR #302 open, for 0.6865) |

Best lower bound: just above 1/2 (Biró). Numerics (Cheer–Goldston) suggest C42 ≈ 0.7 for small n, so the true value is unknown.

A new record means a rigorously certified upper bound **strictly below 0.68637**.
Re-check the table page and the open PRs at the start of every session; if someone posted a better bound, that becomes the target.

## Bakeas framework in brief (read NOTES.md in his repo for the real thing)

1. Harcos's linearization: prescribe the first ⌊n/2⌋ power sums freely; the rest can be chosen with a common modulus F_n so that z = 1 is in the tuple. So R_n ≤ max(T, F_n), with T = largest prescribed modulus.
2. Prescribe s_k = T·e^{iφ(k/n)}.
3. Theorem A: F_n → F, the solution of an explicit linear integral equation; the proof needs sup|d| < 1 (a convolution operator is a contraction).
4. Arb (python-flint) ball arithmetic certifies F ≤ 0.68637.

## Directions, in priority order (you may reorder with justification in LOG.md)

A. **Richer profile, same proof.** More phase pieces (48, 96, 192…), or splines / Fourier phase, optimized jointly with T. Goal: drive the fixed point where F = T lower. Use a fast float evaluator of F plus gradient-based optimization (autodiff or adjoint), then L-BFGS / CMA-ES polish.
B. **Let the modulus vary too.** Prescribe s_k = T·ρ(k/n)·e^{iφ(k/n)} with 0 ≤ ρ ≤ 1. Since R_n ≤ max(max_k |s_k|, F_n), this is a strict generalization *if* the proof of Theorem A still goes through. Check this carefully in NOTES.md before investing compute; write out exactly which lemma uses |s_k| = T.
C. **Change the split.** Check whether the linearization allows prescribing ⌊βn⌋ sums with β ≠ 1/2, or a different correction structure. Only pursue if the math genuinely allows it.
D. **Hybrid.** Combine the varying-phase start with Koshta's coefficient-cancellation correction terms.

Always confirm in float, at finite n, with Bakeas's `crosscheck/finite_n.py` (or your own exact finite-n construction) that a candidate's finite-n values approach the claimed limit. A limit formula that disagrees with finite n means a bug.

## Rigor rules (non-negotiable)

- Floats are for search only. A bound is claimed only after a rigorous Arb check prints PASS with margin, at a resolution high enough that the result is stable when M is doubled.
- Every hypothesis of Theorem A (the contraction condition sup|d| < 1, positivity of denominators, etc.) must be checked rigorously for the new profile, not assumed. Note: Bakeas v1.0.0 had a bug where denominator positivity was not checked — check everything his checker checks, and more.
- If the proof changes at all (direction B, C or D), write the full modified proof in `NOTES.md`, mark every change, and state honestly what is new and unreviewed.
- The final certificate needs two independently written checkers. The second must be written in a separate session from NOTES.md and the certificate format only, without reading the search or first-checker code.
- Include corruption tests: perturbed profiles and a slightly-too-small claimed bound must FAIL.
- Never round a bound in your favor. State the claimed bound as an exact rational slightly above the certified value.

## Working habits

- Log every experiment in `LOG.md` (date/time, direction, parameters, float F, verdict). Keep it terse.
- Save the best candidate so far to `results/best_profile.json` with its float F, and commit + push at least every 30–45 minutes, and immediately whenever the best improves. Sessions can end without warning; uncommitted work is lost.
- Run long optimizations as background processes that checkpoint to disk, so a timeout doesn't lose progress.
- Use exact decimals / rationals in certificate files.
- Prefer small, fast experiments first; scale up only what shows promise.
- If after a solid effort a direction gives nothing, write down why in LOG.md and move on.
- Be honest in all summaries: "no improvement found" is an acceptable result.

## Setup

```
pip install python-flint==0.9.0 numpy scipy mpmath jax jaxlib
git clone https://github.com/Bakeas/turan-c42-bound third_party/bakeas
git clone https://github.com/DennisKoshta/c42-turan-power-sum-certificate third_party/koshta
```
`third_party/` is for reading and reproduction; credit all prior work (MIT-licensed code may be reused with attribution).

## Final deliverables (repo root)

- `certificate/` — exact profile JSON + SHA-256
- `checkers/rigorous_check.py`, `checkers/independent_check.py`, expected outputs, `checkers/corruptions/`
- `NOTES.md` — statement, full proof (with changes from Bakeas marked), exactly what the programs check
- `README.md` — claim, how to verify (commands + runtime), credits, AI-use statement
- `PR_DRAFT.md` — text for a pull request to teorth/optimizationproblems updating constants/42a.md: new upper-bound row with asterisk, reference entry, contribution/AI-use note crediting Harcos, Griego, Röhrig, Plata, Pinho, Koshta and Bakeas. Follow the repo's CONTRIBUTING conventions (read them first).
