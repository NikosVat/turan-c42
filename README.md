# C42 ≤ 0.686360751: an upper bound for Turán's pure power sum constant

**Claim (unrefereed).**

  C42 = limsup_{n→∞} R_n ≤ C = **686360751/1000000000** = 0.686360751,

where R_n = min over z_1, …, z_n ∈ ℂ with max_i |z_i| = 1 of max_{1≤k≤n} |Σ_i z_i^k| (Erdős problem #519;
[table entry 42a](https://teorth.github.io/optimizationproblems/constants/42a.html)).

- The bound is 4.40·10⁻⁸ below the best bound posted so far, 0.6863607950* (Pinho, teorth/optimizationproblems
  [PR #305](https://github.com/teorth/optimizationproblems/pull/305), open). It is 9.249·10⁻⁶ below Bakeas's 0.68637* ([PR #302](https://github.com/teorth/optimizationproblems/pull/302), open) and 9.750553·10⁻⁴ below the value in
  the merged table, 0.6873358063* (Koshta). These are exact differences of the decimals.
- On 4 Nov 2026 no open or merged pull request in teorth/optimizationproblems claimed a lower upper bound for C42.

**Status.**
- Two independently written rigorous checkers certify the bound (Arb ball arithmetic, python-flint 0.9.0).
- The proof is Bakeas's Theorem A, with Pinho's arbitrary-knot form and integrated residual bound, plus a new
  a posteriori certifier. It is written out in full in `NOTES.md`, with every new step marked.
- An AI referee reviewed it (`REVIEW.md`) and found no mathematical error. **No human mathematician has checked
  the argument.**

## Certified results (tightest first)

All runs are on the certificate below (SHA-256 `d015a829…5652`), with the checker versions in
`certificate/SHA256SUMS`. F is the limit functional of Theorem A, and PASS means that F < C is proved.

| checker | N | P | verdict | certified F_upper | C − F_upper | certified F_lower | time | peak memory |
|---|---|---|---|---|---|---|---|---|
| #2 | 73728 | 12 | PASS | 0.6863607494249 | 1.575e-09 | – | 10.2 min | 2.9 GB |
| #1 | 73728 | 10 | PASS | 0.6863607494291 | 1.571e-09 | 0.6863607494206 | 13.0 min | 6.8 GB |
| #2 | 36864 | 12 | PASS | 0.6863607494323 | 1.568e-09 | – | 3.9 min | 1.5 GB |
| #1 | 36864 | 12 | PASS | 0.6863607494397 | 1.560e-09 | 0.6863607494100 | 6.6 min | 3.8 GB |
| #1 | 36864 | 10 | PASS | 0.6863607500725 | 9.275e-10 | 0.6863607487772 | 6.9 min | 3.6 GB |
| #2 | 36864 | 10 | PASS | 0.6863607500744 | 9.256e-10 | – | 2.9 min | 1.3 GB |

- Checker #1 also certifies a lower bound F_lower (a diagnostic; the claim uses only F_upper). Its tightest
  enclosure is F ∈ [0.6863607494206, 0.6863607494291], of width 8.38e-12.
- Every checker #2 upper bound lies inside that interval or above it. The margin C − F ≈ 1.6·10⁻⁹ is about 190
  times the enclosure width.
- Times: one core of a 4-vCPU cloud machine, measured with up to four jobs running at once (session S7). With
  less contention runs are faster; see the timed reproduction below.
- Once converged, refining moves F_upper by about 10⁻¹¹. Checker #2 moves by 7.4·10⁻¹² from N = 36864 to 73728
  at P = 12. Checker #1 moves by 1.1·10⁻¹¹ from (N, P) = (36864, 12) to (73728, 10).
- The N = 36864, P = 10 runs are the loosest PASSes (margin 9.3·10⁻¹⁰), but they are the cheapest to reproduce.
  At that setting the two checkers' upper bounds agree to 2·10⁻¹².

## Method

Following Harcos and Bakeas, prescribe the first ⌊n/2⌋ power sums as s_k = T e^{iφ(k/n)}, with T = C and φ
piecewise linear through 34 exact rational knots in [0, ½]. Harcos's linearization then chooses the remaining
power sums with a common modulus F_n so that 1 is one of the z_i, which gives R_n ≤ max(T, F_n). Bakeas's
Theorem A shows that F_n → F = |p(1)| / ∫_0^{1/2} |p(t)| dt/(1−t). Here p is built from the bounded solution ρ of
the Volterra equation t ρ(t) = d(t) + ∫_0^t d(u) ρ(t−u) du, with d = γ − T e^{iφ} on (0, ½]. The hypotheses are
sup|d| < 1, which makes the operator a contraction, and Re γ ≥ 0. The 33-piece profile was found by a
floating-point adjoint-gradient search; its knots are j²/1152, six dyadic knots near 0 and four greedy
insertions, and its fixed point T = F is ≈ 0.6863607503. The new certifier takes a trial function ρ̃ that is a
discontinuous piecewise polynomial on a uniform grid aligned with all knots. It computes the residual of the
Volterra equation *exactly* on every cell, by an identity for convolutions of piecewise polynomials, and turns it
into rigorous bounds on p(1) and on the denominator with the a posteriori bounds of Bakeas and Pinho; the kernel
singularity is handled in closed form. This certifies F to about 10⁻¹¹ with under 7 GB of memory. For
comparison, Pinho's certificate needed 14 million cells and 95 GB for a one-sided gap of 3·10⁻⁸.

## How to verify

Requirements: Python ≥ 3.10 and python-flint 0.9.0. The time and peak memory of each command are given with
it; the two quick settings need less than 4 GB.

```bash
pip install python-flint==0.9.0 numpy scipy mpmath
```

```bash
sha256sum -c certificate/SHA256SUMS
```

Checker #2 (blind, independent), quick setting. About 4 min and 1.5 GB:

```bash
OPENBLAS_NUM_THREADS=1 python checkers/independent_check.py certificate/c42_profile_0.686360751.json --N 36864 --P 12 --out result2.json
```

Checker #1, quick setting. About 6.5 min and 3.6 GB:

```bash
OPENBLAS_NUM_THREADS=1 python checkers/rigorous_check.py certificate/c42_profile_0.686360751.json --N 36864 --P 10 --refine 1 > result1.json
```

- Each checker prints `PASS` and exits with code 0.
- Checker #1 prints a JSON object with `"verdict": "PASS"`, and checker #2 prints a final line
  `PASS: F <= … < C = 686360751/1000000000`.
- The expected outputs are `certificate/expected_N36864_P10.json` (checker #1) and
  `checkers/independent_out/cert_N36864_P12.json` (checker #2).

Timed reproduction from a fresh clone (session S7). The clone was of commit `e6ffc94`; later commits change only
documentation and recorded outputs. The commands were run one after another as above, while two other
single-threaded jobs used two of the four cores:
- `sha256sum -c`: all four files OK.
- Checker #2: PASS in 238 s, peak 1.50 GB. `F_ball` is identical to `checkers/independent_out/cert_N36864_P12.json`.
- Checker #1: PASS in 393 s, peak 3.64 GB. F_upper, F_lower and p̃(1) are identical to
  `certificate/expected_N36864_P10.json`.

Tightest settings (the headline numbers above):

```bash
OPENBLAS_NUM_THREADS=1 python checkers/rigorous_check.py certificate/c42_profile_0.686360751.json --N 73728 --P 10 --refine 1 > result1_tight.json
```

```bash
OPENBLAS_NUM_THREADS=1 python checkers/independent_check.py certificate/c42_profile_0.686360751.json --N 73728 --P 12 --out result2_tight.json
```

They took 13.0 min / 6.8 GB and 10.2 min / 2.9 GB. Checker #1's other stability run is
`--N 36864 --P 12` (6.6 min, 3.8 GB).

**Corruption tests.** Each must give the expected verdict, and each script exits nonzero on a mismatch.

```bash
checkers/corruptions/claim/run_claim.sh
```

```bash
checkers/corruptions/claim/run_claim.sh checkers/corruptions/claim/list2.txt
```

```bash
checkers/corruptions/claim/run_claim.sh checkers/corruptions/claim/list3.txt
```

```bash
checkers/corruptions/run_all.sh
```

```bash
python checkers/independent_corruptions.py --jobs 3
```

- The first three are checker #1 on corrupted copies of the certificate, 2 jobs × 3.9 GB, about 6 min per FAIL
  or PASS case. They cover lowered T = C, perturbed phases, the claim 1e-9 below and above the certified
  enclosure, C < T, C = 1, a < 0, a misaligned knot, and malformed knots or phases.
- The fourth is checker #1 on corrupted Bakeas and Pinho profiles (about 15 min).
- The last is checker #2 on all of them (28 cases, about 25 min with 3 jobs).
- Summaries are in `checkers/corruptions/claim/out/summary_list*.txt`, `checkers/corruptions/run_all_out.txt` and
  `checkers/independent_out/corruptions/SUMMARY.txt`.

**Self-tests:**
- `python checkers/independent_selftest.py` (15 s) checks checker #2 against direct mpmath quadrature and the
  exactly solvable constant-phase case.
- `python checkers/tests/harcos_exact.py` gives the exact F of `checkers/tests/profiles/harcos_const.json`.
  `python checkers/rigorous_check.py checkers/tests/profiles/harcos_const.json --N 256 --P 8 --refine 1` encloses
  it.
- `python checkers/grid_check.py certificate/c42_profile_0.686360751.json 36864` checks, in exact arithmetic, that
  every knot, ½ and every sum of up to three knots lies on the grid.

**Floating point and threads.** The trial function ρ̃ comes from a floating-point solve, so its last bits can
depend on the BLAS thread count; set `OPENBLAS_NUM_THREADS=1` for bit-for-bit reproduction. Each run certifies its
own ρ̃ rigorously, so the verdict does not depend on this, but the last digits of F_upper may.

## The certificate

`certificate/c42_profile_0.686360751.json`, SHA-256
`d015a8295d809fa26c9f18888ece53d77c9fea71712420313e75911065855652` (also in `certificate/SHA256SUMS`, together with
the SHA-256 of both checkers).

### Certificate format

The certificate is a JSON object. Only the following keys carry mathematical content. All other keys (`claim`,
`framework`, `exact_note`, `origin`, `checker`) are descriptive and must not affect a verdict.

| key | type | meaning |
|---|---|---|
| `constant` | string | must be `"C_42"` |
| `T` | decimal string | the modulus T of the prescribed power sums, an exact rational (`"0.686360751"` = 686360751/10⁹) |
| `C` | decimal string | the claimed bound; must satisfy T ≤ C < 1. Here C = T. |
| `breaks` | list of K+1 strings | the knots u_0 < u_1 < … < u_K, exact rationals written `"p/q"` or `"0"`; u_0 = 0, u_K = 1/2. Here K = 33 and the least common denominator is 18432. |
| `phases` | list of K+1 decimal strings | ψ_0, …, ψ_K, exact rationals (finite decimals, 16 significant digits), in radians |

Every number is exact. A decimal string such as `"-1.422143609677031"` denotes exactly the rational
−1422143609677031/10¹⁵, and `"1/18432"` denotes 1/18432. Implementations must parse them as exact rationals
(e.g. Python `fractions.Fraction(s)`), never through a binary float.

**What the certificate asserts.** Let φ: [0,½] → ℝ be the continuous piecewise-linear function with
φ(u_j) = ψ_j. Put g(u) = T·e^{iφ(u)}, γ = g(0), a = Re γ, and

  d(u) = γ − g(u) for 0 < u ≤ ½,   d(u) = γ for ½ < u ≤ 1,   d(0) = 0,   κ = sup_{0<u≤1} |d(u)|.

Let ρ be the bounded solution on (0,1] of t ρ(t) = d(t) + ∫_0^t d(u) ρ(t−u) du, and put

  p(t) = t^{−γ} + ∫_0^t (t−v)^{−γ} ρ(v) dv,   F = |p(1)| / ∫_0^{1/2} |p(t)| dt/(1−t).

A checker must prove, rigorously (e.g. in interval or ball arithmetic):
1. 0 < T ≤ C < 1, and the knots strictly increase from 0 to ½.
2. The hypotheses of Theorem A. Theorem A needs κ < 1 and a = Re γ ≥ 0; then a ≤ T < 1 automatically. Both
   checkers certify the stronger condition 0 < a < 1, together with κ̂ < 1 for a number κ̂ ≥ κ. Any certified
   upper bound κ̂ < 1 for κ may be used (NOTES.md §3, remark on κ̂).
3. F < C. Equivalently, it must find a certified upper bound F_upper < C, with a certified positive lower bound
   for the denominator.

Theorem A (NOTES.md §3) then gives limsup R_n ≤ max(T, F) ≤ C. NOTES.md §4 lists valid a posteriori error bounds
for any trial function ρ̃. §§5–7 describe one way to evaluate everything rigorously, the way
`checkers/rigorous_check.py` does it. An independent checker may use any rigorous method.

**Reference values** (for comparison only):
- a = Re γ = T cos ψ_0 = 0.10165403756107…;
- κ = T, attained on (½,1]. The largest |d| on (0,½] is 2T sin(E/2) = 0.57902, with
  E = max|ψ_j − ψ_0| = 0.87087 < π;
- |p̃(1)| ≈ 0.2201694;
- ∫_0^{1/2}|p| dt/(1−t) ≈ 0.3207780;
- F ∈ [0.6863607494206, 0.6863607494291] (checker #1, N = 73728, P = 10, the tightest enclosure).

## Findings (summary of FINDINGS.md)

Session S4 studied the finite-n problem itself (`FINDINGS.md`; float numerics except where stated):
- **Proved:** Harcos's reduction is exact at every finite n.
  - R_n = min over first halves s ∈ ℂ^{⌊n/2⌋} of max(max_k|s_k|, F_n(s)).
  - Equivalently, R_n is the fixed point of T ↦ min_{max|s_k| ≤ T} F_n(s).
  - So every construction (any split point, Koshta-type correction terms, a varying modulus) is dominated by the
    completion of its own first half.
- **Rigorous, finite n only:** Arb-certified upper bounds such as R_400 ≤ 0.690120598 and R_3200 ≤ 0.687129253.
- **Numerical evidence (n = 200 to 3200):**
  - about 2,000 local searches, including fully non-smooth and varying-modulus starts, find a single basin at
    constant modulus;
  - the optimum is the smooth limit profile plus a boundary layer whose gain decays like n^{−0.9};
  - extrapolation puts lim R_n at about 0.68636 ± 2·10⁻⁵, the value of the smooth phase-only family.
- **Consequence (heuristic):** if these local optima are global, then C42 ≈ 0.68636, so the present upper bound
  would be within about 10⁻⁵ of the truth. The directions "varying modulus", "different split" and "hybrid with
  correction terms" would then add nothing beyond the phase-only family.
- **Not proved:** any new lower bound. The best lower bound is still Biró's, just above ½.

## Files

- `NOTES.md` contains the statement and the full proof: Bakeas's Lemmas 1–3 and Theorem A, Pinho's integrated
  form, and the new certifier analysis, with new steps marked. It also says exactly what the programs check.
- `REVIEW.md` is the referee report and the description of checker #2 (session S6, written blind).
- `certificate/` holds the profile JSON, `SHA256SUMS` and the expected outputs of checker #1 (see
  `certificate/INDEX.md`).
- `checkers/rigorous_check.py` (with `spectral_witness.py`, a floating-point witness only) is checker #1.
- `checkers/independent_check.py` is checker #2, with `independent_selftest.py`, `independent_corruptions.py`
  and its outputs in `checkers/independent_out/`.
- `checkers/corruptions/` holds the corrupted certificates and the runners. `checkers/tests/` holds checker #1's
  validation scripts.
- `FINDINGS.md` and `results/finite_n/` are the finite-n study (session S4).
- `NOTES_CERTIFIER.md` is the certifier derivation and its validation log (S3). `results/SATURATION.md` and
  `results/candidate_*.json` are the float search (S2). `src/` holds the search and float cross-check code.
- `submission/` contains the proposed update to teorth/optimizationproblems, written against its current `main`
  (`42a.md`, the README changes, `PR_DRAFT.md`).
- `LOG.md` and `PLAN.md` are the experiment log and the plan.

## Credits

- **G. Harcos**: the linearization (Lemma 2 in NOTES.md) and the constant-phase bound 0.69368 (in Biró 2000).
- **A. Biró**: the upper bound 5/6 and the lower bounds 1/2 and > 1/2 (the best lower bound to date).
- **S. Griego** (0.6906538*, two-block coefficient cancellation), **A. Röhrig** (0.688983*, refined step profile,
  [PR #184](https://github.com/teorth/optimizationproblems/pull/184)), **K. Plata** (six-level step profile), **R. Pinho** (0.6888863712*, 256 middle blocks, [PR #251](https://github.com/teorth/optimizationproblems/pull/251)) and
  **D. Koshta** (0.6873358063*, first-block cut-off τ = 41/200, [PR #263](https://github.com/teorth/optimizationproblems/pull/263)): the 2026 certified block constructions
  in the Optimization Constants repository.
- **V. Bakeas** (turan-c42-bound v1.1.0, MIT; [PR #302](https://github.com/teorth/optimizationproblems/pull/302)): Theorem A, Lemma 3, the phase-varying profile with
  τ = 0, and the checkers this work started from.
- **R. Pinho** (c42-certificate @009120a, MIT; **[PR #305](https://github.com/teorth/optimizationproblems/pull/305)**): the observation that Theorem A holds for arbitrary
  knots, the quadratic knots j²/4608 (this profile uses j²/1152), the integrated (Gronwall) residual bound, and
  the previous record 0.6863607950*.
- No code from these repositories is copied into `checkers/`. `src/finite_n_check.py` imports Bakeas's
  `crosscheck/finite_n.py` (MIT) for a float cross-check only.

## AI use

- **Who did what.** This bound was found, proved and checked by Claude agents (Claude Code, Anthropic), directed
  by Nikolaos Vatopoulos. The agents wrote the profile search, the certifier and its error analysis, both
  checkers, the corruption tests, the referee report and all texts, working autonomously in cloud sessions.
- **The second checker was written blind.** Checker #2 (`checkers/independent_check.py`) and `REVIEW.md` were
  written in a separate session from `NOTES.md`, `certificate/` and the certificate-format section of this README
  only. That session did not see the search code, checker #1 or the log.
- **Limits of the independence.** The two checkers share no code, witness or intermediate data, but they share the
  mathematics of NOTES.md §§3–4. Both are AI-written, so the independence is procedural.
- **No human mathematician has checked the argument.** The director did not check the mathematics by hand.
  Before the bound is treated as established, a human should read at least Theorem A (NOTES.md §3), Lemma C1
  (§5.2) and §6.
