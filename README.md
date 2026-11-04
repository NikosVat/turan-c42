# C42 ≤ 0.686360751: a candidate upper bound for Turán's pure power sum constant

**Claim (candidate, unrefereed).**

  C42 = limsup_{n→∞} R_n ≤ C = 0.686360751 = 686360751/10⁹,

where R_n = min over z_1..z_n ∈ ℂ with max_i |z_i| = 1 of max_{1≤k≤n} |Σ_i z_i^k| (Erdős problem #519).

This is 4.40·10⁻⁸ below the best bound posted so far, 0.6863607950* (Pinho, teorth/optimizationproblems PR #305,
open), and 9.7·10⁻⁴ below the bound in the merged table (0.6873358063*, Koshta).

**Status.**
- The bound rests on Bakeas's Theorem A, with Pinho's arbitrary-knot form and integrated residual bound. To
  this it adds a new profile (33 phase pieces) and a new rigorous checker (`checkers/rigorous_check.py`, Arb
  ball arithmetic). Both are **new and unreviewed**.
- The project rules (CLAUDE.md) require a **second, independently written checker** before the bound is
  claimed publicly. That checker does not exist yet (planned for session S6). Until it exists and agrees, treat
  0.686360751 as a candidate.

## How to verify

```bash
pip install python-flint==0.9.0 numpy scipy mpmath
```

```bash
sha256sum -c certificate/SHA256SUMS
```

```bash
OPENBLAS_NUM_THREADS=1 python checkers/rigorous_check.py certificate/c42_profile_0.686360751.json --N 36864 --P 10 --refine 1
```

The last command must print `"verdict": "PASS"` and exit with code 0. Expected output:
`certificate/expected_N36864_P10.json`, with F ∈ [0.6863607487772, 0.6863607500725] and C − F_upper = 9.3e-10.
Runtime: about 6 min on one core of a 4-vCPU cloud machine, peak memory 3.7 GB.

The trial function ρ̃ is computed in floating point (GMRES), so its last bits can depend on the BLAS thread
count. Each run then certifies its own ρ̃ rigorously, so the verdict is unaffected, but the last digits of F_upper
may change. Our 4-thread and 1-thread runs at these settings differed by only 5e-22
(0.68636075007245447 vs 0.68636075007245403). At N = 18432, P = 14 they differed by 8e-11.

Stability reruns (PASS, enclosures nested inside the one above):

```bash
OPENBLAS_NUM_THREADS=1 python checkers/rigorous_check.py certificate/c42_profile_0.686360751.json --N 36864 --P 12 --refine 1
```

```bash
OPENBLAS_NUM_THREADS=1 python checkers/rigorous_check.py certificate/c42_profile_0.686360751.json --N 73728 --P 10 --refine 1
```

They give F ∈ [0.6863607494100, 0.6863607494397] (6 min, 3.8 GB) and F ∈ [0.6863607494207, 0.6863607494290]
(13 min, 6.8 GB). See NOTES.md §8.3 for all runs, including coarser settings that FAIL because the bound is
too loose.

Corruption tests (each must give the expected verdict; about 25 min):

```bash
checkers/corruptions/claim/run_claim.sh
```

```bash
checkers/corruptions/claim/run_claim.sh checkers/corruptions/claim/list2.txt
```

`python checkers/grid_check.py certificate/c42_profile_0.686360751.json 36864` checks, in exact arithmetic, that every
knot, ½ and every sum of up to three knots lies on the grid.

## Certificate format

The certificate is the single file `certificate/c42_profile_0.686360751.json` (SHA-256 in
`certificate/SHA256SUMS`). It is a JSON object. Only the following keys carry mathematical content; all other
keys (`claim`, `framework`, `exact_note`, `origin`, `checker`) are descriptive and must not affect a verdict.

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
1. 0 < T ≤ C < 1, and the knots strictly increase from 0 to ½;
2. 0 < a < 1 and κ < 1 (the hypotheses of Theorem A);
3. F < C. Equivalently, it must find a certified upper bound F_upper < C, with a certified positive lower bound
   for the denominator.

Theorem A (NOTES.md §3) then gives limsup R_n ≤ max(T, F) ≤ C. NOTES.md §4 lists valid a posteriori error
bounds for any trial function ρ̃. §§5–7 describe one way to evaluate everything rigorously, the way
`checkers/rigorous_check.py` does it. An independent checker may use any rigorous method.

**Reference values** (from `checkers/rigorous_check.py`, for comparison only):
- a = Re γ = 0.10165403…;
- κ = T (attained on (½,1]; the largest |d| on (0,½] is 2T sin(E/2) with E = max|ψ_j − ψ_0| = 0.8709 < π);
- |p̃(1)| ≈ 0.2201694;
- ∫_0^{1/2}|p| dt/(1−t) ≈ 0.3207780;
- F ∈ [0.6863607494207, 0.6863607494290] (N = 73728, P = 10, the tightest enclosure).

## Files

- `NOTES.md`: statement, the full proof (Bakeas's Lemmas 1–3 and Theorem A, Pinho's integrated form, the new
  certifier analysis, with new steps marked), exactly what the program checks, and the certificate data.
- `certificate/`: the profile JSON, `SHA256SUMS`, and expected checker outputs.
- `checkers/rigorous_check.py` (+ `spectral_witness.py`, a floating-point witness only): checker #1.
- `checkers/corruptions/claim/`: corrupted versions of the certificate that must FAIL, and the runner.
- `checkers/tests/`: validation against direct mpmath quadrature and the exactly solvable constant-phase case.
- `NOTES_CERTIFIER.md`: the certifier derivation and its validation log (session S3).
- `results/SATURATION.md`, `results/candidate_*.json`: the float search (session S2).
- `LOG.md`, `PLAN.md`: experiment log and plan.

## Credits

- **G. Harcos**: the linearization (Lemma 2) and the constant-phase bound 0.69368 (in Biró 2000).
- **A. Biró**: the first proof that C42 < 1, and the lower bound > 1/2.
- **S. Griego, A. Röhrig, Plata, R. Pinho, D. Koshta**: the 2026 certified block constructions in the
  Optimization Constants repository.
- **Bakeas** (turan-c42-bound v1.1.0, MIT): Theorem A, Lemma 3, the phase-varying profile with τ = 0, and the
  checkers this work started from.
- **R. Pinho** (c42-certificate @009120a, MIT): quadratic knots, the integrated (Gronwall) residual bound, the
  arbitrary-knot form of Theorem A, and the previous record 0.6863607950.
- No code from these repositories is copied into `checkers/`. `src/finite_n_check.py` imports Bakeas's
  `crosscheck/finite_n.py` (MIT) for a float cross-check only.

## AI use

All search code, the checker, its derivation and these texts were produced by AI agents (Claude Code,
Anthropic), working autonomously in cloud sessions directed by the repository owner. The owner did not check
the mathematics by hand. No human mathematician has reviewed the new parts (NOTES.md §§5–7).
