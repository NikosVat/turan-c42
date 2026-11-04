# LOG

Terse experiment log. F values are floats unless marked "Arb". Times are UTC.

## S1 — 2026-11-03 (reproduce + evaluator)

Environment: 4 vCPU, 15 GB RAM, Python 3.13.18, python-flint 0.9.0, jax 0.12.0.1, numpy 2.5.3, scipy 1.18.1.
teorth.github.io is **not reachable** from this sandbox (network policy). The table was read from the source
repo `teorth/optimizationproblems@87b4ebb` (constants/42a.md), and PRs were found through the GitHub search API.

### 15:45 Target check
- Merged table (42a.md @87b4ebb): best upper bound 0.6873358063* (Koshta).
- Open PRs:
  - [#302](https://github.com/teorth/optimizationproblems/pull/302) (Bakeas, 0.68637*, phase-varying first block);
  - **[#305](https://github.com/teorth/optimizationproblems/pull/305) (Pinho, 0.6863607950*)**: Bakeas's Theorem A, 48 phase pieces on knots j²/4608, sharper Arb error
    propagation, F ≤ 0.6863607774, repo rpinho/c42-certificate@009120a/volterra-quadratic-knots;
  - [#310](https://github.com/teorth/optimizationproblems/pull/310) (Pinho, lower bound 1/2 + 2^-4096*, irrelevant here);
  - [#184](https://github.com/teorth/optimizationproblems/pull/184) (Röhrig, old).
- **Target updated in CLAUDE.md: 0.6863607950.** Pinho and Bakeas both report the float optimum of the
  phase-only family as ≈ 0.686360744.

### 15:50 Reproduction of Bakeas v1.1.0 (commit dc9c491, profile SHA-256 dcea4edb…ded395, matches README)

| checker | M | N | result | F ≤ (Arb) | wall time |
|---|---|---|---|---|---|
| `rigorous_limit.py` | 2560 | 122880 | **PASS**, exit 0 | 0.6863659159547926 | 69.5 s (program reports 68.8 s) |
| `independent_check.py` | 2560 | 122880 | **PASS**, exit 0 | 0.6863614529 | 23.0 s (program reports 22.3 s) |

- Both outputs are identical to `expected_*_M2560.txt`, except for timings and the trailing digits of the p1 ball
  (radius ~1e-46).
- Logs: `logs/repro_rigorous_M2560.txt`, `logs/repro_independent_M2560.txt`.

### 16:00 Evaluator (`src/evaluator.py`) validation
- Method: product-trapezoid solve of t p = ∫ c(u) p(t−u) du with p = s^{−γ} q, plus a numpy O(N²) recursion.
  The JAX dense solve is used for N ≤ 4096. The exact discrete adjoint gradient costs ≈ 2× a forward solve.
- Harcos constant phase (exact, mpmath: 0.693676303073):
  - F_N for N = 2048…32768: 0.69367616, 0.69367605, 0.69367616, 0.69367624, 0.69367628.
  - The error is non-monotone, with terms ~N^{−(2−a)} and ~N^{−2}. The jump of ρ at ½ gives q a (t−½)^{1−γ}
    singularity.
  - Two-exponent Richardson extrapolation over (N, 2N, 4N) with exponents (2−a, 2):
    - 0.6936763062 from (4096, 8192, 16384), error 3.1e-9;
    - 0.6936763065 from (8192, 16384, 32768), error 3.4e-9.
- Bakeas profile, N = 3072·2^j, extrapolated: **0.6863600838**. The NOTES report 0.68636008 (direct solver).
- Pinho profile, N = 4608·2^j:
  - extrapolated over (9216, 18432, 36864): **0.6863607433**;
  - over (18432, 36864, 73728): 0.6863607439;
  - Pinho and Bakeas report 0.686360744.
- Gradient checks:
  - adjoint vs JAX dense autodiff (N = 768, 50 params): max difference 3.6e-16;
  - adjoint vs central finite differences: ~1e-10.
- Timing: forward numpy N = 18432 ≈ 1 s; N = 36864 ≈ 4 s.

### 16:05 Direction B, first-order test (`results/sensitivity_*.npz`)
- Quantity: the radial density r(u) = δF/δ log|g|(u), from the node-wise adjoint.
- Pinho profile (N = 9216): r < 0 on all of (0,½], from −13.4 (u→0) to −1.16 (u = ½). Its integral is
  T·∂F/∂T = −0.839.
- Bakeas profile (N = 6144): the same shape, with max −1.163.
- Verdict: **lowering the modulus anywhere raises F**. ρ ≡ 1 is KKT for ρ ≤ 1.
- Non-local check (`optimize.py --rho --rho0 0.85`, N = 4608, 98 params): F 0.81628 → 0.6863612. L-BFGS-B returns
  **every** ρ_j to its bound 1.
- Verdict: no gain from B, locally or from this start.

### 16:08 Direction A probes (Pinho start, T = 0.6863607950 fixed)
- Optimising F_N at fixed N overfits the discretisation:
  - re-opt at N = 18432 lowers F_18432 by 5e-9, but the extrapolated value rises 0.6863607439 → 0.6863607481;
  - adding 3 knots at 1/18432..3/18432 lowers F_18432 by 1.7e-8, but the extrapolated value rises to 0.6863607626.
  - **Lesson: optimise the extrapolated F.** Added `--extrap` to `src/optimize.py`.
- Extrapolated objective at base 4608, 49 knots (the refine0 knots coincided with existing ones):
  F_x 0.6863607374 → 0.6863607370 (Δ = −4e-10). Pinho's profile is stationary for its knot family.
- 96 pieces (each Pinho interval halved), extrapolated objective at base 18432: start F_x = 0.6863607439.
  ~42 s per evaluation. RUNNING in background → `results/probeA_x2_extrap.json`, `logs/probeA_x2_extrap.log`.
  Result: see the next entry.

### 16:09 Rescaling check (Lemma 1), `src/finite_n_roots.py`
- Question: does the finite-n tuple have max|z_j| > 1? Then max_k |s_k|/λ^k would beat max(T, F_n).
- Bakeas profile, n = 200, 400, 800, 1600: max|z_j| = 1.00000000. z = 1 is the largest root and all other roots
  are strictly inside.
- The power sums recomputed from the roots match the σ_k: the construction works.
- F_n = 0.70128, 0.69527, 0.69164, 0.68947, decreasing towards F ≈ 0.68636.
- Verdict: no free gain.

### 16:15 Direction E probes: exact finite-n first half, `src/finite_n_free.py`
- Setup: Harcos's reduction is exact (PLAN.md §1), so R_n = min over s_1..s_{n/2} of max(max|s_k|, F_n).
- Constant modulus T = 0.68637, all n/2 phases free, L-BFGS/BFGS from the smooth Bakeas profile:

  | n | smooth F_n | free optimum | Δ |
  |---|---|---|---|
  | 200 | 0.70127916 | 0.70020743 | −1.07e-3 |
  | 400 | 0.69526940 | 0.69465442 | −6.15e-4 |
  | 800 | 0.69163889 | 0.69128953 | −3.49e-4 |

  - Δ ∝ n^{−0.8}: a finite-size effect.
  - The difference is a boundary layer at k ≲ 20 (≈ −0.12 rad at k ≤ 4) plus a smooth correction that shrinks
    with n at fixed u.
- Multistart at n = 400, constant modulus, 120 starts (iid uniform, random smooth, random linear phases): **all**
  reach F_400 = 0.69465442. One basin. (`logs/finite_n_multistart.log`)
- Multistart at n = 400, free modulus 0 ≤ |s_k| ≤ T, 60 starts:
  - best F_400 = 0.69465442 with ρ ≡ 1, reached by 2 of 60 starts;
  - the rest are stuck at 0.736–1.12 with some ρ_k = 0;
  - nothing beats ρ ≡ 1.
- Verdict: no evidence for non-smooth first halves (E) or varying modulus (B).

### 16:20 Pinho's certification error budget (from his receipts, N = 14,155,776, 224 bits, 1110 s on 3 threads)
- sup|R/t| = 2.85e-6, with the worst cell at index 3072, i.e. t = 1/4608, the first knot. The global
  B3 = |R'''| bound is 1.49e8, dominated by the steep first pieces (first slope ≈ 195).
- The integrated and local-contraction bounds reduce the errors to: numerator 7.3e-9, denominator 3.8e-9. This
  gives F_upper = 0.6863607742 against float 0.686360744, a gap of 3.0e-8.
- Lesson for any tighter certifier: replace the global B3 by per-cell R''' bounds, and/or use a higher-order trial
  function or a graded grid near 0.

### 16:25 Evaluator `F_extrapolated` switched to the validated two-exponent fit
- Output: `logs/evaluator_validation.txt`.
- Harcos 0.6936763062 (exact 0.6936763031); Bakeas 0.6863600848; Pinho 0.6863607374 at base 4608.
- Pinho at base 9216 or 18432 gives 0.6863607433 / 0.6863607439. **Use base ≥ 9216 for profiles with knots near 1/4608.**

### 16:41 Direction A: 96-piece probe (each Pinho interval halved), verified at independent levels
- The optimizer was stopped after 11 evaluations of the extrapolated objective at base 18432: CPU contention made
  each evaluation take > 100 s. At that point |grad| = 4.3e-6 and F was still decreasing.
- Extrapolated F at fixed T = 0.6863607950:
  - optimization levels (18432, 36864, 73728): 0.6863606904;
  - **independent levels (36864, 73728, 147456): 0.6863606919**;
  - raw F_147456: 0.6863606966.
- Pinho's 48-piece profile on the same independent levels: 0.6863607440 (raw F_147456 = 0.6863607451).
- **Gain at fixed T: ΔF = −5.2e-8.** It survives the change of levels and is visible in the raw F at N = 147456.
- Fixed-point estimate, with ∂F/∂T ≈ −1.22: T* ≈ T − (T−F)/2.22.
  - 96 pieces: ≈ 0.68636075;
  - Pinho's family: ≈ 0.68636077.
  - The room below the target 0.6863607950 is therefore ~4.6e-8 before certification losses (Pinho's
    certificate lost 3e-8).
- Saved as `results/best_profile.json`: float candidate only, not certified.
- Finite-n confirmation was not possible at this precision: the finite-n error is ~1e-5 at n = 4e6.
- Next (S2): continue this run, then refine to 192 pieces and run the T fixed-point loop.

### 16:30 Spline path sanity check
- A natural cubic spline through Bakeas's 25 knot values interpolates exactly and its adjoint gradient matches
  finite differences.
- At N = 3072 it gives F = 0.68635915 vs 0.68636007 for the piecewise-linear profile, at the same T and the same
  knot values. This is consistent with the known near-0 resolution gain, not a new effect.
- Theorem A's analysis allows C² phases (H6/H7), but the checkers' closed forms (H9/H10) do not.

**S1 wall time ≈ 1 h; S1 verdict: reproduction OK; evaluator OK (≤ 3e-9 vs exact Harcos); B, C, D, E show
no gain; A has ~5e-8 of float room at fixed T, and certification is the bottleneck.**

## S3 — 2026-11-03 (A-certifier, branch `certifier`)

### 17:15 Machine
- `nproc` = 4; `free -g`: 15 GB total, 14 GB free, no swap; `df -h /`: 30 GB available.
- Everything built in this session must fit 4 vCPU / 15 GB. Pinho's N = 14.2M run needed 95 GB and 32 vCPU.

### 17:20 Choice of technique (reasoning)
- Pinho's error is O(h²) because the trial ρ̃ is piecewise linear; every doubling of accuracy costs ~1.4× N,
  and N is memory-bound (Python lists of Arb balls). Per-cell R''' bounds (technique 2) or FFT (4) alone keep h².
- Key observation: ρ is **piecewise analytic with breakpoints only at finite sums of knots and 1/2**, and for
  both profiles those are all on the knot grid (j²/4608 + k²/4608 = (j²+k²)/4608; j/48 + k/48). So a
  discontinuous piecewise polynomial of degree P on a knot-aligned uniform grid converges spectrally
  (techniques 1 + 3 combined).
- Convolutions of piecewise polynomials on a uniform grid are again piecewise polynomials, computable
  **exactly** by two Kronecker-packed polynomial products (python-flint acb_poly, outward rounded). So the
  residual R = d + d*ρ̃ − tρ̃ is an explicit polynomial per cell (d replaced by its Taylor polynomial on each
  cell with a rigorous remainder); sup|R| on a cell ≤ Σ|Chebyshev coefficients|.
- Same trick for p̃ = t^-γ + ∫(t−v)^-γ ρ̃: kernel cell 0 in closed form (Beta functions, θ^{1−γ} terms),
  kernel cells ≥ 1 by Taylor polynomials with rigorous remainder; denominator via the exact polynomial
  representation of p̃ per cell and sqrt(X) ≥ (3/2)Xu − (1/2)X²u³ (u > 0 certified), X = |p̃|².
- Propagation: Lemma 3 (sup) and the integrated Gronwall form (Pinho), min of the two.

### 17:30 Validation of the formulas (`checkers/tests/`)
- Float pipeline F from the spectral trial: Bakeas 0.68636008313–0.68636008372 (N=192..768, P=6);
  Pinho 0.6863607439575 (N=18432) / 0.6863607439536 (N=36864).
- `tests/mp_pointwise.py` (mpmath, 30 digits, direct quadrature, no convolution identity):
  Bakeas N=192 P=6: max sampled |R/t| = 1.1e-12 (cells 0–3, knots, 95/96 around 1/2);
  N=96: p̃ cell representation vs direct quadrature ≤ 1.9e-15, p̃(1) diff 1.7e-16.
- Harcos constant phase (`tests/profiles/harcos_const.json`, exact F = 0.69368452670658032 by mpmath):
  rigorous F_upper − F_exact = +5.2e-12 (N=64), +2.6e-12 (N=256), +1.5e-12 (N=1024, before weight fix).
  Two slack sources found and fixed on the way: cubic u fit near 0 (now deg 8/6/3), and the 1/(1−t) weight
  series truncated at 4 terms (now adaptive, tail < 1e-22).

### 17:45 Gate (a): Bakeas profile c42_profile_0.68637.json — PASS
- `python checkers/rigorous_check.py third_party/bakeas/c42_profile_0.68637.json --N 768 --P 6`
- **PASS, F ≤ 0.6863600829256854** (Arb), 4.2 s, residual sup|R/t| = 3.6e-12, EN = 3.2e-12, ED = 1.5e-12,
  denominator slack 1.5e-12. Bakeas's own checkers: F ≤ 0.6863659 and 0.6863615 (N = 122880).
- Note: the S1 evaluator's extrapolated value 0.6863600838 was off by ~1e-9. Its raw F_N at N=98304 is
  0.6863600830081 and still moving (complex-exponent error terms from (t−t_k)^{2−γ}); the 4-term fit over
  12288..98304 gives 0.6863600829277, i.e. 1.4e-12 above our F_upper, within that fit's accuracy. Our
  enclosure says F ∈ [F_upper − ~1e-11, F_upper].

### 17:48 Gate (b), first point: Pinho [PR #305](https://github.com/teorth/optimizationproblems/pull/305) profile
- N=18432 (4608·4), P=6: **PASS, F ≤ 0.6863607723298**, 69.5 s, max RSS 1.24 GB.
  (Pinho: F ≤ 0.6863607742 at N = 14.2M, 95 GB, 1110 s on 3 threads.)
- Error budget: residual peak 3.6e-7 in cell 4 (first cell after the knot 1/4608: the continued piece has
  a 1/t pole one knot-cell away, so convergence there is only (4s+2)^-P); EN = 7.9e-9, ED = 3.4e-9.
  Denominator slack 2.1e-12. So the gap 2.8e-8 is all residual; fix = larger s or P.

### 17:53 Gate (b): Pinho [PR #305](https://github.com/teorth/optimizationproblems/pull/305) profile — PASS, far tighter and with 2.6 GB
- `python checkers/rigorous_check.py third_party/pinho/volterra-quadratic-knots/certificates/C42_compact_best.json --N 36864 --P 8`
- **PASS, F ≤ 0.6863607448924165** (Arb), 155.5 s single-threaded, **max RSS 2.62 GB**.
  Float F of the same family (spectral float pipeline, N=18432/36864): 0.6863607439575 / 0.6863607439536.
  **Gap F_upper − F_float = 9.4e-10** (Pinho's certificate: 3.0e-8 with N = 14.2M, 95 GB, 1110 s × 3 threads).
- Budget: residual sup|R/t| = 1.74e-9 (cell 1); EN = 2.6e-10 (integrated form), ED = 1.1e-10;
  denominator slack 1.1e-12 + kernel remainder 1.6e-12; 63 knot cells by geometric subdivision.
- The residual is now float-noise-limited near t = 0 (mpmath direct quadrature on cells 0–9:
  2e-10 … 9e-10; |ρ| ≈ 134 there, P = 8 monomial conditioning), not approximation-limited.
  Next: iterative refinement of the witness with Arb-computed nodal residuals.

### 18:10 Witness refinement + rigorous lower bound F_lower
- Trial coefficients are now obtained by Arb interpolation of the float nodal values at the (exact dyadic)
  collocation nodes (midpoints taken as exact), which removes the float Vandermonde conditioning loss.
- `--refine k`: k passes of (Arb nodal residual → float GMRES correction → add to nodal values).
  Only the trial changes; every proof step is unchanged.
- Pinho, N=36864, P=8, refine=1: **PASS, F ≤ 0.6863607439516052**; sup|R/t| = 3.5e-12 (was 1.7e-9),
  EN = 4.9e-14, ED = 2.1e-14, denominator slack 1.1e-12 + kernel remainder 1.6e-12. 388 s (CPU shared),
  3.19 GB. This is *below* the float spectral estimate 0.6863607439536: the float quadrature drops
  θ ∈ [0, 2^-30] on cell 0, inflating its F by ~2e-12 (checked by size estimate). The Arb value is the reliable one.
- Added a rigorous **upper** bound for the denominator: sqrt(X) − f = s(s−1)²(s+2)/(2u) ≤ η²(1+η)(3+η)/(2u_min)
  with η = sup|u²X − 1| < 1 (certified per cell), plus the dropped geometric piece bounded by its sup.
  This gives F_lower = (|p̃(1)| − EN)/(D_up + ED), i.e. a two-sided enclosure of F.
  - Harcos exact 0.69368452670658 ∈ [0.69368452670057, 0.69368452670886] (N=256), width 8.3e-12. ✓
  - Bakeas N=1536 P=6 r=1: F ∈ [0.6863600829150, 0.6863600829174], width 2.4e-12, 8.1 s.

### 18:35 Gate (c): tightness vs N (certified enclosure F ∈ [F_lower, F_upper]; width = F_upper − F_lower)

Bakeas c42_profile_0.68637.json (`logs/cert/sweep_bakeas_*.json`):

| N | P | refine | verdict | sup R/t | EN | ED | F_upper | F_lower | width | s | GB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 96 | 6 | 0 | PASS | 5.79e-07 | 9.71e-08 | 5.02e-08 | 0.68636031375472072 | 0.68635985207547038 | 4.62e-07 | 4.8 | 0.091 |
| 96 | 6 | 1 | PASS | 5.79e-07 | 9.71e-08 | 5.02e-08 | 0.68636031375350531 | 0.68635985207669850 | 4.62e-07 | 4.7 | 0.091 |
| 192 | 6 | 1 | PASS | 8.98e-09 | 9.34e-10 | 4.81e-10 | 0.68636008514044798 | 0.68636008069053328 | 4.45e-09 | 5.9 | 0.098 |
| 384 | 6 | 1 | PASS | 1.03e-10 | 7.62e-12 | 3.92e-12 | 0.68636008293781560 | 0.68636008289389647 | 4.39e-11 | 5.0 | 0.112 |
| 768 | 4 | 1 | PASS | 4.32e-09 | 3.34e-10 | 1.72e-10 | 0.68636008371053953 | 0.68636008212174790 | 1.59e-09 | 9.6 | 0.133 |
| 768 | 6 | 0 | PASS | 3.58e-12 | 1.84e-12 | 8.43e-13 | 0.68636008292250530 | 0.68636008291043196 | 1.21e-11 | 5.5 | 0.142 |
| 768 | 6 | 1 | PASS | 9.87e-13 | 5.94e-14 | 3.05e-14 | 0.68636008291840569 | 0.68636008291378900 | 4.62e-12 | 5.6 | 0.142 |
| 768 | 8 | 1 | PASS | 2.26e-16 | 1.15e-17 | 5.87e-18 | 0.68636008291823135 | 0.68636008291394228 | 4.29e-12 | 5.3 | 0.148 |
| 1536 | 6 | 1 | PASS | 8.57e-15 | 4.61e-16 | 2.37e-16 | 0.68636008291744467 | 0.68636008291503417 | 2.41e-12 | 7.8 | 0.196 |

Pinho C42_compact_best.json (`logs/cert/sweep_pinho_*.json`; 36864/73728 rows pending at time of writing):

| N | P | refine | verdict | sup R/t | EN | ED | F_upper | F_lower | width | s | GB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 9216 | 6 | 0 | FAIL | 2.29e-05 | 8.17e-07 | 3.50e-07 | 0.68636367644701533 | 0.68635781145728377 | 5.86e-06 | 54.2 | 0.864 |
| 9216 | 8 | 1 | PASS | 2.34e-07 | 8.10e-09 | 3.47e-09 | 0.68636077302774936 | 0.68636071487079846 | 5.82e-08 | 75.6 | 0.945 |
| 18432 | 6 | 0 | PASS | 3.56e-07 | 7.89e-09 | 3.39e-09 | 0.68636077229099971 | 0.68636071560755823 | 5.67e-08 | 89.5 | 1.461 |
| 18432 | 6 | 1 | PASS | 3.56e-07 | 7.89e-09 | 3.38e-09 | 0.68636077226801144 | 0.68636071563054958 | 5.66e-08 | 130.7 | 1.605 |
| 18432 | 8 | 1 | PASS | 1.10e-09 | 2.27e-11 | 9.73e-12 | 0.68636074403471125 | 0.68636074386384734 | 1.71e-10 | 132.9 | 1.682 |
| 36864 | 8 | None | PASS | 1.74e-09 | 2.62e-10 | 1.13e-10 | 0.68636074489241650 |  | nan | 155.5 | 2.62 |
| 36864 | 8 | 1 | PASS | 3.54e-12 | 4.88e-14 | 2.09e-14 | 0.68636074395160519 |  | nan | 388.3 | 3.188 |

- Reading: with P = 8 and one refinement pass the width falls from 5.8e-8 (N=9216) to 1.7e-10 (N=18432); at
  N=36864 the residual terms are < 1e-13 and the width (~5e-12) is set by the denominator bookkeeping
  (kernel-Taylor remainder 1.6e-12, tails 1.1e-12). Doubling N moves F_upper by < 1e-11 once converged
  (Bakeas 768→1536: −1e-12). F_upper − F_float(S1 evaluator, 4-term extrapolation) is below that evaluator's
  own accuracy (~1e-11); the certified F_lower is the better reference.
- Rows without F_lower (pinho_N36864_P8*.json) were made before F_lower was added.

### 18:40 Gate (d): corruptions (`checkers/corruptions/`, runner `run_all.sh`)
| file | N, P, refine | expected | got | enclosure |
|---|---|---|---|---|
| bakeas_phase0_plus_5e-2 | 768, 6, 1 | FAIL | FAIL | F ≤ 0.68639554 > T |
| bakeas_T_0.68636 | 768, 6, 1 | FAIL | FAIL | F ≤ 0.68637231 > T = 0.68636 |
| bakeas_shift_a_negative | 768, 6, 1 | REFUSED | REFUSED (0 < a not certified) | — |
| pinho_claim_below_T (C = T − 1e-10) | — | REFUSED | REFUSED (C < T) | — |
| pinho_T_below_fixed_point (T = 0.68636076) | 36864, 8, 1 | FAIL | FAIL | F ∈ [0.68636078674847, 0.68636078675311] |
| pinho_phase3_plus_1e-2 | 36864, 8, 1 | FAIL | FAIL | F ∈ [0.68636099228135, 0.68636099228601] |
| pinho_T76_claim_1e-9_below_F (C = 0.686360785748 = F_lower − 1e-9, rounded down) | 18432, 8, 1 | FAIL | FAIL | F ∈ [0.68636078667, 0.68636078684] |
| pinho_T76_claim_1e-9_above_F (C = 0.686360787754 = F_upper + 1e-9, rounded up; resolution control) | 18432, 8, 1 | PASS | PASS | same |
- A first attempt "Bakeas phase0 + 0.01" PASSed *correctly* (F ≤ 0.68636150 < T: Bakeas has 1e-5 margin) and
  was replaced by +0.05; "Pinho phase3 + 0.002" was replaced by +0.01 for the same reason (float F < T).

### 18:55 Record feasibility test (checker #1 only; NOT a claim)
- Profile: S1's 96-piece float candidate (`results/best_profile.json`, phases optimized at T = 0.686360795,
  knots exactly j²/18432), with T lowered: `certificate/candidates/s1_96piece_T0.68636076.json`.
- `rigorous_check.py --N 73728 --P 8 --refine 1`: **PASS, F ∈ [0.6863607347625, 0.6863607351611]**
  (width 4.0e-10; sup|R/t| = 3.7e-9, EN = 4.8e-11, ED = 2.0e-11), margin T − F_upper = 2.48e-8.
  621 s, **6.29 GB**. I.e. by this checker C42 ≤ 0.68636076, 3.5e-8 below Pinho's 0.6863607950.
- Implied float fixed point of this phase vector: T* ≈ T − (T − F)/(1 + 1.22) ≈ 0.6863607487.
- Pinho rerun (with F_lower): N=36864 P=8 r=1: F ∈ [0.6863607439470, 0.6863607439516], width 4.6e-12, 283 s.
- Status per CLAUDE.md rules: one checker, new method, unreviewed; no second independent checker yet; phases
  not re-optimized at the lower T; no finite-n check at this precision. Not a claim.
- Same phases, T = 0.68636075 (`certificate/candidates/s1_96piece_T0.68636075.json`), N=73728 P=8 r=1:
  **PASS, F ∈ [0.6863607469915, 0.6863607473901]**, margin T − F_upper = 2.6e-9, 597 s, 6.29 GB.
  The two runs give dF/dT = −1.223 (consistent with S1's −1.22). So with this phase vector, checker #1 certifies
  T = 0.68636075, i.e. 4.5e-8 below Pinho. Still NOT a claim (same caveats as above).
- Doubling check on Pinho: N=73728 P=6 r=1: F ∈ [0.6863607439463, 0.6863607439522] (width 5.9e-12, 530 s,
  5.03 GB), overlapping the N=36864 P=8 enclosure [0.6863607439470, 0.6863607439516]. Stable.

### 19:10 Gate (c), final Pinho table (all with F_lower)

| N | P | refine | verdict | sup R/t | EN | ED | F_upper | F_lower | width | s | GB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 9216 | 6 | 0 | FAIL | 2.29e-05 | 8.17e-07 | 3.50e-07 | 0.68636367644701533 | 0.68635781145728377 | 5.86e-06 | 54.2 | 0.864 |
| 9216 | 8 | 1 | PASS | 2.34e-07 | 8.10e-09 | 3.47e-09 | 0.68636077302774936 | 0.68636071487079846 | 5.82e-08 | 75.6 | 0.945 |
| 18432 | 6 | 0 | PASS | 3.56e-07 | 7.89e-09 | 3.39e-09 | 0.68636077229099971 | 0.68636071560755823 | 5.67e-08 | 89.5 | 1.461 |
| 18432 | 6 | 1 | PASS | 3.56e-07 | 7.89e-09 | 3.38e-09 | 0.68636077226801144 | 0.68636071563054958 | 5.66e-08 | 130.7 | 1.605 |
| 18432 | 8 | 1 | PASS | 1.10e-09 | 2.27e-11 | 9.73e-12 | 0.68636074403471125 | 0.68636074386384734 | 1.71e-10 | 132.9 | 1.682 |
| 36864 | 8 | 0 | PASS | 2.94e-10 | 4.10e-11 | 1.75e-11 | 0.68636074409860560 | 0.68636074379999257 | 2.99e-10 | 208.5 | 3.165 |
| 36864 | 8 | 1 | PASS | 3.54e-12 | 4.88e-14 | 2.09e-14 | 0.68636074395160519 | 0.68636074394696189 | 4.64e-12 | 283.2 | 3.188 |
| 73728 | 6 | 1 | PASS | 3.92e-11 | 5.02e-13 | 2.15e-13 | 0.68636074395224924 | 0.68636074394632366 | 5.93e-12 | 530.3 | 5.034 |
- Candidate T = 0.68636076 at a different discretisation (N=36864, P=10, r=1): F ∈ [0.6863607342526,
  0.6863607356709] (width 1.4e-9, 312 s, 3.65 GB), which contains the N=73728 P=8 enclosure. Consistent.

**S3 verdict.** The spectral a-posteriori certifier (`checkers/rigorous_check.py`, derivation in
NOTES_CERTIFIER.md) certifies Pinho's profile to a two-sided enclosure of width 4.6e-12 with 3.2 GB in ~5 min,
versus Pinho's one-sided gap of 3.0e-8 with 95 GB. That is ~6000× tighter at ~30× less memory. All four gates
pass. With it, S1's 96-piece phase vector at T = 0.68636075 PASSes (margin 2.6e-9). A record ≈ 4.5e-8 below
Pinho is certifiable *by this checker*. The remaining steps before any claim are a second checker written
independently in a separate session, review of NOTES_CERTIFIER.md, and re-optimisation at the lower T.

## S2 — 2026-11-03, branch search-A (direction A; extrapolated objective only)

### 17:15 Fast solver (`src/fastsolve.py`)
- Divide-and-conquer FFT ("online convolution") for the forward and adjoint product-trapezoid recursions:
  O(N log² N) instead of O(N²). Same discretisation as `evaluator.py`; agreement on Pinho's profile:
  F to 2e-16, q to 7e-16, adjoint to 3e-13 relative (N = 4608, 18432).
- Timing (forward + adjoint, 1 thread): N = 73728: 0.5 + 0.75 s (was ~16 + 16 s); N = 589824: 6.6 + 7.1 s.

### 17:17 Extrapolation study, Pinho 48-piece profile at T = 0.6863607950 (`logs/levels_study_pinho.log`)
- F_N for N = 4608·2^k, k = 0..7, and 3-level fits (exponents 2−a, 2):
  base 4608 (= knot denominator): 0.6863607374; 9216: 0.6863607433; 18432: 0.6863607439463;
  36864: …9580; 73728: …9492; 147456: …9484. 4-level fits (extra N^{−(3−a)}): …9478–…9482.
- **F(Pinho) = 0.68636074395 ± 1e-11.** Rule: base ≥ 2× (better 4×) the lcm of the knot denominators.
  At base = 1× the error is ~6e-9 (first piece resolved by a single cell).

### 17:20 Driver `src/opt2.py` (exact rational knots, decimal phases, extrapolated objective, fixed point)
- Pinho's profile, verify only: Fx_ind = 0.6863607439580, **T* = 0.6863607720381** (err est 5e-12).
- All runs: L-BFGS-B on the 3-level extrapolated F (exponents 2−a, 2) at fixed T, then verification on
  independent levels (2N0, 4N0, 8N0) and a 4-level fit; T* solved with phases frozen (secant, dF/dT = −1.2229).
  err(T*) = max(|Fx_opt − Fx_ind|, |Fx_ind − Fx_4fit|)/(1 − dF/dT).

### 17:20 Quadratic-knot family quad:m (u_j = j²/(2m²)), float T* (search only)
| run | pieces | T used | base N0 | evals | T* | err(T*) |
|---|---|---|---|---|---|---|
| A24_q (start: Pinho restricted) | 24 | 0.6863610 | 9216 | ~30 | 0.6863608822895 | 1.2e-10 |
| A48_q (start: Pinho) | 48 | 0.68636077 | 18432 | ~30 | 0.6863607718846 | 5e-12 |
| Pinho as published | 48 | 0.6863607950 | 18432 | — | 0.6863607720381 | 5e-12 |
| A96_q (start: S1 eval-11 probe) | 96 | 0.68636075 | 36864 | 36 | **0.6863607482663** | 9e-11 |
- The S1 96-piece probe (stopped at eval 11) was already within 3e-10 of this.
- Knot gradient (`src/knotgrad.py`) on A24_q: dF/dx_j·h_j = 2.9e-7 for the first knot vs ≤ 1.3e-8 for all
  others. Knots j ≤ 16 want to move toward 0 and j ≥ 17 toward ½. **The first interval dominates.**
- Near 0 the optimal phase behaves like φ − φ(0) ∝ u^ν with local exponent ν ≈ 0.28–0.34 (A96_q knots),
  so the slope is unbounded at 0 and resolution there is what matters.

### 17:24 Near-0 refinements of quad:48 (T = 0.68636076, base 36864)
- q48_h4 (first 4 intervals halved, 52 pieces): Fx 0.6863607511850 → T* ≈ 0.686360756.
- q48_g1 (two geometric knots 1/18432, 1/9216 below 1/4608, 50 pieces): Fx 0.6863607405335 → T* ≈ 0.6863607512.
- **Geometric refinement at u → 0 beats uniform refinement.** 2 extra knots give 88% of the 48 → 96 gain.
- Added a stall stop to opt2 (12 evals without a 1e-13 improvement).
- q24_g4 (quad:24 + 4 geometric knots 1/18432..1/2304 below 1/1152, 28 pieces; base 36864):
  **T* = 0.6863607644762** (err 8.7e-11). It beats the 48-piece quadratic profile.
- Knot gradient on q48_g1: bulk knots are stationary (|dF/dx·h| ≲ 3e-11). The remaining signal sits at the
  geometric → quadratic transition: the ratio-4 gap 1/4608 → 1/1152 wants an extra knot.
  → New family D(m,K): dyadic knots x_2/2^k (k = 1..K) below x_2 = 2/m², plus quad:m knots for j ≥ 2.

### 17:45 OOM incident
- Four runs at base 147456 (top level N = 589824) were killed by the memory cgroup (4–7 GB RSS each).
- Cause: in `fastsolve._online_solve` the recursive closure referenced itself. The cycle kept x, E and K of
  every call alive until a full GC, a leak of ~55 MB per call at N = 147456. Fixed by breaking the cycle;
  RSS is now flat at 0.39 GB.
- Runs resumed from their 5-eval checkpoints, losing the L-BFGS memory only.

### 17:48–18:00 Results (all float, verified on independent levels; T* with phases frozen)
| run | pieces | first knot | base N0 | T* | err(T*) |
|---|---|---|---|---|---|
| A192_q (quad:192, polished at base 147456 from the base-73728 run) | 192 | 1/73728 | 147456 | **0.6863607431492** | 2.1e-11 |
| A_D96_4 (quad:96 + 1/9216, 1/36864, 1/73728) | 99 | 1/73728 | 147456 | 0.6863607433391 | 2.0e-11 |
| A_Rs2_24 (ratio-√2 lattice knots 1/18432..1/288 + quad:24) | 34 | 1/18432 | 36864 | 0.6863607534227 | 8.8e-11 |
| A_D24_6 (dyadic 1/18432..1/576 + quad:24 from 1/288) | 29 | 1/18432 | 36864 | 0.6863607538339 | 8.6e-11 |
| A_R4_24 (quad:24 + 1/18432, 1/4608) | 26 | 1/18432 | 36864 | 0.6863607650046 | 8.7e-11 |
- Base 73728 (1× the knot denominator) had underestimated quad:192 by 3.7e-10. The polish at base 147456 moved it
  only 6e-13.
- Finite n (`logs/finite_n_S2a.log`, common T = 0.68636075; F_n − F_n(Pinho)):
  - n = 2.5e5: A96 −1.85e-7, A192 −1.81e-7, q24_g4 −1.40e-7;
  - n = 1e6: −1.71e-7, −2.13e-7, −1.31e-7;
  - the limit differences are −5.3e-8, −6.4e-8, −1.7e-8.
  - F_n − F(Pinho) = 5.1e-5 → 1.6e-5 (exponent ≈ 0.85). Every candidate is below Pinho at every n, but the
    differences are dominated by small-k finite-size effects. That is consistent with S1 (a near-0 boundary layer
    with advantage decaying like n^{−0.8}). **Not a precision test of the limit.**
- **Lessons:**
  - The near-0 ratio (2 vs 4 vs √2 below 1/1152) hardly matters.
  - A ratio-4 gap at u ~ 1e-3 (quad:24's 1/1152 → 1/288) costs ~1.1e-8 in T*.
  - Insertion scores on D24_6 (`knotgrad.py --lattice`): the top candidate is [1/288, 1/128] at 7e-9,
    10× the next one. The region u ≈ 3e-3 to 2e-2 is the bottleneck for small bulks.

### 18:00–18:06 Greedy knot insertion (free-knot search on the lattice) and depth
| run | pieces | first knot | base | T* | err |
|---|---|---|---|---|---|
| A_G1_24 = D24_6 + 13/2304 | 30 | 1/18432 | 36864 | 0.6863607517785 | 8.6e-11 |
| A_G4_24 = D24_6 + 13/2304, 25/2304, 41/2304, 1/768 | 33 | 1/18432 | 36864 | 0.6863607502855 | 8.6e-11 |
| A_q24_g6 (quad:24 + 6 dyadic below 1/1152; keeps the 1/1152→1/288 gap) | 30 | 1/73728 | 147456 | 0.6863607595812 | 2.0e-11 |
| A_q48_g4 (quad:48 + 4 dyadic below 1/4608) | 52 | 1/73728 | 147456 | 0.6863607465193 | 2.0e-11 |
- Greedy rule: insert knots at the lattice midpoints with the largest score g_new²/width, where g_new is the
  adjoint gradient with respect to the value of a new knot inserted at its PL value; then re-optimise.
  Round 2 on G4_24 → G8_24 (+1/384, 61/2304, 1/1536, 1/6144; 37 pieces).
- Finite n at n = 4e6 (`logs/finite_n_S2a.log`), F_n − F_n(Pinho): A96 −1.19e-7, A192 −1.58e-7,
  q24_g4 −0.82e-7 (limit differences −0.53, −0.64, −0.17e-7). The excess shrinks by a factor of 0.56–0.63 per
  4× in n. That is consistent with slow convergence (steep first pieces: only ~54 indices in A192's first piece
  at n = 4e6). A precision match needs n ≫ 1e7.
- **Finite-n consistency** (common T = 0.68636075; F_lim from the limit evaluator at that T):
  - F_n − F_lim at n = 1e6 / 4e6 / 8e6:
    - Pinho: 1.580e-5 / 4.836e-6 / 2.667e-6;
    - A96: 1.568e-5 / 4.770e-6;
    - A192: 1.565e-5 / 4.742e-6 / 2.600e-6;
    - q24_g4: 1.568e-5 / 4.771e-6.
  - Every profile approaches **its own** limit with the same law n^{−0.86} (ratio 1e6→4e6 = 3.27–3.30). The
    constants agree within 2%.
  - The A192 − Pinho difference is −2.13e-7 / −1.58e-7 / −1.32e-7 at n = 1e6 / 4e6 / 8e6, moving toward the
    limit difference −6.4e-8.
  - No sign of a limit-formula bug.
- `levels_study` on A192_q up to N = 2,359,296: the 3-fit over 589824..2359296 = 0.6863607470035, vs 0.6863607470004
  on the verification levels (4-fit …0040). **The reported err (4.7e-11) is conservative; the true error is ~5e-12.**

### 18:05–18:55 Deep near-0 refinement (first knot 1/73728 … 1/294912) and small greedy profiles
| run | pieces | first knot | base | T* | err | max |slope| |
|---|---|---|---|---|---|---|
| A_G8_24 = G4_24 + 1/384, 61/2304, 1/1536, 1/6144 | 37 | 1/18432 | 36864 | 0.6863607494011 | 8.7e-11 | 617 |
| A_S35 = G4_24 + 1/36864, 1/73728 | 35 | 1/73728 | 147456 | 0.6863607453798 | 2.0e-11 | ~2000 |
| A_S36 = S35 + 1/147456 | 36 | 1/147456 | 294912 | 0.6863607446461 | 9.6e-12 | ~3600 |
| A_S39 = G8_24 + 1/36864, 1/73728 | 39 | 1/73728 | 147456 | 0.6863607445336 | 2.0e-11 | 2009 |
| A_D96_5 = D96_4 + 1/147456 | 100 | 1/147456 | 294912 | 0.6863607426060 | 9.7e-12 | 3658 |
| **A_D96_6 = D96_5 + 1/294912** (base 1×; Fx_opt − Fx_ind = 8e-11) | 101 | 1/294912 | 294912 | **0.6863607422566** | 3.7e-11 | 6817 |
- Depth series at bulk quad:96 (T*): 1/18432 …7482663 → 1/73728 …7433391 → 1/147456 …7426060 →
  1/294912 …7422566.
  - Gains: 4.9e-9 (for two halvings), then 7.3e-10, then 3.5e-10. The ratio is ~0.48 per halving.
  - Estimated remaining depth gain: ≈ 3e-10.
  - Bulk beyond 96 is worth ≤ 2e-10 (quad:192 vs D96_4 at equal depth).
  - **Float infimum of the PL-phase family ≈ 0.686360742 ± 3e-10.**
- Insertion scores on S39: max 7e-10, down from 7e-9 (D24_6) and 2e-9 (G4_24). The small profile is near
  saturation.
- `results/best_profile.json` ← A_D96_6 (float only).
- Candidate exports: `results/candidate_101pc.json`, `results/candidate_33pc.json`.
- Certification arithmetic: a claim at T_c needs F_upper(T_c) ≤ T_c, i.e. the gap F_upper − F ≤ 2.2229·(T_c − T*).
  To beat 0.6863607950, the allowed certifier gap is 2.2229·margin: 1.17e-7 for D96_6 and 9.9e-8 for G4_24.
  Pinho's certificate lost 3.3e-8 on his profile, but his first slope was 196, not 600–6800.

### 19:01 S40 and wrap-up
- A_S40 = S39 + 1/147456 (40 pieces, base 294912): **T* = 0.6863607438010** (err 9.6e-12).
- Candidates: `results/candidate_101pc.json` (D96_6), `candidate_40pc.json` (S40), `candidate_33pc.json` (G4_24).
  Each contains exact rational breaks, decimal phases, T, float F with error, T* and float Theorem A inputs.
  For all three: a ≈ 0.086–0.102, κ = T, min slope ≈ 1.2 > 0, phase range < 0.9.
- `results/SATURATION.md` and `results/summary_table.txt` written; PLAN.md has an "S2 outcome" section.
- Search stopped at 19:01 (1 h 46 min). The family is saturated: ~3e-10 of estimated depth gain remains, and the
  next depth level (first knot 1/589824) would cost ~80 min of compute.

**S2 verdict.** Direction A in float:
- best T* = 0.6863607422566 ± 4e-11 (101 pieces), 5.27e-8 below the target 0.6863607950;
- 40 pieces reach 0.6863607438 (95% of the gain over Pinho), 33 pieces reach 0.6863607503 (73%);
- nothing is certified.

## S5 — 2026-11-03 (certificate, branch `certificate` = main + certifier + search-A)

### 23:20 Target check (step 0)
- GitHub search API, teorth/optimizationproblems open PRs (56): C42 upper-bound PRs are [#302](https://github.com/teorth/optimizationproblems/pull/302) (Bakeas 0.68637*) and
  [#305](https://github.com/teorth/optimizationproblems/pull/305) (Pinho 0.6863607950*, title unchanged, last touched 18:41 UTC today). [#310](https://github.com/teorth/optimizationproblems/pull/310) is a lower bound.
  Nothing below 0.6863607950. Target unchanged.
- Machine: 4 vCPU, 15 GB, no swap. Peak memory kept < 11 GB (one ≤ 7.5 GB run, or two ≤ 4 GB runs at a time).

### 23:25 Grid alignment (`checkers/grid_check.py`, exact)
- 33pc: lcm denominator 18432; N ∈ {18432, 36864, 73728, …} align all knots, ½, and all 3180 sums of ≤ 3 knots.
- 40pc: needs N ≥ 147456; 101pc: needs N ≥ 294912 (not attempted: the 33pc claim is already certified,
  and N = 147456 at P = 8–10 extrapolates to ≥ 12 GB from S3's 6.3 GB at 73728).

### 23:30 Step 2: re-optimise 33pc phases at T = 0.68636075 (src/opt2.py, base 36864)
- Start gradient 5.8e-9; L-BFGS finds no descent. Phases move ≤ 2.5e-6, T* = 0.6863607502854 (was …2855).
  **No gain**; certificate keeps S2's phases. `results/S5_33pc_reopt_T75.json`.

### 23:40 Checker #1 on 33pc at its own T = 0.68636076 (N 36864, P 10, refine 1)
- PASS, F ∈ [0.6863607377711, 0.6863607390664] (width 1.3e-9), 352 s, 3.65 GB. Float F 0.6863607384 inside.
- ⇒ T* ∈ [0.6863607500, 0.6863607506] (linearised, dF/dT = −1.2229).

### 23:50 Claim C = T = 0.686360751 (`certificate/c42_profile_0.686360751.json`)
| N | P | refine | verdict | sup R/t | EN | ED | F_upper | F_lower | width | s | GB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 18432 | 12 | 1 | FAIL | 7.27e-8 | 2.82e-9 | 1.19e-9 | 0.6863607607554 | 0.6863607380943 | 2.3e-8 | 242 | 2.02 |
| **36864** | **10** | 1 | **PASS** | 7.28e-9 | 1.59e-10 | 6.72e-11 | **0.6863607500725** | 0.6863607487772 | 1.3e-9 | 349 | 3.65 |
- Margin C − F_upper = 9.28e-10. Margin below Pinho: 0.6863607950 − 0.686360751 = 4.4e-8.
| 18432 | **14** | 1 | **PASS** | 2.86e-9 | 2.17e-10 | 9.15e-11 | **0.6863607503089** | 0.6863607485408 | 1.8e-9 | 432 | 2.15 |
| 18432 | 16 | 1 | FAIL | 2.28e-5 (cell 0) | 5.7e-6 | 2.4e-6 | 0.6863835 | 0.6863380 | 4.6e-5 | 1281 | 2.41 |
- **N = 18432 (the smallest aligned grid) with P = 14 PASSes**, margin C − F_upper = 6.91e-10. This is the
  primary setting (certificate "checker" field; file SHA changed only through that field).
- P = 16: the float witness breaks down at cell 0 (monomial-basis conditioning; GMRES/refinement cannot fix it).
  The rigorous bound is correct but loose. Not a rigor problem; P ≤ 14 is the usable range of this witness.
- N = 36864, P = 14: witness GMRES needed 225 iterations, collocation residual 2.8e-8 (vs 4.7e-11 at P=10):
  conditioning starts to bite at P = 14. Run interrupted by a container restart; requeued.

### 00:12 Finite-n cross-check (float), `logs/S5/finite_n_claim.log`
- n = 125k…4M at T = 0.686360751: F_n decreasing monotonically (0.68645191 → 0.68636552); our profile below
  Pinho's at every n (−1.2e-7 … −1.9e-7). Aitken extrapolates still rising (ours 0.68636068, Pinho 0.68636074
  at the top level); last difference −5.5e-8 vs expected limit difference −4.8e-8. Consistent.

### 00:15 Container restart (all processes killed). Requeued with `checkers/run_queue.sh` (resumable, 2 jobs,
1 BLAS thread each; load had been ~10 on 4 vCPU).

### 00:20–01:20 Final runs (1 BLAS thread each; `checkers/run_queue.sh`, resumable)
- N = 18432 only passes at P = 14 (P = 12, 13 FAIL loosely; P = 16 witness breakdown), and N = 36864 at P = 14
  is loose (witness: 318 GMRES iterations, collocation residual 4e-8). So **primary moved to N = 36864, P = 10**
  (well-conditioned witness). Certificate `checker` field updated; earlier runs moved to certificate/runs_superseded.
- Final file SHA-256 d015a829…5652 (certificate/SHA256SUMS; matches the expected output's certificate_sha256).

| run | N | P | verdict | F_lower | F_upper | s | GB |
|---|---|---|---|---|---|---|---|
| primary | 36864 | 10 | PASS | 0.6863607487772 | 0.6863607500725 | 359 | 3.64 |
| different P | 36864 | 12 | PASS | 0.6863607494100 | 0.6863607494397 | 346 | 3.84 |
| doubled N | 73728 | 10 | PASS | 0.6863607494207 | 0.6863607494290 | 748 | 6.78 |
- Nested enclosures, all < C. Margin C − F_upper(primary) = 9.28e-10; tightest F ≈ 0.68636074942 (float 0.6863607494).
- Corruptions at N = 36864, P = 10 (`checkers/corruptions/claim/out/SUMMARY.txt`): T=C=0.68636074 FAIL;
  T=C=0.68636075 FAIL (F_lower > T by 1.3e-13); ψ_5+0.002 FAIL; ψ_12+0.005 FAIL; T=0.68636074 with
  C = F_lower − 1e-9 FAIL; with C = F_upper + 1e-9 PASS (control). All as expected.
- Thread dependence: witness bits depend on BLAS threads; at N=36864 P=10 4-thread vs 1-thread F_upper differ by
  5e-22, at N=18432 P=14 by 8e-11. Verdicts unaffected. README tells verifiers to set OPENBLAS_NUM_THREADS=1.
- **Status: C42 ≤ 0.686360751 is certified by checker #1 only. Not a public claim until checker #2 (S6) agrees.**

## S4 — 2026-11-03/04 (exactness of the reduction; finite-n study; FINDINGS.md)

Branch `claude/harcos-reduction-findings-53k0p2`. Machine: 4 vCPU, 15 GB.

- **Exactness of Harcos's reduction: PROVED** (FINDINGS.md §2, elementary).
  - R_n = min_s max(‖s‖, F_n(s)), equivalently R_n = the unique fixed point of G_n(T) = min_{‖s‖≤T} F_n.
  - Float cross-check (`src/reduction_crosscheck.py`): identity (★) to 3.5e-16 on 2000 random tuples. Direct
    z-space multistart agrees with the first-half minimum at n = 2..7 to 8 digits.
- **S1 wording corrected.**
  - 0.69465442 at n = 400 is a local min of F_400 on the torus at T_0 = 0.68637. It is not R_400.
  - The fixed point gives R_400 ≤ 0.690120597996 (Arb).
  - S1's free-modulus "stuck" runs were mostly a polar-coordinate artifact (ρ_k = 0 is a spurious KKT point).
    They were redone with the disk map s = T·2w/(1+|w|²).
- **`src/finite_n_study.py`**, n = 200, 283, 400, 566, 800, 1131, 1600, 2263, 3200. The full table is in
  `logs/finite_n_fit.txt`.
  - 1404/1404 constant-modulus starts end at one minimiser (± conjugation).
  - 470/484 free-modulus starts end at ρ ≡ 1. The remaining 14 are worse, with 1–5 of the s_k ≈ 0.
  - ∂F/∂ρ_k < 0 for all k at every n.
  - Arb: R_n ≤ 0.692648924532 (200) … 0.687129252420 (3200).
- **Trends.**
  - Δ(T_0) ≈ 0.070 n^−0.79.
  - Δ* = T*_smooth(Pinho φ) − T*_free ≈ 0.058 n^−0.88, with local exponents rising to 0.92.
  - The expected exponents are 1−a, 1, 2−a with a = Re g(0) (0.166 for Bakeas, 0.112 for Pinho).
  - Model exp3 was validated on the φ-frozen smooth families (n to 102400, known limits 0.6863655395 and
    0.6863607720): window error +3.5e-6 / +1.2e-5.
  - Extrapolated free fixed point: 0.686356 – 0.686376.
- **Boundary layer.** The deviation is fixed in k (−0.16 rad at k = 1), and the u-scale part shrinks. Grafting the
  k ≤ 8 layer: gain −3.6e-4 (n=400) → −2.3e-5 (n=25600).
- **Verdict.**
  - No evidence for non-smooth or varying-modulus first halves. B, C and D add nothing beyond A (proved for
    β ≥ ½, numerical otherwise).
  - Conditional on local = global and on the trend: lim R_n ≈ 0.68636 ± 2e-5.
- Caveat: the background limit-reference job hit the 30-min tool limit once under CPU contention. It was rerun
  cheaply (S1's F values + a fresh slope).

## S7 — 2026-11-04 (package; branch `certificate`)

Machine: 4 vCPU, 15 GB. Record mode.

- Merged `claude/certificate-validation-checker-0jbs2a` (S6: checker #2, REVIEW.md) and
  `claude/harcos-reduction-findings-53k0p2` (S4: FINDINGS.md; this is the branch S4 was told to call `findings`)
  into `certificate`. LOG.md conflict: kept both sections.
- **Open PRs re-checked** (teorth/optimizationproblems, search API + PR heads by git fetch): the 42a PRs are
  [#302](https://github.com/teorth/optimizationproblems/pull/302) (0.68637*), [#305](https://github.com/teorth/optimizationproblems/pull/305) (0.6863607950*, head bac1d85, body unchanged, 0 comments), [#310](https://github.com/teorth/optimizationproblems/pull/310) (lower bound),
  [#184](https://github.com/teorth/optimizationproblems/pull/184) (Röhrig). Merged: [#85](https://github.com/teorth/optimizationproblems/pull/85), [#251](https://github.com/teorth/optimizationproblems/pull/251), [#263](https://github.com/teorth/optimizationproblems/pull/263). **No C42 upper bound below 0.686360751.**
- REVIEW.md §2 / NOTES §5.3 (cell 0): checker #1 already set the constant term of d_{Q,0} to the exact `acb(0)`
  (radius 0; `fr = [zero if i == 0 else gamma - g]`) and formed R_0/θ by shifting coefficients, never dividing a
  ball by θ. Added an assertion (`is_zero() and is_exact()`). Note: `g - g` in Arb is a ball ±1e-16 at 53 bits,
  so the concern was real in principle; the code was not affected.
- REVIEW.md §2 / NOTES §6.2 (series truncation): checker #1 keeps l = 0..L (Horner `range(L, -1, -1)`, L+1
  terms) with tail n'^{−a} n'^{−(L+1)}/(1−1/n') — the smaller bound, correct for that reading. Switched to the
  conservative n'^{−a} n'^{−L}/(1−1/n') (valid under either reading). Tail < 1e-22 either way. NOTES §6.2 now
  states the truncation and both bounds explicitly.
- Also from REVIEW: κ̂ remark (§3; checker #1's κ̂ = T is an Arb ball containing T, valid by inclusion
  monotonicity), hypothesis on a stated identically in NOTES §3/§7 and README, coefficient balls vs rounding
  (§5.3), §7 piecewise-linear wording, min/inf note (§1), NOTES §9 (checker #2 exists), headline = tightest PASS.
- Checker #1 SHA-256 now 93fdcf68…6c7d; checker #2 unchanged (26e12388…8093). SHA256SUMS lists both.
- New corruption list for checker #1: `checkers/corruptions/claim/list3.txt` (copies of checker #2's certificate
  corruptions: C < T, C = 1, a < 0, misaligned knot, phase-count mismatch, knots not increasing, T = C =
  0.686360749, ψ_20 + 1e-3).
- Removed 10 optimizer checkpoint/snapshot files (`results/*_ckpt.json`, `*_snap.json`; final JSONs kept).
- Reruns (memory-aware queue, ≤ 13 GB, 4 single-threaded jobs, OPENBLAS_NUM_THREADS=1): both checkers at all
  certificate settings, all corruption lists of both checkers, checker #2 self-test. Results below.
- **Results (all 12 queue jobs rc = 0):**
  - checker #1 (SHA 93fdcf68): N=73728 P=10 PASS F ∈ [0.6863607494206, 0.6863607494291] (780 s, 6.78 GB);
    N=36864 P=12 PASS [0.6863607494100, 0.6863607494397] (398 s, 3.84 GB); N=36864 P=10 PASS
    [0.6863607487772, 0.6863607500725] (416 s, 3.64 GB). All identical to S5 to 25 digits; only the radius of
    p̃(1) grew (2e-31 → 2e-30) from the conservative tail.
  - checker #2 (SHA 26e12388, unchanged): N=73728 P=12 PASS F ≤ 0.6863607494249 (612 s, 2.90 GB); N=36864 P=12
    PASS F ≤ 0.6863607494323 (236 s, 1.50 GB); N=36864 P=10 PASS F ≤ 0.6863607500744 (171 s, 1.35 GB). Same as
    S6 to ≤ 1e-17. Self-test OK.
  - Corruptions: checker #1 claim lists 14/14 as expected (list 4, list2 2, list3 8), run_all.sh 8/8;
    checker #2 28/28. Every FAIL of checker #1 has F_lower > C.
  - Times measured with up to 4 single-threaded jobs at once.
- Fresh clone of `certificate` @ e6ffc94 → README commands: sha256sum OK; checker #2 PASS 238 s / 1.50 GB;
  checker #1 PASS 393 s / 3.64 GB; both outputs identical to the committed expected outputs.
- NOTES §8.2 now leads with the tightest PASS of both checkers; §8.3 tables from S7 runs, with outward rounding
  (the S5 tables had rounded some F_upper values down in the last digit; display only).
- README.md, submission/ (42a.md on top of [PR #305](https://github.com/teorth/optimizationproblems/pull/305) head bac1d85, README row 42 + Recent-progress line,
  PR_DRAFT.md), PR_DRAFT.md pointer at root.

### S7b — 2026-11-04 (owner follow-up)

- Name for all credits confirmed by the owner: **Nikolaos Vatopoulos**. It is now used in README, NOTES (AI use),
  FINDINGS (AI use) and submission/ (row reference column "Vatopoulos [Vat26]", reference "Vatopoulos, Nikolaos",
  contribution note, Recent-progress line).
- teorth PR links: no file contained `/turan-c42/pull/`. Bare mentions "PR #n" / "#n" of teorth PRs (85, 184, 251,
  263, 302, 305, 310) in all markdown files are now explicit links to
  https://github.com/teorth/optimizationproblems/pull/n (GitHub autolinks bare #n in PR bodies and commit messages
  to this repo, not to teorth). Erdős problem numbers (#519, #165) left alone.
- submission/ rebuilt against teorth **main** 87b4ebb (not on top of #305): 42a.md = main + our row (after
  Koshta's) + [Vat26] + our contribution note (diff: 5 added lines, 0 removed); README row 42 changed from
  `0.69368 (0.6873358063*)` to `0.69368 (0.686360751*)` + one Recent-progress line. Bakeas (#302) and Pinho (#305)
  are cited by inline links; the row now defines F itself (it can no longer lean on Bakeas's row). PR_DRAFT.md
  says #302/#305 are pending, that the construction builds on them, and that we rebase if they merge first.
- URL / hash check (submission/ and README.md):
  - commits by git fetch: Bakeas dc9c491, Pinho 009120a (path volterra-quadratic-knots) and a8a9bc3, Koshta
    8e1797e (= repo HEAD), Griego c8ddce1 (= tag v1.0.0), teorth main 87b4ebb, own e6ffc94: all resolve;
  - teorth PR refs 184, 251, 263, 302, 305, 310 exist; profile SHA-256 705a…cafcab = rpinho certificates/c42_256.json
    @a8a9bc3, be0c…cbef = Koshta certificates/c42_tau41_200_M64_0.6873358063.json @8e1797e, d015a829… = ours;
  - full teorth README copy (253 URLs): 120 teorth.github.io pages all have a source file on main; 44/44 GitHub
    repos exist (one parse artifact from a BibTeX `url = {…}`); 18/18 pinned refs/paths resolve;
  - not checkable here: web hosts are outside this session's network policy (github.com web pages 403 via the
    proxy; teorth.github.io, erdosproblems.com, arxiv.org, doi.org etc. not resolvable). Those 80 URLs are
    teorth's existing content, copied unchanged.

### S8 — 2026-11-04 (audit of teorth/optimizationproblems#327; credit wording)

- PR teorth/optimizationproblems#327 (fork branch 42a-turan-0.686360751, head 6c971f0): open, not draft, 0 comments,
  no reviews, no labels. Touches only constants/42a.md (+5) and README.md (+2/−1). Base = teorth main 87b4ebb,
  which is unchanged; merges cleanly into main (git merge-tree). teorth has no CI workflows.
- Consistency at 9668a60: knots and phases in the row = certificate (34 + 34, LCD 18432); SHA-256 d015a829…5652;
  F intervals, a, κ, margins and run times in the row and description = committed outputs; table arity 3 on every
  row; `scripts/check_pages.py`: 0 errors / 0 warnings on main and on the PR.
- Links: teorth PR refs 263, 302, 305, 310 exist (search API + refs/pull); Bakeas dc9c491 (= tag v1.1.0), Pinho
  009120a (path volterra-quadratic-knots) and our 9668a60 resolve. teorth.github.io and other github.com web pages
  are blocked by this session's proxy (not fetchable).
- Fresh clone at 9668a60, python-flint 0.9.0, OPENBLAS_NUM_THREADS=1, 4 vCPU (two jobs at once):
  - `sha256sum -c certificate/SHA256SUMS`: 4/4 OK; grid_check N = 36864 OK; independent_selftest OK;
  - checker #2, N = 36864, P = 12: PASS, F ≤ 0.6863607494322799, 181 s, 1.54 GB; F_ball identical to
    checkers/independent_out/cert_N36864_P12.json;
  - checker #1, N = 36864, P = 10, refine 1: PASS, F ∈ [0.6863607487772, 0.6863607500725], 355 s, 3.49 GB;
    F_upper, F_lower, p̃(1), trial_sha256 identical to certificate/expected_N36864_P10.json;
  - checker #2, N = 73728, P = 12: PASS, F ≤ 0.6863607494249, 374 s, 2.96 GB; F_ball identical to
    checkers/independent_out/cert_N73728_P12.json;
  - checker #1, N = 73728, P = 10, refine 1: PASS, F ∈ [0.6863607494206, 0.6863607494291], 655 s, 6.48 GB;
    F_upper, F_lower, p̃(1), trial_sha256 identical to certificate/stability_N73728_P10.json (finished after
    commit 7270346, which the PR pins; this line was added afterwards).
- Other PRs: only 302, 305, 310 (and ours) touch 42a.md among all PR heads ≥ 200. #305 (bac1d85) and #310 (e0393b2)
  unchanged since S7b. #302 updated today (3887fbb): adds a lower-bound row 1/2 + 2^(−32176926823186696)* [Bak26c]
  and references [Bak26b], [Bak26c]; its upper bound is still 0.68637*. No PR claims an upper bound below
  0.686360751. Ours still conflicts textually with #302/#305 (README, 42a.md) and #310 (42a.md), as the PR says.
- Credit wording rewritten to lead with Nikolaos Vatopoulos, with one AI-use statement (README "Credits and AI use",
  NOTES/FINDINGS AI-use paragraphs, submission/ contribution note); [Vat26] reference and Recent-progress line no
  longer mention AI. certificate/ and checkers/ unchanged.
