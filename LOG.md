# LOG

Terse experiment log. F values are floats unless marked "Arb". Times are UTC.

## S1 — 2026-11-03 (reproduce + evaluator)

Environment: 4 vCPU, 15 GB RAM, Python 3.13.18, python-flint 0.9.0, jax 0.12.0.1, numpy 2.5.3, scipy 1.18.1.
teorth.github.io is **not reachable** from this sandbox (network policy). The table was read from the source
repo `teorth/optimizationproblems@87b4ebb` (constants/42a.md), and PRs were found through the GitHub search API.

### 15:45 Target check
- Merged table (42a.md @87b4ebb): best upper bound 0.6873358063* (Koshta).
- Open PRs:
  - #302 (Bakeas, 0.68637*, phase-varying first block);
  - **#305 (Pinho, 0.6863607950*)**: Bakeas's Theorem A, 48 phase pieces on knots j²/4608, sharper Arb error
    propagation, F ≤ 0.6863607774, repo rpinho/c42-certificate@009120a/volterra-quadratic-knots;
  - #310 (Pinho, lower bound 1/2 + 2^-4096*, irrelevant here);
  - #184 (Röhrig, old).
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

### 17:48 Gate (b), first point: Pinho PR #305 profile
- N=18432 (4608·4), P=6: **PASS, F ≤ 0.6863607723298**, 69.5 s, max RSS 1.24 GB.
  (Pinho: F ≤ 0.6863607742 at N = 14.2M, 95 GB, 1110 s on 3 threads.)
- Error budget: residual peak 3.6e-7 in cell 4 (first cell after the knot 1/4608: the continued piece has
  a 1/t pole one knot-cell away, so convergence there is only (4s+2)^-P); EN = 7.9e-9, ED = 3.4e-9.
  Denominator slack 2.1e-12. So the gap 2.8e-8 is all residual; fix = larger s or P.

### 17:53 Gate (b): Pinho PR #305 profile — PASS, far tighter and with 2.6 GB
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
