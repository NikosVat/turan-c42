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

### 17:35 Validation of the formulas (`checkers/tests/`)
- Float pipeline F from the spectral trial: Bakeas 0.68636008313–0.68636008372 (N=192..768, P=6);
  Pinho 0.6863607439575 (N=18432) / 0.6863607439536 (N=36864).
- `tests/mp_pointwise.py` (mpmath, 30 digits, direct quadrature, no convolution identity):
  Bakeas N=192 P=6: max sampled |R/t| = 1.1e-12 (cells 0–3, knots, 95/96 around 1/2);
  N=96: p̃ cell representation vs direct quadrature ≤ 1.9e-15, p̃(1) diff 1.7e-16.
- Harcos constant phase (`tests/profiles/harcos_const.json`, exact F = 0.69368452670658032 by mpmath):
  rigorous F_upper − F_exact = +5.2e-12 (N=64), +2.6e-12 (N=256), +1.5e-12 (N=1024, before weight fix).
  Two slack sources found and fixed on the way: cubic u fit near 0 (now deg 8/6/3), and the 1/(1−t) weight
  series truncated at 4 terms (now adaptive, tail < 1e-22).

### 17:50 Gate (a): Bakeas profile c42_profile_0.68637.json — PASS
- `python checkers/rigorous_check.py third_party/bakeas/c42_profile_0.68637.json --N 768 --P 6`
- **PASS, F ≤ 0.6863600829256854** (Arb), 4.2 s, residual sup|R/t| = 3.6e-12, EN = 3.2e-12, ED = 1.5e-12,
  denominator slack 1.5e-12. Bakeas's own checkers: F ≤ 0.6863659 and 0.6863615 (N = 122880).
- Note: the S1 evaluator's extrapolated value 0.6863600838 was off by ~1e-9. Its raw F_N at N=98304 is
  0.6863600830081 and still moving (complex-exponent error terms from (t−t_k)^{2−γ}); the 4-term fit over
  12288..98304 gives 0.6863600829277, i.e. 1.4e-12 above our F_upper, within that fit's accuracy. Our
  enclosure says F ∈ [F_upper − ~1e-11, F_upper].

### 18:00 Gate (b), first point: Pinho PR #305 profile
- N=18432 (4608·4), P=6: **PASS, F ≤ 0.6863607723298**, 69.5 s, max RSS 1.24 GB.
  (Pinho: F ≤ 0.6863607742 at N = 14.2M, 95 GB, 1110 s on 3 threads.)
- Error budget: residual peak 3.6e-7 in cell 4 (first cell after the knot 1/4608: the continued piece has
  a 1/t pole one knot-cell away, so convergence there is only (4s+2)^-P); EN = 7.9e-9, ED = 3.4e-9.
  Denominator slack 2.1e-12. So the gap 2.8e-8 is all residual; fix = larger s or P.
