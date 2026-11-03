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
