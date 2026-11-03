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
