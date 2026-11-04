# PLAN.md — ranking of directions A–D (written end of S1, 2026-11-03)

**Target: certified bound < 0.6863607950** (Pinho, teorth [PR #305](https://github.com/teorth/optimizationproblems/pull/305), Bakeas's Theorem A with 48 quadratic knots).

Every number below is a float (search only). The evidence is in LOG.md (S1) and `results/`.

## What S1 established

1. **Harcos's reduction is exact at finite n.**
   - Take any tuple with z_1 = 1. Its power sums satisfy W(1) = P_n − Σ_{k>n/2}(s_k/k)P_{n−k} = 0, because
     2⌊n/2⌋+2 > n. So max_{k>n/2}|s_k| ≥ F_n(s_1..s_{⌊n/2⌋}), and Lemma 2 shows equality is attained. Hence
     **R_n = min over s ∈ ℂ^{⌊n/2⌋} of max(max_k|s_k|, F_n(s))**.
   - So Bakeas's framework loses nothing except through the *smooth ansatz* s_k = g(k/n). Every direction A–D is
     a choice of first half, so each is either inside "smooth g with |g| ≤ T" (= direction B) or a non-smooth
     first half (called E below).
2. **Direction B (modulus) shows no gain.**
   - The radial sensitivity δF/δlog|g|(u) is **strictly negative on all of (0,½]**: from −13.4 to −1.16, for both
     the Pinho and Bakeas profiles. So lowering |g| anywhere raises F, and ρ ≡ 1 is a KKT point.
   - Limit evaluator, start ρ ≡ 0.85: L-BFGS-B returns every ρ_j to 1.
   - Exact finite n = 400, 60 random starts with 0 ≤ ρ_k ≤ 1 free: nothing beats ρ ≡ 1. The best runs (2 of 60)
     end at ρ ≡ 1, and the others are trapped in worse optima with some ρ_k = 0.
3. **At constant modulus the exact finite-n landscape has one basin.**
   - n = 400, 120 random starts (iid phases, random smooth phases, random linear phases): all converge to
     F_400 = 0.69465442, the optimum of the smooth-profile basin.
   - The free first-half optimum differs from the smooth profile only by a boundary layer at small k plus a smooth
     correction. Its advantage decays like n^{−0.8}: −1.1e-3, −6.2e-4, −3.5e-4 at n = 200, 400, 800. That looks
     like a finite-size effect, not a structure that survives n → ∞.
   - *Heuristic, not a claim:* this suggests that the smooth-profile optimum ≈ 0.6863607 is close to lim R_n
     itself, and that no construction will beat it by much.
4. **Rescaling (Lemma 1) gives nothing.** For n = 200..1600, z = 1 is the root of largest modulus and all other
   roots lie strictly inside the disk.
5. **Direction A still has a little room.**
   - Pinho's 48-knot profile is stationary for its own knot family: the extrapolated objective moves by 4e-10.
   - With every interval halved (96 pieces) the extrapolated F at fixed T = 0.6863607950 dropped from
     0.6863607440 to 0.6863606919 within 11 evaluations. That value was **re-verified on independent levels**
     36864/73728/147456; the run was stopped while still descending. Saved in `results/best_profile.json`.
   - A drop δF at fixed T moves the fixed point T* by about δF/2.22, since ∂F/∂T = −1.22.
   - Pinho's family has float T* ≈ 0.68636077, so the remaining room is a few × 10⁻⁸.
6. **Pitfall.** Optimising F_N at a fixed N overfits the discretisation error, which is ~5e-8 at N = 18432.
   Optimise the Richardson-extrapolated value (`src/optimize.py --extrap`) and confirm at other levels.

## Ranking

| rank | direction | assessment | P(certified record) |
|---|---|---|---|
| 1 | **A — richer phase, same proof** | Only direction with a measured float gain. That gain is tiny (T* ~ 2–4e-8 below Pinho's family). A record needs a certificate whose gap F_upper − F is ≲ 2e-8 on 4 vCPU / 15 GB. Pinho needed N = 14.2M, 95 GB and 32 vCPU for a gap of 3e-8. **Certification is the bottleneck, not search.** | ~30% |
| 2 | B — modulus ρ ≤ 1 | Strict generalization, and Theorem A extends easily (MATH.md O2). But every test (first order, non-local start, finite-n multistart) gives ρ ≡ 1. Keep it as a free flag in the A searches. Don't spend a session on it. | ~2% |
| 3 | D — hybrid with Koshta corrections | Koshta's family is a sub-family of B: a constant first block, steps with \|η_j\| ≤ C (so varying modulus), jumps allowed. Bakeas's Volterra solution already contains all correction orders exactly. The hybrid is therefore inside B, and B shows no gain. | ~1% |
| 4 | C — change the split | β > ½ is exactly equivalent to β = ½ (MATH.md O1: the extra sums enter linearly, so the relaxation is convex with the same optimum). β < ½ is only a re-parametrisation of the first half, so it is covered by B/E. | ~0% |
| — | E (new) — non-smooth first halves | The only way to a non-marginal gain. But S1's finite-n evidence (single basin; non-smooth gain decays like n^{−0.8}) is against it. A new limit theory would be needed. Interesting as research ("is C42 ≈ 0.68636?"), not as a route to a record. | ~2% |

## Recommended use of S2–S4

The template has three parallel search sessions, A / B / (C or D). S1's evidence says to replace them with:

- **S2 — A-search.**
  - Profile family: 96–192 phase pieces with graded knots (quadratic and cubic grading; also try a few extra
    knots near 0).
  - Objective: `--extrap` with base N ≥ 18432 and knot-aligned levels. At ~45 s per gradient, run it in the
    background with checkpoints.
  - Also run the fixed-point outer loop in T: re-optimise at T ← T − (T − F)/2.22.
  - Verify every candidate at *different* N levels, e.g. 36864/73728/147456. A gain must survive the change of
    levels.
  - Check finite n with `third_party/bakeas/crosscheck/finite_n.py` or `src/finite_n_roots.py`.
  - Deliverable: `results/best_profile.json` with float T* and the safety margin.
- **S3 — A-certifier.** Build a rigorous checker that is ≥ 10× tighter per unit of compute than Pinho's.
  Pinho's error budget at N = 14.2M:
  - sup|R/t| = 2.85e-6, with the worst cell at the first knot 1/4608. The global R''' bound is B3 = 1.5e8.
  - The integrated residual bound brings the numerator error to 7.3e-9 and the denominator error to 3.8e-9.

  Clear wins:
  - (i) per-cell R''' bounds instead of the global B3, since B3 is set by the steep first pieces;
  - (ii) a piecewise-quadratic trial ρ̃, giving an O(h³) residual;
  - (iii) a composite grid, fine near 0 and coarse elsewhere, keeping Pinho's integrated residual and local
    contraction bounds.

  Validate against Bakeas's and Pinho's profiles first: it must reproduce PASS and give tighter F_upper.
- **S4 — optional, cheap.** Either skip it to save budget, or run B/E only as a negative-result confirmation
  (finite-n multistart at n = 800–1600 with free modulus) to document that the smooth constant-modulus basin is
  the only one.

## Honest expectation

The Bakeas–Pinho family is very close to saturated. A new certified record is plausible only in the 8th
significant digit, roughly 0.68636076–0.68636079, and only if S3 delivers a much tighter certifier. If S2 cannot
get the float fixed point below ~0.68636076, or S3 cannot certify within ~2e-8 of the float value, the right
outcome is to write up the negative results (B, C, D, E) and stop.

## S3 update (certifier, branch `certifier`)

Certification is no longer the bottleneck. `checkers/rigorous_check.py` (spectral a-posteriori, Arb):
- Pinho's profile: F ∈ [0.6863607439470, 0.6863607439516] at N=36864, P=8 (3.2 GB, ~5 min);
- S1's 96-piece candidate at T = 0.68636076 PASSes with F ≤ 0.68636073516 (6.3 GB, 10 min).

The remaining work for a record is: re-optimize at the lower T (search), a second independently written checker
(separate session, from NOTES_CERTIFIER.md only), review, and packaging.

## S2 outcome (direction A search, branch `search-A`, 2026-11-03). Float only, not certified.

- **Best float fixed point: T* = 0.6863607422566 ± 4e-11** (D96_6, 101 pieces), i.e. **5.27e-8 below 0.6863607950**.
- **Saturation.** See `results/SATURATION.md`.
  - The family's float infimum is ≈ 0.686360742 ± 3e-10, so direction A is exhausted in float.
  - Piece count saturates at ~35–40 with adjoint-greedy knots. What still matters is the first-knot depth:
    the optimal phase has a u^0.28 singularity at 0.
- **Candidates** (`results/candidate_{101,40,33}pc.json`):
  - 101 pieces: T* = 0.6863607423, first break 1/294912, max slope 6817;
  - 40 pieces: T* = 0.6863607438, first break 1/147456, max slope 3653;
  - 33 pieces: T* = 0.6863607503, first break 1/18432, max slope 615.
- **Correction to the S1 estimate above.** A claim at T_c needs F_upper(T_c) ≤ T_c. Since dF/dT = −1.2229, the
  certifier may lose up to **2.2229 × margin**:
  - 1.17e-7 (101 pieces), 1.14e-7 (40 pieces), 9.9e-8 (33 pieces).
  - So the allowed gap is not "≲ 2e-8". Pinho's certifier lost 3.3e-8 on his profile.
  - The difficulty for S3 is the steep first pieces (slopes 600–6800 vs Pinho's 196). The requirement is not
    an extra-tight certifier.
  - Suggested S3 order: certify the 33-piece candidate first (first break 1/18432), then try the 40- and 101-piece
    ones with a graded near-0 grid.
- Tools added: `src/fastsolve.py` (O(N log² N) forward/adjoint), `src/opt2.py` (driver + fixed point),
  `src/knotgrad.py` (knot gradients and insertion scores), `src/finite_n_check.py`, `src/export_candidate.py`,
  `src/summarize.py`.

## S5 outcome (certificate, branch `certificate`, 2026-11-04)

- **Checker #1 certifies C42 ≤ C = 0.686360751** (33-piece S2 profile, T = C) at N = 36864, P = 10:
  F ∈ [0.6863607487772, 0.6863607500725], margin 9.3e-10. Stable: N = 73728 (P 10) and P = 12 give nested
  enclosures around F ≈ 0.68636074942. Corruptions all as expected. 4.40e-8 below Pinho's 0.6863607950.
- 40pc / 101pc not certified: they need N ≥ 147456 / 294912 (est. ≥ 13 GB / ≥ 26 GB at P = 10) and would gain
  ≤ 6.5e-9 more. Not worth it before checker #2 exists.
- Next (S6): independent checker #2 written from NOTES.md, certificate/ and README "Certificate format" only.
  Then PR_DRAFT.md.

## S7 outcome (package, branch `certificate`, 2026-11-04)

- REVIEW.md points addressed (NOTES §§1, 3, 5.3, 6.2, 7, 8, 9; README). Checker #1: exact-zero assertion on cell 0
  and conservative series tail (no change in F to 25 digits).
- Both checkers rerun at all settings and on all corruptions: everything as expected. Fresh-clone reproduction OK.
- No C42 claim below 0.686360751 in teorth/optimizationproblems (4 Nov 2026).
- Remaining before a public claim: a human reading of Theorem A, Lemma C1 and NOTES §6; then submit
  `submission/` (fill `<REPO_URL>`, `<COMMIT>`).
