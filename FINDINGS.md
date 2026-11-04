# Findings: Harcos's reduction is exact at finite n, and evidence that smooth phase profiles are near-optimal

Status: unrefereed note from this project (session S4, 2026-11-03). Section 2 has complete, elementary proofs.
Sections 5–7 report **floating-point numerics** (local optimisation). The only rigorous numbers in them are the Arb
upper bounds for R_n at the finite n listed, which say nothing rigorous about C42. Nothing here changes the
certified upper bound for C42. The current record is Pinho's 0.6863607950* (teorth [PR #305](https://github.com/teorth/optimizationproblems/pull/305)).

Notation follows MATH.md.
- R_n = min over z ∈ ℂⁿ with max_j|z_j| = 1 of max_{1≤k≤n}|s_k(z)|, where s_k(z) = Σ_j z_j^k.
- C42 = limsup R_n.
- K = ⌊n/2⌋. ‖s‖ = max_k|s_k| on ℂ^K.

**Summary.**
- **Proved (§2):** R_n = min over first halves s ∈ ℂ^{⌊n/2⌋} of max(max_k|s_k|, F_n(s)), and equivalently R_n is the
  unique fixed point of G_n(T) = min_{‖s‖≤T} F_n(s). Harcos's linearization loses nothing at any finite n. Every
  construction (any split, Koshta-type corrections, any modulus profile) is dominated by the completion of its own
  first half.
- **Not proved:** any lower bound for R_n, or that the optima below are global.
- **Rigorous but finite-n only:** Arb-certified upper bounds such as R_400 ≤ 0.690120598 and
  R_3200 ≤ 0.687129253.
- **Numerical evidence (n = 200…3200):**
  - about 2,000 local searches from random, including fully non-smooth, first halves find a single basin, at
    constant modulus;
  - the optimum is the smooth limit profile plus a boundary layer whose gain decays;
  - validated extrapolation puts lim T*_n at 0.68636 ± ~2e-5, the same as the smooth phase-only family.
- **Directions:** B, C and D add nothing beyond direction A. This is proved for β ≥ ½; for the others it rests on
  the evidence above.

## 1. Objects

For a first half s = (s_1..s_K) ∈ ℂ^K put

  A_s(x) = exp(−Σ_{k≤K} s_k x^k/k) = Σ_m α_m x^m,  P_m = α_0 + … + α_m  (so Σ P_m x^m = A_s(x)/(1−x)),

  D_n(s) = Σ_{k=K+1}^{n} |P_{n−k}|/k,  F_n(s) = |P_n| / D_n(s).

D_n(s) ≥ |P_0|/n = 1/n > 0, so F_n is a continuous function on ℂ^K. P_m is a polynomial in s_1..s_m.

For σ ∈ ℂⁿ let W_σ(x) be the truncation to degree ≤ n of exp(−Σ_{k≤n} σ_k x^k/k).

## 2. Theorem: the reduction is exact

**Lemma 0 (Newton, standard).**
(a) For every tuple z ∈ ℂⁿ, Π_j(1 − z_j x) = W_{s(z)}(x).
(b) For every σ ∈ ℂⁿ there is a tuple z ∈ ℂⁿ with s_k(z) = σ_k for all k ≤ n.
(c) For every σ ∈ ℂⁿ, 1 is an entry of every such z iff W_σ(1) = 0.

*Proof.*
(a) As formal power series, Π_j(1 − z_j x) = exp(Σ_j log(1 − z_j x)) = exp(−Σ_{k≥1} s_k(z) x^k/k). The coefficient of
x^m on the right depends only on s_1..s_m. The left side is a polynomial of degree ≤ n. So it equals the degree-≤n
truncation of exp(−Σ_{k≤n} s_k x^k/k), which is W_{s(z)}.

(b) W_σ(0) = 1. Factor W_σ(x) = Π_{j≤d}(1 − z_j x) over ℂ, where d = deg W_σ ≤ n and the z_j are the reciprocals of
the roots. Put z_{d+1} = … = z_n = 0. By (a), W_{s(z)} = W_σ. The coefficients w_m of W_σ satisfy
m·w_m = −Σ_{k=1}^{m} σ_k w_{m−k}, with w_0 = 1. This recursion lets one read off σ_1..σ_n from w_1..w_n, one at a
time. So σ ↦ W_σ is injective, and s_k(z) = σ_k for k ≤ n.

(c) If s(z) = σ, then by (a) W_σ(1) = Π_j(1 − z_j). This vanishes iff some z_j = 1. ∎

**Lemma 1 (normalisation, standard).** R_n = inf { max_k |s_k(z)| : z ∈ ℂⁿ, z_1 = 1 }, and the minimum defining
R_n is attained.

*Proof.* The set {max_j|z_j| = 1} is compact and z ↦ max_k|s_k(z)| is continuous, so R_n is attained.
- Rotating z by a unimodular factor so that a root of modulus 1 becomes 1 does not change any |s_k|. So the
  infimum on the right is ≤ R_n.
- Conversely, take z with z_1 = 1 and let λ = max|z_j| ≥ 1. Then z/λ has maximum modulus 1, and
  |s_k(z/λ)| = |s_k(z)|/λ^k ≤ |s_k(z)|. So R_n is ≤ the infimum. ∎

**Theorem 1 (Harcos's reduction is exact).** For every n ≥ 2,

  R_n = min_{s ∈ ℂ^K} max( ‖s‖, F_n(s) ),

and the minimum is attained. More precisely:
- (i) Every tuple z with z_1 = 1 satisfies max_{K<k≤n}|s_k(z)| ≥ F_n(s_1(z)..s_K(z)).
- (ii) For every s ∈ ℂ^K there is a tuple z with z_1 = 1, s_k(z) = s_k for k ≤ K, and |s_k(z)| = F_n(s) for all
  K < k ≤ n.

*Proof.*
Let σ ∈ ℂⁿ and write s = (σ_1..σ_K) and E(x) = Σ_{K<k≤n} σ_k x^k/k. Then

  exp(−Σ_{k≤n} σ_k x^k/k) = A_s(x)·exp(−E(x)).

Every monomial of E² has degree ≥ 2(K+1) ≥ n+1, because 2⌊n/2⌋ + 2 > n. Hence exp(−E) ≡ 1 − E mod x^{n+1}, and

  W_σ(x) ≡ A_s(x)·(1 − Σ_{K<k≤n} σ_k x^k/k)  (mod x^{n+1}).

Put x = 1 in the truncation. The sum of the coefficients of degree ≤ n is

  W_σ(1) = P_n − Σ_{K<k≤n} (σ_k/k)·P_{n−k}.                                                     (★)

(i) Let z_1 = 1 and σ = s(z). By Lemma 0(a),(c), W_σ(1) = 0. So P_n = Σ_{k>K}(σ_k/k)P_{n−k}, and

  |P_n| ≤ max_{k>K}|σ_k| · D_n(s).

Divide by D_n(s) > 0.

(ii) Put λ = P_n/D_n(s). For K < k ≤ n let σ_k = λ·conj(P_{n−k})/|P_{n−k}|, or σ_k = λ if P_{n−k} = 0. Then

  Σ_{k>K}(σ_k/k)P_{n−k} = λ·D_n(s) = P_n,

so W_σ(1) = 0 by (★). Every |σ_k| = |λ| = F_n(s). Lemma 0(b),(c) gives a tuple z with s(z) = σ that contains 1.
Re-index it so that z_1 = 1.

*The identity.* By Lemma 1, R_n is the infimum of max_k|s_k(z)| over z with z_1 = 1.
- For such z, (i) gives max_k|s_k(z)| ≥ max(‖s‖, F_n(s)), where s is the first half of s(z).
- By (ii), every value max(‖s‖, F_n(s)) is attained by some z.

So R_n = inf_s max(‖s‖, F_n(s)). The infimum is attained at the first half of a minimiser of R_n, which exists by
Lemma 1 after rotation. ∎

The proof uses nothing beyond Newton's identities and 2⌊n/2⌋ + 2 > n.
- Inequality (ii) is Harcos's linearization (Lemma 2 in MATH.md).
- The full-tuple statement R_n = min{max_k|S_k| : b_n(S) = 0} is in Koshta's NOTES §2.
- What Theorem 1 adds is the matching lower bound (i): eliminating the second half loses nothing. We do not claim
  the observation is new.

**Corollary 2 (fixed-point form).** Let G_n(T) = min_{‖s‖≤T} F_n(s). Then:
- G_n is continuous and nonincreasing on [0,∞);
- G_n(0) = 1/(H_n − H_K) > 0, where H_m is the m-th harmonic number;
- R_n is the unique T with G_n(T) = T.

Moreover, for every T_0:
- if G_n(T_0) > T_0, then T_0 < R_n ≤ G_n(T_0);
- if G_n(T_0) ≤ T_0, then R_n ≤ T_0.

*Proof.*
- Write s = Tu with u in the closed unit polydisc 𝔻^K. (T,u) ↦ F_n(Tu) is continuous and 𝔻^K is compact, so
  G_n is continuous. It is nonincreasing because the feasible sets grow with T.
- At s = 0, P_m = 1 for all m, which gives G_n(0).
- h(T) = G_n(T) − T is continuous and strictly decreasing, with h(0) > 0 and h(G_n(0)) ≤ 0. So h has exactly one
  root T*.
- By Theorem 1, R_n = min_T max(T, G_n(T)).
  - For T < T*: max(T, G_n(T)) ≥ G_n(T) ≥ G_n(T*) = T*.
  - For T ≥ T*: max(T, G_n(T)) ≥ T ≥ T*.
  - At T* the value is T*.
- The two bracketing statements follow in the same way from the monotonicity of h. ∎

**Corollary 3 (every construction is a choice of first half).** Take any construction of a tuple z with z_1 = 1.
This covers any split β, any correction scheme (Koshta's T_3/T_4 terms) and any modulus profile. Its value
max_k|s_k(z)| is ≥ max(‖s‖, F_n(s)), the value of the Harcos completion of its own first half s = (s_k(z))_{k≤K}.
This is immediate from (i).

So at every n the only freedom is the first half s ∈ ℂ^K. Bakeas's family is the sub-family
s_k = T e^{iφ(k/n)} with φ independent of n.

## 3. Remarks on rigor and scope

- Theorem 1 is exact at each fixed n. It does **not** by itself identify lim R_n or C42 = limsup R_n.
- An upper bound for C42 still needs a limit theorem for a family of first halves (Bakeas's Theorem A).
- A lower bound for R_n needs a *global* lower bound for G_n(T): a non-convex problem in 2K real variables. Nothing
  in this project proves one for any n ≥ 3.
- Corollary 2 says exactly what a global computation would give: R_n = T*_n. A local computation gives only
  R_n ≤ T*_n (local) and, more generally, R_n ≤ max(T, F_n(s)) for the specific s found.

## 4. What S1's n = 400 experiment shows, precisely

S1 reported the following. At T_0 = 0.68637, 120 random starts of L-BFGS on the n/2 = 200 phases, with
|s_k| = T_0, all ended at F_400 = 0.69465442. The free first half beats the smooth profile by an amount that
shrinks like n^{−0.8}.

- **What the number is.** 0.69465442 is a local minimum value of F_400 on the torus {|s_k| = T_0}.
  - Nothing proves it is the global minimum on the torus. Nothing proves the polydisc {|s_k| ≤ T_0} has no lower
    value.
  - So it is a candidate for G_400(T_0), not a value of R_400.
- **What it proves about R_400.** Only an upper bound. Even that needs a rigorous evaluation of the specific tuple,
  which S1 did not do. Theorem 1(ii) gives R_400 ≤ max(T_0, F_400(s)) ≈ 0.69465.
  - This is not the best bound of this kind, because T_0 is the *limit* fixed point and not the n = 400 one.
  - Redoing the optimisation at the fixed point T*_400 (Corollary 2) and checking that tuple in Arb gives
    **R_400 ≤ 0.690120597996 (rigorous)**. The other n are in the table in §5.
- **Is the optimisation global?** No. It is local optimisation from many random starts. Every start reaching the same
  minimiser (up to complex conjugation, which is an exact symmetry since conj(1) = 1) is evidence of a single basin.
  It is not a proof. The problem is non-convex in 2K real variables, and the free-modulus runs show that worse
  local minima do exist (§5).
- **Conditional reading (unproved hypothesis).** Suppose the minimiser found at T*_400 is the global minimiser of
  F_400 on the polydisc {‖s‖ ≤ T*_400}. Then Corollary 2 gives R_400 = T*_400 = 0.6901205980 to float accuracy.
  Similarly, if G_400(T_0) really were 0.69465 > T_0, then Corollary 2 would give 0.68637 < R_400.
- **About C42.** Nothing rigorous. Finite-n values of R_n do not bound limsup R_n in either direction.

## 5. Finite-n experiments (this session)

Script: `src/finite_n_study.py`. Raw results: `results/finite_n/n*.json`. Table: `logs/finite_n_fit.txt`.
All numbers are floats unless marked Arb. The steps for each n:
- **(a)** Constant modulus, T_0 = 0.68637. L-BFGS-B with exact JAX gradients, starting from the smooth Bakeas
  phases and from N random starts. The starts cycle through four kinds:
  - iid uniform phases;
  - random smooth phases (6-term Fourier series in k/n);
  - random linear phases (slopes up to ±12);
  - smooth plus iid noise of 0.5 rad.
- **(b)** Free modulus |s_k| ≤ T_0, via the smooth unconstrained map s_k = T_0·2w_k/(1+|w_k|²) onto the closed
  disc. Random moduli in [0,1] (iid or constant) and random phases of the four kinds.
  - S1 used polar coordinates with 0 ≤ ρ_k ≤ 1. In those coordinates ρ_k = 0 is a spurious KKT point, because the
    phase of s_k is frozen there. Most of S1's "stuck at 0.736–1.12" runs were this artifact.
- **(c)** The sign of ∂F_n/∂ρ_k at the best constant-modulus optimum (first-order KKT test for ρ ≡ 1).
- **(d)** The fixed point T*_n = G_n(T*_n) of the best basin. This is Newton in T, with dG/dT from the envelope
  theorem and a warm start. Then a small multistart *at* T*_n.
- **(e)** The fixed points of the smooth ansatz s_k = T e^{iφ(k/n)}, with φ frozen at Bakeas's or Pinho's limit
  phases.
- **(f)** An Arb check, at 256 bits with python-flint `acb_series.exp`, of the decimal-rounded fixed-point tuple
  (`results/finite_n/tuple_fp_n*.json`). This gives a certified decimal B with R_n ≤ B.

| n | (a) starts → same minimiser (±conj) | F_n smooth (T_0) | F_n free (T_0) | (b) free-modulus starts → const. optimum | max_k ∂F/∂ρ_k | T*_n free | T*_n smooth (Pinho φ) | **Arb: R_n ≤** |
|---|---|---|---|---|---|---|---|---|
| 200 | 240/240 | 0.7012791587 | 0.7002074330 | 80/80 | −6.1e-3 | 0.6926489245 | 0.6931844434 | 0.692648924532 |
| 283 | 240/240 | 0.6978617171 | 0.6970489430 | 79/80 | −4.3e-3 | 0.6912250401 | 0.6916300881 | 0.691225040151 |
| 400 | 240/240 | 0.6952694034 | 0.6946544191 | 77/80 | −3.0e-3 | 0.6901205980 | 0.6904232564 | 0.690120597996 |
| 566 | 240/240 | 0.6932215836 | 0.6927584209 | 77/80 | −2.1e-3 | 0.6892593677 | 0.6894846675 | 0.689259367693 |
| 800 | 240/240 | 0.6916388854 | 0.6912895253 | 79/80 | −1.5e-3 | 0.6885930700 | 0.6887590118 | 0.688593070017 |
| 1131 | 100/100 | 0.6904103725 | 0.6901459026 | 37/40 | −1.0e-3 | 0.6880762784 | 0.6881988096 | 0.688076278412 |
| 1600 | 80/80 | 0.6894657230 | 0.6892638369 | 31/32 | −7.4e-4 | 0.6876758286 | 0.6877647203 | 0.687675828651 |
| 2263 | 16/16 | 0.6887354120 | 0.6885798062 | 6/8 | −5.2e-4 | 0.6873669817 | 0.6874318084 | 0.687366981675 |
| 3200 | 8/8 | 0.6881758821 | 0.6880543110 | 4/4 | −3.7e-4 | 0.6871292524 | 0.6871763603 | 0.687129252420 |

Observations (numerical):
1. **One basin at constant modulus.** At every n, *every* constant-modulus start, 1,404 in total, ended at the same
   minimiser or its complex conjugate (max phase distance < 1e-4). The same holds for the 24/24/…/2 extra starts
   at T*_n: none went below the fixed-point basin. These starts include iid-random phases, i.e. maximally
   non-smooth first halves.
2. **Free modulus does not help.**
   - 470 of 484 free-modulus starts reached the constant-modulus optimum, with every |s_k| = T_0.
   - The other 14 ended at genuinely worse stationary points (F_400 = 0.70526, F_800 = 0.69745,
     F_1600 = 0.70413, …). In each of these, 1 to 5 of the s_k are ≈ 0 (|s_k| < 4e-5·T_0) and the rest
     have modulus T_0.
   - No run beat the constant-modulus optimum.
   - At that optimum, ∂F_n/∂ρ_k < 0 for every k and every n: ρ ≡ 1 satisfies the first-order KKT conditions. This
     is the finite-n counterpart of S1's limit sensitivity (MATH.md O3).
3. **The optimum is "smooth + boundary layer".** Here is the phase difference θ_free − θ_smooth(Bakeas) at T_0
   (`logs/finite_n_fit.txt`):
   - In the k-scale it does **not** shrink. It is ≈ −0.16 at k = 1, −0.13 at k = 2, −0.09 at k = 4 for every n
     from 200 to 3200.
   - At fixed u = k/n it **does** shrink. At u = 0.1 it is −0.054 (n = 200), −0.023 (800), −0.009 (3200), roughly
     like n^{−0.65}.

   So the free optimum is a finite-n re-tuning of the smooth profile plus a boundary layer at fixed k.
   - A fixed-k boundary layer changes A_s(x) by an n-independent factor E(x) = exp(−Σ_{k≤L} δ_k x^k/k).
   - *Heuristic:* to leading order this multiplies every P_m with m → ∞ by the same constant E(1), which cancels
     in F_n = |P_n|/D_n. So its gain should vanish as n → ∞.
   - *Numerical check* (`src/boundary_layer_test.py`): graft the k ≤ 8 deviation of the n = 1131 optimum onto the
     smooth profile at larger n. The gain is −3.6e-4, −1.3e-4, −5.6e-5, −2.3e-5 at n = 400, 1600, 6400, 25600. It
     decays steadily, with local exponent ≈ 0.6–0.65.

## 6. Trends in n and extrapolation (heuristic)

Fit script: `src/finite_n_fit.py`. Output: `logs/finite_n_fit.txt`.

- **Gain of the free first half at fixed T_0.** Δ_n = F_free − F_smooth(Bakeas) = −1.07e-3 … −1.22e-4 for
  n = 200 … 3200.
  - Power fit: 0.070 n^{−0.79}. This confirms S1's n^{−0.8} over a 16× range of n.
  - The local exponents drift from 0.80 to 0.71 at the largest n.
- **Gap between fixed points.** Δ*_n = T*_smooth(Pinho φ) − T*_free = 5.4e-4 … 4.7e-5.
  - Power fit: 0.058 n^{−0.88}. The local exponents *increase* from 0.80 to 0.92.
  - So there is no sign of a non-zero limit at the 1e-5 level.
- **The expected exponents.**
  - The leading finite-n correction comes from Riemann sums of |p(t)| ~ t^{−a} near t = 0, with a = Re g(0). The
    Euler–Maclaurin "ζ-term" has size n^{−(1−a)}.
  - The next corrections are n^{−1} and n^{−(2−a)}.
  - Here a = 0.166 (Bakeas) and 0.112 (Pinho), so 1 − a = 0.83 and 0.89. This explains S1's empirical n^{−0.8};
    nothing special about free first halves is needed.
- **Validated extrapolation.**
  - We fitted three models over the window n ∈ [200, 3200]:
    - pow3: L + c n^{−α};
    - exp2: L + c_1 n^{−(1−a)} + c_2 n^{−1};
    - exp3: exp2 plus c_3 n^{−(2−a)}.
  - Each model was first tested on the φ-frozen smooth families. Their limits T* are known from the limit
    evaluator: 0.6863655395 (Bakeas φ) and 0.6863607720 (Pinho φ). The Bakeas family was computed up to
    n = 102400.

  | model | error of window fit, Bakeas φ | error of window fit, Pinho φ | L_free (window fit of T*_free) |
  |---|---|---|---|
  | pow3 | −5.9e-5 | −8.0e-5 | 0.686293 |
  | exp2 | +1.4e-5 | +3.7e-5 | 0.686373 / 0.686408 (a of Bakeas / Pinho) |
  | exp3 | +3.5e-6 | +1.2e-5 | **0.686356 / 0.686376** |

  - The best-validated model (exp3) puts the n → ∞ limit of the free-first-half fixed points at
    **0.68636 ± ~2e-5**.
  - The same model extrapolates the gap Δ*_n to −3e-6 … −4e-6. That is zero within the model's own validation
    error.
  - The free-form pow3 fit is visibly biased low on the families with known limits. Its 0.68629 should not be
    trusted.

## 7. Directions B, C, D: what is proved and what is evidence

By Corollary 3, **at every n every construction is dominated by Harcos's completion of its own first half**
(proved). So a direction can only help through the first half it induces.

- **C, split β ≥ ½: proved to give nothing.**
  - Prescribing more sums with modulus ≤ T on (K, βn] and solving for the rest can never beat max(T, F_n(s)) of the
    same first half.
  - (MATH.md O1 had an explicit convex-relaxation version.)
- **C, split β < ½: no new freedom.**
  - Such a construction (Koshta's τ = 41/200 is the extreme case) produces some first half (s_k)_{k≤K}. Its value
    is ≥ max(‖s‖, F_n(s)) by Theorem 1(i).
  - It is therefore a restricted way of choosing a first half. The products of free sums that need Koshta's
    T_3/T_4 terms are an artefact of not using the exact completion.
  - Whether its first half can beat the smooth one is the question settled numerically in §5–6 below.
- **D, hybrid with Koshta's correction terms: no new freedom.** By the same argument, the hybrid is again just
  some first half.
  - Koshta's first halves are piecewise constant, with jumps and modulus < T on the middle steps. They therefore lie
    in the "varying modulus / non-smooth" part of ℂ^K.
- **B, varying modulus ρ ≤ 1: not proved useless, but no evidence of gain.**
  - It is a genuine part of the first-half space, and Theorem A extends to it (MATH.md O2).
  - Evidence against a gain:
    - the limit radial sensitivity is < 0 everywhere (S1, O3);
    - at finite n the constant-modulus optimum satisfies the KKT sign condition at every n tested;
    - 470/484 unconstrained polydisc starts end at ρ ≡ 1;
    - the rest end at worse points.

**Bottom line (numerical, not a theorem).** The unrestricted first-half problem was searched locally with many
non-smooth, varying-modulus starts. At every n from 200 to 3200 it has one basin at constant modulus. Its minimiser
is the smooth limit profile plus a boundary layer whose gain vanishes with n, and the gap to the smooth family
decays like n^{−0.9}.

So directions B, C and D have nothing to add beyond the phase-only family of direction A. The record question
reduces to direction A plus certification (PLAN.md).

## 8. What this says about C42

- **Proved:** R_n is exactly the fixed point of G_n (Theorem 1, Corollary 2).
- **Proved, numerically certified:** R_n ≤ B_n for the n in §5 (Arb), e.g. R_3200 ≤ 0.687129252420. These are
  statements about finite n only.
- **Not proved:** any lower bound on R_n beyond what is known (Biró: just above ½). Any statement that these local
  optima are global.
- **Evidence:** suppose that (i) the single basin found at each n is the global optimum (so R_n = T*_n, Corollary 2),
  and (ii) the finite-n trend continues as validated in §6. Then
  - lim R_n exists and equals **≈ 0.68636 (± ~2e-5)**;
  - this is indistinguishable from the best smooth phase-only value 0.6863607 (Bakeas/Pinho family);
  - so C42 would be ≈ 0.68636, the current upper bound would be within ~1e-5 of the truth, and further improvements
    of the upper bound would be confined to roughly the 6th–8th significant digit.

  This is consistent with:
  - R_n decreasing in n through the whole range: 0.874, 0.825, 0.798, … at n = 2, 3, 4 (float multistart, `logs/reduction_crosscheck.log`),
    down to 0.6871 at n = 3200;
  - Cheer–Goldston's small-n numerics (≈ 0.7).

  It is **not** evidence for C42 near Biró's lower bound ½. Assumption (i) is the weak point: a better basin that
  random local search never finds cannot be excluded at n ≥ 3.

## 9. Reproduction

```
pip install python-flint==0.9.0 numpy scipy jax jaxlib
git clone https://github.com/Bakeas/turan-c42-bound third_party/bakeas   # profiles only, see third_party/README.md
git clone https://github.com/rpinho/c42-certificate third_party/pinho
python src/reduction_crosscheck.py                 # identity (★) on random tuples; tiny-n direct vs reduced, ~1 min
python src/finite_n_study.py 400 --starts 240 --rho-starts 80 --fp-starts 24   # ~7 min on 4 vCPU
python src/smooth_fixed_points.py 200 400 800 1600 # smooth phi-frozen fixed points (fast)
python src/limit_fixed_points.py                   # limit references, ~2 min
python src/finite_n_fit.py                         # tables and fits
```
The Arb check inside `finite_n_study.py` (step f) takes < 1 s for n ≤ 3200. It evaluates the stored decimal tuple
and certifies a decimal B > F_n(s) with B ≥ T.

## Credits and AI use

- The linearization is due to Harcos (in Biró 2000).
- The limit theory (Theorem A) and the phase-varying profile are Bakeas's.
- The refined profile and certificate are Pinho's ([PR #305](https://github.com/teorth/optimizationproblems/pull/305)).
- The exact full-tuple reformulation is in Koshta's NOTES §2.
- Griego, Röhrig and Plata developed the coefficient-cancellation line.

This note, the proofs in §2 and all code in `src/` were written by Claude (Anthropic) in an autonomous session for
Nikolaos Vatopoulos. The proofs are elementary but have not been reviewed by a human.
