# MATH.md — Harcos linearization, Bakeas's limit equation, and what Theorem A assumes

This is a summary of Bakeas, *turan-c42-bound* v1.1.0 (`third_party/bakeas/NOTES.md`, commit `dc9c491`),
with extensions by Pinho (`third_party/pinho/volterra-quadratic-knots`, commit `009120a`). Section 7 has
observations of this project. They are **new and unreviewed**. All credit for Lemmas 1–3 and Theorem A
goes to Bakeas, and for the linearization to Harcos (and Biró).

Tags used in Section 6:
- **[T]**: the step uses |s_k| = T exactly, i.e. constant modulus.
- **[½]**: the step uses the split K = ⌊n/2⌋, i.e. the breakpoint of the profile at u = ½.
- **[PL]**: the step uses that φ is piecewise linear.

---

## 1. Problem and normalization

R_n = inf over z_1 = 1, z_2..z_n ∈ ℂ of max_{k≤n} |s_k(z)|, where s_k(z) = Σ_j z_j^k, and C42 = limsup R_n.

**Lemma 1.** This equals the normalization max_j|z_j| = 1. Divide by the root of largest modulus. This does not
increase any |s_k|.

## 2. Harcos's linearization (exact for every n)

For prescribed s_1..s_K put A(x) = exp(−Σ_{k≤K} s_k x^k/k) = Σ α_m x^m and P_m = α_0+…+α_m. Then
Σ P_m x^m = A(x)/(1−x).

**Lemma 2.** Let n ≥ 2 and ⌊n/2⌋ ≤ K < n, and put D_n = Σ_{k=K+1}^{n} |P_{n−k}|/k (≥ 1/n). Then some tuple with
z_1 = 1 has s_k(z) = s_k for k ≤ K and |s_k(z)| = |P_n|/D_n for K < k ≤ n. Hence

  R_n ≤ max( max_{k≤K}|s_k| , F_n ),  F_n = |P_n|/D_n.

*Mechanism.* We need W(1) = 0, where W is the degree-≤n truncation of exp(−Σ_{k≤n} σ_k x^k/k). The free indices
satisfy k ≥ K+1 > n/2, so no product of two free terms reaches degree ≤ n. So mod x^{n+1},

  exp(−Σ σ_k x^k/k) ≡ A(x)·(1 − Σ_{k>K} σ_k x^k/k),

and W(1) = P_n − Σ_{k>K} (σ_k/k) P_{n−k} is **affine** in the free sums. The choice σ_k = λ·conj(P_{n−k})/|P_{n−k}|
with λ = P_n/D_n solves it with common modulus |λ| = F_n. By the triangle inequality this is the smallest
possible value of max_{k>K}|σ_k|. The roots of x^n W(1/x) are the tuple, and 1 is among them. The inequality
2K+2 > n is the only place where K ≥ ⌊n/2⌋ is used.

## 3. The profile and the limit objects

- Profile on [0,½]: g(u) = T·e^{iφ(u)}. Bakeas: φ piecewise linear with knots j/48, j = 0..24. Pinho: knots
  j²/4608, j = 0..48.
- s_k = g(k/n) for k ≤ K = ⌊n/2⌋. Put γ = g(0) and a = Re γ.
- d(u) = γ − g(u) on (0,½], d(u) = γ on (½,1], d(0) = 0. Put κ = sup_{(0,1]}|d|.
- Slopes β_j, Λ = max|β_j|, M = max(TΛ, 2T). Then |d(u)| ≤ Mu on (0,1].

**Exact discrete identities.** Put c_k = 1 − s_k (k ≤ K) and c_k = 1 (k > K). Then

  Σ_m P_m x^m = exp(Σ_k c_k x^k/k) = (1−x)^{γ−1} D(x),  D(x) = exp(Σ d_k x^k/k),  d_k = d(k/n) for all k ≥ 1.

The last equality needs "k > K ⇔ k/n > ½", i.e. K = ⌊n/2⌋ and the breakpoint of d at exactly ½.
Equivalently m P_m = Σ_{k=1}^m c_k P_{m−k}, and P_j = Σ_i a_{j−i} δ_i with a_j the coefficients of (1−x)^{γ−1}.
Here δ_m are the coefficients of D, and m δ_m = Σ_k d_k δ_{m−k}. With ρ_n(m) = n δ_m and t_m = m/n:

  t_m ρ_n(m) = d(t_m) + (1/n) Σ_{k=1}^{m−1} d(t_k) ρ_n(m−k).       (discrete Volterra equation)

**The integral equation.** Equation (2):

  t ρ(t) = d(t) + ∫_0^t d(u) ρ(t−u) du,  0 < t ≤ 1.

Equivalently ρ = d(t)/t + 𝒦ρ with (𝒦f)(t) = t^{−1}∫_0^t d(u) f(t−u) du, and ‖𝒦‖_∞ ≤ κ. Then

  p(t) = t^{−γ} + ∫_0^t (t−v)^{−γ} ρ(v) dv,   F = |p(1)| / ∫_0^{½} |p(t)| dt/(1−t).

**Equivalent homogeneous form** (used by `src/evaluator.py`). Put c(u) = 1 − g(u) on [0,½] and c = 1 on (½,1].
Then p solves

  t p(t) = ∫_0^t c(u) p(t−u) du,  with p(t) ~ t^{−γ} at 0,

and this solution is unique up to a scalar, which cancels in F. *Check by Laplace transform.* Put
P = ℒp = Γ(1−γ)s^{γ−1}(1+ℒρ). Equation (2) reads −(ℒρ)' = ℒd·(1+ℒρ). Then −P' = P·((1−γ)/s + ℒd) = ℒc·P,
which is ℒ of the homogeneous equation. This is the continuum version of m P_m = Σ c_k P_{m−k}.

**Harcos's case.** For constant φ: d = 0 on (0,½], ρ = 0 there, ρ = γ/t on (½,1], and with w = −γ,
F = |1 − w∫_0^{½} t^w/(1−t)dt| / ∫_0^{½}|t^w|/(1−t)dt. At w = −0.43246 − 0.54237i this is
0.693676303073… (mpmath, this project).

## 4. Theorem A (Bakeas)

**Theorem A.** Assume κ < 1 and a ≥ 0. Then automatically a ≤ |γ| ≤ κ < 1, because |d| = |γ| on (½,1]. Under
these assumptions:
1. (2) has exactly one bounded solution ρ.
2. The denominator of F is in (0,∞), and F_n → F.
3. Hence limsup R_n ≤ max(T, F).

**Proof outline (Bakeas Steps 1–5).**
1. Exact identities (Section 3).
2. Neumann series in the sup norm, since ‖𝒦‖ ≤ κ < 1. This gives ‖ρ‖ ≤ B = M/(1−κ). Regularity: ρ is uniformly
   continuous on (0,½] and on (½,1], with one jump at ½.
3. Discrete stability. Induction gives |ρ_n(m)| ≤ B. The error e_m = ρ_n(m) − ρ(t_m) satisfies
   |e_m| ≤ κ max_{l<m}|e_l| + |τ_m|/t_m, where τ_m is a Riemann-sum defect. Hence max|e_m| → 0. At most two cells
   are discontinuous, and only when t_m > ½.
4. Asymptotics of P_j. Π_n(j) = Γ(1−γ) n^γ P_j. (B2) domination: |Π_n(j)| ≤ C t^{−a}. (B1) uniform convergence to
   p on [ε,1], via η_l = Γ(l+1−γ) l^γ/Γ(l+1) → 1 and a Riemann-sum argument with one jump cell.
5. The numerator converges by (B1). The denominator converges by dominated convergence, with the majorant
   2C t^{−a} ∈ L¹ because a < 1.

## 5. The certified inequality F < T (Lemma 3 and (8))

- **Lemma 3.** For any bounded ρ̃ with residual R(t) = d + d*ρ̃ − tρ̃:
  ‖ρ − ρ̃‖_∞ ≤ ε := sup|R/t| / (1−κ).
- Then |p − p̃| ≤ ε t^{1−a}/(1−a), and

  F ≤ (|p̃(1)| + ε/(1−a)) / ( ∫_0^{½}|p̃|/(1−t) − ε ∫_0^{½} t^{1−a}/((1−a)(1−t)) ).      (8)

  The denominator bound must be checked positive (the v1.0.0 bug). Either way the bound claimed is T, never F.
- `rigorous_limit.py` (192-bit Arb) works as follows:
  - ρ̃ is piecewise linear on N = 48M cells, with a jump at ½.
  - R is evaluated at the nodes exactly via closed-form moments of d against hats. This divides by β_j, so β_j ≠ 0
    is checked.
  - The cell interiors are controlled by a bound on R'' and B_3 ≥ |R'''|, which uses |d^{(r)}| ≤ Tβ^r.
  - The first cell uses R(0) = 0 and R'(0⁺).
  - The numerator uses exact moments of u^{−γ}.
  - Denominator: a lower bound t^{−a} − P_* t^{1−a}/(1−a) on [0,t_*], and the linear interpolant minus a Peano
    bound with p̃'' on [t_*,½].
  - Input checks: 25 phases, knots j/48, 0 < a < 1, |ψ_j−ψ_0| < 3, κ < 1, β_j ≠ 0, D2 positivity, finite bounds.
- `independent_check.py` follows the same route (Lemma 3 + (8)) with different code: a ramp representation of ρ̃,
  cell-wise enclosure of R'', and a smaller cut near 0.
- **Pinho's extensions** (unreviewed by a human; Bakeas checked them):
  - non-uniform rational knots;
  - a factored denominator q = t^γ p̃;
  - a sharpened first cell, using a Hermite remainder and the exact R''' on [0,h];
  - an *integrated* residual bound from the Gronwall/integrating-factor inequality
    e(t) ≤ r(t) + T t^{T−1}∫_0^t r v^{−T} dv. This uses κ = T;
  - a local contraction constant k_δ ≤ T on [0,δ].

## 6. Hypotheses and where they are used

| # | Hypothesis / fact | Used in | [T] | [½] | [PL] |
|---|---|---|:-:|:-:|:-:|
| H1 | ⌊n/2⌋ ≤ K < n, so 2K+2 > n | Lemma 2, Step 4 (W(1) affine in the free sums) | | **yes** | |
| H2 | s_k = g(k/n) and d_k = d(k/n) for all k (k > K ⇔ k/n > ½) | identity (4), discrete Volterra eq. | | **yes** | |
| H3 | max_{k≤K}|s_k| = T, giving the form max(T, F_n) | (1), conclusion of Theorem A | **yes** (in general: sup|g|) | | |
| H4 | κ = sup|d| < 1 (sup-norm contraction) | Step 2 (Neumann series, B = M/(1−κ)); Step 3 (|ρ_n| ≤ B, error recursion); Lemma 3 | partly: κ = T on (½,1] since |γ| = T; on (0,½] |d| = 2T|sin(Δφ/2)| | **yes** (d ≡ γ after ½) | the checker's "max at a knot" argument |
| H5 | 0 ≤ a < 1 | t^{−γ} ∈ L¹; (B2) (v^{−a} decreasing); D2 positivity; (8) | a < 1 follows from |γ| ≤ κ < 1 | | |
| H6 | |d(u)| ≤ M u (Lipschitz at 0) | d/t bounded ⇒ ρ bounded; Step 3 bounds; (B2) | the bound T|1−e^{iΔφ}| ≤ TΛu | 2Tu ≥ T on (½,1] | Λ = max slope (any Lipschitz φ works) |
| H7 | d uniformly continuous on (0,½] and on (½,1]; one jump at ½ | Step 3 (≤ 2 bad cells, only for t_m > ½); (B1) (≤ 1 jump cell) | | **yes** | only continuity |
| H8 | Re(1−γ) > 0 | η_l → 1 (Gamma ratio), C_η < ∞ | | | |
| H9 | R'' formula, |d^{(r)}| ≤ Tβ^r, d' = 0 after ½ | checker C2/C3 only | **yes** | **yes** | **yes** |
| H10 | closed-form moments of d over cells (division by β_j) | `rigorous_limit.py` C1 only | **yes** | | **yes** |
| H11 | knots aligned with the grid; ½ a grid point | both checkers | | **yes** | **yes** (breakpoints) |
| H12 | denominator lower bound in (8) is positive | (8) | | | |
| H13 | Pinho's Gronwall factor t^T, Beta(T, 1−a) | Pinho's integrated residual bound | **yes** (uses |d| ≤ T) | | |

**Summary.**
- The *analysis* (Steps 1–5) needs only:
  - g piecewise continuous on [0,½], Lipschitz at 0;
  - κ < 1 and 0 ≤ a < 1.

  Constant modulus is used only through (i) the final form max(T, F) and (ii) the easy evaluation of κ and M.
  Piecewise linearity is used only for M (any Lipschitz φ works) and for the checkers' closed forms. The ½ split
  is used structurally: in Lemma 2, in (4), and in the jump location.
- The *checkers* are much more specific. They depend on [T], [PL] and the grid (H9–H11, H13).

## 7. Observations of this project (new, unreviewed, not part of any claim)

**O1. A split β > ½ gives nothing new (finite n, exact).** Take K' > K ≥ ⌊n/2⌋. The sums s_k with K < k ≤ K'
enter only linearly, again because 2K+2 > n:
- P'_n = P_n − Σ_{K<k≤K'} (s_k/k) P_{n−k};
- P'_{n−k} = P_{n−k} for k > K'.

So prescribing s_k on (K,K'] with |s_k| ≤ T and leaving k > K' free is a convex relaxation of the same linear
constraint. Its optimum F' satisfies F'·D_2 = max(0, |P_n| − T·D_1), where D_1 and D_2 are the parts of D_n
over (K,K'] and (K',n]. At F = T this gives F' = T. So for the purpose of max(T,·), β ≥ ½ is equivalent to
β = ½. A split β < ½ breaks Lemma 2: products of free sums reach degree n, and W(1) becomes a polynomial of
degree ⌊1/β⌋ in the free sums. This is the source of Koshta's T_3/T_4 terms. With a common free modulus it
again reduces to the β = ½ problem at its fixed point.

**O2. Varying modulus (direction B) and the proof.** Allow g = Tρe^{iφ} with 0 ≤ ρ ≤ 1 piecewise Lipschitz.
Then:
- H3 becomes max(sup|g|, F) = max(T, F);
- H6 holds with M = T(Λ_φ + Λ_ρ) on (0,½] and 2|γ| on (½,1];
- H4 needs a direct bound on sup|γ − g(u)|;
- H5 holds as before, since |γ| = |d| on (½,1].

Steps 1–5 do not use |g| = T otherwise. So Theorem A extends verbatim, *except* that the checkers' closed forms
and derivative bounds (H9, H10, H13) must be redone. This has not been reviewed.

**O3. First-order test of direction B (numerical, `results/sensitivity_*.npz`).** The radial sensitivity is the
density r(u) = δF/δ log|g|(u). It was computed with the exact discrete adjoint of the evaluator, at N = 9216
(Pinho) and N = 6144 (Bakeas). It is **strictly negative on all of (0,½]**:
- it ranges from −13.4 (near u = 0) to −1.16 (at u = ½);
- its integral is T·∂F/∂T = −0.839.

So lowering |g| anywhere raises F to first order, and ρ ≡ 1 satisfies the KKT conditions of the problem with
ρ ≤ 1. The tangential density is ≤ 6·10⁻² near 0 and ~10⁻⁵ in the bulk, which confirms phase-stationarity.

**O4. The contraction hypothesis can probably be weakened.** In the weighted norm sup e^{−λt}|f(t)|, 𝒦 has norm
≤ sup_t t^{−1}∫_0^t |d(u)|e^{−λu}du. With |d| ≤ min(Mu, κ_∞) this tends to 0 as λ → ∞. So κ < 1 should be
replaceable by "d bounded and Lipschitz at 0". The discrete Step 3 and Lemma 3 need the same weights. This
matters only for profiles with |γ − g(u)| ≥ 1 somewhere, which the current optima are far from
(κ = T = 0.686). Unproved; listed only as a possible tool.

**O5. Harcos's reduction loses nothing at finite n.** Take any tuple with z_1 = 1. Its power sums satisfy
W(1) = P_n − Σ_{k>⌊n/2⌋}(s_k/k)P_{n−k} = 0, by the same degree count as in Lemma 2. Hence
max_{k>⌊n/2⌋}|s_k| ≥ |P_n|/D_n = F_n(s_1..s_{⌊n/2⌋}). Lemma 2 gives the reverse inequality. So

  R_n = min over (s_1..s_{⌊n/2⌋}) ∈ ℂ^{⌊n/2⌋} of max( max_{k≤n/2}|s_k| , F_n ).

Koshta's §2 states an equivalent fact. Consequences:
- every construction is a choice of the first half;
- Bakeas's limit family restricts only by the ansatz s_k = g(k/n);
- directions C (any β) and D are reparametrisations within B, or non-smooth first halves.

The finite-n experiments in LOG.md (S1, 16:15) find a single basin at constant modulus for n = 400. They also find
that the advantage of a free first half over the smooth one decays like n^{−0.8}.
