# NOTES_CERTIFIER.md: a spectral a-posteriori certifier for F in Bakeas's Theorem A

Status: **new and unreviewed** (written in session S3, 3 Nov 2026, by an AI agent). Theorem A, Lemma 3 and the
limit equation are Bakeas's (turan-c42-bound v1.1.0). The integrated (Gronwall) propagation bound is Pinho's
(PR #305 bundle, VOLTERRA_VERIFIER_MATH.md). Everything else below is the new part: the trial space, the exact
residual, the representation of p̃, and the denominator bound. The program is `checkers/rigorous_check.py`. Its only
floating-point part is the witness `checkers/spectral_witness.py`.

## 0. What is certified

The input is a profile (T, knots 0 = u_0 < … < u_K = ½, phases ψ_0..ψ_K) with exact rationals, and a claim C with
T ≤ C < 1 (default C = T). Put φ = the piecewise linear interpolant of ψ, g = T e^{iφ}, γ = g(0), a = Re γ, and

  d(u) = γ − g(u) on (0,½],   d(u) = γ on (½,1],   d(0) = 0.

The program certifies, in outward-rounded ball arithmetic, the hypotheses of Theorem A and the inequality

  F = |p(1)| / ∫_0^½ |p(t)| dt/(1−t)  <  C,     p(t) = t^{−γ} + ∫_0^t (t−v)^{−γ} ρ(v) dv,

where ρ is the bounded solution of  t ρ(t) = d(t) + ∫_0^t d(u) ρ(t−u) du  on (0,1].
By Theorem A, limsup R_n ≤ max(T, F) ≤ C.

**Hypotheses checked (all rigorous):**
- T, C, knots and phases are exact rationals; 0 < T ≤ C < 1; the knots strictly increase from 0 to ½.
- The grid size N is even and N·u_j ∈ ℤ for every knot (so ½ and all knots are grid points).
- a = Re γ satisfies 0 < a < 1.
- κ := sup_{(0,1]} |d| < 1. Its computation:
  - |d| = T on (½,1].
  - On (0,½], |d(u)| = 2T|sin((φ(u)−ψ_0)/2)|. The function |φ − ψ_0| is piecewise linear, so its maximum E is
    attained at a knot.
  - If E < π (certified), then sup = 2T sin(E/2), because x ↦ sin(x/2) is increasing on [0, π]. Otherwise the
    program uses 2T.
  - κ = max(T, that value).
- All final quantities are finite balls, and the denominator lower bound is > 0.
- PASS iff the upper endpoint of the F-ball is < C.

These are the same as the checks of Bakeas's and Pinho's checkers, except:
- no check β_j ≠ 0 (nothing divides by a slope);
- no restriction to 25 phases or knots j/48;
- no restriction max|ψ_j − ψ_0| < 1 (κ is computed instead).

## 1. Grid and trial function (the only floating step)

- Grid: h = 1/N, cells I_m = [mh, (m+1)h], m = 0..N−1, local variable θ ∈ [0,1], t = h(m+θ). H = N/2.
- **Trial ρ̃.** On each cell a polynomial ρ̃_m(θ) = Σ_{b≤P} c_{m,b} θ^b, discontinuous across cells.
  - The c_{m,b} are exact (high-precision dyadic) numbers. They are the midpoints of an Arb interpolation of
    floating nodal values, which come from GMRES on the collocation equations, optionally refined with nodal
    residuals computed in Arb.
  - **Nothing below depends on how the c_{m,b} were obtained.**
  - Convention: ρ̃(½) is the value of the left cell (cell H−1) at θ = 1. Every other grid value is either
    neighbour's value; both cases are covered, because every bound below is taken over the closed cell [0,1].
- **Why this converges fast.** The breakpoints of ρ are the knots, ½, and finite sums of these. Away from them
  ρ is analytic. For knots j²/4608 (Pinho) or j/48 (Bakeas) all such sums are grid points.
  - So a piecewise polynomial on a knot-aligned grid is a spectrally accurate trial.
  - The worst cells are those just after the first knots. There the analytic continuation of the piece has a
    singularity about one knot-spacing away, so the convergence rate is about (4s+2)^{−P}, where s is the
    number of cells per knot spacing.
  - This is heuristic and is used only to choose N and P.

## 2. Convolution of piecewise polynomials on a uniform grid (exact identity)

**Lemma C1.** Let f, g be piecewise polynomials on the grid, with f_i(σ) = Σ_a f_{i,a}σ^a and
g_j(τ) = Σ_b g_{j,b}τ^b. Write f̌_{i,a}, ǧ_{j,b} for the coefficients in the reversed variable w = 1 − θ
(f_i(1−w) = Σ_a f̌_{i,a} w^a). Then for t = h(m+θ), θ ∈ [0,1]:

  (f*g)(t) = ∫_0^t f(u) g(t−u) du
           = h [ Σ_c θ^{c+1}/(c+1)! · U_{m,c}  +  Σ_c (1−θ)^{c+1}/(c+1)! · V_{m−1,c} ],

  U_{m,c} = Σ_{i=0}^{m} Σ_{a+b=c} (a! f_{i,a})(b! g_{m−i,b}),   V_{m,c} = Σ_{i=0}^{m} Σ_{a+b=c} (a! f̌_{i,a})(b! ǧ_{m−i,b}),

with V_{−1} := 0.

*Proof.*
1. Substitute u = h(i+σ) and split cell i at σ = θ.
   - For σ < θ: t−u lies in cell m−i, with local variable θ−σ.
   - For σ > θ: t−u lies in cell m−i−1, with local variable 1+θ−σ. This case does not occur for i = m.
2. The first part uses ∫_0^θ σ^a (θ−σ)^b dσ = a! b! θ^{a+b+1}/(a+b+1)!.
3. In the second part substitute s = 1−σ, and write λ = 1−θ. Then f_i(σ) = Σ f̌_{i,a} s^a and
   g_{m−1−i}(1+θ−σ) = Σ ǧ_b (λ−s)^b. The integral is ∫_0^λ, so the same Beta identity applies, with λ in
   place of θ. ∎

Packing (cell index, degree) into one polynomial variable with stride ≥ (deg f + deg g + 1) turns the double sums
into one univariate polynomial product (Kronecker substitution). This product is computed by `acb_poly`
multiplication, which is outward rounded. The identity was also checked independently by direct mpmath quadrature,
see `checkers/tests/mp_pointwise.py`.

## 3. The residual

**d on a cell.** On cell i < H, with φ_i the exact phase at the left end and z_i = iβ_i h:
- d(h(i+θ)) = γ − g_i e^{z_i θ}, where g_i = T e^{iφ_i}.
- Replace e^{z_iθ} by its Taylor polynomial of degree Q. Since z_i is imaginary, |e^{iy} − Σ_{k≤Q}(iy)^k/k!| ≤
  |y|^{Q+1}/(Q+1)! for real y. So |d − d_Q| ≤ ε_i := T|β_i h|^{Q+1}/(Q+1)! on cell i.
- On cell 0 the constant term is exactly 0, and |d − d_Q|(t) ≤ T(|β_0| t)^{Q+1}/(Q+1)!.
- On cells i ≥ H, d = γ exactly. The single point u = ½ does not affect any integral.

**Residual polynomial.** On cell m,

  R(t) = d(t) + (d*ρ̃)(t) − t ρ̃(t) = R_m(θ) + E_m(θ),
  R_m = d_{Q,m} + (d_Q*ρ̃)|_m − h(m+θ)ρ̃_m,

where (d_Q*ρ̃)|_m is given exactly by Lemma C1, and

  |E_m| ≤ ε_m + max_i ε_i · ∫_0^t |ρ̃| ≤ ε_m + ε_max · t · ‖ρ̃‖_∞.

R_m is a polynomial of degree D = P+Q+1. Its Chebyshev coefficients on [0,1] (in T_n(2θ−1)) are computed by an
exact rational linear map. The monomial-to-Chebyshev matrix on [0,1] has entries in [0,1]. Hence
sup_{[0,1]}|R_m| ≤ Σ_n |χ_n(R_m)|, and similarly ‖ρ̃‖_∞ ≤ max_m Σ_n |χ_n(ρ̃_m)|. The bounds per cell are:
- m ≥ 1:  r_m = (Σ|χ(R_m)| + ε_m)/(mh) + 2 ε_max ‖ρ̃‖, since (m+1)/m ≤ 2.
- m = 0:  R_0 has zero constant term, so R_0/θ is a polynomial. Then
  r_0 = Σ|χ(R_0/θ)|/h + T|β_0|^{Q+1}h^Q/(Q+1)! + ε_max‖ρ̃‖.

Then sup_{t∈I_m} |R(t)/t| ≤ r_m. Q is chosen so that T|β h|^{Q+1}/(Q+1)! < 2^{−110}.

## 4. From the residual to |ρ − ρ̃| (Bakeas's Lemma 3; Pinho's integrated form)

Put e = ρ − ρ̃. It is bounded and measurable, and t e = d*e + R. With |d| ≤ κ,

  |e(t)| ≤ r(t) + (κ/t) ∫_0^t |e|,  where r(t) = r_m on I_m.

- (sup) ‖e‖_∞ ≤ ε_∞ := max r_m / (1−κ).
- (integrated) Put E(t) = ∫_0^t |e|. Then (t^{−κ}E)' ≤ t^{−κ} r. Since E = O(t) and κ < 1, t^{−κ}E → 0 at 0. So
  E(t) ≤ t^κ A(t) with A(t) = ∫_0^t r v^{−κ} dv, and |e(t)| ≤ r(t) + κ t^{κ−1} A(t).

**Error in the numerator.** |p(1) − p̃(1)| ≤ ∫_0^1 (1−v)^{−a}|e(v)| dv ≤ EN, where EN is the smaller of
- ε_∞/(1−a), and
- Σ_m r_m ∫_{I_m}(1−v)^{−a} dv + κ A(1) B(κ, 1−a).

**Error in the denominator.**
∫_0^½ |p − p̃|/(1−t) ≤ 2∫_0^½ ∫_0^t (t−v)^{−a}|e(v)| dv dt = (2/(1−a)) ∫_0^½ |e(v)|(½−v)^{1−a} dv ≤ ED, where ED is
the smaller of
- 2ε_∞(½)^{2−a}/((1−a)(2−a)), and
- (2/(1−a)) [ (½)^{1−a} Σ_{m<H} r_m h + κ A(½)(½)^{κ+1−a} B(κ, 2−a) ].

All cell moments are explicit power differences, evaluated in Arb.

## 5. The trial integral p̃ = t^{−γ} + ∫_0^t (t−v)^{−γ} ρ̃(v) dv

Two elementary bounds (|γ| = T < 1):
- |binom(−γ,k)| = Π_{j<k} |γ+j|/(j+1) ≤ 1, and likewise |(γ)_l/l!| ≤ 1;
- |binom(1−γ,k)| ≤ Π_{j<k}(|1−γ|+j)/(j+1) ≤ k+1, since |1−γ| ≤ 2.

**Numerator.** p̃(1) = 1 + h^{1−γ} Σ_k ∫_0^1 (N−k−σ)^{−γ} ρ̃_k(σ) dσ, with the cases n' = N−k:
- n' = 1: ∫_0^1 (1−σ)^{−γ}σ^b dσ = B(b+1, 1−γ).
- n' = 2: ∫_0^1 (2−σ)^{−γ}σ^b dσ = Σ_e binom(b,e) 2^{b−e} (−1)^e (2^{e+1−γ} − 1)/(e+1−γ).
- n' ≥ 3: (n'−σ)^{−γ} = n'^{−γ} Σ_l (γ)_l/l! (σ/n')^l. Truncate at L(n') terms. The tail is at most
  n'^{−a} n'^{−L−1}/(1−1/n') · sup|ρ̃_k|, times h^{1−a}.

**Denominator: p̃ on a cell m ≥ 1.** Split the kernel u^{−γ} into kernel cell 0, which is treated exactly, and
kernel cells i ≥ 1.
- **Kernel cells i ≥ 1.** Approximate h^{−γ}(i+σ)^{−γ} by its Taylor polynomial of degree P_k about σ = ½.
  - The remainder is ≤ h^{−a}(i+½)^{−a} q^{P_k+1}/(1−q), with q = 1/(2i+1).
  - Convolve these piecewise polynomials with ρ̃ by Lemma C1, using one more pair of Kronecker products.
  - The total error is uniform in t: E_kern = h^{1−a} ‖ρ̃‖ Σ_i (i+½)^{−a} q_i^{P_k+1}/(1−q_i).
- **Kernel cell 0, exactly.** Its contribution is the U-part from ρ̃_m plus the V-part from ρ̃_{m−1}:
  - U-part: h^{1−γ} Σ_b c_{m,b} B(b+1,1−γ) θ^{b+1−γ}.
  - V-part: h^{1−γ} Σ_b č_{m−1,b} [ Poly_b(θ) − (−1)^b B(b+1,1−γ) θ^{b+1−γ} ], where
    Poly_b(θ) = Σ_e binom(b,e)(−θ)^{b−e}/(e+1−γ).
  - *Proof.* With w = σ−θ and y = θ+w, the V-part integral is ∫_θ^1 y^{−γ}(y−θ)^b dy. Expand (y−θ)^b. The
    identity Σ_e binom(b,e)(−1)^{b−e}/(e+1−γ) = ∫_0^1 y^{−γ}(y−1)^b dy = (−1)^b B(b+1,1−γ) finishes it. ∎
- **The singular part.** The two θ^{b+1−γ} pieces combine to
  S_m(θ) = h^{1−γ} θ^{1−γ} Σ_b B(b+1,1−γ) Δ_{m,b} θ^b, with Δ_{m,b} = c_{m,b} − (−1)^b č_{m−1,b}.
  - Δ_{m,b} is the jump of the b-th Taylor coefficient of ρ̃ at mh.
  - |S_m| ≤ σ_m := h^{1−a} Σ_b |B(b+1,1−γ) Δ_{m,b}| on the cell.
- **The term t^{−γ}.** Write h^{−γ}(m+½+x)^{−γ} as its binomial series in x = θ−½, truncated at K_m terms. The
  tail is ≤ h^{−a}(m+½)^{−a} q^{K_m+1}/(1−q).

Collecting terms, on cell m we have p̃ = A_m(x) + S_m + (error), where A_m is an explicit polynomial in x. The
explicit error is the sum of
- the series tails;
- E_kern;
- the dropped top coefficients of A_m (dropped when their contribution is below 1e-19·|A_0|).

The representation was checked against direct mpmath quadrature; the differences were ≤ 2e-15.

## 6. Lower bound for the denominator

**Inequality.** For X ≥ 0 and u > 0,

  sqrt(X) ≥ (3/2) X u − (1/2) X² u³.

*Proof.* Put s = u·sqrt(X). Then sqrt(X) − RHS = s(s−1)²(s+2)/(2u) ≥ 0. ∎

The loss is second order in s − 1, so u should be close to 1/|A|.

**Regular cells.** Here σ_m·∫_{I_m} w is below a budget of 10^{−12}/H. Then
|p̃| ≥ |A_m| − σ_m − (errors). Write w(x) = h/(1−(m+½)h−hx).
- Take X = A_m·Ā_m. This is a polynomial, and it equals |A_m|² for real x.
- Take u as a float least-squares fit of 1/|A_m| at 24 Chebyshev points, of degree 8 / 6 / 3 for
  m < 16 / < 256 / otherwise, with exact dyadic coefficients.
- Certify u > 0 on [−½,½] by u_0 − Σ_{k≥1}|u_k| 2^{−k} > 0.
- w is expanded as a geometric series in x with a rigorous tail, and the series length adapts so the tail is
  < 1e-22.

Then ∫_{I_m}|p̃|/(1−t) dt ≥ ∫_{−½}^{½} [(3/2)Xu − (1/2)X²u³] w dx − sup|f|·(tail of w) − (errors + σ_m)·sup w.

**Knot cells.** These are cells where the derivative jumps of ρ̃ are significant, i.e. σ_m·∫w is above the
budget. Typically they are the cells starting at a knot.
- Keep S_m exactly. On each geometric piece θ ∈ [2^{−j−1}, 2^{−j}] (j < 40), θ^{1−γ} = θ_c^{1−γ}(1+ℓx'/θ_c)^{1−γ}
  with |ℓx'/θ_c| ≤ 1/3. Its binomial series has the tail bound |θ_c^{1−γ}| Σ_{k>K}(k+1)3^{−k}.
- Apply the same sqrt inequality on each piece.
- The piece [0, 2^{−40}] is dropped; this is valid for a lower bound, since the integrand is ≥ 0.

**Cell 0.** Here p̃ = h^{−γ}θ^{−γ} q_0(θ), where q_0 = 1 + h Σ_b B(b+1,1−γ)c_{0,b}θ^{b+1} is a polynomial. Apply
the sqrt inequality to |q_0| with the weight θ^{−a}, using the exact moments ∫_0^1 θ^{k−a} = 1/(k+1−a).

**Conclusion.** D̃_low ≤ ∫_0^½ |p̃|/(1−t), and hence

  F ≤ (|p̃(1)| + EN) / (D̃_low − ED),

with the denominator checked positive. Every quantity is an Arb ball, and the program takes upper endpoints for
the numerator and lower endpoints for the denominator.

## 6b. Two-sided enclosure (diagnostic, also used by the corruption tests)

**Upper bound for the denominator.**
- On every regular piece put s = u·sqrt(X). Then sqrt(X) − f = s(s−1)²(s+2)/(2u).
- If η := sup|u²X − 1| < 1 (certified by coefficient sums), then |s−1| ≤ η and s ≤ 1+η. Hence
  sqrt(X) ≤ f + η²(1+η)(3+η)/(2u_min).
- With |p̃| ≤ |A_m| + errors, and the dropped knot-cell piece [0, 2^{−40}] bounded by its length times
  sup(|A_m| + |S_m|) times sup w, this gives D̃_up ≥ ∫_0^½ |p̃|/(1−t).
- Then D ≤ D̃_up + ED and |p(1)| ≥ |p̃(1)| − EN, so

  F ≥ F_lower := (|p̃(1)| − EN)/(D̃_up + ED).

The claim never uses F_lower. It is reported to show the width of the enclosure.

## 6c. Witness refinement (not a proof step)

`--refine k` repeats k times:
1. compute the nodal values R/t at the collocation nodes of the current exact trial, using the same Arb products;
2. solve the float collocation operator for a correction;
3. add the correction to the nodal values.

The final trial coefficients are the midpoints of an Arb interpolation of the summed nodal values at the exact
dyadic nodes. This changes only which trial function is certified.

## 7. What is new and unreviewed

New and unreviewed:
- Lemma C1 and its use for the residual and for p̃;
- the closed forms for kernel cell 0;
- the singular-part bookkeeping;
- the sqrt inequality approach to the denominator;
- the geometric treatment of knot cells.

Validation done (logs in LOG.md, S3):
1. Direct mpmath quadrature of R(t) and p̃(t) at sample points.
2. The exactly solvable constant-phase (Harcos) profile: F_upper − F_exact = +2.6e-12 at N = 256.
3. Agreement with the independent product-trapezoid evaluator (src/evaluator.py) to its extrapolation accuracy
   (~1e-12) for Bakeas's profile.
4. Corruption tests (checkers/corruptions/).

A second, independently written checker has **not** been written yet. CLAUDE.md requires one before any claim.

## 8. How to run

```
pip install python-flint==0.9.0 numpy scipy mpmath
python checkers/rigorous_check.py <profile.json> --N <grid> --P <degree> [--refine 1] [--bits 256]
```
- Profile format: Bakeas's (25 phases, knots j/48) or Pinho's (explicit rational "breaks"). An optional "C" sets a
  claim with T ≤ C < 1.
- Exit code 0 iff PASS. The output JSON has F_upper, F_lower, every error term, the SHA-256 of the certificate and
  the checker, and max RSS.
- Recommended settings (4 vCPU / 15 GB machine):
  - Bakeas: `--N 768 --P 6 --refine 1` (6 s, 0.14 GB).
  - Pinho (knots j²/4608): `--N 36864 --P 8 --refine 1` (5–6 min, 3.2 GB).
  - Knots j²/18432: N = 73728 (s = 4 cells per first knot spacing) with P = 8.
- Rule of thumb: about 8 cells per first-knot spacing at P = 8, or 4 at P = 10.

Tests:
- `checkers/tests/mp_pointwise.py <profile> N P` checks R and p̃ against direct mpmath quadrature.
- `checkers/tests/profiles/harcos_const.json` is the exactly solvable case; `checkers/tests/harcos_exact.py` gives
  its exact value.
- `checkers/corruptions/*.json` must FAIL or be REFUSED, except `*_claim_1e-9_above_F.json`, a resolution test that
  should PASS.
