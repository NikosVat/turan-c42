# Referee report on NOTES.md (C42 ≤ 0.686360751), with checker #2

Session S6. Written from `NOTES.md`, `certificate/` and the "Certificate format" section of `README.md` only.
I did not open `src/`, `checkers/rigorous_check.py` (or any other file of checker #1), `LOG.md` or other branches.
`checkers/corruptions/**/*.json` were read only as input data (they are certificates in the same format).

## 0. Results

**Verdict of checker #2 on `certificate/c42_profile_0.686360751.json` (SHA-256 `d015a829…5652`): PASS** at
three settings. The hypotheses of Theorem A hold: a = Re γ = 0.10165403756107818… > 0,
E = max|ψ_j − ψ_0| = 0.8708652 < π, sup_{(0,½]}|d| = 2T sin(E/2) ≤ 0.5790176 < T, so κ̂ = T < 1; T ≤ C < 1.

| N | P | certified F_upper | C − F_upper | sup\|R/t\| | EN | ED | D̃_low | time | peak RSS |
|---|---|---|---|---|---|---|---|---|---|
| **36864** | **12** | **0.6863607494322799** | 1.568·10⁻⁹ | 1.66·10⁻¹⁰ | 1.85·10⁻¹² | 7.80·10⁻¹³ | 0.3207780226746401 | 253 s | 1.5 GB |
| 73728 | 12 | 0.6863607494248632 | 1.575·10⁻⁹ | 7.29·10⁻¹⁴ | 4.17·10⁻¹⁵ | 1.42·10⁻¹⁵ | 0.3207780226746401 | 464 s | 3.0 GB |
| 36864 | 10 | 0.6863607500743508 | 9.26·10⁻¹⁰ | 1.11·10⁻⁸ | 1.62·10⁻¹⁰ | 6.81·10⁻¹¹ | 0.3207780226746400 | 171 s | 1.4 GB |

(EN, ED: the smaller of the integrated and sup forms, which is the integrated form in every run. Times measured
with two runs sharing 4 cores; python-flint 0.9.0, Arb precision 448 bits. Outputs in
`checkers/independent_out/cert_N*_P*.json`, logs in `*.err`.)

- **Stability.** Doubling N moves F_upper down by 7.4·10⁻¹², exactly the shrinkage of EN/D̃ + F·ED/D̃; D̃_low
  and p̃(1) agree to 4·10⁻¹⁷ between the two grids. The float Gauss–Legendre estimate of ∫|p̃|w agrees with
  the certified D̃_low to 4·10⁻¹⁶, i.e. the linear minorant loses nothing measurable.
- **Agreement with checker #1.** NOTES §8.3 reports checker #1's tightest enclosure
  F ∈ [0.6863607494207, 0.6863607494290] (N = 73728, P = 10). Checker #2's N = 73728 upper bound
  0.6863607494249 lies *inside* that interval, and at the same setting as checker #1's primary run (N = 36864,
  P = 10) the two upper bounds are 0.6863607500744 (#2) and 0.6863607500725 (#1). Against NOTES §8.2:
  p̃(1) agrees to 4·10⁻¹⁷ in each component (different witnesses), |p̃(1)| to 6·10⁻¹⁵, and D̃_low is
  3.2·10⁻¹² larger in checker #2 (0.3207780226746401 vs 0.3207780226714513), which matches the 3.2·10⁻¹²
  "bookkeeping error" checker #1 reports for its inequality-S minorant. No upper bound of either checker lies
  below a lower bound of the other.
- The worst residual piece is the first cell after the first knot (t ∈ [2h, 3h] at N = 36864, [4h, 5h] at
  N = 73728), the same place checker #1 reports (m = 2).

**Corruption tests:** see §4 (all verdicts as expected).

**Referee conclusion.** I found no mathematical error in NOTES.md. The proof of Theorem A (Bakeas), the
a posteriori bounds (Bakeas, Pinho) and the new certifier method (§§5–7) are correct as written, up to the
clarifications listed in §2. The bound C42 ≤ 0.686360751 is now confirmed by two independently written rigorous
checkers. It remains **unrefereed by any human**; the two checkers share the mathematics of NOTES §§3–4 (which
I reviewed but which is not peer-reviewed), so a gap there would affect both.


---

## 1. Summary of the claim

NOTES.md claims limsup R_n ≤ C = 0.686360751, via
1. Lemma 1 (normalization) and Lemma 2 (Harcos's linearization at fixed n);
2. Theorem A (Bakeas): with s_k = T e^{iφ(k/n)} for k ≤ ⌊n/2⌋, F_n → F, an explicit functional of the bounded
   solution ρ of t ρ = d + d∗ρ, provided κ = sup|d| < 1 and a = Re γ ≥ 0;
3. a posteriori bounds (Lemma 3 / Lemma 3') turning any trial ρ̃ into rigorous bounds on F;
4. a new certifier (§§5–7) and a 33-piece profile with C = T.

My overall assessment: **§§1–4 are correct as written** (I re-derived every step; minor presentation points
below). §§5–7 describe a valid method; I re-derived Lemma C1, the kernel-cell-0 closed forms, the singular-part
bookkeeping, inequality S and the error budget, and found no error. Checker #2 (`checkers/independent_check.py`)
implements the same *framework* (Theorem A + §4 + piecewise-polynomial ρ̃ on the knot-aligned grid + Lemma C1)
but uses different rigorous sub-methods for every step where a choice exists (§3 below), so that a bug in one
implementation is unlikely to be replicated in the other.

## 2. Mathematical review, section by section

### §1 Lemma 1 — correct.
The two normalizations (max|z_j| = 1 versus z_1 = 1) give the same R_n. Both directions check. Note the problem
statement uses "min", NOTES uses "inf"; nothing depends on attainment.

### §2 Lemma 2 — correct.
- Step 1: with u_k = conj(P_{n−k})/|P_{n−k}|, Σ_{k>K} (σ_k/k) P_{n−k} = λ Σ |P_{n−k}|/k = λ D_n = P_n. ✓
- Step 4 needs y² ≡ 0 mod x^{n+1} for y = Σ_{k>K} σ_k x^k/k, i.e. 2K+2 > n, which holds for K ≥ ⌊n/2⌋. ✓
- D_n ≥ 1/n because P_0 = 1. ✓

### §3 Theorem A — correct.
I checked each step:
- Step 1: Σ P_m x^m = exp(Σ c_k x^k/k) = (1−x)^{γ−1} D(x), with d_k = d(k/n) for *all* 1 ≤ k ≤ n, including
  k = n/2 (d(½) = γ − g(½)) and k > ⌊n/2⌋ (d = γ). ✓ The recursion m δ_m = Σ d_k δ_{m−k} is xD' = (Σ d_k x^k) D. ✓
- Step 2: 𝒦 is a contraction on bounded Borel functions with constant κ; d/t is bounded by M = max(TΛ, 2T)
  (on (½,1], T ≤ 2Tu). ✓ Regularity argument: d/t extends continuously to 0; |𝒦ρ(t)| ≤ MBt/2. ✓
- Step 3: the uniform bound |ρ_n(m)| ≤ M + κB = B is right; the error recursion and the bound on τ_m are right
  (at most two discontinuous cells, only when t_m > ½, so dividing by t_m costs a factor ≤ 2). ✓
- Step 4: the decomposition of Π_n(j) into the l = 0 ("middle") term, the j-term and the convolution sum is
  right; (B2) uses that v^{−a} is decreasing (this is where a ≥ 0 is used) and t^{1−a} ≤ t^{−a},
  n^{a−1} ≤ t^{−a} for t ∈ [1/n, 1]. ✓ (B1): items 1–6 check; the jump of ρ(t − ·) is handled in item 6. ✓
- Step 5: D_n = Σ_{j=0}^{L} |P_j|/(n−j), L = ⌈n/2⌉ − 1; the j = 0 term → 0; domination h_n ≤ 2C t^{−a} uses
  ⌈nt⌉ ≤ L < n/2. ✓

Presentation points (no effect on validity):
- §3 Theorem A says "Assume κ < 1 and a ≥ 0"; README says the checker must prove "0 < a < 1". Both checkers prove
  the stronger 0 < a. Fine, but the two documents should state the same hypothesis.
- The proof never needs κ to be *equal* to sup|d|: every use is |d| ≤ κ < 1. Saying this explicitly would make it
  obvious that a certified *upper bound* κ̂ (as the programs use) is enough, including in the exponents of
  Lemma 3'. (With ball arithmetic the result is still valid as long as the ball contains one admissible κ̂;
  checker #2 avoids the question by using an exact rational κ̂.)

### §4 A posteriori bounds — correct.
- (9) from te = d∗e + R. ✓
- Lemma 3' : E' ≤ r + κE/t a.e., (t^{−κ}E)' ≤ t^{−κ} r, t^{−κ}E → 0 because e is bounded and κ < 1. ✓
- EN (integrated) uses A(v) ≤ A(1) and ∫_0^1 v^{κ−1}(1−v)^{−a} dv = B(κ, 1−a). ✓
- ED uses 1/(1−t) ≤ 2 on [0,½], Tonelli, and ∫_0^{1/2} v^{κ−1}(½−v)^{1−a} dv = (½)^{κ+1−a} B(κ, 2−a). ✓
- (10) ✓ (|p(1)| ≤ |p̃(1)| + EN and ∫|p|w ≥ ∫|p̃|w − ED).
- Remark: r only needs to majorize |R/t| almost everywhere and may be piecewise constant on any partition; both
  checkers exploit this (checker #2 uses 16 sub-pieces on each of the first 64 cells, see §3).

### §5 Trial space and residual — correct, two clarifications requested.
- Lemma C1: re-derived independently (split of kernel cell i at σ = θ; the reversed variable in the second part
  is λ − s). ✓ Kronecker stride S ≥ deg f + deg g + 1 avoids aliasing. ✓
- Taylor bound for e^{iy}, y real: |y|^{Q+1}/(Q+1)!. ✓
- **Clarification 1 (cell 0).** The bound r_0 requires R_0(θ)/θ to be a polynomial, i.e. the constant term of
  d_{Q,0} must be *exactly* 0. If γ − g_0 is computed as a ball (radius ≠ 0) it is not exactly 0 and dividing by
  θ is not justified. NOTES says "the constant term of d_{Q,0} is exactly 0"; the program must set it
  symbolically (checker #2 does: its d^ on cell 0 has constant term the integer 0, and it asserts R^_0(0) = 0
  exactly). I could not check checker #1 on this point (by the rules of this session).
- **Clarification 2.** The error of using balls (rather than exact numbers) for the Taylor coefficients of d is
  covered automatically if they are balls; if they are rounded to dyadics (as in checker #2) the rounding error
  must be added to ε_i. NOTES should say which.
- r_m for m ≥ 1 uses t ≥ mh. For m = 1, 2, … this loses up to a factor (m+1)/m; it is valid, just loose. The
  worst cell of checker #1 is m = 2 (§8.2), so sub-pieces would tighten EN.

### §6 Rigorous evaluation — correct; one ambiguity.
- 6.1: |binom(−γ,k)| ≤ 1, |(γ)_l/l!| ≤ 1, |binom(1−γ,k)| ≤ k+1. ✓
- 6.2: the n' = 2 closed form ✓; **ambiguity:** "truncated after L(n') terms" with tail
  n'^{−a} n'^{−L−1}/(1−1/n') is correct if the terms l = 0..L are kept (L+1 terms). If only L terms
  (l = 0..L−1) are kept the tail is n'^{−a} n'^{−L}/(1−1/n'), a factor n' larger. The text should say which.
  (Checker #2 does not use this series: it evaluates p̃(1) in closed form, §3.)
- 6.3: kernel cell 0 (U-part, V-part, Poly_b, the (−1)^b B θ^{b+1−γ} identity) re-derived ✓; singular part S_m and
  Δ_{m,b} as the jump of the Taylor coefficients ✓; E_kern ✓; t^{−γ} tail ✓.
- 6.4: inequality S re-derived: √X − RHS = s(s−1)²(s+2)/(2u) with s = u√X. ✓ The treatment of regular cells
  (singular part as error), knot cells (geometric pieces, |ℓx'/θ_c| ≤ 1/3, tail Σ_{k>K}(k+1)3^{−k}), the dropped
  piece [0, 2^{−40}], cell 0, and the weight tails are all valid lower-bound steps. ✓
- 6.5 diagnostic F_lower ✓ (|s² − 1| ≤ η ⇒ |s − 1| ≤ η).

### §7 Decision rule — correct.
- Minor wording: "|φ − ψ_0| is piecewise linear" — it is piecewise linear only after adding the zero crossings as
  breakpoints; the conclusion (maximum at a knot) holds anyway because |·| of a linear function is convex.
- The rule "PASS iff upper endpoint of (|p̃(1)| + EN)/(D̃_low − ED) < C", together with T ≤ C and Theorem A, gives
  limsup R_n ≤ max(T, F) ≤ C. ✓

### §8 Certificate, stability, corruption tests, finite n
- The arithmetic in §8 is consistent: 0.6863607950 − 0.686360751 = 4.40·10⁻⁸; a = T cos ψ_0 = 0.1016540375…;
  β_0 = (ψ_1 − ψ_0)·18432 = 614.89; E = 0.87087 < π; 2T sin(E/2) = 0.57902 < T, so κ = T.
  Checker #2 reproduces a, E, sup_{(0,½]}|d| and κ independently.
- The primary run of checker #1 has width 1.3·10⁻⁹ against a margin 9.3·10⁻¹⁰; the claim rests only on the
  upper end, so this is fine, but the README/NOTES should present the tightest PASS (N = 73728, P = 10, or
  N = 36864, P = 12) as the headline, since those show the bound is not borderline. Checker #2 agrees (§3).
- Finite-n (§8.4): honest as written. It is a consistency check only; at n = 4·10⁶ the finite-n error (~6·10⁻⁶)
  is four orders of magnitude larger than the margin.
- The FAIL rows of §8.3 are reported with F_lower > C, which makes them certifiably false claims — good practice.

### Other remarks
- The margin C − F ≈ 7·10⁻¹⁰ is tiny in absolute terms but large compared with the certified enclosure widths
  (8·10⁻¹² for checker #1 at N = 73728; see §3 for checker #2). The bound is therefore not "at the edge of
  numerical resolution", provided both checkers are correct.
- Theorem A, Lemma 3' and the profile are the only ingredients; nothing in the claim depends on the search code
  or on the finite-n experiments.
- The bound is a limsup bound; nothing is claimed for finite n.

## 3. What checker #2 does (`checkers/independent_check.py`)

Same mathematics as NOTES §§3–4 (Theorem A, Lemmas 3/3', bound (10)); same trial space (piecewise polynomials of
degree P on the grid h = 1/N with N even and N·u_j ∈ ℤ) and Lemma C1. Every *implementation* choice differs
from NOTES §§5–7 where a choice exists:

| step | NOTES §§5–7 (checker #1) | checker #2 |
|---|---|---|
| witness ρ̃ | float collocation, GMRES | own float collocation (Chebyshev nodes, LU per cell, GMRES), then 2 passes of mixed-precision iterative refinement driven by the *exact* residual; coefficients dyadic (2⁻⁵⁰) |
| d on a cell | Taylor degree Q, Arb balls | Taylor degree Q (remainder T\|βh\|^{Q+1}/(Q+1)! ≤ 10⁻³⁴), coefficients rounded to 2⁻¹²⁸ with an explicit rounding bound; constant term on cell 0 the integer 0 |
| convolutions (Lemma C1) | `acb_poly` products (outward rounded) | **exact integer** Kronecker products (`fmpz_poly`, 3-multiplication complex product); all rational post-processing (θ^{c+1}/(c+1)!, (1−θ)^{c+1}, Chebyshev conversion) in exact integers (`fmpz_mat`) |
| sup\|R/t\| per cell | Σ\|χ_n\| in balls, t ≥ mh | Σ\|χ_n\| in exact integers (integer square roots rounded up); first 64 cells split into 16 sub-pieces each, with t ≥ h(m + s/16) |
| EN, ED | min(sup, integrated) | same formulas, re-derived; κ̂ an exact rational (= T here) |
| p̃(1) | series in 1/n' (n' ≥ 3) + closed forms for n' = 1, 2 | **closed form, no truncation**: p̃(1) = 1 + h^{1−γ} Σ_k Σ_e ρ̌_{k,e} I_e(N−1−k), with I_k(m) = ∫_0^1 (m+s)^{−γ} s^k ds from (k+1−γ) I_k = (m+1)^{1−γ} − [k=0] m^{1−γ} − k m I_{k−1} (forward for m < 64; backward from a rigorous starting ball \|I_k(m)\| ≤ m^{−a}/(k+1) for m ≥ 64) |
| p̃ on cell m ≥ 1 | t^{−γ} by binomial series; kernel cell 0 exact; kernel cells ≥ 1 Taylor degree 20 | t^{−γ} kept exact (its moments are I_k(m)); kernel cell 0 exact (same closed forms, re-derived); kernel cells ≥ 1 Taylor degree 30 about the centre (tail ≤ 10⁻¹⁶), exact integer Kronecker products |
| denominator lower bound | inequality S (√X ≥ (3/2)Xu − (1/2)X²u³), regular/knot cells, geometric splitting of knot cells, dropped piece [0, 2⁻⁴⁰] | **linear minorant** \|p\| ≥ Re(p z̄)(3 − \|z\|²)/2, valid for \|z\| ≤ √3 (because 0 ≤ x(3−x²)/2 ≤ 1 on [0, √3]); z a dyadic polynomial fit of p̃/\|p̃\| per cell, \|z\| ≤ √3 certified by Σ\|Chebyshev coefficients\|. The integrand is then linear in p̃, so each cell integral is an exact combination of moments: polynomial moments, singular moments ∫θ^{1−γ+b+k} = 1/(b+k+2−γ), and I_k(m). No cell classification, no dropped piece; the loss is second order in the fit error |
| weight 1/(1−t) | geometric series with tail | same idea, J = 6 terms, tail bounded by sup\|p̃\| ≤ (mh)^{−a} + ‖ρ̃‖/(1−a) |

Cell 0 of the denominator: p̃ = h^{−γ} θ^{−γ} q_0(θ), so \|p̃\| = h^{−a} θ^{−a} \|q_0\| ≥ h^{−a} θ^{−a} Re(q_0 z̄)(3−\|z\|²)/2,
integrated with the moments 1/(j+k+1−a).

Validation of checker #2 itself (`checkers/independent_selftest.py`, toy 5-piece profile, N = 64, P = 10):
- R(t) computed by direct mpmath quadrature (with the exact d and the certified ρ̃) satisfies \|R/t\| ≤ r at
  random points in cells 0, 1, 2, 3, N/4, N/2 ± 1, N − 1;
- p̃(t) by direct quadrature agrees with the cell representation to ≤ 2·10⁻¹⁶;
- p̃(1) by direct quadrature agrees with the closed form to 2·10⁻¹⁷;
- a float Gauss–Legendre estimate of ∫\|p̃\|w exceeds the certified D̃_low (as it must).
- Constant-phase (Harcos) profile T = 0.69, ψ ≡ −1.3, N = 256: certified F ≤ 0.79514335162406, exact
  F = 0.79514335162396 (mpmath, from the closed formula in NOTES §3); D̃_low is 1.1·10⁻¹³ below the exact
  denominator.

What checker #2 does **not** share with checker #1: no code, no witness, no data other than the certificate file.
It was written from NOTES.md and the README format section; it agrees with NOTES §§3–4 because those are the
mathematics being checked.


## 4. Corruption tests (checker #2)

`python checkers/independent_corruptions.py --jobs 3` runs checker #2 on
- every file in `checkers/corruptions/` and `checkers/corruptions/claim/` as is;
- adapted copies (certificate format, C := T) of the old-format Bakeas files (no `C`, knots given as the text
  "j/48, j = 0..24") and of the Pinho files without `C`;
- new corruptions of the certificate made by the script.

Meaning of the verdicts: PASS = F < C proved; FAIL = the certified upper bound is not below C (checker #2 does
**not** compute a rigorous lower bound for F, so a FAIL means "not proved", not "disproved"; checker #1's
F_lower > C in NOTES §8.3 is what shows those claims are false); REFUSED = invalid input or a hypothesis of
Theorem A not certified.

__CORRUPTIONS__

## 5. How to run checker #2

```bash
pip install python-flint==0.9.0 numpy scipy mpmath
```

```bash
python checkers/independent_check.py certificate/c42_profile_0.686360751.json --N 36864 --P 12 --out result.json
```

Prints the JSON result and a last line `PASS: F <= … < C = …`; exit code 0 = PASS, 1 = FAIL, 2 = REFUSED.
About 4 minutes and 1.5 GB (N = 36864, P = 12); `--N 73728` about 8 minutes and 3 GB.

```bash
python checkers/independent_selftest.py
```

```bash
python checkers/independent_corruptions.py --jobs 3
```

The self-test takes about 15 s; the corruption run about 25 minutes with 3 jobs.

## 6. Recommendations

1. Update README.md and NOTES.md §9: the second checker now exists and agrees (this report, §0).
2. Address the two clarifications in §2 (§5.3 exact zero constant term on cell 0; §6.2 "after L terms").
3. Present the tightest PASS runs (checker #1 N = 73728/P = 10 or N = 36864/P = 12; checker #2 N = 73728/P = 12)
   as the headline numbers, rather than the loosest PASS.
4. State the hypothesis of Theorem A uniformly (a ≥ 0 in NOTES, 0 < a < 1 in README).
5. Before a public claim, a human should read Theorem A (Bakeas) and Lemma C1/§6; both checkers rest on them.
