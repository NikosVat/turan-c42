# C42 ≤ 0.686360751: proof and computer verification

**Main result.** For Turán's pure power sum problem,

  limsup_{n→∞} R_n ≤ 0.686360751,   R_n = inf_{z_1 = 1, z_2..z_n ∈ ℂ} max_{1≤k≤n} |Σ_j z_j^k|.

**Status: unrefereed. Written with AI assistance (see the end of the file).**
- Sections 1–3 (Lemmas 1–2, Theorem A) are Bakeas's (turan-c42-bound v1.1.0, commit dc9c491, MIT). They are
  reproduced here with small notational changes, so that this file is self-contained.
  - The extension from the knots j/48 to arbitrary rational knots is Pinho's (c42-certificate @009120a,
    `volterra-quadratic-knots/VOLTERRA_VERIFIER_MATH.md`). It changes nothing in the proof; see the remark in §3.
- Section 4: Lemma 3 is Bakeas's; the integrated (Gronwall) form of it is Pinho's.
- Sections 5–7 (the certifier) are **NEW** and **unreviewed**. Every new step is marked **[NEW]**.
- Section 8 gives the certificate and exactly what the programs check.

Notation:
- s_k(z) = Σ_j z_j^k.
- For t > 0 and w ∈ ℂ, t^w = e^{w log t}.
- B(x, y) = Γ(x)Γ(y)/Γ(x+y) is the Beta function.
- (y)_l = y(y+1)…(y+l−1) is the rising factorial, and binom(y, k) = y(y−1)…(y−k+1)/k! for complex y.

---

## 1. Normalization (Bakeas, Lemma 1)

**Lemma 1.** Let R'_n be the infimum of max_{k≤n}|s_k(z)| over all z ∈ ℂ^n with max_j |z_j| = 1. Then R'_n = R_n.
In particular C42 = limsup R_n in either normalization.

*Proof.*
- (R_n ≤ R'_n.) Suppose max_j |z_j| = |z_{j0}| = 1, and put w_j = z_j/z_{j0}. Then w_{j0} = 1, and
  |s_k(w)| = |s_k(z)|.
- (R'_n ≤ R_n.) Suppose z_1 = 1. Let λ = max_j |z_j| = |z_{j0}| ≥ 1 and put w_j = z_j/z_{j0}. Then
  max_j |w_j| = 1 and |s_k(w)| = λ^{−k}|s_k(z)| ≤ |s_k(z)|. ∎

## 2. The construction for fixed n (Harcos's linearization; Bakeas, Lemma 2)

For s_1,…,s_K ∈ ℂ put

  A(x) = exp(−Σ_{k=1}^K s_k x^k/k) = Σ_{m≥0} α_m x^m,   P_m = α_0 + … + α_m,   so Σ_m P_m x^m = A(x)/(1−x).

**Lemma 2.** Let n ≥ 2 and ⌊n/2⌋ ≤ K < n, and let s_1,…,s_K ∈ ℂ. Put D_n = Σ_{k=K+1}^{n} |P_{n−k}|/k. Note
that D_n ≥ 1/n > 0, because P_0 = 1. Then there are z_1 = 1, z_2,…,z_n ∈ ℂ such that
- s_k(z) = s_k for 1 ≤ k ≤ K;
- |s_k(z)| = |P_n|/D_n for K < k ≤ n.

Hence R_n ≤ max(max_{k≤K}|s_k|, |P_n|/D_n).

*Proof.*
1. **Choose the remaining power sums.** For K < k ≤ n put σ_k = λu_k, where
   - λ = P_n/D_n;
   - u_k = conj(P_{n−k})/|P_{n−k}| if P_{n−k} ≠ 0, and u_k = 1 otherwise.

   Put σ_k = s_k for k ≤ K. Then Σ_{k=K+1}^n (σ_k/k) P_{n−k} = λD_n = P_n.
2. **Build the tuple.** Let W(x) = Σ_{m=0}^n w_m x^m be the truncation to degree ≤ n of exp(−Σ_{k=1}^n σ_k x^k/k).
   Since w_0 = 1, the polynomial x^n W(1/x) is monic of degree n. Let z_1,…,z_n be its roots, with
   multiplicity. Then W(x) = Π_j (1 − z_j x).
3. **The power sums are the σ_k.** Take logarithms of power series with constant term 1:
   −Σ_k s_k(z) x^k/k ≡ −Σ_{k≤n} σ_k x^k/k (mod x^{n+1}). So s_k(z) = σ_k for all k ≤ n.
4. **W(1) = 0.**
   - Two free indices k, k' ≥ K+1 have k + k' ≥ 2K+2 > n. Hence, mod x^{n+1},
     exp(−Σ_{k≤n} σ_k x^k/k) ≡ A(x)(1 − Σ_{k=K+1}^n σ_k x^k/k).
   - So w_m = α_m − Σ_{K<k≤m} (σ_k/k) α_{m−k}.
   - Summing over m ≤ n gives W(1) = P_n − Σ_{k=K+1}^n (σ_k/k) P_{n−k} = 0.

   So 1 is one of the z_j. Renumber so that z_1 = 1. ∎

For a constant first block this is the argument in G. Harcos's note "On a result of András Biró".

## 3. The profile and the limit theorem (Bakeas, Theorem A)

**The profile.**
- Fix T ∈ (0,1), rational knots 0 = u_0 < u_1 < … < u_K = ½ (here K denotes the number of pieces), and real
  phases ψ_0,…,ψ_K.
- Let φ: [0,½] → ℝ be the piecewise linear function with φ(u_j) = ψ_j. Its slopes are
  β_j = (ψ_{j+1} − ψ_j)/(u_{j+1} − u_j). Put Λ = max_j |β_j|.
- Put g(u) = T e^{iφ(u)}, γ = g(0) and a = Re γ.
- Define d(u) = γ − g(u) for 0 < u ≤ ½, d(u) = γ for ½ < u ≤ 1, and d(0) = 0. Put κ = sup_{0<u≤1}|d(u)|.

**The construction.** For n ≥ 2 apply Lemma 2 with K_n = ⌊n/2⌋ and s_k = g(k/n), 1 ≤ k ≤ K_n (note that
k/n ≤ ½). Since |s_k| = T,

  R_n ≤ max(T, F_n),  F_n = |P_n|/D_n.   (1)

**Theorem A.** Assume κ < 1 and a ≥ 0. (Then automatically a ≤ |γ| = T ≤ κ < 1, because |d| = T on (½,1].)
1. The equation

     t ρ(t) = d(t) + ∫_0^t d(u) ρ(t−u) du   (0 < t ≤ 1)   (2)

   has exactly one bounded solution ρ on (0,1].
2. Put
     p(t) = t^{−γ} + ∫_0^t (t−v)^{−γ} ρ(v) dv,   F = |p(1)| / ∫_0^{1/2} |p(t)| dt/(1−t).
   Then the denominator of F is positive and finite, and F_n → F as n → ∞.
3. Hence limsup_n R_n ≤ max(T, F).

**Remark (knots).** Bakeas states Theorem A for the knots j/48. The proof below uses only two facts about φ:
- φ is continuous and piecewise linear with finitely many pieces, so |d(u)| ≤ TΛu on (0,½];
- d is uniformly continuous on (0,½].

So the proof holds verbatim for any finite set of knots in [0,½]. This was already observed by Pinho (PR #305).
The knots need not be rational for the theorem; they are rational only so that the certifier can align its grid.

**Proof.** Throughout, M = max(TΛ, 2T). For 0 < u ≤ ½, |d(u)| = T|1 − e^{i(φ(u)−φ(0))}| ≤ TΛu. For u > ½,
|d(u)| = T ≤ 2Tu. Hence

  |d(u)| ≤ Mu on (0,1],  |d| ≤ κ.   (3)

### Step 1: exact discrete identities

Put c_k = 1 − s_k for k ≤ K_n and c_k = 1 for k > K_n. Since −log(1−x) = Σ_k x^k/k,

  Σ_m P_m x^m = exp(Σ_{k≥1} c_k x^k/k).

Write c_k = (1−γ) + d_k.
- For k ≤ K_n, d_k = γ − g(k/n) = d(k/n).
- For k > K_n, d_k = γ = d(k/n), because k > K_n implies k/n > ½.

So
  d_k = d(k/n) for every k ≥ 1,   (4)
and Σ_m P_m x^m = (1−x)^{γ−1} D(x), where
- (1−x)^{γ−1} = Σ_j a_j x^j with a_j = Γ(j+1−γ)/(Γ(1−γ) j!);
- D(x) = exp(Σ_k d_k x^k/k) = Σ_m δ_m x^m, with δ_0 = 1 and m δ_m = Σ_{k=1}^m d_k δ_{m−k} (from xD' = (Σ_k d_k x^k) D).

Therefore
  P_j = Σ_{i=0}^{j} a_{j−i} δ_i.   (5)

### Step 2: the continuous problem

Let X be the space of bounded Borel-measurable functions on (0,1] with the sup norm (a Banach space). Put
(𝒦f)(t) = (1/t) ∫_0^t d(u) f(t−u) du. Then:
- ‖𝒦f‖ ≤ κ‖f‖;
- 𝒦f is continuous on (0,1], since d*f is the convolution of two bounded functions on [0,1];
- equation (2) reads ρ = d(t)/t + 𝒦ρ, and d(t)/t is bounded by (3).

Since κ < 1, there is a unique bounded solution, given by the Neumann series, and ‖ρ‖ ≤ B := M/(1−κ).

*Regularity of ρ.*
- d(t)/t is uniformly continuous on (0,½]: it extends continuously to 0, with value −iTβ_0e^{iψ_0}. It is also
  uniformly continuous on (½,1].
- |𝒦ρ(t)| ≤ MBt/2, and 𝒦ρ = (d*ρ)/t is continuous.

Hence ρ is uniformly continuous on (0,½] and on (½,1]. Its only jump is at ½, and it is left-continuous there.

*The function p.*
- t ↦ ∫_0^t u^{−γ}ρ(t−u) du is continuous on [0,1], since it is the convolution of u^{−γ} ∈ L¹ with a bounded
  function. So p is continuous on (0,1].
- |p(t)| ≥ t^{−a} − Bt^{1−a}/(1−a) > 0 for small t, so the denominator of F is positive.
- The denominator is finite, because |p(t)| ≤ t^{−a} + B/(1−a) and a < 1.

### Step 3: the discrete solution converges to ρ

Let ρ_n(m) = nδ_m and t_m = m/n. By (4), the recursion for δ_m reads

  t_m ρ_n(m) = d(t_m) + (1/n) Σ_{k=1}^{m−1} d(t_k) ρ_n(m−k)   (1 ≤ m ≤ n).   (6)

*Uniform bound.* We have |d| ≤ κ, |d(t_m)| ≤ Mt_m and (1/n)Σ_{k<m}|d(t_k)| ≤ κt_m. So induction gives
|ρ_n(m)| ≤ B.

*Error recursion.* Let e_m = ρ_n(m) − ρ(t_m). Subtracting (2) at t = t_m from (6),

  t_m e_m = (1/n) Σ_{k=1}^{m−1} d(t_k) e_{m−k} + τ_m,   τ_m = (1/n)Σ_{k=1}^{m−1} f_m(t_k) − ∫_0^{t_m} f_m(u) du,

where f_m(u) = d(u)ρ(t_m − u). Hence |e_m| ≤ κ max_{l<m}|e_l| + |τ_m|/t_m, and so
max_{m≤n}|e_m| ≤ (sup_m |τ_m|/t_m)/(1−κ).

*Bounding τ_m.* Write
  τ_m = Σ_{k=1}^{m−1} ∫_{t_{k−1}}^{t_k} (f_m(t_k) − f_m(u)) du − ∫_{t_{m−1}}^{t_m} f_m.
- The last integral has modulus ≤ MBt_m/n, by (3).
- f_m is the product of d, which is continuous on (0,½] and on (½,1], and of u ↦ ρ(t_m − u), which has at most
  one jump. So at most two cells contain a discontinuity, and this happens only when t_m > ½. On each of them
  the oscillation of f_m is ≤ 2κB.
- On every other cell the oscillation is ≤ Bω_d(1/n) + κω_ρ(1/n). Here ω_d and ω_ρ are the moduli of uniform
  continuity of d and ρ on their two pieces.

Dividing by t_m, and using t_m > ½ when there are discontinuous cells,

  sup_m |τ_m|/t_m ≤ Bω_d(1/n) + κω_ρ(1/n) + (MB + 8κB)/n → 0.

Therefore max_{1≤m≤n} |ρ_n(m) − ρ(m/n)| → 0.   (7)

### Step 4: asymptotics of P_j

For l ≥ 1 put η_l = Γ(1−γ) a_l l^γ = Γ(l+1−γ) l^γ/Γ(l+1). The classical asymptotic
Γ(l+α)/Γ(l+β) ~ l^{α−β} gives η_l → 1. Put C_η = sup_l |η_l| < ∞.

Let Π_n(j) = Γ(1−γ) n^γ P_j. By (5), for 1 ≤ j ≤ n,

  Π_n(j) = η_j (j/n)^{−γ} + (Γ(1−γ)n^γ/n) ρ_n(j) + (1/n) Σ_{l=1}^{j−1} η_l (l/n)^{−γ} ρ_n(j−l).

The middle term is the term l = 0. Its modulus is ≤ |Γ(1−γ)| B n^{a−1} → 0.

**(B2) Domination.** v^{−a} is decreasing, so (1/n)Σ_{l=1}^{j−1}(l/n)^{−a} ≤ ∫_0^{j/n} v^{−a} dv. Hence, for
1 ≤ j ≤ n and t = j/n,

  |Π_n(j)| ≤ C t^{−a},   C = C_η + |Γ(1−γ)|B + C_η B/(1−a).

**(B1) Convergence.** For every ε > 0, sup_{εn≤j≤n} |Π_n(j) − p(j/n)| → 0.

*Proof.* Let t = j/n ∈ [ε,1] and fix δ ∈ (0, ε/2). Split

  p(t) = t^{−γ} + I_{≤δ}(t) + I_{>δ}(t),   I_{≤δ} = ∫_0^δ u^{−γ}ρ(t−u) du,   I_{>δ} = ∫_δ^t u^{−γ}ρ(t−u) du.

Split the last sum in Π_n(j) the same way: Σ_{≤δ} runs over 1 ≤ l ≤ δn, and Σ_{>δ} over δn < l ≤ j−1.
1. *First term.* |η_j(j/n)^{−γ} − t^{−γ}| = |η_j − 1| t^{−a} ≤ |η_j − 1| ε^{−a} → 0 uniformly, since j ≥ εn.
2. *Middle term.* It tends to 0, as shown above.
3. *Near part.* |Σ_{≤δ}| ≤ C_ηBδ^{1−a}/(1−a), and |I_{≤δ}| ≤ Bδ^{1−a}/(1−a) ≤ C_ηBδ^{1−a}/(1−a). Here C_η ≥ 1,
   since η_l → 1.
4. *Replacing ρ_n by ρ.* Let Σ' be Σ_{>δ} with ρ_n(j−l) replaced by ρ((j−l)/n). Then
   |Σ_{>δ} − Σ'| ≤ C_η max_m|e_m| · (1/n)Σ_{l≤n}(l/n)^{−a} ≤ C_η max_m|e_m|/(1−a) → 0, by (7).
5. *Replacing η_l by 1.* Let Σ'' be Σ' with η_l replaced by 1. Then
   |Σ' − Σ''| ≤ sup_{l>δn}|η_l − 1| · B/(1−a) → 0.
6. *Riemann sum.* Σ'' = (1/n)Σ_{δn<l≤j−1}(l/n)^{−γ}ρ(t − l/n) is a right-endpoint Riemann sum for I_{>δ}, with
   cells of width 1/n.
   - Take n ≥ 2/δ. Then every cell lies in [δ/2, 1], where |u^{−γ}| ≤ (δ/2)^{−a}.
   - The two end cells contribute ≤ 2(δ/2)^{−a}B/n.
   - On any other cell, the oscillation of u ↦ u^{−γ}ρ(t−u) is ≤ Bω_1(1/n) + (δ/2)^{−a}ω_ρ(1/n). Here ω_1 is
     the modulus of continuity of u^{−γ} on [δ/2,1].
   - The exception is the at most one cell containing the jump of ρ(t−·), at u = t − ½. There the oscillation
     is ≤ 2(δ/2)^{−a}B, so the contribution is ≤ 2(δ/2)^{−a}B/n.
   - Hence |Σ'' − I_{>δ}| → 0 uniformly in t ∈ [ε,1].

So limsup_n sup_{εn≤j≤n} |Π_n(j) − p(j/n)| ≤ 2C_ηBδ^{1−a}/(1−a) for every small δ > 0. Letting δ → 0 proves
(B1). ∎

### Step 5: conclusion

Since |Γ(1−γ)n^γ| = |Γ(1−γ)| n^a, (1) gives

  F_n = |Π_n(n)| / ( (1/n) Σ_{j=0}^{L} |Π_n(j)|/(1 − j/n) ),   L = n − K_n − 1 = ⌈n/2⌉ − 1.

- **Numerator.** Π_n(n) → p(1), by (B1).
- **Denominator.**
  - The term j = 0 equals |Γ(1−γ)| n^{a−1} → 0.
  - The terms j ≥ 1 equal ∫_0^{L/n} h_n(t) dt, where h_n(t) = |Π_n(⌈nt⌉)|/(1 − ⌈nt⌉/n).
  - By (B1) and the continuity of p, h_n(t) → |p(t)|/(1−t) for every t ∈ (0,½).
  - By (B2), h_n(t) ≤ 2C t^{−a}, which is integrable.
  - Since L/n → ½, dominated convergence shows that the denominator tends to ∫_0^{1/2}|p(t)|dt/(1−t) > 0.

Hence F_n → F. With (1), limsup R_n ≤ max(T, F + ε) for every ε > 0. ∎

**Sanity check (Harcos's case).** Suppose φ ≡ ψ_0 is constant. Then d = 0 on (0,½], so ρ = 0 there and
ρ(t) = γ/t on (½,1]. With w = −γ,

  p(1) = 1 − w ∫_0^{1/2} t^w/(1−t) dt,   ∫_0^{1/2}|p| dt/(1−t) = ∫_0^{1/2} |t^w|/(1−t) dt.

This is Harcos's limit formula.

## 4. A posteriori bounds (Bakeas, Lemma 3; Pinho, integrated form)

Let ρ̃ be any bounded measurable function on (0,1], and define its residual

  R(t) = d(t) + ∫_0^t d(u) ρ̃(t−u) du − t ρ̃(t).

Let e = ρ − ρ̃. Subtracting (2) gives te = d*e + R. With |d| ≤ κ this yields, for every t ∈ (0,1],

  |e(t)| ≤ r(t) + (κ/t) ∫_0^t |e(v)| dv,   (9)

for any function r with |R(t)/t| ≤ r(t).

**Lemma 3 (Bakeas, sup form).** If sup|R/t| < ∞, then ‖e‖_∞ ≤ ε_∞ := sup_t |R(t)/t| / (1−κ).

*Proof.* e = 𝒦e + R/t, so ‖e‖ ≤ κ‖e‖ + ‖R/t‖. ∎

**Lemma 3' (Pinho, integrated form).** Let r ≥ 0 be piecewise constant with |R(t)/t| ≤ r(t), and put
A(t) = ∫_0^t r(v) v^{−κ} dv. Then

  |e(t)| ≤ r(t) + κ t^{κ−1} A(t)  for 0 < t ≤ 1.

*Proof.* Let E(t) = ∫_0^t|e|. By (9), E' ≤ r + κE/t almost everywhere, so (t^{−κ}E)' ≤ t^{−κ}r. Since e is
bounded, E(t) = O(t). With κ < 1 this gives t^{−κ}E(t) → 0 as t → 0. Integrating, E(t) ≤ t^κ A(t), and (9)
gives the claim. ∎

**Consequences for p and F.** Put p̃(t) = t^{−γ} + ∫_0^t (t−v)^{−γ} ρ̃(v) dv. Then
p − p̃ = ∫_0^t (t−v)^{−γ} e(v) dv.
- **Numerator.** |p(1) − p̃(1)| ≤ ∫_0^1 (1−v)^{−a} |e(v)| dv ≤ EN, where EN is either of the following.
  - (sup) EN = ε_∞/(1−a).
  - (integrated) EN = ∫_0^1 r(v)(1−v)^{−a} dv + κ A(1) B(κ, 1−a). This uses A(v) ≤ A(1) and
    ∫_0^1 v^{κ−1}(1−v)^{−a} dv = B(κ, 1−a).
- **Denominator.** Use 1/(1−t) ≤ 2 on [0,½] and Tonelli:
  ∫_0^{1/2} |p − p̃| dt/(1−t) ≤ 2 ∫_0^{1/2} ∫_0^t (t−v)^{−a}|e(v)| dv dt = (2/(1−a)) ∫_0^{1/2} |e(v)| (½−v)^{1−a} dv ≤ ED.
  Here ED is either of the following.
  - (sup) ED = 2ε_∞ (½)^{2−a}/((1−a)(2−a)).
  - (integrated) ED = (2/(1−a)) [ (½)^{1−a} ∫_0^{1/2} r + κ A(½) (½)^{κ+1−a} B(κ, 2−a) ]. This uses
    ∫_0^{1/2} v^{κ−1}(½−v)^{1−a} dv = (½)^{κ+1−a} B(κ, 2−a).
- **Bound for F.** Let D̃_low ≤ ∫_0^{1/2}|p̃| dt/(1−t), and suppose D̃_low − ED > 0. Then

  F ≤ (|p̃(1)| + EN) / (D̃_low − ED).   (10)

The certifier takes, for EN and for ED, the smaller of the sup and integrated bounds. Both are valid.

Sections 5–7 explain how ρ̃, sup|R/t| (cell by cell), p̃(1) and D̃_low are computed rigorously. **All of it is
[NEW].**

---

## 5. [NEW] Trial space and exact residual

### 5.1 Grid and trial function

- **Grid.** Choose an even N such that N·u_j ∈ ℤ for every knot. Then ½ and all knots are grid points, and
  hence so is every finite sum of knots that is ≤ 1.
- **Cells.** h = 1/N, H = N/2, and cells I_m = [mh, (m+1)h] for m = 0..N−1. On a cell, t = h(m+θ) with θ ∈ [0,1].
- **Trial ρ̃.** On cell m, ρ̃(h(m+θ)) = ρ̃_m(θ) = Σ_{b=0}^{P} c_{m,b} θ^b. The c_{m,b} are exact (dyadic) complex
  numbers, and ρ̃ may be discontinuous at grid points.
  - The values of ρ̃ at the grid points themselves are irrelevant: every bound below is a supremum over the
    closed cell θ ∈ [0,1], so both one-sided values are covered.
  - How the c_{m,b} were found is irrelevant to the proof. In practice they come from a floating-point
    collocation and are then rounded to exact dyadics.
- **Why this converges fast (heuristic, used only to choose N and P).** The true ρ is analytic away from the
  points that are finite sums of knots and ½. Those are all grid points, so a piecewise polynomial on this
  grid converges spectrally in P. The worst cells are those just after a small knot.

### 5.2 Lemma C1 (convolution of piecewise polynomials on a uniform grid) [NEW]

Let f, g be piecewise polynomials on the grid: f(h(i+σ)) = f_i(σ) = Σ_a f_{i,a}σ^a and
g(h(j+τ)) = g_j(τ) = Σ_b g_{j,b}τ^b, for σ, τ ∈ [0,1). Write f̌_{i,a}, ǧ_{j,b} for the coefficients in the
reversed variable w = 1−θ, i.e. f_i(1−w) = Σ_a f̌_{i,a} w^a. Then for t = h(m+θ), θ ∈ [0,1],

  (f*g)(t) = ∫_0^t f(u) g(t−u) du = h [ Σ_c θ^{c+1}/(c+1)! · U_{m,c} + Σ_c (1−θ)^{c+1}/(c+1)! · V_{m−1,c} ],

  U_{m,c} = Σ_{i=0}^{m} Σ_{a+b=c} (a! f_{i,a})(b! g_{m−i,b}),   V_{m,c} = Σ_{i=0}^{m} Σ_{a+b=c} (a! f̌_{i,a})(b! ǧ_{m−i,b}),

with V_{−1,c} := 0.

*Proof.*
1. Substitute u = h(i+σ), i = 0..m, and split the cell i at σ = θ.
   - If σ < θ, then t−u lies in cell m−i, with local variable θ−σ.
   - If σ > θ, then t−u lies in cell m−i−1, with local variable 1+θ−σ. For i = m this part is empty.
2. The first part is h Σ_i ∫_0^θ f_i(σ) g_{m−i}(θ−σ) dσ. Use ∫_0^θ σ^a(θ−σ)^b dσ = a!b! θ^{a+b+1}/(a+b+1)!.
3. In the second part (σ ∈ (θ,1), i ≤ m−1) substitute s = 1−σ and write λ = 1−θ.
   - Then f_i(σ) = Σ_a f̌_{i,a} s^a.
   - The local variable in cell m−1−i is τ = 1+θ−σ, and its reversed variable is 1−τ = σ−θ = λ−s. So
     g_{m−1−i}(τ) = Σ_b ǧ_{m−1−i,b}(λ−s)^b.
   - The range is s ∈ (0, λ), so the same Beta identity gives Σ_{a,b} a!b! f̌_{i,a} ǧ_{m−1−i,b} λ^{a+b+1}/(a+b+1)!.
   - Summing over i = 0..m−1 gives the V_{m−1} term. ∎

**Computation.** The double sums U, V are coefficients of products of two univariate polynomials, after
Kronecker substitution: (cell index, degree) ↦ index·S + degree, with stride S ≥ deg f + deg g + 1. The
products are computed with `acb_poly` multiplication, which is outward rounded.

### 5.3 The residual polynomial [NEW]

**d on a cell.** On cell i < H, let φ_i be the exact phase at the left end, β_i the slope on that cell, and
z_i = iβ_i h. Then d(h(i+θ)) = γ − g_i e^{z_i θ}, where g_i = Te^{iφ_i}.
- Replace e^{z_iθ} by its Taylor polynomial of degree Q, giving d_{Q,i}. Since z_iθ is imaginary,
  |e^{iy} − Σ_{k≤Q}(iy)^k/k!| ≤ |y|^{Q+1}/(Q+1)! for real y. So |d − d_Q| ≤ ε_i := T|β_i h|^{Q+1}/(Q+1)! on cell i.
- On cell 0 the constant term of d_{Q,0} is exactly 0, and |d − d_Q|(t) ≤ T(|β_0|t)^{Q+1}/(Q+1)!.
- On cells i ≥ H, d = γ exactly (ε_i = 0). The single point u = ½ does not affect any integral.

Q is the smallest integer with T(Λh)^{Q+1}/(Q+1)! < 2^{−110}.

**Residual.** On cell m, R(t) = R_m(θ) + E_m(θ), where

  R_m = d_{Q,m} + (d_Q*ρ̃)|_m − h(m+θ) ρ̃_m(θ)

is a polynomial of degree ≤ D := P+Q+1. Here (d_Q*ρ̃)|_m is given exactly by Lemma C1. The remainder satisfies

  |E_m| ≤ ε_m + ε_max ∫_0^t |ρ̃| ≤ ε_m + ε_max t ‖ρ̃‖_∞,   ε_max = max_i ε_i.

**Sup of a polynomial on [0,1].** Expand q(θ) = Σ_n χ_n T_n(2θ−1) in Chebyshev polynomials. The map from
monomial to Chebyshev coefficients is an exact rational matrix. Since |T_n| ≤ 1 on [−1,1],
sup_{[0,1]}|q| ≤ Σ_n |χ_n|. The same bound gives ‖ρ̃‖_∞ ≤ max_m Σ_n |χ_n(ρ̃_m)|.

**Cell bounds r_m ≥ sup_{t∈I_m} |R(t)/t|.**
- For m ≥ 1, t ≥ mh, so
  r_m = (Σ_n|χ_n(R_m)| + ε_m)/(mh) + 2ε_max‖ρ̃‖_∞.
  The factor 2 is slack; ε_max‖ρ̃‖_∞ suffices.
- For m = 0, R_0 has zero constant term: d_{Q,0}(0) = 0, the convolution part carries a factor θ, and
  t ρ̃ = hθρ̃_0. So R_0(θ)/θ is a polynomial, and with t = hθ,
  r_0 = Σ_n|χ_n(R_0/θ)|/h + T|β_0|^{Q+1}h^Q/(Q+1)! + ε_max‖ρ̃‖_∞.

Then r(t) := r_m on I_m is a valid majorant in Lemmas 3 and 3'. sup|R/t| ≤ max_m r_m, and ε_∞ = max_m r_m/(1−κ).

### 5.4 Propagation [NEW only in the bookkeeping]

The integrals in Lemma 3' and §4 are evaluated cell by cell. On I_m, r is the constant r_m, so:
- ∫_{I_m} v^{−κ} dv = ((m+1)^{1−κ} − m^{1−κ}) h^{1−κ}/(1−κ);
- ∫_{I_m} (1−v)^{−a} dv = ((1−mh)^{1−a} − (1−(m+1)h)^{1−a})/(1−a);
- ∫_{I_m} dv = h.

Each of these is evaluated in Arb. B(κ, 1−a) and B(κ, 2−a) are evaluated by Arb's Gamma function.

## 6. [NEW] Rigorous evaluation of p̃(1) and of the denominator

### 6.1 Two elementary bounds

Since |γ| = T < 1:
- |binom(−γ, k)| = Π_{j<k} |γ+j|/(j+1) ≤ 1, and likewise |(γ)_l/l!| ≤ 1;
- |binom(1−γ, k)| ≤ Π_{j<k}(|1−γ|+j)/(j+1) ≤ k+1, since |1−γ| ≤ 2.

### 6.2 Numerator p̃(1)

  p̃(1) = 1 + h^{1−γ} Σ_{k=0}^{N−1} ∫_0^1 (N−k−σ)^{−γ} ρ̃_k(σ) dσ.

Put n' = N−k. The cell integrals are, per monomial σ^b:
- n' = 1: ∫_0^1 (1−σ)^{−γ}σ^b dσ = B(b+1, 1−γ).
- n' = 2: ∫_0^1 (2−σ)^{−γ}σ^b dσ = Σ_{e=0}^{b} binom(b,e) 2^{b−e} (−1)^e (2^{e+1−γ} − 1)/(e+1−γ). This follows by
  substituting y = 2−σ and expanding σ^b = (2−y)^b.
- n' ≥ 3: (n'−σ)^{−γ} = n'^{−γ} Σ_{l≥0} ((γ)_l/l!)(σ/n')^l, truncated after L(n') terms. With |(γ)_l/l!| ≤ 1, the
  tail is ≤ n'^{−a} n'^{−L−1}/(1−1/n') · sup|ρ̃_k|, times h^{1−a}.

### 6.3 p̃ on a cell m ≥ 1 of [0,½]

Write t = h(m+θ). Then p̃(t) = t^{−γ} + ∫_0^t u^{−γ} ρ̃(t−u) du. Split the kernel u^{−γ} into kernel cell 0
(u ∈ [0,h)) and kernel cells i ≥ 1.

**Kernel cells i ≥ 1.**
- On kernel cell i, h^{−γ}(i+σ)^{−γ} is replaced by its Taylor polynomial of degree P_k about σ = ½, in powers
  of σ.
- With q_i = 1/(2i+1), the remainder is ≤ h^{−a}(i+½)^{−a} q_i^{P_k+1}/(1−q_i), by 6.1.
- The convolution of this piecewise polynomial (restricted to cells ≥ 1) with ρ̃ is exact by Lemma C1, via one
  more pair of Kronecker products.
- The total error is uniform in t:
  E_kern = h^{1−a}‖ρ̃‖_∞ Σ_{i≥1} (i+½)^{−a} q_i^{P_k+1}/(1−q_i).

**Kernel cell 0, exactly.** Its contribution to p̃(h(m+θ)) is ∫_0^{min(h,t)} u^{−γ}ρ̃(t−u) du. Split it at u = hθ.
- *U-part* (u < hθ, so t−u is in cell m): h^{1−γ} Σ_b c_{m,b} B(b+1, 1−γ) θ^{b+1−γ}.
- *V-part* (hθ < u < h, so t−u is in cell m−1):
  h^{1−γ} Σ_b č_{m−1,b} [ Poly_b(θ) − (−1)^b B(b+1, 1−γ) θ^{b+1−γ} ], where
  Poly_b(θ) = Σ_{e=0}^{b} binom(b,e) (−θ)^{b−e}/(e+1−γ).

  *Proof.* Write u = hy with y ∈ (θ,1). The local variable in cell m−1 is 1+θ−y, and its reversed variable is
  y−θ. So the V-part equals h^{1−γ} Σ_b č_{m−1,b} ∫_θ^1 y^{−γ}(y−θ)^b dy. Expanding (y−θ)^b gives
  ∫_θ^1 y^{−γ}(y−θ)^b dy = Poly_b(θ) − ∫_0^θ y^{−γ}(y−θ)^b dy, and the last integral equals
  (−1)^b B(b+1, 1−γ) θ^{b+1−γ}. ∎

**The singular part.** The two θ^{b+1−γ} pieces combine to

  S_m(θ) = h^{1−γ} θ^{1−γ} Σ_b B(b+1, 1−γ) Δ_{m,b} θ^b,   Δ_{m,b} = c_{m,b} − (−1)^b č_{m−1,b}.

Δ_{m,b} is the jump at mh of the b-th Taylor coefficient of ρ̃. On the cell, |S_m| ≤ σ_m := h^{1−a} Σ_b |B(b+1, 1−γ) Δ_{m,b}|.

**The term t^{−γ}.** With x = θ − ½ ∈ [−½, ½], h^{−γ}(m+½+x)^{−γ} is replaced by its binomial series in x,
truncated after K_m terms. The tail is ≤ h^{−a}(m+½)^{−a} q^{K_m+1}/(1−q), with q = 1/(2m+1).

**Collected.** On cell m, p̃ = A_m(x) + S_m(θ) + Err_m, where:
- A_m is an explicit polynomial in x;
- |Err_m| ≤ (series tails) + E_kern + (top coefficients of A_m that were dropped because they contribute
  < 10^{−19}|A_m(0)|; the sum of their moduli is added to the error).

### 6.4 Lower bound for the denominator ∫_0^{1/2} |p̃| dt/(1−t)

**Inequality S.** For X ≥ 0 and u > 0,  √X ≥ (3/2)Xu − (1/2)X²u³.
*Proof.* Put s = u√X. Then √X − RHS = s(s−1)²(s+2)/(2u) ≥ 0. ∎

The loss is second order in s − 1, so u is chosen close to 1/|A_m|.

**Weight.** w(x) = h/(1−(m+½)h − hx). It is expanded as a geometric series in x, whose tail is bounded
rigorously; the number of terms is chosen so that the tail is < 10^{−22}.

**Regular cells** (σ_m · sup_{I_m} 1/(1−t) · h ≤ 10^{−12}/H).
- |p̃| ≥ |A_m| − σ_m − |Err_m|.
- Put X(x) = A_m(x)·Ā_m(x), where Ā_m is A_m with conjugated coefficients. This is a polynomial with real
  values equal to |A_m(x)|² for real x.
- u is a real polynomial: a float least-squares fit of 1/|A_m| at 24 Chebyshev points, of degree 8 for m < 16,
  6 for m < 256 and 3 otherwise. Its coefficients are exact dyadics.
- u > 0 on [−½,½] is certified by u_0 − Σ_{k≥1}|u_k|2^{−k} > 0.
- Then ∫_{I_m}|p̃|/(1−t) ≥ ∫_{−1/2}^{1/2} [(3/2)Xu − (1/2)X²u³] w dx − sup|f|·(tail of w) − (σ_m + |Err_m|)·∫w.
  The first integral is the exact integral of a polynomial; f denotes the integrand polynomial.

**Knot cells** (all other cells m ≥ 1; typically those starting at a knot).
- S_m is kept exactly.
- θ ∈ [0,1] is split into geometric pieces [2^{−j−1}, 2^{−j}], j = 0..39. On each piece, with centre θ_c and
  half-length ℓ, write θ = θ_c + ℓx' with |x'| ≤ 1. Then θ^{1−γ} = θ_c^{1−γ}(1 + ℓx'/θ_c)^{1−γ}, with
  |ℓx'/θ_c| ≤ 1/3. Its binomial series has the tail bound |θ_c^{1−γ}| Σ_{k>K}(k+1)3^{−k}, by 6.1.
- On each piece, A_m + S_m is a polynomial in x' plus a bounded error, and inequality S is applied as above.
- The piece θ ∈ [0, 2^{−40}] is dropped. This is valid for a lower bound, since the integrand is ≥ 0.

**Cell 0.** Here p̃ = h^{−γ} θ^{−γ} q_0(θ), with the polynomial q_0(θ) = 1 + h Σ_b B(b+1, 1−γ) c_{0,b} θ^{b+1}.
(Kernel cell 0 is the whole integral when t ≤ h.)
- So |p̃| = h^{−a} θ^{−a} |q_0(θ)|, and the cell-0 integral is h^{−a} ∫_0^1 θ^{−a}|q_0(θ)| · h/(1−hθ) dθ.
- Apply inequality S to |q_0| (u a float fit of 1/|q_0| of degree 6 on [0,1]; u > 0 certified by
  u_0 − Σ_{k≥1}|u_k| > 0).
- Expand h/(1−hθ) = Σ_{j<J} h^{j+1}θ^j + rem, with 0 ≤ rem ≤ h^{J+1}/(1−h) on [0,1].
- Integrate with the exact moments ∫_0^1 θ^{k−a} dθ = 1/(k+1−a). The remainder contributes at most
  sup_{[0,1]}|f| · h^{J+1}/((1−h)(1−a)), where f = (3/2)Xu − (1/2)X²u³ and sup|f| ≤ Σ|coefficients|.

The sum over all cells m < H is D̃_low.

### 6.5 Two-sided enclosure (diagnostic only)

On each regular piece put η = sup|u²X − 1|, bounded by the sum of the moduli of the coefficients. If η < 1,
then |s−1| ≤ η and √X − ((3/2)Xu − (1/2)X²u³) ≤ η²(1+η)(3+η)/(2u_min). Add the dropped piece [0, 2^{−40}],
bounded by its length times sup(|A_m| + |S_m|) times sup w. This gives D̃_up ≥ ∫_0^{1/2}|p̃|/(1−t), and then

  F ≥ F_lower := (|p̃(1)| − EN)/(D̃_up + ED).

The claim never uses F_lower. It only shows how tight the certified upper bound is.

## 7. [NEW] Hypotheses checked and the decision rule

The input is T, C, the knots and the phases, all exact rationals (decimal strings are read exactly).
`checkers/rigorous_check.py` checks, in outward-rounded Arb ball arithmetic:
1. 0 < T ≤ C < 1; the knots strictly increase from 0 to ½; there is one phase per knot.
2. N is even and N·u_j ∈ ℤ for every knot.
3. a = Re γ = T cos ψ_0 satisfies 0 < a < 1.
4. κ < 1. It is computed as follows.
   - On (½,1], |d| = T.
   - On (0,½], |d(u)| = 2T|sin((φ(u)−ψ_0)/2)|. The function |φ − ψ_0| is piecewise linear, so its maximum
     E = max_j |ψ_j − ψ_0| is attained at a knot.
   - If E < π (certified), then sup_{(0,½]}|d| = 2T sin(E/2), because sin(x/2) is increasing on [0,π].
     Otherwise the program uses 2T.
   - κ = max(T, that value).
5. Every quantity entering (10) is a finite ball, and D̃_low − ED > 0 (lower endpoint).
6. **PASS iff the upper endpoint of the ball (|p̃(1)| + EN)/(D̃_low − ED) is < C.**

The program also reports, but does not need, F_lower and the error budget.

Compared with Bakeas's and Pinho's checkers:
- there is no β_j ≠ 0 check, because nothing divides by a slope;
- there is no restriction to j/48 knots or to max|ψ_j − ψ_0| < 1, because κ is computed instead.

By Theorem A, a PASS proves limsup R_n ≤ max(T, F) ≤ max(T, C) = C.

---

## 8. The certificate

### 8.1 The profile

File `certificate/c42_profile_0.686360751.json` (SHA-256 in `certificate/SHA256SUMS`):
- T = C = 0.686360751 = 686360751/10⁹;
- K = 33 pieces, with the exact knots u_j and exact decimal phases ψ_j (radians) below. The least common
  denominator of the knots is 18432.

| j | u_j | ψ_j |
|---|---|---|
| 0 | 0 | -1.422143609677031 |
| 1 | 1/18432 | -1.388783825383378 |
| 2 | 1/9216 | -1.381084544438781 |
| 3 | 1/4608 | -1.368351615437954 |
| 4 | 1/2304 | -1.354497472120591 |
| 5 | 1/1152 | -1.33879324681703 |
| 6 | 1/768 | -1.32889661088203 |
| 7 | 1/576 | -1.319911449156991 |
| 8 | 1/288 | -1.297988460091246 |
| 9 | 13/2304 | -1.28028980383792 |
| 10 | 1/128 | -1.266590112297936 |
| 11 | 25/2304 | -1.251202931199948 |
| 12 | 1/72 | -1.238411064787167 |
| 13 | 41/2304 | -1.224388301760243 |
| 14 | 25/1152 | -1.211779864148879 |
| 15 | 1/32 | -1.186173465289945 |
| 16 | 49/1152 | -1.161121902414134 |
| 17 | 1/18 | -1.136007953927681 |
| 18 | 9/128 | -1.110646152046127 |
| 19 | 25/288 | -1.084806526597323 |
| 20 | 121/1152 | -1.058306952419107 |
| 21 | 1/8 | -1.030970091538424 |
| 22 | 169/1152 | -1.002621341389417 |
| 23 | 49/288 | -0.9730792935981849 |
| 24 | 25/128 | -0.9421494103796814 |
| 25 | 2/9 | -0.9096171352870314 |
| 26 | 289/1152 | -0.8752405215067437 |
| 27 | 9/32 | -0.8387413359444171 |
| 28 | 361/1152 | -0.7997941834415409 |
| 29 | 25/72 | -0.7580116838145782 |
| 30 | 49/128 | -0.7129266622002852 |
| 31 | 121/288 | -0.6639595265100425 |
| 32 | 529/1152 | -0.610401510709484 |
| 33 | 1/2 | -0.5512784196680292 |

Origin. These are the knots and phases of session S2's 33-piece candidate (`results/candidate_33pc.json`, run
A_G4_24, optimized in float at T = 0.68636076). Its float fixed point is T* = 0.6863607503 ± 9e-11. Only T was
changed, to C. Re-optimizing the phases at T = 0.68636075 (S5) moved T* by 2e-14, so the phases were kept.

Knot structure:
- near 0: 1/18432, 1/9216, 1/4608, 1/2304, 1/1152, 1/576 (S2 calls these "dyadic");
- the quadratic knots j²/1152 for j = 2..24 (Pinho's family with 24 pieces);
- four adjoint-greedy insertions: 1/768, 13/2304, 25/2304, 41/2304.

Theorem A inputs (certified by the program):
- a = Re γ = T cos ψ_0 = 0.1016540375610781… ∈ (0,1);
- E = max_j|ψ_j − ψ_0| = 0.87087 < π, so sup_{(0,½]}|d| = 2T sin(E/2) = 0.57902;
- κ = max(T, 0.57902) = T = 0.686360751 < 1;
- the slopes satisfy 1.2 ≤ |β_j| ≤ 615 (not used by the proof).

### 8.2 Certified result (checker #1)

`OPENBLAS_NUM_THREADS=1 python checkers/rigorous_check.py certificate/c42_profile_0.686360751.json --N 36864 --P 10 --refine 1`
(256-bit Arb, python-flint 0.9.0, Python 3.13; Q = 13, P_k = 20; full output in
`certificate/expected_N36864_P10.json`):

| quantity | value |
|---|---|
| sup_t \|R(t)/t\| ≤ max_m r_m | 7.27826e-9 (worst cell m = 2) |
| ‖ρ̃‖_∞ ≤ | 422.819 |
| ε_∞ = sup\|R/t\|/(1−κ) | 2.32058e-8 |
| EN (integrated form; sup form 2.583e-8) | 1.59407e-10 |
| ED (integrated form; sup form 7.300e-9) | 6.72278e-11 |
| p̃(1) | 0.130644422135623288746790986588 − 0.177219127225608731395717733545 i |
| \|p̃(1)\| + EN ≤ | 0.2201694442013934931421864 |
| D̃_low (69 knot cells; bookkeeping error 3.18633e-12) | 0.3207780226714513265564244 |
| D̃_low − ED ≥ | 0.3207780226042235570125772 |
| **F ≤ F_upper** | **0.6863607500724540266028192** |
| F ≥ F_lower (diagnostic) | 0.6863607487772309291757098 |
| **C − F_upper** | **9.275e-10** |
| runtime, peak memory | 359.2 s, 3.638 GB |

The claim C is an exact decimal with 9 digits. The tightest enclosure (§8.3, N = 36864, P = 12) is
F(C) ∈ [0.68636074941002, 0.68636074943967]. Linearized with the float slope dF/dT = −1.2229, this puts the fixed
point at T* ≈ 0.6863607503. That agrees with S2's float value 0.6863607502855 ± 9e-11. This is an estimate, not a
proof step. So C is ≈ 7e-10 above T*, and 4.40e-8 below Pinho's 0.6863607950.

### 8.3 Stability, resolution and corruption tests

**Stability.** All runs use the certificate file with C = T = 0.686360751, refine = 1 and 256 bits. The runs
in `certificate/runs_superseded/` were made on an earlier copy of the file that differed only in the descriptive
`checker` field.

| run | N | P | verdict | F_lower | F_upper | width | s | GB |
|---|---|---|---|---|---|---|---|---|
| **primary** | 36864 | 10 | PASS | 0.6863607487772 | 0.6863607500724 | 1.29522e-9 | 359.2 | 3.638 |
| different P | 36864 | 12 | PASS | 0.6863607494100 | 0.6863607494396 | 2.96471e-11 | 345.7 | 3.84 |
| doubled N | 73728 | 10 | PASS | 0.6863607494206 | 0.6863607494290 | 8.37737e-12 | 748.2 | 6.778 |
| coarse grid (superseded primary) | 18432 | 14 | PASS | 0.6863607486167 | 0.6863607502330 | 1.61629e-9 | 295.6 | 2.272 |
| coarse grid | 18432 | 13 | FAIL | 0.6863607474421 | 0.6863607514075 | 3.96545e-9 | 275.4 | 2.077 |
| high P | 36864 | 14 | FAIL | 0.6863607422378 | 0.6863607566116 | 1.43739e-8 | 571.3 | 4.088 |
| coarse grid | 18432 | 12 | FAIL | 0.6863607380942 | 0.6863607607554 | 2.26612e-8 | 241.5 | 2.022 |
| coarse grid, high P | 18432 | 16 | FAIL | 0.6863379598988 | 0.6863835394937 | 4.55796e-5 | 1280.7 | 2.413 |

- The three PASS runs give nested enclosures:
  [0.6863607494207, 0.6863607494290] ⊂ [0.6863607494100, 0.6863607494397] ⊂ [0.6863607487772, 0.6863607500725].
  All lie below C.
- Every FAIL row is a *loose* but valid bound: its enclosure contains the PASS enclosures, and only its upper end
  exceeds C. No pair of enclosures is disjoint.
- Two causes of looseness:
  - N = 18432 is a coarse grid: one cell per first-knot spacing. It passes only at P = 14.
  - At P = 14 on N = 36864, and at P = 16 on N = 18432, the floating-point witness degrades (monomial-basis
    conditioning; at N = 36864, P = 14 GMRES needs 318 iterations and the collocation residual is 4e-8). The
    checker then certifies a poorer ρ̃.
- This is why the primary setting is N = 36864, P = 10. There the witness is well conditioned: 29 GMRES
  iterations and collocation residual 5e-11.

**Corruption tests** (`checkers/corruptions/claim/`, N = 36864, P = 10; outputs in `out/`, summary in
`out/SUMMARY.txt`):

| corrupted certificate | expected | got | F_lower | F_upper |
|---|---|---|---|---|
| T = C = 0.68636074 (below T*) | FAIL | FAIL | 0.6863607622291 | 0.6863607635243 |
| T = C = 0.68636075 (3e-10 below T*) | FAIL | FAIL | 0.6863607500001 | 0.6863607512953 |
| ψ_5 + 0.002 (float F ≈ 0.6863607537) | FAIL | FAIL | 0.6863607530854 | 0.6863607543815 |
| ψ_12 + 0.005 (float F ≈ 0.6863608512) | FAIL | FAIL | 0.6863608506045 | 0.6863608519008 |
| T = 0.68636074, C = F_lower − 1e-9 = 0.686360761229 | FAIL | FAIL | 0.6863607622291 | 0.6863607635243 |
| T = 0.68636074, C = F_upper + 1e-9 = 0.686360764525 (resolution control) | PASS | PASS | 0.6863607622291 | 0.6863607635243 |

- In every FAIL row F_lower > C, so each corrupted claim is certifiably false, not merely uncertified.
  - The closest case is T = C = 0.68636075: F_lower exceeds it by 1.3e-13, so that T is rigorously below the
    fixed point.
- The last two rows show that the checker resolves a claim to 1e-9 at a fixed profile: 1e-9 below the enclosure
  FAILs, 1e-9 above PASSes.
- Hypothesis violations (C < T, a ≤ 0, misaligned grid) are REFUSED. This was tested in S3
  (`checkers/corruptions/run_all.sh`), and the checks are unchanged.

### 8.4 Finite-n cross-check (floating point, not part of the proof)

`src/finite_n_check.py` (float; FFT power-series exp from Bakeas's `crosscheck/finite_n.py`) evaluates the exact
finite-n construction of §3 with K = ⌊n/2⌋ and s_k = T e^{iφ(k/n)}, at T = 0.686360751, for this profile and for
Pinho's profile (C42_compact_best.json). Log: `logs/S5/finite_n_claim.log`.

| n | F_n (this profile) | F_n (Pinho) | difference |
|---|---|---|---|
| 125000 | 0.6864519118 | 0.6864520357 | −1.24e-7 |
| 250000 | 0.6864116410 | 0.6864118216 | −1.81e-7 |
| 500000 | 0.6863890455 | 0.6863892306 | −1.85e-7 |
| 1000000 | 0.6863764269 | 0.6863765926 | −1.66e-7 |
| 2000000 | 0.6863694088 | 0.6863695480 | −1.39e-7 |
| 4000000 | 0.6863655194 | 0.6863656337 | −1.14e-7 |

- F_n decreases monotonically. The ratios of successive increments are constant (≈ 1.81), as expected for
  F_n − F ∝ n^{−0.86}.
- 3-point Aitken extrapolates are still rising towards the limits:
  - this profile: 0.68636016, 0.68636047, 0.68636061, 0.68636068;
  - Pinho's profile: 0.68636027, 0.68636055, 0.68636068, 0.68636074.
- Their last difference, −5.5e-8, agrees with the difference of the limits at this T: F ≈ 0.6863607494 here,
  against ≈ 0.6863607978 for Pinho's profile (his F at T = 0.6863607950 plus 1.2229 × 4.4e-8), i.e. −4.8e-8.
- So the finite-n values are consistent with the certified limit. At these n the finite-n error (~5e-6) is far
  larger than the margin, so this is a consistency check only, not evidence at the 1e-9 level.


## 9. What is new and unreviewed

- **Not new:** Lemmas 1–3 and Theorem A (Bakeas); the arbitrary-knot form of Theorem A and Lemma 3' (Pinho).
- **New and unreviewed:**
  - Lemma C1 and its use for the residual (§5) and for p̃ (§6.3);
  - the closed forms for kernel cell 0, and the singular-part bookkeeping;
  - the denominator via inequality S, and the geometric treatment of knot cells;
  - the profile itself: 33 phase pieces found by S2's adjoint search.
- **Validation of the new code** (details in NOTES_CERTIFIER.md and LOG.md):
  - direct mpmath quadrature agrees at sample points: |R/t| ≤ 1.1e-12, p̃ to 2e-15, p̃(1) to 2e-16;
  - for the exactly solvable constant-phase profile (Harcos), the exact F lies inside the certified enclosure;
  - Bakeas's and Pinho's profiles PASS, with enclosures consistent with their published values;
  - doubling N or changing P moves F_upper by < 1e-11 once converged;
  - corruption tests FAIL as they should (§8).
- **Second checker.** CLAUDE.md requires a second checker, written independently from this file and the
  certificate format only. At the time of writing it has **not** been written (planned for session S6).
  Until it exists and agrees, the bound must be treated as a candidate.

## AI use

The search, the certifier, its derivation and this text were produced by AI agents (Claude Code), directed by
the repository owner. The mathematics of Sections 1–4 is by Bakeas (with AI assistance, per his notes), Harcos
and Pinho. Nothing in Sections 5–8 has been reviewed by a human mathematician. Corrections are welcome.
