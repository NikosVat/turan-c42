# Findings: Harcos's reduction is exact at finite n, and evidence that smooth phase profiles are near-optimal

Status: unrefereed note from this project (session S4, 2026-11-03). Section 2 has complete, elementary proofs.
Sections 5–7 report **floating-point numerics** (local optimisation). The only rigorous numbers in them are the Arb
upper bounds for R_n at the finite n listed, which say nothing rigorous about C42. Nothing here changes the
certified upper bound for C42. The current record is Pinho's 0.6863607950* (teorth PR #305).

Notation follows MATH.md.
- R_n = min over z ∈ ℂⁿ with max_j|z_j| = 1 of max_{1≤k≤n}|s_k(z)|, where s_k(z) = Σ_j z_j^k.
- C42 = limsup R_n.
- K = ⌊n/2⌋. ‖s‖ = max_k|s_k| on ℂ^K.

<!-- RESULTS-SUMMARY -->

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
