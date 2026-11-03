# Direction A — saturation study (S2, 2026-11-03, branch `search-A`)

**All numbers are floats from the search evaluator. Nothing here is certified.**

Family: Bakeas's Theorem A, constant modulus T, piecewise-linear phase φ on exact rational breaks.

Method:
- **Objective.** L-BFGS-B on the Richardson-extrapolated F over (N0, 2N0, 4N0) at fixed T, using the exact
  discrete adjoint and the O(N log² N) solver `src/fastsolve.py`.
- **Verification.** Re-evaluate on independent levels (2N0, 4N0, 8N0), with phases rounded to the 16-digit
  decimals that are saved.
- **Fixed point.** T* solves F(T*) = T* with the phases frozen; dF/dT = −1.2229.
- **Error.** err(T*) = max(|fit_opt − fit_ind|, |fit_ind − fit_4level|)/(1 − dF/dT).
- **Check of the error estimate.** For quad:192, pushing the levels 2× further (up to N = 2,359,296) moved F by
  3e-12, well inside the quoted 2e-11.

"Gain" is measured against Pinho's published 48-piece profile, float T* = 0.6863607720381.

"Max certifier gap" = (1 − dF/dT)·(0.6863607950 − T*) = 2.2229·margin. This is the largest F_upper − F a rigorous
check may lose at the claimed T while still giving a bound below 0.6863607950. For comparison, Pinho's certificate
lost 3.3e-8 on his own profile.

## 1. Quadratic knots u_j = j²/(2m²) (Pinho's family)

| pieces | first knot | T* | err | gain vs Pinho |
|---|---|---|---|---|
| 24 | 1/1152 | 0.6863608822895 | 1.2e-10 | −1.10e-7 |
| 48 | 1/4608 | 0.6863607718846 | 5e-12 | 1.5e-10 (re-optimised Pinho) |
| 96 | 1/18432 | 0.6863607482663 | 9e-11 | 2.38e-8 |
| 192 | 1/73728 | 0.6863607431492 | 2e-11 | 2.89e-8 |

The gains per doubling are 1.10e-7, 2.36e-8 and 5.1e-9, a ratio of about 4.6 each time. But most of each
doubling's gain comes from the **finer first knot**, not from the piece count (Section 2).

## 2. Where the knots matter

Near u = 0 the optimal phase is singular: φ(u) − φ(0) ∝ u^ν with ν ≈ 0.28. The adjoint knot gradient and the
insertion scores (`src/knotgrad.py`) show three things:
1. On A24_q the first interval dominates, by 20×.
2. Once the first knot is small, the bottleneck moves to **u ≈ 1e-3 … 3e-2**. Ratio-4 gaps there (quad:24's
   1/1152 → 1/288) cost ~1e-8.
3. Below ~1e-3 the spacing ratio (√2, 2 or 4) hardly matters (A_Rs2_24 vs A_D24_6: 4e-10 for 5 extra pieces).

The bulk u > 0.05 is stationary: knot gradients are ≲ 3e-11.

## 3. T* by piece count and first-knot depth (best runs)

FILLED IN BELOW (Section 5) FROM `python src/summarize.py`.

## 4. Conclusions

- **Piece count saturates at ~35–40 pieces** when the knots are placed by the greedy adjoint rule
  (S35/S39/S40). At equal first-knot depth, they come within ~1e-9 of 99–192 pieces.
- **First-knot depth keeps mattering, with diminishing returns.** At bulk quad:96 (D96_k):
  - 1/18432 → 1/73728 gains 5.0e-9;
  - 1/73728 → 1/147456 gains 0.7e-9;
  - the remaining room is estimated at ≲ 0.5e-9.
- **The float infimum of this family is ≈ 0.686360742.** The best run (D96_5, 100 pieces) is within ~1e-9 of it.
- **Certification cost.** Deeper first knots mean steeper first pieces: max |slope| is ≈ 615 at a first knot of
  1/18432, ≈ 2000 at 1/73728 and ≈ 3000 at 1/147456 (Pinho: 196). Pinho's error budget is dominated by the
  first-knot cell. A certifier with a graded grid near 0 should therefore care about the first-knot depth much
  more than about the piece count.
