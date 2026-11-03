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

Rows: depth of the first knot. Entries: pieces → float T*. err(T*) ≤ 1e-10 everywhere, ≤ 4e-11 for the deep runs.
The full table is in `results/summary_table.txt`.

| first knot | small (greedy / dyadic, quad:24 bulk) | quad:48 bulk | quad:96 bulk | quad:192 |
|---|---|---|---|---|
| 1/18432 | 29 → 0.6863607538 (D24_6); 33 → 0.6863607503 (G4_24); 37 → 0.6863607494 (G8_24) | 50 → 0.6863607513 (q48_g1) | 96 → 0.6863607483 | — |
| 1/73728 | 35 → 0.6863607454 (S35); 39 → 0.6863607445 (S39) | 52 → 0.6863607465 (q48_g4) | 99 → 0.6863607433 (D96_4) | 192 → 0.6863607431 |
| 1/147456 | 36 → 0.6863607446 (S36); 40 → 0.6863607438 (S40) | — | 100 → 0.6863607426 (D96_5) | — |
| 1/294912 | — | — | **101 → 0.6863607423 (D96_6)** | — |

Pinho's published profile (48 pieces, first knot 1/4608) has T* = 0.6863607720. Measured against the best value
found, 0.6863607423 (a total gain of 2.98e-8):
- **33 pieces / 1/18432** capture 73% of the gain;
- **35 pieces / 1/73728** capture 89%;
- **40 pieces / 1/147456** capture 95%.

## 4. Conclusions

- **Piece count saturates at ~35–40 pieces** when the knots are placed by the greedy adjoint rule
  (S35/S39/S40). At equal first-knot depth, they come within ~1e-9 of 99–192 pieces.
- **First-knot depth keeps mattering, with diminishing returns.** At bulk quad:96 (D96_k):
  - 1/18432 → 1/73728 gains 5.0e-9;
  - 1/73728 → 1/147456 gains 0.7e-9;
  - 1/147456 → 1/294912 gains 0.35e-9;
  - the remaining room is estimated at ≈ 3e-10.
- **The float infimum of this family is ≈ 0.686360742 ± 3e-10.**
  - Depth gains at bulk 96 per halving of the first knot: 7.3e-10, then 3.5e-10 (ratio ~0.48).
  - Bulk beyond 96 is worth ≤ 2e-10.
  - The best run, D96_6 with 101 pieces (0.6863607423), is within ~3e-10 of the infimum.
- **Candidates exported** (`results/candidate_{101,40,33}pc.json`): exact rational breaks, decimal phases,
  float F with error, T*, and float values of the Theorem A inputs (a, κ, slopes).
- **Certification cost.** Deeper first knots mean steeper first pieces: max |slope| is ≈ 615 at a first knot of
  1/18432, ≈ 2000 at 1/73728, ≈ 3650 at 1/147456 and ≈ 6800 at 1/294912 (Pinho: 196). Pinho's error budget is dominated by the
  first-knot cell. A certifier with a graded grid near 0 should therefore care about the first-knot depth much
  more than about the piece count.
