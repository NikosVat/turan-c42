#!/usr/bin/env python3
"""Spectral a-posteriori certifier for F in Bakeas's Theorem A (Arb ball arithmetic via python-flint).

Claim checked:  F < C, where C (default T) is the claimed bound, T <= C < 1, for the profile
g(u) = T exp(i phi(u)) on [0,1/2], phi piecewise linear through the given rational knots.
By Theorem A (Bakeas), limsup R_n <= max(T, F) <= C.

Method (see NOTES_CERTIFIER.md for the full derivation):
  1. A FLOATING witness rho~ (spectral_witness.py): discontinuous piecewise polynomial of degree P on N
     uniform cells of [0,1] (all knots and 1/2 are grid points).  Its coefficients are then treated as
     exact dyadic numbers; nothing below depends on how they were found.
  2. Residual R = d + d*rho~ - t rho~, computed EXACTLY as a piecewise polynomial on every cell
     (d replaced by its degree-Q Taylor polynomial on every cell, with a rigorous remainder), using
     the identity for convolutions of piecewise polynomials on a uniform grid (two Kronecker products).
     sup_cell |R| <= sum |Chebyshev coefficients|.
  3. Lemma 3 (Bakeas) in sup norm, and the integrated (Gronwall) form, give |rho - rho~| bounds.
  4. p~ = t^-gamma + int (t-v)^-gamma rho~(v) dv: numerator p~(1) by exact cell moments; denominator
     int_0^1/2 |p~|/(1-t) by an exact polynomial representation of p~ on every cell (kernel cell 0
     exact, kernel cells >= 1 by Taylor polynomials with rigorous remainders, one more pair of Kronecker
     products) and the inequality sqrt(X) >= (3/2) X u - (1/2) X^2 u^3 (u > 0), X = |p~|^2.
  5. F <= (|p~(1)| + EN) / (D_low - ED).  PASS iff this is < C (exact rational comparison).

All proof steps use outward-rounded Arb balls.  Only the witness is floating point.
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import argparse
import hashlib
import json
import math
import sys
import time
from fractions import Fraction

import numpy as np
from flint import acb, acb_mat, acb_poly, arb, ctx, fmpq

if not __debug__:
    sys.exit("run this program without python -O (it relies on assert)")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spectral_witness as SW  # noqa: E402  (floating witness only)


# ------------------------------------------------------------------------------------------------
# small helpers
# ------------------------------------------------------------------------------------------------
def R(q):
    q = Fraction(q)
    return arb(fmpq(q.numerator, q.denominator))


def up(x):
    """exact upper endpoint (as an exact arb) of a real ball."""
    return arb(x.upper())


def lo(x):
    return arb(x.lower())


def absup(z):
    return up(abs(z))


def exact_acb(z):
    return acb(float(z.real), float(z.imag))


def amax(xs):
    m = arb(0)
    for x in xs:
        x = up(x)
        if x > m:
            m = x
    return m


def cheb_of_monomials(D):
    """Exact matrix M[n][k]: theta^k = sum_n M[n][k] T_n(2 theta - 1), theta in [0,1], k <= D."""
    # x^j in Chebyshev basis
    X = [[Fraction(0)] * (D + 1) for _ in range(D + 1)]  # X[n][j]
    for j in range(D + 1):
        for r in range(j // 2 + 1):
            n = j - 2 * r
            c = Fraction(math.comb(j, r), 2 ** (j - 1)) if j > 0 else Fraction(1)
            if j > 0 and n == 0:
                c = Fraction(math.comb(j, r), 2 ** j)
            X[n][j] += c
    M = [[Fraction(0)] * (D + 1) for _ in range(D + 1)]
    for k in range(D + 1):
        # theta^k = 2^-k (1+x)^k
        for j in range(k + 1):
            cj = Fraction(math.comb(k, j), 2 ** k)
            for n in range(j + 1):
                if X[n][j]:
                    M[n][k] += cj * X[n][j]
    return M


def rows_times(rows, LT):
    """rows: list of equal-length lists of acb (one per cell); LT: acb_mat (len x k). Returns flat
    row-major list of the products (one row of length k per cell)."""
    if not rows:
        return []
    M = acb_mat(len(rows), len(rows[0]), [x for r in rows for x in r])
    return (M * LT).entries()


def mat_from(rows):
    return acb_mat(len(rows), len(rows[0]), [x if isinstance(x, acb) else acb(x) for r in rows for x in r])


# ------------------------------------------------------------------------------------------------
# profile
# ------------------------------------------------------------------------------------------------
class Profile:
    def __init__(self, data):
        if data.get("constant", "C_42") != "C_42":
            raise ValueError("wrong constant")
        self.Tq = Fraction(data["T"])
        self.ph = [Fraction(x) for x in data["phases"]]
        K = len(self.ph) - 1
        if isinstance(data.get("breaks"), list):
            self.kn = [Fraction(x) for x in data["breaks"]]
        elif isinstance(data.get("knots"), list):
            self.kn = [Fraction(x) for x in data["knots"]]
        else:
            if data.get("knots", "j/48, j = 0..24") != "j/48, j = 0..24" or K != 24:
                raise ValueError("explicit rational knots required unless Bakeas's j/48 format")
            self.kn = [Fraction(j, 48) for j in range(25)]
        if not (0 < self.Tq < 1) or K < 1:
            raise ValueError("need 0 < T < 1 and at least one phase piece")
        if len(self.kn) != len(self.ph) or self.kn[0] != 0 or self.kn[-1] != Fraction(1, 2):
            raise ValueError("knots must run from 0 to 1/2, one phase per knot")
        if any(u >= v for u, v in zip(self.kn, self.kn[1:])):
            raise ValueError("knots must be strictly increasing")
        self.slopes = [(y - x) / (v - u) for x, y, u, v in zip(self.ph, self.ph[1:], self.kn, self.kn[1:])]
        self.Cq = Fraction(data.get("C", data["T"]))
        if not (self.Tq <= self.Cq < 1):
            raise ValueError("claim C must satisfy T <= C < 1")
        self.lcd = 1
        for u in self.kn:
            self.lcd = self.lcd * u.denominator // math.gcd(self.lcd, u.denominator)

    def check_grid(self, N):
        if N % 2 or any((u * N).denominator != 1 for u in self.kn):
            raise ValueError(f"N={N} must be even and align every knot (lcd {self.lcd})")

    def phase_at_cell(self, N):
        """per cell i < N/2: (exact phase at left end, exact slope)."""
        out = []
        j = 0
        for i in range(N // 2):
            u = Fraction(i, N)
            while not (self.kn[j] <= u < self.kn[j + 1]):
                j += 1
            out.append((self.ph[j] + self.slopes[j] * (u - self.kn[j]), self.slopes[j]))
        return out


# ------------------------------------------------------------------------------------------------
# hypotheses of Theorem A
# ------------------------------------------------------------------------------------------------
def hypotheses(pr):
    T = R(pr.Tq)
    gamma = T * acb(0, R(pr.ph[0])).exp()
    a = gamma.real
    # kappa = sup_(0,1] |d|; |d| = T on (1/2,1]; on (0,1/2]: |d(u)| = 2T|sin((phi(u)-phi(0))/2)|.
    E = max(abs(x - pr.ph[0]) for x in pr.ph)  # max |phi - phi(0)| is attained at a knot (pw linear)
    if R(E) < arb.pi():
        k1 = 2 * T * (R(E) / 2).sin()  # 2T sin(x/2) increasing on [0, pi]
    else:
        k1 = 2 * T
    kappa = up(k1) if up(k1) > T else T
    checks = {
        "0 < a = Re gamma": bool(a > 0),
        "a < 1": bool(a < 1),
        "kappa < 1": bool(kappa < 1),
        "T < 1": bool(T < 1),
    }
    if not all(checks.values()):
        raise ArithmeticError(f"hypothesis of Theorem A not certified: {checks}")
    Lam = max(abs(s) for s in pr.slopes)
    return dict(T=T, gamma=gamma, a=a, kappa=kappa, max_phase_excursion=E, Lambda=Lam, checks=checks)


# ------------------------------------------------------------------------------------------------
# 1. residual
# ------------------------------------------------------------------------------------------------
def residual_bounds(pr, hy, N, P, Crows, log, nodes=None):
    T, gamma = hy["T"], hy["gamma"]
    h = R(Fraction(1, N))
    H = N // 2
    # Taylor degree of d on cells: T |beta h|^{Q+1}/(Q+1)! < 2^-110
    zmax = max(abs(s) for s in pr.slopes) / N
    Q = 1
    while float(pr.Tq) * float(zmax) ** (Q + 1) / math.factorial(Q + 1) > 2.0 ** -110:
        Q += 1
    S = P + Q + 1
    D = P + Q + 1
    cells = pr.phase_at_cell(N)
    zero = acb(0)
    f_rows = []  # Taylor coefficients of d_Q per cell (powers of theta)
    ahat, arev = [], []
    eps = []
    for i in range(N):
        if i < H:
            phi, beta = cells[i]
            z = acb(0, R(beta / N))
            g = T * acb(0, R(phi)).exp()
            ge = T * acb(0, R(phi + beta / N)).exp()
            # On cell 0, phi_0 = psi_0 and g_0 = gamma, so the constant term gamma - g_0 of d_{Q,0} is
            # mathematically 0. It is set to the EXACT zero acb(0) (radius 0), not computed as the
            # ball gamma - g (which would only contain 0); see NOTES.md 5.3 and the assertion below.
            fr = [zero if i == 0 else gamma - g]
            hr = [fr[0]]
            rr = [gamma - ge]
            zp = acb(1)
            mz = -z
            mzp = acb(1)
            fac = 1
            for k in range(1, Q + 1):
                zp = zp * z
                mzp = mzp * mz
                fac *= k
                hr.append(-g * zp)
                fr.append(-g * zp / fac)
                rr.append(-ge * mzp)
            eps.append(T * abs(R(beta / N)) ** (Q + 1) / math.factorial(Q + 1))
        else:
            fr = [gamma] + [zero] * Q
            hr = list(fr)
            rr = list(fr)
            eps.append(arb(0))
        f_rows.append(fr)
        ahat.extend(hr + [zero] * (S - Q - 1))
        arev.extend(rr + [zero] * (S - Q - 1))
    eps_max = amax(eps)
    log("residual: d coefficients", Q=Q)
    # exact trial coefficients and their reversed / factorial-scaled versions
    rev = [[Fraction((-1) ** a * math.comb(b, a)) for b in range(P + 1)] for a in range(P + 1)]
    revT = mat_from([[R(rev[a][b]) for a in range(P + 1)] for b in range(P + 1)])  # (P+1)x(P+1) = Rev^T
    Crev = []
    for m0 in range(0, N, 4096):
        blk = Crows[m0:m0 + 4096]
        ent = rows_times(blk, revT)
        Crev.extend([ent[r * (P + 1):(r + 1) * (P + 1)] for r in range(len(blk))])
    fb = [math.factorial(b) for b in range(P + 1)]
    ghat, grev = [], []
    for m in range(N):
        ghat.extend([Crows[m][b] * fb[b] for b in range(P + 1)] + [zero] * (S - P - 1))
        grev.extend([Crev[m][b] * fb[b] for b in range(P + 1)] + [zero] * (S - P - 1))
    U = (acb_poly(ahat) * acb_poly(ghat)).coeffs()
    log("residual: U product")
    V = (acb_poly(arev) * acb_poly(grev)).coeffs()
    log("residual: V product")
    del ahat, arev, ghat, grev
    U += [zero] * max(0, N * S - len(U))
    V += [zero] * max(0, N * S - len(V))
    # Chebyshev matrices
    M = cheb_of_monomials(D + 1)
    MD = lambda n, k: R(M[n][k])  # noqa: E731
    C = P + Q + 1  # number of U/V coefficients per cell (c = 0..P+Q)
    # L^T rows: [f (Q+1)] [U (C)] [V (C)] [m*c (P+1)] [c (P+1)]  ->  columns: cheb(R) (D+1), cheb(rho~) (P+1)
    LT = []
    for k in range(Q + 1):
        LT.append([MD(n, k) for n in range(D + 1)] + [arb(0)] * (P + 1))
    for c in range(C):
        f1 = math.factorial(c + 1)
        LT.append([h * MD(n, c + 1) / f1 for n in range(D + 1)] + [arb(0)] * (P + 1))
    for c in range(C):
        f1 = math.factorial(c + 1)
        LT.append([(-1) ** n * h * MD(n, c + 1) / f1 for n in range(D + 1)] + [arb(0)] * (P + 1))
    for b in range(P + 1):
        LT.append([-h * MD(n, b) for n in range(D + 1)] + [arb(0)] * (P + 1))
    for b in range(P + 1):
        LT.append([-h * MD(n, b + 1) for n in range(D + 1)] + [MD(n, b) for n in range(P + 1)])
    nn = 0
    if nodes is not None:  # extra columns: R_m(theta_l) at the collocation nodes (exact dyadic nodes)
        nn = len(nodes)
        th = [R(Fraction(float(x))) for x in nodes]
        k = 0
        for q in range(Q + 1):
            LT[k] += [t_ ** q for t_ in th]; k += 1
        for c in range(C):
            LT[k] += [h * t_ ** (c + 1) / math.factorial(c + 1) for t_ in th]; k += 1
        for c in range(C):
            LT[k] += [h * (1 - t_) ** (c + 1) / math.factorial(c + 1) for t_ in th]; k += 1
        for b in range(P + 1):
            LT[k] += [-h * t_ ** b for t_ in th]; k += 1
        for b in range(P + 1):
            LT[k] += [-h * t_ ** (b + 1) for t_ in th]; k += 1
    LT = mat_from(LT)
    node_res = np.zeros((N, nn), complex)
    chebR = [None] * N
    rho_sup = [None] * N
    W = D + 1 + P + 1 + nn
    for m0 in range(0, N, 2048):
        rows = []
        for m in range(m0, min(N, m0 + 2048)):
            Vm = V[(m - 1) * S:(m - 1) * S + C] if m >= 1 else [zero] * C
            rows.append(f_rows[m] + U[m * S:m * S + C] + Vm + [Crows[m][b] * m for b in range(P + 1)] + Crows[m])
        ent = rows_times(rows, LT)
        for r in range(len(rows)):
            row = ent[r * W:(r + 1) * W]
            chebR[m0 + r] = sum((abs(x) for x in row[:D + 1]), arb(0))
            rho_sup[m0 + r] = sum((abs(x) for x in row[D + 1:D + 2 + P]), arb(0))
            if nn:
                m = m0 + r
                node_res[m] = [complex(float(z.real.mid()), float(z.imag.mid())) / (float(h.mid()) * (m + float(x)))
                               for z, x in zip(row[D + 2 + P:], nodes)]
    if nodes is not None:
        return node_res
    rho_norm = amax(rho_sup)
    # cell 0: R_0(theta)/theta. The constant term of R_0 is exactly 0: d_{Q,0}(0) is the exact zero
    # (asserted here), the convolution part on cell 0 is h sum_c theta^{c+1} U_{0,c}/(c+1)! (Lemma C1,
    # V_{-1} = 0) and t rho~ = h theta rho~_0. So R_0/theta is formed by shifting coefficients
    # (k -> k-1); nothing is divided by theta, and no ball around 0 is ever dropped.
    assert f_rows[0][0].is_zero() and f_rows[0][0].is_exact(), "d_{Q,0}(0) must be the exact zero"
    mono = [acb(0)] * (D + 1)
    for k in range(1, Q + 1):
        mono[k - 1] += f_rows[0][k]
    for c in range(C):
        mono[c] += h * U[c] / math.factorial(c + 1)
    for b in range(P + 1):
        mono[b] -= h * Crows[0][b]
    cheb0 = [sum((MD(n, k) * mono[k] for k in range(D + 1)), acb(0)) for n in range(D + 1)]
    beta0 = abs(R(pr.slopes[0]))
    r = [None] * N
    r[0] = up(sum((abs(x) for x in cheb0), arb(0)) / h + T * beta0 ** (Q + 1) * h ** Q / math.factorial(Q + 1)
              + eps_max * rho_norm)
    for m in range(1, N):
        r[m] = up((chebR[m] + eps[m]) / (m * h) + 2 * eps_max * rho_norm)
    worst = max(range(N), key=lambda m: float(r[m].upper()))
    log("residual: certified", sup=r[worst].str(5), worst=worst)
    return dict(r=r, rho_sup=rho_sup, rho_norm=rho_norm, Q=Q, worst=worst, sup=up(r[worst]), Crows=Crows,
                Crev=Crev, eps_max=eps_max)


# ------------------------------------------------------------------------------------------------
# 2. propagation (Lemma 3, sup norm; and the integrated Gronwall form)
# ------------------------------------------------------------------------------------------------
def propagation(hy, N, res):
    a, kappa = hy["a"], hy["kappa"]
    br = 1 - a
    h = R(Fraction(1, N))
    H = N // 2
    r = res["r"]
    sup = res["sup"]
    eps_inf = up(sup / (1 - kappa))
    EN_sup = up(eps_inf / br)
    half = R(Fraction(1, 2))
    ED_sup = up(2 * eps_inf * half ** (2 - a) / (br * (2 - a)))
    # integrated: e(t) <= r(t) + kappa t^{kappa-1} int_0^t r v^-kappa dv
    A1 = arb(0)
    Ah = arb(0)
    Rh = arb(0)
    RN = arb(0)
    prevA = arb(0)
    prevN = arb(1)
    for m in range(N):
        t1 = R(Fraction(m + 1, N))
        nA = t1 ** (1 - kappa)
        nN = (1 - t1) ** br if m + 1 < N else arb(0)
        A1 += r[m] * (nA - prevA) / (1 - kappa)
        RN += r[m] * (prevN - nN) / br
        if m < H:
            Ah += r[m] * (nA - prevA) / (1 - kappa)
            Rh += r[m] * h
        prevA, prevN = nA, nN
    B1 = kappa.gamma() * br.gamma() / (kappa + br).gamma()
    B2 = kappa.gamma() * (br + 1).gamma() / (kappa + br + 1).gamma()
    EN_int = up(RN + kappa * A1 * B1)
    ED_int = up(2 / br * (half ** br * Rh + kappa * Ah * half ** (kappa + br) * B2))
    EN = EN_int if EN_int < EN_sup else EN_sup
    ED = ED_int if ED_int < ED_sup else ED_sup
    return dict(eps_inf=eps_inf, EN=EN, ED=ED, EN_sup=EN_sup, ED_sup=ED_sup, EN_int=EN_int, ED_int=ED_int)


# ------------------------------------------------------------------------------------------------
# 3. numerator p~(1)
# ------------------------------------------------------------------------------------------------
def numerator(hy, N, P, res):
    gamma, a = hy["gamma"], hy["a"]
    b1 = 1 - gamma
    Crows = res["Crows"]
    Bb = [acb(math.factorial(b)) * b1.gamma() / (b + 1 + b1).gamma() for b in range(P + 1)]
    tot = sum((Crows[N - 1][b] * Bb[b] for b in range(P + 1)), acb(0))
    J = []
    for b in range(P + 1):
        s = acb(0)
        for e in range(b + 1):
            s += math.comb(b, e) * 2 ** (b - e) * (-1) ** e * (acb(2) ** (e + b1) - 1) / (e + b1)
        J.append(s)
    tot += sum((Crows[N - 2][b] * J[b] for b in range(P + 1)), acb(0))
    # n' >= 3 : series in sigma/n'
    Lmax = int(math.ceil(25 * math.log(10) / math.log(3))) + 1
    rf = [acb(1)]
    for l in range(1, Lmax + 1):
        rf.append(rf[-1] * (gamma + l - 1) / l)  # (gamma)_l / l!
    hilb = mat_from([[R(Fraction(1, l + b + 1)) for l in range(Lmax + 1)] for b in range(P + 1)])
    tail = arb(0)
    for k0 in range(0, N - 2, 4096):
        blk = Crows[k0:min(N - 2, k0 + 4096)]
        ent = rows_times(blk, hilb)
        for r_, k in enumerate(range(k0, k0 + len(blk))):
            n1 = N - k
            L = min(Lmax, int(math.ceil(25 * math.log(10) / math.log(n1))) + 1)
            w = ent[r_ * (Lmax + 1):(r_ + 1) * (Lmax + 1)]
            x = arb(1) / n1
            s = acb(0)
            for l in range(L, -1, -1):
                s = s * x + rf[l] * w[l]
            tot += s * acb(n1) ** (-gamma)
            # The Horner loop keeps the L+1 terms l = 0..L, so the omitted tail is
            # <= n'^-a n'^-(L+1)/(1-1/n') sup|rho~_k|. We use the conservative bound
            # n'^-a n'^-L/(1-1/n') sup|rho~_k| (one factor n' larger), which is valid even if only
            # l = 0..L-1 were kept (NOTES.md 6.2; REVIEW.md 2, section 6.2).
            tail += arb(n1) ** (-a) * x ** L / (1 - x) * res["rho_sup"][k]
    h = R(Fraction(1, N))
    hg = acb(h) ** (1 - gamma)
    p1 = 1 + hg * tot
    tail = up(tail * h ** (1 - a))
    p1 = p1 + acb(arb(0, tail), arb(0, tail))
    return p1


# ------------------------------------------------------------------------------------------------
# 4. denominator lower bound
# ------------------------------------------------------------------------------------------------
def poly_conj(p):
    return acb_poly([c.conjugate() for c in p.coeffs()])


def poly_absmax(p, rad):
    """upper bound of sup_{|x|<=rad} |p(x)| (rad exact arb)."""
    s = arb(0)
    rp = arb(1)
    for c in p.coeffs():
        s += abs(c) * rp
        rp = rp * rad
    return up(s)


def integrate_sym(p):
    """exact integral over x in [-1/2, 1/2] of a polynomial (acb coefficients)."""
    s = acb(0)
    for k, c in enumerate(p.coeffs()):
        if k % 2 == 0:
            s += c / (arb(2) ** k * (k + 1))
    return s


def fit_u_sym(Af, deg=3):
    """float positive polynomial u(x) ~ 1/|A(x)| on [-1/2,1/2] (exact dyadic coefficients)."""
    xs = np.cos((2 * np.arange(24) + 1) * np.pi / 48) / 2
    vals = np.abs(np.polyval(Af[::-1], xs))
    c = np.polynomial.polynomial.polyfit(xs, 1.0 / vals, deg)
    return [float(v) for v in c]


def wterms(ew):
    """number of terms J+1 of the geometric weight series with (ew/2)^(J+1) < 1e-22."""
    e = float(ew.upper()) / 2
    if e <= 0:
        return 1
    return max(4, int(math.ceil(-22 * math.log(10) / math.log(e))))


def sqrt_lower_integral(Ap, wser, wrem, rad, u_coef, wsup=None):
    """Bounds for int_{-rad}^{rad} |A(x)| w(x) dx given w = wser(x) + r(x), |r| <= wrem, rad = 1/2.
    Lower: sqrt(X) >= f := (3/2) X u - (1/2) X^2 u^3 for u > 0 (X = |A|^2).
    Upper (if wsup >= sup w is given): sqrt(X) - f = s(s-1)^2(s+2)/(2u), s = u sqrt(X); with
    eta = sup|u^2 X - 1| < 1 this is <= eta^2 (1+eta)(3+eta)/(2 u_min).  Returns lower, or (lower, upper)."""
    u = acb_poly([acb(c) for c in u_coef])
    # certify u > 0 on [-rad, rad]
    u0 = u.coeffs()[0].real
    rest = arb(0)
    rp = arb(1)
    for c in u.coeffs()[1:]:
        rp = rp * rad
        rest += abs(c) * rp
    umin = u0 - rest
    if not (umin > 0):
        return None
    X = Ap * poly_conj(Ap)
    u3 = u * u * u
    f = X * u * acb(1.5) - X * X * u3 * acb(0.5)
    val = integrate_sym(f * wser).real
    err = poly_absmax(f, rad) * wrem  # times length 1
    low = lo(val - err)
    if wsup is None:
        return low
    eta = poly_absmax(u * u * X - acb(1), rad)
    if not (eta < 1):
        return low, None
    ex = eta ** 2 * (1 + eta) * (3 + eta) / (2 * lo(umin))
    return low, up(val + err + ex * wsup)


def denominator_lower(pr, hy, N, P, res, Pk, log, geo_J=40, budget=1e-12):
    T, gamma, a = hy["T"], hy["gamma"], hy["a"]
    b1 = 1 - gamma
    H = N // 2
    h = R(Fraction(1, N))
    half = R(Fraction(1, 2))
    Crows, Crev = res["Crows"], res["Crev"]
    zero = acb(0)
    hmg = acb(h) ** (-gamma)
    h1g = acb(h) ** (1 - gamma)
    h1a = h ** (1 - a)
    # ---- kernel cells i = 1..H-1: Taylor (centered) of h^-gamma (i+sigma)^-gamma, monomials in sigma
    bs = [acb(1)]
    for k in range(1, Pk + 1):
        bs.append(bs[-1] * (-gamma - k + 1) / k)
    shiftT = mat_from([[R(Fraction(math.comb(k, j)) * Fraction(-1, 2) ** (k - j)) if j <= k else arb(0)
                        for j in range(Pk + 1)] for k in range(Pk + 1)])  # row k -> monomial j
    revK = mat_from([[R(Fraction((-1) ** aa * math.comb(bb, aa))) if aa <= bb else arb(0)
                      for aa in range(Pk + 1)] for bb in range(Pk + 1)])
    Sk = Pk + P + 1
    Kh, Kr = [], []
    epsk = arb(0)
    for i0 in range(0, H, 4096):
        rows = []
        for i in range(i0, min(H, i0 + 4096)):
            if i == 0:
                rows.append([zero] * (Pk + 1))
                continue
            ci = R(Fraction(2 * i + 1, 2))
            base = hmg * acb(ci) ** (-gamma)
            inv = 1 / ci
            row = []
            p_ = acb(1)
            for k in range(Pk + 1):
                row.append(base * bs[k] * p_)
                p_ = p_ * inv
            rows.append(row)
            q = arb(1) / (2 * i + 1)
            epsk += ci ** (-a) * q ** (Pk + 1) / (1 - q)  # sum over kernel cells of relative remainders
        mono = rows_times(rows, shiftT)
        monor = rows_times([mono[r * (Pk + 1):(r + 1) * (Pk + 1)] for r in range(len(rows))], revK)
        for r_ in range(len(rows)):
            mrow = mono[r_ * (Pk + 1):(r_ + 1) * (Pk + 1)]
            rrow = monor[r_ * (Pk + 1):(r_ + 1) * (Pk + 1)]
            Kh.extend([mrow[k] * math.factorial(k) for k in range(Pk + 1)] + [zero] * (Sk - Pk - 1))
            Kr.extend([rrow[k] * math.factorial(k) for k in range(Pk + 1)] + [zero] * (Sk - Pk - 1))
    fb = [math.factorial(b) for b in range(P + 1)]
    gh, gr = [], []
    for m in range(H):
        gh.extend([Crows[m][b] * fb[b] for b in range(P + 1)] + [zero] * (Sk - P - 1))
        gr.extend([Crev[m][b] * fb[b] for b in range(P + 1)] + [zero] * (Sk - P - 1))
    UK = (acb_poly(Kh) * acb_poly(gh)).coeffs()
    log("denominator: kernel U product")
    VK = (acb_poly(Kr) * acb_poly(gr)).coeffs()
    log("denominator: kernel V product")
    del Kh, Kr, gh, gr
    UK += [zero] * max(0, H * Sk - len(UK))
    VK += [zero] * max(0, H * Sk - len(VK))
    # kernel-approximation error on p~ (uniform in t): h^{1-a} ||rho~|| sum_i eps_i
    E_kern = up(h1a * res["rho_norm"] * epsk)
    # ---- exact kernel cell 0 pieces
    Bb = [acb(math.factorial(b)) * b1.gamma() / (b + 1 + b1).gamma() for b in range(P + 1)]
    # Poly_b(theta) = sum_e binom(b,e) (-theta)^{b-e} / (e+1-gamma)   (in powers of theta)
    PolyB = []
    for b in range(P + 1):
        cf = [acb(0)] * (b + 1)
        for e in range(b + 1):
            cf[b - e] += math.comb(b, e) * (-1) ** (b - e) / (e + b1)
        PolyB.append(acb_poly(cf))
    xpoly = acb_poly([acb(half), acb(1)])  # theta = 1/2 + x
    C = Pk + P + 1
    DK = C  # max degree in theta of U/V part is C (theta^{c+1}, c <= C-1)
    # centered matrices: rows = source index, columns = power of x
    LT = []
    for c in range(C):
        f1 = math.factorial(c + 1)
        LT.append([h * R(Fraction(math.comb(c + 1, k), 2 ** (c + 1 - k))) / f1 if k <= c + 1 else arb(0)
                   for k in range(DK + 1)])
    for c in range(C):
        f1 = math.factorial(c + 1)
        LT.append([(-1) ** k * h * R(Fraction(math.comb(c + 1, k), 2 ** (c + 1 - k))) / f1 if k <= c + 1
                   else arb(0) for k in range(DK + 1)])
    for b in range(P + 1):
        cc = PolyB[b](xpoly).coeffs()
        LT.append([h1g * cc[k] if k < len(cc) else acb(0) for k in range(DK + 1)])
    LT = mat_from(LT)
    bsT = [acb(1)]
    for k in range(1, 61):
        bsT.append(bsT[-1] * (-gamma - k + 1) / k)  # binom(-gamma, k)
    # ---- loop over cells
    total = arb(0)
    total_up = arb(0)
    err_total = arb(0)
    n_geo = 0
    worst_loss = []
    for m0 in range(1, H, 2048):
        rows = []
        ms = list(range(m0, min(H, m0 + 2048)))
        for m in ms:
            rows.append(UK[m * Sk:m * Sk + C] + VK[(m - 1) * Sk:(m - 1) * Sk + C] + Crev[m - 1])
        ent = rows_times(rows, LT)
        for r_, m in enumerate(ms):
            cen = list(ent[r_ * (DK + 1):(r_ + 1) * (DK + 1)])
            # t^-gamma part: h^-gamma (m + 1/2 + x)^-gamma
            cm = R(Fraction(2 * m + 1, 2))
            q = arb(1) / (2 * m + 1)
            Km = max(4, min(60, int(math.ceil(60 / math.log2(2 * m + 1)))))
            base = hmg * acb(cm) ** (-gamma)
            p_ = acb(1)
            inv = 1 / cm
            coeffs = cen + [acb(0)] * max(0, Km + 1 - len(cen))
            for k in range(Km + 1):
                coeffs[k] += base * bsT[k] * p_
                p_ = p_ * inv
            tail_t = h ** (-a) * cm ** (-a) * q ** (Km + 1) / (1 - q)
            # truncate the polynomial: keep degree <= Kt where the dropped part is tiny
            Kt = len(coeffs) - 1
            amp = abs(coeffs[0])
            acc = arb(0)
            while Kt > 3:
                tv = abs(coeffs[Kt]) * half ** Kt
                if up(acc + tv) < amp * 1e-19:
                    acc += tv
                    Kt -= 1
                else:
                    break
            tail_trunc = up(acc)
            Ap = acb_poly(coeffs[:Kt + 1])
            # singular part S_m = theta^{1-gamma} sum_b Bb Delta_b theta^b, Delta = c_m - (-1)^b crev_{m-1}
            Delta = [Crows[m][b] - (-1) ** b * Crev[m - 1][b] for b in range(P + 1)]
            sig = up(h1a * sum((abs(Bb[b] * Delta[b]) for b in range(P + 1)), arb(0)))
            tcen = (m + half) * h
            w0 = h / (1 - tcen)
            ew = h / (1 - tcen)
            nw = wterms(ew)
            wser = acb_poly([acb(w0 * ew ** j) for j in range(nw)])
            wrem = up(w0 * (ew / 2) ** nw / (1 - ew / 2))
            wint = up(h / (1 - (m + 1) * h))  # >= int_cell w d theta
            err_cell = tail_t + tail_trunc + E_kern
            if sig * wint < budget / H:
                err_cell = err_cell + sig
                Af = np.array([complex(float(c.real.mid()), float(c.imag.mid())) for c in coeffs[:Kt + 1]])
                ucoef = fit_u_sym(Af, 8 if m < 16 else (6 if m < 256 else 3))
                bb = sqrt_lower_integral(Ap, wser, wrem, half, ucoef, wsup=up(w0 / (1 - ew / 2)))
                if bb is None:
                    raise ArithmeticError(f"u not positive on cell {m}")
                total += bb[0]
                total_up = None if (total_up is None or bb[1] is None) else total_up + bb[1]
            else:
                n_geo += 1
                kl, ku = knot_cell_lower(Ap, Bb, Delta, h1g, m, h, half, geo_J, gamma)
                total += kl
                total_up = None if (total_up is None or ku is None) else total_up + ku
            err_total += err_cell * wint
    # ---- cell 0: p~ = h^-gamma theta^-gamma q0(theta), q0 = 1 + h sum_b Bb c_{0,b} theta^{b+1}
    q0 = [acb(1)] + [h * Bb[b] * Crows[0][b] for b in range(P + 1)]
    c0l, c0u = cell0_lower(q0, a, h)
    total += c0l
    total_up = None if (total_up is None or c0u is None) else total_up + c0u
    D_low = lo(total - err_total)
    D_up = None if total_up is None else up(total_up + err_total)
    log("denominator: certified", n_geo=n_geo, D_low=D_low.str(15), err=up(err_total).str(4),
        E_kern=E_kern.str(4))
    return dict(D_low=D_low, D_up=D_up, err=up(err_total), n_geo=n_geo, E_kern=E_kern)


def knot_cell_lower(Ap, Bb, Delta, h1g, m, h, half, J, gamma):
    """Lower bound of int_0^1 |A(theta-1/2) + h^{1-gamma} theta^{1-gamma} S(theta)| w(theta) dtheta on
    a cell with a significant derivative jump at its left end, using the geometric pieces
    [2^-(j+1), 2^-j], j < J; the piece [0, 2^-J] is dropped (its contribution is >= 0)."""
    b1 = 1 - gamma
    Spoly = acb_poly([h1g * Bb[b] * Delta[b] for b in range(len(Delta))])  # S(theta), times theta^{1-gamma}
    Ks = 40
    bsr = [acb(1)]
    for k in range(1, Ks + 1):
        bsr.append(bsr[-1] * (b1 - k + 1) / k)  # binom(1-gamma, k)
    # |binom(1-gamma,k)| <= k+1 since |1-gamma| <= 2; tail sum_{k>Ks} (k+1) 3^-k, closed form
    x3 = arb(1) / 3
    tail_ser = x3 ** (Ks + 1) * ((Ks + 2) - (Ks + 1) * x3) / (1 - x3) ** 2
    total = arb(0)
    total_up = arb(0)
    for j in range(J):
        al = R(Fraction(1, 2 ** (j + 1)))
        be = R(Fraction(1, 2 ** j))
        thc = (al + be) / 2
        ell = be - al
        xp = acb_poly([acb(thc - half), acb(ell)])  # cell-centred x as a function of the piece variable
        th = acb_poly([acb(thc), acb(ell)])  # theta as a function of the piece variable
        Aj = Ap(xp)
        rr = ell / thc  # |ell x'/thc| <= 1/3 for |x'| <= 1/2
        tc = acb(thc) ** b1
        ser = acb_poly([tc * bsr[k] * rr ** k for k in range(Ks + 1)])
        Sj = Spoly(th)
        fpoly = Aj + ser * Sj
        err = up(abs(tc) * tail_ser * poly_absmax(Sj, half))  # |theta^{1-gamma} - ser| |S| on the piece
        tcen = (m + thc) * h
        w0 = h / (1 - tcen)
        ew = h * ell / (1 - tcen)
        nw = wterms(ew)
        wser = acb_poly([acb(w0 * ew ** k) for k in range(nw)])
        wrem = up(w0 * (ew / 2) ** nw / (1 - ew / 2))
        Af = np.array([complex(float(c.real.mid()), float(c.imag.mid())) for c in fpoly.coeffs()])
        wsup = up(h / (1 - (m + 1) * h))
        bb = sqrt_lower_integral(fpoly, wser, wrem, half, fit_u_sym(Af, 8), wsup=wsup)
        if bb is None:
            raise ArithmeticError(f"u not positive on knot cell {m} piece {j}")
        total += ell * (bb[0] - err * wsup)
        total_up = None if (total_up is None or bb[1] is None) else total_up + ell * (bb[1] + err * wsup)
    if total_up is not None:  # the dropped piece [0, 2^-J]: |p~| <= sup|A| + sup|S| there
        supc = poly_absmax(Ap, half) + sum((abs(c) for c in Spoly.coeffs()), arb(0))
        total_up = up(total_up + R(Fraction(1, 2 ** J)) * supc * up(h / (1 - (m + 1) * h)))
    return lo(total), total_up


def cell0_lower(q0, a, h):
    """Lower bound of int_0^1 |h^-gamma theta^-gamma q0(theta)| h/(1-h theta) d theta."""
    qp = acb_poly(q0)
    xs = (1 - np.cos((2 * np.arange(24) + 1) * np.pi / 48)) / 2
    qf = np.array([complex(float(c.real.mid()), float(c.imag.mid())) for c in q0])
    vals = np.abs(np.polyval(qf[::-1], xs))
    uc = np.polynomial.polynomial.polyfit(xs, 1.0 / vals, 6)
    u = acb_poly([acb(float(c)) for c in uc])
    u0 = u.coeffs()[0].real
    rest = sum((abs(c) for c in u.coeffs()[1:]), arb(0))
    # u(theta) >= u0 - sum |u_k| on [0,1]
    assert u0 - rest > 0, "u not positive on cell 0"
    X = qp * poly_conj(qp)
    f = X * u * acb(1.5) - X * X * u * u * u * acb(0.5)
    nw = wterms(2 * h)
    wser = acb_poly([acb(h * h ** j) for j in range(nw)])
    g = f * wser
    val = acb(0)
    for k, c in enumerate(g.coeffs()):
        val += c / (k + 1 - a)
    supf = sum((abs(c) for c in f.coeffs()), arb(0))
    wrem = h * h ** nw / (1 - h)
    lb = (val.real - supf * wrem / (1 - a)) * h ** (-a)
    # upper: sqrt(X) - f <= eta^2 (1+eta)(3+eta)/(2 u_min), eta = sup_[0,1] |u^2 X - 1|
    eta = sum((abs(c) for c in (u * u * X - acb(1)).coeffs()), arb(0))
    if not (eta < 1):
        return lo(lb), None
    ex = eta ** 2 * (1 + eta) * (3 + eta) / (2 * lo(u0 - rest))
    ub = (val.real + supf * wrem / (1 - a) + ex * h / (1 - h) / (1 - a)) * h ** (-a)
    return lo(lb), up(ub)


def exact_coefficients(parts, nodes, P):
    """Trial coefficients (powers of theta) from nodal values sum(parts), interpolation at the exact dyadic
    nodes done in Arb; the MIDPOINTS are taken as the (exact) trial coefficients.  This only defines the
    trial function; no proof step depends on how it was obtained."""
    th = [R(Fraction(float(x))) for x in nodes]
    Vm = mat_from([[t_ ** b for b in range(P + 1)] for t_ in th])
    ViT = Vm.inv().transpose()
    N = parts[0].shape[0]
    out = []
    for m0 in range(0, N, 4096):
        rows = []
        for m in range(m0, min(N, m0 + 4096)):
            rows.append([sum((exact_acb(p[m, l]) for p in parts), acb(0)) for l in range(P + 1)])
        ent = rows_times(rows, ViT)
        for r_ in range(len(rows)):
            out.append([z.mid() for z in ent[r_ * (P + 1):(r_ + 1) * (P + 1)]])
    return out


# ------------------------------------------------------------------------------------------------
# driver
# ------------------------------------------------------------------------------------------------
def verify(data, N, P, Pk=20, bits=256, verbose=True, refine=0):
    t0 = time.time()
    ctx.prec = bits
    logs = []

    def log(stage, **kw):
        rec = dict(stage=stage, seconds=round(time.time() - t0, 2), **{k: (str(v)) for k, v in kw.items()})
        logs.append(rec)
        if verbose:
            print(json.dumps(rec), file=sys.stderr, flush=True)

    pr = Profile(data)
    pr.check_grid(N)
    hy = hypotheses(pr)
    log("hypotheses", a=hy["a"].str(12), kappa=hy["kappa"].str(12))
    wit = SW.Witness(data["T"], [str(u) for u in pr.kn], [str(p) for p in pr.ph], N, P)
    log("witness", gmres=wit.gmres_iters, coll=f"{wit.coll_res:.2e}")
    parts = [wit.vals]
    for it in range(refine):
        ctx.prec = 128
        Crows = exact_coefficients(parts, wit.nodes, P)
        nres = residual_bounds(pr, hy, N, P, Crows, lambda *a, **k: None, nodes=wit.nodes)
        log("refine", it=it, max_nodal_R_over_t=f"{np.abs(nres).max():.3e}")
        parts.append(wit.correct(nres))
        ctx.prec = bits
    Crows = exact_coefficients(parts, wit.nodes, P)
    hsh = hashlib.sha256()
    for row in Crows:
        for z in row:
            hsh.update((z.real.mid().str(80) + z.imag.mid().str(80)).encode())
    trial_hash = hsh.hexdigest()
    res = residual_bounds(pr, hy, N, P, Crows, log)
    prop = propagation(hy, N, res)
    log("propagation", eps_inf=prop["eps_inf"].str(5), EN=prop["EN"].str(5), ED=prop["ED"].str(5))
    p1 = numerator(hy, N, P, res)
    log("numerator", p1=p1.str(20))
    den = denominator_lower(pr, hy, N, P, res, Pk, log)
    num_up = up(abs(p1) + prop["EN"])
    D = lo(den["D_low"] - prop["ED"])
    if not D > 0:
        raise ArithmeticError("denominator lower bound not positive")
    Fup = up(num_up / D)
    F_low = None
    if den["D_up"] is not None:
        F_low = lo((abs(p1) - prop["EN"]) / (den["D_up"] + prop["ED"]))
    Fq = Fraction(str(Fup.fmpq())) if hasattr(Fup, "fmpq") else None
    verdict = "PASS" if Fup < R(pr.Cq) else "FAIL"
    out = dict(verdict=verdict, C=str(pr.Cq), T=str(pr.Tq), N=N, P=P, Q=res["Q"], Pk=Pk, bits=bits, refine=refine,
               a=hy["a"].str(20), kappa=hy["kappa"].str(20), hypotheses=hy["checks"],
               residual_sup=res["sup"].str(6), residual_worst_cell=res["worst"], rho_norm=res["rho_norm"].str(6),
               eps_inf=prop["eps_inf"].str(6), EN=prop["EN"].str(6), ED=prop["ED"].str(6),
               EN_sup=prop["EN_sup"].str(4), EN_int=prop["EN_int"].str(4), ED_sup=prop["ED_sup"].str(4),
               ED_int=prop["ED_int"].str(4),
               p_tilde_1=p1.str(30), numerator_upper=num_up.str(25), denominator_trial_lower=den["D_low"].str(25),
               denominator_lower=D.str(25), denominator_err=den["err"].str(6), knot_cells=den["n_geo"],
               F_upper=Fup.str(25), F_lower=(F_low.str(25) if F_low is not None else None),
               enclosure_width=(up(Fup - F_low).str(6) if F_low is not None else None), trial_sha256=trial_hash, seconds=round(time.time() - t0, 1), log=logs)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("certificate")
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--P", type=int, default=6)
    ap.add_argument("--Pk", type=int, default=20)
    ap.add_argument("--bits", type=int, default=256)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--refine", type=int, default=0, help="witness refinement passes (Arb nodal residuals)")
    args = ap.parse_args()
    raw = open(args.certificate, "rb").read()
    try:
        out = verify(json.loads(raw), args.N, args.P, args.Pk, args.bits, not args.quiet, args.refine)
    except Exception as e:  # any failure is a refusal
        out = dict(verdict="REFUSED", reason=f"{type(e).__name__}: {e}")
    import resource
    out["max_rss_GB"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 3)
    out["certificate_sha256"] = hashlib.sha256(raw).hexdigest()
    out["checker_sha256"] = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
    print(json.dumps(out, indent=1))
    sys.exit(0 if out["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()
