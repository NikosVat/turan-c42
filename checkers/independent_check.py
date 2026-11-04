#!/usr/bin/env python3
"""
checkers/independent_check.py -- checker #2 for the C42 certificate format.

Written from NOTES.md, the "Certificate format" section of README.md and the
certificate file only (session S6), without reading checker #1 or the search code.

What it proves (if it prints PASS): with the profile (T, knots, phases) of the
certificate, Theorem A of NOTES.md applies (0 < a < 1, kappa < 1) and the limit
F = |p(1)| / int_0^{1/2} |p(t)| dt/(1-t) satisfies F < C, with T <= C < 1.
Hence limsup R_n <= max(T, F) <= C.

Method (see REVIEW.md, section "What checker #2 does"):
  * trial function rho~: piecewise polynomial of degree P on the uniform grid
    h = 1/N (N even, N*u_j integer), found by a float collocation solve with
    mixed-precision iterative refinement; coefficients are exact dyadics.
  * residual R = d + d*rho~ - t rho~ is computed EXACTLY in integer arithmetic
    (Lemma C1 of NOTES.md via Kronecker substitution, FLINT fmpz_poly
    products), with d replaced by a dyadic polynomial d^ and a rigorous bound on
    |d - d^|.  sup|R/t| on each (sub)cell: sum of moduli of Chebyshev
    coefficients, computed in exact integers.
  * a posteriori bounds: NOTES.md section 4 (Lemma 3 and Lemma 3'), taking the
    smaller of the sup and integrated forms.
  * p~(1): exact closed form, using the moments
        I_k(m) = int_0^1 (m+s)^(-gamma) s^k ds,
    (k+1-gamma) I_k = (m+1)^(1-gamma) - [k=0] m^(1-gamma) - k m I_{k-1};
    no series truncation.
  * denominator lower bound: for any polynomial z with |z| <= sqrt(3),
        |p| >= Re(p conj(z)) (3 - |z|^2)/2
    (since 0 <= x(3-x^2)/2 <= 1 for 0 <= x <= sqrt(3))
    with z a fit of p~/|p~| on each cell.  This is LINEAR in p~, so each cell
    integral is an exact combination of polynomial moments, of the moments
    I_k(m) of t^(-gamma), and of the moments of the singular parts theta^(1-gamma+b).
    p~ on a cell uses: kernel cell 0 exactly (Beta integrals), kernel cells >= 1
    by a degree-Pk Taylor polynomial about the cell centre (rigorous tail), and
    Lemma C1 (exact integer Kronecker products).
All real-number arithmetic other than the exact integer parts is Arb ball
arithmetic (python-flint).  Floats are used only to choose rho~ and the phase
fits z; neither choice affects validity.

Usage:
  python checkers/independent_check.py certificate/c42_profile_0.686360751.json \
      [--N 36864] [--P 12] [--out result.json]
Exit code: 0 PASS, 1 FAIL, 2 REFUSED (input or hypothesis violation).
"""
import argparse
import hashlib
import json
import math
import sys
import time
from fractions import Fraction
from math import comb, factorial

import numpy as np
import scipy.fft as sfft
import scipy.linalg as sla
from scipy.sparse.linalg import LinearOperator, gmres

import flint
from flint import acb, acb_poly, arb, arb_mat, ctx, fmpq, fmpz_mat, fmpz_poly

T0 = time.time()


def log(*a):
    print("[%7.1fs]" % (time.time() - T0), *a, file=sys.stderr, flush=True)


class Refused(Exception):
    pass


# ----------------------------------------------------------------------------
# small exact helpers
# ----------------------------------------------------------------------------

def farb(x):
    """Fraction -> arb ball (exact if representable, else rounded outward)."""
    x = Fraction(x)
    return arb(fmpq(x.numerator, x.denominator))


def round_arb(x, W):
    """Nearest integer to mid(x)*2^W.  Caller bounds |x - result*2^-W|."""
    m, e = x.mid().man_exp()
    m = int(m)
    s = int(e) + W
    if s >= 0:
        return m << s
    return (m + (1 << (-s - 1))) >> (-s)


def check_rad(x, W, what):
    if not (x.rad() < arb(2) ** (-W - 2)):
        raise RuntimeError("precision too low for %s (rad %s)" % (what, x.rad()))


def isqrt_up(v):
    """integer >= sqrt(v) for integer v >= 0"""
    r = math.isqrt(v)
    return r if r * r == v else r + 1


def shifted_cheb_int(n):
    """Integer monomial coefficients (in theta) of T*_k(theta) = T_k(2 theta - 1), k = 0..n."""
    T = [[1], [-1, 2]]
    for k in range(1, n):
        a, b = T[k], T[k - 1]
        nxt = [0] * (k + 2)
        for i, c in enumerate(a):          # 2(2 theta - 1) T_k
            nxt[i] -= 2 * c
            nxt[i + 1] += 4 * c
        for i, c in enumerate(b):
            nxt[i] -= c
        T.append(nxt)
    return [t + [0] * (n + 1 - len(t)) for t in T[: n + 1]]


def mono_to_cheb(n):
    """Exact rational matrix M (Fractions), (n+1)x(n+1): theta^j = sum_k M[j][k] T*_k(theta)."""
    Ts = shifted_cheb_int(n)
    # theta^j = sum_k M[j][k] T*_k ; solve the triangular system column by column
    M = [[Fraction(0)] * (n + 1) for _ in range(n + 1)]
    for j in range(n + 1):
        rem = [Fraction(0)] * (n + 1)
        rem[j] = Fraction(1)
        for k in range(j, -1, -1):
            coef = rem[k] / Ts[k][k]
            M[j][k] = coef
            if coef:
                for i in range(k + 1):
                    rem[i] -= coef * Ts[k][i]
        assert all(r == 0 for r in rem)
    return M


def sub_interval_matrix(n, lo, hi):
    """Exact matrix A: coefficients of q(lo + (hi-lo) tau) in tau from coefficients of q(theta)
    (row vector convention: newcoef = oldcoef * A)."""
    ell = hi - lo
    A = [[Fraction(0)] * (n + 1) for _ in range(n + 1)]
    for j in range(n + 1):           # theta^j = (lo + ell tau)^j
        for k in range(j + 1):
            A[j][k] = comb(j, k) * lo ** (j - k) * ell ** k
    return A


def matmul_frac(A, B):
    n, m, p = len(A), len(B), len(B[0])
    return [[sum(A[i][k] * B[k][j] for k in range(m)) for j in range(p)] for i in range(n)]


def frac_matrix_to_int(A):
    """Return (integer matrix, common denominator D) with A = intM / D."""
    D = 1
    for row in A:
        for x in row:
            D = D * x.denominator // math.gcd(D, x.denominator)
    return [[int(x * D) for x in row] for row in A], D


def fmat(rows, cols, flat):
    return fmpz_mat(rows, cols, [int(v) for v in flat])


def to_int_list(mat):
    return [int(v) for v in mat.entries()]


def cplx_kronecker_mul(Ar, Ai, Br, Bi, n):
    """(Ar + i Ai)(Br + i Bi) mod x^n with 3 real products; inputs fmpz_poly."""
    k1 = Br.mul_low(Ar + Ai, n)
    k2 = Ar.mul_low(Bi - Br, n)
    k3 = Ai.mul_low(Br + Bi, n)
    re = k1 - k3
    im = k1 + k2
    def pad(p):
        c = [int(v) for v in p.coeffs()]
        return c + [0] * (n - len(c))
    return pad(re), pad(im)


# ----------------------------------------------------------------------------
# certificate
# ----------------------------------------------------------------------------

def parse_exact(s, what):
    if not isinstance(s, str):
        raise Refused("%s must be a string (exact rational), got %r" % (what, s))
    try:
        return Fraction(s.strip())
    except (ValueError, ZeroDivisionError):
        raise Refused("cannot parse %s = %r as an exact rational" % (what, s))


def load_certificate(path):
    raw = open(path, "rb").read()
    cert = json.loads(raw)
    if cert.get("constant") != "C_42":
        raise Refused('"constant" must be "C_42"')
    for k in ("T", "C", "breaks", "phases"):
        if k not in cert:
            raise Refused("missing key %s" % k)
    T = parse_exact(cert["T"], "T")
    C = parse_exact(cert["C"], "C")
    u = [parse_exact(s, "breaks[%d]" % j) for j, s in enumerate(cert["breaks"])]
    psi = [parse_exact(s, "phases[%d]" % j) for j, s in enumerate(cert["phases"])]
    if len(u) < 2 or len(u) != len(psi):
        raise Refused("need K+1 >= 2 breaks and exactly one phase per break")
    if not (0 < T <= C < 1):
        raise Refused("need 0 < T <= C < 1 (T=%s, C=%s)" % (T, C))
    if u[0] != 0 or u[-1] != Fraction(1, 2):
        raise Refused("breaks must start at 0 and end at 1/2")
    if any(u[j + 1] <= u[j] for j in range(len(u) - 1)):
        raise Refused("breaks must be strictly increasing")
    return dict(T=T, C=C, u=u, psi=psi, sha256=hashlib.sha256(raw).hexdigest())


# ----------------------------------------------------------------------------
# main verification
# ----------------------------------------------------------------------------

class Checker:
    def __init__(self, cert, N=None, P=12, prec=448, Pk=30, J=6, Msub=64, nsub=16,
                 refine=2, Wr=50, Wd=128, Wk=128, Wz=52):
        self.cert = cert
        self.T, self.C, self.u, self.psi = cert["T"], cert["C"], cert["u"], cert["psi"]
        self.P, self.prec, self.Pk, self.J = P, prec, Pk, J
        self.Msub, self.nsub, self.refine = Msub, nsub, refine
        self.Wr, self.Wd, self.Wk, self.Wz = Wr, Wd, Wk, Wz
        lcd = 1
        for x in self.u:
            lcd = lcd * x.denominator // math.gcd(lcd, x.denominator)
        step = lcd * 2 // math.gcd(lcd, 2)
        if N is None:
            N = step * max(1, -(-36864 // step))
        if N % 2 or any((x * N).denominator != 1 for x in self.u):
            raise Refused("grid N=%d is not even or not aligned with the knots (lcd %d)" % (N, lcd))
        self.N, self.H = N, N // 2
        self.h = Fraction(1, N)
        self.out = dict(N=N, P=P, Pk=Pk, prec=prec, knots_lcd=lcd)
        ctx.prec = prec

    # ---------------- hypotheses of Theorem A ----------------
    def hypotheses(self):
        T, psi = self.T, self.psi
        Ta = farb(T)
        p0 = farb(psi[0])
        self.gam = Ta * acb(p0.cos(), p0.sin())          # gamma = T e^{i psi_0}
        self.a = self.gam.real                            # a = Re gamma
        if not (self.a > 0):
            raise Refused("a = Re gamma = T cos psi_0 is not certified > 0 (a = %s)" % self.a)
        if not (self.a < 1):
            raise Refused("a not certified < 1")
        # sup_{(0,1/2]} |d| = 2T sup |sin((phi - psi_0)/2)|, phi - psi_0 in [min_j, max_j](psi_j - psi_0)
        E = max(abs(x - psi[0]) for x in psi)
        Ea = farb(E)
        if Ea < arb.pi():
            dmax = 2 * Ta * (Ea / 2).sin()
        else:
            dmax = 2 * Ta
        # kappa_hat: exact rational >= sup |d| on (0,1]; |d| = T on (1/2, 1]
        if dmax < Ta:
            kap = T
        else:
            up = dmax.upper()
            m, e = up.man_exp()
            kap = max(T, Fraction(int(m)) * Fraction(2) ** int(e))
        if not (kap < 1):
            raise Refused("kappa = sup|d| not certified < 1 (bound %s)" % float(kap))
        self.kap = kap
        self.kapa = farb(kap)
        self.out.update(a=str(self.a), E=float(E), sup_d_first_half=str(dmax),
                        kappa_hat=str(kap))
        self.out["checks"] = {
            "0 < T <= C < 1 (exact)": True,
            "knots strictly increasing from 0 to 1/2, one phase per knot (exact)": True,
            "grid N even and N*u_j integer (exact)": True,
            "a = T cos(psi_0) > 0 (Arb)": True,
            "E = max|psi_j - psi_0| < pi (Arb)": bool(Ea < arb.pi()),
            "kappa_hat = max(T, sup_(0,1/2]|d|) < 1 (exact rational upper bound)": True}
        log("a = %s, E = %.6f, sup_(0,1/2]|d| <= %s, kappa_hat = %s" %
            (self.a.str(12), float(E), dmax.str(8), kap))

    # ---------------- d on cells, exact dyadic polynomial d^ ----------------
    def build_d(self):
        T, N, H, h = self.T, self.N, self.H, self.h
        u, psi = self.u, self.psi
        phis, betas = [], []
        j = 0
        for i in range(H):
            x = Fraction(i, N)
            while not (u[j] <= x < u[j + 1]):
                j += 1
            beta = (psi[j + 1] - psi[j]) / (u[j + 1] - u[j])
            phis.append(psi[j] + beta * (x - u[j]))
            betas.append(beta)
        Lam = max(abs(b) for b in betas)
        # Taylor degree Q: T (Lam h)^{Q+1}/(Q+1)! <= 1e-34
        Q = 1
        while T * (Lam * h) ** (Q + 1) / factorial(Q + 1) > Fraction(1, 10 ** 34):
            Q += 1
        self.Q = Q
        Wd = self.Wd
        Ta = farb(T)
        Fr = np.zeros((N, Q + 1), dtype=object)
        Fi = np.zeros((N, Q + 1), dtype=object)
        eps = [None] * N
        rnd = Fraction(2 * (Q + 1), 2 ** Wd)         # rounding: each coefficient |err| <= 2^{1-Wd}
        for i in range(H):
            ph = farb(phis[i])
            g = Ta * acb(ph.cos(), ph.sin())
            z = acb(0, farb(betas[i] * h))           # i beta h
            term = g
            for a in range(Q + 1):
                if a == 0:
                    c = self.gam - g
                else:
                    term = term * z / a
                    c = -term
                if i == 0 and a == 0:
                    Fr[i, a] = 0
                    Fi[i, a] = 0
                    continue
                check_rad(c.real, Wd, "d coeff")
                check_rad(c.imag, Wd, "d coeff")
                Fr[i, a] = round_arb(c.real, Wd)
                Fi[i, a] = round_arb(c.imag, Wd)
            tay = T * (abs(betas[i]) * h) ** (Q + 1) / factorial(Q + 1)
            eps[i] = tay + rnd
        check_rad(self.gam.real, Wd, "gamma")
        gr, gi = round_arb(self.gam.real, Wd), round_arb(self.gam.imag, Wd)
        for i in range(H, N):
            Fr[i, 0], Fi[i, 0] = gr, gi
            eps[i] = Fraction(2, 2 ** Wd)
        # cell 0: constant term exactly 0, so |d - d^|(h theta) <= theta * eps0p
        self.eps0p = T * (abs(betas[0]) * h) ** (Q + 1) / factorial(Q + 1) + Fraction(2 * Q, 2 ** Wd)
        self.Fr, self.Fi, self.eps = Fr, Fi, eps
        self.eps_max = max(eps[1:] + [self.eps0p])
        self.betas = betas
        self.out.update(Q=Q, Lambda=float(Lam), eps_d_max=float(self.eps_max))
        log("d^ built: Q = %d, Lambda = %.3f, eps_max = %.3g" % (Q, float(Lam), float(self.eps_max)))

    # ---------------- float collocation solver (witness only) ----------------
    def setup_float(self):
        N, P, Q = self.N, self.P, self.Q
        S = Q + P + 1
        self.S = S
        h = 1.0 / N
        fd = (self.Fr.astype(float) + 1j * self.Fi.astype(float)) / 2.0 ** self.Wd
        self.fd = fd
        fact = np.array([float(factorial(k)) for k in range(S + 2)])
        RevQ = np.array([[comb(b, e) * (-1) ** e if e <= b else 0 for e in range(Q + 1)]
                         for b in range(Q + 1)], dtype=float)
        RevP = np.array([[comb(b, e) * (-1) ** e if e <= b else 0 for e in range(P + 1)]
                         for b in range(P + 1)], dtype=float)
        self.RevP_f = RevP
        Lfft = sfft.next_fast_len(2 * N * S)
        self.Lfft = Lfft
        Fk = np.zeros((N, S), complex)
        Fk[:, :Q + 1] = fd * fact[:Q + 1]
        Fkr = np.zeros((N, S), complex)
        Fkr[:, :Q + 1] = (fd @ RevQ) * fact[:Q + 1]
        self.FFd = sfft.fft(Fk.ravel(), Lfft, workers=-1)
        self.FFdr = sfft.fft(Fkr.ravel(), Lfft, workers=-1)
        j = np.arange(P + 1)
        th = (1 - np.cos((2 * j + 1) * np.pi / (2 * (P + 1)))) / 2
        self.th = th
        self.Vm = th[:, None] ** np.arange(P + 1)[None, :]
        self.Vlu = sla.lu_factor(self.Vm)          # backward-stable solves (monomial basis)
        self.EU = np.array([[th[q] ** (c + 1) / fact[c + 1] for c in range(S)] for q in range(P + 1)])
        self.EV = np.array([[(1 - th[q]) ** (c + 1) / fact[c + 1] for c in range(S)] for q in range(P + 1)])
        self.tnodes = h * (np.arange(N)[:, None] + th[None, :])
        self.fact = fact
        Vd = th[:, None] ** np.arange(Q + 1)[None, :]
        self.rhs0 = fd @ Vd.T
        self.Vfull = th[:, None] ** np.arange(S + 1)[None, :]

    def fconv(self, c):
        N, P, S, L = self.N, self.P, self.S, self.Lfft
        G = np.zeros((N, S), complex)
        G[:, :P + 1] = c * self.fact[:P + 1]
        Gr = np.zeros((N, S), complex)
        Gr[:, :P + 1] = (c @ self.RevP_f) * self.fact[:P + 1]
        U = sfft.ifft(self.FFd * sfft.fft(G.ravel(), L, workers=-1), workers=-1)[:N * S].reshape(N, S)
        V = sfft.ifft(self.FFdr * sfft.fft(Gr.ravel(), L, workers=-1), workers=-1)[:N * S].reshape(N, S)
        Vs = np.vstack([np.zeros((1, S), complex), V[:-1]])
        return (U @ self.EU.T + Vs @ self.EV.T) / N

    def c_from_y(self, y):
        return sla.lu_solve(self.Vlu, (y / self.tnodes).T).T

    def fsolve(self, rhs, rtol, maxiter=20):
        N, P = self.N, self.P
        n = N * (P + 1)

        def mv(yf):
            y = yf.reshape(N, P + 1)
            return (y - self.fconv(self.c_from_y(y))).ravel()
        A = LinearOperator((n, n), matvec=mv, dtype=complex)
        it = [0]

        def cb(_):
            it[0] += 1
        y, info = gmres(A, rhs.ravel(), rtol=rtol, atol=0.0, restart=80, maxiter=maxiter,
                        callback=cb, callback_type="pr_norm")
        log("  GMRES: info=%d, inner iterations=%d" % (info, it[0]))
        return self.c_from_y(y.reshape(N, P + 1))

    # ---------------- exact residual ----------------
    def exact_residual(self):
        """Returns (Rre, Rim) fmpz_mat N x (S+1) and the scale: R^_m(theta) = sum_n (Rre+iRim)[m,n] theta^n / scale."""
        N, P, Q, S = self.N, self.P, self.Q, self.S
        Wr, Wd = self.Wr, self.Wd
        n = N * S
        fact = [factorial(k) for k in range(S + 2)]
        Cr = self.Cr.astype(object)
        Ci = self.Ci.astype(object)
        # Kronecker vectors
        def kron(Mr, Mi, deg):
            ar = np.zeros((N, S), dtype=object)
            ai = np.zeros((N, S), dtype=object)
            fa = np.array(fact[:deg + 1], dtype=object)
            ar[:, :deg + 1] = Mr * fa
            ai[:, :deg + 1] = Mi * fa
            return fmpz_poly([int(v) for v in ar.ravel()]), fmpz_poly([int(v) for v in ai.ravel()])
        RevQ = fmat(Q + 1, Q + 1, [comb(b, e) * (-1) ** e if e <= b else 0
                                     for b in range(Q + 1) for e in range(Q + 1)])
        RevP = fmat(P + 1, P + 1, [comb(b, e) * (-1) ** e if e <= b else 0
                                     for b in range(P + 1) for e in range(P + 1)])
        def rev(Mr, Mi, Rv, deg):
            r = fmat(N, deg + 1, Mr.ravel()) * Rv
            i = fmat(N, deg + 1, Mi.ravel()) * Rv
            return (np.array(to_int_list(r), dtype=object).reshape(N, deg + 1),
                    np.array(to_int_list(i), dtype=object).reshape(N, deg + 1))
        Fr_, Fi_ = kron(self.Fr, self.Fi, Q)
        Gr_, Gi_ = kron(Cr, Ci, P)
        Ure, Uim = cplx_kronecker_mul(Fr_, Fi_, Gr_, Gi_, n)
        Fvr, Fvi = rev(self.Fr, self.Fi, RevQ, Q)
        Cvr, Cvi = rev(Cr, Ci, RevP, P)
        self.Cvr, self.Cvi = Cvr, Cvi
        Fr_, Fi_ = kron(Fvr, Fvi, Q)
        Gr_, Gi_ = kron(Cvr, Cvi, P)
        Vre, Vim = cplx_kronecker_mul(Fr_, Fi_, Gr_, Gi_, n)
        L = factorial(S)
        LTU = [[0] * (S + 1) for _ in range(S)]
        LTV = [[0] * (S + 1) for _ in range(S)]
        for c in range(S):
            LTU[c][c + 1] = L // fact[c + 1]
            for k in range(c + 2):
                LTV[c][k] = L // fact[c + 1] * comb(c + 1, k) * (-1) ** k
        LTU = fmat(S, S + 1, sum(LTU, []))
        LTV = fmat(S, S + 1, sum(LTV, []))
        mcol = np.arange(N, dtype=object)[:, None]
        out = []
        for (U, V, Fx, Cx) in ((Ure, Vre, self.Fr, Cr), (Uim, Vim, self.Fi, Ci)):
            Rest = np.zeros((N, S + 1), dtype=object)
            Rest[:, :Q + 1] += Fx * (L * 2 ** Wr * N)
            Rest[:, :P + 1] -= mcol * Cx * (L * 2 ** Wd)
            Rest[:, 1:P + 2] -= Cx * (L * 2 ** Wd)
            Um = fmat(N, S, U)
            Vm = fmat(N, S, [0] * S + V[: (N - 1) * S])
            out.append(Um * LTU + Vm * LTV + fmat(N, S + 1, Rest.ravel()))
        scale = 2 ** (Wd + Wr) * N * L
        return out[0], out[1], scale

    def residual_nodes_float(self, Rre, Rim, scale):
        N, S = self.N, self.S
        rr = np.array([v / scale for v in to_int_list(Rre)]).reshape(N, S + 1)
        ri = np.array([v / scale for v in to_int_list(Rim)]).reshape(N, S + 1)
        R = rr + 1j * ri
        return R @ self.Vfull.T, R

    # ---------------- witness ----------------
    def witness(self):
        self.setup_float()
        log("float solve: N=%d, P=%d, Q=%d, stride S=%d, FFT length %d" %
            (self.N, self.P, self.Q, self.S, self.Lfft))
        c = self.fsolve(self.rhs0, 1e-14)
        W = 2.0 ** self.Wr
        self.Cr = np.rint(c.real * W).astype(np.int64)
        self.Ci = np.rint(c.imag * W).astype(np.int64)
        hist = []
        for it in range(self.refine + 1):
            log("exact residual (pass %d)" % it)
            Rre, Rim, scale = self.exact_residual()
            Rn, Rcoef = self.residual_nodes_float(Rre, Rim, scale)
            tn = self.tnodes
            col = float(np.max(np.abs(Rn) / tn))
            log("  collocation max|R/t| at the nodes (float) = %.3e" % col)
            hist.append(col)
            if it == self.refine:
                break
            dc = self.fsolve(Rn, 1e-9, maxiter=3)
            self.Cr += np.rint(dc.real * W).astype(np.int64)
            self.Ci += np.rint(dc.imag * W).astype(np.int64)
        self.Rre, self.Rim, self.Rscale = Rre, Rim, scale
        self.out["collocation_residual_history"] = hist

    # ---------------- rigorous residual bounds r ----------------
    def residual_bounds(self):
        N, P, S, h = self.N, self.P, self.S, self.h
        Ds = S  # residual degree
        # || rho~ ||_inf
        M = mono_to_cheb(P)
        Mi, den = frac_matrix_to_int(M)
        Mi = fmat(P + 1, P + 1, sum(Mi, []))
        chr_ = to_int_list(fmat(N, P + 1, self.Cr.ravel()) * Mi)
        chi_ = to_int_list(fmat(N, P + 1, self.Ci.ravel()) * Mi)
        best = 0
        for m in range(N):
            s = 0
            for k in range(P + 1):
                q = m * (P + 1) + k
                s += isqrt_up(chr_[q] ** 2 + chi_[q] ** 2)
            best = max(best, s)
        self.rho_norm = Fraction(best, den * 2 ** self.Wr)
        log("||rho~||_inf <= %.6f" % float(self.rho_norm))
        # whole-cell Chebyshev bound of R^_m
        M = mono_to_cheb(Ds)
        Mi, den = frac_matrix_to_int(M)
        Mi = fmat(Ds + 1, Ds + 1, sum(Mi, []))
        cr = to_int_list(self.Rre * Mi)
        ci = to_int_list(self.Rim * Mi)
        sc = self.Rscale * den
        Rb = []
        for m in range(N):
            s = 0
            base = m * (Ds + 1)
            for k in range(Ds + 1):
                s += isqrt_up(cr[base + k] ** 2 + ci[base + k] ** 2)
            Rb.append(Fraction(s, sc))
        # sub-interval bounds for small m (and the divided polynomial for m = 0)
        Msub, ns = min(self.Msub, N), self.nsub
        rre = to_int_list(self.Rre)
        rim = to_int_list(self.Rim)
        row = lambda L, m: L[m * (Ds + 1):(m + 1) * (Ds + 1)]
        r0re, r0im = row(rre, 0), row(rim, 0)
        if r0re[0] != 0 or r0im[0] != 0:
            raise RuntimeError("R^_0(0) != 0")
        sub_rows_re = [r0re[1:] + [0]] + [row(rre, m) for m in range(1, Msub)]
        sub_rows_im = [r0im[1:] + [0]] + [row(rim, m) for m in range(1, Msub)]
        Ar = fmat(Msub, Ds + 1, sum(sub_rows_re, []))
        Ai = fmat(Msub, Ds + 1, sum(sub_rows_im, []))
        Msubb = [[None] * ns for _ in range(Msub)]
        for s in range(ns):
            A = matmul_frac(sub_interval_matrix(Ds, Fraction(s, ns), Fraction(s + 1, ns)), M)
            Ai_, den2 = frac_matrix_to_int(A)
            Ai_ = fmat(Ds + 1, Ds + 1, sum(Ai_, []))
            xr = to_int_list(Ar * Ai_)
            xi = to_int_list(Ai * Ai_)
            for m in range(Msub):
                ssum = sum(isqrt_up(xr[m * (Ds + 1) + k] ** 2 + xi[m * (Ds + 1) + k] ** 2)
                           for k in range(Ds + 1))
                Msubb[m][s] = Fraction(ssum, self.Rscale * den2)
        # assemble piecewise constant r on pieces [lo, hi] (in t units, Fractions)
        tail = self.eps_max * self.rho_norm      # | ((d - d^) * rho~)(t) | / t
        pieces = []                              # (lo, hi, r)
        for m in range(N):
            if m < Msub:
                for s in range(ns):
                    th0 = Fraction(s, ns)
                    lo, hi = h * (m + th0), h * (m + th0 + Fraction(1, ns))
                    if m == 0:
                        r = (Msubb[0][s] + self.eps0p) / h + tail
                    else:
                        r = (Msubb[m][s] + self.eps[m]) / lo + tail
                    pieces.append((lo, hi, r))
            else:
                lo, hi = h * m, h * (m + 1)
                pieces.append((lo, hi, (Rb[m] + self.eps[m]) / lo + tail))
        self.pieces = pieces
        rmax = max(p[2] for p in pieces)
        imax = max(range(len(pieces)), key=lambda q: pieces[q][2])
        self.out.update(rho_norm=float(self.rho_norm), sup_R_over_t=float(rmax),
                        worst_piece_t=[float(pieces[imax][0]), float(pieces[imax][1])])
        log("sup |R/t| <= %.4e (worst piece t in [%.3e, %.3e])" %
            (float(rmax), float(pieces[imax][0]), float(pieces[imax][1])))
        # first cells (diagnostic)
        self.out["r_first_cells"] = [float(max(p[2] for p in pieces[m * ns:(m + 1) * ns]))
                                     for m in range(min(8, Msub))]

    # ---------------- error propagation (NOTES section 4) ----------------
    def error_bounds(self):
        a, k = self.a, self.kapa
        one = arb(1)
        half = arb(1) / 2
        A1 = arb(0)
        Ahalf = arb(0)
        sN = arb(0)
        sD = arb(0)
        rmax = arb(0)
        oma, omk = one - a, one - k
        for (lo, hi, r) in self.pieces:
            ra = farb(r)
            loa, hia = farb(lo), farb(hi)
            iv = (hia ** omk - (loa ** omk if lo > 0 else arb(0))) / omk
            A1 += ra * iv
            top = (one - hia) ** oma if hi < 1 else arb(0)
            sN += ra * ((one - loa) ** oma - top) / oma
            if hi <= Fraction(1, 2):
                Ahalf += ra * iv
                top = (half - hia) ** (2 - a) if hi < Fraction(1, 2) else arb(0)
                sD += ra * ((half - loa) ** (2 - a) - top) / (2 - a)
            rmax = rmax.max(ra)
        B1 = k.gamma() * oma.gamma() / (k + oma).gamma()
        B2 = k.gamma() * (2 - a).gamma() / (k + 2 - a).gamma()
        EN_int = sN + k * A1 * B1
        ED_int = 2 / oma * (sD + k * Ahalf * half ** (k + 1 - a) * B2)
        eps_inf = rmax / omk
        EN_sup = eps_inf / oma
        ED_sup = 2 * eps_inf * half ** (2 - a) / (oma * (2 - a))
        self.EN = EN_int if EN_int.upper() < EN_sup.upper() else EN_sup
        self.ED = ED_int if ED_int.upper() < ED_sup.upper() else ED_sup
        self.out.update(EN_integrated=float(EN_int.upper()), EN_sup=float(EN_sup.upper()),
                        ED_integrated=float(ED_int.upper()), ED_sup=float(ED_sup.upper()),
                        A1=float(A1.upper()))
        log("EN: integrated %.4e, sup %.4e ; ED: integrated %.4e, sup %.4e" %
            (float(EN_int.upper()), float(EN_sup.upper()), float(ED_int.upper()), float(ED_sup.upper())))

    # ---------------- moments I_k(m) ----------------
    def setup_powers(self, upto):
        g1 = 1 - self.gam
        self.pw = [acb(0)] + [acb(m) ** g1 for m in range(1, upto + 2)]

    def I_moments(self, m, K):
        """I_k(m) = int_0^1 (m+s)^(-gamma) s^k ds, k = 0..K, rigorous balls."""
        g = self.gam
        pw1 = self.pw[m + 1]
        if m < 64:
            res = []
            prev = None
            for k in range(K + 1):
                num = pw1 - (self.pw[m] if k == 0 else 0)
                if k > 0:
                    num -= k * m * prev
                prev = num / (k + 1 - g)
                res.append(prev)
            return res
        # backward recurrence, stable for m >= 64: I_{k-1} = (pw1 - (k+1-g) I_k)/(k m)
        extra = int(math.ceil(self.prec / math.log2(m / 3.0))) + 4
        K0 = K + extra
        R = arb(m) ** (-self.a) / (K0 + 1)         # |I_k(m)| <= m^{-a}/(k+1)
        cur = acb(arb(0, R), arb(0, R))
        res = [None] * (K + 1)
        for k in range(K0, 0, -1):
            if k <= K:
                res[k] = cur
            cur = (pw1 - (k + 1 - g) * cur) / (k * m)
        res[0] = cur
        return res

    # ---------------- numerator p~(1) ----------------
    def numerator(self):
        N, P = self.N, self.P
        h = farb(self.h)
        Cvr, Cvi = self.Cvr, self.Cvi
        self.setup_powers(N + 1)
        tot = acb(0)
        for k in range(N):
            m = N - 1 - k
            I = self.I_moments(m, P)
            s = acb(0)
            for e in range(P + 1):
                s += acb(int(Cvr[k, e]), int(Cvi[k, e])) * I[e]
            tot += s
        p1 = 1 + h ** (1 - self.gam) * tot / arb(2) ** self.Wr
        self.p1 = p1
        self.out["p1"] = [str(p1.real), str(p1.imag)]
        log("p~(1) = %s" % p1.str(20))

    # ---------------- denominator lower bound ----------------
    def kernel_polys(self):
        """Kernel u^(-gamma) on kernel cells i = 1..H-1 as dyadic polynomials in sigma, degree Pk."""
        H, Pk, Wk = self.H, self.Pk, self.Wk
        g = self.gam
        ha = farb(self.h)
        hg = ha ** (-g)
        bins = [acb(1)]
        for l in range(Pk):
            bins.append(bins[-1] * (-g - l) / (l + 1))
        # shift matrix x = sigma - 1/2 :  sigma^j coeff = sum_l c_l comb(l,j) (-1/2)^(l-j)
        Sh = [[Fraction(comb(l, j)) * Fraction(-1, 2) ** (l - j) if j <= l else Fraction(0)
               for j in range(Pk + 1)] for l in range(Pk + 1)]
        Shm = arb_mat(Pk + 1, Pk + 1, [farb(x) for row in Sh for x in row])
        rows_re, rows_im = [], []
        for i in range(1, H):
            c = arb(2 * i + 1) / 2
            base = hg * acb(c) ** (-g)
            q = 1 / c
            cur = base
            rr, ri = [], []
            for l in range(Pk + 1):
                v = cur * bins[l]
                rr.append(v.real)
                ri.append(v.imag)
                cur = cur * q
            rows_re += rr
            rows_im += ri
        Kre = arb_mat(H - 1, Pk + 1, rows_re) * Shm
        Kim = arb_mat(H - 1, Pk + 1, rows_im) * Shm
        KR = np.zeros((H, Pk + 1), dtype=object)
        KI = np.zeros((H, Pk + 1), dtype=object)
        for i in range(1, H):
            for l in range(Pk + 1):
                x, y = Kre[i - 1, l], Kim[i - 1, l]
                check_rad(x, Wk, "kernel")
                check_rad(y, Wk, "kernel")
                KR[i, l] = round_arb(x, Wk)
                KI[i, l] = round_arb(y, Wk)
        # error bound sum_i eps^K_i
        a = self.a
        tot = arb(0)
        for i in range(1, H):
            q = arb(1) / (2 * i + 1)
            tot += (arb(2 * i + 1) / 2) ** (-a) * q ** (Pk + 1) / (1 - q)
        tot = ha ** (-a) * tot + farb(Fraction(2 * (Pk + 1) * (H - 1), 2 ** Wk))
        self.E_kern = ha * farb(self.rho_norm) * tot
        self.KR, self.KI = KR, KI
        self.out["E_kern"] = float(self.E_kern.upper())
        log("kernel polynomials built (Pk = %d), E_kern <= %.3e" % (Pk, float(self.E_kern.upper())))

    def kernel_convolution(self):
        """Integer matrices for h*[sum theta^{c+1}/(c+1)! U^K + sum (1-theta)^{c+1}/(c+1)! V^K_{m-1}]."""
        H, P, Pk = self.H, self.P, self.Pk
        SK = Pk + P + 1
        n = H * SK
        fact = [factorial(k) for k in range(SK + 2)]
        def kron(Mr, Mi, deg):
            ar = np.zeros((H, SK), dtype=object)
            ai = np.zeros((H, SK), dtype=object)
            fa = np.array(fact[:deg + 1], dtype=object)
            ar[:, :deg + 1] = Mr[:H] * fa
            ai[:, :deg + 1] = Mi[:H] * fa
            return fmpz_poly([int(v) for v in ar.ravel()]), fmpz_poly([int(v) for v in ai.ravel()])
        RevK = fmat(Pk + 1, Pk + 1, [comb(b, e) * (-1) ** e if e <= b else 0
                                       for b in range(Pk + 1) for e in range(Pk + 1)])
        KvR = np.array(to_int_list(fmat(H, Pk + 1, self.KR.ravel()) * RevK), dtype=object).reshape(H, Pk + 1)
        KvI = np.array(to_int_list(fmat(H, Pk + 1, self.KI.ravel()) * RevK), dtype=object).reshape(H, Pk + 1)
        Cr = self.Cr.astype(object)
        Ci = self.Ci.astype(object)
        a1, a2 = kron(self.KR, self.KI, Pk)
        b1, b2 = kron(Cr, Ci, P)
        Ure, Uim = cplx_kronecker_mul(a1, a2, b1, b2, n)
        a1, a2 = kron(KvR, KvI, Pk)
        b1, b2 = kron(self.Cvr, self.Cvi, P)
        Vre, Vim = cplx_kronecker_mul(a1, a2, b1, b2, n)
        L = factorial(SK)
        LTU = [[0] * (SK + 1) for _ in range(SK)]
        LTV = [[0] * (SK + 1) for _ in range(SK)]
        for c in range(SK):
            LTU[c][c + 1] = L // fact[c + 1]
            for k in range(c + 2):
                LTV[c][k] = L // fact[c + 1] * comb(c + 1, k) * (-1) ** k
        LTU = fmat(SK, SK + 1, sum(LTU, []))
        LTV = fmat(SK, SK + 1, sum(LTV, []))
        Gre = fmat(H, SK, Ure) * LTU + fmat(H, SK, [0] * SK + Vre[:(H - 1) * SK]) * LTV
        Gim = fmat(H, SK, Uim) * LTU + fmat(H, SK, [0] * SK + Vim[:(H - 1) * SK]) * LTV
        self.SK = SK
        self.Gre = to_int_list(Gre)
        self.Gim = to_int_list(Gim)
        self.Gscale = 2 ** (self.Wk + self.Wr) * self.N * L
        log("kernel convolution done (stride %d)" % SK)

    def ptilde_float(self, m, th):
        """Float evaluation of the cell representation of p~ at t = h(m+th), m >= 1 (for tests)."""
        P, DG = self.P, self.SK
        N = self.N
        hf = 1.0 / N
        gf = complex(float(self.gam.real), float(self.gam.imag))
        G = np.array([self.Gre[m * (DG + 1) + j] / self.Gscale + 1j * self.Gim[m * (DG + 1) + j] / self.Gscale
                      for j in range(DG + 1)])
        cv = (self.Cvr[m - 1].astype(float) + 1j * self.Cvi[m - 1].astype(float)) / 2.0 ** self.Wr
        c = (self.Cr[m].astype(float) + 1j * self.Ci[m].astype(float)) / 2.0 ** self.Wr
        val = np.polynomial.polynomial.polyval(th, G)
        poly = 0
        sing = 0
        B = 1 / (1 - gf)
        for b in range(P + 1):
            if b > 0:
                B = B * b / (b + 1 - gf)
            pb = sum(comb(b, e) * (-th) ** (b - e) / (e + 1 - gf) for e in range(b + 1))
            poly += cv[b] * pb
            sing += B * (c[b] - (-1) ** b * cv[b]) * th ** b
        val += hf ** (1 - gf) * (poly + sing * th ** (1 - gf))
        val += (hf * (m + th)) ** (-gf)
        return val

    def fit_degree(self, m):
        if m < 4:
            return 20
        if m < 16:
            return 14
        if m < 128:
            return 9
        if m < 1024:
            return 6
        return 4

    def phase_fit(self, vals_fn, kf, cells):
        """Fit z ~ p/|p| on [0,1] (Chebyshev basis, degree kf) for a batch of cells.
        vals_fn(theta) -> complex array (len(cells), len(theta)).  Returns exact dyadic
        Chebyshev coefficients as integer arrays (scale 2^Wz)."""
        nf = 2 * kf + 8
        x = np.cos((2 * np.arange(nf) + 1) * np.pi / (2 * nf))
        th = (x + 1) / 2
        p = vals_fn(th)
        uu = p / np.abs(p)
        Tm = np.polynomial.chebyshev.chebvander(x, kf)
        chi = np.linalg.lstsq(Tm, uu.T, rcond=None)[0]          # (kf+1, cells)
        W = 2.0 ** self.Wz
        return np.rint(chi.real * W).astype(np.int64), np.rint(chi.imag * W).astype(np.int64)

    def make_v(self, chr_, chi_, kf, Wcoef):
        """v = W conj(z) (3 - |z|^2)/2 as acb_poly in theta; certifies sup|z| <= sqrt(3)."""
        Wz = self.Wz
        sc = arb(2) ** (-Wz)
        sabs = arb(0)
        for n in range(kf + 1):
            sabs += abs(acb(int(chr_[n]), int(chi_[n])))
        sabs = sabs * sc
        if not (sabs * sabs < 3):
            raise RuntimeError("phase fit: cannot certify sup|z| <= sqrt(3)")
        Ts = self.Tstar[kf]
        zr = [sum(int(chr_[n]) * Ts[n][k] for n in range(kf + 1)) for k in range(kf + 1)]
        zi = [sum(int(chi_[n]) * Ts[n][k] for n in range(kf + 1)) for k in range(kf + 1)]
        z = acb_poly([acb(zr[k], zi[k]) * sc for k in range(kf + 1)])
        zc = acb_poly([acb(zr[k], -zi[k]) * sc for k in range(kf + 1)])
        X = z * zc
        v = zc * (3 - X) * acb_poly(Wcoef) * acb(arb(1) / 2)
        return v.coeffs()

    def denominator(self):
        H, P, Pk, J = self.H, self.P, self.Pk, self.J
        N = self.N
        g = self.gam
        a = self.a
        ha = farb(self.h)
        hf = 1.0 / N
        h1g = ha ** (1 - g)
        hmg = ha ** (-g)
        gf = complex(float(g.real), float(g.imag))
        self.Tstar = {k: [row[:k + 1] for row in shifted_cheb_int(k)] for k in (4, 6, 9, 14, 20, 24)}
        # Beta values B_b = B(b+1, 1-gamma) = b!/((1-g)(2-g)...(b+1-g))
        Bb = []
        cur = 1 / (1 - g)
        for b in range(P + 1):
            if b > 0:
                cur = cur * b / (b + 1 - g)
            Bb.append(cur)
        Bbf = np.array([complex(float(x.real), float(x.imag)) for x in Bb])
        Wr2 = arb(2) ** (-self.Wr)
        Cr, Ci = self.Cr, self.Ci
        Cvr, Cvi = self.Cvr, self.Cvi
        # ---- cell 0 ----
        q0 = [acb(1)] + [ha * Bb[b] * acb(int(Cr[0, b]), int(Ci[0, b])) * Wr2 for b in range(P + 1)]
        q0f = np.array([complex(float(x.real), float(x.imag)) for x in q0])
        kf = 9
        fr, fi = self.phase_fit(lambda th: np.polynomial.polynomial.polyval(th, q0f)[None, :], kf, [0])
        Wc = [ha ** j for j in range(J)]
        v0 = self.make_v(fr[:, 0], fi[:, 0], kf, Wc)
        s = acb(0)
        for j, qj in enumerate(q0):
            for k, vk in enumerate(v0):
                s += qj * vk / (j + k + 1 - a)
        L0 = ha ** (1 - a) * s.real
        q0abs = sum(abs(x) for x in q0)
        tail0 = ha ** (1 - a) * q0abs * ha ** J / (1 - ha) / (1 - a)
        Dlow = L0 - tail0
        self.out["denominator_cell0"] = float(L0.mid())
        # ---- cells m >= 1 ----
        SK = self.SK
        DG = SK                                   # degree of G
        Gs = self.Gscale
        groups = []
        m0 = 1
        for thr in (4, 16, 128, 1024, H):
            if m0 < min(thr, H):
                groups.append((m0, min(thr, H)))
                m0 = min(thr, H)
        Lsum = arb(0)
        tailsum = arb(0)
        Dest = float(L0.mid())
        rho_n = farb(self.rho_norm)
        for (ma, mb) in groups:
            kf = self.fit_degree(ma)
            K = 3 * kf + J - 1
            cells = list(range(ma, mb))
            nc = len(cells)
            # float data for the fit
            Gf = np.array([[self.Gre[m * (DG + 1) + j] / Gs + 1j * self.Gim[m * (DG + 1) + j] / Gs
                            for j in range(DG + 1)] for m in cells])
            # poly part from kernel cell 0, V-side: h^{1-g} sum_b cv_{m-1,b} Poly_b(theta)
            cvf = (Cvr[np.array(cells) - 1].astype(float) + 1j * Cvi[np.array(cells) - 1].astype(float)) / 2.0 ** self.Wr
            MPf = np.zeros((P + 1, P + 1), complex)
            for b in range(P + 1):
                for jj in range(b + 1):
                    MPf[b, jj] = comb(b, jj) * (-1) ** jj / (b - jj + 1 - gf)
            h1gf = hf ** (1 - gf)
            Gf[:, :P + 1] += h1gf * (cvf @ MPf)
            Delta = (Cr[cells].astype(object) - np.array([(-1) ** b for b in range(P + 1)], dtype=object)
                     * Cvr[np.array(cells) - 1].astype(object),
                     Ci[cells].astype(object) - np.array([(-1) ** b for b in range(P + 1)], dtype=object)
                     * Cvi[np.array(cells) - 1].astype(object))
            Df = (Delta[0].astype(float) + 1j * Delta[1].astype(float)) / 2.0 ** self.Wr
            mvec = np.array(cells, dtype=float)

            def vals(th):
                pw = th[None, :] ** np.arange(DG + 1)[:, None]          # (DG+1, nf)
                out = Gf @ pw
                out += np.exp(-gf * np.log(hf * (mvec[:, None] + th[None, :])))
                sing = (Df * Bbf[None, :]) @ (th[None, :] ** np.arange(P + 1)[:, None])
                out += h1gf * sing * th[None, :] ** (1 - gf)
                return out
            fr, fi = self.phase_fit(vals, kf, cells)
            # diagnostic only (float, not rigorous): Gauss-Legendre estimate of int |p~| w on these cells
            xg, wg = np.polynomial.legendre.leggauss(24)
            tg = (xg + 1) / 2
            pv = np.abs(vals(tg)) / (1 - hf * (mvec[:, None] + tg[None, :]))
            Dest += float(np.sum(pv * wg[None, :] / 2) * hf)
            # exact moments: G part
            Gre_m = arb_mat(nc, DG + 1, [arb(self.Gre[m * (DG + 1) + j]) for m in cells for j in range(DG + 1)])
            Gim_m = arb_mat(nc, DG + 1, [arb(self.Gim[m * (DG + 1) + j]) for m in cells for j in range(DG + 1)])
            Hil = arb_mat(DG + 1, K + 1, [arb(1) / (j + k + 1) for j in range(DG + 1) for k in range(K + 1)])
            mu_re = (Gre_m * Hil) * (arb(1) / Gs)
            mu_im = (Gim_m * Hil) * (arb(1) / Gs)
            # poly part (V-side of kernel cell 0): coefficients h^{1-g} * cv_{m-1} * MP, then Hilbert
            MPc = [[(comb(b, jj) * (-1) ** jj / (b - jj + 1 - g)) if jj <= b else acb(0)
                    for jj in range(P + 1)] for b in range(P + 1)]
            HilP = [[arb(1) / (jj + k + 1) for k in range(K + 1)] for jj in range(P + 1)]
            MPH = [[sum((MPc[b][jj] * HilP[jj][k] for jj in range(P + 1)), acb(0)) * h1g * Wr2
                    for k in range(K + 1)] for b in range(P + 1)]
            MPHr = arb_mat(P + 1, K + 1, [x.real for row in MPH for x in row])
            MPHi = arb_mat(P + 1, K + 1, [x.imag for row in MPH for x in row])
            CVr = arb_mat(nc, P + 1, [arb(int(Cvr[m - 1, b])) for m in cells for b in range(P + 1)])
            CVi = arb_mat(nc, P + 1, [arb(int(Cvi[m - 1, b])) for m in cells for b in range(P + 1)])
            mu_re = mu_re + CVr * MPHr - CVi * MPHi
            mu_im = mu_im + CVr * MPHi + CVi * MPHr
            # singular part: h^{1-g} sum_b B_b Delta_b /(b+k+2-g)
            MS = [[Bb[b] / (b + k + 2 - g) * h1g * Wr2 for k in range(K + 1)] for b in range(P + 1)]
            MSr = arb_mat(P + 1, K + 1, [x.real for row in MS for x in row])
            MSi = arb_mat(P + 1, K + 1, [x.imag for row in MS for x in row])
            DLr = arb_mat(nc, P + 1, [arb(int(x)) for x in Delta[0].ravel()])
            DLi = arb_mat(nc, P + 1, [arb(int(x)) for x in Delta[1].ravel()])
            mu_re = mu_re + DLr * MSr - DLi * MSi
            mu_im = mu_im + DLr * MSi + DLi * MSr
            for idx, m in enumerate(cells):
                tm = ha * m
                Wc = [ha ** j / (1 - tm) ** (j + 1) for j in range(J)]
                v = self.make_v(fr[:, idx], fi[:, idx], kf, Wc)
                I = self.I_moments(m, K)
                s = arb(0)
                for k in range(min(K + 1, len(v))):
                    mu = acb(mu_re[idx, k], mu_im[idx, k]) + hmg * I[k]
                    s += (v[k] * mu).real
                Lsum += ha * s
                tm1 = ha * (m + 1)
                pmax = (tm) ** (-a) + rho_n / (1 - a)
                tailsum += ha * pmax * (ha / (1 - tm)) ** J / (1 - tm1)
            log("  denominator cells %d..%d done (fit degree %d, v degree %d), running sum %s" %
                (ma, mb - 1, kf, K, (Lsum + L0).str(15)))
        Dlow = Dlow + Lsum - tailsum - self.E_kern   # sum_m h E_kern /(1-t_{m+1}) <= E_kern
        self.Dlow = Dlow
        self.out.update(D_low=str(Dlow), denominator_tails=float(tailsum.upper()),
                        D_float_estimate_minus_D_low=Dest - float(Dlow.mid()))
        log("float estimate of int|p~|w minus D~_low (cells >= 1; diagnostic): %.3e" % (Dest - float(Dlow.mid())))
        log("D~_low = %s" % Dlow.str(20))

    # ---------------- run ----------------
    def run(self):
        self.hypotheses()
        self.build_d()
        self.witness()
        self.residual_bounds()
        self.error_bounds()
        self.numerator()
        self.kernel_polys()
        self.kernel_convolution()
        self.denominator()
        num = abs(self.p1) + self.EN
        den = self.Dlow - self.ED
        if not (den > 0):
            raise RuntimeError("denominator not certified positive")
        self.out["checks"]["D_low - ED > 0 (Arb)"] = True
        self.out["checks"]["F_upper < C (Arb)"] = bool(num / den < farb(self.C))
        F = num / den
        Ca = farb(self.C)
        Fu = arb(F.upper())
        self.out.update(abs_p1_plus_EN=str(num), D_low_minus_ED=str(den), F_ball=str(F),
                        F_upper=float(F.upper()), F_upper_str=F.upper().str(25, radius=False)
                        if hasattr(F.upper(), "str") else str(F.upper()))
        passed = bool(F < Ca)
        self.out["margin_C_minus_F_upper"] = float((Ca - Fu).mid())
        self.out["verdict"] = "PASS" if passed else "FAIL"
        return passed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("certificate")
    ap.add_argument("--N", type=int, default=None, help="grid size (even multiple of the knot LCD)")
    ap.add_argument("--P", type=int, default=12, help="trial polynomial degree per cell")
    ap.add_argument("--Pk", type=int, default=30, help="kernel Taylor degree")
    ap.add_argument("--prec", type=int, default=448, help="Arb precision (bits)")
    ap.add_argument("--refine", type=int, default=2, help="iterative refinement passes")
    ap.add_argument("--out", default=None, help="write JSON result here")
    args = ap.parse_args()
    res = dict(checker="checkers/independent_check.py", certificate=args.certificate,
               python_flint=flint.__version__, numpy=np.__version__)
    try:
        res["checker_sha256"] = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
    except OSError:
        pass
    code = 2
    try:
        cert = load_certificate(args.certificate)
        res.update(certificate_sha256=cert["sha256"], T=str(cert["T"]), C=str(cert["C"]),
                   pieces=len(cert["u"]) - 1)
        log("certificate %s: T = %s, C = %s, %d pieces" %
            (args.certificate, cert["T"], cert["C"], len(cert["u"]) - 1))
        ch = Checker(cert, N=args.N, P=args.P, Pk=args.Pk, prec=args.prec, refine=args.refine)
        ok = ch.run()
        res.update(ch.out)
        code = 0 if ok else 1
    except Refused as e:
        res["verdict"] = "REFUSED"
        res["reason"] = str(e)
    res["runtime_s"] = round(time.time() - T0, 1)
    try:
        import resource
        res["peak_rss_GB"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 3)
    except Exception:
        pass
    txt = json.dumps(res, indent=1)
    if args.out:
        open(args.out, "w").write(txt + "\n")
    print(txt)
    v = res.get("verdict")
    if v == "PASS":
        print("PASS: F <= %s < C = %s" % (res.get("F_upper_str"), res.get("C")))
    elif v == "FAIL":
        print("FAIL: certified F_upper = %s is not < C = %s" % (res.get("F_upper_str"), res.get("C")))
    else:
        print("REFUSED: %s" % res.get("reason"))
    sys.exit(code)


if __name__ == "__main__":
    main()
