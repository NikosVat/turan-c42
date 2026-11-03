"""Floating-point witness for the spectral a-posteriori certifier (NOT a proof step).

Builds a trial function rho~ for Bakeas's equation (2)

    t rho(t) = d(t) + int_0^t d(u) rho(t-u) du,   0 < t <= 1,

as a discontinuous piecewise polynomial of degree P on N uniform cells of [0,1] (cell m = [m h, (m+1) h],
local variable theta in [0,1]).  The grid must contain every phase knot and 1/2.  Collocation at the
P+1 Chebyshev points of every cell; the collocation system is solved by GMRES.

Everything here is floating point.  `rigorous_check.py` re-does every proof step in Arb ball arithmetic;
this module only supplies the coefficients of rho~ (which are then treated as exact dyadic numbers).

Convolution of piecewise polynomials on a uniform grid (exact identity, used both here and in Arb):
for f, g piecewise polynomials with cell coefficients f_{i,a}, g_{j,b} (powers of theta),

  (f*g)(h(m+theta)) = h [ sum_c theta^{c+1}/(c+1)!   U_{m,c}  +  sum_c (1-theta)^{c+1}/(c+1)!  V_{m-1,c} ],
  U_{m,c} = sum_i sum_{a+b=c} (a! f_{i,a}) (b! g_{m-i,b}),
  V_{m,c} = sum_i sum_{a+b=c} (a! fr_{i,a}) (b! gr_{m-i,b}),

where fr, gr are the coefficients of the same cell polynomials in the reversed variable w = 1 - theta.
"""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
from scipy.fft import next_fast_len, fft, ifft
from scipy.sparse.linalg import LinearOperator, gmres


def cheb_nodes(P):
    l = np.arange(P + 1)
    return (1 - np.cos((2 * l + 1) * np.pi / (2 * (P + 1)))) / 2


def rev_matrix(n):
    """Rev[a, b] = (-1)^a binom(b, a):  sum_b c_b (1-w)^b = sum_a w^a (Rev c)_a."""
    M = np.zeros((n, n))
    for b in range(n):
        for a in range(b + 1):
            M[a, b] = (-1) ** a * math.comb(b, a)
    return M


class Grid:
    """Profile data on the uniform grid (floats)."""

    def __init__(self, T, knots, phases, N, Q):
        self.T, self.N, self.Q = float(T), N, Q
        self.h = 1.0 / N
        knots = [Fraction(k) for k in knots]
        assert N % 2 == 0 and all((k * N).denominator == 1 for k in knots), "grid must align knots and 1/2"
        kidx = [int(k * N) for k in knots]
        self.knot_pos = [float(k) for k in knots]
        ph = np.array([float(Fraction(p)) for p in phases])
        H = N // 2
        self.H = H
        self.gamma = self.T * np.exp(1j * ph[0])
        # per-cell phase at left end and slope
        beta = np.zeros(N)
        phl = np.zeros(N)
        for j in range(len(knots) - 1):
            s = float((Fraction(phases[j + 1]) - Fraction(phases[j])) / (knots[j + 1] - knots[j]))
            for i in range(kidx[j], kidx[j + 1]):
                beta[i] = s
                phl[i] = ph[j] + s * (i - kidx[j]) * self.h
        self.beta, self.phl = beta, phl
        # d Taylor coefficients per cell: d_i(theta) = gamma - g_i exp(i beta h theta)
        a = np.arange(Q + 1)
        f = np.zeros((N, Q + 1), complex)
        gl = self.T * np.exp(1j * phl[:H])
        z = 1j * beta[:H] * self.h
        fact = np.array([math.factorial(k) for k in a], float)
        f[:H] = -gl[:, None] * z[:, None] ** a[None, :] / fact[None, :]
        f[:H, 0] += self.gamma
        f[0, 0] = 0.0
        f[H:, 0] = self.gamma
        self.f = f
        # reversed: d_i(1-w) = gamma - g_i e^{z} e^{-z w}
        fr = np.zeros((N, Q + 1), complex)
        fr[:H] = -(gl * np.exp(z))[:, None] * (-z[:, None]) ** a[None, :] / fact[None, :]
        fr[:H, 0] += self.gamma
        fr[H:, 0] = self.gamma
        self.fr = fr
        self.fhat = f * fact[None, :]
        self.frhat = fr * fact[None, :]

    def d_exact(self, m, theta):
        """Exact d at t = h(m + theta) (vectorised over m, theta arrays of same shape)."""
        m = np.asarray(m)
        out = np.where(m >= self.H, self.gamma + 0j,
                       self.gamma - self.T * np.exp(1j * (self.phl[np.minimum(m, self.N - 1)]
                                                          + self.beta[np.minimum(m, self.N - 1)] * self.h * theta)))
        return out


class Convolver:
    """Float FFT (Kronecker) bivariate convolution of fixed d-coefficients with variable g-coefficients."""

    def __init__(self, fhat, P):
        N, Q1 = fhat.shape
        self.N, self.Q1, self.P1 = N, Q1, P + 1
        self.S = Q1 + P  # stride: a+b <= Q + P < S
        self.L = next_fast_len(2 * N * self.S)
        self.cache = {}

    def set_kernel(self, key, fhat):
        pk = np.zeros((self.N, self.S), complex)
        pk[:, :self.Q1] = fhat
        self.cache[key] = fft(pk.ravel(), self.L)

    def conv(self, key, ghat):
        pg = np.zeros((self.N, self.S), complex)
        pg[:, :self.P1] = ghat
        prod = ifft(self.cache[key] * fft(pg.ravel(), self.L))[: self.N * self.S]
        return prod.reshape(self.N, self.S)  # [m, c], c = 0..P+Q


class Witness:
    def __init__(self, T, knots, phases, N, P, Q=None, tol=1e-14, verbose=False):
        self.P = P
        if Q is None:
            # Taylor degree of d: remainder T (beta h)^{Q+1}/(Q+1)! < 1e-20
            bmax = 0.0
            kn = [Fraction(k) for k in knots]
            for j in range(len(kn) - 1):
                bmax = max(bmax, abs(float((Fraction(phases[j + 1]) - Fraction(phases[j])) / (kn[j + 1] - kn[j]))))
            z = bmax / N
            Q = 1
            while float(Fraction(T)) * z ** (Q + 1) / math.factorial(Q + 1) > 1e-20:
                Q += 1
        self.Q = Q
        self.grid = G = Grid(T, knots, phases, N, Q)
        self.N, self.h, self.H = N, G.h, G.H
        self.nodes = cheb_nodes(P)
        self.V = np.vander(self.nodes, P + 1, increasing=True)  # V[l,b] = theta_l^b
        self.Vinv = np.linalg.inv(self.V)
        self.Rev = rev_matrix(P + 1)
        self.factP = np.array([math.factorial(b) for b in range(P + 1)], float)
        self.cv = Convolver(G.fhat, P)
        self.cv.set_kernel("U", G.fhat)
        self.cv.set_kernel("V", G.frhat)
        C = P + Q + 1  # c = 0..P+Q
        cc = np.arange(C)
        fc1 = np.array([math.factorial(c + 1) for c in cc], float)
        th = self.nodes
        self.Bu = th[:, None] ** (cc[None, :] + 1) / fc1[None, :]  # [l, c]
        self.Bv = (1 - th)[:, None] ** (cc[None, :] + 1) / fc1[None, :]
        m = np.arange(N)
        self.tn = self.h * (m[:, None] + th[None, :])  # [m, l]
        self.dn = G.d_exact(np.repeat(m[:, None], P + 1, 1), np.repeat(th[None, :], N, 0))
        self.verbose = verbose
        self.solve(tol)

    def conv_at(self, coef, Bu, Bv):
        """(d*rho~) on every cell evaluated with basis tables Bu, Bv ([points, c])."""
        ghat = coef * self.factP[None, :]
        grhat = (coef @ self.Rev.T) * self.factP[None, :]
        U = self.cv.conv("U", ghat)
        Vv = self.cv.conv("V", grhat)
        out = U @ Bu.T
        out[1:] += Vv[:-1] @ Bv.T
        return self.h * out

    def solve(self, tol):
        N, P1 = self.N, self.P + 1

        def mv(x):
            X = x.reshape(N, P1)
            coef = X @ self.Vinv.T
            y = X - self.conv_at(coef, self.Bu, self.Bv) / self.tn
            return y.ravel()

        A = LinearOperator((N * P1, N * P1), matvec=mv, dtype=complex)
        b = (self.dn / self.tn).ravel()
        it = [0]

        def cb(r):
            it[0] += 1

        x, info = gmres(A, b, x0=b.copy(), rtol=tol, atol=0, restart=150, maxiter=4, callback=cb,
                        callback_type="pr_norm")
        if info != 0:
            raise RuntimeError(f"gmres info={info}")
        self.gmres_iters = it[0]
        X = x.reshape(N, P1)
        self.vals = X
        self.coef = X @ self.Vinv.T  # monomial coefficients per cell, powers of theta
        r = mv(x) - b
        self.coll_res = np.abs(r).max()

    def residual_float(self, npts=16):
        """Float residual R = d + d*rho~ - t rho~ divided by t, sampled at npts points per cell."""
        th = (np.arange(npts) + 0.5) / npts
        C = self.P + self.Q + 1
        cc = np.arange(C)
        fc1 = np.array([math.factorial(c + 1) for c in cc], float)
        Bu = th[:, None] ** (cc[None, :] + 1) / fc1[None, :]
        Bv = (1 - th)[:, None] ** (cc[None, :] + 1) / fc1[None, :]
        conv = self.conv_at(self.coef, Bu, Bv)
        m = np.arange(self.N)
        t = self.h * (m[:, None] + th[None, :])
        rho = self.coef @ (th[:, None] ** np.arange(self.P + 1)[None, :]).T
        dd = self.grid.d_exact(np.repeat(m[:, None], npts, 1), np.repeat(th[None, :], self.N, 0))
        R = dd + conv - t * rho
        return np.abs(R / t)


# ------------------------------------------------------------------------------------------------
# Float evaluation of F from the trial rho~ (validation of the formulas used in Arb; NOT rigorous)
# ------------------------------------------------------------------------------------------------
def binom_series(alpha, K):
    """binom(alpha, k), k = 0..K (complex alpha)."""
    b = np.zeros(K + 1, complex)
    b[0] = 1.0
    for k in range(1, K + 1):
        b[k] = b[k - 1] * (alpha - k + 1) / k
    return b


def shift_matrix(K):
    """S[j, k] = coefficient of sigma^j in (sigma - 1/2)^k."""
    S = np.zeros((K + 1, K + 1))
    for k in range(K + 1):
        for j in range(k + 1):
            S[j, k] = math.comb(k, j) * (-0.5) ** (k - j)
    return S


def kernel_cells_float(gamma, ncell, Pk):
    """Monomial coefficients (powers of sigma) of a degree-Pk approximation of (i+sigma)^(-gamma) on cells
    i = 1..ncell-1; row 0 is zero (cell 0 is treated exactly)."""
    bs = binom_series(-gamma, Pk)
    i = np.arange(1, ncell)
    c = i + 0.5
    cen = (c[:, None] ** (-gamma)) * bs[None, :] * c[:, None] ** (-np.arange(Pk + 1.0))[None, :]
    S = shift_matrix(Pk)
    out = np.zeros((ncell, Pk + 1), complex)
    out[1:] = cen @ S.T
    return out


def pair_conv(F, G, weights_fn):
    """sum_{a,b} w(a,b) * (F[:,a] * G[:,b]) (discrete causal convolution over cells), grouped by c=a+b.
    Returns array [m, c]."""
    n, A = F.shape
    B = G.shape[1]
    L = next_fast_len(2 * n)
    FF = fft(F, L, axis=0)
    GG = fft(G, L, axis=0)
    out = np.zeros((n, A + B - 1), complex)
    for c in range(A + B - 1):
        acc = np.zeros(L, complex)
        for a in range(max(0, c - B + 1), min(A, c + 1)):
            acc += weights_fn(a, c - a) * FF[:, a] * GG[:, c - a]
        out[:, c] = ifft(acc)[:n]
    return out


def beta_fn(x, y):
    from scipy.special import gamma as G
    return G(x) * G(y) / G(x + y)


def F_float(w: "Witness", Pk=22, npts=24, geo=40):
    """Float F from the trial: p~ = t^-gamma + int (t-v)^-gamma rho~(v) dv, exact cell formulas."""
    from scipy.special import gamma as Gfun
    G = w.grid
    gam, h, H, P, N = G.gamma, w.h, w.H, w.P, w.N
    a_re = gam.real
    c = w.coef  # [m, b]
    cr = c @ w.Rev.T  # reversed coefficients
    Bb = np.array([Gfun(b + 1) * Gfun(1 - gam) / Gfun(b + 2 - gam) for b in range(P + 1)])  # B(b+1,1-gam)
    # --- numerator p~(1)
    tot = 0.0 + 0j
    for k in range(N):
        n1 = N - k
        if n1 == 1:
            tot += (c[k] * Bb).sum()
        else:
            l = np.arange(0, 120)
            rf = np.ones(120, complex)
            for j in range(1, 120):
                rf[j] = rf[j - 1] * (gam + j - 1) / j
            wmom = np.array([(c[k] / (ll + np.arange(P + 1) + 1)).sum() for ll in l])
            tot += n1 ** (-gam) * (rf * n1 ** (-l.astype(float)) * wmom).sum()
    p1 = 1.0 + h ** (1 - gam) * tot
    # --- p~ on cells m < H as callable pieces
    Kc = kernel_cells_float(gam, H, Pk) * h ** (-gam)
    fa = np.array([math.factorial(a) for a in range(Pk + 1)], float)
    fb = np.array([math.factorial(b) for b in range(P + 1)], float)
    wfn = lambda a, b: fa[a] * fb[b]
    Kr = Kc @ rev_matrix(Pk + 1).T
    U = pair_conv(Kc, c[:H], wfn)
    V = pair_conv(Kr, cr[:H], wfn)
    C = Pk + P + 1
    fc1 = np.array([math.factorial(cc + 1) for cc in range(C)], float)
    # Gauss-Legendre on [0,1]
    xg, wg = np.polynomial.legendre.leggauss(npts)
    xg = (xg + 1) / 2
    wg = wg / 2
    # polyB_b(theta) = sum_e binom(b,e) (-theta)^{b-e} / (e+1-gam)
    def p_cell(m, th):
        th = np.asarray(th, float)
        val = (h * (m + th)) ** (-gam)
        cc = np.arange(C)
        val = val + h * ((th[:, None] ** (cc + 1) / fc1) @ U[m])
        if m >= 1:
            val = val + h * (((1 - th)[:, None] ** (cc + 1) / fc1) @ V[m - 1])
        # exact kernel cell 0
        sing = (c[m] * Bb)[None, :] * th[:, None] ** (np.arange(P + 1) + 1 - gam)[None, :]
        val = val + h ** (1 - gam) * sing.sum(1)
        if m >= 1:
            pv = np.zeros_like(th, complex)
            for b in range(P + 1):
                poly = sum(math.comb(b, e) * (-th) ** (b - e) / (e + 1 - gam) for e in range(b + 1))
                pv += cr[m - 1, b] * (poly - (-1) ** b * Bb[b] * th ** (b + 1 - gam))
            val = val + h ** (1 - gam) * pv
        return val

    knot_cells = set(int(round(k * N)) for k in G.knot_pos)
    den = 0.0
    for m in range(H):
        # geometric subdivision toward theta=0 on the first cells (singular terms), plain GL elsewhere
        if m < 4 or m in knot_cells:
            pieces = [(2.0 ** (-j - 1), 2.0 ** (-j)) for j in range(geo)]
        else:
            pieces = [(0.0, 1.0)]
        for (lo, hi) in pieces:
            th = lo + (hi - lo) * xg
            ww = wg * (hi - lo)
            t = h * (m + th)
            den += h * (np.abs(p_cell(m, th)) / (1 - t) * ww).sum()
    return abs(p1) / den, p1, den, p_cell
