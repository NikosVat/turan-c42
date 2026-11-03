"""
Fast floating-point evaluator of the limit F in Bakeas's Theorem A (SEARCH ONLY, NOT RIGOROUS).

    F(g) = |p(1)| / int_0^{1/2} |p(t)| dt/(1-t),

where p solves the homogeneous Volterra equation

    t p(t) = int_0^t c(u) p(t-u) du,   c(u) = 1 - g(u) on [0, 1/2],  c(u) = 1 on (1/2, 1],

normalised by p(t) ~ t^{-gamma} at 0, gamma = g(0).  (Equivalent to Theorem A's
p = t^{-gamma} + int (t-v)^{-gamma} rho(v) dv; see MATH.md section 3.)  The prescribed power sums are
s_k = g(k/n), k <= n/2, and the bound is limsup R_n <= max(sup|g|, F).

Numerics.  Write p(s) = s^{-gamma} q(s) (q(0) = 1, q is smooth near 0).  On the uniform grid s_j = j/N
the product c(t_i - s) q(s) is replaced by its linear interpolant on every cell and integrated exactly
against s^{-gamma} ("product trapezoidal rule").  The jump of c at u = 1/2 is handled with one-sided
limits, so N must be even.  The resulting lower-triangular system is solved
  * in JAX (dense matrix + triangular solve; differentiable; N up to ~4096), or
  * in numpy (O(N^2) forward recursion; any N; forward only).
The denominator uses the same product rule with weight s^{-a}, a = Re gamma, on |q(s)|/(1-s).
Discretisation error ~ A N^{-(2-a)} + B N^{-2}; `F_extrapolated` fits both terms over N, 2N, 4N.
Optimising F_N at a fixed N overfits this error: optimise the extrapolated value instead (optimize.py --extrap).

This file was written for this project (it is not copied from Bakeas's crosscheck/limit_direct.py,
although it uses the same product-integration idea, which is credited to that script).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable, Optional, Sequence

import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


# ----------------------------------------------------------------------------------------------
# Profiles
# ----------------------------------------------------------------------------------------------
@dataclass
class Profile:
    """g(u) = T * rho(u) * exp(i phi(u)) on [0, 1/2].

    kind = "pwlinear": phi, rho piecewise linear on `knots` (0 = u_0 < ... < u_m = 1/2).
    kind = "spline":   phi, rho natural cubic splines through the knot values.
    rho = None means rho == 1 (Bakeas's constant-modulus family).
    """

    T: float
    knots: np.ndarray
    phases: np.ndarray
    rho: Optional[np.ndarray] = None
    kind: str = "pwlinear"
    meta: dict = field(default_factory=dict)

    def g(self, u, phases=None, rho=None, T=None):
        """Evaluate g at points u (jax-traceable in phases / rho / T)."""
        phases = self.phases if phases is None else phases
        rho = self.rho if rho is None else rho
        T = self.T if T is None else T
        kn = jnp.asarray(self.knots)
        if self.kind == "pwlinear":
            phi = jnp.interp(u, kn, phases)
            r = 1.0 if rho is None else jnp.interp(u, kn, rho)
        elif self.kind == "spline":
            phi = natural_cubic_spline(kn, phases, u)
            r = 1.0 if rho is None else natural_cubic_spline(kn, rho, u)
        else:
            raise ValueError(self.kind)
        return T * r * jnp.exp(1j * phi)


def natural_cubic_spline(x, y, u):
    """Natural cubic spline through (x_k, y_k), evaluated at u (jax-differentiable in y)."""
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    n = x.shape[0]
    hx = jnp.diff(x)
    # tridiagonal system for second derivatives m_1..m_{n-2}; m_0 = m_{n-1} = 0
    A = jnp.zeros((n, n))
    A = A.at[0, 0].set(1.0).at[n - 1, n - 1].set(1.0)
    idx = jnp.arange(1, n - 1)
    A = A.at[idx, idx - 1].set(hx[:-1]).at[idx, idx].set(2 * (hx[:-1] + hx[1:])).at[idx, idx + 1].set(hx[1:])
    rhs = jnp.zeros(n).at[idx].set(6 * ((y[2:] - y[1:-1]) / hx[1:] - (y[1:-1] - y[:-2]) / hx[:-1]))
    m = jnp.linalg.solve(A, rhs)
    k = jnp.clip(jnp.searchsorted(x, u, side="right") - 1, 0, n - 2)
    x0, x1, y0, y1, m0, m1, hk = x[k], x[k + 1], y[k], y[k + 1], m[k], m[k + 1], hx[k]
    A1 = (x1 - u) / hk
    B1 = (u - x0) / hk
    return A1 * y0 + B1 * y1 + ((A1**3 - A1) * m0 + (B1**3 - B1) * m1) * hk**2 / 6


def _parse_number(s) -> float:
    if isinstance(s, (int, float)):
        return float(s)
    return float(Fraction(s))


def load_profile(path: str) -> Profile:
    """Load Bakeas's format (knots j/48) or Pinho's format (explicit rational 'breaks')."""
    d = json.load(open(path))
    T = _parse_number(d["T"])
    ph = np.array([_parse_number(x) for x in d["phases"]])
    if "breaks" in d:
        kn = np.array([_parse_number(x) for x in d["breaks"]])
    elif "knots" in d and isinstance(d["knots"], list):
        kn = np.array([_parse_number(x) for x in d["knots"]])
    else:  # Bakeas: "j/48, j = 0..24"
        m = len(ph) - 1
        kn = np.arange(m + 1) / (2.0 * m)
    rho = np.array([_parse_number(x) for x in d["rho"]]) if "rho" in d else None
    kind = d.get("kind", "pwlinear")
    assert abs(kn[0]) < 1e-15 and abs(kn[-1] - 0.5) < 1e-15 and len(kn) == len(ph)
    return Profile(T=T, knots=kn, phases=ph, rho=rho, kind=kind, meta={"path": path})


def harcos_profile(w=-0.43246 - 0.54237j) -> Profile:
    """Constant g = -w (Harcos): F should be 0.693676."""
    gam = -w
    T = abs(gam)
    ph = np.angle(gam)
    return Profile(T=T, knots=np.array([0.0, 0.5]), phases=np.array([ph, ph]))


# ----------------------------------------------------------------------------------------------
# Core: F from samples of g on the grid u_k = k/N, k = 0..N/2
# ----------------------------------------------------------------------------------------------
def _weights(xp, expo, s, h):
    """Hat-function moments of s^{-expo} on cells [s_j, s_{j+1}] (j = 0..len(s)-2).
    Returns (w0, w1): w0_j = int s^{-expo}(s_{j+1}-s)/h, w1_j = int s^{-expo}(s-s_j)/h."""
    e1 = 1.0 - expo
    e2 = 2.0 - expo
    sc = s.astype(complex) if xp is np else s.astype(jnp.complex128)
    pos = s > 0
    safe = xp.where(pos, sc, 1.0)
    P1 = xp.where(pos, safe**e1, 0.0)
    P2 = xp.where(pos, safe**e2, 0.0)
    M0 = (P1[1:] - P1[:-1]) / e1
    M1 = (P2[1:] - P2[:-1]) / e2
    w0 = (s[1:] * M0 - M1) / h
    w1 = (M1 - s[:-1] * M0) / h
    return w0, w1, P1


def _c_arrays(xp, gnodes, N):
    """c(u) = 1 - g(u) on [0,1/2], 1 after; left/right limits at u_k = k/N, k = 0..N."""
    H = N // 2
    c_half = 1.0 - gnodes  # k = 0..H
    ones = xp.ones(N - H, dtype=c_half.dtype)
    cL = xp.concatenate([c_half, ones])  # c((k/N)^-): at k = H the left limit 1 - g(1/2)
    cR = xp.concatenate([c_half[:H], xp.ones(N - H + 1, dtype=c_half.dtype)])  # right limit: 1 at k = H
    return cL, cR


def _denominator(xp, q, a, s, h, H):
    """int_0^{1/2} s^{-a} |q(s)|/(1-s) ds, product rule with linear interpolation of |q|/(1-s)."""
    f = xp.abs(q[: H + 1]) / (1.0 - s[: H + 1])
    w0, w1, _ = _weights(xp, a, s[: H + 1], h)
    return xp.real(xp.sum(w0 * f[:-1] + w1 * f[1:]))


def F_from_gnodes_jax(gnodes, N: int):
    """F for samples gnodes[k] = g(k/N), k = 0..N/2 (complex jax array). Differentiable. O(N^2) memory."""
    assert N % 2 == 0
    H = N // 2
    h = 1.0 / N
    gam = gnodes[0]
    a = jnp.real(gam)
    s = jnp.arange(N + 1) * h
    w0, w1, P1 = _weights(jnp, gam, s, h)
    cL, cR = _c_arrays(jnp, gnodes, N)
    # Matrix rows i = 1..N, columns j = 0..N
    i = jnp.arange(1, N + 1)[:, None]
    j = jnp.arange(0, N + 1)[None, :]
    k = jnp.clip(i - j, 0, N)
    w0full = jnp.concatenate([w0, jnp.zeros(1, w0.dtype)])  # w0_j, j = 0..N (w0_N unused)
    w1shift = jnp.concatenate([jnp.zeros(1, w1.dtype), w1])  # w1_{j-1}, j = 0..N
    A = jnp.where(j < i, w0full[None, :] * cL[k] + w1shift[None, :] * cR[k], 0.0)
    diag = w1[i[:, 0] - 1] * cR[0] - P1[1:]
    A = A + jnp.where(j == i, diag[:, None], 0.0)
    L = A[:, 1:]  # N x N lower triangular
    rhs = -A[:, 0]  # q_0 = 1
    qrest = jax.scipy.linalg.solve_triangular(L, rhs, lower=True)
    q = jnp.concatenate([jnp.ones(1, qrest.dtype), qrest])
    num = jnp.abs(q[N])
    den = _denominator(jnp, q, a, s, h, H)
    return num / den


def F_from_gnodes_np(gnodes, N: int, return_q: bool = False):
    """Same discretisation, numpy O(N^2) forward recursion (any N; not differentiable)."""
    assert N % 2 == 0
    gnodes = np.asarray(gnodes, dtype=complex)
    H = N // 2
    h = 1.0 / N
    gam = gnodes[0]
    a = gam.real
    s = np.arange(N + 1) * h
    w0, w1, P1 = _weights(np, gam, s, h)
    cL, cR = _c_arrays(np, gnodes, N)
    q = np.zeros(N + 1, complex)
    q[0] = 1.0
    # y0_j = w0_j q_j ; y1_l = w1_{l-1} q_l
    y0 = np.zeros(N + 1, complex)
    y1 = np.zeros(N + 1, complex)
    y0[0] = w0[0]
    for i in range(1, N + 1):
        # sum_{j=0}^{i-1} cL_{i-j} y0_j + sum_{l=1}^{i-1} cR_{i-l} y1_l
        S = np.dot(cL[i:0:-1], y0[:i])
        if i > 1:
            S += np.dot(cR[i - 1 : 0 : -1], y1[1:i])
        q[i] = S / (P1[i] - w1[i - 1] * cR[0])
        if i < N:
            y0[i] = w0[i] * q[i]
        y1[i] = w1[i - 1] * q[i]
    num = abs(q[N])
    den = _denominator(np, q, a, s, h, H)
    if return_q:
        return num / den, q
    return num / den


# ----------------------------------------------------------------------------------------------
# Exact gradient of the discrete F by the adjoint method (fast; any N)
# ----------------------------------------------------------------------------------------------
def _setup_np(gnodes, N):
    H = N // 2
    h = 1.0 / N
    gam = gnodes[0]
    s = np.arange(N + 1) * h
    w0, w1, P1 = _weights(np, gam, s, h)
    cL, cR = _c_arrays(np, gnodes, N)
    return H, h, gam, s, w0, w1, P1, cL, cR


def _adjoint_np(gnodes, q, Fval, N):
    """lambda solving L^T lambda = G, where dF = Re(sum_j G_j dq_j) (q_0 = 1 fixed)."""
    H, h, gam, s, w0, w1, P1, cL, cR = _setup_np(gnodes, N)
    a = gam.real
    wa0, wa1, _ = _weights(np, a, s[: H + 1], h)
    om = np.zeros(H + 1)
    om[:-1] += wa0.real
    om[1:] += wa1.real
    num = abs(q[N])
    den = num / Fval
    G = np.zeros(N + 1, complex)
    G[N] += np.conj(q[N]) / num / den
    jj = np.arange(1, H + 1)
    G[jj] -= Fval / den * om[jj] / (1 - s[jj]) * np.conj(q[jj]) / np.abs(q[jj])
    D = np.zeros(N + 1, complex)
    D[1:] = P1[1:] - w1[: N] * cR[0]
    lam = np.zeros(N + 1, complex)
    w0f = np.concatenate([w0, [0.0]])
    for j in range(N, 0, -1):
        if j < N:
            tail = lam[j + 1 :]  # i = j+1..N, k = i - j = 1..N-j
            SL = np.dot(cL[1 : N - j + 1], tail)
            SR = np.dot(cR[1 : N - j + 1], tail)
            lam[j] = (G[j] + w0f[j] * SL + w1[j - 1] * SR) / D[j]
        else:
            lam[j] = G[j] / D[j]
    return lam


def _causal_conv_jax(c, y):
    """(c * y)_i = sum_{j<=i} c_{i-j} y_j for i = 0..len-1 (FFT)."""
    n = c.shape[0]
    m = 1 << int(np.ceil(np.log2(2 * n)))
    return jnp.fft.ifft(jnp.fft.fft(c, m) * jnp.fft.fft(y, m))[:n]


def _phi_jax(gnodes, q, lam, N):
    """-Re(sum_i lam_i E_i(g; q)) + F(g; q) with q, lam frozen.  grad of this = total dF/dg."""
    H = N // 2
    h = 1.0 / N
    gam = gnodes[0]
    s = jnp.arange(N + 1) * h
    w0, w1, P1 = _weights(jnp, gam, s, h)
    cL, cR = _c_arrays(jnp, gnodes, N)
    y0 = jnp.concatenate([w0 * q[:N], jnp.zeros(1, q.dtype)])  # y0_j, j = 0..N-1
    y1 = jnp.concatenate([jnp.zeros(1, q.dtype), w1 * q[1:]])  # y1_l, l = 1..N
    cL0 = cL.at[0].set(0.0)
    cR0 = cR.at[0].set(0.0)
    conv = _causal_conv_jax(cL0, y0) + _causal_conv_jax(cR0, y1)
    D = jnp.concatenate([jnp.zeros(1, q.dtype), P1[1:] - w1 * cR[0]])
    E = D * q - conv
    contract = jnp.real(jnp.sum(lam[1:] * E[1:]))
    a = jnp.real(gam)
    den = _denominator(jnp, q, a, s, h, H)
    return -contract + jnp.abs(q[N]) / den


_phi_grad = jax.jit(jax.grad(_phi_jax, argnums=0, holomorphic=False), static_argnums=3)


def F_and_grad(param_to_gnodes: Callable, theta, N: int):
    """F and dF/dtheta for a jax-traceable map theta (real vector) -> gnodes (complex, length N/2+1)."""
    theta = jnp.asarray(theta)
    gfun = jax.jit(param_to_gnodes)
    gn = np.asarray(gfun(theta))
    Fval, q = F_from_gnodes_np(gn, N, return_q=True)
    lam = _adjoint_np(gn, q, Fval, N)
    qj = jnp.asarray(q)
    lj = jnp.asarray(lam)

    def phi(th):
        return _phi_jax(param_to_gnodes(th), qj, lj, N)

    grad = jax.grad(phi)(theta)
    return Fval, np.asarray(grad)


class Objective:
    """Reusable (jit-compiled once) F and dF/dtheta for theta -> gnodes; use for optimisation."""

    def __init__(self, param_to_gnodes: Callable, N: int):
        self.N = N
        self.gfun = jax.jit(param_to_gnodes)

        def phi(th, q, lam):
            return _phi_jax(param_to_gnodes(th), q, lam, N)

        self.grad_phi = jax.jit(jax.grad(phi, argnums=0))

    def F(self, theta):
        return F_from_gnodes_np(np.asarray(self.gfun(jnp.asarray(theta))), self.N)

    def value_and_grad(self, theta):
        theta = jnp.asarray(theta)
        gn = np.asarray(self.gfun(theta))
        Fval, q = F_from_gnodes_np(gn, self.N, return_q=True)
        lam = _adjoint_np(gn, q, Fval, self.N)
        g = self.grad_phi(theta, jnp.asarray(q), jnp.asarray(lam))
        return Fval, np.asarray(g)


def node_sensitivity(gnodes, N: int):
    """Returns (F, Gc) with Gc_k = dF/dRe g_k + i dF/dIm g_k  (node-wise gradient, k = 0..N/2)."""
    gn = jnp.asarray(gnodes)
    x = jnp.concatenate([jnp.real(gn), jnp.imag(gn)])
    m = gn.shape[0]
    Fval, gr = F_and_grad(lambda th: th[:m] + 1j * th[m:], x, N)
    return Fval, gr[:m] + 1j * gr[m:]


def gnodes_of(profile: Profile, N: int, **kw):
    u = jnp.arange(N // 2 + 1) / N
    return profile.g(u, **kw)


def F(profile: Profile, N: int = 2048, backend: str = "auto") -> float:
    if backend == "auto":
        backend = "jax" if N <= 4096 else "np"
    gn = gnodes_of(profile, N)
    if backend == "jax":
        return float(_F_jit(gn, N))
    return float(F_from_gnodes_np(np.asarray(gn), N))


_F_jit = jax.jit(F_from_gnodes_jax, static_argnums=1)


def F_extrapolated(profile: Profile, N0: int = 4608, levels: int = 3, backend: str = "auto"):
    """Two-exponent Richardson extrapolation over N0, 2N0, 4N0: fits F_N = F + A N^{-(2-a)} + B N^{-2}
    (a = Re gamma; the N^{-(2-a)} term comes from the (t-1/2)^{1-gamma} singularity of q caused by the jump
    of c at 1/2).  Validated on Harcos's closed form: error ~3e-9 at N0 = 4096.  Choose N0 so that all knots
    are grid points.  Returns (F_extrap, [(N, F_N)])."""
    Ns = [N0 * 2**l for l in range(levels)]
    Fs = np.array([F(profile, N, backend) for N in Ns])
    a = float(np.real(profile.g(0.0)))
    if levels == 3:
        A = np.array([[1.0, N ** -(2 - a), N ** -2.0] for N in Ns])
        Fx = float(np.linalg.solve(A, Fs)[0])
    else:
        Fx = float(Fs[-1] + (Fs[-1] - Fs[-2]) / (2 ** (2 - a) - 1))
    return Fx, list(zip(Ns, Fs.tolist()))


# ----------------------------------------------------------------------------------------------
# Differentiable objective helpers
# ----------------------------------------------------------------------------------------------
def make_F_of_params(profile: Profile, N: int, with_rho: bool = False):
    """Returns f(params) -> F, with params = (T, phases[, rho]) as one flat vector; jit + grad ready."""
    u = jnp.arange(N // 2 + 1) / N
    m = len(profile.phases)

    def f(x):
        T = x[0]
        ph = x[1 : 1 + m]
        rho = x[1 + m : 1 + 2 * m] if with_rho else None
        gn = profile.g(u, phases=ph, rho=rho, T=T)
        return F_from_gnodes_jax(gn, N)

    return jax.jit(f), jax.jit(jax.value_and_grad(f))


def pack(profile: Profile, with_rho: bool = False):
    parts = [np.array([profile.T]), np.asarray(profile.phases)]
    if with_rho:
        parts.append(np.ones(len(profile.phases)) if profile.rho is None else np.asarray(profile.rho))
    return np.concatenate(parts)


if __name__ == "__main__":
    import sys
    import time

    t0 = time.time()
    Fx, vals = F_extrapolated(harcos_profile(), 4096, 3)
    print(f"Harcos constant phase: F_N = {[(n, round(v, 10)) for n, v in vals]}  extrap = {Fx:.10f}"
          f"   (exact 0.6936763031)   [{time.time() - t0:.1f}s]")
    for p in sys.argv[1:]:
        prof = load_profile(p)
        # smallest N0 >= 4096 that is a multiple of every knot denominator (knots must be grid points)
        from math import lcm
        den = 2
        for x in json.load(open(p)).get("breaks", []):
            den = lcm(den, Fraction(x).denominator)
        if "breaks" not in json.load(open(p)):
            den = lcm(den, 2 * (len(prof.knots) - 1))
        N0 = den * max(1, -(-4096 // den))
        t0 = time.time()
        Fx, vals = F_extrapolated(prof, N0, 3)
        print(f"{p}: T = {prof.T}, F_N = {[(n, round(v, 12)) for n, v in vals]}, extrap F = {Fx:.10f} "
              f"[{time.time() - t0:.1f}s]")
