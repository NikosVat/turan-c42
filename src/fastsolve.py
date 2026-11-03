"""
O(N log^2 N) forward and adjoint solves for the discretisation of src/evaluator.py (SEARCH ONLY, float).

The product-trapezoid system of evaluator.F_from_gnodes_np is an "online convolution":

    D_i q_i = sum_{j<i} ( cL_{i-j} w0_j q_j + cR_{i-j} w1_{j-1} q_j ),   q_0 = 1,

and its adjoint (evaluator._adjoint_np) is the same recursion run backwards with the weights moved to the
output side.  Both are solved here by the standard divide-and-conquer (CDQ) scheme: solve the left half,
push its contribution to the right half with one FFT convolution per kernel, solve the right half; blocks
of size <= BASE are solved as dense lower-triangular systems.  Same arithmetic as the O(N^2) recursion up
to FFT round-off (validated in __main__ against evaluator.py to ~1e-14).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import scipy.fft as sfft
from scipy.linalg import solve_triangular

sys.path.insert(0, os.path.dirname(__file__))
import evaluator as ev  # noqa: E402

BASE = 128
WORKERS = int(os.environ.get("FFT_WORKERS", "1"))


def _online_solve(G, D, K, a, b):
    """Solve x_i = (G_i + sum_k b[k,i] * sum_{j<i} K[k][i-j] a[k,j] x_j) / D_i, i = 0..n-1.

    K: (r, n) kernels (only offsets >= 1 are used), a, b: (r, n) weights.  Returns x (complex, n)."""
    n = G.shape[0]
    r = K.shape[0]
    x = np.zeros(n, complex)
    E = np.zeros((r, n), complex)  # external accumulated sums  sum_{j < block start} K[i-j] a_j x_j

    # precompute small-block Toeplitz index
    def base(lo, hi):
        m = hi - lo
        ii = np.arange(m)[:, None]
        jj = np.arange(m)[None, :]
        off = ii - jj
        mask = off > 0
        offc = np.where(mask, off, 0)
        Mt = np.zeros((m, m), complex)
        for k in range(r):
            Mt -= np.where(mask, K[k][offc], 0.0) * a[k, lo:hi][None, :] * b[k, lo:hi][:, None]
        Mt[np.arange(m), np.arange(m)] += D[lo:hi]
        rhs = G[lo:hi] + np.sum(b[:, lo:hi] * E[:, lo:hi], axis=0)
        x[lo:hi] = solve_triangular(Mt, rhs, lower=True, check_finite=False)

    def rec(lo, hi):
        if hi - lo <= BASE:
            base(lo, hi)
            return
        mid = (lo + hi) // 2
        rec(lo, mid)
        # contributions of j in [lo, mid) to i in [mid, hi): offsets i - j in [1, hi - lo - 1]
        L = hi - lo
        nfft = sfft.next_fast_len(L + (mid - lo), real=False)
        for k in range(r):
            y = a[k, lo:mid] * x[lo:mid]
            conv = sfft.ifft(sfft.fft(y, nfft, workers=WORKERS) * sfft.fft(K[k][:L], nfft, workers=WORKERS),
                             workers=WORKERS)
            # conv[t] = sum_j y_{j} K[t - j], index t = i - lo
            E[k, mid:hi] += conv[mid - lo : L]
        rec(mid, hi)

    rec(0, n)
    return x


def forward(gnodes, N):
    """Returns (F, q) with the same discretisation as evaluator.F_from_gnodes_np."""
    gnodes = np.asarray(gnodes, dtype=complex)
    H, h, gam, s, w0, w1, P1, cL, cR = ev._setup_np(gnodes, N)
    n = N + 1
    D = np.zeros(n, complex)
    D[0] = 1.0
    D[1:] = P1[1:] - w1[:N] * cR[0]
    G = np.zeros(n, complex)
    G[0] = 1.0
    K = np.stack([cL, cR])
    a = np.zeros((2, n), complex)
    a[0, :N] = w0  # y0_j = w0_j q_j (j < N)
    a[1, 1:] = w1  # y1_j = w1_{j-1} q_j
    b = np.ones((2, n), complex)
    b[:, 0] = 0.0
    q = _online_solve(G, D, K, a, b)
    num = abs(q[N])
    den = ev._denominator(np, q, gam.real, s, h, H)
    return num / den, q


def adjoint(gnodes, q, Fval, N):
    """Same as evaluator._adjoint_np (lam_0 is returned as 0)."""
    gnodes = np.asarray(gnodes, dtype=complex)
    H, h, gam, s, w0, w1, P1, cL, cR = ev._setup_np(gnodes, N)
    a_ = gam.real
    wa0, wa1, _ = ev._weights(np, a_, s[: H + 1], h)
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
    D[1:] = P1[1:] - w1[:N] * cR[0]
    w0f = np.concatenate([w0, [0.0]])
    # reversed index J = N - j, J = 0..N-1  (j = N..1)
    j = N - np.arange(N)
    Gr = G[j]
    Dr = D[j]
    K = np.stack([cL[:N], cR[:N]])
    a = np.ones((2, N), complex)
    b = np.stack([w0f[j], w1[j - 1]])
    lamr = _online_solve(Gr, Dr, K, a, b)
    lam = np.zeros(N + 1, complex)
    lam[j] = lamr
    return lam


class FastObjective:
    """Drop-in replacement for evaluator.Objective using the O(N log^2 N) solves."""

    def __init__(self, param_to_gnodes, N: int):
        import jax
        import jax.numpy as jnp

        self.N = N
        self.jnp = jnp
        self.gfun = jax.jit(param_to_gnodes)

        def phi(th, q, lam):
            return ev._phi_jax(param_to_gnodes(th), q, lam, N)

        self.grad_phi = jax.jit(jax.grad(phi, argnums=0))

    def F(self, theta):
        return forward(np.asarray(self.gfun(self.jnp.asarray(theta))), self.N)[0]

    def value_and_grad(self, theta):
        jnp = self.jnp
        theta = jnp.asarray(theta)
        gn = np.asarray(self.gfun(theta))
        Fval, q = forward(gn, self.N)
        lam = adjoint(gn, q, Fval, self.N)
        g = self.grad_phi(theta, jnp.asarray(q), jnp.asarray(lam))
        return Fval, np.asarray(g)


if __name__ == "__main__":
    import time

    prof = ev.load_profile(sys.argv[1] if len(sys.argv) > 1 else
                           "third_party/pinho/volterra-quadratic-knots/certificates/C42_compact_best.json")
    for N in [4608, 18432]:
        gn = np.asarray(ev.gnodes_of(prof, N))
        t0 = time.time()
        F0, q0 = ev.F_from_gnodes_np(gn, N, return_q=True)
        t1 = time.time()
        lam0 = ev._adjoint_np(gn, q0, F0, N)
        t2 = time.time()
        F1, q1 = forward(gn, N)
        t3 = time.time()
        lam1 = adjoint(gn, q1, F1, N)
        t4 = time.time()
        print(f"N={N}: F np {F0:.15f} fast {F1:.15f} diff {F1 - F0:.2e}; max|dq| {np.max(np.abs(q1 - q0)):.2e}; "
              f"max|dlam|/max|lam| {np.max(np.abs(lam1[1:] - lam0[1:])) / np.max(np.abs(lam0)):.2e}; "
              f"times np {t1 - t0:.2f}+{t2 - t1:.2f}s fast {t3 - t2:.2f}+{t4 - t3:.2f}s", flush=True)
    for N in [73728, 147456, 294912, 589824]:
        gn = np.asarray(ev.gnodes_of(prof, N))
        t0 = time.time()
        F1, q1 = forward(gn, N)
        t1 = time.time()
        lam1 = adjoint(gn, q1, F1, N)
        print(f"N={N}: F fast {F1:.15f}  forward {t1 - t0:.2f}s adjoint {time.time() - t1:.2f}s", flush=True)
