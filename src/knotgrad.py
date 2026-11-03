"""
Knot-position gradient of the extrapolated F for a piecewise-linear phase profile (FLOAT, search only).

For phi piecewise linear with knots x_j and values phi_j (moving x_j with the values fixed):
    d phi(u)/d x_j = -sl_L hat_j(u) on (x_{j-1}, x_j),  -sl_R hat_j(u) on (x_j, x_{j+1}),
and -(sl_L + sl_R)/2 at u = x_j (a grid node; the average of the one-sided derivatives, i.e. the derivative of
the continuum objective).  With the node phase sensitivities s_k = dF/dphi(u_k) from the exact adjoint this
gives dF/dx_j = sum_k s_k dphi(u_k)/dx_j and dF/dphi_j = sum_k s_k hat_j(u_k).  Both are combined over the
extrapolation levels with the Richardson coefficients.

Usage: python src/knotgrad.py PROFILE.json N0 [--out file.npz]
"""
import json
import os
import sys
from fractions import Fraction

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)
import evaluator as ev  # noqa: E402
import fastsolve as fs  # noqa: E402
from opt2 import load_start  # noqa: E402

_gphi = jax.jit(jax.grad(lambda x, q, lam, N: ev._phi_jax(x[: x.shape[0] // 2] + 1j * x[x.shape[0] // 2:], q, lam, N)),
                static_argnums=3)


def node_phase_sens(gn, N):
    F, q = fs.forward(gn, N)
    lam = fs.adjoint(gn, q, F, N)
    x = jnp.concatenate([jnp.real(jnp.asarray(gn)), jnp.imag(jnp.asarray(gn))])
    gr = np.asarray(_gphi(x, jnp.asarray(q), jnp.asarray(lam), N))
    m = len(gn)
    G = gr[:m] + 1j * gr[m:]
    # dF = Re(conj(G) dg), dg = i g dphi
    return F, np.real(np.conj(G) * 1j * gn)


def knot_grad(knf, ph, T, N):
    u = np.arange(N // 2 + 1) / N
    gn = T * np.exp(1j * np.interp(u, knf, ph))
    F, s = node_phase_sens(gn, N)
    m = len(knf) - 1
    sl = np.diff(ph) / np.diff(knf)
    dx = np.zeros(m + 1)
    dphi = np.zeros(m + 1)
    kidx = np.rint(knf * N).astype(int)
    assert np.allclose(kidx / N, knf, atol=1e-14), "knots must be grid points"
    for j in range(m + 1):
        k = kidx[j]
        dphi[j] += s[k]
        if j > 0:
            kl = np.arange(kidx[j - 1] + 1, k)
            hat = (u[kl] - knf[j - 1]) / (knf[j] - knf[j - 1])
            dphi[j] += np.sum(s[kl] * hat)
            dx[j] += -sl[j - 1] * np.sum(s[kl] * hat)
        if j < m:
            kr = np.arange(k + 1, kidx[j + 1])
            hat = (knf[j + 1] - u[kr]) / (knf[j + 1] - knf[j])
            dphi[j] += np.sum(s[kr] * hat)
            dx[j] += -sl[j] * np.sum(s[kr] * hat)
        slL = sl[j - 1] if j > 0 else 0.0
        slR = sl[j] if j < m else 0.0
        if 0 < j < m:
            dx[j] += -0.5 * (slL + slR) * s[k]
    return F, dx, dphi, s


if __name__ == "__main__":
    path, N0 = sys.argv[1], int(sys.argv[2])
    kn, ph, T, d = load_start(path)
    knf = np.array([float(k) for k in kn])
    a = T * np.cos(ph[0])
    Ns = [N0, 2 * N0, 4 * N0]
    A = np.array([[1.0, N ** -(2 - a), N ** -2.0] for N in Ns])
    coef = np.linalg.solve(A.T, np.array([1.0, 0.0, 0.0]))
    res = [knot_grad(knf, ph, T, N) for N in Ns]
    F = sum(c * r[0] for c, r in zip(coef, res))
    dx = sum(c * r[1] for c, r in zip(coef, res))
    dphi = sum(c * r[2] for c, r in zip(coef, res))
    print(f"{path}: Fx = {F:.13f}, |dF/dphi| = {np.linalg.norm(dphi):.2e}, |dF/dx| = {np.linalg.norm(dx):.2e}")
    hs = np.diff(knf)
    print(" j        x_j        h_left       dF/dx_j     dF/dx_j*h_left   dF/dphi_j")
    for j in range(len(knf)):
        hl = hs[j - 1] if j > 0 else 0.0
        print(f"{j:3d} {knf[j]:.8f} {hl:.3e} {dx[j]: .3e} {dx[j] * hl: .3e} {dphi[j]: .3e}")
    if "--out" in sys.argv:
        np.savez(sys.argv[sys.argv.index("--out") + 1], knots=knf, phases=ph, dx=dx, dphi=dphi, F=F)
