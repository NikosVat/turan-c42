"""
Float search (NOT rigorous): minimise F over the phase (and optionally modulus) profile at fixed T,
using the exact discrete adjoint gradient of src/evaluator.py and scipy L-BFGS-B.  Checkpoints to JSON.

Usage:
  python src/optimize.py --start PROFILE.json --N 18432 --T 0.6863607950 [--rho] [--refine0 K]
                         [--iters 200] --out results/run.json

--refine0 K : add K extra knots at the first K grid points 1/N..K/N (below the first knot of the start
              profile), to probe how much a finer phase near u = 0 helps (direction A).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(__file__))
import jax.numpy as jnp  # noqa: E402

from evaluator import Profile, Objective, load_profile, F_extrapolated  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--N", type=int, default=18432)
    ap.add_argument("--T", type=float, default=None)
    ap.add_argument("--rho", action="store_true", help="also optimise modulus factor rho <= 1 (direction B)")
    ap.add_argument("--refine0", type=int, default=0)
    ap.add_argument("--rho0", type=float, default=1.0, help="initial rho (with --rho)")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--extrap", action="store_true",
                    help="minimise the Richardson extrapolation over N, 2N, 4N (exponents 2-a and 2) instead of F_N")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    prof = load_profile(args.start)
    T = prof.T if args.T is None else args.T
    kn = np.asarray(prof.knots, float)
    ph = np.asarray(prof.phases, float)
    if args.refine0:
        extra = np.arange(1, args.refine0 + 1) / args.N
        extra = extra[extra < kn[1] - 1e-15]
        newkn = np.concatenate([[0.0], extra, kn[1:]])
        ph = np.interp(newkn, kn, ph)
        kn = newkn
    m = len(kn)
    N = args.N
    u = jnp.arange(N // 2 + 1) / N
    base = Profile(T=T, knots=kn, phases=ph)

    def make_gfun(Nk):
        uk = jnp.arange(Nk // 2 + 1) / Nk
        if args.rho:
            return lambda x: base.g(uk, phases=x[:m], rho=x[m:], T=T)
        return lambda x: base.g(uk, phases=x, T=T)

    if args.rho:
        x0 = np.concatenate([ph, args.rho0 * np.ones(m)])
        bounds = [(None, None)] * m + [(0.0, 1.0)] * m
    else:
        x0 = ph.copy()
        bounds = None

    if args.extrap:
        a0 = float(T * (args.rho0 if args.rho else 1.0) * np.cos(ph[0]))
        Ns = [N, 2 * N, 4 * N]
        A = np.array([[1.0, Nk ** -(2 - a0), Nk ** -2.0] for Nk in Ns])
        coef = np.linalg.solve(A.T, np.array([1.0, 0.0, 0.0]))  # F_x = coef . F_N
        objs = [Objective(make_gfun(Nk), Nk) for Nk in Ns]
        print("extrapolation over", Ns, "coefficients", coef, flush=True)

        class _Ex:
            def F(self, x):
                return float(sum(c * o.F(x) for c, o in zip(coef, objs)))

            def value_and_grad(self, x):
                vs = [o.value_and_grad(x) for o in objs]
                return float(sum(c * v[0] for c, v in zip(coef, vs))), sum(c * v[1] for c, v in zip(coef, vs))

        obj = _Ex()
    else:
        obj = Objective(make_gfun(N), N)
    hist = []
    t0 = time.time()

    def fg(x):
        f, g = obj.value_and_grad(x)
        hist.append(f)
        if len(hist) % 10 == 1:
            print(f"[{time.time() - t0:7.1f}s] eval {len(hist):4d}  F = {f:.12f}  |g| = {np.linalg.norm(g):.3e}",
                  flush=True)
            save(x, f)
        return f, g

    def save(x, f):
        out = {"T": repr(T), "knots": [repr(float(v)) for v in kn], "phases": [repr(float(v)) for v in x[:m]],
               "F_float": f, "N": N, "extrap": bool(args.extrap), "evals": len(hist), "start": args.start, "kind": "pwlinear",
               "note": "float search output, not certified"}
        if args.rho:
            out["rho"] = [repr(float(v)) for v in x[m:]]
        json.dump(out, open(args.out, "w"), indent=1)

    f0 = obj.F(x0)
    print(f"start F(N={N}) = {f0:.12f}, T = {T}, params = {len(x0)}", flush=True)
    res = minimize(fg, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": args.iters, "maxfun": args.iters * 2, "ftol": 1e-15, "gtol": 1e-12})
    save(res.x, res.fun)
    print(f"done: F(N={N}) {f0:.12f} -> {res.fun:.12f}  (delta {res.fun - f0:.3e}); {res.message}", flush=True)
    if args.rho:
        print("rho min/max:", res.x[m:].min(), res.x[m:].max())


if __name__ == "__main__":
    main()
