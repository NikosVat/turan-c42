"""Finite-n study of the exact first-half problem (S4).  FLOAT SEARCH, except `arb_upper_bound`.

By the exact reduction (FINDINGS.md, Prop. 1),

    R_n = min_{s in C^K} max( max_k |s_k| , F_n(s) ),   K = floor(n/2),
    F_n(s) = |P_n| / sum_{k=K+1}^{n} |P_{n-k}|/k,   sum_m P_m x^m = exp(-sum_{k<=K} s_k x^k/k) / (1-x).

We parametrise s_k = T rho_k e^{i theta_k} and run local optimisation (L-BFGS-B, exact JAX gradients) from
many random starts.  Everything here is a LOCAL search; no global optimality is proved.

For each n this script computes and writes results/finite_n/n{n}.json:
  (a) multistart at fixed T0 (constant modulus): distribution of local optima of F_n on the torus |s_k| = T0;
  (b) multistart at fixed T0 with free modulus 0 <= rho_k <= 1;
  (c) the KKT sign test d F_n / d rho_k at the best constant-modulus optimum;
  (d) the finite-n fixed point T*_n = G_n(T*_n) of the best basin (Newton in T, envelope derivative),
      plus a small multistart AT T*_n;
  (e) the same fixed point for the smooth ansatz s_k = T e^{i phi(k/n)} with phi from a given limit profile;
  (f) an Arb (python-flint, 256 bits) upper bound for max(T*, F_n) of the best free tuple, i.e. a rigorous
      upper bound for R_n (rigorous for R_n only, says nothing rigorous about C42).

Usage: python src/finite_n_study.py n [--starts 200] [--rho-starts 60] [--fp-starts 20] [--T0 0.68637]
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
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
from evaluator import load_profile  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BAKEAS = os.path.join(ROOT, "third_party/bakeas/c42_profile_0.68637.json")
PINHO = os.path.join(ROOT, "third_party/pinho/volterra-quadratic-knots/certificates/C42_compact_best.json")


# ------------------------------------------------------------------------------------------------
# F_n in JAX
# ------------------------------------------------------------------------------------------------
def make_Fn(n: int):
    K = n // 2
    idx_i = jnp.arange(1, n + 1)[:, None]
    idx_j = jnp.arange(1, n + 1)[None, :]
    lower = idx_j < idx_i
    diff = jnp.clip(idx_i - idx_j, 0, n)
    mdiag = jnp.diag(jnp.arange(1, n + 1).astype(complex))
    kf = jnp.arange(K + 1, n + 1)

    def Fn_of_s(s):
        sfull = jnp.zeros(n + 1, complex).at[1:K + 1].set(s)
        # m a_m + sum_{k=1}^{m-1} s_k a_{m-k} = -s_m,  a_0 = 1
        L = jnp.where(lower, sfull[diff], 0.0) + mdiag
        a = jax.scipy.linalg.solve_triangular(L, -sfull[1:], lower=True)
        P = jnp.cumsum(jnp.concatenate([jnp.ones(1, complex), a]))
        return jnp.abs(P[n]) / jnp.sum(jnp.abs(P[n - kf]) / kf)

    def F_const(theta, T):
        return Fn_of_s(T * jnp.exp(1j * theta))

    def F_rho(x, T):  # x = [theta (K), rho (K)]  (polar; used only for the KKT sign test at rho = 1)
        return Fn_of_s(T * x[K:] * jnp.exp(1j * x[:K]))

    def F_disk(x, T):
        # x = [Re w (K), Im w (K)],  s_k = T * 2 w_k / (1 + |w_k|^2): smooth, unconstrained, onto the closed
        # disk |s_k| <= T (|w|=1 <-> |s|=T).  Local minima in w <-> local minima of F_n on the closed polydisc.
        # (Polar coordinates with 0 <= rho <= 1 create spurious KKT points at rho_k = 0; this map does not.)
        w = x[:K] + 1j * x[K:]
        return Fn_of_s(T * 2 * w / (1 + jnp.abs(w) ** 2))

    vg_const = jax.jit(jax.value_and_grad(F_const, argnums=0))
    dT_const = jax.jit(jax.grad(F_const, argnums=1))
    vg_rho = jax.jit(jax.value_and_grad(F_rho, argnums=0))
    vg_disk = jax.jit(jax.value_and_grad(F_disk, argnums=0))
    return jax.jit(F_const), vg_const, dT_const, jax.jit(F_rho), vg_rho, vg_disk


def _np(fun):
    def f(x, *a):
        v, g = fun(x, *a)
        return float(v), np.asarray(g, dtype=float)
    return f


# ------------------------------------------------------------------------------------------------
# starts
# ------------------------------------------------------------------------------------------------
def random_start(kind: str, K: int, n: int, rng):
    u = np.arange(1, K + 1) / n
    if kind == "iid":
        return rng.uniform(0, 2 * np.pi, K)
    if kind == "smooth":  # random Fourier series in u, amplitude O(1)
        th = rng.uniform(0, 2 * np.pi) * np.ones(K)
        for j in range(1, 7):
            a, b = rng.normal(0, 1.5 / j, 2)
            th += a * np.cos(2 * np.pi * j * u) + b * np.sin(2 * np.pi * j * u)
        return th
    if kind == "linear":
        return rng.uniform(0, 2 * np.pi) + rng.uniform(-12, 12) * u
    if kind == "rough":  # smooth + iid noise of size ~0.5 rad on every k
        return random_start("smooth", K, n, rng) + rng.normal(0, 0.5, K)
    raise ValueError(kind)


KINDS = ["iid", "smooth", "linear", "rough"]


def lbfgs(fun, x0, bounds=None, maxiter=5000):
    r = minimize(fun, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                 options={"maxiter": maxiter, "maxfun": 2 * maxiter, "ftol": 1e-15, "gtol": 1e-11, "maxcor": 30})
    return r


def wrapped_dist(a, b):
    d = np.angle(np.exp(1j * (a - b)))
    return float(np.max(np.abs(d)))


# ------------------------------------------------------------------------------------------------
# Arb upper bound for R_n
# ------------------------------------------------------------------------------------------------
def arb_upper_bound(theta_dec: list[str], T_dec: str, n: int, prec: int = 256):
    """Rigorous: s_k = T exp(i theta_k) with T, theta_k the given exact decimals.  Returns an upper bound
    (as a float rounded UP, via arb upper endpoint) for max(T, F_n(s)), hence for R_n by Lemma 2."""
    import flint
    from flint import acb, arb, acb_series

    old_prec, old_cap = flint.ctx.prec, flint.ctx.cap
    flint.ctx.prec, flint.ctx.cap = prec, n + 1
    try:
        K = n // 2
        T = arb(T_dec)
        coeffs = [acb(0)]
        for k in range(1, K + 1):
            th = arb(theta_dec[k - 1])
            coeffs.append(-T * acb(th.cos(), th.sin()) / k)
        coeffs += [acb(0)] * (n - K)
        A = acb_series(coeffs).exp()  # exp(-sum s_k x^k/k) mod x^{n+1}, rigorous
        alpha = A.coeffs()
        alpha += [acb(0)] * (n + 1 - len(alpha))
        P = [alpha[0]]
        for m in range(1, n + 1):
            P.append(P[-1] + alpha[m])
        D = arb(0)
        for k in range(K + 1, n + 1):
            D += abs(P[n - k]) / k
        assert D > 0
        Fn = abs(P[n]) / D
        # decimal upper bound B (12 digits after the point), accepted only if arb certifies B > F_n and B >= T
        mid = float(Fn.mid())
        for digits in (12, 11, 10):
            B = f"{np.ceil(max(mid, float(T.mid())) * 10**digits + 2) / 10**digits:.{digits}f}"
            if (arb(B) > Fn) and (arb(B) >= T):
                break
        else:
            raise RuntimeError("could not certify a decimal upper bound")
        return {"F_n_ball": Fn.str(25, radius=True), "D_lower": float(D.lower()), "T": T_dec,
                "R_n_upper_bound": B, "certified": "arb: B > F_n(s) and B >= T = max|s_k|"}
    finally:
        flint.ctx.prec, flint.ctx.cap = old_prec, old_cap


# ------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int)
    ap.add_argument("--starts", type=int, default=200)
    ap.add_argument("--rho-starts", type=int, default=60)
    ap.add_argument("--fp-starts", type=int, default=20)
    ap.add_argument("--T0", type=float, default=0.68637)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    n, K, T0 = args.n, args.n // 2, args.T0
    rng = np.random.default_rng(args.seed + 1000 * n)
    out = args.out or os.path.join(ROOT, f"results/finite_n/n{n}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    res: dict = {"n": n, "K": K, "T0": T0, "status": "FLOAT SEARCH (local optimisation); only 'arb' entries are rigorous"}

    def save():
        json.dump(res, open(out, "w"), indent=1)

    Fc, vgc, dTc, Fr, vgr, vgd = make_Fn(n)
    u = np.arange(1, K + 1) / n
    bak, pin = load_profile(BAKEAS), load_profile(PINHO)
    th_bak = np.unwrap(np.angle(np.asarray(bak.g(u))))
    th_pin = np.unwrap(np.angle(np.asarray(pin.g(u))))
    t0 = time.time()

    # ---- (a) smooth reference and multistart at T0, constant modulus
    F_smooth_T0 = float(Fc(th_bak, T0))
    fun = lambda x: _np(vgc)(x, T0)
    r = lbfgs(fun, th_bak.copy())
    best_th, best_F = r.x.copy(), float(r.fun)
    res["a_smooth_bakeas_F_T0"] = F_smooth_T0
    res["a_from_smooth"] = {"F": best_F, "nit": int(r.nit), "gradnorm": float(np.linalg.norm(r.jac))}
    runs = []
    for i in range(args.starts):
        kind = KINDS[i % len(KINDS)]
        x0 = random_start(kind, K, n, rng)
        r = lbfgs(fun, x0)
        runs.append({"kind": kind, "F": float(r.fun), "nit": int(r.nit), "gradnorm": float(np.linalg.norm(r.jac)),
                     "dist_best": wrapped_dist(r.x, best_th), "dist_conj_best": wrapped_dist(-r.x, best_th)})
        if r.fun < best_F - 1e-12:
            best_F, best_th = float(r.fun), r.x.copy()
        if i % 20 == 19:
            res["a_runs"] = runs; save()
            print(f"n={n} (a) {i+1}/{args.starts} best {best_F:.10f}  {time.time()-t0:.0f}s", flush=True)
    res["a_runs"] = runs
    Fs = np.array([q["F"] for q in runs])
    res["a_summary"] = {"best": float(best_F), "n_starts": len(runs),
                        "n_within_1e-8_of_best": int(np.sum(Fs < best_F + 1e-8)),
                        "n_same_minimiser_or_conj(dist<1e-4)": int(sum(min(q["dist_best"], q["dist_conj_best"]) < 1e-4 for q in runs)),
                        "worst": float(Fs.max()), "distinct_values_1e-8": sorted(set(np.round(Fs, 8).tolist()))[:20]}
    np.save(os.path.join(os.path.dirname(out), f"theta_T0_n{n}.npy"), best_th)
    save()
    print(f"n={n} (a) done: {res['a_summary']}", flush=True)

    # ---- (c) KKT sign test for rho at the best constant-modulus optimum
    x = np.concatenate([best_th, np.ones(K)])
    v, g = _np(vgr)(x, T0)
    grho = g[K:]
    res["c_kkt_rho"] = {"F": v, "max_dF_drho": float(grho.max()), "min_dF_drho": float(grho.min()),
                        "n_positive": int(np.sum(grho > 0)), "grad_theta_norm": float(np.linalg.norm(g[:K]))}
    save()

    # ---- (b) free modulus multistart at T0, |s_k| <= T0, via the disk map (unconstrained L-BFGS)
    fund = lambda x: _np(vgd)(x, T0)
    rruns = []
    for i in range(args.rho_starts):
        kind = KINDS[i % len(KINDS)]
        th0 = random_start(kind, K, n, rng)
        rho0 = rng.uniform(0.0, 1.0, K) if i % 2 == 0 else np.full(K, rng.uniform(0.2, 1.0))
        r0 = rho0 / (1 + np.sqrt(np.maximum(1 - rho0 ** 2, 0)))       # |w| with 2|w|/(1+|w|^2) = rho0, |w| <= 1
        w0 = r0 * np.exp(1j * th0)
        r = lbfgs(fund, np.concatenate([w0.real, w0.imag]))
        w = r.x[:K] + 1j * r.x[K:]
        rho = 2 * np.abs(w) / (1 + np.abs(w) ** 2)
        rruns.append({"kind": kind, "F": float(r.fun), "min_rho": float(rho.min()), "gradnorm": float(np.linalg.norm(r.jac)),
                      "frac_rho_ge_1-1e-6": float(np.mean(rho > 1 - 1e-6)), "nit": int(r.nit)})
        if i % 10 == 9:
            res["b_runs"] = rruns; save()
            print(f"n={n} (b) {i+1}/{args.rho_starts} best {min(q['F'] for q in rruns):.10f}  {time.time()-t0:.0f}s", flush=True)
    res["b_runs"] = rruns
    Fb = np.array([q["F"] for q in rruns]) if rruns else np.array([np.nan])
    res["b_summary"] = {"best": float(np.min(Fb)), "beats_const_modulus_by_>1e-9": bool(np.min(Fb) < best_F - 1e-9),
                        "n_reaching_const_optimum_1e-8": int(np.sum(Fb < best_F + 1e-8)), "n": len(rruns),
                        "worst": float(np.max(Fb)), "distinct_values_1e-6": sorted(set(np.round(Fb, 6).tolist()))[:20]}
    save()
    print(f"n={n} (b) done: {res['b_summary']}", flush=True)

    # ---- (d) fixed point of the best basin: T = G_n(T), Newton with envelope derivative dG/dT = dF/dT
    T, th = T0, best_th.copy()
    hist = []
    for it in range(12):
        r = lbfgs(lambda x: _np(vgc)(x, T), th)
        th, G = r.x.copy(), float(r.fun)
        dG = float(dTc(th, T))
        hist.append({"T": T, "G": G, "dG_dT": dG})
        if abs(G - T) < 1e-13:
            break
        T = T - (G - T) / (dG - 1.0)
    res["d_fixed_point_free"] = {"T_star": T, "G_at_T_star": G, "history": hist}
    np.save(os.path.join(os.path.dirname(out), f"theta_fp_n{n}.npy"), th)
    th_fp, T_fp = th.copy(), T
    save()
    print(f"n={n} (d) free fixed point T*={T:.10f}  {time.time()-t0:.0f}s", flush=True)
    # small multistart AT T*_n
    fp_runs = []
    for i in range(args.fp_starts):
        kind = KINDS[i % len(KINDS)]
        r = lbfgs(lambda x: _np(vgc)(x, T_fp), random_start(kind, K, n, rng))
        fp_runs.append({"kind": kind, "F": float(r.fun)})
    if fp_runs:
        Ff = np.array([q["F"] for q in fp_runs])
        res["d_multistart_at_T_star"] = {"best": float(Ff.min()), "n": len(fp_runs), "worst": float(Ff.max()),
                                         "n_within_1e-8_of_fp": int(np.sum(Ff < G + 1e-8)),
                                         "any_below_fp_by_1e-9": bool(Ff.min() < G - 1e-9)}
    save()

    # ---- (e) smooth-ansatz fixed points (phi fixed from the limit profile, T varies)
    for name, thp in (("bakeas", th_bak), ("pinho", th_pin)):
        T = T0
        for it in range(30):
            G = float(Fc(thp, T)); dG = float(dTc(thp, T))
            if abs(G - T) < 1e-14:
                break
            T = T - (G - T) / (dG - 1.0)
        res[f"e_fixed_point_smooth_{name}"] = {"T_star": T, "F_at_T_star": G}
    res["e_smooth_F_at_T0"] = {"bakeas": F_smooth_T0, "pinho": float(Fc(th_pin, T0))}
    save()

    # ---- (f) Arb upper bound for R_n from the free fixed-point tuple (rounded decimals)
    T_dec = f"{T_fp + 2e-12:.13f}"           # round T up a little; F then only decreases (G nonincreasing) -- but we
    th_dec = [f"{t:.17f}" for t in th_fp]    # do not rely on that: the Arb run evaluates THIS exact tuple.
    ta = time.time()
    arb = arb_upper_bound(th_dec, T_dec, n)
    arb["seconds"] = time.time() - ta
    res["f_arb"] = arb
    json.dump({"n": n, "T": T_dec, "theta": th_dec,
               "meaning": "s_k = T exp(i theta_k), k=1..n/2; completed by Harcos/Lemma 2. R_n <= max(T, F_n)."},
              open(os.path.join(os.path.dirname(out), f"tuple_fp_n{n}.json"), "w"))
    res["seconds"] = time.time() - t0
    save()
    print(f"n={n} (f) Arb: R_n <= {arb['R_n_upper_bound']}  F_n ball {arb['F_n_ball']}  total {res['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
