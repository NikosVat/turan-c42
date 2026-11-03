"""
Direction-A search driver (FLOAT ONLY, NOT RIGOROUS).

Minimises the Richardson-extrapolated F over the phase values of a piecewise-linear phase profile with
exact rational knots, at fixed T, using the O(N log^2 N) solver of fastsolve.py and the exact discrete
adjoint gradient.  At the end (or with --verify-only) it

  * re-evaluates the extrapolation on the optimisation levels (N0, 2N0, 4N0) and on independent levels
    (2N0, 4N0, 8N0), plus a 4-level fit with an extra N^{-(3-a)} term over (N0..8N0);
  * solves the fixed point T* = F(T*) with the phases frozen (secant in T on the independent levels);
  * writes an exact-decimal JSON (rational "breaks", decimal-string "phases", decimal T).

The Richardson error estimate is err = max(|Fx_opt - Fx_ind|, |Fx_ind - Fx_4fit|).

Knot specs (--knots):  start | quad:m (u_j = j^2/(2m^2)) | pow:p:m:L (u_j = round(L (j/m)^p / 2)/L, made strictly
increasing) | file:path (JSON list of rationals) | refine:k (split every start interval into k equal parts).

Usage example:
  python src/opt2.py --start results/probeA_x2_extrap_eval11.json --knots quad:96 --T 0.68636075 \
      --base 36864 --iters 300 --out results/A96.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from fractions import Fraction

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(__file__))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)
import fastsolve as fs  # noqa: E402

PH_DIGITS = 16  # significant digits of the decimal phases written to JSON


def to_frac(x) -> Fraction:
    if isinstance(x, str):
        return Fraction(x)
    f = Fraction(x)
    return f.limit_denominator(10**8) if abs(float(f.limit_denominator(10**8)) - x) < 1e-15 else f


def load_start(path):
    d = json.load(open(path))
    ph = np.array([float(Fraction(str(x))) for x in d["phases"]])
    if "breaks" in d:
        kn = [Fraction(str(x)) for x in d["breaks"]]
    elif "knots" in d and isinstance(d["knots"], list):
        kn = [to_frac(float(x)) for x in d["knots"]]
    else:
        m = len(ph) - 1
        kn = [Fraction(j, 2 * m) for j in range(m + 1)]
    T = float(Fraction(str(d["T"])))
    return kn, ph, T, d


def make_knots(spec, start_kn):
    if spec == "start":
        return list(start_kn)
    kind, *rest = spec.split(":")
    if kind == "quad":
        m = int(rest[0])
        return [Fraction(j * j, 2 * m * m) for j in range(m + 1)]
    if kind == "pow":
        p, m, L = float(rest[0]), int(rest[1]), int(rest[2])
        v = [int(round(L * 0.5 * (j / m) ** p)) for j in range(m + 1)]
        v[0], v[-1] = 0, L // 2
        for j in range(1, m + 1):
            v[j] = max(v[j], v[j - 1] + 1)
        assert v[-1] == L // 2, "lattice too coarse for this grading"
        return [Fraction(x, L) for x in v]
    if kind == "file":
        return [Fraction(str(x)) for x in json.load(open(rest[0]))]
    if kind == "refine":
        k = int(rest[0])
        out = [start_kn[0]]
        for a, b in zip(start_kn[:-1], start_kn[1:]):
            out += [a + (b - a) * Fraction(i, k) for i in range(1, k + 1)]
        return out
    raise ValueError(spec)


def lcm_den(kn):
    from math import lcm

    d = 2
    for x in kn:
        d = lcm(d, x.denominator)
    return d


def gfun_factory(knf, T, Nk):
    uk = jnp.arange(Nk // 2 + 1) / Nk
    kj = jnp.asarray(knf)

    def gf(x, T=T):
        return T * jnp.exp(1j * jnp.interp(uk, kj, x))

    return gf


def F_levels(knf, x, T, Ns):
    out = []
    for N in Ns:
        gn = np.asarray(gfun_factory(knf, T, N)(jnp.asarray(x)))
        out.append(fs.forward(gn, N)[0])
    return out


def fit(Ns, Fs, a, exps):
    A = np.array([[1.0] + [N ** -e for e in exps] for N in Ns])
    if len(Ns) == len(exps) + 1:
        return float(np.linalg.solve(A, np.array(Fs))[0])
    return float(np.linalg.lstsq(A, np.array(Fs), rcond=None)[0][0])


def extrap_set(knf, x, T, N0):
    """Fx over (N0,2N0,4N0), (2N0,4N0,8N0) and a 4-level fit; returns dict."""
    Ns = [N0 * 2**k for k in range(4)]
    Fs = F_levels(knf, x, T, Ns)
    a = T * np.cos(x[0])
    e3 = [2 - a, 2.0]
    r = {
        "levels": Ns,
        "F_N": Fs,
        "Fx_opt": fit(Ns[:3], Fs[:3], a, e3),
        "Fx_ind": fit(Ns[1:], Fs[1:], a, e3),
        "Fx_4fit": fit(Ns, Fs, a, e3 + [3 - a]),
    }
    r["err"] = max(abs(r["Fx_opt"] - r["Fx_ind"]), abs(r["Fx_ind"] - r["Fx_4fit"]))
    return r


def fixed_point(knf, x, T0, N0, F0=None):
    """T* with F_ind(T*) = T*, phases frozen.  Secant from T0 (2 extra extrapolations)."""
    Ns = [N0 * 2**k for k in range(1, 4)]

    def Fx(T):
        a = T * np.cos(x[0])
        return fit(Ns, F_levels(knf, x, T, Ns), a, [2 - a, 2.0])

    f0 = Fx(T0) if F0 is None else F0
    dT = -1e-7
    f1 = Fx(T0 + dT)
    slope = (f1 - f0) / dT
    Ts = T0 + (f0 - T0) / (1 - slope)
    fs_ = Fx(Ts)
    # one more secant correction (tiny)
    Ts2 = Ts + (fs_ - Ts) / (1 - slope)
    return {"T_star": Ts2, "dF_dT": slope, "F_at_Tstar_minus_Tstar": fs_ - Ts}


def fmt_phase(v):
    return f"{v:.{PH_DIGITS}g}"


def write_json(path, kn, x, T, info, extra=None):
    out = {
        "status": "FLOAT SEARCH CANDIDATE ONLY - NOT CERTIFIED, NOT A CLAIM",
        "family": "Bakeas Theorem A, constant modulus T, piecewise-linear phase on rational breaks",
        "pieces": len(kn) - 1,
        "T": repr(T) if not isinstance(T, str) else T,
        "breaks": [str(k) for k in kn],
        "phases": [fmt_phase(v) for v in x],
        "kind": "pwlinear",
        "exact_note": "the profile is DEFINED by these exact rational breaks and decimal phases",
    }
    out.update(info)
    if extra:
        out.update(extra)
    tmp = path + ".tmp"
    json.dump(out, open(tmp, "w"), indent=1)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--knots", default="start")
    ap.add_argument("--T", type=str, default=None)
    ap.add_argument("--base", type=int, default=None, help="N0; optimisation levels N0, 2N0, 4N0")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--maxtime", type=float, default=1e9, help="seconds; stop optimisation after this")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stall", type=int, default=12, help="stop after this many evals without a 1e-13 improvement")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--no-fixed-point", action="store_true")
    args = ap.parse_args()

    skn, sph, sT, _ = load_start(args.start)
    kn = make_knots(args.knots, skn)
    knf = np.array([float(k) for k in kn])
    sknf = np.array([float(k) for k in skn])
    x0 = np.interp(knf, sknf, sph)
    Tstr = args.T if args.T is not None else repr(sT)
    T = float(Fraction(Tstr))
    den = lcm_den(kn)
    N0 = args.base or den * max(1, -(-18432 // den))
    assert N0 % den == 0, f"base {N0} is not a multiple of knot denominator {den}"
    print(f"{len(kn) - 1} pieces, knot lcm denominator {den}, N0 = {N0}, T = {Tstr}", flush=True)

    t0 = time.time()
    if not args.verify_only:
        Ns = [N0, 2 * N0, 4 * N0]
        a0 = float(T * np.cos(x0[0]))
        A = np.array([[1.0, Nk ** -(2 - a0), Nk ** -2.0] for Nk in Ns])
        coef = np.linalg.solve(A.T, np.array([1.0, 0.0, 0.0]))
        objs = [fs.FastObjective(gfun_factory(knf, T, Nk), Nk) for Nk in Ns]
        SCALE = 1e6
        state = {"n": 0, "best": np.inf, "bestx": x0.copy(), "f0": None, "stop": False}

        def fg(x):
            if state["stop"]:
                return state["best_scaled"], np.zeros_like(x)
            vs = [o.value_and_grad(x) for o in objs]
            f = float(sum(c * v[0] for c, v in zip(coef, vs)))
            g = sum(c * v[1] for c, v in zip(coef, vs))
            state["n"] += 1
            if state["f0"] is None:
                state["f0"] = f
            if f < state["best"] - 1e-13:
                state["last_improve"] = state["n"]
            if f < state["best"]:
                state["best"], state["bestx"] = f, np.array(x)
                state["best_scaled"] = (f - state["f0"]) * SCALE
            if state["n"] % 5 == 1 or state["n"] < 3:
                print(f"[{time.time() - t0:7.1f}s] eval {state['n']:4d} Fx = {f:.13f} best {state['best']:.13f} "
                      f"|g| = {np.linalg.norm(g):.3e}", flush=True)
                write_json(args.out, kn, state["bestx"], Tstr,
                           {"Fx_opt_levels": state["best"], "opt_levels": Ns, "evals": state["n"],
                            "start": args.start, "stage": "optimising"})
            if time.time() - t0 > args.maxtime or state["n"] - state.get("last_improve", 0) >= args.stall:
                state["stop"] = True
            return (f - state["f0"]) * SCALE, g * SCALE

        res = minimize(fg, x0, jac=True, method="L-BFGS-B",
                       options={"maxiter": args.iters, "maxfun": int(args.iters * 1.5), "ftol": 1e-14,
                                "gtol": 1e-14, "maxcor": 30})
        x = state["bestx"]
        print(f"optimisation done: {state['n']} evals, Fx {state['f0']:.13f} -> {state['best']:.13f} "
              f"(delta {state['best'] - state['f0']:.3e}); {res.message} [{time.time() - t0:.0f}s]", flush=True)
    else:
        x = x0
    # round to the decimals that are written, and verify those exact values
    x = np.array([float(fmt_phase(v)) for v in x])
    ver = extrap_set(knf, x, T, N0)
    print(f"verify: F_N {['%.13f' % f for f in ver['F_N']]} at {ver['levels']}", flush=True)
    print(f"verify: Fx_opt {ver['Fx_opt']:.13f}  Fx_ind {ver['Fx_ind']:.13f}  Fx_4fit {ver['Fx_4fit']:.13f}  "
          f"err {ver['err']:.2e}", flush=True)
    info = {"T_used": Tstr, "verify": ver, "start": args.start, "knot_spec": args.knots}
    if not args.no_fixed_point:
        fp = fixed_point(knf, x, T, N0, ver["Fx_ind"])
        fp["T_star_err"] = ver["err"] / (1 - fp["dF_dT"])
        fp["margin_below_0.6863607950"] = 0.6863607950 - fp["T_star"]
        info["fixed_point"] = fp
        print(f"fixed point: T* = {fp['T_star']:.13f}  (dF/dT = {fp['dF_dT']:.4f}, residual "
              f"{fp['F_at_Tstar_minus_Tstar']:.1e}, err est {fp['T_star_err']:.1e}); margin below 0.6863607950: "
              f"{fp['margin_below_0.6863607950']:.3e}", flush=True)
    write_json(args.out, kn, x, Tstr, info, {"stage": "final"})
    print(f"wrote {args.out} [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
