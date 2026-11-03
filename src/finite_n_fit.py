"""Trend fits over n for results/finite_n/n*.json (FLOAT; heuristic extrapolation, not a proof)."""
import glob, json, os, sys
import numpy as np
from scipy.optimize import curve_fit
sys.path.insert(0, os.path.dirname(__file__))
from evaluator import load_profile
from finite_n_study import BAKEAS

R = os.path.join(os.path.dirname(__file__), "..", "results", "finite_n")
rows = []
for f in sorted(glob.glob(os.path.join(R, "n*.json")), key=lambda p: int(os.path.basename(p)[1:-5])):
    d = json.load(open(f))
    if "f_arb" not in d:
        continue
    rows.append(d)
ns = np.array([d["n"] for d in rows], float)
lim = json.load(open(os.path.join(R, "limit_fixed_points.json")))
out = {"n": ns.tolist()}

print("n | starts(a) same-minimiser | F_smooth(T0) | F_free(T0) | Delta(T0) | rho-starts reaching const opt | "
      "max dF/drho | T*_free | T*_smooth(Pinho phi) | T*_smooth(Bakeas phi) | Arb R_n <=")
for d in rows:
    a, b, c = d["a_summary"], d["b_summary"], d["c_kkt_rho"]
    print(f'{d["n"]} | {a["n_starts"]} {a["n_same_minimiser_or_conj(dist<1e-4)"]} | {d["a_smooth_bakeas_F_T0"]:.10f} | '
          f'{a["best"]:.10f} | {a["best"]-d["a_smooth_bakeas_F_T0"]:.4e} | {b["n_reaching_const_optimum_1e-8"]}/{b["n"]} | '
          f'{c["max_dF_drho"]:.2e} | {d["d_fixed_point_free"]["T_star"]:.10f} | {d["e_fixed_point_smooth_pinho"]["T_star"]:.10f} | '
          f'{d["e_fixed_point_smooth_bakeas"]["T_star"]:.10f} | {d["f_arb"]["R_n_upper_bound"]}')

def powfit(x, y):
    p = np.polyfit(np.log(x), np.log(np.abs(y)), 1)
    return -p[0], np.exp(p[1])

def lfit(x, y, sl=slice(None)):
    f = lambda n, L, c, al: L + c * n ** (-al)
    p, cov = curve_fit(f, x[sl], y[sl], p0=[y[-1] - 0.01, 1.0, 0.8], maxfev=20000)
    return p, np.sqrt(np.diag(cov))

D0 = np.array([d["a_summary"]["best"] - d["a_smooth_bakeas_F_T0"] for d in rows])
al, c = powfit(ns, D0)
loc = [np.log(D0[i] / D0[i + 1]) / np.log(ns[i + 1] / ns[i]) for i in range(len(ns) - 1)]
print(f"\nDelta(T0) = F_free - F_smooth at T0: power fit {c:.3g} n^-{al:.3f}; local exponents {np.round(loc, 3).tolist()}")
out["Delta_T0"] = {"values": D0.tolist(), "alpha": al, "c": c, "local_exponents": loc}

Tf = np.array([d["d_fixed_point_free"]["T_star"] for d in rows])
Tp = np.array([d["e_fixed_point_smooth_pinho"]["T_star"] for d in rows])
Tb = np.array([d["e_fixed_point_smooth_bakeas"]["T_star"] for d in rows])
Ds = Tp - Tf
al2, c2 = powfit(ns, Ds)
loc2 = [np.log(Ds[i] / Ds[i + 1]) / np.log(ns[i + 1] / ns[i]) for i in range(len(ns) - 1)]
print(f"Delta* = T*_smooth(Pinho) - T*_free: {np.array2string(Ds, precision=4)}; power fit {c2:.3g} n^-{al2:.3f}; local exps {np.round(loc2,3).tolist()}")
out["Delta_star"] = {"values": Ds.tolist(), "alpha": al2, "c": c2, "local_exponents": loc2}

print("\n3-parameter fits X_n = L + c n^-alpha (all n, and n >= 400):")
for name, y, known in (("T*_smooth(Pinho phi)", Tp, lim["pinho"]["T_star_limit"]),
                       ("T*_smooth(Bakeas phi)", Tb, lim["bakeas"]["T_star_limit"]),
                       ("T*_free", Tf, None)):
    for lab, sl in (("all", slice(None)), ("n>=400", ns >= 400)):
        if np.sum(np.ones_like(ns)[sl]) < 4:
            continue
        p, e = lfit(ns, y, sl)
        err = f"  (known limit {known:.10f}, fit error {p[0]-known:+.2e})" if known else ""
        print(f"  {name:22s} [{lab}]: L = {p[0]:.8f} +- {e[0]:.1e}, alpha = {p[2]:.3f}{err}")
        out.setdefault("fits", []).append({"series": name, "range": lab, "L": p[0], "L_se": e[0], "c": p[1],
                                           "alpha": p[2], "known_limit": known})
# difference fit with offset
for lab, sl in (("all", slice(None)), ("n>=400", ns >= 400)):
    f = lambda n, d0, c, al: d0 + c * n ** (-al)
    try:
        p, cov = curve_fit(f, ns[sl], Ds[sl], p0=[0, 0.05, 0.8], maxfev=20000)
        print(f"  Delta* = d_inf + c n^-alpha [{lab}]: d_inf = {p[0]:.2e} +- {np.sqrt(cov[0,0]):.1e}, alpha = {p[2]:.3f}")
        out.setdefault("fits", []).append({"series": "Delta*", "range": lab, "d_inf": p[0], "alpha": p[2]})
    except Exception as ex:
        print("  Delta* offset fit failed:", ex)

# boundary-layer structure of the free optimum at T0 relative to the smooth Bakeas profile
bak = load_profile(BAKEAS)
print("\nphase deviation theta_free - theta_smooth(Bakeas) at T0:")
print("  n | k=1 | k=2 | k=4 | k=8 | k=16 | k=32 | u=0.1 | u=0.25 | u=0.4 | u=0.5")
bl = []
for d in rows:
    n = d["n"]; K = n // 2
    th = np.load(os.path.join(R, f"theta_T0_n{n}.npy"))
    th0 = np.unwrap(np.angle(np.asarray(bak.g(np.arange(1, K + 1) / n))))
    dev = np.angle(np.exp(1j * (th - th0)))
    if np.max(np.abs(dev)) > 1:   # conjugate minimiser
        dev = np.angle(np.exp(1j * (-th - th0)))
    ks = [1, 2, 4, 8, 16, 32]; us = [int(round(x * n)) for x in (0.1, 0.25, 0.4, 0.5)]
    vals = [dev[k - 1] for k in ks] + [dev[k - 1] for k in us]
    bl.append({"n": n, "dev": vals})
    print(f"  {n} | " + " | ".join(f"{v:+.4f}" for v in vals))
out["boundary_layer"] = bl
json.dump(out, open(os.path.join(R, "fit.json"), "w"), indent=1)
