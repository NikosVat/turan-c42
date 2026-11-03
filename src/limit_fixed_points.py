"""Limit (n -> infinity) fixed points T* = F(T, phi) of the phi-fixed smooth families used in finite_n_study.py
(FLOAT, evaluator.F_extrapolated).  These are the known targets against which finite-n extrapolations are tested."""
import os, sys, json, dataclasses
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from evaluator import load_profile, F_extrapolated
from finite_n_study import BAKEAS, PINHO

out = {}
for name, path, N0 in (("bakeas", BAKEAS, 6144), ("pinho", PINHO, 9216)):
    p = load_profile(path)
    Ts, Fs = [], []
    for dT in (-2e-4, 0.0, 2e-4):
        q = dataclasses.replace(p, T=p.T + dT)
        Fs.append(F_extrapolated(q, N0)[0]); Ts.append(q.T)
    c = np.polyfit(np.array(Ts) - p.T, np.array(Fs) - np.array(Ts), 2)
    r = np.roots(c); r = r[np.argmin(np.abs(r))].real
    slope = np.polyfit(Ts, Fs, 1)[0]
    out[name] = {"T_profile": p.T, "F_at_T_profile": Fs[1], "dF_dT": slope, "T_star_limit": p.T + r}
    print(name, out[name], flush=True)
json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results/finite_n/limit_fixed_points.json"), "w"), indent=1)
