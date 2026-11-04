"""Limit (n -> infinity) fixed points T* = F(T*, phi) of the phi-fixed smooth families used in finite_n_study.py
(FLOAT).  F at the profile's own T is taken from the S1-validated extrapolations (LOG.md, 16:25 / 16:00):
Bakeas F(0.68637) = 0.6863600848 (base 3072), Pinho F(0.6863607950) = 0.6863607439 (base 18432); the slope dF/dT
is computed here at lower resolution (it enters T* only through (F-T)/(1-dF/dT), so 1e-3 relative accuracy suffices).
Usage: python src/limit_fixed_points.py"""
import os, sys, json, dataclasses
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from evaluator import load_profile, F_extrapolated
from finite_n_study import BAKEAS, PINHO

S1_F = {"bakeas": 0.6863600848, "pinho": 0.6863607439}
out = {}
for name, path, N0 in (("bakeas", BAKEAS, 3072), ("pinho", PINHO, 4608)):
    p = load_profile(path)
    Fp = F_extrapolated(dataclasses.replace(p, T=p.T + 2e-4), N0)[0]
    Fm = F_extrapolated(dataclasses.replace(p, T=p.T - 2e-4), N0)[0]
    slope = (Fp - Fm) / 4e-4
    F0 = S1_F[name]
    Tstar = p.T + (F0 - p.T) / (1 - slope)
    out[name] = {"T_profile": p.T, "F_at_T_profile(S1)": F0, "dF_dT": slope, "T_star_limit": Tstar}
    print(name, out[name], flush=True)
json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results/finite_n/limit_fixed_points.json"), "w"), indent=1)
