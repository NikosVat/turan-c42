"""Export an opt2.py result as a self-contained exact-decimal candidate JSON (FLOAT SEARCH OUTPUT, NOT CERTIFIED).

Adds float values of the quantities the Theorem A checkers test (NOT rigorous; the certifier must redo them):
  a = Re g(0) = T cos phi_0 (need 0 < a < 1), kappa = sup|g(0) - g(u)| = max(T, 2T max_j |sin((phi_j-phi_0)/2)|)
  (need < 1; |d| = |gamma| = T on (1/2, 1]), slopes beta_j (Bakeas's checker divides by beta_j: need beta_j != 0),
  max |phi_j - phi_0| (Bakeas input check < 3).

Usage: python src/export_candidate.py results/A_xxx.json results/candidate_xxx.json "description"
"""
import hashlib
import json
import sys
from fractions import Fraction

import numpy as np

src, dst, desc = sys.argv[1], sys.argv[2], sys.argv[3]
d = json.load(open(src))
assert d.get("stage") == "final" and "fixed_point" in d, "run not finished"
kn = [Fraction(x) for x in d["breaks"]]
ph = [Fraction(x) for x in d["phases"]]  # exact decimals
knf = np.array([float(k) for k in kn])
phf = np.array([float(p) for p in ph])
T = Fraction(d["T_used"])
Tf = float(T)
sl = np.diff(phf) / np.diff(knf)
kappa = max(Tf, 2 * Tf * np.max(np.abs(np.sin((phf - phf[0]) / 2))))
v, fp = d["verify"], d["fixed_point"]
out = {
    "status": "FLOAT SEARCH CANDIDATE ONLY - NOT CERTIFIED, NOT A CLAIM",
    "description": desc,
    "framework": "Bakeas Theorem A (Harcos linearization, s_k = T exp(i phi(k/n)), k <= n/2), constant modulus T, "
                 "phi piecewise linear on the rational breaks; F = |p(1)| / int_0^{1/2} |p(t)| dt/(1-t)",
    "pieces": len(kn) - 1,
    "T": str(d["T_used"]),
    "breaks": [str(k) for k in kn],
    "phases": [d["phases"][i] for i in range(len(ph))],
    "exact_note": "The profile is DEFINED by the exact rational breaks, the decimal phases and the decimal T above.",
    "float_results": {
        "F_at_T": f"{v['Fx_ind']:.13f}",
        "F_at_T_richardson_err": f"{v['err']:.2e}",
        "extrapolation": "3-level fit F_N = F + A N^-(2-a) + B N^-2 on independent levels (2N0, 4N0, 8N0); "
                         "err = max(|fit(N0..4N0) - fit(2N0..8N0)|, |fit(2N0..8N0) - 4-level fit with N^-(3-a)|)",
        "levels": v["levels"],
        "F_N": [f"{x:.15f}" for x in v["F_N"]],
        "fit_opt_levels": f"{v['Fx_opt']:.13f}",
        "fit_independent_levels": f"{v['Fx_ind']:.13f}",
        "fit_4level": f"{v['Fx_4fit']:.13f}",
        "fixed_point_T_star": f"{fp['T_star']:.13f}",
        "T_star_err": f"{fp['T_star_err']:.1e}",
        "dF_dT": f"{fp['dF_dT']:.5f}",
        "margin_below_0.6863607950": f"{0.6863607950 - fp['T_star']:.3e}",
    },
    "theorem_A_inputs_float": {
        "a_Re_gamma": f"{Tf * np.cos(phf[0]):.6f}",
        "kappa_sup_abs_d": f"{kappa:.6f}",
        "min_abs_slope": f"{np.min(np.abs(sl)):.6g}",
        "max_abs_slope": f"{np.max(np.abs(sl)):.6g}",
        "max_abs_phase_minus_phase0": f"{np.max(np.abs(phf - phf[0])):.6f}",
        "first_break": str(kn[1]),
        "lcm_break_denominator": int(np.lcm.reduce([k.denominator for k in kn])),
    },
    "source_run": src,
}
payload = json.dumps({"T": out["T"], "breaks": out["breaks"], "phases": out["phases"]}, separators=(",", ":"))
out["sha256_of_T_breaks_phases"] = hashlib.sha256(payload.encode()).hexdigest()
json.dump(out, open(dst, "w"), indent=1)
print(dst, out["pieces"], out["float_results"]["fixed_point_T_star"], out["float_results"]["T_star_err"],
      out["theorem_A_inputs_float"])
