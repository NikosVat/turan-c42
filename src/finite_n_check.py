"""
Finite-n cross-check (FLOAT, NOT rigorous) of the exact construction for profiles with arbitrary breaks:
K = floor(n/2), s_k = T exp(i phi(k/n)) (k <= K), P_m = partial sums of the coefficients of
exp(-sum_{k<=K} s_k x^k/k), F_n = |P_n| / sum_{k>K} |P_{n-k}|/k.  Then R_n <= max(T, F_n).

Uses Bakeas's FFT power-series exp (third_party/bakeas/crosscheck/finite_n.py, MIT), imported, not copied.
Prints F_n for each profile and the differences to the first (reference) profile, which should approach the
difference of the limits.

Usage: python src/finite_n_check.py n1,n2,... [T=0.68636075] REF.json CAND.json [...]
"""
import importlib.util
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from opt2 import load_start  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "bakeas_finite_n", os.path.join(os.path.dirname(__file__), "..", "third_party", "bakeas", "crosscheck", "finite_n.py"))
_bk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bk)


def F_n(knf, ph, T, n):
    K = n // 2
    k = np.arange(1, K + 1)
    S = T * np.exp(1j * np.interp(k / n, knf, ph))
    hh = np.zeros(n + 1, complex)
    hh[1:K + 1] = -S / k
    P = np.cumsum(_bk._exp(hh, n + 1))
    kk = np.arange(K + 1, n + 1)
    return abs(P[n]) / np.sum(np.abs(P[n - kk]) / kk)


if __name__ == "__main__":
    ns = [int(float(x)) for x in sys.argv[1].split(",")]
    Tall = None
    args = sys.argv[2:]
    if args[0].startswith("T="):
        Tall = float(args[0][2:])
        args = args[1:]
    profs = []
    for p in args:
        kn, ph, T, d = load_start(p)
        if "T_used" in d:
            T = float(d["T_used"])
        if Tall is not None:
            T = Tall
        profs.append((p, np.array([float(x) for x in kn]), ph, T))
    for n in ns:
        vals = [F_n(kn, ph, T, n) for _, kn, ph, T in profs]
        print(f"n = {n:>8d}: " + "  ".join(f"{os.path.basename(p)}: F_n = {v:.10f} (T={T})"
                                         for (p, _, _, T), v in zip(profs, vals)), flush=True)
        print(f"            differences to {os.path.basename(profs[0][0])}: "
              + "  ".join(f"{v - vals[0]: .3e}" for v in vals[1:]), flush=True)
