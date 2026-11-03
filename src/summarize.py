"""Table of opt2.py results: pieces, first knot, max |slope|, T*, Richardson error, margin (float, search only)."""
import glob
import json
import sys
from fractions import Fraction

import numpy as np

TARGET = 0.6863607950
rows = []
for p in sorted(sys.argv[1:] or glob.glob("results/A*.json")):
    d = json.load(open(p))
    if d.get("stage") != "final" or "fixed_point" not in d:
        continue
    kn = np.array([float(Fraction(x)) for x in d["breaks"]])
    ph = np.array([float(x) for x in d["phases"]])
    sl = np.abs(np.diff(ph) / np.diff(kn))
    fp = d["fixed_point"]
    v = d["verify"]
    rows.append((fp["T_star"], p, len(kn) - 1, d["breaks"][1], sl.max(), fp["T_star_err"], v["err"],
                 TARGET - fp["T_star"], v["levels"][0]))
rows.sort()
print(f"{'file':34s} {'pcs':>4s} {'1st knot':>9s} {'max|slope|':>10s} {'T* (float)':>15s} {'err(T*)':>8s} "
      f"{'margin':>9s} {'N0':>7s}")
for T, p, m, k1, sl, eT, eF, mg, N0 in rows:
    print(f"{p:34s} {m:4d} {k1:>9s} {sl:10.1f} {T:.13f} {eT:8.1e} {mg:9.3e} {N0:7d}")
