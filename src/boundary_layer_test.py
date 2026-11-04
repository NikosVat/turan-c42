"""Does a fixed-k (boundary-layer) phase correction keep its gain as n grows?  (FLOAT, heuristic test.)
Take the free optimum at n0 and T0, keep only its deviation from the smooth Bakeas phases on k <= L, graft that
deviation (indexed by k, not by k/n) onto the smooth profile at larger n, and measure the gain in F_n at T0."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from evaluator import load_profile
from finite_n_study import BAKEAS
from smooth_fixed_points import Fn_np

n0, T0 = int(sys.argv[1]) if len(sys.argv) > 1 else 400, 0.68637
R = os.path.join(os.path.dirname(__file__), "..", "results/finite_n")
bak = load_profile(BAKEAS)
def th_s(n):
    return np.unwrap(np.angle(np.asarray(bak.g(np.arange(1, n // 2 + 1) / n))))
th = np.load(os.path.join(R, f"theta_T0_n{n0}.npy"))
dev = np.angle(np.exp(1j * (th - th_s(n0))))
if np.max(np.abs(dev)) > 1:
    dev = np.angle(np.exp(1j * (-th - th_s(n0))))
res = []
for L in (8, 32):
    for n in (400, 800, 1600, 3200, 6400, 12800, 25600, 51200):
        e = th_s(n); e2 = e.copy(); e2[:L] += dev[:L]
        F0 = Fn_np(T0 * np.exp(1j * e), n); F1 = Fn_np(T0 * np.exp(1j * e2), n)
        res.append({"L": L, "n": n, "gain": F1 - F0})
        print(f"L={L:3d} n={n:6d}: F_n smooth {F0:.10f}, with k<={L} graft {F1:.10f}, gain {F1-F0:+.3e}", flush=True)
json.dump(res, open(os.path.join(R, "boundary_layer_graft.json"), "w"), indent=1)
