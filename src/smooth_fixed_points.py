"""Finite-n fixed points T*_n = F_n(T e^{i phi(k/n)}) of the phi-fixed smooth families, for large n (FLOAT).
Used to validate the n-extrapolation models of finite_n_fit.py against the known limits."""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from evaluator import load_profile
from finite_n_study import BAKEAS, PINHO

def Fn_np(s, n):
    """F_n(s) for a first half s (length n//2), by the forward recursion m a_m = -sum s_k a_{m-k} (complex128)."""
    K = len(s)
    a = np.zeros(n + 1, complex); a[0] = 1
    sr = np.asarray(s)[::-1]              # sr[K-k] = s_k
    for m in range(1, n + 1):
        k = min(m, K)
        a[m] = -np.dot(sr[K - k:], a[m - k:m]) / m   # sum_{j=1..k} s_j a_{m-j}
    P = np.cumsum(a); kf = np.arange(K + 1, n + 1)
    return abs(P[n]) / np.sum(np.abs(P[n - kf]) / kf)


if __name__ == "__main__":
    ns = [int(x) for x in sys.argv[1:]] or [200, 283, 400, 566, 800, 1131, 1600, 2263, 3200, 4525, 6400, 9051, 12800]
    out_path = os.path.join(os.path.dirname(__file__), "..", "results/finite_n/smooth_fixed_points.json")
    out = json.load(open(out_path)) if os.path.exists(out_path) else {}
    for name, path in (("bakeas", BAKEAS), ("pinho", PINHO)):
        prof = load_profile(path)
        for n in ns:
            K = n // 2; t0 = time.time()
            e = np.exp(1j * np.angle(np.asarray(prof.g(np.arange(1, K + 1) / n))))
            f = lambda T: Fn_np(T * e, n) - T
            T0, T1 = prof.T, prof.T + 1e-3
            f0, f1 = f(T0), f(T1)
            for it in range(20):
                T2 = T1 - f1 * (T1 - T0) / (f1 - f0)
                T0, f0, T1, f1 = T1, f1, T2, f(T2)
                if abs(f1) < 1e-14 or abs(T1 - T0) < 1e-15:
                    break
            Fprof = Fn_np(prof.T * e, n)
            out.setdefault(name, {})[str(n)] = {"T_star": T1, "F_at_T_profile": Fprof}
            print(f"{name} n={n}: T*_n = {T1:.12f}  F_n(T_profile) = {Fprof:.12f}  ({time.time()-t0:.1f}s)", flush=True)
            json.dump(out, open(out_path, "w"), indent=1)
