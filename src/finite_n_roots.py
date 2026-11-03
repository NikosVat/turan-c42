"""Exact finite-n Harcos/Bakeas construction (float, NOT rigorous): build W, its roots z_j, and compare
max(T, F_n) with the actual normalized bound max_k |s_k| / lambda^k, lambda = max|z_j| (Lemma 1)."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from evaluator import load_profile

def construct(prof, n):
    K = n // 2
    k = np.arange(1, K + 1)
    s = np.asarray(prof.g(k / n))                     # prescribed s_k, k <= K
    # A(x) = exp(-sum s_k x^k / k) mod x^{n+1} via m a_m = -sum_k s_k a_{m-k}
    a = np.zeros(n + 1, complex); a[0] = 1
    for m in range(1, n + 1):
        kk = np.arange(1, min(m, K) + 1)
        a[m] = -np.dot(s[kk - 1], a[m - kk]) / m
    P = np.cumsum(a)
    kf = np.arange(K + 1, n + 1)
    D = np.sum(np.abs(P[n - kf]) / kf)
    lam = P[n] / D
    sig = np.concatenate([s, lam * np.conj(P[n - kf]) / np.abs(P[n - kf])])   # sigma_1..sigma_n
    # W = exp(-sum sigma_k x^k/k) mod x^{n+1}
    w = np.zeros(n + 1, complex); w[0] = 1
    for m in range(1, n + 1):
        kk = np.arange(1, m + 1)
        w[m] = -np.dot(sig[kk - 1], w[m - kk]) / m
    z = np.roots(w)            # roots of z^n W(1/z): coefficients w_0..w_n as highest..lowest
    return abs(P[n]) / D, sig, w, z

if __name__ == "__main__":
    prof = load_profile(sys.argv[1])
    for n in [int(x) for x in sys.argv[2:]]:
        Fn, sig, w, z = construct(prof, n)
        W1 = abs(np.sum(w)); i1 = np.argmin(np.abs(z - 1))
        lam = np.max(np.abs(z)); kk = np.arange(1, n + 1)
        ps = np.array([np.sum(z ** k) for k in (1, 2, n // 2, n)])
        bound_norm = np.max(np.abs(sig) / lam ** kk)
        print(f"n={n}: F_n={Fn:.6f}  max(T,F_n)={max(prof.T, Fn):.6f}  |W(1)|={W1:.1e}  dist(1,roots)={abs(z[i1]-1):.1e}  "
              f"max|z|={lam:.8f}  n*log(max|z|)={n*np.log(lam):.4f}  #|z|>1: {np.sum(np.abs(z) > 1 + 1e-9)}  "
              f"normalized bound max|s_k|/lam^k={bound_norm:.6f}  check |s_1|,|s_2|,|s_n/2|,|s_n| from roots: "
              f"{np.abs(ps).round(5)} vs {np.abs(sig[[0, 1, n // 2 - 1, n - 1]]).round(5)}", flush=True)
