"""Numerical cross-checks of the exact reduction (FINDINGS.md Prop. 1).  FLOAT; illustrates, proves nothing.
(1) random tuples z with z_1 = 1: the identity P_n - sum_{k>K} (s_k/k) P_{n-k} = 0 and the inequality
    max_{k>K} |s_k(z)| >= F_n(s_1..s_K);
(2) tiny n: minimum of max_k |s_k(z)| over z (z_1 = 1) by direct multistart in z-space, versus the minimum of
    max(max_{k<=K}|s_k|, F_n(s)) over the first half s in C^K."""
import numpy as np
from scipy.optimize import minimize

def P_of_s(s, n):
    K = len(s); a = np.zeros(n + 1, complex); a[0] = 1
    for m in range(1, n + 1):
        kk = np.arange(1, min(m, K) + 1)
        a[m] = -np.dot(s[kk - 1], a[m - kk]) / m
    return np.cumsum(a)

def Fn(s, n):
    K = n // 2; P = P_of_s(s, n); kf = np.arange(K + 1, n + 1)
    return abs(P[n]) / np.sum(np.abs(P[n - kf]) / kf), P

rng = np.random.default_rng(1)
worst_id, min_slack = 0.0, np.inf
for trial in range(2000):
    n = int(rng.integers(2, 60)); K = n // 2
    z = np.concatenate([[1.0], rng.normal(size=n - 1) + 1j * rng.normal(size=n - 1)]) * np.r_[1, rng.uniform(0.2, 1.2, n - 1)]
    s = np.array([np.sum(z ** k) for k in range(1, n + 1)])
    F, P = Fn(s[:K], n)
    ident = P[n] - sum(s[k - 1] / k * P[n - k] for k in range(K + 1, n + 1))
    scale = abs(P[n]) + sum(abs(s[k - 1] / k * P[n - k]) for k in range(K + 1, n + 1))
    worst_id = max(worst_id, abs(ident) / scale)
    min_slack = min(min_slack, np.max(np.abs(s[K:])) / F - 1)
print(f"(1) 2000 random tuples, n in [2,60): max relative |identity residual| = {worst_id:.1e}; "
      f"min of max_(k>K)|s_k| / F_n - 1 = {min_slack:.3e} (must be >= 0 up to rounding)")

def direct(n, starts=400):
    best = np.inf
    for _ in range(starts):
        x0 = np.concatenate([rng.normal(size=2 * (n - 1)) * rng.uniform(0.3, 1), [2.0]])
        def S(x):
            z = np.concatenate([[1.0], x[:n - 1] + 1j * x[n - 1:2 * n - 2]])
            return np.array([abs(np.sum(z ** k)) ** 2 for k in range(1, n + 1)])
        cons = {"type": "ineq", "fun": lambda x: x[-1] - S(x)}
        r = minimize(lambda x: x[-1], x0, constraints=[cons], method="SLSQP", options={"maxiter": 500, "ftol": 1e-14})
        if r.success and np.all(S(r.x) <= r.x[-1] + 1e-10):
            best = min(best, np.sqrt(np.max(S(r.x))))
    return best

def first_half(n, starts=400):
    K = n // 2; best = np.inf
    for _ in range(starts):
        x0 = np.concatenate([rng.normal(size=2 * K) * 0.7, [1.0]])
        def S(x):
            s = x[:K] + 1j * x[K:2 * K]
            return np.concatenate([np.abs(s) ** 2, [Fn(s, n)[0] ** 2]])
        cons = {"type": "ineq", "fun": lambda x: x[-1] - S(x)}
        r = minimize(lambda x: x[-1], x0, constraints=[cons], method="SLSQP", options={"maxiter": 500, "ftol": 1e-14})
        if r.success and np.all(S(r.x) <= r.x[-1] + 1e-10):
            best = min(best, np.sqrt(np.max(S(r.x))))
    return best

for n in (2, 3, 4, 5, 6, 7):
    print(f"(2) n={n}: direct multistart min_z max_k|s_k| = {direct(n):.8f}   first-half min = {first_half(n):.8f}", flush=True)
