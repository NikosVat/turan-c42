"""Direction-E probe (float, NOT rigorous): at finite n, optimise ALL first-half phases s_k = T e^{i theta_k},
k <= n/2, to minimise F_n = |P_n| / sum_{k>n/2} |P_{n-k}|/k (exact Harcos reduction), and compare with the
smooth profile s_k = g(k/n) at the same n.  Usage: python src/finite_n_free.py PROFILE.json n [iters]"""
import sys, os, json, time, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import jax, jax.numpy as jnp
from scipy.optimize import minimize
from evaluator import load_profile
jax.config.update("jax_enable_x64", True)

def make_Fn(n):
    K = n // 2
    m = jnp.arange(1, n + 1)
    def Fn(theta, T):
        s = T * jnp.exp(1j * theta)                     # s_1..s_K
        sfull = jnp.zeros(n + 1, complex).at[1:K + 1].set(s)
        # (m) a_m + sum_{k=1}^{m} s_k a_{m-k} = 0, a_0 = 1  -> lower-triangular Toeplitz-plus-diagonal system
        i = jnp.arange(1, n + 1)[:, None]; j = jnp.arange(1, n + 1)[None, :]
        L = jnp.where(j < i, sfull[jnp.clip(i - j, 0, n)], 0.0) + jnp.diag(m.astype(complex))
        a = jax.scipy.linalg.solve_triangular(L, -sfull[1:], lower=True)
        P = jnp.cumsum(jnp.concatenate([jnp.ones(1, complex), a]))
        kf = jnp.arange(K + 1, n + 1)
        return jnp.abs(P[n]) / jnp.sum(jnp.abs(P[n - kf]) / kf)
    return jax.jit(Fn), jax.jit(jax.value_and_grad(Fn))

if __name__ == "__main__":
    prof = load_profile(sys.argv[1]); n = int(sys.argv[2]); iters = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    K = n // 2; T = prof.T
    th0 = np.unwrap(np.angle(np.asarray(prof.g(np.arange(1, K + 1) / n))))
    Fn, vg = make_Fn(n)
    f0 = float(Fn(th0, T)); t0 = time.time()
    res = minimize(lambda x: tuple(np.asarray(v, dtype=float) for v in vg(x, T)), th0, jac=True, method="L-BFGS-B",
                   options={"maxiter": iters, "ftol": 1e-15, "gtol": 1e-12})
    d = res.x - th0
    print(f"n={n} T={T}: smooth-profile F_n = {f0:.8f}; free-phase optimum F_n = {res.fun:.8f} "
          f"(delta {res.fun - f0:.3e}); max|theta-theta_smooth| = {np.max(np.abs(d)):.3e}; "
          f"rms of 2nd difference of theta: smooth {np.sqrt(np.mean(np.diff(th0, 2)**2)):.2e}, "
          f"opt {np.sqrt(np.mean(np.diff(res.x, 2)**2)):.2e}; {res.nit} it, {time.time() - t0:.0f}s", flush=True)
    np.save(f"results/finite_n_free_n{n}.npy", res.x)
