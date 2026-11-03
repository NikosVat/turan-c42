"""Re-evaluate profiles with 2-exponent Richardson extrapolation at user-chosen N levels (float, not rigorous)."""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from evaluator import load_profile, F
Ns = [int(x) for x in sys.argv[1].split(",")]
for path in sys.argv[2:]:
    p = load_profile(path); a = float(np.real(p.g(0.0))); t0 = time.time()
    Fs = [F(p, N, "np") for N in Ns]
    A = np.array([[1, N ** -(2 - a), N ** -2.0] for N in Ns]); Fx = np.linalg.solve(A, np.array(Fs))[0]
    print(f"{path}: knots={len(p.knots)} T={p.T} levels={Ns} F_N={['%.12f' % f for f in Fs]} extrap={Fx:.12f} "
          f"[{time.time() - t0:.0f}s]", flush=True)
