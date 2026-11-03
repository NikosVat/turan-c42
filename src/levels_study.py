"""Richardson-extrapolation study: F_N for N = N0*2^k and fits over consecutive triples / quadruples (float)."""
import sys, os, time, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import evaluator as ev
import fastsolve as fs

def Fx3(Ns, Fs, a):
    A = np.array([[1, N ** -(2 - a), N ** -2.0] for N in Ns]); return np.linalg.solve(A, np.array(Fs))[0]
def Fx4(Ns, Fs, a):
    A = np.array([[1, N ** -(2 - a), N ** -2.0, N ** -(3 - a)] for N in Ns]); return np.linalg.solve(A, np.array(Fs))[0]

if __name__ == "__main__":
    path = sys.argv[1]; N0 = int(sys.argv[2]); K = int(sys.argv[3])
    T = float(sys.argv[4]) if len(sys.argv) > 4 else None
    p = ev.load_profile(path)
    if T is not None: p.T = T
    a = float(np.real(p.g(0.0)))
    Ns, Fs = [], []
    for k in range(K):
        N = N0 * 2 ** k; t0 = time.time()
        F, _ = fs.forward(np.asarray(ev.gnodes_of(p, N)), N)
        Ns.append(N); Fs.append(F)
        print(f"N={N:8d} F_N={F:.15f} [{time.time()-t0:.1f}s]", flush=True)
    for i in range(K - 2):
        print(f"3-fit {Ns[i]}..{Ns[i+2]}: {Fx3(Ns[i:i+3], Fs[i:i+3], a):.13f}")
    for i in range(K - 3):
        print(f"4-fit {Ns[i]}..{Ns[i+3]}: {Fx4(Ns[i:i+4], Fs[i:i+4], a):.13f}")
