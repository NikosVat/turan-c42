"""mpmath check of |R/t| on the first cells (direct quadrature) for a given witness."""
import sys, os, json
from fractions import Fraction
import numpy as np, mpmath as mp
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import spectral_witness as SW
mp.mp.dps = 30
prof, N, P = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); cells = [int(x) for x in sys.argv[4].split(",")]
d = json.load(open(prof)); kn = d["breaks"] if "breaks" in d else [str(Fraction(j, 48)) for j in range(25)]
w = SW.Witness(d["T"], kn, d["phases"], N, P)
T = mp.mpf(Fraction(d["T"]).numerator) / Fraction(d["T"]).denominator
knf = [Fraction(k) for k in kn]; phf = [Fraction(p) for p in d["phases"]]
F2m = lambda q: mp.mpf(q.numerator) / q.denominator
gam = T * mp.expj(F2m(phf[0]))
def dfun(u):
    if u <= 0: return mp.mpc(0)
    for j in range(len(knf) - 1):
        if F2m(knf[j]) <= u <= F2m(knf[j + 1]):
            s = (phf[j + 1] - phf[j]) / (knf[j + 1] - knf[j])
            return gam - T * mp.expj(F2m(phf[j]) + F2m(s) * (u - F2m(knf[j])))
    return gam
cf = [[mp.mpc(complex(c)) for c in row[::-1]] for row in w.coef[:max(cells) + 2]]
rc = lambda v, m: mp.polyval(cf[m], v * N - m)
for m in cells:
    best = 0
    for th in np.linspace(0.02, 0.98, 13):
        t = (m + mp.mpf(th)) / N
        pts = sorted(set([mp.mpf(0), t] + [F2m(k) for k in knf if 0 < F2m(k) < t] + [t - mp.mpf(j) / N for j in range(m + 1) if t - mp.mpf(j) / N > 0]))
        conv = sum(mp.quad(lambda u: dfun(u) * rc(t - u, int(mp.floor((t - (a_ + b_) / 2) * N))), [a_, b_]) for a_, b_ in zip(pts, pts[1:]))
        best = max(best, abs((dfun(t) + conv - t * rc(t, m)) / t))
    print(m, mp.nstr(best, 4), flush=True)
