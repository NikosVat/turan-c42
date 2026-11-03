"""Independent pointwise check (mpmath quadrature, 30 digits) of the formulas used by rigorous_check.py:
  R(t) = d(t) + int_0^t d(u) rho~(t-u) du - t rho~(t)   and   p~(t) = t^-gamma + int_0^t (t-v)^-gamma rho~(v) dv
evaluated directly from the trial coefficients (no convolution identity used).  Compares with the
witness' own float evaluation via the convolution identity."""
import sys, os, json, math, random
from fractions import Fraction
import numpy as np, mpmath as mp
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import spectral_witness as SW

mp.mp.dps = 30
prof, N, P = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
d = json.load(open(prof))
kn = d["breaks"] if "breaks" in d else [str(Fraction(j, 48)) for j in range(25)]
w = SW.Witness(d["T"], kn, d["phases"], N, P)
T = mp.mpf(Fraction(d["T"]).numerator) / Fraction(d["T"]).denominator
knf = [Fraction(k) for k in kn]; phf = [Fraction(p) for p in d["phases"]]
gam = T * mp.expj(mp.mpf(phf[0].numerator) / phf[0].denominator)
def phi(u):
    for j in range(len(knf) - 1):
        if knf[j] <= u <= knf[j + 1]:
            s = (phf[j + 1] - phf[j]) / (knf[j + 1] - knf[j])
            return mp.mpf(phf[j].numerator)/phf[j].denominator + (mp.mpf(s.numerator)/s.denominator) * (u - mp.mpf(knf[j].numerator)/knf[j].denominator)
    raise ValueError
def dfun(u):
    if u <= 0: return mp.mpc(0)
    if u > mp.mpf(1)/2: return gam
    return gam - T * mp.expj(phi(u))
coef = w.coef
def rho(t, side="left"):
    m = int(mp.floor(t * N))
    if m >= N: m = N - 1
    th = t * N - m
    return mp.polyval([mp.mpc(complex(c)) for c in coef[m][::-1]], th)
def conv(t):
    # breakpoints: knots, 1/2, t - grid points
    pts = set([mp.mpf(0), t])
    for k in knf + [Fraction(1, 2)]:
        x = mp.mpf(k.numerator) / k.denominator
        if 0 < x < t: pts.add(x)
    m = int(mp.floor(t * N))
    for j in range(m + 1):
        x = t - mp.mpf(j) / N
        if 0 < x < t: pts.add(x)
    pts = sorted(pts)
    tot = mp.mpc(0)
    for a_, b_ in zip(pts, pts[1:]):
        mid = (a_ + b_) / 2
        tot += mp.quad(lambda u: dfun(u) * rho_cell(t - u, t - mid), [a_, b_])
    return tot
def rho_cell(v, vref):
    m = int(mp.floor(vref * N)); th = v * N - m
    return mp.polyval([mp.mpc(complex(c)) for c in coef[m][::-1]], th)
random.seed(1)
cells = sorted(set([0, 1, 2, 3, N // 2 - 1, N // 2, N - 1] + random.sample(range(N), 6)))
print("cell  theta   |R/t| (mpmath direct)")
worst = 0
for m in cells:
    for th in (mp.mpf(1)/7, mp.mpf(1)/2, mp.mpf(6)/7):
        t = (m + th) / N
        Rv = dfun(t) + conv(t) - t * rho_cell(t, t)
        worst = max(worst, abs(Rv / t))
        print(m, float(th), mp.nstr(abs(Rv / t), 5))
print("max |R/t| at sampled points:", mp.nstr(worst, 5))

# ---- p~(t) direct vs the cell representation (float pipeline, same formulas as the Arb code)
F, p1, den, p_cell = SW.F_float(w, geo=45)
def p_direct(t):
    pts = sorted(set([mp.mpf(0), t] + [mp.mpf(j) / N for j in range(int(mp.floor(t * N)) + 1)]))
    tot = mp.mpc(0)
    for a_, b_ in zip(pts, pts[1:]):
        if a_ >= t: break
        mid = (a_ + b_) / 2
        tot += mp.quad(lambda v: (t - v) ** (-gam) * rho_cell(v, mid), [a_, b_])
    return t ** (-gam) + tot
print("cell theta  |p_direct - p_cell|  |p|")
worstp = 0
for m in [0, 1, 2, 3, 16, 17, 30, N // 2 - 1]:
    for th in (0.13, 0.5, 0.91):
        t = (m + mp.mpf(th)) / N
        pd = p_direct(t)
        pc = p_cell(m, np.array([th]))[0]
        worstp = max(worstp, abs(pd - mp.mpc(pc)))
        print(m, th, mp.nstr(abs(pd - mp.mpc(pc)), 4), mp.nstr(abs(pd), 6))
print("max |p_direct - p_cell|:", mp.nstr(worstp, 4))
p1d = p_direct(mp.mpf(1))
print("p~(1) direct", mp.nstr(p1d, 15), " cell-sum", p1, " diff", mp.nstr(abs(p1d - mp.mpc(p1)), 3))
