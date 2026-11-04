#!/usr/bin/env python3
"""Self-test for checkers/independent_check.py (checker #2): compares its exact/cellwise
representations with direct mpmath quadrature of the definitions, on a small toy profile.

  python checkers/independent_selftest.py [profile.json] [N]

Checks, at random points:
  * R(t) = d(t) + int_0^t d(u) rho~(t-u) du - t rho~(t)   (d exact, rho~ the certified witness)
    against the rigorous per-piece bound r >= |R/t|;
  * p~(t) = t^-gamma + int_0^t (t-v)^-gamma rho~(v) dv against the cell representation;
  * p~(1) against the closed form used by the checker.
"""
import json, random, sys, os
from fractions import Fraction
import numpy as np
import mpmath as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import independent_check as ic

prof = sys.argv[1] if len(sys.argv) > 1 else None
N = int(sys.argv[2]) if len(sys.argv) > 2 else 64
if prof is None:
    cert = dict(T=Fraction("0.6864"), C=Fraction("0.687"),
                u=[Fraction(x) for x in ["0", "1/64", "1/16", "1/8", "1/4", "1/2"]],
                psi=[Fraction(x) for x in ["-1.42", "-1.36", "-1.25", "-1.15", "-0.98", "-0.55"]],
                sha256="toy")
else:
    cert = ic.load_certificate(prof)
ch = ic.Checker(cert, N=N, P=10)
ch.hypotheses(); ch.build_d(); ch.witness(); ch.residual_bounds(); ch.error_bounds()
ch.numerator(); ch.kernel_polys(); ch.kernel_convolution(); ch.denominator()

mp.mp.dps = 40
T = mp.mpf(cert["T"].numerator) / cert["T"].denominator
u = [mp.mpf(x.numerator) / x.denominator for x in cert["u"]]
psi = [mp.mpf(x.numerator) / x.denominator for x in cert["psi"]]
gam = T * mp.exp(1j * psi[0])
h = mp.mpf(1) / N
W = mp.mpf(2) ** ch.Wr
coef = [[(mp.mpf(int(ch.Cr[m, b])) + 1j * mp.mpf(int(ch.Ci[m, b]))) / W for b in range(ch.P + 1)] for m in range(N)]

def phi(x):
    for j in range(len(u) - 1):
        if x <= u[j + 1]:
            return psi[j] + (psi[j + 1] - psi[j]) * (x - u[j]) / (u[j + 1] - u[j])
    return psi[-1]

def d(x):
    if x <= 0: return mp.mpc(0)
    if x <= mp.mpf(1) / 2: return gam - T * mp.exp(1j * phi(x))
    return gam

def rho(v, side=+1):
    m = int(mp.floor(v / h))
    th = v / h - m
    if m >= N: m, th = N - 1, mp.mpf(1)
    return mp.polyval(coef[m][::-1], th)

def conv(f, t):
    # int_0^t f(u) rho(t-u) du, split at grid points of u and of t-u, and at knots
    pts = set([mp.mpf(0), t])
    k = 0
    while k * h < t:
        pts.add(k * h); pts.add(t - k * h); k += 1
    for x in u:
        if x < t: pts.add(x)
    pts = sorted(p for p in pts if 0 <= p <= t)
    s = 0
    for a, b in zip(pts[:-1], pts[1:]):
        if b - a > 0:
            mid = (a + b) / 2
            s += mp.quad(lambda x: f(x) * rho(t - x), [a, b])
    return s

random.seed(1)
worst = 0
print("residual checks (|R/t| vs rigorous r):")
for trial in range(6):
    m = random.choice([0, 1, 2, 3, N // 4, N // 2 - 1, N // 2 + 1, N - 1])
    th = mp.mpf(random.random())
    t = h * (m + th)
    R = d(t) + conv(d, t) - t * rho(t)
    rb = [p[2] for p in ch.pieces if p[0] <= Fraction(str(float(t))) <= p[1]]
    print("  t=%.6f  |R/t| = %.3e   r = %.3e" % (float(t), float(abs(R / t)), float(max(rb))))
    assert abs(R / t) <= max(rb)
print("p~ representation checks:")
for trial in range(6):
    m = random.choice([1, 2, 3, 5, N // 8, N // 4 + 1, N // 2 - 1])
    th = random.random()
    t = h * (m + mp.mpf(th))
    pt = t ** (-gam) + conv(lambda x: x ** (-gam), t)
    rep = ch.ptilde_float(m, th)
    print("  m=%d th=%.3f  |direct - repr| = %.2e  (|p|=%.3f)" % (m, th, abs(complex(pt) - rep), abs(complex(pt))))
    assert abs(complex(pt) - rep) < 1e-11
s = 1
for k in range(N):
    s += mp.quad(lambda v: (1 - v) ** (-gam) * mp.polyval(coef[k][::-1], v / h - k), [k * h, (k + 1) * h])
print("p~(1): direct %s, checker %s, diff %.2e" % (mp.nstr(s, 18), ch.p1.str(18), float(abs(s - complex(float(ch.p1.real), float(ch.p1.imag))))))
# denominator: float Gauss-Legendre estimate vs certified lower bound
xg, wg = np.polynomial.legendre.leggauss(30)
th = (xg + 1) / 2
D = 0.0
for m in range(1, N // 2):
    v = np.array([ch.ptilde_float(m, x) for x in th])
    D += np.sum(wg / 2 * np.abs(v) / (1 - (m + th) / N)) / N
# cell 0 in mpmath (theta^-a singularity)
c0 = coef[0]
D0 = mp.quad(lambda t: abs(t ** (-gam) * (1 + h * sum(c0[b] * mp.beta(b + 1, 1 - gam) * (t / h) ** (b + 1) for b in range(ch.P + 1)))) / (1 - t), [0, h])
print("denominator: GL estimate %.15f, certified D_low %s, estimate - D_low = %.2e" %
      (D + float(D0), ch.Dlow.str(16), D + float(D0) - float(ch.Dlow.mid())))
print("SELFTEST OK")
