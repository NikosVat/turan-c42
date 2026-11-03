#!/usr/bin/env python3
"""Exact check that every knot, 1/2, and every finite sum of knots lies on the grid (1/N)Z, and the
smallest such even N.  Sums of grid points are grid points, so it suffices that N*u_j is an integer for
every knot and N is even; this script verifies that explicitly (and also all pairwise/triple sums below 1).
Usage: python checkers/grid_check.py profile.json [N ...]"""
import json, math, sys
from fractions import Fraction
d = json.load(open(sys.argv[1]))
kn = [Fraction(x) for x in d.get("breaks", d.get("knots"))]
lcd = 1
for u in kn:
    lcd = lcd * u.denominator // math.gcd(lcd, u.denominator)
Nmin = lcd if lcd % 2 == 0 else 2 * lcd
print(f"{len(kn)} knots, lcm denominator {lcd}, smallest admissible even N = {Nmin}")
pts = set(kn) | {Fraction(1, 2)}
sums = set(pts)
for _ in range(2):  # sums of up to 3 knots, restricted to [0,1]
    sums |= {a + b for a in sums for b in pts if a + b <= 1}
for N in [int(x) for x in sys.argv[2:]] or [Nmin]:
    bad = [s for s in sums if (s * N).denominator != 1]
    print(f"N={N}: even={N % 2 == 0}, {len(sums)} points (knots, 1/2, sums of <=3 knots in [0,1]) checked, "
          f"off-grid: {len(bad)}  ->  {'OK' if N % 2 == 0 and not bad else 'NOT ALIGNED'}")
