import mpmath as mp
mp.mp.dps = 40
T = mp.mpf('0.69367'); g = T * mp.expj(mp.mpf('0.8977')); a = mp.re(g)
p1 = 1 + mp.quad(lambda v: (1 - v) ** (-g) * g / v, [0.5, 1])
den = mp.quad(lambda t: t ** (-a) / (1 - t), [0, 0.5])
print("exact F =", mp.nstr(abs(p1) / den, 25), " p1 =", mp.nstr(p1, 20), " den =", mp.nstr(den, 20))
