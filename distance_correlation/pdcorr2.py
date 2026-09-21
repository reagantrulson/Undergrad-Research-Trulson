import numpy as np
from pdcorr import pdcor, dcor_u

def r(u, v):
    return np.corrcoef(u, v)[0, 1]
# regular partial correlation
def partial_pearson(A, B, Z):
    rAB, rAZ, rBZ = r(A, B), r(A, Z), r(B, Z)
    return (rAB - rAZ*rBZ) / np.sqrt((1-rAZ**2)*(1-rBZ**2))

def battery(A, B, Z, label):
    print(label)
    print("raw pearson   =", round(r(A, B), 4))
    print("raw dcor      =", round(dcor_u(A, B), 4))
    print("pcorr(;Z)     =", round(partial_pearson(A, B, Z), 4))
    print("pdCor(;Z)     =", round(pdcor(A, B, Z), 4))
    print()

def full_C_comparison(g1, g2, y, z, label):
    print(label)
    C_pearson = np.array([
        [r(g1, y), r(g1, z)],
        [r(g2, y), r(g2, z)]
    ])
    C_dcor = np.array([
        [dcor_u(g1, y), dcor_u(g1, z)],
        [dcor_u(g2, y), dcor_u(g2, z)]
    ])
    print("Pearson-based C:\n", np.round(C_pearson, 4))
    print("dCor-based C:\n", np.round(C_dcor, 4))
    print()

# false positive case
rng = np.random.default_rng(7)
a, b, c = 2.0, 1.0, 1.0
s1, s2 = 1.0, 1.7
n = 1500

y = rng.normal(0, s1, n)
z = rng.normal(0, s2, n)
w = a*y + b*z + c*y*z
h1, h2 = w, w**2

battery(h2, y, h1, "False positive: h2 vs y, controlling for h1")
battery(h2, z, h1, "False positive: h2 vs z, controlling for h1")

# weakly observable cases
# case2a
rho = 0   # tune this: 0 = your original independent case, closer to 1 = strong confound
z_corr = rho*y + np.sqrt(1-rho**2)*rng.normal(0, s2, n)

g1, g2 = y, z_corr**2
battery(g2, z_corr, y, "Case 2a (confounded): g2=z^2 vs z, controlling for y")

# case2b
g1, g2 = y, z_corr**3
battery(g2, z_corr, y, "Case 2b (confounded): g2=z^3 vs z, controlling for y")

full_C_comparison(y, z_corr**2, y, z_corr, "Case 2a: Pearson C vs dCor C")
