import numpy as np
import matplotlib.pyplot as plt

# derived on paper
def C_case1(s1, s2):
    return np.array([[s1**2, 0],
                      [0,     s2**2]])

def C_case2a(s1, s2):
    # g = (y, z^2)
    return np.array([[s1**2, 0],
                      [0,     0]])

def C_case2b(s1, s2):
    # g = (y, z^3)
    return np.array([[s1**2, 0],
                      [0,     3*s2**4]])

def C_case3(s1, s2):
    # g = (y, y^3)
    return np.array([[s1**2,    0],
                      [3*s1**4, 0]])

def C_false_positive(s, a=2.0, b=1.0, c=1.0):
    # Structurally unobservable everywhere (Jacobian rank <= 1 for all y,z),
    # but falsely shows full rank at any fixed sigma.
    # Construction: w = a*y + b*z + c*y*z,  h = (w, w^2)
    return np.array([[a*s**2,        b*s**2],
                      [2*b*c*s**4,    2*a*c*s**4]])

cases = {
    "Case 1 (observable):": C_case1,
    "Case 2a (weakly observable):": C_case2a,
    "Case 2b (weakly observable):": C_case2b,
    "Case 3 (not observable):": C_case3,
    "False positive":  C_false_positive,
}

# vary the variance and loop
s1_fixed = 1.0
sigma2 = np.linspace(0.05, 3.0, 200)

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
axes = axes.ravel()

for ax, (name, Cfun) in zip(axes, cases.items()):
    sv_max, sv_min, cond = [], [], []
    for s2 in sigma2:
        C = Cfun(s1_fixed, s2)
        svals = np.linalg.svd(C, compute_uv=False)
        sv_max.append(svals[0])
        sv_min.append(svals[-1])

    ax.plot(sigma2, sv_max, label=r"$\sigma_{\max}(C)$")
    ax.plot(sigma2, sv_min, label=r"$\sigma_{\min}(C)$")
    ax.set_title(name, fontsize=9)
    ax.set_xlabel(r"$\sigma_2$")
    ax.set_ylabel("singular value")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("cross-covariance.png", dpi=150, bbox_inches="tight")
plt.show()
















