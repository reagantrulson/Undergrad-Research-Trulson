import numpy as np
from matplotlib import pyplot as plt
from crosscorr import cases
# partial correlation test
# set up with derivations and conditions that make it valid
a, b, c = 2, 1, 1
y = np.array([1, -1, 2, 0, -2])
z = np.array([1, 2, -1, -2, 0])
# arbitrary nonlinear set up
w = a*y + b*z + c*y*z
# 1st and second observables for nonlinear situation
h_1 = w
h_2 = w**2
# pearson coefficients
r_h2_y  = np.corrcoef(h_2, y)[0, 1]
r_h2_h1 = np.corrcoef(h_2, h_1)[0, 1]
r_y_h1  = np.corrcoef(y, h_1)[0, 1]
r_h2_z  = np.corrcoef(h_2, z)[0, 1]
r_z_h1  = np.corrcoef(z, h_1)[0, 1]
# regular partial correlations
pcorr_h2_y = (r_h2_y - r_h2_h1 * r_y_h1) / np.sqrt((1-r_h2_h1**2)*(1-r_y_h1**2))
print(pcorr_h2_y)
pcorr_h2_z = (r_h2_z - r_h2_h1 * r_z_h1) / np.sqrt((1-r_h2_h1**2)*(1-r_z_h1**2))
print(pcorr_h2_z)

rng = np.random.default_rng(0)


# building delay embeddings
def delay_embedding(signal, tau, d):

    N = len(signal) - (d-1)*tau

    return np.column_stack([
        signal[i*tau : i*tau+N]
        for i in range(d)
    ])

# all things used for biased distance correlation
# euclidean distance matrix
def pairwise_dist(X):
    X = np.atleast_2d(X)
    if X.shape[0] == 1:  # allow column vectors passed as 1-D
        X = X.reshape(-1, 1)
    diff = X[:, None, :] - X[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))

# double centering (can be used in regular partial correlation)
def double_center(a):
    # subtract row mean and column mean and add back whole mean to center distance matrix
    n = a.shape[0]
    row_mean = a.mean(axis=1, keepdims=True)
    col_mean = a.mean(axis=0, keepdims=True)
    grand_mean = a.mean()
    return a - row_mean - col_mean + grand_mean

def dcov2_biased(A, B):
    n = A.shape[0]
    return (A * B).sum() / n**2

def dcor_biased(X, Y):
    # take distance matrices, center them, make biased distance correlation with 1/n^2 inner product
    A = double_center(pairwise_dist(X))
    B = double_center(pairwise_dist(Y))
    num = dcov2_biased(A, B)
    denom = np.sqrt(dcov2_biased(A, A) * dcov2_biased(B, B))
    return np.sqrt(num / denom) if denom > 0 else 0.0


# all things used in u-centered unbiased distance correlation
# u-centering from u-statistics directly from paper, eq. (2.1)-(2.2)
def u_center(a):
    n = a.shape[0]
    row_sum = a.sum(axis=1)
    col_sum = a.sum(axis=0)
    total = a.sum()

    A = (
        a
        - row_sum[:, None] / (n - 2)
        - col_sum[None, :] / (n - 2)
        + total / ((n - 1) * (n - 2))
    )
    np.fill_diagonal(A, 0.0)
    return A
# inner product that is in the right hilbert space now because of u centering
def inner_product_u(A, B):
    # Eq. (2.3)
    n = A.shape[0]
    return (A * B).sum() / (n * (n - 3))

# unbiased distance correlation
def dcor_u(X, Y):
    # take distance matrices, center them with u centering, make unbiased distance correlation with inner product
    A = u_center(pairwise_dist(X))
    B = u_center(pairwise_dist(Y))
    numerator = inner_product_u(A, B) # how similar are the distance patterns in X and Y?
    denom = np.sqrt(inner_product_u(A, A) * inner_product_u(B, B)) # normalization of above
    return numerator / denom if denom > 0 else 0.0 #"cosine formula"


# partial distance correlation
def pdcor(X, Y, Z):
    """#Partial distance correlation of X and Y given Z,
    via the projection formula (Szekely & Rizzo 2014, Thm 3):

        R*(X,Y;Z) = (Rxy - Rxz*Ryz) / sqrt(1-Rxz^2) / sqrt(1-Ryz^2)

    where R* is the UNBIASED distance correlation from Step 3.
    """
    # find unbiased distance correlation of each piece
    Rxy = dcor_u(X, Y)
    Rxz = dcor_u(X, Z)
    Ryz = dcor_u(Y, Z)
    denom = np.sqrt(1 - Rxz**2) * np.sqrt(1 - Ryz**2)
    return (Rxy - Rxz * Ryz) / denom if denom != 0 else np.nan

def run(x, y, z, title):
    print("\n"+ title)
    print("Pearson correlations")
    print("corr(X,Y)      =", np.round(np.corrcoef(x, y)[0, 1], 4))
    print("corr(X,Z)      =", np.round(np.corrcoef(x, z)[0, 1], 4))
    print("corr(Y,Z)      =", np.round(np.corrcoef(y, z)[0, 1], 4))

    # classical partial correlation, closed form, as a ground-truth check
    rxy, rxz, ryz = [np.corrcoef(a, b)[0, 1] for a, b in [(x, y), (x, z), (y, z)]]
    partial_pearson = (rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    print("partial corr(X,Y;Z) [Pearson] =", np.round(partial_pearson, 4))

    print("\nDistance correlation")
    print("dCor(X,Y) [biased, always >=0]   =", np.round(dcor_biased(x, y), 4))
    print("R*(X,Y) [unbiased, can be neg]    =", np.round(dcor_u(x, y), 4))

    print("\nPartial distance correlation")
    pd_formula = pdcor(x, y, z)
    print("pdCor(X,Y;Z) via Thm-3 formula   =", np.round(pd_formula, 4))

def henon(n, a=1.4, b=0.3):
    x = np.zeros(n)
    y = np.zeros(n)
    x[0] = 0.1
    y[0] = 0.0
    for k in range(n - 1):
        x[k + 1] = 1 - a * x[k] ** 2 + y[k]
        y[k + 1] = b * x[k]

    return x, y

def sigma_min(C):
    return np.linalg.svd(C, compute_uv=False)[-1]

if __name__ == "__main__":
    n = 1000

    # case 1: noisy functions of shared confounder z, no direct x-y link
    Z_1 = rng.normal(size=n)
    X_1 = 0.8 * Z_1 + rng.normal(scale=0.6, size=n)
    Y_1 = 0.8 * Z_1 + rng.normal(scale=0.6, size=n)
    run(X_1, Y_1, Z_1,"Case 1: Noisy uncorrelated")

    # case 2: independent gaussians
    X_2 = np.random.normal(size=n)
    Y_2 = np.random.normal(size=n)
    Z_2 = rng.normal(size=n)
    run(X_2, Y_2, Z_2, "Case 2: Independent Gaussians")

    # case 3: linear relationship
    X_3 = np.random.normal(size=n)
    Y_3 = 3 * X_3 + 0.1 * np.random.normal(size=n)
    run(X_3, Y_3, Z_2, "Case 3: Linear Relationship")

    # case 4: quadratic
    X_4 = np.random.normal(size=n)
    Y_4 = X_4 ** 2
    run(X_4, Y_4, Z_2, "Case 4: Quadratic Relationship")

    # case 5: circle
    theta = np.random.uniform(0, 2 * np.pi, n)
    X_5 = np.cos(theta)
    Y_5 = np.sin(theta)
    run(X_5, Y_5, Z_2, "Case 4: Circle")


sigmas = np.logspace(-3, -0.3, 30)  # from 0.001 to ~0.5, log-spaced

sweep_results = {}
for name, Cfunc in cases.items():
    smins = np.array([sigma_min(Cfunc(s, s)) for s in sigmas])
    sweep_results[name] = smins

print("Decay exponent/log-log slope of sigma_min(C) vs sigma")
slopes = {}
for name, smins in sweep_results.items():
    if np.all(smins < 1e-14):
        print(f"{name:28s}  identically 0 (undefined slope - structural unobservability)")
        slopes[name] = None
        continue
    slope, intercept = np.polyfit(np.log(sigmas), np.log(smins + 1e-300), 1)
    slopes[name] = slope
    print(f"{name:28s}  slope = {slope:.4f}")

# plot regular and log-log plots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for name, smins in sweep_results.items():
    axes[0].plot(sigmas, smins, label=name)
    if np.any(smins > 1e-14):
        axes[1].plot(sigmas, smins, label=name,markevery=5, markersize=5,
                     linewidth=2, alpha=0.85)


axes[0].set_xlabel("sigma")
axes[0].set_ylabel("sigma_min(C)")
axes[0].set_title("Regular scale")
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3)

axes[1].set_xscale("log")
axes[1].set_yscale("log")
axes[1].set_xlabel("sigma (log scale)")
axes[1].set_ylabel("sigma_min(C) (log scale)")
axes[1].set_title("Log-log scale (slope = decay exponent)")
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3, which="both")

plt.tight_layout()
plt.savefig("sigma_decay.png", dpi=150, bbox_inches="tight")
plt.show()
