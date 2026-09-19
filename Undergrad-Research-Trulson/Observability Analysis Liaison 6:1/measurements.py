import numpy as np
from orbits import twob_scenario

def range_measure(x): # scalar range between the two satellites from full state
    r1, r2 = x[0:3], x[6:9]
    delta_r = r2 - r1
    return np.linalg.norm(delta_r) #  return magnitude of range

def range_jacobian(x): # jacobian of range with respect to full 12-d state
    r1, r2 = x[0:3], x[6:9]
    delta_r = r2 - r1
    rho = np.linalg.norm(delta_r)

    H = np.zeros((1,12)) # derived on paper, returns jacobian of range
    H[0, 0:3] = -delta_r/rho #[deltax/rho, deltay/rho, deltaz/rho] respect to satellite 1
    H[0, 6:9] = delta_r/rho # same as above, but with respect to satellite 2
    return H # changing velocity does not change current distance/range, ∂ρ/∂v = 0


def angle_measure(x): # line of sight angles from satellite 1 to 2
    r1, r2 = x[0:3], x[6:9]
    los = r2 - r1 # line-of-sight vector
    rho = np.linalg.norm(los) # range = ||los||
    l_hat = los / rho # los unit vector

    elevation = np.arcsin(np.clip(l_hat[2], -1.0, 1.0)) #arcsin(z/l_hat)
    azimuth = np.arctan2(l_hat[1], l_hat[0]) #arcttan(y/x)

    return np.array([azimuth, elevation])

def angle_jacobian(x):
    r1, r2 = x[0:3], x[6:9]
    los = r2 - r1
    rho = np.linalg.norm(los)
    dx, dy, dz  = los

    # short-hands to simplify calculations
    dxy2 = dx ** 2 + dy ** 2

    # derived on paper:
    # change in azimuth with respect to x, y, z
    dax = -dy / dxy2
    day = dx / dxy2
    da = np.array([dax,day, 0])

    # change in elevation with respect to x, y, z
    dex = (-dx*dz) / (rho**2 * np.sqrt(dxy2))
    dey = (-dy*dz) / (rho**2 * np.sqrt(dxy2))
    dez = np.sqrt(dxy2) / (rho**2)
    de = np.array([dex, dey, dez])

    # assemble jacobian with respect to each satellite
    H = np.zeros((2, 12))
    H[0, 0:3] = -da
    H[1, 0:3] = -de
    H[0, 6:9] = da
    H[1, 6:9] = de

    return H

def jacobian_fd(H_func, x, epsilon = 1): # epsilon = 1 meter
    H_0 = H_func(x)
    m = len(np.atleast_1d(H_0)) # set dimensions for jacobian, make sure scalars convert into 1-element arrays
    n = len(x)
    H_fd = np.zeros((m, n)) # either 2 x 12 or 1 x 12 depending on range/angle

    for j in range(n):
        step = epsilon if j < 6 else epsilon * 1e-3  # smaller step for velocities
        xpos, xneg = x.copy(), x.copy()
        xpos[j] += step
        xneg[j] -= step
        H_fd[:, j] = (np.atleast_1d(H_func(xpos)) - np.atleast_1d(H_func(xneg))) / (2 * step) # central finite difference formula
        # adjust all measurements according to perturbation
    return H_fd


def compute_measurement_history(r1_hist, v1_hist, r2_hist, v2_hist):
    # evaluate both measurement models at every time step.
    N = len(r1_hist)
    rho_hist = np.zeros(N)
    angle_hist = np.zeros((N, 2))
    H_range = np.zeros((N, 1, 12))
    H_angle = np.zeros((N, 2, 12))

    for k in range(N):
        x = np.concatenate([r1_hist[k], v1_hist[k], r2_hist[k], v2_hist[k]])
        rho_hist[k] = range_measure(x)
        angle_hist[k] = angle_measure(x)
        H_range[k] = range_jacobian(x)
        H_angle[k] = angle_jacobian(x)

    return rho_hist, angle_hist, H_range, H_angle


def verify_jacobians(x, verbose = True): # compare derivations with finite difference jacobians
    H_range = range_jacobian(x)
    H_range_fd = jacobian_fd(lambda s: np.array([range_measure(s)]), x) # temporary function with an array of range measure, not just a scalar
    H_angle = angle_jacobian(x)
    H_angle_fd = jacobian_fd(angle_measure, x)

    # find errors
    err_range = np.abs(H_range - H_range_fd)
    err_angle = np.abs(H_angle - H_angle_fd)

    if verbose:
        print("Jacobian Verification")
        print(f"Range  — max |analytic - FD| : {err_range.max():.2e}")
        print(f"Angles — max |analytic - FD| : {err_angle.max():.2e}")
        print("\nRange H (analytic) — non-zero columns [r1x,r1y,r1z, r2x,r2y,r2z]:")
        print(f"  r1: {H_range[0, 0:3]}")
        print(f"  r2: {H_range[0, 6:9]}")
        print(f"  v1: {H_range[0, 3:6]}  (should be zero)")
        print(f"  v2: {H_range[0, 9:12]} (should be zero)")
        print("\nAngle H rows [az; el] — r1 and r2 blocks:")
        print(f"  az/r1: {H_angle[0, 0:3]}")
        print(f"  az/r2: {H_angle[0, 6:9]}")
        print(f"  el/r1: {H_angle[1, 0:3]}")
        print(f"  el/r2: {H_angle[1, 6:9]}")

    return err_range, err_angle


# test
if __name__ == "__main__":
    print("Measurement Model")

    # Build trajectory from orbits
    t, r1, v1, r2, v2, T = twob_scenario(n_points=500)

    # Pick a single state vector for Jacobian checks
    k_test = 100
    x_test = np.concatenate([r1[k_test], v1[k_test], r2[k_test], v2[k_test]])

    #Measurement values
    rho = range_measure(x_test)
    alpha = angle_measure(x_test)
    print(f"  Range       ρ  = {rho:.2f} km")
    print(f"  Range       ρ  = {rho} km")
    print(f"  Azimuth     az = {np.rad2deg(alpha[0]):.3f} °")
    print(f"  Elevation   el = {np.rad2deg(alpha[1]):.3f} °")

    # Jacobian verification
    verify_jacobians(x_test)

