import numpy as np
from scipy.integrate import solve_ivp
from orbits import twob_scenario, two_body_ode, MU_EARTH
from measurements import compute_measurement_history

def gravity_gradient_tensor(r, mu = MU_EARTH): # derived on paper, ∂²U/∂r² or ∂F/∂r
    r_norm = np.linalg.norm(r)
    r_norm3 = r_norm ** 3
    r_norm5 = r_norm ** 5

    I_3 = np.eye(3)
    rrt = np.outer(r, r)  # r · r^T, outer product 3x3 matrix

    G = (-mu / r_norm3) * I_3 + (3 * mu / r_norm5) * rrt
    return G # how much gravity changes when a satellite moves slightly

def dynamics_jacobian(r, mu = MU_EARTH): # full 6x6 dynamics Jacobian = ∂f/∂x for two-body gravity
    A = np.zeros((6, 6))
    A[0:3, 3:6] = np.eye(3) # top right
    A[3:6, 0:3] = gravity_gradient_tensor(r, mu) # gravity gradient bottom left
    return A

def stm_ode(t, z, mu = MU_EARTH): # ode for combined state stm system, t is current time, z is stacked state + stm
    x = z[0:6]
    phi = z[6:42].reshape(6,6) # unflatten stm

    # state derivative
    dx = two_body_ode(t, x, mu)

    #stm derivative
    r = x[0:3]
    A = dynamics_jacobian(r, mu)
    dphi = A @ phi

    return np.concatenate([dx, dphi.flatten()])

def propagate_stm(r_0, v_0, t_span, t_eval, mu=MU_EARTH):
    x_0 = np.concatenate([r_0, v_0])
    phi_0 = np.eye(6).flatten() # long vector inside ode solver
    z_0 = np.concatenate([x_0, phi_0])

    sol = solve_ivp(stm_ode,t_span, z_0,t_eval = t_eval, method = "RK45",rtol=1e-10, atol=1e-12,args=(mu,))

    r_hist = sol.y[0:3].T # all positions for each t
    v_hist = sol.y[3:6].T # all velocities for each t
    phi_hist = sol.y[6:42].T.reshape(-1, 6, 6) # unflatten, all stms for each t

    return r_hist, v_hist, phi_hist

def verify_stm(r_0, v_0, t_span, t_eval, delta_scale = 1.0, mu=MU_EARTH):
    # original propagation
    r_n, v_n, phi_hist = propagate_stm(r_0, v_0, t_span, t_eval, mu)

    dx_0 = np.zeros(6)
    dx_0[0] = delta_scale # create tiny perturbation at beginning
    r_0p = r_0 + dx_0[0:3]
    v_0p = v_0 + dx_0[3:6]
    # propagate the perturbation, nonlinear trajectory
    r_pert, v_pert, _ = propagate_stm(r_0p, v_0p, t_span, t_eval, mu)

    # deviation
    dr = r_pert - r_n
    dv = v_pert - v_n
    dx = np.hstack([dr, dv])

    # stm-predicted deviation
    dx_stm = np.einsum('nij,j->ni', phi_hist, dx_0)

    d = dx  - dx_stm
    errors = np.linalg.norm(d, axis=1) / np.linalg.norm(dx_0)

    return errors

def build_combined_stm(phi1_hist, phi2_hist):
    N = len(phi1_hist)
    phi_comb = np.zeros((N,12,12))
    phi_comb[:, 0:6, 0:6] = phi1_hist
    phi_comb[:, 6:12, 6:12] = phi2_hist
    return phi_comb #(N,12,12)  block-diagonal combined STM

def build_observability_matrix(phi_hist, h_hist, t_eval):
    N = len(t_eval)
    integrand = np.zeros((N, 12, 12))
    # integrate gramian, W = integral(phi^TH^THphi dt)
    for k in range(N):
        phi_k = phi_hist[k]
        H_k = h_hist[k]
        HTH = H_k.T @ H_k
        integrand[k] = phi_k.T @ HTH @ phi_k # build matrix with values that are inside the integral at each t

    W_hist = np.zeros((N, 12, 12))
    W = np.zeros((12, 12))

    # approximate the integral W with the trapezoidal rule
    for k in range(1, N):
        dt = t_eval[k] - t_eval[k - 1]
        dW = 0.5 * (integrand[k] + integrand[k - 1]) * dt # trapezoidal rule, approximates W for each t_k-1 to t_k
        W += dW # add each approximation
        W_hist[k] = W.copy()

    return W, W_hist, integrand

def analyze(W, label="", verbose=True):
    U, s, v = np.linalg.svd(W)
    rank = np.linalg.matrix_rank(W)
    cond = s[0] / s[-1] if s[-1] > 0 else np.inf
    null_vectors = v[rank:]

    if verbose:
        print(f"Gramian analysis — {label}")
        print(f"  Rank              : {rank} / 12")
        print(f"  Condition number  : {cond:.3e}")
        print(f"  Singular values   :")

        for i, sv in enumerate(s):
            print(f"σ{i + 1:02d} = {sv:12.4e}")

        if len(null_vectors) > 0:
            print(f"\n  Null space ({len(null_vectors)} vector(s)):")
            print("  These state directions are unobservable:")

            labels = [
                'r1x', 'r1y', 'r1z',
                'v1x', 'v1y', 'v1z',
                'r2x', 'r2y', 'r2z',
                'v2x', 'v2y', 'v2z'
            ]

            for i, nv in enumerate(null_vectors):
                dominant = np.argsort(np.abs(nv))[::-1][:3]

                print(
                    f"    null[{i}]: dominant components = "
                    f"{labels[dominant[0]]}({nv[dominant[0]]:.3f}), "
                    f"{labels[dominant[1]]}({nv[dominant[1]]:.3f}), "
                    f"{labels[dominant[2]]}({nv[dominant[2]]:.3f})"
                )

    return rank, s, null_vectors


if __name__ == "__main__":

    print("STM Integration & Observability Gramian")

    # build trajectory
    N_POINTS = 500
    t, r1, v1, r2, v2, T = twob_scenario(n_points=N_POINTS)
    t_span = (t[0], t[-1])

    # propagate stms
    r1_s, v1_s, Phi1 = propagate_stm(r1[0], v1[0], t_span, t, MU_EARTH)
    r2_s, v2_s, Phi2 = propagate_stm(r2[0], v2[0], t_span, t, MU_EARTH)

    # verify stm
    rel_err = verify_stm(r1[0], v1[0], t_span, t)
    print(f"  Max relative error over orbit: {rel_err.max():.3e}")
    print(f"  Mean relative error          : {rel_err.mean():.3e}")
    print(f"  (Should be << 1 for 1m perturbation on ~7000 km orbit)")

    # combine stm
    Phi_combined = build_combined_stm(Phi1, Phi2)
    print(f"\nCombined STM shape: {Phi_combined.shape}  (N, 12, 12)")

    # measurement jacobians
    _, _, H_range_hist, H_angle_hist = compute_measurement_history(r1, v1, r2, v2)
    print(f"  H_range shape: {H_range_hist.shape}   (N, 1, 12)")
    print(f"  H_angle shape: {H_angle_hist.shape}   (N, 2, 12)")

    # build gramians
    W_range, W_range_hist, _ = build_observability_matrix(Phi_combined, H_range_hist, t)
    W_angle, W_angle_hist, _ = build_observability_matrix(Phi_combined, H_angle_hist, t)

    # analyze gramians
    rank_r, svs_r, null_r = analyze(W_range, label="Range Measurements")
    rank_a, svs_a, null_a = analyze(W_angle, label="Angle Measurements")



