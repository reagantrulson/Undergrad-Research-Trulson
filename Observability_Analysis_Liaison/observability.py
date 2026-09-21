import numpy as np
from orbits import twob_scenario
from measurements  import compute_measurement_history
from stm import propagate_stm, build_combined_stm, build_observability_matrix

RANK_TOL = 1e-13       # essentially disabled — just a safety net
RANK_TOL_ABS = 1e-2    # tied to STM integration error
STATE= ['r1x', 'r1y', 'r1z', 'v1x', 'v1y', 'v1z','r2x', 'r2y', 'r2z', 'v2x', 'v2y', 'v2z']

def svd_analysis(W, label = "", tol_rel = RANK_TOL, tol_abs = RANK_TOL_ABS):
    U, s, Vt = np.linalg.svd(W, full_matrices = True) # svd

    sigma_max = s[0] # largest singular value
    rank = np.linalg.matrix_rank(W)
    threshold = max(tol_rel * sigma_max, tol_abs)  # take the larger of the two
    cond = sigma_max / s[rank-1] if rank > 0 else np.inf # how observable each state direction is



    # smallest nonzero singular value
    obs_index = s[rank-1] if rank > 0  else 0.0

    # subspace decomposition
    obs_subspace = Vt[:rank].T # observable directions, rank # of singular vectors/nonzero singular values (12, rank)
    unobs_subspace = Vt[rank:].T # unobservable directions, singular values near 0 (12, 12-rank)

    return {"label": label,
        "rank": rank,
        "singular_values": s,
        "U": U,
        "Vt": Vt,
        "observable_subspace": obs_subspace,
        "unobservable_subspace": unobs_subspace,
        "condition_number": cond,
        "observability_index": obs_index,
    }

def interpret_null(unobs_subspace, threshold = 0.1):
    interpretations = []
    k = unobs_subspace.shape[1]

    for i in range(k):
        # find important components of each col of the unobservable subspace
        null = unobs_subspace[:, i]
        sigma_idx = np.where(np.abs(null)>threshold)[0]
        if len(sigma_idx) == 0: # keep largest component if none are above threshold
            sigma_idx = [np.argmax(np.abs(null))]

        parts = []
        for j in sigma_idx: # loop over significant components and find their state
            parts.append(f"{null[j]:+.3f}{STATE[j]}")
        interpretation = f" null[{i+1}]: " + "+".join(parts)
        interpretations.append(interpretation) # show each null value and its components/ why
    return interpretations

def observability_over_time(W_hist, t_eval, tol=RANK_TOL,tol_abs=RANK_TOL_ABS, stride =10):
    # compute rank and svs of W at each index (every 10 points)
    idx = np.arange(0, len(t_eval), stride)
    t_sub = t_eval[idx]
    rank_hist = np.zeros(len(idx), dtype=int)
    sigma_hist = np.zeros((len(idx), 12))

    for k, n in enumerate(idx): # go through each index, k and value of t, n
        W_k = W_hist[n]
        _,s,_ = np.linalg.svd(W_k) # find singular values with svd
        sigma_max = s[0] if s[0] > 0 else 1.0 # largest singular value
        threshold = max(tol * sigma_max, tol_abs)
        rank_hist[k] = int(np.sum(s > threshold))
        sigma_hist[k] = s # store all singular values
    return t_sub, rank_hist, sigma_hist # return times, observable rank over time, singular vals over time

def local_injectivity(W, n_trials=1000, perturbation_scale=1.0):
    _, s, Vt = np.linalg.svd(W)
    sigma_max = s[0] if s[0] > 0 else 1.0
    threshold = max(RANK_TOL * sigma_max, RANK_TOL_ABS)
    rank = int(np.sum(s > threshold))

    responses = []
    for i in range(n_trials):
        dx = np.random.randn(12) * perturbation_scale # small random perturbation
        dx = dx/np.linalg.norm(dx) # make length 1
        Wdx = W @dx # how strongly W responds to the perturbation
        responses.append(np.linalg.norm(Wdx))

    min_response = min(responses) # weakest observable random direction

    null_responses = []
    for i in range(rank,12):
        null = Vt[i] # null space basis vector
        w_null = W @ null
        null_responses.append(np.linalg.norm(w_null)) # should be nearly 0 if there are nontrivial solutions ( rank < 12)

    return min_response, null_responses


if __name__ == "__main__":
    print("Observability Analysis")
    N = 500
    t, r1, v1, r2, v2, T = twob_scenario(n_points=N)
    t_span = (t[0], t[-1])

    # propagate stms
    r1_s, v1_s, Phi1 = propagate_stm(r1[0], v1[0], t_span, t)
    r2_s, v2_s, Phi2 = propagate_stm(r2[0], v2[0], t_span, t)
    Phi_combined = build_combined_stm(Phi1, Phi2)
    _, _, H_range_hist, H_angle_hist = compute_measurement_history(r1, v1, r2, v2) # measurement jacobians
    # observability gramians
    W_range, W_range_hist, _ = build_observability_matrix(Phi_combined, H_range_hist, t)
    W_angle, W_angle_hist, _ = build_observability_matrix(Phi_combined, H_angle_hist, t)
    # svd analysis
    result_range = svd_analysis(W_range, label="Range")
    result_angle = svd_analysis(W_angle, label="Angles")
    # observability over time
    t_sub_r, rank_hist_r, sigma_hist_r = observability_over_time(W_range_hist, t, stride=5)
    t_sub_a, rank_hist_a, sigma_hist_a = observability_over_time(W_angle_hist, t, stride=5)

    # null space interpretation
    print("Unobservable directions (range)")
    unobs_interps = interpret_null(result_range["unobservable_subspace"])
    if unobs_interps:
        for u in unobs_interps:
            print(u)
    else:
        print(" None - full rank")

    print(f"  sigma_hist_r shape : {sigma_hist_r.shape}")
    print(f"  sigma_hist_a shape : {sigma_hist_a.shape}")
    print(f"  rank_hist_r        : {rank_hist_r[:5]} ... (first 5 values)")
    print(f"  rank_hist_a        : {rank_hist_a[:5]} ...")





