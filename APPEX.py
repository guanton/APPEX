import numpy as np
import time
from scipy.linalg import expm, sqrtm
from scipy.stats import multivariate_normal
from utils import *
from scipy.special import logsumexp


def APPEX(X_measured, dt, T,
          linearization, report_time_splits, log_sinkhorn,
          max_its,
          check_convergence = False,
          metrics_to_check=[],
          verbose_rejection=False,
          true_A=None, true_H=None, eps=0.01):
    """
    Runs max_its iterations of the APPEX algorithm

    Parameters:
      X_measured          : The observed data.
      dt, T               : Time discretization parameters.
      linearization, report_time_splits, log_sinkhorn : Additional parameters required by the iteration functions.
      max_its             : The maximum number of accepted iterations.
      check_convergence    : If true, convergence check is performed.
      metrics_to_check    : List of metric keys to check for improvement, options include 'nll', 'w2', 'w1', 'mmd'.
      verbose_rejection   : If True, print detailed rejection information including candidate MAEs.
      true_A, true_H      : Ground truth matrices (if provided, used to compute MAEs).

    Returns:
      accepted_A_list     : List of accepted A estimates (first iteration included).
      accepted_H_list     : List of accepted H estimates.
      accepted_scores     : List of score dictionaries corresponding to accepted iterations.
    """
    # ---- First iteration (always accepted) ----
    X_OT, A_OT, H_OT = APPEX_iteration(X_measured, dt, T,
                                 cur_est_A=None, cur_est_H=None,
                                 linearization=linearization,
                                 report_time_splits=report_time_splits,
                                 log_sinkhorn=log_sinkhorn)
    if check_convergence:
        current_score = {
            'nll': compute_nll(X_OT, A_OT, H_OT, dt),
            'w2': compute_w2(X_OT, A_OT, H_OT, dt),
            'w1': compute_w1(X_OT, A_OT, H_OT, dt),
            'mmd': compute_mmd(X_OT, A_OT, H_OT, dt)
        }
        attempt_count = 1  # Counts all attempts (accepted and rejected)
        accepted_scores = [current_score]
    else:
        accepted_scores = []
    accepted_iterations = 1  # First iteration accepted.
    accepted_A_list = [A_OT]
    accepted_H_list = [H_OT]



    # Print initial MAEs if ground truth provided.
    if verbose_rejection and (true_A is not None and true_H is not None):
        mae_A = compute_mae(A_OT, true_A)
        mae_H = compute_mae(H_OT, true_H)
        print(f"Iteration 1 accepted: MAE A: {mae_A:.4f}, MAE H: {mae_H:.4f}")

    # ---- Subsequent iterations with rejection rule ----
    while accepted_iterations < max_its:


        # Generate candidate using APPEX_iteration
        X_OT_candidate, candidate_A, candidate_H = APPEX_iteration(
            X_measured, dt, T,
            cur_est_A=accepted_A_list[-1],
            cur_est_H=accepted_H_list[-1],
            linearization=linearization,
            report_time_splits=report_time_splits,
            log_sinkhorn=log_sinkhorn)

        if check_convergence:
            attempt_count += 1
            candidate_score = {
                'nll': compute_nll(X_OT_candidate, candidate_A, candidate_H, dt),
                'w2': compute_w2(X_OT_candidate, candidate_A, candidate_H, dt),
                'w1': compute_w1(X_OT_candidate, candidate_A, candidate_H, dt),
                'mmd': compute_mmd(X_OT_candidate, candidate_A, candidate_H, dt)
            }

        # If ground truth is provided, compute candidate MAEs.
        if true_A is not None and true_H is not None:
            candidate_mae_A = compute_mae(candidate_A, true_A)
            candidate_mae_H = compute_mae(candidate_H, true_H)
        else:
            candidate_mae_A = candidate_mae_H = None

        # Check for improvement on specified metrics (lower is better).
        reject = False
        if check_convergence:
            rejection_details = []
            for metric in metrics_to_check:
                if candidate_score[metric] > current_score[metric] + eps:
                    diff = candidate_score[metric] - current_score[metric]
                    reject = True
                    rejection_details.append(
                        f"{metric.upper()} increased by {diff:.4f} (from {current_score[metric]:.4f} to {candidate_score[metric]:.4f})"
                    )

            if reject:
                if verbose_rejection:
                    msg = f"Attempt {attempt_count}: Candidate rejected because: " + "; ".join(rejection_details)
                    if candidate_mae_A is not None and candidate_mae_H is not None:
                        msg += f" | Candidate MAE A: {candidate_mae_A:.4f}, MAE H: {candidate_mae_H:.4f}"
                    print(msg)
                # Do not accept this candidate; continue to the next attempt.
                continue

        # Accept candidate: update accepted estimates and scores.
        accepted_iterations += 1
        accepted_A_list.append(candidate_A)
        accepted_H_list.append(candidate_H)
        if check_convergence:
            accepted_scores.append(candidate_score)
            current_score = candidate_score
            if verbose_rejection:
                msg = f"Attempt {attempt_count}: Candidate accepted as iteration {accepted_iterations}."
                if candidate_mae_A is not None and candidate_mae_H is not None:
                    msg += f" | MAE A: {candidate_mae_A:.4f}, MAE H: {candidate_mae_H:.4f}"
                print(msg)

    return accepted_A_list, accepted_H_list, accepted_scores

def APPEX_iteration(X, dt, T=1, cur_est_A=None, cur_est_H=None, linearization=True,
                    report_time_splits=False, log_sinkhorn=False):
    '''
    Performs one iteration of the APPEX algorithm given current estimates of drift A and diffusion H.
    Now, convergence is gauged by a user-specified metric: 'nll', 'w1', or 'w2'.

    Returns:
         (A_OT, H_OT, convergence_score)
    '''
    num_trajectories, num_steps, d = X.shape
    # Initialize estimates as Brownian motion if not provided
    if cur_est_A is None:
        cur_est_A = np.zeros((d, d))
    if cur_est_H is None:
        cur_est_H = np.eye(d)
    # Perform trajectory inference via generalized entropic optimal transport
    X_OT = AEOT_trajectory_inference(X, dt, cur_est_A, cur_est_H, linearization=linearization,
                                     report_time_splits=report_time_splits, log_sinkhorn=log_sinkhorn)
    # # Compute convergence score after the trajectory inference step
    # convergence_score_OT = compute_convergence_score(X_OT, cur_est_A, cur_est_H, dt, metric=score_metric)
    # print(f'{score_metric} after traj inference step:', convergence_score_OT)
    # Estimate drift and diffusion from inferred trajectories via closed-form MLEs
    if linearization:
        A_OT = estimate_A(X_OT, dt)
        # convergence_score_A = compute_convergence_score(X_OT, A_OT, cur_est_H, dt, metric=score_metric)
        # print(f'{score_metric} after A step:', convergence_score_A)
        H_OT = estimate_GGT(X_OT, T, est_A=A_OT)
    else:
        # only supported for dimension 1
        A_OT = estimate_A_exact_1d(X_OT, dt)
        H_OT = estimate_GGT_exact_1d(X_OT, T, est_A=A_OT)
    return X_OT, A_OT, H_OT


def AEOT_trajectory_inference(X, dt, est_A, est_GGT, linearization=True, report_time_splits=False,
                              epsilon=1e-8, log_sinkhorn=False, N_sample_traj=1000):
    '''
    Leverages anisotropic entropic optimal transport to infer trajectories from marginal samples
    :param X: measured population snapshots
    :param dt: time step
    :param est_A: pre-estimated drift for reference SDE
    :param est_GGT: pre-estimated observational diffusion for reference SDE
    :param linearization: whether to use linearization for drift estimation
    :param report_time_splits: whether to report time splits
    :param epsilon: regularization parameter for numerical stability of covariance matrix
    :param log_sinkhorn: whether to use log-domain sinkhorn
    :param N_sample_traj: number of trajectories to sample from the estimated (discretized) law on paths
    :return: array of sampled trajectories from the estimated law
    '''
    marginal_samples = extract_marginal_samples(X)
    num_time_steps = len(marginal_samples)
    d = marginal_samples[0].shape[1]
    num_trajectories = marginal_samples[0].shape[0]
    ps = []  # transport plans for each pair of consecutive marginals
    sinkhorn_time = 0
    K_time = 0
    for t in range(num_time_steps - 1):
        # extract marginal samples
        X_t = marginal_samples[t]
        X_t1 = marginal_samples[t + 1]
        a = np.ones(len(X_t)) / len(X_t)
        b = np.ones(len(X_t1)) / len(X_t1)
        if linearization:
            A_X_t = np.matmul(est_A, X_t.T) * dt
        else:
            assert d == 1, "exact solvers are only implemented for dimension d=1"
            exp_A_dt = expm(est_A * dt)
        H_reg = est_GGT + np.eye(est_GGT.shape[0]) * epsilon
        Sigma_dt = H_reg * dt  # Precompute D * dt once to avoid repeated computation
        K = np.zeros((num_trajectories, num_trajectories))
        # Loop over trajectories, vectorize inner calculations
        for i in range(num_trajectories):
            t1 = time.time()
            if linearization:
                dX_ij = X_t1 - X_t[i] - A_X_t[:, i].T
            else:
                dX_ij = X_t1 - np.matmul(exp_A_dt, X_t[i])
            # Flatten the differences for all pairs (vectorized)
            dX_ij_flattened = dX_ij.reshape(num_trajectories, d)
            try:
                # Vectorized PDF computation for all j's
                K[i, :] = multivariate_normal.pdf(dX_ij_flattened, mean=np.zeros(d), cov=Sigma_dt)
            except np.linalg.LinAlgError:
                # If numerical issues, regularize again and compute PDFs
                print(f"Numerical issue in multivariate normal pdf at i={i}")
                Sigma_dt += np.eye(est_GGT.shape[0]) * epsilon  # Further regularize if needed
                K[i, :] = multivariate_normal.pdf(dX_ij_flattened, mean=np.zeros(d), cov=Sigma_dt)
            t2 = time.time()
            K_time += t2 - t1
        t1 = time.time()
        if log_sinkhorn:
            p = sinkhorn_log(a=a, b=b, K=K)
        else:
            p = sinkhorn(a=a, b=b, K=K)
        t2 = time.time()
        sinkhorn_time += t2 - t1
        ps.append(p)

    t1 = time.time()
    X_OT = np.zeros(shape=(N_sample_traj, num_time_steps, d))
    OT_index_propagation = np.zeros(shape=(N_sample_traj, num_time_steps - 1))
    # obtain OT plans for each time
    normalized_ps = np.array([normalize_rows(ps[t]) for t in range(num_time_steps - 1)])
    indices = np.arange(num_trajectories)
    for _ in range(N_sample_traj):
        for t in range(num_time_steps - 1):
            pt_normalized = normalized_ps[t]
            if t == 0:
                k = np.random.randint(num_trajectories)
                X_OT[_, 0, :] = marginal_samples[0][k]
            else:
                # retrieve where _th observation at time 0 was projected to at time t
                k = int(OT_index_propagation[_, t - 1])
            j = np.random.choice(indices, p=pt_normalized[k])
            OT_index_propagation[_, t] = int(j)
            X_OT[_, t + 1, :] = marginal_samples[t + 1][j]
    t2 = time.time()
    ot_traj_time = t2 - t1
    if report_time_splits:
        print('Time setting up K:', K_time)
        print('Time doing Sinkhorn:', sinkhorn_time)
        print('Time creating trajectories', ot_traj_time)
    return X_OT


def sinkhorn(a, b, K, maxiter=1000, stopThr=1e-9, epsilon=1e-2):
    '''
    Sinkhorn algorithm given Gibbs kernel K
    :param a: first marginal
    :param b: second marginal
    :param K: Gibbs kernel
    :param maxiter: max number of iteraetions
    :param stopThr: threshold for stopping
    :param epsilon: second stopping threshold
    :return:
    '''
    u = np.ones(K.shape[0])
    v = np.ones(K.shape[1])

    for _ in range(maxiter):
        u_prev = u
        # Perform standard Sinkhorn update
        u = a / (K @ v)
        v = b / (K.T @ u)
        tmp = np.diag(u) @ K @ np.diag(v)

        # Check for convergence based on the error
        err = np.linalg.norm(tmp.sum(axis=1) - a)
        if err < stopThr or np.linalg.norm(u - u_prev) / np.linalg.norm(u_prev) < epsilon:
            break

    return tmp


def sinkhorn_log(a, b, K, maxiter=500, stopThr=1e-9, epsilon=1e-5):
    '''
    Logarithm-domain Sinkhorn algorithm given Gibbs kernel K
    :param a: first marginal
    :param b: second marginal
    :param K: Gibbs kernel K
    :param maxiter: max number of iterations
    :param stopThr: threshold for stopping
    :param epsilon: second stopping threshold
    :return:
    '''
    # Initialize log-domain variables
    log_K = np.log(K + 1e-300)  # Small constant to prevent log(0)
    log_a = np.log(a + 1e-300)
    log_b = np.log(b + 1e-300)
    log_u = np.zeros(K.shape[0])
    log_v = np.zeros(K.shape[1])

    for _ in range(maxiter):
        log_u_prev = log_u.copy()

        # Perform updates in the log domain using logsumexp
        log_u = log_a - logsumexp(log_K + log_v, axis=1)
        log_v = log_b - logsumexp(log_K.T + log_u[:, np.newaxis], axis=0)

        # Calculate the transport plan in the log domain
        log_tmp = log_K + log_u[:, np.newaxis] + log_v

        # Check for convergence based on the error
        tmp = np.exp(log_tmp)
        err = np.linalg.norm(tmp.sum(axis=1) - a)
        if err < stopThr or np.linalg.norm(log_u - log_u_prev) < epsilon:
            break

    return tmp


def estimate_A(X, dt, pinv=False):
    """
    Calculate the approximate closed form estimator A_hat for time homogeneous linear drift from multiple trajectories

    Parameters:
        trajectories (numpy.ndarray): 3D array (num_trajectories, num_steps, d),
        where each slice corresponds to a single trajectory.
        dt (float): Discretization time step.
        pinv: whether to use pseudo-inverse. Otherwise, we use left_Var_Equation

    Returns:
        numpy.ndarray: Estimated drift matrix A given the set of trajectories
    """
    num_trajectories, num_steps, d = X.shape
    sum_Edxt_Ext = np.zeros((d, d))
    sum_Ext_ExtT = np.zeros((d, d))
    for t in range(num_steps - 1):
        sum_dxt_xt = np.zeros((d, d))
        sum_xt_xt = np.zeros((d, d))
        for n in range(num_trajectories):
            xt = X[n, t, :]
            dxt = X[n, t + 1, :] - X[n, t, :]
            sum_dxt_xt += np.outer(dxt, xt)
            sum_xt_xt += np.outer(xt, xt)
        sum_Edxt_Ext += sum_dxt_xt / num_trajectories
        sum_Ext_ExtT += sum_xt_xt / num_trajectories

    if pinv:
        return np.matmul(sum_Edxt_Ext, np.linalg.pinv(sum_Ext_ExtT)) * (1 / dt)
    else:
        return left_Var_Equation(sum_Ext_ExtT, sum_Edxt_Ext * (1 / dt))


def estimate_A_exact_1d(X, dt):
    """
    Calculate the exact closed form estimator A_hat using observed data from multiple trajectories
    Applicable only for dimension d=1

    Parameters:
        X (numpy.ndarray): 3D array where each slice corresponds to a single trajectory (num_trajectories, num_steps, d).
        dt (float): Discretization time step.

    Returns:
        numpy.ndarray: Estimated drift matrix A given the set of trajectories.
    """
    num_trajectories, num_steps, d = X.shape
    assert d == 1, "the exact MLE estimator is only implemented for d=1"
    # Initialize cumulative sums
    sum_Xtp1_XtT = np.zeros((d, d))  # Sum of X_{t+1} * X_t^T
    sum_Xt_XtT = np.zeros((d, d))  # Sum of X_t * X_t^T

    for t in range(num_steps - 1):
        sum_Xtp1_Xt = np.zeros((d, d))
        sum_Xt_Xt = np.zeros((d, d))
        for n in range(num_trajectories):
            Xt = X[n, t, :]  # X_t for trajectory n
            Xtp1 = X[n, t + 1, :]  # X_{t+1} for trajectory n
            sum_Xtp1_Xt += np.outer(Xtp1, Xt)  # X_{t+1} * X_t^T
            sum_Xt_Xt += np.outer(Xt, Xt)  # X_t * X_t^T
        sum_Xtp1_XtT += sum_Xtp1_Xt / num_trajectories
        sum_Xt_XtT += sum_Xt_Xt / num_trajectories
    return (np.log(sum_Xtp1_XtT) - np.log(sum_Xt_XtT)) * (1 / dt)


def estimate_GGT(trajectories, T, est_A=None):
    """
    Estimate the observational diffusion GG^T for a multidimensional linear
    additive noise SDE from multiple trajectories

    Parameters:
        trajectories (numpy.ndarray): 3D array (num_trajectories, num_steps, d),
        where each "slice" (2D array) corresponds to a single trajectory.
        T (float): Total time period.
        est_A (numpy.ndarray, optional): pre-estimated drift A.
        If none provided, est_A = 0, modeling a pure diffusion process

    Returns:
        numpy.ndarray: Estimated GG^T matrix.
    """
    num_trajectories, num_steps, d = trajectories.shape
    dt = T / (num_steps - 1)

    # Initialize the GG^T matrix
    GGT = np.zeros((d, d))

    if est_A is None:
        # Compute increments ΔX for each trajectory (no drift adjustment)
        increments = np.diff(trajectories, axis=1)
    else:
        # Adjust increments by subtracting the deterministic drift: ΔX - A * X_t * dt
        increments = np.diff(trajectories, axis=1) - dt * np.einsum('ij,nkj->nki', est_A, trajectories[:, :-1, :])

    # Sum up the products of increments for each dimension pair across all trajectories and steps
    for i in range(d):
        for j in range(d):
            GGT[i, j] = np.sum(increments[:, :, i] * increments[:, :, j])

    # Divide by total time T*num_trajectories to normalize
    GGT /= T * num_trajectories
    return GGT


def estimate_GGT_exact_1d(X, T, est_A=None):
    """
    Calculate the exact MLE estimator for the matrix GG^T from multiple trajectories of a multidimensional linear
    additive noise SDE. Applicable only for dimension d=1.

    Parameters:
        X (numpy.ndarray): A 3D array where each "slice" (2D array) corresponds to a single trajectory.
        T (float): Total time period.
        est_A (numpy.ndarray, optional): pre-estimated drift matrix A.

    Returns:
        numpy.ndarray: Estimated GG^T matrix.
    """
    num_trajectories, num_steps, d = X.shape
    print(X.shape)
    assert d == 1, "the exact MLE estimator is only implemented for d=1"
    dt = T / (num_steps - 1)

    if est_A is None:
        # Compute increments ΔX for each trajectory (no drift adjustment)
        increments = np.diff(X, axis=1)
    else:
        # Precompute exp(A * dt)
        exp_Adt = expm(est_A * dt)
        # Adjust increments: X_{t+1} - exp(A * dt) * X_t
        increments = X[:, 1:, :] - np.einsum('ij,nkj->nki', exp_Adt, X[:, :-1, :])

        # Efficient computation of GG^T using einsum
    GGT = np.einsum('nti,ntj->ij', increments, increments)

    GGT /= T * num_trajectories

    return GGT
