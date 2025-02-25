import os
from utils import *
from data_generation import *
from APPEX import *
import math
import pickle


def default_measurement_settings(exp_number, N=None):
    """
    Default settings for the measurement process.
    """
    dt = 0.05
    dt_EM = 0.01
    T = 1
    if N is None:
        N = 500
    max_its = 30
    if exp_number == 1:
        linearization = False
    else:
        linearization = True
    if exp_number != 'random':
        log_sinkhorn = False
    else:
        log_sinkhorn = True
    killed = False
    report_time_splits = True
    print(
        f"Experiment settings: \n dt: {dt}, \n dt_EM: {dt_EM}, \n T: {T}, \n N: {N}, \n max_its: {max_its}, "
        f"\n linearization: {linearization}, \n killed: {killed}, \n log Sinkhorn: {log_sinkhorn}")
    return dt, dt_EM, T, N, max_its, linearization, killed, log_sinkhorn, report_time_splits


def run_experiment_1(version=1, N=None, score_metric='w2'):
    '''
    Run experiment 1.
    '''
    d = 1
    if version == 1:
        A = np.array([[-1]])
        G = np.eye(d)
    else:
        A = np.array([[-10]])
        G = math.sqrt(10) * np.eye(d)
    return run_generic_experiment(A, G, d, N, verbose=True, exp_number=1, score_metric=score_metric)


def run_experiment_2(version=1, N=500, score_metric='w2'):
    '''
    Run experiment 2.
    '''
    d = 2
    if version == 1:
        A = np.zeros((d, d))
    else:
        A = np.array([[0, 1], [-1, 0]])
    G = np.eye(d)
    return run_generic_experiment(A, G, d, N, verbose=True, exp_number=2, score_metric=score_metric)


def run_experiment_3(version=1, N=500, score_metric='w2'):
    '''
    Run experiment 3.
    '''
    d = 2
    if version == 1:
        A = np.array([[1, 2], [1, 0]])
    else:
        A = np.array([[1 / 3, 4 / 3], [2 / 3, -1 / 3]])
    G = np.array([[1, 2], [-1, -2]])
    return run_generic_experiment(A, G, d, N, verbose=True, exp_number=3, score_metric=score_metric)


def run_experiment_random(d, N=None, causal_sufficiency=True, causal_experiment=True, p=0.5, score_metric='w2'):
    '''
    Run higher-dimensional random experiments.
    '''
    if causal_experiment:
        A = generate_random_matrix_with_eigenvalue_constraint(d, eigenvalue_threshold=1, sparsity_threshold=p,
                                                              epsilon=0.5)
        G = generate_G_causal_experiment(causal_sufficiency, d)
    else:
        A = generate_random_matrix_with_eigenvalue_constraint(d, eigenvalue_threshold=1, sparsity_threshold=1,
                                                              epsilon=0)
        G = np.random.uniform(low=-1, high=1, size=(d, d))
    return run_generic_experiment(A, G, d, N, exp_number='random', verbose=True, score_metric=score_metric)


def run_generic_experiment(A, G, d, N=None, verbose=False, gaussian_start=False, exp_number='random',
                           score_metric='w2'):
    """
    Run an experiment with specified drift (A) and diffusion (G) matrices.

    This version computes three convergence metrics (nll, w2, w1) at every iteration.
    Since APPEX_iteration returns only (A_OT, H_OT, current_score), we re-run
    the trajectory inference step (AEOT_trajectory_inference) to obtain X_OT for metric computation.
    """
    # Compute true observational diffusion matrix
    H = np.matmul(G, G.T)
    print(rf'Generating data for experiment: dX_t = {A} X_t \, dt + {G} \, dW_t')
    if gaussian_start:
        mean = np.zeros(d)
        gaussian_points = np.random.multivariate_normal(mean, H, size=N)
        X0_dist = [(point, 1 / N) for point in gaussian_points]
        print(rf'X0 is initialised uniformly from N(0, {H})')
    else:
        points = generate_independent_points(d, d)
        X0_dist = [(point, 1 / len(points)) for point in points]
        print(rf'X0 is initialised uniformly from the points: {points}')
    dt, dt_EM, T, N, max_its, linearization, killed, log_sinkhorn, report_time_splits = default_measurement_settings(
        exp_number)

    # Generate measured data
    X_measured = linear_additive_noise_data(N, d=d, T=T, dt_EM=dt_EM, dt=dt, A=A, G=G, X0_dist=X0_dist,
                                            destroyed_samples=killed)

    # --- Initial iteration ---
    # Call APPEX_iteration (which returns 3 values: A_OT, H_OT, current_score)
    A_OT, H_OT = APPEX_iteration(X_measured, dt, T, cur_est_A=None, cur_est_H=None,
                                    linearization=linearization, report_time_splits=report_time_splits,
                                    log_sinkhorn=log_sinkhorn)
    # Recompute X_OT using current estimates (AEOT_trajectory_inference)
    X_OT = AEOT_trajectory_inference(X_measured, dt, A_OT, H_OT, linearization=linearization,
                                     report_time_splits=report_time_splits, log_sinkhorn=log_sinkhorn)
    score_nll = compute_nll(X_OT, A_OT, H_OT, dt)
    score_w2 = compute_w2(X_OT, A_OT, H_OT, dt)
    score_w1 = compute_w1(X_OT, A_OT, H_OT, dt)
    score_mmd = compute_mmd(X_OT, A_OT, H_OT, dt)
    score_dict = {'nll': score_nll, 'w2': score_w2, 'w1': score_w1, 'mmd': score_mmd}
    if verbose:
        print(f"Iteration 1 metrics: nll = {score_nll:.4f}, w2 = {score_w2:.4f}, w1 = {score_w1:.4f}, mmd = {score_mmd:.4f}")
        print(f"Iteration 1 MAE A: {compute_mae(A_OT, A):.4f}, MAE H: {compute_mae(H_OT, H):.4f}")


    est_A_list = [A_OT]
    est_GGT_list = [H_OT]
    convergence_score_list = [score_dict]

    its = 1
    while its < max_its:
        A_OT, H_OT = APPEX_iteration(X_measured, dt, T, cur_est_A=A_OT, cur_est_H=H_OT,
                                        linearization=linearization, report_time_splits=report_time_splits,
                                        log_sinkhorn=log_sinkhorn)
        X_OT = AEOT_trajectory_inference(X_measured, dt, A_OT, H_OT, linearization=linearization,
                                         report_time_splits=report_time_splits, log_sinkhorn=log_sinkhorn)
        score_nll = compute_nll(X_OT, A_OT, H_OT, dt)
        score_w2 = compute_w2(X_OT, A_OT, H_OT, dt)
        score_w1 = compute_w1(X_OT, A_OT, H_OT, dt)
        score_mmd = compute_mmd(X_OT, A_OT, H_OT, dt)
        score_dict = {'nll': score_nll, 'w2': score_w2, 'w1': score_w1, 'mmd': score_mmd}
        est_A_list.append(A_OT)
        est_GGT_list.append(H_OT)
        convergence_score_list.append(score_dict)
        its += 1
        if verbose:
            print(f"Iteration {its} metrics: nll = {score_nll:.4f}, w2 = {score_w2:.4f}, w1 = {score_w1:.4f}, mmd = {score_mmd:.4f}")
        print(f"Iteration {its} MAE A: {compute_mae(A_OT, A):.4f}, MAE H: {compute_mae(H_OT, H):.4f}")

    results_data = {
        'true_A': A,
        'true_G': G,
        'true_H': H,
        'X0 points': points if not gaussian_start else None,
        'est A values': est_A_list,
        'est H values': est_GGT_list,
        'convergence scores': convergence_score_list,  # list of dicts with keys 'nll', 'w2', 'w1', 'mmd'
        'N': N,
        'score_metric': score_metric
    }
    return results_data


# --- Replicate Running Function ---
def run_generic_experiment_replicates(exp_number, num_replicates, N_list=None, version=1, d=None, p=0.5,
                                      causal_sufficiency=False, causal_experiment=False, seed=None, score_metric='w2'):
    if seed is not None:
        np.random.seed(seed)
    else:
        seed = np.random.randint(0, 1000)
        np.random.seed(seed)
    print('Seed:', seed)
    if N_list is None:
        N_list = [None]
    for N in N_list:
        if N is None:
            N = 500
        print(f'\nRunning experiments for N={N}')
        for i in range(1, num_replicates + 1):
            print(f'\nRunning replicate {i} of experiment {exp_number} with d={d} and N={N}')
            if exp_number == 1:
                results_data = run_experiment_1(version=version, N=N, score_metric=score_metric)
            elif exp_number == 2:
                results_data = run_experiment_2(version=version, N=N, score_metric=score_metric)
            elif exp_number == 3:
                results_data = run_experiment_3(version=version, N=N, score_metric=score_metric)
            elif exp_number == 'random':
                if causal_experiment:
                    if causal_sufficiency:
                        print('Causal discovery experiment with causal sufficiency')
                    else:
                        print('Causal discovery experiment with latent confounders')
                results_data = run_experiment_random(d, N=N, p=p, causal_sufficiency=causal_sufficiency,
                                                     causal_experiment=causal_experiment, score_metric=score_metric)
            if exp_number != 'random':
                results_dir = f'Results_experiment_{exp_number}_seed-{seed}'
                filename = f'version-{version}_N-{N}_replicate-{i}.pkl'
            else:
                if causal_experiment:
                    if causal_sufficiency:
                        results_dir = f'Results_experiment_causal_sufficiency_random_{d}_sparsity_{p}_seed-{seed}'
                    else:
                        results_dir = f'Results_experiment_latent_confounder_random_{d}_sparsity_{p}_seed-{seed}'
                else:
                    results_dir = f'Results_experiment_random_{d}_seed-{seed}'
                filename = f'replicate-{i}_N-{N}.pkl'
            os.makedirs(results_dir, exist_ok=True)
            filepath = os.path.join(results_dir, filename)
            with open(filepath, 'wb') as f:
                pickle.dump(results_data, f)


# # # First experiments based on previously non-identifiable SDEs
# run_generic_experiment_replicates(exp_number=1, version=1, num_replicates=10, d=1)
# run_generic_experiment_replicates(exp_number=2, version=1, num_replicates=10, d=2)
# run_generic_experiment_replicates(exp_number=3, version=1, num_replicates=10, d=2)
# run_generic_experiment_replicates(exp_number=1, version=2, num_replicates=10, d=1)
# run_generic_experiment_replicates(exp_number=2, version=2, num_replicates=10, d=2)
# run_generic_experiment_replicates(exp_number=3, version=2, num_replicates=10, d=2)
#
# # Random higher dimensional experiments
# ds = [3, 4, 5, 10]
# for d in ds:
#     run_generic_experiment_replicates(exp_number='random', d=d, num_replicates=10, seed=69)
#
# # Causal discovery experiments (causal sufficiency)
ds_cd = [3, 5, 10]
# ps = [0.1, 0.25, 0.5]
# for d in ds_cd:
#     for p in ps:
#         run_generic_experiment_replicates(exp_number='random', d=d, num_replicates=10, p=p, causal_sufficiency=True,
#                                           causal_experiment=True, seed = 7)
#
#

# Causal discovery experiments (latent confounder)
ps_latent = [0.25]
for d in ds_cd:
    for p in ps_latent:
        run_generic_experiment_replicates(exp_number='random', d=d, num_replicates=10, p=p, causal_sufficiency=False,
                                          causal_experiment=True, seed=1)