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
    report_time_splits = False
    print(
        f"Experiment settings: \n dt: {dt}, \n dt_EM: {dt_EM}, \n T: {T}, \n N: {N}, \n max_its: {max_its}, "
        f"\n linearization: {linearization}, \n killed: {killed}, \n log Sinkhorn: {log_sinkhorn}")
    return dt, dt_EM, T, N, max_its, linearization, killed, log_sinkhorn, report_time_splits


def run_experiment_1(version=1, N=None, metrics_to_check=['nll', 'w2', 'w1', 'mmd']):
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
    return run_generic_experiment(A, G, d, N, verbose=True, exp_number=1, metrics_to_check=metrics_to_check)


def run_experiment_2(version=1, N=500, metrics_to_check=['nll', 'w2', 'w1', 'mmd']):
    '''
    Run experiment 2.
    '''
    d = 2
    if version == 1:
        A = np.zeros((d, d))
    else:
        A = np.array([[0, 1], [-1, 0]])
    G = np.eye(d)
    return run_generic_experiment(A, G, d, N, verbose=True, exp_number=2, metrics_to_check=metrics_to_check)


def run_experiment_3(version=1, N=500, metrics_to_check=['nll', 'w2', 'w1', 'mmd']):
    '''
    Run experiment 3.
    '''
    d = 2
    if version == 1:
        A = np.array([[1, 2], [1, 0]])
    else:
        A = np.array([[1 / 3, 4 / 3], [2 / 3, -1 / 3]])
    G = np.array([[1, 2], [-1, -2]])
    return run_generic_experiment(A, G, d, N, verbose=True, exp_number=3, metrics_to_check=metrics_to_check)


def run_experiment_random(d, N=None, causal_sufficiency=True, causal_experiment=True, p=0.5, metrics_to_check=['nll', 'w2', 'w1', 'mmd']):
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
    return run_generic_experiment(A, G, d, N, exp_number='random', verbose=True, metrics_to_check=metrics_to_check)


def run_generic_experiment(A, G, d, N=None, verbose=False, gaussian_start=False, exp_number='random',
                           metrics_to_check=['nll', 'w2', 'w1', 'mmd'], check_convergence=False):
    """
    Run an experiment with specified drift (A) and diffusion (G) matrices.

    This version computes the convergence metrics (nll, w2, w1, mmd) at every iteration.
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

    # Run the iterative procedure via the new APPEX function.
    est_A_list, est_H_list, convergence_score_list = APPEX(
        X_measured, dt, T,
        linearization=linearization,
        check_convergence=check_convergence,
        report_time_splits=report_time_splits,
        log_sinkhorn=log_sinkhorn,
        max_its=max_its,
        metrics_to_check=metrics_to_check,
        verbose_rejection=verbose,
        true_A=A, true_H=H)

    # Print MAE metrics for each accepted iteration.
    for i, (est_A, est_H) in enumerate(zip(est_A_list, est_H_list), start=1):
        print(f"Accepted Iteration {i}: MAE A: {compute_mae(est_A, A):.4f}, MAE H: {compute_mae(est_H, H):.4f}")

    results_data = {
        'true_A': A,
        'true_G': G,
        'true_H': H,
        'X0 points': points if not gaussian_start else None,
        'est A values': est_A_list,
        'est H values': est_H_list,
        'convergence scores': convergence_score_list,  # list of dicts with keys 'nll', 'w2', 'w1', 'mmd'
        'N': N,
        'metrics': metrics_to_check
    }
    return results_data


# --- Replicate Running Function ---
def run_generic_experiment_replicates(exp_number, num_replicates, N_list=None, version=1, d=None, p=0.5,
                                      causal_sufficiency=False, causal_experiment=False, seed=None, metrics_to_check=['nll', 'w2', 'w1', 'mmd']):
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
                results_data = run_experiment_1(version=version, N=N, metrics_to_check=metrics_to_check)
            elif exp_number == 2:
                results_data = run_experiment_2(version=version, N=N, metrics_to_check=metrics_to_check)
            elif exp_number == 3:
                results_data = run_experiment_3(version=version, N=N, metrics_to_check=metrics_to_check)
            elif exp_number == 'random':
                if causal_experiment:
                    if causal_sufficiency:
                        print('Causal discovery experiment with causal sufficiency')
                    else:
                        print('Causal discovery experiment with latent confounders')
                results_data = run_experiment_random(d, p=p, causal_sufficiency=causal_sufficiency,
                                                     causal_experiment=causal_experiment, metrics_to_check=metrics_to_check)
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
run_generic_experiment_replicates(exp_number=3, version=1, num_replicates=10, d=2)
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
# ds_cd = [3]
#
# #
# #
#
# # Causal discovery experiments (latent confounder)
# ps_latent = [0.25]
# for d in ds_cd:
#     for p in ps_latent:
#         run_generic_experiment_replicates(exp_number='random', d=d, num_replicates=10, p=p, causal_sufficiency=False,
#                                           causal_experiment=True, seed=0, metrics_to_check=['w2'])

# ps = [0.1, 0.25, 0.5]
# for d in ds_cd:
#     for p in ps:
#         run_generic_experiment_replicates(exp_number='random', d=d, num_replicates=10, p=p, causal_sufficiency=True,
#                                           causal_experiment=True, seed=0, metrics_to_check=['nll', 'w2', 'w1', 'mmd'])
