import os
from utils import *
from data_generation import *
from APPEX import *
import math
import pickle


def default_measurement_settings(exp_number, N=None):
    """
    Default settings for the measurement process
    :param exp_number: 1,2,3 or "random"
    :param N: specified number of samples per time marginal
    :return: list of initialized variables
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


def run_experiment_1(version=1, N=None):
    '''
    Run experiment 1 with specified version and number of samples
    :param version: if version 1, the drift is -1. If version 2, the drift is -10. In both, drift:diffusivity = 1
    :param N: number of samples per time marginal
    :return:
    '''
    d = 1
    if version == 1:
        A = np.array([[-1]])
        G = np.eye(d)
    else:
        A = np.array([[-10]])
        G = math.sqrt(10) * np.eye(d)
    return run_generic_experiment(A, G, d, N, verbose=True, exp_number=1)


def run_experiment_2(version=1, N=500):
    '''
    Run experiment 2 with specified version and number of samples
    :param version: if version 1, the drift is 0, if version 2 the drift is rotational
    :param N: number of samples per marginal
    :return:
    '''
    d = 2
    if version == 1:
        A = np.zeros((d, d))
    else:
        A = np.array([[0, 1], [-1, 0]])
    G = np.eye(d)
    return run_generic_experiment(A, G, d, N, verbose=True, exp_number=2)


def run_experiment_3(version=1, N=500):
    '''
    Run experiment 3 with specified version and number of samples
    :param version: if version 1, the drift is [[1, 2], [1, 0]], if version 2 the drift is [[1/3, 4/3], [2/3, -1/3]]
    :param N: number of samples per marginal
    :return:
    '''
    d = 2
    if version == 1:
        A = np.array([[1, 2], [1, 0]])
    else:
        A = np.array([[1 / 3, 4 / 3], [2 / 3, -1 / 3]])
    G = np.array([[1, 2], [-1, -2]])
    return run_generic_experiment(A, G, d, N, verbose=True, exp_number=3)


def run_experiment_random(d, N=None, causal_sufficiency=True, causal_experiment=True, p=0.5):
    '''
    Used for running the higher dimensional random experiments, including the causal discovery experiments
    :param d: dimension of process
    :param N: number of samples per time marginal
    :param causal_sufficiency: if True, the ground truth diffusion matrix will only have entries on main diagonal
    :param causal_experiment: if True, we will generate drift and diffusion according to sparsity threshold p
    :param p: the probability of a non-zero entry in the drift matrix
    :return:
    '''
    if causal_experiment:
        A = generate_random_matrix_with_eigenvalue_constraint(d, eigenvalue_threshold=1, sparsity_threshold=p,
                                                              epsilon=0.5)
        G = generate_G_causal_experiment(causal_sufficiency, d)
    else:
        A = generate_random_matrix_with_eigenvalue_constraint(d, eigenvalue_threshold=1, sparsity_threshold=1,
                                                              epsilon=0)
        G = np.random.uniform(low=-1, high=1, size=(d, d))
    return run_generic_experiment(A, G, d, N, exp_number='random', verbose=False)


def run_generic_experiment(A, G, d, N=None, verbose=False, gaussian_start=False, exp_number='random'):
    '''
    Run an experiment iterate with specified drift and diffusion matrices
    :param A: true drift matrix
    :param G: true diffusion matrix
    :param d: dimension of process
    :param N: number of observations per marginal
    :param verbose: if True, print the estimated drift and diffusion matrices at each iteration
    :param gaussian_start: if True, uses a Gaussian distribution to initialise the process, otherwise uses Uniform over independent points
    :param exp_number: 1, 2, 3, or 'random'
    :return: dictionary object of results
    '''
    # ground truth observational diffusion matrix
    H = np.matmul(G, G.T)
    print(rf'Generating data for experiment: dX_t = {A} X_t \, dt + {G} \, dW_t')
    if gaussian_start:
        mean = np.zeros(d)
        gaussian_points = np.random.multivariate_normal(mean, H, size=N)
        # Assign equal probabilities to each point
        X0_dist = [(point, 1 / N) for point in gaussian_points]
        print(rf'X0 is initialised uniformly from N(0, {H})')
    else:
        points = generate_independent_points(d, d)
        X0_dist = [(point, 1 / len(points)) for point in points]
        print(rf'X0 is initialised uniformly from the points: {points}')
    dt, dt_EM, T, N, max_its, linearization, killed, log_sinkhorn, report_time_splits = default_measurement_settings(
        exp_number)
    X_measured = linear_additive_noise_data(N, d=d, T=T, dt_EM=dt_EM, dt=dt, A=A, G=G, X0_dist=X0_dist,
                                            destroyed_samples=killed)
    print('NLL with ground truth data:', compute_nll(X_measured, A, H, dt))
    print('shape of marginal observation data:', X_measured.shape)
    print('Estimating parameters')
    its = 1
    mean = np.trace(H) / d  # get average magnitude of main diagonal entry of true diffusion
    order_magnitude = np.random.uniform(low=-1, high=1)
    random_scale = 10 ** order_magnitude
    initial_H = random_scale * mean * np.eye(d)
    print('Initial guess for D:', initial_H)
    # run first iteration of APPEX (equivalent to WOT)
    est_A_list, est_GGT_list, nll_list = [], [], []
    est_A, est_GGT, nll = APPEX_iteration(X_measured, dt, T, cur_est_H=initial_H, linearization=linearization,
                                     report_time_splits=report_time_splits, log_sinkhorn=log_sinkhorn)
    est_A_list.append(est_A)
    est_GGT_list.append(est_GGT)
    nll_list.append(nll)

    if verbose:
        print(f'Estimated A at iteration {its}:', est_A)
        print(f'Estimated H at iteration {its}:', est_GGT)

    print(f'MAE to true A at iteration {its}: {compute_mae(est_A, A)}')
    print(f'MAE to true H at iteration {its}: {compute_mae(est_GGT, H)}')
    print(f'NLL at iteration {its}: {nll}')

    # run subsequent iterations of APPEX
    while its < max_its:
        est_A, est_GGT, nll = APPEX_iteration(X_measured, dt, T, cur_est_A=est_A, cur_est_H=est_GGT,
                                         linearization=linearization, report_time_splits=report_time_splits,
                                         log_sinkhorn=log_sinkhorn)
        est_A_list.append(est_A)
        est_GGT_list.append(est_GGT)
        nll_list.append(nll)
        its += 1
        if verbose:
            print(f'Estimated A at iteration {its}:', est_A)
            print(f'Estimated D at iteration {its}:', est_GGT)
        print(f'MAE to true A at iteration {its}: {compute_mae(est_A, A)}')
        print(f'MAE to true D at iteration {its}: {compute_mae(est_GGT, H)}')
        print(f'NLL at iteration {its}: {nll}')

    results_data = {
        'true_A': A,
        'true_G': G,
        'true_D': np.matmul(G, G.T),
        'X0 points': points,
        'initial D': initial_H,
        'est D values': est_GGT_list,
        'est A values': est_A_list,
        'nll values': nll_list,
        'N': N
    }
    return results_data


def run_generic_experiment_replicates(exp_number, num_replicates, N_list=None, version=1, d=None, p=0.5,
                                      causal_sufficiency=False, causal_experiment=False, seed=None):
    '''
    :param exp_number: 1, 2, 3, or 'random'
    :param num_replicates: number of replicates to run
    :param N_list: list of N values to test, where N is the number of samples per time marginal
    :param version: only applicable for experiments 1, 2, and 3. Determines which SDE to use for experiments
    :param d: dimension of process
    :param p: sparsity threshold (only relevant for causal discovery random experiments)
    :param causal_sufficiency: whether to generate a ground truth diffusion matrix with only diagonal entries
    :param causal_experiment: whether to generate a ground truth drift matrix with sparsity threshold p
    :param seed: random seed
    :return:
    '''
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
                results_data = run_experiment_1(version=version, N=N)
            elif exp_number == 2:
                results_data = run_experiment_2(version=version, N=N)
            elif exp_number == 3:
                results_data = run_experiment_3(version=version, N=N)
            elif exp_number == 'random':
                if causal_experiment:
                    if causal_sufficiency:
                        print('Causal discovery experiment with causal sufficiency')
                    else:
                        print('Causal discovery experiment with latent confounders')
                results_data = run_experiment_random(d, N=N, p=p, causal_sufficiency=causal_sufficiency,
                                                     causal_experiment=causal_experiment)
            # Save the data
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


# # First experiments based on previously non-identifiable SDEs
# # run_generic_experiment_replicates(exp_number=1, version=1, num_replicates=10, d=1)
# run_generic_experiment_replicates(exp_number=2, version=1, num_replicates=10, d=2)
# run_generic_experiment_replicates(exp_number=3, version=1, num_replicates=10, d=2)
# run_generic_experiment_replicates(exp_number=1, version=2, num_replicates=10, d=1)
# run_generic_experiment_replicates(exp_number=2, version=2, num_replicates=10, d=2)
# run_generic_experiment_replicates(exp_number=3, version=2, num_replicates=10, d=2)

# Random higher dimensional experiments
# ds = [3, 4, 5]
# for d in ds:
#     run_generic_experiment_replicates(exp_number='random', d=d, num_replicates=10, seed=69)

# Causal discovery experiments (causal sufficiency)
ds_cd = [3, 5, 10]
ps = [0.1, 0.25, 0.5]
# for d in ds_cd:
#     for p in ps:
#         run_generic_experiment_replicates(exp_number='random', d=d, num_replicates=10, p=p, causal_sufficiency=True,
#                                           causal_experiment=True)

# Causal discovery experiments (latent confounder)
ps_latent = [0.25]
for d in ds_cd:
    for p in ps_latent:
        run_generic_experiment_replicates(exp_number='random', d=d, num_replicates=10, p=p, causal_sufficiency=False,
                                          causal_experiment=True, seed=69)

