import numpy as np
import os
import re
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import ot
# For the sklearn option:
from sklearn.metrics.pairwise import pairwise_kernels

def angle_between(v1, v2):
    """
    Helper function to compute the angle between two vectors in radians.
    """
    dot_product = np.dot(v1, v2)
    norms = np.linalg.norm(v1) * np.linalg.norm(v2)
    cos_angle = dot_product / norms
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Clip for numerical stability
    return np.arccos(cos_angle)


def generate_independent_points(d, num_points, min_magnitude=2, max_magnitude=10, min_angle_degrees=30, max_its=1000):
    '''
    Theorem 1 implies that a d-dimensional 0-mean linear additive noise is identifiable from X0 if X0 is supported on
    d linearly independent points. This function generates linearly independent points in d-dimensional space.
    Args:
        d: dimension
        num_points: number of points to generate (if > d, then additional points are generated without constraints)
        min_magnitude: minimum Euclidean norm for considered points
        max_magnitude: maximum Euclidean norm for considered points
        min_angle_degrees: minimum pairwise angle required between considered points in degrees
        max_its: maximum number of iterations to attempt to generate linearly independent points with min_angle_degrees
    Returns:
        list of numpy.ndarray: list of linearly independent points
    '''
    # Convert minimum angle from degrees to radians
    min_angle_radians = np.radians(min_angle_degrees)

    points = []
    # Generate first random point
    point = np.random.uniform(-1, 1, d)
    point = point / np.linalg.norm(point)  # Normalize
    scale = np.random.uniform(min_magnitude, max_magnitude)
    point = point * scale
    points.append(point)
    # Generate remaining linearly independent points with at least min_angle_radians between each pair
    for _ in range(1, min(num_points, d)):
        its = 0
        independent = False
        while not independent:
            its += 1
            # Generate candidate point
            candidate_point = np.random.uniform(-1, 1, d)
            candidate_point = candidate_point / np.linalg.norm(candidate_point)
            scale = np.random.uniform(min_magnitude, max_magnitude)
            candidate_point = candidate_point * scale

            # check for linear independence
            matrix = np.vstack(points + [candidate_point])
            rank = np.linalg.matrix_rank(matrix)
            if rank == len(points) + 1:
                # Check the angles with all existing points to also ensure that pairwise angles are sufficiently large
                independent = True
                for existing_point in points:
                    angle = angle_between(candidate_point, existing_point)
                    if angle < min_angle_radians:
                        independent = False
                        if its > max_its:
                            print(
                                f'max number of iterations {max_its} exceeded. Consider increasing max_its or '
                                f'decreasing min_angle_degrees')
                            break
                if independent:
                    points.append(candidate_point)

    # If num_points > d, generate additional points without constraints
    for _ in range(d, num_points):
        candidate_point = np.random.uniform(-1, 1, d)
        candidate_point = candidate_point / np.linalg.norm(candidate_point)  # Normalize
        scale = np.random.uniform(min_magnitude, max_magnitude)
        candidate_point = candidate_point * scale
        points.append(candidate_point)
    return points


def generate_random_matrix_with_eigenvalue_constraint(d, eigenvalue_threshold=1, sparsity_threshold=0,
                                                      epsilon=0, max_iterations=1e5):
    '''
    Args:
        d: dimension of square matrix to be generated
        eigenvalue_threshold: maximal real part of eigenvalue
        sparsity_threshold: fraction of elements to set to zero, used for causal discovery experiment
        epsilon: minimum magnitude for matrix entries (set to 0.5 for causal discovery experiment)
        max_iterations: maximum number of iterations to attempt to generate a matrix with eigenvalue constraint
    Returns:
        np.array: d x d random matrix with eigenvalue constraint
    '''
    for _ in range(int(max_iterations)):
        M = np.random.uniform(low=epsilon, high=5, size=(d, d))
        sign_matrix = np.random.choice([-1, 1], size=M.shape)
        M = M * sign_matrix

        # Introduce sparsity if applicable
        if sparsity_threshold > 0:
            mask = np.random.rand(d, d) < sparsity_threshold
            M = np.multiply(M, mask)

        # Eigenvalue check
        eigenvalues = np.linalg.eigvals(M)
        max_eigenvalue = np.max(eigenvalues.real)
        if max_eigenvalue < eigenvalue_threshold:
            return M  # Return the matrix if the condition is satisfied

    # Step 5: Raise an exception if no valid matrix was found within the iteration limit
    raise ValueError(
        f"Failed to generate a matrix of dimension {d} with max real eigenvalue {eigenvalue_threshold} after {max_iterations} iterations. Consider lowering eigenvalue threshold or increasing max_iterations")


def extract_marginal_samples(X, shuffle=True):
    """
    Extract marginal distributions per time from measured population snapshots

    Parameters:
        X (numpy.ndarray): 3D array of trajectories (num_trajectories, num_steps, d).
        shuffle: whether to shuffle the time marginals (X should already break dependencies between trajectories)

    Returns:
        list of numpy.ndarray: Each element is an array containing samples from the marginal distribution at each time step.
    """
    num_trajectories, num_steps, d = X.shape
    marginal_samples = []

    for t in range(num_steps):
        # Extract all samples at time t from each trajectory
        samples_at_t = X[:, t, :]
        if shuffle:
            samples_at_t_copy = samples_at_t.copy()
            np.random.shuffle(samples_at_t_copy)
            marginal_samples.append(samples_at_t_copy)
        else:
            marginal_samples.append(samples_at_t)
    return marginal_samples


def shuffle_trajectories_within_time(trajectories, return_as_list=False):
    """
    Shuffles the trajectories within each time step.

    Args:
        trajectories (numpy.ndarray): Array of shape (num_trajectories, num_steps, d).

    Returns:
        numpy.ndarray: Array containing the shuffled samples for each time step.
    """
    num_trajectories, num_steps, d = trajectories.shape
    marginal_samples = []

    for t in range(num_steps):
        # Extract all samples at time t from each trajectory
        samples_at_t = trajectories[:, t, :]
        # Make a copy and shuffle the trajectories at time t
        samples_at_t_copy = samples_at_t.copy()
        np.random.shuffle(samples_at_t_copy)
        marginal_samples.append(samples_at_t_copy)

    if return_as_list:
        return marginal_samples
    else:
        return np.array(marginal_samples)
def normalize_rows(matrix):
    """
    Normalize each row of the matrix to sum to 1.

    Parameters:
        matrix (numpy.ndarray): The matrix to normalize.

    Returns:
        numpy.ndarray: The row-normalized matrix.
    """
    row_sums = matrix.sum(axis=1, keepdims=True)
    return matrix / row_sums

def left_Var_Equation(A1, B1):
    """
    Stable solver for np.matmul(sum_Edxt_Ext, np.linalg.pinv(sum_Ext_ExtT)) * (1 / dt)
    via least squares formulation of XA = B  <=> A^T X^T = B^T
    """
    m = B1.shape[0]
    n = A1.shape[0]
    X = np.zeros((m, n))
    for i in range(m):
        X[i, :] = np.linalg.lstsq(np.transpose(A1), B1[i, :], rcond=None)[0]
    return X


def compute_mae(estimated, ground_truth):
    """Compute Mean Absolute Percentage Error (MAE)"""
    mae = np.mean(np.abs((estimated - ground_truth)))
    return mae



def compute_mmd(X_OT, A_OT, H_OT, dt, num_samples=1000, sigma=None, method='closed'):
    """
    Compute the Maximum Mean Discrepancy (MMD) between the empirical residual distribution
    (computed at each time step) and the theoretical Gaussian distribution N(0, H_OT * dt).

    The MMD is computed using a Gaussian kernel:
         k(x,y) = exp(-||x-y||^2/(2*sigma^2))
    where sigma is a bandwidth parameter (if not provided, it defaults to dt).
    The function returns the maximum MMD value over all time steps.

    Parameters:
      X_OT: np.ndarray, shape (num_trajectories, num_steps, d)
            Inferred trajectories.
      A_OT: np.ndarray, shape (d, d)
            Drift matrix.
      H_OT: np.ndarray, shape (d, d)
            Diffusion matrix.
      dt: float
            Time step size.
      num_samples: int, maximum number of residual samples to use per time step.
      sigma: float, kernel bandwidth; if None, defaults to dt.
      method: str, one of {'closed', 'sklearn', 'keops'}.

    Returns:
      mmd: float, the maximum (supremum) MMD over all time steps.
    """
    if sigma is None:
        sigma = dt  # default choice; can be tuned

    num_trajectories, num_steps, d = X_OT.shape
    mmd_list = []

    # Define the three kernel-computation branches.
    def mmd_closed(empirical_samples, theoretical_samples, sigma):
        # Use cdist to compute squared Euclidean distances.
        K_xx = np.exp(-cdist(empirical_samples, empirical_samples, metric='sqeuclidean') / (2 * sigma ** 2))
        K_yy = np.exp(-cdist(theoretical_samples, theoretical_samples, metric='sqeuclidean') / (2 * sigma ** 2))
        K_xy = np.exp(-cdist(empirical_samples, theoretical_samples, metric='sqeuclidean') / (2 * sigma ** 2))
        m = empirical_samples.shape[0]
        mmd_sq = ((np.sum(K_xx) - np.trace(K_xx)) / (m * (m - 1)) +
                  (np.sum(K_yy) - np.trace(K_yy)) / (m * (m - 1)) -
                  2 * np.mean(K_xy))
        return np.sqrt(max(mmd_sq, 0))

    def mmd_sklearn(empirical_samples, theoretical_samples, sigma):
        # Use sklearn's pairwise_kernels with the RBF (Gaussian) kernel.
        gamma = 1.0 / (2 * sigma ** 2)
        K_xx = pairwise_kernels(empirical_samples, empirical_samples, metric='rbf', gamma=gamma)
        K_yy = pairwise_kernels(theoretical_samples, theoretical_samples, metric='rbf', gamma=gamma)
        K_xy = pairwise_kernels(empirical_samples, theoretical_samples, metric='rbf', gamma=gamma)
        m = empirical_samples.shape[0]
        mmd_sq = ((np.sum(K_xx) - np.trace(K_xx)) / (m * (m - 1)) +
                  (np.sum(K_yy) - np.trace(K_yy)) / (m * (m - 1)) -
                  2 * np.mean(K_xy))
        return np.sqrt(max(mmd_sq, 0))

    # Loop over time steps:
    for t in range(num_steps - 1):
        # Compute residuals for time step t:
        residuals = X_OT[:, t + 1, :] - (X_OT[:, t, :] + (A_OT @ X_OT[:, t, :].T).T * dt)
        N_total = residuals.shape[0]
        n = min(num_samples, N_total)
        idx = np.random.choice(N_total, n, replace=False)
        empirical_samples = residuals[idx]  # shape (n, d)
        theoretical_samples = np.random.multivariate_normal(np.zeros(d), H_OT * dt, size=n)

        if method == 'closed':
            mmd_val = mmd_closed(empirical_samples, theoretical_samples, sigma)
        elif method == 'sklearn':
            mmd_val = mmd_sklearn(empirical_samples, theoretical_samples, sigma)
        else:
            raise ValueError("Invalid method. Choose 'closed' or 'sklearn'.")
        mmd_list.append(mmd_val)

    # Return the maximum MMD value over all time steps.
    return np.max(mmd_list)



def compute_nll(X_OT, A_OT, H_OT, dt):
    """
    Compute the negative log-likelihood (NLL) of the inferred trajectories under the current model.
    (Original code; retained for metric='nll')
    """
    num_trajectories, num_steps, d = X_OT.shape
    total_nll = 0.0
    H_dt = H_OT * dt
    # Precompute inverse and determinant of H_dt
    H_dt_inv = np.linalg.pinv(H_dt)
    sign, logdet_H_dt = np.linalg.slogdet(H_dt)
    const_term = 0.5 * (d * np.log(2 * np.pi) + logdet_H_dt)
    for traj in X_OT:
        nll = 0.0
        for t in range(num_steps - 1):
            X_t = traj[t]
            X_tp1 = traj[t + 1]
            # Using Euler-Maruyama: mu_t = X_t + A_OT @ X_t * dt
            mu_t = X_t + A_OT @ X_t * dt
            diff = X_tp1 - mu_t
            exponent = 0.5 * diff.T @ H_dt_inv @ diff
            nll += exponent + const_term
        total_nll += nll
    avg_nll = total_nll / num_trajectories
    return avg_nll

def compute_w2(X_OT, A_OT, H_OT, dt, num_samples=1000, pot=False):
    """
    Compute the per-time-step squared Wasserstein-2 distance.

    For each time step, compute the residuals (using Euler–Maruyama)
    and then compute the W2 distance between the empirical residual distribution
    and the theoretical N(0, H_OT * dt) distribution.

    Parameters:
        X_OT: array-like, shape (num_trajectories, num_steps, d)
              Inferred trajectories.
        A_OT: array-like, shape (d, d)
              Drift matrix.
        H_OT: array-like, shape (d, d)
              Diffusion matrix (used in covariance H_OT * dt).
        dt: float
              Time step.
        num_samples: int, optional
              Maximum number of residual samples to use per time step.
        pot: bool, optional (default=False)
              If False use POT's ot.emd2; otherwise, use linear_sum_assignment

    Returns:
        max_w2: float
             The maximum squared W2 distance across time steps.
    """
    num_trajectories, num_steps, d = X_OT.shape
    w2_list = []

    for t in range(num_steps - 1):
        # Compute residuals for time step t (Euler–Maruyama update).
        residuals = X_OT[:, t + 1, :] - (X_OT[:, t, :] + (A_OT @ X_OT[:, t, :].T).T * dt)
        N_total = residuals.shape[0]
        n = min(num_samples, N_total)
        idx = np.random.choice(N_total, n, replace=False)
        empirical_samples = residuals[idx]

        # Generate theoretical samples from N(0, H_OT * dt).
        theoretical_samples = np.random.multivariate_normal(np.zeros(d), H_OT * dt, size=n)

        # Compute cost matrix (squared Euclidean distances).
        cost_matrix = cdist(empirical_samples, theoretical_samples, metric='sqeuclidean')

        if pot:
            # Use POT's optimal transport solver.
            p = np.ones(n) / n
            q = np.ones(n) / n
            w2 = ot.emd2(p, q, cost_matrix)
            # w2 = np.sqrt(W2_squared)
        else:
            # Use the Hungarian algorithm from linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            w2 = np.mean(cost_matrix[row_ind, col_ind])
            # w2 = np.sqrt(avg_squared_distance)

        w2_list.append(w2)

    return np.max(w2_list)


def compute_w1(X_OT, A_OT, H_OT, dt, num_samples=1000, pot=False):
    """
    Compute the per-time-step Wasserstein-1 (Earth Mover's) distance.

    For each time step, compute the residuals and then compute the W1 distance
    between the empirical residual distribution and the theoretical N(0, H_OT * dt)
    distribution.

    Parameters:
        X_OT: array-like, shape (num_trajectories, num_steps, d)
              Inferred trajectories.
        A_OT: array-like, shape (d, d)
              Inferred drift matrix.
        H_OT: array-like, shape (d, d)
              Inferred diffusion matrix (used in covariance H_OT * dt).
        dt: float
              Time step.
        num_samples: int, optional
              Maximum number of residual samples to use per time step.
        pot: bool, optional (default=False)
              If False use use POT's ot.emd2; otherwise, use linear_sum_assignment

    Returns:
        max_w1: float
             The maximum W1 distance across time steps.
    """
    num_trajectories, num_steps, d = X_OT.shape
    w1_list = []

    for t in range(num_steps - 1):
        # Compute residuals for time step t.
        residuals = X_OT[:, t + 1, :] - (X_OT[:, t, :] + (A_OT @ X_OT[:, t, :].T).T * dt)
        N_total = residuals.shape[0]
        n = min(num_samples, N_total)
        idx = np.random.choice(N_total, n, replace=False)
        empirical_samples = residuals[idx]

        # Generate theoretical samples from N(0, H_OT * dt)
        theoretical_samples = np.random.multivariate_normal(np.zeros(d), H_OT * dt, size=n)

        # Compute cost matrix (Euclidean distances)
        cost_matrix = cdist(empirical_samples, theoretical_samples, metric='euclidean')

        if pot:
            # Use POT's optimal transport solver
            p = np.ones(n) / n
            q = np.ones(n) / n
            w1 = ot.emd2(p, q, cost_matrix)
        else:
            # Use the Hungarian algorithm from linear sum assignment
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            avg_distance = np.mean(cost_matrix[row_ind, col_ind])
            w1 = avg_distance

        w1_list.append(w1)

    return np.max(w1_list)


def compute_convergence_score(X_OT, A_OT, H_OT, dt, metric='nll'):
    """
    Compute a convergence score based on the chosen metric.

    Parameters:
        metric: one of 'nll' (negative log-likelihood), 'w1' (Wasserstein-1), or 'w2' (Wasserstein-2)
    """
    if metric == 'nll':
        return compute_nll(X_OT, A_OT, H_OT, dt)
    elif metric == 'w2':
        # print('W2 difference in implementation:', compute_w2(X_OT, A_OT, H_OT, dt) - compute_w2_pot(X_OT, A_OT, H_OT, dt))
        return compute_w2(X_OT, A_OT, H_OT, dt)
    elif metric == 'w1':
        return compute_w1(X_OT, A_OT, H_OT, dt)
    elif metric == 'mmd':
        return compute_mmd(X_OT, A_OT, H_OT, dt)
    else:
        raise ValueError("Invalid metric. Choose from 'nll', 'w1', 'w2', or 'mmd'.")