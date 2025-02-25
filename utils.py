import numpy as np
import os
import re

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