import numpy as np
import os
import re

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