import os
import matplotlib.pyplot as plt
from scipy.linalg import expm
import numpy as np

def sample_X0(d, X0_dist=None):
    """
    Samples the initial marginal X0 based on the provided distribution. If none provided, uses standard normal
    Args:
        d (int): Dimension of the process.
        X0_dist (list of tuples, optional): List of tuples containing initial values and their associated probabilities.
    Returns:
        numpy.ndarray: The sampled initial condition X0.
    """
    if X0_dist is not None:
        # Sample X0 from the provided distribution
        return X0_dist[np.random.choice(len(X0_dist), p=[prob for _, prob in X0_dist])][0]
    else:
        # Sample X0 from a standard normal distribution
        cov_matrix = np.eye(d)
        return np.random.multivariate_normal(np.zeros(d), cov_matrix)


def linear_additive_noise_data(num_trajectories, d, T, dt_EM, dt, A, G, X0_dist=None, destroyed_samples=False, shuffle=False):
    '''
    Args:
        num_trajectories: number of trajectories
        d: dimension of process
        T: Total time period
        dt_EM: Euler-Maruyama discretization time step used for simulating the raw trajectories
        dt: discretization time step of the measurements
        A (numpy.ndarray): Drift matrix.
        G (numpy.ndarray): Variance matrix.
        X0_dist (list of tuples): List of tuples, each containing an initial value and its associated probability.
        destroyed_samples: if True, the trajectories will be destroyed at each time step and new ones will be generated
        shuffle: if True, shuffle observations within each time step
    Returns:
        numpy.ndarray: Measured trajectories.
    '''
    n_measured_times = int(T / dt) + 1
    X_measured = np.zeros((num_trajectories, n_measured_times, d))
    rate = int(dt / dt_EM)

    # Generate trajectories
    for n in range(num_trajectories):
        # biological setting to simulate single-cell dataset
        if destroyed_samples:
            for i in range(n_measured_times):
                X0_ = sample_X0(d, X0_dist)  # Sample new X0 for each step
                if i == 0:
                    X_measured[n, i, :] = X0_
                else:
                    measured_T = i * dt
                    X_measured[n, i, :] = linear_additive_noise_trajectory(measured_T, dt_EM, A, G, X0_)[-1]
        else:
            X0_ = sample_X0(d, X0_dist)  # Sample X0 once for the trajectory
            X_true = linear_additive_noise_trajectory(T, dt_EM, A, G, X0_)
            for i in range(n_measured_times):
                X_measured[n, i, :] = X_true[i * rate, :]

    # Shuffle the trajectories within each time step
    if shuffle:
        np.random.shuffle(X_measured)

    return X_measured

def linear_additive_noise_trajectory(T, dt, A, G, X0, seed=None):
    """
    Simulate a single trajectory of a multidimensional linear additive noise process:
    dX_t = AX_tdt + GdW_t
    via Euler Maruyama discretization with time step dt.

    Parameters:
        T (float): Total time period.
        dt (float): Time step size.
        A (numpy.ndarray): Drift matrix.
        G (numpy.ndarray): Variance matrix.
        X0 (numpy.ndarray): Initial value.

    Returns:
        numpy.ndarray: Array of simulated trajectories.
    """
    if seed is not None:
        np.random.seed(seed)
    num_steps = int(T / dt) + 1
    d = len(X0)
    m = G.shape[0]
    dW = np.sqrt(dt) * np.random.randn(num_steps, m)
    X = np.zeros((num_steps, d))
    X[0] = X0

    for t in range(1, num_steps):
        X[t] = X[t - 1] + dt * (A.dot(X[t - 1])) + G.dot(dW[t])

    return X

def generate_G_causal_experiment(causal_sufficiency, d):
    if causal_sufficiency:
        G = np.eye(d)
        np.fill_diagonal(G, np.random.uniform(low=-1, high=1, size=d))
    else:
        G = np.eye(d)
        if d == 3:
            num_columns = np.random.choice([1, 2])
        elif d == 5:
            num_columns = np.random.choice([1, 2, 3])
        elif d == 10:
            num_columns = np.random.choice([1, 2, 3, 4, 5, 6])

        columns_with_shared_noise = np.random.choice(d, num_columns, replace=False)

        for col in columns_with_shared_noise:
            # Create the set I = {0, 1, ..., d-1} and remove the diagonal index col.
            available_rows = list(range(d))
            available_rows.remove(col)
            # Randomly select one row from the off-diagonal entries.
            row = np.random.choice(available_rows)
            # Set the off-diagonal entry to 1.
            G[row, col] = 1
    return G


def trajectory_general_drift(T, dt, drift_func, G, X0, seed=None, interpretation="ito"):
    r"""
    Simulate a single trajectory of an SDE with a general drift function:

        dX_t = f(t, X_t) dt + G dW_t,

    using either an Itô (Euler–Maruyama) scheme or a Stratonovich (Heun) scheme.

    Parameters:
        T (float): Total simulation time.
        dt (float): Time step for discretization.
        drift_func (function): A function f(t, X) returning the drift vector.
        G (np.ndarray): Diffusion matrix.
        X0 (np.ndarray): Initial state (d-dimensional vector).
        seed (int, optional): Random seed for reproducibility.
        interpretation (str): "ito" (default) or "stratonovich".

    Returns:
        np.ndarray: Trajectory array of shape (num_steps, d).
    """
    if seed is not None:
        np.random.seed(seed)
    num_steps = int(T / dt) + 1
    d = len(X0)
    m = G.shape[0]
    X = np.zeros((num_steps, d))
    X[0] = X0

    if interpretation == "ito":
        # Euler–Maruyama (Itô) update
        dW = np.sqrt(dt) * np.random.randn(num_steps, m)
        for t in range(1, num_steps):
            current_time = (t - 1) * dt
            drift = drift_func(current_time, X[t - 1])
            X[t] = X[t - 1] + drift * dt + G.dot(dW[t])
    elif interpretation == "stratonovich":
        # Heun scheme for Stratonovich SDE.
        for t in range(1, num_steps):
            current_time = (t - 1) * dt
            # Sample Brownian increment for this step
            dW = np.sqrt(dt) * np.random.randn(m)
            # Predictor step (Euler-Maruyama)
            X_pred = X[t - 1] + drift_func(current_time, X[t - 1]) * dt + G.dot(dW)
            # Evaluate drift at predicted point at t+dt
            drift_pred = drift_func(current_time + dt, X_pred)
            # Heun update: average drift
            avg_drift = 0.5 * (drift_func(current_time, X[t - 1]) + drift_pred)
            X[t] = X[t - 1] + avg_drift * dt + G.dot(dW)
    else:
        raise ValueError("interpretation must be 'ito' or 'stratonovich'")

    return X


def general_drift_data(num_trajectories, d, T, dt_EM, dt, drift_func, G, X0_dist=None,
                       destroyed_samples=False, shuffle=False, interpretation="ito"):
    r"""
    Generate measurement data from an SDE with a general drift function:

        dX_t = f(t, X_t) dt + G dW_t.

    The simulation uses a fine discretization time step (dt_EM) and
    samples measurements every dt.

    Parameters:
        num_trajectories (int): Number of trajectories.
        d (int): Dimension of the process.
        T (float): Total simulation time.
        dt_EM (float): Euler–Maruyama discretization time step.
        dt (float): Measurement time step.
        drift_func (function): Function f(t, X) defining the drift.
        G (np.ndarray): Diffusion matrix.
        X0_dist (list of tuples, optional): List of (X0, probability) tuples.
        destroyed_samples (bool): If True, sample a new X0 at each measurement.
        shuffle (bool): If True, shuffle the trajectories along the first axis.
        interpretation (str): "ito" (default) or "stratonovich".

    Returns:
        np.ndarray: Measured trajectories of shape (num_trajectories, n_measured_times, d).
    """
    n_measured_times = int(T / dt) + 1
    X_measured = np.zeros((num_trajectories, n_measured_times, d))
    rate = int(dt / dt_EM)

    for n in range(num_trajectories):
        if destroyed_samples:
            for i in range(n_measured_times):
                X0_ = sample_X0(d, X0_dist)
                if i == 0:
                    X_measured[n, i, :] = X0_
                else:
                    measured_T = i * dt
                    X_measured[n, i, :] = trajectory_general_drift(measured_T, dt_EM, drift_func, G, X0_,
                                                                   interpretation=interpretation)[-1]
        else:
            X0_ = sample_X0(d, X0_dist)
            X_true = trajectory_general_drift(T, dt_EM, drift_func, G, X0_,
                                              interpretation=interpretation)
            for i in range(n_measured_times):
                X_measured[n, i, :] = X_true[i * rate, :]

    if shuffle:
        np.random.shuffle(X_measured)
    return X_measured


def polynomial_drift_function(poly_coefs):
    r"""
    Creates a drift function based on polynomial expressions for each coordinate.

    Parameters:
        poly_coefs (list of lists or 2D array-like): Each entry corresponds to one coordinate
            and contains the polynomial coefficients in descending order.
            For example, for a 2D process, poly_coefs could be:
                [ [a_n, ..., a_1, a_0],   # coefficients for coordinate 0
                  [b_m, ..., b_1, b_0] ]   # coefficients for coordinate 1
            The drift in the i-th coordinate will be computed as:
                drift_i(X_i) = polyval(poly_coefs[i], X_i).

    Returns:
        function: A function f(t, X) that ignores t (i.e. time-independent drift) and returns
                  a vector where each element is the polynomial evaluated at the corresponding coordinate.

    Note:
        This drift function is time-independent. If you require explicit time dependence,
        you can modify the returned function accordingly.
    """

    def drift(t, X):
        # Evaluate each coordinate's polynomial drift independently.
        drift_vals = np.array([np.polyval(poly_coefs[i], X[i]) for i in range(len(X))])
        return drift_vals

    return drift

