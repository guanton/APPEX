import os
import matplotlib.pyplot as plt
from scipy.linalg import expm
import numpy as np

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

def additive_noise_trajectory_inhomogeneous(T, dt, A_func, G, X0, seed=None):
    """
    Simulate a single trajectory of a multidimensional linear additive noise process with time-inhomogeneous drift:
    dX_t = A(t)X_tdt + GdW_t
    via Euler Maruyama discretization with time step dt.

    Parameters:
        T (float): Total time period.
        dt (float): Time step size.
        A_func (function): Function that takes in time t and returns the drift matrix A(t).
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
        current_time = (t - 1) * dt
        A_t = A_func(current_time)
        X[t] = X[t - 1] + dt * (A_t.dot(X[t - 1])) + G.dot(dW[t])

    return X

def example_A_func(t):
    # Define a simple time-inhomogeneous drift matrix A(t) that changes with time
    A_t = np.array([[-0.1 * (1 + t), 0], [0, -0.2 * (1 + 0.5 * t)]])
    return A_t

def generate_G_causal_experiment(causal_sufficiency, d):
    if causal_sufficiency:
        G = np.eye(d)
        np.fill_diagonal(G, np.random.uniform(low=0.1, high=1, size=d))
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
            num_nonzero_entries = 2  # latent confounder will be over 2 variables
            nonzero_indices = np.random.choice(d, num_nonzero_entries, replace=False)
            G[nonzero_indices, col] = 1
    return G
def plot_trajectories(X, T, save_file=False, N_truncate=None, title = 'SDE trajectories'):
    """
    Plot the trajectories of a multidimensional process.

    Parameters:
        X (numpy.ndarray): Array of trajectories.
        T (float): Total time period.
        dt (float): Time step size.
    """
    num_trajectories, num_steps, num_dimensions = X.shape
    if N_truncate is not None:
        num_trajectories = N_truncate

    time_steps = np.linspace(0, T, num_steps)  # Generate time steps corresponding to [0, T]

    # Plot trajectories
    plt.figure(figsize=(12, 8))

    for n in range(num_trajectories):
        for d in range(num_dimensions):
            plt.plot(time_steps, X[n, :, d], label=f'{n}th trajectory dim {d}')

    plt.title(title, fontsize=20)
    plt.xlabel('Time', fontsize=16)
    plt.ylabel('Value', fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    if save_file:
        os.makedirs('Raw_trajectory_figures', exist_ok=True)
        plot_filename = os.path.join('Raw_trajectory_figures', f"raw_trajectory_d-{num_dimensions}_stationary.png")
        plt.savefig(plot_filename)
    plt.show()