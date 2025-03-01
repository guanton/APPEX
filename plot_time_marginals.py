import matplotlib.gridspec as gridspec
import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from scipy.stats import multivariate_normal
from data_generation import *
from APPEX import AEOT_trajectory_inference
from plot_trajectories import plot_trajectories

# Parameters
N = 500  # Number of trajectories
dt = 0.05  # Measurement time step
T = 2 # Total time
dt_EM = 0.01  # Euler-Maruyama discretization time step

# Configure plotting options
make_gif = True
plot_individual_trajectories = False
perform_AEOT = False
hist_time_jump = 1
confidence_interval = 0.99
gif_frame_paths = []


def classic_isotropic(identifiable):
    A_list = [np.array([[0, 0], [0, 0]]), np.array([[0, 1], [-1, 0]])]
    G_list = [np.eye(2), np.eye(2)]
    if identifiable:
        points = [np.array([2, 0]), np.array([2, 0.1])]
        X0_dist = [(point, 1 / 2) for point in points]
        gif_filename = "classic_isotropic_identifiable.gif"
    else:
        mean = np.array([0, 0])
        cov = np.eye(2)
        points = np.random.multivariate_normal(mean, cov, size=N)
        X0_dist = [(point, 1 / N) for point in points]
        gif_filename = "classic_isotropic_non_identifiable.gif"
    return A_list, G_list, X0_dist, gif_filename


def skewed_ellipse(identifiable):
    A_list = [np.array([[0, 0], [0, 0]]), np.array([[1, -1], [2, -1]])]
    G_list = [np.array([[1, 1], [2, 0]]), np.array([[1, 1], [2, 0]])]
    if identifiable:
        points = generate_independent_points(2, 2)
        X0_dist = [(point, 1 / 2) for point in points]
        gif_filename = "skewed_ellipse_identifiable.gif"
    else:
        mean = np.array([0, 0])
        cov = np.array([[2, 2], [2, 4]])
        points = np.random.multivariate_normal(mean, cov, size=N)
        X0_dist = [(point, 1 / N) for point in points]
        gif_filename = "skewed_ellipse_non_identifiable.gif"
    return A_list, G_list, X0_dist, gif_filename


def rank_degeneracy_non_identifiability(identifiable):
    A_list = [np.array([[1, 2], [1, 0]]), np.array([[1 / 3, 4 / 3], [2 / 3, -1 / 3]])]
    G_list = [np.array([[1, 2], [-1, -2]]), np.array([[1, 2], [-1, -2]])]
    if identifiable:
        points = [np.array([1,0]),np.array([0,1])] #generate_independent_points(d, d)
        gif_filename = "rank_degeneracy_identifiable.gif"
    else:
        points = [np.array([1, -1])]
        gif_filename = "rank_degeneracy_non_identifiable.gif"
    X0_dist = [(point, 1 / len(points)) for point in points]
    return A_list, G_list, X0_dist, gif_filename


'''
Implement custom list of drift matrices A, diffusion matrices G, to go along with X0_dist and a gif_filename
or use a pre-set (some examples below
'''
# A_list = [np.array([[0]])]
# G_list = [np.eye(1)]
# points = generate_independent_points(1, 1)
# X0_dist = [(point, 1 / len(points)) for point in points]

A_list, G_list, X0_dist, gif_filename = classic_isotropic(True)
# A_list, G_list, X0_dist, gif_filename = rank_degeneracy_non_identifiability(False)
# A_list, G_list, X0_dist, gif_filename = skewed_ellipse(False)


# Temporal marginals storage
X_measured_list = []

# Generate trajectories for each SDE
np.random.seed(42)  # Fix seed for reproducibility
for i, (A, G) in enumerate(zip(A_list, G_list)):
    print(f"Generating trajectories for SDE {i + 1}: A = {A}, G = {G}")
    d = A.shape[0]
    X_measured = linear_additive_noise_data(N, d, T, dt_EM, dt, A, G, X0_dist=X0_dist, destroyed_samples=False,
                                            shuffle=False)
    X_measured_list.append(X_measured)
    if plot_individual_trajectories:
        plot_trajectories(X_measured, T, N_truncate=5, title=f'Raw trajectories with A={A}, G={G}')
    if perform_AEOT:
        X_OT = AEOT_trajectory_inference(X_measured, dt, A, np.matmul(G, G.T))
        plot_trajectories(X_OT, T, N_truncate=5, title=f'AEOT trajectories from marginals given A={A}, G={G}')


# Define helper function to ensure covariance matrix is positive definite
def ensure_positive_definite(cov_matrix, epsilon=1e-6):
    try:
        np.linalg.cholesky(cov_matrix)
    except np.linalg.LinAlgError:
        cov_matrix += np.eye(cov_matrix.shape[0]) * epsilon
    return cov_matrix

# Function to round matrices and convert them to string representation
def matrix_to_text(matrix):
    return np.array2string(np.round(matrix, 2), separator=', ')
def filter_outliers(data, confidence_interval):
    """
    Filters outliers beyond a given confidence interval.
    Returns a boolean mask indicating which rows are within the range.
    """
    lower_bound = np.percentile(data, (1 - confidence_interval) * 50, axis=0)
    upper_bound = np.percentile(data, confidence_interval * 100 + (1 - confidence_interval) * 50, axis=0)
    mask = np.all((data >= lower_bound) & (data <= upper_bound), axis=1)
    return mask


# Determine global axis limits for consistent plotting
num_steps = int(T / dt) + 1
# Padding for the axes to make them less square

# Determine global axis limits for consistent plotting with padding
x_min, x_max, y_min, y_max = None, None, None, None

for i in range(0, num_steps, hist_time_jump):
    for X_measured in X_measured_list:
        time_marginal = X_measured[:, i, :]
        if time_marginal.size > 0:
            x_min_current, x_max_current = np.min(time_marginal[:, 0]), np.max(time_marginal[:, 0])
            y_min_current, y_max_current = np.min(time_marginal[:, 1]), np.max(time_marginal[:, 1])
            x_min = min(x_min, x_min_current) if x_min is not None else x_min_current
            x_max = max(x_max, x_max_current) if x_max is not None else x_max_current
            y_min = min(y_min, y_min_current) if y_min is not None else y_min_current
            y_max = max(y_max, y_max_current) if y_max is not None else y_max_current

# Add padding to the limits to avoid overly square plots
x_min, x_max = (x_min), (x_max)
y_min, y_max = (y_min), (y_max)

# Plot the temporal marginals with adjusted axes and bold/red contours
for i in range(0, num_steps, hist_time_jump):
    num_sdes = len(X_measured_list)
    fig = plt.figure(figsize=(6 * num_sdes, 6))
    gs = gridspec.GridSpec(1, num_sdes)

    plt.suptitle(f'Marginal at time {round(i * dt, 2)}', fontsize=18, y=0.92)

    for m, X_measured in enumerate(X_measured_list):
        ax = fig.add_subplot(gs[0, m])
        time_marginal = X_measured[:, i, :]

        # Filter outliers more aggressively
        mask = filter_outliers(time_marginal, confidence_interval)
        filtered_time_marginal = time_marginal[mask]

        # Create a grid for the Gaussian PDF
        xgrid = np.linspace(x_min, x_max, 100)
        ygrid = np.linspace(y_min, y_max, 100)
        X, Y = np.meshgrid(xgrid, ygrid)

        # Compute Gaussian PDF over the grid
        empirical_mean = np.mean(filtered_time_marginal, axis=0)
        empirical_covariance = np.cov(filtered_time_marginal, rowvar=False)
        empirical_covariance = ensure_positive_definite(empirical_covariance)

        pos = np.dstack((X, Y))
        rv = multivariate_normal(empirical_mean, empirical_covariance, allow_singular=True)
        Z = rv.pdf(pos)

        #
        # Plot bold red contours with more levels and ensure strictly increasing levels
        num_contour_levels = 10  # Increase the number of contours for more detail
        contour_levels = np.linspace(Z.min(), Z.max(), num_contour_levels)

        # Ensure contour_levels is strictly increasing by checking unique levels
        contour_levels = np.unique(np.clip(contour_levels, Z.min() + 1e-1000, Z.max()))
        ax.contour(X, Y, Z, levels=contour_levels, colors='red', linewidths=2.0, zorder=10)  # Thicker contours

        # Plot distinct points after the contours with a lower zorder
        ax.scatter(filtered_time_marginal[:, 0], filtered_time_marginal[:, 1],
                   color='grey', s=20, zorder=5, marker='o', alpha=0.8)

        # Set uniform axis limits across all frames with added padding
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.axhline(0, color='grey', linewidth=0.5)
        ax.axvline(0, color='grey', linewidth=0.5)
        ax.set_xlabel('X', fontsize=18)
        ax.set_ylabel('Y', fontsize=18)
        ax.tick_params(labelsize=10)
        ax.tick_params(axis='both', which='major', labelsize=12)

        drift_text = f'A={matrix_to_text(A_list[m])}'
        diffusion_text = f'G={matrix_to_text(G_list[m])}'
        if m == 0:
            x_adjust = 0.25
        else:
            x_adjust = 0.27
        ax.text(x_adjust, 0.95, drift_text, transform=ax.transAxes, fontsize=15, verticalalignment='top', horizontalalignment='right',
                bbox=dict(facecolor='lightgrey', alpha=0.7))
        ax.text(0.95, 0.95, diffusion_text, transform=ax.transAxes, fontsize=15, verticalalignment='top',
                horizontalalignment='right', bbox=dict(facecolor='lightgrey', alpha=0.7))

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save each frame for the GIF
    if make_gif:
        frame_filename = f"marginals_gifs/frame_{i}.png"
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(frame_filename)
        gif_frame_paths.append(frame_filename)
    plt.close()


if make_gif:
    os.makedirs('marginals_gifs', exist_ok=True)
    gif_file_path = os.path.join('marginals_gifs', gif_filename)
    with imageio.get_writer(gif_file_path, mode='I', duration=1, loop=0) as writer:
        for frame_path in gif_frame_paths:
            image = imageio.imread(frame_path)
            writer.append_data(image)

    for frame_path in gif_frame_paths:
        os.remove(frame_path)
    print(f"GIF saved as {gif_filename}")


