import matplotlib.gridspec as gridspec
import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from data_generation import *
from APPEX import GEOT_trajectory_inference

# Parameters
N = 500  # Number of trajectories
dt = 0.05  # Measurement time step
T = 1  # Total time
dt_EM = 0.01  # Euler-Maruyama discretization time step

# Configure plotting options
make_gif = True
plot_individual_trajectories = False
perform_GEOT = False
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


A_list, G_list, X0_dist, gif_filename = classic_isotropic(False)
# A_list, G_list, X0_dist, gif_filename = rank_degeneracy_non_identifiability(True)
# A_list, G_list, X0_dist, gif_filename = skewed_ellipse(True)
# A_list = [np.array([[0]])]
# G_list = [np.eye(1)]
# points = generate_independent_points(1, 1)
# X0_dist = [(point, 1 / len(points)) for point in points]


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
    if perform_GEOT:
        X_OT = GEOT_trajectory_inference(X_measured, dt, A, np.matmul(G, G.T))
        plot_trajectories(X_OT, T, N_truncate=5, title=f'GEOT trajectories from marginals given A={A}, G={G}')


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
x_min, x_max, y_min, y_max = None, None, None, None

for i in range(0, num_steps, hist_time_jump):
    for X_measured in X_measured_list:
        time_marginal = X_measured[:, i, :]
        x_min_current, x_max_current = np.min(time_marginal[:, 0]), np.max(time_marginal[:, 0])
        y_min_current, y_max_current = np.min(time_marginal[:, 1]), np.max(time_marginal[:, 1])
        x_min = x_min_current if x_min is None or x_min_current < x_min else x_min
        x_max = x_max_current if x_max is None or x_max_current > x_max else x_max
        y_min = y_min_current if y_min is None or y_min_current < y_min else y_min
        y_max = y_max_current if y_max is None or y_max_current > y_max else y_max

# Plot the temporal marginals at each time step
for i in range(0, num_steps, hist_time_jump):
    num_sdes = len(X_measured_list)
    fig = plt.figure(figsize=(6 * num_sdes, 6))
    gs = gridspec.GridSpec(1, num_sdes)

    # Global title about time marginal
    global_title = f'Marginal at time {round(i * dt, 2)}'
    plt.suptitle(global_title, fontsize=16, y=0.95)

    for m, X_measured in enumerate(X_measured_list):
        ax = fig.add_subplot(gs[0, m])
        time_marginal = X_measured[:, i, :]

        # Filter outliers
        mask = filter_outliers(time_marginal, confidence_interval)
        filtered_time_marginal = time_marginal[mask]

        # Add artificial points at the corners to ensure no white space
        artificial_points = np.array([
            [x_min, y_min],
            [x_min, y_max],
            [x_max, y_min],
            [x_max, y_max]
        ])
        extended_data = np.vstack([filtered_time_marginal, artificial_points])

        # Create a 2D histogram of the marginal data
        H, xedges, yedges = np.histogram2d(extended_data[:, 0], extended_data[:, 1], bins=(100, 100), density=True)
        # Create meshgrid for the plot
        x_midpoints = (xedges[:-1] + xedges[1:]) / 2
        y_midpoints = (yedges[:-1] + yedges[1:]) / 2
        X, Y = np.meshgrid(x_midpoints, y_midpoints)

        # Plot the filled contour to fill the background
        contour = ax.contourf(X, Y, H.T, levels=100, cmap='viridis')

        # Calculate the empirical mean and covariance
        empirical_mean = np.mean(filtered_time_marginal, axis=0)
        empirical_covariance = np.cov(filtered_time_marginal, rowvar=False)
        empirical_covariance = ensure_positive_definite(empirical_covariance)

        # Create a grid for the Gaussian PDF
        pos = np.dstack((X, Y))
        rv = multivariate_normal(empirical_mean, empirical_covariance, allow_singular=True)
        Z = rv.pdf(pos)

        # Overlay the Gaussian PDF as a contour plot with a few contour levels
        num_contour_levels = 5  # Choose the number of contour levels (e.g., 5 for a few rings)
        contour_levels = np.linspace(Z.min(), Z.max(), num_contour_levels)

        # Ensure that contour levels are strictly increasing and within the range of the data
        contour_levels = np.sort(np.unique(contour_levels))
        contour_levels = contour_levels[contour_levels > Z.min()]  # Remove levels at or below Z.min()
        ax.contour(X, Y, Z, levels=contour_levels, colors='red')

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.axhline(0, color='white', linewidth=.5)
        ax.axvline(0, color='white', linewidth=.5)
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')

        # Set subplot title with equation
        # Add drift and diffusion matrices to the plot
        drift_text = f'A={matrix_to_text(A_list[m])}'
        diffusion_text = f'G={matrix_to_text(G_list[m])}'
        ax.text(0.05, 0.95, drift_text, transform=ax.transAxes, fontsize=16, verticalalignment='top',
                bbox=dict(facecolor='lightblue', alpha=0.5))
        ax.text(0.95, 0.95, diffusion_text, transform=ax.transAxes, fontsize=16, verticalalignment='top',
                horizontalalignment='right', bbox=dict(facecolor='lightcoral', alpha=0.5))

    # Save each frame for the GIF
    if make_gif:
        frame_filename = f"marginals_gifs/frame_{i}.png"
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(frame_filename)
        gif_frame_paths.append(frame_filename)
    plt.close()

if make_gif:
    # Create GIF from saved frames
    os.makedirs('marginals_gifs', exist_ok=True)
    gif_file_path = os.path.join('marginals_gifs', gif_filename)
    with imageio.get_writer(gif_file_path, mode='I', duration=1, loop=0) as writer:
        for frame_path in gif_frame_paths:
            image = imageio.imread(frame_path)
            writer.append_data(image)

    print(f"GIF saved as {gif_filename}")

    # Delete saved frames after GIF creation
    for frame_path in gif_frame_paths:
        os.remove(frame_path)
