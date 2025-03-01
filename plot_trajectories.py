import numpy as np
import matplotlib.pyplot as plt
import os
import imageio.v2 as imageio
from data_generation import *

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



# --- Define function to produce a side-by-side animated GIF ---
def plot_side_by_side_trajectories_gif(X1, X2, T, dt, save_path="comparison.gif", fps=5,
                                       title1="SDE 1", title2="SDE 2", N_truncate=None):
    """
    Create an animated GIF that shows side-by-side cumulative trajectories for two 2D SDEs.

    Each frame corresponds to a time snapshot. For each SDE, the cumulative paths (up to that time)
    are plotted with a red dot marking the current position.

    Parameters:
        X1 (np.ndarray): Trajectories from SDE 1 of shape (num_traj, num_steps, 2)
        X2 (np.ndarray): Trajectories from SDE 2 of shape (num_traj, num_steps, 2)
        T (float): Total simulation time.
        dt (float): Measurement time step.
        save_path (str): Path to save the resulting GIF.
        fps (int): Frames per second.
        title1 (str): Title for left subplot.
        title2 (str): Title for right subplot.
        N_truncate (int, optional): Plot only this many trajectories (default: all).
    """
    num_traj1, num_steps, _ = X1.shape
    num_traj2, _, _ = X2.shape
    if N_truncate is not None:
        num_traj1 = min(num_traj1, N_truncate)
        num_traj2 = min(num_traj2, N_truncate)

    # Determine global axis limits (same for both plots)
    pts1 = X1[:num_traj1].reshape(-1, 2)
    pts2 = X2[:num_traj2].reshape(-1, 2)
    all_pts = np.vstack((pts1, pts2))
    x_min, x_max = np.min(all_pts[:, 0]), np.max(all_pts[:, 0])
    y_min, y_max = np.min(all_pts[:, 1]), np.max(all_pts[:, 1])

    # Create temporary folder for frames
    frames_folder = "temp_frames"
    os.makedirs(frames_folder, exist_ok=True)
    frame_paths = []

    # Create time array for annotation
    time_array = np.linspace(0, T, num_steps)

    for frame in range(num_steps):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Left subplot: SDE 1
        ax = axes[0]
        ax.set_title(f"{title1} (t={time_array[frame]:.2f})", fontsize=16)
        for i in range(num_traj1):
            traj = X1[i, :frame + 1, :]
            ax.plot(traj[:, 0], traj[:, 1], color='blue', alpha=0.7)
            ax.scatter(traj[-1, 0], traj[-1, 1], color='red', s=30)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True)

        # Right subplot: SDE 2
        ax = axes[1]
        ax.set_title(f"{title2} (t={time_array[frame]:.2f})", fontsize=16)
        for i in range(num_traj2):
            traj = X2[i, :frame + 1, :]
            ax.plot(traj[:, 0], traj[:, 1], color='green', alpha=0.7)
            ax.scatter(traj[-1, 0], traj[-1, 1], color='red', s=30)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True)

        plt.tight_layout()
        frame_filename = os.path.join(frames_folder, f"frame_{frame:04d}.png")
        plt.savefig(frame_filename)
        frame_paths.append(frame_filename)
        plt.close(fig)

    # Write frames into a GIF
    with imageio.get_writer(save_path, mode='I', duration=1 / fps) as writer:
        for frame_filename in frame_paths:
            image = imageio.imread(frame_filename)
            writer.append_data(image)

    # Clean up temporary files
    for frame_filename in frame_paths:
        os.remove(frame_filename)
    os.rmdir(frames_folder)

    print(f"GIF saved at {save_path}")

# --- Define the SDE parameters for the degenerate counterexample ---
T = 2 # Total simulation time
dt_EM = 0.001  # Fine Euler–Maruyama time step
dt = 0.01  # Measurement time step
num_trajectories = 1
d = 2

def wang_example(T, dt_EM, dt, d, X0, num_trajectories=1):
    # Drift matrices (linear case) from the counterexample
    A1 = np.array([[1, 2],
                   [1, 0]])
    A2 = np.array([[1 / 3, 4 / 3],
                   [2 / 3, -1 / 3]])

    # Define the drift functions as lambda functions: f(t, X) = A.dot(X)
    drift1 = lambda t, X: A1.dot(X)
    drift2 = lambda t, X: A2.dot(X)

    # Diffusion matrix (rank-degenerate)
    G = np.array([[1, 2],
                  [-1, -2]])

    # Use a distribution with a single point for X0.
    X0_dist = [(X0, 1.0)]

    # --- Generate trajectories using the generalized function ---
    np.random.seed(42)
    data1 = general_drift_data(num_trajectories, d, T, dt_EM, dt, drift1, G, X0_dist,
                               destroyed_samples=False, shuffle=False)

    # Reset seed so that both systems use the same noise realization.
    np.random.seed(42)
    data2 = general_drift_data(num_trajectories, d, T, dt_EM, dt, drift2, G, X0_dist,
                               destroyed_samples=False, shuffle=False)


    # --- Produce the animated GIF ---
    plot_side_by_side_trajectories_gif(data1, data2, T, dt, save_path="degenerate_comparison.gif", fps=5,
                                       title1="SDE: A1", title2="SDE: A2", N_truncate=1)


def polynomial_example(T, dt_EM, dt, d, X0, num_trajectories=1):
    # --- Define our drift functions using polynomial evaluations ---
    # We want to use a polynomial for f1(x)=0.5*x^2 - x.
    # The coefficients (in descending order) are: [0.5, -1.0, 0.0]
    poly_coefs_f1 = [-0.5, -1.0, 0.0]

    # Although we have polynomial_drift_function, our canonical drift requires
    # using f1 on the first coordinate and then 2*x*f1(x) on the second.
    def drift1(t, Y):
        """
        Canonical quadratic drift.
        Y[0] is interpreted as x.
        """
        x = Y[0]
        f1 = np.polyval(poly_coefs_f1, x)
        return np.array([f1, 2 * x * f1])

    def drift2(t, Y):
        """
        Modified quadratic drift with extra (vanishing) dynamics.
        Extra term is (x^2 - Y[1]), so that when Y[1]==x^2 the extra term vanishes.
        """
        x = Y[0]
        extra = (x ** 2 - Y[1])
        f1 = np.polyval(poly_coefs_f1, x)
        return np.array([f1 + extra, 2 * x * f1 + extra])

    # Use zero diffusion to remain exactly on the parabola.
    sigma = 0.1
    G = np.array([[sigma, 0],
                  [2 * alpha * sigma, 0]])

    X0_dist = [(X0, 1.0)]

    # --- Generate trajectories using the provided general_drift_data function ---
    np.random.seed(42)
    data_canonical = general_drift_data(num_trajectories, d, T, dt_EM, dt, drift1, G, X0_dist,
                                        destroyed_samples=False, shuffle=False, interpretation="stratonovich")
    np.random.seed(42)
    data_modified = general_drift_data(num_trajectories, d, T, dt_EM, dt, drift2, G, X0_dist,
                                       destroyed_samples=False, shuffle=False, interpretation="stratonovich")

    # Extract the single trajectory (shape: (num_steps, 2))
    Y_canonical = data_canonical[0][None, :]  # Shape becomes (1, num_steps, 2)
    Y_modified = data_modified[0][None, :]
    plot_side_by_side_trajectories_gif(Y_canonical, Y_modified, T, dt,
                                       save_path="quadratic_invariant_comparison.gif",
                                       fps=5,
                                       title1="Canonical Quadratic Process",
                                       title2="Modified Quadratic Process")


# # Fixed initial condition X0 = [1, -1]
# X0 = np.array([1, -1])
# wang_example(T, dt_EM, dt, d, X0, num_trajectories=num_trajectories)

alpha = 0.5
X0 = np.array([alpha, alpha ** 2])
polynomial_example(T, dt_EM, dt, d, X0, num_trajectories=num_trajectories)