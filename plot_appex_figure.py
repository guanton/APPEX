import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def simulate_OU_paths(x0_vals, mu, sigma, T=1.0, N=50, paths_per_init=1):
    """
    Simulate sample paths of the Ornstein–Uhlenbeck SDE
        dX_t = -mu * X_t dt + sigma dW_t
    via Euler–Maruyama on a time grid of length N+1.

    Returns
    -------
    t_grid : (N+1,) array
        Times from 0 to T in steps of T/N.
    X : 2D array of shape (M, N+1)
        M = len(x0_vals)*paths_per_init distinct sample paths (rows).
    """
    dt = T / N
    t_grid = np.linspace(0, T, N + 1)

    x0_vals = np.asarray(x0_vals)
    M = len(x0_vals) * paths_per_init
    X = np.zeros((M, N + 1))

    idx = 0
    for x0 in x0_vals:
        for _ in range(paths_per_init):
            X[idx, 0] = x0
            idx += 1

    # Euler–Maruyama updates
    for n in range(N):
        Xn = X[:, n]
        dW = np.random.randn(M) * np.sqrt(dt)
        X[:, n + 1] = Xn - mu * Xn * dt + sigma * dW

    return t_grid, X


# -------------------------------------------------------------------
# 1) Generate "inferred" paths => also used as observation data
# -------------------------------------------------------------------
np.random.seed(123)

mu_inferred = 1.8
sigma_inferred = 0.5
x0_vals = np.array([-3, 0, 3])

t_grid, X_inferred = simulate_OU_paths(
    x0_vals, mu_inferred, sigma_inferred,
    T=1.0, N=50, paths_per_init=10
)

times_for_obs = np.linspace(0, 1.0, 6)
obs_indices = (times_for_obs / (1.0 / 50)).astype(int)

obs_t_list = []
obs_x_list = []

# At t=0: pick exactly 1 path from each of the 3 blocks [0..9], [10..19], [20..29]
paths_for_t0 = [0, 10, 20]
for pidx in paths_for_t0:
    obs_t_list.append(t_grid[0])
    obs_x_list.append(X_inferred[pidx, 0])

# For t=0.2,...,1.0: use ALL 30 paths => 30 obs/time
for idx_t in obs_indices[1:]:
    for pidx in range(X_inferred.shape[0]):
        obs_t_list.append(t_grid[idx_t])
        obs_x_list.append(X_inferred[pidx, idx_t])

obs_t = np.array(obs_t_list)
obs_x = np.array(obs_x_list)

X_updated = X_inferred  # same paths for middle & right

# -------------------------------------------------------------------
# 2) Make the 3‐panel figure
# -------------------------------------------------------------------
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

# Left panel: Observations
axes[0].scatter(obs_t, obs_x, s=25, color='blue', label="Observations")
axes[0].set_title("Temporal marginals from underlying SDE", fontsize=12)
axes[0].set_xlabel("Time (t)")

# Middle panel: Inferred trajectory
for i in range(X_inferred.shape[0]):
    axes[1].plot(t_grid, X_inferred[i], color='purple', alpha=0.5)
axes[1].scatter(obs_t, obs_x, s=25, color='blue')
axes[1].set_title(
    "Trajectory inference of $q^{(k)}$ given \n"
    + "reference SDE: $p_{A^{(k-1)},H^{(k-1)}}$",
    fontsize=12
)
axes[1].set_xlabel("Time (t)")

# Right panel: Updated reference SDE
for i in range(X_updated.shape[0]):
    axes[2].plot(t_grid, X_updated[i], color='purple', alpha=0.5)
axes[2].set_title(
    "Obtain MLEs $A^{(k)},H^{(k)}$ given $q^{(k)}$ \n"
    + "and update reference SDE: $p_{A^{(k)},H^{(k)}}$",
    fontsize=12
)
axes[2].set_xlabel("Time (t)")

# Remove y‐axis ticks
for ax in axes:
    ax.set_yticks([])

# -------------------------------------------------------------------
# 3) Reserve some space on top so the heading won't overlap
# -------------------------------------------------------------------
# tight_layout uses only 88% of figure's vertical space => 12% left on top
plt.tight_layout(rect=[0, 0, 1, 0.88])

# -------------------------------------------------------------------
# 4) Draw a dashed vertical line between left subplot & the other two
# -------------------------------------------------------------------
fig.canvas.draw()  # finalize positions

pos_left  = axes[0].get_position()
pos_mid   = axes[1].get_position()

# x = midpoint between left subplot's right edge and middle subplot's left edge
sep_x = 0.5 * (pos_left.x1 + pos_mid.x0)

line = Line2D(
    [sep_x, sep_x],
    [0.05, 0.95],   # draw line from 10% to 90% of figure height
    color='black',
    linewidth=1.5,
    linestyle='--',
    transform=fig.transFigure,
    clip_on=False
)
fig.add_artist(line)

# -------------------------------------------------------------------
# 5) Put "APPEX iteration k" above the second & third subplots
# -------------------------------------------------------------------
pos_mid   = axes[1].get_position()
pos_right = axes[2].get_position()

# We can horizontally center above subplots 2 & 3:
center_x = 0.5 * (pos_mid.x0 + pos_right.x1)
# We'll place the text at ~0.94 in figure coords (above the top margin = 0.88)
top_y = 0.88

fig.text(
    center_x,
    top_y,
    "APPEX iteration k",
    ha='center',
    va='bottom',
    fontsize=14,
    transform=fig.transFigure
)

first_center_x = 0.5*(pos_mid.x0)

fig.text(
    first_center_x,
    top_y,
    "Observational data",
    ha='center',
    va='bottom',
    fontsize=14,
    transform=fig.transFigure
)


plt.show()
