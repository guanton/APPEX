import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting

# -------------------------------------------------------------------
# 1. Sphere with Multiple Rotations (light blue surface)
# -------------------------------------------------------------------
def plot_sphere():
    # Create a unit sphere using spherical coordinates.
    phi = np.linspace(0, np.pi, 30)
    theta = np.linspace(0, 2 * np.pi, 30)
    phi, theta = np.meshgrid(phi, theta)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, color='lightgrey', alpha=0.2,
                    edgecolor='k', linewidth=0.5)

    # Define a set of rotation angles.
    t_rot = np.linspace(0, 2 * np.pi, 100)
    dash_indices = np.linspace(0, len(t_rot) - 1, 20, dtype=int)

    # Rotation in the xy–plane.
    for idx in dash_indices:
        bx, by, bz = np.cos(t_rot[idx]), np.sin(t_rot[idx]), 0
        dx, dy, dz = -np.sin(t_rot[idx]), np.cos(t_rot[idx]), 0
        ax.quiver(bx, by, bz, dx, dy, dz,
                  color='red', length=0.15, normalize=True,
                  linewidth=3.0, alpha=1.0, zorder=10)

    # Rotation in the xz–plane.
    for idx in dash_indices:
        bx, by, bz = np.cos(t_rot[idx]), 0, np.sin(t_rot[idx])
        dx, dy, dz = -np.sin(t_rot[idx]), 0, np.cos(t_rot[idx])
        ax.quiver(bx, by, bz, dx, dy, dz,
                  color='red', length=0.15, normalize=True,
                  linewidth=3.0, alpha=1.0, zorder=10)

    # Rotation in the yz–plane.
    for idx in dash_indices:
        bx, by, bz = 0, np.cos(t_rot[idx]), np.sin(t_rot[idx])
        dx, dy, dz = 0, -np.sin(t_rot[idx]), np.cos(t_rot[idx])
        ax.quiver(bx, by, bz, dx, dy, dz,
                  color='red', length=0.15, normalize=True,
                  linewidth=3.0, alpha=1.0, zorder=10)

    # ax.set_axis_off()
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.tight_layout()
    plt.savefig("sphere_rv.pdf", format="pdf", bbox_inches='tight')
    plt.show()


# -------------------------------------------------------------------
# 2. Tilted Cylinder with Elliptical Cross–Sections
# -------------------------------------------------------------------
def plot_cylinder():
    # Parameterize a standard cylinder (radius = 1, height in [-0.5, 0.5]).
    theta = np.linspace(0, 2 * np.pi, 40)
    z_vals = np.linspace(-0.5, 0.5, 30)
    theta, z_vals = np.meshgrid(theta, z_vals)
    x = np.cos(theta)
    y = np.sin(theta)
    z = z_vals

    # Build a transformation: first scale (to get elliptical stacks)...
    S = np.diag([0.5,  0.2, 0.9])
    # ... then rotate about the x–axis by 20° to tilt the cylinder.
    angle = np.deg2rad(20)
    R = np.array([[1, 0, 0],
                  [0, np.cos(angle), -np.sin(angle)],
                  [0, np.sin(angle), np.cos(angle)]])
    T = R @ S  # overall transformation

    pts = np.vstack((x.flatten(), y.flatten(), z.flatten()))
    pts_t = T @ pts
    x_t = pts_t[0].reshape(x.shape)
    y_t = pts_t[1].reshape(y.shape)
    z_t = pts_t[2].reshape(z.shape)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x_t, y_t, z_t, color='lightgrey', alpha=0.2,
                    edgecolor='k', linewidth=0.5)

    # Overlay rotation arrows on a transformed base circle.
    t_rot = np.linspace(0, 2 * np.pi, 100)
    offset = 0.02  # small offset to lift arrows above the surface
    base_circle = np.vstack((np.cos(t_rot),
                             np.sin(t_rot),
                             np.full_like(t_rot, offset)))
    base_circle_t = T @ base_circle
    # Compute tangent directions on the base circle.
    d_circle = np.vstack((-np.sin(t_rot),
                          np.cos(t_rot),
                          np.zeros_like(t_rot)))
    d_circle_t = T @ d_circle
    dash_indices = np.linspace(0, len(t_rot) - 1, 20, dtype=int)
    for idx in dash_indices:
        ax.quiver(base_circle_t[0, idx],
                  base_circle_t[1, idx],
                  base_circle_t[2, idx],
                  d_circle_t[0, idx],
                  d_circle_t[1, idx],
                  d_circle_t[2, idx],
                  color='red', length=0.05, normalize=True,
                  linewidth=2.0, alpha=1.0, zorder=10)

    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.tight_layout()
    plt.savefig("cylinder_rv.pdf", format="pdf", bbox_inches='tight')
    plt.show()


# -------------------------------------------------------------------
# 3. Gaussian Pancake: Scatter of Points from N(0,1) x Unif(-1,1)
# -------------------------------------------------------------------
def plot_gaussian_pancake():
    np.random.seed(42)
    n_points = 10000
    # x ~ N(0,1), y ~ Unif(-1,1)
    x = np.random.normal(0, 1, n_points)
    y = np.random.uniform(-1, 1, n_points)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.scatter(x, y, color='dimgrey', alpha=0.6, s=20)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig("gaussian_pancake_rv.pdf", format="pdf", bbox_inches='tight')
    plt.show()


# -------------------------------------------------------------------
# 4. Degenerate Distribution: Points Supported on a Line
# -------------------------------------------------------------------
def plot_degenerate():
    # Create 5 points exactly on the line y = x from -0.5 to 0.5.
    import numpy as np
    import matplotlib.pyplot as plt

    t = np.linspace(-0.5, 0.5, 5)  # 5 points
    points = np.column_stack((t, t))  # Each row is [t, t]

    fig, ax = plt.subplots(figsize=(8, 6))
    # Overlay the individual points.
    ax.scatter(points[:, 0], points[:, 1], color='dimgrey', s=30, zorder=3)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ticks = np.linspace(-0.5, 0.5, 5)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig("degenerate_rv.pdf", format="pdf", bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    plot_sphere()
    plot_cylinder()
    plot_gaussian_pancake()
    plot_degenerate()
