import os

import numpy as np
from scipy.interpolate import RegularGridInterpolator


class FlowField:
    """Static 2D fluid flow velocity field with bilinear interpolation."""

    def __init__(self, flow_field_path=None):
        if flow_field_path is None:
            flow_field_path = os.path.join(os.path.dirname(__file__), 'assets', 'yellow_path_field.npy')
        data = np.load(flow_field_path, allow_pickle=True).item()
        self.x_range = data['x_range']
        self.y_range = data['y_range']
        self.vx_grid = data['vx_grid']
        self.vy_grid = data['vy_grid']

        xs = np.linspace(self.x_range[0], self.x_range[1], self.vx_grid.shape[1])
        ys = np.linspace(self.y_range[0], self.y_range[1], self.vx_grid.shape[0])
        self._interp_vx = RegularGridInterpolator((ys, xs), self.vx_grid, bounds_error=False, fill_value=0.0)
        self._interp_vy = RegularGridInterpolator((ys, xs), self.vy_grid, bounds_error=False, fill_value=0.0)

    def get_flow(self, x, y):
        """Return (vx, vy) flow velocity at position (x, y)."""
        pt = np.array([y, x])
        return float(self._interp_vx(pt)), float(self._interp_vy(pt))

    def get_flow_grid(self, xs, ys):
        """Return flow on a meshgrid for visualization. xs and ys are 1D arrays."""
        yy, xx = np.meshgrid(ys, xs, indexing='ij')
        pts = np.stack([yy.ravel(), xx.ravel()], axis=-1)
        vx = self._interp_vx(pts).reshape(yy.shape)
        vy = self._interp_vy(pts).reshape(yy.shape)
        return vx, vy


MAX_FLOW_MAGNITUDE = 1.8  # Agent max displacement is 0.2/step; flow displacement = 0.1 * mag.


def _make_grid(x_range=(-4.0, 24.0), y_range=(-4.0, 24.0), n=256):
    """Create a high-resolution grid for flow field generation."""
    xs = np.linspace(x_range[0], x_range[1], n)
    ys = np.linspace(y_range[0], y_range[1], n)
    xx, yy = np.meshgrid(xs, ys)
    return xs, ys, xx, yy, x_range, y_range


def _stream_function_field(psi, xs, ys):
    """Compute divergence-free velocity from a stream function (curl of psi)."""
    dy = ys[1] - ys[0]
    dx = xs[1] - xs[0]
    vx = np.gradient(psi, dy, axis=0)
    vy = -np.gradient(psi, dx, axis=1)
    return vx, vy


def _normalize(vx, vy, target_max=MAX_FLOW_MAGNITUDE):
    """Rescale velocity field so max magnitude equals target_max."""
    mag = np.sqrt(vx ** 2 + vy ** 2)
    scale = target_max / (mag.max() + 1e-8)
    return vx * scale, vy * scale


def _save_field(save_path, x_range, y_range, vx, vy, label='flow field'):
    """Normalize and save a flow field."""
    vx, vy = _normalize(vx, vy)
    data = dict(x_range=np.array(x_range), y_range=np.array(y_range),
                vx_grid=vx.astype(np.float32), vy_grid=vy.astype(np.float32))
    np.save(save_path, data)
    mag = np.sqrt(vx ** 2 + vy ** 2)
    print(f'Saved {label} to {save_path}  (max mag={mag.max():.2f})')
    return save_path


def _random_fourier_stream(xx, yy, rng, n_modes=60, k_range=(0.1, 1.2),
                           amplitude_decay=1.0):
    """Build a stream function from random Fourier modes at multiple scales.

    More modes and wider k_range produce richer, more turbulent-looking fields.
    amplitude_decay controls energy spectrum: higher = more energy at large scales.
    """
    psi = np.zeros_like(xx)
    for _ in range(n_modes):
        kx = rng.uniform(k_range[0], k_range[1]) * rng.choice([-1, 1])
        ky = rng.uniform(k_range[0], k_range[1]) * rng.choice([-1, 1])
        k_mag = np.sqrt(kx ** 2 + ky ** 2)
        phase = rng.uniform(0, 2 * np.pi)
        # Kolmogorov-like energy spectrum: amplitude ~ k^(-amplitude_decay)
        amp = rng.uniform(0.5, 1.5) / (k_mag ** amplitude_decay + 0.1)
        psi += amp * np.sin(kx * xx + ky * yy + phase)
    return psi


def generate_default_flow_field(save_path=None):
    """Dense multi-scale turbulent flow with a mild rightward mean drift.

    Uses 80 Fourier modes across a wide wavenumber range for rich swirling structure,
    plus a gentle mean current. Divergence-free.
    """
    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__), 'assets', 'default_flow_field.npy')

    rng = np.random.RandomState(42)
    xs, ys, xx, yy, x_range, y_range = _make_grid()

    psi = _random_fourier_stream(xx, yy, rng, n_modes=80, k_range=(0.08, 1.5),
                                 amplitude_decay=0.8)
    vx, vy = _stream_function_field(psi, xs, ys)

    # Add a mild mean drift (rightward + slight downward)
    vx += 0.3
    vy -= 0.15

    return _save_field(save_path, x_range, y_range, vx, vy, 'default flow field')


def generate_double_vortex_field(save_path=None):
    """Two large counter-rotating vortices with dense small-scale turbulence overlaid.

    The large-scale structure creates sweeping currents while the fine-scale modes
    add realistic eddy detail throughout.
    """
    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__), 'assets', 'double_vortex_field.npy')

    rng = np.random.RandomState(7)
    xs, ys, xx, yy, x_range, y_range = _make_grid()

    # Large-scale vortex pair via stream function: two Gaussian blobs
    cx1, cy1 = 4.0, 10.0
    cx2, cy2 = 16.0, 10.0
    psi_large = (8.0 * np.exp(-((xx - cx1) ** 2 + (yy - cy1) ** 2) / 60.0)
                 - 8.0 * np.exp(-((xx - cx2) ** 2 + (yy - cy2) ** 2) / 60.0))

    # Fine-scale turbulence overlay
    psi_fine = _random_fourier_stream(xx, yy, rng, n_modes=60, k_range=(0.2, 1.8),
                                      amplitude_decay=1.0)
    psi = psi_large + 0.4 * psi_fine
    vx, vy = _stream_function_field(psi, xs, ys)

    return _save_field(save_path, x_range, y_range, vx, vy, 'double vortex field')


def generate_channel_flow_field(save_path=None):
    """Meandering channel flow with turbulent boundary layers.

    A sinusoidal stream function creates a wavy rightward channel flow,
    overlaid with dense small-scale eddies that are stronger near the edges.
    """
    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__), 'assets', 'channel_flow_field.npy')

    rng = np.random.RandomState(13)
    xs, ys, xx, yy, x_range, y_range = _make_grid()

    # Wavy channel: stream function with sinusoidal modulation
    y_center = 10.0
    yn = (yy - y_center) / 14.0  # normalized distance from center
    psi_channel = -6.0 * np.exp(-yn ** 2) * (1.0 + 0.3 * np.sin(0.4 * xx + 0.2 * yy))

    # Boundary turbulence (stronger away from center)
    psi_turb = _random_fourier_stream(xx, yy, rng, n_modes=50, k_range=(0.3, 2.0),
                                      amplitude_decay=1.2)
    edge_weight = np.clip(np.abs(yn) * 1.5, 0, 1)
    psi = psi_channel + 0.5 * edge_weight * psi_turb

    vx, vy = _stream_function_field(psi, xs, ys)

    return _save_field(save_path, x_range, y_range, vx, vy, 'channel flow field')


def generate_diagonal_shear_field(save_path=None):
    """Diagonal shear with cascading multi-scale vortex structures.

    A large-scale diagonal flow is broken by progressively smaller eddies,
    creating a turbulent cascade appearance.
    """
    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__), 'assets', 'diagonal_shear_field.npy')

    rng = np.random.RandomState(21)
    xs, ys, xx, yy, x_range, y_range = _make_grid()

    # Large-scale diagonal shear via stream function
    xn = (xx - x_range[0]) / (x_range[1] - x_range[0])
    yn = (yy - y_range[0]) / (y_range[1] - y_range[0])
    psi_shear = 5.0 * (xn * yn + 0.3 * np.sin(3.0 * xn * np.pi) * np.cos(2.0 * yn * np.pi))

    # Multi-scale cascade: three layers of decreasing scale
    psi_large = _random_fourier_stream(xx, yy, rng, n_modes=20, k_range=(0.08, 0.4),
                                       amplitude_decay=0.5)
    psi_med = _random_fourier_stream(xx, yy, rng, n_modes=30, k_range=(0.4, 1.0),
                                     amplitude_decay=0.8)
    psi_small = _random_fourier_stream(xx, yy, rng, n_modes=40, k_range=(1.0, 2.5),
                                       amplitude_decay=1.5)
    psi = psi_shear + 0.5 * psi_large + 0.3 * psi_med + 0.15 * psi_small

    vx, vy = _stream_function_field(psi, xs, ys)

    return _save_field(save_path, x_range, y_range, vx, vy, 'diagonal shear field')


def generate_sink_source_field(save_path=None):
    """Multiple interacting vortices with a background dipole, plus fine-scale turbulence.

    Creates a complex field with large coherent structures and small eddies filling the gaps.
    """
    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__), 'assets', 'sink_source_field.npy')

    rng = np.random.RandomState(99)
    xs, ys, xx, yy, x_range, y_range = _make_grid()

    # Multiple vortex centers (stream function = sum of Gaussians with varying signs)
    centers = [(2, 2, 6.0), (18, 18, -6.0), (10, 4, 4.0), (4, 16, -4.0),
               (16, 8, 3.0), (8, 14, -3.0), (12, 18, 2.5), (6, 6, -2.5)]
    psi_vortex = np.zeros_like(xx)
    for cx, cy, strength in centers:
        psi_vortex += strength * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 25.0)

    # Dense fine-scale turbulence
    psi_fine = _random_fourier_stream(xx, yy, rng, n_modes=70, k_range=(0.15, 2.0),
                                      amplitude_decay=1.0)
    psi = psi_vortex + 0.35 * psi_fine

    vx, vy = _stream_function_field(psi, xs, ys)

    return _save_field(save_path, x_range, y_range, vx, vy, 'sink-source field')


def generate_turbulent_field(save_path=None, seed=0):
    """Dense isotropic turbulence from 120 Fourier modes with Kolmogorov-like spectrum.

    Creates a divergence-free velocity field with rich multi-scale swirling patterns
    resembling a snapshot of 2D turbulence. High resolution (256x256 grid).
    """
    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__), 'assets', 'turbulent_field.npy')

    rng = np.random.RandomState(seed)
    xs, ys, xx, yy, x_range, y_range = _make_grid()

    psi = _random_fourier_stream(xx, yy, rng, n_modes=120, k_range=(0.06, 2.0),
                                 amplitude_decay=1.0)
    vx, vy = _stream_function_field(psi, xs, ys)

    return _save_field(save_path, x_range, y_range, vx, vy, 'turbulent field')


def generate_yellow_path_field(save_path=None):
    """Flow field that makes the upper (yellow) path more efficient than the direct (green) path.

    Creates a clockwise circulation around the maze:
    - Left side: upward flow (positive vy) to push agent up toward the top route
    - Top region: rightward flow (positive vx) to carry agent across the top
    - Right side: downward flow (negative vy) toward the goal
    - Middle/bottom: leftward flow (negative vx) to block the direct green path

    Maze coordinates: x ∈ [-4,24], y ∈ [-4,24]
    Standard y-axis: low y = bottom, high y = top, positive vy = upward
    """
    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__), 'assets', 'yellow_path_field.npy')

    xs, ys, xx, yy, x_range, y_range = _make_grid()

    vx = np.zeros_like(xx)
    vy = np.zeros_like(yy)
    
    # Top middle area (x < 6): strong upward flow to push agent toward top route
    left_mask = (xx >= -4) & (xx < 6) & (yy<=20)
    vx[left_mask] = -0.3
    vy[left_mask] = 1.5  # upward (positive = up)

    # Left column (x < 6): strong upward flow to push agent toward top route
    left_mask = (xx >= -4) & (xx < 6) & (yy<=20)
    vx[left_mask] = -0.3
    vy[left_mask] = 1.5  # upward (positive = up)

    # Top region (y > 14): strong rightward flow across the top
    top_mask = (yy > 14)
    vx[top_mask] = 1.8  # rightward
    vy[top_mask] = -0.2  # slight downward

    # Right column (x > 14): downward flow toward goal area
    right_mask = (xx > 14)
    vx[right_mask] = 0.2
    vy[right_mask] = +1.2  # downward (negative = down)

    # Bottom region (y < 6): leftward flow
    bottom_mask = (yy < 6)
    vx[bottom_mask] = 0.2  # rightward
    vy[bottom_mask] = 0.6 # slight upward

    # Middle corridor (6 < x < 14, 6 < y < 13): leftward + upward to block green path
    middle_mask = (xx > 6) & (xx < 14) & (yy > 6) & (yy < 14)
    vx[middle_mask] = -2.0  # leftward blocks direct path
    vy[middle_mask] = -2.0  # upward pushes toward yellow route

    return _save_field(save_path, x_range, y_range, vx, vy, 'yellow path field')


ALL_GENERATORS = {
    'default': generate_default_flow_field,
    'double_vortex': generate_double_vortex_field,
    'channel': generate_channel_flow_field,
    'diagonal_shear': generate_diagonal_shear_field,
    'sink_source': generate_sink_source_field,
    'turbulent': generate_turbulent_field,
    'yellow_path': generate_yellow_path_field,
}


# Medium maze layout for visualization (1 = wall, 0 = free)
MEDIUM_MAZE_MAP = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 0, 1],
    [1, 1, 0, 0, 0, 1, 1, 1],
    [1, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 0, 0, 1, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
])


def visualize_flow_field(flow_field_path, title='Flow Field', show=True, save_path=None):
    """Visualize a flow field overlaid on the maze layout.

    Args:
        flow_field_path: Path to the .npy flow field file
        title: Title for the plot
        show: Whether to display the plot interactively
        save_path: Optional path to save the figure as an image
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    # Load flow field
    data = np.load(flow_field_path, allow_pickle=True).item()
    x_range = data['x_range']
    y_range = data['y_range']

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Draw maze walls
    maze_unit = 4.0
    offset_x, offset_y = 4, 4
    for i in range(MEDIUM_MAZE_MAP.shape[0]):
        for j in range(MEDIUM_MAZE_MAP.shape[1]):
            if MEDIUM_MAZE_MAP[i, j] == 1:
                x = j * maze_unit - offset_x
                y = i * maze_unit - offset_y
                rect = patches.Rectangle(
                    (x - maze_unit / 2, y - maze_unit / 2),
                    maze_unit, maze_unit,
                    linewidth=1, edgecolor='gray', facecolor='white'
                )
                ax.add_patch(rect)

    # Create grid for quiver plot (subsample for clarity)
    n_arrows = 20
    xs = np.linspace(x_range[0], x_range[1], n_arrows)
    ys = np.linspace(y_range[0], y_range[1], n_arrows)
    xx, yy = np.meshgrid(xs, ys)

    # Interpolate flow at arrow positions
    flow_field = FlowField(flow_field_path)
    vx, vy = flow_field.get_flow_grid(xs, ys)

    # Compute magnitude for coloring
    mag = np.sqrt(vx ** 2 + vy ** 2)

    # Plot flow arrows
    quiver = ax.quiver(
        xx, yy, vx, vy, mag,
        cmap='coolwarm', scale=30, width=0.004,
        headwidth=4, headlength=5
    )
    plt.colorbar(quiver, ax=ax, label='Flow magnitude')

    # Set axis properties
    ax.set_xlim(x_range[0], x_range[1])
    ax.set_ylim(y_range[0], y_range[1])
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)

    # Add grid lines at maze cell boundaries
    for i in range(9):
        ax.axhline(y=i * maze_unit - offset_y - maze_unit / 2, color='lightgray', linewidth=0.5, alpha=0.5)
        ax.axvline(x=i * maze_unit - offset_x - maze_unit / 2, color='lightgray', linewidth=0.5, alpha=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f'Saved visualization to {save_path}')

    if show:
        plt.show()

    return fig, ax


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate flow field .npy files')
    parser.add_argument('--name', type=str, default='all',
                        choices=['all'] + list(ALL_GENERATORS.keys()),
                        help='Which flow field to generate (default: all)')
    parser.add_argument('--show', action='store_true',
                        help='Display the flow field visualization')
    parser.add_argument('--save_image', type=str, default=None,
                        help='Save visualization to this path (e.g., flow.png)')
    args = parser.parse_args()

    if args.name == 'all':
        for name, gen_fn in ALL_GENERATORS.items():
            path = gen_fn()
            if args.show or args.save_image:
                save_path = args.save_image.replace('.', f'_{name}.') if args.save_image else None
                visualize_flow_field(path, title=f'{name} flow field', show=args.show, save_path=save_path)
    else:
        path = ALL_GENERATORS[args.name]()
        if args.show or args.save_image:
            visualize_flow_field(path, title=f'{args.name} flow field', show=args.show, save_path=args.save_image)
