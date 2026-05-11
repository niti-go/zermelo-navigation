import glob
import os
import re

import numpy as np
from scipy.interpolate import RegularGridInterpolator

try:
    import xarray as _xr
except ImportError:
    _xr = None

#this will be the default value if we don't provide an override in config
MAX_FLOW_MAGNITUDE = 1.8  # Agent max displacement is 0.2/step; flow displacement = 0.1 * mag.

# Process-wide cache for preloaded netcdf frames so multiple env instances in
# the same process share one copy of the (multi-GB) array.
_PRELOAD_CACHE = {}

"""
To generate a flow field:
python zermelo_env/zermelo_flow.py --name yellow_path
"""

class FlowField:
    """Static 2D fluid flow velocity field with bilinear interpolation."""

    def __init__(self, flow_field_path=None):
        if flow_field_path is None:
            flow_field_path = os.path.join(os.path.dirname(__file__), 'assets', 'yellow_path_field.npz')
        if flow_field_path.endswith('.npz'):
            data = dict(np.load(flow_field_path))
        else:
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


class DynamicTGVFlowField:
    """Time-dependent 2D Taylor-Green vortex flow field.

    The base velocity field follows the analytical TGV solution to the
    incompressible Navier-Stokes equations:

        u =  U0 * sin(kx * x) * cos(ky * y) * exp(-nu*(kx^2+ky^2)*t)
        v = -U0 * cos(kx * x) * sin(ky * y) * exp(-nu*(kx^2+ky^2)*t)

    On top of this, two optional extensions are available (set to 0 to disable):

      1. **Oscillation** (periodic reversal):  cos(omega * t)
         Controlled by ``omega``.  Not part of the NS solution but useful
         for creating periodically reversing currents.

      2. **Translation** (drifting vortex lattice):
         The vortex pattern slides at constant velocity (Ux, Uy).
         Set both to 0 to keep the pattern stationary.

    Full velocity field:

        xc = (x - cx) - Ux * t          # translated coordinate
        yc = (y - cy) - Uy * t
        temporal = cos(omega * t) * exp(-nu*(kx^2+ky^2) * t)

        vx =  A * sin(kx * xc) * cos(ky * yc) * temporal
        vy = -A * cos(kx * xc) * sin(ky * yc) * temporal

    Remains divergence-free for all t (the spatial structure of the TGV is
    preserved; translation shifts it and the temporal factor is spatially
    uniform).
    """

    def __init__(self, dynamic_cfg):
        cfg = dynamic_cfg
        self.amplitude = cfg.get('amplitude', 1.8)
        self.nu = cfg.get('nu', 0.01)
        self.omega = cfg.get('omega', 0.0)
        self.Ux = cfg.get('Ux', 0.0)
        self.Uy = cfg.get('Uy', 0.0)

        # Domain — match the static grid defaults.
        self.x_range = np.array(cfg.get('x_range', [-4.0, 24.0]), dtype=np.float64)
        self.y_range = np.array(cfg.get('y_range', [-4.0, 24.0]), dtype=np.float64)

        Lx = self.x_range[1] - self.x_range[0]
        Ly = self.y_range[1] - self.y_range[0]

        n_vortices = cfg.get('n_vortices', 2)  # full wavelengths per dimension
        self.kx = 2.0 * np.pi * n_vortices / Lx
        self.ky = 2.0 * np.pi * n_vortices / Ly

        # Center of the pattern.
        self._cx = (self.x_range[0] + self.x_range[1]) / 2.0
        self._cy = (self.y_range[0] + self.y_range[1]) / 2.0

        # Viscous decay rate: nu * (kx^2 + ky^2) from the Navier-Stokes
        # analytical solution.  With kx = ky = 1 this gives 2*nu, matching
        # the classical TGV decay exp(-2*nu*t).
        self._decay = self.nu * (self.kx ** 2 + self.ky ** 2)

    def _temporal(self, t):
        """Combined temporal modulation factor."""
        return np.cos(self.omega * t) * np.exp(-self._decay * t)

    def get_flow(self, x, y, t=0.0):
        """Return (vx, vy) at position (x, y) and time t."""
        xc = (x - self._cx) - self.Ux * t
        yc = (y - self._cy) - self.Uy * t
        mod = self._temporal(t)
        vx = self.amplitude * np.sin(self.kx * xc) * np.cos(self.ky * yc) * mod
        vy = -self.amplitude * np.cos(self.kx * xc) * np.sin(self.ky * yc) * mod
        return float(vx), float(vy)

    def get_flow_grid(self, xs, ys, t=0.0):
        """Return flow on a meshgrid for visualization at time t."""
        yy, xx = np.meshgrid(ys, xs, indexing='ij')
        xc = (xx - self._cx) - self.Ux * t
        yc = (yy - self._cy) - self.Uy * t
        mod = self._temporal(t)
        vx = self.amplitude * np.sin(self.kx * xc) * np.cos(self.ky * yc) * mod
        vy = -self.amplitude * np.cos(self.kx * xc) * np.sin(self.ky * yc) * mod
        return vx, vy


def _resolve_nc_paths(nc_dir=None, nc_paths=None, nc_path=None):
    """Resolve the netcdf config fields to an ordered list of absolute paths.

    Exactly one of ``nc_dir`` / ``nc_paths`` / ``nc_path`` should be set.
    When ``nc_dir`` is used, files are sorted by the integer found in the
    filename (HIT1, HIT2, ..., HIT9, HIT10 — not lexicographic).
    """
    if nc_dir is not None:
        if not os.path.isabs(nc_dir):
            nc_dir = os.path.abspath(nc_dir)
        if not os.path.isdir(nc_dir):
            raise ValueError(f'dynamic.netcdf.nc_dir does not exist: {nc_dir}')
        files = glob.glob(os.path.join(nc_dir, '*.nc'))
        if not files:
            raise ValueError(f'No .nc files found in {nc_dir}')

        def _num_key(p):
            m = re.search(r'(\d+)', os.path.basename(p))
            return (int(m.group(1)) if m else 0, os.path.basename(p))

        return sorted(files, key=_num_key)

    if nc_paths is not None:
        if isinstance(nc_paths, str):
            nc_paths = [nc_paths]
        return [p if os.path.isabs(p) else os.path.abspath(p) for p in nc_paths]

    if nc_path is not None:
        return [nc_path if os.path.isabs(nc_path) else os.path.abspath(nc_path)]

    raise ValueError(
        'dynamic.netcdf must set one of nc_dir, nc_paths, or nc_path '
        'when using netcdf flow mode.'
    )


class DynamicNetCDFFlowField:
    """Time-varying 2D flow field loaded from one or more NetCDF (.nc) files.

    Expects each dataset to have variables ``u(time, x, y)`` and
    ``v(time, x, y)`` and 1-D coords ``time``, ``x``, ``y`` (e.g. the
    Diff-FlowFSI HIT datasets: 1000 snapshots per file on a 512x512 periodic
    box with x,y in [0, 2pi]).

    Multiple files are concatenated along time into one long virtual clip;
    all files must share the same spatial grid.  You can point to:
      - ``nc_dir``: a directory — every ``*.nc`` inside is sorted by the
        integer suffix in the filename (e.g. ``HIT1.nc``, ``HIT2.nc``, …) and
        chained in that order;
      - ``nc_paths``: an explicit ordered list of files;
      - ``nc_path``: a single file (back-compat shorthand).

    The simulation's native domain is remapped to the Zermelo arena
    ``(x_range, y_range)`` via an affine rescale, with periodic wrap by
    default (HIT data is periodic).

    Temporal playback is controlled by ``frames_per_step`` — how many HIT
    snapshots the flow advances per env step.  If ``frames_per_step`` is not
    given but ``frames_per_episode`` + ``max_episode_steps`` are, they're
    used to derive it (so one playthrough = frames_per_episode frames over
    max_episode_steps steps).  Time wraps back to frame 0 after the last
    frame of the last file.

    The velocities are also rescaled to ``target_max`` (default 1.8) so they
    match the magnitude regime the agent expects (agent step = 0.2, flow
    displacement per step = dt * flow).
    """

    def __init__(self, cfg):
        if _xr is None:
            raise ImportError(
                'xarray is required for NetCDF flow fields. Install with: '
                'pip install xarray netCDF4'
            )

        nc_paths = _resolve_nc_paths(
            nc_dir=cfg.get('nc_dir'),
            nc_paths=cfg.get('nc_paths'),
            nc_path=cfg.get('nc_path'),
        )

        # Lazy-open every file; verify they share the same spatial grid.
        self._datasets = [_xr.open_dataset(p) for p in nc_paths]
        for p, ds in zip(nc_paths, self._datasets):
            if 'u' not in ds or 'v' not in ds:
                raise ValueError(f"NetCDF file {p} must contain 'u' and 'v' variables.")

        ref_x = np.asarray(self._datasets[0]['x'].values, dtype=np.float64)
        ref_y = np.asarray(self._datasets[0]['y'].values, dtype=np.float64)
        for p, ds in zip(nc_paths[1:], self._datasets[1:]):
            x = np.asarray(ds['x'].values, dtype=np.float64)
            y = np.asarray(ds['y'].values, dtype=np.float64)
            if x.shape != ref_x.shape or y.shape != ref_y.shape \
                    or not np.allclose(x, ref_x) or not np.allclose(y, ref_y):
                raise ValueError(
                    f'NetCDF file {p} has a different spatial grid than {nc_paths[0]}; '
                    'all files must share the same (x, y) grid.'
                )

        # Per-file frame counts and cumulative offsets for frame-index -> file routing.
        self._frames_per_file = np.array(
            [ds.sizes['time'] for ds in self._datasets], dtype=np.int64
        )
        self._file_offsets = np.concatenate(([0], np.cumsum(self._frames_per_file)))
        self._n_frames_total = int(self._file_offsets[-1])

        # Native coordinates (time is the concatenation of each file's time axis).
        self._t_native = np.concatenate(
            [np.asarray(ds['time'].values, dtype=np.float64) for ds in self._datasets]
        )
        self._x_native = ref_x
        self._y_native = ref_y

        self._t0 = float(self._t_native[0])
        self._t_span = float(self._t_native[-1] - self._t_native[0])
        self._nx = len(self._x_native)
        self._ny = len(self._y_native)

        # Arena domain the agent lives in.
        self.x_range = np.array(cfg.get('x_range', [-4.0, 24.0]), dtype=np.float64)
        self.y_range = np.array(cfg.get('y_range', [-4.0, 24.0]), dtype=np.float64)
        self._Lx_arena = self.x_range[1] - self.x_range[0]
        self._Ly_arena = self.y_range[1] - self.y_range[0]

        # Native box size (assume uniform; wrap periodically).
        self._x_native_min = float(self._x_native[0])
        self._y_native_min = float(self._y_native[0])
        self._x_native_max = float(self._x_native[-1])
        self._y_native_max = float(self._y_native[-1])
        # Extend by one cell so wrap point matches a periodic grid.
        self._dx_native = (self._x_native_max - self._x_native_min) / max(self._nx - 1, 1)
        self._dy_native = (self._y_native_max - self._y_native_min) / max(self._ny - 1, 1)
        self._Lx_native = (self._x_native_max - self._x_native_min) + self._dx_native
        self._Ly_native = (self._y_native_max - self._y_native_min) + self._dy_native

        # Spatial tiling: how many copies of the native periodic box fit across
        # the arena.  1 = one big copy (largest eddies).  N>1 = NxN tiling
        # (eddies look N times smaller).  Field stays seamless because the
        # native data is periodic.
        self._n_tiles = float(cfg.get('n_tiles', 1.0))

        # Temporal playback: frames of HIT data advanced per env step.
        # Precedence: frames_per_step > (frames_per_episode / max_episode_steps)
        #           > time_scale (legacy, continuous-time mapping)
        # Time always loops — at the end of the 800-frame clip we wrap back to
        # the start so episodes longer than the dataset don't freeze.

        if cfg.get('frames_per_step') is not None:
            self._frames_per_step = float(cfg['frames_per_step'])
            self._time_mode = 'frames_per_step'
        elif cfg.get('frames_per_episode') is not None and cfg.get('max_episode_steps') is not None:
            self._frames_per_step = (
                float(cfg['frames_per_episode']) / float(cfg['max_episode_steps'])
            )
            self._time_mode = 'frames_per_episode'
        else:
            self.time_scale = float(cfg.get('time_scale', 1.0))
            self._frames_per_step = None
            self._time_mode = 'time_scale'

        # Env step duration, used to convert the API's sim-time (seconds)
        # into step count.  The maze wrapper runs at frame_skip=5,
        # mj_timestep=0.02 => 0.1s/step.
        self._env_dt = float(cfg.get('env_dt', 0.1))

        # Velocity normalization so magnitudes match the static-field regime.
        # Uses the first frame of the first file as a proxy for the global max
        # (HIT is statistically stationary, so frame 0 is representative).
        # Individual frames may occasionally exceed target_max after scaling.
        target_max = float(cfg.get('target_max', MAX_FLOW_MAGNITUDE))
        u0 = self._datasets[0]['u'].isel(time=0).values
        v0 = self._datasets[0]['v'].isel(time=0).values
        native_max = max(float(np.max(np.abs(u0))), float(np.max(np.abs(v0))), 1e-8)
        self._vel_scale = target_max / native_max

        # Cache of loaded time slices, keyed by global frame index.
        self._slice_cache = {}
        self._cache_max = int(cfg.get('slice_cache_size', 4))

        # Optional bulk preload: read every (u, v) frame into RAM once so
        # _get_slice becomes a free array index. Costs ~2 * n_frames * nx * ny *
        # 4 bytes (float32). For the 9-file HIT dataset that's ~18 GB.
        self._preloaded_u = None
        self._preloaded_v = None
        if bool(cfg.get('preload_all_frames', False)):
            self._preload_all_frames()

    def _preload_all_frames(self):
        """Load every u/v frame across all files into one contiguous float32
        array. After this, _get_slice indexes RAM instead of reading disk.

        Cached process-wide so multiple env instances share one copy.
        """
        total = self._n_frames_total
        nx, ny = self._nx, self._ny
        cache_key = (total, nx, ny, tuple(self._frames_per_file.tolist()))
        cached = _PRELOAD_CACHE.get(cache_key)
        if cached is not None:
            self._preloaded_u, self._preloaded_v = cached
            print('  [DynamicNetCDFFlowField] preload reused from cache.')
            return
        bytes_total = 2 * total * nx * ny * 4
        print(
            f'  [DynamicNetCDFFlowField] preloading {total} frames '
            f'({nx}x{ny}, ~{bytes_total / 1e9:.1f} GB) into RAM...'
        )
        u_full = np.empty((total, nx, ny), dtype=np.float32)
        v_full = np.empty((total, nx, ny), dtype=np.float32)
        for file_i, ds in enumerate(self._datasets):
            start = int(self._file_offsets[file_i])
            end = int(self._file_offsets[file_i + 1])
            u_full[start:end] = ds['u'].values.astype(np.float32, copy=False)
            v_full[start:end] = ds['v'].values.astype(np.float32, copy=False)
        self._preloaded_u = u_full
        self._preloaded_v = v_full
        _PRELOAD_CACHE[cache_key] = (u_full, v_full)
        print('  [DynamicNetCDFFlowField] preload complete.')

    def _native_time(self, t):
        """Convert arena time (seconds) to a continuous native frame index in [0, n_frames).

        When the user specifies ``frames_per_step`` (or a frames-per-episode
        budget), we first convert seconds -> env-step count (t / env_dt),
        then multiply by frames_per_step.  Otherwise we fall back to the
        legacy continuous-time rescale (``time_scale`` native seconds per
        sim second).

        The index wraps modulo the total frame count so episodes longer than
        the concatenated clip keep replaying it seamlessly.
        """
        n_frames = self._n_frames_total
        if n_frames <= 1:
            return 0.0

        if self._frames_per_step is not None:
            step_count = t / max(self._env_dt, 1e-12)
            frac_idx = step_count * self._frames_per_step
        else:
            native_t = t * self.time_scale
            frac_idx = native_t / (self._t_span / (n_frames - 1))

        frac_idx = frac_idx % n_frames
        return frac_idx

    def _arena_to_native_xy(self, x, y):
        """Map arena (x,y) -> native (x,y) with periodic wrap.

        ``n_tiles`` controls how many copies of the native periodic box tile
        across the arena.  n_tiles=1 stretches one copy across the arena
        (biggest eddies); n_tiles=4 packs 4x4 copies into the arena (eddies
        appear 4x smaller).
        """
        fx = (np.asarray(x) - self.x_range[0]) / self._Lx_arena * self._n_tiles
        fy = (np.asarray(y) - self.y_range[0]) / self._Ly_arena * self._n_tiles
        # Wrap periodically so out-of-arena queries still return a valid flow.
        fx = fx - np.floor(fx)
        fy = fy - np.floor(fy)
        nx = self._x_native_min + fx * self._Lx_native
        ny = self._y_native_min + fy * self._Ly_native
        return nx, ny

    def _get_slice(self, idx):
        """Load u/v arrays for global frame idx (periodic in T), cached."""
        idx = int(idx) % self._n_frames_total
        if self._preloaded_u is not None:
            return self._preloaded_u[idx], self._preloaded_v[idx]
        if idx in self._slice_cache:
            return self._slice_cache[idx]
        # Route the global frame index to the right file + local index.
        file_i = int(np.searchsorted(self._file_offsets, idx, side='right') - 1)
        local_idx = idx - int(self._file_offsets[file_i])
        ds = self._datasets[file_i]
        u = ds['u'].isel(time=local_idx).values.astype(np.float32)
        v = ds['v'].isel(time=local_idx).values.astype(np.float32)
        if len(self._slice_cache) >= self._cache_max:
            self._slice_cache.pop(next(iter(self._slice_cache)))
        self._slice_cache[idx] = (u, v)
        return u, v

    def _interp_frame(self, u_frame, v_frame, nx_pts, ny_pts):
        """Bilinear interpolation of one (u,v) snapshot at native coords."""
        # Fractional grid index along each axis (native grid is uniform).
        ix = (nx_pts - self._x_native_min) / self._dx_native
        iy = (ny_pts - self._y_native_min) / self._dy_native
        # Periodic wrap on the integer indices.
        i0 = np.floor(ix).astype(np.int64) % self._nx
        j0 = np.floor(iy).astype(np.int64) % self._ny
        i1 = (i0 + 1) % self._nx
        j1 = (j0 + 1) % self._ny
        fx = ix - np.floor(ix)
        fy = iy - np.floor(iy)
        # Bilinear.
        def _bi(arr):
            a00 = arr[i0, j0]
            a10 = arr[i1, j0]
            a01 = arr[i0, j1]
            a11 = arr[i1, j1]
            return ((1 - fx) * (1 - fy) * a00 + fx * (1 - fy) * a10
                    + (1 - fx) * fy * a01 + fx * fy * a11)
        return _bi(u_frame), _bi(v_frame)

    def get_flow(self, x, y, t=0.0):
        """Return (vx, vy) at arena position (x, y) and arena time t."""
        nx_pt, ny_pt = self._arena_to_native_xy(x, y)
        frac_idx = self._native_time(t)
        i0 = int(np.floor(frac_idx))
        i1 = i0 + 1
        alpha = frac_idx - i0

        u0, v0 = self._get_slice(i0)
        u_a, v_a = self._interp_frame(u0, v0, np.atleast_1d(nx_pt), np.atleast_1d(ny_pt))
        if alpha > 1e-9:
            u1, v1 = self._get_slice(i1)
            u_b, v_b = self._interp_frame(u1, v1, np.atleast_1d(nx_pt), np.atleast_1d(ny_pt))
            vx_n = (1 - alpha) * u_a + alpha * u_b
            vy_n = (1 - alpha) * v_a + alpha * v_b
        else:
            vx_n = u_a
            vy_n = v_a
        vx = float(vx_n.ravel()[0]) * self._vel_scale
        vy = float(vy_n.ravel()[0]) * self._vel_scale
        return vx, vy

    def get_flow_grid(self, xs, ys, t=0.0):
        """Return (vx, vy) on a meshgrid at arena time t (ij indexing)."""
        xs = np.asarray(xs)
        ys = np.asarray(ys)
        yy, xx = np.meshgrid(ys, xs, indexing='ij')
        nx_pts, ny_pts = self._arena_to_native_xy(xx, yy)

        frac_idx = self._native_time(t)
        i0 = int(np.floor(frac_idx))
        i1 = i0 + 1
        alpha = frac_idx - i0

        u0, v0 = self._get_slice(i0)
        u_a, v_a = self._interp_frame(u0, v0, nx_pts, ny_pts)
        if alpha > 1e-9:
            u1, v1 = self._get_slice(i1)
            u_b, v_b = self._interp_frame(u1, v1, nx_pts, ny_pts)
            vx = (1 - alpha) * u_a + alpha * u_b
            vy = (1 - alpha) * v_a + alpha * v_b
        else:
            vx = u_a
            vy = v_a
        return vx * self._vel_scale, vy * self._vel_scale



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
    
    # Top middle area (x < 6): upward flow to push agent toward top route
    left_mask = (xx >= -4) & (xx < 6) & (yy<=20)
    vx[left_mask] = -0.15
    vy[left_mask] = 0.75  # upward (positive = up)

    # Top region (y > 14): rightward flow across the top
    top_mask = (yy > 14)
    vx[top_mask] = 0.9  # rightward
    vy[top_mask] = -0.1  # slight downward

    # Right column (x > 14): downward flow toward goal area
    right_mask = (xx > 14)
    vx[right_mask] = 0.1
    vy[right_mask] = 0.6

    # Bottom region (y < 6): rightward flow
    bottom_mask = (yy < 6)
    vx[bottom_mask] = 0.1
    vy[bottom_mask] = 0.3

    # Middle corridor (6 < x < 14, 6 < y < 13): leftward + downward to block green path
    middle_mask = (xx > 6) & (xx < 14) & (yy > 6) & (yy < 14)
    vx[middle_mask] = -0.5
    vy[middle_mask] = -0.5

    return _save_field(save_path, x_range, y_range, vx, vy, 'yellow path field')


def generate_taylor_green_field(save_path=None):
    """2D Taylor-Green vortex flow field.

    The Taylor-Green vortex is an exact solution to the incompressible
    Navier-Stokes equations. It produces a periodic array of counter-rotating
    vortices that create an interesting navigation challenge: the agent can
    ride favorable vortex currents or must fight against opposing ones.

        vx =  A * sin(kx * x) * cos(ky * y)
        vy = -A * cos(kx * x) * sin(ky * y)

    Divergence-free by construction (∂vx/∂x + ∂vy/∂y = 0).

    With kx = ky = 2π/L, this produces a 2×2 grid of vortices across
    the maze domain, giving the agent multiple route choices through
    alternating favorable and adverse currents.
    """
    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__), 'assets', 'taylor_green_field.npy')

    xs, ys, xx, yy, x_range, y_range = _make_grid()

    # Domain length.
    Lx = x_range[1] - x_range[0]  # 28.0
    Ly = y_range[1] - y_range[0]  # 28.0

    # Wavenumbers: 2 full vortex wavelengths across each dimension.
    kx = 2.0 * np.pi * 2.0 / Lx
    ky = 2.0 * np.pi * 2.0 / Ly

    # Center the pattern on the domain.
    xc = xx - (x_range[0] + x_range[1]) / 2.0
    yc = yy - (y_range[0] + y_range[1]) / 2.0

    vx = np.sin(kx * xc) * np.cos(ky * yc)
    vy = -np.cos(kx * xc) * np.sin(ky * yc)

    return _save_field(save_path, x_range, y_range, vx, vy, 'Taylor-Green vortex field')


ALL_GENERATORS = {
    'default': generate_default_flow_field,
    'double_vortex': generate_double_vortex_field,
    'channel': generate_channel_flow_field,
    'diagonal_shear': generate_diagonal_shear_field,
    'sink_source': generate_sink_source_field,
    'turbulent': generate_turbulent_field,
    'yellow_path': generate_yellow_path_field,
    'taylor_green': generate_taylor_green_field,
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
