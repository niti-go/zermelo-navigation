"""Chained HIT flow field — single source of truth for fluid flow.

Loads HIT{1..max_file}.nc from a directory and exposes them as one continuous
clip. Spatial domain is mapped onto the arena (x_range, y_range) with
periodic wrap. Time is parameterized by a continuous `frame` index in
[0, n_frames_total); fractional values are linearly interpolated between the
two nearest snapshots.

Memory model
------------
At runtime we never read NetCDF. Every source file is transcoded once into a
local-SSD float32 binary of shape ``(time, x, y, 2)`` (u and v interleaved on
the innermost axis); see ``zermelo_env.hit_cache``. Each ``HITChainFlow``
instance ``np.memmap``s every cache file it covers. mmap of a shared file is
the inter-process sharing layer: 16 worker processes hitting the same cache
files map to a single set of physical pages in the OS page cache.

Random ``arr[i, j]`` lookups against the memmap fault one 4 KB page on first
access. ``prewarm_range(frame_lo, frame_hi)`` reads a contiguous slab so all
the pages a worker will touch are resident before its episodes start.

This module is policy-agnostic: it knows nothing about train/test splits or
which frames are "safe" to query. Callers (e.g. dataset generators) are
responsible for choosing start frames that keep their queries within the
train segment.
"""

import glob
import os
import re

import numpy as np

from zermelo_env import hit_cache


def _list_hit_files(nc_dir, max_file=None):
    """Return HIT*.nc files in nc_dir sorted by integer suffix.

    If max_file is given, only files HIT1..HIT{max_file} (inclusive) are
    returned; this is how callers reserve the tail of the chain for testing.
    """
    if not os.path.isabs(nc_dir):
        nc_dir = os.path.abspath(nc_dir)
    if not os.path.isdir(nc_dir):
        raise ValueError(f'HIT flow dir does not exist: {nc_dir}')
    files = glob.glob(os.path.join(nc_dir, '*.nc'))
    if not files:
        raise ValueError(f'No .nc files found in {nc_dir}')

    def _key(p):
        m = re.search(r'(\d+)', os.path.basename(p))
        return int(m.group(1)) if m else 0

    files = sorted(files, key=_key)
    if max_file is not None:
        files = [f for f in files if _key(f) <= max_file]
        if not files:
            raise ValueError(f'No HIT files with index <= {max_file} in {nc_dir}')
    return files


class HITChainFlow:
    """Continuous time-varying 2-D flow built from chained HIT NetCDF files.

    Args:
        nc_dir: directory containing ``HIT*.nc`` files. Files are ordered by
            the integer suffix in the filename (HIT1, HIT2, ...).
        max_file: include only ``HIT1..HIT{max_file}``. ``None`` = all files.
        x_range, y_range: arena spatial extent. Native HIT box is mapped
            onto this rectangle (with periodic wrap).
        n_tiles: number of native-box copies tiling the arena
            (1 = single copy stretched, 2 = 2x2 tiling, ...).
        target_max: if set, rescale velocities so the peak in frame 0 has
            magnitude ``target_max``. Use ``None`` to keep raw native units.
        cache_dir: optional override for the local-SSD cache directory.
            Default is ``$ZERMELO_HIT_CACHE_DIR`` or
            ``/tmp/zermelo_hit_cache_<uid>``. The cache must be populated
            first via ``scripts/build_hit_cache.py``; missing files raise
            ``FileNotFoundError`` immediately.
    """

    def __init__(self, nc_dir, max_file=None, x_range=(-4.0, 24.0),
                 y_range=(-4.0, 24.0), n_tiles=1.0, target_max=None,
                 cache_dir=None):
        nc_paths = _list_hit_files(nc_dir, max_file=max_file)
        self._nc_paths = nc_paths

        if cache_dir is None:
            cache_dir = hit_cache.default_cache_dir()
        self._cache_dir = cache_dir

        # One small read at startup; never touch NFS again.
        ref_x, ref_y, file_shape = hit_cache.load_coords(cache_dir)

        self._memmaps = [
            hit_cache.open_memmap(p, cache_dir, file_shape) for p in nc_paths
        ]

        # All cache files share the same shape (validated at build time);
        # n_frames is just len(files) * frames_per_file.
        frames_per_file = int(file_shape[0])
        self._frames_per_file = np.full(len(nc_paths), frames_per_file,
                                        dtype=np.int64)
        self._file_offsets = np.concatenate(([0], np.cumsum(self._frames_per_file)))
        self.n_frames = int(self._file_offsets[-1])

        self._x_native = ref_x
        self._y_native = ref_y
        self._nx = len(ref_x)
        self._ny = len(ref_y)

        self.x_range = np.array(x_range, dtype=np.float64)
        self.y_range = np.array(y_range, dtype=np.float64)
        self._Lx_arena = self.x_range[1] - self.x_range[0]
        self._Ly_arena = self.y_range[1] - self.y_range[0]

        self._x_native_min = float(ref_x[0])
        self._y_native_min = float(ref_y[0])
        self._dx_native = (float(ref_x[-1]) - float(ref_x[0])) / max(self._nx - 1, 1)
        self._dy_native = (float(ref_y[-1]) - float(ref_y[0])) / max(self._ny - 1, 1)
        # +1 cell so the wrap point matches a periodic grid.
        self._Lx_native = (float(ref_x[-1]) - float(ref_x[0])) + self._dx_native
        self._Ly_native = (float(ref_y[-1]) - float(ref_y[0])) + self._dy_native

        self._n_tiles = float(n_tiles)

        # Velocity rescale (peak of frame 0 -> target_max).
        if target_max is not None:
            slab0 = self._memmaps[0][0]  # (nx, ny, 2)
            native_max = max(float(np.max(np.abs(slab0[..., 0]))),
                             float(np.max(np.abs(slab0[..., 1]))), 1e-8)
            self._vel_scale = float(target_max) / native_max
        else:
            self._vel_scale = 1.0

    @property
    def cache_dir(self):
        return self._cache_dir

    def prewarm_range(self, frame_lo, frame_hi, verbose=False):
        """Force pages for frames ``[frame_lo, frame_hi)`` resident in RAM.

        Walks each covered cache file and triggers a sequential read of the
        slab the worker will actually touch. On warm OS page cache this is a
        no-op; on cold local SSD it streams at device speed (~5–10 s for
        ~12 GB). Workers with overlapping ranges share pages via the OS
        cache, so the host pulls each page from disk at most once.

        Out-of-range bounds are clamped; ``frame_hi <= frame_lo`` is a no-op.
        """
        lo = max(0, int(np.floor(frame_lo)))
        hi = min(self.n_frames, int(np.ceil(frame_hi)))
        if hi <= lo:
            return
        # Map the global frame range to per-file local ranges.
        for fi in range(len(self._memmaps)):
            f_start = int(self._file_offsets[fi])
            f_end = int(self._file_offsets[fi + 1])
            local_lo = max(0, lo - f_start)
            local_hi = min(f_end - f_start, hi - f_start)
            if local_lo >= local_hi:
                continue
            if verbose:
                print(f'  [hit_chain] prewarm file {fi + 1}/{len(self._memmaps)} '
                      f'frames [{local_lo}, {local_hi})', flush=True)
            # ``.sum()`` forces every page in the slab to be faulted in;
            # the result is discarded.
            self._memmaps[fi][local_lo:local_hi].sum()

    def prewarm(self, verbose=False):
        """Force every frame in this chain resident in RAM (whole-dataset).

        Prefer ``prewarm_range`` when the worker's frame range is known —
        it touches a fraction of the data. ``prewarm()`` exists for callers
        with no a-priori bound on which frames they will query.
        """
        self.prewarm_range(0, self.n_frames, verbose=verbose)

    def _get_slice(self, frame_idx):
        """Return (u, v) snapshot views for an integer frame index."""
        idx = int(frame_idx) % self.n_frames
        file_i = int(np.searchsorted(self._file_offsets, idx, side='right') - 1)
        local_idx = idx - int(self._file_offsets[file_i])
        slab = self._memmaps[file_i][local_idx]  # (nx, ny, 2) view
        return slab[..., 0], slab[..., 1]

    def _arena_to_native_xy(self, x, y):
        fx = (np.asarray(x) - self.x_range[0]) / self._Lx_arena * self._n_tiles
        fy = (np.asarray(y) - self.y_range[0]) / self._Ly_arena * self._n_tiles
        fx = fx - np.floor(fx)
        fy = fy - np.floor(fy)
        nx = self._x_native_min + fx * self._Lx_native
        ny = self._y_native_min + fy * self._Ly_native
        return nx, ny

    def _interp_frame(self, u_frame, v_frame, nx_pts, ny_pts):
        ix = (nx_pts - self._x_native_min) / self._dx_native
        iy = (ny_pts - self._y_native_min) / self._dy_native
        i0 = np.floor(ix).astype(np.int64) % self._nx
        j0 = np.floor(iy).astype(np.int64) % self._ny
        i1 = (i0 + 1) % self._nx
        j1 = (j0 + 1) % self._ny
        fx = ix - np.floor(ix)
        fy = iy - np.floor(iy)

        def _bi(arr):
            a00 = arr[i0, j0]
            a10 = arr[i1, j0]
            a01 = arr[i0, j1]
            a11 = arr[i1, j1]
            return ((1 - fx) * (1 - fy) * a00 + fx * (1 - fy) * a10
                    + (1 - fx) * fy * a01 + fx * fy * a11)
        return _bi(u_frame), _bi(v_frame)

    def get_flow(self, x, y, frame=0.0):
        """Return (vx, vy) at arena (x, y) and continuous frame index `frame`.

        `frame` may be fractional; values are linearly interpolated between
        adjacent snapshots. Out-of-range frames wrap modulo n_frames.
        """
        nx_pt, ny_pt = self._arena_to_native_xy(x, y)
        f = float(frame) % self.n_frames
        i0 = int(np.floor(f))
        alpha = f - i0
        i1 = (i0 + 1) % self.n_frames

        u0, v0 = self._get_slice(i0)
        u_a, v_a = self._interp_frame(u0, v0, np.atleast_1d(nx_pt), np.atleast_1d(ny_pt))
        if alpha > 1e-9:
            u1, v1 = self._get_slice(i1)
            u_b, v_b = self._interp_frame(u1, v1, np.atleast_1d(nx_pt), np.atleast_1d(ny_pt))
            vx_n = (1 - alpha) * u_a + alpha * u_b
            vy_n = (1 - alpha) * v_a + alpha * v_b
        else:
            vx_n, vy_n = u_a, v_a
        return float(vx_n.ravel()[0]) * self._vel_scale, float(vy_n.ravel()[0]) * self._vel_scale

    def get_flow_grid(self, xs, ys, frame=0.0):
        """Return (vx, vy) on the meshgrid of (xs, ys) at frame `frame`."""
        xs = np.asarray(xs)
        ys = np.asarray(ys)
        yy, xx = np.meshgrid(ys, xs, indexing='ij')
        nx_pts, ny_pts = self._arena_to_native_xy(xx, yy)

        f = float(frame) % self.n_frames
        i0 = int(np.floor(f))
        alpha = f - i0
        i1 = (i0 + 1) % self.n_frames

        u0, v0 = self._get_slice(i0)
        u_a, v_a = self._interp_frame(u0, v0, nx_pts, ny_pts)
        if alpha > 1e-9:
            u1, v1 = self._get_slice(i1)
            u_b, v_b = self._interp_frame(u1, v1, nx_pts, ny_pts)
            vx = (1 - alpha) * u_a + alpha * u_b
            vy = (1 - alpha) * v_a + alpha * v_b
        else:
            vx, vy = u_a, v_a
        return vx * self._vel_scale, vy * self._vel_scale
