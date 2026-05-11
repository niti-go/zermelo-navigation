"""Local-SSD cache for HIT u/v fields.

Layout (flat, no subdirs):

    $cache_dir/
        coords.npz    # x, y, shape_per_file — read once at startup
        HIT1.bin      # raw float32 (time, x, y, 2)
        HIT2.bin
        ...

The default cache dir is ``$ZERMELO_HIT_CACHE_DIR`` or
``/tmp/zermelo_hit_cache_<uid>``. Workers mmap the .bin files directly;
no NFS access at runtime. Build with ``scripts/build_hit_cache.py``.

Missing cache → loud FileNotFoundError. There is no lazy build path.
"""

import os

import numpy as np


DTYPE = np.float32

_MISSING_HINT = (
    'Run scripts/build_hit_cache.py to populate the cache, '
    'or set $ZERMELO_HIT_CACHE_DIR if it lives elsewhere.'
)


def default_cache_dir():
    env = os.environ.get('ZERMELO_HIT_CACHE_DIR')
    if env:
        return env
    return f'/tmp/zermelo_hit_cache_{os.getuid()}'


def bin_path(cache_dir, src_or_basename):
    base = os.path.basename(src_or_basename)
    if base.endswith('.nc'):
        base = base[:-3]
    return os.path.join(cache_dir, base + '.bin')


def coords_path(cache_dir):
    return os.path.join(cache_dir, 'coords.npz')


def load_coords(cache_dir):
    """Return ``(x, y, shape_per_file)``. Loud error if missing."""
    p = coords_path(cache_dir)
    if not os.path.exists(p):
        raise FileNotFoundError(
            f'Missing HIT cache coords: {p}\n{_MISSING_HINT}'
        )
    z = np.load(p)
    return z['x'], z['y'], tuple(int(s) for s in z['shape'])


def open_memmap(src_or_basename, cache_dir, shape):
    """mmap one cache file. Loud error if missing."""
    p = bin_path(cache_dir, src_or_basename)
    if not os.path.exists(p):
        raise FileNotFoundError(
            f'Missing HIT cache file: {p}\n{_MISSING_HINT}'
        )
    return np.memmap(p, dtype=DTYPE, mode='r', shape=shape)


def write_coords(cache_dir, x, y, shape):
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(coords_path(cache_dir),
             x=np.asarray(x, np.float64),
             y=np.asarray(y, np.float64),
             shape=np.asarray(shape, np.int64))


def build_one(src_nc_path, cache_dir, *, verbose=True):
    """Transcode one HIT*.nc into the cache. Skip if already present.

    Returns ``(x, y, shape)`` from the source file, or ``(None, None, None)``
    if the cache file already existed (so the caller can collect coords
    from whichever file it actually touched).
    """
    out = bin_path(cache_dir, src_nc_path)
    os.makedirs(cache_dir, exist_ok=True)
    if os.path.exists(out):
        if verbose:
            print(f'  [skip] {os.path.basename(src_nc_path)} (cached)', flush=True)
        return None, None, None

    import xarray as xr
    if verbose:
        print(f'  [build] {os.path.basename(src_nc_path)} -> {out}', flush=True)
    with xr.open_dataset(src_nc_path) as ds:
        u = np.asarray(ds['u'].values, dtype=DTYPE)
        v = np.asarray(ds['v'].values, dtype=DTYPE)
        x = np.asarray(ds['x'].values, dtype=np.float64)
        y = np.asarray(ds['y'].values, dtype=np.float64)
    arr = np.stack([u, v], axis=-1)  # (time, x, y, 2)

    tmp = out + '.tmp'
    arr.tofile(tmp)
    os.replace(tmp, out)
    return x, y, tuple(arr.shape)
