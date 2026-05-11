"""Build the local-SSD HIT u/v cache from the source NetCDF files.

Run this once after first cloning the dataset (or after the source NetCDFs
change). At runtime, ``HITChainFlow`` mmaps the cache files written here
and never touches NFS.

Usage:
    python scripts/build_hit_cache.py
    python scripts/build_hit_cache.py --max_file=49

The cache dir defaults to ``$ZERMELO_HIT_CACHE_DIR`` or
``/tmp/zermelo_hit_cache_<uid>``. Override with ``--cache_dir``.

The build is idempotent: existing cache files are skipped. If the cache
already contains all .bin files but is missing ``coords.npz`` (e.g. from
an older layout), this script reads coords from one source NetCDF and
writes the file without rebuilding any .bin.
"""

import glob
import os
import re
import time

from absl import app, flags

import zermelo_env  # noqa
from zermelo_env import hit_cache
from zermelo_env.zermelo_config import build_hit_flow_cfg, load_config

FLAGS = flags.FLAGS
flags.DEFINE_string('config', None,
                    'Path to zermelo_config.yaml (uses built-in defaults if omitted).')
flags.DEFINE_integer('max_file', None,
                     'Build only HIT1..HIT{max_file}. Default: every HIT*.nc found.')
flags.DEFINE_string('cache_dir', None,
                    'Override the cache directory (default: $ZERMELO_HIT_CACHE_DIR '
                    'or /tmp/zermelo_hit_cache_<uid>).')


def _list_sources(nc_dir, max_file):
    files = glob.glob(os.path.join(nc_dir, '*.nc'))

    def _key(p):
        m = re.search(r'(\d+)', os.path.basename(p))
        return int(m.group(1)) if m else 0

    files = sorted(files, key=_key)
    if max_file is not None:
        files = [f for f in files if _key(f) <= max_file]
    return files


def main(_):
    cfg = load_config(FLAGS.config)
    flow_kwargs = build_hit_flow_cfg(cfg, max_file=FLAGS.max_file)
    nc_dir = flow_kwargs['nc_dir']
    cache_dir = FLAGS.cache_dir or hit_cache.default_cache_dir()

    print(f'Source dir: {nc_dir}')
    print(f'Cache dir:  {cache_dir}')

    src_paths = _list_sources(nc_dir, FLAGS.max_file)
    if not src_paths:
        raise SystemExit(f'No HIT*.nc files found in {nc_dir}')
    print(f'Found {len(src_paths)} source HIT*.nc file(s).')

    t0 = time.time()
    coords = None
    for p in src_paths:
        x, y, shape = hit_cache.build_one(p, cache_dir, verbose=True)
        if x is not None:
            coords = (x, y, shape)

    coords_p = hit_cache.coords_path(cache_dir)
    if not os.path.exists(coords_p):
        if coords is None:
            # Everything was already cached but coords.npz never got written
            # (e.g. older cache layout). Read coords from one source NetCDF.
            import xarray as xr
            print(f'Reading coords from {os.path.basename(src_paths[0])}...')
            with xr.open_dataset(src_paths[0]) as ds:
                import numpy as np
                x = np.asarray(ds['x'].values, dtype=np.float64)
                y = np.asarray(ds['y'].values, dtype=np.float64)
                shape = (int(ds.sizes['time']), int(ds.sizes['x']),
                         int(ds.sizes['y']), 2)
            coords = (x, y, shape)
        hit_cache.write_coords(cache_dir, *coords)
        print(f'Wrote coords -> {coords_p}')
    else:
        print(f'Coords already present: {coords_p}')

    print(f'Done in {time.time() - t0:.1f}s. Cache ready at {cache_dir}.')


if __name__ == '__main__':
    app.run(main)
