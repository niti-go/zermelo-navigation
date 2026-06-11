"""Trim an offline dataset to approximate a target return distribution.

Edit the constants at the top of this file to configure paths, the
target distribution shape, and the random seed.  No config file needed.

HOW THE TARGET DISTRIBUTION WORKS
-----------------------------------
TARGET_RETURNS and TARGET_DENSITIES define a piecewise-linear acceptance
probability curve — same length, TARGET_RETURNS are x-axis positions
(episode return values), TARGET_DENSITIES are heights in [0, 1]:

  0.0  → never keep an episode with this return
  1.0  → always keep an episode with this return

Straight lines are drawn between consecutive points.
Duplicate return values produce a near-vertical cliff.

Example — flat plateau from -200 to -50, sharp cliffs on both sides:
  TARGET_RETURNS   = [-600, -200, -200,  -50,  -50,    0]
  TARGET_DENSITIES = [ 0.0,  0.0,  1.0,  1.0,  0.0,  0.0]

  density
    1 |        ┌──────────┐
      |        │          │
    0 |────────┘          └────
         -600  -200      -50   0

Example — gradual ramps instead of cliffs:
  TARGET_RETURNS   = [-600, -300, -200,  -50,    0,   50]
  TARGET_DENSITIES = [ 0.0,  0.0,  1.0,  1.0,  0.0,  0.0]

  density
    1 |           /──────────\
      |          /            \
    0 |─────────/              \────
         -600 -300 -200  -50   0   50

HOW TRIMMING WORKS
-------------------
Each episode is kept with probability = target_density(episode_return).
Episodes in the density=1 region are always kept.
Episodes in the density=0 region are always discarded.
The output size is not predetermined — it depends on how many episodes
in the source fall in each density region.  The resulting distribution
shape will match the target.
"""

import numpy as np
import pathlib

# ---------------------------------------------------------------------------
# ~~~ EDIT THESE ~~~
# ---------------------------------------------------------------------------

INPUT_PATH  = 'datasets/waypointpoorquality_1initialcondition_gridsensors.npz'
OUTPUT_PATH = 'datasets/newest_thinned.npz'

# Piecewise-linear target distribution.
# Duplicate return values = near-vertical cliff.
TARGET_RETURNS   = [-600, -300, -250,  -100,  50,    0]
TARGET_DENSITIES = [ 0.0,  0.0,  1.0,  1.0,  0.5,  0.5]

SEED = 42

# ---------------------------------------------------------------------------
# Internals — no need to edit below this line
# ---------------------------------------------------------------------------


def _density_fn(xs, ys):
    """Build a vectorised piecewise-linear function from control points.

    Consecutive duplicate x values are separated by a tiny epsilon so that
    np.interp handles them as near-vertical cliffs.
    """
    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)
    assert len(xs) == len(ys) >= 2, "Need at least 2 control points"
    assert np.all(ys >= 0) and np.all(ys <= 1), "Densities must be in [0, 1]"
    for i in range(1, len(xs)):
        if xs[i] <= xs[i - 1]:
            xs[i] = xs[i - 1] + 1e-9
    left_val  = float(ys[0])
    right_val = float(ys[-1])
    def fn(values):
        return np.interp(np.asarray(values, dtype=float),
                         xs, ys, left=left_val, right=right_val)
    return fn


def _episode_slices(terminals):
    """Return list of (start, end) index pairs (inclusive) for each episode."""
    ends   = np.where(np.asarray(terminals, dtype=float) >= 0.5)[0]
    starts = np.concatenate([[0], ends[:-1] + 1])
    return list(zip(starts.tolist(), ends.tolist()))


def _ascii_histogram(values, lo, hi, n_bins=40, width=50):
    """Print a simple ASCII histogram to the terminal."""
    counts, edges = np.histogram(values, bins=n_bins, range=(lo, hi))
    max_count = max(counts.max(), 1)
    bar_scale = width / max_count
    for i in range(n_bins):
        bar_len = int(round(counts[i] * bar_scale))
        label = f'{edges[i]:>7.0f}'
        print(f'  {label} | {"█" * bar_len} {counts[i]}')


def main():
    rng = np.random.default_rng(SEED)

    # --- Load ---------------------------------------------------------------
    print(f'Loading {INPUT_PATH} ...', flush=True)
    raw  = np.load(INPUT_PATH, allow_pickle=False)
    data = {k: raw[k] for k in raw.files}
    keys = list(data.keys())
    n_steps_src = len(data[keys[0]])
    print(f'  {n_steps_src:,} steps  |  keys: {keys}')

    # --- Segment into episodes ----------------------------------------------
    slices = _episode_slices(data['terminals'])
    n_src  = len(slices)
    print(f'  {n_src:,} episodes')

    # --- Per-episode returns & success rate ---------------------------------
    rewards = np.asarray(data['rewards'], dtype=float)
    masks   = np.asarray(data['masks'],   dtype=float)
    returns = np.array([rewards[s:e + 1].sum() for s, e in slices])
    success = np.array([masks[e] == 0.0         for s, e in slices])

    print(f'\nSource return distribution:')
    print(f'  mean={returns.mean():.1f}  '
          f'min={returns.min():.1f}  max={returns.max():.1f}  '
          f'success={success.mean():.1%}')

    lo = min(returns.min(), TARGET_RETURNS[0])
    hi = max(returns.max(), TARGET_RETURNS[-1])
    _ascii_histogram(returns, lo, hi)

    # --- Acceptance probabilities & rejection sampling ----------------------
    density  = _density_fn(TARGET_RETURNS, TARGET_DENSITIES)
    probs    = np.clip(density(returns), 0.0, 1.0)
    keep     = rng.random(n_src) < probs
    kept_idx = np.where(keep)[0]
    n_kept   = len(kept_idx)

    print(f'\nKept {n_kept:,} / {n_src:,} episodes '
          f'({100 * n_kept / n_src:.1f}%)')

    if n_kept == 0:
        print('\nERROR: no episodes kept.  '
              'Check that TARGET_RETURNS covers the source return range.')
        return

    # --- Stats for kept episodes --------------------------------------------
    kept_returns = returns[kept_idx]
    kept_success = success[kept_idx]
    print(f'\nTrimmed return distribution:')
    print(f'  mean={kept_returns.mean():.1f}  '
          f'min={kept_returns.min():.1f}  max={kept_returns.max():.1f}  '
          f'success={kept_success.mean():.1%}')
    _ascii_histogram(kept_returns, lo, hi)

    # --- Reconstruct flat arrays from kept episodes -------------------------
    print('\nAssembling trimmed dataset...', flush=True)
    kept_slices = [slices[i] for i in kept_idx]
    out = {k: np.concatenate([arr[s:e + 1] for s, e in kept_slices], axis=0)
           for k, arr in data.items()}

    n_steps_out = len(out[keys[0]])
    print(f'  {n_steps_out:,} steps in trimmed dataset')

    # --- Save ---------------------------------------------------------------
    pathlib.Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT_PATH, **out)
    print(f'\nSaved to {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
