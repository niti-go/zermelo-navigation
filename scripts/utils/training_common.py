"""Shared helpers for offline-RL training scripts (bc, dt, meanflowql).

Three concerns each script duplicates:
  1. Loading a Zermelo .npz dataset and splitting it episode-wise into
     train/val. Each script wants a slightly different view (BC and MFQL
     want flat transition dicts; DT wants per-episode dicts with RTG).
  2. Building an eval env from `zermelo_config.yaml` with fixed start/goal.
  3. Computing observation-normalization statistics.

Helpers below provide the common pieces; each script composes the view it
needs from `load_episode_segments`.
"""
import os

import gymnasium
import numpy as np

import zermelo_env  # noqa — registers gymnasium envs
from zermelo_env.zermelo_config import load_config, config_to_env_kwargs

# Repo root — this file lives at scripts/utils/, two levels below.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_episode_segments(dataset_path, train_test_split=0.8, rng=None):
    """Load a .npz dataset and split episode indices into train/val.

    Returns:
      data: dict of raw arrays (whatever the .npz contains, cast to float32
            where appropriate).
      train_segments: list of (start, end_exclusive) tuples for train episodes.
      val_segments:   list of (start, end_exclusive) tuples for val episodes.

    Reproducible if you pass an `rng` (np.random.Generator); otherwise uses
    the global numpy RNG.
    """
    print(f"Loading dataset: {dataset_path}")
    data = dict(np.load(dataset_path))

    # Cast common keys to float32; leave the rest alone.
    for k in ('observations', 'next_observations', 'actions', 'rewards',
              'terminals', 'masks', 'dist_to_goal'):
        if k in data:
            data[k] = data[k].astype(np.float32)

    terminals = data['terminals']
    ends = np.where(terminals > 0.5)[0]
    starts = np.concatenate([[0], ends[:-1] + 1])
    segments = [(int(s), int(e) + 1) for s, e in zip(starts, ends)]

    n_ep = len(segments)
    idx = np.arange(n_ep)
    if rng is not None:
        rng.shuffle(idx)
    else:
        np.random.shuffle(idx)

    n_train = int(n_ep * train_test_split)
    train_idx = sorted(idx[:n_train].tolist())
    val_idx = sorted(idx[n_train:].tolist())
    train_segments = [segments[i] for i in train_idx]
    val_segments = [segments[i] for i in val_idx]

    print(f"  Total: {len(terminals)} transitions, {n_ep} episodes "
          f"(train={len(train_segments)}, val={len(val_segments)})")
    print(f"  Obs shape: {data['observations'].shape[1:]}, "
          f"Act shape: {data['actions'].shape[1:]}")
    return data, train_segments, val_segments


def flatten_segments(data, segments, keys):
    """Concatenate `data[k][s:e]` over all segments for each key in `keys`."""
    if not segments:
        return {k: data[k][:0] for k in keys}
    rows = np.concatenate([np.arange(s, e) for (s, e) in segments])
    return {k: data[k][rows] for k in keys}


def episode_views(data, segments, with_rtg=False):
    """Return one dict per episode for sequence-model scripts (e.g. DT).

    When `with_rtg=True`, each dict gets an `rtg` array (cumulative sum from
    end-of-episode back to step 0).
    """
    out = []
    for (s, e) in segments:
        ep = {
            'observations': data['observations'][s:e],
            'actions': data['actions'][s:e],
            'rewards': data['rewards'][s:e],
        }
        if with_rtg:
            r = ep['rewards']
            rtg = np.zeros_like(r)
            rtg[-1] = r[-1]
            for t in reversed(range(len(r) - 1)):
                rtg[t] = r[t] + rtg[t + 1]
            ep['rtg'] = rtg
        out.append(ep)
    return out


# ---------------------------------------------------------------------------
# Observation normalization
# ---------------------------------------------------------------------------

def compute_obs_norm(observations):
    """Mean/std of obs, with a 1e-6 floor on std to avoid divide-by-zero."""
    obs_mean = observations.mean(axis=0)
    obs_std = observations.std(axis=0) + 1e-6
    return obs_mean.astype(np.float32), obs_std.astype(np.float32)


def save_obs_norm(save_dir, obs_mean, obs_std):
    path = os.path.join(save_dir, 'obs_norm_stats.npz')
    np.savez(path, obs_mean=obs_mean, obs_std=obs_std)
    return path


# ---------------------------------------------------------------------------
# Eval env
# ---------------------------------------------------------------------------

def make_eval_env(zermelo_config_path=None, env_id='zermelo-pointmaze-medium-v0',
                  override_kwargs=None):
    """Create a Zermelo env from `zermelo_config.yaml` with fixed start/goal."""
    cfg = load_config(zermelo_config_path)
    env_kwargs = config_to_env_kwargs(cfg)
    env_kwargs['fixed_start_goal'] = True
    env_kwargs['max_episode_steps'] = cfg['run']['max_episode_steps']
    if override_kwargs:
        env_kwargs.update(override_kwargs)
    return gymnasium.make(env_id, **env_kwargs)


def default_dataset_path(zermelo_cfg):
    """Resolve `cfg['run']['save_path']` against the repo root."""
    return os.path.join(REPO_ROOT, zermelo_cfg['run']['save_path'])


def default_config_src_path(user_path):
    """Path to the YAML config file (user-supplied or the repo default)."""
    return user_path or os.path.join(REPO_ROOT, 'zermelo_config.yaml')
