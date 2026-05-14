"""Evaluate BC, DT, and MeanFlowQL on Zermelo — held-out HIT flow + sanity check.

For each episode, samples a random (start, goal) and runs all three policies
on that same pair. Each eval segment ('heldout' / 'train') pins the flow-clock
start frame inside its segment so policies are tested on the flow regime we
intend (HIT46..HIT49 are reserved by `flow.train_max_file` and were never
seen at dataset-generation time).

Everything is configured by the constants at the top of the file.

Usage
-----
    conda activate flowrl
    cd ~/zermelo-navigation
    python scripts/evaluate_models.py

Outputs
-------
    results/<EXP_PROJECT>/<timestamp>/
        manifest.json          # config + resolved checkpoint paths
        zermelo_config.json    # snapshot of the config used (from BC run dir)
        metrics.csv            # one row per (algo, segment) — means + CIs
        metrics.json           # same data, machine-readable
        raw_episodes.json      # per-episode dicts (no frames)
        plots/                 # png/mp4 outputs (see _plot_* functions)
        videos/                # one stitched mp4 per recorded episode
"""

import csv
import glob
import json
import os
import pickle
import shutil
import sys
import time
from datetime import datetime

os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['MUJOCO_GL'] = 'egl'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

# ── GPU selection (applied before torch/jax import) ────────────────────────
# Pinned device id (e.g. '0', '0,1') or None to auto-pick the least-loaded GPU.
# Priority: env-var override (`CUDA_VISIBLE_DEVICES=0 python …`) > this
# constant > auto-pick.
CUDA_VISIBLE_DEVICES = None


def _pick_free_gpu():
    import subprocess
    out = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,memory.used',
         '--format=csv,noheader,nounits'], text=True,
    )
    rows = [(line.split(',')[0].strip(), int(line.split(',')[1].strip()))
            for line in out.strip().splitlines()]
    rows.sort(key=lambda r: r[1])
    return rows[0][0]


if os.environ.get('CUDA_VISIBLE_DEVICES'):
    print(f'GPU: using env-var CUDA_VISIBLE_DEVICES={os.environ["CUDA_VISIBLE_DEVICES"]}')
elif CUDA_VISIBLE_DEVICES is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(CUDA_VISIBLE_DEVICES)
    print(f'GPU: using pinned CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES}')
else:
    try:
        picked = _pick_free_gpu()
        os.environ['CUDA_VISIBLE_DEVICES'] = picked
        print(f'GPU: auto-picked least-loaded GPU {picked}')
    except Exception as e:
        print(f'GPU: auto-pick failed ({e!r}); leaving CUDA visibility default.')

import gymnasium
import imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _REPO_ROOT)
import zermelo_env  # noqa — registers gymnasium envs
from zermelo_env.zermelo_config import (
    load_config, config_to_env_kwargs, build_hit_flow_cfg,
)
from zermelo_env.hit_chain import HITChainFlow


# ─────────────────────────────────────────────────────────────────────────────
# User config — edit these, then run.
# ─────────────────────────────────────────────────────────────────────────────

# (For GPU selection, see CUDA_VISIBLE_DEVICES at the top of this file.)

# Which training project / runs to evaluate. RUN_TAG=None auto-picks the most
# recent run dir under exp/<EXP_PROJECT>/<algo>/.
EXP_PROJECT        = 'straight_general_v1'
RUN_TAG            = None
DEVICE             = 'cuda'
SEED               = 42

# How many episodes to evaluate per segment, and how many of those to record as videos.
NUM_EVAL_EPISODES  = 200
NUM_VIDEO_EPISODES = 3

# Flow segments to evaluate. Any subset of ('heldout', 'train'):
#   'heldout' — start_frames in [n_train_frames, n_total_frames). Reserved by
#               flow.train_max_file. This is the primary evaluation.
#   'train'   — start_frames in [0, n_train_frames). Sanity check / generalization
#               gap comparison.
EVAL_FLOW_SEGMENTS = ('heldout', 'train')

# How start_frames are scheduled within a segment. Mirrors dataset gen.
#   'deterministic_spread' — linspace across the segment
#   'random'               — uniform random (seeded)
START_FRAME_MODE   = 'deterministic_spread'

# Per-algo checkpoint selection.
#   'last'      — highest step number
#   'best_eval' — read eval.csv (MFQL only) and pick the step with highest
#                 evaluation/episode.return; falls back to 'last' if no csv.
#   int         — explicit step number
CHECKPOINT_POLICY  = {'BC': 'last', 'DT': 'last', 'MeanFlowQL': 'best_eval'}

# Whether to plot the offline-dataset return distribution alongside policy
# returns. The dataset was generated on training flow only; comparing held-out
# eval to it is intentional (it's the demonstration distribution the policies
# learned from).
COMPARE_TO_OFFLINE_DATASET = True

# Rendering / I/O.
RENDER_SIZE        = 200
VIDEO_FPS          = 30
RESULTS_ROOT       = os.path.join(_REPO_ROOT, 'results')


# Visual constants (changing these only affects plots, not eval).
ALGO_ORDER  = ('BC', 'DT', 'MeanFlowQL')
ALGO_COLORS = {'BC': '#1f77b4', 'DT': '#ff7f0e', 'MeanFlowQL': '#2ca02c'}
SEGMENT_LABEL = {'heldout': 'Held-out flow (HIT46..HIT49)',
                 'train':   'Training flow (HIT1..HIT45)'}


# ─────────────────────────────────────────────────────────────────────────────
# Model definitions — must match the training scripts byte-for-byte so the
# state_dicts load without surprises.
# ─────────────────────────────────────────────────────────────────────────────

class BCPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dims=(256, 256)):
        super().__init__()
        layers, in_dim = [], obs_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers += [nn.Linear(in_dim, act_dim), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        return self.net(obs)


class DecisionTransformer(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=128, n_heads=4,
                 n_layers=3, context_len=100, max_ep_len=1024, dropout=0.1):
        super().__init__()
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.hidden_dim = hidden_dim
        self.context_len = context_len
        self.embed_rtg    = nn.Linear(1, hidden_dim)
        self.embed_state  = nn.Linear(obs_dim, hidden_dim)
        self.embed_action = nn.Linear(act_dim, hidden_dim)
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_dim)
        self.embed_ln = nn.LayerNorm(hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.predict_action = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, act_dim), nn.Tanh(),
        )

    def forward(self, rtgs, states, actions, timesteps):
        B, T = states.shape[0], states.shape[1]
        rtg_e   = self.embed_rtg(rtgs)
        state_e = self.embed_state(states)
        act_e   = self.embed_action(actions)
        t_e     = self.embed_timestep(timesteps)
        stacked = torch.stack([rtg_e + t_e, state_e + t_e, act_e + t_e], dim=2)
        stacked = stacked.reshape(B, 3 * T, self.hidden_dim)
        stacked = self.embed_ln(stacked)
        L = 3 * T
        mask = torch.triu(torch.ones(L, L, device=stacked.device), diagonal=1).bool()
        h = self.transformer(stacked, mask=mask)
        return self.predict_action(h[:, 1::3, :])


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint discovery
# ─────────────────────────────────────────────────────────────────────────────

_ALGO_SUBDIR = {'BC': 'bc', 'DT': 'dt', 'MeanFlowQL': 'meanflowql'}
_CKPT_GLOB   = {'BC': 'policy_*.pt', 'DT': 'model_*.pt', 'MeanFlowQL': 'params_*.pkl'}


def _step_of(p):
    return int(os.path.basename(p).rsplit('_', 1)[1].split('.')[0])


def find_run_dir(algo):
    base = os.path.join(_REPO_ROOT, 'exp', EXP_PROJECT, _ALGO_SUBDIR[algo])
    if RUN_TAG is not None:
        cand = os.path.join(base, RUN_TAG)
        if not os.path.isdir(cand):
            raise FileNotFoundError(f'Run dir not found: {cand}')
        return cand
    if not os.path.isdir(base):
        raise FileNotFoundError(f'No {algo} runs under {base}')
    subs = [os.path.join(base, d) for d in os.listdir(base)
            if os.path.isdir(os.path.join(base, d))]
    if not subs:
        raise FileNotFoundError(f'No {algo} runs under {base}')
    # The trainer names dirs with a YYYYMMDD_HHMMSS suffix — sort is fine.
    return sorted(subs)[-1]


def select_checkpoint(run_dir, algo, policy):
    files = sorted(glob.glob(os.path.join(run_dir, _CKPT_GLOB[algo])))
    if not files:
        raise FileNotFoundError(f'No {_CKPT_GLOB[algo]} in {run_dir}')

    if isinstance(policy, int):
        match = [f for f in files if _step_of(f) == policy]
        if not match:
            raise FileNotFoundError(f'No {algo} checkpoint at step {policy} in {run_dir}')
        return match[0]

    if policy == 'last':
        return max(files, key=_step_of)

    if policy == 'best_eval':
        csv_path = os.path.join(run_dir, 'eval.csv')
        if not os.path.isfile(csv_path):
            print(f'  [{algo}] no eval.csv → falling back to last checkpoint')
            return max(files, key=_step_of)
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        valid = [r for r in rows if r.get('evaluation/episode.return', '') != '']
        if not valid:
            return max(files, key=_step_of)
        best = max(valid, key=lambda r: float(r['evaluation/episode.return']))
        best_step = int(float(best['step']))
        match = [f for f in files if _step_of(f) == best_step]
        if match:
            return match[0]
        # Save interval may not match eval interval — snap to nearest saved step.
        nearest = min(files, key=lambda p: abs(_step_of(p) - best_step))
        print(f'  [{algo}] best eval step {best_step} not saved; using '
              f'nearest saved step {_step_of(nearest)}')
        return nearest

    raise ValueError(f'Unknown CHECKPOINT_POLICY for {algo}: {policy!r}')


# ─────────────────────────────────────────────────────────────────────────────
# Policy loaders + uniform wrapper
# ─────────────────────────────────────────────────────────────────────────────

class _Policy:
    """Uniform interface. `reset()` clears any per-episode state."""
    def reset(self):
        pass

    def act(self, obs):
        raise NotImplementedError

    def observe_reward(self, r):
        pass  # only DT cares


class _BCPolicy(_Policy):
    def __init__(self, net, obs_mean, obs_std, device):
        self.net, self.device = net, device
        self.obs_mean, self.obs_std = obs_mean, obs_std

    def act(self, obs):
        x = torch.tensor((obs - self.obs_mean) / self.obs_std,
                         dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            a = self.net(x).cpu().numpy()[0]
        return np.clip(a, -1, 1)


class _DTPolicy(_Policy):
    def __init__(self, net, obs_mean, obs_std, context_len, max_ep_len,
                 rtg_scale, target_return, device):
        self.net = net
        self.obs_mean, self.obs_std = obs_mean, obs_std
        self.ctx_len, self.max_ep_len = context_len, max_ep_len
        self.rtg_scale = rtg_scale
        self.target_return = target_return
        self.device = device
        self.obs_dim, self.act_dim = len(obs_mean), 2

    def reset(self):
        K, D, A = self.ctx_len, self.obs_dim, self.act_dim
        self._states    = np.zeros((K, D), dtype=np.float32)
        self._actions   = np.zeros((K, A), dtype=np.float32)
        self._rtgs      = np.zeros((K, 1), dtype=np.float32)
        self._timesteps = np.zeros(K, dtype=np.int64)
        self._ep_len    = 0
        self._remaining = self.target_return

    def act(self, obs):
        K = self.ctx_len
        t = min(self._ep_len, K - 1)
        if self._ep_len >= K:
            self._states[:-1]    = self._states[1:]
            self._actions[:-1]   = self._actions[1:]
            self._rtgs[:-1]      = self._rtgs[1:]
            self._timesteps[:-1] = self._timesteps[1:]
            t = K - 1
        self._states[t] = (obs - self.obs_mean) / self.obs_std
        self._rtgs[t, 0] = self._remaining / self.rtg_scale
        self._timesteps[t] = min(self._ep_len, self.max_ep_len - 1)
        seq = t + 1
        s  = torch.tensor(self._states[:seq],    device=self.device).unsqueeze(0)
        a  = torch.tensor(self._actions[:seq],   device=self.device).unsqueeze(0)
        r  = torch.tensor(self._rtgs[:seq],      device=self.device).unsqueeze(0)
        ts = torch.tensor(self._timesteps[:seq], device=self.device).unsqueeze(0)
        with torch.no_grad():
            preds = self.net(r, s, a, ts)
        action = np.clip(preds[0, -1].cpu().numpy(), -1, 1)
        self._actions[t] = action
        self._ep_len += 1
        return action

    def observe_reward(self, r):
        self._remaining -= r


class _MFQLPolicy(_Policy):
    def __init__(self, get_action):
        self._get = get_action  # already handles obs norm internally

    def act(self, obs):
        return self._get(obs)


def load_bc(run_dir, ckpt_path, device):
    with open(os.path.join(run_dir, 'flags.json')) as f:
        flags = json.load(f)
    stats = np.load(os.path.join(run_dir, 'obs_norm_stats.npz'))
    obs_mean, obs_std = stats['obs_mean'], stats['obs_std']
    net = BCPolicy(len(obs_mean), 2,
                   hidden_dims=flags.get('hidden_dims', [256, 256])).to(device)
    net.load_state_dict(torch.load(ckpt_path, map_location=device))
    net.eval()
    print(f'  BC loaded: step={_step_of(ckpt_path)}')
    return _BCPolicy(net, obs_mean, obs_std, device)


def load_dt(run_dir, ckpt_path, device, dataset_path):
    with open(os.path.join(run_dir, 'flags.json')) as f:
        flags = json.load(f)
    stats = np.load(os.path.join(run_dir, 'obs_norm_stats.npz'))
    obs_mean, obs_std = stats['obs_mean'], stats['obs_std']
    net = DecisionTransformer(
        obs_dim=len(obs_mean), act_dim=2,
        hidden_dim=flags['hidden_dim'], n_heads=flags['n_heads'],
        n_layers=flags['n_layers'], context_len=flags['context_len'],
        max_ep_len=flags['max_ep_len'], dropout=flags['dropout'],
    ).to(device)
    net.load_state_dict(torch.load(ckpt_path, map_location=device))
    net.eval()
    # target_return: max episode return in the offline dataset.
    target_return = _dataset_max_return(dataset_path)
    print(f'  DT loaded: step={_step_of(ckpt_path)}, '
          f'target_return={target_return:.1f}')
    return _DTPolicy(net, obs_mean, obs_std,
                     context_len=flags['context_len'],
                     max_ep_len=flags['max_ep_len'],
                     rtg_scale=flags['rtg_scale'],
                     target_return=target_return,
                     device=device), target_return


def load_mfql(run_dir, ckpt_path):
    import jax
    import flax
    mfql_root = os.path.join(_REPO_ROOT, 'ext', 'MeanFlowQL')
    if mfql_root not in sys.path:
        sys.path.insert(0, mfql_root)
    from agents import agents as agent_registry
    from absl import flags as absl_flags

    FLAGS = absl_flags.FLAGS
    # MeanFlowQL's agent factory pokes at a few global flags; declare them
    # idempotently so we work whether or not absl has been initialized.
    for name, default in [('offline_steps', 1_000_000), ('online_steps', 0)]:
        if name not in FLAGS:
            absl_flags.DEFINE_integer(name, default, '')
    for name, default in [('pretrain_factor', 0.0)]:
        if name not in FLAGS:
            absl_flags.DEFINE_float(name, default, '')
    try:
        FLAGS(sys.argv[:1])
    except Exception:
        pass

    with open(os.path.join(run_dir, 'flags.json')) as f:
        flags_dict = json.load(f)
    config = flags_dict['agent']
    seed = flags_dict.get('seed', 0)
    FLAGS['offline_steps'].value   = flags_dict.get('offline_steps', 1_000_000)
    FLAGS['online_steps'].value    = flags_dict.get('online_steps', 0)
    FLAGS['pretrain_factor'].value = flags_dict.get('pretrain_factor', 0.0)

    stats = np.load(os.path.join(run_dir, 'obs_norm_stats.npz'))
    obs_mean, obs_std = stats['obs_mean'], stats['obs_std']
    obs_dim = len(obs_mean)

    dummy_obs = np.zeros((1, obs_dim), dtype=np.float32)
    dummy_act = np.zeros((1, 2),       dtype=np.float32)
    agent = agent_registry[config['agent_name']].create(
        seed, dummy_obs, dummy_act, config)

    with open(ckpt_path, 'rb') as f:
        load_dict = pickle.load(f)
    agent = flax.serialization.from_state_dict(agent, load_dict['agent'])
    print(f'  MeanFlowQL loaded: step={_step_of(ckpt_path)}')

    rng = jax.random.PRNGKey(seed)
    use_norm = flags_dict.get('use_observation_normalization', True)

    def get_action(obs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        ob_b = obs[None, :] if obs.ndim == 1 else obs
        if use_norm:
            ob_b = (ob_b - obs_mean) / obs_std
        a = np.array(agent.sample_actions(observations=ob_b, seed=key))
        if a.ndim > 1 and a.shape[0] == 1:
            a = a[0]
        return np.clip(a, -1, 1)

    return _MFQLPolicy(get_action)


# ─────────────────────────────────────────────────────────────────────────────
# Env + flow-segment helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_env(cfg, render_mode=None):
    """Eval env: random start/goal each reset, no within-cell jitter.

    Does NOT set max_file on the flow, so HIT46..HIT49 are accessible.
    """
    kw = config_to_env_kwargs(cfg)
    kw['fixed_start_goal']    = False
    kw['add_noise_to_goal']   = False
    kw['add_noise_to_start']  = False
    kw['max_episode_steps']   = cfg['run']['max_episode_steps']
    extra = {}
    if render_mode is not None:
        extra = dict(render_mode=render_mode, width=RENDER_SIZE, height=RENDER_SIZE)
    return gymnasium.make('zermelo-pointmaze-medium-v0', **kw, **extra)


def n_train_frames(cfg, env):
    """Flow frames corresponding to HIT1..HIT{train_max_file}."""
    n_total = int(env.unwrapped.n_frames)
    nc_dir = cfg['flow']['nc_dir']
    if not os.path.isabs(nc_dir):
        nc_dir = os.path.join(_REPO_ROOT, nc_dir)
    n_files = len(glob.glob(os.path.join(nc_dir, 'HIT*.nc')))
    train_max = cfg['flow'].get('train_max_file', n_files)
    return int(n_total * train_max / max(n_files, 1))


def episode_start_frames(segment, n_episodes, cfg, env):
    """Return (start_frames[n_episodes], wraps_into_train) for `segment`.

    Spread across the segment per START_FRAME_MODE. If the segment is shorter
    than one max-length episode, episodes may wrap modulo n_frames (back into
    the training segment); the second return flags that case.
    """
    fps = float(env.unwrapped.frames_per_step)
    max_steps = int(cfg['run']['max_episode_steps'])
    span = max_steps * fps
    n_total = int(env.unwrapped.n_frames)
    n_train = n_train_frames(cfg, env)

    if segment == 'train':
        lo, hi_full = 0.0, float(n_train)
    elif segment == 'heldout':
        lo, hi_full = float(n_train), float(n_total)
    else:
        raise ValueError(f'Unknown segment: {segment!r}')

    hi_safe = max(lo, hi_full - span)
    can_avoid_wrap = hi_safe > lo
    hi = hi_safe if can_avoid_wrap else hi_full
    wraps = not can_avoid_wrap

    if START_FRAME_MODE == 'random':
        rng = np.random.default_rng(SEED + hash(segment) % (2**31))
        starts = rng.uniform(lo, hi, size=n_episodes)
    elif START_FRAME_MODE == 'deterministic_spread':
        if n_episodes <= 1 or hi <= lo:
            starts = np.full(n_episodes, lo, dtype=np.float64)
        else:
            starts = np.linspace(lo, hi, n_episodes)
    else:
        raise ValueError(f'Unknown START_FRAME_MODE: {START_FRAME_MODE!r}')

    return starts.astype(np.float64), wraps


def free_cells(env):
    """Free maze cells (i, j). Reads the actual maze map so it stays correct
    whether maze.enabled is true or false."""
    return list(env.unwrapped._free_cells)


def sample_episode_tasks(cells, n_episodes, seed):
    """Per-episode (init_ij, goal_ij), distinct, reproducible."""
    rng = np.random.default_rng(seed)
    tasks = []
    for _ in range(n_episodes):
        a = cells[rng.integers(len(cells))]
        b = cells[rng.integers(len(cells))]
        while b == a:
            b = cells[rng.integers(len(cells))]
        tasks.append((a, b))
    return tasks


# ─────────────────────────────────────────────────────────────────────────────
# Episode runner
# ─────────────────────────────────────────────────────────────────────────────

def run_episode(env, policy, start_frame, init_ij, goal_ij, record_video):
    """One episode of `policy` on `env`. Returns metrics + trajectory + frames."""
    opts = {'task_info': {'init_ij': init_ij, 'goal_ij': goal_ij},
            'start_frame': float(start_frame)}
    obs, _ = env.reset(options=opts)
    policy.reset()

    goal_xy = np.asarray(env.unwrapped.cur_goal_xy, dtype=np.float64)
    init_dist = float(np.linalg.norm(np.asarray(obs[:2]) - goal_xy))

    done = False
    ep_ret = 0.0
    action_effort = 0.0
    traj, frames = [], []
    info = {}
    while not done:
        traj.append([float(obs[0]), float(obs[1])])
        if record_video:
            frames.append(env.render())
        action = policy.act(obs)
        action_effort += float(np.linalg.norm(action))
        obs, reward, terminated, truncated, info = env.step(action)
        policy.observe_reward(reward)
        done = terminated or truncated
        ep_ret += reward

    return {
        'return':        float(ep_ret),
        'length':        int(len(traj)),
        'success':       float(info.get('success', 0.0)),
        'action_effort': float(action_effort),
        'init_dist':     init_dist,
        'final_dist':    float(info.get('dist_to_goal', float('nan'))),
        'trajectory':    traj,
        'frames':        frames,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Dataset metrics (for the offline-distribution comparison panel)
# ─────────────────────────────────────────────────────────────────────────────

def _dataset_episode_metrics(cfg):
    """Per-episode return / success / length / mean ||a|| from the offline .npz.
    Returns None if the dataset isn't present."""
    p = os.path.join(_REPO_ROOT, cfg['run']['save_path'])
    if not os.path.exists(p):
        return None
    data = np.load(p)
    rewards = data['rewards'].astype(np.float64)
    terminals = data['terminals'].astype(np.float32)
    ends = np.where(terminals > 0.5)[0]
    if len(ends) == 0:
        return None
    starts = np.concatenate([[0], ends[:-1] + 1])
    goal_tol = float(cfg['env']['goal_tolerance'])
    actions = data['actions'].astype(np.float32)
    amag = np.linalg.norm(actions, axis=1)
    return {
        'return':            np.array([rewards[s:e + 1].sum() for s, e in zip(starts, ends)]),
        'length':            np.array([e - s + 1 for s, e in zip(starts, ends)], dtype=np.float64),
        'success':           (data['dist_to_goal'][ends] <= goal_tol).astype(np.float32),
        'mean_action_mag':   np.array([amag[s:e + 1].mean() for s, e in zip(starts, ends)]),
    }


def _dataset_max_return(dataset_path):
    data = np.load(dataset_path)
    rewards = data['rewards'].astype(np.float32)
    terminals = data['terminals'].astype(np.float32)
    ends = np.where(terminals > 0.5)[0]
    starts = np.concatenate([[0], ends[:-1] + 1])
    return float(max(rewards[s:e + 1].sum() for s, e in zip(starts, ends)))


# ─────────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_ci(values, n_boot=1000, alpha=0.05, seed=0):
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = arr[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _wall_patches(ax, env):
    """Boundary walls (and any internal walls) from the actual maze map."""
    maze_map = env.unwrapped.maze_map
    unit = float(env.unwrapped._maze_unit)
    ox, oy = float(env.unwrapped._offset_x), float(env.unwrapped._offset_y)
    H, W = maze_map.shape
    for i in range(H):
        for j in range(W):
            if maze_map[i, j] == 1:
                x = j * unit - ox - unit / 2
                y = i * unit - oy - unit / 2
                ax.add_patch(plt.Rectangle(
                    (x, y), unit, unit,
                    facecolor='#cccccc', edgecolor='#999999', linewidth=0.5))


def plot_trajectories(results, cfg, env, segment, out_path):
    """One subplot per algo. Flow snapshot at the median start_frame for visual
    consistency. Successes solid (algo color), failures dashed (red)."""
    flow_kwargs = build_hit_flow_cfg(cfg)
    flow_kwargs.pop('frames_per_step', None)
    flow = HITChainFlow(**flow_kwargs)
    median_frame = float(np.median([ep['start_frame'] for ep in results[ALGO_ORDER[0]]]))

    xs = np.linspace(-2, 26, 25)
    ys = np.linspace(-2, 26, 25)
    xx, yy = np.meshgrid(xs, ys)
    vx, vy = flow.get_flow_grid(xs, ys, frame=median_frame)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, algo in zip(axes, ALGO_ORDER):
        _wall_patches(ax, env)
        ax.quiver(xx, yy, vx, vy, alpha=0.12, color='gray', scale=30)
        for ep in results[algo]:
            traj = np.array(ep['trajectory'])
            if len(traj) == 0:
                continue
            color = ALGO_COLORS[algo] if ep['success'] > 0.5 else '#d62728'
            ls = '-' if ep['success'] > 0.5 else '--'
            ax.plot(traj[:, 0], traj[:, 1], color=color, alpha=0.3,
                    linewidth=0.8, linestyle=ls)
            ax.plot(traj[0, 0],  traj[0, 1],  'o', color=color, markersize=3, alpha=0.5)
            ax.plot(traj[-1, 0], traj[-1, 1], '*', color=color, markersize=4, alpha=0.5)
        sr = np.mean([e['success'] for e in results[algo]])
        mr = np.mean([e['return']  for e in results[algo]])
        ax.set_title(f'{algo}  (success={sr:.0%}, return={mr:.1f})')
        ax.set_xlim(-4, 24); ax.set_ylim(-4, 24)
        ax.set_aspect('equal')
        ax.set_xlabel('x'); ax.set_ylabel('y')

    fig.suptitle(f'Trajectories — {SEGMENT_LABEL[segment]}', fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_comparison(results, cfg, segment, out_path):
    """Bar chart: success / return / length / mean action, with bootstrap CIs.
    Adds the offline-dataset distribution as a reference bar (always — the
    dataset is the demonstration distribution the policies learned from)."""
    metrics = [('success',          'Success rate'),
               ('return',           'Episode return'),
               ('length',           'Episode length'),
               ('mean_action_mag',  'Mean ||action|| / step')]
    ds = _dataset_episode_metrics(cfg) if COMPARE_TO_OFFLINE_DATASET else None
    labels = list(ALGO_ORDER) + (['Offline (train flow)'] if ds is not None else [])
    colors = [ALGO_COLORS[a] for a in ALGO_ORDER] + (
        ['#7f7f7f'] if ds is not None else [])

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    for ax, (key, ylabel) in zip(axes, metrics):
        means, los, his = [], [], []
        for algo in ALGO_ORDER:
            if key == 'mean_action_mag':
                vals = [e['action_effort'] / e['length']
                        for e in results[algo] if e['length'] > 0]
            else:
                vals = [e[key] for e in results[algo]]
            m = float(np.mean(vals)) if vals else 0.0
            lo, hi = bootstrap_ci(vals)
            means.append(m); los.append(m - lo); his.append(hi - m)
        if ds is not None:
            v = ds[key]
            means.append(float(v.mean()))
            lo, hi = bootstrap_ci(v)
            los.append(means[-1] - lo); his.append(hi - means[-1])
        x = np.arange(len(labels))
        bars = ax.bar(x, means, yerr=[los, his], capsize=4, color=colors, alpha=0.85)
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{m:.2f}', ha='center', va='bottom', fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha='right')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3, axis='y')
    fig.suptitle(f'Policy performance — {SEGMENT_LABEL[segment]}  '
                 f'(error bars: 95% bootstrap CI)', fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_return_histogram(results, cfg, segment, out_path):
    """Per-algo return histograms + dataset distribution (shared x-axis)."""
    ds = _dataset_episode_metrics(cfg) if COMPARE_TO_OFFLINE_DATASET else None
    all_r = [e['return'] for a in ALGO_ORDER for e in results[a]]
    if ds is not None:
        all_r = all_r + ds['return'].tolist()
    if not all_r:
        return
    lo = max(-300.0, float(np.min(all_r)))
    hi = max(float(np.max(all_r)), lo + 1.0)
    bins = np.linspace(lo, hi, 51)

    panels = []
    for algo in ALGO_ORDER:
        rs = [e['return']  for e in results[algo]]
        ss = [e['success'] for e in results[algo]]
        panels.append((algo, rs, ALGO_COLORS[algo],
                       f'{algo}  (mean={np.mean(rs):.1f}, '
                       f'success={np.mean(ss):.0%}, n={len(rs)})'))
    if ds is not None:
        panels.append(('Offline (train flow)', ds['return'].tolist(), '#7f7f7f',
                       f'Offline dataset  (n={len(ds["return"])}, '
                       f'mean={ds["return"].mean():.1f}, '
                       f'success={ds["success"].mean():.0%})'))

    fig, axes = plt.subplots(len(panels), 1,
                             figsize=(8, 2.2 * len(panels)), sharex=True)
    if len(panels) == 1:
        axes = [axes]
    for ax, (_, rs, color, title) in zip(axes, panels):
        ax.hist(np.clip(rs, lo, hi), bins=bins, color=color, alpha=0.85,
                edgecolor='black', linewidth=0.4)
        ax.set_xlim(lo, hi)
        ax.set_ylabel('Episodes')
        ax.set_title(title, loc='left', fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
    axes[-1].set_xlabel('Episode return  (clipped to ≥ −300 for readability)')
    fig.suptitle(f'Return distribution — {SEGMENT_LABEL[segment]}', fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_success_vs_init_dist(results, segment, out_path):
    """Per-algo: rolling success rate as a function of initial distance to goal.
    Tells you whether each algo's wins are easy short tasks vs. genuinely
    long-route generalization."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for algo in ALGO_ORDER:
        d  = np.array([e['init_dist'] for e in results[algo]])
        s  = np.array([e['success']   for e in results[algo]])
        order = np.argsort(d)
        d, s = d[order], s[order]
        # Sliding window of 25 episodes.
        w = max(5, min(25, len(d) // 4))
        kernel = np.ones(w) / w
        rolling = np.convolve(s, kernel, mode='valid')
        d_mid = d[w - 1:]
        ax.plot(d_mid, rolling, color=ALGO_COLORS[algo], label=algo, linewidth=1.8)
        ax.scatter(d, s + (ALGO_ORDER.index(algo) - 1) * 0.02,
                   s=8, color=ALGO_COLORS[algo], alpha=0.25)
    ax.set_xlabel('Initial distance to goal')
    ax.set_ylabel(f'Rolling success rate (window={w})')
    ax.set_ylim(-0.05, 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower left')
    ax.set_title(f'Success vs. task difficulty — {SEGMENT_LABEL[segment]}')
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_energy_vs_return(results, segment, out_path):
    """Energy (mean ||action||) vs episode return, colored by success.
    Reveals whether returns correlate with effort and whether any algo gets
    away with low-effort high-return episodes."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True, sharey=True)
    for ax, algo in zip(axes, ALGO_ORDER):
        rets   = np.array([e['return']        for e in results[algo]])
        effort = np.array([e['action_effort'] / max(e['length'], 1)
                           for e in results[algo]])
        succ   = np.array([e['success']       for e in results[algo]])
        ax.scatter(effort[succ > 0.5], rets[succ > 0.5],
                   s=14, c=ALGO_COLORS[algo], alpha=0.75, label='success')
        ax.scatter(effort[succ < 0.5], rets[succ < 0.5],
                   s=14, c='#d62728', alpha=0.5, label='fail', marker='x')
        ax.set_title(algo)
        ax.set_xlabel('Mean ||action|| per step')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', fontsize=9)
    axes[0].set_ylabel('Episode return')
    fig.suptitle(f'Energy vs. return — {SEGMENT_LABEL[segment]}', fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_train_vs_heldout(by_segment, out_path):
    """Side-by-side success rate and mean return: train vs held-out. The gap
    is the flow-overfitting penalty for each algo."""
    segs = [s for s in ('train', 'heldout') if s in by_segment]
    if len(segs) < 2:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(ALGO_ORDER))
    width = 0.35
    for ax, key, ylabel in [(axes[0], 'success', 'Success rate'),
                            (axes[1], 'return',  'Mean episode return')]:
        for off, seg in zip([-width / 2, width / 2], segs):
            vals  = [np.mean([e[key] for e in by_segment[seg][a]]) for a in ALGO_ORDER]
            cis   = [bootstrap_ci([e[key] for e in by_segment[seg][a]]) for a in ALGO_ORDER]
            errs_lo = [v - c[0] for v, c in zip(vals, cis)]
            errs_hi = [c[1] - v for v, c in zip(vals, cis)]
            color = '#bbbbbb' if seg == 'train' else '#444444'
            bars = ax.bar(x + off, vals, width=width, yerr=[errs_lo, errs_hi],
                          capsize=4, color=color, label=SEGMENT_LABEL[seg],
                          edgecolor='black', linewidth=0.5)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f'{v:.2f}', ha='center', va='bottom', fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(ALGO_ORDER)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=9)
    fig.suptitle('Generalization gap — training flow vs. held-out flow',
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Video helpers
# ─────────────────────────────────────────────────────────────────────────────

def _add_text_overlay(frames, label, success):
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype(
            '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 12)
        small = ImageFont.truetype(
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 11)
    except OSError:
        font = small = ImageFont.load_default()
    status = 'SUCCESS' if success else 'FAIL'
    sc = (0, 200, 0) if success else (220, 50, 50)
    bar_h, status_w = 20, 55
    out = []
    for f in frames:
        im = Image.fromarray(f)
        d = ImageDraw.Draw(im)
        d.rectangle([(0, 0), (RENDER_SIZE, bar_h)], fill=(0, 0, 0))
        d.text((6, 4), label, fill=(255, 255, 255), font=font)
        d.text((RENDER_SIZE - status_w, 4), status, fill=sc, font=small)
        out.append(np.array(im))
    return out


def _stitch_video(per_algo_frames, out_path):
    all_frames = []
    for frames in per_algo_frames:
        all_frames.extend(frames)
        if frames:
            all_frames.extend([np.zeros_like(frames[0])] * 8)
    if all_frames:
        imageio.mimsave(out_path, all_frames, fps=VIDEO_FPS)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def _maybe_drop_legacy_results():
    """One-off: the old hard-coded results dir from the previous version of
    this script. Cleaned up here so it doesn't linger. We never touch
    old_experiments/."""
    stale = os.path.join(_REPO_ROOT, 'results',
                         'zermelo_hit_poor_quality_offline_results')
    if os.path.isdir(stale):
        shutil.rmtree(stale)
        print(f'  cleaned up stale results dir: {stale}')


def main():
    t0 = time.time()
    _maybe_drop_legacy_results()

    # ── Output dir ──────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(RESULTS_ROOT, EXP_PROJECT, timestamp)
    plots_dir  = os.path.join(out_dir, 'plots')
    videos_dir = os.path.join(out_dir, 'videos')
    os.makedirs(plots_dir,  exist_ok=True)
    os.makedirs(videos_dir, exist_ok=True)
    print(f'Output dir: {out_dir}')

    # ── Resolve checkpoints ─────────────────────────────────────────────────
    print('\nResolving checkpoints…')
    run_dirs = {a: find_run_dir(a) for a in ALGO_ORDER}
    ckpts    = {a: select_checkpoint(run_dirs[a], a, CHECKPOINT_POLICY[a])
                for a in ALGO_ORDER}
    for a in ALGO_ORDER:
        print(f'  {a:12s}  run_dir={os.path.relpath(run_dirs[a], _REPO_ROOT)}')
        print(f'  {"":12s}  ckpt   ={os.path.relpath(ckpts[a],    _REPO_ROOT)}')

    # ── Config + env ────────────────────────────────────────────────────────
    cfg = load_config(None)
    device = torch.device(DEVICE)
    dataset_path = os.path.join(_REPO_ROOT, cfg['run']['save_path'])

    # ── Load policies ───────────────────────────────────────────────────────
    print('\nLoading policies…')
    bc_pi   = load_bc(run_dirs['BC'], ckpts['BC'], device)
    dt_pi, dt_target = load_dt(run_dirs['DT'], ckpts['DT'], device, dataset_path)
    mfql_pi = load_mfql(run_dirs['MeanFlowQL'], ckpts['MeanFlowQL'])
    policies = {'BC': bc_pi, 'DT': dt_pi, 'MeanFlowQL': mfql_pi}

    # ── Envs ────────────────────────────────────────────────────────────────
    render_env   = make_env(cfg, render_mode='rgb_array') if NUM_VIDEO_EPISODES > 0 else None
    headless_env = make_env(cfg) if NUM_VIDEO_EPISODES < NUM_EVAL_EPISODES else None
    probe_env = render_env or headless_env
    n_total = int(probe_env.unwrapped.n_frames)
    n_train = n_train_frames(cfg, probe_env)
    print(f'\nFlow chain: total={n_total} frames, train={n_train} '
          f'(held-out={n_total - n_train}), '
          f'fps={probe_env.unwrapped.frames_per_step:g}, '
          f'max_steps={cfg["run"]["max_episode_steps"]}')

    cells = free_cells(probe_env)
    tasks = sample_episode_tasks(cells, NUM_EVAL_EPISODES, SEED)

    # ── Run each segment ────────────────────────────────────────────────────
    by_segment = {}
    segment_wraps = {}
    for segment in EVAL_FLOW_SEGMENTS:
        print(f'\n══ Evaluating segment: {SEGMENT_LABEL[segment]} ══')
        start_frames, wraps = episode_start_frames(
            segment, NUM_EVAL_EPISODES, cfg, probe_env)
        segment_wraps[segment] = wraps
        if wraps:
            print(f'  ⚠ episodes longer than the segment ({len(start_frames)} '
                  f'frames headroom) will wrap modulo n_frames.')

        results = {a: [] for a in ALGO_ORDER}
        policy_time = {a: 0.0 for a in ALGO_ORDER}

        for i in tqdm(range(NUM_EVAL_EPISODES),
                      desc=f'  episodes [{segment}]', dynamic_ncols=True):
            init_ij, goal_ij = tasks[i]
            sf = float(start_frames[i])
            record = i < NUM_VIDEO_EPISODES
            env = render_env if record else headless_env

            algo_eps = {}
            for algo in ALGO_ORDER:
                t_a = time.time()
                ep = run_episode(env, policies[algo], sf, init_ij, goal_ij, record)
                policy_time[algo] += time.time() - t_a
                ep['start_frame'] = sf
                ep['init_ij']     = list(init_ij)
                ep['goal_ij']     = list(goal_ij)
                algo_eps[algo] = ep
                results[algo].append({k: v for k, v in ep.items() if k != 'frames'})

            if record:
                stitched = []
                for algo in ALGO_ORDER:
                    ep = algo_eps[algo]
                    lbl = f'{algo} (ret={ep["return"]:.1f})'
                    stitched.append(
                        _add_text_overlay(ep['frames'], lbl, ep['success'] > 0.5))
                _stitch_video(stitched, os.path.join(
                    videos_dir, f'{segment}_ep{i + 1:02d}.mp4'))

        by_segment[segment] = results

        # Per-segment console summary.
        print(f'\n  Results ({segment}, n={NUM_EVAL_EPISODES}):')
        print(f'    {"":>12}  {"success":>8}  {"return":>17}  {"length":>8}  {"time/ep":>8}')
        for algo in ALGO_ORDER:
            eps = results[algo]
            sr  = np.mean([e['success'] for e in eps])
            r   = np.array([e['return'] for e in eps])
            r_lo, r_hi = bootstrap_ci(r)
            length = np.mean([e['length'] for e in eps])
            print(f'    {algo:>12s}  {sr:>7.1%}  '
                  f'{r.mean():>7.1f} [{r_lo:>5.1f},{r_hi:>5.1f}]  '
                  f'{length:>8.0f}  {policy_time[algo] / NUM_EVAL_EPISODES:>7.2f}s')

        # Per-segment plots.
        plot_trajectories(results, cfg, probe_env, segment,
                          os.path.join(plots_dir, f'trajectories_{segment}.png'))
        plot_comparison(results, cfg, segment,
                        os.path.join(plots_dir, f'comparison_{segment}.png'))
        plot_return_histogram(results, cfg, segment,
                              os.path.join(plots_dir, f'return_histogram_{segment}.png'))
        plot_success_vs_init_dist(results, segment,
                                  os.path.join(plots_dir, f'success_vs_init_dist_{segment}.png'))
        plot_energy_vs_return(results, segment,
                              os.path.join(plots_dir, f'energy_vs_return_{segment}.png'))

    # ── Cross-segment plot ──────────────────────────────────────────────────
    if 'train' in by_segment and 'heldout' in by_segment:
        plot_train_vs_heldout(
            by_segment, os.path.join(plots_dir, 'train_vs_heldout.png'))

    # ── Persist metrics + manifest ──────────────────────────────────────────
    metrics_rows = []
    for segment, results in by_segment.items():
        for algo in ALGO_ORDER:
            eps = results[algo]
            ret = np.array([e['return']  for e in eps])
            suc = np.array([e['success'] for e in eps])
            ln  = np.array([e['length']  for e in eps])
            eff = np.array([e['action_effort'] / max(e['length'], 1) for e in eps])
            ret_lo, ret_hi = bootstrap_ci(ret)
            suc_lo, suc_hi = bootstrap_ci(suc)
            metrics_rows.append({
                'segment':           segment,
                'algo':              algo,
                'n':                 len(eps),
                'success_mean':      float(suc.mean()),
                'success_ci_lo':     suc_lo,
                'success_ci_hi':     suc_hi,
                'return_mean':       float(ret.mean()),
                'return_ci_lo':      ret_lo,
                'return_ci_hi':      ret_hi,
                'length_mean':       float(ln.mean()),
                'action_mean':       float(eff.mean()),
            })
    csv_path = os.path.join(out_dir, 'metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()))
        w.writeheader()
        w.writerows(metrics_rows)
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_rows, f, indent=2)
    print(f'\nWrote {csv_path}')

    # Raw per-episode dicts (no frames) — sometimes useful for ad-hoc analysis.
    with open(os.path.join(out_dir, 'raw_episodes.json'), 'w') as f:
        json.dump({s: {a: results_for_a for a, results_for_a in by_segment[s].items()}
                   for s in by_segment}, f)

    # Manifest: everything needed to reproduce this run.
    manifest = {
        'timestamp':           timestamp,
        'exp_project':         EXP_PROJECT,
        'run_tag':             RUN_TAG,
        'seed':                SEED,
        'num_eval_episodes':   NUM_EVAL_EPISODES,
        'num_video_episodes':  NUM_VIDEO_EPISODES,
        'eval_flow_segments':  list(EVAL_FLOW_SEGMENTS),
        'start_frame_mode':    START_FRAME_MODE,
        'checkpoint_policy':   CHECKPOINT_POLICY,
        'compare_to_offline_dataset': COMPARE_TO_OFFLINE_DATASET,
        'device':              DEVICE,
        'dataset_path':        os.path.relpath(dataset_path, _REPO_ROOT),
        'flow':                {'n_total_frames': n_total,
                                'n_train_frames': n_train,
                                'frames_per_step': float(probe_env.unwrapped.frames_per_step),
                                'max_episode_steps': cfg['run']['max_episode_steps'],
                                'segment_wraps_modulo': segment_wraps},
        'dt_target_return':    dt_target,
        'run_dirs':            {a: os.path.relpath(run_dirs[a], _REPO_ROOT) for a in ALGO_ORDER},
        'checkpoints':         {a: {'path': os.path.relpath(ckpts[a], _REPO_ROOT),
                                    'step': _step_of(ckpts[a])} for a in ALGO_ORDER},
        'wall_time_seconds':   time.time() - t0,
    }
    with open(os.path.join(out_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

    # Snapshot the zermelo_config that the policies were *trained* under
    # (taken from the BC run dir — all three are trained against the same yaml).
    train_cfg_src = os.path.join(run_dirs['BC'], 'zermelo_config.json')
    if os.path.isfile(train_cfg_src):
        shutil.copy2(train_cfg_src, os.path.join(out_dir, 'zermelo_config.json'))

    # ── Cleanup + final summary ─────────────────────────────────────────────
    if render_env is not None:   render_env.close()
    if headless_env is not None: headless_env.close()

    # Winner line for the primary segment (held-out if present, else train).
    primary = 'heldout' if 'heldout' in by_segment else EVAL_FLOW_SEGMENTS[0]
    succ = {a: np.mean([e['success'] for e in by_segment[primary][a]])
            for a in ALGO_ORDER}
    winner = max(ALGO_ORDER, key=lambda a: succ[a])
    others = ', '.join(f'{a}={succ[a]:.0%}' for a in ALGO_ORDER if a != winner)
    print('\n' + '═' * 70)
    print(f'  Primary segment ({primary}) winner: {winner} '
          f'(success={succ[winner]:.0%} | {others})')
    print('═' * 70)
    print(f'Done in {time.time() - t0:.0f}s. Outputs in: {out_dir}')


if __name__ == '__main__':
    main()
