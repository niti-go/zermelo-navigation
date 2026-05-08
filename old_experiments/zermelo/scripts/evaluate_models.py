"""Evaluate final policies of BC, DT, and MeanFlowQL on Zermelo.

For each episode, samples a random start/goal, then runs all 3 policies on
that same start/goal. Produces one comparison video per episode (3 clips
stitched together with algorithm labels), trajectory plots, and metrics.

Usage:
    conda activate flowrl
    cd ~/zermelo-navigation
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 python scripts/evaluate_models.py
"""
import argparse
import glob
import json
import os
import pickle
import sys
import time

os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['MUJOCO_GL'] = 'egl'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import gymnasium
import imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _repo_root)
import zermelo_env  # noqa — registers gymnasium envs
from zermelo_env.zermelo_config import load_config, config_to_env_kwargs, get_flow_runtime
from zermelo_env.zermelo_flow import DynamicNetCDFFlowField, DynamicTGVFlowField, FlowField

# ── Config ───────────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(_repo_root, 'results', 'zermelo_hit_poor_quality_offline_results')
NUM_EVAL_EPISODES = 200
NUM_VIDEO_EPISODES = 0  # Only render videos for the first N episodes; rest run headless.
VIDEO_FPS = 30
RENDER_SIZE = 200

#EDIT THIS TO FIND WANDB MODEL CHECKPOINTS
BC_DIR = os.path.join(_repo_root, 'exp', 'zermelo_hit_dynamic_poordataset', 'bc', 'bc_sd000_20260429_211818')
DT_DIR = os.path.join(_repo_root, 'exp', 'zermelo_hit_dynamic_poordataset', 'dt', 'dt_sd000_20260429_211834')
MFQL_DIR = os.path.join(_repo_root, 'exp', 'zermelo_hit_dynamic_poordataset', 'meanflowql', 'sd000_20260429_211845')

ALGO_COLORS = {'BC': '#1f77b4', 'DT': '#ff7f0e', 'MeanFlowQL': '#2ca02c'}
ALGO_ORDER = ['BC', 'DT', 'MeanFlowQL']

#This script reads the offline dataset from the config file


# ── Model definitions ────────────────────────────────────────────────────────

class BCPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dims=(256, 256)):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, act_dim))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        return self.net(obs)


class DecisionTransformer(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=128, n_heads=4,
                 n_layers=3, context_len=100, max_ep_len=1024, dropout=0.1):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.context_len = context_len
        self.embed_rtg = nn.Linear(1, hidden_dim)
        self.embed_state = nn.Linear(obs_dim, hidden_dim)
        self.embed_action = nn.Linear(act_dim, hidden_dim)
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_dim)
        self.embed_ln = nn.LayerNorm(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.predict_action = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, act_dim), nn.Tanh(),
        )

    def forward(self, rtgs, states, actions, timesteps):
        B, T = states.shape[0], states.shape[1]
        rtg_emb = self.embed_rtg(rtgs)
        state_emb = self.embed_state(states)
        action_emb = self.embed_action(actions)
        time_emb = self.embed_timestep(timesteps)
        rtg_emb = rtg_emb + time_emb
        state_emb = state_emb + time_emb
        action_emb = action_emb + time_emb
        stacked = torch.stack([rtg_emb, state_emb, action_emb], dim=2)
        stacked = stacked.reshape(B, 3 * T, self.hidden_dim)
        stacked = self.embed_ln(stacked)
        seq_len = 3 * T
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=stacked.device), diagonal=1).bool()
        hidden = self.transformer(stacked, mask=causal_mask)
        state_hidden = hidden[:, 1::3, :]
        return self.predict_action(state_hidden)


# ── Environment ──────────────────────────────────────────────────────────────

def make_eval_env(render_mode=None):
    cfg = load_config(None)
    env_kwargs = config_to_env_kwargs(cfg)
    env_kwargs['fixed_start_goal'] = False
    # Eval samples fresh start/goal cells each episode; suppress within-cell
    # jitter so both endpoints sit at their cell centers deterministically.
    env_kwargs['add_noise_to_goal'] = False
    env_kwargs['add_noise_to_start'] = False
    env_kwargs['max_episode_steps'] = cfg['run']['max_episode_steps']
    kw = {}
    if render_mode:
        kw = dict(render_mode=render_mode, width=RENDER_SIZE, height=RENDER_SIZE)
    return gymnasium.make('zermelo-pointmaze-medium-v0', **env_kwargs, **kw)


def get_free_cells():
    """Return list of (i, j) free cells in the maze grid."""
    cfg = load_config(None)
    env_kwargs = config_to_env_kwargs(cfg)
    # Open arena: boundary walls only. Free cells are (1,1) to (6,6).
    cells = []
    for i in range(1, 7):
        for j in range(1, 7):
            cells.append((i, j))
    return cells


def sample_start_goal(free_cells, rng):
    """Sample random distinct start and goal grid cells."""
    init_ij = free_cells[rng.integers(len(free_cells))]
    goal_ij = free_cells[rng.integers(len(free_cells))]
    while goal_ij == init_ij:
        goal_ij = free_cells[rng.integers(len(free_cells))]
    return init_ij, goal_ij


# ── Episode runners (accept reset options for controlled start/goal) ─────────

def run_bc_episode(policy, obs_mean, obs_std, env, device, reset_options=None,
                   record_video=False):
    ob, _ = env.reset(options=reset_options)
    done, ep_ret, action_effort, frames, traj = False, 0.0, 0.0, [], []
    while not done:
        traj.append(ob[:2].tolist())
        if record_video:
            frames.append(env.render())
        ob_t = torch.tensor(
            (ob - obs_mean) / obs_std, dtype=torch.float32, device=device
        ).unsqueeze(0)
        with torch.no_grad():
            action = policy(ob_t).cpu().numpy()[0]
        action = np.clip(action, -1, 1)
        action_effort += float(np.linalg.norm(action))
        ob, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        ep_ret += reward
    return {'return': ep_ret, 'length': len(traj), 'success': info.get('success', 0.0),
            'action_effort': action_effort, 'trajectory': traj, 'frames': frames}


def run_dt_episode(model, obs_mean, obs_std, flags, env, target_return, device,
                   reset_options=None, record_video=False):
    context_len = flags['context_len']
    max_ep_len = flags['max_ep_len']
    rtg_scale = flags['rtg_scale']
    obs_dim = len(obs_mean)
    act_dim = 2

    ob, _ = env.reset(options=reset_options)
    done, ep_ret, ep_len, action_effort = False, 0.0, 0, 0.0
    frames, traj = [], []
    states_buf = np.zeros((context_len, obs_dim), dtype=np.float32)
    actions_buf = np.zeros((context_len, act_dim), dtype=np.float32)
    rtgs_buf = np.zeros((context_len, 1), dtype=np.float32)
    timesteps_buf = np.zeros(context_len, dtype=np.int64)
    remaining_return = target_return

    while not done:
        traj.append(ob[:2].tolist())
        if record_video:
            frames.append(env.render())
        t = min(ep_len, context_len - 1)
        if ep_len >= context_len:
            states_buf[:-1] = states_buf[1:]
            actions_buf[:-1] = actions_buf[1:]
            rtgs_buf[:-1] = rtgs_buf[1:]
            timesteps_buf[:-1] = timesteps_buf[1:]
            t = context_len - 1
        states_buf[t] = (ob - obs_mean) / obs_std
        rtgs_buf[t, 0] = remaining_return / rtg_scale
        timesteps_buf[t] = min(ep_len, max_ep_len - 1)

        seq_len = t + 1
        s = torch.tensor(states_buf[:seq_len], device=device).unsqueeze(0)
        a = torch.tensor(actions_buf[:seq_len], device=device).unsqueeze(0)
        r = torch.tensor(rtgs_buf[:seq_len], device=device).unsqueeze(0)
        ts = torch.tensor(timesteps_buf[:seq_len], device=device).unsqueeze(0)
        with torch.no_grad():
            action_preds = model(r, s, a, ts)
        action = action_preds[0, -1].cpu().numpy()
        action = np.clip(action, -1.0, 1.0)
        actions_buf[t] = action
        action_effort += float(np.linalg.norm(action))

        ob, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        ep_ret += reward
        remaining_return -= reward
        ep_len += 1
    return {'return': ep_ret, 'length': ep_len, 'success': info.get('success', 0.0),
            'action_effort': action_effort, 'trajectory': traj, 'frames': frames}


def run_mfql_episode(get_action, env, reset_options=None, record_video=False):
    ob, _ = env.reset(options=reset_options)
    done, ep_ret, action_effort = False, 0.0, 0.0
    frames, traj = [], []
    while not done:
        traj.append(ob[:2].tolist())
        if record_video:
            frames.append(env.render())
        action = get_action(ob)
        action_effort += float(np.linalg.norm(action))
        ob, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        ep_ret += reward
    return {'return': ep_ret, 'length': len(traj), 'success': info.get('success', 0.0),
            'action_effort': action_effort, 'trajectory': traj, 'frames': frames}


# ── Model loading ────────────────────────────────────────────────────────────

def latest_checkpoint(directory, pattern):
    paths = glob.glob(os.path.join(directory, pattern))
    def get_step(p):
        return int(os.path.basename(p).split('_')[-1].split('.')[0])
    return max(paths, key=get_step)


def load_bc(device):
    ckpt = latest_checkpoint(BC_DIR, 'policy_*.pt')
    with open(os.path.join(BC_DIR, 'flags.json')) as f:
        flags = json.load(f)
    stats = np.load(os.path.join(BC_DIR, 'obs_norm_stats.npz'))
    obs_mean, obs_std = stats['obs_mean'], stats['obs_std']
    policy = BCPolicy(len(obs_mean), 2, hidden_dims=flags.get('hidden_dims', [256, 256])).to(device)
    policy.load_state_dict(torch.load(ckpt, map_location=device))
    policy.eval()
    print(f'  BC: loaded {os.path.basename(ckpt)}')
    return policy, obs_mean, obs_std


def load_dt(device):
    ckpt = latest_checkpoint(DT_DIR, 'model_*.pt')
    with open(os.path.join(DT_DIR, 'flags.json')) as f:
        flags = json.load(f)
    stats = np.load(os.path.join(DT_DIR, 'obs_norm_stats.npz'))
    obs_mean, obs_std = stats['obs_mean'], stats['obs_std']
    model = DecisionTransformer(
        obs_dim=len(obs_mean), act_dim=2,
        hidden_dim=flags['hidden_dim'], n_heads=flags['n_heads'],
        n_layers=flags['n_layers'], context_len=flags['context_len'],
        max_ep_len=flags['max_ep_len'], dropout=flags['dropout'],
    ).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    # Compute target return from dataset
    cfg = load_config(None)
    dataset_path = os.path.join(_repo_root, cfg['run']['save_path'])
    data = dict(np.load(dataset_path))
    rewards = data['rewards'].astype(np.float32)
    terminals = data['terminals'].astype(np.float32)
    ends = np.where(terminals > 0.5)[0]
    starts = np.concatenate([[0], ends[:-1] + 1])
    target_return = float(max(rewards[s:e + 1].sum() for s, e in zip(starts, ends)))
    print(f'  DT: loaded {os.path.basename(ckpt)}, target_return={target_return:.1f}')
    return model, obs_mean, obs_std, flags, target_return


def load_mfql():
    import jax
    import flax
    mfql_root = os.path.join(_repo_root, 'ext', 'MeanFlowQL')
    if mfql_root not in sys.path:
        sys.path.insert(0, mfql_root)
    from agents import agents as agent_registry
    from absl import flags as absl_flags

    FLAGS = absl_flags.FLAGS
    for name, default in [('offline_steps', 1000000), ('online_steps', 0)]:
        if name not in FLAGS:
            absl_flags.DEFINE_integer(name, default, '')
    for name, default in [('pretrain_factor', 0.0)]:
        if name not in FLAGS:
            absl_flags.DEFINE_float(name, default, '')
    try:
        FLAGS(sys.argv[:1])
    except Exception:
        pass

    with open(os.path.join(MFQL_DIR, 'flags.json')) as f:
        flags_dict = json.load(f)
    config = flags_dict['agent']
    seed = flags_dict.get('seed', 0)

    FLAGS['offline_steps'].value = flags_dict.get('offline_steps', 1000000)
    FLAGS['online_steps'].value = flags_dict.get('online_steps', 0)
    FLAGS['pretrain_factor'].value = flags_dict.get('pretrain_factor', 0.0)

    stats = np.load(os.path.join(MFQL_DIR, 'obs_norm_stats.npz'))
    obs_mean, obs_std = stats['obs_mean'], stats['obs_std']
    obs_dim = len(obs_mean)

    dummy_obs = np.zeros((1, obs_dim), dtype=np.float32)
    dummy_act = np.zeros((1, 2), dtype=np.float32)
    agent = agent_registry[config['agent_name']].create(seed, dummy_obs, dummy_act, config)

    ckpt = latest_checkpoint(MFQL_DIR, 'params_*.pkl')
    with open(ckpt, 'rb') as f:
        load_dict = pickle.load(f)
    agent = flax.serialization.from_state_dict(agent, load_dict['agent'])
    print(f'  MeanFlowQL: loaded {os.path.basename(ckpt)}')

    rng = jax.random.PRNGKey(seed)
    use_obs_norm = flags_dict.get('use_observation_normalization', True)

    def get_action(obs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        obs_batch = obs[None, :] if obs.ndim == 1 else obs
        if use_obs_norm:
            obs_batch = (obs_batch - obs_mean) / obs_std
        action = agent.sample_actions(observations=obs_batch, seed=key)
        action = np.array(action)
        if action.ndim > 1 and action.shape[0] == 1:
            action = action[0]
        return np.clip(action, -1, 1)

    return get_action


# ── Video helpers ────────────────────────────────────────────────────────────

def add_text_overlay(frames, text, success):
    """Add algorithm name and outcome to top of each frame using PIL."""
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except OSError:
        font = ImageFont.load_default()
        small_font = font

    status = 'SUCCESS' if success else 'FAIL'
    status_color = (0, 200, 0) if success else (220, 50, 50)
    bar_h = 20
    status_w = 55
    result = []
    for frame in frames:
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        # Black bar at top
        draw.rectangle([(0, 0), (RENDER_SIZE, bar_h)], fill=(0, 0, 0))
        # Algorithm name (white)
        draw.text((6, 4), text, fill=(255, 255, 255), font=font)
        # Success/fail on right
        draw.text((RENDER_SIZE - status_w, 4), status, fill=status_color, font=small_font)
        result.append(np.array(img))
    return result


def stitch_episode_video(episode_frames, output_path):
    """Stitch frames from 3 algorithms into one video with black gaps."""
    all_frames = []
    for frames in episode_frames:
        all_frames.extend(frames)
        # Brief black gap between algorithms
        if frames:
            all_frames.extend([np.zeros_like(frames[0])] * 8)
    if all_frames:
        imageio.mimsave(output_path, all_frames, fps=VIDEO_FPS)


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_trajectories(all_results):
    """Plot 2D trajectories with flow field background."""
    cfg = load_config(None)
    dyn_cfg, static_path = get_flow_runtime(cfg)
    if dyn_cfg.get('enabled', False):
        mode = dyn_cfg['mode']
        if mode == 'netcdf':
            flow = DynamicNetCDFFlowField({**dyn_cfg.get('netcdf', {}),
                                            'x_range': dyn_cfg['x_range'],
                                            'y_range': dyn_cfg['y_range']})
        else:
            flow = DynamicTGVFlowField(dyn_cfg)
        flow_t = 0.0
    else:
        flow = FlowField(os.path.join(_repo_root, static_path))
        flow_t = None
    maze_unit, offset = 4.0, 4.0

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, algo in zip(axes, ALGO_ORDER):
        # Boundary walls
        for i in range(8):
            for j in range(8):
                if i == 0 or i == 7 or j == 0 or j == 7:
                    x = j * maze_unit - offset - maze_unit / 2
                    y = i * maze_unit - offset - maze_unit / 2
                    ax.add_patch(plt.Rectangle((x, y), maze_unit, maze_unit,
                                               facecolor='#cccccc', edgecolor='#999999',
                                               linewidth=0.5))
        # Flow field (snapshot at t=0 for dynamic flows).
        xs = np.linspace(-2, 26, 25)
        ys = np.linspace(-2, 26, 25)
        if flow_t is None:
            vx, vy = flow.get_flow_grid(xs, ys)
        else:
            vx, vy = flow.get_flow_grid(xs, ys, t=flow_t)
        xx, yy = np.meshgrid(xs, ys)
        ax.quiver(xx, yy, vx, vy, alpha=0.12, color='gray', scale=30)

        # Trajectories
        for ep in all_results[algo]:
            traj = np.array(ep['trajectory'])
            if len(traj) == 0:
                continue
            color = ALGO_COLORS[algo] if ep['success'] > 0.5 else '#d62728'
            ls = '-' if ep['success'] > 0.5 else '--'
            ax.plot(traj[:, 0], traj[:, 1], color=color, alpha=0.3, linewidth=0.8, linestyle=ls)
            ax.plot(traj[0, 0], traj[0, 1], 'o', color=color, markersize=3, alpha=0.5)
            ax.plot(traj[-1, 0], traj[-1, 1], '*', color=color, markersize=4, alpha=0.5)

        sr = np.mean([e['success'] for e in all_results[algo]])
        mr = np.mean([e['return'] for e in all_results[algo]])
        ax.set_title(f'{algo}  (success={sr:.0%}, return={mr:.1f})')
        ax.set_xlim(-4, 24)
        ax.set_ylim(-4, 24)
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    fig.suptitle('Evaluated Trajectories (same start/goal per episode)', fontsize=14, y=1.02)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, 'trajectories.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


def animate_trajectories(all_results, fps=30, env_dt=0.1):
    """Animate all 3 algorithms drawing trajectories simultaneously over an
    evolving flow field. Output: one MP4 with three side-by-side panels.

    Skipped for netcdf flows: per-frame slice loads make this prohibitively
    slow at full eval scale.
    """
    cfg = load_config(None)
    dyn_cfg, static_path = get_flow_runtime(cfg)
    if dyn_cfg.get('enabled', False) and dyn_cfg.get('mode') == 'netcdf':
        print('  Skipping animate_trajectories (netcdf flow).')
        return
    if dyn_cfg.get('enabled', False):
        flow = DynamicTGVFlowField(dyn_cfg)
        flow_is_dynamic = True
    else:
        flow = FlowField(os.path.join(_repo_root, static_path))
        flow_is_dynamic = False

    # Pre-compute trajectories as fixed-length arrays. After an episode ends,
    # its position stays pinned at the final point (so the "pen" stops moving
    # but the line stays on the page).
    max_len = max(
        max((len(ep['trajectory']) for ep in all_results[a]), default=0)
        for a in ALGO_ORDER
    )
    if max_len == 0:
        print('  No trajectories to animate.')
        return

    padded = {a: [] for a in ALGO_ORDER}
    for algo in ALGO_ORDER:
        for ep in all_results[algo]:
            traj = np.asarray(ep['trajectory'], dtype=np.float32)
            if len(traj) == 0:
                continue
            pad = np.broadcast_to(traj[-1], (max_len - len(traj), 2))
            full = np.concatenate([traj, pad], axis=0) if len(pad) else traj
            padded[algo].append({'xy': full, 'real_len': len(traj),
                                 'success': ep['success']})

    maze_unit, offset = 4.0, 4.0
    xs = np.linspace(-2, 26, 25)
    ys = np.linspace(-2, 26, 25)
    xx, yy = np.meshgrid(xs, ys)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    quivers = []
    line_artists = []   # list of dict per algo: {'lines': [...], 'heads': [...]}
    for ax, algo in zip(axes, ALGO_ORDER):
        # Boundary walls
        for i in range(8):
            for j in range(8):
                if i == 0 or i == 7 or j == 0 or j == 7:
                    x = j * maze_unit - offset - maze_unit / 2
                    y = i * maze_unit - offset - maze_unit / 2
                    ax.add_patch(plt.Rectangle((x, y), maze_unit, maze_unit,
                                               facecolor='#cccccc', edgecolor='#999999',
                                               linewidth=0.5))

        # Initial flow snapshot at t=0 (gets updated each frame).
        if flow_is_dynamic:
            vx, vy = flow.get_flow_grid(xs, ys, t=0.0)
        else:
            vx, vy = flow.get_flow_grid(xs, ys)
        q = ax.quiver(xx, yy, vx, vy, alpha=0.18, color='gray', scale=30)
        quivers.append(q)

        lines, heads = [], []
        for ep in padded[algo]:
            color = ALGO_COLORS[algo] if ep['success'] > 0.5 else '#d62728'
            ls = '-' if ep['success'] > 0.5 else '--'
            ln, = ax.plot([], [], color=color, alpha=0.45, linewidth=1.0, linestyle=ls)
            head, = ax.plot([], [], 'o', color=color, markersize=4, alpha=0.85)
            # Plant a marker at the start so you see the "pens" before they move.
            ax.plot(ep['xy'][0, 0], ep['xy'][0, 1], 'o',
                    color=color, markersize=3, alpha=0.4)
            lines.append(ln)
            heads.append(head)
        line_artists.append({'lines': lines, 'heads': heads})

        sr = np.mean([e['success'] for e in all_results[algo]])
        mr = np.mean([e['return'] for e in all_results[algo]])
        ax.set_title(f'{algo}  (success={sr:.0%}, return={mr:.1f})')
        ax.set_xlim(-4, 24)
        ax.set_ylim(-4, 24)
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    fig.suptitle('Trajectories drawing in real time', fontsize=14, y=1.02)
    fig.tight_layout()

    path = os.path.join(RESULTS_DIR, 'trajectories.mp4')
    print(f'  Animating {max_len} frames -> {path}')
    with imageio.get_writer(path, fps=fps, macro_block_size=1) as writer:
        for f in tqdm(range(max_len), desc='  frames'):
            t = f * env_dt
            if flow_is_dynamic:
                vx, vy = flow.get_flow_grid(xs, ys, t=t)
                for q in quivers:
                    q.set_UVC(vx, vy)
            for algo_idx, algo in enumerate(ALGO_ORDER):
                arts = line_artists[algo_idx]
                for ep, ln, head in zip(padded[algo], arts['lines'], arts['heads']):
                    upto = min(f + 1, ep['real_len'])
                    xy = ep['xy'][:upto]
                    ln.set_data(xy[:, 0], xy[:, 1])
                    head.set_data([ep['xy'][min(f, ep['real_len'] - 1), 0]],
                                  [ep['xy'][min(f, ep['real_len'] - 1), 1]])
            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
            writer.append_data(frame)

    plt.close(fig)
    print(f'  Saved {path}')


def plot_comparison(all_results):
    """Bar chart of success rate, return, length, mean action magnitude."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    metrics = [('success', 'Success Rate'), ('return', 'Episode Return'),
               ('length', 'Episode Length'),
               ('mean_action_mag', 'Mean Action Magnitude\n(sum ||a|| / steps)')]
    ds_metrics = _dataset_episode_metrics()
    labels = list(ALGO_ORDER) + (['Offline'] if ds_metrics is not None else [])
    colors = [ALGO_COLORS[a] for a in ALGO_ORDER] + (
        ['#7f7f7f'] if ds_metrics is not None else []
    )
    for ax, (key, ylabel) in zip(axes, metrics):
        means, stds = [], []
        for algo in ALGO_ORDER:
            if key == 'mean_action_mag':
                vals = [e['action_effort'] / e['length']
                        for e in all_results[algo]
                        if 'action_effort' in e and e.get('length', 0) > 0]
            else:
                vals = [e[key] for e in all_results[algo] if key in e]
            if not vals:
                means.append(0.0); stds.append(0.0)
                continue
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        if ds_metrics is not None:
            ds_vals = ds_metrics[key]
            means.append(float(ds_vals.mean()))
            stds.append(float(ds_vals.std()))
        x = np.arange(len(labels))
        bars = ax.bar(x, means, yerr=stds, capsize=5,
                      color=colors, alpha=0.85)
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{m:.2f}', ha='center', va='bottom', fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha='right')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3, axis='y')
    fig.suptitle('Final Policy Performance vs Offline Dataset', fontsize=14)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, 'comparison.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


def _dataset_episode_metrics():
    """Per-episode return / success / length / mean action magnitude from the
    offline dataset. Returns None if the dataset isn't present.
    Success := terminal step's dist_to_goal is within goal_tolerance.
    """
    cfg = load_config(None)
    dataset_path = os.path.join(_repo_root, cfg['run']['save_path'])
    if not os.path.exists(dataset_path):
        return None
    data = np.load(dataset_path)
    rewards = data['rewards'].astype(np.float64)
    terminals = data['terminals'].astype(np.float32)
    ends = np.where(terminals > 0.5)[0]
    if len(ends) == 0:
        return None
    starts = np.concatenate([[0], ends[:-1] + 1])
    goal_tol = float(cfg['env']['goal_tolerance'])

    actions = data['actions'].astype(np.float32)
    action_mag = np.linalg.norm(actions, axis=1)

    returns = np.array([rewards[s:e + 1].sum() for s, e in zip(starts, ends)])
    lengths = np.array([e - s + 1 for s, e in zip(starts, ends)], dtype=np.int64)
    mean_act = np.array(
        [action_mag[s:e + 1].mean() for s, e in zip(starts, ends)]
    )
    successes = (data['dist_to_goal'][ends] <= goal_tol).astype(np.float32)
    return {
        'return': returns,
        'success': successes,
        'length': lengths.astype(np.float64),
        'mean_action_mag': mean_act,
    }


def _dataset_episode_returns():
    """Backward-compatible wrapper used by plot_reward_histogram."""
    m = _dataset_episode_metrics()
    if m is None:
        return None, None
    return m['return'], m['success']


def plot_reward_histogram(all_results):
    """Stacked per-algorithm histograms of episode returns plus the offline
    dataset distribution (shared x-axis)."""
    dataset_returns, dataset_successes = _dataset_episode_returns()

    all_returns = [e['return'] for algo in ALGO_ORDER for e in all_results[algo]]
    if dataset_returns is not None:
        all_returns = list(all_returns) + dataset_returns.tolist()
    if not all_returns:
        return
    lo = max(-300.0, float(np.min(all_returns)))
    hi = float(np.max(all_returns))
    if hi <= lo:
        hi = lo + 1.0
    bins = np.linspace(lo, hi, 51)

    panels = [(algo, [e['return'] for e in all_results[algo]],
               ALGO_COLORS[algo],
               f'{algo}  (mean={np.mean([e["return"] for e in all_results[algo]]):.1f}, '
               f'success={np.mean([e["success"] for e in all_results[algo]]):.0%})')
              for algo in ALGO_ORDER]
    if dataset_returns is not None:
        ds_success = float(dataset_successes.mean()) if dataset_successes is not None else None
        title = (f'Offline dataset  (n={len(dataset_returns)}, '
                 f'mean={dataset_returns.mean():.1f}')
        if ds_success is not None:
            title += f', success={ds_success:.0%}'
        title += ')'
        panels.append(('Offline dataset', dataset_returns, '#7f7f7f', title))

    fig, axes = plt.subplots(len(panels), 1, figsize=(8, 2.4 * len(panels)),
                             sharex=True)
    for ax, (_, rets, color, title) in zip(axes, panels):
        # Clip values to the visible range so episodes below the cutoff pile
        # into the leftmost bin instead of being silently dropped.
        rets_clipped = np.clip(rets, lo, hi)
        ax.hist(rets_clipped, bins=bins, color=color, alpha=0.85,
                edgecolor='black', linewidth=0.4)
        ax.set_xlim(lo, hi)
        # Each panel has its own y-axis since the dataset has ~100x more
        # episodes than the eval set; a shared y would flatten the eval rows.
        ax.set_ylabel('Episodes')
        ax.set_title(title, loc='left', fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
    axes[-1].set_xlabel('Episode return')
    fig.suptitle('Distribution of episode returns', fontsize=13)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, 'reward_histogram.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=NUM_EVAL_EPISODES)
    parser.add_argument('--num-videos', type=int, default=NUM_VIDEO_EPISODES,
                        help='Render videos only for the first N episodes; '
                             'remaining episodes run headless. Set 0 to disable.')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--no-video', action='store_true')
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device(args.device)
    num_videos = 0 if args.no_video else min(args.num_videos, args.episodes)
    t0 = time.time()

    # Load all models once
    print('\nLoading models...')
    bc_policy, bc_obs_mean, bc_obs_std = load_bc(device)
    dt_model, dt_obs_mean, dt_obs_std, dt_flags, target_return = load_dt(device)
    mfql_get_action = load_mfql()

    # Create envs: a rendering one for the first `num_videos` episodes, and a
    # headless one for the rest (skips MuJoCo render calls — much faster).
    render_env = make_eval_env(render_mode='rgb_array') if num_videos > 0 else None
    headless_env = make_eval_env(render_mode=None) if num_videos < args.episodes else None
    free_cells = get_free_cells()
    rng = np.random.default_rng(42)

    if num_videos > 0:
        video_dir = os.path.join(RESULTS_DIR, 'videos')
        os.makedirs(video_dir, exist_ok=True)

    # Per-algorithm results
    all_results = {algo: [] for algo in ALGO_ORDER}
    policy_time_totals = {algo: 0.0 for algo in ALGO_ORDER}

    print(f'\nRunning {args.episodes} episodes (same start/goal for all 3 algos); '
          f'rendering videos for first {num_videos}.\n')
    for ep_idx in tqdm(range(args.episodes), desc='Episodes'):
        # Sample start/goal for this episode
        init_ij, goal_ij = sample_start_goal(free_cells, rng)
        reset_opts = {'task_info': {'init_ij': init_ij, 'goal_ij': goal_ij}}

        record_this = ep_idx < num_videos
        env = render_env if record_this else headless_env

        # Run all 3 policies on the same start/goal
        t_bc = time.time()
        bc_ep = run_bc_episode(bc_policy, bc_obs_mean, bc_obs_std, env, device,
                               reset_options=reset_opts, record_video=record_this)
        policy_time_totals['BC'] += time.time() - t_bc

        t_dt = time.time()
        dt_ep = run_dt_episode(dt_model, dt_obs_mean, dt_obs_std, dt_flags, env,
                               target_return, device, reset_options=reset_opts,
                               record_video=record_this)
        policy_time_totals['DT'] += time.time() - t_dt

        t_mf = time.time()
        mfql_ep = run_mfql_episode(mfql_get_action, env, reset_options=reset_opts,
                                   record_video=record_this)
        policy_time_totals['MeanFlowQL'] += time.time() - t_mf

        if ep_idx < 5 or (ep_idx + 1) % 25 == 0:
            tqdm.write(
                f'  ep {ep_idx + 1}: '
                f'BC={policy_time_totals["BC"] / (ep_idx + 1):.2f}s/ep  '
                f'DT={policy_time_totals["DT"] / (ep_idx + 1):.2f}s/ep  '
                f'MFQL={policy_time_totals["MeanFlowQL"] / (ep_idx + 1):.2f}s/ep  '
                f'(this ep lengths: BC={bc_ep["length"]} DT={dt_ep["length"]} MFQL={mfql_ep["length"]})',
                file=sys.stdout,
            )
            sys.stdout.flush()

        # Store results
        for algo, ep in zip(ALGO_ORDER, [bc_ep, dt_ep, mfql_ep]):
            all_results[algo].append({k: v for k, v in ep.items() if k != 'frames'})

        # Stitch video for this episode
        if record_this:
            labeled_frames = []
            for algo, ep in zip(ALGO_ORDER, [bc_ep, dt_ep, mfql_ep]):
                label = f'{algo} (ret={ep["return"]:.1f})'
                labeled_frames.append(
                    add_text_overlay(ep['frames'], label, ep['success'] > 0.5))
            video_path = os.path.join(video_dir, f'episode_{ep_idx + 1:02d}.mp4')
            stitch_episode_video(labeled_frames, video_path)

    if render_env is not None:
        render_env.close()
    if headless_env is not None:
        headless_env.close()

    print('\nPer-policy total time:')
    for algo in ALGO_ORDER:
        total = policy_time_totals[algo]
        print(f'  {algo:>12}: {total:7.1f}s  ({total / args.episodes:.2f}s/ep)')

    # ── Print results ──
    print('\n' + '=' * 60)
    print(f'RESULTS ({args.episodes} episodes, same start/goal per episode)')
    print('=' * 60)
    print(f'{"":>12} {"Success":>10} {"Return":>15} {"Length":>15}')
    print('-' * 55)
    for algo in ALGO_ORDER:
        eps = all_results[algo]
        sr = np.mean([e['success'] for e in eps])
        ret = np.mean([e['return'] for e in eps])
        ret_std = np.std([e['return'] for e in eps])
        length = np.mean([e['length'] for e in eps])
        print(f'{algo:>12} {sr:>9.0%} {ret:>8.1f} +/- {ret_std:<5.1f} {length:>8.0f}')
    print('=' * 60)

    # ── Plots ──
    print('\nGenerating plots...')
    plot_trajectories(all_results)
    animate_trajectories(all_results)
    plot_comparison(all_results)
    plot_reward_histogram(all_results)

    # Save results JSON
    results_path = os.path.join(RESULTS_DIR, 'eval_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f)
    print(f'  Saved {results_path}')

    if num_videos > 0:
        print(f'  Videos saved to {os.path.join(RESULTS_DIR, "videos")}')

    print(f'\nDone in {time.time() - t0:.0f}s. All outputs in: {RESULTS_DIR}')


if __name__ == '__main__':
    main()
