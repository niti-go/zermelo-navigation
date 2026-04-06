"""Evaluate final policies of BC, DT, and MeanFlowQL on Zermelo.

For each episode, samples a random start/goal, then runs all 3 policies on
that same start/goal. Produces one comparison video per episode (3 clips
stitched together with algorithm labels), trajectory plots, and metrics.

Usage:
    conda activate flowrl
    cd ~/zermelo-navigation
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=4 python scripts/evaluate_models.py
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
from zermelo_env.zermelo_config import load_config, config_to_env_kwargs
from zermelo_env.zermelo_flow import FlowField

# ── Config ───────────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(_repo_root, 'results', 'zermelo_offline_results')
NUM_EVAL_EPISODES = 20
VIDEO_FPS = 30
RENDER_SIZE = 400

BC_DIR = os.path.join(_repo_root, 'exp', 'zermelo', 'bc', 'bc_sd000_20260403_140851')
DT_DIR = os.path.join(_repo_root, 'exp', 'zermelo', 'dt', 'dt_sd000_20260403_141030')
MFQL_DIR = os.path.join(_repo_root, 'exp', 'zermelo', 'meanflowql', 'sd000_20260403_143654')

ALGO_COLORS = {'BC': '#1f77b4', 'DT': '#ff7f0e', 'MeanFlowQL': '#2ca02c'}
ALGO_ORDER = ['BC', 'DT', 'MeanFlowQL']


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
    env_kwargs['max_episode_steps'] = cfg['dataset']['max_episode_steps']
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

def run_bc_episode(policy, obs_mean, obs_std, env, device, reset_options=None):
    ob, _ = env.reset(options=reset_options)
    done, ep_ret, frames, traj = False, 0.0, [], []
    while not done:
        traj.append(ob[:2].tolist())
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        ob_t = torch.tensor(
            (ob - obs_mean) / obs_std, dtype=torch.float32, device=device
        ).unsqueeze(0)
        with torch.no_grad():
            action = policy(ob_t).cpu().numpy()[0]
        ob, reward, terminated, truncated, info = env.step(np.clip(action, -1, 1))
        done = terminated or truncated
        ep_ret += reward
    return {'return': ep_ret, 'length': len(traj), 'success': info.get('success', 0.0),
            'trajectory': traj, 'frames': frames}


def run_dt_episode(model, obs_mean, obs_std, flags, env, target_return, device,
                   reset_options=None):
    context_len = flags['context_len']
    max_ep_len = flags['max_ep_len']
    rtg_scale = flags['rtg_scale']
    obs_dim = len(obs_mean)
    act_dim = 2

    ob, _ = env.reset(options=reset_options)
    done, ep_ret, ep_len = False, 0.0, 0
    frames, traj = [], []
    states_buf = np.zeros((context_len, obs_dim), dtype=np.float32)
    actions_buf = np.zeros((context_len, act_dim), dtype=np.float32)
    rtgs_buf = np.zeros((context_len, 1), dtype=np.float32)
    timesteps_buf = np.zeros(context_len, dtype=np.int64)
    remaining_return = target_return

    while not done:
        traj.append(ob[:2].tolist())
        frame = env.render()
        if frame is not None:
            frames.append(frame)
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

        ob, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        ep_ret += reward
        remaining_return -= reward
        ep_len += 1
    return {'return': ep_ret, 'length': ep_len, 'success': info.get('success', 0.0),
            'trajectory': traj, 'frames': frames}


def run_mfql_episode(get_action, env, reset_options=None):
    ob, _ = env.reset(options=reset_options)
    done, ep_ret = False, 0.0
    frames, traj = [], []
    while not done:
        traj.append(ob[:2].tolist())
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        ob, reward, terminated, truncated, info = env.step(get_action(ob))
        done = terminated or truncated
        ep_ret += reward
    return {'return': ep_ret, 'length': len(traj), 'success': info.get('success', 0.0),
            'trajectory': traj, 'frames': frames}


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
    dataset_path = os.path.join(_repo_root, cfg['dataset']['save_path'])
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
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
        small_font = font

    status = 'SUCCESS' if success else 'FAIL'
    status_color = (0, 200, 0) if success else (220, 50, 50)
    result = []
    for frame in frames:
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        # Black bar at top
        draw.rectangle([(0, 0), (RENDER_SIZE, 32)], fill=(0, 0, 0))
        # Algorithm name (white)
        draw.text((8, 6), text, fill=(255, 255, 255), font=font)
        # Success/fail on right
        draw.text((RENDER_SIZE - 90, 7), status, fill=status_color, font=small_font)
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
    flow_path = os.path.join(_repo_root, cfg['flow']['field_path'])
    flow = FlowField(flow_path)
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
        # Flow field
        xs = np.linspace(-2, 26, 25)
        ys = np.linspace(-2, 26, 25)
        vx, vy = flow.get_flow_grid(xs, ys)
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


def plot_comparison(all_results):
    """Bar chart of success rate, return, length."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    metrics = [('success', 'Success Rate'), ('return', 'Episode Return'),
               ('length', 'Episode Length')]
    for ax, (key, ylabel) in zip(axes, metrics):
        means, stds = [], []
        for algo in ALGO_ORDER:
            vals = [e[key] for e in all_results[algo]]
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        x = np.arange(len(ALGO_ORDER))
        bars = ax.bar(x, means, yerr=stds, capsize=5,
                      color=[ALGO_COLORS[a] for a in ALGO_ORDER], alpha=0.85)
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{m:.2f}', ha='center', va='bottom', fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(ALGO_ORDER)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3, axis='y')
    fig.suptitle('Final Policy Performance (same start/goal per episode)', fontsize=14)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, 'comparison.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=NUM_EVAL_EPISODES)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--no-video', action='store_true')
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device(args.device)
    record_video = not args.no_video
    t0 = time.time()

    # Load all models once
    print('\nLoading models...')
    bc_policy, bc_obs_mean, bc_obs_std = load_bc(device)
    dt_model, dt_obs_mean, dt_obs_std, dt_flags, target_return = load_dt(device)
    mfql_get_action = load_mfql()

    # Create env
    env = make_eval_env(render_mode='rgb_array' if record_video else None)
    free_cells = get_free_cells()
    rng = np.random.default_rng(42)

    if record_video:
        video_dir = os.path.join(RESULTS_DIR, 'videos')
        os.makedirs(video_dir, exist_ok=True)

    # Per-algorithm results
    all_results = {algo: [] for algo in ALGO_ORDER}

    print(f'\nRunning {args.episodes} episodes (same start/goal for all 3 algos)...\n')
    for ep_idx in tqdm(range(args.episodes), desc='Episodes'):
        # Sample start/goal for this episode
        init_ij, goal_ij = sample_start_goal(free_cells, rng)
        reset_opts = {'task_info': {'init_ij': init_ij, 'goal_ij': goal_ij}}

        # Run all 3 policies on the same start/goal
        bc_ep = run_bc_episode(bc_policy, bc_obs_mean, bc_obs_std, env, device,
                               reset_options=reset_opts)
        dt_ep = run_dt_episode(dt_model, dt_obs_mean, dt_obs_std, dt_flags, env,
                               target_return, device, reset_options=reset_opts)
        mfql_ep = run_mfql_episode(mfql_get_action, env, reset_options=reset_opts)

        # Store results
        for algo, ep in zip(ALGO_ORDER, [bc_ep, dt_ep, mfql_ep]):
            all_results[algo].append({k: v for k, v in ep.items() if k != 'frames'})

        # Stitch video for this episode
        if record_video:
            labeled_frames = []
            for algo, ep in zip(ALGO_ORDER, [bc_ep, dt_ep, mfql_ep]):
                label = f'{algo} (ret={ep["return"]:.1f})'
                labeled_frames.append(
                    add_text_overlay(ep['frames'], label, ep['success'] > 0.5))
            video_path = os.path.join(video_dir, f'episode_{ep_idx + 1:02d}.mp4')
            stitch_episode_video(labeled_frames, video_path)

    env.close()

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
    plot_comparison(all_results)

    # Save results JSON
    results_path = os.path.join(RESULTS_DIR, 'eval_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f)
    print(f'  Saved {results_path}')

    if record_video:
        print(f'  Videos saved to {os.path.join(RESULTS_DIR, "videos")}')

    print(f'\nDone in {time.time() - t0:.0f}s. All outputs in: {RESULTS_DIR}')


if __name__ == '__main__':
    main()
