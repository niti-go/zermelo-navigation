"""Shared helpers for offline dataset generators.

Used by `generate_dataset.py` (diverse-trajectory generator) and
`generate_straight_dataset.py` (straight-line oracle). Both scripts have
the same skeleton:

    for episode i in [0, num_episodes):
        env.reset(start_frame = schedule(i))
        while not done:
            action = <policy-specific>(env state, target)
            env.step(action)
            accumulate per-step components

The skeleton lives here; the two scripts supply only their action policy
and per-step bookkeeping.
"""
from collections import defaultdict, deque

import gymnasium
import numpy as np

import zermelo_env  # noqa — registers gymnasium envs
from zermelo_env.zermelo_config import config_to_env_kwargs


# ---------------------------------------------------------------------------
# Maze BFS helpers (only invoked when cfg['maze']['enabled'] is true)
# ---------------------------------------------------------------------------

def bfs_distance_map(maze_map, start_ij):
    """BFS distance map from start_ij. Unreachable cells = -1."""
    dist = np.full_like(maze_map, -1)
    dist[start_ij[0], start_ij[1]] = 0
    queue = deque([start_ij])
    while queue:
        i, j = queue.popleft()
        for di, dj in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
            ni, nj = i + di, j + dj
            if (0 <= ni < maze_map.shape[0] and 0 <= nj < maze_map.shape[1]
                    and maze_map[ni, nj] == 0 and dist[ni, nj] == -1):
                dist[ni, nj] = dist[i, j] + 1
                queue.append((ni, nj))
    return dist


def precompute_bfs_cache(maze_map, all_cells):
    return {cell: bfs_distance_map(maze_map, cell) for cell in all_cells}


def oracle_subgoal(agent_xy, target_ij, bfs_cache, maze_map, xy_to_ij, ij_to_xy):
    """Pick the adjacent free cell that minimizes BFS distance to target_ij."""
    agent_ij = xy_to_ij(agent_xy)
    bfs_map = bfs_cache[target_ij]
    best_ij = agent_ij
    rows, cols = maze_map.shape
    for di, dj in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
        ni, nj = agent_ij[0] + di, agent_ij[1] + dj
        if (0 <= ni < rows and 0 <= nj < cols
                and maze_map[ni, nj] == 0
                and bfs_map[ni, nj] < bfs_map[best_ij[0], best_ij[1]]):
            best_ij = (ni, nj)
    return np.array(ij_to_xy(best_ij))


# ---------------------------------------------------------------------------
# Flow-frame scheduling
# ---------------------------------------------------------------------------

def start_frame_upper(env, cfg):
    """Largest valid start_frame keeping a full episode inside the train segment."""
    fps = float(env.unwrapped.frames_per_step)
    max_steps = int(cfg['run']['max_episode_steps'])
    n_train = int(env.unwrapped.n_frames)
    return max(0.0, float(n_train) - max_steps * fps)


def deterministic_start_frame(global_idx, num_episodes, upper):
    """Evenly spaced start frame for `global_idx` in [0, num_episodes).

    Maps episode index linearly onto [0, upper] so the population of starts
    uniformly covers the train segment.
    """
    if num_episodes <= 1 or upper <= 0:
        return 0.0
    return float(global_idx) * upper / float(num_episodes - 1)


def sample_start_frame(env, cfg, global_idx, num_episodes):
    """Per-episode flow start frame within the train segment.

    Mode selected by `flow.start_frame_mode`:
      - 'deterministic_spread' (default): uniform sweep across the train
        segment using the global episode index.
      - 'random': i.i.d. uniform in the valid range.
    """
    upper = start_frame_upper(env, cfg)
    mode = cfg['flow'].get('start_frame_mode', 'deterministic_spread')
    if mode == 'random':
        if upper <= 0:
            return 0.0
        return float(np.random.uniform(0.0, upper))
    if mode == 'deterministic_spread':
        return deterministic_start_frame(global_idx, num_episodes, upper)
    raise ValueError(f"Unknown flow.start_frame_mode={mode!r}; "
                     f"expected 'deterministic_spread' or 'random'.")


def worker_frame_range(cfg, start_idx, n_episodes, num_episodes_total,
                       n_train_frames):
    """Tightest [frame_lo, frame_hi) the worker will ever access.

    Used to prewarm only the slab of HIT pages this worker will touch.
    Falls back to the full range under `start_frame_mode='random'`.
    """
    mode = cfg['flow'].get('start_frame_mode', 'deterministic_spread')
    fps = float(cfg['flow'].get('frames_per_step', 1.0))
    max_steps = int(cfg['run']['max_episode_steps'])
    span = max_steps * fps
    upper = max(0.0, float(n_train_frames) - span)
    if mode != 'deterministic_spread' or num_episodes_total <= 1 or upper <= 0:
        return (0.0, float(n_train_frames))
    last_idx = min(num_episodes_total - 1, start_idx + n_episodes - 1)
    f_lo = float(start_idx) * upper / float(num_episodes_total - 1)
    f_hi = float(last_idx) * upper / float(num_episodes_total - 1) + span
    return (f_lo, min(float(n_train_frames), f_hi))


# ---------------------------------------------------------------------------
# Env + maze caches
# ---------------------------------------------------------------------------

def build_env_and_caches(cfg, worker_id=0, frame_range=None,
                         need_xy_bounds=False):
    """Open env, prewarm flow pages, build maze BFS cache (if maze on).

    Returns a dict with keys: env, maze_map, all_cells, bfs_cache, xy_bounds.
    `xy_bounds` is only set when `need_xy_bounds=True` and maze is disabled
    (the diverse generator uses it for continuous-XY waypoint sampling).
    """
    run_cfg = cfg['run']
    train_max = cfg['flow'].get('train_max_file')
    env_kwargs = config_to_env_kwargs(cfg, max_file=train_max)
    env_kwargs['max_episode_steps'] = run_cfg['max_episode_steps']

    if worker_id == 0:
        print(f'  [worker 0] Opening HIT flow files via local-SSD cache '
              f'(HIT1..HIT{train_max})...', flush=True)
    env = gymnasium.make('zermelo-pointmaze-medium-v0', **env_kwargs)
    if worker_id == 0:
        print(f'  [worker 0] Flow field ready ({env.unwrapped.n_frames} frames, '
              f'frames_per_step={env.unwrapped.frames_per_step}).', flush=True)

    if frame_range is not None:
        f_lo, f_hi = frame_range
        if worker_id == 0:
            print(f'  [worker 0] Prewarming frames [{f_lo:.0f}, {f_hi:.0f}) '
                  f'(scoped to this worker\'s episodes)...', flush=True)
        env.unwrapped._flow_field.prewarm_range(
            f_lo, f_hi, verbose=(worker_id == 0))
    else:
        if worker_id == 0:
            print(f'  [worker 0] Prewarming full chain '
                  f'(no frame range supplied)...', flush=True)
        env.unwrapped._flow_field.prewarm(verbose=(worker_id == 0))

    maze_map = env.unwrapped.maze_map
    all_cells = [
        (i, j)
        for i in range(maze_map.shape[0])
        for j in range(maze_map.shape[1])
        if maze_map[i, j] == 0
    ]

    maze_enabled = cfg['maze']['enabled']
    if maze_enabled:
        if worker_id == 0:
            print(f'  [worker 0] Precomputing BFS cache for {len(all_cells)} cells...',
                  flush=True)
        bfs_cache = precompute_bfs_cache(maze_map, all_cells)
    else:
        bfs_cache = None

    xy_bounds = None
    if need_xy_bounds and not maze_enabled:
        corner_min = np.array(env.unwrapped.ij_to_xy((1, 1)))
        corner_max = np.array(env.unwrapped.ij_to_xy((6, 6)))
        xy_bounds = (min(corner_min[0], corner_max[0]),
                     max(corner_min[0], corner_max[0]),
                     min(corner_min[1], corner_max[1]),
                     max(corner_min[1], corner_max[1]))

    if worker_id == 0:
        print('  [worker 0] Setup complete; starting episode rollout.', flush=True)
    return dict(env=env, maze_map=maze_map, all_cells=all_cells,
                bfs_cache=bfs_cache, xy_bounds=xy_bounds)


# ---------------------------------------------------------------------------
# Save / stats
# ---------------------------------------------------------------------------

def new_ep_data():
    """Default-dict the generators use to accumulate per-step arrays."""
    return defaultdict(list)


def save_dataset(dataset, save_path):
    """Stack the per-step lists into typed arrays and save as compressed .npz."""
    import pathlib
    pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    out = {}
    for k, v in dataset.items():
        if 'observations' in k and v[0].dtype == np.uint8:
            dtype = np.uint8
        else:
            dtype = np.float32
        out[k] = np.array(v, dtype=dtype)
    np.savez_compressed(save_path, **out)
    total_steps = len(next(iter(out.values()))) if out else 0
    print(f'Saved dataset ({total_steps} steps) to {save_path}')
    return total_steps


def print_stats(stats_list, reward_cfg, header):
    """Pretty-print per-episode statistics and the reward breakdown."""
    ep_lengths = np.array([s['length'] for s in stats_list])
    ep_action_norms = np.array([s['action_norm'] for s in stats_list])
    ep_successes = np.array([s['success'] for s in stats_list])
    ep_goal_rewards = np.array([s['goal_r'] for s in stats_list])
    ep_energy_costs = np.array([s['energy_r'] for s in stats_list])
    ep_progress_rewards = np.array([s['progress_r'] for s in stats_list])
    ep_dist_sums = np.array([s['dist_sum'] for s in stats_list])
    total_steps = int(ep_lengths.sum())
    ep_total = ep_goal_rewards + ep_energy_costs + ep_progress_rewards

    energy_cfg = reward_cfg['energy']
    print(f'\n=== {header} ===')
    print(f'Total steps: {total_steps}')
    print(f'Episodes: {len(ep_lengths)}  |  Success rate: {ep_successes.mean():.1%}')
    print(f'Episode length   — mean: {ep_lengths.mean():.0f}  '
          f'median: {np.median(ep_lengths):.0f}  '
          f'min: {ep_lengths.min()}  max: {ep_lengths.max()}')
    print(f'Action norm/step — mean: {ep_action_norms.mean():.3f}  '
          f'min: {ep_action_norms.min():.3f}  max: {ep_action_norms.max():.3f}')
    print(f'Dist-to-goal sum — mean: {ep_dist_sums.mean():.1f}  '
          f'min: {ep_dist_sums.min():.1f}  max: {ep_dist_sums.max():.1f}')

    print(f'\n--- Reward breakdown (goal={reward_cfg["goal_reward"]}, '
          f'action_weight={energy_cfg["action_weight"]}, '
          f'fixed_hover_cost={energy_cfg["fixed_hover_cost"]}, '
          f'progress={reward_cfg["progress_weight"]}) ---')
    abs_total = abs(ep_total).mean() + 1e-9
    print(f'  Total reward    — mean: {ep_total.mean():.3f}  '
          f'min: {ep_total.min():.3f}  max: {ep_total.max():.3f}')
    print(f'  Goal portion    — mean: {ep_goal_rewards.mean():.3f}  '
          f'(% of |total|: {100*abs(ep_goal_rewards.mean()) / abs_total:.0f}%)')
    print(f'  Energy cost     — mean: {ep_energy_costs.mean():.3f}  '
          f'(action + hover; % of |total|: '
          f'{100*abs(ep_energy_costs.mean()) / abs_total:.0f}%)')
    print(f'  Progress reward — mean: {ep_progress_rewards.mean():.3f}  '
          f'(% of |total|: {100*abs(ep_progress_rewards.mean()) / abs_total:.0f}%)')

    return dict(total_steps=total_steps, ep_lengths=ep_lengths,
                ep_action_norms=ep_action_norms)


# ---------------------------------------------------------------------------
# Worker scaffolding
# ---------------------------------------------------------------------------

def plan_worker_slices(num_episodes, n_workers, base_seed):
    """Split `num_episodes` across `n_workers` and assign each a slice + seed.

    Returns a list of (start_idx, n_episodes, seed, worker_id) tuples covering
    [0, num_episodes) with no overlap. Each worker's seed is base_seed-deterministic
    yet distinct across workers.
    """
    n_workers = min(n_workers, num_episodes)
    per_worker = [num_episodes // n_workers] * n_workers
    for i in range(num_episodes % n_workers):
        per_worker[i] += 1
    start_indices = np.cumsum([0] + per_worker[:-1]).tolist()
    return [
        (start_indices[wid], per_worker[wid], base_seed + 1000 * wid, wid)
        for wid in range(n_workers)
    ], n_workers
