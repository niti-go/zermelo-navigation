"""Generate offline datasets of straight-line trajectories to the goal.

This agent is an oracle: at every step it reads the true flow vector at its
current position and commands the action that cancels the flow component
perpendicular to the goal-direction, producing a trajectory that travels
in a straight line toward the (sub)goal at the highest speed achievable.

Dynamics (see zermelo_env/zermelo_point.py):
    new_xy = xy + dt * (2 * action + flow)
We want the resulting motion to point along d = (goal - xy) / ||goal - xy||
at some non-negative speed s, i.e. (2 * action + flow) = s * d. So:
    action = (s * d - flow) / 2,  clipped to [-1, 1]^2.
We choose the largest s ≥ 0 such that the unclipped action stays in the
box; that's the fastest straight-line motion the agent can achieve at this
state. If the flow has a component along -d larger than the agent's max
push along +d, s saturates at 0 (or the action clips and the trajectory
deviates slightly).

In maze mode, the BFS oracle picks an in-corridor subgoal each step and
the agent steers straight at *that* subgoal — so the path is piecewise-
straight between cells rather than literally straight through walls.

Usage:
    python scripts/generate_straight_dataset.py
    python scripts/generate_straight_dataset.py --num_workers=16
"""
import multiprocessing as mp
import pathlib
from collections import defaultdict, deque

import gymnasium
import numpy as np
from absl import app, flags
from tqdm import trange

import zermelo_env  # noqa — registers gymnasium envs
from zermelo_env.zermelo_config import load_config, config_to_env_kwargs

FLAGS = flags.FLAGS


def _bfs_distance_map(maze_map, start_ij):
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


def _precompute_bfs_cache(maze_map, all_cells):
    return {cell: _bfs_distance_map(maze_map, cell) for cell in all_cells}


def _oracle_subgoal(agent_xy, target_ij, bfs_cache, maze_map, xy_to_ij, ij_to_xy):
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

flags.DEFINE_string('config', None, 'Path to zermelo_config.yaml (optional; uses built-in defaults if omitted).')
flags.DEFINE_integer('num_workers', 16,
                     'Number of parallel worker processes. 1 = run serially in-process.')


def _start_frame_upper(env, cfg):
    """Largest valid start_frame keeping the whole episode inside the train segment."""
    fps = float(env.unwrapped.frames_per_step)
    max_steps = int(cfg['run']['max_episode_steps'])
    n_train = int(env.unwrapped.n_frames)
    return max(0.0, float(n_train) - max_steps * fps)


def _deterministic_start_frame(global_idx, num_episodes, upper):
    """Evenly spaced start frame for ``global_idx`` in ``[0, num_episodes)``.

    Maps episode index linearly onto ``[0, upper]`` so the population of
    starts uniformly covers the train segment. With ``num_episodes`` larger
    than ``upper / episode_span`` adjacent episodes overlap, but coverage
    stays uniform.
    """
    if num_episodes <= 1 or upper <= 0:
        return 0.0
    return float(global_idx) * upper / float(num_episodes - 1)


def _sample_start_frame(env, cfg, global_idx, num_episodes):
    """Per-episode flow start frame within the train segment."""
    upper = _start_frame_upper(env, cfg)
    mode = cfg['flow'].get('start_frame_mode', 'deterministic_spread')
    if mode == 'random':
        if upper <= 0:
            return 0.0
        return float(np.random.uniform(0.0, upper))
    if mode == 'deterministic_spread':
        return _deterministic_start_frame(global_idx, num_episodes, upper)
    raise ValueError(f"Unknown flow.start_frame_mode={mode!r}; "
                     f"expected 'deterministic_spread' or 'random'.")


def _straight_action(direction, flow_xy, action_scale):
    """Return the action that produces motion along `direction` (unit vector).

    Solves (action_scale * action + flow) = s * direction for the largest
    s ≥ 0 keeping action ∈ [-1, 1]^2. If no s ≥ 0 satisfies the box (flow
    drags us away too hard), return the clipped action at s=0 (best-effort
    against flow).
    """
    fx, fy = float(flow_xy[0]), float(flow_xy[1])
    dx, dy = float(direction[0]), float(direction[1])
    a = float(action_scale)

    # action_i = (s * d_i - f_i) / a ∈ [-1, 1]  ⇔  s * d_i ∈ [f_i - a, f_i + a].
    # Solve each component for s assuming d_i > 0 / < 0; ignore axes with d_i ≈ 0.
    s_max = np.inf
    for d, f in ((dx, fx), (dy, fy)):
        if abs(d) < 1e-9:
            # No constraint from this axis on s, but action_i = -f/a must be in [-1, 1].
            # If |f| > a the action will clip; we accept that.
            continue
        if d > 0:
            s_max = min(s_max, (f + a) / d)
        else:
            s_max = min(s_max, (f - a) / d)

    s = max(0.0, s_max if np.isfinite(s_max) else 0.0)
    action = np.array([(s * dx - fx) / a, (s * dy - fy) / a])
    return np.clip(action, -1.0, 1.0)


def _run_one_episode(env, cfg, maze_map, all_cells, bfs_cache,
                     global_idx, num_episodes):
    """Run a single straight-line episode and return per-step data + summary stats."""
    maze_enabled = cfg['maze']['enabled']
    task_cfg = cfg['task']
    run_cfg = cfg['run']
    reward_cfg = cfg['reward']

    action_scale = float(cfg['env'].get('action_scale', 2.0))

    if maze_enabled:
        xy_to_ij = env.unwrapped.xy_to_ij
        ij_to_xy = env.unwrapped.ij_to_xy

    if task_cfg['start_goal_mode'] == 'fixed':
        init_ij = tuple(task_cfg['fixed_start_ij'])
        goal_ij = tuple(task_cfg['fixed_goal_ij'])
    else:
        init_ij = all_cells[np.random.randint(len(all_cells))]
        goal_ij = all_cells[np.random.randint(len(all_cells))]
        while goal_ij == init_ij:
            goal_ij = all_cells[np.random.randint(len(all_cells))]

    start_frame = _sample_start_frame(env, cfg, global_idx, num_episodes)
    ob, _ = env.reset(options=dict(
        task_info=dict(init_ij=init_ij, goal_ij=goal_ij),
        start_frame=start_frame,
    ))
    goal_xy = np.array(env.unwrapped.cur_goal_xy)

    # Initial distance for the progress-shaping term (Ng et al. 1999): the
    # per-step reward is k * (prev_dist - curr_dist), which telescopes to
    # k * (initial_dist - final_dist) over the episode — policy-invariant
    # in contrast to a raw -k*dist penalty.
    prev_dist = float(np.linalg.norm(env.unwrapped.get_xy() - goal_xy))

    ep_data = defaultdict(list)
    done = False
    step = 0
    ep_anorms = []
    ep_dists = []
    ep_goal_r = 0.0
    ep_energy_r = 0.0
    ep_time_r = 0.0
    ep_progress_r = 0.0

    info = {}
    while not done:
        agent_xy = env.unwrapped.get_xy()

        # Pick the point we want to head toward: in maze mode, a BFS subgoal
        # in an adjacent free cell; otherwise the goal itself.
        if maze_enabled:
            target_ij = xy_to_ij(goal_xy)
            target_xy = _oracle_subgoal(
                agent_xy, target_ij, bfs_cache, maze_map, xy_to_ij, ij_to_xy)
        else:
            target_xy = goal_xy

        delta = target_xy - agent_xy
        norm = float(np.linalg.norm(delta))
        if norm < 1e-9:
            direction = np.zeros(2)
        else:
            direction = delta / norm

        # `ob` already contains the flow at the agent's current xy (the env
        # appends [flow_vx, flow_vy] to the state when include_flow_in_obs).
        # Reusing it avoids a third per-step flow lookup on top of the two
        # that env.step() makes internally (dynamics + next get_ob).
        flow_xy = (float(ob[2]), float(ob[3]))
        action = _straight_action(direction, flow_xy, action_scale)

        next_ob, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        ep_anorms.append(np.linalg.norm(action))

        step_goal_r = info.get('success', 0.0) * reward_cfg['goal_reward']
        step_energy_r = info.get('energy_cost', 0.0)
        step_time_r = -reward_cfg['time_weight']
        step_dist = info.get('dist_to_goal', 0.0)
        # Progress shaping (potential-based, Ng et al. 1999): positive when
        # the agent closes distance to goal, negative when it moves away.
        # Telescopes to (initial_dist - final_dist) — policy-invariant.
        step_progress_r = reward_cfg['progress_weight'] * (prev_dist - step_dist)
        prev_dist = step_dist
        ep_goal_r += step_goal_r
        ep_energy_r += step_energy_r
        ep_time_r += step_time_r
        ep_progress_r += step_progress_r
        ep_dists.append(step_dist)

        ep_data['observations'].append(ob)
        ep_data['next_observations'].append(next_ob)
        ep_data['actions'].append(action)
        ep_data['rewards'].append(reward)
        ep_data['terminals'].append(float(done))
        ep_data['masks'].append(0.0 if terminated else 1.0)
        ep_data['qpos'].append(info['prev_qpos'])
        ep_data['qvel'].append(info['prev_qvel'])
        ep_data['frame'].append(np.float64(info['frame'] - env.unwrapped.frames_per_step))
        ep_data['goal_xy'].append(np.array(env.unwrapped.cur_goal_xy))
        ep_data['dist_to_goal'].append(step_dist)
        ep_data['goal_reward_components'].append(step_goal_r)
        ep_data['energy_reward_components'].append(step_energy_r)
        ep_data['time_reward_components'].append(step_time_r)
        ep_data['progress_reward_components'].append(step_progress_r)

        ob = next_ob
        step += 1

    stats = dict(
        length=step,
        action_norm=float(np.mean(ep_anorms)) if ep_anorms else 0.0,
        success=info.get('success', 0.0),
        goal_r=ep_goal_r,
        energy_r=ep_energy_r,
        time_r=ep_time_r,
        progress_r=ep_progress_r,
        dist_sum=float(np.sum(ep_dists)),
    )
    return ep_data, stats


def _build_env_and_caches(cfg, worker_id=0, frame_range=None):
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
    # Prewarm just this worker's frame range; mmap of the cache file is shared
    # across workers via the OS page cache, so overlapping ranges collapse
    # to a single page-in.
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
        bfs_cache = _precompute_bfs_cache(maze_map, all_cells)
    else:
        bfs_cache = None
    if worker_id == 0:
        print('  [worker 0] Setup complete; starting episode rollout.', flush=True)
    return dict(env=env, maze_map=maze_map, all_cells=all_cells, bfs_cache=bfs_cache)


def _worker_frame_range(cfg, start_idx, n_episodes, num_episodes_total,
                        n_train_frames):
    """Tightest [frame_lo, frame_hi) the worker will ever access.

    Episodes are scheduled deterministically (see ``_deterministic_start_frame``):
    episode ``i`` starts at ``i * upper / (num_episodes - 1)`` and runs for
    ``max_episode_steps * frames_per_step`` frames. So a worker covering
    [start_idx, start_idx + n_episodes) touches exactly
    [start_idx_frame, last_idx_frame + episode_span).

    Falls back to the full range under ``start_frame_mode='random'``, where
    we can't bound which frames any given episode will hit.
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


def _worker_main(args):
    cfg, seed, start_idx, n_episodes, num_episodes_total, worker_id = args
    np.random.seed(seed)

    # We don't have the env yet, so derive n_train_frames from the cap if
    # set, else fall back to the conservative total of all train files
    # (1000 frames each). The exact number is only needed to clamp the
    # upper bound; over-shooting is harmless.
    train_max = cfg['flow'].get('train_max_file')
    n_train_frames_est = (train_max or 0) * 1000
    frame_range = _worker_frame_range(
        cfg, start_idx, n_episodes, num_episodes_total, n_train_frames_est,
    )
    ctx = _build_env_and_caches(cfg, worker_id=worker_id, frame_range=frame_range)

    dataset = defaultdict(list)
    all_stats = []
    # Iterate global indices in increasing order so flow-frame access is
    # monotonic across episodes within a worker (matches the 2-slot file
    # cache in HITChainFlow).
    global_indices = range(start_idx, start_idx + n_episodes)
    iterator = trange(n_episodes, position=worker_id, desc=f'worker {worker_id}') \
        if worker_id == 0 else range(n_episodes)
    for local_i in iterator:
        global_idx = global_indices[local_i]
        ep_data, stats = _run_one_episode(
            ctx['env'], cfg, ctx['maze_map'], ctx['all_cells'], ctx['bfs_cache'],
            global_idx, num_episodes_total,
        )
        for k, v in ep_data.items():
            dataset[k].extend(v)
        all_stats.append(stats)

    ctx['env'].close()
    return dict(dataset), all_stats


def main(_):
    print(f'Loading config from {FLAGS.config or "<built-in defaults>"}...', flush=True)
    cfg = load_config(FLAGS.config)

    run_cfg = cfg['run']
    reward_cfg = cfg['reward']
    num_episodes = run_cfg['num_episodes']

    n_workers = min(FLAGS.num_workers, num_episodes)

    per_worker = [num_episodes // n_workers] * n_workers
    for i in range(num_episodes % n_workers):
        per_worker[i] += 1

    base_seed = run_cfg['seed']
    # Assign each worker a contiguous slice of global episode indices so the
    # full population still spans [0, num_episodes); together with the
    # deterministic start-frame schedule this gives a uniform sweep across
    # the train flow segment.
    start_indices = np.cumsum([0] + per_worker[:-1]).tolist()
    worker_args = [
        (cfg, base_seed + 1000 * wid, start_indices[wid], per_worker[wid],
         num_episodes, wid)
        for wid in range(n_workers)
    ]

    print(f'Generating {num_episodes} straight-line episodes across {n_workers} worker(s)...',
          flush=True)
    if cfg['maze']['enabled']:
        print('(each worker will precompute its own BFS cache at startup)', flush=True)

    if n_workers == 1:
        print('Running serially in-process...', flush=True)
        results = [_worker_main(worker_args[0])]
    else:
        print(f'Spawning {n_workers} worker process(es); '
              f'only worker 0 will print progress.', flush=True)
        mp_ctx = mp.get_context('spawn')
        with mp_ctx.Pool(n_workers) as pool:
            results = pool.map(_worker_main, worker_args)
    print('All workers finished. Merging datasets...', flush=True)

    dataset = defaultdict(list)
    stats_list = []
    for worker_dataset, worker_stats in results:
        for k, v in worker_dataset.items():
            dataset[k].extend(v)
        stats_list.extend(worker_stats)

    ep_lengths = np.array([s['length'] for s in stats_list])
    ep_action_norms = np.array([s['action_norm'] for s in stats_list])
    ep_successes = np.array([s['success'] for s in stats_list])
    ep_goal_rewards = np.array([s['goal_r'] for s in stats_list])
    ep_energy_costs = np.array([s['energy_r'] for s in stats_list])
    ep_time_costs = np.array([s['time_r'] for s in stats_list])
    ep_progress_rewards = np.array([s['progress_r'] for s in stats_list])
    ep_dist_sums = np.array([s['dist_sum'] for s in stats_list])
    total_steps = int(ep_lengths.sum())
    ep_total_rewards = (ep_goal_rewards + ep_energy_costs
                        + ep_time_costs + ep_progress_rewards)

    print(f'\n=== Dataset stats (straight-line oracle) ===')
    print(f'Total steps: {total_steps}')
    print(f'Episodes: {len(ep_lengths)}  |  Success rate: {ep_successes.mean():.1%}')
    print(f'Episode length   — mean: {ep_lengths.mean():.0f}  median: {np.median(ep_lengths):.0f}  '
          f'min: {ep_lengths.min()}  max: {ep_lengths.max()}')
    print(f'Action norm/step — mean: {ep_action_norms.mean():.3f}  '
          f'min: {ep_action_norms.min():.3f}  max: {ep_action_norms.max():.3f}')
    print(f'Dist-to-goal sum — mean: {ep_dist_sums.mean():.1f}  '
          f'min: {ep_dist_sums.min():.1f}  max: {ep_dist_sums.max():.1f}')

    print(f'\n--- Reward breakdown (current weights: goal={reward_cfg["goal_reward"]}, '
          f'energy={reward_cfg["energy_weight"]}, time={reward_cfg["time_weight"]}, '
          f'progress={reward_cfg["progress_weight"]}) ---')
    abs_total = abs(ep_total_rewards).mean() + 1e-9
    print(f'  Total reward    — mean: {ep_total_rewards.mean():.3f}  '
          f'min: {ep_total_rewards.min():.3f}  max: {ep_total_rewards.max():.3f}')
    print(f'  Goal portion    — mean: {ep_goal_rewards.mean():.3f}  '
          f'(% of |total|: {100*abs(ep_goal_rewards.mean()) / abs_total:.0f}%)')
    print(f'  Energy cost     — mean: {ep_energy_costs.mean():.3f}  '
          f'(% of |total|: {100*abs(ep_energy_costs.mean()) / abs_total:.0f}%)')
    print(f'  Time cost       — mean: {ep_time_costs.mean():.3f}  '
          f'(% of |total|: {100*abs(ep_time_costs.mean()) / abs_total:.0f}%)')
    print(f'  Progress reward — mean: {ep_progress_rewards.mean():.3f}  '
          f'(% of |total|: {100*abs(ep_progress_rewards.mean()) / abs_total:.0f}%)')

    save_path = run_cfg['save_path']
    pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    save_dataset = {}
    for k, v in dataset.items():
        if 'observations' in k and v[0].dtype == np.uint8:
            dtype = np.uint8
        elif k == 'terminals':
            dtype = np.float32
        else:
            dtype = np.float32
        save_dataset[k] = np.array(v, dtype=dtype)

    np.savez_compressed(save_path, **save_dataset)
    print(f'Saved dataset ({total_steps} steps) to {save_path}')


if __name__ == '__main__':
    app.run(main)
