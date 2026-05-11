"""Generate diverse offline datasets for the Zermelo point maze environment.

Produces episodes of a point agent navigating a maze with background fluid
flow. Each episode picks a random start and goal, then the agent tries to reach
the goal. The resulting dataset contains a variety of trajectories that reach the
goal but take different paths, lengths, and energy. It mimics a real-world
scenario where drones already exist in the real world and take slightly noisy
diverse trajectories toward a goal. We can attach sensors to the drones,
collect a dataset, and train an RL algorithm to learn an optimal trajectory
from the existing drones.

Trajectory diversity comes from two orthogonal mechanisms:

1. Route diversity (which path the agent takes) — controlled by `route.*`:
   - Each episode samples a random number of intermediate waypoints in
     [route.waypoints[0], route.waypoints[1]].
   - Waypoints are accepted only if they keep total path length under
     `route.detour_budget × direct distance`. Same semantics in maze (BFS cell
     count) and open arena (Euclidean distance).
   - Maze: BFS grid-cell waypoints, agent follows BFS oracle to each one.
   - Open arena: random XY waypoints in continuous space, agent steers directly.

2. Agent competence (how well the agent executes) — controlled by `agent.*`:
   - Three independent dimensions: `noise`, `inertia`, `speed`. Each is a
     `[min, max]` range; one value is sampled per dimension at episode start.
   - `agent.drift ∈ [0, 1]` controls within-episode variation (0 = each
     dimension fixed, 1 = each random-walks across its full sampled range).

All parameters are controlled by a single YAML config file (zermelo_config.yaml).

Usage:
    python scripts/generate_dataset.py
    python scripts/generate_dataset.py --num_workers=16

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


def sample_waypoints_maze(all_cells, start_ij, goal_ij, n_target,
                          detour_budget, bfs_cache):
    """Sample-then-accept waypoints in cell space.

    Picks up to `n_target` random free cells, accepting each if it keeps the
    cumulative path-through-waypoints + remaining-to-goal under
    `detour_budget × direct distance`. Identical structure to
    `sample_waypoints_open` so the two modes produce comparable distributions.
    """
    if n_target == 0:
        return []

    dist_to_goal = bfs_cache[goal_ij]
    direct = bfs_cache[start_ij][goal_ij[0], goal_ij[1]]
    if direct <= 0:
        return []
    budget = detour_budget * direct

    waypoints = []
    current = start_ij
    path_so_far = 0.0
    for _ in range(n_target):
        wp = all_cells[np.random.randint(len(all_cells))]
        added = bfs_cache[current][wp[0], wp[1]]
        remaining = dist_to_goal[wp[0], wp[1]]
        if added <= 0 or remaining < 0:
            continue  # unreachable from current or unreachable to goal
        if path_so_far + added + remaining <= budget:
            waypoints.append(wp)
            path_so_far += added
            current = wp
    return waypoints


def sample_waypoints_open(start_xy, goal_xy, xy_bounds, n_target, detour_budget):
    """Sample-then-accept waypoints in continuous XY space.

    Mirror of `sample_waypoints_maze` but with Euclidean distances.
    """
    if n_target == 0:
        return []

    direct = float(np.linalg.norm(np.array(goal_xy) - np.array(start_xy)))
    if direct < 1e-6:
        return []
    budget = detour_budget * direct

    x_min, x_max, y_min, y_max = xy_bounds
    waypoints = []
    current = np.array(start_xy)
    path_so_far = 0.0
    for _ in range(n_target):
        wp = np.array([np.random.uniform(x_min, x_max),
                       np.random.uniform(y_min, y_max)])
        added = float(np.linalg.norm(wp - current))
        remaining = float(np.linalg.norm(np.array(goal_xy) - wp))
        if path_so_far + added + remaining <= budget:
            waypoints.append(tuple(wp))
            path_so_far += added
            current = wp
    return waypoints


class AgentPersonality:
    """Per-episode randomized agent with three independent dimensions.

    Each of (noise, inertia, speed) is sampled once per episode from its own
    [min, max] range in agent.*. drift ∈ [0, 1] then random-walks each value
    within that same range during the episode (0 = fixed, 1 = full range).
    """

    def __init__(self, noise_range, inertia_range, speed_range, drift):
        self._dims = {
            'noise':   self._init_dim(noise_range, drift),
            'inertia': self._init_dim(inertia_range, drift),
            'speed':   self._init_dim(speed_range, drift),
        }
        self._smoothed_dir = None

    @staticmethod
    def _init_dim(rng, drift):
        lo, hi = float(rng[0]), float(rng[1])
        d = float(np.clip(drift, 0.0, 1.0))
        v0 = float(np.random.uniform(lo, hi))
        span = max(hi - lo, 1e-9)
        # Scale walk size to this dimension's range so drift=1 covers it in
        # ~tens of steps regardless of absolute scale.
        sigma = d * 0.15 * span
        revert = 0.05 * (1.0 - d)
        # At drift=0, pin to the sampled value; otherwise walk within [lo, hi].
        clip = (lo, hi) if d > 0.0 else (v0, v0)
        return {'value': v0, 'v0': v0, 'sigma': sigma, 'revert': revert, 'clip': clip}

    def _step_dim(self, dim):
        dim['value'] += (
            dim['revert'] * (dim['v0'] - dim['value'])
            + np.random.normal(0, dim['sigma'])
        )
        dim['value'] = float(np.clip(dim['value'], *dim['clip']))
        return dim['value']

    def get_action(self, subgoal_dir):
        if self._smoothed_dir is None:
            self._smoothed_dir = subgoal_dir.copy()

        noise = self._step_dim(self._dims['noise'])
        inertia = self._step_dim(self._dims['inertia'])
        speed = self._step_dim(self._dims['speed'])

        # Smooth heading with exponential moving average.
        self._smoothed_dir = inertia * self._smoothed_dir + (1 - inertia) * subgoal_dir
        norm = np.linalg.norm(self._smoothed_dir)
        if norm > 1e-6:
            self._smoothed_dir = self._smoothed_dir / norm

        action = speed * self._smoothed_dir + np.random.normal(0, noise, 2)
        return np.clip(action, -1, 1)


def _sample_start_frame(env, cfg):
    """Pick a per-episode flow start frame within the train segment.

    Restricts to [0, n_train_frames - max_episode_steps * frames_per_step]
    so a full-length episode never queries past the train cutoff (HIT
    files beyond `flow.train_max_file` are reserved for held-out eval).
    """
    fps = float(env.unwrapped.frames_per_step)
    max_steps = int(cfg['run']['max_episode_steps'])
    n_train = int(env.unwrapped.n_frames)
    upper = n_train - max_steps * fps
    if upper <= 0:
        return 0.0
    return float(np.random.uniform(0.0, upper))


def _run_one_episode(env, cfg, maze_map, all_cells, bfs_cache, xy_bounds):
    """Run a single episode and return per-step data plus episode summary stats."""
    maze_enabled = cfg['maze']['enabled']
    task_cfg = cfg['task']
    route_cfg = cfg['route']
    agent_cfg = cfg['agent']
    run_cfg = cfg['run']
    reward_cfg = cfg['reward']

    if maze_enabled:
        xy_to_ij = env.unwrapped.xy_to_ij
        ij_to_xy = env.unwrapped.ij_to_xy

    # Choose start and goal for this episode.
    if task_cfg['start_goal_mode'] == 'fixed':
        init_ij = tuple(task_cfg['fixed_start_ij'])
        goal_ij = tuple(task_cfg['fixed_goal_ij'])
    else:
        init_ij = all_cells[np.random.randint(len(all_cells))]
        goal_ij = all_cells[np.random.randint(len(all_cells))]
        while goal_ij == init_ij:
            goal_ij = all_cells[np.random.randint(len(all_cells))]

    # Sample intermediate waypoints to force diverse routes.
    wp_min, wp_max = route_cfg['waypoints']
    n_waypoints = np.random.randint(wp_min, wp_max + 1)
    detour_budget = route_cfg['detour_budget']

    if maze_enabled:
        waypoints_ij = sample_waypoints_maze(
            all_cells, init_ij, goal_ij, n_waypoints, detour_budget, bfs_cache,
        )
        route_ij = waypoints_ij + [goal_ij]
        route_xy = [np.array(env.unwrapped.ij_to_xy(ij)) for ij in route_ij]
    else:
        start_xy = np.array(env.unwrapped.ij_to_xy(init_ij))
        goal_xy = np.array(env.unwrapped.ij_to_xy(goal_ij))
        waypoints_xy = sample_waypoints_open(
            start_xy, goal_xy, xy_bounds, n_waypoints, detour_budget,
        )
        route_xy = [np.array(wp) for wp in waypoints_xy] + [goal_xy]

    route_idx = 0
    current_target_xy = route_xy[route_idx]

    personality = AgentPersonality(
        noise_range=agent_cfg['noise'],
        inertia_range=agent_cfg['inertia'],
        speed_range=agent_cfg['speed'],
        drift=agent_cfg.get('drift', 0.0),
    )

    start_frame = _sample_start_frame(env, cfg)
    ob, _ = env.reset(options=dict(
        task_info=dict(init_ij=init_ij, goal_ij=goal_ij),
        start_frame=start_frame,
    ))

    ep_data = defaultdict(list)
    done = False
    step = 0
    ep_anorms = []
    ep_dists = []
    ep_goal_r = 0.0
    ep_energy_r = 0.0
    ep_time_r = 0.0
    ep_dist_r = 0.0
    wp_tol = route_cfg['waypoint_tolerance']
    # Per-waypoint timeout as a fraction of the episode budget; this auto-
    # scales with max_episode_steps so the route can never silently exceed
    # the episode budget.
    wp_timeout = max(1, int(route_cfg['waypoint_step_budget'] * run_cfg['max_episode_steps']))
    wp_steps = 0

    info = {}
    while not done:
        agent_xy = env.unwrapped.get_xy()

        dist_to_waypoint = np.linalg.norm(agent_xy - current_target_xy)
        wp_steps += 1

        timed_out = wp_steps >= wp_timeout and route_idx < len(route_xy) - 1
        reached = dist_to_waypoint < wp_tol and route_idx < len(route_xy) - 1

        if reached or timed_out:
            route_idx += 1
            current_target_xy = route_xy[route_idx]
            wp_steps = 0

        if maze_enabled:
            target_ij = xy_to_ij(current_target_xy)
            subgoal_xy = _oracle_subgoal(
                agent_xy, target_ij, bfs_cache, maze_map, xy_to_ij, ij_to_xy)
        else:
            subgoal_xy = current_target_xy
        subgoal_dir = subgoal_xy - agent_xy
        norm = np.linalg.norm(subgoal_dir)
        if norm > 1e-6:
            subgoal_dir = subgoal_dir / norm

        action = personality.get_action(subgoal_dir)

        next_ob, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        ep_anorms.append(np.linalg.norm(action))

        step_goal_r = info.get('success', 0.0) * reward_cfg['goal_reward']
        step_energy_r = info.get('energy_cost', 0.0)
        step_time_r = -reward_cfg['time_weight']
        step_dist = info.get('dist_to_goal', 0.0)
        step_dist_r = -reward_cfg['distance_weight'] * step_dist
        ep_goal_r += step_goal_r
        ep_energy_r += step_energy_r
        ep_time_r += step_time_r
        ep_dist_r += step_dist_r
        ep_dists.append(step_dist)

        ep_data['observations'].append(ob)
        ep_data['next_observations'].append(next_ob)
        ep_data['actions'].append(action)
        ep_data['rewards'].append(reward)
        ep_data['terminals'].append(float(done))
        ep_data['masks'].append(0.0 if terminated else 1.0)
        ep_data['qpos'].append(info['prev_qpos'])
        ep_data['qvel'].append(info['prev_qvel'])
        # Flow-clock frame at the *start* of this step (before advance).
        ep_data['frame'].append(np.float64(info['frame'] - env.unwrapped.frames_per_step))
        ep_data['goal_xy'].append(np.array(env.unwrapped.cur_goal_xy))
        ep_data['dist_to_goal'].append(step_dist)
        ep_data['goal_reward_components'].append(step_goal_r)
        ep_data['energy_reward_components'].append(step_energy_r)
        ep_data['time_reward_components'].append(step_time_r)
        ep_data['distance_reward_components'].append(step_dist_r)

        ob = next_ob
        step += 1

    stats = dict(
        length=step,
        action_norm=float(np.mean(ep_anorms)) if ep_anorms else 0.0,
        success=info.get('success', 0.0),
        goal_r=ep_goal_r,
        energy_r=ep_energy_r,
        time_r=ep_time_r,
        dist_r=ep_dist_r,
        dist_sum=float(np.sum(ep_dists)),
    )
    return ep_data, stats


def _build_env_and_caches(cfg):
    """Build env + per-worker BFS cache / xy bounds. Returned as a dict."""
    run_cfg = cfg['run']
    train_max = cfg['flow'].get('train_max_file')
    env_kwargs = config_to_env_kwargs(cfg, max_file=train_max)
    env_kwargs['max_episode_steps'] = run_cfg['max_episode_steps']
    env = gymnasium.make('zermelo-pointmaze-medium-v0', **env_kwargs)
    # Prewarm OS page cache for HIT flow files. This generator samples
    # start_frames uniformly across the train segment, so we can't bound
    # the worker's frame range — touch everything. The data is read from
    # the local-SSD cache (see zermelo_env.hit_cache), not NFS, and pages
    # are shared across workers via the OS page cache.
    env.unwrapped._flow_field.prewarm()

    maze_map = env.unwrapped.maze_map
    all_cells = [
        (i, j)
        for i in range(maze_map.shape[0])
        for j in range(maze_map.shape[1])
        if maze_map[i, j] == 0
    ]

    maze_enabled = cfg['maze']['enabled']
    bfs_cache = _precompute_bfs_cache(maze_map, all_cells) if maze_enabled else None

    xy_bounds = None
    if not maze_enabled:
        corner_min = np.array(env.unwrapped.ij_to_xy((1, 1)))
        corner_max = np.array(env.unwrapped.ij_to_xy((6, 6)))
        xy_bounds = (min(corner_min[0], corner_max[0]),
                     max(corner_min[0], corner_max[0]),
                     min(corner_min[1], corner_max[1]),
                     max(corner_min[1], corner_max[1]))

    return dict(env=env, maze_map=maze_map, all_cells=all_cells,
                bfs_cache=bfs_cache, xy_bounds=xy_bounds)


def _worker_main(args):
    """Run a chunk of episodes in a worker process and return the collected data."""
    cfg, seed, n_episodes, worker_id = args
    np.random.seed(seed)

    ctx = _build_env_and_caches(cfg)

    dataset = defaultdict(list)
    all_stats = []
    iterator = trange(n_episodes, position=worker_id, desc=f'worker {worker_id}') \
        if worker_id == 0 else range(n_episodes)
    for _ in iterator:
        ep_data, stats = _run_one_episode(
            ctx['env'], cfg, ctx['maze_map'], ctx['all_cells'],
            ctx['bfs_cache'], ctx['xy_bounds'],
        )
        for k, v in ep_data.items():
            dataset[k].extend(v)
        all_stats.append(stats)

    ctx['env'].close()
    return dict(dataset), all_stats


def main(_):
    cfg = load_config(FLAGS.config)

    run_cfg = cfg['run']
    reward_cfg = cfg['reward']
    num_episodes = run_cfg['num_episodes']

    n_workers = min(FLAGS.num_workers, num_episodes)

    # Split episodes across workers as evenly as possible.
    per_worker = [num_episodes // n_workers] * n_workers
    for i in range(num_episodes % n_workers):
        per_worker[i] += 1

    base_seed = run_cfg['seed']
    worker_args = [
        (cfg, base_seed + 1000 * wid, per_worker[wid], wid)
        for wid in range(n_workers)
    ]

    print(f'Generating {num_episodes} episodes across {n_workers} worker(s)...')
    if cfg['maze']['enabled']:
        print('(each worker will precompute its own BFS cache at startup)')

    if n_workers == 1:
        results = [_worker_main(worker_args[0])]
    else:
        # spawn avoids MuJoCo fork-safety issues.
        mp_ctx = mp.get_context('spawn')
        with mp_ctx.Pool(n_workers) as pool:
            results = pool.map(_worker_main, worker_args)

    # Merge per-worker datasets and stats.
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
    ep_dist_costs = np.array([s['dist_r'] for s in stats_list])
    ep_dist_sums = np.array([s['dist_sum'] for s in stats_list])
    total_steps = int(ep_lengths.sum())
    ep_total_rewards = ep_goal_rewards + ep_energy_costs + ep_time_costs + ep_dist_costs

    print(f'\n=== Dataset stats (use these to set reward params) ===')
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
          f'distance={reward_cfg["distance_weight"]}) ---')
    abs_total = abs(ep_total_rewards).mean() + 1e-9
    print(f'  Total reward  — mean: {ep_total_rewards.mean():.3f}  '
          f'min: {ep_total_rewards.min():.3f}  max: {ep_total_rewards.max():.3f}')
    print(f'  Goal portion  — mean: {ep_goal_rewards.mean():.3f}  '
          f'(% of |total|: {100*abs(ep_goal_rewards.mean()) / abs_total:.0f}%)')
    print(f'  Energy cost   — mean: {ep_energy_costs.mean():.3f}  '
          f'(% of |total|: {100*abs(ep_energy_costs.mean()) / abs_total:.0f}%)')
    print(f'  Time cost     — mean: {ep_time_costs.mean():.3f}  '
          f'(% of |total|: {100*abs(ep_time_costs.mean()) / abs_total:.0f}%)')
    print(f'  Distance cost — mean: {ep_dist_costs.mean():.3f}  '
          f'(% of |total|: {100*abs(ep_dist_costs.mean()) / abs_total:.0f}%)')

    avg_len = ep_lengths.mean()
    avg_anorm = ep_action_norms.mean()
    avg_dist_sum = ep_dist_sums.mean()
    print(f'\n--- Suggested reward params (targeting ~50% of goal_reward as total penalty) ---')
    print(f'  time_weight     ≈ {0.5 / avg_len:.5f}   (so {avg_len:.0f} steps × time_weight ≈ 0.5)')
    print(f'  energy_weight   ≈ {0.5 / (avg_len * avg_anorm):.5f}   '
          f'(so {avg_len:.0f} steps × {avg_anorm:.2f} avg_norm × energy_weight ≈ 0.5)')
    print(f'  distance_weight ≈ {0.5 / max(avg_dist_sum, 1e-9):.5f}   '
          f'(so Σdist={avg_dist_sum:.0f} × distance_weight ≈ 0.5)')
    print(f'  Run: python scripts/analyze_rewards.py  to visualize and sweep weights\n')

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
