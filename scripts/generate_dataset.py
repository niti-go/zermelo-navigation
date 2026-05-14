"""Generate diverse offline datasets for the Zermelo point maze environment.

Produces episodes of a point agent navigating a (maze or open-arena) world
with background fluid flow. Each episode picks a random start and goal, then
the agent tries to reach the goal via a personality-driven policy. The
resulting dataset contains a variety of trajectories that reach the goal but
take different paths, lengths, and energy. Mimics a real-world scenario
where drones already exist and take noisy diverse trajectories toward a
goal; sensor logs can then train an offline RL algorithm.

Trajectory diversity comes from two orthogonal mechanisms:

1. Route diversity — controlled by `dataset_diverse.route.*`:
   - Each episode samples a random number of intermediate waypoints in
     [waypoints[0], waypoints[1]].
   - Accepted only if total path length stays under
     `detour_budget × direct distance` (BFS cells in maze, Euclidean otherwise).

2. Agent competence — controlled by `dataset_diverse.agent.*`:
   - Three independent dimensions: `noise`, `inertia`, `speed`.
   - `drift ∈ [0, 1]` random-walks each dimension within its range during
     the episode (0 = fixed per episode, 1 = walks the full range).

All parameters live in `zermelo_config.yaml`.

Usage:
    python scripts/generate_dataset.py
    python scripts/generate_dataset.py --num_workers=16
"""
import multiprocessing as mp

import numpy as np
from absl import app, flags
from tqdm import trange

from utils import dataset_common as dc
from zermelo_env.zermelo_config import load_config

FLAGS = flags.FLAGS

flags.DEFINE_string('config', None,
                    'Path to zermelo_config.yaml (optional; uses built-in defaults if omitted).')
flags.DEFINE_integer('num_workers', 16,
                     'Number of parallel worker processes. 1 = run serially in-process.')


# ---------------------------------------------------------------------------
# Waypoint sampling
# ---------------------------------------------------------------------------

def sample_waypoints_maze(all_cells, start_ij, goal_ij, n_target,
                          detour_budget, bfs_cache):
    """Sample-then-accept waypoints in cell space, respecting BFS detour budget."""
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
            continue
        if path_so_far + added + remaining <= budget:
            waypoints.append(wp)
            path_so_far += added
            current = wp
    return waypoints


def sample_waypoints_open(start_xy, goal_xy, xy_bounds, n_target, detour_budget):
    """Sample-then-accept waypoints in continuous XY space (Euclidean detour budget)."""
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


# ---------------------------------------------------------------------------
# Agent personality
# ---------------------------------------------------------------------------

class AgentPersonality:
    """Per-episode randomized agent with three independent dimensions.

    Each of (noise, inertia, speed) is sampled once per episode from its own
    [min, max] range. `drift ∈ [0, 1]` then random-walks each value within
    that same range during the episode (0 = fixed, 1 = full range).
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
        sigma = d * 0.15 * span
        revert = 0.05 * (1.0 - d)
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

        self._smoothed_dir = inertia * self._smoothed_dir + (1 - inertia) * subgoal_dir
        norm = np.linalg.norm(self._smoothed_dir)
        if norm > 1e-6:
            self._smoothed_dir = self._smoothed_dir / norm

        action = speed * self._smoothed_dir + np.random.normal(0, noise, 2)
        return np.clip(action, -1, 1)


# ---------------------------------------------------------------------------
# Episode rollout
# ---------------------------------------------------------------------------

def _run_one_episode(env, cfg, maze_map, all_cells, bfs_cache, xy_bounds,
                     global_idx, num_episodes):
    """Run a single episode and return per-step data + summary stats."""
    maze_enabled = cfg['maze']['enabled']
    task_cfg = cfg['task']
    diverse_cfg = cfg['dataset_diverse']
    route_cfg = diverse_cfg['route']
    agent_cfg = diverse_cfg['agent']
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

    start_frame = dc.sample_start_frame(env, cfg, global_idx, num_episodes)
    ob, _ = env.reset(options=dict(
        task_info=dict(init_ij=init_ij, goal_ij=goal_ij),
        start_frame=start_frame,
    ))

    prev_dist = float(np.linalg.norm(
        env.unwrapped.get_xy() - np.array(env.unwrapped.cur_goal_xy)))

    ep_data = dc.new_ep_data()
    done = False
    step = 0
    ep_anorms, ep_dists = [], []
    ep_goal_r = ep_energy_r = ep_progress_r = 0.0
    wp_tol = route_cfg['waypoint_tolerance']
    wp_timeout = max(1, int(route_cfg['waypoint_step_budget']
                            * run_cfg['max_episode_steps']))
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
            subgoal_xy = dc.oracle_subgoal(
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
        step_dist = info.get('dist_to_goal', 0.0)
        step_progress_r = reward_cfg['progress_weight'] * (prev_dist - step_dist)
        prev_dist = step_dist
        ep_goal_r += step_goal_r
        ep_energy_r += step_energy_r
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
        ep_data['progress_reward_components'].append(step_progress_r)

        ob = next_ob
        step += 1

    stats = dict(
        length=step,
        action_norm=float(np.mean(ep_anorms)) if ep_anorms else 0.0,
        success=info.get('success', 0.0),
        goal_r=ep_goal_r,
        energy_r=ep_energy_r,
        progress_r=ep_progress_r,
        dist_sum=float(np.sum(ep_dists)),
    )
    return ep_data, stats


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _worker_main(args):
    cfg, seed, start_idx, n_episodes, num_episodes_total, worker_id = args
    np.random.seed(seed)

    train_max = cfg['flow'].get('train_max_file')
    n_train_frames_est = (train_max or 0) * 1000
    frame_range = dc.worker_frame_range(
        cfg, start_idx, n_episodes, num_episodes_total, n_train_frames_est,
    )
    ctx = dc.build_env_and_caches(
        cfg, worker_id=worker_id, frame_range=frame_range,
        need_xy_bounds=not cfg['maze']['enabled'],
    )

    dataset = dc.new_ep_data()
    all_stats = []
    global_indices = range(start_idx, start_idx + n_episodes)
    iterator = trange(n_episodes, position=worker_id, desc=f'worker {worker_id}') \
        if worker_id == 0 else range(n_episodes)
    for local_i in iterator:
        global_idx = global_indices[local_i]
        ep_data, stats = _run_one_episode(
            ctx['env'], cfg, ctx['maze_map'], ctx['all_cells'],
            ctx['bfs_cache'], ctx['xy_bounds'],
            global_idx, num_episodes_total,
        )
        for k, v in ep_data.items():
            dataset[k].extend(v)
        all_stats.append(stats)

    ctx['env'].close()
    return dict(dataset), all_stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(_):
    print(f'Loading config from {FLAGS.config or "<built-in defaults>"}...', flush=True)
    cfg = load_config(FLAGS.config)

    run_cfg = cfg['run']
    reward_cfg = cfg['reward']
    num_episodes = run_cfg['num_episodes']

    slices, n_workers = dc.plan_worker_slices(
        num_episodes, FLAGS.num_workers, run_cfg['seed'])
    worker_args = [
        (cfg, seed, start_idx, n_eps, num_episodes, wid)
        for (start_idx, n_eps, seed, wid) in slices
    ]

    print(f'Generating {num_episodes} episodes across {n_workers} worker(s)...',
          flush=True)
    if cfg['maze']['enabled']:
        print('(each worker will precompute its own BFS cache at startup)', flush=True)

    if n_workers == 1:
        print('Running serially in-process...', flush=True)
        results = [_worker_main(worker_args[0])]
    else:
        print(f'Spawning {n_workers} worker process(es); only worker 0 prints progress.',
              flush=True)
        mp_ctx = mp.get_context('spawn')
        with mp_ctx.Pool(n_workers) as pool:
            results = pool.map(_worker_main, worker_args)
    print('All workers finished. Merging datasets...', flush=True)

    dataset = dc.new_ep_data()
    stats_list = []
    for worker_dataset, worker_stats in results:
        for k, v in worker_dataset.items():
            dataset[k].extend(v)
        stats_list.extend(worker_stats)

    summary = dc.print_stats(stats_list, reward_cfg,
                             header='Dataset stats (diverse generator)')

    # Per-step weight suggestions (helps a user pick reward weights from scratch).
    avg_len = summary['ep_lengths'].mean()
    avg_anorm = summary['ep_action_norms'].mean()
    print(f'\n--- Suggested reward params (targeting ~50% of goal_reward per term) ---')
    print(f'  fixed_hover_cost ≈ {0.5 / avg_len:.5f}   '
          f'(so {avg_len:.0f} steps × fixed_hover_cost ≈ 0.5)')
    print(f'  action_weight    ≈ {0.5 / (avg_len * avg_anorm):.5f}   '
          f'(so {avg_len:.0f} steps × {avg_anorm:.2f} avg_norm × action_weight ≈ 0.5)')
    print(f'  progress_weight  ≈ 0.5 / (typical_start_distance ~ 15) ≈ 0.03   '
          f'(scale-invariant: total = progress_weight × travelled_distance)\n')

    dc.save_dataset(dataset, run_cfg['save_path'])


if __name__ == '__main__':
    app.run(main)
