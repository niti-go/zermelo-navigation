"""Generate diverse offline datasets for the Zermelo point maze environment.

Produces ~1000 episodes of a point agent navigating a maze with background fluid
flow. Each episode picks a random start and goal, then the agent tries to reach
the goal. The resulting dataset contains a variety of trajectories that reach the goal
but take different paths, lengths, and energy. It mimics a real-world scenario where
drones already exist in the real world and
take slightly noisy diverse trajectories toward a goal. We can attach sensors to
the drones, collect a dataset, and train an RL algorithm to learn an optimal trajectory
from the existing drones.

Trajectory diversity comes from two independent mechanisms:

1. Route diversity (which path the agent takes):
   - Each episode samples 0-3 random intermediate waypoints between start and goal.
   - Waypoints are filtered by a detour budget so alternate routes are explored
     without absurd backtracking.
   - With maze enabled: BFS grid-cell waypoints, agent follows BFS oracle.
   - With maze disabled (open arena): random XY waypoints in continuous space,
     agent steers directly toward each waypoint.

2. Agent diversity (how the agent executes the route):
   Each episode rolls a fresh "personality" with three knobs: noise, inertia, speed.

All parameters are controlled by a single YAML config file (zermelo_config.yaml).

Usage:
    python scripts/generate_dataset.py
"""
import pathlib
from collections import defaultdict

import gymnasium
import numpy as np
from absl import app, flags
from tqdm import trange

import zermelo_env  # noqa — registers gymnasium envs
from zermelo_env.zermelo_config import load_config, config_to_env_kwargs

FLAGS = flags.FLAGS

flags.DEFINE_string('config', None, 'Path to zermelo_config.yaml (optional; uses built-in defaults if omitted).')


def bfs_reachable(maze_map, start_ij):
    """Return BFS distance map from start_ij. Unreachable cells have value -1."""
    dist = np.full_like(maze_map, -1)
    dist[start_ij[0], start_ij[1]] = 0
    queue = [start_ij]
    while queue:
        i, j = queue.pop(0)
        for di, dj in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
            ni, nj = i + di, j + dj
            if (0 <= ni < maze_map.shape[0] and 0 <= nj < maze_map.shape[1]
                    and maze_map[ni, nj] == 0 and dist[ni, nj] == -1):
                dist[ni, nj] = dist[i, j] + 1
                queue.append((ni, nj))
    return dist


def sample_reachable_waypoints(maze_map, all_cells, start_ij, goal_ij,
                               max_waypoints, max_detour):
    """Sample 0..max_waypoints waypoints that don't create absurd detours."""
    n_waypoints = np.random.randint(0, max_waypoints + 1)
    if n_waypoints == 0:
        return []

    dist_to_goal = bfs_reachable(maze_map, goal_ij)

    waypoints = []
    current = start_ij
    for _ in range(n_waypoints):
        dist_from_current = bfs_reachable(maze_map, current)
        direct = dist_from_current[goal_ij[0], goal_ij[1]]
        if direct <= 0:
            break

        budget = max_detour * direct
        candidates = [
            c for c in all_cells
            if dist_from_current[c[0], c[1]] > 0
            and dist_from_current[c[0], c[1]] + dist_to_goal[c[0], c[1]] <= budget
        ]
        if not candidates:
            break

        wp = candidates[np.random.randint(len(candidates))]
        waypoints.append(wp)
        current = wp

    return waypoints


def sample_xy_waypoints(start_xy, goal_xy, xy_bounds, max_waypoints, max_detour):
    """Sample random XY waypoints in continuous open space (no maze walls).

    Returns a list of (x, y) tuples. The detour budget limits total path length
    relative to the straight-line start→goal distance.
    """
    n_waypoints = np.random.randint(0, max_waypoints + 1)
    if n_waypoints == 0:
        return []

    x_min, x_max, y_min, y_max = xy_bounds
    direct_dist = np.linalg.norm(np.array(goal_xy) - np.array(start_xy))
    if direct_dist < 1e-6:
        return []

    budget = max_detour * direct_dist

    waypoints = []
    current = np.array(start_xy)
    for _ in range(n_waypoints):
        # Sample a random point in the arena.
        wp = np.array([np.random.uniform(x_min, x_max),
                       np.random.uniform(y_min, y_max)])

        # Check detour budget: path so far + this waypoint + remaining to goal.
        path_so_far = sum(
            np.linalg.norm(np.array(waypoints[i]) - (np.array(waypoints[i-1]) if i > 0 else current))
            for i in range(len(waypoints))
        )
        added = np.linalg.norm(wp - (np.array(waypoints[-1]) if waypoints else current))
        remaining = np.linalg.norm(np.array(goal_xy) - wp)

        if path_so_far + added + remaining <= budget:
            waypoints.append(tuple(wp))

    return waypoints


class AgentPersonality:
    """Per-episode randomized agent with three knobs: noise, inertia, speed."""

    def __init__(self, noise_range, inertia_range, speed_range):
        self.noise = np.random.uniform(*noise_range)
        self.inertia = np.random.uniform(*inertia_range)
        self._speed_range = speed_range
        self._speed = np.random.uniform(*speed_range)
        self._smoothed_dir = None

    def get_action(self, subgoal_dir):
        if self._smoothed_dir is None:
            self._smoothed_dir = subgoal_dir.copy()

        # Smooth heading with exponential moving average.
        self._smoothed_dir = self.inertia * self._smoothed_dir + (1 - self.inertia) * subgoal_dir
        norm = np.linalg.norm(self._smoothed_dir)
        if norm > 1e-6:
            self._smoothed_dir = self._smoothed_dir / norm

        # Random-walk the speed (mean-reverting toward center of range).
        mid = 0.5 * (self._speed_range[0] + self._speed_range[1])
        self._speed += 0.05 * (mid - self._speed) + np.random.normal(0, 0.03)
        self._speed = np.clip(self._speed, self._speed_range[0], self._speed_range[1])

        action = self._speed * self._smoothed_dir + np.random.normal(0, self.noise, 2)
        return np.clip(action, -1, 1)


def main(_):
    cfg = load_config(FLAGS.config)

    # Unpack sections for readability.
    ds_cfg = cfg['dataset']
    traj_cfg = cfg['trajectory']
    pers_cfg = cfg['personality']
    sg_cfg = cfg['start_goal']

    np.random.seed(ds_cfg['seed'])

    # Build env kwargs from config.
    env_kwargs = config_to_env_kwargs(cfg)
    env_kwargs['max_episode_steps'] = ds_cfg['max_episode_steps']

    env = gymnasium.make('zermelo-pointmaze-medium-v0', **env_kwargs)

    # Collect free cells.
    all_cells = []
    maze_map = env.unwrapped.maze_map
    for i in range(maze_map.shape[0]):
        for j in range(maze_map.shape[1]):
            if maze_map[i, j] == 0:
                all_cells.append((i, j))

    maze_enabled = cfg['maze']['enabled']

    # Compute continuous XY bounds for open-arena waypoint sampling.
    if not maze_enabled:
        # Inner free cells are rows/cols 1..6 in an 8x8 grid.
        corner_min = np.array(env.unwrapped.ij_to_xy((1, 1)))
        corner_max = np.array(env.unwrapped.ij_to_xy((6, 6)))
        xy_bounds = (min(corner_min[0], corner_max[0]),
                     max(corner_min[0], corner_max[0]),
                     min(corner_min[1], corner_max[1]),
                     max(corner_min[1], corner_max[1]))

    dataset = defaultdict(list)
    total_steps = 0
    num_episodes = ds_cfg['num_episodes']

    # Per-episode stats for reward parameter tuning.
    ep_lengths = []
    ep_action_norms = []
    ep_successes = []

    # Fixed start/goal from config.
    fixed_init_ij = tuple(sg_cfg['start_ij'])
    fixed_goal_ij = tuple(sg_cfg['goal_ij'])

    for ep_idx in trange(num_episodes):

        # Choose start and goal for this episode.
        if sg_cfg['fixed']:
            init_ij = fixed_init_ij
            goal_ij = fixed_goal_ij
        else:
            init_ij = all_cells[np.random.randint(len(all_cells))]
            goal_ij = all_cells[np.random.randint(len(all_cells))]
            # Ensure start != goal.
            while goal_ij == init_ij:
                goal_ij = all_cells[np.random.randint(len(all_cells))]

        # Sample intermediate waypoints to force diverse routes.
        if maze_enabled:
            waypoints_ij = sample_reachable_waypoints(
                maze_map, all_cells, init_ij, goal_ij,
                traj_cfg['max_waypoints'], traj_cfg['max_detour'],
            )
            # Full route as grid cells: start -> waypoints -> goal.
            route_ij = waypoints_ij + [goal_ij]
            route_xy = [np.array(env.unwrapped.ij_to_xy(ij)) for ij in route_ij]
        else:
            start_xy = np.array(env.unwrapped.ij_to_xy(init_ij))
            goal_xy = np.array(env.unwrapped.ij_to_xy(goal_ij))
            waypoints_xy = sample_xy_waypoints(
                start_xy, goal_xy, xy_bounds,
                traj_cfg['max_waypoints'], traj_cfg['max_detour'],
            )
            route_xy = [np.array(wp) for wp in waypoints_xy] + [goal_xy]

        route_idx = 0
        current_target_xy = route_xy[route_idx]

        # Each episode gets a fresh agent personality.
        personality = AgentPersonality(
            noise_range=pers_cfg['noise'],
            inertia_range=pers_cfg['inertia'],
            speed_range=pers_cfg['speed'],
        )

        ob, _ = env.reset(options=dict(task_info=dict(init_ij=init_ij, goal_ij=goal_ij)))
        # Point the env's goal at the final destination (for success/reward).
        env.unwrapped.set_goal(goal_ij)

        done = False
        step = 0
        ep_anorms = []
        wp_tol = traj_cfg['waypoint_tolerance']
        wp_timeout = traj_cfg.get('waypoint_timeout', 200)
        wp_steps = 0  # steps spent pursuing the current waypoint

        while not done:
            agent_xy = env.unwrapped.get_xy()

            # Check if we've reached the current waypoint, or timed out on it.
            dist_to_waypoint = np.linalg.norm(agent_xy - current_target_xy)
            wp_steps += 1

            timed_out = wp_steps >= wp_timeout and route_idx < len(route_xy) - 1
            reached = dist_to_waypoint < wp_tol and route_idx < len(route_xy) - 1

            if reached or timed_out:
                route_idx += 1
                current_target_xy = route_xy[route_idx]
                wp_steps = 0

            # Get direction toward the current target.
            if maze_enabled:
                subgoal_xy, _ = env.unwrapped.get_oracle_subgoal(agent_xy, current_target_xy)
            else:
                # Open arena: steer directly toward the target.
                subgoal_xy = current_target_xy
            subgoal_dir = subgoal_xy - agent_xy
            norm = np.linalg.norm(subgoal_dir)
            if norm > 1e-6:
                subgoal_dir = subgoal_dir / norm

            # Let the personality produce the actual action.
            action = personality.get_action(subgoal_dir)

            next_ob, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_anorms.append(np.linalg.norm(action))

            dataset['observations'].append(ob)
            dataset['actions'].append(action)
            dataset['rewards'].append(reward)
            dataset['terminals'].append(done)
            dataset['qpos'].append(info['prev_qpos'])
            dataset['qvel'].append(info['prev_qvel'])
            dataset['goal_xy'].append(np.array(env.unwrapped.cur_goal_xy))

            ob = next_ob
            step += 1

        ep_lengths.append(step)
        ep_action_norms.append(np.mean(ep_anorms))
        ep_successes.append(info.get('success', 0.0))

        total_steps += step

    # --- Dataset stats for reward parameter tuning ---
    ep_lengths = np.array(ep_lengths)
    ep_action_norms = np.array(ep_action_norms)
    ep_successes = np.array(ep_successes)

    print(f'\n=== Dataset stats (use these to set reward params) ===')
    print(f'Total steps: {total_steps}')
    print(f'Episodes: {len(ep_lengths)}  |  Success rate: {ep_successes.mean():.1%}')
    print(f'Episode length   — mean: {ep_lengths.mean():.0f}  median: {np.median(ep_lengths):.0f}  '
          f'min: {ep_lengths.min()}  max: {ep_lengths.max()}')
    print(f'Action norm/step — mean: {ep_action_norms.mean():.3f}  '
          f'min: {ep_action_norms.min():.3f}  max: {ep_action_norms.max():.3f}')

    avg_len = ep_lengths.mean()
    avg_anorm = ep_action_norms.mean()
    print(f'\n--- Suggested reward params (targeting ~50% of goal_reward as total penalty) ---')
    print(f'  time_weight   ≈ {0.5 / avg_len:.5f}   (so {avg_len:.0f} steps × time_weight ≈ 0.5)')
    print(f'  energy_weight ≈ {0.5 / (avg_len * avg_anorm):.5f}   '
          f'(so {avg_len:.0f} steps × {avg_anorm:.2f} avg_norm × energy_weight ≈ 0.5)')
    print(f'  (adjust up/down to make the gap between good and bad episodes larger/smaller)\n')

    save_path = ds_cfg['save_path']
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
