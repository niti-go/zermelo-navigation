"""Generate offline datasets of straight-line trajectories to the goal.

An oracle that at every step reads the true flow vector at its current
position and commands the action that cancels the flow component
perpendicular to the goal direction, producing the fastest straight-line
motion the agent can achieve.

Dynamics (see zermelo_env/zermelo_point.py):
    new_xy = xy + dt * (action_scale * action + flow)
We want the resulting motion to point along d = (goal - xy) / ||goal - xy||
at some non-negative speed s, i.e. (action_scale * action + flow) = s * d.
So:
    action = (s * d - flow) / action_scale,  clipped to [-1, 1]^2.
We choose the largest s ≥ 0 such that the unclipped action stays in the
box; that's the fastest straight-line motion at this state. If the flow has
a component along -d larger than the agent's max push along +d, s saturates
at 0 (or the action clips and the trajectory deviates slightly).

In maze mode, the BFS oracle picks an in-corridor subgoal each step and
the agent steers straight at *that* subgoal — so the path is piecewise-
straight between cells rather than literally straight through walls.

Usage:
    python scripts/generate_straight_dataset.py
    python scripts/generate_straight_dataset.py --num_workers=16
"""
import multiprocessing as mp

import numpy as np
from absl import app, flags
from tqdm import trange

import _dataset_common as dc
from zermelo_env.zermelo_config import load_config

FLAGS = flags.FLAGS

flags.DEFINE_string('config', None,
                    'Path to zermelo_config.yaml (optional; uses built-in defaults if omitted).')
flags.DEFINE_integer('num_workers', 16,
                     'Number of parallel worker processes. 1 = run serially in-process.')


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

    s_max = np.inf
    for d, f in ((dx, fx), (dy, fy)):
        if abs(d) < 1e-9:
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

    start_frame = dc.sample_start_frame(env, cfg, global_idx, num_episodes)
    ob, _ = env.reset(options=dict(
        task_info=dict(init_ij=init_ij, goal_ij=goal_ij),
        start_frame=start_frame,
    ))
    goal_xy = np.array(env.unwrapped.cur_goal_xy)

    prev_dist = float(np.linalg.norm(env.unwrapped.get_xy() - goal_xy))

    ep_data = dc.new_ep_data()
    done = False
    step = 0
    ep_anorms, ep_dists = [], []
    ep_goal_r = ep_energy_r = ep_progress_r = 0.0

    info = {}
    while not done:
        agent_xy = env.unwrapped.get_xy()

        if maze_enabled:
            target_ij = xy_to_ij(goal_xy)
            target_xy = dc.oracle_subgoal(
                agent_xy, target_ij, bfs_cache, maze_map, xy_to_ij, ij_to_xy)
        else:
            target_xy = goal_xy

        delta = target_xy - agent_xy
        norm = float(np.linalg.norm(delta))
        direction = np.zeros(2) if norm < 1e-9 else delta / norm

        # `ob` already contains the flow at the agent's current xy (the env
        # appends [flow_vx, flow_vy] when include_flow_in_obs). Reusing it
        # avoids a third per-step flow lookup beyond the two env.step does.
        flow_xy = (float(ob[2]), float(ob[3]))
        action = _straight_action(direction, flow_xy, action_scale)

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
    )

    dataset = dc.new_ep_data()
    all_stats = []
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

    slices, n_workers = dc.plan_worker_slices(
        num_episodes, FLAGS.num_workers, run_cfg['seed'])
    worker_args = [
        (cfg, seed, start_idx, n_eps, num_episodes, wid)
        for (start_idx, n_eps, seed, wid) in slices
    ]

    print(f'Generating {num_episodes} straight-line episodes across {n_workers} worker(s)...',
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

    dc.print_stats(stats_list, reward_cfg,
                   header='Dataset stats (straight-line oracle)')
    dc.save_dataset(dataset, run_cfg['save_path'])


if __name__ == '__main__':
    app.run(main)
