"""Recompute rewards in a static dataset with new energy / progress weights.

Trajectories are unchanged — only `data['rewards']` (and its component
arrays) get rewritten. Per-step reward formula matches the env:

    reward = goal_reward * (reached goal this step)
           - (action_weight * ||action|| + fixed_hover_cost)
           + progress_weight * (prev_dist - curr_dist)

Both energy terms (dynamic action cost + fixed hover cost) are paid every
step, including the goal step.

Usage:
    cd ~/zermelo-navigation
    PYTHONPATH=. python scripts/recompute_rewards.py \
        --action_weight=0.015 \
        --fixed_hover_cost=0.088 \
        --progress_weight=0.5

    # Or read all weights from the config file:
    PYTHONPATH=. python scripts/recompute_rewards.py --from_config
"""
import os
import sys

import numpy as np
from absl import app, flags

# Make scripts/ importable for the shared helpers, repo root for `import zermelo_env`.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(_SCRIPT_DIR))

import _training_common as tc  # noqa: E402
from zermelo_env.zermelo_config import load_config  # noqa: E402

FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', None, 'Path to .npz dataset. Defaults to config save_path.')
flags.DEFINE_string('config', None, 'Path to zermelo_config.yaml.')
flags.DEFINE_float('goal_reward', None, 'Goal reward. Defaults to config value.')
flags.DEFINE_float('action_weight', None, 'Per-unit-action energy weight.')
flags.DEFINE_float('fixed_hover_cost', None, 'Per-step baseline hover energy cost.')
flags.DEFINE_float('progress_weight', None, 'Potential-based progress shaping weight.')
flags.DEFINE_bool('from_config', False, 'Read all weights from config file.')


def _compute_progress_components(dist_to_goal, terminals, progress_weight):
    """Per-step progress shaping: pw * (prev_dist - curr_dist).

    Resets at episode boundaries: the first step of each episode contributes
    0 (no prior distance). The first episode's first step is treated as a
    boundary because there is no prior episode.
    """
    n = len(dist_to_goal)
    # Vectorized via shifted-by-one: prev_dist = dist_to_goal shifted right.
    prev_dist = np.empty_like(dist_to_goal)
    prev_dist[0] = dist_to_goal[0]
    prev_dist[1:] = dist_to_goal[:-1]
    progress = progress_weight * (prev_dist - dist_to_goal)
    # Step i is the start of a new episode iff terminals[i-1] == 1 (and i > 0).
    is_episode_start = np.zeros(n, dtype=bool)
    is_episode_start[0] = True
    is_episode_start[1:] = terminals[:-1] > 0.5
    progress[is_episode_start] = 0.0
    return progress.astype(np.float32)


def main(_):
    cfg = load_config(FLAGS.config)
    reward_cfg = cfg['reward']
    energy_cfg = reward_cfg['energy']

    goal_reward = (FLAGS.goal_reward if FLAGS.goal_reward is not None
                   else reward_cfg['goal_reward'])
    if FLAGS.from_config:
        aw = energy_cfg['action_weight']
        hc = energy_cfg['fixed_hover_cost']
        pw = reward_cfg['progress_weight']
    else:
        aw = FLAGS.action_weight if FLAGS.action_weight is not None else 0.0
        hc = FLAGS.fixed_hover_cost if FLAGS.fixed_hover_cost is not None else 0.0
        pw = FLAGS.progress_weight if FLAGS.progress_weight is not None else 0.0

    dataset_path = FLAGS.dataset or tc.default_dataset_path(cfg)
    print(f'Loading: {dataset_path}')
    data = dict(np.load(dataset_path))

    actions = data['actions']
    dist_to_goal = data['dist_to_goal']
    terminals = data['terminals']
    goal_components_in = data['goal_reward_components']

    action_norms = np.linalg.norm(actions, axis=1)
    reached_goal = (goal_components_in > 0.5).astype(np.float32)

    energy_components = -(aw * action_norms + hc).astype(np.float32)
    progress_components = _compute_progress_components(dist_to_goal, terminals, pw)
    goal_components = (goal_reward * reached_goal).astype(np.float32)
    new_rewards = (goal_components + energy_components + progress_components).astype(np.float32)

    data['rewards'] = new_rewards
    data['energy_reward_components'] = energy_components
    data['progress_reward_components'] = progress_components
    data['goal_reward_components'] = goal_components

    print(f'\nWeights: goal={goal_reward}, action_weight={aw}, '
          f'fixed_hover_cost={hc}, progress_weight={pw}')
    print(f'Reward range: [{new_rewards.min():.3f}, {new_rewards.max():.3f}]')
    print(f'Reward mean:  {new_rewards.mean():.3f}')
    print(f'Steps with goal reached: {int(reached_goal.sum())}')

    np.savez_compressed(dataset_path, **data)
    print(f'\nSaved to {dataset_path}')


if __name__ == '__main__':
    app.run(main)
