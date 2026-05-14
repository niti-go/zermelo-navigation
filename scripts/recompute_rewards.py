"""Recompute rewards in a static dataset with new energy / progress weights.

The trajectories (observations, actions, etc.) are unchanged — only the
rewards array is overwritten. The per-step reward formula matches the env:

    reward = goal_reward * (reached goal this step)
           - (action_weight * ||action|| + fixed_hover_cost)
           + progress_weight * (prev_dist - curr_dist)

Both energy terms (dynamic action cost + fixed hover cost) are paid every
step, including the goal step.

Usage:
    cd ~/zermelo-navigation
    PYTHONPATH=. python scripts/recompute_rewards.py \
        --action_weight=0.015438 \
        --fixed_hover_cost=0.087846 \
        --progress_weight=0.5

    # Or read weights from the config file:
    PYTHONPATH=. python scripts/recompute_rewards.py --from_config
"""
import numpy as np
from absl import app, flags

import zermelo_env  # noqa
from zermelo_env.zermelo_config import load_config

FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', None, 'Path to .npz dataset. Defaults to config save_path.')
flags.DEFINE_string('config', None, 'Path to zermelo_config.yaml.')
flags.DEFINE_float('goal_reward', None, 'Goal reward. Defaults to config value.')
flags.DEFINE_float('action_weight', None, 'Per-unit-action energy weight.')
flags.DEFINE_float('fixed_hover_cost', None, 'Per-step baseline hover energy cost.')
flags.DEFINE_float('progress_weight', None, 'Potential-based progress shaping weight.')
flags.DEFINE_bool('from_config', False, 'Read all weights from config file.')


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

    dataset_path = FLAGS.dataset or cfg['run']['save_path']
    print(f'Loading: {dataset_path}')
    data = dict(np.load(dataset_path))

    actions = data['actions']
    dist_to_goal = data['dist_to_goal']
    goal_components = data['goal_reward_components']
    terminals = data['terminals']

    # Recompute rewards.
    action_norms = np.linalg.norm(actions, axis=1)
    reached_goal = (goal_components > 0.5).astype(np.float32)

    energy_components = -(aw * action_norms + hc).astype(np.float32)

    # Progress shaping: per-step progress_weight * (prev_dist - curr_dist),
    # with prev_dist reset at each episode boundary (terminals[i-1] == 1
    # means a new episode starts at i, so progress at i is 0).
    progress_components = np.zeros_like(action_norms, dtype=np.float32)
    prev_dist = float(dist_to_goal[0])
    new_episode = True
    for i in range(len(action_norms)):
        if new_episode:
            progress_components[i] = 0.0
            new_episode = False
        else:
            progress_components[i] = pw * (prev_dist - float(dist_to_goal[i]))
        prev_dist = float(dist_to_goal[i])
        if terminals[i] > 0.5:
            new_episode = True
    progress_components *= 1.0  # already weighted

    new_rewards = (
        goal_reward * reached_goal
        + energy_components
        + progress_components
    ).astype(np.float32)

    data['rewards'] = new_rewards
    data['energy_reward_components'] = energy_components
    data['progress_reward_components'] = progress_components
    data['goal_reward_components'] = (goal_reward * reached_goal).astype(np.float32)

    print(f'\nWeights: goal={goal_reward}, action_weight={aw}, '
          f'fixed_hover_cost={hc}, progress_weight={pw}')
    print(f'Reward range: [{new_rewards.min():.3f}, {new_rewards.max():.3f}]')
    print(f'Reward mean:  {new_rewards.mean():.3f}')
    print(f'Steps with goal reached: {int(reached_goal.sum())}')

    np.savez_compressed(dataset_path, **data)
    print(f'\nSaved to {dataset_path}')


if __name__ == '__main__':
    app.run(main)
