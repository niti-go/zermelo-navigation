"""Recompute rewards in a static dataset with new penalty weights.

The trajectories (observations, actions, etc.) are unchanged — only the
rewards array is overwritten. The per-step reward formula is:

    reward = goal_reward * (reached goal this step)
           - energy_weight * ||action||
           - time_weight * 1
           - distance_weight * dist_to_goal

Usage:
    cd ~/zermelo-navigation
    PYTHONPATH=. python scripts/recompute_rewards.py \
        --energy_weight=0.015438 \
        --time_weight=0.087846 \
        --distance_weight=0.015088

    # Or read weights from the config file:
    PYTHONPATH=. python scripts/recompute_rewards.py --from_config
"""
import pathlib

import numpy as np
from absl import app, flags

import zermelo_env  # noqa
from zermelo_env.zermelo_config import load_config

FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', None, 'Path to .npz dataset. Defaults to config save_path.')
flags.DEFINE_string('config', None, 'Path to zermelo_config.yaml.')
flags.DEFINE_float('goal_reward', None, 'Goal reward. Defaults to config value.')
flags.DEFINE_float('energy_weight', None, 'Energy penalty weight.')
flags.DEFINE_float('time_weight', None, 'Time penalty weight.')
flags.DEFINE_float('distance_weight', None, 'Distance penalty weight.')
flags.DEFINE_bool('from_config', False, 'Read all weights from config file.')


def main(_):
    cfg = load_config(FLAGS.config)
    reward_cfg = cfg['reward']

    # Resolve weights: flags override config, --from_config uses config for all.
    goal_reward = FLAGS.goal_reward if FLAGS.goal_reward is not None else reward_cfg['goal_reward']
    if FLAGS.from_config:
        ew = reward_cfg['energy_weight']
        tw = reward_cfg['time_weight']
        dw = reward_cfg['distance_weight']
    else:
        ew = FLAGS.energy_weight if FLAGS.energy_weight is not None else 0.0
        tw = FLAGS.time_weight if FLAGS.time_weight is not None else 0.0
        dw = FLAGS.distance_weight if FLAGS.distance_weight is not None else 0.0

    dataset_path = FLAGS.dataset or cfg['dataset']['save_path']
    print(f'Loading: {dataset_path}')
    data = dict(np.load(dataset_path))

    actions = data['actions']
    dist_to_goal = data['dist_to_goal']
    goal_components = data['goal_reward_components']

    # Recompute rewards.
    action_norms = np.linalg.norm(actions, axis=1)
    reached_goal = (goal_components > 0.5).astype(np.float32)

    new_rewards = (
        goal_reward * reached_goal
        - ew * action_norms
        - tw
        - dw * dist_to_goal
    ).astype(np.float32)

    # Also update the component arrays for consistency.
    data['rewards'] = new_rewards
    data['energy_reward_components'] = (-ew * action_norms).astype(np.float32)
    data['time_reward_components'] = np.full_like(new_rewards, -tw)
    data['distance_reward_components'] = (-dw * dist_to_goal).astype(np.float32)
    data['goal_reward_components'] = (goal_reward * reached_goal).astype(np.float32)

    # Print summary.
    print(f'\nWeights: goal={goal_reward}, ew={ew}, tw={tw}, dw={dw}')
    print(f'Reward range: [{new_rewards.min():.3f}, {new_rewards.max():.3f}]')
    print(f'Reward mean:  {new_rewards.mean():.3f}')
    print(f'Steps with goal reached: {int(reached_goal.sum())}')

    # Save back.
    np.savez_compressed(dataset_path, **data)
    print(f'\nSaved to {dataset_path}')


if __name__ == '__main__':
    app.run(main)
