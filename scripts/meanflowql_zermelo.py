'''
Trains MeanFlowQL on an offline dataset of the Zermelo navigation environment.

=== FULL WORKFLOW (run these in order) ===

# 1. Start a tmux session
tmux new -s meanflowql_zermelo

# 2. Generate the offline dataset (uses zermelo conda env)
conda activate zermelo
cd ~/zermelo-navigation
PYTHONPATH=. python scripts/generate_dataset.py

# 3. Find good reward weights using the analysis script
PYTHONPATH=. python scripts/analyze_rewards.py
#    Check the recommended weights in the output and plots in datasets/hyperparameter_tuning/.

# 4. Recompute rewards in the existing dataset (no regeneration needed)
PYTHONPATH=. python scripts/recompute_rewards.py \
    --energy_weight=<recommended ew> \
    --time_weight=<recommended tw> \
    --distance_weight=<recommended dw>
    
# 5 Visualize trajectories from the dataset (optional)
PYTHONPATH=. python scripts/visualize.py
This saves a video of 5 random trajectories to datasets/video.mp4

# 6. Train MeanFlowQL (uses flowrl conda env, can run from any directory)
conda activate flowrl
wandb login
CUDA_VISIBLE_DEVICES=4 python ~/zermelo-navigation/experiments/zermelo/meanflowql_zermelo.py \
    --offline_steps=1000000 \
    --seed=0 \
    --save_interval=50000 \
    --agent.alpha=2000 \
    --agent.num_candidates=5 \
    --proj_wandb=zermelo_hit_dynamic_poordataset \
    --run_group=meanflowql \
    --wandb_online=True

#    The --zermelo_dataset flag defaults to the path in zermelo_config.yaml
#    (datasets/zermelo_pointmaze_medium.npz). Pass explicitly if using a different file:
#    --zermelo_dataset=/path/to/my_custom_dataset.npz

# 7. Detach tmux: Ctrl+b, then d
# 8. Reattach later: tmux attach -t meanflowql_zermelo

=== WANDB LOGGING ===
  Entity:  --wandb_entity (default: RL_Control_JX)
  Project: --proj_wandb   (default: zermelo)
  Group:   --run_group    (default: meanflowql)
  Mode:    --wandb_online (default: True, set False for offline logging)

=== CHECKPOINTS ===
  Saved to: exp/<proj_wandb>/<run_group>/<exp_name>/
  Includes: agent checkpoints, obs_norm_stats.npz, train.csv, eval.csv, flags.json
'''
import os
import platform
# Set OpenGL platform to EGL for headless rendering
os.environ['PYOPENGL_PLATFORM'] = 'egl'
# Configure MuJoCo to use EGL renderer
os.environ['MUJOCO_GL'] = 'egl'
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import json
import random
import sys
import time

import jax
import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

# Path setup — make both MeanFlowQL and zermelo_env importable regardless of CWD.
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_meanflowql_root = os.path.join(_repo_root, 'ext', 'MeanFlowQL')
sys.path.insert(0, _meanflowql_root)
sys.path.insert(0, _repo_root)

# MeanFlowQL imports.
from agents import agents
from envs.env_utils import EpisodeMonitor
from utils.datasets import Dataset, ReplayBuffer
from utils.evaluation import evaluate, flatten
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, get_wandb_video, setup_wandb

# Zermelo env imports.
import gymnasium
import zermelo_env  # noqa — registers gymnasium envs
from zermelo_env.zermelo_config import load_config, config_to_env_kwargs


FLAGS = flags.FLAGS

# Experiment configuration flags
flags.DEFINE_string('run_group', 'meanflowql', 'Run group in wandb')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'zermelo-pointmaze-medium-v0', 'Environment name.')
flags.DEFINE_string('proj_wandb', 'zermelo', 'wandb project name')
flags.DEFINE_string('wandb_entity', 'RL_Control_JX', 'wandb entity (team/org). Set to None for personal account.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('wandb_save_dir', 'debug/', 'Wandb offline data save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')
flags.DEFINE_boolean('use_observation_normalization', True, 'Whether to normalize observations')

# Zermelo-specific flags
flags.DEFINE_string('zermelo_dataset', None, 'Path to Zermelo .npz dataset.')
flags.DEFINE_string('zermelo_config', None, 'Path to zermelo_config.yaml (uses defaults if None).')
flags.DEFINE_float('reward_shift', 0.0, 'Constant added to all rewards. 0.0 for Zermelo (already has dense penalties).')

# Training configuration flags
flags.DEFINE_integer('offline_steps', 1000000, 'Number of offline steps.')
flags.DEFINE_integer('online_steps', 0, 'Number of online steps.')
flags.DEFINE_integer('buffer_size', 2000000, 'Replay buffer size.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 50000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 50000, 'Saving interval.')

# Evaluation configuration flags
flags.DEFINE_integer('eval_episodes', 50, 'Number of evaluation episodes.')
flags.DEFINE_integer('video_episodes', 0, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')

# Data augmentation and preprocessing flags
flags.DEFINE_float('p_aug', None, 'Probability of applying image augmentation.')
flags.DEFINE_integer('frame_stack', None, 'Number of frames to stack.')
flags.DEFINE_integer('balanced_sampling', 0, 'Whether to use balanced sampling for online fine-tuning.')
flags.DEFINE_float('pretrain_factor', 0.0, 'Fraction of offline steps used for pretraining')
flags.DEFINE_boolean('wandb_online', True, 'Whether to use wandb online mode')

# Early stopping configuration flags
flags.DEFINE_bool('enable_early_stopping', True, 'Whether to enable early stopping.')
flags.DEFINE_integer('early_stopping_patience', 6, 'Number of evaluations to wait before early stopping.')
flags.DEFINE_float('early_stopping_min_delta', 0.01, 'Minimum change in validation metric to qualify as improvement.')
flags.DEFINE_string('early_stopping_metric', 'evaluation/episode.return', 'Metric to monitor for early stopping.')

config_flags.DEFINE_config_file(
    'agent',
    os.path.join(_meanflowql_root, 'agents', 'meanflowql.py'),
    lock_config=False,
)


def load_zermelo_dataset(dataset_path, reward_shift=0.0, train_test_split=0.8):
    """Load a Zermelo .npz dataset and convert to MeanFlowQL format.

    Expected .npz keys (from generate_dataset.py):
      observations, next_observations, actions, rewards, terminals, masks,
      plus optional: qpos, qvel, goal_xy, dist_to_goal, *_reward_components

    Returns:
      (train_dataset_dict, val_dataset_dict) — each a dict ready for Dataset.create().
    """
    print(f"Loading Zermelo dataset: {dataset_path}")
    data = dict(np.load(dataset_path))

    # Validate required keys.
    required = ['observations', 'next_observations', 'actions', 'rewards', 'terminals', 'masks']
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(
            f"Dataset missing required keys: {missing}. "
            f"Regenerate with the updated generate_dataset.py."
        )

    # Build the dataset dict with only the fields MeanFlowQL needs.
    dataset_dict = {
        'observations': data['observations'].astype(np.float32),
        'next_observations': data['next_observations'].astype(np.float32),
        'actions': data['actions'].astype(np.float32),
        'rewards': data['rewards'].astype(np.float32) + reward_shift,
        'terminals': data['terminals'].astype(np.float32),
        'masks': data['masks'].astype(np.float32),
    }

    # Episode-level train/test split (preserves temporal structure within episodes).
    terminals = dataset_dict['terminals']
    ends = np.where(terminals > 0.5)[0]
    episode_starts = np.concatenate([[0], ends[:-1] + 1])
    episode_ends = ends + 1  # exclusive

    num_episodes = len(episode_starts)
    episode_indices = np.arange(num_episodes)
    np.random.shuffle(episode_indices)

    train_ep_count = int(num_episodes * train_test_split)
    train_ep_indices = sorted(episode_indices[:train_ep_count])
    val_ep_indices = sorted(episode_indices[train_ep_count:])

    train_trans = np.concatenate([
        np.arange(episode_starts[ep], episode_ends[ep]) for ep in train_ep_indices
    ])
    val_trans = np.concatenate([
        np.arange(episode_starts[ep], episode_ends[ep]) for ep in val_ep_indices
    ])

    train_dataset = {k: v[train_trans] for k, v in dataset_dict.items()}
    val_dataset = {k: v[val_trans] for k, v in dataset_dict.items()}

    n_samples = len(terminals)
    print(f"\nDataset statistics:")
    print(f"  Total transitions: {n_samples}")
    print(f"  Total episodes: {num_episodes}")
    print(f"  Training: {len(train_trans)} transitions from {len(train_ep_indices)} episodes")
    print(f"  Validation: {len(val_trans)} transitions from {len(val_ep_indices)} episodes")
    print(f"  Observation shape: {dataset_dict['observations'][0].shape}")
    print(f"  Action shape: {dataset_dict['actions'][0].shape}")
    print(f"  Reward range (after shift): [{dataset_dict['rewards'].min():.3f}, {dataset_dict['rewards'].max():.3f}]")
    print(f"  Masks: {dataset_dict['masks'].sum():.0f} bootstrap, "
          f"{n_samples - dataset_dict['masks'].sum():.0f} no-bootstrap")

    return train_dataset, val_dataset


def make_zermelo_eval_env(zermelo_config_path=None):
    """Create a Zermelo eval environment wrapped with EpisodeMonitor."""
    cfg = load_config(zermelo_config_path)
    env_kwargs = config_to_env_kwargs(cfg)
    # Use fixed start/goal for reproducible evaluation.
    env_kwargs['fixed_start_goal'] = True
    env_kwargs['max_episode_steps'] = cfg['run']['max_episode_steps']

    env = gymnasium.make('zermelo-pointmaze-medium-v0', **env_kwargs)
    env = EpisodeMonitor(env)
    return env


def main(_):
    """Main training function for MeanFlowQL on Zermelo navigation."""
    # Load the zermelo YAML config up front so we can log it alongside flags.
    zermelo_cfg = load_config(FLAGS.zermelo_config)
    zermelo_cfg_src = FLAGS.zermelo_config or os.path.join(
        _repo_root, 'configs', 'zermelo_config.yaml')

    # Initialize experiment logging
    exp_name = get_exp_name(FLAGS.seed)
    if FLAGS.wandb_online:
        os.environ["WANDB_MODE"] = "online"
    else:
        os.environ["WANDB_MODE"] = "offline"
    entity = FLAGS.wandb_entity if FLAGS.wandb_entity != 'None' else None
    setup_wandb(project=FLAGS.proj_wandb, group=FLAGS.run_group, name=exp_name,
                entity=entity, mode=os.environ["WANDB_MODE"],
                wandb_output_dir=FLAGS.wandb_save_dir)

    # setup_wandb only logs argparse-style flags; attach the parsed YAML config too.
    wandb.config.update({'zermelo_config_yaml': zermelo_cfg}, allow_val_change=True)
    if os.path.isfile(zermelo_cfg_src):
        wandb.save(zermelo_cfg_src, policy='now')

    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, exp_name)
    print(f"Saving results to {FLAGS.save_dir}")
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f, default=str)
    with open(os.path.join(FLAGS.save_dir, 'zermelo_config.json'), 'w') as f:
        json.dump(zermelo_cfg, f, indent=2, default=str)

    config = FLAGS.agent

    # Resolve dataset path.
    if FLAGS.zermelo_dataset is None:
        # save_path in config is relative to repo root, not CWD.
        FLAGS.zermelo_dataset = os.path.join(_repo_root, zermelo_cfg['run']['save_path'])

    # Create eval environment.
    eval_env = make_zermelo_eval_env(FLAGS.zermelo_config)
    env = make_zermelo_eval_env(FLAGS.zermelo_config)  # for online fine-tuning

    # Load dataset.
    train_dataset, val_dataset = load_zermelo_dataset(
        FLAGS.zermelo_dataset, reward_shift=FLAGS.reward_shift)

    # Set random seeds.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # Clip actions to [-1+eps, 1-eps] (same as MeanFlowQL's make_env_and_datasets).
    action_clip_eps = 1e-5
    train_dataset['actions'] = np.clip(train_dataset['actions'], -1 + action_clip_eps, 1 - action_clip_eps)
    val_dataset['actions'] = np.clip(val_dataset['actions'], -1 + action_clip_eps, 1 - action_clip_eps)

    # Configure datasets and replay buffer.
    train_dataset = Dataset.create(**train_dataset)
    val_dataset = Dataset.create(**val_dataset)
    if FLAGS.balanced_sampling:
        example_transition = {k: v[0] for k, v in train_dataset.items()}
        replay_buffer = ReplayBuffer.create(example_transition, size=FLAGS.buffer_size)
    else:
        train_dataset = ReplayBuffer.create_from_initial_dataset(
            dict(train_dataset), size=max(FLAGS.buffer_size, train_dataset.size + 1)
        )
        replay_buffer = train_dataset

    for dataset in [train_dataset, val_dataset, replay_buffer]:
        if dataset is not None:
            dataset.p_aug = FLAGS.p_aug
            dataset.frame_stack = FLAGS.frame_stack
            if config['agent_name'] == 'rebrac':
                dataset.return_next_actions = True

    if FLAGS.use_observation_normalization:
        print("Computing observation normalization statistics...")
        if train_dataset is not None:
            train_dataset.compute_normalization_stats()
            train_dataset.enable_normalization(True)
            norm_stats_path = os.path.join(FLAGS.save_dir, 'obs_norm_stats.npz')
            np.savez(norm_stats_path, obs_mean=train_dataset.obs_mean, obs_std=train_dataset.obs_std)
            print(f"Saved observation normalization stats to {norm_stats_path}")
        if val_dataset is not None and train_dataset is not None:
            val_dataset.obs_mean = train_dataset.obs_mean
            val_dataset.obs_std = train_dataset.obs_std
            val_dataset.enable_normalization(True)
        if replay_buffer is not None and train_dataset is not None and replay_buffer != train_dataset:
            replay_buffer.obs_mean = train_dataset.obs_mean
            replay_buffer.obs_std = train_dataset.obs_std
            replay_buffer.enable_normalization(True)
    else:
        print("Observation normalization disabled")
    print("Observation normalization setup complete.")

    # Initialize agent.
    example_batch = train_dataset.sample(1)
    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )
    agent.print_param_stats()

    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    # Initialize loggers.
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()

    step = 0
    done = True
    expl_metrics = dict()
    online_rng = jax.random.PRNGKey(FLAGS.seed)

    # Early stopping setup.
    best_metric_value = float('-inf')
    patience_counter = 0
    early_stopped = False
    maximize_metric = ('return' in FLAGS.early_stopping_metric
                       or 'reward' in FLAGS.early_stopping_metric
                       or 'success' in FLAGS.early_stopping_metric)
    if not maximize_metric:
        best_metric_value = float('inf')

    # Pretraining phase.
    pretrain_steps = int(FLAGS.offline_steps * FLAGS.pretrain_factor)
    for i in tqdm.tqdm(range(1, pretrain_steps + 1), smoothing=0.1, dynamic_ncols=True, desc="Pretrain"):
        batch = train_dataset.sample(config['batch_size'])
        agent, update_info = agent.pretrain(batch, current_step=i)

        if i % FLAGS.log_interval == 0:
            train_metrics = {f'pretraining/{k}': v for k, v in update_info.items()}
            wandb.log(train_metrics, step=i)

        if FLAGS.eval_interval != 0 and (i == 1 or i % FLAGS.eval_interval == 0):
            eval_info, trajs, cur_renders = evaluate(
                agent=agent, env=eval_env, config=config,
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=0, video_frame_skip=FLAGS.video_frame_skip,
                train_dataset=train_dataset,
            )
            eval_metrics = {f'pretrain_evaluation/{k}': v for k, v in eval_info.items()}
            wandb.log(eval_metrics, step=i)

    # Q-Learning phase.
    q_learning_steps = FLAGS.offline_steps + FLAGS.online_steps
    best_metric_value = float('-inf') if maximize_metric else float('inf')
    patience_counter = 0
    early_stopped = False

    for i in tqdm.tqdm(range(pretrain_steps, q_learning_steps + pretrain_steps + 1),
                       smoothing=0.1, dynamic_ncols=True, desc="Q-Learning"):
        if i <= FLAGS.offline_steps + pretrain_steps:
            batch = train_dataset.sample(config['batch_size'])
            if config['agent_name'] == 'rebrac':
                agent, update_info = agent.update(batch, full_update=(i % config['actor_freq'] == 0))
            else:
                agent, update_info = agent.update(batch, current_step=i)
        else:
            # Online fine-tuning phase.
            online_rng, key = jax.random.split(online_rng)
            if done:
                step = 0
                ob, _ = env.reset()

            ob_batch = ob[None, :] if len(ob.shape) == 1 else ob
            if (FLAGS.use_observation_normalization and train_dataset is not None
                    and hasattr(train_dataset, 'normalize_obs') and train_dataset.normalize_obs):
                ob_batch = (ob_batch - train_dataset.obs_mean) / train_dataset.obs_std

            action = agent.sample_actions(observations=ob_batch, seed=key)
            action = np.array(action)
            if action.ndim > 1 and action.shape[0] == 1:
                action = action[0]

            next_ob, reward, terminated, truncated, info = env.step(action.copy())
            done = terminated or truncated
            reward = reward + FLAGS.reward_shift

            replay_buffer.add_transition(dict(
                observations=ob, actions=action, rewards=reward,
                terminals=float(done), masks=1.0 - float(terminated),
                next_observations=next_ob,
            ))
            ob = next_ob

            if done:
                expl_metrics = {f'exploration/{k}': np.mean(v) for k, v in flatten(info).items()}
            step += 1

            if FLAGS.balanced_sampling:
                dataset_batch = train_dataset.sample(config['batch_size'] // 2)
                replay_batch = replay_buffer.sample(config['batch_size'] // 2)
                batch = {k: np.concatenate([dataset_batch[k], replay_batch[k]], axis=0)
                         for k in dataset_batch}
            else:
                batch = train_dataset.sample(config['batch_size'])

            if config['agent_name'] == 'rebrac':
                agent, update_info = agent.update(batch, full_update=(i % config['actor_freq'] == 0),
                                                  current_step=i)
            else:
                agent, update_info = agent.update(batch, current_step=i)

        # Log training metrics.
        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            if val_dataset is not None:
                val_batch = val_dataset.sample(config['batch_size'])
                _, val_info = agent.total_loss(val_batch, grad_params=None, rng=agent.rng, current_step=i)
                train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})
            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            train_metrics.update(expl_metrics)
            last_time = time.time()
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        # Evaluate.
        if FLAGS.eval_interval != 0 and (i % FLAGS.eval_interval == 0):
            eval_info, trajs, cur_renders = evaluate(
                agent=agent, env=eval_env, config=config,
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=0, video_frame_skip=FLAGS.video_frame_skip,
                train_dataset=train_dataset,
            )
            eval_metrics = {f'evaluation/{k}': v for k, v in eval_info.items()}
            wandb.log(eval_metrics, step=i)
            eval_logger.log(eval_metrics, step=i)

            # Early stopping.
            if FLAGS.enable_early_stopping and FLAGS.early_stopping_metric in eval_metrics:
                current_metric = eval_metrics[FLAGS.early_stopping_metric]
                if maximize_metric:
                    improved = current_metric > best_metric_value + FLAGS.early_stopping_min_delta
                else:
                    improved = current_metric < best_metric_value - FLAGS.early_stopping_min_delta

                if improved:
                    best_metric_value = current_metric
                    patience_counter = 0
                    print(f"Q-Learning: New best {FLAGS.early_stopping_metric}: {current_metric:.4f}")
                else:
                    patience_counter += 1
                    print(f"Q-Learning: No improvement in {FLAGS.early_stopping_metric}. "
                          f"Patience: {patience_counter}/{FLAGS.early_stopping_patience}")

                if patience_counter >= FLAGS.early_stopping_patience:
                    print(f"Early stopping triggered during Q-Learning at step {i}")
                    early_stopped = True
                    break

        # Save checkpoint.
        if i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, i)

        if early_stopped:
            print(f"Q-Learning: Early stopping triggered at step {i}")
            wandb.log({
                'early_stopping/step': i,
                'early_stopping/metric': FLAGS.early_stopping_metric,
                'early_stopping/best_metric_value': best_metric_value,
            })
            train_logger.log({
                'early_stopping/step': i,
                'early_stopping/metric': FLAGS.early_stopping_metric,
                'early_stopping/best_metric_value': best_metric_value,
            })
            break

    wandb.finish()
    train_logger.close()
    eval_logger.close()


if __name__ == '__main__':
    app.run(main)
