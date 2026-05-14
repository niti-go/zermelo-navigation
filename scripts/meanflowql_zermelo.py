'''
Trains MeanFlowQL on an offline dataset of the Zermelo navigation environment.

=== FULL WORKFLOW ===
# (Identical to bc_zermelo.py except step 5 trains MeanFlowQL.)
# See bc_zermelo.py docstring for the full sequence.

CUDA_VISIBLE_DEVICES=4 python ~/zermelo-navigation/scripts/meanflowql_zermelo.py \
    --offline_steps=1000000 \
    --seed=0 \
    --save_interval=50000 \
    --agent.alpha=2000 \
    --agent.num_candidates=5 \
    --proj_wandb=zermelo_hit_dynamic_poordataset \
    --run_group=meanflowql \
    --wandb_online=True

The --zermelo_dataset flag defaults to the path in zermelo_config.yaml
(datasets/<save_path>). Pass explicitly if using a different file.
'''
import os
# Set OpenGL platform to EGL for headless rendering (must precede MuJoCo import).
os.environ['PYOPENGL_PLATFORM'] = 'egl'
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

# Make scripts/ importable for shared helpers, repo root for `import zermelo_env`,
# and ext/MeanFlowQL for the MeanFlowQL package.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
_MEANFLOWQL_ROOT = os.path.join(_REPO_ROOT, 'ext', 'MeanFlowQL')
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _MEANFLOWQL_ROOT)

from helpers import training_common as tc  # noqa: E402
from zermelo_env.zermelo_config import load_config  # noqa: E402

# MeanFlowQL imports.
from agents import agents  # noqa: E402
from envs.env_utils import EpisodeMonitor  # noqa: E402
from utils.datasets import Dataset, ReplayBuffer  # noqa: E402
from utils.evaluation import evaluate, flatten  # noqa: E402
from utils.flax_utils import restore_agent, save_agent  # noqa: E402
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, setup_wandb  # noqa: E402


FLAGS = flags.FLAGS

# Experiment configuration flags
flags.DEFINE_string('run_group', 'meanflowql', 'Run group in wandb')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'zermelo-pointmaze-medium-v0', 'Environment name.')
flags.DEFINE_string('proj_wandb', 'zermelo', 'wandb project name')
flags.DEFINE_string('wandb_entity', 'RL_Control_JX',
                    'wandb entity (team/org). Set to None for personal account.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('wandb_save_dir', 'debug/', 'Wandb offline data save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')
flags.DEFINE_boolean('use_observation_normalization', True,
                     'Whether to normalize observations')

# Zermelo-specific flags
flags.DEFINE_string('zermelo_dataset', None, 'Path to Zermelo .npz dataset.')
flags.DEFINE_string('zermelo_config', None,
                    'Path to zermelo_config.yaml (uses defaults if None).')
flags.DEFINE_float('reward_shift', 0.0,
                   'Constant added to all rewards. 0.0 for Zermelo (already has dense penalties).')

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
flags.DEFINE_integer('balanced_sampling', 0, 'Balanced sampling for online fine-tuning.')
flags.DEFINE_float('pretrain_factor', 0.0, 'Fraction of offline steps used for pretraining')
flags.DEFINE_boolean('wandb_online', True, 'Whether to use wandb online mode')

# Early stopping configuration flags
flags.DEFINE_bool('enable_early_stopping', True, 'Whether to enable early stopping.')
flags.DEFINE_integer('early_stopping_patience', 6,
                     'Number of evaluations to wait before early stopping.')
flags.DEFINE_float('early_stopping_min_delta', 0.01,
                   'Minimum change in validation metric to qualify as improvement.')
flags.DEFINE_string('early_stopping_metric', 'evaluation/episode.return',
                    'Metric to monitor for early stopping.')

config_flags.DEFINE_config_file(
    'agent',
    os.path.join(_MEANFLOWQL_ROOT, 'agents', 'meanflowql.py'),
    lock_config=False,
)


# Keys MeanFlowQL needs from the .npz dataset.
_MFQL_KEYS = ('observations', 'next_observations', 'actions', 'rewards',
              'terminals', 'masks')


def load_zermelo_dataset_for_mfql(dataset_path, reward_shift=0.0):
    """Load .npz and return (train_dict, val_dict) shaped for MeanFlowQL."""
    missing_required = ('observations', 'next_observations', 'actions',
                        'rewards', 'terminals', 'masks')
    data, train_segs, val_segs = tc.load_episode_segments(dataset_path)
    for k in missing_required:
        if k not in data:
            raise ValueError(
                f"Dataset missing required key: {k!r}. Regenerate with "
                f"the updated generate_dataset.py."
            )

    train_dict = tc.flatten_segments(data, train_segs, _MFQL_KEYS)
    val_dict = tc.flatten_segments(data, val_segs, _MFQL_KEYS)
    train_dict['rewards'] = train_dict['rewards'] + reward_shift
    val_dict['rewards'] = val_dict['rewards'] + reward_shift

    n_samples = len(data['terminals'])
    print(f"  Reward range (after shift): "
          f"[{train_dict['rewards'].min():.3f}, {train_dict['rewards'].max():.3f}]")
    print(f"  Masks: {train_dict['masks'].sum():.0f} bootstrap, "
          f"{len(train_dict['masks']) - train_dict['masks'].sum():.0f} no-bootstrap "
          f"(train only; {n_samples} total transitions)")
    return train_dict, val_dict


def main(_):
    zermelo_cfg = load_config(FLAGS.zermelo_config)
    zermelo_cfg_src = tc.default_config_src_path(FLAGS.zermelo_config)

    # Wandb.
    exp_name = get_exp_name(FLAGS.seed)
    os.environ["WANDB_MODE"] = "online" if FLAGS.wandb_online else "offline"
    entity = FLAGS.wandb_entity if FLAGS.wandb_entity != 'None' else None
    setup_wandb(project=FLAGS.proj_wandb, group=FLAGS.run_group, name=exp_name,
                entity=entity, mode=os.environ["WANDB_MODE"],
                wandb_output_dir=FLAGS.wandb_save_dir)
    wandb.config.update({'zermelo_config_yaml': zermelo_cfg}, allow_val_change=True)
    if os.path.isfile(zermelo_cfg_src):
        wandb.save(zermelo_cfg_src, policy='now')

    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project,
                                  FLAGS.run_group, exp_name)
    print(f"Saving results to {FLAGS.save_dir}")
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(get_flag_dict(), f, default=str)
    with open(os.path.join(FLAGS.save_dir, 'zermelo_config.json'), 'w') as f:
        json.dump(zermelo_cfg, f, indent=2, default=str)

    config = FLAGS.agent

    # Dataset & env.
    dataset_path = FLAGS.zermelo_dataset or tc.default_dataset_path(zermelo_cfg)
    eval_env = EpisodeMonitor(tc.make_eval_env(FLAGS.zermelo_config))
    env = EpisodeMonitor(tc.make_eval_env(FLAGS.zermelo_config))  # for online fine-tuning

    train_dict, val_dict = load_zermelo_dataset_for_mfql(
        dataset_path, reward_shift=FLAGS.reward_shift)

    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # Clip actions to [-1+eps, 1-eps] (same as MeanFlowQL's make_env_and_datasets).
    action_clip_eps = 1e-5
    train_dict['actions'] = np.clip(train_dict['actions'],
                                    -1 + action_clip_eps, 1 - action_clip_eps)
    val_dict['actions'] = np.clip(val_dict['actions'],
                                  -1 + action_clip_eps, 1 - action_clip_eps)

    train_dataset = Dataset.create(**train_dict)
    val_dataset = Dataset.create(**val_dict)
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
            norm_path = os.path.join(FLAGS.save_dir, 'obs_norm_stats.npz')
            np.savez(norm_path, obs_mean=train_dataset.obs_mean,
                     obs_std=train_dataset.obs_std)
            print(f"Saved observation normalization stats to {norm_path}")
        if val_dataset is not None and train_dataset is not None:
            val_dataset.obs_mean = train_dataset.obs_mean
            val_dataset.obs_std = train_dataset.obs_std
            val_dataset.enable_normalization(True)
        if (replay_buffer is not None and train_dataset is not None
                and replay_buffer is not train_dataset):
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

    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()

    step = 0
    done = True
    expl_metrics = dict()
    online_rng = jax.random.PRNGKey(FLAGS.seed)

    # Early stopping setup.
    maximize_metric = ('return' in FLAGS.early_stopping_metric
                       or 'reward' in FLAGS.early_stopping_metric
                       or 'success' in FLAGS.early_stopping_metric)
    best_metric_value = float('-inf') if maximize_metric else float('inf')
    patience_counter = 0
    early_stopped = False

    # Pretraining phase.
    pretrain_steps = int(FLAGS.offline_steps * FLAGS.pretrain_factor)
    for i in tqdm.tqdm(range(1, pretrain_steps + 1), smoothing=0.1,
                       dynamic_ncols=True, desc="Pretrain"):
        batch = train_dataset.sample(config['batch_size'])
        agent, update_info = agent.pretrain(batch, current_step=i)
        if i % FLAGS.log_interval == 0:
            wandb.log({f'pretraining/{k}': v for k, v in update_info.items()}, step=i)
        if FLAGS.eval_interval != 0 and (i == 1 or i % FLAGS.eval_interval == 0):
            eval_info, _, _ = evaluate(
                agent=agent, env=eval_env, config=config,
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=0, video_frame_skip=FLAGS.video_frame_skip,
                train_dataset=train_dataset,
            )
            wandb.log({f'pretrain_evaluation/{k}': v for k, v in eval_info.items()}, step=i)

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
                agent, update_info = agent.update(
                    batch, full_update=(i % config['actor_freq'] == 0))
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
                    and hasattr(train_dataset, 'normalize_obs')
                    and train_dataset.normalize_obs):
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
                expl_metrics = {f'exploration/{k}': np.mean(v)
                                for k, v in flatten(info).items()}
            step += 1

            if FLAGS.balanced_sampling:
                dataset_batch = train_dataset.sample(config['batch_size'] // 2)
                replay_batch = replay_buffer.sample(config['batch_size'] // 2)
                batch = {k: np.concatenate([dataset_batch[k], replay_batch[k]], axis=0)
                         for k in dataset_batch}
            else:
                batch = train_dataset.sample(config['batch_size'])

            if config['agent_name'] == 'rebrac':
                agent, update_info = agent.update(
                    batch, full_update=(i % config['actor_freq'] == 0), current_step=i)
            else:
                agent, update_info = agent.update(batch, current_step=i)

        # Log training metrics.
        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            if val_dataset is not None:
                val_batch = val_dataset.sample(config['batch_size'])
                _, val_info = agent.total_loss(
                    val_batch, grad_params=None, rng=agent.rng, current_step=i)
                train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})
            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            train_metrics.update(expl_metrics)
            last_time = time.time()
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        # Evaluate.
        if FLAGS.eval_interval != 0 and (i % FLAGS.eval_interval == 0):
            eval_info, _, _ = evaluate(
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
                    print(f"Q-Learning: New best {FLAGS.early_stopping_metric}: "
                          f"{current_metric:.4f}")
                else:
                    patience_counter += 1
                    print(f"Q-Learning: No improvement in {FLAGS.early_stopping_metric}. "
                          f"Patience: {patience_counter}/{FLAGS.early_stopping_patience}")

                if patience_counter >= FLAGS.early_stopping_patience:
                    print(f"Early stopping triggered during Q-Learning at step {i}")
                    early_stopped = True
                    break

        if i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, i)

        if early_stopped:
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
