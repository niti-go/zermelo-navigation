'''
Trains MeanFlowQL on an offline dataset of pointmaze umaze from minari.
I will compare this to the performances of online RL baselines on the online environment.
(previously saved to tensorboard logs)

Command to set GPU and run:

cd ~/offline_training/MeanFlowQL
CUDA_VISIBLE_DEVICES=0 python meanflowql_pointmaze.py \
--env_name=pointmaze-umaze-v3 \
--agent=agents/meanflowql.py \
--offline_steps=1000000 \
--seed=0 \
--save_interval=50000 \
--agent.alpha=2000 \
--agent.num_candidates=5 \
--proj_wandb=meanflowql_pointmaze_offline \
--run_group=meanflowql_pointmaze_offline \
--dataset_source=minari \
--wandb_online=True

# 1. Start tmux
tmux new -s meanflowql_pointmaze
conda activate flowrl
wandb login

# 7. Detach: Ctrl+b, then d

# 8. Reattach later: tmux attach -t baselines
'''
import os
import platform
# Set OpenGL platform to EGL for headless rendering
os.environ['PYOPENGL_PLATFORM'] = 'egl'
# Configure MuJoCo to use EGL renderer
os.environ['MUJOCO_GL'] = 'egl'
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# # Limit GPU memory usage to 50%
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.5'
import json
import random
import time

import jax
import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags
import gymnasium
from agents import agents
from envs.env_utils import EpisodeMonitor
from utils.datasets import Dataset, ReplayBuffer
from utils.evaluation import evaluate, flatten
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, get_wandb_video, setup_wandb


FLAGS = flags.FLAGS

# Experiment configuration flags
flags.DEFINE_string('run_group', 'debug', 'Run group in wandb')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'pointmaze-umaze-v3', 'Environment (dataset) name. Used for reference; actual env is recovered from Minari dataset.')
flags.DEFINE_string('proj_wandb', 'pointmaze_comparison', 'wandb project name')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('wandb_save_dir', 'debug/', 'Wandb offline data save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')
flags.DEFINE_boolean('use_observation_normalization', True, 'Whether to normalize observations')

# Dataset configuration flags
flags.DEFINE_string('dataset_source', 'minari', 'Dataset source: minari, d4rl, or custom')
flags.DEFINE_string('minari_dataset_name', None, 'Minari dataset name (auto-inferred from env_name if None)')

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
flags.DEFINE_boolean('wandb_online', True , 'Whether to use wandb online mode')

# Early stopping configuration flags
flags.DEFINE_bool('enable_early_stopping', True, 'Whether to enable early stopping.')
flags.DEFINE_integer('early_stopping_patience', 6, 'Number of evaluations to wait before early stopping.')
flags.DEFINE_float('early_stopping_min_delta', 0.01, 'Minimum change in validation metric to qualify as improvement.')
flags.DEFINE_string('early_stopping_metric', 'evaluation/episode.return', 'Metric to monitor for early stopping.')

config_flags.DEFINE_config_file('agent', 'agents/meanfql_dit_inv_simp_dyna_alpha.py', lock_config=False)


def load_minari_dataset(minari_dataset_name, train_test_split=0.8):
    """
    Load dataset from Minari and convert to the format expected by Dataset/ReplayBuffer.

    The expected format matches what d4rl_utils.get_dataset() produces:
      - observations: (N, obs_dim) float32
      - actions: (N, act_dim) float32
      - next_observations: (N, obs_dim) float32
      - rewards: (N,) float32
      - terminals: (N,) float32 - marks episode boundaries (1.0 on last step)
      - masks: (N,) float32 - bootstrapping mask (1.0 = bootstrap, 0.0 = don't)
        For truncated episodes, mask=1 (should bootstrap).
        For terminated episodes, mask=0 (absorbing state).

    Pointmaze observations are dicts with keys:
      'achieved_goal' (2,), 'desired_goal' (2,), 'observation' (4,)
    These are flattened into a single (8,) vector.
    """
    try:
        import minari
    except ImportError:
        raise ImportError("Minari not installed. Install with: pip install minari")

    print(f"Loading Minari dataset: {minari_dataset_name}")
    dataset = minari.load_dataset(minari_dataset_name, download=True)

    observations = []
    actions = []
    rewards = []
    terminals = []
    masks = []
    next_observations = []

    def flatten_obs(obs_dict, i):
        """Flatten a dict observation at timestep i into a 1D float32 array."""
        if isinstance(obs_dict, dict):
            return np.concatenate([obs_dict[k][i] for k in sorted(obs_dict.keys())]).astype(np.float32)
        return obs_dict[i].astype(np.float32)

    print("Converting Minari dataset to standard format...")
    # Track episode boundaries for episode-level train/test split
    episode_start_indices = []
    episode_end_indices = []  # exclusive
    transition_count = 0

    for episode_idx, episode in enumerate(dataset.iterate_episodes()):
        obs = episode.observations   # dict with T+1 entries
        acts = episode.actions        # (T, act_dim)
        rews = episode.rewards        # (T,)
        terms = episode.terminations  # (T,) bool
        truncs = episode.truncations  # (T,) bool
        T = len(rews)

        episode_start_indices.append(transition_count)
        for i in range(T):
            observations.append(flatten_obs(obs, i))
            next_observations.append(flatten_obs(obs, i + 1))
            actions.append(acts[i].astype(np.float32))
            rewards.append(float(rews[i]))
            # terminals: marks episode boundaries (1.0 on last step of episode)
            terminals.append(1.0 if i == T - 1 else 0.0)
            # masks: bootstrapping flag.
            # Only set mask=0 for TRUE terminations (absorbing states).
            # Truncations (goal reached in continuing_task) should still bootstrap
            # (mask=1), same as antmaze handling in d4rl_utils.py.
            masks.append(0.0 if terms[i] else 1.0)
            transition_count += 1
        episode_end_indices.append(transition_count)

        if (episode_idx + 1) % 1000 == 0:
            print(f"  Processed {episode_idx + 1} episodes...")

    # Shift rewards: sparse binary (0/1) -> (-1/0), matching antmaze handling in d4rl_utils.py.
    # This gives the Q-function a strong negative cost-to-go signal in the continuing task setting.
    rewards = np.array(rewards, dtype=np.float32) - 1.0

    dataset_dict = {
        'observations': np.array(observations, dtype=np.float32),
        'actions': np.array(actions, dtype=np.float32),
        'next_observations': np.array(next_observations, dtype=np.float32),
        'rewards': rewards,
        'terminals': np.array(terminals, dtype=np.float32),
        'masks': np.array(masks, dtype=np.float32),
    }

    # Train-test split by episode (preserves temporal structure within each split)
    n_samples = len(observations)
    num_episodes = len(episode_start_indices)
    episode_indices = np.arange(num_episodes)
    np.random.shuffle(episode_indices)

    train_ep_count = int(num_episodes * train_test_split)
    train_ep_indices = episode_indices[:train_ep_count]
    val_ep_indices = episode_indices[train_ep_count:]

    # Gather transition indices for each split
    train_trans_indices = np.concatenate([
        np.arange(episode_start_indices[ep], episode_end_indices[ep])
        for ep in sorted(train_ep_indices)
    ])
    val_trans_indices = np.concatenate([
        np.arange(episode_start_indices[ep], episode_end_indices[ep])
        for ep in sorted(val_ep_indices)
    ])

    train_dataset = {k: v[train_trans_indices] for k, v in dataset_dict.items()}
    val_dataset = {k: v[val_trans_indices] for k, v in dataset_dict.items()}

    print(f"\nDataset statistics:")
    print(f"  Total transitions: {n_samples}")
    print(f"  Total episodes: {num_episodes}")
    print(f"  Training: {len(train_trans_indices)} transitions from {len(train_ep_indices)} episodes ({100*len(train_trans_indices)/n_samples:.1f}%)")
    print(f"  Validation: {len(val_trans_indices)} transitions from {len(val_ep_indices)} episodes ({100*len(val_trans_indices)/n_samples:.1f}%)")
    print(f"  Observation shape: {dataset_dict['observations'][0].shape}")
    print(f"  Action shape: {dataset_dict['actions'][0].shape}")
    print(f"  Reward range: [{dataset_dict['rewards'].min():.3f}, {dataset_dict['rewards'].max():.3f}]")
    print(f"  Masks: {dataset_dict['masks'].sum():.0f}/{n_samples} bootstrap, {n_samples - dataset_dict['masks'].sum():.0f} no-bootstrap (episode boundaries)")

    return train_dataset, val_dataset


def infer_minari_dataset_name(env_name):
    """
    Infer Minari dataset name from environment name.
    
    Uses the D4RL naming convention: "D4RL/domain/task-vN"
    
    Args:
        env_name: Environment name (e.g., 'pointmaze-umaze-v3')
        
    Returns:
        Minari dataset name in D4RL format
    """
    # Mapping of environment names to D4RL Minari dataset names
    env_to_minari = {
        'pointmaze-umaze-v3': 'D4RL/pointmaze/umaze-v2',
        'pointmaze-medium-v3': 'D4RL/pointmaze/medium-v2',
        'pointmaze-large-v3': 'D4RL/pointmaze/large-v2',
        'pointmaze-umaze-dense-v3': 'D4RL/pointmaze/umaze-dense-v2',
        'pointmaze-medium-dense-v3': 'D4RL/pointmaze/medium-dense-v2',
        'pointmaze-large-dense-v3': 'D4RL/pointmaze/large-dense-v2',
        'antmaze-umaze-v2': 'D4RL/antmaze/umaze-v1',
        'antmaze-medium-v2': 'D4RL/antmaze/medium-v1',
    }
    
    if env_name in env_to_minari:
        return env_to_minari[env_name]
    else:
        print(f"Warning: No explicit mapping for {env_name}.")
        print("Available D4RL/pointmaze datasets:")
        print("  - D4RL/pointmaze/umaze-v2")
        print("  - D4RL/pointmaze/medium-v2")
        print("  - D4RL/pointmaze/large-v2")
        print("  - D4RL/pointmaze/umaze-dense-v2")
        print("  - D4RL/pointmaze/medium-dense-v2")
        print("  - D4RL/pointmaze/large-dense-v2")
        # Default to umaze if unsure
        return 'D4RL/pointmaze/umaze-v2'


def main(_):
    """
    Main training function for MeanFlow RL on Pointmaze with early stopping.
    
    This code is adapted for Pointmaze environments with Minari dataset support.
    """
    # Initialize experiment logging
    exp_name = get_exp_name(FLAGS.seed)
    import os
    # Configure wandb mode based on command line arguments
    if FLAGS.wandb_online:
        os.environ["WANDB_MODE"] = "online"
    else:
        os.environ["WANDB_MODE"] = "offline"
    setup_wandb(project=FLAGS.proj_wandb, group=FLAGS.run_group, name=exp_name, mode=os.environ["WANDB_MODE"], wandb_output_dir=FLAGS.wandb_save_dir)
    
    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, exp_name)
    print(f"Saving results to {FLAGS.save_dir}") 
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    # Initialize environment and datasets
    config = FLAGS.agent
    
    # Resolve dataset name before any loading
    if FLAGS.minari_dataset_name is None:
        FLAGS.minari_dataset_name = infer_minari_dataset_name(FLAGS.env_name)

    # Load Minari dataset first to recover environment
    print(f"Loading Minari dataset: {FLAGS.minari_dataset_name}")
    import minari
    minari_dataset = minari.load_dataset(FLAGS.minari_dataset_name, download=True)

    # Recover the environment from dataset metadata (creates Gymnasium-Robotics env).
    # The pointmaze env returns dict observations {achieved_goal, desired_goal, observation}.
    # We wrap with FlattenObservation to get a flat (8,) array matching our dataset format,
    # and EpisodeMonitor to track episode stats needed for evaluation metrics.
    # The recovered env has max_episode_steps=1M which is unusable for evaluation,
    # so we re-wrap with a reasonable time limit.
    print("Reconstructing environment from dataset metadata...")

    def _make_pointmaze_env(minari_ds, max_episode_steps=300):
        """Recover pointmaze env with proper wrappers."""
        raw = minari_ds.recover_environment()
        # Strip the existing TimeLimit (max=1M) and re-wrap with a sane limit.
        # Wrapper chain: TimeLimit -> OrderEnforcing -> PassiveEnvChecker -> PointMazeEnv
        inner = raw.env  # unwrap TimeLimit
        env = gymnasium.wrappers.TimeLimit(inner, max_episode_steps=max_episode_steps)
        env = gymnasium.wrappers.FlattenObservation(env)
        env = EpisodeMonitor(env)
        return env

    eval_env = _make_pointmaze_env(minari_dataset)
    env = _make_pointmaze_env(minari_dataset)

    # Load and convert dataset
    if FLAGS.dataset_source == 'minari':
        train_dataset, val_dataset = load_minari_dataset(FLAGS.minari_dataset_name)
    else:
        raise ValueError(f"Dataset source '{FLAGS.dataset_source}' not yet supported. Use 'minari'.")

    # Set random seeds for reproducibility
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # Clip dataset actions to [-1+eps, 1-eps] (same as make_env_and_datasets does)
    action_clip_eps = 1e-5
    train_dataset['actions'] = np.clip(train_dataset['actions'], -1 + action_clip_eps, 1 - action_clip_eps)
    val_dataset['actions'] = np.clip(val_dataset['actions'], -1 + action_clip_eps, 1 - action_clip_eps)

    # Configure datasets and replay buffer
    train_dataset = Dataset.create(**train_dataset)
    val_dataset = Dataset.create(**val_dataset)
    if FLAGS.balanced_sampling:
        # Create separate replay buffer for balanced sampling between training dataset and replay buffer
        example_transition = {k: v[0] for k, v in train_dataset.items()}
        replay_buffer = ReplayBuffer.create(example_transition, size=FLAGS.buffer_size)
    else:
        # Use training dataset as the replay buffer
        train_dataset = ReplayBuffer.create_from_initial_dataset(
            dict(train_dataset), size=max(FLAGS.buffer_size, train_dataset.size + 1)
        )
        replay_buffer = train_dataset
    
    # Configure data augmentation and frame stacking for all datasets
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
            # Save normalization stats alongside checkpoints for eval-time use
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
    
    # Initialize agent
    example_batch = train_dataset.sample(1)

    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )
    
    # Print agent parameter statistics
    agent.print_param_stats()

    # Restore agent from checkpoint if specified
    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    # Initialize training loggers and timing
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()

    step = 0
    done = True
    expl_metrics = dict()
    online_rng = jax.random.PRNGKey(FLAGS.seed)
    
    # Initialize early stopping variables
    best_metric_value = float('-inf')  # For metrics where higher is better (like success)
    patience_counter = 0
    early_stopped = False
    
    # Determine if metric should be maximized or minimized
    maximize_metric = 'return' in FLAGS.early_stopping_metric or 'reward' in FLAGS.early_stopping_metric or 'success' in FLAGS.early_stopping_metric
    if not maximize_metric:
        best_metric_value = float('inf')  # For metrics where lower is better (like loss)

    # Pretraining phase
    pretrain_steps = int(FLAGS.offline_steps * FLAGS.pretrain_factor)
    for i in tqdm.tqdm(range(1, pretrain_steps + 1), smoothing=0.1, dynamic_ncols=True, desc="Pretrain"):
        batch = train_dataset.sample(config['batch_size'])
        agent, update_info = agent.pretrain(batch, current_step=i)
        
        # Log pretraining metrics
        if i % FLAGS.log_interval == 0:
            train_metrics = {f'pretraining/{k}': v for k, v in update_info.items()}
            wandb.log(train_metrics, step=i)
        
        # Evaluate agent during pretraining
        if FLAGS.eval_interval != 0 and (i == 1 or i % FLAGS.eval_interval == 0):
            renders = []
            eval_metrics = {}
                                       
            eval_info, trajs, cur_renders = evaluate(
                agent=agent,
                env=eval_env,
                config=config,
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=0,  # Disabled for Pointmaze
                video_frame_skip=FLAGS.video_frame_skip,
                train_dataset=train_dataset,
            )
            renders.extend(cur_renders)
            for k, v in eval_info.items():
                eval_metrics[f'pretrain_evaluation/{k}'] = v
                
            wandb.log(eval_metrics, step=i)


    # Q-Learning phase
    q_learning_steps = FLAGS.offline_steps + FLAGS.online_steps
    
    # Reset early stopping variables for Q-Learning phase
    best_metric_value = float('-inf') if maximize_metric else float('inf')
    patience_counter = 0
    early_stopped = False
    
    for i in tqdm.tqdm(range(pretrain_steps, q_learning_steps + pretrain_steps + 1), smoothing=0.1, dynamic_ncols=True, desc="Q-Learning"):
        if i <= FLAGS.offline_steps + pretrain_steps:
            # Offline reinforcement learning
            batch = train_dataset.sample(config['batch_size'])
    
            if config['agent_name'] == 'rebrac':
                agent, update_info = agent.update(
                    batch, full_update=(i % config['actor_freq'] == 0)
                )
            else:
                agent, update_info = agent.update(
                    batch, current_step=i
                )
        else:
            # Online fine-tuning phase
            online_rng, key = jax.random.split(online_rng)

            if done:
                step = 0
                ob, _ = env.reset()
            
            # Ensure observation has correct shape before calling sample_actions
            if len(ob.shape) == 1:
                ob_batch = ob[None, :]  # Add batch dimension
            else:
                ob_batch = ob
            
            # Apply observation normalization if enabled
            if FLAGS.use_observation_normalization and train_dataset is not None and hasattr(train_dataset, 'normalize_obs') and train_dataset.normalize_obs:
                ob_batch = (ob_batch - train_dataset.obs_mean) / train_dataset.obs_std
            
            action = agent.sample_actions(observations=ob_batch, seed=key)
            action = np.array(action)
            
            # Remove batch dimension if present for environment step
            if action.ndim > 1 and action.shape[0] == 1:
                action = action[0]  # Remove batch dimension
            
            next_ob, reward, terminated, truncated, info = env.step(action.copy())
            done = terminated or truncated
            reward = reward - 1.0

            # Store transition in replay buffer (use original unnormalized observations)
            replay_buffer.add_transition(
                dict(
                    observations=ob,
                    actions=action,
                    rewards=reward,
                    terminals=float(done),
                    masks=1.0 - terminated,
                    next_observations=next_ob,
                )
            )
            ob = next_ob

            if done:
                expl_metrics = {f'exploration/{k}': np.mean(v) for k, v in flatten(info).items()}

            step += 1

            # Update agent with appropriate sampling strategy
            if FLAGS.balanced_sampling:
                # Sample half from training dataset and half from replay buffer
                dataset_batch = train_dataset.sample(config['batch_size'] // 2)
                replay_batch = replay_buffer.sample(config['batch_size'] // 2)
                batch = {k: np.concatenate([dataset_batch[k], replay_batch[k]], axis=0) for k in dataset_batch}
            else:
                batch = train_dataset.sample(config['batch_size'])

            if config['agent_name'] == 'rebrac':
                agent, update_info = agent.update(
                    batch, full_update=(i % config['actor_freq'] == 0),
                    current_step=i
                )
            else:
                agent, update_info = agent.update(
                    batch, current_step=i
                )

        # Log training metrics
        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            if val_dataset is not None:
                val_batch = val_dataset.sample(config['batch_size'])
                # Compute validation loss with required rng parameter
                _, val_info = agent.total_loss(val_batch, grad_params=None, rng=agent.rng, current_step=i)
                train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})
            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            train_metrics.update(expl_metrics)
            last_time = time.time()
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        # Evaluate agent performance
        if FLAGS.eval_interval != 0 and (i % FLAGS.eval_interval == 0):
            renders = []
            eval_metrics = {}
            eval_info, trajs, cur_renders = evaluate(
                agent=agent,
                env=eval_env,
                config=config,
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=0,  # Disabled for Pointmaze
                video_frame_skip=FLAGS.video_frame_skip,
                train_dataset=train_dataset,
            )
            renders.extend(cur_renders)
            for k, v in eval_info.items():
                eval_metrics[f'evaluation/{k}'] = v
                
            wandb.log(eval_metrics, step=i)
            eval_logger.log(eval_metrics, step=i)
            
            # Early stopping check
            if FLAGS.enable_early_stopping and FLAGS.early_stopping_metric in eval_metrics:
                current_metric = eval_metrics[FLAGS.early_stopping_metric]
                
                # Standard early stopping logic based on metric improvement
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
                    print(f"Q-Learning: No improvement in {FLAGS.early_stopping_metric}. Patience: {patience_counter}/{FLAGS.early_stopping_patience}")

                if patience_counter >= FLAGS.early_stopping_patience:
                    print(f"Early stopping triggered during Q-Learning at step {i}")
                    early_stopped = True
                    break

        # Save agent checkpoint
        if i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, i)
            
        # Exit training loop if early stopping was triggered
        if early_stopped:
            print(f"Q-Learning: Early stopping triggered at step {i}")
            wandb.log({
                'early_stopping/step': i,
                'early_stopping/metric': FLAGS.early_stopping_metric,
                'early_stopping/best_metric_value': best_metric_value,
            })
            train_logger.log(
                {
                    'early_stopping/step': i,
                    'early_stopping/metric': FLAGS.early_stopping_metric,
                    'early_stopping/best_metric_value': best_metric_value,
                }
            )
            break

    # Cleanup and finalize logging
    wandb.finish()
    train_logger.close()
    eval_logger.close()


if __name__ == '__main__':
    app.run(main)