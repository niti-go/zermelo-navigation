#!/usr/bin/env python3
"""
Evaluate trained MeanFlowQL agent on PointMaze_UMaze-v3.

Must be run in the 'flowrl' conda environment (which has JAX/Flax).

Usage:
    conda activate flowrl
    CUDA_VISIBLE_DEVICES=0 python ~/evaluate_meanflowql.py

Output:
    /ehome/niti/Results/
        meanflowql_results.json  (avg return + all episode returns)
        eval_videos/
            MeanFlowQL/          (3 videos)
"""

import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['MUJOCO_GL'] = 'egl'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import sys
import json
import pickle
import numpy as np
import gymnasium as gym
import gymnasium_robotics
import imageio
from tqdm import tqdm

# =========================================================
# Configuration
# =========================================================
RESULTS_DIR = '/ehome/niti/Results'
VIDEO_DIR = os.path.join(RESULTS_DIR, 'eval_videos')
MEANFLOWQL_RESULTS_FILE = os.path.join(RESULTS_DIR, 'meanflowql_results.json')

ENV_NAME = 'PointMaze_UMaze-v3'
NUM_EVAL_EPISODES = int(os.environ.get('NUM_EVAL_EPISODES', 50))
NUM_VIDEO_EPISODES = 3
MAX_EPISODE_STEPS = int(os.environ.get('MAX_EPISODE_STEPS', 300))
VIDEO_FPS = 30

# MeanFlowQL checkpoint (latest from wandb run 8ujgukn9)
MEANFLOWQL_CHECKPOINT_DIR = '/ehome/niti/offline_training/MeanFlowQL/exp/meanflowql_pointmaze_offline/meanflowql_pointmaze_offline/sd000_20260227_001807'


# =========================================================
# Environment helpers
# =========================================================

def make_flat_eval_env(render_mode=None):
    """Create flattened PointMaze env for MeanFlowQL (expects flat 8D obs)."""
    gym.register_envs(gymnasium_robotics)
    env = gym.make(
        ENV_NAME,
        continuing_task=True,
        reset_target=True,
        render_mode=render_mode,
        max_episode_steps=MAX_EPISODE_STEPS,
    )
    env = gym.wrappers.FlattenObservation(env)
    return env


# =========================================================
# MeanFlowQL evaluation
# =========================================================

def find_latest_checkpoint(checkpoint_dir):
    """Find the checkpoint with the highest step number in a directory."""
    import glob
    checkpoints = glob.glob(os.path.join(checkpoint_dir, 'params_*.pkl'))
    if not checkpoints:
        raise FileNotFoundError(f'No checkpoints found in {checkpoint_dir}')
    def get_step(path):
        basename = os.path.basename(path)
        return int(basename.replace('params_', '').replace('.pkl', ''))
    latest = max(checkpoints, key=get_step)
    print(f'  Latest checkpoint: {latest} (step {get_step(latest)})')
    return latest


def evaluate_meanflowql(checkpoint_dir):
    """Evaluate MeanFlowQL agent. Returns (avg_return, list_of_returns)."""
    import jax
    # Add MeanFlowQL source to path so its imports resolve from anywhere
    mfql_dir = '/ehome/niti/offline_training/MeanFlowQL'
    if mfql_dir not in sys.path:
        sys.path.insert(0, mfql_dir)
    # Also chdir temporarily so relative imports inside MeanFlowQL work
    original_cwd = os.getcwd()
    os.chdir(mfql_dir)

    import flax
    from agents import agents as agent_registry
    from utils.datasets import Dataset

    # Load config
    flags_path = os.path.join(checkpoint_dir, 'flags.json')
    with open(flags_path, 'r') as f:
        flags_dict = json.load(f)

    config = flags_dict['agent']
    seed = flags_dict.get('seed', 0)
    use_obs_norm = flags_dict.get('use_observation_normalization', True)

    # ---------------------------------------------------------------
    # Observation normalization (input standardization)
    # ---------------------------------------------------------------
    # During training, the MeanFlowQL script computes obs_mean and obs_std
    # across the entire offline dataset, then transforms every observation
    # before feeding it to the neural network:
    #
    #     obs_normalized = (obs - obs_mean) / obs_std
    #
    # For pointmaze, the raw flat observation is 8D:
    #   [achieved_goal_x, achieved_goal_y, desired_goal_x, desired_goal_y,
    #    pos_x, pos_y, vel_x, vel_y]
    #
    # These dimensions have different scales (positions ~0-3, velocities
    # ~-0.1 to 0.1), so normalizing to zero-mean unit-variance helps the
    # network learn more stably.
    #
    # This is NOT batch normalization. Batch norm is a learnable layer
    # inside the network that normalizes hidden activations on-the-fly,
    # with running stats stored in the model parameters. What we have here
    # is a fixed preprocessing step applied to the *inputs* before they
    # reach the network — like sklearn's StandardScaler. The stats are
    # computed once from the training data and never updated.
    #
    # Because the network was trained on normalized inputs, we MUST apply
    # the exact same transform at eval time. Feeding raw observations would
    # produce meaningless actions. The stats are NOT saved inside the model
    # checkpoint — they lived on the Dataset object during training. We now
    # save them to obs_norm_stats.npz alongside checkpoints (added to
    # meanflowql_pointmaze.py). For older runs without this file, we fall
    # back to recomputing from the Minari dataset (slow but only once).
    # ---------------------------------------------------------------
    norm_stats_path = os.path.join(checkpoint_dir, 'obs_norm_stats.npz')
    obs_mean, obs_std = None, None
    if use_obs_norm and os.path.exists(norm_stats_path):
        stats = np.load(norm_stats_path)
        obs_mean = stats['obs_mean']
        obs_std = stats['obs_std']
        print(f'  Loaded normalization stats from {norm_stats_path}')
    elif use_obs_norm:
        # Fallback: recompute from dataset (slow, for old checkpoints without saved stats)
        print(f'  obs_norm_stats.npz not found, recomputing from dataset (slow)...')
        from meanflowql_pointmaze import load_minari_dataset
        train_data, _ = load_minari_dataset('D4RL/pointmaze/umaze-v2')
        tmp_dataset = Dataset.create(**train_data)
        tmp_dataset.compute_normalization_stats()
        obs_mean = tmp_dataset.obs_mean
        obs_std = tmp_dataset.obs_std
        # Save for next time
        np.savez(norm_stats_path, obs_mean=obs_mean, obs_std=obs_std)
        print(f'  Saved normalization stats to {norm_stats_path} for future use')
        del tmp_dataset, train_data

    if use_obs_norm:
        print(f'  Observation normalization enabled')

    # ---------------------------------------------------------------
    # Workaround: setting absl FLAGS for agent.create()
    # ---------------------------------------------------------------
    # During training, the learning rate follows a schedule:
    #   1. Warmup phase: LR ramps up from ~0 to base_lr (first 5% of steps)
    #   2. Cosine decay: LR gradually decreases to a small min_lr
    # This helps training converge — a high LR early on would cause
    # instability, and a decaying LR later helps fine-tune.
    #
    # To build this schedule, agent.create() (in meanflowql.py) needs to
    # know the total training duration, so it reads these absl FLAGS:
    #   - offline_steps (1M)   — how many offline gradient updates
    #   - online_steps (0)     — how many online fine-tuning steps
    #   - pretrain_factor (0)  — fraction of steps for pretraining
    #
    # During inference, we don't need any learning rates at all,
    # but this is a code design issue in the MeanFlowQL codebase.
    # To build the agent object, even just for inference, we have to provide
    # network architecture, optimizer, AND LR schedule. There's no way
    # to create just the network without also building the optimizer.
    # So even though we're only evaluating (no learning, no gradients,
    # no LR needed), we still have to provide these values or it crashes.
    #
    # The optimizer and LR schedule created here are immediately
    # discarded when we overwrite the agent with checkpoint weights.
    # They have zero effect on evaluation — this is purely to be
    # able to create the agent: agent.create().
    # ---------------------------------------------------------------
    from absl import flags as absl_flags
    FLAGS = absl_flags.FLAGS
    for flag_name, default in [
        ('offline_steps', flags_dict.get('offline_steps', 1000000)),
        ('online_steps', flags_dict.get('online_steps', 0)),
    ]:
        if flag_name not in FLAGS:
            absl_flags.DEFINE_integer(flag_name, default, '')
    for flag_name, default in [
        ('pretrain_factor', flags_dict.get('pretrain_factor', 0.0)),
    ]:
        if flag_name not in FLAGS:
            absl_flags.DEFINE_float(flag_name, default, '')
    try:
        FLAGS(sys.argv[:1])
    except Exception:
        pass
    FLAGS['offline_steps'].value = flags_dict.get('offline_steps', 1000000)
    FLAGS['online_steps'].value = flags_dict.get('online_steps', 0)
    FLAGS['pretrain_factor'].value = flags_dict.get('pretrain_factor', 0.0)

    # Create agent skeleton — only need obs/action shapes, use dummy data
    obs_dim = len(obs_mean) if obs_mean is not None else 8
    act_dim = 2
    dummy_obs = np.zeros((1, obs_dim), dtype=np.float32)
    dummy_act = np.zeros((1, act_dim), dtype=np.float32)
    agent_class = agent_registry[config['agent_name']]
    agent = agent_class.create(seed, dummy_obs, dummy_act, config)

    # Restore checkpoint
    checkpoint_path = find_latest_checkpoint(checkpoint_dir)
    with open(checkpoint_path, 'rb') as f:
        load_dict = pickle.load(f)
    agent = flax.serialization.from_state_dict(agent, load_dict['agent'])
    print(f'  Agent restored successfully')

    # Helper to get action from agent
    rng = jax.random.PRNGKey(seed)
    def get_action(obs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        obs_batch = obs[None, :] if obs.ndim == 1 else obs
        if use_obs_norm:
            obs_batch = (obs_batch - obs_mean) / obs_std
        action = agent.sample_actions(observations=obs_batch, seed=key)
        action = np.array(action)
        if action.ndim > 1 and action.shape[0] == 1:
            action = action[0]
        return np.clip(action, -1, 1)

    # --- eval episodes (no render) ---
    env = make_flat_eval_env(render_mode=None)
    all_returns = []
    pbar = tqdm(range(NUM_EVAL_EPISODES), desc='  MeanFlowQL', unit='ep')
    for ep in pbar:
        obs, info = env.reset()
        ep_return = 0.0
        done = False
        while not done:
            action = get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            done = terminated or truncated
        all_returns.append(ep_return)
        avg_so_far = np.mean(all_returns)
        pbar.set_postfix(avg=f'{avg_so_far:.2f}', last=f'{ep_return:.0f}')
    pbar.close()
    env.close()

    # --- 3 video episodes ---
    video_env = make_flat_eval_env(render_mode='rgb_array')
    video_dir = os.path.join(VIDEO_DIR, 'MeanFlowQL')
    os.makedirs(video_dir, exist_ok=True)

    for ep in range(NUM_VIDEO_EPISODES):
        obs, info = video_env.reset()
        frames = []
        done = False
        while not done:
            action = get_action(obs)
            obs, reward, terminated, truncated, info = video_env.step(action)
            frame = video_env.render()
            if frame is not None:
                frames.append(frame)
            done = terminated or truncated
        if frames:
            path = os.path.join(video_dir, f'episode_{ep + 1}.mp4')
            imageio.mimsave(path, frames, fps=VIDEO_FPS)
            print(f'    Saved video: {path}')
    video_env.close()

    avg_ret = float(np.mean(all_returns))

    # Restore original working directory
    os.chdir(original_cwd)

    return avg_ret, all_returns


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(VIDEO_DIR, exist_ok=True)

    print('\n' + '=' * 60)
    print('Evaluating: MeanFlowQL')
    print('=' * 60)
    avg_ret, all_returns = evaluate_meanflowql(MEANFLOWQL_CHECKPOINT_DIR)
    print(f'  Avg return over {NUM_EVAL_EPISODES} episodes: {avg_ret:.3f}')

    # Save results as JSON for the merge script
    results = {
        'MeanFlowQL': {
            'avg_return': avg_ret,
            'all_returns': all_returns,
        }
    }
    with open(MEANFLOWQL_RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nMeanFlowQL results saved to: {MEANFLOWQL_RESULTS_FILE}')


if __name__ == '__main__':
    main()
