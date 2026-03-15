#!/usr/bin/env python3
"""
Evaluate SB3 baseline models (PPO, SAC, TD3, DDPG) on PointMaze_UMaze-v3.

Must be run in the 'pointmaze' conda environment (which has the matching
numpy version used when these models were saved).

Usage:
    conda activate pointmaze
    python ~/evaluate_sb3.py

Output:
    /ehome/niti/Results/
        sb3_results.json       (per-algorithm avg returns + all episode returns)
        eval_videos/
            DDPG/              (3 videos)
            TD3/               (3 videos)
            SAC/               (3 videos)
            PPO_config_1/      (3 videos)
            PPO_config_2/      (3 videos)
"""

import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['MUJOCO_GL'] = 'egl'

import json
import numpy as np
import gymnasium as gym
import gymnasium_robotics
import imageio
from tqdm import tqdm
from stable_baselines3 import PPO, SAC, TD3, DDPG

# =========================================================
# Configuration
# =========================================================
RESULTS_DIR = '/ehome/niti/Results'
VIDEO_DIR = os.path.join(RESULTS_DIR, 'eval_videos')
SB3_RESULTS_FILE = os.path.join(RESULTS_DIR, 'sb3_results.json')

ENV_NAME = 'PointMaze_UMaze-v3'
NUM_EVAL_EPISODES = int(os.environ.get('NUM_EVAL_EPISODES', 50))
NUM_VIDEO_EPISODES = 3
MAX_EPISODE_STEPS = int(os.environ.get('MAX_EPISODE_STEPS', 300))
VIDEO_FPS = 30

SB3_MODELS = {
    'DDPG':         '/ehome/niti/newattempt/experiments/DDPG/config_1/models/best_model.zip',
    'TD3':          '/ehome/niti/newattempt/experiments/TD3/config_1/models/best_model.zip',
    'SAC':          '/ehome/niti/newattempt/experiments/SAC/config_1/models/best_model.zip',
    'PPO_config_1': '/ehome/niti/newattempt/experiments/PPO/config_1/models/best_model.zip',
    'PPO_config_2': '/ehome/niti/newattempt/experiments/PPO/config_2/models/best_model.zip',
}

ALGO_MAP = {'PPO': PPO, 'SAC': SAC, 'TD3': TD3, 'DDPG': DDPG}


def make_eval_env(render_mode=None):
    gym.register_envs(gymnasium_robotics)
    return gym.make(
        ENV_NAME,
        continuing_task=True,
        reset_target=True,
        render_mode=render_mode,
        max_episode_steps=MAX_EPISODE_STEPS,
    )


def evaluate_sb3(algo_name, model_path):
    """Evaluate one SB3 model. Returns (avg_return, list_of_returns)."""
    # Determine SB3 class from algo name
    for key, cls in ALGO_MAP.items():
        if key in algo_name.upper():
            AlgoClass = cls
            break

    model = AlgoClass.load(model_path)
    print(f'  Loaded model from {model_path}')

    # --- eval episodes (no render) ---
    env = make_eval_env(render_mode=None)
    all_returns = []
    pbar = tqdm(range(NUM_EVAL_EPISODES), desc=f'  {algo_name}', unit='ep')
    for ep in pbar:
        obs, info = env.reset()
        ep_return = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            done = terminated or truncated
        all_returns.append(ep_return)
        avg_so_far = np.mean(all_returns)
        pbar.set_postfix(avg=f'{avg_so_far:.2f}', last=f'{ep_return:.0f}')
    pbar.close()
    env.close()

    # --- 3 video episodes ---
    video_env = make_eval_env(render_mode='rgb_array')
    video_dir = os.path.join(VIDEO_DIR, algo_name)
    os.makedirs(video_dir, exist_ok=True)

    for ep in range(NUM_VIDEO_EPISODES):
        obs, info = video_env.reset()
        frames = []
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
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
    return avg_ret, all_returns


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(VIDEO_DIR, exist_ok=True)

    results = {}

    for algo_name, model_path in SB3_MODELS.items():
        print('\n' + '=' * 60)
        print(f'Evaluating: {algo_name}')
        print('=' * 60)
        if not os.path.exists(model_path):
            print(f'  WARNING: Model not found at {model_path}, skipping.')
            continue
        avg_ret, all_returns = evaluate_sb3(algo_name, model_path)
        results[algo_name] = {
            'avg_return': avg_ret,
            'all_returns': all_returns,
        }
        print(f'  Avg return over {NUM_EVAL_EPISODES} episodes: {avg_ret:.3f}')

    # Save results as JSON for the merge script
    with open(SB3_RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSB3 results saved to: {SB3_RESULTS_FILE}')


if __name__ == '__main__':
    main()
