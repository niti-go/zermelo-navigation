"""Replay 5 random episodes from the most recent .npz dataset and save to video.mp4.

Usage:
    cd ~/zermelo-navigation
    python scripts/visualize.py
"""
import glob
import os
import sys

os.environ['MUJOCO_GL'] = 'egl'

import gymnasium
import imageio
import mujoco
import numpy as np

import zermelo_env  # noqa: register envs
from zermelo_env.zermelo_config import load_config, config_to_env_kwargs

# --- Settings ---
NUM_EPISODES = 3
OUT_PATH = 'datasets/video.mp4'
FPS = 30
RENDER_SIZE = 400
ENV_NAME = 'zermelo-pointmaze-medium-v0'


def find_latest_dataset():
    """Find the most recently modified .npz file under the project root."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    candidates = glob.glob(os.path.join(project_root, '**', '*.npz'), recursive=True)
    # Exclude val datasets.
    candidates = [c for c in candidates if '-val.npz' not in c]
    if not candidates:
        print('No .npz dataset found. Run generate_zermelo.py first.')
        sys.exit(1)
    latest = max(candidates, key=os.path.getmtime)
    return latest


def find_episode_boundaries(terminals):
    ends = np.where(terminals > 0.5)[0]
    episodes = []
    start = 0
    for end in ends:
        episodes.append((start, end + 1))
        start = end + 1
    return episodes


def replay_episode(env, qpos_array, actions_array=None):
    dt = env.unwrapped.frame_skip * env.unwrapped.model.opt.timestep
    frames = []
    for t in range(len(qpos_array)):
        # Set the env's sim clock so render() draws the flow at this episode time.
        env.unwrapped._sim_time = t * dt
        env.unwrapped.data.qpos[:] = qpos_array[t]
        if actions_array is not None and t < len(actions_array):
            env.unwrapped._last_action = np.asarray(actions_array[t], dtype=np.float64)
        mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)
        frame = env.unwrapped.render()
        frames.append(frame)
    return frames


def main():
    dataset_path = find_latest_dataset()
    print(f'Using dataset: {dataset_path}')

    data = np.load(dataset_path)
    terminals = data['terminals']
    qpos = data['qpos']
    actions = data['actions'] if 'actions' in data.files else None
    goal_xy_all = data.get('goal_xy', None)

    episodes = find_episode_boundaries(terminals)
    print(f'{len(episodes)} episodes, {len(terminals)} total steps.')

    # Pick random episodes.
    n = min(NUM_EPISODES, len(episodes))
    chosen = np.random.choice(len(episodes), size=n, replace=False)
    chosen.sort()

    cfg = load_config()
    env_kwargs = config_to_env_kwargs(cfg)
    env = gymnasium.make(ENV_NAME, render_mode='rgb_array', width=RENDER_SIZE, height=RENDER_SIZE, **env_kwargs)
    env.reset()

    all_frames = []
    for idx in chosen:
        start, end = episodes[idx]
        print(f'  Episode {idx}: {end - start} steps')

        # Set goal marker to the actual goal for this episode.
        if goal_xy_all is not None:
            env.unwrapped.model.geom('target').pos[:2] = goal_xy_all[start]
        else:
            # Fallback for old datasets without goal_xy.
            goal_xy = env.unwrapped.ij_to_xy(tuple(cfg['task']['fixed_goal_ij']))
            env.unwrapped.model.geom('target').pos[:2] = goal_xy

        ep_actions = actions[start:end] if actions is not None else None
        frames = replay_episode(env, qpos[start:end], actions_array=ep_actions)
        all_frames.extend(frames)
        # Brief black gap between episodes.
        if idx != chosen[-1]:
            all_frames.extend([np.zeros_like(frames[0])] * 10)

    imageio.mimsave(OUT_PATH, all_frames, fps=FPS)
    print(f'Saved {len(all_frames)} frames to {OUT_PATH}')
    env.close()


if __name__ == '__main__':
    main()
