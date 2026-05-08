#!/usr/bin/env python3
"""
Record evaluation videos for trained online RL models.
=======================================================
Loads best_model.zip from PPO_online/, TD3_online/, SAC_online/ and
records ~30-second evaluation videos in each folder as eval_video.mp4.

Each video runs multiple episodes back-to-back until ~900 frames
(30 seconds at 30 fps) are collected.

Usage:
  python record_eval_videos.py
  python record_eval_videos.py --algos PPO_online SAC_online
  python record_eval_videos.py --max_frames 1800  # 60 seconds
"""

import os
#os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
#os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ["MUJOCO_GL"] = "egl"
os.environ["EGL_DEVICE_ID"] = "0"

import argparse
import traceback
from pathlib import Path

import numpy as np
import imageio

import gym
import d4rl  # noqa: registers antmaze envs

import gymnasium
from stable_baselines3 import PPO, SAC, TD3

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

ENV_NAME   = "antmaze-umaze-v2"
FPS        = 30
MAX_FRAMES = 900   # 30 seconds at 30 fps

ALGO_CLASSES = {
    "PPO_online": PPO,
    "SAC_online": SAC,
    "TD3_online": TD3,
}

# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────

def make_env():
    """Create antmaze env via gymnasium bridge (same as training scripts)."""
    return gymnasium.make("GymV21Environment-v0", env_id=ENV_NAME)


def render_frame(env):
    """Try multiple render methods to get an RGB frame."""
    for render_fn in [
        lambda: env.render(),
        lambda: env.unwrapped.render(mode="rgb_array"),
        lambda: env.unwrapped.env.render(mode="rgb_array"),
    ]:
        try:
            f = render_fn()
            if isinstance(f, np.ndarray) and f.ndim == 3:
                return f
        except Exception:
            pass
    return None

# ─────────────────────────────────────────────────────────────────────────────
# Video recording
# ─────────────────────────────────────────────────────────────────────────────

def record_video(model, output_path: Path, max_frames: int = MAX_FRAMES, fps: int = FPS):
    """Record multiple episodes until max_frames is reached.

    Prints per-episode stats (return, steps, success) and overall summary.
    """
    env = make_env()
    frames = []
    episode = 0
    total_successes = 0

    print(f"  Recording (target: {max_frames} frames / {max_frames/fps:.0f}s) ...")

    while len(frames) < max_frames:
        obs, _ = env.reset()
        episode += 1
        ep_return = 0.0
        ep_steps = 0
        success = False
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            ep_steps += 1
            done = terminated or truncated

            # Raw antmaze reward: 1.0 at goal, 0.0 elsewhere
            if terminated and reward >= 0.5:
                success = True

            frame = render_frame(env)
            if frame is not None:
                frames.append(frame)

            # Stop if we've collected enough frames
            if len(frames) >= max_frames:
                break

        total_successes += int(success)
        status = "GOAL" if success else "timeout"
        print(f"    Episode {episode}: {ep_steps} steps, "
              f"return {ep_return:+.1f}, {status}")

    env.close()

    if not frames:
        print("  WARNING: No frames captured — rendering may not be available.")
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(output_path), frames, fps=fps)

    duration = len(frames) / fps
    print(f"  Saved: {output_path} ({len(frames)} frames, {duration:.1f}s, "
          f"{episode} episodes, {total_successes}/{episode} successes)")
    return len(frames)

# ─────────────────────────────────────────────────────────────────────────────
# Quick evaluation (success rate)
# ─────────────────────────────────────────────────────────────────────────────

def quick_eval(model, n_episodes: int = 20):
    """Run a quick eval to print success rate before recording video."""
    env = make_env()
    successes = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        success = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if terminated and reward >= 0.5:
                success = True
        successes.append(float(success))

    env.close()
    rate = np.mean(successes)
    print(f"  Quick eval ({n_episodes} episodes): {rate:.0%} success rate")
    return rate

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Record eval videos for trained models")
    p.add_argument("--algos", nargs="+", default=list(ALGO_CLASSES.keys()),
                   choices=list(ALGO_CLASSES.keys()),
                   help="Which model folders to process")
    p.add_argument("--max_frames", type=int, default=MAX_FRAMES,
                   help=f"Frames to record per video (default: {MAX_FRAMES} = {MAX_FRAMES/FPS:.0f}s)")
    p.add_argument("--fps", type=int, default=FPS)
    p.add_argument("--skip_eval", action="store_true",
                   help="Skip quick eval, just record video")
    return p.parse_args()


def main():
    args = parse_args()

    for algo_name in args.algos:
        model_path = Path(algo_name) / "best_model.zip"
        video_path = Path(algo_name) / "eval_video.mp4"

        print(f"\n{'='*60}")
        print(f"  {algo_name}")
        print(f"{'='*60}")

        if not model_path.exists():
            print(f"  SKIPPED: {model_path} not found")
            continue

        print(f"  Loading {model_path} ...")
        AlgoClass = ALGO_CLASSES[algo_name]
        model = AlgoClass.load(str(model_path))
        print(f"  Model loaded.")

        if not args.skip_eval:
            quick_eval(model)

        try:
            record_video(model, video_path, max_frames=args.max_frames, fps=args.fps)
        except Exception:
            print(f"  ERROR during video recording:")
            traceback.print_exc()

    print(f"\nDone.")


if __name__ == "__main__":
    main()