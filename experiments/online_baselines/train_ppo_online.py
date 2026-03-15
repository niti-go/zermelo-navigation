#!/usr/bin/env python3
"""
PPO Online Training on AntMaze-UMaze-v2
========================================
Trains PPO from scratch with full environment interaction (online RL).
This is the standard online baseline to compare against offline RL methods.

Logs training metrics (policy loss, value loss, entropy, clip fraction,
gradient norms) and evaluation metrics (success rate, return, episode
length) to wandb.

Usage:
  python train_ppo_online.py
  python train_ppo_online.py --total_timesteps 2000000 --seed 42
  
  
# 1. Start tmux
tmux new -s baselines
conda activate flowrl
wandb login
# 6. Set GPU and run
CUDA_VISIBLE_DEVICES=0 python train_baselines.py
# 7. Detach: Ctrl+b, then d
# 8. Reattach later: tmux attach -t baselines
"""

import os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import wandb
import imageio

import gym
import d4rl  # noqa: registers antmaze envs

import gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

ENV_NAME            = "antmaze-umaze-v2"
TOTAL_TIMESTEPS     = 1_000_000
EVAL_FREQ           = 10_000
LOG_FREQ            = 2_048       # log every rollout (= n_steps)
N_EVAL_EPISODES     = 50
VIDEO_GOALS         = 3
MAX_STEPS_GOAL      = 700
VIDEO_FPS           = 30

WANDB_ENTITY        = "RL_Control_JX"

# ─────────────────────────────────────────────────────────────────────────────
# Environment helpers
# ─────────────────────────────────────────────────────────────────────────────

class AntmazeRewardWrapper(gymnasium.RewardWrapper):
    """Shift reward by -1 to match offline RL convention."""
    def reward(self, reward: float) -> float:
        return reward - 1.0


class ContinuingTaskWrapper(gymnasium.Wrapper):
    """Video wrapper: on goal-reach, reset to new goal and continue recording."""
    def __init__(self, env, num_goals=VIDEO_GOALS, max_steps_per_goal=MAX_STEPS_GOAL):
        super().__init__(env)
        self.num_goals = num_goals
        self.max_steps_per_goal = max_steps_per_goal
        self._n_goals = 0
        self._goal_steps = 0

    def reset(self, **kwargs):
        self._n_goals = 0
        self._goal_steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._goal_steps += 1
        goal_reached = terminated and reward >= -0.5
        if goal_reached:
            self._n_goals += 1
            info["goals_reached"] = self._n_goals
            if self._n_goals >= self.num_goals:
                return obs, reward, True, False, info
            obs, _ = self.env.reset()
            self._goal_steps = 0
            terminated = False
        elif terminated or truncated or self._goal_steps >= self.max_steps_per_goal:
            obs, _ = self.env.reset()
            self._goal_steps = 0
            terminated = False
            truncated = False
        return obs, reward, terminated, truncated, info


def _bridge_env():
    return gymnasium.make("GymV21Environment-v0", env_id=ENV_NAME)

def make_train_env():
    env = _bridge_env()
    env = AntmazeRewardWrapper(env)
    env = Monitor(env)
    return env

def make_eval_env():
    env = _bridge_env()
    env = AntmazeRewardWrapper(env)
    env = Monitor(env)
    return env

def make_video_env():
    env = _bridge_env()
    env = AntmazeRewardWrapper(env)
    env = ContinuingTaskWrapper(env)
    return env

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_antmaze(model, n_episodes=N_EVAL_EPISODES):
    env = make_eval_env()
    returns, successes, ep_lengths = [], [], []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total_r, success, done, steps = 0.0, False, False, 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, _ = env.step(action)
            total_r += r
            steps += 1
            done = terminated or truncated
            if terminated and r >= -0.5:
                success = True
        returns.append(total_r)
        successes.append(float(success))
        ep_lengths.append(steps)
    env.close()
    return {
        "evaluation/mean_return": float(np.mean(returns)),
        "evaluation/std_return": float(np.std(returns)),
        "evaluation/success_rate": float(np.mean(successes)),
        "evaluation/mean_ep_length": float(np.mean(ep_lengths)),
    }

# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic helpers
# ─────────────────────────────────────────────────────────────────────────────

def extract_ppo_metrics(model) -> dict:
    metrics = {}
    try:
        for key in ["train/policy_gradient_loss", "train/value_loss",
                     "train/entropy_loss", "train/clip_fraction",
                     "train/clip_range", "train/approx_kl",
                     "train/explained_variance", "train/loss"]:
            val = model.logger.name_to_value.get(key)
            if val is not None:
                metrics[key.replace("train/", "training/")] = val
    except Exception:
        pass
    return metrics

# ─────────────────────────────────────────────────────────────────────────────
# Wandb callback
# ─────────────────────────────────────────────────────────────────────────────

class WandbLoggingCallback(BaseCallback):
    """Logs training metrics + periodic evaluation to wandb during online training."""

    def __init__(self, eval_freq=EVAL_FREQ, log_freq=LOG_FREQ, verbose=1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.log_freq = log_freq
        self.best_success = -1.0
        self.t0 = None
        self._last_log_step = 0

    def _on_training_start(self):
        self.t0 = time.time()
        eval_metrics = evaluate_antmaze(self.model)
        wandb.log(eval_metrics, step=0)
        print(f"  [        0]  return {eval_metrics['evaluation/mean_return']:+.2f}  "
              f"success {eval_metrics['evaluation/success_rate']:.1%}  "
              f"ep_len {eval_metrics['evaluation/mean_ep_length']:.0f}")

    def _on_rollout_end(self):
        """Called after each PPO rollout collection. Log training metrics here."""
        step = self.num_timesteps
        metrics = extract_ppo_metrics(self.model)

        if len(self.model.ep_info_buffer) > 0:
            ep_rewards = [ep["r"] for ep in self.model.ep_info_buffer]
            ep_lengths = [ep["l"] for ep in self.model.ep_info_buffer]
            metrics["rollout/mean_ep_reward"] = np.mean(ep_rewards)
            metrics["rollout/mean_ep_length"] = np.mean(ep_lengths)

        if metrics:
            wandb.log(metrics, step=step)

    def _on_step(self) -> bool:
        step = self.num_timesteps

        if step % self.eval_freq == 0 and step != self._last_log_step:
            self._last_log_step = step
            eval_metrics = evaluate_antmaze(self.model)
            wandb.log(eval_metrics, step=step)

            elapsed = (time.time() - self.t0) / 60
            success = eval_metrics["evaluation/success_rate"]
            print(
                f"  [{step:>9,}]  return {eval_metrics['evaluation/mean_return']:+.2f}  "
                f"success {success:.1%}  "
                f"ep_len {eval_metrics['evaluation/mean_ep_length']:.0f}  "
                f"({elapsed:.1f} min)"
            )

            if success > self.best_success:
                self.best_success = success
                self.model.save(str(Path("PPO_online") / "best_model"))

        return True

# ─────────────────────────────────────────────────────────────────────────────
# Video recording
# ─────────────────────────────────────────────────────────────────────────────

def record_video(model, output_path: Path, fps=VIDEO_FPS) -> int:
    env = make_video_env()
    frames = []
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        frame = None
        for render_fn in [lambda: env.render(),
                          lambda: env.unwrapped.render(mode="rgb_array"),
                          lambda: env.unwrapped.env.render(mode="rgb_array")]:
            try:
                f = render_fn()
                if isinstance(f, np.ndarray) and f.ndim == 3:
                    frame = f
                    break
            except Exception:
                pass
        if frame is not None:
            frames.append(frame)
    env.close()
    if not frames:
        print("    Warning: no frames captured.")
        return 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(output_path), frames, fps=fps)
    return len(frames)

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PPO online training on antmaze-umaze-v2"
    )
    parser.add_argument("--total_timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--project", default="antmaze_online_baselines")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--n_steps", type=int, default=2048,
                        help="Number of steps per rollout before update")
    parser.add_argument("--n_epochs", type=int, default=10,
                        help="Number of PPO epochs per rollout")
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01,
                        help="Entropy bonus to encourage exploration")
    args = parser.parse_args()

    out_dir = Path("PPO_online")
    out_dir.mkdir(parents=True, exist_ok=True)

    run = wandb.init(
        project=args.project,
        entity=WANDB_ENTITY,
        name=f"PPO_online_seed{args.seed}",
        group="PPO",
        config=dict(
            algo="PPO", mode="online", env=ENV_NAME, seed=args.seed,
            total_timesteps=args.total_timesteps,
            learning_rate=args.learning_rate, batch_size=args.batch_size,
            gamma=args.gamma, n_steps=args.n_steps, n_epochs=args.n_epochs,
            gae_lambda=args.gae_lambda, clip_range=args.clip_range,
            ent_coef=args.ent_coef,
        ),
        reinit=True,
    )

    print(f"\n{'='*60}")
    print(f"  PPO / online  seed={args.seed}")
    print(f"  total_timesteps={args.total_timesteps:,}")
    print(f"  n_steps={args.n_steps}  n_epochs={args.n_epochs}  "
          f"batch_size={args.batch_size}  ent_coef={args.ent_coef}")
    print(f"{'='*60}\n")

    train_env = make_train_env()

    model = PPO(
        "MlpPolicy",
        train_env,
        seed=args.seed,
        verbose=0,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gamma=args.gamma,
        n_steps=args.n_steps,
        n_epochs=args.n_epochs,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
    )

    callback = WandbLoggingCallback(eval_freq=EVAL_FREQ, log_freq=LOG_FREQ)

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callback,
        log_interval=None,
        progress_bar=True,
    )

    train_env.close()

    print("\n  === Final Evaluation ===")
    eval_metrics = evaluate_antmaze(model)
    for k, v in eval_metrics.items():
        print(f"  {k}: {v:.4f}")
    wandb.log(eval_metrics, step=args.total_timesteps)

    model.save(str(out_dir / "final_model"))

    video_path = out_dir / "videos" / "eval_video.mp4"
    print(f"  Recording video -> {video_path}")
    try:
        n_frames = record_video(model, video_path)
        if n_frames > 0:
            print(f"  Video saved ({n_frames} frames)")
            wandb.log({"video": wandb.Video(str(video_path), fps=VIDEO_FPS, format="mp4")})
        else:
            print("  No video saved.")
    except Exception:
        print("  Video recording failed:")
        traceback.print_exc()

    wandb.finish()
    print("Done.")


if __name__ == "__main__":
    main()