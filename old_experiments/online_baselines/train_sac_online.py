#!/usr/bin/env python3
"""
SAC Online Training on AntMaze-UMaze-v2
========================================
Trains SAC from scratch with full environment interaction (online RL).
This is the standard online baseline to compare against offline RL methods.

Logs training metrics (critic loss, actor loss, entropy coef, gradient norms)
and evaluation metrics (success rate, return, episode length) to wandb.

Usage:
  python train_sac_online.py
  python train_sac_online.py --total_timesteps 2000000 --seed 42
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
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

ENV_NAME            = "antmaze-umaze-v2"
TOTAL_TIMESTEPS     = 1_000_000
EVAL_FREQ           = 10_000      # eval every N env steps
LOG_FREQ            = 1_000       # log training metrics every N env steps
N_EVAL_EPISODES     = 50
VIDEO_GOALS         = 3
MAX_STEPS_GOAL      = 700
VIDEO_FPS           = 30

WANDB_ENTITY        = "RL_Control_JX"

# ─────────────────────────────────────────────────────────────────────────────
# Environment helpers
# ─────────────────────────────────────────────────────────────────────────────

class AntmazeRewardWrapper(gymnasium.RewardWrapper):
    """Shift reward by -1 to match offline RL convention.
    Raw: 1.0 at goal, 0.0 elsewhere. After shift: 0.0 at goal, -1.0 elsewhere.
    """
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
    """Training env with reward shift + Monitor for episode stats."""
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

def compute_gradient_norms(model) -> dict:
    norms = {}
    for name, param_group in [("actor", model.actor.parameters()),
                               ("critic", model.critic.parameters())]:
        total_norm = 0.0
        n_params = 0
        for p in param_group:
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
                n_params += 1
        norms[f"grad_norm/{name}"] = total_norm ** 0.5 if n_params > 0 else 0.0
    return norms


def extract_sac_metrics(model) -> dict:
    metrics = {}
    try:
        for key in ["train/critic_loss", "train/actor_loss", "train/ent_coef",
                     "train/ent_coef_loss", "train/log_ent_coef"]:
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

    def _on_training_start(self):
        self.t0 = time.time()
        # Initial eval at step 0
        eval_metrics = evaluate_antmaze(self.model)
        wandb.log(eval_metrics, step=0)
        print(f"  [        0]  return {eval_metrics['evaluation/mean_return']:+.2f}  "
              f"success {eval_metrics['evaluation/success_rate']:.1%}  "
              f"ep_len {eval_metrics['evaluation/mean_ep_length']:.0f}")

    def _on_step(self) -> bool:
        step = self.num_timesteps

        # Log training metrics
        if step % self.log_freq == 0:
            metrics = extract_sac_metrics(self.model)
            grad_norms = compute_gradient_norms(self.model)
            metrics.update(grad_norms)

            # Log episode info from Monitor wrapper if available
            if len(self.model.ep_info_buffer) > 0:
                ep_rewards = [ep["r"] for ep in self.model.ep_info_buffer]
                ep_lengths = [ep["l"] for ep in self.model.ep_info_buffer]
                metrics["rollout/mean_ep_reward"] = np.mean(ep_rewards)
                metrics["rollout/mean_ep_length"] = np.mean(ep_lengths)

            wandb.log(metrics, step=step)

        # Evaluation
        if step % self.eval_freq == 0:
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
                self.model.save(str(Path("SAC_online") / "best_model"))

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
        description="SAC online training on antmaze-umaze-v2"
    )
    parser.add_argument("--total_timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--project", default="antmaze_online_baselines")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    args = parser.parse_args()

    out_dir = Path("SAC_online")
    out_dir.mkdir(parents=True, exist_ok=True)

    run = wandb.init(
        project=args.project,
        entity=WANDB_ENTITY,
        name=f"SAC_online_seed{args.seed}",
        group="SAC",
        config=dict(
            algo="SAC", mode="online", env=ENV_NAME, seed=args.seed,
            total_timesteps=args.total_timesteps,
            learning_rate=args.learning_rate, batch_size=args.batch_size,
            gamma=args.gamma, tau=args.tau,
        ),
        reinit=True,
    )

    print(f"\n{'='*60}")
    print(f"  SAC / online  seed={args.seed}")
    print(f"  total_timesteps={args.total_timesteps:,}")
    print(f"{'='*60}\n")

    train_env = make_train_env()

    model = SAC(
        "MlpPolicy",
        train_env,
        seed=args.seed,
        verbose=0,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gamma=args.gamma,
        tau=args.tau,
        ent_coef="auto",
        learning_starts=10_000,   # collect random transitions before training
        buffer_size=1_000_000,
    )

    callback = WandbLoggingCallback(eval_freq=EVAL_FREQ, log_freq=LOG_FREQ)

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callback,
        log_interval=None,       # we handle logging ourselves
        progress_bar=True,
    )

    train_env.close()

    # Final eval
    print("\n  === Final Evaluation ===")
    eval_metrics = evaluate_antmaze(model)
    for k, v in eval_metrics.items():
        print(f"  {k}: {v:.4f}")
    wandb.log(eval_metrics, step=args.total_timesteps)

    model.save(str(out_dir / "final_model"))

    # Record video
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