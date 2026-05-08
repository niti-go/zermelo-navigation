#!/usr/bin/env python3
"""
PPO Online Training on OGBench Pointmaze
=========================================
Trains PPO from scratch with full environment interaction (online RL).
This is the standard online baseline to compare against offline RL methods
(e.g., MeanFlowQL trained on the same OGBench pointmaze dataset).

Logs training metrics (policy loss, value loss, entropy, clip fraction,
gradient norms) and evaluation metrics (success rate, return, episode
length) to wandb.

Usage:
  cd ~/online_training_ogbench
  conda activate flowrl
  CUDA_VISIBLE_DEVICES=4 python train_ppo_online.py --total_timesteps 2000000 --seed 42

# 1. Start tmux
tmux new -s ppo_ogbench
conda activate flowrl
wandb login
# 6. Set GPU and run
CUDA_VISIBLE_DEVICES=4 python train_ppo_online.py
# 7. Detach: Ctrl+b, then d
# 8. Reattach later: tmux attach -t ppo_ogbench
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

import gymnasium
import ogbench
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

ENV_NAME            = "pointmaze-medium-navigate-singletask-v0"
TOTAL_TIMESTEPS     = 1_000_000
EVAL_FREQ           = 10_000
LOG_FREQ            = 2_048       # log every rollout (= n_steps)
N_EVAL_EPISODES     = 50
MAX_EPISODE_STEPS   = 1000        # ogbench default for pointmaze

WANDB_ENTITY        = "RL_Control_JX"

# ─────────────────────────────────────────────────────────────────────────────
# Environment helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_train_env(env_name=ENV_NAME):
    """Create ogbench pointmaze training environment."""
    env = ogbench.make_env_and_datasets(env_name, env_only=True)
    env = Monitor(env)
    return env

def make_eval_env(env_name=ENV_NAME):
    """Create ogbench pointmaze evaluation environment."""
    env = ogbench.make_env_and_datasets(env_name, env_only=True)
    env = Monitor(env)
    return env

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, env_name=ENV_NAME, n_episodes=N_EVAL_EPISODES):
    """Evaluate agent on ogbench pointmaze and return metrics."""
    env = make_eval_env(env_name)
    returns, successes, ep_lengths = [], [], []
    for _ in range(n_episodes):
        obs, info = env.reset()
        total_r, done, steps = 0.0, False, 0
        episode_success = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(action)
            total_r += r
            steps += 1
            done = terminated or truncated
            # ogbench reports success in info
            if info.get("success", 0.0) > 0.5:
                episode_success = True
        returns.append(total_r)
        successes.append(float(episode_success))
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
    """Extract PPO internal training metrics from the logger."""
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
    """Logs training metrics + periodic evaluation to wandb."""

    def __init__(self, env_name=ENV_NAME, eval_freq=EVAL_FREQ, log_freq=LOG_FREQ, verbose=1):
        super().__init__(verbose)
        self.env_name = env_name
        self.eval_freq = eval_freq
        self.log_freq = log_freq
        self.best_success = -1.0
        self.t0 = None
        self._last_log_step = 0

    def _on_training_start(self):
        self.t0 = time.time()
        eval_metrics = evaluate(self.model, self.env_name)
        wandb.log(eval_metrics, step=0)
        print(f"  [        0]  return {eval_metrics['evaluation/mean_return']:+.2f}  "
              f"success {eval_metrics['evaluation/success_rate']:.1%}  "
              f"ep_len {eval_metrics['evaluation/mean_ep_length']:.0f}")

    def _on_rollout_end(self):
        """Called after each PPO rollout collection. Log training metrics."""
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
            eval_metrics = evaluate(self.model, self.env_name)
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
                self.model.save(str(Path("PPO_online_ogbench") / "best_model"))

        return True

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PPO online training on OGBench pointmaze"
    )
    parser.add_argument("--env_name", default=ENV_NAME)
    parser.add_argument("--total_timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--project", default="online_baselines_pointmaze_ogbench")
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

    out_dir = Path("PPO_online_ogbench")
    out_dir.mkdir(parents=True, exist_ok=True)

    run = wandb.init(
        project=args.project,
        entity=WANDB_ENTITY,
        name=f"PPO_online_{args.env_name}_seed{args.seed}",
        group="PPO",
        config=dict(
            algo="PPO", mode="online", env=args.env_name, seed=args.seed,
            total_timesteps=args.total_timesteps,
            learning_rate=args.learning_rate, batch_size=args.batch_size,
            gamma=args.gamma, n_steps=args.n_steps, n_epochs=args.n_epochs,
            gae_lambda=args.gae_lambda, clip_range=args.clip_range,
            ent_coef=args.ent_coef,
        ),
        reinit=True,
    )

    print(f"\n{'='*60}")
    print(f"  PPO / online  env={args.env_name}  seed={args.seed}")
    print(f"  total_timesteps={args.total_timesteps:,}")
    print(f"  n_steps={args.n_steps}  n_epochs={args.n_epochs}  "
          f"batch_size={args.batch_size}  ent_coef={args.ent_coef}")
    print(f"{'='*60}\n")

    train_env = make_train_env(args.env_name)

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

    callback = WandbLoggingCallback(
        env_name=args.env_name,
        eval_freq=EVAL_FREQ,
        log_freq=LOG_FREQ,
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callback,
        log_interval=None,
        progress_bar=True,
    )

    train_env.close()

    print("\n  === Final Evaluation ===")
    eval_metrics = evaluate(model, args.env_name)
    for k, v in eval_metrics.items():
        print(f"  {k}: {v:.4f}")
    wandb.log(eval_metrics, step=args.total_timesteps)

    model.save(str(out_dir / "final_model"))

    wandb.finish()
    print("Done.")


if __name__ == "__main__":
    main()
