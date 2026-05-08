#!/usr/bin/env python3
"""
TD3 Online Training on OGBench Pointmaze
=========================================
Trains TD3 from scratch with full environment interaction (online RL).
This is the standard online baseline to compare against offline RL methods
(e.g., MeanFlowQL trained on the same OGBench pointmaze dataset).

Logs training metrics (actor loss, critic loss) and evaluation metrics
(success rate, return, episode length) to wandb.

Usage:
  cd ~/online_training_ogbench
  conda activate flowrl
  CUDA_VISIBLE_DEVICES=0 python train_td3_online.py
  CUDA_VISIBLE_DEVICES=7 python train_td3_online.py --total_timesteps 2000000 --seed 42

# 1. Start tmux
tmux new -s td3_ogbench
conda activate flowrl
wandb login
# 6. Set GPU and run
CUDA_VISIBLE_DEVICES=0 python train_td3_online.py
# 7. Detach: Ctrl+b, then d
# 8. Reattach later: tmux attach -t td3_ogbench
"""

import os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import wandb

import gymnasium
import ogbench
from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

ENV_NAME            = "pointmaze-medium-navigate-singletask-v0"
TOTAL_TIMESTEPS     = 1_000_000
EVAL_FREQ           = 10_000
N_EVAL_EPISODES     = 50

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

def extract_td3_metrics(model) -> dict:
    """Extract TD3 internal training metrics from the logger."""
    metrics = {}
    try:
        for key in ["train/actor_loss", "train/critic_loss"]:
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

    def __init__(self, env_name=ENV_NAME, eval_freq=EVAL_FREQ, verbose=1):
        super().__init__(verbose)
        self.env_name = env_name
        self.eval_freq = eval_freq
        self.best_success = -1.0
        self.t0 = None
        self._last_log_step = 0
        self._last_train_log_step = 0

    def _on_training_start(self):
        self.t0 = time.time()
        eval_metrics = evaluate(self.model, self.env_name)
        wandb.log(eval_metrics, step=0)
        print(f"  [        0]  return {eval_metrics['evaluation/mean_return']:+.2f}  "
              f"success {eval_metrics['evaluation/success_rate']:.1%}  "
              f"ep_len {eval_metrics['evaluation/mean_ep_length']:.0f}")

    def _on_step(self) -> bool:
        step = self.num_timesteps

        # Log training metrics periodically
        if step - self._last_train_log_step >= 1000:
            self._last_train_log_step = step
            metrics = extract_td3_metrics(self.model)

            if len(self.model.ep_info_buffer) > 0:
                ep_rewards = [ep["r"] for ep in self.model.ep_info_buffer]
                ep_lengths = [ep["l"] for ep in self.model.ep_info_buffer]
                metrics["rollout/mean_ep_reward"] = np.mean(ep_rewards)
                metrics["rollout/mean_ep_length"] = np.mean(ep_lengths)

            if metrics:
                wandb.log(metrics, step=step)

        # Evaluation
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
                self.model.save(str(Path("TD3_online_ogbench") / "best_model"))

        return True

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="TD3 online training on OGBench pointmaze"
    )
    parser.add_argument("--env_name", default=ENV_NAME)
    parser.add_argument("--total_timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--project", default="online_baselines_pointmaze_ogbench")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005,
                        help="Target network soft update coefficient")
    parser.add_argument("--learning_starts", type=int, default=1000,
                        help="Number of random steps before training starts")
    parser.add_argument("--noise_std", type=float, default=0.1,
                        help="Standard deviation of exploration noise")
    parser.add_argument("--policy_delay", type=int, default=2,
                        help="Delay between actor updates (TD3 feature)")
    args = parser.parse_args()

    out_dir = Path("TD3_online_ogbench")
    out_dir.mkdir(parents=True, exist_ok=True)

    run = wandb.init(
        project=args.project,
        entity=WANDB_ENTITY,
        name=f"TD3_online_{args.env_name}_seed{args.seed}",
        group="TD3",
        config=dict(
            algo="TD3", mode="online", env=args.env_name, seed=args.seed,
            total_timesteps=args.total_timesteps,
            learning_rate=args.learning_rate, batch_size=args.batch_size,
            gamma=args.gamma, tau=args.tau,
            learning_starts=args.learning_starts, noise_std=args.noise_std,
            policy_delay=args.policy_delay,
        ),
        reinit=True,
    )

    print(f"\n{'='*60}")
    print(f"  TD3 / online  env={args.env_name}  seed={args.seed}")
    print(f"  total_timesteps={args.total_timesteps:,}")
    print(f"  batch_size={args.batch_size}  tau={args.tau}  "
          f"noise_std={args.noise_std}  policy_delay={args.policy_delay}")
    print(f"{'='*60}\n")

    train_env = make_train_env(args.env_name)

    # TD3 uses deterministic policy + Gaussian exploration noise
    action_dim = train_env.action_space.shape[0]
    action_noise = NormalActionNoise(
        mean=np.zeros(action_dim),
        sigma=args.noise_std * np.ones(action_dim),
    )

    model = TD3(
        "MlpPolicy",
        train_env,
        seed=args.seed,
        verbose=0,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gamma=args.gamma,
        tau=args.tau,
        learning_starts=args.learning_starts,
        action_noise=action_noise,
        policy_delay=args.policy_delay,
    )

    callback = WandbLoggingCallback(
        env_name=args.env_name,
        eval_freq=EVAL_FREQ,
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
