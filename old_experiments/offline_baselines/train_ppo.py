#!/usr/bin/env python3
"""
PPO (Behavioral Cloning) Offline Baseline on AntMaze-UMaze-v2 (v2 — with diagnostics)
=======================================================================================
Trains ONLY PPO via behavioral cloning on the D4RL offline antmaze-umaze-v2
dataset for comparison with MeanFlowQL.

PPO is an on-policy algorithm so it cannot use a static replay buffer.
Instead we maximize log-likelihood of recorded actions (= behavioral cloning)
with an entropy bonus to prevent premature collapse.

v2 diagnostics:
  - tqdm progress bar
  - BC loss / log-prob / entropy / gradient norms logged to wandb every LOG_FREQ
  - Dataset sanity checks at startup
  - Policy vs dataset action diagnostics at eval time

Usage:
  python train_ppo_v2.py
  python train_ppo_v2.py --configs config_1 config_2 --seed 42
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
from tqdm import tqdm

import gym
import d4rl

import gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

# ─────────────────────────────────────────────────────────────────────────────
# Global constants
# ─────────────────────────────────────────────────────────────────────────────

ENV_NAME               = "antmaze-umaze-v2"
TOTAL_GRADIENT_STEPS   = 1_000_000
EVAL_FREQ              = 100_000
LOG_FREQ               = 1_000
N_EVAL_EPISODES        = 50
VIDEO_GOALS            = 3
MAX_STEPS_GOAL         = 700
VIDEO_FPS              = 30

WANDB_ENTITY           = "RL_Control_JX"

CONFIGS = {
    # config_1: standard BC with default lr
    "config_1": dict(
        bc_lr=3e-4,
        batch_size=256,
        ent_coef=0.01,
    ),
    # config_2: smaller lr, larger batch, less entropy regularization
    "config_2": dict(
        bc_lr=1e-4,
        batch_size=512,
        ent_coef=0.001,
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading + sanity checks
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset() -> dict:
    """Load the D4RL offline dataset with reward shift and terminal detection."""
    print(f"Loading D4RL offline dataset for {ENV_NAME} ...")
    raw_env = gym.make(ENV_NAME)
    data = d4rl.qlearning_dataset(raw_env)
    raw_env.close()

    N   = len(data["observations"])
    obs = data["observations"].astype(np.float32)
    act = data["actions"].astype(np.float32)
    rwd = data["rewards"].astype(np.float32) - 1.0
    nxt = data["next_observations"].astype(np.float32)

    terminals = np.zeros(N, dtype=np.float32)
    for i in range(N - 1):
        terminals[i] = float(np.linalg.norm(obs[i + 1] - nxt[i]) > 1e-6)
    terminals[-1] = 1.0

    print(f"  {N:,} transitions loaded.")

    print(f"\n  === Dataset Sanity Checks ===")
    print(f"  obs  shape: {obs.shape}  range: [{obs.min():.2f}, {obs.max():.2f}]  mean: {obs.mean():.4f}  std: {obs.std():.4f}")
    print(f"  act  shape: {act.shape}  range: [{act.min():.2f}, {act.max():.2f}]  mean: {act.mean():.4f}  std: {act.std():.4f}")
    print(f"  rwd  shape: {rwd.shape}  range: [{rwd.min():.2f}, {rwd.max():.2f}]  mean: {rwd.mean():.4f}")
    print(f"  nxt  shape: {nxt.shape}  range: [{nxt.min():.2f}, {nxt.max():.2f}]")
    print(f"  terminals: {terminals.sum():.0f} / {N} ({terminals.mean():.4f})")
    print(f"  reward=0 (goal) count: {(rwd == 0.0).sum()} / {N}")
    print(f"  reward=-1 count:       {(rwd == -1.0).sum()} / {N}")

    for name, arr in [("obs", obs), ("act", act), ("rwd", rwd), ("nxt", nxt)]:
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            print(f"  WARNING: {name} contains NaN or Inf!")
        else:
            print(f"  {name}: no NaN/Inf")

    print()
    return dict(
        observations=obs, actions=act, rewards=rwd,
        next_observations=nxt, terminals=terminals,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Environment helpers
# ─────────────────────────────────────────────────────────────────────────────

class AntmazeRewardWrapper(gymnasium.RewardWrapper):
    def reward(self, reward: float) -> float:
        return reward - 1.0

class ContinuingTaskWrapper(gymnasium.Wrapper):
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
# Diagnostic helpers
# ─────────────────────────────────────────────────────────────────────────────

def diagnose_policy(model, dataset: dict, device, n_samples: int = 1000):
    idx = np.random.randint(0, len(dataset["observations"]), size=n_samples)
    dataset_actions = dataset["actions"][idx]

    with torch.no_grad():
        policy_actions = model.predict(dataset["observations"][idx], deterministic=True)[0]

    return {
        "diag/policy_action_mean": float(np.mean(policy_actions)),
        "diag/policy_action_std":  float(np.std(policy_actions)),
        "diag/policy_action_min":  float(np.min(policy_actions)),
        "diag/policy_action_max":  float(np.max(policy_actions)),
        "diag/dataset_action_mean": float(np.mean(dataset_actions)),
        "diag/dataset_action_std":  float(np.std(dataset_actions)),
        "diag/action_mse_vs_data": float(np.mean((policy_actions - dataset_actions) ** 2)),
    }

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_antmaze(model, n_episodes: int = N_EVAL_EPISODES):
    env = make_eval_env()
    returns, successes, ep_lengths = [], [], []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        total_r = 0.0
        success = False
        done = False
        steps = 0

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
    return (
        float(np.mean(returns)),
        float(np.std(returns)),
        float(np.mean(successes)),
        float(np.mean(ep_lengths)),
    )

# ─────────────────────────────────────────────────────────────────────────────
# Video recording
# ─────────────────────────────────────────────────────────────────────────────

def record_video(model, output_path: Path, fps: int = VIDEO_FPS) -> int:
    env = make_video_env()
    frames = []
    obs, _ = env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        frame = None
        for render_fn in [
            lambda: env.render(),
            lambda: env.unwrapped.render(mode="rgb_array"),
            lambda: env.unwrapped.env.render(mode="rgb_array"),
        ]:
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
# PPO behavioral cloning training — with diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def train_offline_ppo(
    config_name: str,
    hyperparams: dict,
    dataset: dict,
    project: str,
    seed: int,
) -> None:
    """Offline training for PPO via behavioral cloning (BC).

    Each step: sample a minibatch from the offline dataset and update
    the actor with  loss = -E[log pi(a|s)] - ent_coef * H[pi(.|s)].
    """
    out_dir = Path("PPO") / config_name
    out_dir.mkdir(parents=True, exist_ok=True)

    bc_lr      = hyperparams.get("bc_lr", 3e-4)
    batch_size = hyperparams.get("batch_size", 256)
    ent_coef   = hyperparams.get("ent_coef", 0.01)

    run = wandb.init(
        project=project,
        entity=WANDB_ENTITY,
        name=f"PPO_{config_name}_seed{seed}",
        group="PPO",
        config=dict(
            algo="PPO", config=config_name, env=ENV_NAME, seed=seed,
            total_gradient_steps=TOTAL_GRADIENT_STEPS,
            eval_freq=EVAL_FREQ, log_freq=LOG_FREQ,
            n_eval_episodes=N_EVAL_EPISODES, **hyperparams,
        ),
        reinit=True,
    )

    print(f"\n{'='*60}")
    print(f"  PPO / {config_name}  (offline behavioral cloning)  seed={seed}")
    print(f"{'='*60}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    init_env = make_eval_env()
    model = PPO("MlpPolicy", init_env, seed=seed, verbose=0, learning_rate=bc_lr)
    init_env.close()

    policy    = model.policy
    device    = model.device
    optimizer = torch.optim.Adam(policy.parameters(), lr=bc_lr)

    obs_np = dataset["observations"]
    act_np = dataset["actions"]
    N      = len(obs_np)

    # Initial eval
    mean_r, std_r, success, mean_len = evaluate_antmaze(model)
    wandb.log({
        "evaluation/mean_return": mean_r, "evaluation/std_return": std_r,
        "evaluation/success_rate": success, "evaluation/mean_ep_length": mean_len,
    }, step=0)
    print(f"  [        0]  return {mean_r:+.2f} +/- {std_r:.2f}  "
          f"success {success:.1%}  ep_len {mean_len:.0f}")

    t0 = time.time()
    pbar = tqdm(range(1, TOTAL_GRADIENT_STEPS + 1), desc="PPO-BC", unit="step",
                dynamic_ncols=True, miniters=LOG_FREQ)

    for step in pbar:
        idx   = np.random.randint(0, N, size=batch_size)
        obs_t = torch.FloatTensor(obs_np[idx]).to(device)
        act_t = torch.FloatTensor(act_np[idx]).to(device)

        policy.set_training_mode(True)
        dist      = policy.get_distribution(obs_t)
        log_probs = dist.log_prob(act_t)
        entropy   = dist.entropy()

        loss = -(log_probs + ent_coef * entropy).mean()

        optimizer.zero_grad()
        loss.backward()

        # Compute gradient norm before clipping
        total_grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)

        optimizer.step()
        policy.set_training_mode(False)

        if step % LOG_FREQ == 0:
            metrics = {
                "training/bc_loss": loss.item(),
                "training/log_prob": log_probs.mean().item(),
                "training/entropy": entropy.mean().item() if entropy.ndim > 0 else entropy.item(),
                "training/grad_norm": total_grad_norm.item() if torch.is_tensor(total_grad_norm) else total_grad_norm,
                "training/log_prob_min": log_probs.min().item(),
                "training/log_prob_max": log_probs.max().item(),
            }
            wandb.log(metrics, step=step)
            pbar.set_postfix(
                loss=f"{loss.item():.3f}",
                logp=f"{log_probs.mean().item():.2f}",
                ent=f"{(entropy.mean().item() if entropy.ndim > 0 else entropy.item()):.3f}",
                gnorm=f"{(total_grad_norm.item() if torch.is_tensor(total_grad_norm) else total_grad_norm):.2f}",
            )

        if step % EVAL_FREQ == 0:
            mean_r, std_r, success, mean_len = evaluate_antmaze(model)
            diag = diagnose_policy(model, dataset, device)

            eval_metrics = {
                "evaluation/mean_return": mean_r,
                "evaluation/std_return": std_r,
                "evaluation/success_rate": success,
                "evaluation/mean_ep_length": mean_len,
            }
            eval_metrics.update(diag)
            wandb.log(eval_metrics, step=step)

            elapsed = (time.time() - t0) / 60
            tqdm.write(
                f"  [{step:>9,}]  return {mean_r:+.2f} +/- {std_r:.2f}  "
                f"success {success:.1%}  ep_len {mean_len:.0f}  "
                f"act_mse {diag['diag/action_mse_vs_data']:.4f}  "
                f"({elapsed:.1f} min)"
            )

    pbar.close()
    print(f"  Training done ({(time.time()-t0)/60:.1f} min total)")

    # Final diagnosis
    print(f"\n  === Final Policy Diagnosis ===")
    diag = diagnose_policy(model, dataset, device)
    for k, v in diag.items():
        print(f"  {k}: {v:.6f}")

    model.save(str(out_dir / "model"))

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

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="PPO (behavioral cloning) offline baseline on antmaze-umaze-v2 — v2 with diagnostics"
    )
    p.add_argument("--configs", nargs="+", default=list(CONFIGS.keys()),
                   help="Config names to run (default: all — config_1 and config_2)")
    p.add_argument("--project", default="antmaze_offline_baselines")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    dataset = load_dataset()

    for config_name in args.configs:
        if config_name not in CONFIGS:
            print(f"Warning: unknown config '{config_name}' (available: {list(CONFIGS.keys())})")
            continue
        try:
            train_offline_ppo(config_name, CONFIGS[config_name], dataset,
                              args.project, args.seed)
        except Exception:
            print(f"\nERROR during PPO/{config_name}:")
            traceback.print_exc()
            print("Continuing with next config...\n")


if __name__ == "__main__":
    main()