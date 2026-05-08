#!/usr/bin/env python3
"""
Offline RL Baselines on AntMaze-UMaze-v2 (v2 — with diagnostics)
=================================================================
Trains SAC, PPO, TD3 on the D4RL offline antmaze-umaze-v2 dataset
for comparison with MeanFlowQL.

v2 changes:
  - tqdm progress bars for training loops
  - Training loss / Q-value / actor loss / gradient norms logged to wandb
  - Dataset sanity checks at startup
  - Action distribution diagnostics at eval time
  - Console + wandb training metrics every LOG_FREQ steps

Usage:
  python train_baselines_v2.py
  python train_baselines_v2.py --algos SAC TD3 --seed 42
  python train_baselines_v2.py --algos PPO --configs config_2
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
from stable_baselines3 import SAC, TD3, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure as sb3_configure

# ─────────────────────────────────────────────────────────────────────────────
# Global constants
# ─────────────────────────────────────────────────────────────────────────────

ENV_NAME               = "antmaze-umaze-v2"
TOTAL_GRADIENT_STEPS   = 1_000_000
EVAL_FREQ              = 100_000
LOG_FREQ               = 1_000      # ← more frequent logging for debugging
N_EVAL_EPISODES        = 50
GRADIENT_STEPS_PER_ITER = 1
VIDEO_GOALS            = 3
MAX_STEPS_GOAL         = 700
VIDEO_FPS              = 30

WANDB_ENTITY           = "RL_Control_JX"

ALGO_CLASSES = {"SAC": SAC, "TD3": TD3, "PPO": PPO}

CONFIGS = {
    "SAC": {
        "config_1": dict(
            learning_rate=3e-4,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            ent_coef="auto",
        ),
    },
    "TD3": {
        "config_1": dict(
            learning_rate=1e-3,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            policy_delay=2,
        ),
    },
    "PPO": {
        "config_1": dict(
            bc_lr=3e-4,
            batch_size=256,
            ent_coef=0.01,
        ),
        "config_2": dict(
            bc_lr=1e-4,
            batch_size=512,
            ent_coef=0.001,
        ),
    },
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

    # ── Sanity checks ──────────────────────────────────────────────────────
    print(f"\n  === Dataset Sanity Checks ===")
    print(f"  obs  shape: {obs.shape}  range: [{obs.min():.2f}, {obs.max():.2f}]  mean: {obs.mean():.4f}  std: {obs.std():.4f}")
    print(f"  act  shape: {act.shape}  range: [{act.min():.2f}, {act.max():.2f}]  mean: {act.mean():.4f}  std: {act.std():.4f}")
    print(f"  rwd  shape: {rwd.shape}  range: [{rwd.min():.2f}, {rwd.max():.2f}]  mean: {rwd.mean():.4f}")
    print(f"  nxt  shape: {nxt.shape}  range: [{nxt.min():.2f}, {nxt.max():.2f}]")
    print(f"  terminals: {terminals.sum():.0f} / {N} ({terminals.mean():.4f})")
    print(f"  reward=0 (goal) count: {(rwd == 0.0).sum()} / {N}")
    print(f"  reward=-1 count:       {(rwd == -1.0).sum()} / {N}")

    # Check for NaN/Inf
    for name, arr in [("obs", obs), ("act", act), ("rwd", rwd), ("nxt", nxt)]:
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            print(f"  ⚠️  WARNING: {name} contains NaN or Inf!")
        else:
            print(f"  ✓ {name}: no NaN/Inf")

    print()
    return dict(
        observations=obs, actions=act, rewards=rwd,
        next_observations=nxt, terminals=terminals,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Environment helpers (unchanged)
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
# Replay buffer filling + verification
# ─────────────────────────────────────────────────────────────────────────────

def fill_replay_buffer(model, dataset: dict) -> int:
    """Fill SB3 replay buffer and verify the data was written correctly."""
    buf = model.replay_buffer
    N = min(len(dataset["observations"]), buf.buffer_size)

    buf.observations[:N, 0]      = dataset["observations"][:N]
    buf.next_observations[:N, 0] = dataset["next_observations"][:N]
    buf.actions[:N, 0]           = dataset["actions"][:N]
    buf.rewards[:N, 0]           = dataset["rewards"][:N]
    buf.dones[:N, 0]             = dataset["terminals"][:N]

    buf.pos  = N % buf.buffer_size
    buf.full = N >= buf.buffer_size

    print(f"  Replay buffer filled: {N:,} transitions "
          f"({'full' if buf.full else f'{N/buf.buffer_size:.0%}'})")

    # ── Verify buffer contents ─────────────────────────────────────────────
    print(f"  === Buffer Verification ===")
    print(f"  buf.observations[:5, 0, :3] = {buf.observations[:5, 0, :3]}")
    print(f"  dataset obs[:5, :3]         = {dataset['observations'][:5, :3]}")
    match = np.allclose(buf.observations[:100, 0], dataset["observations"][:100])
    print(f"  First 100 obs match dataset: {'✓ YES' if match else '⚠️  NO — BUFFER FILL BROKEN'}")

    # Verify rewards
    rwd_buf = buf.rewards[:N, 0].flatten()
    print(f"  buf rewards  — mean: {rwd_buf.mean():.4f}  min: {rwd_buf.min():.2f}  max: {rwd_buf.max():.2f}")
    print(f"  buf dones    — sum: {buf.dones[:N, 0].sum():.0f}")

    # Sample a batch the same way SB3 will
    sample = buf.sample(256)
    print(f"  Sample batch — obs shape: {sample.observations.shape}  "
          f"rwd range: [{sample.rewards.min():.2f}, {sample.rewards.max():.2f}]  "
          f"done frac: {sample.dones.float().mean():.3f}")

    return N

# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_gradient_norms(model) -> dict:
    """Compute gradient norms for actor and critic networks."""
    norms = {}
    for name, param_group in [
        ("actor", model.actor.parameters()),
        ("critic", model.critic.parameters()),
    ]:
        total_norm = 0.0
        n_params = 0
        for p in param_group:
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
                n_params += 1
        norms[f"grad_norm/{name}"] = total_norm ** 0.5 if n_params > 0 else 0.0
    return norms


def diagnose_policy(model, dataset: dict, device, n_samples: int = 1000):
    """Check what actions the policy produces vs the dataset."""
    idx = np.random.randint(0, len(dataset["observations"]), size=n_samples)
    obs_t = torch.FloatTensor(dataset["observations"][idx]).to(device)
    dataset_actions = dataset["actions"][idx]

    with torch.no_grad():
        # Get deterministic actions from the policy
        policy_actions = model.predict(dataset["observations"][idx], deterministic=True)[0]

    diag = {
        "diag/policy_action_mean": float(np.mean(policy_actions)),
        "diag/policy_action_std":  float(np.std(policy_actions)),
        "diag/policy_action_min":  float(np.min(policy_actions)),
        "diag/policy_action_max":  float(np.max(policy_actions)),
        "diag/dataset_action_mean": float(np.mean(dataset_actions)),
        "diag/dataset_action_std":  float(np.std(dataset_actions)),
        "diag/action_mse_vs_data": float(np.mean((policy_actions - dataset_actions) ** 2)),
    }

    return diag


def extract_sac_metrics(model) -> dict:
    """Extract internal SAC training metrics from the logger."""
    metrics = {}
    try:
        logger = model.logger
        # SB3 SAC logs these internally; we read from the model's internal tracking
        if hasattr(model, '_n_updates'):
            metrics["training/n_updates"] = model._n_updates
    except Exception:
        pass

    # Access critic/actor losses from the last training step via model internals
    # SB3 stores these in the logger name_to_value dict
    try:
        for key in ["train/critic_loss", "train/actor_loss", "train/ent_coef",
                     "train/ent_coef_loss", "train/log_ent_coef"]:
            val = model.logger.name_to_value.get(key)
            if val is not None:
                wandb_key = key.replace("train/", "training/")
                metrics[wandb_key] = val
    except Exception:
        pass

    return metrics


def extract_td3_metrics(model) -> dict:
    """Extract internal TD3 training metrics."""
    metrics = {}
    try:
        for key in ["train/critic_loss", "train/actor_loss"]:
            val = model.logger.name_to_value.get(key)
            if val is not None:
                wandb_key = key.replace("train/", "training/")
                metrics[wandb_key] = val
    except Exception:
        pass
    return metrics

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
# Off-policy offline training (SAC / TD3) — with diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def train_offline_off_policy(
    algo_name: str,
    config_name: str,
    hyperparams: dict,
    dataset: dict,
    project: str,
    seed: int,
) -> None:
    out_dir = Path(algo_name) / config_name
    out_dir.mkdir(parents=True, exist_ok=True)

    run = wandb.init(
        project=project,
        entity=WANDB_ENTITY,
        name=f"{algo_name}_{config_name}_seed{seed}",
        group=algo_name,
        config=dict(
            algo=algo_name, config=config_name, env=ENV_NAME, seed=seed,
            total_gradient_steps=TOTAL_GRADIENT_STEPS,
            eval_freq=EVAL_FREQ, log_freq=LOG_FREQ,
            n_eval_episodes=N_EVAL_EPISODES, **hyperparams,
        ),
        reinit=True,
    )

    print(f"\n{'='*60}")
    print(f"  {algo_name} / {config_name}  (offline off-policy)  seed={seed}")
    print(f"{'='*60}")

    init_env = make_eval_env()
    n_data = len(dataset["observations"])
    AlgoClass = ALGO_CLASSES[algo_name]

    model = AlgoClass(
        "MlpPolicy", init_env, seed=seed, verbose=0,
        buffer_size=max(n_data + 1, 2_000_000),
        learning_starts=0,
        **hyperparams,
    )
    init_env.close()

    # Use SB3's logger so we can read internal metrics (critic_loss etc.)
    # but redirect its output to /dev/null — we log to wandb ourselves
    model.set_logger(sb3_configure(str(out_dir / "sb3_logs"), format_strings=["csv"]))

    fill_replay_buffer(model, dataset)

    # ── Initial policy diagnosis ───────────────────────────────────────────
    print(f"\n  === Initial Policy Diagnosis (before training) ===")
    diag = diagnose_policy(model, dataset, model.device)
    for k, v in diag.items():
        print(f"  {k}: {v:.6f}")
    wandb.log(diag, step=0)

    # ── Training loop with tqdm ────────────────────────────────────────────
    batch_size = hyperparams.get("batch_size", 256)
    t0 = time.time()
    extract_metrics = extract_sac_metrics if algo_name == "SAC" else extract_td3_metrics

    # Initial eval at step 0
    mean_r, std_r, success, mean_len = evaluate_antmaze(model)
    wandb.log({
        "evaluation/mean_return": mean_r, "evaluation/std_return": std_r,
        "evaluation/success_rate": success, "evaluation/mean_ep_length": mean_len,
    }, step=0)
    print(f"  [        0]  return {mean_r:+.2f} ± {std_r:.2f}  "
          f"success {success:.1%}  ep_len {mean_len:.0f}")

    pbar = tqdm(range(1, TOTAL_GRADIENT_STEPS + 1), desc=f"{algo_name}", unit="step",
                dynamic_ncols=True, miniters=LOG_FREQ)

    for step in pbar:
        model.policy.set_training_mode(True)
        model.train(gradient_steps=GRADIENT_STEPS_PER_ITER, batch_size=batch_size)
        model.policy.set_training_mode(False)

        # ── Log training metrics ───────────────────────────────────────────
        if step % LOG_FREQ == 0:
            metrics = extract_metrics(model)
            grad_norms = compute_gradient_norms(model)
            metrics.update(grad_norms)
            metrics["training/step"] = step

            wandb.log(metrics, step=step)

            # Update tqdm postfix with key metrics
            postfix = {}
            if "training/critic_loss" in metrics:
                postfix["crit"] = f"{metrics['training/critic_loss']:.3f}"
            if "training/actor_loss" in metrics:
                postfix["act"] = f"{metrics['training/actor_loss']:.3f}"
            if "grad_norm/critic" in grad_norms:
                postfix["g_crit"] = f"{grad_norms['grad_norm/critic']:.2f}"
            if "grad_norm/actor" in grad_norms:
                postfix["g_act"] = f"{grad_norms['grad_norm/actor']:.2f}"
            if algo_name == "SAC" and "training/ent_coef" in metrics:
                postfix["α"] = f"{metrics['training/ent_coef']:.4f}"
            pbar.set_postfix(postfix)

        # ── Evaluation ─────────────────────────────────────────────────────
        if step % EVAL_FREQ == 0:
            mean_r, std_r, success, mean_len = evaluate_antmaze(model)

            # Also run policy diagnosis
            diag = diagnose_policy(model, dataset, model.device)

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
                f"  [{step:>9,}]  return {mean_r:+.2f} ± {std_r:.2f}  "
                f"success {success:.1%}  ep_len {mean_len:.0f}  "
                f"act_mse_vs_data {diag['diag/action_mse_vs_data']:.4f}  "
                f"({elapsed:.1f} min)"
            )

    pbar.close()
    print(f"  Training done ({(time.time()-t0)/60:.1f} min total)")

    # ── Final diagnosis ────────────────────────────────────────────────────
    print(f"\n  === Final Policy Diagnosis ===")
    diag = diagnose_policy(model, dataset, model.device)
    for k, v in diag.items():
        print(f"  {k}: {v:.6f}")

    model.save(str(out_dir / "model"))

    video_path = out_dir / "videos" / "eval_video.mp4"
    print(f"  Recording video → {video_path}")
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
# PPO Behavioral Cloning — with diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def train_offline_ppo(
    config_name: str,
    hyperparams: dict,
    dataset: dict,
    project: str,
    seed: int,
) -> None:
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
    print(f"  [        0]  return {mean_r:+.2f} ± {std_r:.2f}  "
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

            # Policy diagnosis
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
                f"  [{step:>9,}]  return {mean_r:+.2f} ± {std_r:.2f}  "
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
    print(f"  Recording video → {video_path}")
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
        description="Offline RL baselines (SAC/PPO/TD3) on antmaze-umaze-v2 — v2 with diagnostics"
    )
    p.add_argument("--algos", nargs="+", default=list(CONFIGS.keys()),
                   choices=list(CONFIGS.keys()))
    p.add_argument("--configs", nargs="+", default=None)
    p.add_argument("--project", default="antmaze_offline_baselines")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    dataset = load_dataset()

    for algo in args.algos:
        algo_configs = dict(CONFIGS[algo])
        if args.configs is not None:
            algo_configs = {k: v for k, v in algo_configs.items() if k in args.configs}
            if not algo_configs:
                print(f"Warning: no matching configs for {algo}")
                continue

        for config_name, hyperparams in algo_configs.items():
            try:
                if algo == "PPO":
                    train_offline_ppo(config_name, hyperparams, dataset,
                                      args.project, args.seed)
                else:
                    train_offline_off_policy(algo, config_name, hyperparams, dataset,
                                              args.project, args.seed)
            except Exception:
                print(f"\nERROR during {algo}/{config_name}:")
                traceback.print_exc()
                print("Continuing with next config...\n")


if __name__ == "__main__":
    main()