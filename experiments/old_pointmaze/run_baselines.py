#!/usr/bin/env python3
"""
Run multiple RL algorithms (PPO, SAC, TD3, DDPG) on PointMaze UMaze,
with multiple hyperparameter configurations each.

- Auto-organized logs/checkpoints/videos per experiment
- Weights & Biases logging for every run (losses, evals, videos)
- EvalCallback for best-model saving
- Video generation after each experiment (one video per episode)

Usage:
    tmux new -s NEW_pointmaze_baselines
    conda activate pointmaze
    wandb login        # one-time setup
    cd ~/newattempt
    CUDA_VISIBLE_DEVICES=4 python run_baselines.py
    
    # 7. Detach: Ctrl+b, then d
   # 8. Reattach later: tmux attach -t NEW_pointmaze_baselines

"""

import os
os.environ["MUJOCO_GL"] = "egl"

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import imageio

import wandb
from wandb.integration.sb3 import WandbCallback

from stable_baselines3 import PPO, SAC, TD3, DDPG
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CallbackList

# =========================================================
# 1. GLOBAL CONFIG
# =========================================================
ENV_NAME = "PointMaze_UMaze-v3"

BASE_DIR = "./experiments"
TOTAL_TIMESTEPS = 1_000_000
N_ENVS = 8

WANDB_PROJECT = "pointmaze-baselines"

# =========================================================
# 2. ENV FACTORY
# =========================================================
def make_pointmaze_env():
    gym.register_envs(gymnasium_robotics)
    env = gym.make(ENV_NAME, continuing_task=True, reset_target=True)
    return Monitor(env)


# =========================================================
# 3. ALGORITHM REGISTRY
# =========================================================
ALGO_REGISTRY = {
    "PPO": PPO,
    "SAC": SAC,
    "TD3": TD3,
    "DDPG": DDPG,
}

# Each algorithm can have multiple experiments
ALGO_CONFIGS = {
    "PPO": [
        {
            "n_steps": 2048,
            "batch_size": 64,
            "learning_rate": 3e-4,
            "gae_lambda": 0.95,
            "gamma": 0.99,
            "clip_range": 0.2,
        },
        {
            "n_steps": 1024,
            "batch_size": 128,
            "learning_rate": 1e-4,
            "gae_lambda": 0.92,
        },
    ],

    "SAC": [
        {
            "batch_size": 256,
            "learning_rate": 3e-4,
            "gamma": 0.99,
        }
    ],

    "TD3": [
        {
            "batch_size": 256,
            "learning_rate": 1e-3,
            "gamma": 0.98,
        }
    ],

    "DDPG": [
        {
            "batch_size": 256,
            "learning_rate": 1e-3,
            "gamma": 0.98,
        }
    ],
}


# =========================================================
# 4. UTILITY: Eval + Video (one video per episode)
# =========================================================
def evaluate_and_record(model, save_dir, n_episodes=3):
    """Run eval episodes & save one mp4 video per episode. Returns list of video paths."""
    eval_env = gym.make(
        ENV_NAME,
        render_mode="rgb_array",
        continuing_task=True,
        reset_target=True,
    )

    obs, info = eval_env.reset(seed=0)
    all_returns = []
    video_paths = []

    for ep in range(n_episodes):
        done = False
        truncated = False
        ep_ret = 0
        episode_frames = []

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = eval_env.step(action)
            ep_ret += reward

            frame = eval_env.render()
            if frame is not None:
                episode_frames.append(frame)

        all_returns.append(ep_ret)

        # Save this episode as its own video
        video_path = os.path.join(save_dir, f"eval_episode_{ep}.mp4")
        if episode_frames:
            imageio.mimsave(video_path, episode_frames, fps=30)
            video_paths.append(video_path)
            print(f"  Episode {ep}: return={ep_ret:.1f}, frames={len(episode_frames)}, saved => {video_path}")

        obs, info = eval_env.reset()

    eval_env.close()

    avg_return = float(np.mean(all_returns))
    print(f"[Eval] Average return over {n_episodes} episodes: {avg_return:.3f}")

    return avg_return, video_paths

# =========================================================
# 5. RUN EXPERIMENT
# =========================================================
def run_experiment(algo_name, config_dict, exp_id):
    print(f"\n========== RUNNING {algo_name} | config {exp_id} ==========\n")

    # Create directory structure
    exp_dir = os.path.join(BASE_DIR, algo_name, f"config_{exp_id}")
    tb_log_dir = os.path.join(exp_dir, "tb")
    model_dir = os.path.join(exp_dir, "models")
    video_dir = os.path.join(exp_dir, "videos")

    os.makedirs(tb_log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)

    # Initialize W&B run for this experiment
    run = wandb.init(
        project=WANDB_PROJECT,
        name=f"{algo_name}_config_{exp_id}",
        config={
            "algorithm": algo_name,
            "config_id": exp_id,
            "env": ENV_NAME,
            "total_timesteps": TOTAL_TIMESTEPS,
            "n_envs": N_ENVS,
            **config_dict,
        },
        sync_tensorboard=True,
        monitor_gym=True,
        reinit=True,
    )

    # Vectorized training env
    vec_env = make_vec_env(make_pointmaze_env, n_envs=N_ENVS)

    # Eval env
    eval_env = Monitor(gym.make(ENV_NAME, continuing_task=True, reset_target=True))

    # Instantiate model (keep tensorboard_log so SB3 emits loss metrics that W&B syncs)
    AlgoClass = ALGO_REGISTRY[algo_name]

    model = AlgoClass(
        policy="MultiInputPolicy",
        env=vec_env,
        verbose=1,
        tensorboard_log=tb_log_dir,
        **config_dict
    )

    # Callbacks: EvalCallback (frequent evals + best model) + WandbCallback (syncs to W&B)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=model_dir,
        eval_freq=50_000,
        n_eval_episodes=3,
        deterministic=True,
        render=False,
    )

    wandb_callback = WandbCallback(
        verbose=2,
    )

    callbacks = CallbackList([eval_callback, wandb_callback])

    # Train
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callbacks,
        tb_log_name=f"{algo_name}_config_{exp_id}"
    )

    # Save final model
    final_path = os.path.join(model_dir, "final_model.zip")
    model.save(final_path)
    print(f"Saved final model => {final_path}")

    # Load best model (if exists) for evaluation
    best_model_path = os.path.join(model_dir, "best_model.zip")
    if os.path.exists(best_model_path):
        model = AlgoClass.load(best_model_path)
        print("Loaded best model for evaluation.")
    else:
        print("No best_model found. Using final model.")

    # Evaluate + record one video per episode
    avg_return, video_paths = evaluate_and_record(model, video_dir)

    # Log eval results and videos to W&B
    wandb.log({"eval/final_avg_return": avg_return})

    for i, vpath in enumerate(video_paths):
        wandb.log({f"eval/video_episode_{i}": wandb.Video(vpath, fps=30, format="mp4")})

    # Upload best model as W&B artifact
    if os.path.exists(best_model_path):
        artifact = wandb.Artifact(
            name=f"{algo_name}_config_{exp_id}_best_model",
            type="model",
            description=f"Best model for {algo_name} config {exp_id} (eval return={avg_return:.1f})",
        )
        artifact.add_file(best_model_path)
        wandb.log_artifact(artifact)
        print(f"Uploaded best model to W&B as artifact.")

    wandb.finish()

    # Return summary
    return {
        "algo": algo_name,
        "config_id": exp_id,
        "avg_return": avg_return,
        "experiment_path": exp_dir,
    }


# =========================================================
# 6. MAIN LOOP OVER ALL ALGORITHMS
# =========================================================
def main():
    all_results = []

    for algo_name, config_list in ALGO_CONFIGS.items():
        for i, config in enumerate(config_list, start=1):
            result = run_experiment(algo_name, config, exp_id=i)
            all_results.append(result)

    print("\n==================== SUMMARY =====================")
    for r in all_results:
        print(r)


if __name__ == "__main__":
    main()
