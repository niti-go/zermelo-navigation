'''
Trains Behavior Cloning on an offline dataset of the Zermelo navigation environment.

=== FULL WORKFLOW (run these in order) ===

# 1. Start a tmux session
tmux new -s bc_zermelo

# 2. Generate the offline dataset (uses zermelo conda env)
conda activate zermelo
cd ~/zermelo-navigation
PYTHONPATH=. python scripts/generate_dataset.py

# 3. (Optional) Recompute rewards with new weights — no regeneration needed
PYTHONPATH=. python scripts/recompute_rewards.py \
    --action_weight=<aw> \
    --fixed_hover_cost=<hc> \
    --progress_weight=<pw>

# 4. (Optional) Visualize trajectories from the dataset
PYTHONPATH=. python scripts/visualize.py
#    This saves a video of N random trajectories to datasets/video.mp4

# 5. Train BC (uses flowrl conda env, can run from any directory)
conda activate flowrl
wandb login
CUDA_VISIBLE_DEVICES=6 python ~/zermelo-navigation/scripts/bc_zermelo.py \
    --train_steps=500000 \
    --seed=0 \
    --proj_wandb=zermelo_hit_dynamic_poordataset \
    --run_group=bc \
    --wandb_online=True

# 6. Detach tmux: Ctrl+b, then d
# 7. Reattach later: tmux attach -t bc_zermelo

=== WANDB LOGGING ===
  Entity:  --wandb_entity (default: RL_Control_JX)
  Project: --proj_wandb   (default: zermelo)
  Group:   --run_group    (default: bc)
  Mode:    --wandb_online (default: True, set False for offline logging)

=== CHECKPOINTS ===
  Saved to: exp/<proj_wandb>/<run_group>/<exp_name>/
  Includes: model checkpoints, obs_norm_stats.npz, flags.json
'''
import argparse
import json
import os
import random
import sys
import time

os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['MUJOCO_GL'] = 'egl'

import numpy as np
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm

# Make scripts/ importable for the shared helpers, and the repo root for
# `import zermelo_env` regardless of CWD.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(_SCRIPT_DIR))

import _training_common as tc  # noqa: E402
from zermelo_env.zermelo_config import load_config  # noqa: E402


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class BCPolicy(nn.Module):
    """Simple MLP policy for behavior cloning."""

    def __init__(self, obs_dim, act_dim, hidden_dims=(256, 256)):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, act_dim))
        layers.append(nn.Tanh())  # actions in [-1, 1]
        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        return self.net(obs)


class OfflineDataset:
    """Random-batch sampler over a dict of arrays."""

    def __init__(self, data_dict, device='cpu'):
        self.data = {k: torch.tensor(v, device=device) for k, v in data_dict.items()}
        self.size = len(next(iter(self.data.values())))

    def sample(self, batch_size):
        idx = torch.randint(0, self.size, (batch_size,))
        return {k: v[idx] for k, v in self.data.items()}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(policy, env, num_episodes, obs_mean, obs_std, device):
    """Run policy in env, return dict of mean metrics."""
    returns, lengths, successes = [], [], []
    for _ in range(num_episodes):
        ob, _ = env.reset()
        done = False
        ep_ret, ep_len = 0.0, 0
        while not done:
            ob_t = torch.tensor((ob - obs_mean) / obs_std,
                                dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = policy(ob_t).cpu().numpy()[0]
            action = np.clip(action, -1.0, 1.0)
            ob, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_ret += reward
            ep_len += 1
        returns.append(ep_ret)
        lengths.append(ep_len)
        successes.append(info.get('success', 0.0))
    return {
        'episode.return': np.mean(returns),
        'episode.length': np.mean(lengths),
        'success': np.mean(successes),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Behavior Cloning on Zermelo')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--train_steps', type=int, default=500000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[256, 256])
    parser.add_argument('--eval_interval', type=int, default=50000)
    parser.add_argument('--eval_episodes', type=int, default=50)
    parser.add_argument('--save_interval', type=int, default=50000)
    parser.add_argument('--log_interval', type=int, default=5000)
    parser.add_argument('--zermelo_dataset', type=str, default=None)
    parser.add_argument('--zermelo_config', type=str, default=None)
    parser.add_argument('--proj_wandb', type=str, default='zermelo')
    parser.add_argument('--run_group', type=str, default='bc')
    parser.add_argument('--wandb_entity', type=str, default='RL_Control_JX')
    parser.add_argument('--wandb_online', type=bool, default=True)
    parser.add_argument('--save_dir', type=str, default='exp/')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    zermelo_cfg = load_config(args.zermelo_config)
    zermelo_cfg_src = tc.default_config_src_path(args.zermelo_config)

    # Wandb.
    exp_name = f'bc_sd{args.seed:03d}_{time.strftime("%Y%m%d_%H%M%S")}'
    entity = args.wandb_entity if args.wandb_entity != 'None' else None
    os.environ["WANDB_MODE"] = "online" if args.wandb_online else "offline"
    wandb.init(project=args.proj_wandb, group=args.run_group, name=exp_name,
               entity=entity,
               config={**vars(args), 'zermelo_config_yaml': zermelo_cfg})
    if os.path.isfile(zermelo_cfg_src):
        wandb.save(zermelo_cfg_src, policy='now')

    save_dir = os.path.join(args.save_dir, args.proj_wandb, args.run_group, exp_name)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'flags.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    with open(os.path.join(save_dir, 'zermelo_config.json'), 'w') as f:
        json.dump(zermelo_cfg, f, indent=2)

    # Dataset.
    dataset_path = args.zermelo_dataset or tc.default_dataset_path(zermelo_cfg)
    data, train_segs, val_segs = tc.load_episode_segments(dataset_path)
    train_flat = tc.flatten_segments(data, train_segs, ['observations', 'actions'])
    val_flat = tc.flatten_segments(data, val_segs, ['observations', 'actions'])

    # Observation normalization.
    obs_mean, obs_std = tc.compute_obs_norm(train_flat['observations'])
    train_flat['observations'] = (train_flat['observations'] - obs_mean) / obs_std
    val_flat['observations'] = (val_flat['observations'] - obs_mean) / obs_std
    tc.save_obs_norm(save_dir, obs_mean, obs_std)

    train_ds = OfflineDataset(train_flat, device)
    val_ds = OfflineDataset(val_flat, device)

    # Model.
    obs_dim = train_flat['observations'].shape[1]
    act_dim = train_flat['actions'].shape[1]
    policy = BCPolicy(obs_dim, act_dim, hidden_dims=args.hidden_dims).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    print(f"Policy params: {sum(p.numel() for p in policy.parameters()):,}")

    eval_env = tc.make_eval_env(args.zermelo_config)

    # Training.
    first_time = time.time()
    last_time = time.time()
    for step in tqdm(range(1, args.train_steps + 1), desc="BC Training", dynamic_ncols=True):
        batch = train_ds.sample(args.batch_size)
        pred_actions = policy(batch['observations'])
        loss = nn.functional.mse_loss(pred_actions, batch['actions'])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % args.log_interval == 0:
            with torch.no_grad():
                val_batch = val_ds.sample(args.batch_size)
                val_loss = nn.functional.mse_loss(
                    policy(val_batch['observations']), val_batch['actions']).item()
            wandb.log({
                'training/loss': loss.item(),
                'validation/loss': val_loss,
                'time/epoch_time': (time.time() - last_time) / args.log_interval,
                'time/total_time': time.time() - first_time,
            }, step=step)
            last_time = time.time()

        if step % args.eval_interval == 0:
            policy.eval()
            eval_info = evaluate(policy, eval_env, args.eval_episodes,
                                 obs_mean, obs_std, device)
            policy.train()
            wandb.log({f'evaluation/{k}': v for k, v in eval_info.items()}, step=step)
            print(f"  Step {step}: return={eval_info['episode.return']:.2f}, "
                  f"success={eval_info['success']:.2f}, "
                  f"length={eval_info['episode.length']:.0f}")

        if step % args.save_interval == 0:
            torch.save(policy.state_dict(), os.path.join(save_dir, f'policy_{step}.pt'))

    wandb.finish()
    print("Done.")


if __name__ == '__main__':
    main()
