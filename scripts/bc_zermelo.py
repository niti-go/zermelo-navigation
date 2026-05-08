'''
Trains Behavior Cloning on an offline dataset of the Zermelo navigation environment.

=== FULL WORKFLOW (run these in order) ===

# 1. Start a tmux session
tmux new -s bc_zermelo

# 2. Generate the offline dataset (uses zermelo conda env)
conda activate zermelo
cd ~/zermelo-navigation
PYTHONPATH=. python scripts/generate_dataset.py

# 3. Find good reward weights using the analysis script
PYTHONPATH=. python scripts/analyze_rewards.py
#    Check the recommended weights in the output and plots in datasets/hyperparameter_tuning/.

# 4. Recompute rewards in the existing dataset (no regeneration needed)
PYTHONPATH=. python scripts/recompute_rewards.py \
    --energy_weight=<recommended ew> \
    --time_weight=<recommended tw> \
    --distance_weight=<recommended dw>

# 5. Visualize trajectories from the dataset (optional)
PYTHONPATH=. python scripts/visualize.py
#    This saves a video of 5 random trajectories to datasets/video.mp4

# 6. Train BC (uses flowrl conda env, can run from any directory)
conda activate flowrl
wandb login
CUDA_VISIBLE_DEVICES=6 python ~/zermelo-navigation/experiments/zermelo/bc_zermelo.py \
    --train_steps=500000 \
    --seed=0 \
    --proj_wandb=zermelo_hit_dynamic_poordataset \
    --run_group=bc \
    --wandb_online=True

# 7. Detach tmux: Ctrl+b, then d
# 8. Reattach later: tmux attach -t bc_zermelo

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

import gymnasium
import numpy as np
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm

# Zermelo env imports.
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, _repo_root)
import zermelo_env  # noqa — registers gymnasium envs
from zermelo_env.zermelo_config import load_config, config_to_env_kwargs


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


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_zermelo_dataset(dataset_path, train_test_split=0.8):
    """Load Zermelo .npz and split by episode into train/val."""
    print(f"Loading dataset: {dataset_path}")
    data = dict(np.load(dataset_path))

    obs = data['observations'].astype(np.float32)
    actions = data['actions'].astype(np.float32)
    terminals = data['terminals'].astype(np.float32)

    # Episode-level split.
    ends = np.where(terminals > 0.5)[0]
    starts = np.concatenate([[0], ends[:-1] + 1])
    num_ep = len(starts)
    ep_idx = np.arange(num_ep)
    np.random.shuffle(ep_idx)

    n_train = int(num_ep * train_test_split)
    train_trans = np.concatenate([np.arange(starts[e], ends[e] + 1) for e in sorted(ep_idx[:n_train])])
    val_trans = np.concatenate([np.arange(starts[e], ends[e] + 1) for e in sorted(ep_idx[n_train:])])

    train = {'observations': obs[train_trans], 'actions': actions[train_trans]}
    val = {'observations': obs[val_trans], 'actions': actions[val_trans]}

    print(f"  Train: {len(train_trans)} transitions ({n_train} episodes)")
    print(f"  Val:   {len(val_trans)} transitions ({num_ep - n_train} episodes)")
    print(f"  Obs shape: {obs.shape[1:]}, Act shape: {actions.shape[1:]}")
    return train, val


class OfflineDataset:
    """Simple random-batch sampler over a dict of arrays."""

    def __init__(self, data_dict, device='cpu'):
        self.data = {k: torch.tensor(v, device=device) for k, v in data_dict.items()}
        self.size = len(next(iter(self.data.values())))

    def sample(self, batch_size):
        idx = torch.randint(0, self.size, (batch_size,))
        return {k: v[idx] for k, v in self.data.items()}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def make_eval_env(zermelo_config_path=None):
    cfg = load_config(zermelo_config_path)
    env_kwargs = config_to_env_kwargs(cfg)
    env_kwargs['fixed_start_goal'] = True
    env_kwargs['max_episode_steps'] = cfg['run']['max_episode_steps']
    return gymnasium.make('zermelo-pointmaze-medium-v0', **env_kwargs)


def evaluate(policy, env, num_episodes, obs_mean, obs_std, device):
    """Run policy in env, return dict of mean metrics."""
    returns, lengths, successes = [], [], []
    for _ in range(num_episodes):
        ob, _ = env.reset()
        done = False
        ep_ret, ep_len = 0.0, 0
        while not done:
            ob_t = torch.tensor((ob - obs_mean) / obs_std, dtype=torch.float32, device=device).unsqueeze(0)
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

    # Seeds.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load the zermelo YAML config up front so we can log it alongside argparse flags.
    zermelo_cfg = load_config(args.zermelo_config)
    zermelo_cfg_src = args.zermelo_config or os.path.join(
        _repo_root, 'configs', 'zermelo_config.yaml')

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
    if args.zermelo_dataset is None:
        args.zermelo_dataset = os.path.join(_repo_root, zermelo_cfg['run']['save_path'])

    train_data, val_data = load_zermelo_dataset(args.zermelo_dataset)

    # Observation normalization.
    obs_mean = train_data['observations'].mean(axis=0)
    obs_std = train_data['observations'].std(axis=0) + 1e-6
    train_data['observations'] = (train_data['observations'] - obs_mean) / obs_std
    val_data['observations'] = (val_data['observations'] - obs_mean) / obs_std
    np.savez(os.path.join(save_dir, 'obs_norm_stats.npz'), obs_mean=obs_mean, obs_std=obs_std)

    train_ds = OfflineDataset(train_data, device)
    val_ds = OfflineDataset(val_data, device)

    # Model.
    obs_dim = train_data['observations'].shape[1]
    act_dim = train_data['actions'].shape[1]
    policy = BCPolicy(obs_dim, act_dim, hidden_dims=args.hidden_dims).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    print(f"Policy params: {sum(p.numel() for p in policy.parameters()):,}")

    # Eval env.
    eval_env = make_eval_env(args.zermelo_config)

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

        # Log.
        if step % args.log_interval == 0:
            with torch.no_grad():
                val_batch = val_ds.sample(args.batch_size)
                val_loss = nn.functional.mse_loss(policy(val_batch['observations']), val_batch['actions']).item()

            metrics = {
                'training/loss': loss.item(),
                'validation/loss': val_loss,
                'time/epoch_time': (time.time() - last_time) / args.log_interval,
                'time/total_time': time.time() - first_time,
            }
            wandb.log(metrics, step=step)
            last_time = time.time()

        # Eval.
        if step % args.eval_interval == 0:
            policy.eval()
            eval_info = evaluate(policy, eval_env, args.eval_episodes, obs_mean, obs_std, device)
            policy.train()
            eval_metrics = {f'evaluation/{k}': v for k, v in eval_info.items()}
            wandb.log(eval_metrics, step=step)
            print(f"  Step {step}: return={eval_info['episode.return']:.2f}, "
                  f"success={eval_info['success']:.2f}, length={eval_info['episode.length']:.0f}")

        # Save.
        if step % args.save_interval == 0:
            ckpt_path = os.path.join(save_dir, f'policy_{step}.pt')
            torch.save(policy.state_dict(), ckpt_path)

    wandb.finish()
    print("Done.")


if __name__ == '__main__':
    main()
